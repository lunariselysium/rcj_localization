#include "rcj_localization/particle_filter.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace rcj_loc {

ParticleFilter::ParticleFilter(int num_particles) : num_particles_(num_particles) {
    std::random_device rd;
    gen_ = std::mt19937(rd());
}

void ParticleFilter::initRandom(double field_width, double field_height) {
    std::uniform_real_distribution<double> dist_x(-field_width / 2.0, field_width / 2.0);
    std::uniform_real_distribution<double> dist_y(-field_height / 2.0, field_height / 2.0);
    std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

    particles_.clear();
    for (int i = 0; i < num_particles_; ++i) {
        particles_.push_back({dist_x(gen_), dist_y(gen_), dist_theta(gen_), 1.0 / num_particles_});
    }
}

void ParticleFilter::setMap(const nav_msgs::msg::OccupancyGrid::SharedPtr& map_msg) {
    map_resolution_ = map_msg->info.resolution;
    map_origin_x_ = map_msg->info.origin.position.x;
    map_origin_y_ = map_msg->info.origin.position.y;
    int width = map_msg->info.width;
    int height = map_msg->info.height;

    // Convert ROS OccupancyGrid to OpenCV Matrix
    cv::Mat binary_map(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = x + (y * width);
            int cell_value = map_msg->data[i];
            
            // In ROS: 100 = occupied (black lines), 0 = free (green carpet)
            if (cell_value > 50) {
                binary_map.at<uchar>(y, x) = 0;   // Obstacle is 0 in OpenCV
            } else {
                binary_map.at<uchar>(y, x) = 255; // Free space is 255
            }
        }
    }

    // --- LIKELIHOOD FIELD MAGIC ---
    // This function creates a gradient glow around the black lines.
    // If a pixel is on a line, distance is 0. If it's 10 pixels away, distance is 10.
    cv::distanceTransform(binary_map, distance_map_, cv::DIST_L2, 5);
    
    map_initialized_ = true;
}

void ParticleFilter::predict(double absolute_yaw) {
    // Because we don't have odometry yet, we use a "Random Walk".
    // We add a tiny bit of random noise to X and Y every frame to spread the particles out.
    std::normal_distribution<double> noise_xy(0.0, 0.05); // 5cm standard deviation noise
    std::normal_distribution<double> noise_theta(0.0, 0.1); // Small noise for yaw

    for (auto& p : particles_) {
        p.x += noise_xy(gen_);
        p.y += noise_xy(gen_);
        
        // --- BREAKING SYMMETRY ---
        // We force the particle to face the direction the IMU/Compass says we are facing.
        p.theta = absolute_yaw + noise_theta(gen_);
    }
}

void ParticleFilter::updateWeights(const std::vector<Point2D>& local_observations) {
    if (!map_initialized_ || local_observations.empty()) return;

    double weight_sum = 0.0;
    double sigma_hit = 0.10; // We expect camera points to be accurate within 10cm of real lines

    #pragma omp parrallel for reduction(+:weight_sum)
    for (auto& p : particles_) {
        double log_weight = 0.0;

        // Preconpute trig
        double cos_t = cos(p.theta);
        double sin_t = sin(p.theta);

        for (const auto& obs : local_observations) {
            // 1. Transform the local camera point to the global map frame using the particle's pose
            double global_x = p.x + (obs.x * cos_t) - (obs.y * sin_t);
            double global_y = p.y + (obs.x * sin_t) + (obs.y * cos_t);

            // 2. Convert global coordinate (meters) to map pixel
            int px = std::round((global_x - map_origin_x_) / map_resolution_);
            int py = std::round((global_y - map_origin_y_) / map_resolution_);

            // 3. Lookup distance to nearest line
            double dist_to_line = 1.0; // Default penalty if point is off the map
            if (px >= 0 && px < distance_map_.cols && py >= 0 && py < distance_map_.rows) {
                // Get the distance in pixels, convert back to meters
                dist_to_line = distance_map_.at<float>(py, px) * map_resolution_;
            }

            // 4. Calculate Gaussian probability
            // If dist is 0, prob is 1.0. If dist is large, prob drops towards 0.
            log_weight += -(dist_to_line * dist_to_line) / (2.0 * sigma_hit * sigma_hit);
        }

        p.weight = std::exp(log_weight);
        weight_sum += p.weight;
    }

    // Calculate the average weight for alpha tracking
    double avg_weight = weight_sum / num_particles_;
    
    // If it's the first run, initialize them
    if (alpha_slow_ == 0.0) {
        alpha_slow_ = avg_weight;
        alpha_fast_ = avg_weight;
    } else {
        // Update the rolling averages
        alpha_fast_ += alpha_fast_rate_ * (avg_weight - alpha_fast_);
        alpha_slow_ += alpha_slow_rate_ * (avg_weight - alpha_slow_);
    }

    // Normalize weights so they all add up to 1.0
    if (weight_sum > 0.0) {
        for (auto& p : particles_) {
            p.weight /= weight_sum;
        }
    } else {
        // If all particles are terribly wrong (weight_sum is 0), reset weights evenly
        for (auto& p : particles_) p.weight = 1.0 / num_particles_;
    }
}

void ParticleFilter::resample() {
    // "Low Variance Resampling" (Roulette Wheel selection)
    // Particles with high weight get copied multiple times. Low weight particles die.
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);

    std::uniform_real_distribution<double> random_dist(0.0, 1.0 / num_particles_);
    double r = random_dist(gen_);
    double c = particles_[0].weight;
    int i = 0;

    for (int m = 0; m < num_particles_; ++m) {
        double U = r + static_cast<double>(m) / num_particles_;
        while (U > c) {
            i = (i + 1) % num_particles_;
            c += particles_[i].weight;
        }
        // Copy the winning particle, but reset its weight
        Particle survivor = particles_[i];
        survivor.weight = 1.0 / num_particles_;
        new_particles.push_back(survivor);
    }

    // Alpha tracking to add particles when needed
    // If fast average drops below slow average, lost
    double p_random = std::max(0.0, 1.0 - (alpha_fast_ / alpha_slow_));
    
    // Limit to a maximum of 25% random injection to prevent total chaos
    p_random = std::min(p_random, 0.25); 

    int random_count = num_particles_ * p_random; 
    
    // Calculate the map boundaries dynamically based on loaded map
    double map_end_x = map_origin_x_ + (distance_map_.cols * map_resolution_);
    double map_end_y = map_origin_y_ + (distance_map_.rows * map_resolution_);

    std::uniform_real_distribution<double> dist_x(map_origin_x_, map_end_x);
    std::uniform_real_distribution<double> dist_y(map_origin_y_, map_end_y);
    std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

    for(int j = 0; j < random_count; j++) {
        new_particles[num_particles_ - 1 - j] = {
            dist_x(gen_), dist_y(gen_), dist_theta(gen_), 1.0 / num_particles_
        };
    }

    particles_ = new_particles;
}

Particle ParticleFilter::getBestPose() const {
    // Return the particle with the highest weight
    Particle best = particles_[0];
    for (const auto& p : particles_) {
        if (p.weight > best.weight) best = p;
    }
    return best;
}

} // namespace rcj_loc