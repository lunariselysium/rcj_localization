#include "rcj_localization/particle_filter_v2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rcj_loc {

namespace {

double safeExp(double value) {
    if (value < -700.0) {
        return 0.0;
    }
    return std::exp(value);
}

}  // namespace

ParticleFilterV2::ParticleFilterV2(const ParticleFilterV2Config &config) : config_(config) {
    std::random_device rd;
    gen_ = std::mt19937(rd());
    initRandom();
}

void ParticleFilterV2::setConfig(const ParticleFilterV2Config &config) {
    config_ = config;
}

void ParticleFilterV2::initRandom() {
    std::uniform_real_distribution<double> dist_x(
        -config_.init_field_width / 2.0,
        config_.init_field_width / 2.0);
    std::uniform_real_distribution<double> dist_y(
        -config_.init_field_height / 2.0,
        config_.init_field_height / 2.0);
    std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

    particles_.clear();
    particles_.reserve(static_cast<std::size_t>(config_.num_particles));
    for (int i = 0; i < config_.num_particles; ++i) {
        particles_.push_back(
            {dist_x(gen_), dist_y(gen_), dist_theta(gen_), 1.0 / config_.num_particles});
    }

    alpha_slow_ = 0.0;
    alpha_fast_ = 0.0;
}

void ParticleFilterV2::setMap(const nav_msgs::msg::OccupancyGrid::SharedPtr &map_msg) {
    map_resolution_ = map_msg->info.resolution;
    map_origin_x_ = map_msg->info.origin.position.x;
    map_origin_y_ = map_msg->info.origin.position.y;
    const int width = static_cast<int>(map_msg->info.width);
    const int height = static_cast<int>(map_msg->info.height);

    cv::Mat binary_map(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int i = x + (y * width);
            const int cell_value = map_msg->data[i];
            binary_map.at<uchar>(y, x) =
                cell_value > config_.occupancy_threshold ? 0 : 255;
        }
    }

    cv::distanceTransform(
        binary_map,
        distance_map_,
        cv::DIST_L2,
        config_.distance_transform_mask_size);

    map_initialized_ = true;
}

void ParticleFilterV2::predict(double absolute_yaw) {
    std::normal_distribution<double> noise_xy(0.0, config_.noise_xy);
    std::normal_distribution<double> noise_theta(0.0, config_.noise_theta);

    for (auto &particle : particles_) {
        particle.x += noise_xy(gen_);
        particle.y += noise_xy(gen_);
        particle.theta = absolute_yaw + noise_theta(gen_);
    }
}

bool ParticleFilterV2::updateWeights(const std::vector<Point2D> &local_observations) {
    if (!map_initialized_ || local_observations.empty() || particles_.empty()) {
        return false;
    }

    std::vector<double> mean_log_likelihoods(particles_.size(), 0.0);
    double max_mean_log_likelihood = -std::numeric_limits<double>::infinity();
    double raw_weight_sum = 0.0;

    for (std::size_t i = 0; i < particles_.size(); ++i) {
        auto &particle = particles_[i];
        double log_weight_sum = 0.0;

        const double cos_t = std::cos(particle.theta);
        const double sin_t = std::sin(particle.theta);

        for (const auto &observation : local_observations) {
            const double global_x =
                particle.x + (observation.x * cos_t) - (observation.y * sin_t);
            const double global_y =
                particle.y + (observation.x * sin_t) + (observation.y * cos_t);

            const int px = static_cast<int>(
                std::round((global_x - map_origin_x_) / map_resolution_));
            const int py = static_cast<int>(
                std::round((global_y - map_origin_y_) / map_resolution_));

            double dist_to_line = config_.off_map_penalty;
            if (px >= 0 && px < distance_map_.cols && py >= 0 && py < distance_map_.rows) {
                dist_to_line = distance_map_.at<float>(py, px) * map_resolution_;
            }

            log_weight_sum +=
                -(dist_to_line * dist_to_line) / (2.0 * config_.sigma_hit * config_.sigma_hit);
        }

        const double mean_log_likelihood =
            log_weight_sum / static_cast<double>(local_observations.size());
        mean_log_likelihoods[i] = mean_log_likelihood;
        max_mean_log_likelihood = std::max(max_mean_log_likelihood, mean_log_likelihood);
        raw_weight_sum += safeExp(mean_log_likelihood);
    }

    const double avg_weight = raw_weight_sum / static_cast<double>(particles_.size());
    if (alpha_slow_ == 0.0) {
        alpha_slow_ = avg_weight;
        alpha_fast_ = avg_weight;
    } else {
        alpha_fast_ += config_.alpha_fast_rate * (avg_weight - alpha_fast_);
        alpha_slow_ += config_.alpha_slow_rate * (avg_weight - alpha_slow_);
    }

    double normalized_weight_sum = 0.0;
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        particles_[i].weight = safeExp(mean_log_likelihoods[i] - max_mean_log_likelihood);
        normalized_weight_sum += particles_[i].weight;
    }

    if (normalized_weight_sum > 0.0) {
        for (auto &particle : particles_) {
            particle.weight /= normalized_weight_sum;
        }
    } else {
        for (auto &particle : particles_) {
            particle.weight = 1.0 / static_cast<double>(particles_.size());
        }
    }

    return true;
}

void ParticleFilterV2::resample() {
    if (particles_.empty()) {
        return;
    }

    std::vector<Particle> new_particles;
    new_particles.reserve(particles_.size());

    std::uniform_real_distribution<double> random_dist(
        0.0,
        1.0 / static_cast<double>(particles_.size()));
    double r = random_dist(gen_);
    double c = particles_.front().weight;
    std::size_t i = 0;

    for (std::size_t m = 0; m < particles_.size(); ++m) {
        const double u = r + static_cast<double>(m) / static_cast<double>(particles_.size());
        while (u > c && !particles_.empty()) {
            i = (i + 1) % particles_.size();
            c += particles_[i].weight;
        }
        Particle survivor = particles_[i];
        survivor.weight = 1.0 / static_cast<double>(particles_.size());
        new_particles.push_back(survivor);
    }

    double p_random = 0.0;
    if (alpha_slow_ > std::numeric_limits<double>::epsilon()) {
        p_random = std::max(0.0, 1.0 - (alpha_fast_ / alpha_slow_));
    }
    p_random = std::min(p_random, config_.random_injection_max_ratio);

    const int random_count =
        static_cast<int>(static_cast<double>(particles_.size()) * p_random);

    double min_x = -config_.init_field_width / 2.0;
    double max_x = config_.init_field_width / 2.0;
    double min_y = -config_.init_field_height / 2.0;
    double max_y = config_.init_field_height / 2.0;
    if (map_initialized_) {
        min_x = map_origin_x_;
        max_x = map_origin_x_ + (distance_map_.cols * map_resolution_);
        min_y = map_origin_y_;
        max_y = map_origin_y_ + (distance_map_.rows * map_resolution_);
    }

    std::uniform_real_distribution<double> dist_x(min_x, max_x);
    std::uniform_real_distribution<double> dist_y(min_y, max_y);
    std::uniform_real_distribution<double> dist_theta(-M_PI, M_PI);

    for (int j = 0; j < random_count; ++j) {
        new_particles[new_particles.size() - 1 - j] = {
            dist_x(gen_),
            dist_y(gen_),
            dist_theta(gen_),
            1.0 / static_cast<double>(particles_.size())};
    }

    particles_ = new_particles;
}

Particle ParticleFilterV2::getBestPose() const {
    Particle best = particles_.front();
    for (const auto &particle : particles_) {
        if (particle.weight > best.weight) {
            best = particle;
        }
    }
    return best;
}

}  // namespace rcj_loc
