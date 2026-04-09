#pragma once

#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace rcj_loc {

// A simple structure to hold 2D points from the camera
struct Point2D {
    double x; // Forward distance in meters
    double y; // Lateral distance in meters
};

// The state of a single particle
struct Particle {
    double x;
    double y;
    double theta; // Yaw angle in radians
    double weight;
};

class ParticleFilter {
public:
    ParticleFilter(int num_particles);

    // 1. Initialization
    void initRandom(double field_width, double field_height);

    // 2. Map Processing (Likelihood Field)
    void setMap(const nav_msgs::msg::OccupancyGrid::SharedPtr& map_msg);

    // 3. Motion Model (The "Spread")
    // If dx,dy are provided (non-zero), move particles by odometry delta plus noise.
    // Otherwise use random walk for position.
    void predict(double absolute_yaw, double dx = 0.0, double dy = 0.0);

    // 4. Sensor Model (Scoring against the map)
    void updateWeights(const std::vector<Point2D>& local_observations);

    // 5. Survival of the fittest
    void resample();

    // Getters
    const std::vector<Particle>& getParticles() const { return particles_; }
    Particle getBestPose() const;

private:
    int num_particles_;
    std::vector<Particle> particles_;

    // Map data for the Likelihood Field
    cv::Mat distance_map_;
    double map_resolution_;
    double map_origin_x_;
    double map_origin_y_;
    bool map_initialized_ = false;

    // Random number generators for noise
    std::mt19937 gen_;

    // Augmented MCL variables
    double alpha_slow_ = 0.0;
    double alpha_fast_ = 0.0;
    double alpha_slow_rate_ = 0.001; // Adapts very slowly
    double alpha_fast_rate_ = 0.05;   // Adapts very quickly
};

} // namespace rcj_loc