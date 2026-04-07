#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace rcj_loc::vision {

struct Point2D {
    double x;  // Forward distance from robot (meters)
    double y;  // Lateral distance from robot (meters), left is positive
};

class VisionProcessor {
public:
    // Camera extrinsics (physical setup on the robot)
    double camera_height = 0.20;
    double camera_pitch = 30.0 * (M_PI / 180.0);

    // Camera intrinsics
    double fx = 549.45489;
    double fy = 556.93243;
    double cx = 492.11415;
    double cy = 312.57320;

    // Process image and return a list of points on the floor (in meters)
    std::vector<Point2D> extractFieldLines(cv::Mat &img);
};

}  // namespace rcj_loc::vision
