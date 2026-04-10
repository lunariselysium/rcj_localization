#define _USE_MATH_DEFINES
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



// Global variables for trackbar access
int height_cm = 20;           // camera height in cm (default 20 cm)
int pitch_deg_trackbar = 120; // trackbar 0-180, where 0 = -90°, 90 = 0°, 180 = +90°

struct Point2D {
    double x; // Forward distance from robot (meters)
    double y; // Lateral distance from robot (meters) (Left is positive)
};

class VisionProcessor {
public:
    // Camera Extrinsics (Physical setup on the robot)
    double camera_height = 0.20; // 20 cm off the ground
    double camera_pitch = 30.0 * (M_PI / 180.0); // Tilted down 30 degrees

    // Camera Intrinsics
    double fx = 549.45489; // data[0] from projection_matrix
    double fy = 556.93243; // data[5] from projection_matrix
    double cx = 492.11415; // data[2] from projection_matrix
    double cy = 312.57320; // data[6] from projection_matrix

    // Process image and return a list of points on the floor (in meters)
    std::vector<Point2D> extractFieldLines(cv::Mat& img) {
        std::vector<Point2D> local_points;

        // 1. Horizon Crop (Ignore the top part of the image where the field isn't)
        int horizon_y = img.rows / 10; // Crop top 1/xth
        cv::Rect roi(0, horizon_y, img.cols, img.rows - horizon_y);
        cv::Mat cropped_img = img(roi);

        // 2. Convert to HSV
        cv::Mat hsv;
        cv::cvtColor(cropped_img, hsv, cv::COLOR_BGR2HSV);

        // 3. Threshold for Lines
        cv::Mat white_mask;
        cv::Scalar lower_white(0, 0, 158);
        cv::Scalar upper_white(180, 180, 205);
        cv::inRange(hsv, lower_white, upper_white, white_mask);

        cv::Mat black_mask;
        cv::inRange(hsv, cv::Scalar(0, 10, 42), cv::Scalar(180, 255, 102), black_mask); //10 filters out everything for now

        // Combine
        cv::Mat feature_mask;
        cv::bitwise_or(white_mask, black_mask, feature_mask);

        // Optional: Erode/Dilate to remove tiny noise pixels
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(feature_mask, feature_mask, cv::MORPH_OPEN, kernel);

        // --- Blob / Contour Filtering ---
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(feature_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Create a new black mask to draw only the valid contours onto
        cv::Mat filtered_mask = cv::Mat::zeros(feature_mask.size(), CV_8UC1);

        double min_area = 50.0; // Min pixel area to be considered a line segment
        double max_area = 80000.0; // Max area to filter out huge white blobs (like a wall)

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > min_area && area < max_area) {
                // Draw this valid contour onto our new mask
                cv::drawContours(filtered_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
            }
        }

        // 4. Extract points and apply IPM (Inverse Perspective Mapping)
        int rows = filtered_mask.rows;
        int cols = filtered_mask.cols;

        for (int v_cropped = 0; v_cropped < rows; v_cropped++) {
            // Get a pointer to the start of this row
            const uchar* row_ptr = filtered_mask.ptr<uchar>(v_cropped);

            for (int u = 0; u < cols; u++) {
                if (row_ptr[u] == 255) {
                    
                    // Re-add the cropped height to get original image Y coordinate
                    int v = v_cropped + horizon_y;

                    // --- INVERSE PERSPECTIVE MAPPING MATH ---
                    // Calculate ray angle vertically based on pixel 'v'
                    double ray_angle_y = atan((v - cy) / fy);

                    // If the ray is pointing above the horizon, skip it (can't hit the floor)
                    if (camera_pitch + ray_angle_y <= 0) continue;

                    // Forward distance (X in ROS)
                    double distance_x = camera_height / tan(camera_pitch + ray_angle_y);

                    // Lateral distance (Y in ROS)
                    // Note: OpenCV 'u' goes left-to-right. ROS 'Y' goes right-to-left.
                    double distance_y = distance_x * (cx - u) / fx;

                    // Save the projected 2D floor point
                    local_points.push_back({distance_x, distance_y});
                }
            }
        }

        // --- FOR DEBUG VISUALIZATION ONLY ---
        cv::Mat green_color(cropped_img.size(), CV_8UC3, cv::Scalar(0, 255, 0));
        green_color.copyTo(cropped_img, filtered_mask);

        // Cap amount of points
        if (local_points.size() > 5000) {
            std::vector<Point2D> decimated_points;
            int step = local_points.size() / 5000;
            for (size_t i = 0; i < local_points.size(); i += step) {
                decimated_points.push_back(local_points[i]);
                if (decimated_points.size() >= 5000) break;
            }
            return decimated_points;
        }

        return local_points;
    }
};

// Global publisher pointers (initialized in main)
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg,
                   rclcpp::Node::SharedPtr node) {
    cv::Mat frame;
    try {
        frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(node->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Read current trackbar positions
    height_cm = cv::getTrackbarPos("Height (cm)", "Camera Tuner");
    pitch_deg_trackbar = cv::getTrackbarPos("Pitch (deg)", "Camera Tuner");

    // Update VisionProcessor parameters
    VisionProcessor vp;
    vp.camera_height = height_cm / 100.0; // cm -> meters
    // Convert trackbar value (0-180) to degrees (-90 to +90)
    double pitch_deg = static_cast<double>(pitch_deg_trackbar) - 90.0;
    vp.camera_pitch = pitch_deg * (M_PI / 180.0); // deg -> radians

    // Process image
    auto points = vp.extractFieldLines(frame);

    // Publish debug image
    std_msgs::msg::Header header;
    header.stamp = node->now();
    header.frame_id = "camera_link";
    sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
    debug_image_pub->publish(*debug_msg);

    // Publish RViz 3D Marker
    visualization_msgs::msg::Marker p_marker;
    p_marker.header.frame_id = "base_link";
    p_marker.header.stamp = node->now();
    p_marker.ns = "ipm_projection";
    p_marker.id = 0;
    p_marker.type = visualization_msgs::msg::Marker::POINTS;
    p_marker.action = visualization_msgs::msg::Marker::ADD;
    p_marker.scale.x = 0.02; p_marker.scale.y = 0.02;
    p_marker.color.r = 0.0f; p_marker.color.g = 1.0f; p_marker.color.b = 1.0f; p_marker.color.a = 1.0f;
    
    for (const auto& p : points) {
        geometry_msgs::msg::Point gp;
        gp.x = p.x; gp.y = p.y; gp.z = 0.0;
        p_marker.points.push_back(gp);
    }
    marker_pub->publish(p_marker);

    // Show OpenCV windows
    cv::imshow("Original Image", frame);
    // Create a mask visualization by re-thresholding (simplified)
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar lower_white(0, 0, 158);
    cv::Scalar upper_white(180, 180, 205);
    cv::inRange(hsv, lower_white, upper_white, mask);
    cv::Scalar lower_black(0, 10, 42);
    cv::Scalar upper_black(180, 255, 102);
    cv::Mat black_mask;
    cv::inRange(hsv, lower_black, upper_black, black_mask);
    cv::bitwise_or(mask, black_mask, mask);
    cv::imshow("Line Mask", mask);
    // cv::waitKey(1);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("camera_tuner_node");

    // Create GUI windows and trackbars
    cv::namedWindow("Camera Tuner", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Height (cm)", "Camera Tuner", &height_cm, 100); // 0-100 cm
    // Pitch: trackbar 0-180 maps to -90° to +90°, default 120 (i.e., 30° down)
    cv::createTrackbar("Pitch (deg)", "Camera Tuner", &pitch_deg_trackbar, 180);

    // Create publishers
    debug_image_pub = node->create_publisher<sensor_msgs::msg::Image>("/vision_debug/image_raw", 10);
    marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("/projected_lines", 10);

    auto sub = node->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", rclcpp::SensorDataQoS(),
        [node](const sensor_msgs::msg::Image::SharedPtr msg) {
            imageCallback(msg, node);
        });
    
    RCLCPP_INFO(node->get_logger(), "Camera Tuner Started. Adjust height and pitch.");
    RCLCPP_INFO(node->get_logger(), "Pitch: 0 = -90°, 90 = 0°, 180 = +90°");

    while (rclcpp::ok()) {
        // Process any pending ROS callbacks (like imageCallback)
        rclcpp::spin_some(node);
        
        // Process OpenCV GUI events (wait 10ms). This unfreezes the window!
        cv::waitKey(10); 
    }

    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}