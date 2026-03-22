#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/float32.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>


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
        int horizon_y = img.rows / 5; // Crop top 1/5th
        cv::Rect roi(0, horizon_y, img.cols, img.rows - horizon_y);
        cv::Mat cropped_img = img(roi);

        // 2. Convert to HSV
        cv::Mat hsv;
        cv::cvtColor(cropped_img, hsv, cv::COLOR_BGR2HSV);

        // 3. Threshold for White Lines
        // White has low saturation (0-40) and high value (brightness, 200-255)
        cv::Mat white_mask;
        cv::Scalar lower_white(0, 0, 132);
        cv::Scalar upper_white(180, 40, 191);
        cv::inRange(hsv, lower_white, upper_white, white_mask);

        // Optional: Erode/Dilate to remove tiny noise pixels
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(white_mask, white_mask, cv::MORPH_OPEN, kernel);

        // --- Blob / Contour Filtering ---
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(white_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Create a new black mask to draw only the valid contours onto
        cv::Mat filtered_mask = cv::Mat::zeros(white_mask.size(), CV_8UC1);

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
        for (int v_cropped = 0; v_cropped < filtered_mask.rows; v_cropped++) {
            for (int u = 0; u < filtered_mask.cols; u++) {
                
                // If pixel is white (value is 255)
                if (filtered_mask.at<uchar>(v_cropped, u) == 255) {
                    
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

        return local_points;
    }
};



class RcjLocalizationNode : public rclcpp::Node
{
public:
    RcjLocalizationNode() : Node("rcj_localization_node")
    {
        // --- 1. Parameters ---
        this->declare_parameter("num_particles", 1000);
        num_particles_ = this->get_parameter("num_particles").as_int();

        // --- 2. Subscribers ---
        // Best practice in ROS 2: Use SensorDataQoS for high-frequency topics like camera feeds
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", qos,
            std::bind(&RcjLocalizationNode::imageCallback, this, std::placeholders::_1));

        yaw_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/robot/yaw", 10,
            std::bind(&RcjLocalizationNode::yawCallback, this, std::placeholders::_1));

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(),
            std::bind(&RcjLocalizationNode::mapCallback, this, std::placeholders::_1));

        // --- 3. Publishers (Including Visualization) ---
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/amcl_pose", 10);
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/particlecloud", 10);
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/vision_debug/image_raw", 10);

        // --- 4. TF Broadcaster ---
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // --- 5. Main Filter Loop Timer (e.g., 20 Hz) ---
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&RcjLocalizationNode::filterLoop, this));

        RCLCPP_INFO(this->get_logger(), "RCJ Localization Node initialized with %d particles.", num_particles_);
    }

private:
    int num_particles_;
    float current_yaw_ = 0.0;
    cv::Mat latest_image_;
    VisionProcessor vision_processor_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr yaw_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

    // TF Broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;

    // --- CALLBACKS ---

    void yawCallback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        // Convert degrees to radians assuming incoming is 0-360
        current_yaw_ = msg->data * (M_PI / 180.0);
    }

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Map received. Building likelihood field...");
        // TODO: Store map and build Distance Transform map (Likelihood field)
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            latest_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // 1. Run the Vision Processor to get the 2D floor points
        std::vector<Point2D> line_points = vision_processor_.extractFieldLines(latest_image_);

        // 2. VISUALIZATION: Publish the debug image showing extracted lines
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_link";
        // Convert the modified image back to ROS format
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(header, "bgr8", latest_image_).toImageMsg();
        debug_image_pub_->publish(*debug_msg);

        // 3. Optional for later: Pass these points to the Particle Filter Update step
        // particle_filter_.update(line_points);
    }

    void filterLoop()
    {
        // This is the core AMCL algorithm loop

        // 1. PREDICT STEP: Spread particles based on Random Walk + current_yaw_
        // TODO: particle_filter.predict(current_yaw_);

        // 2. UPDATE STEP: Score particles using visual features vs Likelihood field
        // TODO: particle_filter.update(local_visual_features);

        // 3. RESAMPLE STEP: Weed out bad particles, multiply good ones
        // TODO: particle_filter.resample();

        // 4. PUBLISH VISUALIZATIONS & TF
        publishVisualizationsAndTF();
    }

    void publishVisualizationsAndTF()
    {
        rclcpp::Time now = this->now();

        // --- A. Publish Particle Cloud for RViz2 ---
        geometry_msgs::msg::PoseArray cloud_msg;
        cloud_msg.header.stamp = now;
        cloud_msg.header.frame_id = "map"; // Particles exist in the global map frame

        // TODO: Loop through your actual particles here. 
        // Example inserting dummy particles based on an estimated pose:
        for(int i = 0; i < num_particles_; i++){
            geometry_msgs::msg::Pose p;
            p.position.x = 1.0 + ((rand() % 100) / 100.0 * 0.5); // Dummy spread
            p.position.y = 1.0 + ((rand() % 100) / 100.0 * 0.5); 
            // Yaw visualization
            tf2::Quaternion q;
            q.setRPY(0, 0, current_yaw_);
            p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();
            cloud_msg.poses.push_back(p);
        }
        particle_pub_->publish(cloud_msg);

        // --- B. Publish Best Estimated Pose ---
        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
        pose_msg.header.stamp = now;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.pose.position.x = 1.0; // Dummy best X
        pose_msg.pose.pose.position.y = 1.0; // Dummy best Y
        tf2::Quaternion q_best;
        q_best.setRPY(0, 0, current_yaw_);
        pose_msg.pose.pose.orientation.x = q_best.x(); pose_msg.pose.pose.orientation.y = q_best.y(); 
        pose_msg.pose.pose.orientation.z = q_best.z(); pose_msg.pose.pose.orientation.w = q_best.w();
        pose_pub_->publish(pose_msg);

        // --- C. Broadcast TF (map -> base_link) ---
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = now;
        t.header.frame_id = "map";
        t.child_frame_id = "base_link";
        t.transform.translation.x = 1.0;
        t.transform.translation.y = 1.0;
        t.transform.translation.z = 0.0;
        t.transform.rotation = pose_msg.pose.pose.orientation;
        
        tf_broadcaster_->sendTransform(t);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RcjLocalizationNode>());
    rclcpp::shutdown();
    return 0;
}