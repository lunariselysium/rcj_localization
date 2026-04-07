#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>

#include "rcj_localization/vision_processor.hpp"

class VisionProcessorNode : public rclcpp::Node {
public:
    VisionProcessorNode() : Node("vision_processor_node") {
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", qos,
            std::bind(&VisionProcessorNode::imageCallback, this, std::placeholders::_1));

        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/vision_debug/image_raw", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/projected_lines", 10);

        RCLCPP_INFO(this->get_logger(), "vision_processor_node started.");
    }

private:
    rcj_loc::vision::VisionProcessor vision_processor_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception &) {
            return;
        }

        const auto points = vision_processor_.extractFieldLines(frame);

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_link";
        sensor_msgs::msg::Image::SharedPtr debug_msg =
            cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        debug_image_pub_->publish(*debug_msg);

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = this->now();
        marker.ns = "ipm_projection";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::POINTS;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.02;
        marker.scale.y = 0.02;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0f;

        for (const auto &p : points) {
            geometry_msgs::msg::Point gp;
            gp.x = p.x;
            gp.y = p.y;
            gp.z = 0.0;
            marker.points.push_back(gp);
        }
        marker_pub_->publish(marker);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisionProcessorNode>());
    rclcpp::shutdown();
    return 0;
}
