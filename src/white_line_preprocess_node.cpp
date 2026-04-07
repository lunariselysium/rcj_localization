#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_preprocessing.hpp"

class WhiteLinePreprocessNode : public rclcpp::Node {
public:
    WhiteLinePreprocessNode() : Node("white_line_preprocess_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLinePreprocessNode::imageCallback, this, std::placeholders::_1));

        filtered_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/filtered_bgr", 10);
        luminance_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/luminance", 10);
        enhanced_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/enhanced", 10);

        cv::namedWindow("Preprocess Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Preprocess Filtered", cv::WINDOW_NORMAL);
        cv::namedWindow("Preprocess Luminance", cv::WINDOW_NORMAL);
        cv::namedWindow("Preprocess Enhanced", cv::WINDOW_NORMAL);

        RCLCPP_INFO(this->get_logger(), "white_line_preprocess_node started.");
    }

    ~WhiteLinePreprocessNode() override {
        cv::destroyWindow("Preprocess Original");
        cv::destroyWindow("Preprocess Filtered");
        cv::destroyWindow("Preprocess Luminance");
        cv::destroyWindow("Preprocess Enhanced");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr luminance_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr enhanced_pub_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "cv_bridge failed: %s", e.what());
            return;
        }

        const auto params = rcj_loc::vision::white_line::getPreprocessParams(*this);
        const auto preprocessed = rcj_loc::vision::white_line::preprocessFrame(frame, params);

        filtered_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", preprocessed.filtered_bgr).toImageMsg());
        luminance_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", preprocessed.luminance).toImageMsg());
        enhanced_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", preprocessed.enhanced).toImageMsg());

        cv::imshow("Preprocess Original", frame);
        cv::imshow("Preprocess Filtered", preprocessed.filtered_bgr);
        cv::imshow("Preprocess Luminance", preprocessed.luminance);
        cv::imshow("Preprocess Enhanced", preprocessed.enhanced);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLinePreprocessNode>());
    rclcpp::shutdown();
    return 0;
}
