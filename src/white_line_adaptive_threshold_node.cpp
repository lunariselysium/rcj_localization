#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

class WhiteLineAdaptiveThresholdNode : public rclcpp::Node {
public:
    WhiteLineAdaptiveThresholdNode() : Node("white_line_adaptive_threshold_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);
        this->declare_parameter("adaptive_block_size", 31);
        this->declare_parameter("adaptive_c", 5.0);
        this->declare_parameter("open_kernel", 3);
        this->declare_parameter("close_kernel", 5);
        this->declare_parameter("min_area", 40);
        this->declare_parameter("max_area", 50000);
        this->declare_parameter("min_major_axis", 10);
        this->declare_parameter("max_minor_axis", 30);
        this->declare_parameter("min_aspect_ratio", 1.0);

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLineAdaptiveThresholdNode::imageCallback, this, std::placeholders::_1));

        candidate_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/candidate_mask", 10);
        mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/mask", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

        cv::namedWindow("Adaptive Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Adaptive Enhanced", cv::WINDOW_NORMAL);
        cv::namedWindow("Adaptive Candidate Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Adaptive Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Adaptive Overlay", cv::WINDOW_NORMAL);

        RCLCPP_INFO(this->get_logger(), "white_line_adaptive_threshold_node started.");
    }

    ~WhiteLineAdaptiveThresholdNode() override {
        cv::destroyWindow("Adaptive Original");
        cv::destroyWindow("Adaptive Enhanced");
        cv::destroyWindow("Adaptive Candidate Mask");
        cv::destroyWindow("Adaptive Mask");
        cv::destroyWindow("Adaptive Overlay");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr candidate_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "cv_bridge failed: %s", e.what());
            return;
        }

        const auto preprocess_params = rcj_loc::vision::white_line::getPreprocessParams(*this);
        const auto preprocessed = rcj_loc::vision::white_line::preprocessFrame(frame, preprocess_params);

        cv::Mat candidate_mask;
        const int adaptive_block_size = rcj_loc::vision::debug::makeOdd(
            static_cast<int>(this->get_parameter("adaptive_block_size").as_int()),
            3);
        cv::adaptiveThreshold(
            preprocessed.enhanced,
            candidate_mask,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY,
            adaptive_block_size,
            this->get_parameter("adaptive_c").as_double());

        const cv::Mat raw_candidate_mask = candidate_mask.clone();

        const int open_kernel = rcj_loc::vision::debug::makeOdd(
            static_cast<int>(this->get_parameter("open_kernel").as_int()),
            1);
        const int close_kernel = rcj_loc::vision::debug::makeOdd(
            static_cast<int>(this->get_parameter("close_kernel").as_int()),
            1);

        const cv::Mat open_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(open_kernel, open_kernel));
        const cv::Mat close_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(close_kernel, close_kernel));
        cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_OPEN, open_element);
        cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_CLOSE, close_element);

        const rcj_loc::vision::debug::ComponentStatsFilter filter{
            static_cast<int>(this->get_parameter("min_area").as_int()),
            static_cast<int>(this->get_parameter("max_area").as_int()),
            static_cast<int>(this->get_parameter("min_major_axis").as_int()),
            static_cast<int>(this->get_parameter("max_minor_axis").as_int()),
            this->get_parameter("min_aspect_ratio").as_double()};
        const cv::Mat filtered_mask = rcj_loc::vision::debug::filterComponentsByStats(candidate_mask, filter);
        const cv::Mat overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, filtered_mask, cv::Scalar(255, 255, 255));

        candidate_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", raw_candidate_mask).toImageMsg());
        mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", filtered_mask).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        cv::imshow("Adaptive Original", frame);
        cv::imshow("Adaptive Enhanced", preprocessed.enhanced);
        cv::imshow("Adaptive Candidate Mask", raw_candidate_mask);
        cv::imshow("Adaptive Mask", filtered_mask);
        cv::imshow("Adaptive Overlay", overlay);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineAdaptiveThresholdNode>());
    rclcpp::shutdown();
    return 0;
}
