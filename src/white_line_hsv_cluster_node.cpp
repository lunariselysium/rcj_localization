#include <algorithm>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

namespace {

constexpr int kHueMax = 179;
constexpr int kByteMax = 255;
constexpr char kControlsWindow[] = "HSV Cluster Controls";

int clampToByte(int value) {
    return std::clamp(value, 0, kByteMax);
}

int clampHue(int value) {
    return std::clamp(value, 0, kHueMax);
}

cv::Mat takeFromRemaining(const cv::Mat &candidate, cv::Mat &remaining) {
    cv::Mat assigned;
    cv::bitwise_and(candidate, remaining, assigned);

    cv::Mat assigned_inv;
    cv::bitwise_not(assigned, assigned_inv);
    cv::bitwise_and(remaining, assigned_inv, remaining);
    return assigned;
}

}  // namespace

class WhiteLineHsvClusterNode : public rclcpp::Node {
public:
    WhiteLineHsvClusterNode() : Node("white_line_hsv_cluster_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);

        this->declare_parameter("black_v_max", 70);
        this->declare_parameter("green_h_min", 35);
        this->declare_parameter("green_h_max", 95);
        this->declare_parameter("green_s_min", 40);
        this->declare_parameter("green_v_min", 40);
        this->declare_parameter("white_s_max", 60);
        this->declare_parameter("white_v_min", 170);

        loadThresholdsFromParameters();

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLineHsvClusterNode::imageCallback, this, std::placeholders::_1));

        white_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/white_mask", 10);
        green_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/green_mask", 10);
        black_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/black_mask", 10);
        noise_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/noise_mask", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

        cv::namedWindow("HSV Cluster Original", cv::WINDOW_NORMAL);
        cv::namedWindow("HSV Cluster Enhanced", cv::WINDOW_NORMAL);
        cv::namedWindow("HSV Cluster White Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("HSV Cluster Green Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("HSV Cluster Black Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("HSV Cluster Noise Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("HSV Cluster Overlay", cv::WINDOW_NORMAL);
        cv::namedWindow(kControlsWindow, cv::WINDOW_AUTOSIZE);

        cv::createTrackbar("black_v_max", kControlsWindow, &black_v_max_, kByteMax);
        cv::createTrackbar("green_h_min", kControlsWindow, &green_h_min_, kHueMax);
        cv::createTrackbar("green_h_max", kControlsWindow, &green_h_max_, kHueMax);
        cv::createTrackbar("green_s_min", kControlsWindow, &green_s_min_, kByteMax);
        cv::createTrackbar("green_v_min", kControlsWindow, &green_v_min_, kByteMax);
        cv::createTrackbar("white_s_max", kControlsWindow, &white_s_max_, kByteMax);
        cv::createTrackbar("white_v_min", kControlsWindow, &white_v_min_, kByteMax);

        RCLCPP_INFO(this->get_logger(), "white_line_hsv_cluster_node started. Use the '%s' window to tune HSV thresholds and press 'p' to print the current values.", kControlsWindow);
    }

    ~WhiteLineHsvClusterNode() override {
        cv::destroyWindow("HSV Cluster Original");
        cv::destroyWindow("HSV Cluster Enhanced");
        cv::destroyWindow("HSV Cluster White Mask");
        cv::destroyWindow("HSV Cluster Green Mask");
        cv::destroyWindow("HSV Cluster Black Mask");
        cv::destroyWindow("HSV Cluster Noise Mask");
        cv::destroyWindow("HSV Cluster Overlay");
        cv::destroyWindow(kControlsWindow);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr green_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr black_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr noise_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    int black_v_max_;
    int green_h_min_;
    int green_h_max_;
    int green_s_min_;
    int green_v_min_;
    int white_s_max_;
    int white_v_min_;

    void loadThresholdsFromParameters() {
        black_v_max_ = clampToByte(static_cast<int>(this->get_parameter("black_v_max").as_int()));
        green_h_min_ = clampHue(static_cast<int>(this->get_parameter("green_h_min").as_int()));
        green_h_max_ = clampHue(static_cast<int>(this->get_parameter("green_h_max").as_int()));
        green_s_min_ = clampToByte(static_cast<int>(this->get_parameter("green_s_min").as_int()));
        green_v_min_ = clampToByte(static_cast<int>(this->get_parameter("green_v_min").as_int()));
        white_s_max_ = clampToByte(static_cast<int>(this->get_parameter("white_s_max").as_int()));
        white_v_min_ = clampToByte(static_cast<int>(this->get_parameter("white_v_min").as_int()));
    }

    void readThresholdsFromTrackbars() {
        black_v_max_ = cv::getTrackbarPos("black_v_max", kControlsWindow);
        green_h_min_ = cv::getTrackbarPos("green_h_min", kControlsWindow);
        green_h_max_ = cv::getTrackbarPos("green_h_max", kControlsWindow);
        green_s_min_ = cv::getTrackbarPos("green_s_min", kControlsWindow);
        green_v_min_ = cv::getTrackbarPos("green_v_min", kControlsWindow);
        white_s_max_ = cv::getTrackbarPos("white_s_max", kControlsWindow);
        white_v_min_ = cv::getTrackbarPos("white_v_min", kControlsWindow);
    }

    void logCurrentThresholds() const {
        RCLCPP_INFO(
            this->get_logger(),
            "Current HSV thresholds: black_v_max=%d, green_h_min=%d, green_h_max=%d, green_s_min=%d, green_v_min=%d, white_s_max=%d, white_v_min=%d",
            black_v_max_,
            green_h_min_,
            green_h_max_,
            green_s_min_,
            green_v_min_,
            white_s_max_,
            white_v_min_);
    }

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

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        readThresholdsFromTrackbars();

        cv::Mat black_candidate;
        cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(kHueMax, kByteMax, black_v_max_), black_candidate);

        cv::Mat green_candidate = cv::Mat::zeros(frame.size(), CV_8UC1);
        if (green_h_min_ <= green_h_max_) {
            cv::inRange(
                hsv,
                cv::Scalar(green_h_min_, green_s_min_, green_v_min_),
                cv::Scalar(green_h_max_, kByteMax, kByteMax),
                green_candidate);
        } else {
            cv::Mat green_low;
            cv::Mat green_high;
            cv::inRange(
                hsv,
                cv::Scalar(0, green_s_min_, green_v_min_),
                cv::Scalar(green_h_max_, kByteMax, kByteMax),
                green_low);
            cv::inRange(
                hsv,
                cv::Scalar(green_h_min_, green_s_min_, green_v_min_),
                cv::Scalar(kHueMax, kByteMax, kByteMax),
                green_high);
            cv::bitwise_or(green_low, green_high, green_candidate);
        }

        cv::Mat white_candidate;
        cv::inRange(
            hsv,
            cv::Scalar(0, 0, white_v_min_),
            cv::Scalar(kHueMax, white_s_max_, kByteMax),
            white_candidate);

        cv::Mat remaining(frame.size(), CV_8UC1, cv::Scalar(255));

        // 只主动筛绿色、白色、黑色，最后剩余像素统一归到 noise。
        const cv::Mat green_mask = takeFromRemaining(green_candidate, remaining);
        const cv::Mat white_mask = takeFromRemaining(white_candidate, remaining);
        const cv::Mat black_mask = takeFromRemaining(black_candidate, remaining);
        const cv::Mat noise_mask = remaining.clone();

        cv::Mat overlay = frame.clone();
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, black_mask, cv::Scalar(0, 0, 255));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, green_mask, cv::Scalar(0, 255, 0));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, noise_mask, cv::Scalar(255, 0, 255));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, white_mask, cv::Scalar(255, 255, 255));

        white_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", white_mask).toImageMsg());
        green_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", green_mask).toImageMsg());
        black_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", black_mask).toImageMsg());
        noise_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", noise_mask).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        cv::imshow("HSV Cluster Original", frame);
        cv::imshow("HSV Cluster Enhanced", preprocessed.enhanced);
        cv::imshow("HSV Cluster White Mask", white_mask);
        cv::imshow("HSV Cluster Green Mask", green_mask);
        cv::imshow("HSV Cluster Black Mask", black_mask);
        cv::imshow("HSV Cluster Noise Mask", noise_mask);
        cv::imshow("HSV Cluster Overlay", overlay);

        const int key = cv::waitKey(1);
        if (key == 'p' || key == 'P') {
            logCurrentThresholds();
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineHsvClusterNode>());
    rclcpp::shutdown();
    return 0;
}
