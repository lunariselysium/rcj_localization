#include <algorithm>
#include <cmath>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

class WhiteLineProbabilityNode : public rclcpp::Node {
public:
    WhiteLineProbabilityNode() : Node("white_line_probability_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);
        this->declare_parameter("white_l_mean", 220.0);
        this->declare_parameter("white_l_sigma", 30.0);
        this->declare_parameter("white_a_mean", 128.0);
        this->declare_parameter("white_a_sigma", 10.0);
        this->declare_parameter("white_b_mean", 128.0);
        this->declare_parameter("white_b_sigma", 10.0);
        this->declare_parameter("probability_threshold", 0.35);
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
            std::bind(&WhiteLineProbabilityNode::imageCallback, this, std::placeholders::_1));

        mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/mask", 10);
        probability_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/probability_image", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

        cv::namedWindow("Probability Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Probability Enhanced", cv::WINDOW_NORMAL);
        cv::namedWindow("Probability Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Probability Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Probability Overlay", cv::WINDOW_NORMAL);

        RCLCPP_INFO(this->get_logger(), "white_line_probability_node started.");
    }

    ~WhiteLineProbabilityNode() override {
        cv::destroyWindow("Probability Original");
        cv::destroyWindow("Probability Enhanced");
        cv::destroyWindow("Probability Map");
        cv::destroyWindow("Probability Mask");
        cv::destroyWindow("Probability Overlay");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr probability_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    static double gaussianProbability(double value, double mean, double sigma) {
        const double safe_sigma = std::max(sigma, 1e-6);
        const double delta = (value - mean) / safe_sigma;
        return std::exp(-0.5 * delta * delta);
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

        cv::Mat probability_map(frame.size(), CV_8UC1, cv::Scalar(0));
        for (int row = 0; row < frame.rows; ++row) {
            const uchar *enhanced_ptr = preprocessed.enhanced.ptr<uchar>(row);
            const uchar *a_ptr = preprocessed.lab_a.ptr<uchar>(row);
            const uchar *b_ptr = preprocessed.lab_b.ptr<uchar>(row);
            uchar *probability_ptr = probability_map.ptr<uchar>(row);

            for (int col = 0; col < frame.cols; ++col) {
                const double l_prob = gaussianProbability(
                    static_cast<double>(enhanced_ptr[col]),
                    this->get_parameter("white_l_mean").as_double(),
                    this->get_parameter("white_l_sigma").as_double());
                const double a_prob = gaussianProbability(
                    static_cast<double>(a_ptr[col]),
                    this->get_parameter("white_a_mean").as_double(),
                    this->get_parameter("white_a_sigma").as_double());
                const double b_prob = gaussianProbability(
                    static_cast<double>(b_ptr[col]),
                    this->get_parameter("white_b_mean").as_double(),
                    this->get_parameter("white_b_sigma").as_double());

                const double probability = l_prob * a_prob * b_prob;
                const double probability_byte = std::max(0.0, std::min(probability * 255.0, 255.0));
                probability_ptr[col] = static_cast<uchar>(probability_byte);
            }
        }

        cv::Mat mask;
        const double threshold_ratio = std::max(
            0.0,
            std::min(this->get_parameter("probability_threshold").as_double(), 1.0));
        const double probability_threshold = threshold_ratio * 255.0;
        cv::threshold(probability_map, mask, probability_threshold, 255, cv::THRESH_BINARY);

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
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_element);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_element);

        const rcj_loc::vision::debug::ComponentStatsFilter filter{
            static_cast<int>(this->get_parameter("min_area").as_int()),
            static_cast<int>(this->get_parameter("max_area").as_int()),
            static_cast<int>(this->get_parameter("min_major_axis").as_int()),
            static_cast<int>(this->get_parameter("max_minor_axis").as_int()),
            this->get_parameter("min_aspect_ratio").as_double()};
        const cv::Mat filtered_mask = rcj_loc::vision::debug::filterComponentsByStats(mask, filter);
        const cv::Mat overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, filtered_mask, cv::Scalar(255, 255, 255));

        mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", filtered_mask).toImageMsg());
        probability_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", probability_map).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        cv::imshow("Probability Original", frame);
        cv::imshow("Probability Enhanced", preprocessed.enhanced);
        cv::imshow("Probability Map", probability_map);
        cv::imshow("Probability Mask", filtered_mask);
        cv::imshow("Probability Overlay", overlay);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineProbabilityNode>());
    rclcpp::shutdown();
    return 0;
}
