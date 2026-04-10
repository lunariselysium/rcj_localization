#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

class WhiteLineEdgeNode : public rclcpp::Node {
public:
    WhiteLineEdgeNode() : Node("white_line_edge_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);
        this->declare_parameter("canny_low", 50.0);
        this->declare_parameter("canny_high", 150.0);
        this->declare_parameter("close_kernel", 5);
        this->declare_parameter("pair_kernel", 0);
        this->declare_parameter("min_area", 20);
        this->declare_parameter("max_area", 30000);
        this->declare_parameter("min_major_axis", 8);
        this->declare_parameter("max_minor_axis", 30);
        this->declare_parameter("min_aspect_ratio", 1.0);

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLineEdgeNode::imageCallback, this, std::placeholders::_1));

        cv::namedWindow("Edge Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Edge Luminance", cv::WINDOW_NORMAL);
        cv::namedWindow("Edge Enhanced", cv::WINDOW_NORMAL);
        cv::namedWindow("Edge Canny", cv::WINDOW_NORMAL);
        cv::namedWindow("Edge Connected", cv::WINDOW_NORMAL);
        cv::namedWindow("Edge Overlay", cv::WINDOW_NORMAL);

        RCLCPP_INFO(this->get_logger(), "white_line_edge_node started.");
    }

    ~WhiteLineEdgeNode() override {
        cv::destroyWindow("Edge Original");
        cv::destroyWindow("Edge Luminance");
        cv::destroyWindow("Edge Enhanced");
        cv::destroyWindow("Edge Canny");
        cv::destroyWindow("Edge Connected");
        cv::destroyWindow("Edge Overlay");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "cv_bridge failed: %s", e.what());
            return;
        }

        const auto preprocess_params = rcj_loc::vision::white_line::getPreprocessParams(*this);
        const int close_kernel_param = static_cast<int>(this->get_parameter("close_kernel").as_int());
        const int pair_kernel_param = static_cast<int>(this->get_parameter("pair_kernel").as_int());
        const int min_area = static_cast<int>(this->get_parameter("min_area").as_int());
        const int max_area = static_cast<int>(this->get_parameter("max_area").as_int());
        const int min_major_axis = static_cast<int>(this->get_parameter("min_major_axis").as_int());
        const int max_minor_axis = static_cast<int>(this->get_parameter("max_minor_axis").as_int());
        const auto preprocessed = rcj_loc::vision::white_line::preprocessFrame(frame, preprocess_params);

        cv::Mat edges;
        cv::Canny(
            preprocessed.enhanced,
            edges,
            this->get_parameter("canny_low").as_double(),
            this->get_parameter("canny_high").as_double());

        const int close_kernel = rcj_loc::vision::debug::makeOdd(close_kernel_param, 1);
        const cv::Mat close_element =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(close_kernel, close_kernel));
        cv::Mat connected_edges;
        cv::morphologyEx(edges, connected_edges, cv::MORPH_CLOSE, close_element);

        const int pair_kernel = rcj_loc::vision::debug::makeOdd(pair_kernel_param, 1);
        if (pair_kernel > 1) {
            const cv::Mat pair_element =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(pair_kernel, pair_kernel));
            cv::morphologyEx(connected_edges, connected_edges, cv::MORPH_CLOSE, pair_element);
        }

        const rcj_loc::vision::debug::ComponentStatsFilter filter{
            min_area,
            max_area,
            min_major_axis,
            max_minor_axis,
            this->get_parameter("min_aspect_ratio").as_double()};
        const cv::Mat filtered_mask = rcj_loc::vision::debug::filterComponentsByStats(connected_edges, filter);
        const cv::Mat overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, filtered_mask, cv::Scalar(0, 255, 255));

        cv::imshow("Edge Original", frame);
        cv::imshow("Edge Luminance", preprocessed.luminance);
        cv::imshow("Edge Enhanced", preprocessed.enhanced);
        cv::imshow("Edge Canny", edges);
        cv::imshow("Edge Connected", filtered_mask);
        cv::imshow("Edge Overlay", overlay);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineEdgeNode>());
    rclcpp::shutdown();
    return 0;
}
