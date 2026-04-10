#include <algorithm>
#include <cmath>
#include <array>
#include <limits>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

class WhiteLineClusterNode : public rclcpp::Node {
public:
    WhiteLineClusterNode() : Node("white_line_cluster_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);
        this->declare_parameter("cluster_count", 4);
        this->declare_parameter("cluster_attempts", 3);
        this->declare_parameter("cluster_iterations", 15);
        this->declare_parameter("cluster_epsilon", 1.0);

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLineClusterNode::imageCallback, this, std::placeholders::_1));

        white_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/white_mask", 10);
        green_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/green_mask", 10);
        black_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/black_mask", 10);
        noise_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/noise_mask", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

        cv::namedWindow("Cluster Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Enhanced", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster White Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Green Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Black Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Noise Mask", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Overlay", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Green Overlay", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Black Overlay", cv::WINDOW_NORMAL);
        cv::namedWindow("Cluster Noise Overlay", cv::WINDOW_NORMAL);

        RCLCPP_INFO(this->get_logger(), "white_line_cluster_node started.");
    }

    ~WhiteLineClusterNode() override {
        cv::destroyWindow("Cluster Original");
        cv::destroyWindow("Cluster Enhanced");
        cv::destroyWindow("Cluster White Mask");
        cv::destroyWindow("Cluster Green Mask");
        cv::destroyWindow("Cluster Black Mask");
        cv::destroyWindow("Cluster Noise Mask");
        cv::destroyWindow("Cluster Overlay");
        cv::destroyWindow("Cluster Green Overlay");
        cv::destroyWindow("Cluster Black Overlay");
        cv::destroyWindow("Cluster Noise Overlay");
    }

private:
    struct ClusterStats {
        int label = 0;
        float brightness = 0.0f;
        float a = 0.0f;
        float b = 0.0f;
        float chroma_distance = 0.0f;
        float white_score = 0.0f;
    };

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr green_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr black_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr noise_mask_pub_;
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

        const int total_pixels = frame.rows * frame.cols;
        cv::Mat samples(total_pixels, 3, CV_32F);
        for (int row = 0; row < frame.rows; ++row) {
            const uchar *enhanced_ptr = preprocessed.enhanced.ptr<uchar>(row);
            const uchar *a_ptr = preprocessed.lab_a.ptr<uchar>(row);
            const uchar *b_ptr = preprocessed.lab_b.ptr<uchar>(row);

            for (int col = 0; col < frame.cols; ++col) {
                const int idx = row * frame.cols + col;
                samples.at<float>(idx, 0) = static_cast<float>(enhanced_ptr[col]);
                samples.at<float>(idx, 1) = static_cast<float>(a_ptr[col]);
                samples.at<float>(idx, 2) = static_cast<float>(b_ptr[col]);
            }
        }

        cv::Mat labels;
        cv::Mat centers;
        const int cluster_count = std::max(4, static_cast<int>(this->get_parameter("cluster_count").as_int()));
        const cv::TermCriteria criteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
            static_cast<int>(this->get_parameter("cluster_iterations").as_int()),
            this->get_parameter("cluster_epsilon").as_double());
        cv::kmeans(
            samples,
            cluster_count,
            labels,
            criteria,
            static_cast<int>(this->get_parameter("cluster_attempts").as_int()),
            cv::KMEANS_PP_CENTERS,
            centers);

        std::vector<ClusterStats> stats(cluster_count);
        for (int i = 0; i < cluster_count; ++i) {
            stats[i].label = i;
            stats[i].brightness = centers.at<float>(i, 0);
            stats[i].a = centers.at<float>(i, 1);
            stats[i].b = centers.at<float>(i, 2);
            const float da = stats[i].a - 128.0f;
            const float db = stats[i].b - 128.0f;
            stats[i].chroma_distance = std::sqrt(da * da + db * db);
            stats[i].white_score = stats[i].brightness - 1.5f * stats[i].chroma_distance;
        }

        int black_label = 0;
        float darkest = std::numeric_limits<float>::max();
        for (const auto &stat : stats) {
            if (stat.brightness < darkest) {
                darkest = stat.brightness;
                black_label = stat.label;
            }
        }

        int white_label = 0;
        float best_white_score = -std::numeric_limits<float>::max();
        for (const auto &stat : stats) {
            if (stat.label == black_label) {
                continue;
            }
            if (stat.white_score > best_white_score) {
                best_white_score = stat.white_score;
                white_label = stat.label;
            }
        }

        int green_label = 0;
        float smallest_a = std::numeric_limits<float>::max();
        for (const auto &stat : stats) {
            if (stat.label == black_label || stat.label == white_label) {
                continue;
            }
            if (stat.a < smallest_a) {
                smallest_a = stat.a;
                green_label = stat.label;
            }
        }

        cv::Mat white_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat green_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat black_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat noise_mask = cv::Mat::zeros(frame.size(), CV_8UC1);

        for (int row = 0; row < frame.rows; ++row) {
            uchar *white_ptr = white_mask.ptr<uchar>(row);
            uchar *green_ptr = green_mask.ptr<uchar>(row);
            uchar *black_ptr = black_mask.ptr<uchar>(row);
            uchar *noise_ptr = noise_mask.ptr<uchar>(row);
            for (int col = 0; col < frame.cols; ++col) {
                const int idx = row * frame.cols + col;
                const int label = labels.at<int>(idx, 0);
                if (label == white_label) {
                    white_ptr[col] = 255;
                } else if (label == green_label) {
                    green_ptr[col] = 255;
                } else if (label == black_label) {
                    black_ptr[col] = 255;
                } else {
                    noise_ptr[col] = 255;
                }
            }
        }

        const cv::Mat overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, white_mask, cv::Scalar(255, 255, 255));
        const cv::Mat green_overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, green_mask, cv::Scalar(0, 255, 0));
        const cv::Mat black_overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, black_mask, cv::Scalar(0, 0, 255));
        const cv::Mat noise_overlay =
            rcj_loc::vision::debug::createMaskOverlay(frame, noise_mask, cv::Scalar(255, 0, 255));

        white_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", white_mask).toImageMsg());
        green_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", green_mask).toImageMsg());
        black_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", black_mask).toImageMsg());
        noise_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", noise_mask).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        cv::imshow("Cluster Original", frame);
        cv::imshow("Cluster Enhanced", preprocessed.enhanced);
        cv::imshow("Cluster White Mask", white_mask);
        cv::imshow("Cluster Green Mask", green_mask);
        cv::imshow("Cluster Black Mask", black_mask);
        cv::imshow("Cluster Noise Mask", noise_mask);
        cv::imshow("Cluster Overlay", overlay);
        cv::imshow("Cluster Green Overlay", green_overlay);
        cv::imshow("Cluster Black Overlay", black_overlay);
        cv::imshow("Cluster Noise Overlay", noise_overlay);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineClusterNode>());
    rclcpp::shutdown();
    return 0;
}
