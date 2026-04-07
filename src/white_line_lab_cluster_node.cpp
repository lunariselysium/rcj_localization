#include <algorithm>
#include <filesystem>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

namespace {

constexpr int kByteMax = 255;
constexpr char kControlsWindow[] = "Lab Cluster Controls";

int clampToByte(int value) {
    return std::clamp(value, 0, kByteMax);
}

cv::Size fitWithinBounds(const cv::Size &image_size, int max_width, int max_height) {
    const int safe_max_width = std::max(1, max_width);
    const int safe_max_height = std::max(1, max_height);
    if (image_size.width <= 0 || image_size.height <= 0) {
        return cv::Size(safe_max_width, safe_max_height);
    }

    const double width_scale =
        static_cast<double>(safe_max_width) / static_cast<double>(image_size.width);
    const double height_scale =
        static_cast<double>(safe_max_height) / static_cast<double>(image_size.height);
    const double scale = std::min(1.0, std::min(width_scale, height_scale));

    return cv::Size(
        std::max(1, static_cast<int>(std::round(image_size.width * scale))),
        std::max(1, static_cast<int>(std::round(image_size.height * scale))));
}

void resizeWindowToFitImage(
    const std::string &window_name,
    const cv::Mat &image,
    int max_width,
    int max_height) {
    const cv::Size fitted_size = fitWithinBounds(image.size(), max_width, max_height);
    cv::resizeWindow(window_name, fitted_size.width, fitted_size.height);
}

cv::Mat takeFromRemaining(const cv::Mat &candidate, cv::Mat &remaining) {
    cv::Mat assigned;
    cv::bitwise_and(candidate, remaining, assigned);

    cv::Mat assigned_inv;
    cv::bitwise_not(assigned, assigned_inv);
    cv::bitwise_and(remaining, assigned_inv, remaining);
    return assigned;
}

cv::Mat makeWhiteCandidate(
    const cv::Mat &enhanced,
    const cv::Mat &lab_a,
    const cv::Mat &lab_b,
    int enhanced_min,
    int a_center,
    int b_center,
    int a_tol,
    int b_tol) {
    cv::Mat bright_mask;
    cv::inRange(enhanced, cv::Scalar(enhanced_min), cv::Scalar(kByteMax), bright_mask);

    cv::Mat a_delta;
    cv::Mat b_delta;
    cv::absdiff(lab_a, cv::Scalar(a_center), a_delta);
    cv::absdiff(lab_b, cv::Scalar(b_center), b_delta);

    cv::Mat a_mask;
    cv::Mat b_mask;
    cv::inRange(a_delta, cv::Scalar(0), cv::Scalar(a_tol), a_mask);
    cv::inRange(b_delta, cv::Scalar(0), cv::Scalar(b_tol), b_mask);

    cv::Mat neutral_mask;
    cv::bitwise_and(a_mask, b_mask, neutral_mask);

    cv::Mat white_candidate;
    cv::bitwise_and(bright_mask, neutral_mask, white_candidate);
    return white_candidate;
}

cv::Mat filterByGreenAdjacency(
    const cv::Mat &white_mask,
    const cv::Mat &green_neighbor_mask,
    int min_overlap_pixels) {
    CV_Assert(white_mask.type() == CV_8UC1);
    CV_Assert(green_neighbor_mask.type() == CV_8UC1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(white_mask, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered = cv::Mat::zeros(white_mask.size(), CV_8UC1);
    for (int label = 1; label < stats.rows; ++label) {
        const cv::Mat component_mask = labels == label;
        cv::Mat overlap_mask;
        cv::bitwise_and(component_mask, green_neighbor_mask, overlap_mask);
        if (cv::countNonZero(overlap_mask) < min_overlap_pixels) {
            continue;
        }

        filtered.setTo(255, component_mask);
    }

    return filtered;
}

struct GeometryFilterDebugResult {
    cv::Mat area_mask;
    cv::Mat axis_mask;
    cv::Mat final_mask;
};

GeometryFilterDebugResult filterComponentsByStatsDebug(
    const cv::Mat &binary,
    const rcj_loc::vision::debug::ComponentStatsFilter &filter) {
    CV_Assert(binary.type() == CV_8UC1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    GeometryFilterDebugResult result{
        cv::Mat::zeros(binary.size(), CV_8UC1),
        cv::Mat::zeros(binary.size(), CV_8UC1),
        cv::Mat::zeros(binary.size(), CV_8UC1)};

    for (int label = 1; label < stats.rows; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int major_axis = std::max(width, height);
        const int minor_axis = std::min(width, height);
        const double aspect_ratio = static_cast<double>(major_axis) / std::max(1, minor_axis);
        const cv::Mat component_mask = labels == label;

        if (area < filter.min_area || area > filter.max_area) {
            continue;
        }
        result.area_mask.setTo(255, component_mask);

        if (major_axis < filter.min_major_axis || minor_axis > filter.max_minor_axis) {
            continue;
        }
        result.axis_mask.setTo(255, component_mask);

        if (aspect_ratio < filter.min_aspect_ratio) {
            continue;
        }
        result.final_mask.setTo(255, component_mask);
    }

    return result;
}

}  // namespace

class WhiteLineLabClusterNode : public rclcpp::Node {
public:
    WhiteLineLabClusterNode() : Node("white_line_lab_cluster_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);

        this->declare_parameter("white_enhanced_min", 170);
        this->declare_parameter("white_a_center", 128);
        this->declare_parameter("white_b_center", 128);
        this->declare_parameter("white_a_tol", 14);
        this->declare_parameter("white_b_tol", 14);
        this->declare_parameter("green_a_max", 120);
        this->declare_parameter("green_b_min", 83);
        this->declare_parameter("green_b_max", 170);
        this->declare_parameter("black_enhanced_max", 123);
        this->declare_parameter("white_open_kernel", 4);
        this->declare_parameter("white_close_kernel", 5);
        this->declare_parameter("sideline_min_area", 1000);
        this->declare_parameter("sideline_max_area", 5000);
        this->declare_parameter("sideline_min_major_axis", 25);
        this->declare_parameter("sideline_max_minor_axis", 20);
        this->declare_parameter("sideline_min_aspect_ratio", 2.0);
        this->declare_parameter("green_neighbor_kernel", 11);
        this->declare_parameter("green_neighbor_min_pixels", 15);
        this->declare_parameter("display_max_width", 960);
        this->declare_parameter("display_max_height", 720);
        this->declare_parameter<std::string>(
            "area_debug_save_dir",
            "/home/terry/RCJ/localization_ws/src/rcj_localization/debugImage");

        loadThresholdsFromParameters();

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLineLabClusterNode::imageCallback, this, std::placeholders::_1));

        white_candidate_mask_pub_ =
            this->create_publisher<sensor_msgs::msg::Image>("~/white_candidate_mask", 10);
        white_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/white_mask", 10);
        green_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/green_mask", 10);
        black_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/black_mask", 10);
        noise_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/noise_mask", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

        // Disabled to reduce VM display load: original camera frame view.
        // cv::namedWindow("Lab Cluster Original", cv::WINDOW_NORMAL);
        // Disabled to reduce VM display load: filtered BGR preprocessing view.
        // cv::namedWindow("Lab Cluster Filtered", cv::WINDOW_NORMAL);
        // Disabled to reduce VM display load: enhanced luminance view.
        // cv::namedWindow("Lab Cluster Enhanced", cv::WINDOW_NORMAL);
        cv::namedWindow("Lab Cluster White Candidate", cv::WINDOW_NORMAL);
        cv::namedWindow("Lab Cluster White Morph", cv::WINDOW_NORMAL);
        cv::namedWindow("Lab Cluster White Area", cv::WINDOW_NORMAL);
        cv::namedWindow("Lab Cluster White Axis", cv::WINDOW_NORMAL);
        cv::namedWindow("Lab Cluster White Geom", cv::WINDOW_NORMAL);
        cv::namedWindow("Lab Cluster White Final", cv::WINDOW_NORMAL);
        // Disabled to reduce VM display load: green-class mask view.
        // cv::namedWindow("Lab Cluster Green Mask", cv::WINDOW_NORMAL);
        // Disabled to reduce VM display load: black-class mask view.
        // cv::namedWindow("Lab Cluster Black Mask", cv::WINDOW_NORMAL);
        // Disabled to reduce VM display load: noise-class mask view.
        // cv::namedWindow("Lab Cluster Noise Mask", cv::WINDOW_NORMAL);
        // Disabled to reduce VM display load: final overlay view on the filtered image.
        // cv::namedWindow("Lab Cluster Overlay", cv::WINDOW_NORMAL);
        cv::namedWindow(kControlsWindow, cv::WINDOW_AUTOSIZE);

        cv::createTrackbar("white_enh_min", kControlsWindow, &white_enhanced_min_, kByteMax);
        cv::createTrackbar("white_a_center", kControlsWindow, &white_a_center_, kByteMax);
        cv::createTrackbar("white_b_center", kControlsWindow, &white_b_center_, kByteMax);
        cv::createTrackbar("white_a_tol", kControlsWindow, &white_a_tol_, kByteMax);
        cv::createTrackbar("white_b_tol", kControlsWindow, &white_b_tol_, kByteMax);
        cv::createTrackbar("green_a_max", kControlsWindow, &green_a_max_, kByteMax);
        cv::createTrackbar("green_b_min", kControlsWindow, &green_b_min_, kByteMax);
        cv::createTrackbar("green_b_max", kControlsWindow, &green_b_max_, kByteMax);
        cv::createTrackbar("black_enh_max", kControlsWindow, &black_enhanced_max_, kByteMax);

        RCLCPP_INFO(
            this->get_logger(),
            "white_line_lab_cluster_node started. Tune '%s' and press 'p' to print thresholds.",
            kControlsWindow);
    }

    ~WhiteLineLabClusterNode() override {
        // Disabled to reduce VM display load: original camera frame view.
        // cv::destroyWindow("Lab Cluster Original");
        // Disabled to reduce VM display load: filtered BGR preprocessing view.
        // cv::destroyWindow("Lab Cluster Filtered");
        // Disabled to reduce VM display load: enhanced luminance view.
        // cv::destroyWindow("Lab Cluster Enhanced");
        cv::destroyWindow("Lab Cluster White Candidate");
        cv::destroyWindow("Lab Cluster White Morph");
        cv::destroyWindow("Lab Cluster White Area");
        cv::destroyWindow("Lab Cluster White Axis");
        cv::destroyWindow("Lab Cluster White Geom");
        cv::destroyWindow("Lab Cluster White Final");
        // Disabled to reduce VM display load: green-class mask view.
        // cv::destroyWindow("Lab Cluster Green Mask");
        // Disabled to reduce VM display load: black-class mask view.
        // cv::destroyWindow("Lab Cluster Black Mask");
        // Disabled to reduce VM display load: noise-class mask view.
        // cv::destroyWindow("Lab Cluster Noise Mask");
        // Disabled to reduce VM display load: final overlay view on the filtered image.
        // cv::destroyWindow("Lab Cluster Overlay");
        cv::destroyWindow(kControlsWindow);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_candidate_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr green_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr black_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr noise_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

    int white_enhanced_min_;
    int white_a_center_;
    int white_b_center_;
    int white_a_tol_;
    int white_b_tol_;
    int green_a_max_;
    int green_b_min_;
    int green_b_max_;
    int black_enhanced_max_;
    int white_open_kernel_;
    int white_close_kernel_;
    int sideline_min_area_;
    int sideline_max_area_;
    int sideline_min_major_axis_;
    int sideline_max_minor_axis_;
    double sideline_min_aspect_ratio_;
    int green_neighbor_kernel_;
    int green_neighbor_min_pixels_;
    int display_max_width_;
    int display_max_height_;
    std::string area_debug_save_dir_;

    void loadThresholdsFromParameters() {
        white_enhanced_min_ = clampToByte(static_cast<int>(this->get_parameter("white_enhanced_min").as_int()));
        white_a_center_ = clampToByte(static_cast<int>(this->get_parameter("white_a_center").as_int()));
        white_b_center_ = clampToByte(static_cast<int>(this->get_parameter("white_b_center").as_int()));
        white_a_tol_ = clampToByte(static_cast<int>(this->get_parameter("white_a_tol").as_int()));
        white_b_tol_ = clampToByte(static_cast<int>(this->get_parameter("white_b_tol").as_int()));
        green_a_max_ = clampToByte(static_cast<int>(this->get_parameter("green_a_max").as_int()));
        green_b_min_ = clampToByte(static_cast<int>(this->get_parameter("green_b_min").as_int()));
        green_b_max_ = clampToByte(static_cast<int>(this->get_parameter("green_b_max").as_int()));
        black_enhanced_max_ = clampToByte(static_cast<int>(this->get_parameter("black_enhanced_max").as_int()));
        white_open_kernel_ = static_cast<int>(this->get_parameter("white_open_kernel").as_int());
        white_close_kernel_ = static_cast<int>(this->get_parameter("white_close_kernel").as_int());
        sideline_min_area_ = static_cast<int>(this->get_parameter("sideline_min_area").as_int());
        sideline_max_area_ = static_cast<int>(this->get_parameter("sideline_max_area").as_int());
        sideline_min_major_axis_ = static_cast<int>(this->get_parameter("sideline_min_major_axis").as_int());
        sideline_max_minor_axis_ = static_cast<int>(this->get_parameter("sideline_max_minor_axis").as_int());
        sideline_min_aspect_ratio_ = this->get_parameter("sideline_min_aspect_ratio").as_double();
        green_neighbor_kernel_ = static_cast<int>(this->get_parameter("green_neighbor_kernel").as_int());
        green_neighbor_min_pixels_ = static_cast<int>(this->get_parameter("green_neighbor_min_pixels").as_int());
        display_max_width_ = static_cast<int>(this->get_parameter("display_max_width").as_int());
        display_max_height_ = static_cast<int>(this->get_parameter("display_max_height").as_int());
        area_debug_save_dir_ = this->get_parameter("area_debug_save_dir").as_string();
    }

    void saveAreaMaskAsPng(const cv::Mat &white_area_mask) {
        try {
            std::filesystem::create_directories(area_debug_save_dir_);
        } catch (const std::exception &e) {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to create area debug directory '%s': %s",
                area_debug_save_dir_.c_str(),
                e.what());
            return;
        }

        const auto stamp_ns = this->get_clock()->now().nanoseconds();
        const std::string output_path =
            area_debug_save_dir_ + "/white_area_" + std::to_string(stamp_ns) + ".png";

        if (!cv::imwrite(output_path, white_area_mask)) {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to save white area debug image to '%s'",
                output_path.c_str());
            return;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Saved white area debug image to '%s'",
            output_path.c_str());
    }

    void readThresholdsFromTrackbars() {
        white_enhanced_min_ = cv::getTrackbarPos("white_enh_min", kControlsWindow);
        white_a_center_ = cv::getTrackbarPos("white_a_center", kControlsWindow);
        white_b_center_ = cv::getTrackbarPos("white_b_center", kControlsWindow);
        white_a_tol_ = cv::getTrackbarPos("white_a_tol", kControlsWindow);
        white_b_tol_ = cv::getTrackbarPos("white_b_tol", kControlsWindow);
        green_a_max_ = cv::getTrackbarPos("green_a_max", kControlsWindow);
        green_b_min_ = cv::getTrackbarPos("green_b_min", kControlsWindow);
        green_b_max_ = cv::getTrackbarPos("green_b_max", kControlsWindow);
        black_enhanced_max_ = cv::getTrackbarPos("black_enh_max", kControlsWindow);
    }

    void logCurrentThresholds() const {
        RCLCPP_INFO(
            this->get_logger(),
            "Current Lab thresholds: white_enhanced_min=%d, white_a_center=%d, white_b_center=%d, white_a_tol=%d, white_b_tol=%d, green_a_max=%d, green_b_min=%d, green_b_max=%d, black_enhanced_max=%d",
            white_enhanced_min_,
            white_a_center_,
            white_b_center_,
            white_a_tol_,
            white_b_tol_,
            green_a_max_,
            green_b_min_,
            green_b_max_,
            black_enhanced_max_);
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

        readThresholdsFromTrackbars();

        cv::Mat green_candidate;
        const int green_b_low = std::min(green_b_min_, green_b_max_);
        const int green_b_high = std::max(green_b_min_, green_b_max_);
        cv::Mat green_a_mask;
        cv::Mat green_b_mask;
        cv::inRange(preprocessed.lab_a, cv::Scalar(0), cv::Scalar(green_a_max_), green_a_mask);
        cv::inRange(preprocessed.lab_b, cv::Scalar(green_b_low), cv::Scalar(green_b_high), green_b_mask);
        cv::bitwise_and(green_a_mask, green_b_mask, green_candidate);

        const cv::Mat white_candidate = makeWhiteCandidate(
            preprocessed.enhanced,
            preprocessed.lab_a,
            preprocessed.lab_b,
            white_enhanced_min_,
            white_a_center_,
            white_b_center_,
            white_a_tol_,
            white_b_tol_);

        cv::Mat black_candidate;
        cv::inRange(
            preprocessed.enhanced,
            cv::Scalar(0),
            cv::Scalar(black_enhanced_max_),
            black_candidate);

        cv::Mat remaining(frame.size(), CV_8UC1, cv::Scalar(255));
        const cv::Mat green_mask = takeFromRemaining(green_candidate, remaining);
        const cv::Mat raw_white_mask = takeFromRemaining(white_candidate, remaining);
        const cv::Mat black_mask = takeFromRemaining(black_candidate, remaining);
        const cv::Mat noise_mask = remaining;

        const int white_open_kernel =
            rcj_loc::vision::debug::makeOdd(white_open_kernel_, 1);
        const int white_close_kernel =
            rcj_loc::vision::debug::makeOdd(white_close_kernel_, 1);

        cv::Mat white_morph_mask = raw_white_mask.clone();
        const cv::Mat open_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(white_open_kernel, white_open_kernel));
        const cv::Mat close_element =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(white_close_kernel, white_close_kernel));
        cv::morphologyEx(white_morph_mask, white_morph_mask, cv::MORPH_OPEN, open_element);
        cv::morphologyEx(white_morph_mask, white_morph_mask, cv::MORPH_CLOSE, close_element);

        const rcj_loc::vision::debug::ComponentStatsFilter sideline_filter{
            sideline_min_area_,
            sideline_max_area_,
            sideline_min_major_axis_,
            sideline_max_minor_axis_,
            sideline_min_aspect_ratio_};
        const GeometryFilterDebugResult geom_debug =
            filterComponentsByStatsDebug(white_morph_mask, sideline_filter);
        const cv::Mat &white_area_mask = geom_debug.area_mask;
        const cv::Mat &white_axis_mask = geom_debug.axis_mask;
        const cv::Mat &white_geom_mask = geom_debug.final_mask;

        const int green_neighbor_kernel =
            rcj_loc::vision::debug::makeOdd(green_neighbor_kernel_, 1);
        const cv::Mat green_neighbor_element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(green_neighbor_kernel, green_neighbor_kernel));
        cv::Mat green_neighbor_mask;
        cv::dilate(green_mask, green_neighbor_mask, green_neighbor_element);

        const cv::Mat sideline_mask =
            filterByGreenAdjacency(white_geom_mask, green_neighbor_mask, green_neighbor_min_pixels_);

        cv::Mat overlay = preprocessed.filtered_bgr.clone();
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, black_mask, cv::Scalar(0, 0, 255));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, green_mask, cv::Scalar(0, 255, 0));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, noise_mask, cv::Scalar(255, 0, 255));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, sideline_mask, cv::Scalar(255, 255, 255));

        white_candidate_mask_pub_->publish(
            *cv_bridge::CvImage(msg->header, "mono8", raw_white_mask).toImageMsg());
        white_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", sideline_mask).toImageMsg());
        green_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", green_mask).toImageMsg());
        black_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", black_mask).toImageMsg());
        noise_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", noise_mask).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        // Disabled to reduce VM display load: original camera frame view.
        // cv::imshow("Lab Cluster Original", frame);
        // Disabled to reduce VM display load: filtered BGR preprocessing view.
        // cv::imshow("Lab Cluster Filtered", preprocessed.filtered_bgr);
        // Disabled to reduce VM display load: enhanced luminance view.
        // cv::imshow("Lab Cluster Enhanced", preprocessed.enhanced);
        cv::imshow("Lab Cluster White Candidate", raw_white_mask);
        cv::imshow("Lab Cluster White Morph", white_morph_mask);
        cv::imshow("Lab Cluster White Area", white_area_mask);
        cv::imshow("Lab Cluster White Axis", white_axis_mask);
        cv::imshow("Lab Cluster White Geom", white_geom_mask);
        cv::imshow("Lab Cluster White Final", sideline_mask);
        // Disabled to reduce VM display load: green-class mask view.
        // cv::imshow("Lab Cluster Green Mask", green_mask);
        // Disabled to reduce VM display load: black-class mask view.
        // cv::imshow("Lab Cluster Black Mask", black_mask);
        // Disabled to reduce VM display load: noise-class mask view.
        // cv::imshow("Lab Cluster Noise Mask", noise_mask);
        // Disabled to reduce VM display load: final overlay view on the filtered image.
        // cv::imshow("Lab Cluster Overlay", overlay);

        // Disabled to reduce VM display load: original camera frame view.
        // resizeWindowToFitImage("Lab Cluster Original", frame, display_max_width_, display_max_height_);
        // Disabled to reduce VM display load: filtered BGR preprocessing view.
        // resizeWindowToFitImage(
        //     "Lab Cluster Filtered",
        //     preprocessed.filtered_bgr,
        //     display_max_width_,
        //     display_max_height_);
        // Disabled to reduce VM display load: enhanced luminance view.
        // resizeWindowToFitImage(
        //     "Lab Cluster Enhanced",
        //     preprocessed.enhanced,
        //     display_max_width_,
        //     display_max_height_);
        resizeWindowToFitImage(
            "Lab Cluster White Candidate",
            raw_white_mask,
            display_max_width_,
            display_max_height_);
        resizeWindowToFitImage(
            "Lab Cluster White Morph",
            white_morph_mask,
            display_max_width_,
            display_max_height_);
        resizeWindowToFitImage(
            "Lab Cluster White Area",
            white_area_mask,
            display_max_width_,
            display_max_height_);
        resizeWindowToFitImage(
            "Lab Cluster White Axis",
            white_axis_mask,
            display_max_width_,
            display_max_height_);
        resizeWindowToFitImage(
            "Lab Cluster White Geom",
            white_geom_mask,
            display_max_width_,
            display_max_height_);
        resizeWindowToFitImage(
            "Lab Cluster White Final",
            sideline_mask,
            display_max_width_,
            display_max_height_);
        // Disabled to reduce VM display load: green-class mask view.
        // resizeWindowToFitImage(
        //     "Lab Cluster Green Mask",
        //     green_mask,
        //     display_max_width_,
        //     display_max_height_);
        // Disabled to reduce VM display load: black-class mask view.
        // resizeWindowToFitImage(
        //     "Lab Cluster Black Mask",
        //     black_mask,
        //     display_max_width_,
        //     display_max_height_);
        // Disabled to reduce VM display load: noise-class mask view.
        // resizeWindowToFitImage(
        //     "Lab Cluster Noise Mask",
        //     noise_mask,
        //     display_max_width_,
        //     display_max_height_);
        // Disabled to reduce VM display load: final overlay view on the filtered image.
        // resizeWindowToFitImage(
        //     "Lab Cluster Overlay",
        //     overlay,
        //     display_max_width_,
        //     display_max_height_);

        const int key = cv::waitKey(1);
        if (key == 'p' || key == 'P') {
            logCurrentThresholds();
        } else if (key == 's' || key == 'S') {
            saveAreaMaskAsPng(white_area_mask);
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineLabClusterNode>());
    rclcpp::shutdown();
    return 0;
}
