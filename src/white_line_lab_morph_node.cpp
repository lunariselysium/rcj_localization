#include <algorithm>
#include <chrono>
#include <string>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#elif __has_include(<cv_bridge/cv_bridge.h>)
#include <cv_bridge/cv_bridge.h>
#else
#error "cv_bridge header not found"
#endif
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"
#include "rcj_localization/white_line_preprocessing.hpp"

namespace {

constexpr int kByteMax = 255;
constexpr char kControlsWindow[] = "Lab Morph Controls";
constexpr char kOriginalWindow[] = "Lab Morph Original";
constexpr char kWhiteCandidateWindow[] = "Lab Morph White Candidate";
constexpr char kWhiteMorphWindow[] = "Lab Morph White Morph";
constexpr char kGreenMaskWindow[] = "Lab Morph Green Mask";
constexpr char kBlackMaskWindow[] = "Lab Morph Black Mask";
constexpr char kNoiseMaskWindow[] = "Lab Morph Noise Mask";
constexpr char kOverlayWindow[] = "Lab Morph Overlay";

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

}  // namespace

class WhiteLineLabMorphNode : public rclcpp::Node {
public:
    WhiteLineLabMorphNode() : Node("white_line_lab_morph_node") {
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");
        rcj_loc::vision::white_line::declarePreprocessParameters(*this);

        this->declare_parameter("white_enhanced_min", 170);
        this->declare_parameter("white_a_center", 128);
        this->declare_parameter("white_b_center", 128);
        this->declare_parameter("white_a_tol", 14);
        this->declare_parameter("white_b_tol", 14);
        this->declare_parameter("green_a_max", 115);
        this->declare_parameter("green_b_min", 83);
        this->declare_parameter("green_b_max", 170);
        this->declare_parameter("black_enhanced_max", 123);
        this->declare_parameter("white_open_kernel", 4);
        this->declare_parameter("white_close_kernel", 5);
        this->declare_parameter("enable_image_view", false);
        this->declare_parameter("show_input_image", true);
        this->declare_parameter("show_white_candidate_mask", true);
        this->declare_parameter("show_white_morph_mask", true);
        this->declare_parameter("show_green_mask", true);
        this->declare_parameter("show_black_mask", false);
        this->declare_parameter("show_noise_mask", false);
        this->declare_parameter("show_debug_image", false);
        this->declare_parameter("display_max_width", 960);
        this->declare_parameter("display_max_height", 720);

        loadThresholdsFromParameters();

        const auto topic = this->get_parameter("input_topic").as_string();
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::SensorDataQoS(),
            std::bind(&WhiteLineLabMorphNode::imageCallback, this, std::placeholders::_1));

        white_candidate_mask_pub_ =
            this->create_publisher<sensor_msgs::msg::Image>("~/white_candidate_mask", 10);
        white_morph_mask_pub_ =
            this->create_publisher<sensor_msgs::msg::Image>("~/white_morph_mask", 10);
        green_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/green_mask", 10);
        black_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/black_mask", 10);
        noise_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/noise_mask", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

        syncImageViewState();

        RCLCPP_INFO(
            this->get_logger(),
            "white_line_lab_morph_node started. enable_image_view=%s",
            enable_image_view_ ? "true" : "false");
    }

    ~WhiteLineLabMorphNode() override {
        destroyDebugWindows();
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_candidate_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_morph_mask_pub_;
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
    bool enable_image_view_ = false;
    bool show_input_image_ = true;
    bool show_white_candidate_mask_ = true;
    bool show_white_morph_mask_ = true;
    bool show_green_mask_ = true;
    bool show_black_mask_ = false;
    bool show_noise_mask_ = false;
    bool show_debug_image_ = false;
    bool controls_window_created_ = false;
    bool original_window_created_ = false;
    bool white_candidate_window_created_ = false;
    bool white_morph_window_created_ = false;
    bool green_mask_window_created_ = false;
    bool black_mask_window_created_ = false;
    bool noise_mask_window_created_ = false;
    bool overlay_window_created_ = false;
    int display_max_width_ = 960;
    int display_max_height_ = 720;
    unsigned long long frame_count_ = 0;
    long long total_morph_time_us_ = 0;

    void loadThresholdsFromParameters() {
        white_enhanced_min_ =
            clampToByte(static_cast<int>(this->get_parameter("white_enhanced_min").as_int()));
        white_a_center_ =
            clampToByte(static_cast<int>(this->get_parameter("white_a_center").as_int()));
        white_b_center_ =
            clampToByte(static_cast<int>(this->get_parameter("white_b_center").as_int()));
        white_a_tol_ =
            clampToByte(static_cast<int>(this->get_parameter("white_a_tol").as_int()));
        white_b_tol_ =
            clampToByte(static_cast<int>(this->get_parameter("white_b_tol").as_int()));
        green_a_max_ =
            clampToByte(static_cast<int>(this->get_parameter("green_a_max").as_int()));
        green_b_min_ =
            clampToByte(static_cast<int>(this->get_parameter("green_b_min").as_int()));
        green_b_max_ =
            clampToByte(static_cast<int>(this->get_parameter("green_b_max").as_int()));
        black_enhanced_max_ =
            clampToByte(static_cast<int>(this->get_parameter("black_enhanced_max").as_int()));
        white_open_kernel_ = static_cast<int>(this->get_parameter("white_open_kernel").as_int());
        white_close_kernel_ = static_cast<int>(this->get_parameter("white_close_kernel").as_int());
        enable_image_view_ = this->get_parameter("enable_image_view").as_bool();
        show_input_image_ = this->get_parameter("show_input_image").as_bool();
        show_white_candidate_mask_ = this->get_parameter("show_white_candidate_mask").as_bool();
        show_white_morph_mask_ = this->get_parameter("show_white_morph_mask").as_bool();
        show_green_mask_ = this->get_parameter("show_green_mask").as_bool();
        show_black_mask_ = this->get_parameter("show_black_mask").as_bool();
        show_noise_mask_ = this->get_parameter("show_noise_mask").as_bool();
        show_debug_image_ = this->get_parameter("show_debug_image").as_bool();
        display_max_width_ = std::max(1, static_cast<int>(this->get_parameter("display_max_width").as_int()));
        display_max_height_ = std::max(1, static_cast<int>(this->get_parameter("display_max_height").as_int()));
    }

    bool anyImageWindowRequested() const {
        return show_input_image_ || show_white_candidate_mask_ || show_white_morph_mask_ ||
               show_green_mask_ || show_black_mask_ || show_noise_mask_ || show_debug_image_;
    }

    void syncWindow(const std::string &window_name, bool should_show, bool &created) {
        if (should_show && !created) {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);
            created = true;
        } else if (!should_show && created) {
            cv::destroyWindow(window_name);
            created = false;
        }
    }

    void syncControlsWindow(bool should_show) {
        if (should_show && !controls_window_created_) {
            cv::namedWindow(kControlsWindow, cv::WINDOW_AUTOSIZE);

            cv::createTrackbar("white_enh_min", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("white_a_center", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("white_b_center", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("white_a_tol", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("white_b_tol", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("green_a_max", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("green_b_min", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("green_b_max", kControlsWindow, nullptr, kByteMax);
            cv::createTrackbar("black_enh_max", kControlsWindow, nullptr, kByteMax);

            cv::setTrackbarPos("white_enh_min", kControlsWindow, white_enhanced_min_);
            cv::setTrackbarPos("white_a_center", kControlsWindow, white_a_center_);
            cv::setTrackbarPos("white_b_center", kControlsWindow, white_b_center_);
            cv::setTrackbarPos("white_a_tol", kControlsWindow, white_a_tol_);
            cv::setTrackbarPos("white_b_tol", kControlsWindow, white_b_tol_);
            cv::setTrackbarPos("green_a_max", kControlsWindow, green_a_max_);
            cv::setTrackbarPos("green_b_min", kControlsWindow, green_b_min_);
            cv::setTrackbarPos("green_b_max", kControlsWindow, green_b_max_);
            cv::setTrackbarPos("black_enh_max", kControlsWindow, black_enhanced_max_);

            controls_window_created_ = true;
        } else if (!should_show && controls_window_created_) {
            cv::destroyWindow(kControlsWindow);
            controls_window_created_ = false;
        }
    }

    void destroyDebugWindows() {
        syncWindow(kOriginalWindow, false, original_window_created_);
        syncWindow(kWhiteCandidateWindow, false, white_candidate_window_created_);
        syncWindow(kWhiteMorphWindow, false, white_morph_window_created_);
        syncWindow(kGreenMaskWindow, false, green_mask_window_created_);
        syncWindow(kBlackMaskWindow, false, black_mask_window_created_);
        syncWindow(kNoiseMaskWindow, false, noise_mask_window_created_);
        syncWindow(kOverlayWindow, false, overlay_window_created_);
        syncControlsWindow(false);
    }

    void syncImageViewState() {
        loadThresholdsFromParameters();
        syncWindow(kOriginalWindow, enable_image_view_ && show_input_image_, original_window_created_);
        syncWindow(
            kWhiteCandidateWindow,
            enable_image_view_ && show_white_candidate_mask_,
            white_candidate_window_created_);
        syncWindow(
            kWhiteMorphWindow,
            enable_image_view_ && show_white_morph_mask_,
            white_morph_window_created_);
        syncWindow(kGreenMaskWindow, enable_image_view_ && show_green_mask_, green_mask_window_created_);
        syncWindow(kBlackMaskWindow, enable_image_view_ && show_black_mask_, black_mask_window_created_);
        syncWindow(kNoiseMaskWindow, enable_image_view_ && show_noise_mask_, noise_mask_window_created_);
        syncWindow(kOverlayWindow, enable_image_view_ && show_debug_image_, overlay_window_created_);
        syncControlsWindow(enable_image_view_ && anyImageWindowRequested());
    }

    void readThresholdsFromTrackbars() {
        if (!controls_window_created_) {
            return;
        }
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
            "Current Lab morph thresholds: white_enhanced_min=%d, white_a_center=%d, white_b_center=%d, white_a_tol=%d, white_b_tol=%d, green_a_max=%d, green_b_min=%d, green_b_max=%d, black_enhanced_max=%d, white_open_kernel=%d, white_close_kernel=%d",
            white_enhanced_min_,
            white_a_center_,
            white_b_center_,
            white_a_tol_,
            white_b_tol_,
            green_a_max_,
            green_b_min_,
            green_b_max_,
            black_enhanced_max_,
            white_open_kernel_,
            white_close_kernel_);
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        syncImageViewState();

        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "cv_bridge failed: %s",
                e.what());
            return;
        }

        const auto morph_start = std::chrono::steady_clock::now();
        const auto preprocess_params = rcj_loc::vision::white_line::getPreprocessParams(*this);
        const auto preprocessed = rcj_loc::vision::white_line::preprocessFrame(frame, preprocess_params);

        if (controls_window_created_) {
            readThresholdsFromTrackbars();
        }

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

        const int white_open_kernel = rcj_loc::vision::debug::makeOdd(white_open_kernel_, 1);
        const int white_close_kernel = rcj_loc::vision::debug::makeOdd(white_close_kernel_, 1);

        cv::Mat white_morph_mask = raw_white_mask.clone();
        const cv::Mat open_element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(white_open_kernel, white_open_kernel));
        const cv::Mat close_element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(white_close_kernel, white_close_kernel));
        cv::morphologyEx(white_morph_mask, white_morph_mask, cv::MORPH_OPEN, open_element);
        cv::morphologyEx(white_morph_mask, white_morph_mask, cv::MORPH_CLOSE, close_element);

        cv::Mat overlay = preprocessed.filtered_bgr.clone();
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, black_mask, cv::Scalar(0, 0, 255));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, green_mask, cv::Scalar(0, 255, 0));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, noise_mask, cv::Scalar(255, 0, 255));
        overlay = rcj_loc::vision::debug::createMaskOverlay(overlay, white_morph_mask, cv::Scalar(255, 255, 255));
        const auto morph_end = std::chrono::steady_clock::now();
        const auto morph_duration_us =
            std::chrono::duration_cast<std::chrono::microseconds>(morph_end - morph_start).count();
        ++frame_count_;
        total_morph_time_us_ += morph_duration_us;
        const double average_morph_us =
            static_cast<double>(total_morph_time_us_) / static_cast<double>(frame_count_);

        white_candidate_mask_pub_->publish(
            *cv_bridge::CvImage(msg->header, "mono8", raw_white_mask).toImageMsg());
        white_morph_mask_pub_->publish(
            *cv_bridge::CvImage(msg->header, "mono8", white_morph_mask).toImageMsg());
        green_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", green_mask).toImageMsg());
        black_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", black_mask).toImageMsg());
        noise_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", noise_mask).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        bool displayed_any_window = false;
        if (original_window_created_) {
            cv::imshow(kOriginalWindow, frame);
            resizeWindowToFitImage(kOriginalWindow, frame, display_max_width_, display_max_height_);
            displayed_any_window = true;
        }
        if (white_candidate_window_created_) {
            cv::imshow(kWhiteCandidateWindow, raw_white_mask);
            resizeWindowToFitImage(
                kWhiteCandidateWindow,
                raw_white_mask,
                display_max_width_,
                display_max_height_);
            displayed_any_window = true;
        }
        if (white_morph_window_created_) {
            cv::imshow(kWhiteMorphWindow, white_morph_mask);
            resizeWindowToFitImage(
                kWhiteMorphWindow,
                white_morph_mask,
                display_max_width_,
                display_max_height_);
            displayed_any_window = true;
        }
        if (green_mask_window_created_) {
            cv::imshow(kGreenMaskWindow, green_mask);
            resizeWindowToFitImage(
                kGreenMaskWindow,
                green_mask,
                display_max_width_,
                display_max_height_);
            displayed_any_window = true;
        }
        if (black_mask_window_created_) {
            cv::imshow(kBlackMaskWindow, black_mask);
            resizeWindowToFitImage(
                kBlackMaskWindow,
                black_mask,
                display_max_width_,
                display_max_height_);
            displayed_any_window = true;
        }
        if (noise_mask_window_created_) {
            cv::imshow(kNoiseMaskWindow, noise_mask);
            resizeWindowToFitImage(
                kNoiseMaskWindow,
                noise_mask,
                display_max_width_,
                display_max_height_);
            displayed_any_window = true;
        }
        if (overlay_window_created_) {
            cv::imshow(kOverlayWindow, overlay);
            resizeWindowToFitImage(
                kOverlayWindow,
                overlay,
                display_max_width_,
                display_max_height_);
            displayed_any_window = true;
        }

        if (displayed_any_window || controls_window_created_) {
            const int key = cv::waitKey(1);
            if (key == 'p' || key == 'P') {
                logCurrentThresholds();
            }
        }

        RCLCPP_INFO(
            this->get_logger(),
            "frame=%llu morph_us=%lld morph_ms=%.3f avg_morph_ms=%.3f",
            static_cast<unsigned long long>(frame_count_),
            static_cast<long long>(morph_duration_us),
            static_cast<double>(morph_duration_us) / 1000.0,
            average_morph_us / 1000.0);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineLabMorphNode>());
    rclcpp::shutdown();
    return 0;
}
