#include <algorithm>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
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
// Disabled debug window: visualizes the black-class mask after mutually exclusive assignment.
// constexpr char kBlackMaskWindow[] = "Lab Morph Black Mask";
// Disabled debug window: visualizes pixels that were left over as uncategorized noise.
// constexpr char kNoiseMaskWindow[] = "Lab Morph Noise Mask";
// Disabled debug window: visualizes the combined overlay of all classes on the filtered image.
// constexpr char kOverlayWindow[] = "Lab Morph Overlay";

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
    bool windows_initialized_ = false;
    int display_max_width_;
    int display_max_height_;

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
        display_max_width_ = static_cast<int>(this->get_parameter("display_max_width").as_int());
        display_max_height_ = static_cast<int>(this->get_parameter("display_max_height").as_int());
    }

    void createDebugWindows() {
        if (windows_initialized_) {
            return;
        }

        cv::namedWindow(kOriginalWindow, cv::WINDOW_NORMAL);
        cv::namedWindow(kWhiteCandidateWindow, cv::WINDOW_NORMAL);
        cv::namedWindow(kWhiteMorphWindow, cv::WINDOW_NORMAL);
        cv::namedWindow(kGreenMaskWindow, cv::WINDOW_NORMAL);
        // Disabled debug window: black mask view used to inspect dark-region classification.
        // cv::namedWindow(kBlackMaskWindow, cv::WINDOW_NORMAL);
        // Disabled debug window: noise mask view used to inspect uncategorized leftover pixels.
        // cv::namedWindow(kNoiseMaskWindow, cv::WINDOW_NORMAL);
        // Disabled debug window: overlay view used to inspect all class masks together.
        // cv::namedWindow(kOverlayWindow, cv::WINDOW_NORMAL);
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

        windows_initialized_ = true;
    }

    void destroyDebugWindows() {
        if (!windows_initialized_) {
            return;
        }

        cv::destroyWindow(kOriginalWindow);
        cv::destroyWindow(kWhiteCandidateWindow);
        cv::destroyWindow(kWhiteMorphWindow);
        cv::destroyWindow(kGreenMaskWindow);
        // Disabled debug window teardown: black mask view.
        // cv::destroyWindow(kBlackMaskWindow);
        // Disabled debug window teardown: noise mask view.
        // cv::destroyWindow(kNoiseMaskWindow);
        // Disabled debug window teardown: combined overlay view.
        // cv::destroyWindow(kOverlayWindow);
        cv::destroyWindow(kControlsWindow);
        windows_initialized_ = false;
    }

    void syncImageViewState() {
        loadThresholdsFromParameters();
        if (enable_image_view_) {
            createDebugWindows();
        } else {
            destroyDebugWindows();
        }
    }

    void readThresholdsFromTrackbars() {
        if (!windows_initialized_) {
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

        const auto preprocess_params = rcj_loc::vision::white_line::getPreprocessParams(*this);
        const auto preprocessed = rcj_loc::vision::white_line::preprocessFrame(frame, preprocess_params);

        if (windows_initialized_) {
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

        white_candidate_mask_pub_->publish(
            *cv_bridge::CvImage(msg->header, "mono8", raw_white_mask).toImageMsg());
        white_morph_mask_pub_->publish(
            *cv_bridge::CvImage(msg->header, "mono8", white_morph_mask).toImageMsg());
        green_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", green_mask).toImageMsg());
        black_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", black_mask).toImageMsg());
        noise_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", noise_mask).toImageMsg());
        debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());

        if (windows_initialized_) {
            cv::imshow(kOriginalWindow, frame);
            cv::imshow(kWhiteCandidateWindow, raw_white_mask);
            cv::imshow(kWhiteMorphWindow, white_morph_mask);
            cv::imshow(kGreenMaskWindow, green_mask);
            // Disabled debug window: black mask visualization.
            // cv::imshow(kBlackMaskWindow, black_mask);
            // Disabled debug window: noise mask visualization.
            // cv::imshow(kNoiseMaskWindow, noise_mask);
            // Disabled debug window: combined overlay visualization.
            // cv::imshow(kOverlayWindow, overlay);

            resizeWindowToFitImage(
                kOriginalWindow,
                frame,
                display_max_width_,
                display_max_height_);
            resizeWindowToFitImage(
                kWhiteCandidateWindow,
                raw_white_mask,
                display_max_width_,
                display_max_height_);
            resizeWindowToFitImage(
                kWhiteMorphWindow,
                white_morph_mask,
                display_max_width_,
                display_max_height_);
            resizeWindowToFitImage(
                kGreenMaskWindow,
                green_mask,
                display_max_width_,
                display_max_height_);
            // Disabled debug window resize: black mask visualization.
            // resizeWindowToFitImage(
            //     kBlackMaskWindow,
            //     black_mask,
            //     display_max_width_,
            //     display_max_height_);
            // Disabled debug window resize: noise mask visualization.
            // resizeWindowToFitImage(
            //     kNoiseMaskWindow,
            //     noise_mask,
            //     display_max_width_,
            //     display_max_height_);
            // Disabled debug window resize: combined overlay visualization.
            // resizeWindowToFitImage(
            //     kOverlayWindow,
            //     overlay,
            //     display_max_width_,
            //     display_max_height_);

            const int key = cv::waitKey(1);
            if (key == 'p' || key == 'P') {
                logCurrentThresholds();
            }
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WhiteLineLabMorphNode>());
    rclcpp::shutdown();
    return 0;
}
