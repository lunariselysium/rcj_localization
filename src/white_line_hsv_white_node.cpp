#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kHueMax = 179;
constexpr int kByteMax = 255;
constexpr char kInputWindowName[] = "HSV White Input";
constexpr char kMaskWindowName[] = "HSV White Mask";
constexpr char kOverlayWindowName[] = "HSV White Overlay";

int clampHue(int value)
{
  return std::clamp(value, 0, kHueMax);
}

int clampByte(int value)
{
  return std::clamp(value, 0, kByteMax);
}

cv::Size fitWithinBounds(const cv::Size & image_size, int max_width, int max_height)
{
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
    std::max(1, static_cast<int>(image_size.width * scale)),
    std::max(1, static_cast<int>(image_size.height * scale)));
}

void resizeWindowToFitImage(
  const std::string & window_name,
  const cv::Mat & image,
  int max_width,
  int max_height)
{
  const cv::Size fitted_size = fitWithinBounds(image.size(), max_width, max_height);
  cv::resizeWindow(window_name, fitted_size.width, fitted_size.height);
}

}  // namespace

class WhiteLineHsvWhiteNode : public rclcpp::Node
{
public:
  WhiteLineHsvWhiteNode()
  : Node("white_line_hsv_white_node")
  {
    declare_parameter<std::string>("input_topic", "/camera/image_remapped");
    declare_parameter("white_h_min", 0);
    declare_parameter("white_h_max", kHueMax);
    declare_parameter("white_s_max", 60);
    declare_parameter("white_v_min", 170);
    declare_parameter("enable_timing_log", true);
    declare_parameter("timing_log_interval", 30);
    declare_parameter("enable_image_view", false);
    declare_parameter("show_input_image", true);
    declare_parameter("show_white_mask", true);
    declare_parameter("show_overlay_image", true);
    declare_parameter("display_max_width", 960);
    declare_parameter("display_max_height", 720);

    loadParameters();
    syncImageViewState();

    const auto input_topic = get_parameter("input_topic").as_string();
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic,
      rclcpp::SensorDataQoS(),
      std::bind(&WhiteLineHsvWhiteNode::imageCallback, this, std::placeholders::_1));

    white_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/white_mask", 10);

    RCLCPP_INFO(
      get_logger(),
      "white_line_hsv_white_node started. input_topic=%s, white_h_min=%d, white_h_max=%d, "
      "white_s_max=%d, white_v_min=%d, enable_timing_log=%s, timing_log_interval=%d, "
      "enable_image_view=%s",
      input_topic.c_str(),
      white_h_min_,
      white_h_max_,
      white_s_max_,
      white_v_min_,
      enable_timing_log_ ? "true" : "false",
      timing_log_interval_,
      enable_image_view_ ? "true" : "false");
  }

  ~WhiteLineHsvWhiteNode() override
  {
    destroyDebugWindows();
  }

private:
  void loadParameters()
  {
    white_h_min_ = clampHue(static_cast<int>(get_parameter("white_h_min").as_int()));
    white_h_max_ = clampHue(static_cast<int>(get_parameter("white_h_max").as_int()));
    white_s_max_ = clampByte(static_cast<int>(get_parameter("white_s_max").as_int()));
    white_v_min_ = clampByte(static_cast<int>(get_parameter("white_v_min").as_int()));
    enable_timing_log_ = get_parameter("enable_timing_log").as_bool();
    timing_log_interval_ =
      std::max(1, static_cast<int>(get_parameter("timing_log_interval").as_int()));
  }

  void syncWindow(const std::string & window_name, bool should_show, bool & created)
  {
    if (should_show && !created) {
      cv::namedWindow(window_name, cv::WINDOW_NORMAL);
      created = true;
    } else if (!should_show && created) {
      cv::destroyWindow(window_name);
      created = false;
    }
  }

  void syncImageViewState()
  {
    enable_image_view_ = get_parameter("enable_image_view").as_bool();
    show_input_image_ = get_parameter("show_input_image").as_bool();
    show_white_mask_ = get_parameter("show_white_mask").as_bool();
    show_overlay_image_ = get_parameter("show_overlay_image").as_bool();
    display_max_width_ =
      std::max(1, static_cast<int>(get_parameter("display_max_width").as_int()));
    display_max_height_ =
      std::max(1, static_cast<int>(get_parameter("display_max_height").as_int()));

    const bool display_available =
      std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
    if (enable_image_view_ && !display_available) {
      if (!headless_warned_) {
        RCLCPP_WARN(
          get_logger(),
          "enable_image_view=true but no DISPLAY/WAYLAND_DISPLAY is available; "
          "disabling OpenCV image view for this process.");
        headless_warned_ = true;
      }
      enable_image_view_ = false;
    }

    syncWindow(kInputWindowName, enable_image_view_ && show_input_image_, input_window_created_);
    syncWindow(kMaskWindowName, enable_image_view_ && show_white_mask_, mask_window_created_);
    syncWindow(
      kOverlayWindowName, enable_image_view_ && show_overlay_image_, overlay_window_created_);
  }

  void destroyDebugWindows()
  {
    syncWindow(kInputWindowName, false, input_window_created_);
    syncWindow(kMaskWindowName, false, mask_window_created_);
    syncWindow(kOverlayWindowName, false, overlay_window_created_);
  }

  void showDebugImages(const cv::Mat & frame, const cv::Mat & white_mask)
  {
    if (input_window_created_) {
      cv::imshow(kInputWindowName, frame);
      resizeWindowToFitImage(
        kInputWindowName, frame, display_max_width_, display_max_height_);
    }

    if (mask_window_created_) {
      cv::imshow(kMaskWindowName, white_mask);
      resizeWindowToFitImage(
        kMaskWindowName, white_mask, display_max_width_, display_max_height_);
    }

    if (overlay_window_created_) {
      cv::Mat overlay = frame.clone();
      overlay.setTo(cv::Scalar(0, 255, 0), white_mask);
      cv::imshow(kOverlayWindowName, overlay);
      resizeWindowToFitImage(
        kOverlayWindowName, overlay, display_max_width_, display_max_height_);
    }

    if (input_window_created_ || mask_window_created_ || overlay_window_created_) {
      cv::waitKey(1);
    }
  }

  void logTiming(std::int64_t frame_duration_us)
  {
    if (!enable_timing_log_) {
      return;
    }

    ++frame_count_;
    total_duration_us_ += frame_duration_us;

    if (frame_count_ % static_cast<std::size_t>(timing_log_interval_) != 0U) {
      return;
    }

    const double current_ms = static_cast<double>(frame_duration_us) / 1000.0;
    const double average_ms =
      static_cast<double>(total_duration_us_) / 1000.0 / static_cast<double>(frame_count_);

    RCLCPP_INFO(
      get_logger(),
      "HSV white extraction timing: current=%.3f ms, average=%.3f ms, frames=%zu",
      current_ms,
      average_ms,
      frame_count_);
  }

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    const auto start = std::chrono::steady_clock::now();
    syncImageViewState();

    cv::Mat frame;
    try {
      frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "cv_bridge failed: %s",
        e.what());
      return;
    }

    cv::Mat hsv_image;
    cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat white_mask;
    if (white_h_min_ <= white_h_max_) {
      cv::inRange(
        hsv_image,
        cv::Scalar(white_h_min_, 0, white_v_min_),
        cv::Scalar(white_h_max_, white_s_max_, kByteMax),
        white_mask);
    } else {
      cv::Mat low_range_mask;
      cv::Mat high_range_mask;
      cv::inRange(
        hsv_image,
        cv::Scalar(0, 0, white_v_min_),
        cv::Scalar(white_h_max_, white_s_max_, kByteMax),
        low_range_mask);
      cv::inRange(
        hsv_image,
        cv::Scalar(white_h_min_, 0, white_v_min_),
        cv::Scalar(kHueMax, white_s_max_, kByteMax),
        high_range_mask);
      cv::bitwise_or(low_range_mask, high_range_mask, white_mask);
    }

    white_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", white_mask).toImageMsg());
    if (input_window_created_ || mask_window_created_ || overlay_window_created_) {
      showDebugImages(frame, white_mask);
    }

    const auto end = std::chrono::steady_clock::now();
    const auto duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    logTiming(duration_us);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_mask_pub_;

  int white_h_min_{0};
  int white_h_max_{kHueMax};
  int white_s_max_{60};
  int white_v_min_{170};
  bool enable_timing_log_{true};
  int timing_log_interval_{30};
  bool enable_image_view_{false};
  bool show_input_image_{true};
  bool show_white_mask_{true};
  bool show_overlay_image_{true};
  bool headless_warned_{false};
  bool input_window_created_{false};
  bool mask_window_created_{false};
  bool overlay_window_created_{false};
  int display_max_width_{960};
  int display_max_height_{720};
  std::size_t frame_count_{0};
  std::int64_t total_duration_us_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WhiteLineHsvWhiteNode>());
  rclcpp::shutdown();
  return 0;
}
