#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#elif __has_include(<cv_bridge/cv_bridge.h>)
#include <cv_bridge/cv_bridge.h>
#else
#error "cv_bridge header not found"
#endif
#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/synchronizer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include "rcj_localization/white_line_debug_utils.hpp"

namespace {

constexpr char kMorphWindow[] = "DT Ridge Filter Input Morph Mask";
constexpr char kGreenWindow[] = "DT Ridge Filter Input Green Mask";
constexpr char kBlackWindow[] = "DT Ridge Filter Input Black Mask";
constexpr char kNoiseWindow[] = "DT Ridge Filter Input Noise Mask";
constexpr char kRidgeWindow[] = "DT Ridge Filter Ridge";
constexpr char kOrientationValidWindow[] = "DT Ridge Filter Orientation Valid";
constexpr char kSideSupportWindow[] = "DT Ridge Filter Side Color Filter";
constexpr char kWidthSupportedWindow[] = "DT Ridge Filter Width Filter";
constexpr char kLengthFilteredRidgeWindow[] = "DT Ridge Filter Length-Filtered Ridge";
constexpr char kReconstructedWindow[] = "DT Ridge Filter Reconstruction";
constexpr char kWhiteFinalMaskWindow[] = "DT Ridge Filter White Final Mask";
constexpr char kDebugWindow[] = "DT Ridge Filter Composite Debug";
constexpr bool kDefaultShowLengthFilteredRidgeMask = true;

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
    std::max(1, static_cast<int>(std::round(image_size.width * scale))),
    std::max(1, static_cast<int>(std::round(image_size.height * scale))));
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

double computeMedian(std::vector<float> values)
{
  if (values.empty()) {
    return 0.0;
  }

  const auto mid_it = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid_it, values.end());
  const double upper = *mid_it;
  if ((values.size() % 2U) == 1U) {
    return upper;
  }

  const auto lower_it = std::max_element(values.begin(), mid_it);
  return 0.5 * (upper + *lower_it);
}

bool estimateOrientation(
  const cv::Mat & centerline_mask,
  const cv::Point & center,
  int radius,
  int min_neighbors,
  cv::Point2f & tangent,
  cv::Point2f & normal)
{
  const int clamped_radius = std::max(1, radius);
  const int x_min = std::max(0, center.x - clamped_radius);
  const int x_max = std::min(centerline_mask.cols - 1, center.x + clamped_radius);
  const int y_min = std::max(0, center.y - clamped_radius);
  const int y_max = std::min(centerline_mask.rows - 1, center.y + clamped_radius);
  const int radius_sq = clamped_radius * clamped_radius;

  std::vector<cv::Point2f> neighbors;
  neighbors.reserve(static_cast<std::size_t>((2 * clamped_radius + 1) * (2 * clamped_radius + 1)));
  for (int y = y_min; y <= y_max; ++y) {
    for (int x = x_min; x <= x_max; ++x) {
      if (centerline_mask.at<uchar>(y, x) == 0) {
        continue;
      }
      const int dx = x - center.x;
      const int dy = y - center.y;
      if (dx * dx + dy * dy > radius_sq) {
        continue;
      }
      neighbors.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
  }

  if (static_cast<int>(neighbors.size()) < std::max(2, min_neighbors)) {
    return false;
  }

  cv::Point2f mean(0.0F, 0.0F);
  for (const auto & neighbor : neighbors) {
    mean += neighbor;
  }
  mean.x /= static_cast<float>(neighbors.size());
  mean.y /= static_cast<float>(neighbors.size());

  double cov_xx = 0.0;
  double cov_xy = 0.0;
  double cov_yy = 0.0;
  for (const auto & neighbor : neighbors) {
    const double dx = neighbor.x - mean.x;
    const double dy = neighbor.y - mean.y;
    cov_xx += dx * dx;
    cov_xy += dx * dy;
    cov_yy += dy * dy;
  }

  const double trace = cov_xx + cov_yy;
  const double det = cov_xx * cov_yy - cov_xy * cov_xy;
  const double delta = std::sqrt(std::max(0.0, 0.25 * trace * trace - det));
  const double lambda1 = 0.5 * trace + delta;

  cv::Point2f eigenvector;
  if (std::abs(cov_xy) > 1e-6 || std::abs(lambda1 - cov_xx) > 1e-6) {
    eigenvector = cv::Point2f(
      static_cast<float>(cov_xy),
      static_cast<float>(lambda1 - cov_xx));
  } else if (cov_xx >= cov_yy) {
    eigenvector = cv::Point2f(1.0F, 0.0F);
  } else {
    eigenvector = cv::Point2f(0.0F, 1.0F);
  }

  const float norm = std::sqrt(eigenvector.x * eigenvector.x + eigenvector.y * eigenvector.y);
  if (norm < 1e-6F) {
    return false;
  }

  tangent = cv::Point2f(eigenvector.x / norm, eigenvector.y / norm);
  normal = cv::Point2f(-tangent.y, tangent.x);
  return true;
}

struct SideSampleStats
{
  int total_samples = 0;
  int green_samples = 0;
  int black_samples = 0;
  int noise_samples = 0;
  int outside_samples = 0;

  double greenRatio() const
  {
    return total_samples > 0 ? static_cast<double>(green_samples) / total_samples : 0.0;
  }

  double boundaryRatio() const
  {
    return total_samples > 0
             ? static_cast<double>(black_samples + noise_samples + outside_samples) / total_samples
             : 0.0;
  }
};

SideSampleStats sampleSideSupport(
  const cv::Point & center,
  const cv::Point2f & tangent,
  const cv::Point2f & normal,
  float local_width_px,
  int side_sign,
  const cv::Mat & green_mask,
  const cv::Mat & black_mask,
  const cv::Mat & noise_mask,
  int side_margin_px,
  int side_band_depth_px)
{
  SideSampleStats stats;

  const float half_width = 0.5F * std::max(local_width_px, 0.0F);
  const int start_offset =
    std::max(1, static_cast<int>(std::round(half_width + static_cast<float>(side_margin_px))));
  const int depth = std::max(1, side_band_depth_px);
  const int tangent_half_span =
    std::max(1, static_cast<int>(std::round(std::max(1.0F, half_width))));

  const cv::Point2f center_f(static_cast<float>(center.x), static_cast<float>(center.y));
  for (int tangential_step = -tangent_half_span; tangential_step <= tangent_half_span;
    ++tangential_step)
  {
    const cv::Point2f tangential_offset = static_cast<float>(tangential_step) * tangent;
    for (int depth_step = 0; depth_step < depth; ++depth_step) {
      const float offset = static_cast<float>(start_offset + depth_step);
      const cv::Point2f sample =
        center_f + tangential_offset + static_cast<float>(side_sign) * offset * normal;
      const int sample_x = static_cast<int>(std::round(sample.x));
      const int sample_y = static_cast<int>(std::round(sample.y));

      ++stats.total_samples;
      if (
        sample_x < 0 || sample_x >= green_mask.cols || sample_y < 0 ||
        sample_y >= green_mask.rows)
      {
        ++stats.outside_samples;
        continue;
      }

      if (green_mask.at<uchar>(sample_y, sample_x) != 0) {
        ++stats.green_samples;
      }
      if (black_mask.at<uchar>(sample_y, sample_x) != 0) {
        ++stats.black_samples;
      }
      if (noise_mask.at<uchar>(sample_y, sample_x) != 0) {
        ++stats.noise_samples;
      }
    }
  }

  return stats;
}

cv::Mat filterMaskByLength(const cv::Mat & binary_mask, int min_length_pixels)
{
  CV_Assert(binary_mask.type() == CV_8UC1);

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);

  cv::Mat filtered = cv::Mat::zeros(binary_mask.size(), CV_8UC1);
  for (int label = 1; label < stats.rows; ++label) {
    const int component_area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (component_area < min_length_pixels) {
      continue;
    }
    filtered.setTo(255, labels == label);
  }

  return filtered;
}

bool isLocalMaxPair(float center, float first, float second)
{
  return center >= first && center >= second && (center > first || center > second);
}

cv::Mat extractDistanceTransformRidgeMask(
  const cv::Mat & morph_mask,
  const cv::Mat & distance_transform)
{
  CV_Assert(morph_mask.type() == CV_8UC1);
  CV_Assert(distance_transform.type() == CV_32FC1);

  cv::Mat ridge_mask = cv::Mat::zeros(morph_mask.size(), CV_8UC1);

  auto sample_distance = [&](int x, int y) -> float {
      if (x < 0 || x >= distance_transform.cols || y < 0 || y >= distance_transform.rows) {
        return 0.0F;
      }
      return distance_transform.at<float>(y, x);
    };

  for (int y = 0; y < morph_mask.rows; ++y) {
    for (int x = 0; x < morph_mask.cols; ++x) {
      if (morph_mask.at<uchar>(y, x) == 0) {
        continue;
      }

      const float center = distance_transform.at<float>(y, x);
      if (center <= 0.0F) {
        continue;
      }

      const bool is_ridge =
        isLocalMaxPair(center, sample_distance(x - 1, y), sample_distance(x + 1, y)) ||
        isLocalMaxPair(center, sample_distance(x, y - 1), sample_distance(x, y + 1)) ||
        isLocalMaxPair(center, sample_distance(x - 1, y - 1), sample_distance(x + 1, y + 1)) ||
        isLocalMaxPair(center, sample_distance(x - 1, y + 1), sample_distance(x + 1, y - 1));

      if (is_ridge) {
        ridge_mask.at<uchar>(y, x) = 255;
      }
    }
  }

  return ridge_mask;
}

cv::Mat createDebugComposite(
  const cv::Mat & green_mask,
  const cv::Mat & black_mask,
  const cv::Mat & white_final_mask)
{
  cv::Mat debug_image(green_mask.size(), CV_8UC3, cv::Scalar(30, 70, 30));
  debug_image =
    rcj_loc::vision::debug::createMaskOverlay(debug_image, black_mask, cv::Scalar(0, 0, 255));
  debug_image =
    rcj_loc::vision::debug::createMaskOverlay(debug_image, green_mask, cv::Scalar(0, 200, 0));
  debug_image =
    rcj_loc::vision::debug::createMaskOverlay(debug_image, white_final_mask, cv::Scalar(255, 255, 255));
  return debug_image;
}

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

long long elapsedUs(const TimePoint & start, const TimePoint & end)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

enum class RidgeTimingStage : std::size_t
{
  RuntimeSync = 0,
  CvBridgeConvert,
  DistanceTransform,
  RidgeExtract,
  SideSupportScan,
  WidthRangeEstimate,
  WidthFilter,
  LengthFilter,
  Reconstruct,
  FinalMask,
  DebugComposite,
  PublishOutputs,
  GuiDisplay,
  CallbackTotal,
  Count
};

constexpr std::array<const char *, static_cast<std::size_t>(RidgeTimingStage::Count)>
  kRidgeTimingLabels = {
    "runtime_sync",
    "cv_bridge",
    "distance_transform",
    "ridge_extract",
    "side_support_scan",
    "width_range",
    "width_filter",
    "length_filter",
    "reconstruct",
    "final_mask",
    "debug_composite",
    "publish_outputs",
    "gui_display",
    "callback_total",
  };

using RidgeTimingArray =
  std::array<long long, static_cast<std::size_t>(RidgeTimingStage::Count)>;

struct RidgeFrameTiming
{
  RidgeTimingArray stage_us{};
};

void recordStageDuration(
  RidgeTimingArray & target,
  RidgeTimingStage stage,
  const TimePoint & start,
  const TimePoint & end)
{
  target[static_cast<std::size_t>(stage)] = elapsedUs(start, end);
}

template<typename PublisherT>
std::size_t subscriptionCount(const std::shared_ptr<PublisherT> & publisher)
{
  return publisher == nullptr ? 0U : publisher->get_subscription_count();
}

template<typename PublisherT>
bool hasSubscribers(const std::shared_ptr<PublisherT> & publisher)
{
  return subscriptionCount(publisher) > 0U;
}

template<typename PublisherT>
bool publishImageIfSubscribed(
  const std::shared_ptr<PublisherT> & publisher,
  const std_msgs::msg::Header & header,
  const std::string & encoding,
  const cv::Mat & image)
{
  if (!hasSubscribers(publisher)) {
    return false;
  }
  publisher->publish(*cv_bridge::CvImage(header, encoding, image).toImageMsg());
  return true;
}

void appendTimingTable(
  std::ostringstream & oss,
  const RidgeFrameTiming & timing,
  const RidgeTimingArray & interval_totals,
  std::size_t interval_frame_count)
{
  const double callback_average_ms =
    interval_frame_count > 0
      ? static_cast<double>(
      interval_totals[static_cast<std::size_t>(RidgeTimingStage::CallbackTotal)]) /
      static_cast<double>(interval_frame_count) / 1000.0
      : 0.0;

  oss << std::left << std::setw(22) << "阶段"
      << std::right << std::setw(12) << "当前ms"
      << std::setw(14) << "区间平均ms"
      << std::setw(12) << "占比%" << '\n';

  for (std::size_t i = 0; i < kRidgeTimingLabels.size(); ++i) {
    const double current_ms = static_cast<double>(timing.stage_us[i]) / 1000.0;
    const double interval_average_ms =
      interval_frame_count > 0
        ? static_cast<double>(interval_totals[i]) / static_cast<double>(interval_frame_count) /
        1000.0
        : 0.0;
    const double ratio =
      callback_average_ms > 0.0 ? (interval_average_ms / callback_average_ms) * 100.0 : 0.0;

    oss << std::left << std::setw(22) << kRidgeTimingLabels[i]
        << std::right << std::setw(12) << std::fixed << std::setprecision(3) << current_ms
        << std::setw(14) << std::fixed << std::setprecision(3) << interval_average_ms
        << std::setw(12) << std::fixed << std::setprecision(1) << ratio << '\n';
  }
}

}  // namespace

class WhiteLineDtRidgeFilterNode : public rclcpp::Node
{
public:
  WhiteLineDtRidgeFilterNode()
  : Node("white_line_dt_ridge_filter_node")
  {
    declare_parameter<std::string>("morph_mask_topic", "/white_line_hsv_white_node/white_mask");
    declare_parameter<std::string>("green_mask_topic", "/white_line_hsv_white_node/green_mask");
    declare_parameter<std::string>("black_mask_topic", "/white_line_hsv_white_node/black_mask");
    declare_parameter<std::string>("noise_mask_topic", "/white_line_hsv_white_node/noise_mask");
    declare_parameter("orientation_window_radius_px", 5);
    declare_parameter("min_orientation_neighbors", 6);
    declare_parameter("side_margin_px", 1);
    declare_parameter("side_band_depth_px", 4);
    declare_parameter("min_green_ratio", 0.35);
    declare_parameter("min_boundary_ratio", 0.35);
    declare_parameter("enable_boundary_mode", true);
    declare_parameter("width_floor_px", 2.0);
    declare_parameter("width_ceil_px", 40.0);
    declare_parameter("width_mad_scale", 2.5);
    declare_parameter("min_width_samples", 25);
    declare_parameter("min_skeleton_length_px", 12);
    declare_parameter("reconstruction_margin_px", 1.0);
    declare_parameter("enable_image_view", false);
    declare_parameter("show_morph_mask", true);
    declare_parameter("show_green_mask", false);
    declare_parameter("show_black_mask", false);
    declare_parameter("show_noise_mask", false);
    declare_parameter("show_ridge_mask", true);
    declare_parameter("show_skeleton_mask", true);
    declare_parameter("show_orientation_valid_mask", true);
    declare_parameter("show_side_support_mask", true);
    declare_parameter("show_width_supported_ridge_mask", true);
    declare_parameter("show_width_supported_skeleton_mask", true);
    declare_parameter("show_length_filtered_ridge_mask", kDefaultShowLengthFilteredRidgeMask);
    declare_parameter("show_length_filtered_skeleton_mask", kDefaultShowLengthFilteredRidgeMask);
    declare_parameter("show_supported_skeleton_mask", kDefaultShowLengthFilteredRidgeMask);
    declare_parameter("show_reconstructed_mask", true);
    declare_parameter("show_white_final_mask", true);
    declare_parameter("show_white_mask", true);
    declare_parameter("show_debug_image", true);
    declare_parameter("enable_timing_debug", false);
    declare_parameter("timing_summary_interval", 10);
    declare_parameter("display_max_width", 960);
    declare_parameter("display_max_height", 720);

    syncImageViewState();
    setupSubscribers();

    ridge_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/ridge_mask", 10);
    legacy_skeleton_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/skeleton_mask", 10);
    orientation_valid_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/orientation_valid_mask", 10);
    side_support_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/side_support_mask", 10);
    width_supported_ridge_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/width_supported_ridge_mask", 10);
    legacy_width_supported_skeleton_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/width_supported_skeleton_mask", 10);
    length_filtered_ridge_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/length_filtered_ridge_mask", 10);
    legacy_length_filtered_skeleton_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/length_filtered_skeleton_mask", 10);
    legacy_supported_skeleton_mask_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/supported_skeleton_mask", 10);
    reconstructed_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/reconstructed_mask", 10);
    white_final_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/white_final_mask", 10);
    legacy_white_mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/white_mask", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

    RCLCPP_INFO(
      get_logger(),
      "white_line_dt_ridge_filter_node started. enable_image_view=%s, enable_timing_debug=%s, "
      "timing_summary_interval=%d. Waiting for morph masks on '%s'.",
      enable_image_view_ ? "true" : "false",
      enable_timing_debug_ ? "true" : "false",
      timing_summary_interval_,
      morph_mask_topic_.c_str());
  }

  ~WhiteLineDtRidgeFilterNode() override
  {
    destroyDebugWindows();
  }

private:
  using ImageMsg = sensor_msgs::msg::Image;
  using ExactPolicy =
    message_filters::sync_policies::ExactTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg>;

  message_filters::Subscriber<ImageMsg> morph_sub_;
  message_filters::Subscriber<ImageMsg> green_sub_;
  message_filters::Subscriber<ImageMsg> black_sub_;
  message_filters::Subscriber<ImageMsg> noise_sub_;
  std::shared_ptr<message_filters::Synchronizer<ExactPolicy>> synchronizer_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ridge_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr legacy_skeleton_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr orientation_valid_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr side_support_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr width_supported_ridge_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr legacy_width_supported_skeleton_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr length_filtered_ridge_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr legacy_length_filtered_skeleton_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr legacy_supported_skeleton_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr reconstructed_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr white_final_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr legacy_white_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;

  std::string morph_mask_topic_;
  std::string green_mask_topic_;
  std::string black_mask_topic_;
  std::string noise_mask_topic_;
  int orientation_window_radius_px_ = 5;
  int min_orientation_neighbors_ = 6;
  int side_margin_px_ = 1;
  int side_band_depth_px_ = 4;
  double min_green_ratio_ = 0.35;
  double min_boundary_ratio_ = 0.35;
  bool enable_boundary_mode_ = true;
  double width_floor_px_ = 2.0;
  double width_ceil_px_ = 40.0;
  double width_mad_scale_ = 2.5;
  int min_width_samples_ = 25;
  int min_skeleton_length_px_ = 12;
  double reconstruction_margin_px_ = 1.0;
  bool enable_image_view_ = false;
  bool show_morph_mask_ = true;
  bool show_green_mask_ = false;
  bool show_black_mask_ = false;
  bool show_noise_mask_ = false;
  bool show_ridge_mask_ = true;
  bool show_orientation_valid_mask_ = true;
  bool show_side_support_mask_ = true;
  bool show_width_supported_ridge_mask_ = true;
  bool show_length_filtered_ridge_mask_ = kDefaultShowLengthFilteredRidgeMask;
  bool show_reconstructed_mask_ = true;
  bool show_white_final_mask_ = true;
  bool show_debug_image_ = true;
  bool enable_timing_debug_ = false;
  bool morph_window_created_ = false;
  bool green_window_created_ = false;
  bool black_window_created_ = false;
  bool noise_window_created_ = false;
  bool ridge_window_created_ = false;
  bool orientation_valid_window_created_ = false;
  bool side_support_window_created_ = false;
  bool width_supported_window_created_ = false;
  bool length_filtered_ridge_window_created_ = false;
  bool reconstructed_window_created_ = false;
  bool white_final_mask_window_created_ = false;
  bool debug_window_created_ = false;
  bool warned_deprecated_white_mask_topic_ = false;
  int display_max_width_ = 960;
  int display_max_height_ = 720;
  unsigned long long frame_count_ = 0;
  int timing_summary_interval_ = 10;
  std::size_t timing_frames_in_interval_ = 0;
  unsigned long long timing_interval_start_frame_ = 0;
  long long timing_interval_total_us_ = 0;
  long long timing_interval_max_us_ = 0;
  RidgeTimingArray timing_stage_interval_totals_{};

  bool isParameterOverridden(const char * name)
  {
    const auto & overrides = get_node_parameters_interface()->get_parameter_overrides();
    return overrides.find(name) != overrides.end();
  }

  bool resolveBoolParameter(const char * current_name, std::initializer_list<const char *> legacy_names)
  {
    if (isParameterOverridden(current_name)) {
      return get_parameter(current_name).as_bool();
    }
    for (const char * legacy_name : legacy_names) {
      if (isParameterOverridden(legacy_name)) {
        return get_parameter(legacy_name).as_bool();
      }
    }
    return get_parameter(current_name).as_bool();
  }

  void loadRuntimeParameters()
  {
    morph_mask_topic_ = get_parameter("morph_mask_topic").as_string();
    green_mask_topic_ = get_parameter("green_mask_topic").as_string();
    black_mask_topic_ = get_parameter("black_mask_topic").as_string();
    noise_mask_topic_ = get_parameter("noise_mask_topic").as_string();
    orientation_window_radius_px_ =
      std::max(1, static_cast<int>(get_parameter("orientation_window_radius_px").as_int()));
    min_orientation_neighbors_ =
      std::max(2, static_cast<int>(get_parameter("min_orientation_neighbors").as_int()));
    side_margin_px_ = std::max(0, static_cast<int>(get_parameter("side_margin_px").as_int()));
    side_band_depth_px_ =
      std::max(1, static_cast<int>(get_parameter("side_band_depth_px").as_int()));
    min_green_ratio_ = std::clamp(get_parameter("min_green_ratio").as_double(), 0.0, 1.0);
    min_boundary_ratio_ = std::clamp(get_parameter("min_boundary_ratio").as_double(), 0.0, 1.0);
    enable_boundary_mode_ = get_parameter("enable_boundary_mode").as_bool();
    width_floor_px_ = std::max(0.0, get_parameter("width_floor_px").as_double());
    width_ceil_px_ = std::max(width_floor_px_, get_parameter("width_ceil_px").as_double());
    width_mad_scale_ = std::max(0.0, get_parameter("width_mad_scale").as_double());
    min_width_samples_ = std::max(1, static_cast<int>(get_parameter("min_width_samples").as_int()));
    min_skeleton_length_px_ =
      std::max(1, static_cast<int>(get_parameter("min_skeleton_length_px").as_int()));
    reconstruction_margin_px_ =
      std::max(0.0, get_parameter("reconstruction_margin_px").as_double());
    enable_image_view_ = get_parameter("enable_image_view").as_bool();
    show_morph_mask_ = get_parameter("show_morph_mask").as_bool();
    show_green_mask_ = get_parameter("show_green_mask").as_bool();
    show_black_mask_ = get_parameter("show_black_mask").as_bool();
    show_noise_mask_ = get_parameter("show_noise_mask").as_bool();
    show_ridge_mask_ = resolveBoolParameter("show_ridge_mask", {"show_skeleton_mask"});
    show_orientation_valid_mask_ = get_parameter("show_orientation_valid_mask").as_bool();
    show_side_support_mask_ = get_parameter("show_side_support_mask").as_bool();
    show_width_supported_ridge_mask_ = resolveBoolParameter(
      "show_width_supported_ridge_mask",
      {"show_width_supported_skeleton_mask"});
    show_length_filtered_ridge_mask_ = resolveBoolParameter(
      "show_length_filtered_ridge_mask",
      {"show_length_filtered_skeleton_mask", "show_supported_skeleton_mask"});
    show_reconstructed_mask_ = get_parameter("show_reconstructed_mask").as_bool();
    show_white_final_mask_ = resolveBoolParameter("show_white_final_mask", {"show_white_mask"});
    show_debug_image_ = get_parameter("show_debug_image").as_bool();
    enable_timing_debug_ = get_parameter("enable_timing_debug").as_bool();
    timing_summary_interval_ =
      std::max(1, static_cast<int>(get_parameter("timing_summary_interval").as_int()));
    display_max_width_ = std::max(1, static_cast<int>(get_parameter("display_max_width").as_int()));
    display_max_height_ =
      std::max(1, static_cast<int>(get_parameter("display_max_height").as_int()));
  }

  void resetTimingSummary()
  {
    timing_frames_in_interval_ = 0;
    timing_interval_start_frame_ = 0;
    timing_interval_total_us_ = 0;
    timing_interval_max_us_ = 0;
    timing_stage_interval_totals_.fill(0);
  }

  void maybeLogTimingSummary(
    const RidgeFrameTiming & timing,
    unsigned long long current_frame_index)
  {
    if (!enable_timing_debug_) {
      return;
    }

    const long long callback_duration_us =
      timing.stage_us[static_cast<std::size_t>(RidgeTimingStage::CallbackTotal)];

    if (timing_frames_in_interval_ == 0U) {
      timing_interval_start_frame_ = current_frame_index;
    }

    ++timing_frames_in_interval_;
    timing_interval_total_us_ += callback_duration_us;
    timing_interval_max_us_ = std::max(timing_interval_max_us_, callback_duration_us);
    for (std::size_t i = 0; i < timing_stage_interval_totals_.size(); ++i) {
      timing_stage_interval_totals_[i] += timing.stage_us[i];
    }

    if (timing_frames_in_interval_ < static_cast<std::size_t>(timing_summary_interval_)) {
      return;
    }

    const double avg_ms =
      static_cast<double>(timing_interval_total_us_) /
      static_cast<double>(timing_frames_in_interval_) / 1000.0;

    std::ostringstream oss;
    oss << "\n================ DT Ridge Profiling 摘要 ================\n";
    oss << "帧区间: " << timing_interval_start_frame_ << " - " << current_frame_index
        << " (" << timing_frames_in_interval_ << " 帧)\n";
    oss << "当前帧总耗时: " << static_cast<double>(callback_duration_us) / 1000.0
        << " ms, 区间平均: " << avg_ms
        << " ms, 区间最大: " << static_cast<double>(timing_interval_max_us_) / 1000.0 << " ms\n\n";
    appendTimingTable(oss, timing, timing_stage_interval_totals_, timing_frames_in_interval_);
    oss << "========================================================";
    RCLCPP_INFO_STREAM(get_logger(), oss.str());

    resetTimingSummary();
  }

  bool anyImageWindowRequested() const
  {
    return show_morph_mask_ || show_green_mask_ || show_black_mask_ || show_noise_mask_ ||
           show_ridge_mask_ || show_orientation_valid_mask_ || show_side_support_mask_ ||
           show_width_supported_ridge_mask_ || show_length_filtered_ridge_mask_ ||
           show_reconstructed_mask_ || show_white_final_mask_ || show_debug_image_;
  }

  bool anyImageWindowCreated() const
  {
    return morph_window_created_ || green_window_created_ || black_window_created_ ||
           noise_window_created_ || ridge_window_created_ || orientation_valid_window_created_ ||
           side_support_window_created_ || width_supported_window_created_ ||
           length_filtered_ridge_window_created_ || reconstructed_window_created_ ||
           white_final_mask_window_created_ || debug_window_created_;
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

  void destroyDebugWindows()
  {
    syncWindow(kMorphWindow, false, morph_window_created_);
    syncWindow(kGreenWindow, false, green_window_created_);
    syncWindow(kBlackWindow, false, black_window_created_);
    syncWindow(kNoiseWindow, false, noise_window_created_);
    syncWindow(kRidgeWindow, false, ridge_window_created_);
    syncWindow(kOrientationValidWindow, false, orientation_valid_window_created_);
    syncWindow(kSideSupportWindow, false, side_support_window_created_);
    syncWindow(kWidthSupportedWindow, false, width_supported_window_created_);
    syncWindow(kLengthFilteredRidgeWindow, false, length_filtered_ridge_window_created_);
    syncWindow(kReconstructedWindow, false, reconstructed_window_created_);
    syncWindow(kWhiteFinalMaskWindow, false, white_final_mask_window_created_);
    syncWindow(kDebugWindow, false, debug_window_created_);
  }

  void syncImageViewState()
  {
    loadRuntimeParameters();
    const bool master_enabled = enable_image_view_ && anyImageWindowRequested();
    syncWindow(kMorphWindow, master_enabled && show_morph_mask_, morph_window_created_);
    syncWindow(kGreenWindow, master_enabled && show_green_mask_, green_window_created_);
    syncWindow(kBlackWindow, master_enabled && show_black_mask_, black_window_created_);
    syncWindow(kNoiseWindow, master_enabled && show_noise_mask_, noise_window_created_);
    syncWindow(kRidgeWindow, master_enabled && show_ridge_mask_, ridge_window_created_);
    syncWindow(
      kOrientationValidWindow,
      master_enabled && show_orientation_valid_mask_,
      orientation_valid_window_created_);
    syncWindow(
      kSideSupportWindow,
      master_enabled && show_side_support_mask_,
      side_support_window_created_);
    syncWindow(
      kWidthSupportedWindow,
      master_enabled && show_width_supported_ridge_mask_,
      width_supported_window_created_);
    syncWindow(
      kLengthFilteredRidgeWindow,
      master_enabled && show_length_filtered_ridge_mask_,
      length_filtered_ridge_window_created_);
    syncWindow(
      kReconstructedWindow,
      master_enabled && show_reconstructed_mask_,
      reconstructed_window_created_);
    syncWindow(
      kWhiteFinalMaskWindow,
      master_enabled && show_white_final_mask_,
      white_final_mask_window_created_);
    syncWindow(kDebugWindow, master_enabled && show_debug_image_, debug_window_created_);
  }

  void setupSubscribers()
  {
    morph_sub_.subscribe(this, morph_mask_topic_, rmw_qos_profile_sensor_data);
    green_sub_.subscribe(this, green_mask_topic_, rmw_qos_profile_sensor_data);
    black_sub_.subscribe(this, black_mask_topic_, rmw_qos_profile_sensor_data);
    noise_sub_.subscribe(this, noise_mask_topic_, rmw_qos_profile_sensor_data);

    synchronizer_ = std::make_shared<message_filters::Synchronizer<ExactPolicy>>(
      ExactPolicy(10),
      morph_sub_,
      green_sub_,
      black_sub_,
      noise_sub_);
    synchronizer_->registerCallback(std::bind(
        &WhiteLineDtRidgeFilterNode::maskCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
  }

  bool publishOutputs(
    const std_msgs::msg::Header & header,
    const cv::Mat & ridge_mask,
    const cv::Mat & orientation_valid_mask,
    const cv::Mat & side_support_mask,
    const cv::Mat & width_supported_ridge_mask,
    const cv::Mat & length_filtered_ridge_mask,
    const cv::Mat & reconstructed_mask,
    const cv::Mat & white_final_mask,
    const cv::Mat & debug_image)
  {
    bool published_any = false;

    published_any |=
      publishImageIfSubscribed(white_final_mask_pub_, header, "mono8", white_final_mask);

    if (hasSubscribers(legacy_white_mask_pub_)) {
      if (!warned_deprecated_white_mask_topic_) {
        RCLCPP_WARN(
          get_logger(),
          "Topic '~/white_mask' is deprecated. Use '~/white_final_mask' instead.");
        warned_deprecated_white_mask_topic_ = true;
      }
      legacy_white_mask_pub_->publish(
        *cv_bridge::CvImage(header, "mono8", white_final_mask).toImageMsg());
      published_any = true;
    }

    if (show_ridge_mask_) {
      published_any |= publishImageIfSubscribed(ridge_mask_pub_, header, "mono8", ridge_mask);
      published_any |= publishImageIfSubscribed(
        legacy_skeleton_mask_pub_,
        header,
        "mono8",
        ridge_mask);
    }
    if (show_orientation_valid_mask_) {
      published_any |= publishImageIfSubscribed(
        orientation_valid_mask_pub_,
        header,
        "mono8",
        orientation_valid_mask);
    }
    if (show_side_support_mask_) {
      published_any |= publishImageIfSubscribed(
        side_support_mask_pub_,
        header,
        "mono8",
        side_support_mask);
    }
    if (show_width_supported_ridge_mask_) {
      published_any |= publishImageIfSubscribed(
        width_supported_ridge_mask_pub_,
        header,
        "mono8",
        width_supported_ridge_mask);
      published_any |= publishImageIfSubscribed(
        legacy_width_supported_skeleton_mask_pub_,
        header,
        "mono8",
        width_supported_ridge_mask);
    }
    if (show_length_filtered_ridge_mask_) {
      published_any |= publishImageIfSubscribed(
        length_filtered_ridge_mask_pub_,
        header,
        "mono8",
        length_filtered_ridge_mask);
      published_any |= publishImageIfSubscribed(
        legacy_length_filtered_skeleton_mask_pub_,
        header,
        "mono8",
        length_filtered_ridge_mask);
      published_any |= publishImageIfSubscribed(
        legacy_supported_skeleton_mask_pub_,
        header,
        "mono8",
        length_filtered_ridge_mask);
    }
    if (show_reconstructed_mask_) {
      published_any |= publishImageIfSubscribed(
        reconstructed_mask_pub_,
        header,
        "mono8",
        reconstructed_mask);
    }
    if (show_debug_image_ && !debug_image.empty()) {
      published_any |= publishImageIfSubscribed(debug_pub_, header, "bgr8", debug_image);
    }

    return published_any;
  }

  bool displayOutputs(
    const cv::Mat & morph_mask,
    const cv::Mat & ridge_mask,
    const cv::Mat & orientation_valid_mask,
    const cv::Mat & side_support_mask,
    const cv::Mat & width_supported_ridge_mask,
    const cv::Mat & length_filtered_ridge_mask,
    const cv::Mat & reconstructed_mask,
    const cv::Mat & white_final_mask,
    const cv::Mat & green_mask,
    const cv::Mat & black_mask,
    const cv::Mat & noise_mask,
    const cv::Mat & debug_image)
  {
    bool displayed_any_window = false;
    if (morph_window_created_) {
      cv::imshow(kMorphWindow, morph_mask);
      resizeWindowToFitImage(kMorphWindow, morph_mask, display_max_width_, display_max_height_);
      displayed_any_window = true;
    }
    if (green_window_created_) {
      cv::imshow(kGreenWindow, green_mask);
      resizeWindowToFitImage(kGreenWindow, green_mask, display_max_width_, display_max_height_);
      displayed_any_window = true;
    }
    if (black_window_created_) {
      cv::imshow(kBlackWindow, black_mask);
      resizeWindowToFitImage(kBlackWindow, black_mask, display_max_width_, display_max_height_);
      displayed_any_window = true;
    }
    if (noise_window_created_) {
      cv::imshow(kNoiseWindow, noise_mask);
      resizeWindowToFitImage(kNoiseWindow, noise_mask, display_max_width_, display_max_height_);
      displayed_any_window = true;
    }
    if (ridge_window_created_) {
      cv::imshow(kRidgeWindow, ridge_mask);
      resizeWindowToFitImage(kRidgeWindow, ridge_mask, display_max_width_, display_max_height_);
      displayed_any_window = true;
    }
    if (orientation_valid_window_created_) {
      cv::imshow(kOrientationValidWindow, orientation_valid_mask);
      resizeWindowToFitImage(
        kOrientationValidWindow,
        orientation_valid_mask,
        display_max_width_,
        display_max_height_);
      displayed_any_window = true;
    }
    if (side_support_window_created_) {
      cv::imshow(kSideSupportWindow, side_support_mask);
      resizeWindowToFitImage(
        kSideSupportWindow,
        side_support_mask,
        display_max_width_,
        display_max_height_);
      displayed_any_window = true;
    }
    if (width_supported_window_created_) {
      cv::imshow(kWidthSupportedWindow, width_supported_ridge_mask);
      resizeWindowToFitImage(
        kWidthSupportedWindow,
        width_supported_ridge_mask,
        display_max_width_,
        display_max_height_);
      displayed_any_window = true;
    }
    if (length_filtered_ridge_window_created_) {
      cv::imshow(kLengthFilteredRidgeWindow, length_filtered_ridge_mask);
      resizeWindowToFitImage(
        kLengthFilteredRidgeWindow,
        length_filtered_ridge_mask,
        display_max_width_,
        display_max_height_);
      displayed_any_window = true;
    }
    if (reconstructed_window_created_) {
      cv::imshow(kReconstructedWindow, reconstructed_mask);
      resizeWindowToFitImage(
        kReconstructedWindow,
        reconstructed_mask,
        display_max_width_,
        display_max_height_);
      displayed_any_window = true;
    }
    if (white_final_mask_window_created_) {
      cv::imshow(kWhiteFinalMaskWindow, white_final_mask);
      resizeWindowToFitImage(
        kWhiteFinalMaskWindow,
        white_final_mask,
        display_max_width_,
        display_max_height_);
      displayed_any_window = true;
    }
    if (debug_window_created_ && !debug_image.empty()) {
      cv::imshow(kDebugWindow, debug_image);
      resizeWindowToFitImage(kDebugWindow, debug_image, display_max_width_, display_max_height_);
      displayed_any_window = true;
    }
    if (displayed_any_window) {
      cv::waitKey(1);
    }
    return displayed_any_window;
  }

  void maskCallback(
    const ImageMsg::ConstSharedPtr & morph_msg,
    const ImageMsg::ConstSharedPtr & green_msg,
    const ImageMsg::ConstSharedPtr & black_msg,
    const ImageMsg::ConstSharedPtr & noise_msg)
  {
    const bool previous_timing_debug = enable_timing_debug_;
    const int previous_timing_summary_interval = timing_summary_interval_;
    syncImageViewState();
    const bool timing_enabled = enable_timing_debug_;
    RidgeFrameTiming timing;
    const auto callback_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    auto stage_start = timing_enabled ? callback_start : TimePoint{};
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::RuntimeSync,
        stage_start,
        SteadyClock::now());
    }
    if (
      previous_timing_debug != enable_timing_debug_ ||
      previous_timing_summary_interval != timing_summary_interval_)
    {
      resetTimingSummary();
    }

    cv::Mat morph_mask;
    cv::Mat green_mask;
    cv::Mat black_mask;
    cv::Mat noise_mask;
    try {
      stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
      morph_mask = cv_bridge::toCvCopy(morph_msg, "mono8")->image;
      green_mask = cv_bridge::toCvCopy(green_msg, "mono8")->image;
      black_mask = cv_bridge::toCvCopy(black_msg, "mono8")->image;
      noise_mask = cv_bridge::toCvCopy(noise_msg, "mono8")->image;
      if (timing_enabled) {
        recordStageDuration(
          timing.stage_us,
          RidgeTimingStage::CvBridgeConvert,
          stage_start,
          SteadyClock::now());
      }
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge failed: %s", e.what());
      return;
    }

    if (
      morph_mask.size() != green_mask.size() || morph_mask.size() != black_mask.size() ||
      morph_mask.size() != noise_mask.size())
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "DT ridge filter received mismatched mask sizes.");
      return;
    }

    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    cv::Mat distance_transform;
    cv::distanceTransform(morph_mask, distance_transform, cv::DIST_L2, 3);
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::DistanceTransform,
        stage_start,
        SteadyClock::now());
    }

    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    const cv::Mat ridge_mask = extractDistanceTransformRidgeMask(morph_mask, distance_transform);
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::RidgeExtract,
        stage_start,
        SteadyClock::now());
    }

    cv::Mat orientation_valid_mask = cv::Mat::zeros(morph_mask.size(), CV_8UC1);
    cv::Mat side_support_mask = cv::Mat::zeros(morph_mask.size(), CV_8UC1);
    cv::Mat local_width_map = cv::Mat::zeros(morph_mask.size(), CV_32FC1);
    std::vector<float> supported_widths;
    supported_widths.reserve(static_cast<std::size_t>(cv::countNonZero(ridge_mask)));

    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    for (int y = 0; y < ridge_mask.rows; ++y) {
      for (int x = 0; x < ridge_mask.cols; ++x) {
        if (ridge_mask.at<uchar>(y, x) == 0) {
          continue;
        }

        const float local_width_px = 2.0F * distance_transform.at<float>(y, x);
        if (local_width_px <= 0.0F) {
          continue;
        }

        cv::Point2f tangent;
        cv::Point2f normal;
        if (!estimateOrientation(
            ridge_mask,
            cv::Point(x, y),
            orientation_window_radius_px_,
            min_orientation_neighbors_,
            tangent,
            normal))
        {
          continue;
        }

        orientation_valid_mask.at<uchar>(y, x) = 255;
        local_width_map.at<float>(y, x) = local_width_px;
        const SideSampleStats left_stats = sampleSideSupport(
          cv::Point(x, y),
          tangent,
          normal,
          local_width_px,
          -1,
          green_mask,
          black_mask,
          noise_mask,
          side_margin_px_,
          side_band_depth_px_);
        const SideSampleStats right_stats = sampleSideSupport(
          cv::Point(x, y),
          tangent,
          normal,
          local_width_px,
          1,
          green_mask,
          black_mask,
          noise_mask,
          side_margin_px_,
          side_band_depth_px_);

        const bool interior_supported =
          left_stats.greenRatio() >= min_green_ratio_ &&
          right_stats.greenRatio() >= min_green_ratio_;
        const bool boundary_supported =
          enable_boundary_mode_ &&
          ((left_stats.greenRatio() >= min_green_ratio_ &&
          right_stats.boundaryRatio() >= min_boundary_ratio_) ||
          (right_stats.greenRatio() >= min_green_ratio_ &&
          left_stats.boundaryRatio() >= min_boundary_ratio_));

        if (!interior_supported && !boundary_supported) {
          continue;
        }

        side_support_mask.at<uchar>(y, x) = 255;
        supported_widths.push_back(local_width_px);
      }
    }
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::SideSupportScan,
        stage_start,
        SteadyClock::now());
    }

    double width_lower_bound = width_floor_px_;
    double width_upper_bound = width_ceil_px_;
    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    if (static_cast<int>(supported_widths.size()) >= min_width_samples_) {
      const double width_median = computeMedian(supported_widths);
      std::vector<float> deviations;
      deviations.reserve(supported_widths.size());
      for (const float width : supported_widths) {
        deviations.push_back(static_cast<float>(std::abs(width - width_median)));
      }
      const double width_mad = computeMedian(deviations);
      width_lower_bound = std::max(width_floor_px_, width_median - width_mad_scale_ * width_mad);
      width_upper_bound = std::min(width_ceil_px_, width_median + width_mad_scale_ * width_mad);
      if (width_upper_bound < width_lower_bound) {
        width_lower_bound = width_floor_px_;
        width_upper_bound = width_ceil_px_;
      }
    }
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::WidthRangeEstimate,
        stage_start,
        SteadyClock::now());
    }

    cv::Mat width_supported_ridge_mask = cv::Mat::zeros(morph_mask.size(), CV_8UC1);
    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    for (int y = 0; y < side_support_mask.rows; ++y) {
      for (int x = 0; x < side_support_mask.cols; ++x) {
        if (side_support_mask.at<uchar>(y, x) == 0) {
          continue;
        }
        const float local_width_px = local_width_map.at<float>(y, x);
        if (local_width_px < width_lower_bound || local_width_px > width_upper_bound) {
          continue;
        }
        width_supported_ridge_mask.at<uchar>(y, x) = 255;
      }
    }
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::WidthFilter,
        stage_start,
        SteadyClock::now());
    }

    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    const cv::Mat length_filtered_ridge_mask =
      filterMaskByLength(width_supported_ridge_mask, min_skeleton_length_px_);
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::LengthFilter,
        stage_start,
        SteadyClock::now());
    }

    cv::Mat reconstructed_mask = cv::Mat::zeros(morph_mask.size(), CV_8UC1);
    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    for (int y = 0; y < length_filtered_ridge_mask.rows; ++y) {
      for (int x = 0; x < length_filtered_ridge_mask.cols; ++x) {
        if (length_filtered_ridge_mask.at<uchar>(y, x) == 0) {
          continue;
        }
        const float local_width_px = local_width_map.at<float>(y, x);
        const int radius = std::max(
          1,
          static_cast<int>(std::round(0.5 * static_cast<double>(local_width_px) + reconstruction_margin_px_)));
        cv::circle(
          reconstructed_mask,
          cv::Point(x, y),
          radius,
          cv::Scalar(255),
          cv::FILLED);
      }
    }
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::Reconstruct,
        stage_start,
        SteadyClock::now());
    }

    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    cv::Mat white_final_mask;
    cv::bitwise_and(reconstructed_mask, morph_mask, white_final_mask);
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::FinalMask,
        stage_start,
        SteadyClock::now());
    }
    const unsigned long long current_frame_index = ++frame_count_;

    const bool debug_image_needed =
      show_debug_image_ && (debug_window_created_ || hasSubscribers(debug_pub_));
    cv::Mat debug_image;
    if (debug_image_needed) {
      stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
      debug_image = createDebugComposite(green_mask, black_mask, white_final_mask);
      if (timing_enabled) {
        recordStageDuration(
          timing.stage_us,
          RidgeTimingStage::DebugComposite,
          stage_start,
          SteadyClock::now());
      }
    }

    stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
    const bool published_any = publishOutputs(
      morph_msg->header,
      ridge_mask,
      orientation_valid_mask,
      side_support_mask,
      width_supported_ridge_mask,
      length_filtered_ridge_mask,
      reconstructed_mask,
      white_final_mask,
      debug_image);
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::PublishOutputs,
        stage_start,
        SteadyClock::now());
    }

    const bool display_requested = anyImageWindowCreated();
    bool displayed_any = false;
    if (display_requested) {
      stage_start = timing_enabled ? SteadyClock::now() : TimePoint{};
      displayed_any = displayOutputs(
        morph_mask,
        ridge_mask,
        orientation_valid_mask,
        side_support_mask,
        width_supported_ridge_mask,
        length_filtered_ridge_mask,
        reconstructed_mask,
        white_final_mask,
        green_mask,
        black_mask,
        noise_mask,
        debug_image);
    }
    if (timing_enabled && displayed_any) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::GuiDisplay,
        stage_start,
        SteadyClock::now());
    }
    if (timing_enabled && !published_any) {
      timing.stage_us[static_cast<std::size_t>(RidgeTimingStage::PublishOutputs)] = 0;
    }
    if (timing_enabled) {
      recordStageDuration(
        timing.stage_us,
        RidgeTimingStage::CallbackTotal,
        callback_start,
        SteadyClock::now());
    }

    maybeLogTimingSummary(timing, current_frame_index);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WhiteLineDtRidgeFilterNode>());
  rclcpp::shutdown();
  return 0;
}
