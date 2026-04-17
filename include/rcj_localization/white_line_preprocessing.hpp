#pragma once

#include <array>
#include <cstddef>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

namespace rcj_loc::vision::white_line {

enum class PreprocessTimingStage : std::size_t {
    MedianBlur = 0,
    BgrToLab,
    Split,
    LuminanceSelect,
    ClaheCreate,
    ClaheApply,
    GaussianBlur,
    Sharpen,
    Count,
};

constexpr std::size_t kPreprocessTimingStageCount =
    static_cast<std::size_t>(PreprocessTimingStage::Count);

struct PreprocessParams {
    int median_kernel = 3;
    bool use_lab_l = true;
    double clahe_clip_limit = 2.5;
    int clahe_tile_size = 8;
    int blur_kernel = 3;
    double sharpen_amount = 0.0;
};

struct PreprocessResult {
    cv::Mat filtered_bgr;
    cv::Mat lab_l;
    cv::Mat lab_a;
    cv::Mat lab_b;
    cv::Mat luminance;
    cv::Mat enhanced;
};

struct PreprocessTiming {
    std::array<long long, kPreprocessTimingStageCount> stage_us{};
};

void declarePreprocessParameters(rclcpp::Node &node);
PreprocessParams getPreprocessParams(const rclcpp::Node &node);
PreprocessResult preprocessFrame(
    const cv::Mat &frame,
    const PreprocessParams &params,
    PreprocessTiming *timing = nullptr);

}  // namespace rcj_loc::vision::white_line
