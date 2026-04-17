#include "rcj_localization/white_line_preprocessing.hpp"

#include "rcj_localization/white_line_debug_utils.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

namespace rcj_loc::vision::white_line {

namespace {

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

long long elapsedUs(const TimePoint &start, const TimePoint &end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void recordStage(
    PreprocessTiming *timing,
    PreprocessTimingStage stage,
    const TimePoint &start,
    const TimePoint &end) {
    if (timing == nullptr) {
        return;
    }
    timing->stage_us[static_cast<std::size_t>(stage)] = elapsedUs(start, end);
}

}  // namespace

void declarePreprocessParameters(rclcpp::Node &node) {
    node.declare_parameter("median_kernel", 3);
    node.declare_parameter("use_lab_l", true);
    node.declare_parameter("clahe_clip_limit", 2.5);
    node.declare_parameter("clahe_tile_size", 8);
    node.declare_parameter("blur_kernel", 3);
    node.declare_parameter("sharpen_amount", 0.0);
}

PreprocessParams getPreprocessParams(const rclcpp::Node &node) {
    PreprocessParams params;
    params.median_kernel = static_cast<int>(node.get_parameter("median_kernel").as_int());
    params.use_lab_l = node.get_parameter("use_lab_l").as_bool();
    params.clahe_clip_limit = node.get_parameter("clahe_clip_limit").as_double();
    params.clahe_tile_size = static_cast<int>(node.get_parameter("clahe_tile_size").as_int());
    params.blur_kernel = static_cast<int>(node.get_parameter("blur_kernel").as_int());
    params.sharpen_amount = node.get_parameter("sharpen_amount").as_double();
    return params;
}

PreprocessResult preprocessFrame(
    const cv::Mat &frame,
    const PreprocessParams &params,
    PreprocessTiming *timing) {
    PreprocessResult result;
    if (timing != nullptr) {
        timing->stage_us.fill(0);
    }

    const int median_kernel = rcj_loc::vision::debug::makeOdd(params.median_kernel, 1);
    if (median_kernel > 1) {
        const auto stage_start = SteadyClock::now();
        cv::medianBlur(frame, result.filtered_bgr, median_kernel);
        recordStage(timing, PreprocessTimingStage::MedianBlur, stage_start, SteadyClock::now());
    } else {
        const auto stage_start = SteadyClock::now();
        result.filtered_bgr = frame.clone();
        recordStage(timing, PreprocessTimingStage::MedianBlur, stage_start, SteadyClock::now());
    }

    cv::Mat lab;
    auto stage_start = SteadyClock::now();
    cv::cvtColor(result.filtered_bgr, lab, cv::COLOR_BGR2Lab);
    recordStage(timing, PreprocessTimingStage::BgrToLab, stage_start, SteadyClock::now());

    std::vector<cv::Mat> lab_channels;
    stage_start = SteadyClock::now();
    cv::split(lab, lab_channels);
    recordStage(timing, PreprocessTimingStage::Split, stage_start, SteadyClock::now());
    result.lab_l = lab_channels[0];
    result.lab_a = lab_channels[1];
    result.lab_b = lab_channels[2];

    stage_start = SteadyClock::now();
    if (params.use_lab_l) {
        result.luminance = result.lab_l;
    } else {
        cv::cvtColor(result.filtered_bgr, result.luminance, cv::COLOR_BGR2GRAY);
    }
    recordStage(
        timing,
        PreprocessTimingStage::LuminanceSelect,
        stage_start,
        SteadyClock::now());

    stage_start = SteadyClock::now();
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        params.clahe_clip_limit,
        cv::Size(std::max(1, params.clahe_tile_size), std::max(1, params.clahe_tile_size)));
    recordStage(timing, PreprocessTimingStage::ClaheCreate, stage_start, SteadyClock::now());

    stage_start = SteadyClock::now();
    clahe->apply(result.luminance, result.enhanced);
    recordStage(timing, PreprocessTimingStage::ClaheApply, stage_start, SteadyClock::now());

    const int blur_kernel = rcj_loc::vision::debug::makeOdd(params.blur_kernel, 1);
    if (blur_kernel > 1) {
        stage_start = SteadyClock::now();
        cv::GaussianBlur(result.enhanced, result.enhanced, cv::Size(blur_kernel, blur_kernel), 0.0);
        recordStage(
            timing,
            PreprocessTimingStage::GaussianBlur,
            stage_start,
            SteadyClock::now());
    }

    if (params.sharpen_amount > 0.0) {
        cv::Mat blurred_for_sharpen;
        stage_start = SteadyClock::now();
        cv::GaussianBlur(result.enhanced, blurred_for_sharpen, cv::Size(0, 0), 1.0);
        cv::addWeighted(
            result.enhanced,
            1.0 + params.sharpen_amount,
            blurred_for_sharpen,
            -params.sharpen_amount,
            0.0,
            result.enhanced);
        recordStage(timing, PreprocessTimingStage::Sharpen, stage_start, SteadyClock::now());
    }

    return result;
}

}  // namespace rcj_loc::vision::white_line
