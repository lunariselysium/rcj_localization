#include "rcj_localization/white_line_preprocessing.hpp"

#include "rcj_localization/white_line_debug_utils.hpp"

#include <algorithm>
#include <vector>

namespace rcj_loc::vision::white_line {

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

PreprocessResult preprocessFrame(const cv::Mat &frame, const PreprocessParams &params) {
    PreprocessResult result;

    const int median_kernel = rcj_loc::vision::debug::makeOdd(params.median_kernel, 1);
    if (median_kernel > 1) {
        cv::medianBlur(frame, result.filtered_bgr, median_kernel);
    } else {
        result.filtered_bgr = frame.clone();
    }

    cv::Mat lab;
    cv::cvtColor(result.filtered_bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);
    result.lab_l = lab_channels[0];
    result.lab_a = lab_channels[1];
    result.lab_b = lab_channels[2];

    if (params.use_lab_l) {
        result.luminance = result.lab_l;
    } else {
        cv::cvtColor(result.filtered_bgr, result.luminance, cv::COLOR_BGR2GRAY);
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        params.clahe_clip_limit,
        cv::Size(std::max(1, params.clahe_tile_size), std::max(1, params.clahe_tile_size)));
    clahe->apply(result.luminance, result.enhanced);

    const int blur_kernel = rcj_loc::vision::debug::makeOdd(params.blur_kernel, 1);
    if (blur_kernel > 1) {
        cv::GaussianBlur(result.enhanced, result.enhanced, cv::Size(blur_kernel, blur_kernel), 0.0);
    }

    if (params.sharpen_amount > 0.0) {
        cv::Mat blurred_for_sharpen;
        cv::GaussianBlur(result.enhanced, blurred_for_sharpen, cv::Size(0, 0), 1.0);
        cv::addWeighted(
            result.enhanced,
            1.0 + params.sharpen_amount,
            blurred_for_sharpen,
            -params.sharpen_amount,
            0.0,
            result.enhanced);
    }

    return result;
}

}  // namespace rcj_loc::vision::white_line
