#include "rcj_localization/white_line_debug_utils.hpp"

#include <algorithm>
namespace rcj_loc::vision::debug {

int makeOdd(int value, int minimum) {
    int clamped = std::max(value, minimum);
    if (clamped % 2 == 0) {
        ++clamped;
    }
    return clamped;
}

cv::Mat filterComponentsByStats(const cv::Mat &binary, const ComponentStatsFilter &filter) {
    CV_Assert(binary.type() == CV_8UC1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered = cv::Mat::zeros(binary.size(), CV_8UC1);
    for (int label = 1; label < stats.rows; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int major_axis = std::max(width, height);
        const int minor_axis = std::min(width, height);
        const double aspect_ratio = static_cast<double>(major_axis) / std::max(1, minor_axis);

        if (area < filter.min_area || area > filter.max_area) {
            continue;
        }
        if (major_axis < filter.min_major_axis || minor_axis > filter.max_minor_axis) {
            continue;
        }
        if (aspect_ratio < filter.min_aspect_ratio) {
            continue;
        }

        filtered.setTo(255, labels == label);
    }

    return filtered;
}

cv::Mat createMaskOverlay(const cv::Mat &frame, const cv::Mat &mask, const cv::Scalar &color, double alpha) {
    CV_Assert(frame.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8UC1);

    cv::Mat overlay = frame.clone();
    cv::Mat solid(frame.size(), CV_8UC3, color);
    cv::Mat blended;
    cv::addWeighted(frame, 1.0 - alpha, solid, alpha, 0.0, blended);
    blended.copyTo(overlay, mask);
    return overlay;
}

}  // namespace rcj_loc::vision::debug
