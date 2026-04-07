#include "rcj_localization/vision_processor.hpp"

namespace rcj_loc::vision {

std::vector<Point2D> VisionProcessor::extractFieldLines(cv::Mat &img) {
    std::vector<Point2D> local_points;

    // 1. Horizon Crop (Ignore the top part of the image where the field isn't)
    int horizon_y = img.rows / 10;
    cv::Rect roi(0, horizon_y, img.cols, img.rows - horizon_y);
    cv::Mat cropped_img = img(roi);

    // 2. Convert to HSV
    cv::Mat hsv;
    cv::cvtColor(cropped_img, hsv, cv::COLOR_BGR2HSV);

    // 3. Threshold for lines
    cv::Mat white_mask;
    cv::Scalar lower_white(0, 0, 158);
    cv::Scalar upper_white(180, 180, 205);
    cv::inRange(hsv, lower_white, upper_white, white_mask);

    cv::Mat black_mask;
    cv::inRange(hsv, cv::Scalar(0, 10, 42), cv::Scalar(180, 255, 102), black_mask);

    cv::Mat feature_mask;
    cv::bitwise_or(white_mask, black_mask, feature_mask);

    // Remove tiny noise pixels
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(feature_mask, feature_mask, cv::MORPH_OPEN, kernel);

    // Blob / contour filtering
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(feature_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat filtered_mask = cv::Mat::zeros(feature_mask.size(), CV_8UC1);

    double min_area = 50.0;
    double max_area = 80000.0;
    for (const auto &contour : contours) {
        double area = cv::contourArea(contour);
        if (area > min_area && area < max_area) {
            cv::drawContours(filtered_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 4. Extract points and apply IPM (Inverse Perspective Mapping)
    int rows = filtered_mask.rows;
    int cols = filtered_mask.cols;

    for (int v_cropped = 0; v_cropped < rows; v_cropped++) {
        const uchar *row_ptr = filtered_mask.ptr<uchar>(v_cropped);

        for (int u = 0; u < cols; u++) {
            if (row_ptr[u] != 255) {
                continue;
            }

            int v = v_cropped + horizon_y;

            double ray_angle_y = atan((v - cy) / fy);
            if (camera_pitch + ray_angle_y <= 0) {
                continue;
            }

            double distance_x = camera_height / tan(camera_pitch + ray_angle_y);
            double distance_y = distance_x * (cx - u) / fx;
            local_points.push_back({distance_x, distance_y});
        }
    }

    // Debug visualization: paint detected pixels green on the output frame
    cv::Mat green_color(cropped_img.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    green_color.copyTo(cropped_img, filtered_mask);

    if (local_points.size() > 5000) {
        std::vector<Point2D> decimated_points;
        int step = local_points.size() / 5000;
        for (size_t i = 0; i < local_points.size(); i += step) {
            decimated_points.push_back(local_points[i]);
            if (decimated_points.size() >= 5000) {
                break;
            }
        }
        return decimated_points;
    }

    return local_points;
}

}  // namespace rcj_loc::vision
