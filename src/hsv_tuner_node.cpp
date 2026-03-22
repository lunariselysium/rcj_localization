#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

// Globals for trackbar access
int h_min = 0, s_min = 0, v_min = 180;
int h_max = 180, s_max = 40, v_max = 255;

void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv::Mat frame;
    try {
        frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("hsv_tuner"), "cv_bridge exception: %s", e.what());
        return;
    }

    // Read current trackbar positions
    h_min = cv::getTrackbarPos("H_min", "Trackbars");
    h_max = cv::getTrackbarPos("H_max", "Trackbars");
    s_min = cv::getTrackbarPos("S_min", "Trackbars");
    s_max = cv::getTrackbarPos("S_max", "Trackbars");
    v_min = cv::getTrackbarPos("V_min", "Trackbars");
    v_max = cv::getTrackbarPos("V_max", "Trackbars");

    // Process Image
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar lower(h_min, s_min, v_min);
    cv::Scalar upper(h_max, s_max, v_max);
    cv::inRange(hsv, lower, upper, mask);
    
    // Optional noise reduction
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Show images
    cv::imshow("Original Image", frame);
    cv::imshow("Result Mask", mask);
    cv::waitKey(1);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("hsv_tuner_node");

    // Create GUI windows and trackbars
    cv::namedWindow("Trackbars", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("H_min", "Trackbars", &h_min, 180);
    cv::createTrackbar("H_max", "Trackbars", &h_max, 180);
    cv::createTrackbar("S_min", "Trackbars", &s_min, 255);
    cv::createTrackbar("S_max", "Trackbars", &s_max, 255);
    cv::createTrackbar("V_min", "Trackbars", &v_min, 255);
    cv::createTrackbar("V_max", "Trackbars", &v_max, 255);

    auto sub = node->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10, imageCallback);
    
    RCLCPP_INFO(node->get_logger(), "HSV Tuner Started. Adjust sliders to isolate white lines.");

    rclcpp::spin(node);
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}