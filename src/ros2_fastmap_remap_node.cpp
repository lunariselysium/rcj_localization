#include <functional>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#elif __has_include(<cv_bridge/cv_bridge.h>)
#include <cv_bridge/cv_bridge.h>
#else
#error "cv_bridge header not found"
#endif
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class FastMapRemapNode : public rclcpp::Node
{
public:
    FastMapRemapNode()
        : Node("fastmap_remap_node")
    {
        const auto fastMapPath = declare_parameter<std::string>("fastmap_file", "/home/terry/RCJ/localization_ws/src/rcj_localization/config/undistort_map_20260413_115400_fast.xml");
        const auto inputTopic = declare_parameter<std::string>("input_topic", "/camera/image_raw");
        const auto outputTopic = declare_parameter<std::string>("output_topic", "/camera/image_remapped");
        const auto inputTransport = declare_parameter<std::string>("input_transport", "raw");
        const auto interpolation = declare_parameter<std::string>("interpolation", "linear");

        loadFastMaps(fastMapPath);
        interpolationMode_ = parseInterpolation(interpolation);

        publisher_ = image_transport::create_publisher(
            this,
            outputTopic,
            rmw_qos_profile_sensor_data);

        subscription_ = image_transport::create_subscription(
            this,
            inputTopic,
            std::bind(&FastMapRemapNode::imageCallback, this, std::placeholders::_1),
            inputTransport,
            rmw_qos_profile_sensor_data);

        RCLCPP_INFO(get_logger(), "Loaded fast maps from: %s", fastMapPath.c_str());
        RCLCPP_INFO(get_logger(), "Subscribed to: %s", inputTopic.c_str());
        RCLCPP_INFO(get_logger(), "Publishing to: %s", outputTopic.c_str());
        RCLCPP_INFO(get_logger(), "Input transport: %s", inputTransport.c_str());
        RCLCPP_INFO(get_logger(), "Interpolation: %s", interpolation.c_str());
    }

private:
    int parseInterpolation(const std::string& interpolation) const
    {
        if (interpolation == "nearest") {
            return cv::INTER_NEAREST;
        }
        if (interpolation == "linear") {
            return cv::INTER_LINEAR;
        }

        throw std::runtime_error(
            "Unsupported interpolation '" + interpolation +
            "'. Supported values: nearest, linear.");
    }

    void loadFastMaps(const std::string& fastMapPath)
    {
        if (fastMapPath.empty()) {
            throw std::runtime_error("Parameter 'fastmap_file' must be set.");
        }

        cv::FileStorage fs(fastMapPath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Cannot open fast map file: " + fastMapPath);
        }

        fs["source_width"] >> sourceWidth_;
        fs["source_height"] >> sourceHeight_;
        fs["fast_map_1"] >> fastMap1_;
        fs["fast_map_2"] >> fastMap2_;
        fs.release();

        if (fastMap1_.empty() || fastMap2_.empty()) {
            throw std::runtime_error("Fast map file is missing fast_map_1 or fast_map_2: " + fastMapPath);
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        try {
            const auto cvInput = cv_bridge::toCvShare(msg, msg->encoding);

            if (sourceWidth_ > 0 && sourceHeight_ > 0 &&
                (cvInput->image.cols != sourceWidth_ || cvInput->image.rows != sourceHeight_)) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    2000,
                    "Input image size %dx%d does not match fast map source size %dx%d. Frame dropped.",
                    cvInput->image.cols,
                    cvInput->image.rows,
                    sourceWidth_,
                    sourceHeight_);
                return;
            }

            const auto remapStart = std::chrono::steady_clock::now();
            cv::remap(
                cvInput->image,
                remappedImage_,
                fastMap1_,
                fastMap2_,
                interpolationMode_,
                cv::BORDER_CONSTANT);
            const auto remapEnd = std::chrono::steady_clock::now();

            const auto remapDurationUs =
                std::chrono::duration_cast<std::chrono::microseconds>(remapEnd - remapStart).count();
            ++frameCount_;
            totalRemapTimeUs_ += remapDurationUs;
            const double averageRemapUs =
                static_cast<double>(totalRemapTimeUs_) / static_cast<double>(frameCount_);

            auto outputMsg = cv_bridge::CvImage(msg->header, msg->encoding, remappedImage_).toImageMsg();
            publisher_.publish(*outputMsg);

            RCLCPP_INFO(
                get_logger(),
                "frame=%llu remap_us=%lld remap_ms=%.3f avg_remap_ms=%.3f",
                static_cast<unsigned long long>(frameCount_),
                static_cast<long long>(remapDurationUs),
                static_cast<double>(remapDurationUs) / 1000.0,
                averageRemapUs / 1000.0);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "cv_bridge error: %s",
                e.what());
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "OpenCV remap error: %s",
                e.what());
        }
    }

    image_transport::Subscriber subscription_;
    image_transport::Publisher publisher_;
    cv::Mat fastMap1_;
    cv::Mat fastMap2_;
    cv::Mat remappedImage_;
    int sourceWidth_ = 0;
    int sourceHeight_ = 0;
    int interpolationMode_ = cv::INTER_LINEAR;
    uint64_t frameCount_ = 0;
    int64_t totalRemapTimeUs_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<FastMapRemapNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("fastmap_remap_node"), "%s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
