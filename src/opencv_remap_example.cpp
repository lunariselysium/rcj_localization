#include <iostream>
#include <filesystem>
#include <exception>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

std::string pathToUtf8(const std::filesystem::path& path)
{
    return path.u8string();
}

std::filesystem::path resolveExistingPath(
    const std::vector<std::filesystem::path>& candidates,
    const char* label)
{
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    std::string message = "Cannot locate ";
    message += label;
    throw std::runtime_error(message);
}

std::filesystem::path resolveLatestMapPath(
    const std::vector<std::filesystem::path>& candidateDirs)
{
    std::filesystem::path bestPath;
    std::filesystem::file_time_type bestTime{};

    for (const auto& dir : candidateDirs) {
        if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
            continue;
        }

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const auto filename = entry.path().filename().string();
            if (filename.rfind("undistort_map_", 0) != 0 ||
                entry.path().extension() != ".xml" ||
                filename.find("_fast") != std::string::npos) {
                continue;
            }

            const auto writeTime = entry.last_write_time();
            if (bestPath.empty() || writeTime > bestTime) {
                bestPath = entry.path();
                bestTime = writeTime;
            }
        }
    }

    if (!bestPath.empty()) {
        return bestPath;
    }

    throw std::runtime_error("Cannot locate XML map file");
}

std::filesystem::path deriveFastMapPath(const std::filesystem::path& rawMapPath)
{
    return rawMapPath.parent_path() / (rawMapPath.stem().string() + "_fast.xml");
}

bool tryLoadFastMaps(
    const std::filesystem::path& fastMapPath,
    cv::Mat& fastMap1,
    cv::Mat& fastMap2)
{
    if (!std::filesystem::exists(fastMapPath)) {
        return false;
    }

    cv::FileStorage fs(pathToUtf8(fastMapPath), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    fs["fast_map_1"] >> fastMap1;
    fs["fast_map_2"] >> fastMap2;
    fs.release();

    return !fastMap1.empty() && !fastMap2.empty();
}

void saveFastMaps(
    const std::filesystem::path& fastMapPath,
    const cv::Mat& fastMap1,
    const cv::Mat& fastMap2,
    const cv::Size& sourceSize)
{
    cv::FileStorage fs(pathToUtf8(fastMapPath), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("Cannot write fast map file: " + pathToUtf8(fastMapPath));
    }

    fs << "source_width" << sourceSize.width;
    fs << "source_height" << sourceSize.height;
    fs << "fast_map_1" << fastMap1;
    fs << "fast_map_2" << fastMap2;
    fs.release();
}

int main()
{
    try {
        const std::filesystem::path cwd = std::filesystem::current_path();
        std::vector<std::filesystem::path> mapSearchDirs;
        std::filesystem::path current = cwd;
        for (int depth = 0; depth < 4; ++depth) {
            mapSearchDirs.push_back(current);
            mapSearchDirs.push_back(current / "undistortGetMapping");

            if (!current.has_parent_path()) {
                break;
            }
            current = current.parent_path();
        }

        const std::filesystem::path imagePath = resolveExistingPath({
            cwd / "../arena2.bmp",
            cwd / "../../arena2.bmp",
            cwd / "../../../arena2.bmp"
        }, "input image");
        const std::filesystem::path mapPath = resolveLatestMapPath(mapSearchDirs);
        const std::filesystem::path fastMapPath = deriveFastMapPath(mapPath);
        const std::filesystem::path outputPath = cwd / "undistorted.png";

        std::cout << "Current working directory: " << pathToUtf8(cwd) << std::endl;
        std::cout << "Image path: " << pathToUtf8(imagePath) << std::endl;
        std::cout << "Map path: " << pathToUtf8(mapPath) << std::endl;
        std::cout << "Fast map path: " << pathToUtf8(fastMapPath) << std::endl;

        cv::Mat src = cv::imread(pathToUtf8(imagePath), cv::IMREAD_COLOR);
        if (src.empty()) {
            std::cerr << "Cannot read image: " << pathToUtf8(imagePath) << std::endl;
            return 1;
        }

        cv::Mat fastMap1;
        cv::Mat fastMap2;

        if (tryLoadFastMaps(fastMapPath, fastMap1, fastMap2)) {
            std::cout << "Loaded precomputed fast maps." << std::endl;
        } else {
            cv::FileStorage fs(pathToUtf8(mapPath), cv::FileStorage::READ);
            if (!fs.isOpened()) {
                std::cerr << "Cannot read map file: " << pathToUtf8(mapPath) << std::endl;
                return 1;
            }

            cv::Mat mapX;
            cv::Mat mapY;
            fs["map_x"] >> mapX;
            fs["map_y"] >> mapY;
            fs.release();

            std::cout << "map_x size: " << mapX.cols << " x " << mapX.rows << std::endl;
            std::cout << "map_y size: " << mapY.cols << " x " << mapY.rows << std::endl;

            if (mapX.empty() || mapY.empty()) {
                std::cerr << "map_x or map_y is empty." << std::endl;
                return 1;
            }

            // Convert once at startup if the same map is reused frame after frame.
            cv::convertMaps(mapX, mapY, fastMap1, fastMap2, CV_16SC2);
            saveFastMaps(fastMapPath, fastMap1, fastMap2, src.size());
            std::cout << "Saved precomputed fast maps." << std::endl;
        }

        cv::Mat dst;
        cv::remap(src, dst, fastMap1, fastMap2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        if (!cv::imwrite(pathToUtf8(outputPath), dst)) {
            std::cerr << "Failed to write output image: " << pathToUtf8(outputPath) << std::endl;
            return 1;
        }

        std::cout << "Saved output image: " << pathToUtf8(outputPath) << std::endl;
        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
