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

struct MapMetadata
{
    int sourceWidth = 0;
    int sourceHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;

    cv::Size sourceSize() const
    {
        return {sourceWidth, sourceHeight};
    }

    cv::Size outputSize() const
    {
        return {outputWidth, outputHeight};
    }
};

struct RawMapData
{
    MapMetadata metadata;
    cv::Mat mapX;
    cv::Mat mapY;
};

struct CommandLineOptions
{
    std::filesystem::path rawMapPath;
    std::filesystem::path previewImagePath;
    std::filesystem::path outputPath;
};

std::string formatSize(int width, int height)
{
    return std::to_string(width) + "x" + std::to_string(height);
}

std::string describeMapMetadata(const MapMetadata& metadata)
{
    return formatSize(metadata.sourceWidth, metadata.sourceHeight) +
           " -> " +
           formatSize(metadata.outputWidth, metadata.outputHeight);
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

std::filesystem::path tryResolveExistingPath(const std::vector<std::filesystem::path>& candidates)
{
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
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

            const auto filename = pathToUtf8(entry.path().filename());
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
    return rawMapPath.parent_path() / (pathToUtf8(rawMapPath.stem()) + "_fast.xml");
}

CommandLineOptions parseCommandLineOptions(int argc, char** argv)
{
    CommandLineOptions options;

    for (int argIndex = 1; argIndex < argc; ++argIndex) {
        const std::string arg = argv[argIndex];

        auto consumeValue = [&](const char* optionName) -> std::filesystem::path {
            if (argIndex + 1 >= argc) {
                throw std::runtime_error("Missing value for option " + std::string(optionName));
            }

            ++argIndex;
            return std::filesystem::path(argv[argIndex]);
        };

        if (arg == "--map") {
            options.rawMapPath = consumeValue("--map");
            continue;
        }

        if (arg == "--preview") {
            options.previewImagePath = consumeValue("--preview");
            continue;
        }

        if (arg == "--output") {
            options.outputPath = consumeValue("--output");
            continue;
        }

        throw std::runtime_error(
            "Unknown argument '" + arg +
            "'. Supported options: --map <path> --preview <path> --output <path>.");
    }

    return options;
}

RawMapData loadRawMapData(const std::filesystem::path& rawMapPath)
{
    cv::FileStorage fs(pathToUtf8(rawMapPath), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Cannot read map file: " + pathToUtf8(rawMapPath));
    }

    RawMapData rawMapData;
    fs["source_width"] >> rawMapData.metadata.sourceWidth;
    fs["source_height"] >> rawMapData.metadata.sourceHeight;
    fs["output_width"] >> rawMapData.metadata.outputWidth;
    fs["output_height"] >> rawMapData.metadata.outputHeight;
    fs["map_x"] >> rawMapData.mapX;
    fs["map_y"] >> rawMapData.mapY;
    fs.release();

    if (rawMapData.mapX.empty() || rawMapData.mapY.empty()) {
        throw std::runtime_error("map_x or map_y is empty in " + pathToUtf8(rawMapPath));
    }

    if (rawMapData.mapX.size() != rawMapData.mapY.size()) {
        throw std::runtime_error("map_x and map_y size mismatch in " + pathToUtf8(rawMapPath));
    }

    if (rawMapData.metadata.sourceWidth <= 0 || rawMapData.metadata.sourceHeight <= 0) {
        throw std::runtime_error("Raw map is missing source_width/source_height: " + pathToUtf8(rawMapPath));
    }

    if (rawMapData.metadata.outputWidth <= 0) {
        rawMapData.metadata.outputWidth = rawMapData.mapX.cols;
    }
    if (rawMapData.metadata.outputHeight <= 0) {
        rawMapData.metadata.outputHeight = rawMapData.mapX.rows;
    }

    if (rawMapData.metadata.outputWidth != rawMapData.mapX.cols ||
        rawMapData.metadata.outputHeight != rawMapData.mapX.rows) {
        throw std::runtime_error(
            "Raw map output_width/output_height does not match map_x/map_y matrix size: " +
            pathToUtf8(rawMapPath));
    }

    return rawMapData;
}

bool tryLoadValidatedFastMaps(
    const std::filesystem::path& fastMapPath,
    const MapMetadata& expectedMetadata,
    cv::Mat& fastMap1,
    cv::Mat& fastMap2,
    std::string& validationMessage)
{
    if (!std::filesystem::exists(fastMapPath)) {
        validationMessage = "Fast map file does not exist yet.";
        return false;
    }

    cv::FileStorage fs(pathToUtf8(fastMapPath), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        validationMessage = "Cannot open existing fast map file.";
        return false;
    }

    int sourceWidth = 0;
    int sourceHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    fs["source_width"] >> sourceWidth;
    fs["source_height"] >> sourceHeight;
    fs["output_width"] >> outputWidth;
    fs["output_height"] >> outputHeight;
    fs["fast_map_1"] >> fastMap1;
    fs["fast_map_2"] >> fastMap2;
    fs.release();

    if (fastMap1.empty() || fastMap2.empty()) {
        validationMessage = "Existing fast map is missing fast_map_1 or fast_map_2.";
        return false;
    }

    if (sourceWidth != expectedMetadata.sourceWidth ||
        sourceHeight != expectedMetadata.sourceHeight) {
        validationMessage =
            "Existing fast map source metadata " + formatSize(sourceWidth, sourceHeight) +
            " does not match raw map source metadata " +
            formatSize(expectedMetadata.sourceWidth, expectedMetadata.sourceHeight) + ".";
        return false;
    }

    if (fastMap1.size() != expectedMetadata.outputSize()) {
        validationMessage =
            "Existing fast_map_1 output size " + formatSize(fastMap1.cols, fastMap1.rows) +
            " does not match raw map output size " +
            formatSize(expectedMetadata.outputWidth, expectedMetadata.outputHeight) + ".";
        return false;
    }

    if (fastMap2.size() != expectedMetadata.outputSize()) {
        validationMessage =
            "Existing fast_map_2 output size " + formatSize(fastMap2.cols, fastMap2.rows) +
            " does not match raw map output size " +
            formatSize(expectedMetadata.outputWidth, expectedMetadata.outputHeight) + ".";
        return false;
    }

    if ((outputWidth > 0 && outputWidth != expectedMetadata.outputWidth) ||
        (outputHeight > 0 && outputHeight != expectedMetadata.outputHeight)) {
        validationMessage =
            "Existing fast map output metadata " + formatSize(outputWidth, outputHeight) +
            " does not match raw map output metadata " +
            formatSize(expectedMetadata.outputWidth, expectedMetadata.outputHeight) + ".";
        return false;
    }

    validationMessage = "Existing fast map metadata is valid.";
    return true;
}

void saveFastMaps(
    const std::filesystem::path& fastMapPath,
    const cv::Mat& fastMap1,
    const cv::Mat& fastMap2,
    const MapMetadata& metadata)
{
    cv::FileStorage fs(pathToUtf8(fastMapPath), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("Cannot write fast map file: " + pathToUtf8(fastMapPath));
    }

    fs << "source_width" << metadata.sourceWidth;
    fs << "source_height" << metadata.sourceHeight;
    fs << "output_width" << metadata.outputWidth;
    fs << "output_height" << metadata.outputHeight;
    fs << "fast_map_1" << fastMap1;
    fs << "fast_map_2" << fastMap2;
    fs.release();
}

int main(int argc, char** argv)
{
    try {
        const CommandLineOptions options = parseCommandLineOptions(argc, argv);
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

        const std::filesystem::path mapPath = options.rawMapPath.empty()
            ? resolveLatestMapPath(mapSearchDirs)
            : resolveExistingPath({options.rawMapPath}, "raw XML map");
        const std::filesystem::path fastMapPath = deriveFastMapPath(mapPath);
        const std::filesystem::path outputPath = options.outputPath.empty()
            ? cwd / "undistorted.png"
            : options.outputPath;
        const std::filesystem::path previewImagePath = options.previewImagePath.empty()
            ? tryResolveExistingPath({
                cwd / "../SVGA5.bmp",
                cwd / "../../SVGA5.bmp",
                cwd / "../../../SVGA5.bmp"
            })
            : resolveExistingPath({options.previewImagePath}, "preview image");

        std::cout << "Current working directory: " << pathToUtf8(cwd) << std::endl;
        std::cout << "Map path: " << pathToUtf8(mapPath) << std::endl;
        std::cout << "Fast map path: " << pathToUtf8(fastMapPath) << std::endl;
        if (!previewImagePath.empty()) {
            std::cout << "Preview image path: " << pathToUtf8(previewImagePath) << std::endl;
        } else {
            std::cout << "Preview image path: <not found>" << std::endl;
        }

        const RawMapData rawMapData = loadRawMapData(mapPath);
        std::cout << "Raw map metadata: "
                  << describeMapMetadata(rawMapData.metadata)
                  << std::endl;
        std::cout << "map_x size: "
                  << rawMapData.mapX.cols << " x " << rawMapData.mapX.rows
                  << std::endl;
        std::cout << "map_y size: "
                  << rawMapData.mapY.cols << " x " << rawMapData.mapY.rows
                  << std::endl;

        cv::Mat fastMap1;
        cv::Mat fastMap2;
        std::string validationMessage;

        if (tryLoadValidatedFastMaps(
                fastMapPath,
                rawMapData.metadata,
                fastMap1,
                fastMap2,
                validationMessage)) {
            std::cout << "Loaded precomputed fast maps: "
                      << describeMapMetadata(rawMapData.metadata)
                      << std::endl;
        } else {
            std::cout << "Rebuilding fast maps because: " << validationMessage << std::endl;
            cv::convertMaps(rawMapData.mapX, rawMapData.mapY, fastMap1, fastMap2, CV_16SC2);
            saveFastMaps(fastMapPath, fastMap1, fastMap2, rawMapData.metadata);
            std::cout << "Saved precomputed fast maps: "
                      << describeMapMetadata(rawMapData.metadata)
                      << std::endl;
        }

        if (previewImagePath.empty()) {
            std::cout << "No preview image found. Fast map generation complete." << std::endl;
            return 0;
        }

        cv::Mat src = cv::imread(pathToUtf8(previewImagePath), cv::IMREAD_COLOR);
        if (src.empty()) {
            std::cerr << "Cannot read preview image: " << pathToUtf8(previewImagePath) << std::endl;
            return 1;
        }

        if (src.size() != rawMapData.metadata.sourceSize()) {
            std::cout << "Skipping preview remap: preview image size "
                      << formatSize(src.cols, src.rows)
                      << " does not match raw map source size "
                      << formatSize(rawMapData.metadata.sourceWidth, rawMapData.metadata.sourceHeight)
                      << "." << std::endl;
            return 0;
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
