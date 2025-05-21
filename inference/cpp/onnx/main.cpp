#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include "RF_DETR_ONNX.h"

// Generates random colors for classes (for visualization)
std::vector<cv::Scalar> generateClassColors(int num_classes) {
    std::vector<cv::Scalar> class_colors(num_classes);
    std::srand(42); // Use a fixed seed for consistent colors
    for (int i = 0; i < num_classes; ++i) {
        class_colors[i] = cv::Scalar(std::rand() % 256, std::rand() % 256, std::rand() % 256);
    }
    return class_colors;
}

// Load class names from a file
std::vector<std::string> loadClassNames(const std::string& path, int defaultNumClasses) {
    std::vector<std::string> classNames;
    std::ifstream ifs(path);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            classNames.push_back(line);
        }
        ifs.close();
        std::cout << "[INFO] Loaded " << classNames.size() << " class names from " << path << std::endl;
    }
    else {
        std::cerr << "[WARNING] Could not open class name file: " << path << ". Generating dummy labels." << std::endl;
    }

    if (classNames.empty()) {
        std::cout << "[INFO] Generating " << defaultNumClasses << " dummy class labels (0, 1, 2...)." << std::endl;
        for (int i = 0; i < defaultNumClasses; ++i) {
            classNames.push_back(std::to_string(i));
        }
    }
    return classNames;
}

// --- Drawing Function ---
void drawDetections(
    cv::Mat& image, // Input image (will be modified)
    const std::vector<Detection>& detections,
    const std::vector<std::string>& classNames,
    const std::vector<cv::Scalar>& classColors)
{
    if (classNames.empty()) {
        std::cerr << "Warning: classNames is empty. Cannot draw labels." << std::endl;
        return;
    }
    if (classColors.empty() || classColors.size() < classNames.size()) {
        std::cerr << "Warning: classColors is empty or insufficient. Cannot draw boxes with distinct colors." << std::endl;
        return; // Or use a default color
    }

    for (const auto& det : detections) {
        if (det.class_id < 0 || det.class_id >= classNames.size()) {
            std::cerr << "Warning: Invalid class_id " << det.class_id << " encountered. Skipping box." << std::endl;
            continue;
        }

        cv::Rect box = det.box;
        cv::Scalar color = classColors[det.class_id];
        int classId = det.class_id;
        float score = det.score;

        // Draw bounding box
        cv::rectangle(image, box, color, 2);

        // Create label text
        std::ostringstream oss;
        oss << classNames[classId] << ": " << std::fixed << std::setprecision(2) << score;
        std::string labelText = oss.str();

        // Add text label background and text
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        baseline += 1; // Adjust baseline

        // Ensure text background doesn't go out of bounds (top)
        int textRectY = std::max(box.y - textSize.height - baseline, 0);
        cv::Rect textRect(box.x, textRectY, textSize.width, textSize.height + baseline);

        // Draw filled rectangle for text background
        cv::rectangle(image, textRect, color, cv::FILLED);

        // Put white text on the background
        cv::Point textOrg(box.x, box.y - baseline); // Adjust text position slightly above the box
        // Ensure text org y isn't negative
        textOrg.y = std::max(textOrg.y, textSize.height);
        cv::putText(image, labelText, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

int main(int argc, char* argv[]) {
    // Print ONNX Runtime version using the C API (guaranteed to exist)
    std::cout << "[INFO] ONNXRuntime version: " << OrtGetApiBase()->GetVersionString() << std::endl;


    // --- Configuration Defaults ---
    //std::string modelPath = "rf-detr-base.onnx"; // Provide a sensible default or require via arg
    std::string labelPath = ""; // Default: No labels file, will use generic names
    std::string sourceType = "camera"; // Default source: image
    std::string inputPath = "0"; // Default input image path
    std::string outputPath = "output.jpg"; // Default output path
    float confThreshold = 0.5f;
    bool useCUDA = true;        // Default to CUDA if available
    size_t deviceId = 0;
    int cameraId = 0;
    int intraOpNumThreads = 1; // Default OpenMP threads for ONNX Runtime CPU ops

    // --- Argument Parsing ---
    std::cout << "[INFO] Parsing command line arguments..." << std::endl;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try { // Add try-catch for parsing values
            if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
                modelPath = argv[++i];
            }
            else if ((arg == "--labels" || arg == "-l") && i + 1 < argc) {
                labelPath = argv[++i];
            }
            else if (arg == "--source_type" && i + 1 < argc) {
                sourceType = argv[++i];
                std::transform(sourceType.begin(), sourceType.end(), sourceType.begin(), ::tolower); // Lowercase
            }
            else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
                inputPath = argv[++i]; // Path for image/video, ID for camera
            }
            else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
                outputPath = argv[++i]; // Path for saving image/video
            }
            else if ((arg == "--conf" || arg == "-c") && i + 1 < argc) {
                confThreshold = std::stof(argv[++i]);
            }
            else if (arg == "--use_cuda") {
                useCUDA = true;
            }
            else if (arg == "--use_cpu") {
                useCUDA = false;
            }
            else if (arg == "--device_id" && i + 1 < argc) {
                deviceId = std::stoul(argv[++i]);
            }
            else if (arg == "--threads" && i + 1 < argc) {
                intraOpNumThreads = std::stoi(argv[++i]);
                if (intraOpNumThreads <= 0) intraOpNumThreads = 1; // Ensure at least 1 thread
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "\nUsage: " << argv[0] << " [options]\n\n"
                    << "Options:\n"
                    << "  -m, --model <path>         Path to the ONNX model file (default: " << modelPath << ")\n"
                    << "  -l, --labels <path>        Path to the file containing class names (one per line)\n"
                    << "  --source_type <type>   Input source type: 'image', 'video', or 'camera' (default: " << sourceType << ")\n"
                    << "  -i, --input <value>        Path to image/video file, or camera ID (integer) (default: " << inputPath << ")\n"
                    << "  -o, --output <path>        Path to save the annotated image or video (default: " << outputPath << ")\n"
                    << "  -c, --conf <threshold>     Confidence threshold for detections (default: " << confThreshold << ")\n"
                    << "  --use_cuda             Enable CUDA execution provider (default: " << (useCUDA ? "Yes" : "No") << ")\n"
                    << "  --use_cpu              Use CPU execution provider\n"
                    << "  --device_id <id>       GPU device ID if using CUDA (default: " << deviceId << ")\n"
                    << "  --threads <num>        Number of threads for ONNX Runtime intra-op parallelism (CPU, default: " << intraOpNumThreads << ")\n"
                    << "  -h, --help             Show this help message\n" << std::endl;
                return 0;
            }
            else {
                std::cerr << "[WARNING] Unknown or incomplete argument: " << arg << ". Use --help for options." << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Invalid argument value for " << arg << ": " << e.what() << std::endl;
            return 1;
        }
    }

    // Update camera ID if source type is camera
    if (sourceType == "camera") {
        try {
            cameraId = std::stoi(inputPath);
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] Invalid camera ID '" << inputPath << "'. Please provide an integer ID. " << e.what() << std::endl;
            return 1;
        }
    }


    std::cout << "[INFO] Configuration:" << std::endl;
    std::cout << "  Model Path: " << modelPath << std::endl;
    std::cout << "  Labels Path: " << (labelPath.empty() ? "None (using defaults)" : labelPath) << std::endl;
    std::cout << "  Source Type: " << sourceType << std::endl;
    std::cout << "  Input: " << inputPath << (sourceType == "camera" ? " (Camera ID)" : "") << std::endl;
    if (sourceType == "image" || sourceType == "video") {
        std::cout << "  Output Path: " << outputPath << std::endl;
    }
    std::cout << "  Confidence Threshold: " << confThreshold << std::endl;
    std::cout << "  Execution Provider: " << (useCUDA ? "CUDA (Device ID: " + std::to_string(deviceId) + ")" : "CPU") << std::endl;
    if (!useCUDA) {
        std::cout << "  Intra-op Threads: " << intraOpNumThreads << std::endl;
    }

    try {
        // --- Initialize Detector ---
        std::cout << "[INFO] Initializing detector..." << std::endl;
        auto detector_start = std::chrono::high_resolution_clock::now();
        // Create the detector object using the class
        RF_DETR_ONNX detector(modelPath, useCUDA, deviceId, intraOpNumThreads);
        auto detector_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detector_init_duration = detector_end - detector_start;
        std::cout << "[INFO] Detector initialization time: " << detector_init_duration.count() << " ms" << std::endl;

        // --- Load Class Names & Generate Colors ---
        int numClasses = detector.getNumClasses(); // Get number of classes from the detector
        std::cout << "[INFO] Model expects " << numClasses << " classes." << std::endl;
        std::vector<std::string> classNames = loadClassNames(labelPath, numClasses); // Load or generate names
        std::vector<cv::Scalar> classColors = generateClassColors(classNames.size());


        // --- Process Input Based on Source Type ---

        if (sourceType == "image") {
            std::cout << "[INFO] Processing image: " << inputPath << std::endl;
            cv::Mat imageBGR = cv::imread(inputPath, cv::ImreadModes::IMREAD_COLOR);
            if (imageBGR.empty()) { throw std::runtime_error("Could not read image: " + inputPath); }

            // Perform detection using the detector's detect method
            auto detection_start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = detector.detect(imageBGR, confThreshold);
            auto detection_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> detection_duration = detection_end - detection_start;

            // The detector class already prints timing for preprocess, infer, postprocess
            // std::cout << "[INFO] Total detection pipeline time: " << detection_duration.count() << " ms." << std::endl;

            // Draw detections on the original image
            drawDetections(imageBGR, detections, classNames, classColors);

            bool success = cv::imwrite(outputPath, imageBGR);
            if (!success) { std::cerr << "[ERROR] Failed to save image to " << outputPath << std::endl; }
            else { std::cout << "[INFO] Annotated image saved to: " << outputPath << std::endl; }

        }
        else if (sourceType == "video") {
            std::cout << "[INFO] Processing video: " << inputPath << std::endl;
            cv::VideoCapture cap(inputPath);
            if (!cap.isOpened()) { throw std::runtime_error("Could not open video file: " + inputPath); }

            int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = cap.get(cv::CAP_PROP_FPS);
            if (fps <= 0) { // Handle cases where FPS is not reported correctly
                std::cerr << "[WARNING] Video FPS reported as " << fps << ", defaulting to 30.0 for writer." << std::endl;
                fps = 30.0;
            }
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Common codec, adjust if needed

            std::cout << "[INFO] Video properties: " << frameWidth << "x" << frameHeight << " @ " << fps << " FPS" << std::endl;

            cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight));
            if (!writer.isOpened()) { throw std::runtime_error("Could not open video writer for: " + outputPath); }
            std::cout << "[INFO] Saving annotated video to: " << outputPath << std::endl;

            cv::Mat frame;
            int frameCount = 0;
            auto total_start = std::chrono::high_resolution_clock::now();
            double total_detection_time_ms = 0.0;

            while (cap.read(frame)) {
                frameCount++;
                if (frame.empty()) {
                    std::cerr << "[WARNING] Read empty frame " << frameCount << ", stopping." << std::endl; break;
                }

                auto frame_detect_start = std::chrono::high_resolution_clock::now();
                // Use the detector object
                std::vector<Detection> detections = detector.detect(frame, confThreshold);
                auto frame_detect_end = std::chrono::high_resolution_clock::now();
                total_detection_time_ms += std::chrono::duration<double, std::milli>(frame_detect_end - frame_detect_start).count();

                drawDetections(frame, detections, classNames, classColors);
                writer.write(frame); // Write annotated frame

                // Simple progress indication
                if (frameCount % 30 == 0) { // Print every 30 frames
                    double avg_time_per_frame = total_detection_time_ms / frameCount;
                    std::cout << "[INFO] Processed frame " << frameCount << " (Avg detection time: "
                        << avg_time_per_frame << " ms/frame)" << std::endl;
                }
            }

            auto total_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_duration = total_end - total_start;
            std::cout << "[INFO] Finished processing video. Processed " << frameCount << " frames in "
                << total_duration.count() << " seconds." << std::endl;
            if (frameCount > 0) {
                std::cout << "[INFO] Average detection time per frame: " << (total_detection_time_ms / frameCount) << " ms" << std::endl;
            }

            cap.release();
            writer.release();

        }
        else if (sourceType == "camera") {
            // cameraId was already parsed from inputPath
            std::cout << "[INFO] Starting camera stream: ID " << cameraId << std::endl;
            cv::VideoCapture cap(cameraId);
            if (!cap.isOpened()) { throw std::runtime_error("Could not open camera with ID: " + std::to_string(cameraId)); }

            // Optional: Set desired resolution (camera might ignore or adjust)
            // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

            int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            std::cout << "[INFO] Camera resolution: " << frameWidth << "x" << frameHeight << std::endl;

            const std::string windowName = "RF-DETR Live Detection (ONNX Runtime)";
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

            cv::Mat frame;
            std::cout << "[INFO] Press 'q' or ESC in the window to quit." << std::endl;

            double total_frame_time_ms = 0.0;
            int frame_count_display = 0;
            const int fps_update_interval = 10; // Update FPS every 10 frames

            while (true) {
                auto frame_grab_start = std::chrono::high_resolution_clock::now();
                if (!cap.read(frame) || frame.empty()) {
                    std::cerr << "[ERROR] Failed to grab frame from camera. Exiting." << std::endl; break;
                }

                // Use the detector object
                std::vector<Detection> detections = detector.detect(frame, confThreshold);

                auto frame_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> frame_duration = frame_end - frame_grab_start;
                total_frame_time_ms += frame_duration.count();
                frame_count_display++;

                drawDetections(frame, detections, classNames, classColors);

                // Add FPS display (average over last interval)
                if (frame_count_display >= fps_update_interval) {
                    double avg_fps = (1000.0 * frame_count_display) / total_frame_time_ms;
                    cv::putText(frame, cv::format("FPS: %.2f", avg_fps), cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                    // Reset counters for next interval
                    total_frame_time_ms = 0.0;
                    frame_count_display = 0;
                }
                else {
                    // Optionally display instantaneous FPS or nothing until interval is met
                    double current_fps = 1000.0 / frame_duration.count();
                    cv::putText(frame, cv::format("FPS: %.2f", current_fps), cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                }

                cv::imshow(windowName, frame);

                int key = cv::waitKey(1); // Wait 1ms for key press
                if (key == 'q' || key == 'Q' || key == 27) { // 27 is ASCII for ESC
                    std::cout << "[INFO] Quit key pressed. Exiting camera stream." << std::endl; break;
                }
            }

            cap.release();
            cv::destroyAllWindows();

        }
        else {
            throw std::runtime_error("Invalid source_type specified: '" + sourceType + "'. Use 'image', 'video', or 'camera'.");
        }

        std::cout << "[INFO] Processing finished successfully." << std::endl;

    }
    catch (const Ort::Exception& ort_exception) { // Catch ONNX Runtime specific exceptions
        std::cerr << "[FATAL ERROR][ONNX Runtime] " << ort_exception.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) { // Catch standard exceptions
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }
    catch (...) { // Catch any other unknown exceptions
        std::cerr << "[FATAL ERROR] An unknown error occurred." << std::endl;
        return 1;
    }


    return 0;
}