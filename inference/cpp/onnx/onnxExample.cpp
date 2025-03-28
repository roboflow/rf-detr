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

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>


struct Detection {
    cv::Rect box;        // Bounding box in absolute pixel coordinates (x, y, width, height)
    float score;         // Confidence score
    int class_id;        // Predicted class index
};

class RFDETR_Detector {
public:
    RFDETR_Detector(const std::string& modelPath, bool useCUDA = false, size_t deviceId = 0, int intraOpNumThreads = 1)
        : env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, DEFAULT_INSTANCE_NAME.c_str()),
        sessionOptions_(),
        allocator_(Ort::AllocatorWithDefaultOptions())
    {
        initializeSession(modelPath, useCUDA, deviceId, intraOpNumThreads);
        getInputOutputInfo();
    }

    // Preprocesses the input image
    // Returns a vector<float> representing the NCHW blob
    std::vector<float> preprocess(const cv::Mat& inputImage) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat processedImage;
        cv::resize(inputImage, processedImage, cv::Size(inputWidth_, inputHeight_), 0, 0, cv::INTER_LINEAR);

        cv::cvtColor(processedImage, processedImage, cv::ColorConversionCodes::COLOR_BGR2RGB);

        processedImage.convertTo(processedImage, CV_32FC3, 1.0 / 255.0);

        cv::Mat meanMat(inputHeight_, inputWidth_, CV_32FC3, cv::Scalar(MEANS[0], MEANS[1], MEANS[2]));
        cv::Mat stdMat(inputHeight_, inputWidth_, CV_32FC3, cv::Scalar(STDS[0], STDS[1], STDS[2]));
        cv::subtract(processedImage, meanMat, processedImage);
        cv::divide(processedImage, stdMat, processedImage);

        cv::Mat inputBlob = cv::dnn::blobFromImage(processedImage, 1.0, cv::Size(), cv::Scalar(), false, false);

        std::vector<float> inputTensorValues(inputBlob.ptr<float>(), inputBlob.ptr<float>() + inputTensorSize_);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[INFO] Preprocessing time: " << duration.count() << " ms" << std::endl;

        return inputTensorValues;
    }

    // Runs inference on the preprocessed input tensor
    std::vector<Ort::Value> infer(const std::vector<float>& inputTensorValues) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<Ort::Value> inputTensors;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(inputTensorValues.data()), inputTensorSize_,
            inputDims_.data(), inputDims_.size()
        ));

        std::vector<const char*> inputNamesCStr;
        inputNamesCStr.reserve(inputNames_.size());
        for (const auto& name : inputNames_) {
            inputNamesCStr.push_back(name.c_str());
        }

        std::vector<const char*> outputNamesCStr;
        outputNamesCStr.reserve(outputNames_.size());
        for (const auto& name : outputNames_) {
            outputNamesCStr.push_back(name.c_str());
        }

        std::vector<Ort::Value> outputTensors;
        try {
            outputTensors = ortSession_->Run(
                Ort::RunOptions{ nullptr },
                inputNamesCStr.data(),
                inputTensors.data(),
                inputTensors.size(),
                outputNamesCStr.data(),
                outputNamesCStr.size()
            );
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[ERROR] ONNX Runtime inference failed: " << e.what() << std::endl;
            throw; // Re-throw the exception to be caught in main
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[INFO] Inference time: " << duration.count() << " ms" << std::endl;

        return outputTensors;
    }

    // Postprocesses the raw model outputs
    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputTensors, int originalWidth, int originalHeight, float confThreshold) {
        if (outputTensors.size() != 2) {
            throw std::runtime_error("[ERROR] Expected 2 output tensors from inference, but got " + std::to_string(outputTensors.size()));
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Verify output tensors (basic checks)
        if (!outputTensors[0].IsTensor() || outputTensors[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !outputTensors[1].IsTensor() || outputTensors[1].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("[ERROR] Output tensors are not float tensors as expected.");
        }

        // --- Assuming Output 0 is boxes [1, num_queries, 4] and Output 1 is logits [1, num_queries, num_classes] ---
        // It's crucial this order matches your exported model. Add checks if needed.
        const float* predBoxesData = outputTensors[0].GetTensorData<float>();
        const float* predLogitsData = outputTensors[1].GetTensorData<float>();

        auto boxesShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); // e.g., [1, 300, 4]
        auto logitsShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape(); // e.g., [1, 300, 80]

        if (boxesShape.size() != 3 || boxesShape[0] != 1 || boxesShape[2] != 4) {
            throw std::runtime_error("[ERROR] Unexpected shape for boxes output.");
        }
        if (logitsShape.size() != 3 || logitsShape[0] != 1) {
            throw std::runtime_error("[ERROR] Unexpected shape for logits output.");
        }
        if (boxesShape[1] != logitsShape[1]) {
            throw std::runtime_error("[ERROR] Mismatch in number of queries between boxes and logits outputs.");
        }

        const int64_t numQueries = boxesShape[1];
        const int64_t numClasses = logitsShape[2]; // Get num classes dynamically


        std::vector<std::tuple<float, int, int>> flattenedScores; // (score, query_idx, class_idx)
        flattenedScores.reserve(numQueries * numClasses);

        for (int i = 0; i < numQueries; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                size_t logit_idx = static_cast<size_t>(i) * numClasses + j;
                float score = sigmoid(predLogitsData[logit_idx]);
                flattenedScores.emplace_back(score, i, j);
            }
        }

        // Sort descending by score
        std::sort(flattenedScores.begin(), flattenedScores.end(),
            [](const auto& a, const auto& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

        int numTopK = std::min(static_cast<int>(flattenedScores.size()), MAX_NUMBER_BOXES);

        std::vector<Detection> detections;
        float scaleX = static_cast<float>(originalWidth);
        float scaleY = static_cast<float>(originalHeight);

        for (int k = 0; k < numTopK; ++k) {
            float score = std::get<0>(flattenedScores[k]);

            if (score < confThreshold) {
                // Since the list is sorted descending, we can stop early
                break;
            }

            int queryIdx = std::get<1>(flattenedScores[k]);
            int classIdx = std::get<2>(flattenedScores[k]);

            // Get the raw box data (cxcywh) for this query index
            const float* rawBoxData = predBoxesData + (static_cast<size_t>(queryIdx) * 4); // 4 = box dimensions

            // Convert cxcywh (normalized) to xyxy (normalized)
            std::vector<float> xyxy_norm = box_cxcywh_to_xyxy(rawBoxData);

            // Scale xyxy (normalized) to original image coordinates
            float x1 = xyxy_norm[0] * scaleX;
            float y1 = xyxy_norm[1] * scaleY;
            float x2 = xyxy_norm[2] * scaleX;
            float y2 = xyxy_norm[3] * scaleY;

            // Clip to image boundaries
            x1 = std::max(0.0f, std::min(x1, scaleX - 1.0f));
            y1 = std::max(0.0f, std::min(y1, scaleY - 1.0f));
            x2 = std::max(0.0f, std::min(x2, scaleX - 1.0f));
            y2 = std::max(0.0f, std::min(y2, scaleY - 1.0f));

            // Ensure width and height are positive after clipping
            if (x2 > x1 && y2 > y1) {
                Detection det;
                det.box = cv::Rect(static_cast<int>(std::round(x1)),
                    static_cast<int>(std::round(y1)),
                    static_cast<int>(std::round(x2 - x1)),
                    static_cast<int>(std::round(y2 - y1)));
                det.score = score;
                det.class_id = classIdx;
                detections.push_back(det);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "[INFO] Postprocessing time: " << duration.count() << " ms" << std::endl;
        std::cout << "[INFO] Found " << detections.size() << " detections passing the confidence threshold." << std::endl;

        return detections;
    }

    std::vector<Detection> detect(const cv::Mat& image, float confThreshold) {
        if (image.empty()) {
            throw std::runtime_error("[ERROR] Input image for detection is empty.");
        }

        // 1. Get Original Dimensions
        const int originalWidth = image.cols;
        const int originalHeight = image.rows;

        // 2. Preprocess
        std::vector<float> inputTensorValues = preprocess(image);

        // 3. Infer
        std::vector<Ort::Value> outputTensors = infer(inputTensorValues);

        // 4. Postprocess
        std::vector<Detection> detections = postprocess(outputTensors, originalWidth, originalHeight, confThreshold);

        return detections;
    }

    int getInputWidth() const { return inputWidth_; }
    int getInputHeight() const { return inputHeight_; }
    const std::vector<std::string>& getInputNames() const { return inputNames_; }
    const std::vector<std::string>& getOutputNames() const { return outputNames_; }
    int64_t getNumClasses() const { return numClasses_; }


private:

    const std::vector<float> MEANS = { 0.485f, 0.456f, 0.406f };
    const std::vector<float> STDS = { 0.229f, 0.224f, 0.225f };
    const int MAX_NUMBER_BOXES = 300; // Max proposals from RF-DETR
    const int DEFAULT_NUM_CLASSES = 80; // Example: COCO classes
    const std::string DEFAULT_INSTANCE_NAME = "rtdetr-onnxruntime-cpp-inference";

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> ortSession_;

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<int64_t> inputDims_;
    int64_t inputWidth_ = 0;
    int64_t inputHeight_ = 0;
    size_t inputTensorSize_ = 0;
    int64_t numClasses_ = 80;

    void initializeSession(const std::string& modelPath, bool useCUDA, size_t deviceId, int intraOpNumThreads) {
        sessionOptions_.SetIntraOpNumThreads(intraOpNumThreads);

        if (useCUDA) {
            std::cout << "[INFO] Attempting to use CUDA Execution Provider." << std::endl;
            try {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = deviceId;
                sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
                sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                std::cout << "[INFO] CUDA Execution Provider enabled." << std::endl;
            }
            catch (const Ort::Exception& e) {
                std::cerr << "[ERROR] Failed to initialize CUDA Execution Provider: " << e.what() << std::endl;
                std::cerr << "[INFO] Falling back to CPU." << std::endl;
                sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // Optimize for CPU
                useCUDA = false; // Update flag
            }
        }
        else {
            std::cout << "[INFO] Using CPU Execution Provider." << std::endl;
            sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }

#ifdef _WIN32
        std::wstring wideModelPath = std::wstring(modelPath.begin(), modelPath.end());
        const wchar_t* modelPathW = wideModelPath.c_str();
#else
        const char* modelPathW = modelPath.c_str();
#endif

        try {
            ortSession_ = std::make_unique<Ort::Session>(env_, modelPathW, sessionOptions_);
            std::cout << "[INFO] ONNX Runtime session initialized successfully." << std::endl;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[ERROR] Failed to create ONNX Runtime session: " << e.what() << std::endl;
            throw; // Propagate error
        }
    }

    void getInputOutputInfo() {
        if (!ortSession_) {
            throw std::runtime_error("[ERROR] Session is not initialized.");
        }

        size_t numInputNodes = ortSession_->GetInputCount();
        size_t numOutputNodes = ortSession_->GetOutputCount();

        if (numInputNodes != 1) {
            throw std::runtime_error("[ERROR] Expected 1 input node, but got " + std::to_string(numInputNodes));
        }
        if (numOutputNodes != 2) {
            throw std::runtime_error("[ERROR] Expected 2 output nodes (boxes, logits), but got " + std::to_string(numOutputNodes));
        }

        // Input Info
        inputNames_.resize(numInputNodes);
        auto input_name_ptr = ortSession_->GetInputName(0, allocator_);
        inputNames_[0] = input_name_ptr;

        Ort::TypeInfo inputTypeInfo = ortSession_->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputDims_ = inputTensorInfo.GetShape(); // N C H W

        if (inputDims_.size() != 4) {
            throw std::runtime_error("[ERROR] Expected 4D input tensor (NCHW), but got " + std::to_string(inputDims_.size()) + "D");
        }
        if (inputDims_[0] == -1) inputDims_[0] = 1; // Assume batch size 1 if dynamic
        inputHeight_ = inputDims_[2];
        inputWidth_ = inputDims_[3];
        inputTensorSize_ = vectorProduct_(inputDims_);
        std::cout << "[INFO] Input Name: " << inputNames_[0] << ", Shape: [" << inputDims_[0] << "," << inputDims_[1] << "," << inputDims_[2] << "," << inputDims_[3] << "]" << std::endl;

        // Output Info (Assume order: boxes, logits - verify if necessary)
        outputNames_.resize(numOutputNodes);
        auto output_name_ptr0 = ortSession_->GetOutputName(0, allocator_);
        outputNames_[0] = output_name_ptr0; // Usually boxes
        auto output_name_ptr1 = ortSession_->GetOutputName(1, allocator_);
        outputNames_[1] = output_name_ptr1; // Usually logits


        Ort::TypeInfo outputTypeInfo0 = ortSession_->GetOutputTypeInfo(0);
        auto outputTensorInfo0 = outputTypeInfo0.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims0 = outputTensorInfo0.GetShape();
        if (outputDims0.size() != 3 || outputDims0[2] != 4) {
            std::cerr << "[WARNING] Output 0 (boxes) shape might not be [batch, num_queries, 4]. Got: [";
            for (size_t j = 0; j < outputDims0.size(); ++j) std::cerr << outputDims0[j] << (j == outputDims0.size() - 1 ? "" : ",");
            std::cerr << "]" << std::endl;
        }


        Ort::TypeInfo outputTypeInfo1 = ortSession_->GetOutputTypeInfo(1);
        auto outputTensorInfo1 = outputTypeInfo1.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims1 = outputTensorInfo1.GetShape(); // E.g., [1, 300, 80]
        if (outputDims1.size() != 3) { // Can't check num_classes yet as it's derived
            std::cerr << "[WARNING] Output 1 (logits) shape might not be [batch, num_queries, num_classes]. Got: [";
            for (size_t j = 0; j < outputDims1.size(); ++j) std::cerr << outputDims1[j] << (j == outputDims1.size() - 1 ? "" : ",");
            std::cerr << "]" << std::endl;
        }
        else {
            //numClasses_ = outputDims1[2]; // Determine numClasses from the model's output shape
        }


        std::cout << "[INFO] Output 0 Name: " << outputNames_[0] << ", Shape: [";
        for (size_t j = 0; j < outputDims0.size(); ++j) std::cout << outputDims0[j] << (j == outputDims0.size() - 1 ? "" : ",");
        std::cout << "]" << std::endl;
        std::cout << "[INFO] Output 1 Name: " << outputNames_[1] << ", Shape: [";
        for (size_t j = 0; j < outputDims1.size(); ++j) std::cout << outputDims1[j] << (j == outputDims1.size() - 1 ? "" : ",");
        std::cout << "]" << std::endl;
        std::cout << "[INFO] Determined number of classes: " << numClasses_ << std::endl;

    }

    std::vector<float> box_cxcywh_to_xyxy(const float* box_start) {
        float cx = box_start[0];
        float cy = box_start[1];
        float w = std::max(0.0f, box_start[2]);
        float h = std::max(0.0f, box_start[3]);

        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        return { x1, y1, x2, y2 };
    }

    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    size_t vectorProduct_(const std::vector<int64_t>& vector) {
        if (vector.empty())
            return 0;
        size_t product = 1;
        for (const auto& element : vector) {
            if (element <= 0) return 0; // Avoid issues with non-positive dimensions
            product *= static_cast<size_t>(element);
        }
        return product;
    }
};


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
    std::cout << "[INFO] ONNXRuntime version: " << OrtGetApiBase()->GetVersionString() << std::endl;

    // --- Configuration ---
    std::string modelPath = "rf-detr-base.onnx"; // Default model path
    std::string labelPath = ""; // Default: No labels file
    std::string sourceType = "camera"; // Default source: image
    std::string inputPath = "0";
    std::string outputPath = "output.jpg"; // Default output path (used for image/video)
    float confThreshold = 0.5f;
    bool useCUDA = true;
    size_t deviceId = 0;
    int cameraId = 0; // Default camera ID

    // --- Argument Parsing ---
    std::cout << "[INFO] Parsing command line arguments..." << std::endl;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        }
        else if (arg == "--labels" && i + 1 < argc) {
            labelPath = argv[++i];
        }
        else if (arg == "--source_type" && i + 1 < argc) {
            sourceType = argv[++i]; std::transform(sourceType.begin(), sourceType.end(), sourceType.begin(), ::tolower); // Lowercase
        }
        else if (arg == "--input" && i + 1 < argc) {
            inputPath = argv[++i]; // Path for image/video, ID for camera
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i]; // Path for saving image/video
        }
        else if (arg == "--conf" && i + 1 < argc) {
            try { confThreshold = std::stof(argv[++i]); }
            catch (...) { /* Handle error */ }
        }
        else if (arg == "--use_cuda") {
            useCUDA = true;
        }
        else if (arg == "--use_cpu") {
            useCUDA = false;
        }
        else if (arg == "--device_id" && i + 1 < argc) {
            try { deviceId = std::stoul(argv[++i]); }
            catch (...) { /* Handle error */ }
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "\nUsage: " << argv[0] << " [options]\n\n"
                << "Options:\n"
                << "  --model <path>         Path to the ONNX model file (default: specified in code)\n"
                << "  --labels <path>        Path to the file containing class names (one per line)\n"
                << "  --source_type <type>   Input source type: 'image', 'video', or 'camera' (default: image)\n"
                << "  --input <value>        Path to image/video file, or camera ID (integer) (default: specified in code)\n"
                << "  --output <path>        Path to save the annotated image or video (default: specified in code)\n"
                << "  --conf <threshold>     Confidence threshold for detections (default: 0.5)\n"
                << "  --use_cuda             Enable CUDA execution provider\n"
                << "  --use_cpu              Use CPU execution provider (default)\n"
                << "  --device_id <id>       GPU device ID if using CUDA (default: 0)\n"
                << "  --help, -h             Show this help message\n" << std::endl;
            return 0;
        }
        else {
            std::cerr << "[WARNING] Unknown argument: " << arg << ". Use --help for options." << std::endl;
        }
    }

    std::cout << "[INFO] Configuration:" << std::endl;
    std::cout << "  Model Path: " << modelPath << std::endl;
    std::cout << "  Labels Path: " << (labelPath.empty() ? "None (using defaults)" : labelPath) << std::endl;
    std::cout << "  Source Type: " << sourceType << std::endl;
    std::cout << "  Input: " << inputPath << std::endl;
    if (sourceType == "image" || sourceType == "video") {
        std::cout << "  Output Path: " << outputPath << std::endl;
    }
    std::cout << "  Confidence Threshold: " << confThreshold << std::endl;
    std::cout << "  Execution Provider: " << (useCUDA ? "CUDA (Device ID: " + std::to_string(deviceId) + ")" : "CPU") << std::endl;

    try {
        // --- Initialize Detector ---
        auto detector_start = std::chrono::high_resolution_clock::now();
        RFDETR_Detector detector(modelPath, useCUDA, deviceId);
        auto detector_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detector_init_duration = detector_end - detector_start;
        std::cout << "[INFO] Detector initialization time: " << detector_init_duration.count() << " ms" << std::endl;

        // --- Load Class Names & Generate Colors ---
        int numClasses = detector.getNumClasses();
        std::vector<std::string> classNames = loadClassNames(labelPath, numClasses);
        if (numClasses > 0 && classNames.size() != numClasses) {
            std::cerr << "[WARNING] Mismatch between loaded names (" << classNames.size() << ") and model classes (" << numClasses << "). Adjusting names list." << std::endl;
            if (classNames.size() < numClasses) { for (int i = classNames.size(); i < numClasses; ++i) classNames.push_back("CLS_" + std::to_string(i)); }
            else { classNames.resize(numClasses); }
        }
        std::vector<cv::Scalar> classColors = generateClassColors(classNames.size());


        // --- Process Input Based on Source Type ---

        if (sourceType == "image") {
            std::cout << "[INFO] Processing image: " << inputPath << std::endl;
            cv::Mat imageBGR = cv::imread(inputPath, cv::ImreadModes::IMREAD_COLOR);
            if (imageBGR.empty()) { throw std::runtime_error("Could not read image: " + inputPath); }

            auto detection_start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = detector.detect(imageBGR, confThreshold);
            auto detection_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> detection_duration = detection_end - detection_start;

            std::cout << "[INFO] Found " << detections.size() << " detections in " << detection_duration.count() << " ms." << std::endl;

            drawDetections(imageBGR, detections, classNames, classColors); // Draw on original image

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
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Or use 'X', 'V', 'I', 'D' etc.

            std::cout << "[INFO] Video properties: " << frameWidth << "x" << frameHeight << " @ " << fps << " FPS" << std::endl;

            cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight));
            if (!writer.isOpened()) { throw std::runtime_error("Could not open video writer for: " + outputPath); }
            std::cout << "[INFO] Saving annotated video to: " << outputPath << std::endl;

            cv::Mat frame;
            int frameCount = 0;
            auto total_start = std::chrono::high_resolution_clock::now();

            while (cap.read(frame)) {
                frameCount++;
                if (frame.empty()) {
                    std::cerr << "[WARNING] Read empty frame, stopping." << std::endl; break;
                }

                auto frame_start = std::chrono::high_resolution_clock::now();
                std::vector<Detection> detections = detector.detect(frame, confThreshold);
                drawDetections(frame, detections, classNames, classColors);
                writer.write(frame); // Write annotated frame
                auto frame_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> frame_duration = frame_end - frame_start;

                // Simple progress indication
                if (frameCount % 30 == 0) {
                    std::cout << "[INFO] Processed frame " << frameCount << " (" << frame_duration.count() << " ms/frame)" << std::endl;
                }
            }

            auto total_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_duration = total_end - total_start;
            std::cout << "[INFO] Finished processing video. Processed " << frameCount << " frames in "
                << total_duration.count() << " seconds." << std::endl;

            cap.release();
            writer.release();

        }
        else if (sourceType == "camera") {
            try { cameraId = std::stoi(inputPath); } // Input is camera ID
            catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid camera ID provided: " + inputPath);
            }
            catch (const std::out_of_range& e) {
                throw std::runtime_error("Camera ID out of range: " + inputPath);
            }

            std::cout << "[INFO] Starting camera stream: ID " << cameraId << std::endl;
            cv::VideoCapture cap(cameraId);
            if (!cap.isOpened()) { throw std::runtime_error("Could not open camera with ID: " + std::to_string(cameraId)); }

            // Optional: Set desired resolution (camera might ignore or adjust)
            // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

            int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            std::cout << "[INFO] Camera resolution: " << frameWidth << "x" << frameHeight << std::endl;

            const std::string windowName = "RT-DETR Live Detection";
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

            cv::Mat frame;
            std::cout << "[INFO] Press 'q' or ESC to quit." << std::endl;

            while (true) {
                if (!cap.read(frame) || frame.empty()) {
                    std::cerr << "[ERROR] Failed to grab frame from camera. Exiting." << std::endl; break;
                }

                auto frame_start = std::chrono::high_resolution_clock::now();
                std::vector<Detection> detections = detector.detect(frame, confThreshold);
                auto frame_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> frame_duration = frame_end - frame_start;
                double current_fps = 1000.0 / frame_duration.count();

                drawDetections(frame, detections, classNames, classColors);

                // Add FPS display
                cv::putText(frame, cv::format("FPS: %.2f", current_fps), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

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
            throw std::runtime_error("Invalid source_type specified: " + sourceType + ". Use 'image', 'video', or 'camera'.");
        }

        std::cout << "[INFO] Processing finished successfully." << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}