#include "RF_DETR_ONNX.h"

#include <opencv2/imgproc.hpp> // For cv::resize, cv::cvtColor
#include <opencv2/dnn/dnn.hpp>  // For cv::dnn::blobFromImage

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric> // Potentially useful, e.g., for accumulate, though vectorProduct_ uses a loop
#include <tuple>

// --- Constructor Implementation ---
RF_DETR_ONNX::RF_DETR_ONNX(const std::string& modelPath, bool useCUDA, size_t deviceId, int intraOpNumThreads)
    : env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, DEFAULT_INSTANCE_NAME.c_str()),
    sessionOptions_(),
    allocator_(Ort::AllocatorWithDefaultOptions())
    // MEANS and STDS initialized directly in header (as per original code)
    // numClasses_ initialized with default, updated in getInputOutputInfo
{
    initializeSession(modelPath, useCUDA, deviceId, intraOpNumThreads);
    getInputOutputInfo(); // Get info after session is created
}

// --- Public Method Implementations ---

std::vector<float> RF_DETR_ONNX::preprocess(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat processedImage;
    // Ensure inputWidth_ and inputHeight_ are valid (checked in getInputOutputInfo)
    if (inputWidth_ <= 0 || inputHeight_ <= 0) {
        throw std::runtime_error("[ERROR] Invalid input dimensions obtained from model.");
    }
    cv::resize(inputImage, processedImage, cv::Size(static_cast<int>(inputWidth_), static_cast<int>(inputHeight_)), 0, 0, cv::INTER_LINEAR);

    cv::cvtColor(processedImage, processedImage, cv::ColorConversionCodes::COLOR_BGR2RGB);

    processedImage.convertTo(processedImage, CV_32FC3, 1.0 / 255.0);

    // Apply normalization (Subtract Mean, Divide by Standard Deviation)
    // Ensure MEANS and STDS have the correct size (3)
    if (MEANS.size() != 3 || STDS.size() != 3) {
        throw std::runtime_error("[ERROR] MEANS or STDS vectors do not have size 3.");
    }
    cv::Mat meanMat(inputHeight_, inputWidth_, CV_32FC3, cv::Scalar(MEANS[0], MEANS[1], MEANS[2]));
    cv::Mat stdMat(inputHeight_, inputWidth_, CV_32FC3, cv::Scalar(STDS[0], STDS[1], STDS[2]));

    // Perform normalization using OpenCV functions for potentially better performance/clarity
    cv::subtract(processedImage, meanMat, processedImage);
    cv::divide(processedImage, stdMat, processedImage); // Element-wise division

    // Create blob from image (results in NCHW layout)
    // Scale factor is 1.0 because scaling and normalization are already done.
    // Mean subtraction is (0,0,0) because it's already done.
    // SwapRB is false because we converted BGR->RGB earlier.
    // Crop is false.
    cv::Mat inputBlob = cv::dnn::blobFromImage(processedImage, 1.0, cv::Size(), cv::Scalar(), false, false);

    // Copy blob data to a std::vector<float>
    // Ensure inputTensorSize_ is correctly calculated
    if (inputTensorSize_ == 0) {
        throw std::runtime_error("[ERROR] Input tensor size is zero. Model info might be incorrect.");
    }
    std::vector<float> inputTensorValues(inputTensorSize_);
    memcpy(inputTensorValues.data(), inputBlob.ptr<float>(), inputTensorSize_ * sizeof(float));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[INFO] Preprocessing time: " << duration.count() << " ms" << std::endl;

    return inputTensorValues;
}


std::vector<Ort::Value> RF_DETR_ONNX::infer(const std::vector<float>& inputTensorValues) {
    if (!ortSession_) {
        throw std::runtime_error("[ERROR] Inference called before session initialization or after failure.");
    }
    if (inputTensorValues.size() != inputTensorSize_) {
        throw std::runtime_error("[ERROR] Input tensor value size mismatch. Expected " + std::to_string(inputTensorSize_) + ", got " + std::to_string(inputTensorValues.size()));
    }
    if (inputNames_.empty() || outputNames_.empty()) {
        throw std::runtime_error("[ERROR] Input/Output names not initialized.");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Ort::Value> inputTensors;
    // Use const_cast carefully. Ensure the tensor data is not modified by the inference engine if it's not supposed to be.
    // CreateCpu implies the data is on the CPU. If using CUDA EP with pinned memory, adjust accordingly.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensorValues.data()), inputTensorSize_,
        inputDims_.data(), inputDims_.size()
    ));

    // Prepare C-style strings for ONNX Runtime API
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
            inputTensors.size(), // Should be 1 for this model
            outputNamesCStr.data(),
            outputNamesCStr.size() // Should be 2 for this model
        );
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime inference failed: " << e.what() << std::endl;
        // Consider logging more details, e.g., input shapes/types
        throw; // Re-throw the exception
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[INFO] Inference time: " << duration.count() << " ms" << std::endl;

    return outputTensors;
}


std::vector<Detection> RF_DETR_ONNX::postprocess(const std::vector<Ort::Value>& outputTensors, int originalWidth, int originalHeight, float confThreshold) {
    // Check if output tensors are valid and have expected number
    if (outputTensors.size() != 2) {
        throw std::runtime_error("[ERROR] Expected 2 output tensors from inference, but got " + std::to_string(outputTensors.size()));
    }
    if (!outputTensors[0] || !outputTensors[1]) {
        throw std::runtime_error("[ERROR] One or more output tensors are invalid (null).");
    }
    if (!outputTensors[0].IsTensor() || !outputTensors[1].IsTensor()) {
        throw std::runtime_error("[ERROR] Outputs are not tensors.");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Basic type check
    if (outputTensors[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        outputTensors[1].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("[ERROR] Output tensors are not float tensors as expected.");
    }

    // Get pointers to tensor data
    const float* predBoxesData = outputTensors[0].GetTensorData<float>();
    const float* predLogitsData = outputTensors[1].GetTensorData<float>();

    // Get shapes
    auto boxesShapeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto logitsShapeInfo = outputTensors[1].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> boxesShape = boxesShapeInfo.GetShape(); // e.g., [1, 300, 4]
    std::vector<int64_t> logitsShape = logitsShapeInfo.GetShape(); // e.g., [1, 300, 80]

    // Validate shapes
    if (boxesShape.size() != 3 || boxesShape[0] != 1 || boxesShape[2] != 4) {
        throw std::runtime_error("[ERROR] Unexpected shape for boxes output. Expected [1, N, 4].");
    }
    if (logitsShape.size() != 3 || logitsShape[0] != 1) {
        throw std::runtime_error("[ERROR] Unexpected shape for logits output. Expected [1, N, num_classes].");
    }
    if (boxesShape[1] != logitsShape[1]) {
        throw std::runtime_error("[ERROR] Mismatch in number of queries between boxes ("
            + std::to_string(boxesShape[1]) + ") and logits ("
            + std::to_string(logitsShape[1]) + ") outputs.");
    }
    // Check if the number of classes matches the initialized value
    if (numClasses_ <= 0) {
        throw std::runtime_error("[ERROR] Number of classes not properly initialized.");
    }
    if (logitsShape[2] != numClasses_) {
        std::cerr << "[WARNING] Number of classes in model output (" << logitsShape[2]
            << ") differs from expected value (" << numClasses_
            << "). Using value from model output." << std::endl;
        // Optionally update numClasses_ here if you trust the model output more,
        // or throw an error if they must match.
        // numClasses_ = logitsShape[2]; // Example: Update based on model
    }


    const int64_t numQueries = boxesShape[1];
    // Use the dynamically determined number of classes from the logits shape
    const int64_t actualNumClasses = logitsShape[2];

    // --- Score Calculation and Filtering ---
    // Store scores along with their query and class index: (score, query_idx, class_idx)
    std::vector<std::tuple<float, int, int>> flattenedScores;
    flattenedScores.reserve(static_cast<size_t>(numQueries) * actualNumClasses); // Pre-allocate memory

    for (int64_t i = 0; i < numQueries; ++i) {
        // Optimization: Find max score for this query first (optional)
        // float max_score_for_query = -1.0f;
        // int max_class_idx = -1;

        for (int64_t j = 0; j < actualNumClasses; ++j) {
            // Calculate index in the flattened logits tensor
            size_t logit_idx = static_cast<size_t>(i) * actualNumClasses + j;
            float score = sigmoid(predLogitsData[logit_idx]);

            // Optimization: If you only care about the top class per query:
            // if (score > max_score_for_query) {
            //     max_score_for_query = score;
            //     max_class_idx = j;
            // }

            // Store all scores above a preliminary threshold (or all scores)
            // Adding a small threshold here can reduce sorting time later.
            if (score >= confThreshold) { // Only consider scores above the final threshold
                flattenedScores.emplace_back(score, static_cast<int>(i), static_cast<int>(j));
            }
        }
        // Optimization: Add only the best class for this query if desired
        // if (max_class_idx != -1 && max_score_for_query >= confThreshold) {
        //    flattenedScores.emplace_back(max_score_for_query, i, max_class_idx);
        // }
    }

    // Sort all collected scores in descending order
    std::sort(flattenedScores.rbegin(), flattenedScores.rend()); // Sort descending efficiently

    // --- Box Conversion and Selection ---
    std::vector<Detection> detections;
    detections.reserve(std::min(static_cast<size_t>(MAX_NUMBER_BOXES), flattenedScores.size())); // Pre-allocate memory

    float scaleX = static_cast<float>(originalWidth);
    float scaleY = static_cast<float>(originalHeight);

    // Iterate through the sorted scores and create detections
    // Apply Non-Maximum Suppression (NMS) here if needed (DETR often doesn't require heavy NMS)
    // This current implementation takes the top-K scores directly without NMS.
    int count = 0;
    for (const auto& scoreTuple : flattenedScores) {
        if (count >= MAX_NUMBER_BOXES) {
            break; // Limit the number of detections
        }

        float score = std::get<0>(scoreTuple);
        // We already pre-filtered by confThreshold, but double-check if logic changes
        // if (score < confThreshold) {
        //     continue; // Should not happen if pre-filtered
        // }

        int queryIdx = std::get<1>(scoreTuple);
        int classIdx = std::get<2>(scoreTuple);

        // Get the raw box data (cxcywh normalized) for this query index
        const float* rawBoxData = predBoxesData + (static_cast<size_t>(queryIdx) * 4); // 4 = box dimensions (cx, cy, w, h)

        // Convert cxcywh (normalized) to xyxy (normalized)
        std::vector<float> xyxy_norm = box_cxcywh_to_xyxy(rawBoxData);

        // Scale xyxy (normalized) to original image coordinates
        float x1 = xyxy_norm[0] * scaleX;
        float y1 = xyxy_norm[1] * scaleY;
        float x2 = xyxy_norm[2] * scaleX;
        float y2 = xyxy_norm[3] * scaleY;

        // Clip coordinates to image boundaries to prevent invalid Rect
        x1 = std::max(0.0f, std::min(x1, scaleX - 1.0f));
        y1 = std::max(0.0f, std::min(y1, scaleY - 1.0f));
        x2 = std::max(0.0f, std::min(x2, scaleX - 1.0f));
        y2 = std::max(0.0f, std::min(y2, scaleY - 1.0f));

        // Ensure width and height are non-negative after clipping
        if (x2 > x1 && y2 > y1) {
            Detection det;
            // Convert to integer Rect (x, y, width, height)
            det.box = cv::Rect(static_cast<int>(std::round(x1)),
                static_cast<int>(std::round(y1)),
                static_cast<int>(std::round(x2 - x1)),
                static_cast<int>(std::round(y2 - y1)));
            det.score = score;
            det.class_id = classIdx;
            detections.push_back(det);
            count++;
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[INFO] Postprocessing time: " << duration.count() << " ms" << std::endl;
    std::cout << "[INFO] Found " << detections.size() << " detections passing the confidence threshold (max " << MAX_NUMBER_BOXES << ")." << std::endl;

    return detections;
}


std::vector<Detection> RF_DETR_ONNX::detect(const cv::Mat& image, float confThreshold) {
    if (image.empty()) {
        throw std::runtime_error("[ERROR] Input image for detection is empty.");
    }

    // 1. Get Original Dimensions (needed for postprocessing scaling)
    const int originalWidth = image.cols;
    const int originalHeight = image.rows;

    // 2. Preprocess the image
    std::vector<float> inputTensorValues = preprocess(image);

    // 3. Run inference
    std::vector<Ort::Value> outputTensors = infer(inputTensorValues);

    // 4. Postprocess the results
    std::vector<Detection> detections = postprocess(outputTensors, originalWidth, originalHeight, confThreshold);

    return detections;
}

// --- Getters Implementation ---
int RF_DETR_ONNX::getInputWidth() const {
    return static_cast<int>(inputWidth_);
}

int RF_DETR_ONNX::getInputHeight() const {
    return static_cast<int>(inputHeight_);
}

const std::vector<std::string>& RF_DETR_ONNX::getInputNames() const {
    return inputNames_;
}

const std::vector<std::string>& RF_DETR_ONNX::getOutputNames() const {
    return outputNames_;
}

int64_t RF_DETR_ONNX::getNumClasses() const {
    return numClasses_;
}

void RF_DETR_ONNX::initializeSession(const std::string& modelPath, bool useCUDA, size_t deviceId, int intraOpNumThreads) {
    sessionOptions_.SetIntraOpNumThreads(intraOpNumThreads);
    sessionOptions_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING); // Match env level

    bool cuda_available = false;
    if (useCUDA) {
        std::cout << "[INFO] Attempting to use CUDA Execution Provider." << std::endl;
        try {
            // Check available providers before appending
            auto available_providers = Ort::GetAvailableProviders();
            bool provider_found = false;
            for (const auto& provider_name : available_providers) {
                if (provider_name == "CUDAExecutionProvider") {
                    provider_found = true;
                    break;
                }
            }

            if (provider_found) {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = static_cast<int>(deviceId); // Ensure deviceId fits in int
                // Other options can be set here (e.g., gpu_mem_limit, arena_extend_strategy)
                // cuda_options.gpu_mem_limit = N;
                // cuda_options.arena_extend_strategy = 1; // kNextPowerOfTwo

                sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
                sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                std::cout << "[INFO] CUDA Execution Provider enabled on device " << deviceId << "." << std::endl;
                cuda_available = true;
            }
            else {
                std::cerr << "[WARNING] CUDA Execution Provider is not available in this build." << std::endl;
            }
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[ERROR] Failed to initialize CUDA Execution Provider: " << e.what() << std::endl;
            // Fallback logic is handled below
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] std::exception during CUDA setup: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[ERROR] Unknown exception during CUDA setup." << std::endl;
        }
    }

    if (!cuda_available) {
        if (useCUDA) { // Only print fallback message if CUDA was requested but failed
            std::cerr << "[INFO] Falling back to CPU Execution Provider." << std::endl;
        }
        else {
            std::cout << "[INFO] Using CPU Execution Provider." << std::endl;
        }
        // Set optimization level for CPU
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Optionally disable Per Session Threads if preferring intraOpNumThreads control
        // sessionOptions_.DisablePerSessionThreads();
    }

    // Handle Model Path Encoding (Windows vs. others)
#ifdef _WIN32
    std::wstring wideModelPath = std::wstring(modelPath.begin(), modelPath.end());
    const wchar_t* modelPathW = wideModelPath.c_str();
#else
    const char* modelPathW = modelPath.c_str();
#endif

    // Create the session
    try {
        ortSession_ = std::make_unique<Ort::Session>(env_, modelPathW, sessionOptions_);
        std::cout << "[INFO] ONNX Runtime session initialized successfully for model: " << modelPath << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Failed to create ONNX Runtime session for model '" << modelPath << "': " << e.what() << std::endl;
        throw; // Re-throw to signal failure
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] std::exception during session creation: " << e.what() << std::endl;
        throw;
    }
    catch (...) {
        std::cerr << "[ERROR] Unknown exception during session creation." << std::endl;
        throw;
    }
}


void RF_DETR_ONNX::getInputOutputInfo() {
    if (!ortSession_) {
        throw std::runtime_error("[ERROR] Session is not initialized. Cannot get input/output info.");
    }

    // --- Input Info ---
    size_t numInputNodes = ortSession_->GetInputCount();
    if (numInputNodes != 1) {
        throw std::runtime_error("[ERROR] Expected 1 input node, but found " + std::to_string(numInputNodes));
    }

    inputNames_.resize(numInputNodes);
    // Use Ort::Allocator unique_ptr for automatic memory management
#if ORT_API_VERSION > 11 // Get..NameAllocated introduced around v1.11 (API Version 11)
// Newer ONNX Runtime versions (>= 1.11): Use GetInputNameAllocated
    std::cout << "[INFO] Using GetInputNameAllocated (ORT API Version >= 11)." << std::endl;
    auto input_name_allocated_ptr = ortSession_->GetInputNameAllocated(0, allocator_);
    inputNames_[0] = input_name_allocated_ptr.get(); // Copy string from unique_ptr's managed C-string
#else
    // Older ONNX Runtime versions (< 1.11): Use GetInputName
    std::cout << "[INFO] Using GetInputName (ORT API Version < 11)." << std::endl;
    char* input_name_ptr = ortSession_->GetInputName(0, allocator_); // Returns char*
    inputNames_[0] = input_name_ptr; // Assign to std::string (performs copy)
    allocator_.Free(input_name_ptr); // IMPORTANT: Free the memory allocated by ORT for the name
#endif

    Ort::TypeInfo inputTypeInfo = ortSession_->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    inputDims_ = inputTensorInfo.GetShape(); // N C H W

    // Basic validation of input dimensions
    if (inputDims_.size() != 4) {
        throw std::runtime_error("[ERROR] Expected 4D input tensor (NCHW), but got " + std::to_string(inputDims_.size()) + "D shape.");
    }
    if (inputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("[ERROR] Expected FLOAT input type, but got type " + std::to_string(inputType));
    }

    // Handle dynamic dimensions (e.g., -1 or 'batch')
    if (inputDims_[0] < 1) {
        std::cout << "[INFO] Input batch size is dynamic, assuming batch size = 1." << std::endl;
        inputDims_[0] = 1; // Set batch size to 1
    }
    if (inputDims_[2] <= 0 || inputDims_[3] <= 0) {
        throw std::runtime_error("[ERROR] Input height or width dimension is non-positive or dynamic. Model may need fixed input size or specific handling.");
    }

    inputHeight_ = inputDims_[2];
    inputWidth_ = inputDims_[3];
    inputTensorSize_ = vectorProduct_(inputDims_); // Calculate total elements N*C*H*W

    if (inputTensorSize_ == 0) {
        throw std::runtime_error("[ERROR] Calculated input tensor size is zero. Check model input dimensions.");
    }

    std::cout << "[INFO] Input Name: " << inputNames_[0]
        << ", Type: FLOAT, Shape: [" << inputDims_[0] << "," << inputDims_[1] << "," << inputDims_[2] << "," << inputDims_[3] << "]" << std::endl;

    // --- Output Info ---
    size_t numOutputNodes = ortSession_->GetOutputCount();
    if (numOutputNodes != 2) { // Specific to RF-DETR's expected output (boxes, logits)
        throw std::runtime_error("[ERROR] Expected 2 output nodes (boxes, logits), but found " + std::to_string(numOutputNodes));
    }

    outputNames_.resize(numOutputNodes);
#if ORT_API_VERSION > 11
    // Newer ONNX Runtime versions (>= 1.11)
    std::cout << "[INFO] Using GetOutputNameAllocated (ORT API Version >= 11)." << std::endl;
    auto output_name_ptr0 = ortSession_->GetOutputNameAllocated(0, allocator_);
    outputNames_[0] = output_name_ptr0.get();
    auto output_name_ptr1 = ortSession_->GetOutputNameAllocated(1, allocator_);
    outputNames_[1] = output_name_ptr1.get();
#else
    // Older ONNX Runtime versions (< 1.11)
    std::cout << "[INFO] Using GetOutputName (ORT API Version < 11)." << std::endl;
    char* output_name_ptr0 = ortSession_->GetOutputName(0, allocator_);
    outputNames_[0] = output_name_ptr0; // Copy
    allocator_.Free(output_name_ptr0); // Free memory

    char* output_name_ptr1 = ortSession_->GetOutputName(1, allocator_);
    outputNames_[1] = output_name_ptr1; // Copy
    allocator_.Free(output_name_ptr1); // Free memory
#endif

    // Verify Output 0 (Boxes)
    Ort::TypeInfo outputTypeInfo0 = ortSession_->GetOutputTypeInfo(0);
    auto outputTensorInfo0 = outputTypeInfo0.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType0 = outputTensorInfo0.GetElementType();
    std::vector<int64_t> outputDims0 = outputTensorInfo0.GetShape();

    if (outputType0 != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "[WARNING] Output 0 (Boxes) type is not FLOAT. Got type " << outputType0 << "." << std::endl;
    }
    if (outputDims0.size() != 3 || (outputDims0[0] != inputDims_[0] && outputDims0[0] > 0) || outputDims0[2] != 4) {
        std::cerr << "[WARNING] Output 0 (Boxes) shape might not be [batch, num_queries, 4]. Got shape: [";
        for (size_t j = 0; j < outputDims0.size(); ++j) std::cerr << outputDims0[j] << (j == outputDims0.size() - 1 ? "" : ",");
        std::cerr << "]. Ensure postprocessing logic matches." << std::endl;
    }

    // Verify Output 1 (Logits) and determine numClasses_
    Ort::TypeInfo outputTypeInfo1 = ortSession_->GetOutputTypeInfo(1);
    auto outputTensorInfo1 = outputTypeInfo1.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType1 = outputTensorInfo1.GetElementType();
    std::vector<int64_t> outputDims1 = outputTensorInfo1.GetShape(); // E.g., [batch, num_queries, num_classes]

    if (outputType1 != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "[WARNING] Output 1 (Logits) type is not FLOAT. Got type " << outputType1 << "." << std::endl;
    }
    if (outputDims1.size() != 3) {
        throw std::runtime_error("[ERROR] Output 1 (Logits) is not 3D [batch, num_queries, num_classes]. Got "
            + std::to_string(outputDims1.size()) + "D.");
    }
    if (outputDims1[0] != inputDims_[0] && outputDims1[0] > 0) { // Check batch size consistency
        std::cerr << "[WARNING] Output 1 (Logits) batch size (" << outputDims1[0]
            << ") doesn't match input batch size (" << inputDims_[0] << ")." << std::endl;
    }
    if (outputDims0.size() == 3 && outputDims1.size() == 3 && outputDims0[1] != outputDims1[1] && outputDims0[1] > 0 && outputDims1[1] > 0) { // Check num_queries consistency
        std::cerr << "[WARNING] Mismatch in num_queries between outputs. Boxes: " << outputDims0[1]
            << ", Logits: " << outputDims1[1] << "." << std::endl;
    }

    // Determine number of classes from the last dimension of the logits output
    if (outputDims1[2] <= 0) {
        throw std::runtime_error("[ERROR] Number of classes dimension in logits output is non-positive (" + std::to_string(outputDims1[2]) + ").");
    }
    numClasses_ = outputDims1[2]; // Update numClasses_ based on the model

    // Print output info
    std::cout << "[INFO] Output 0 Name: " << outputNames_[0] << ", Type: FLOAT, Shape: [";
    for (size_t j = 0; j < outputDims0.size(); ++j) std::cout << outputDims0[j] << (j == outputDims0.size() - 1 ? "" : ",");
    std::cout << "]" << std::endl;

    std::cout << "[INFO] Output 1 Name: " << outputNames_[1] << ", Type: FLOAT, Shape: [";
    for (size_t j = 0; j < outputDims1.size(); ++j) std::cout << outputDims1[j] << (j == outputDims1.size() - 1 ? "" : ",");
    std::cout << "]" << std::endl;

    std::cout << "[INFO] Determined number of classes from model: " << numClasses_ << std::endl;
}


std::vector<float> RF_DETR_ONNX::box_cxcywh_to_xyxy(const float* box_start) {
    // Assumes box_start points to [cx, cy, w, h]
    float cx = box_start[0];
    float cy = box_start[1];
    // Ensure width and height are non-negative before calculations
    float w = std::max(0.0f, box_start[2]);
    float h = std::max(0.0f, box_start[3]);

    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;
    float x2 = cx + 0.5f * w;
    float y2 = cy + 0.5f * h;

    return { x1, y1, x2, y2 }; // Return {x_min, y_min, x_max, y_max}
}


inline float RF_DETR_ONNX::sigmoid(float x) {
    // Basic sigmoid implementation
    return 1.0f / (1.0f + std::exp(-x));
    // Consider adding checks for very large/small x to prevent overflow/underflow if necessary
    // e.g., using std::max(-30.0f, std::min(30.0f, -x)) inside exp for stability
}


size_t RF_DETR_ONNX::vectorProduct_(const std::vector<int64_t>& vec) {
    if (vec.empty()) {
        return 0;
    }
    size_t product = 1;
    for (const auto& element : vec) {
        // Ensure dimensions are positive before multiplying
        if (element <= 0) {
            // This indicates an invalid dimension (or dynamic handled incorrectly)
            std::cerr << "[ERROR] Non-positive dimension encountered (" << element << ") in vector product calculation." << std::endl;
            return 0; // Return 0 to signify an error or invalid size
        }
        // Check for potential overflow before multiplication
        if (element > 0 && product > std::numeric_limits<size_t>::max() / static_cast<size_t>(element)) {
            throw std::overflow_error("[ERROR] Size calculation overflowed (vectorProduct_)");
        }
        product *= static_cast<size_t>(element);
    }
    return product;
}