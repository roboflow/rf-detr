#ifndef RF_DETR_ONNX_H_
#define RF_DETR_ONNX_H_

#include <string>
#include <vector>
#include <memory> // For unique_ptr
#include <stdexcept> // For runtime_error
#include <chrono> // For timing (optional in header, but often useful)
#include <iostream> // For logging (optional in header)
#include <algorithm> // For std::min/max etc.
#include <cmath> // For std::exp, std::round
#include <tuple> // For std::tuple

#include <opencv2/core/mat.hpp> // For cv::Mat
#include <opencv2/core/types.hpp> // For cv::Rect

#include <onnxruntime_cxx_api.h> // ONNX Runtime C++ API

// Basic structure to hold detection results
struct Detection {
    cv::Rect box;    // Bounding box (x, y, width, height)
    float score;     // Confidence score
    int class_id;    // Detected class ID
    // std::string class_name; // Optional: Add if you have class names
};

class RF_DETR_ONNX {
public:
    // Constructor: Initializes the ONNX Runtime environment and loads the model.
    RF_DETR_ONNX(const std::string& modelPath, bool useCUDA = false, size_t deviceId = 0, int intraOpNumThreads = 1);

    // Default destructor (unique_ptr handles session cleanup)
    ~RF_DETR_ONNX() = default;

    // Preprocesses the input image into a float vector (NCHW blob).
    std::vector<float> preprocess(const cv::Mat& inputImage);

    // Runs inference on the preprocessed input tensor.
    std::vector<Ort::Value> infer(const std::vector<float>& inputTensorValues);

    // Postprocesses the raw model outputs into a list of detections.
    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputTensors, int originalWidth, int originalHeight, float confThreshold);

    // Performs the full detection pipeline: preprocess, infer, postprocess.
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold);

    // --- Getters ---
    int getInputWidth() const;
    int getInputHeight() const;
    const std::vector<std::string>& getInputNames() const;
    const std::vector<std::string>& getOutputNames() const;
    int64_t getNumClasses() const;


private:
    // --- Constants ---
    // Note: Initializing non-static const vectors directly here might require C++11/14/17 depending on usage/compiler.
    // It's often safer to initialize them in the constructor initializer list or make them static const defined in the .cpp.
    // However, mirroring your original code structure for now.
    const std::vector<float> MEANS = { 0.485f, 0.456f, 0.406f };
    const std::vector<float> STDS = { 0.229f, 0.224f, 0.225f };
    const int MAX_NUMBER_BOXES = 300; // Max proposals from RF-DETR
    const std::string DEFAULT_INSTANCE_NAME = "rfdetr-onnx-cpp-inference";

    // --- ONNX Runtime Members ---
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> ortSession_;

    // --- Model Info Members ---
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<int64_t> inputDims_; // NCHW
    int64_t inputWidth_ = 0;
    int64_t inputHeight_ = 0;
    size_t inputTensorSize_ = 0; // Total number of elements in input tensor
    int64_t numClasses_ = 80; // Default, will be updated from model info

    // --- Private Helper Methods ---

    // Initializes the ONNX Runtime session.
    void initializeSession(const std::string& modelPath, bool useCUDA, size_t deviceId, int intraOpNumThreads);

    // Extracts input and output names, shapes, and other info from the loaded model.
    void getInputOutputInfo();

    // Converts a bounding box from center_x, center_y, width, height to x1, y1, x2, y2 format.
    std::vector<float> box_cxcywh_to_xyxy(const float* box_start);

    // Sigmoid activation function.
    inline float sigmoid(float x);

    // Calculates the product of elements in a vector (for tensor size).
    size_t vectorProduct_(const std::vector<int64_t>& vector);
};

#endif // RF_DETR_ONNX_H_