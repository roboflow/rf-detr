#pragma once  

#include "NvInfer.h"    // TensorRT library for high-performance inference
#include <opencv2/opencv.hpp>  // OpenCV for image processing

using namespace nvinfer1;  // Namespace for TensorRT
using namespace std;        // Use standard library namespace
using namespace cv;         // Use OpenCV namespace

// Struct to store detection results
struct Detection {
    float conf;  // Confidence score of the detection
    int class_id;  // Class ID of the detected object (e.g., person, car, etc.)
    Rect bbox;  // Bounding box coordinates around the detected object
};

// Main class for the RF DETR model
class RF_DETR {

public:
    // Constructor: Loads the TensorRT engine and initializes the model
    RF_DETR(string model_path, nvinfer1::ILogger& logger);

    // Destructor: Cleans up resources used by the model
    ~RF_DETR();

    // Preprocess the input image to match the model's input format
    void preprocess(Mat& image);

    // Run inference on the preprocessed image
    void infer();

    // Postprocess the model's output to extract detection results
    void postprocess(vector<Detection>& output,int originalWidth, int originalHeight);

    // Draw bounding boxes and labels on the original image
    void draw(Mat& image, const vector<Detection>& output);

private:
    // Initialize the TensorRT engine from a serialized model file
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    // Device (GPU) buffers for input and output
    float* gpu_buffers[3];  // Input and output buffers allocated on the GPU

    // Host (CPU) buffer for storing inference output
    float* cpu_output_buffer_1;
    float* cpu_output_buffer_2;


    // CUDA stream for asynchronous execution
    cudaStream_t stream;

    // TensorRT runtime for deserializing the engine from file
    IRuntime* runtime;

    // TensorRT engine used to execute the network
    ICudaEngine* engine;

    // Execution context for running inference with the engine
    IExecutionContext* context;

    // Model parameters
    int input_w;  // Input image width expected by the model
    int input_h;  // Input image height expected by the model
    int num_detections;  // Number of detections output by the model
    int detection_attribute_size;  // Attributes (e.g., bbox, class) per detection
    int num_classes = 80;  // Number of classes (e.g., COCO dataset has 80 classes)

    // Maximum supported image size (used for memory allocation checks)
    const int MAX_IMAGE_SIZE = 4096 * 4096;

    // Confidence threshold for filtering detections
    float conf_threshold = 0.45f;

    // Colors for drawing bounding boxes for each class
    vector<Scalar> colors;

    // Build TensorRT engine from an ONNX model file (if applicable)
    void build(std::string onnxPath, nvinfer1::ILogger& logger);

    // Save the built TensorRT engine to a file
    bool saveEngine(const std::string& filename);
};
