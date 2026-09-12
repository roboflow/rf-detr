
#include "RF_DETR.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include "common.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}


RF_DETR::RF_DETR(string model_path, nvinfer1::ILogger& logger){
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos){
        init(model_path, logger);
    }

    // Build an engine from an onnx model
    else{
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 8
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}

void RF_DETR::init(std::string engine_path, nvinfer1::ILogger& logger){
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the TensorRT engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

#if NV_TENSORRT_MAJOR < 8
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    // Handle dynamic input and output tensors
    auto input_name = engine->getIOTensorName(0);
    auto output_name_1 = engine->getIOTensorName(1);  // "dets"
    auto output_name_2 = engine->getIOTensorName(2);  // "labels"

    auto input_dims = engine->getTensorShape(input_name);
    auto output_dims_1 = engine->getTensorShape(output_name_1); // Shape: ['Concatdets_dim_0', 'Concatdets_dim_1', 4]
    auto output_dims_2 = engine->getTensorShape(output_name_2); // Shape: ['Addlabels_dim_0', 'Addlabels_dim_1', 91]

    // Input shape
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    // Output shapes
    num_detections = output_dims_1.d[1];  // Concatdets_dim_1
    detection_attribute_size = output_dims_1.d[2]; // Should be 4 (x, y, w, h)
    num_classes = output_dims_2.d[2]; // Should be 91 (classes)
#endif

    // Allocate input buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));

    // Allocate output buffers for "dets" and "labels"
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float))); // Bounding boxes
    CUDA_CHECK(cudaMalloc(&gpu_buffers[2], num_detections * num_classes * sizeof(float))); // Class scores
   
    // TWO CPU output buffers
    cpu_output_buffer_1 = new float[num_detections * detection_attribute_size]; // Bounding boxes
    cpu_output_buffer_2 = new float[num_detections * num_classes]; // Class scores


    // Initialize CUDA preprocessing
    cuda_preprocess_init(MAX_IMAGE_SIZE);

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Model warmup
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("Model warmup completed (10 iterations)\n");
    }
}


RF_DETR::~RF_DETR(){
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 3; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));

    delete[] cpu_output_buffer_1;
    delete[] cpu_output_buffer_2;


    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void RF_DETR::preprocess(Mat& image) {
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void RF_DETR::infer() {
    // Ensure input buffer is registered
    const char* input_name = engine->getIOTensorName(0);
    context->setTensorAddress(input_name, gpu_buffers[0]);

    // Retrieve output tensor names
    const char* output_name_1 = engine->getIOTensorName(1); // "dets"
    const char* output_name_2 = engine->getIOTensorName(2); // "labels"

    // Register output buffers
    context->setTensorAddress(output_name_1, gpu_buffers[1]);
    context->setTensorAddress(output_name_2, gpu_buffers[2]);


#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    context->enqueueV3(this->stream);
#endif
}

void RF_DETR::postprocess(vector<Detection>& output, int originalWidth, int originalHeight) {
    if (!cpu_output_buffer_1 || !cpu_output_buffer_2) {
        throw std::runtime_error("[ERROR] Output buffers are not allocated.");
    }

    // --- Copy device output buffers to host ---
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_1, gpu_buffers[1], num_detections * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));  // Boxes
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_2, gpu_buffers[2], num_detections * num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream)); // Class scores
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- Extract pointers to box & class scores ---
    const float* predBoxesData = cpu_output_buffer_1;   // Shape: [num_detections, 4] (cx, cy, w, h)
    const float* predLogitsData = cpu_output_buffer_2;    // Shape: [num_detections, num_classes]

    vector<std::tuple<float, int, int>> flattenedScores; // (score, query_idx, class_idx)
    flattenedScores.reserve(num_detections * num_classes);

    for (int i = 0; i < num_detections; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            size_t logit_idx = i * num_classes + j;
            float score = sigmoid(predLogitsData[logit_idx]);
            flattenedScores.emplace_back(score, i, j);
        }
    }

    // --- Sort descending by confidence score ---
    std::sort(flattenedScores.begin(), flattenedScores.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) > std::get<0>(b);
        });

    // Compute the scaling factor used in preprocessing (letterbox)
    float scale = std::min(input_h / static_cast<float>(originalHeight),
                           input_w / static_cast<float>(originalWidth));

    // Compute offsets added during letterbox
    float pad_w = (input_w - scale * originalWidth) / 2.0f;
    float pad_h = (input_h - scale * originalHeight) / 2.0f;

    // --- Convert boxes & store detections ---
    int numTopK = std::min(static_cast<int>(flattenedScores.size()), num_detections);
    for (int k = 0; k < numTopK; ++k) {
        float score = std::get<0>(flattenedScores[k]);
        if (score < conf_threshold) {
            break;
        }

        int queryIdx = std::get<1>(flattenedScores[k]);
        int classIdx = std::get<2>(flattenedScores[k]);

        // Get box (cx, cy, w, h) predicted in normalized coordinates
        const float* rawBoxData = predBoxesData + (queryIdx * 4);

        // Scale normalized coordinates to network input dimensions
        float cx = rawBoxData[0] * input_w;
        float cy = rawBoxData[1] * input_h;
        float w  = rawBoxData[2] * input_w;
        float h  = rawBoxData[3] * input_h;

        // Convert from center-format to corner-format
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Remove letterbox padding to map back to original image space
        // (x - pad) then divide by scale
        x1 = (x1 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
        x2 = (x2 - pad_w) / scale;
        y2 = (y2 - pad_h) / scale;

        // Clip coordinates to image boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(originalWidth - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(originalHeight - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(originalWidth - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(originalHeight - 1)));

        if (x2 > x1 && y2 > y1) {
            Detection det;
            det.bbox = cv::Rect(static_cast<int>(std::round(x1)),
                                static_cast<int>(std::round(y1)),
                                static_cast<int>(std::round(x2 - x1)),
                                static_cast<int>(std::round(y2 - y1)));
            det.conf = score;
            det.class_id = classIdx - 1; 
            output.push_back(det);
        }
    }
    
    std::cout << "[INFO] Found " << output.size() << " detections passing the confidence threshold." << std::endl;
}



void RF_DETR::build(std::string onnxPath, nvinfer1::ILogger& logger){
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();

    
    if (isFP16){
        config->setFlag(BuilderFlag::kFP16);
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) {
        std::cerr << "Failed to parse ONNX model!" << std::endl;
        exit(EXIT_FAILURE);
    }

    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan) {
        std::cerr << "Failed to build serialized network!" << std::endl;
        exit(EXIT_FAILURE);
    }

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool RF_DETR::saveEngine(const std::string& onnxpath){
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos){
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else{
        return false;
    }

    // Save the engine to the path
    if (engine){
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open()){
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

void RF_DETR::draw(cv::Mat& image, const std::vector<Detection>& output) {
    for (const auto &detection : output) {
        int class_id = detection.class_id;
        cv::Scalar color(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Draw bounding box
        cv::rectangle(image, detection.bbox, color, 2);

        // Prepare label with confidence score
        std::string label = cv::format("%s: %.2f", CLASS_NAMES[class_id].c_str(), detection.conf);
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw background rectangle for label
        cv::rectangle(image,
                      cv::Point(detection.bbox.x, detection.bbox.y - labelSize.height - baseline),
                      cv::Point(detection.bbox.x + labelSize.width, detection.bbox.y),
                      color,
                      cv::FILLED);
        // Draw text label
        cv::putText(image, label,
                    cv::Point(detection.bbox.x, detection.bbox.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1);
    }
}
