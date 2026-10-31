#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <iostream>

// Static buffers
static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

struct AffineMatrix {
    float value[6];
};

// CUDA error checking macro
#define CUDA_CALL(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel with logs
__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx = position % dst_width;
    int dy = position / dst_width;

    // Transform source coordinates
    float src_x = d2s.value[0] * dx + d2s.value[1] * dy + d2s.value[2] + 0.5f;
    float src_y = d2s.value[3] * dx + d2s.value[4] * dy + d2s.value[5] + 0.5f;

    // printf("Thread %d: (dx, dy) = (%d, %d), (src_x, src_y) = (%.2f, %.2f)\n",
    //        position, dx, dy, src_x, src_y);

    float c0, c1, c2;

    // Check if source coordinates are out of bounds
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        c0 = c1 = c2 = const_value_st;  // Default value for out-of-range
    } else {
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;

        // Handle boundary conditions
        uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
        uint8_t* v2 = (x_high < src_width) ? src + y_low * src_line_size + x_high * 3 : v1;
        uint8_t* v3 = (y_high < src_height) ? src + y_high * src_line_size + x_low * 3 : v1;
        uint8_t* v4 = (x_high < src_width && y_high < src_height) ? src + y_high * src_line_size + x_high * 3 : v1;

        // Bilinear interpolation weights
        float lx = src_x - x_low;
        float ly = src_y - y_low;
        float hx = 1 - lx;
        float hy = 1 - ly;
        float w1 = hx * hy, w2 = lx * hy, w3 = hx * ly, w4 = lx * ly;

        // Compute final pixel values
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // Convert BGR to RGB
    float tmp = c0; c0 = c2; c2 = tmp;

    // Normalize pixel values
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;

    // Store the normalized values
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// Host-side preprocessing function
void cuda_preprocess(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream) {

    int img_size = src_width * src_height * 3;

    if (img_buffer_host == nullptr) {
        std::cerr << "Error: img_buffer_host not allocated!" << std::endl;
    }

    if (src == nullptr) {
        std::cerr << "Error: Source image pointer is null!" << std::endl;
    }



    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    cudaDeviceSynchronize();
    std::cout << "Synchronized CUDA device." << std::endl;

    // Copy data to pinned memory
    std::cout << "Copying data to pinned memory..." << std::endl;
    memcpy(img_buffer_host, src, img_size);

    // Copy data to device memory
    std::cout << "Copying data to device memory..." << std::endl;
    CUDA_CALL(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    // Prepare the affine matrices
    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale; s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0; s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_width * dst_height;
    int threads = 256;
    int blocks = (jobs + threads - 1) / threads;

    // Launch the kernel
    std::cout << "Launching kernel..." << std::endl;
    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        img_buffer_device, src_width * 3, src_width, src_height,
        dst, dst_width, dst_height, 128, d2s, jobs);

    // Synchronize and check for errors
    CUDA_CALL(cudaStreamSynchronize(stream));
    std::cout << "Kernel execution completed." << std::endl;
}

void cuda_preprocess_init(int max_image_size) {
    CUDA_CALL(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
    CUDA_CALL(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
    CUDA_CALL(cudaFree(img_buffer_device));
    CUDA_CALL(cudaFreeHost(img_buffer_host));
}
