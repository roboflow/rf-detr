cmake_minimum_required(VERSION 3.5.1)

project(
    onnxExample
    LANGUAGES CXX
    VERSION 1.0.0
)

set(CMAKE_CXX_STANDARD 17)

# Compiler options
if (MSVC)
    add_compile_options(/W4 /permissive-)
    add_definitions(-DNOMINMAX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ONNX Runtime package
find_package(onnxruntime REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED core imgcodecs imgproc)

# Add executable
add_executable(
    onnxExample
    onnxExample.cpp
)

# Link libraries
target_link_libraries(
    onnxExample
    onnxruntime::onnxruntime
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)