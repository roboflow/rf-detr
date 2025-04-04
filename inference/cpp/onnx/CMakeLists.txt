cmake_minimum_required(VERSION 3.10)
project(RF_DETR_ONNX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime/core/session/onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIBRARIES NAMES onnxruntime)

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_DIR to the ONNX Runtime installation path.")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

# Source files
set(SOURCES 
    main.cpp 
    RF_DETR_ONNX.cpp
)

# Header files
set(HEADERS 
    RF_DETR_ONNX.h
)

# Create executable
add_executable(RF_DETR_ONNX ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(RF_DETR_ONNX ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})

# Platform-specific settings
if(WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime libs: ${ONNXRUNTIME_LIBRARIES}")
