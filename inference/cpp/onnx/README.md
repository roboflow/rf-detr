# ONNX Inference in C++

This guide explains how to set up, build, and run the ONNX model inference demo in C++ using ONNX Runtime and OpenCV.

---

## 📌 Prerequisites

Ensure you have the following installed:

- **CMake** (3.5.1 or later)
- **C++ Compiler** (with C++17 support)
- **ONNX Runtime** → Install it following [this guide](https://onnxruntime.ai/docs/genai/howto/install.html).
- **OpenCV** → Install via a package manager:

  - **Windows (vcpkg)** → `vcpkg install opencv`
  - **Ubuntu** → `sudo apt update && sudo apt install libopencv-dev`
  - **macOS (Homebrew)** → `brew install opencv`

---

## ⚙️ Build & Run

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/roboflow/rf-detr
cd inference/cpp/onnx
mkdir build && cd build
```

### 2️⃣ Configure & Compile

#### **🔹 Linux/macOS**
```sh
cmake ..
cmake --build .
```

#### **🔹 Windows (MSVC)**
```sh
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### 3️⃣ Run the Model

This PR introduces a C++ demo for the RF-DETR model, allowing users to perform real-time object detection using an ONNX model. The demo supports various input sources, including images, videos, and live camera streams, with optional CUDA acceleration.

### 🔹 Key Features
✅ Loads an RF-DETR model in ONNX format  
✅ Supports image, video, and live camera inference  
✅ Enables CPU and CUDA (GPU) execution  
✅ Configurable confidence threshold for detections  
✅ Outputs annotated images/videos with detected objects  
✅ Uses COCO class labels for object recognition  

### 🔹 Run Examples

#### **Image Inference**
Detect objects in a static image and save the output:
```sh
./onnxExample --model path/to/model.onnx --source_type image \
  --input path/to/image.jpg --output path/to/output.jpg \
  --conf 0.6 --labels path/to/coco.names
```

#### **Video Inference**
Process a video file and save the annotated output:
```sh
./onnxExample --model path/to/model.onnx --source_type video \
  --input path/to/video.mp4 --output path/to/output.mp4 \
  --conf 0.5 --use_cuda
```

#### **Live Camera Inference (Default ID 0)**
Run inference on the default webcam (ID 0) with GPU acceleration:
```sh
./onnxExample --model path/to/model.onnx --source_type camera \
  --input 0 --conf 0.55 --use_cuda
```

#### **Live Camera Inference (Specific Camera ID 1)**
Run inference on a specific camera (ID 1):
```sh
./onnxExample --model path/to/model.onnx --source_type camera \
  --input 1 --conf 0.55
```

#### **Get Help & Available Options**
```sh
./onnxExample --help
```

### 🔹 Dependencies
- **OpenCV** (for image and video processing)
- **ONNX Runtime** (for model inference)
- **CUDA** (optional, for GPU acceleration)

---

## 📝 Notes

- Ensure the ONNX model and input image are accessible.
- On Windows, make sure `onnxruntime.dll` is in the same directory as `onnxExample.exe` or added to the `PATH`.
- Modify `onnxExample.cpp` as needed for preprocessing or output handling.

