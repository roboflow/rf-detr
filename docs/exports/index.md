---
description: Per-format export guides for RF-DETR — ONNX, TensorRT, TFLite, LiteRT, OpenVINO, ExecuTorch, and native CoreML.
---

# Export Formats

These pages cover each supported export format in detail. For a general overview of the export API, shared parameters, and output-file naming, see [Export RF-DETR Model](../learn/export.md).

- [ONNX Inference](onnx.md) — run an exported ONNX model with ONNX Runtime.
- [TensorRT](tensorrt.md) — build a `.trt` engine for NVIDIA GPUs, directly or from an existing ONNX file.
- [TFLite](tflite.md) — ONNX → TensorFlow → TFLite conversion for mobile and edge devices.
- [LiteRT](litert.md) — PyTorch → `.tflite` via `litert-torch`, no ONNX or TensorFlow step.
- [OpenVINO](openvino.md) — OpenVINO IR for Intel CPUs, GPUs, and NPUs.
- [ExecuTorch](executorch.md) — `.pte` binaries for XNNPACK, CoreML, and QNN backends.
- [Native CoreML](coreml.md) — `.mlpackage` export for Xcode / Apple platforms.
