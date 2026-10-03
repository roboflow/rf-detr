---
description: Overview of exporting RF-DETR models to ONNX, TensorRT, TFLite, LiteRT, ExecuTorch, native CoreML, Apple Core AI and OpenVINO IR (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
---

# Export RF-DETR Model

!!! tip "Key Takeaways"

    - Export to ONNX for cross-platform inference with ONNX Runtime, OpenVINO, or TensorRT
    - Export to OpenVINO IR for optimized inference on CPU (x86, ARM), GPU (Intel integrated & discrete GPU) and AI accelerators (Intel NPU)
    - Export to TFLite (FP32, FP16, INT8) for mobile and edge deployment
    - Export to LiteRT (`.tflite`) straight from PyTorch with `litert-torch` — no ONNX or TensorFlow step
    - TensorRT conversion delivers the lowest latency on NVIDIA GPUs — 2.3 ms for Nano on a T4, TensorRT FP16, model only, batch 1 (the architecture headline on the [Benchmarks page](../learn/benchmarks.md); this page's own L4 end-to-end numbers below are a different measurement, see "Which latency number is this?")
    - INT8 quantization is dynamic-range and needs no calibration data
    - Custom input resolutions supported (must be divisible by `patch_size × num_windows`, which varies by model variant)
    - Export to ExecuTorch for on-device PyTorch inference (XNNPACK, CoreML, QNN)
    - Export directly to native CoreML (`.mlpackage`) for Xcode / Apple-platform deployment
    - Adding a format is an in-tree contribution — see [Exporter Blueprint](blueprint.md)
    - Per-format details are in the format guides below

RF-DETR supports exporting models to ONNX, TFLite, LiteRT, ExecuTorch, native CoreML, Apple Core AI and OpenVINO IR formats, enabling deployment across a wide range of inference frameworks, edge devices, and hardware accelerators.

The export docs are split into three pages:

- **Overview** (this page) — supported formats and measured performance per hardware.
- [Export Basics](basics.md) — installation, a first `model.export()` call, output files, and inference with `inference-models`.
- [Advanced Export](advanced.md) — the full parameter reference, custom resolution, backbone-only export, and how the export pipeline works.

For detailed installation, examples, and inference code per format, see the format guides:

- [ONNX Inference](onnx.md) — run an exported ONNX model with ONNX Runtime.
- [TensorRT](tensorrt.md) — build a `.trt` engine for NVIDIA GPUs, directly or from an existing ONNX file.
- [TFLite](tflite.md) — ONNX → TensorFlow → TFLite conversion for mobile and edge devices.
- [LiteRT](litert.md) — PyTorch → `.tflite` via `litert-torch`, no ONNX or TensorFlow step.
- [OpenVINO](openvino.md) — OpenVINO IR for Intel CPUs, GPUs, and NPUs.
- [ExecuTorch](executorch.md) — `.pte` binaries for XNNPACK, CoreML, and QNN backends.
- [Native CoreML](coreml.md) — `.mlpackage` export for Xcode / Apple platforms.
- [Core AI](coreai.md) — `.aimodel` export for iOS / iPadOS / macOS 27+.

## Measured Performance by Hardware

Which format is fastest depends entirely on the hardware you deploy to. The four per-hardware cookbooks each export every format targeting one class of device, run inference on it, and benchmark it against a PyTorch baseline on the same machine. Below is the fastest end-to-end result per hardware class, plus the PyTorch anchor it was measured against; the cookbooks carry the full tables, including forward-only timings, memory, and the slower configurations.

| Hardware                           | Fastest format                                                   | end2end [ms]   | FPS [img/s] | PyTorch `predict()` anchor | Cookbook                              |
| ---------------------------------- | ---------------------------------------------------------------- | -------------- | ----------- | -------------------------- | ------------------------------------- |
| NVIDIA L4 (pkg `inference-models`) | TensorRT via `inference-models` (managed engine, auto precision) | 3.41 ± 0.06    | 293.5       | 19.88 ms / 50.3 FPS        | [CUDA](../cookbooks/export-cuda/)     |
| NVIDIA L4 (plain `rfdetr`)         | TensorRT raw `.trt` engine (`TRTInference`, fp16, no extra pkg)  | 9.82 ± 0.34    | 101.8       | 19.88 ms / 50.3 FPS        | [CUDA](../cookbooks/export-cuda/)     |
| Apple M-series (ANE/GPU)           | CoreML fp32 (default)                                            | 11.57 ± 0.28   | 86.5        | 24.12 ms / 41.5 FPS        | [Apple](../cookbooks/export-apple/)   |
| x86 CPU (4 cores)                  | OpenVINO fp32 IR                                                 | 357.47 ± 56.88 | 2.8         | 377.34 ms / 2.7 FPS        | [CPU](../cookbooks/export-cpu/)       |
| ARM CPU (edge proxy)               | ExecuTorch XNNPACK, fp32                                         | 80.74 ± 1.90   | 12.4        | 54.54 ms / 18.3 FPS        | [Mobile](../cookbooks/export-mobile/) |

All numbers are batch 1, `RFDETRSmall`, rfdetr `develop` (version not recorded). Warmup and timed-run counts differ per cookbook (GPU uses 20 + 100, CPU 15 + 50, Apple and mobile 5 + 30), and each row was measured on different hardware, so compare *within* a row's hardware class, never across rows. The x86 CPU figures come from a shared Colab vCPU where run-to-run noise is 15–20% of the mean — on that machine no CPU format separates from the others by more than one standard deviation. **Two L4 rows, on purpose**: the first goes through `inference-models`, a separate package that selects and loads its own prebuilt TensorRT engine; the second times the `.trt` file `model.export(format="trt")` itself writes, loaded directly with `rfdetr`'s own `TRTInference` reference runtime — no extra package beyond `rfdetr[tensorrt]`. Both are real end-to-end numbers on the same L4 — see the [CUDA cookbook](../cookbooks/export-cuda/#6-tensorrt) for the full table and the open gap against Roboflow's own product-page number.

!!! note "Which latency number is this?"

    RF-DETR latency appears on three different pages, and they are three different measurements:

    - **Architecture headline** (T4, TensorRT FP16, model only, batch 1) — the number that compares architectures. See the [Benchmarks page](../learn/benchmarks.md).
    - **Deployed product** (L4, Roboflow Inference, scope not stated, batch 1) — what a Roboflow Inference user gets. See [Roboflow's RF-DETR model page](https://docs.roboflow.com/models/supported-models/rf-detr).
    - **Export cookbooks** (this page and below) — end-to-end and forward-only timings for every export format, on one machine per cookbook. Their value is **ratios within one machine**, not absolute milliseconds to set against the other two pages.

    Never compare milliseconds across these three without checking hardware, precision, batch size, and scope (model-only vs end-to-end) match. The CUDA cookbook's own L4 forward-only times (ONNX 7.35 ms, TensorRT 2.16 ms) are lower than the 12.9 and 8.3 ms on Roboflow's model page, which does not state what its timing covers. The two harnesses have not been run side by side yet, so treat all of these as unreconciled — see the [CUDA cookbook's Results section](../cookbooks/export-cuda/#7-results).

!!! warning "fp16 pays off only where the silicon implements it"

    Reduced precision is not a portable speedup, and the cookbooks measure this directly:

    - **GPU — large win.** On an L4, TensorRT's auto-selected precision is 1.62× faster than the same engine forced to fp32 (4.91 vs 7.96 ms), and PyTorch `inference(dtype=torch.float16)` nearly halves the eager fp32 baseline (11.30 vs 19.88 ms).
    - **CPU — no win.** OpenVINO's default FP16 IR came out *slower* than explicit `float32` (354.55 vs 311.92 ms) despite halving the file, and TFLite fp16 matched fp32 exactly (780.86 vs 780.74 ms). A CPU without native fp16 kernels upconverts and computes in fp32, so fp16 saves disk and bandwidth, not arithmetic. On CPU the lever is **INT8**: dynamic-range quantization was ~2.5× faster than fp32 TFLite (319.61 vs 780.74 ms), at the cost of one dropped detection on the sample image.
    - **Apple — runtime-dependent.** Core AI fp16 beat its fp32 default (11.45 vs 12.54 ms), while CoreML fp16 came out *slower* than its fp32 default (20.05 vs 11.62 ms) on the same chip. Both results reproduced across runs. Measure per runtime, not per platform.

## Per-Format Guides

The format guides listed above cover installation, export examples, output files, and inference code.

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](../learn/deploy.md) for cloud-based inference and workflow integration
- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) for multi-backend inference (PyTorch, ONNX, TensorRT) with automatic backend selection
- Deploy TFLite and LiteRT `.tflite` models on mobile/edge devices with the LiteRT runtime
- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
- Read the [Exporter Blueprint](blueprint.md) to add a new export format
- Browse the format guides above for per-format details
