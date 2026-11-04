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
    - INT8 quantization: dynamic-range for TFLite (no calibration data), static for ONNX, OpenVINO and TensorRT (calibration data required; see [TensorRT INT8](tensorrt.md#int8))
    - Custom input resolutions supported (must be divisible by `patch_size × num_windows`, which varies by model variant)
    - Export to ExecuTorch for on-device PyTorch inference (XNNPACK, CoreML, QNN)
    - Export directly to native CoreML (`.mlpackage`) for Xcode / Apple-platform deployment
    - Adding a format is an in-tree contribution — see [Exporter Blueprint](blueprint.md)
    - Per-format details are in the format guides below

RF-DETR supports exporting models to ONNX, TFLite, LiteRT, ExecuTorch, native CoreML, Apple Core AI and OpenVINO IR formats, enabling deployment across a wide range of inference frameworks, edge devices, and hardware accelerators.

The export docs are split into three pages:

- **Overview** (this page) — supported formats and measured performance per hardware.
- [Export Basics](basics.md) — installation, a first `model.export()` call, output files, and prediction with `RFDETRInference`.
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

Which format is fastest depends entirely on the hardware you deploy to. The four per-hardware cookbooks each export every format targeting one class of device, run inference on it, and benchmark it against a PyTorch baseline on the same machine. This page keeps only the best format and configuration per hardware, with the PyTorch anchor it was measured against, and the [PyTorch baseline](#pytorch-baseline) rows of each machine; each format's page carries its own per-configuration, per-batch table (latency, FPS, mAP, size, memory) and no other format's rows.

| Hardware                             | Format          | Package            | Runtime              | Precision | end2end [ms]   | FPS [img/s] | mAP@50:95 [%] | PyTorch `predict()` anchor | Full table                                                                                   |
| ------------------------------------ | --------------- | ------------------ | -------------------- | --------- | -------------- | ----------- | ------------- | -------------------------- | -------------------------------------------------------------------------------------------- |
| NVIDIA L4                            | TensorRT        | `inference-models` | managed engine       | auto      | 3.45 ± 0.15    | 289.8       | —             | 23.46 ms / 42.6 FPS        | [TensorRT](tensorrt.md#measured-latency-on-an-nvidia-l4)                                     |
| Apple M4 Max (40-core GPU)           | PyTorch         | `rfdetr`           | inference() inductor | fp16      | 10.3           | 97.4        | 56.4          | 20.0 ms / 50.0 FPS         | [PyTorch baseline](#pytorch-baseline), [CoreML](coreml.md#measured-latency-on-apple-silicon) |
| x86 CPU (Colab, 4 cores / 8 threads) | OpenVINO        | `rfdetr`           | CPU                  | fp32      | 274.63 ± 23.23 | 3.6         | 56.5          | 345.35 ms / 2.9 FPS        | [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu)                                       |
| Apple M4 Max CPU (12P + 4E cores)    | Pending a rerun | —                  | —                    | —         | —              | —           | —             | 54.54 ms / 18.3 FPS        | [TFLite](tflite.md#measured-latency-on-a-laptop-cpu)                                         |

All numbers are batch 1, `RFDETRSmall`, rfdetr `develop`. Warmup and timed-run counts differ per cookbook (GPU 20 + 100, CPU 15 + 50, Apple and mobile 5 + 30). Compare *within* a row, never across rows. The rows come from three machines: a Colab L4 GPU host, a Colab x86 host, and one Apple M4 Max. The L4 managed engine is not scored for mAP; the `.trt` engine through `RFDETRInference` runs at 5.88 ms with 56.5 mAP@50:95 (see the [TensorRT page](tensorrt.md#measured-latency-on-an-nvidia-l4)). On the x86 host the run-to-run standard deviation is 4 to 10% of the mean, so batch-1 gaps under about 25 ms are noise, not a ranking; ONNX INT8 is the fastest x86 row but costs 1.8 mAP, which is why the best full-accuracy row is listed. The Apple row was measured on macOS 26.7.1. The arm64 CPU row stands in for a phone or edge CPU only approximately: it is a laptop chip, and a phone's own delegate (NNAPI, QNN, Core ML) is not exercised.

### PyTorch Baseline

Each format page holds only that format's rows. The native PyTorch rows every cookbook measures alongside them are collected here, per machine: `predict()` is the eager model, `inference()` the snapshot it returns (TorchScript, plus a CUDA graph on the L4 and inductor on the Mac GPU, which is the MPS default). Latency is batch 1, end to end; `FPS` is images per second at each batch size, `—` where the cookbook did not run that batch. `mAP@50:95` is the same seeded 500-image val2017 subset as the format pages.

| Hardware                             | Runtime                 | Precision | b1 end2end [ms] | mAP@50:95 [%] | FPS b1 | FPS b4 | FPS b8 |
| ------------------------------------ | ----------------------- | --------- | --------------- | ------------- | ------ | ------ | ------ |
| NVIDIA L4                            | predict() eager         | fp32      | 23.46 ± 1.08    | 56.5          | 42.6   | 110.8  | 112.3  |
| NVIDIA L4                            | inference() JIT         | fp16      | 12.87 ± 0.53    | 56.5          | 77.7   | 201.2  | 187.5  |
| NVIDIA L4                            | inference() CUDA Graph  | fp16      | 6.04 ± 0.25     | —             | 165.7  | —      | —      |
| Apple M4 Max (40-core GPU)           | predict() eager         | fp32      | 20.0            | 56.5          | 50.0   | 83.6   | 93.0   |
| Apple M4 Max (40-core GPU)           | inference() TorchScript | fp16      | 12.2            | 56.5          | 81.9   | 131.3  | 143.5  |
| Apple M4 Max (40-core GPU)           | inference() inductor    | fp32      | 11.1            | 56.5          | 90.4   | 101.2  | 120.8  |
| Apple M4 Max (40-core GPU)           | inference() inductor    | fp16      | 10.3            | 56.4          | 97.4   | 127.0  | 141.6  |
| x86 CPU (Colab, 4 cores / 8 threads) | predict() eager         | fp32      | 345.35 ± 24.11  | 56.5          | 2.9    | 3.2    | —      |
| x86 CPU (Colab, 4 cores / 8 threads) | inference() JIT         | fp32      | 320.39 ± 19.53  | 56.5          | 3.1    | 3.3    | —      |
| Apple M4 Max CPU (12P + 4E cores)    | predict() eager         | fp32      | 54.54 ± 0.71    | —             | 18.3   | 25.5   | —      |
| Apple M4 Max CPU (12P + 4E cores)    | inference() JIT         | fp32      | 59.37 ± 5.07    | —             | 16.8   | 26.2   | —      |

On the L4, `inference()` with fp16 and a CUDA graph (6.04 ms) is within 3% of the `.trt` engine at batch 1 and keeps the native `predict()` API. The Apple CPU rows are pending the same rerun as the mobile format pages.

### Side by side with `inference-models`

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is Roboflow's separate deployment package. It serves RF-DETR through PyTorch, ONNX Runtime and TensorRT, so only those formats can be set against `rfdetr`; OpenVINO, TFLite, LiteRT, ExecuTorch, Core AI and CoreML have no `inference-models` backend (it lists a CoreML backend, but publishes no CoreML model for `rfdetr-small`). Both packages ran in the same cookbook on the same machine, with the same image, threshold (0.5; 0.001 for mAP), batch sizes and end-to-end scope. Each cookbook passes `input_color_format="rgb"` to `inference-models`: a NumPy array is BGR to it unless declared, and without that the scores drop with no error.

| Hardware                   | Format  | Package            | Runtime              | Precision | b1 end2end [ms] | mAP@50:95 [%] | FPS b1 | FPS b4 | FPS b8 |
| -------------------------- | ------- | ------------------ | -------------------- | --------- | --------------- | ------------- | ------ | ------ | ------ |
| Apple M4 Max (40-core GPU) | PyTorch | `rfdetr`           | predict() eager      | fp32      | 20.0            | 56.5          | 50.0   | 83.6   | 93.0   |
| Apple M4 Max (40-core GPU) | PyTorch | `rfdetr`           | inference() inductor | fp16      | 10.3            | 56.4          | 97.4   | 127.0  | 141.6  |
| Apple M4 Max (40-core GPU) | PyTorch | `inference-models` | torch (MPS)          | fp32      | 25.6            | 56.4          | 39.1   | 56.6   | 66.4   |
| Apple M4 Max CPU           | ONNX    | `rfdetr`           | CPU EP               | fp32      | 98.7            | 56.5          | 10.1   | 10.7   | 8.9    |
| Apple M4 Max CPU           | ONNX    | `inference-models` | CPU EP               | fp32      | 114.6           | 56.4          | 8.7    | 8.0    | 9.5    |
| Apple M4 Max (CoreML EP)   | ONNX    | `inference-models` | CoreML EP            | fp32      | 22.5            | 56.4          | 44.4   | 46.1   | 36.0   |

On the same runtime the two packages are close, and the accuracy is the same within 0.1 mAP: the ONNX CPU rows differ by 16%, and at batch 1 the `inference-models` PyTorch model (25.6 ms, fp32) is slower than eager `predict()` (20.0 ms) and 2.3 times slower than `inference()` with inductor in fp32 (11.1 ms). The ONNX CoreML EP row runs a different execution provider from the CPU rows, so it is not a like-for-like comparison with them; `rfdetr`'s ONNX loader does not select that provider. These Apple rows come from a busy machine (load average above 10), so treat gaps under about 15% as noise. The [CPU](../cookbooks/export-cpu/) and [CUDA](../cookbooks/export-cuda/) cookbooks produce the same comparison for an x86 CPU and an NVIDIA GPU.

`Memory [MB]` in the cookbooks is comparable only within one cookbook: the CUDA cookbook measures device memory, the CPU, Apple and mobile cookbooks measure host resident memory (RSS).

Each cookbook also scores every exported model and precision at batch 1 on the same seeded 500-image COCO val2017 subset (box mAP@50:95 and mAP@50), so a precision setting that loses accuracy shows up next to its latency. The subset is chosen for run time, so these are not the full-val2017 numbers on the [Benchmarks page](../learn/benchmarks.md); set `FULL_VAL = True` in a cookbook to score all 5000 images.

!!! note "Which latency number is this?"

    RF-DETR latency appears on three different pages, and they are three different measurements:

    - **Architecture headline** (T4, TensorRT FP16, model only, batch 1) — the number that compares architectures. See the [Benchmarks page](../learn/benchmarks.md).
    - **Deployed product** (L4, Roboflow Inference, scope not stated, batch 1) — what a Roboflow Inference user gets. See [Roboflow's RF-DETR model page](https://docs.roboflow.com/models/supported-models/rf-detr).
    - **Export cookbooks** (this page and below) — end-to-end timings for every export format, on one machine per cookbook. Their value is **ratios within one machine**, not absolute milliseconds to set against the other two pages.

    Never compare milliseconds across these three without checking hardware, precision, batch size, and scope (model-only vs end-to-end) match. The cookbook's L4 end-to-end times at batch 1 (3.45 ms for the managed TensorRT engine, 5.88 ms for the `.trt` engine, 15.45 ms for ONNX Runtime) include host preprocessing and postprocessing, and are not comparable with the 8.3 ms TensorRT and 12.9 ms ONNX on Roboflow's model page, which does not state what its timing covers. The two harnesses have not been run side by side yet, so treat all of these as unreconciled.

!!! warning "fp16 pays off only where the silicon implements it"

    Reduced precision is not a portable speedup, and the cookbooks measure this directly:

    - **GPU — large win.** On an L4, TensorRT's auto-selected precision is 1.93× faster than the same engine forced to fp32 (3.45 vs 6.67 ms end-to-end, batch 1), and PyTorch `inference(dtype=torch.float16)` cuts the eager fp32 baseline from 23.46 to 12.87 ms.
    - **CPU — no measured win from fp16.** A CPU without native fp16 kernels upconverts and computes in fp32, so fp16 storage saves disk and bandwidth, not arithmetic. On the Colab x86 host the OpenVINO fp16 IR ran at 291.8 ms against 274.6 ms for the fp32 IR at batch 1, within one standard deviation, at half the size. INT8 is the lever that changes CPU arithmetic: ONNX INT8 ran 1.5 times faster than ONNX fp32 (251.5 against 378.0 ms) for 1.8 mAP, and OpenVINO INT8 matched its fp32 IR's speed at 3.6 times smaller. See the [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu) and [ONNX](onnx.md#on-an-x86-cpu) pages. The TFLite rows are pending a rerun: they ran through `tf.lite.Interpreter` at its default thread count (the same fp32 `.tflite` runs in 119 ms on one thread and 59 ms on eight through `ai_edge_litert` on the M4 Max, against 785 ms in the cookbook).
    - **Apple — runtime-dependent.** On the M4 Max, CoreML fp16 came out *slower* than its fp32 default (23.6 against 12.0 ms at batch 1, macOS 26.7.1) at half the size, and ExecuTorch's CoreML backend (fp16 only) matched CoreML fp16 at batch 1. An earlier macOS 27.0 run measured Core AI fp16 faster than its fp32 default (13.53 against 14.62 ms); Core AI cannot be measured on an older macOS, where it runs on the CPU only. Measure per runtime, not per platform. See the [CoreML](coreml.md#measured-latency-on-apple-silicon) and [ExecuTorch](executorch.md#measured-latency-on-apple-silicon) pages.

## Per-Format Guides

The format guides listed above cover installation, export examples, output files, and inference code.

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](../learn/deploy.md) for cloud-based inference and workflow integration
- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) as a separate option for PyTorch, ONNX, or TensorRT inference
- Deploy TFLite and LiteRT `.tflite` models on mobile/edge devices with the LiteRT runtime
- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
- Read the [Exporter Blueprint](blueprint.md) to add a new export format
- Browse the format guides above for per-format details

\* `TRTInference` and `OpenVINOInference` are retained here to identify the runtimes used for the measured benchmarks. Both are deprecated since v1.12.0 and will be removed in v1.15.0. Use `RFDETRInference` for image prediction; see the [migration guide](../getting-started/migration.md#deprecated-in-v112-remove-in-v115).
