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
| NVIDIA L4                            | TensorRT        | `rfdetr`           | .trt engine          | fp16      | 3.6            | 278.7       | 56.5          | 19.5 ms / 51.2 FPS         | [TensorRT](tensorrt.md#measured-latency-on-an-nvidia-l4)                                     |
| NVIDIA L4                            | TensorRT        | `inference-models` | managed engine       | auto      | 3.6            | 277.7       | 56.3          | 19.5 ms / 51.2 FPS         | [TensorRT](tensorrt.md#why-the-managed-engine-is-faster-end-to-end)                          |
| Apple M4 Max (40-core GPU)           | PyTorch         | `rfdetr`           | inference() inductor | fp16      | 10.3           | 97.4        | 56.4          | 20.0 ms / 50.0 FPS         | [PyTorch baseline](#pytorch-baseline), [CoreML](coreml.md#measured-latency-on-apple-silicon) |
| x86 CPU (Colab, 4 cores / 8 threads) | OpenVINO        | `rfdetr`           | CPU                  | fp32      | 373.65 ± 61.10 | 2.7         | 56.5          | 401.58 ms / 2.5 FPS        | [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu)                                       |
| Apple M4 Max CPU (12P + 4E cores)    | Pending a rerun | —                  | —                    | —         | —              | —           | —             | 54.54 ms / 18.3 FPS        | [TFLite](tflite.md#measured-latency-on-a-laptop-cpu)                                         |

All numbers are batch 1, `RFDETRSmall`, rfdetr `develop`. Warmup and timed-run counts differ per cookbook (GPU 20 + 100, CPU 15 + 50, Apple and mobile 5 + 30). Compare *within* a row, never across rows. The rows come from three machines: a Colab L4 GPU host, a Colab x86 host, and one Apple M4 Max. On the L4 both rows time the same decoded NumPy array, with `include_source_image=False` for `rfdetr`, and they are level at batch 1 (3.6 ms); [the TensorRT page](tensorrt.md#why-the-managed-engine-is-faster-end-to-end) describes how the earlier gap between them closed once both were timed on the same decoded array with no source copy; the rerun changed several things at once, so the gap is not assigned to the input type alone. The `inference()` CUDA graph row (3.5 ms) is not scored; the same weights score 56.5 mAP@50:95 through `inference()` with TorchScript. On the x86 host the run-to-run standard deviation is 10 to 17% of the mean for most rows, so on a CPU batch-1 gaps under about 100 ms are noise, not a ranking: the OpenVINO fp32 row is listed as the best full-accuracy route, but eager PyTorch, TorchScript and `inference-models` ONNX are level with it on this run. The Apple row was measured on macOS 26.7.1. The arm64 CPU row stands in for a phone or edge CPU only approximately: it is a laptop chip, and a phone's own delegate (NNAPI, QNN, Core ML) is not exercised.

### Fastest route across NVIDIA GPUs

The fastest measured route per GPU and size, end to end on one decoded RGB array, `rfdetr` 1.12.0.dev0, TensorRT 10.16.1.11, CUDA 13.2. Cells show `ms` at batch 1 and `img/s` at batch 8, with the winning route: C = `inference()` CUDA graph FP16, F = `.trt` FP16, M = `inference-models` managed engine. Latency only; no accuracy was scored, and INT8 engines are left out because no harness in the repository reproduces their end-to-end figures yet. All routes and the `n/a` cases are in [TensorRT](tensorrt.md#measured-latency-across-nvidia-gpus).

| GPU          | Nano                         | Small                        | Medium                       | Large                        |
| ------------ | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| T4           | 5.12 ms (F) / 334 img/s (F)  | 5.89 ms (M) / 197 img/s (F)  | 7.20 ms (F) / 153 img/s (M)  | 10.22 ms (M) / 99 img/s (M)  |
| L4           | 2.78 ms (C) / 724 img/s (F)  | 3.57 ms (F) / 432 img/s (F)  | 4.13 ms (F) / 349 img/s (F)  | 5.14 ms (F) / 227 img/s (F)  |
| A100         | 3.16 ms (C) / 967 img/s (F)  | 3.44 ms (F) / 783 img/s (F)  | 3.80 ms (C) / 676 img/s (F)  | 4.27 ms (C) / 558 img/s (F)  |
| RTX PRO 6000 | 1.47 ms (F) / 2271 img/s (F) | 1.62 ms (F) / 1782 img/s (F) | 1.76 ms (F) / 1608 img/s (F) | 2.08 ms (F) / 1249 img/s (F) |

### PyTorch Baseline

Each format page holds only that format's rows. The native PyTorch rows every cookbook measures alongside them are collected here, per machine: `predict()` is the eager model, `inference()` the snapshot it returns (TorchScript, plus a CUDA graph on the L4 and inductor on the Mac GPU, which is the MPS default). Latency is batch 1, end to end; `FPS` is images per second at each batch size, `—` where the cookbook did not run that batch. `mAP@50:95` is the same seeded 500-image val2017 subset as the format pages.

| Hardware                             | Runtime                 | Precision | b1 end2end [ms] | mAP@50:95 [%] | FPS b1 | FPS b4 | FPS b8 |
| ------------------------------------ | ----------------------- | --------- | --------------- | ------------- | ------ | ------ | ------ |
| NVIDIA L4                            | predict() eager         | fp32      | 19.5            | 56.5          | 51.2   | 139.8  | 149.2  |
| NVIDIA L4                            | inference() JIT         | fp16      | 9.9             | 56.5          | 101.1  | 340.8  | 329.2  |
| NVIDIA L4                            | inference() CUDA Graph  | fp16      | 3.5             | —             | 283.9  | 362.4  | 337.7  |
| Apple M4 Max (40-core GPU)           | predict() eager         | fp32      | 20.0            | 56.5          | 50.0   | 83.6   | 93.0   |
| Apple M4 Max (40-core GPU)           | inference() TorchScript | fp16      | 12.2            | 56.5          | 81.9   | 131.3  | 143.5  |
| Apple M4 Max (40-core GPU)           | inference() inductor    | fp32      | 11.1            | 56.5          | 90.4   | 101.2  | 120.8  |
| Apple M4 Max (40-core GPU)           | inference() inductor    | fp16      | 10.3            | 56.4          | 97.4   | 127.0  | 141.6  |
| x86 CPU (Colab, 4 cores / 8 threads) | predict() eager         | fp32      | 401.58 ± 66.19  | 56.5          | 2.5    | 2.8    | —      |
| x86 CPU (Colab, 4 cores / 8 threads) | inference() JIT         | fp32      | 377.85 ± 62.10  | 56.5          | 2.6    | 2.8    | —      |
| Apple M4 Max CPU (12P + 4E cores)    | predict() eager         | fp32      | 54.54 ± 0.71    | —             | 18.3   | 25.5   | —      |
| Apple M4 Max CPU (12P + 4E cores)    | inference() JIT         | fp32      | 59.37 ± 5.07    | —             | 16.8   | 26.2   | —      |

On the L4, `inference()` with fp16 and a CUDA graph (3.5 ms) is level with the `.trt` engine loaded through `RFDETRInference` at batch 1 (3.6 ms) and slower at batch 4 and 8 (362.4 and 337.7 against 402.4 and 401.9 img/s), and keeps the native `predict()` API. The Apple CPU rows are pending the same rerun as the mobile format pages.

### Side by side with `inference-models`

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is Roboflow's separate deployment package. It serves RF-DETR through PyTorch, ONNX Runtime and TensorRT, so only those formats can be set against `rfdetr`; OpenVINO, TFLite, LiteRT, ExecuTorch, Core AI and CoreML have no `inference-models` backend (it lists a CoreML backend, but publishes no CoreML model for `rfdetr-small`). Both packages ran in the same cookbook on the same machine, with the same image, threshold (0.5; 0.001 for mAP), batch sizes. The CUDA cookbook times every row on the same decoded RGB array, conversion to `sv.Detections` included. Each cookbook passes `input_color_format="rgb"` to `inference-models`: a NumPy array is BGR to it unless declared, and without that the scores drop with no error.

| Hardware                   | Format   | Package            | Runtime                | Precision | b1 end2end [ms] | mAP@50:95 [%] | FPS b1 | FPS b4 | FPS b8 |
| -------------------------- | -------- | ------------------ | ---------------------- | --------- | --------------- | ------------- | ------ | ------ | ------ |
| Apple M4 Max (40-core GPU) | PyTorch  | `rfdetr`           | predict() eager        | fp32      | 20.0            | 56.5          | 50.0   | 83.6   | 93.0   |
| Apple M4 Max (40-core GPU) | PyTorch  | `rfdetr`           | inference() inductor   | fp16      | 10.3            | 56.4          | 97.4   | 127.0  | 141.6  |
| Apple M4 Max (40-core GPU) | PyTorch  | `inference-models` | torch (MPS)            | fp32      | 25.6            | 56.4          | 39.1   | 56.6   | 66.4   |
| NVIDIA L4                  | PyTorch  | `rfdetr`           | predict() eager        | fp32      | 19.5            | 56.5          | 51.2   | 139.8  | 149.2  |
| NVIDIA L4                  | PyTorch  | `rfdetr`           | inference() CUDA Graph | fp16      | 3.5             | —             | 283.9  | 362.4  | 337.7  |
| NVIDIA L4                  | PyTorch  | `inference-models` | torch (CUDA)           | fp32      | 21.6            | 56.4          | 46.2   | 120.4  | 130.5  |
| NVIDIA L4                  | ONNX     | `rfdetr`           | CUDA EP                | fp32      | 8.4             | 56.5          | 119.2  | 132.9  | 120.5  |
| NVIDIA L4                  | ONNX     | `inference-models` | CUDA EP                | fp32      | 9.4             | 56.4          | 106.0  | 105.9  | 109.3  |
| NVIDIA L4                  | TensorRT | `rfdetr`           | .trt engine            | fp16      | 3.6             | 56.5          | 278.7  | 402.4  | 401.9  |
| NVIDIA L4                  | TensorRT | `inference-models` | managed engine         | auto      | 3.6             | 56.3          | 277.7  | 349.3  | 363.7  |
| NVIDIA L4                  | TensorRT | `inference-models` | managed engine         | fp32      | 7.0             | 56.4          | 142.9  | 152.0  | 154.3  |
| x86 CPU (Colab)            | PyTorch  | `rfdetr`           | predict() eager        | fp32      | 401.6           | 56.5          | 2.5    | 2.8    | —      |
| x86 CPU (Colab)            | PyTorch  | `rfdetr`           | inference() JIT        | fp32      | 377.9           | 56.5          | 2.6    | 2.8    | —      |
| x86 CPU (Colab)            | PyTorch  | `inference-models` | torch (CPU)            | fp32      | 421.8           | 56.4          | 2.4    | 2.7    | —      |
| x86 CPU (Colab)            | ONNX     | `rfdetr`           | CPU EP                 | fp32      | 546.9           | 56.5          | 1.8    | 2.5    | —      |
| x86 CPU (Colab)            | ONNX     | `inference-models` | CPU EP                 | fp32      | 393.8           | 56.4          | 2.5    | 2.4    | —      |
| Apple M4 Max CPU           | ONNX     | `rfdetr`           | CPU EP                 | fp32      | 98.7            | 56.5          | 10.1   | 10.7   | 8.9    |
| Apple M4 Max CPU           | ONNX     | `inference-models` | CPU EP                 | fp32      | 114.6           | 56.4          | 8.7    | 8.0    | 9.5    |
| Apple M4 Max (CoreML EP)   | ONNX     | `inference-models` | CoreML EP              | fp32      | 22.5            | 56.4          | 44.4   | 46.1   | 36.0   |

On the same runtime the two packages are close, and the accuracy is the same within 0.1 mAP: the ONNX CPU rows differ by 16%, and at batch 1 the `inference-models` PyTorch model (25.6 ms, fp32) is slower than eager `predict()` (20.0 ms) and 2.3 times slower than `inference()` with inductor in fp32 (11.1 ms). The ONNX CoreML EP row runs a different execution provider from the CPU rows, so it is not a like-for-like comparison with them; `rfdetr`'s ONNX loader does not select that provider. These Apple rows come from a busy machine (load average above 10), so treat gaps under about 15% as noise. On the L4 the packages are level on TensorRT at batch 1 (3.6 ms each) and `rfdetr` is ahead at batch 4 and 8; on the same ONNX Runtime provider `rfdetr` is 11% faster at batch 1 (8.4 against 9.4 ms), and the `inference-models` PyTorch model (21.6 ms) is slower than eager `predict()` (19.5 ms); see the [TensorRT page](tensorrt.md#why-the-managed-engine-is-faster-end-to-end). On the x86 CPU the two packages are level within the noise for PyTorch (421.8 against 401.6 ms eager, standard deviations of 66 to 73 ms); the one gap outside the noise is ONNX Runtime, where `inference-models` ran at 393.8 ms and `rfdetr`'s loader at 546.9 ms on the same ONNX Runtime 1.22.1. That gap was not investigated.

`Memory [MB]` in the cookbooks is comparable only within one cookbook: the CUDA cookbook measures device memory, the CPU, Apple and mobile cookbooks measure host resident memory (RSS).

Each cookbook also scores every exported model and precision at batch 1 on the same seeded 500-image COCO val2017 subset (box mAP@50:95 and mAP@50), so a precision setting that loses accuracy shows up next to its latency. The subset is chosen for run time, so these are not the full-val2017 numbers on the [Benchmarks page](../learn/benchmarks.md); set `FULL_VAL = True` in a cookbook to score all 5000 images.

!!! note "Which latency number is this?"

    RF-DETR latency appears on three different pages, and they are three different measurements:

    - **Architecture headline** (T4, TensorRT FP16, model only, batch 1) — the number that compares architectures. See the [Benchmarks page](../learn/benchmarks.md).
    - **Deployed product** (L4, Roboflow Inference, scope not stated, batch 1) — what a Roboflow Inference user gets. See [Roboflow's RF-DETR model page](https://docs.roboflow.com/models/supported-models/rf-detr).
    - **Export cookbooks** (this page and below) — end-to-end timings for every export format, on one machine per cookbook. Their value is **ratios within one machine**, not absolute milliseconds to set against the other two pages.

    Never compare milliseconds across these three without checking hardware, precision, batch size, and scope (model-only vs end-to-end) match. The cookbook's L4 end-to-end times at batch 1 (3.6 ms for the managed TensorRT engine, 3.6 ms for the `.trt` engine, 8.4 ms for ONNX Runtime) include host preprocessing and postprocessing, and are not comparable with the 8.3 ms TensorRT and 12.9 ms ONNX on Roboflow's model page, which does not state what its timing covers. The two harnesses have not been run side by side yet, so treat all of these as unreconciled.

!!! warning "fp16 pays off only where the silicon implements it"

    Reduced precision is not a portable speedup, and the cookbooks measure this directly:

    - **GPU — large win.** On an L4, TensorRT's auto-selected precision is 1.94× faster than the same engine forced to fp32 (3.6 vs 7.0 ms end-to-end, batch 1), and PyTorch `inference(dtype=torch.float16)` cuts the eager fp32 baseline from 19.5 to 9.9 ms.
    - **CPU — no measured win from fp16.** A CPU without native fp16 kernels upconverts and computes in fp32, so fp16 storage saves disk and bandwidth, not arithmetic. On the Colab x86 host the OpenVINO fp16 IR ran at 452.3 ms against 373.7 ms for the fp32 IR at batch 1, no faster, at half the size. INT8 is the lever that changes CPU arithmetic, and it did not help consistently: ONNX INT8 ran 1.3 times faster than ONNX fp32 (411.1 against 546.9 ms) for 1.8 mAP, while OpenVINO INT8 was slower than its fp32 IR (542.1 against 373.7 ms) at 3.6 times smaller, and an earlier run on another Colab host had it level with fp32. See the [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu) and [ONNX](onnx.md#on-an-x86-cpu) pages. The TFLite rows are pending a rerun: they ran through `tf.lite.Interpreter` at its default thread count (the same fp32 `.tflite` runs in 119 ms on one thread and 59 ms on eight through `ai_edge_litert` on the M4 Max, against 785 ms in the cookbook).
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
