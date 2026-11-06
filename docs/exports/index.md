---
description: Overview of exporting RF-DETR models to ONNX, TensorRT, TFLite, LiteRT, ExecuTorch, native CoreML, Apple Core AI and OpenVINO IR (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
---

# Export RF-DETR Model

!!! tip "Key Takeaways"

    - Export to ONNX for cross-platform inference with ONNX Runtime, OpenVINO, or TensorRT
    - Export to OpenVINO IR for optimized inference on CPU (x86, ARM), GPU (Intel integrated & discrete GPU) and AI accelerators (Intel NPU)
    - Export to TFLite (FP32, FP16, INT8) for mobile and edge deployment
    - Export to LiteRT (`.tflite`) straight from PyTorch with `litert-torch` — no ONNX or TensorFlow step
    - TensorRT conversion delivers the lowest latency on NVIDIA GPUs — 2.3 ms for Nano on a T4, TensorRT FP16, model only, batch 1 (the architecture headline on the [Benchmarks page](../learn/benchmarks.md); the end-to-end numbers below are a different measurement, see "Which latency number is this?")
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

Which format is fastest depends on the hardware you deploy to. Four per-hardware cookbooks export every format for one class of device, run inference through `RFDETRInference`, and benchmark it against the PyTorch model on the same machine. This page keeps only numbers measured through `RFDETRInference` (and the `model.inference()` snapshot it builds); each format page holds its own per-configuration table.

All numbers are batch 1 unless stated, `RFDETRSmall`, `rfdetr` `develop`, end to end (preprocessing, forward, postprocessing). Compare *within* a row group, never across machines.

| Hardware                             | Fastest route          | Precision | end2end [ms]   | FPS [img/s] | mAP@50:95 [%] | Full table                                                                           |
| ------------------------------------ | ---------------------- | --------- | -------------- | ----------- | ------------- | ------------------------------------------------------------------------------------ |
| NVIDIA L4                            | TensorRT `.trt` engine | fp16      | 3.57           | 280         | —             | [TensorRT](tensorrt.md#latency-across-nvidia-gpus)                                   |
| Apple M4 Max (40-core GPU)           | `inference()` inductor | fp16      | 10.3           | 97.4        | 56.4          | [Baseline](#pytorch-baseline), [CoreML](coreml.md#measured-latency-on-apple-silicon) |
| x86 CPU (Colab, 4 cores / 8 threads) | OpenVINO IR            | fp32      | 373.65 ± 61.10 | 2.7         | 56.5          | [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu)                               |

- Warmup and timed runs differ per cookbook (GPU 20 + 100, CPU 15 + 50, Apple 5 + 30). The rows come from a Colab L4 host, a Colab x86 host and one Apple M4 Max (macOS 26.7.1).
- On the x86 host the run-to-run standard deviation is 10 to 17% of the mean, so CPU batch-1 gaps under about 100 ms are noise: OpenVINO fp32 is the best full-accuracy route listed, but eager PyTorch and TorchScript are level with it.
- No mobile or edge CPU row is listed: the TFLite, LiteRT and ExecuTorch tables are pending a rerun with a pinned thread count.

### Fastest route across NVIDIA GPUs

Fastest measured route per GPU and size, `ms` at batch 1 and `img/s` at batch 8, with the winning route: C = `inference()` CUDA graph FP16, F = `.trt` FP16 engine. Latency only, TensorRT 10.16.1.11. All routes are in [TensorRT](tensorrt.md#latency-across-nvidia-gpus).

| GPU          | Nano                         | Small                        | Medium                       | Large                        |
| ------------ | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| T4           | 5.12 ms (F) / 334 img/s (F)  | 6.21 ms (F) / 197 img/s (F)  | 7.20 ms (F) / 149 img/s (F)  | 10.86 ms (F) / 98 img/s (F)  |
| L4           | 2.78 ms (C) / 724 img/s (F)  | 3.57 ms (F) / 432 img/s (F)  | 4.13 ms (F) / 349 img/s (F)  | 5.14 ms (F) / 227 img/s (F)  |
| A100         | 3.16 ms (C) / 967 img/s (F)  | 3.44 ms (F) / 783 img/s (F)  | 3.80 ms (C) / 676 img/s (F)  | 4.27 ms (C) / 558 img/s (F)  |
| RTX PRO 6000 | 1.47 ms (F) / 2271 img/s (F) | 1.62 ms (F) / 1782 img/s (F) | 1.76 ms (F) / 1608 img/s (F) | 2.08 ms (F) / 1249 img/s (F) |

### PyTorch Baseline

The PyTorch rows every cookbook measures, per machine. `predict()` is the eager model; `inference()` is the snapshot it returns (TorchScript, plus a CUDA graph on CUDA and inductor on the Mac GPU). `FPS` is images per second at each batch size, `—` where that batch was not run. `mAP@50:95` is a seeded 500-image `val2017` subset, scored at batch 1 on the artifact that was timed.

=== "NVIDIA L4"

    `RFDETRSmall`, TensorRT 10.16.1.11, one decoded RGB array per call, latency only (no accuracy scored).

    | Runtime                | Precision | b1 [ms] | b4 [ms] | b8 [ms] | FPS b1 | FPS b4 | FPS b8 |
    | ---------------------- | --------- | ------- | ------- | ------- | ------ | ------ | ------ |
    | inference() CUDA Graph | fp16      | 3.60    | 11.42   | 24.72   | 278    | 350    | 324    |
    | `.trt` engine          | fp16      | 3.57    | 9.36    | 18.53   | 280    | 427    | 432    |

    - `inference()` with fp16 and a CUDA graph is level with the `.trt` engine at batch 1, slower at batch 4 and 8, and keeps the native `predict()` API.

=== "Apple M4 Max"

    | Runtime                 | Precision | b1 [ms] | mAP@50:95 [%] | FPS b1 | FPS b4 | FPS b8 |
    | ----------------------- | --------- | ------- | ------------- | ------ | ------ | ------ |
    | predict() eager         | fp32      | 20.0    | 56.5          | 50.0   | 83.6   | 93.0   |
    | inference() TorchScript | fp16      | 12.2    | 56.5          | 81.9   | 131.3  | 143.5  |
    | inference() inductor    | fp32      | 11.1    | 56.5          | 90.4   | 101.2  | 120.8  |
    | inference() inductor    | fp16      | 10.3    | 56.4          | 97.4   | 127.0  | 141.6  |

    - These rows come from a busy machine (load average above 10): treat gaps under about 15% as noise.

=== "x86 CPU (Colab)"

    | Runtime         | Precision | b1 [ms]        | mAP@50:95 [%] | FPS b1 | FPS b4 |
    | --------------- | --------- | -------------- | ------------- | ------ | ------ |
    | predict() eager | fp32      | 401.58 ± 66.19 | 56.5          | 2.5    | 2.8    |
    | inference() JIT | fp32      | 377.85 ± 62.10 | 56.5          | 2.6    | 2.8    |

`Memory [MB]` in the cookbooks is comparable only within one cookbook: the CUDA cookbook measures device memory, the others host resident memory (RSS). The subset is chosen for run time, so these are not the full-`val2017` numbers on the [Benchmarks page](../learn/benchmarks.md); set `FULL_VAL = True` in a cookbook to score all 5000 images.

!!! note "Which latency number is this?"

    RF-DETR latency appears on three pages, and they are three different measurements:

    - **Architecture headline** (T4, TensorRT FP16, model only, batch 1): compares architectures. See the [Benchmarks page](../learn/benchmarks.md).
    - **Deployed product** (L4, Roboflow Inference, scope not stated, batch 1): see [Roboflow's RF-DETR model page](https://docs.roboflow.com/models/supported-models/rf-detr).
    - **Export cookbooks** (this page and below): end-to-end timings, including host preprocessing and postprocessing, on one machine per cookbook. Their value is **ratios within one machine**. The INT8 figures on the [TensorRT page](tensorrt.md#results) are engine-only GPU time, a fourth scope.

    Never compare milliseconds across these without checking hardware, precision, batch size and scope. The two harnesses have not been run side by side, so treat them as unreconciled.

!!! warning "fp16 pays off only where the silicon implements it"

    - **GPU: large win.** TensorRT fp16 engines and PyTorch `inference(dtype=torch.float16)` on CUDA are the fast routes; see the [TensorRT latency tables](tensorrt.md#latency-across-nvidia-gpus).
    - **CPU: no measured win.** A CPU without native fp16 kernels computes in fp32, so fp16 saves disk and bandwidth, not arithmetic (OpenVINO fp16 452.3 ms against 373.7 ms fp32 on the Colab host, at half the size). INT8 changes CPU arithmetic but not consistently: OpenVINO INT8 was slower than its fp32 IR, and the ONNX x86 INT8 comparison is pending a rerun. See [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu) and [ONNX](onnx.md#on-an-x86-cpu). The TFLite rows are pending a rerun (they ran at the interpreter's default thread count).
    - **Apple: runtime-dependent.** On the M4 Max, CoreML fp16 was *slower* than its fp32 default (23.6 against 12.0 ms) at half the size, and ExecuTorch's CoreML backend (fp16 only) matched CoreML fp16. An earlier macOS 27.0 run had Core AI fp16 faster than fp32 (13.53 against 14.62 ms); Core AI runs on the CPU only on older macOS. Measure per runtime. See [CoreML](coreml.md#measured-latency-on-apple-silicon) and [ExecuTorch](executorch.md#measured-latency-on-apple-silicon).

## Per-Format Guides

The format guides listed above cover installation, export examples, output files, and inference code.

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](../learn/deploy.md) for cloud-based inference and workflow integration
- Deploy TFLite and LiteRT `.tflite` models on mobile/edge devices with the LiteRT runtime
- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
- Read the [Exporter Blueprint](blueprint.md) to add a new export format
- Browse the format guides above for per-format details
