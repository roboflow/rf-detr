---
description: Export RF-DETR models to a TensorRT engine from PyTorch for low-latency inference on NVIDIA GPUs.
---

# TensorRT Export

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
>
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference.

## Prerequisites

- Install the TensorRT extra: `pip install rfdetr[tensorrt]` (provides `tensorrt`, `polygraphy`, `onnx`, `onnxconverter-common` and `onnxruntime-gpu`; `onnx` and `onnxconverter-common` cast the ONNX graph to FP16 on TensorRT 11+, and onnxruntime calibrates [INT8](#int8) engines; no `trtexec` binary needed)
- A CUDA GPU (the engine is built for the local GPU architecture)
- Export an ONNX model first (for example: `output/inference_model.onnx`)

## Export Directly to TensorRT

Pass `format="tensorrt"` to `export()` to export ONNX and convert to a TensorRT engine in one step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt")
```

This exports `output/inference_model.onnx` first and then produces `output/inference_model_fp16.trt` (the `_fp16`/`_fp32` suffix always reflects the precision actually built — see `fp16` in [Export Parameters](advanced.md#export-parameters) — unless `output_name` is set).

!!! note "Dynamic batch"

    Pass `dynamic_batch=True` together with `max_batch_size` to build one engine that accepts any batch from 1 to `max_batch_size`. The engine gets a single TensorRT optimization profile with `min=1`, `opt=batch_size` and `max=max_batch_size`, so `batch_size` should be the batch you serve most often; other sizes inside the range run, TensorRT just tunes its kernels for `opt`. Without `dynamic_batch` the engine accepts only the batch size baked into the intermediate ONNX graph.

    ```python
    model.export(format="tensorrt", dynamic_batch=True, batch_size=4, max_batch_size=16)
    ```

    **Why a single profile with `min=1`, not several.** The engine always builds with one TensorRT optimization profile spanning the full `1 .. max_batch_size` range, rather than several narrower profiles picked at runtime with `set_optimization_profile_async`. This is a deliberate trade-off, not a limitation: it keeps the export API and the runtime simple (one engine, one profile, no profile-selection logic in the caller), and the measured cost at the tuned `opt` batch is small (see the [changelog](https://github.com/roboflow/rf-detr/blob/main/CHANGELOG.md) for per-GPU numbers). A deployment that never serves batches below some floor — for example a DeepStream or Triton pipeline always fed a fixed batch of frames (see [#376](https://github.com/roboflow/rf-detr/issues/376)) — pays for optimizing kernels down to batch 1 even though it never uses them, foreclosing per-batch-band multi-profile support (`set_optimization_profile_async` plus several `Profile()` entries), which is TensorRT's own standard mitigation for the away-from-opt penalty. A `min_batch_size` (paired with `max_batch_size`) or a list of `opt_batch_sizes` each with its own profile may become configurable in a future release if a narrow-band deployment need arises; today, export one profile spanning the batches you plan to serve.

## INT8

Pass `quantization="int8"` and a directory of representative images to build an engine that runs most of the backbone encoder and decoder in INT8 and the rest in FP16:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt", quantization="int8", calibration_data="path/to/images", batch_size=8)
```

This writes `output/rfdetr-small_int8.trt` (next to the intermediate `output/rfdetr-small.onnx`). The export runs the first `max_images` (default `100`) calibration images by file name through the FP32 graph with onnxruntime on the CPU, preprocessed exactly as `predict()` does, and records each quantized tensor's largest absolute value. Those ranges decide the engine's accuracy, so use images from the deployment domain: out-of-domain images give an engine that loads, runs and is quietly less accurate. `calibration_data` also accepts a `.npy` path or an array shaped `(N, C, H, W)`, already normalized the way `predict()` normalizes.

Measured on RF-DETR Nano with an RTX 5070 (sm_120) and TensorRT 11.3, calibrated on 128 COCO `train2017` images (`max_images=128`) and scored on all 5000 `val2017` images (AP at batch 8; the other batch sizes build from the same graph). Latency is GPU time per image (CUDA events, median of 15 rounds of 200 executions, FP16 and INT8 interleaved) from a captured CUDA graph and from a plain `execute_async_v3` call:

| Batch | Precision | AP    | AP50  | Latency, CUDA graph | Latency, plain call | Engine |
| ----- | --------- | ----- | ----- | ------------------- | ------------------- | ------ |
| 1     | FP16      | —     | —     | 0.912 ms            | 1.650 ms            | 62 MB  |
| 1     | INT8      | —     | —     | 0.799 ms (1.14×)    | 1.635 ms (1.01×)    | 49 MB  |
| 8     | FP16      | 48.03 | 67.11 | 0.496 ms            | 0.551 ms            | 73 MB  |
| 8     | INT8      | 47.34 | 66.18 | 0.411 ms (1.21×)    | 0.476 ms (1.16×)    | 110 MB |
| 32    | FP16      | —     | —     | 0.479 ms            | 0.488 ms            | 70 MB  |
| 32    | INT8      | —     | —     | 0.389 ms (1.23×)    | 0.401 ms (1.22×)    | 315 MB |

The INT8 placement follows what was measured, not "quantize every matrix multiply":

- **INT8:** the patch embedding, every backbone MLP, the windowed backbone blocks' attention with its projections, and the decoder's cross-attention, feed-forward and reference-point layers. Q/DQ pairs carry FP16 scales on the FP16 graph, so every layer left unquantized stays FP16.
- **FP16:** the projector, the two-stage proposal head and the detection heads (quantizing them cost accuracy and bought nothing), plus the attention blocks below.
- **Attention is all INT8 or all FP16.** The windowed backbone blocks fit TensorRT's INT8 fused-attention kernel (on Nano: head size 64, 145 tokens), so their projections and both attention matrix multiplies are INT8. The global blocks (580 tokens on Nano, above the kernel's 512) and the decoder self-attention keep projections and attention in FP16.

Requirements and limits:

- Detection models only. Segmentation and keypoint models raise `NotImplementedError`; export them with `quantization=None`.
- A static batch (`dynamic_batch=True` is refused), `fp16=True`, and TensorRT 10 or newer (verified on 10.16 and 11.3).
- It is a latency feature, not a size one: TensorRT stores the INT8 engine larger than the FP16 one at batch 8 and 32 (table above).
- At batch 1 the engine is launch-bound: INT8 is faster only when the engine is replayed from a CUDA graph. `TRTInference` makes a plain `execute_async_v3` call, so the plain-call column is what it gets.
- Measured on Nano and one GPU. Other sizes and GPUs build with the same rules but are not measured; the attention placement relies on TensorRT's INT8 fused-attention kernel, so measure on your GPU before deploying.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. It is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases.

    If you plan to run inference with [`inference-models`](basics.md#run-inference-with-inference-models) (the recommended path), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

## Python API Conversion

Use this only to convert an **already-exported** `.onnx` file without re-running the model export. To go straight from a checkpoint to an engine, use [`format="tensorrt"`](#export-directly-to-tensorrt) above.

!!! warning "Internal API"

    `rfdetr.export._tensorrt.exporter` is a private module — the leading underscore means it carries no stability guarantee and may move or change signature in any release. `RFDETR.export(format="tensorrt")` is the supported entry point; use the class below only when you need to convert an already-exported `.onnx` file.

```python
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter

exporter = TensorRTExporter(TensorRTConfig(fp16=True))
engine_path = exporter.build_engine("output/inference_model.onnx")
# -> "output/inference_model_fp16.trt"
```

`TensorRTExporter.build_engine` builds the engine in-process via the TensorRT Python API (no `trtexec` subprocess) and returns the path to the generated `.trt` engine file. Precision and progress logging come from the `TensorRTConfig` the exporter is constructed with — pass `TensorRTConfig(output_name="my-engine")` to write `output/my-engine.trt` verbatim instead.

An `.onnx` file exported with `dynamic_batch=True` needs `dynamic_batch=True` here as well, plus `max_batch_size`, the largest batch the engine accepts. `opt_batch_size` (default `1`) is the batch its kernels are tuned for. Without `dynamic_batch=True`, `build_engine` raises `ValueError` rather than building an engine that accepts batch 1 only.

```python
exporter = TensorRTExporter(TensorRTConfig(fp16=True, dynamic_batch=True, max_batch_size=16, opt_batch_size=4))
engine_path = exporter.build_engine("output/inference_model.onnx")
```
