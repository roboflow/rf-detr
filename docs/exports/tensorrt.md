---
description: Export RF-DETR models to a TensorRT engine from PyTorch for low-latency inference on NVIDIA GPUs.
---

# TensorRT Export

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
>
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference.

## Prerequisites

- Install the TensorRT extra: `pip install rfdetr[tensorrt]` (provides `tensorrt`, `polygraphy`, `onnx`, and `onnxconverter-common`; the latter two cast the ONNX graph to FP16 on TensorRT 11+; no `trtexec` binary needed)
- A CUDA GPU (the engine is built for the local GPU architecture)
- Export an ONNX model first (for example: `output/inference_model.onnx`)

## Export Directly to TensorRT

Pass `format="tensorrt"` to `export()` to export ONNX and convert to a TensorRT engine in one step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt")
```

This exports `output/inference_model.onnx` first and then produces `output/inference_model_fp16.trt` (the `_fp16`/`_fp32` suffix always reflects the precision actually built — see `fp16` in [Export Parameters](index.md#export-parameters) — unless `output_name` is set).

!!! note "Dynamic batch"

    Pass `dynamic_batch=True` together with `max_batch_size` to build one engine that accepts any batch from 1 to `max_batch_size`. The engine gets a single TensorRT optimization profile with `min=1`, `opt=batch_size` and `max=max_batch_size`, so `batch_size` should be the batch you serve most often; other sizes inside the range run, TensorRT just tunes its kernels for `opt`. Without `dynamic_batch` the engine accepts only the batch size baked into the intermediate ONNX graph.

    ```python
    model.export(format="tensorrt", dynamic_batch=True, batch_size=4, max_batch_size=16)
    ```

    **Why a single profile with `min=1`, not several.** The engine always builds with one TensorRT optimization profile spanning the full `1 .. max_batch_size` range, rather than several narrower profiles picked at runtime with `set_optimization_profile_async`. This is a deliberate trade-off, not a limitation: it keeps the export API and the runtime simple (one engine, one profile, no profile-selection logic in the caller), and the measured cost at the tuned `opt` batch is small (see the [changelog](https://github.com/roboflow/rf-detr/blob/main/CHANGELOG.md) for per-GPU numbers). A deployment that never serves batches below some floor — for example a DeepStream or Triton pipeline always fed a fixed batch of frames (see [#376](https://github.com/roboflow/rf-detr/issues/376)) — pays for optimizing kernels down to batch 1 even though it never uses them, foreclosing per-batch-band multi-profile support (`set_optimization_profile_async` plus several `Profile()` entries), which is TensorRT's own standard mitigation for the away-from-opt penalty. A `min_batch_size` (paired with `max_batch_size`) or a list of `opt_batch_sizes` each with its own profile may become configurable in a future release if a narrow-band deployment need arises; today, export one profile spanning the batches you plan to serve.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. It is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases.

    If you plan to run inference with [`inference-models`](index.md#run-inference-with-inference-models) (the recommended path), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

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
