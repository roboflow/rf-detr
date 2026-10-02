---
description: Export RF-DETR models to a TensorRT engine from PyTorch for low-latency inference on NVIDIA GPUs.
---

# TensorRT Export

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
>
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference, or build a [portable engine](#portable-engines).

## Prerequisites

- Install the TensorRT extra: `pip install rfdetr[tensorrt]` (provides `tensorrt`, `polygraphy`, `onnx`, and `onnxconverter-common`; the latter two cast the ONNX graph to FP16 on TensorRT 11+; no `trtexec` binary needed)
- A CUDA GPU (by default the engine is built for the local GPU architecture)
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

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. By default it is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases; see [Portable Engines](#portable-engines) for the options that widen that.

    If you plan to run inference with [`inference-models`](index.md#run-inference-with-inference-models) (the recommended path), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

## Portable Engines

By default an engine only runs on the kind of GPU and the TensorRT version that built it. Two options ask TensorRT for more portable engines. Both are off by default, and the default build doesn't change. An engine built with either option gets its own file name, like `<variant>_fp16_ampere_plus.trt`, so it doesn't overwrite a default engine in the same folder (unless you set `output_name`, which is used as is).

```python
# Ask for an engine that NVIDIA Ampere GPUs (compute capability 8.x) and newer can run.
model.export(format="tensorrt", trt_hardware_compatibility="ampere_plus")

# Ask for an engine that other releases of the same TensorRT major version may load.
model.export(format="tensorrt", trt_version_compatible=True)
```

`trt_hardware_compatibility` is `"ampere_plus"` or `"same_compute_capability"`. The first targets Ampere and every newer GPU, and has to be built on one of them; the second targets GPUs with the same compute capability as the one you build on. TensorRT records the level in the engine. We only had one GPU, so we haven't tested running such an engine on a second one. NVIDIA doesn't support hardware compatibility on Jetson (JetPack) or DriveOS ([engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)). If your TensorRT doesn't have the level, you get a `ValueError` before the export runs the model.

`trt_version_compatible=True` asks for an engine that other releases of the same TensorRT major version may load. It worked between TensorRT 11.2 and 11.3, in both directions. It didn't work across major versions (10 and 11), or between TensorRT 10.13 and 10.16, so test your pair of releases before you rely on it. The build needs TensorRT's lean runtime, which is a separate package: install the lean wheel that matches your `tensorrt` wheel, for example [`tensorrt-lean-cu13-libs`](https://pypi.org/project/tensorrt-lean-cu13-libs/) next to [`tensorrt-cu13-libs`](https://pypi.org/project/tensorrt-cu13-libs/) (the version may differ by a `.post` suffix), or use the library from the TensorRT archive or system package. Without it the export raises `ImportError` naming the library, before it runs the model.

The options have a cost, so measure on your own model. In our FP16 tests, `ampere_plus` made the build about 3.5 times slower, the engine about 60% larger and inference about 10% slower. `same_compute_capability` cost nothing we could measure. `trt_version_compatible` added about 105 MB to an `RFDETRNano` engine on TensorRT 11 (on TensorRT 10.16 the engine was the size of a default one). The [changelog](https://github.com/roboflow/rf-detr/blob/main/CHANGELOG.md) has the numbers.

An engine built by TensorRT 11 with `trt_version_compatible=True` contains host code, and TensorRT only loads it if you say you trust the file. `TRTInference` and `rfdetr.export.benchmark` take `engine_host_code_allowed` for that; it's off by default:

```python
from rfdetr.export._tensorrt.inference import TRTInference

engine = model.export(format="tensorrt", trt_version_compatible=True)
runtime = TRTInference(str(engine), sync_mode=True, engine_host_code_allowed=True)
```

Only turn it on for a file you built yourself or otherwise trust. `TRTInference` lives in a private module, so like the classes under [Python API Conversion](#python-api-conversion) it can change without notice; `sync_mode=True` avoids the [`pycuda`](https://pypi.org/project/pycuda/) dependency of its default mode.

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

The portability options work here too, as `TensorRTConfig(hardware_compatibility="ampere_plus")` and `TensorRTConfig(version_compatible=True)`; see [Portable Engines](#portable-engines).
