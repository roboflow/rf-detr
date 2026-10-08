---
description: Export RF-DETR models to a TensorRT engine from PyTorch for low-latency inference on NVIDIA GPUs.
---

# TensorRT Export

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
>
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference, or build a [portable engine](#portable-engines).

## Prerequisites

- Install the TensorRT extra: `pip install rfdetr[tensorrt]` (provides `tensorrt`, `polygraphy`, `onnx`, `onnxconverter-common` and `onnxruntime-gpu`; `onnx` and `onnxconverter-common` cast the ONNX graph to FP16 on TensorRT 11+, and onnxruntime calibrates [INT8](#int8) engines; no `trtexec` binary needed)
- A CUDA GPU (by default the engine is built for the local GPU architecture)
- Export an ONNX model first (for example: `output/inference_model.onnx`)

## Export Directly to TensorRT

Pass `format="tensorrt"` to `export()` to export ONNX and convert to a TensorRT engine in one step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt")
```

This exports `output/inference_model.onnx` first and then produces `output/inference_model_fp16.trt` (the `_fp16`/`_fp32`/`_int8` suffix always reflects the precision actually built — see `fp16` in [Export Parameters](advanced.md#export-parameters) — unless `output_name` is set).

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

This writes `output/rfdetr-small_int8.trt` (next to the intermediate `output/rfdetr-small.onnx`). The export runs the first `max_images` (default `100`) calibration images by file name through the FP32 graph with onnxruntime on the CPU, preprocessed exactly as `predict()` does, and records each quantized tensor's largest absolute value. Those ranges decide the engine's accuracy, so use images from the deployment domain: out-of-domain images give an engine that loads, runs and is quietly less accurate. `calibration_data` also accepts a `.npy` path or an array shaped `(N, C, H, W)`, already normalized the way `predict()` normalizes; an integer array, raw pixels, is refused. `max_images` caps only a directory of images; an array or `.npy` file is used whole, so pass as many samples as you want calibrated. Calibration runs the graph at the export's `batch_size`, so its host memory grows with the batch: about 9.5 GB for Nano at batch 32.

Measured with TensorRT 11.3 on an RTX 5070 (sm_120) and a Tesla T4 (sm_75), calibrated on 128 COCO `train2017` images (`max_images=128`) and scored on all 5000 `val2017` images. Latency is GPU time per image (CUDA events, median of 15 rounds of 200 executions, FP16 and INT8 interleaved) from a captured CUDA graph and from back-to-back `execute_async_v3` calls with no synchronization in between. Each speedup in parentheses divides the FP16 latency by the INT8 latency measured the same way, so the two columns have different FP16 bases: a plain call adds CPU launch time that a graph replay removes. That gap is large only where the engine is launch-bound, at batch 1 on the RTX 5070 (1.675 ms plain against 0.922 ms graph for Nano), and small on the T4 and at larger batches. On the RTX 5070, Nano across batch sizes:

| Batch | Precision | AP    | AP50  | Latency, CUDA graph | Latency, plain call | Engine    |
| ----- | --------- | ----- | ----- | ------------------- | ------------------- | --------- |
| 1     | FP16      | 48.03 | 67.10 | 0.922 ms            | 1.675 ms            | 62 MB     |
| 1     | INT8      | 47.40 | 66.23 | 0.693 ms (1.33×)    | 1.582 ms (1.06×)    | 41 MB     |
| 8     | FP16      | 48.02 | 67.11 | 0.497 ms            | 0.554 ms            | 73 MB     |
| 8     | INT8      | 47.29 | 66.19 | 0.363 ms (1.37×)    | 0.426 ms (1.30×)    | 51 MB     |
| 32    | FP16      | 48.03 | 67.12 | 0.482 ms            | 0.494 ms            | 68–108 MB |
| 32    | INT8      | 47.30 | 66.18 | 0.338 ms (1.43×)    | 0.350 ms (1.41×)    | 49–87 MB  |

Every model at batch 1, measured the same way:

| Model  | AP, FP16 | AP, INT8      | CUDA graph, FP16 | CUDA graph, INT8 | Plain call, FP16 | Plain call, INT8 | Engine, FP16 | Engine, INT8 |
| ------ | -------- | ------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------ | ------------ |
| Nano   | 48.03    | 47.40 (−0.63) | 0.922 ms         | 0.693 ms (1.33×) | 1.675 ms         | 1.582 ms (1.06×) | 62 MB        | 41 MB        |
| Small  | 52.79    | 52.18 (−0.61) | 1.432 ms         | 1.157 ms (1.24×) | 2.221 ms         | 1.998 ms (1.11×) | 67 MB        | 45 MB        |
| Medium | 54.69    | 53.84 (−0.85) | 1.625 ms         | 1.370 ms (1.19×) | 2.536 ms         | 2.412 ms (1.05×) | 71 MB        | 48 MB        |
| Large  | 56.52    | 56.02 (−0.50) | 2.000 ms         | 1.820 ms (1.10×) | 2.806 ms         | 2.682 ms (1.05×) | 68 MB        | 51 MB        |

The same on a Tesla T4, from a Colab notebook run (INT8 loses 0.55 to 0.85 AP, 1.0% to 1.6% of the FP16 AP, and is faster everywhere):

| Model, batch | AP, FP16 | AP, INT8      | CUDA graph, FP16 | CUDA graph, INT8 | Plain call, FP16 | Plain call, INT8 | Engine, FP16 | Engine, INT8 |
| ------------ | -------- | ------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------ | ------------ |
| Nano, 1      | 48.07    | 47.31 (−0.77) | 3.21 ms          | 2.65 ms (1.21×)  | 3.26 ms          | 2.68 ms (1.22×)  | 60 MB        | 48 MB        |
| Small, 1     | 52.83    | 52.23 (−0.60) | 5.31 ms          | 4.60 ms (1.16×)  | 5.25 ms          | 4.68 ms (1.12×)  | 64 MB        | 59 MB        |
| Medium, 1    | 54.66    | 53.80 (−0.85) | 6.71 ms          | 5.76 ms (1.16×)  | 6.69 ms          | 5.89 ms (1.14×)  | 69 MB        | 67 MB        |
| Large, 1     | 56.51    | 55.96 (−0.55) | 10.18 ms         | 9.14 ms (1.11×)  | 10.13 ms         | 9.28 ms (1.09×)  | 70 MB        | 83 MB        |
| Nano, 8      | 47.98    | 47.23 (−0.75) | 2.72 ms          | 2.13 ms (1.28×)  | 2.72 ms          | 2.16 ms (1.26×)  | 70 MB        | 120 MB       |
| Nano, 32     | 48.02    | 47.28 (−0.75) | 2.97 ms          | 2.31 ms (1.28×)  | 2.98 ms          | 2.33 ms (1.28×)  | 113 MB       | 366 MB       |

The larger models gain less: their global attention, which stays FP16, is a bigger share of the work, and Large keeps its windowed attention in FP16 as well (see below).

!!! warning "Engine size on the T4"

    On the T4 the INT8 engine is larger than the FP16 one for Large at batch 1 (83 against 70 MB) and for Nano at batch 4, 8 and 32 (79, 120 and 366 MB against 65, 70 and 113 MB), where the RTX 5070 shows the opposite. The cause has not been diagnosed. What TensorRT's engine inspector shows on the T4: the layers' constants are the same at every batch (29 MB for INT8, 51 MB for FP16 on Nano), so it is not the weights; the bytes outside them grow by 10.3 MB per image of batch for INT8 against 1.5 MB for FP16, and by about 31 MB more than FP16 on Large at batch 1, so the cost follows the image size; and it does not change with INT8 attention switched off or with `builder_optimization_level=1`. Speed and accuracy are unaffected, but check the file size if disk or load time matters.

The INT8 weights take 32 MB against FP16's 54 MB on Nano. On the RTX 5070 both engines also store a block of zeros: TensorRT pads the 3-channel image of the FP16 patch embedding to 4 or 8 channels, picked per build, and keeps the added channels as a constant of 0.3 or 1.5 MB per image of batch. Two builds of the same graph can therefore differ by up to 1.2 MB per image, which is why the batch-32 sizes are ranges.

The INT8 placement follows what was measured, not "quantize every matrix multiply":

- **INT8:** every backbone MLP, the windowed backbone blocks' attention with its projections (up to 325 tokens; see below), and the decoder's cross-attention, feed-forward and reference-point layers. Q/DQ pairs carry FP16 scales on the FP16 graph, so every layer left unquantized stays FP16.
- **FP16:** the projector, the two-stage proposal head and the detection heads (quantizing them cost accuracy and bought nothing), plus the attention blocks below.
- **The patch embedding stays FP16.** TensorRT ran it in INT8 on the image padded to 32 channels and stored the 29 added zero channels in the engine, a constant the size of the batch. On Nano that made the engine 110 MB at batch 8 and 315 MB at batch 32, and INT8 1.21× instead of 1.37× faster than FP16 at batch 8. Every convolution whose input channels are not a multiple of 32 stays FP16.
- **Attention is all INT8 or all FP16.** A block with head size 16, 32 or 64 and at most 325 tokens gets INT8 projections and INT8 attention matrix multiplies, which TensorRT fuses into one INT8 attention kernel: the windowed backbone blocks of Nano (145 tokens), Small (257) and Medium (325). 325 is the largest window measured to pay (Medium: 5% faster from a CUDA graph, about equal with a plain call, −0.28 AP); on Large's 485-token windows INT8 attention ran no faster and cost 1.0 AP, so Large keeps them FP16. Windows of 326 to 484 tokens are not measured and stay FP16. The global blocks (580 tokens on Nano) and the decoder self-attention keep projections and attention in FP16 too. A windowed block has `(resolution / patch_size / num_windows)² + 1` tokens, so a higher export resolution can move a model's attention to FP16; the export logs how many backbone attention blocks stay FP16.

Requirements and limits:

- Detection models only. Segmentation and keypoint models raise `NotImplementedError`; export them with `quantization=None`.
- A static batch (`dynamic_batch=True` is refused), `fp16=True`, and TensorRT 10 or newer. Only TensorRT 10.16 and 11.3 were measured; the 10.x releases before 10.16 are accepted but untested.
- On the RTX 5070, at batch 1 the engine is launch-bound: the gain needs a CUDA-graph replay, and without one INT8 gains only 1.05–1.11× (the plain-call column above). The T4 gains about as much without a graph (1.09–1.22×).
- Measured on two GPUs (sm_120 and sm_75), without `trt_hardware_compatibility`, `trt_version_compatible` or a shared `trt_timing_cache`: those settings are accepted with `quantization="int8"`, but INT8 engines built with them were not measured, and a portable engine may not get the fused INT8 attention kernel the gain relies on. The attention placement relies on TensorRT's INT8 fused-attention kernel, which NVIDIA lists for sm_75 to sm_90, sm_120 and sm_121; on other GPUs (for example sm_100, B200) INT8 attention would run unfused, so measure on your GPU before deploying.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. By default it is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases; see [Portable Engines](#portable-engines) for the options that widen that.

    If you plan to run inference with [`inference-models`](basics.md#run-inference-with-inference-models) (the recommended path), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

## Engine Description File

A `.trt` file has no slot for user metadata, so a consumer that does not import the model, such as a C++ service, [Triton](https://developer.nvidia.com/triton-inference-server) or [DeepStream](https://developer.nvidia.com/deepstream-sdk), cannot learn what the engine expects from the file itself. Pass `trt_metadata=True` to write `<engine>.json` next to it, with the same name and a `.json` suffix:

```python
import hashlib
import json
from pathlib import Path

engine = model.export(format="tensorrt", trt_metadata=True)
# engine -> output/rfdetr-medium_fp16.trt, described by output/rfdetr-medium_fp16.json

with open(Path(engine).with_suffix(".json")) as description_file:
    description = json.load(description_file)
engine_bytes = Path(engine).read_bytes()  # deserialize these bytes, not a second read of the file
if hashlib.sha256(engine_bytes).hexdigest() != description["engine"]["sha256"]:
    raise RuntimeError(f"{engine} is not the engine this description was written for")
height, width = description["input"]["height"], description["input"]["width"]
```

| Key              | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version` | Version of this layout, `1`. It changes when a key is removed, renamed or changes meaning. Adding a key does not change it, so a reader ignores the keys it does not know.                                                                                                                                                                                                                                                                                                                                                                                              |
| `rfdetr_version` | The `rfdetr` version that exported the engine, or `null` when run from a source tree that is not installed.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `variant`        | The model variant name, such as `rfdetr-nano`, or `null` when there is none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `backbone_only`  | Whether the engine is a backbone-only export, whose outputs are feature maps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `engine`         | The engine file this description file was written for: its `size` in bytes and `sha256`, the SHA-256 hex digest of its bytes. Compare them with the `.trt` you load, as in the example above.                                                                                                                                                                                                                                                                                                                                                                           |
| `input`          | The input tensor: `name`, `layout` (`NCHW`), `dtype`, `height`, `width`, `channels`, `channel_order` (`"RGB"` for three channels, otherwise `null`), `normalization` and `resize`. Pixel values are divided by `normalization.scale`, then `(x - mean) / std` is applied per channel; `resize` is bilinear with half-pixel centers and no antialiasing, and its `aspect_ratio` is `"stretch"`: the whole image is resized to `height` x `width`, with no letterbox and no padding, so the boxes are normalized to the image and scale by its original width and height. |
| `outputs`        | The output tensors in the order the engine returns them, each with `name` and `dtype`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch`          | `{"dynamic": false, "size": N}` for a static engine, or `{"dynamic": true, "min": 1, "opt": N, "max": M}` for one built with `dynamic_batch=True`.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `build`          | How the engine was built: `precision` (the one actually built, which can differ from `fp16` on a lean TensorRT wheel, and is `int8` for a `quantization="int8"` engine; on TensorRT before 11 it records that the FP16 builder flag was set, so FP16 is allowed but not forced, while on TensorRT 11 and later it records an fp16 cast graph, so read `tensorrt_version` to tell which applies), `opset`, `tensorrt_version` and `gpu` (`name` and `compute_capability`, or `null` when PyTorch has no CUDA).                                                           |
| `notes`          | The `notes` passed to `export()`: a string as it is, any other value JSON-encoded into a string (decode it once more), or `null`.                                                                                                                                                                                                                                                                                                                                                                                                                                       |

All inputs and outputs are `float32`, also for an `fp16` engine. The normalization values are the ones `RFDETR.predict()` applies. The `resize` block records the convention of the export runtime and of `RFDETR.predict()` by default; a checkpoint trained with an antialiased resize may expect `antialias=True`, which is the `antialias` argument of `RFDETR.predict()`, and the file does not record that.

!!! note "What the file does not say"

    It does not record class names or which logit slot is the background class. How a slot maps to a class depends on the checkpoint (official COCO weights use sparse COCO category ids, fine-tuned models use contiguous ids), and `RFDETR.predict()` decides it internally. For `dets` and `labels` outputs, `dets` holds normalized `cx, cy, w, h` boxes and `labels` holds raw class logits, which `RFDETR.predict()` turns into scores with a per-class sigmoid (not a softmax) before it keeps the highest-scoring (query, class) pairs above the confidence threshold; see [`decode_detections`](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/export/_runtime/decode.py) for how the repository turns them into detections.

The file is written after the engine and replaces an older description of the same name in one step, so a reader never sees half a file; if a `.json` of that name that an RF-DETR export did not write is already there, `export()` raises `FileExistsError` before building the engine, and you rename or remove that file or pick another `output_name`. On Linux and macOS it gets the engine's permission bits and, where you are allowed to set it, the engine's group, so a service that can read the engine can read the description too. Only `RFDETR.export` (or calling the exporter on a prepared graph) writes it; `TensorRTExporter.build_engine` on an `.onnx` file has no graph to describe and does not, and with `TensorRTConfig(metadata=True)` it warns that the setting has no effect there. If the file can't be written, `export()` raises `OSError` after the engine is built, and the message says so.

The engine and its description file are still two files, replaced one after the other. A program that reads them while an export is running, or after two exports of the same name ran at the same time, can find an engine next to the description file of another build. That is what `engine` is for: compare its `sha256` with the bytes you load, as the example above does, before you trust the rest.

The engine's file name doesn't include its batch profile or input size, so a later export of the same model overwrites the engine and can leave an old `.json` next to it. When a build doesn't write a description file (`export()` without `trt_metadata=True`, or `build_engine`) and one with the engine's name is already there, you get a warning that it may no longer match. The warning repeats on every rebuild of that engine name for as long as the old description file stays in place. The file is never deleted, because this build didn't write it; a program that checks its `engine.sha256` finds out whether it still matches.

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

## Reuse the Timing Cache

A large share of an engine build goes into timing candidate kernels for each layer. TensorRT can save those timings in a cache file, and a later build that uses the file skips the timing it already did. Pass `trt_timing_cache` to keep one across exports:

```python
model.export(format="tensorrt", trt_timing_cache="output/rfdetr.cache")
```

The first export creates the file (and its folder). Later exports load it and write the merged timings back. It pays off when you rebuild the same model at the same precision and batch profile, on the same GPU and TensorRT version, for example after fine-tuning: the timings depend on the layers' shapes, not on the weights. In our tests, a cache from the pretrained model saved about as much time for a model fine-tuned to 3 classes, whose class head has a different shape. A different resolution changes every layer's shape, and we haven't measured it. A cache from a matching dynamic-batch build (batch 1-8) helps too: it cut the build time by 74-76% in fp16 and 42-43% in fp32. We did not measure reusing a cache between a static and a dynamic batch profile, or across precisions.

A cache TensorRT can't use doesn't stop the export. For a file written by another TensorRT major version, or an empty or corrupt one, TensorRT logs a serialization error, builds as if there were no cache, and the file is replaced with the new timings. A timing cache is specific to the GPU, the CUDA version and the TensorRT version, so keep one cache file per GPU and per software stack. TensorRT's API reference says a cache whose recorded CUDA device properties differ from the current environment is reported as a failure; we had one GPU and could not test that. We tested TensorRT 10.16 and 11.3 with Polygraphy 0.53.4, while the supported range is `tensorrt>=8.6.1` and the Polygraphy version is not pinned, so behavior on other versions is untested.

A cache that can't be written does stop the export, though: Polygraphy writes the timing cache before it returns the engine, so if that write fails (a full disk, say) the export raises and the built engine is not saved. Fix the cause and export again; the engine can simply be rebuilt.

A bad path is refused before the engine is built: a directory, or a path ending in a separator, raises `ValueError`, and a location that can't be created or written raises `OSError`. `~` is expanded, and a relative path is relative to the working directory, not to `output_dir`.

Polygraphy keeps an empty `<file>.lock` next to the cache and locks it while it reads or writes the cache, so the folder has to be writable. We did not test two exports sharing one cache file at the same time. Without `trt_timing_cache`, no file is read or written.

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

`TensorRTExporter.build_engine` builds the engine in-process via the TensorRT Python API (no `trtexec` subprocess) and returns the path to the generated `.trt` engine file. Precision and progress logging come from the `TensorRTConfig` the exporter is constructed with — pass `TensorRTConfig(output_name="my-engine")` to write `output/my-engine.trt` verbatim instead. `TensorRTConfig(timing_cache="output/rfdetr.cache")` reuses a timing cache here too; see [Reuse the Timing Cache](#reuse-the-timing-cache).

An `.onnx` file exported with `dynamic_batch=True` needs `dynamic_batch=True` here as well, plus `max_batch_size`, the largest batch the engine accepts. `opt_batch_size` (default `1`) is the batch its kernels are tuned for. Without `dynamic_batch=True`, `build_engine` raises `ValueError` rather than building an engine that accepts batch 1 only.

```python
exporter = TensorRTExporter(TensorRTConfig(fp16=True, dynamic_batch=True, max_batch_size=16, opt_batch_size=4))
engine_path = exporter.build_engine("output/inference_model.onnx")
```

The portability options work here too, as `TensorRTConfig(hardware_compatibility="ampere_plus")` and `TensorRTConfig(version_compatible=True)`; see [Portable Engines](#portable-engines).
