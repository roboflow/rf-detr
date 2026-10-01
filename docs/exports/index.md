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

This page covers the shared export API, parameters, output-file naming, and the `inference-models` deployment path. For detailed installation, examples, and inference code, see the format guides:

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

| Hardware                           | Fastest format                                                   | end2end [ms] | FPS [img/s] | PyTorch `predict()` anchor | Cookbook                              |
| ---------------------------------- | ---------------------------------------------------------------- | ------------ | ----------- | -------------------------- | ------------------------------------- |
| NVIDIA L4 (pkg `inference-models`) | TensorRT via `inference-models` (managed engine, auto precision) | 3.41 ± 0.06  | 293.5       | 19.88 ms / 50.3 FPS        | [CUDA](../cookbooks/export-cuda/)     |
| NVIDIA L4 (plain `rfdetr`)         | TensorRT raw `.trt` engine (`TRTInference`, fp16, no extra pkg)  | 9.82 ± 0.34  | 101.8       | 19.88 ms / 50.3 FPS        | [CUDA](../cookbooks/export-cuda/)     |
| Apple M4 Max (40-core GPU)         | CoreML fp32 (default)                                            | 11.57 ± 0.28 | 86.5        | 24.12 ms / 41.5 FPS        | [Apple](../cookbooks/export-apple/)   |
| x86 CPU (shared Colab vCPU)        | No clear winner: ONNX, OpenVINO and PyTorch overlap within noise | 357–377      | 2.7–2.8     | 377.34 ms / 2.7 FPS        | [CPU](../cookbooks/export-cpu/)       |
| Apple M4 Max CPU (12P + 4E cores)  | ExecuTorch XNNPACK, fp32                                         | 80.74 ± 1.90 | 12.4        | 54.54 ms / 18.3 FPS        | [Mobile](../cookbooks/export-mobile/) |

All numbers are batch 1, `RFDETRSmall`, rfdetr v1.11.0. Warmup and timed-run counts differ per cookbook (GPU uses 20 + 100, CPU 15 + 50, Apple and mobile 5 + 30). The rows come from three machines: a Colab L4 GPU host, a shared Colab x86 vCPU, and one Apple M4 Max (16-core CPU: 12 performance + 4 efficiency, 40-core GPU) that ran both the Apple row and the arm64 CPU row. Compare *within* a row, never across rows. On the shared x86 vCPU the gap between formats is about one run-to-run standard deviation, and a shared vCPU drifts between sessions, so that row is not a ranking. Its OpenVINO figures were also measured before `OpenVINOInference` computed in float32 (see the [OpenVINO guide](openvino.md#openvino-inference-example)) and are pending a rerun. The arm64 CPU row stands in for a phone or edge CPU only approximately: it is a laptop chip, and a phone's own delegate (NNAPI, QNN, Core ML) is not exercised. **Two L4 rows, on purpose**: the first goes through `inference-models`, a separate package that selects and loads its own prebuilt TensorRT engine; the second times the `.trt` file `model.export(format="trt")` itself writes, loaded directly with `rfdetr`'s own `TRTInference` reference runtime — no extra package beyond `rfdetr[tensorrt]`. Both are real end-to-end numbers on the same L4 — see the [CUDA cookbook](../cookbooks/export-cuda/#6-tensorrt) for the full table and the open gap against Roboflow's own product-page number.

`Memory [MB]` in the cookbooks is comparable only within one cookbook: the CUDA cookbook measures device memory, the CPU, Apple and mobile cookbooks measure host resident memory (RSS).

!!! note "Which latency number is this?"

    RF-DETR latency appears on three different pages, and they are three different measurements:

    - **Architecture headline** (T4, TensorRT FP16, model only, batch 1) — the number that compares architectures. See the [Benchmarks page](../learn/benchmarks.md).
    - **Deployed product** (L4, Roboflow Inference, model only, batch 1) — what a Roboflow Inference user gets. See [Roboflow's RF-DETR model page](https://docs.roboflow.com/models/supported-models/rf-detr).
    - **Export cookbooks** (this page and below) — end-to-end and forward-only timings for every export format, on one machine per cookbook. Their value is **ratios within one machine**, not absolute milliseconds to set against the other two pages.

    Never compare milliseconds across these three without checking hardware, precision, batch size, and scope (model-only vs end-to-end) match. Even matched for scope (model only, batch 1, L4), the CUDA cookbook's forward times are lower than Roboflow's model page for both runtimes: ONNX 7.35 vs 12.9 ms and TensorRT 2.16 vs 8.3 ms. The two harnesses have not been run side by side yet, so treat both as unreconciled — see the [CUDA cookbook's Results section](../cookbooks/export-cuda/#7-results).

!!! warning "fp16 pays off only where the silicon implements it"

    Reduced precision is not a portable speedup, and the cookbooks measure this directly:

    - **GPU — large win.** On an L4, TensorRT's auto-selected precision is 1.94× faster than the same engine forced to fp32 (3.41 vs 6.62 ms end-to-end, batch 1), and PyTorch `inference(dtype=torch.float16)` cuts the eager fp32 baseline from 19.88 to 11.37 ms.
    - **CPU — no measured win.** A CPU without native fp16 kernels upconverts and computes in fp32, so fp16 storage (an fp16 OpenVINO IR or `.tflite`) saves disk and bandwidth, not arithmetic. The cookbooks' CPU fp16 and INT8 timings are pending a rerun: the OpenVINO rows ran at OpenVINO's own default execution precision, which is half precision on CPUs with native bf16 or f16 support (the Colab host's CPU flags were not recorded), and the TFLite rows ran through `tf.lite.Interpreter` at its default thread count (the same fp32 `.tflite` runs in 119 ms on one thread and 59 ms on eight through `ai_edge_litert` on the M4 Max, against 785 ms in the cookbook).
    - **Apple — runtime-dependent.** On the M4 Max, Core AI fp16 beat its fp32 default (13.53 vs 14.62 ms), while CoreML fp16 came out *slower* than its fp32 default (19.86 vs 11.57 ms). The direction of both results reproduced across runs. Measure per runtime, not per platform.

## Installation

Install the export dependencies you need:

=== "ONNX"

    ```bash
    pip install "rfdetr[onnx]"
    ```

=== "OpenVINO"

    ```bash
    pip install "rfdetr[openvino]"
    ```

=== "TFLite"

    ```bash
    pip install "rfdetr[tflite]"
    ```

=== "LiteRT"

    ```bash
    pip install "rfdetr[litert]"
    ```

=== "ExecuTorch"

    ```bash
    pip install "rfdetr[executorch]"
    ```

=== "Native CoreML (macOS)"

    ```bash
    pip install "rfdetr[coreml]"
    ```

=== "Apple Core AI"

    ```bash
    pip install "rfdetr[coreai]"
    ```

## Basic Export

Export your trained model to ONNX format:

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

This command saves the ONNX model to the `output` directory by default.

## Export Parameters

The `export()` method accepts several parameters to customize the export process:

| Parameter            | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_dir`         | `"output"` | Directory where the exported model will be saved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `format`             | `"onnx"`   | Export format: `"onnx"`, `"tflite"`, `"tensorrt"` (alias: `"trt"`), `"executorch"`, `"openvino"`, `"coreml"`, `"coreai"` or `"litert"`, in any case.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `quantization`       | `None`     | TFLite quantization mode: `None`/`"fp32"`, `"fp16"`, or `"int8"`. Only used when `format="tflite"`; `format="litert"` accepts only `None`/`"fp32"` and raises `NotImplementedError` otherwise.                                                                                                                                                                                                                                                                                                                                                                                   |
| `calibration_data`   | `None`     | Optional image directory, `.npy` file path, NumPy array, or `None`. Ignored: passing it warns, since `.tflite` INT8 is dynamic-range.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `max_images`         | `100`      | Maximum number of images to load from a `calibration_data` directory. Ignored for other calibration data formats.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `infer_dir`          | `None`     | Optional directory of sample images for inference validation during export tracing. If not provided, a random dummy image is generated.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `backbone_only`      | `False`    | Export only the backbone feature extractor instead of the full model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `opset_version`      | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `verbose`            | `True`     | Whether to print verbose export information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `shape`              | `None`     | Input shape as tuple `(height, width)`. Each dimension must be divisible by the selected model's block size (`patch_size * num_windows`). If not provided, uses the model's default resolution.                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch_size`         | `1`        | Batch size for the exported model: a positive integer (numpy integers work, `bool` does not); anything else raises `ValueError`. With `dynamic_batch=True` and `format="tensorrt"`, also the batch the engine's optimization profile is tuned for.                                                                                                                                                                                                                                                                                                                               |
| `dynamic_batch`      | `False`    | If `True`, export with a dynamic batch dimension so the model accepts variable batch sizes at runtime. Supported for `format="onnx"`, `format="tflite"` and `format="tensorrt"` (which then needs `max_batch_size`) — ExecuTorch, CoreML, Core AI, OpenVINO and LiteRT bake a fixed batch size.                                                                                                                                                                                                                                                                                  |
| `patch_size`         | `None`     | Backbone patch size override. Defaults to the value from `model_config.patch_size`. Must match the instantiated model's patch size when provided.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `backend`            | `None`     | Backend for ExecuTorch: `"xnnpack"` (CPU, fp32), `"coreml"` (Apple, fp16), or `"qnn"` (Qualcomm HTP, fp16). Required when `format="executorch"`.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `soc`                | `None`     | Target SoC chip identifier for the `"qnn"` backend (e.g. `"SM8650"` for Snapdragon 8 Gen 3). Required when `backend="qnn"`.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `fp16`               | `True`     | Build the TensorRT engine with FP16 precision (only used when `format="tensorrt"`). TensorRT 11+ removed the FP16 builder flag, so there the engine is built from an FP16-cast graph instead; engine inputs and outputs stay FP32 either way. On strongly typed TensorRT (11+), this graph cast requires `onnx`/`onnxconverter-common` — install `rfdetr[tensorrt]` for the complete set, or export raises `ImportError`. A lean/partial TensorRT < 11 wheel lacking the FP16 builder flag falls back to an FP32 engine with a warning instead. Pass `False` for an FP32 engine. |
| `max_batch_size`     | `None`     | Largest batch a dynamic TensorRT engine accepts. Required when `format="tensorrt"` and `dynamic_batch=True`: the engine gets one optimization profile spanning batch `1 .. max_batch_size`, tuned for `batch_size`. Ignored for every other format.                                                                                                                                                                                                                                                                                                                              |
| `notes`              | `None`     | Optional user-defined metadata (string, dict, list, or any JSON-serialisable value) to embed in the exported ONNX model under the `"rfdetr_notes"` metadata property. For a format that embeds it, a value JSON cannot encode raises before the model runs: `ValueError` for `NaN`/`Infinity` or a circular reference, `TypeError` for an arbitrary object.                                                                                                                                                                                                                      |
| `coreml_precision`   | `None`     | Compute precision for `format="coreml"`: `None`/`"float32"` (tight CPU parity with eager PyTorch) or `"float16"` (half the size, and the only precision the Apple Neural Engine runs — at a measured accuracy cost, see [Native CoreML](coreml.md#neural-engine-compute-units-and-the-fallback-boundary)). Ignored for every other format.                                                                                                                                                                                                                                       |
| `coreai_precision`   | `None`     | Compute precision for `format="coreai"`: `None`/`"float32"` (matches eager PyTorch on the GPU) or `"float16"` (half the size, and the precision Core AI runs on the Apple Neural Engine — at a measured accuracy cost, see [Core AI](coreai.md#precision-compute-units-and-latency)); a float16 keypoint model warns, since that asset aborts on the Neural Engine. Ignored for every other format.                                                                                                                                                                              |
| `openvino_precision` | `None`     | IR *storage* weight precision for `format="openvino"`: `None`/`"float16"` (OpenVINO's default FP16 weight compression) or `"float32"` (disables compression). Execution precision still depends on the compiled device — not guaranteed to match eager PyTorch on non-CPU devices. Ignored for every other format. Does not change the output filename.                                                                                                                                                                                                                          |
| `output_name`        | `None`     | Full filename override (without extension). Takes precedence over the model's variant name and suppresses the `_fp32`/`_fp16`/`_{backend}` detail suffix — see [Output Files](#output-files).                                                                                                                                                                                                                                                                                                                                                                                    |

## Advanced Export Examples

### Export with Custom Output Directory

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(output_dir="exports/my_model")
```

### Export with Custom Resolution

Export the model with a specific input resolution. For example, `RFDETRMedium` expects dimensions divisible by `32` (`patch_size=16`, `num_windows=2`):

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(shape=(608, 608))
```

### Export Backbone Only

Export only the backbone feature extractor for use in custom pipelines:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(backbone_only=True)
```

The backbone export contains the encoder and its feature projector, without the detection decoder or prediction heads. ONNX outputs are feature maps in NCHW layout, ordered by `projector_scale`: `features` for the first level, followed by `features_1`, `features_2`, and so on when more levels are configured. Backbones with a second projector also return its levels as `cross_attn_features`, `cross_attn_features_1`, and so on, after the primary levels. These outputs are feature maps, not decoded boxes, masks, or keypoint coordinates.

## Output Files

Filenames are built from the model's variant name (e.g. `rfdetr-medium`, falling back to `inference_model` when no variant or `output_name` is set, or `backbone_model` when `backbone_only=True` in that same case) plus a detail suffix whenever a detail materially changes the artifact — even at its default value, since the file needs to say what it actually is:

| Format       | Filename pattern                                                                                                                                                                                                                     | Detail encoded                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `onnx`       | `{variant}.onnx` (or `{variant}-backbone.onnx` if `backbone_only=True`); without a variant or `output_name`, `inference_model.onnx` (or `backbone_model.onnx` if `backbone_only=True`)                                               | none — `-backbone` is structural, not a precision detail                     |
| `coreml`     | `{variant}_fp32.mlpackage` / `{variant}_fp16.mlpackage` (or `{variant}_fp32-backbone.mlpackage` if `backbone_only=True`); without a variant or `output_name`, `backbone_model_fp32.mlpackage` if `backbone_only=True`                | `coreml_precision`, plus `-backbone` when named                              |
| `coreai`     | `{variant}_fp32.aimodel` / `{variant}_fp16.aimodel`; without a variant or `output_name`, `inference_model_fp32.aimodel`                                                                                                              | `coreai_precision`                                                           |
| `executorch` | `{variant}_xnnpack.pte` / `{variant}_coreml.pte` / `{variant}_qnn_{soc}.pte` (or `{variant}_xnnpack-backbone.pte` if `backbone_only=True`); without a variant or `output_name`, `backbone_model_xnnpack.pte` if `backbone_only=True` | `backend` (+ `soc` for `qnn`), plus `-backbone` when named                   |
| `tensorrt`   | `{variant}_fp16.trt` / `{variant}_fp32.trt` (or `{variant}-backbone_fp16.trt` / `{variant}-backbone_fp32.trt` if `backbone_only=True`)                                                                                               | `fp16`, plus `-backbone` when named                                          |
| `tflite`     | `{variant}_fp32.tflite` + `{variant}_fp16.tflite` (+ `{variant}_dynamic_range_quant.tflite` for `quantization="int8"`)                                                                                                               | precision / quantization mode                                                |
| `openvino`   | `{variant}.xml` + `{variant}.bin` (or `{variant}-backbone.xml`/`.bin` if `backbone_only=True`); without a variant or `output_name`, `inference_model.xml`/`.bin` (or `backbone_model.xml`/`.bin` if `backbone_only=True`)            | none — `openvino_precision` controls IR weight compression, not the filename |

Pass `output_name="my-model"` to override the variant name and write `{output_name}.{ext}` verbatim — this suppresses the detail suffix for every format **except** `tflite`, which always writes multiple files and so keeps its `_fp32`/`_fp16`/`_dynamic_range_quant` suffix even with a custom name (`{output_name}_fp32.tflite`, etc.).

With `backbone_only=True`, ONNX, CoreML, ExecuTorch, TensorRT, and OpenVINO retain a `-backbone` marker before the extension even when `output_name` is set, for example `my-model-backbone.onnx`. This distinguishes the backbone artifact from the full detector exported with the same name.

## Per-Format Guides

The format guides listed above cover installation, export examples, output files, and inference code.

## Run Inference with `inference-models`

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is the recommended library for running RF-DETR inference. It supports multiple backends — PyTorch, ONNX, and TensorRT — with automatic backend selection and a unified API.

### Installation

```bash
# CPU / PyTorch only
pip install inference-models

# With TensorRT support (NVIDIA GPU required)
pip install "inference-models[trt10]"  # TensorRT 10
```

See the [inference-models installation guide](https://inference-models.roboflow.com/getting-started/installation/) for all installation options including Jetson and CUDA 11.x.

### Load a Pre-trained RF-DETR Model

```python
import cv2
from inference_models import AutoModel

# Automatically selects the best available backend for your environment
model = AutoModel.from_pretrained("rfdetr-small")

image = cv2.imread("image.jpg")
predictions = model(image)

# Convert to supervision Detections
detections = predictions[0].to_supervision()
print(detections)
```

### Load a Local RF-DETR Checkpoint

```python
import cv2
from inference_models import AutoModel

# Load from a local .pth checkpoint (same file used by rfdetr for training)
model = AutoModel.from_pretrained(
    "/path/to/checkpoint.pth",
    model_type="rfdetr-small",  # specify the architecture variant
)

image = cv2.imread("image.jpg")
predictions = model(image)
```

### Force TensorRT Backend

```python
import cv2
from inference_models import AutoModel, BackendType

# Explicitly request TensorRT — requires TRT to be installed
model = AutoModel.from_pretrained("rfdetr-small", backend=BackendType.TRT)

image = cv2.imread("image.jpg")
predictions = model(image)
```

`AutoModel.from_pretrained` accepts `backend="onnx"`, `backend="torch"`, or `backend="trt"` to override automatic backend selection.

## How Export Works

Every format is written by an `Exporter` class built from that format's own configuration, and `model.export()` is a facade over them: it resolves the format to an exporter, narrows this method's union-of-every-format signature down to the settings that format actually reads, prepares one format-independent `ExportGraph`, and hands the graph to the exporter. The signature and return value on this page are the supported surface; the classes behind it are internal.

If you want to add a format, or you are reading the export code, see [Exporter Blueprint](blueprint.md) for the contract each format implements and the steps a new one takes.

## Using the Exported Model

Once exported, you can use the ONNX model with various inference frameworks. See [ONNX Inference](onnx.md) for a complete example, or the format-specific pages for other runtimes.

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](../learn/deploy.md) for cloud-based inference and workflow integration
- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) for multi-backend inference (PyTorch, ONNX, TensorRT) with automatic backend selection
- Deploy TFLite and LiteRT `.tflite` models on mobile/edge devices with the LiteRT runtime
- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
- Read the [Exporter Blueprint](blueprint.md) to add a new export format
- Browse the format guides above for per-format details
