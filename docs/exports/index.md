---
description: Overview of exporting RF-DETR models to ONNX, TensorRT, TFLite, LiteRT, ExecuTorch, native CoreML, Apple Core AI and OpenVINO IR (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
---

# Export RF-DETR Model

!!! tip "Key Takeaways"

    - Export to ONNX for cross-platform inference with ONNX Runtime, OpenVINO, or TensorRT
    - Export to OpenVINO IR for optimized inference on CPU (x86, ARM), GPU (Intel integrated & discrete GPU) and AI accelerators (Intel NPU)
    - Export to TFLite (FP32, FP16, INT8) for mobile and edge deployment
    - Export to LiteRT (`.tflite`) straight from PyTorch with `litert-torch` — no ONNX or TensorFlow step
    - TensorRT conversion delivers lowest latency on NVIDIA GPUs (2.3 ms for Nano)
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

## Load an Export for Prediction

`RFDETR.from_export()` loads a full export into the native `predict()` API. It returns an `RFDETRInference` instance. Detection and segmentation return `supervision.Detections`. Keypoint models return `supervision.KeyPoints`.

```python
from rfdetr import RFDETR, RFDETRInference, RFDETRSmall

native_model = RFDETRSmall()
artifact = native_model.export(format="onnx", output_dir="output")

model: RFDETRInference = RFDETR.from_export(artifact, device="auto")
detections = model.predict("image.jpg", threshold=0.5)
print(model.runtime_info)
```

The top-level `from_export(artifact, device="auto")` function provides the same factory. Pass a list to `predict()` to get one result per image. The full list runs as one batch. The artifact must accept its batch size and input shape. If either does not match, `predict()` raises before execution. It does not split or pad the list.

`runtime_info` reports the selected runtime and device policy. Some vendor runtimes choose a physical device after model load. In that case, the policy does not identify one chip.

Exports include versioned inference metadata. ONNX stores it in the model. Other formats use an adjacent `<artifact-name>.rfdetr.json` file. For example, `model.trt` uses `model.trt.rfdetr.json`. Keep this file with the artifact. For an older export, pass a JSON path or a mapping through `metadata=`. The loader rejects missing or conflicting task, class, output, and preprocessing details. User-defined `notes` are separate from inference metadata.

The loader accepts the TensorRT `.trt` extension and the `.engine` alias. Both TFLite export routes use `.tflite`. Their metadata selects the NHWC or NCHW input layout. LiteRT currently cannot export a full keypoint model. Backbone-only exports contain features rather than predictions, so `from_export()` rejects them. `RFDETRInference` exposes `predict()`, `class_names`, and `runtime_info`. It has no training, evaluation, export, or native optimization methods.

Use `device="auto"` to let the runtime select an available policy. An explicit device request must match the runtime and available hardware. For example, TensorRT requires CUDA. Core AI accepts `auto` or `cpu`; its GPU and Neural Engine options express a preference and cannot pin execution to that device. An incompatible request raises an error instead of switching devices. Install the runtime package for the chosen format on the target system. The format guides describe package and hardware limits.

The default `backend="rfdetr"` runs preprocessing and postprocessing on the CPU and copies runtime outputs into owned tensors. The benchmarks below measure the existing per-runtime cookbook pipelines, not this wrapper.

### Complete inference-models pipeline

For detection, segmentation, or keypoint exports in ONNX or TensorRT format, select `backend="inference_models"` to run its complete prediction pipeline:

```python
from rfdetr import RFDETR

model = RFDETR.from_export(
    "output/rfdetr-small.onnx",
    backend="inference_models",
    device="cuda:0",
)
detections = model.predict("image.jpg", threshold=0.5)
print(model.runtime_info)
```

This backend requires `inference-models` 0.39.x and Python 3.10–3.13. Install the runtime extra for your hardware in the same environment:

```bash
# ONNX on CPU
uv pip install "rfdetr[inference-models]" "inference-models[onnx-cpu]>=0.39,<0.40"

# ONNX on CUDA
uv pip install "rfdetr[inference-models]" "inference-models[onnx-cu12]>=0.39,<0.40"
```

For TensorRT, follow the [SDK's TensorRT installation instructions](https://inference-models.roboflow.com/getting-started/installation/) and build the engine with a compatible TensorRT version on the target GPU. The dependency loads only when selected. The `rfdetr[inference-models]` extra includes ONNX for metadata inspection; it does not select CPU or GPU runtime packages.

This backend passes original RGB images to `inference_models.infer()`. That library performs resizing, normalization, execution, and postprocessing. RF-DETR adapts inputs and converts the SDK results to the public result type:

| Task                  | Result                   | Task-specific data                                                                 |
| --------------------- | ------------------------ | ---------------------------------------------------------------------------------- |
| Detection             | `supervision.Detections` | Boxes and detection scores                                                         |
| Instance segmentation | `supervision.Detections` | Boxes, detection scores, and masks                                                 |
| Keypoint detection    | `supervision.KeyPoints`  | Keypoint and detection scores, visibility, covariance, and boxes in `data["xyxy"]` |

The adapter preserves original class IDs, class names, single-image versus list returns, and source-image metadata. It validates the artifact's batch and input-shape limits before inference. Backbone exports and the SDK's separate two-stage keypoint architecture are outside this single-artifact API.

The two backends can produce different coordinates and scores. `inference_models` uses its own resize rules and rounds box coordinates to pixels. Changing the backend does not guarantee identical numbers. Its selected optimizations can also depend on the installed version, hardware, and input type. Inspect `runtime_info` after prediction and measure the complete `predict()` call on your target hardware.

Keypoint selection also differs: the SDK considers every query/class pair, so result counts can differ from the default backend. The SDK supplies pixel-space covariance but does not expose raw precision-Cholesky values. Results therefore omit `data["keypoint_precision_cholesky"]`.

The adapter requires float32 input and output tensors. The exported query count must match `num_select`. ONNX weights must be stored inside the model file. TensorRT outputs must use the standard `dets`, `labels`, and task-specific `masks` or `keypoints` names. Segmentation exports must use `upsample_masks_to_image_size=True`; the SDK returns masks at the original image size. Keypoint exports must use `trace_alpha=0.20`, which matches the SDK's score fusion. Unsupported tasks, formats, and export settings raise an error.

Set `include_source_image=False` when you do not need the original image in result metadata. Source capture can copy GPU tensors to the CPU. Conversion to Supervision results still copies prediction results to the CPU.

## Measured Performance by Hardware

Which format is fastest depends entirely on the hardware you deploy to. The four per-hardware cookbooks each export every format targeting one class of device, run inference on it, and benchmark it against a PyTorch baseline on the same machine. Below is the fastest end-to-end result per hardware class, plus the PyTorch anchor it was measured against; the cookbooks carry the full tables, including forward-only timings, memory, and the slower configurations.

| Hardware                 | Fastest format            | end2end [ms]   | FPS [img/s] | PyTorch `predict()` anchor | Cookbook                              |
| ------------------------ | ------------------------- | -------------- | ----------- | -------------------------- | ------------------------------------- |
| NVIDIA L4                | TensorRT (auto precision) | 4.91 ± 0.14    | 203.5       | 19.88 ms / 50.3 FPS        | [CUDA](../cookbooks/export-cuda/)     |
| Apple M-series (ANE/GPU) | Core AI fp16              | 11.45 ± 0.18   | 87.3        | 22.62 ms / 44.2 FPS        | [Apple](../cookbooks/export-apple/)   |
| x86 CPU (4 cores)        | OpenVINO fp32 IR          | 311.92 ± 46.04 | 3.2         | 345.76 ms / 2.9 FPS        | [CPU](../cookbooks/export-cpu/)       |
| ARM CPU (edge proxy)     | ExecuTorch XNNPACK        | 87.33 ± 1.13   | 11.5        | —                          | [Mobile](../cookbooks/export-mobile/) |

All numbers are batch 1, `RFDETRSmall`, rfdetr v1.11.0. Warmup and timed-run counts differ per cookbook (GPU uses 20 + 100, CPU 15 + 50, Apple and mobile 5 + 30), and each row was measured on different hardware, so compare *within* a row's hardware class, never across rows. The x86 CPU figures come from a shared Colab vCPU where run-to-run noise is 12–17% of the mean — on that machine no CPU format separates from the others by more than one standard deviation.

!!! warning "fp16 pays off only where the silicon implements it"

    Reduced precision is not a portable speedup, and the cookbooks measure this directly:

    - **GPU — large win.** On an L4, TensorRT's auto-selected precision is 1.62× faster than the same engine forced to fp32 (4.91 vs 7.96 ms), and PyTorch `inference(dtype=torch.float16)` nearly halves the eager fp32 baseline (11.30 vs 19.88 ms).
    - **CPU — no win.** OpenVINO's default FP16 IR came out *slower* than explicit `float32` (354.55 vs 311.92 ms) despite halving the file, and TFLite fp16 matched fp32 exactly (780.86 vs 780.74 ms). A CPU without native fp16 kernels upconverts and computes in fp32, so fp16 saves disk and bandwidth, not arithmetic. On CPU the lever is **INT8**: dynamic-range quantization was ~2.5× faster than fp32 TFLite (319.61 vs 780.74 ms), at the cost of one dropped detection on the sample image.
    - **Apple — runtime-dependent.** Core AI fp16 beat its fp32 default (11.45 vs 12.54 ms), while CoreML fp16 came out *slower* than its fp32 default (20.05 vs 11.62 ms) on the same chip. Both results reproduced across runs. Measure per runtime, not per platform.

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
| `calibration_data`   | `None`     | Optional image directory, `.npy` file path, NumPy array, or `None`. Not consumed when building the generated `.tflite` models.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
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

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is a separate deployment option. It supports PyTorch, ONNX, and TensorRT with automatic backend selection and its own API.

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
