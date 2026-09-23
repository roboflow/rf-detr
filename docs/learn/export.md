---
description: Overview of exporting RF-DETR models to ONNX, TensorRT, TFLite, LiteRT, ExecuTorch, native CoreML and OpenVINO IR (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
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
    - Adding a format is an in-tree contribution — see [Exporter Blueprint](../exports/export-blueprint.md)
    - Per-format details live in the [Export Formats](../exports/index.md) subpages

RF-DETR supports exporting models to ONNX, TFLite, LiteRT, ExecuTorch, native CoreML and OpenVINO IR formats, enabling deployment across a wide range of inference frameworks, edge devices, and hardware accelerators.

This page covers the shared export API, parameters, output-file naming, and the `inference-models` deployment path. For detailed installation, examples, and inference code for each format, see the [Export Formats](../exports/index.md) guides:

- [ONNX Inference](../exports/onnx.md)
- [TensorRT](../exports/tensorrt.md)
- [TFLite](../exports/tflite.md)
- [LiteRT](../exports/litert.md)
- [OpenVINO](../exports/openvino.md)
- [ExecuTorch](../exports/executorch.md)
- [Native CoreML](../exports/coreml.md)

## Installation

Install the export dependencies you need:

```bash
# ONNX export only
pip install "rfdetr[onnx]"

# OpenVINO IR export
pip install "rfdetr[openvino]"

# TFLite export
pip install "rfdetr[tflite]"

# LiteRT export (.tflite straight from PyTorch via litert-torch)
pip install "rfdetr[litert]"

# ExecuTorch export (on-device inference: XNNPACK/CoreML/QNN)
pip install "rfdetr[executorch]"

# Native CoreML export (.mlpackage; macOS only)
pip install "rfdetr[coreml]"
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
| `format`             | `"onnx"`   | Export format: `"onnx"`, `"tflite"`, `"tensorrt"` (alias: `"trt"`), `"executorch"`, `"openvino"`, `"coreml"` or `"litert"`.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `quantization`       | `None`     | TFLite quantization mode: `None`/`"fp32"`, `"fp16"`, or `"int8"`. Only used when `format="tflite"`; `format="litert"` accepts only `None`/`"fp32"` and raises `NotImplementedError` otherwise.                                                                                                                                                                                                                                                                                                                                                                                   |
| `calibration_data`   | `None`     | Optional image directory, `.npy` file path, NumPy array, or `None`. Not consumed when building the generated `.tflite` models.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `max_images`         | `100`      | Maximum number of images to load from a `calibration_data` directory. Ignored for other calibration data formats.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `infer_dir`          | `None`     | Optional directory of sample images for inference validation during export tracing. If not provided, a random dummy image is generated.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `backbone_only`      | `False`    | Export only the backbone feature extractor instead of the full model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `opset_version`      | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `verbose`            | `True`     | Whether to print verbose export information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `shape`              | `None`     | Input shape as tuple `(height, width)`. Each dimension must be divisible by the selected model's block size (`patch_size * num_windows`). If not provided, uses the model's default resolution.                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch_size`         | `1`        | Batch size for the exported model. With `dynamic_batch=True` and `format="tensorrt"`, also the batch the engine's optimization profile is tuned for.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `dynamic_batch`      | `False`    | If `True`, export with a dynamic batch dimension so the model accepts variable batch sizes at runtime. Supported for `format="onnx"`, `format="tflite"` and `format="tensorrt"` (which then needs `max_batch_size`) — ExecuTorch, CoreML, OpenVINO and LiteRT bake a fixed batch size.                                                                                                                                                                                                                                                                                           |
| `patch_size`         | `None`     | Backbone patch size override. Defaults to the value from `model_config.patch_size`. Must match the instantiated model's patch size when provided.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `backend`            | `None`     | Backend for ExecuTorch: `"xnnpack"` (CPU, fp32), `"coreml"` (Apple, fp16), or `"qnn"` (Qualcomm HTP, fp16). Required when `format="executorch"`.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `soc`                | `None`     | Target SoC chip identifier for the `"qnn"` backend (e.g. `"SM8650"` for Snapdragon 8 Gen 3). Required when `backend="qnn"`.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `fp16`               | `True`     | Build the TensorRT engine with FP16 precision (only used when `format="tensorrt"`). TensorRT 11+ removed the FP16 builder flag, so there the engine is built from an FP16-cast graph instead; engine inputs and outputs stay FP32 either way. On strongly typed TensorRT (11+), this graph cast requires `onnx`/`onnxconverter-common` — install `rfdetr[tensorrt]` for the complete set, or export raises `ImportError`. A lean/partial TensorRT < 11 wheel lacking the FP16 builder flag falls back to an FP32 engine with a warning instead. Pass `False` for an FP32 engine. |
| `max_batch_size`     | `None`     | Largest batch a dynamic TensorRT engine accepts. Required when `format="tensorrt"` and `dynamic_batch=True`: the engine gets one optimization profile spanning batch `1 .. max_batch_size`, tuned for `batch_size`. Ignored for every other format.                                                                                                                                                                                                                                                                                                                              |
| `notes`              | `None`     | Optional user-defined metadata (string, dict, list, or any JSON-serialisable value) to embed in the exported ONNX model under the `"rfdetr_notes"` metadata property.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `coreml_precision`   | `None`     | Compute precision for `format="coreml"`: `None`/`"float32"` (tight CPU parity with eager PyTorch) or `"float16"` (half the size, and the only precision the Apple Neural Engine runs — at a measured accuracy cost, see [Native CoreML](../exports/coreml.md#neural-engine-compute-units-and-the-fallback-boundary)). Ignored for every other format.                                                                                                                                                                                                                            |
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
| `executorch` | `{variant}_xnnpack.pte` / `{variant}_coreml.pte` / `{variant}_qnn_{soc}.pte` (or `{variant}_xnnpack-backbone.pte` if `backbone_only=True`); without a variant or `output_name`, `backbone_model_xnnpack.pte` if `backbone_only=True` | `backend` (+ `soc` for `qnn`), plus `-backbone` when named                   |
| `tensorrt`   | `{variant}_fp16.trt` / `{variant}_fp32.trt` (or `{variant}-backbone_fp16.trt` / `{variant}-backbone_fp32.trt` if `backbone_only=True`)                                                                                               | `fp16`, plus `-backbone` when named                                          |
| `tflite`     | `{variant}_fp32.tflite` + `{variant}_fp16.tflite` (+ `{variant}_dynamic_range_quant.tflite` for `quantization="int8"`)                                                                                                               | precision / quantization mode                                                |
| `openvino`   | `{variant}.xml` + `{variant}.bin` (or `{variant}-backbone.xml`/`.bin` if `backbone_only=True`); without a variant or `output_name`, `inference_model.xml`/`.bin` (or `backbone_model.xml`/`.bin` if `backbone_only=True`)            | none — `openvino_precision` controls IR weight compression, not the filename |

Pass `output_name="my-model"` to override the variant name and write `{output_name}.{ext}` verbatim — this suppresses the detail suffix for every format **except** `tflite`, which always writes multiple files and so keeps its `_fp32`/`_fp16`/`_dynamic_range_quant` suffix even with a custom name (`{output_name}_fp32.tflite`, etc.).

With `backbone_only=True`, ONNX, CoreML, ExecuTorch, TensorRT, and OpenVINO retain a `-backbone` marker before the extension even when `output_name` is set, for example `my-model-backbone.onnx`. This distinguishes the backbone artifact from the full detector exported with the same name.

## Per-Format Guides

The format-specific installation steps, export examples, output files, and inference code are on dedicated pages:

- [ONNX Inference](../exports/onnx.md)
- [TensorRT](../exports/tensorrt.md)
- [TFLite](../exports/tflite.md)
- [LiteRT](../exports/litert.md)
- [OpenVINO](../exports/openvino.md)
- [ExecuTorch](../exports/executorch.md)
- [Native CoreML](../exports/coreml.md)

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

If you want to add a format, or you are reading the export code, see [Exporter Blueprint](../exports/export-blueprint.md) for the contract each format implements and the steps a new one takes.

## Using the Exported Model

Once exported, you can use the ONNX model with various inference frameworks. See [ONNX Inference](../exports/onnx.md) for a complete example, or the format-specific pages for other runtimes.

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](deploy.md) for cloud-based inference and workflow integration

- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) for multi-backend inference (PyTorch, ONNX, TensorRT) with automatic backend selection

- Deploy TFLite and LiteRT `.tflite` models on mobile/edge devices with the LiteRT runtime

- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime

- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO

- Read the [Exporter Blueprint](../exports/export-blueprint.md) to add a new export format

- Browse the [Export Formats](../exports/index.md) guides for per-format details
