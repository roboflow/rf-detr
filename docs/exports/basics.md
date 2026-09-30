---
description: Export basics for RF-DETR — install the export extras, run model.export(), find the output files, and run inference with the exported model.
---

# Export Basics

This page covers the everyday export path: install an extra, call `model.export()`, locate the output files, and run inference. For the full parameter reference and less common options, see [Advanced Export](advanced.md). For a comparison of formats and measured performance, see the [Overview](index.md).

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
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

This command saves the ONNX model to the `output` directory by default.

## Choose a Format

Pass `format=` to `model.export()` to pick another target; the extra from [Installation](#installation) must be installed first. Which format is fastest depends on the hardware — measured numbers are on the [Overview](index.md#measured-performance-by-hardware).

| Deploying to                | `format=`                      | Guide                                            |
| --------------------------- | ------------------------------ | ------------------------------------------------ |
| Anywhere ONNX Runtime runs  | `"onnx"` (default)             | [ONNX Inference](onnx.md)                        |
| NVIDIA GPU                  | `"tensorrt"` (alias `"trt"`)   | [TensorRT](tensorrt.md)                          |
| Intel CPU, GPU or NPU       | `"openvino"`                   | [OpenVINO](openvino.md)                          |
| Android, embedded, edge CPU | `"tflite"` or `"litert"`       | [TFLite](tflite.md), [LiteRT](litert.md)         |
| On-device PyTorch runtime   | `"executorch"` (alias `"pte"`) | [ExecuTorch](executorch.md)                      |
| Apple platforms (Xcode)     | `"coreml"` or `"coreai"`       | [Native CoreML](coreml.md), [Core AI](coreai.md) |

Format names are case-insensitive. Formats marked experimental in [Advanced Export](advanced.md#format-capabilities) emit a warning when constructed.

## Check the Export

An exported model returns raw tensors — box and logit decoding (sigmoid, background slot, box format) is left to your inference code. The [ONNX guide](onnx.md) spells out the decoding rules and two pitfalls that apply to every format. To confirm an export matches PyTorch, run both on the same image and compare detections; the [export cookbooks](../cookbooks.md) do this per hardware class, with latency and COCO mAP next to each other.

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

## Predict with RFDETRInference

Use `RFDETRInference` to predict with a live native model, a native checkpoint, or any full export supported below. It uses the same image inputs and Supervision results as `RFDETR.predict()`.

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/inference_model.onnx")
detections = model.predict("image.jpg", threshold=0.5)
```

Detection and segmentation return `supervision.Detections`. Keypoint models return `supervision.KeyPoints`. A path ending in `.pth` or `.pt` loads a native checkpoint. To use an existing model instance, pass it directly:

```python
from rfdetr import RFDETRInference, RFDETRSmall

native_model = RFDETRSmall()
model = RFDETRInference(native_model)
```

This wraps the live model without copying its weights. Each prediction reads the model's current state, including changes from training or inference optimization. When you pass a path, the `RFDETRInference` instance loads and owns the native model or exported runtime.

The public constructor supports ONNX, TensorRT (`.trt` and `.engine`), TFLite, LiteRT, ExecuTorch, OpenVINO, native CoreML, Core AI, and native `.pth`/`.pt` checkpoints. Install the runtime package for the format on the target system. LiteRT currently cannot export a full keypoint model. Backbone-only exports contain features rather than predictions and are not supported by this prediction API.

Pass a list to `predict()` to return one result per image. Each call runs the list as one batch. A fixed-shape export must accept that batch size and input shape. If it does not, `predict()` raises before runtime execution. It does not split or pad the list.

`runtime_info` reports the selected runtime and device policy. Some vendor runtimes choose a physical device after model load, so the policy might not identify one chip. An explicit device request must match the runtime and available hardware. For example, TensorRT requires CUDA. Core AI accepts `auto` or `cpu`; its GPU and Neural Engine options express a preference and cannot pin execution to that device.

New exports include versioned inference metadata. ONNX stores it in the model. Other formats use an adjacent `<artifact-name>.rfdetr.json` file. For example, `model.trt` uses `model.trt.rfdetr.json`. Keep this file with the artifact. For an older export, pass a JSON path or mapping through `metadata=`. The metadata must supply missing task, class, output, and preprocessing semantics. The loader rejects missing or conflicting values. User-defined `notes` are separate from inference metadata.

The shared API prepares images and decodes results on the native model's device or the runtime adapter's tensor device. TensorRT keeps tensors on CUDA through decoding. The other exported adapters currently use CPU tensors for these stages. Runtime outputs belong to each call, so later calls cannot change earlier results. The benchmarks in the overview measure the per-runtime cookbook pipelines.

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

## Using the Exported Model

Use `RFDETRInference` for the common prediction path. The format guides also show direct runtime calls for advanced access and forward-only timing.
