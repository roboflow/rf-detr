---
description: Export RF-DETR models to ONNX, TensorRT, and TFLite (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
---

# Export RF-DETR Model

!!! tip "Key Takeaways"

    - Export to ONNX for cross-platform inference with ONNX Runtime, OpenVINO, or TensorRT
    - Export to TFLite (FP32, FP16, INT8) for mobile and edge deployment
    - TensorRT conversion delivers lowest latency on NVIDIA GPUs (2.3 ms for Nano)
    - INT8 quantization requires calibration data from your dataset for accurate results
    - Custom input resolutions supported (must be divisible by 14)

RF-DETR supports exporting models to ONNX and TFLite formats, enabling deployment across a wide range of inference frameworks, edge devices, and hardware accelerators.

## Installation

Install the export dependencies you need:

```bash
# ONNX export only
pip install "rfdetr[onnx]"

# TFLite export (includes ONNX dependency)
pip install "rfdetr[onnx,tflite]"
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

| Parameter          | Default    | Description                                                                                                                                                                                     |
| ------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_dir`       | `"output"` | Directory where the exported model will be saved.                                                                                                                                               |
| `format`           | `"onnx"`   | Export format: `"onnx"` or `"tflite"`.                                                                                                                                                          |
| `quantization`     | `None`     | TFLite quantization mode: `None`/`"fp32"`, `"fp16"`, or `"int8"`. Only used when `format="tflite"`.                                                                                             |
| `calibration_data` | `None`     | Calibration data for TFLite export. Image directory, `.npy` file path, NumPy array, or `None`. See [TFLite Export](#tflite-export).                                                             |
| `max_images`       | `100`      | Maximum number of images to load from a calibration directory for TFLite INT8 quantization. Ignored for other calibration data formats.                                                         |
| `infer_dir`        | `None`     | Path to an image file to use for tracing. If not provided, a random dummy image is generated.                                                                                                   |
| `simplify`         | `False`    | Deprecated and ignored. ONNX simplification is no longer run by `export()`.                                                                                                                     |
| `backbone_only`    | `False`    | Export only the backbone feature extractor instead of the full model.                                                                                                                           |
| `opset_version`    | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                                                                                                  |
| `verbose`          | `True`     | Whether to print verbose export information.                                                                                                                                                    |
| `force`            | `False`    | Deprecated and ignored.                                                                                                                                                                         |
| `shape`            | `None`     | Input shape as tuple `(height, width)`. Each dimension must be divisible by the selected model's block size (`patch_size * num_windows`). If not provided, uses the model's default resolution. |
| `batch_size`       | `1`        | Batch size for the exported model.                                                                                                                                                              |

## Advanced Export Examples

### Export with Custom Output Directory

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(output_dir="exports/my_model")
```

### Deprecated: Export with Simplification

The `simplify` flag is deprecated and ignored:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(simplify=True)  # Deprecated: same result as model.export()
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

## Output Files

After running the export, you will find the following files in your output directory:

- `inference_model.onnx` - The exported ONNX model (or `backbone_model.onnx` if `backbone_only=True`)

## Optional: Convert ONNX to TensorRT

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference.

### Prerequisites

- Install TensorRT (`trtexec` must be available in your `PATH`)
- Export an ONNX model first (for example: `output/inference_model.onnx`)

### Python API Conversion

```python
from argparse import Namespace

from rfdetr.export._tensorrt import trtexec

args = Namespace(
    verbose=True,
    profile=False,
    dry_run=False,
)

trtexec("output/inference_model.onnx", args)
```

This produces `output/inference_model.engine`. If `profile=True`, it also writes an Nsight Systems report (`.nsys-rep`).

## TFLite Export

!!! warning "Experimental — Use with Caution"

    TFLite export is **experimental and work-in-progress**. The pipeline depends on
    several upstream packages (`onnx2tf`, `ai_edge_litert`, `tflite-runtime`) that
    have experienced breaking API changes and installation instabilities across
    releases. You may encounter errors or unexpected results.

    **Known instabilities:**

    - `onnx2tf` output graph structure can change between minor versions, silently
        altering output tensor layout and breaking downstream inference code.
    - `ai_edge_litert` (Google's replacement for `tflite-runtime`) is still
        stabilising its public API; version pinning is strongly recommended.
    - INT8 quantization accuracy is sensitive to calibration data quality — poor
        calibration causes silent precision loss with no error at export time.
    - The ONNX → TF → TFLite conversion chain introduces numerical rounding that
        may produce slightly different predictions from the original PyTorch model.
    - Installation of the `[tflite]` extra may conflict with existing TensorFlow
        or NumPy versions in your environment.

    **Recommendations:**

    - Pin your dependency versions (e.g. `onnx2tf==X.Y.Z`) and test before each upgrade.
    - Validate exported `.tflite` files against a held-out evaluation set before deploying.
    - Prefer ONNX export when your target runtime supports it — it is more stable and
        better tested.
    - If export fails, check the [open issues](https://github.com/roboflow/rf-detr/issues)
        for known workarounds or report a new one with your environment details
        (`pip freeze`, Python version, OS).

Export your model to TFLite for deployment on mobile devices, microcontrollers, and edge hardware via TensorFlow Lite. The TFLite export pipeline converts ONNX → TensorFlow → TFLite using [onnx2tf](https://github.com/PINTO0309/onnx2tf).

### Prerequisites

```bash
pip install "rfdetr[onnx,tflite]"
```

### Basic TFLite Export (FP32)

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    model.export(format="tflite", output_dir="output")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano()

    model.export(format="tflite", output_dir="output")
    ```

This produces both `output/inference_model_float32.tflite` and `output/inference_model_float16.tflite`.

### INT8 Quantization with Calibration Data

For INT8 quantization, provide representative images from your dataset as calibration data. This is **critical** for preserving model accuracy — without real calibration data, the quantizer uses random noise and accuracy will be poor.

#### Option 1: Point to an Image Directory (Recommended)

The simplest approach — just point `calibration_data` to a directory containing JPEG/PNG images. The converter automatically loads, resizes, and prepares the images:

```python
from rfdetr import RFDETRNano

model = RFDETRNano()
model.export(
    format="tflite",
    quantization="int8",
    calibration_data="path/to/val2017/",  # directory of images
    output_dir="output",
)
```

The converter loads up to 100 images from the directory by default, resizes them to the model's input resolution, and uses them for both output validation and INT8 calibration. Supported formats: JPEG, PNG, BMP, WebP.

You can control how many images are loaded with the `max_images` parameter:

```python
model.export(
    format="tflite",
    quantization="int8",
    calibration_data="path/to/val2017/",
    max_images=200,  # load up to 200 images (default: 100)
    output_dir="output",
)
```

#### Option 2: NumPy `.npy` File

Prepare calibration data as a NumPy array and save it to a `.npy` file:

- Shape: `(N, H, W, 3)` — NHWC format with 3 color channels
- Data type: `float32`
- Value range: `[0, 1]` (divide by 255, but do **not** apply ImageNet normalization — the converter handles that automatically)
- Recommended: 20–100 representative images from your dataset

```python
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase

model = RFDETRBase()
target_resolution = model.resolution

# Load representative images from your dataset
images = []
for path in image_paths[:50]:  # 50 representative samples
    img = Image.open(path).convert("RGB").resize((target_resolution, target_resolution))
    images.append(np.array(img, dtype=np.float32) / 255.0)

calibration_data = np.stack(images)  # shape: (50, H, W, 3)

# Save to .npy for reuse
np.save("calibration_data.npy", calibration_data)

# Export with INT8 quantization
model.export(
    format="tflite",
    quantization="int8",
    calibration_data="calibration_data.npy",
    output_dir="output",
)
```

#### Option 3: NumPy Array Directly

You can also pass the NumPy array directly without saving to disk:

```python
model.export(
    format="tflite",
    quantization="int8",
    calibration_data=calibration_data,  # np.ndarray
    output_dir="output",
)
```

### FP16 Export

FP16 models are always produced alongside FP32. You can explicitly request FP16 mode:

```python
model.export(format="tflite", quantization="fp16", output_dir="output")
```

### TFLite Output Files

The `onnx2tf` converter **always** produces both FP32 and FP16 TFLite files, regardless of the requested quantization mode. When `quantization="int8"` is specified, it additionally produces the INT8-quantized model.

| File                                   | Description                             |
| -------------------------------------- | --------------------------------------- |
| `inference_model_float32.tflite`       | FP32 model (always produced)            |
| `inference_model_float16.tflite`       | FP16 model (always produced)            |
| `inference_model_integer_quant.tflite` | INT8 model (when `quantization="int8"`) |

!!! note

    Segmentation models produce TFLite files with three outputs: `dets` (bounding boxes), `labels` (class scores), and `masks` (per-instance segmentation masks).

### TFLite Inference Example

```python
import numpy as np
from PIL import Image

# pip install tflite-runtime  (or use tensorflow.lite)
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="output/inference_model_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input — TFLite model expects NHWC, ImageNet-normalized
input_height, input_width = input_details[0]["shape"][1:3]
image = Image.open("image.jpg").convert("RGB").resize((input_width, input_height))
image_array = np.array(image, dtype=np.float32) / 255.0

# Apply ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

# Add batch dimension: (1, H, W, 3)
image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]["index"], image_array)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]["index"])
labels = interpreter.get_tensor(output_details[1]["index"])
```

## Using the Exported Model

Once exported, you can use the ONNX model with various inference frameworks:

### ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
session = ort.InferenceSession("output/inference_model.onnx")

# Prepare input image
input_height, input_width = session.get_inputs()[0].shape[2:4]
image = Image.open("image.jpg").convert("RGB")
image = image.resize((input_width, input_height))  # Resize to the exported model shape
image_array = np.array(image).astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

# Convert to NCHW format
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Run inference
outputs = session.run(None, {"input": image_array})
boxes, labels = outputs
```

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](deploy.md) for cloud-based inference and workflow integration
- Use the ONNX model with TensorRT for optimized GPU inference
- Deploy TFLite models on mobile/edge devices with TensorFlow Lite
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
