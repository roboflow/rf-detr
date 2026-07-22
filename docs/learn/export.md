---
description: Export RF-DETR models to ONNX, TensorRT, TFLite, and ExecuTorch (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
---

# Export RF-DETR Model

!!! tip "Key Takeaways"

    - Export to ONNX for cross-platform inference with ONNX Runtime, OpenVINO, or TensorRT
    - Export to TFLite (FP32, FP16, INT8) for mobile and edge deployment
    - TensorRT conversion delivers lowest latency on NVIDIA GPUs (2.3 ms for Nano)
    - INT8 quantization requires calibration data from your dataset for accurate results
    - Custom input resolutions supported (must be divisible by `patch_size × num_windows`, which varies by model variant)
    - Export to ExecuTorch for on-device PyTorch inference (XNNPACK, CoreML, QNN)

RF-DETR supports exporting models to ONNX, TFLite, and ExecuTorch formats, enabling deployment across a wide range of inference frameworks, edge devices, and hardware accelerators.

## Installation

Install the export dependencies you need:

```bash
# ONNX export only
pip install "rfdetr[onnx]"

# TFLite export (includes ONNX dependency)
pip install "rfdetr[onnx,tflite]"

# ExecuTorch export (on-device inference: XNNPACK/CoreML/QNN)
pip install "rfdetr[executorch]"
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
| `format`           | `"onnx"`   | Export format: `"onnx"`, `"tflite"`, `"tensorrt"` (alias: `"trt"`), or `"executorch"`.                                                                                                          |
| `quantization`     | `None`     | TFLite quantization mode: `None`/`"fp32"`, `"fp16"`, or `"int8"`. Only used when `format="tflite"`.                                                                                             |
| `calibration_data` | `None`     | Calibration data for TFLite export. Image directory, `.npy` file path, NumPy array, or `None`. See [TFLite Export](#tflite-export).                                                             |
| `max_images`       | `100`      | Maximum number of images to load from a calibration directory for TFLite INT8 quantization. Ignored for other calibration data formats.                                                         |
| `infer_dir`        | `None`     | Optional directory of sample images for inference validation during export tracing. If not provided, a random dummy image is generated.                                                         |
| `backbone_only`    | `False`    | Export only the backbone feature extractor instead of the full model.                                                                                                                           |
| `opset_version`    | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                                                                                                  |
| `verbose`          | `True`     | Whether to print verbose export information.                                                                                                                                                    |
| `shape`            | `None`     | Input shape as tuple `(height, width)`. Each dimension must be divisible by the selected model's block size (`patch_size * num_windows`). If not provided, uses the model's default resolution. |
| `batch_size`       | `1`        | Batch size for the exported model.                                                                                                                                                              |
| `dynamic_batch`    | `False`    | If `True`, export with a dynamic batch dimension so the ONNX model accepts variable batch sizes at runtime.                                                                                     |
| `patch_size`       | `None`     | Backbone patch size override. Defaults to the value from `model_config.patch_size`. Must match the instantiated model's patch size when provided.                                               |
| `backend`          | `None`     | Backend for ExecuTorch: `"xnnpack"` (CPU, fp32), `"coreml"` (Apple, fp16), or `"qnn"` (Qualcomm HTP, fp16). Required when `format="executorch"`.                                                |
| `soc`              | `None`     | Target SoC chip identifier for the `"qnn"` backend (e.g. `"SM8650"` for Snapdragon 8 Gen 3). Required when `backend="qnn"`.                                                                     |
| `notes`            | `None`     | Optional user-defined metadata (string, dict, list, or any JSON-serialisable value) to embed in the exported ONNX model under the `"rfdetr_notes"` metadata property.                           |

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

## Output Files

After running the export, you will find the following files in your output directory:

- `inference_model.onnx` - The exported ONNX model (or `backbone_model.onnx` if `backbone_only=True`)

## Optional: Convert ONNX to TensorRT

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference.

### Prerequisites

- Install the TensorRT extra: `pip install rfdetr[tensorrt]` (provides `tensorrt` + `polygraphy`; no `trtexec` binary needed)
- A CUDA GPU (the engine is built for the local GPU architecture)
- Export an ONNX model first (for example: `output/inference_model.onnx`)

### Export Directly to TensorRT

Pass `format="tensorrt"` to `export()` to export ONNX and convert to a TensorRT engine in one step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt")
```

This exports `output/inference_model.onnx` first and then produces `output/inference_model.trt`.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw
    TensorRT deployment. It is locked to the GPU architecture and
    TensorRT version of the machine that built it, so it is not portable across
    different GPUs or TensorRT releases.

    If you plan to run inference with [`inference-models`](#run-inference-with-inference-models)
    (the recommended path below), do **not** pass `format="tensorrt"` — `inference-models`
    builds and manages its own TensorRT engine internally and does not consume this file.
    Export a plain ONNX model instead and let `inference-models` handle the backend.

### Python API Conversion

```python
from rfdetr.export._tensorrt import build_engine

engine_path = build_engine("output/inference_model.onnx", fp16=True)
```

`build_engine` builds the engine in-process via the TensorRT Python API (no `trtexec` subprocess) and returns the path to the generated `.trt` engine file.

## Run Inference with `inference-models`

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is the
recommended library for running RF-DETR inference. It supports multiple backends — PyTorch,
ONNX, and TensorRT — with automatic backend selection and a unified API.

### Installation

```bash
# CPU / PyTorch only
pip install inference-models

# With TensorRT support (NVIDIA GPU required)
pip install "inference-models[trt10]"  # TensorRT 10
```

See the [inference-models installation guide](https://inference-models.roboflow.com/getting-started/installation/)
for all installation options including Jetson and CUDA 11.x.

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

`AutoModel.from_pretrained` accepts `backend="onnx"`, `backend="torch"`, or
`backend="trt"` to override automatic backend selection.

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
    from rfdetr import RFDETRSmall

    model = RFDETRSmall()

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
from rfdetr import RFDETRSmall

model = RFDETRSmall()
target_resolution = model.model_config.resolution

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

## ExecuTorch Export

!!! warning "Experimental — Use with Caution"

    ExecuTorch export is **experimental**. The `executorch` package is under active
    development and its installation and API are subject to breaking changes between
    releases.

    **Known limitations:**

    - `dynamic_batch=True` is not supported: the runtime cannot resize RF-DETR's
        windowed-attention reshapes, so export one `.pte` per batch size instead.
    - The `"qnn"` backend requires a **source build** of ExecuTorch against the QAIRT
        SDK and cannot be installed via `pip`.
    - CoreML export runs in fp16; top-level detections are correct but raw tensor
        values will differ from the PyTorch fp32 model as expected for fp16 computation.

ExecuTorch is PyTorch's on-device inference runtime. Unlike ONNX export, the model is exported
directly via `torch.export` to a portable `.pte` binary — no intermediate ONNX conversion step
is involved.

### Prerequisites

```bash
pip install "rfdetr[executorch]"
```

### XNNPACK Backend (Portable CPU, fp32)

The `"xnnpack"` backend targets any CPU platform and runs in fp32. It is the recommended,
portable backend and requires only the standard `rfdetr[executorch]` wheel. `backend` has no
default — it must always be passed explicitly for `format="executorch"`.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="executorch", backend="xnnpack")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="executorch", backend="xnnpack")
    ```

This produces `output/rfdetr-seg-medium.pte` — the file is named after the model variant
(`{variant}.pte`), not a generic `inference_model.pte`.

### CoreML Backend (Apple Neural Engine, fp16)

The `"coreml"` backend targets Apple devices (iPhone, iPad, Mac) and runs in fp16 on the
Neural Engine. It requires `coremltools`, which is **not** included in the `rfdetr[executorch]`
extra — install it separately:

```bash
pip install coremltools
```

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="executorch", backend="coreml")
```

!!! note

    CoreML export uses fp16 arithmetic. Top-level detections (bounding boxes and class labels)
    are correct, but raw tensor values will differ from the PyTorch fp32 baseline at the fp16
    precision level — this is expected behavior.

### QNN Backend (Qualcomm Snapdragon HTP, fp16)

The `"qnn"` backend targets the Qualcomm AI Engine (HTP) on Snapdragon SoCs and runs in fp16.
It **requires a source build** of ExecuTorch against the QAIRT SDK and cannot be installed via `pip`.

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="executorch", backend="qnn", soc="SM8650")
```

The `soc` parameter is required for QNN and must be a `QcomChipset` name matching your target
device. For example, `"SM8650"` targets the Snapdragon 8 Gen 3.

!!! warning

    QNN export is validated on-device but cannot be tested in CI (requires QAIRT SDK).
    Validate detections on your target Snapdragon device before deploying to production.

### ExecuTorch Limitations

- **`dynamic_batch=True` is not supported.** The ExecuTorch runtime cannot resize RF-DETR's
    windowed-attention reshapes for a variable batch size. Export one `.pte` file per batch size
    instead (e.g. `batch_size=1` for single-image inference).
- **QNN requires a source build.** The QNN backend is not available via the pip wheel; see the
    ExecuTorch documentation for source-build instructions against the QAIRT SDK.

### ExecuTorch Inference Example

!!! warning "torch/executorch ABI compatibility"

    Loading a `.pte` via `executorch.runtime` (below) requires a `torch` version whose ABI
    matches the `executorch` wheel you installed — `.pte` **export** itself does not need
    `executorch.runtime` and is unaffected. For `executorch==1.3.1`, pin `torch<2.13`
    (`pip install "torch<2.13"`); a newer `torch` release can silently break `executorch.runtime`
    with an `undefined symbol` / `dlopen` error at import time, since ExecuTorch's prebuilt wheels
    are compiled against whichever `torch` ABI existed at their release time.

```python
import numpy as np
import torch
from executorch.runtime import Runtime
from PIL import Image

# Load the exported .pte program
runtime = Runtime.get()
method = runtime.load_program("output/rfdetr-medium.pte").load_method("forward")

# Prepare input — the .pte expects the same NCHW, ImageNet-normalized input as the ONNX export
image = Image.open("image.jpg").convert("RGB").resize((576, 576))
image_array = np.array(image, dtype=np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
image_array = np.expand_dims(image_array, axis=0)  # add batch dimension: (1, 3, H, W)
input_tensor = torch.from_numpy(image_array).float()

# Run inference
outputs = method.execute([input_tensor])
boxes, labels = outputs[0], outputs[1]
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

- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) for
    multi-backend inference (PyTorch, ONNX, TensorRT) with automatic backend selection

- Deploy TFLite models on mobile/edge devices with TensorFlow Lite

- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime

- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
