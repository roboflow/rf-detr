---
description: Export RF-DETR models to ONNX, TensorRT, TFLite, ExecuTorch, and native CoreML (FP32/FP16/INT8) for high-performance inference on GPUs, mobile, and edge devices.
---

# Export RF-DETR Model

!!! tip "Key Takeaways"

    - Export to ONNX for cross-platform inference with ONNX Runtime, OpenVINO, or TensorRT
    - Export to TFLite (FP32, FP16, INT8) for mobile and edge deployment
    - TensorRT conversion delivers lowest latency on NVIDIA GPUs (2.3 ms for Nano)
    - INT8 quantization is dynamic-range and needs no calibration data
    - Custom input resolutions supported (must be divisible by `patch_size × num_windows`, which varies by model variant)
    - Export to ExecuTorch for on-device PyTorch inference (XNNPACK, CoreML, QNN)
    - Export directly to native CoreML (`.mlpackage`) for Xcode / Apple-platform deployment — see [Native CoreML Export](#native-coreml-export-mlpackage)

RF-DETR supports exporting models to ONNX, TFLite, ExecuTorch, and native CoreML formats, enabling deployment across a wide range of inference frameworks, edge devices, and hardware accelerators.

## Installation

Install the export dependencies you need:

```bash
# ONNX export only
pip install "rfdetr[onnx]"

# TFLite export
pip install "rfdetr[tflite]"

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

| Parameter          | Default    | Description                                                                                                                                                                                      |
| ------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output_dir`       | `"output"` | Directory where the exported model will be saved.                                                                                                                                                |
| `format`           | `"onnx"`   | Export format: `"onnx"`, `"tflite"`, `"tensorrt"` (alias: `"trt"`), `"executorch"`, or `"coreml"`.                                                                                               |
| `quantization`     | `None`     | TFLite quantization mode: `None`/`"fp32"`, `"fp16"`, or `"int8"`. Only used when `format="tflite"`.                                                                                              |
| `calibration_data` | `None`     | Optional image directory, `.npy` file path, NumPy array, or `None`. Not consumed when building the generated `.tflite` models.                                                                   |
| `max_images`       | `100`      | Maximum number of images to load from a `calibration_data` directory. Ignored for other calibration data formats.                                                                                |
| `infer_dir`        | `None`     | Optional directory of sample images for inference validation during export tracing. If not provided, a random dummy image is generated.                                                          |
| `backbone_only`    | `False`    | Export only the backbone feature extractor instead of the full model.                                                                                                                            |
| `opset_version`    | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                                                                                                   |
| `verbose`          | `True`     | Whether to print verbose export information.                                                                                                                                                     |
| `shape`            | `None`     | Input shape as tuple `(height, width)`. Each dimension must be divisible by the selected model's block size (`patch_size * num_windows`). If not provided, uses the model's default resolution.  |
| `batch_size`       | `1`        | Batch size for the exported model.                                                                                                                                                               |
| `dynamic_batch`    | `False`    | If `True`, export with a dynamic batch dimension so the ONNX model accepts variable batch sizes at runtime.                                                                                      |
| `patch_size`       | `None`     | Backbone patch size override. Defaults to the value from `model_config.patch_size`. Must match the instantiated model's patch size when provided.                                                |
| `backend`          | `None`     | Backend for ExecuTorch: `"xnnpack"` (CPU, fp32), `"coreml"` (Apple, fp16), or `"qnn"` (Qualcomm HTP, fp16). Required when `format="executorch"`.                                                 |
| `soc`              | `None`     | Target SoC chip identifier for the `"qnn"` backend (e.g. `"SM8650"` for Snapdragon 8 Gen 3). Required when `backend="qnn"`.                                                                      |
| `fp16`             | `True`     | Build the TensorRT engine with FP16 precision (only used when `format="tensorrt"`). Pass `False` to build an FP32 engine — required on TensorRT builds that do not expose the FP16 builder flag. |
| `notes`            | `None`     | Optional user-defined metadata (string, dict, list, or any JSON-serialisable value) to embed in the exported ONNX model under the `"rfdetr_notes"` metadata property.                            |
| `coreml_precision` | `None`     | Compute precision for `format="coreml"`: `None`/`"float32"` (tight CPU parity with eager PyTorch) or `"float16"` (smaller, ANE-oriented bundle). Ignored for every other format.                 |
| `output_name`      | `None`     | Full filename override (without extension). Takes precedence over the model's variant name and suppresses the `_fp32`/`_fp16`/`_{backend}` detail suffix — see [Output Files](#output-files).    |

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

Filenames are built from the model's variant name (e.g. `rfdetr-medium`, falling back to `inference_model` when no variant or `output_name` is set, or `backbone_model` when `backbone_only=True` in that same case) plus a detail suffix whenever a detail materially changes the artifact — even at its default value, since the file needs to say what it actually is:

| Format       | Filename pattern                                                                                                                                                                       | Detail encoded                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `onnx`       | `{variant}.onnx` (or `{variant}-backbone.onnx` if `backbone_only=True`); without a variant or `output_name`, `inference_model.onnx` (or `backbone_model.onnx` if `backbone_only=True`) | none — `-backbone` is structural, not a precision detail |
| `coreml`     | `{variant}_fp32.mlpackage` / `{variant}_fp16.mlpackage`                                                                                                                                | `coreml_precision`                                       |
| `executorch` | `{variant}_xnnpack.pte` / `{variant}_coreml.pte` / `{variant}_qnn_{soc}.pte`                                                                                                           | `backend` (+ `soc` for `qnn`)                            |
| `tensorrt`   | `{variant}_fp16.trt` / `{variant}_fp32.trt`                                                                                                                                            | `fp16`                                                   |
| `tflite`     | `{variant}_fp32.tflite` + `{variant}_fp16.tflite` (+ `{variant}_dynamic_range_quant.tflite` for `quantization="int8"`)                                                                 | precision / quantization mode                            |

Pass `output_name="my-model"` to override the variant name and write `{output_name}.{ext}` verbatim — this suppresses the detail suffix for every format **except** `tflite`, which always writes multiple files and so keeps its `_fp32`/`_fp16`/`_dynamic_range_quant` suffix even with a custom name (`{output_name}_fp32.tflite`, etc.).

## Optional: Convert ONNX to TensorRT

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
>
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

This exports `output/inference_model.onnx` first and then produces `output/inference_model_fp16.trt` (the `_fp16`/`_fp32` suffix always reflects the precision actually built — see `fp16` in [Export Parameters](#export-parameters) — unless `output_name` is set).

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. It is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases.

    If you plan to run inference with [`inference-models`](#run-inference-with-inference-models) (the recommended path below), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

### Python API Conversion

```python
from rfdetr.export._tensorrt import build_engine

engine_path = build_engine("output/inference_model.onnx", fp16=True)
# -> "output/inference_model_fp16.trt"
```

`build_engine` builds the engine in-process via the TensorRT Python API (no `trtexec` subprocess) and returns the path to the generated `.trt` engine file. Pass `output_name="my-engine"` to write `output/my-engine.trt` verbatim instead.

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

## TFLite Export

!!! warning "Experimental — Use with Caution"

    TFLite export is **experimental and work-in-progress**. The pipeline depends on several upstream packages (`onnx2tf`, `ai_edge_litert`, `tflite-runtime`) that have experienced breaking API changes and installation instabilities across releases. You may encounter errors or unexpected results.

    **Known instabilities:**

    - `onnx2tf` output graph structure can change between minor versions, silently altering output tensor layout and breaking downstream inference code.
    - `ai_edge_litert` (Google's replacement for `tflite-runtime`) is still stabilising its public API; version pinning is strongly recommended.
    - INT8 quantization is dynamic-range (INT8 weights, float activations). It is applied without calibration, and quantizing a transformer's weights to 8 bits can still cost accuracy — validate the INT8 model before deploying it.
    - The ONNX → TF → TFLite conversion chain introduces numerical rounding that may produce slightly different predictions from the original PyTorch model.
    - Installation of the `[tflite]` extra may conflict with existing TensorFlow or NumPy versions in your environment.
    - `onnx` and TensorFlow both bundle Abseil and export its symbols weakly, so whichever loads first supplies them to both. RF-DETR imports TensorFlow first on the TFLite route; if your own code imports `onnx` before `tensorflow`, RF-DETR logs a warning and the conversion may block forever while restoring the SavedModel (no error, 0% CPU). Importing `onnx` *after* `tensorflow` is safe; otherwise, in a fresh process, preload/import `tensorflow` before `onnx` and then run the export — freshness alone is not sufficient.

    **Recommendations:**

    - Pin your dependency versions (e.g. `onnx2tf==X.Y.Z`) and test before each upgrade.
    - Validate exported `.tflite` files against a held-out evaluation set before deploying.
    - Prefer ONNX export when your target runtime supports it — it is more stable and better tested.
    - If export fails, check the [open issues](https://github.com/roboflow/rf-detr/issues) for known workarounds or report a new one with your environment details (`pip freeze`, Python version, OS).

Export your model to TFLite for deployment on mobile devices, microcontrollers, and edge hardware via TensorFlow Lite. The TFLite export pipeline converts ONNX → TensorFlow → TFLite using [onnx2tf](https://github.com/PINTO0309/onnx2tf).

### Prerequisites

```bash
pip install "rfdetr[tflite]"
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

This produces both `output/inference_model_fp32.tflite` and `output/inference_model_fp16.tflite`.

### INT8 Quantization

`quantization="int8"` produces a **dynamic-range** INT8 model: weights are stored as INT8, activations stay in float, and the weight scales are derived from the weights themselves. No calibration data is required, and supplying it does not change the result — static/full-integer INT8, the mode that *would* need representative data, is intentionally unsupported because RF-DETR's transformer activations do not survive it.

`calibration_data` accepts a directory of JPEG/PNG images, a path to an `.npy` file of shape `(N, H, W, 3)` (float32, values in `[0, 1]`), or a NumPy array in that format; `max_images` caps how many images are read from a directory. These arguments are not consumed when building the generated `.tflite` models. Omitting them is the normal path:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
model.export(format="tflite", quantization="int8", output_dir="output")
```

This writes `output/inference_model_dynamic_range_quant.tflite` alongside the FP32 and FP16 models.

### FP16 Export

FP16 models are always produced alongside FP32. You can explicitly request FP16 mode:

```python
model.export(format="tflite", quantization="fp16", output_dir="output")
```

### TFLite Output Files

The `onnx2tf` converter **always** produces both FP32 and FP16 TFLite files, regardless of the requested quantization mode. When `quantization="int8"` is specified, it additionally produces the INT8-quantized model.

| File                                         | Description                             |
| -------------------------------------------- | --------------------------------------- |
| `inference_model_fp32.tflite`                | FP32 model (always produced)            |
| `inference_model_fp16.tflite`                | FP16 model (always produced)            |
| `inference_model_dynamic_range_quant.tflite` | INT8 model (when `quantization="int8"`) |

!!! note

    Segmentation models produce TFLite files with three outputs: `dets` (bounding boxes), `labels` (class scores), and `masks` (per-instance segmentation masks).

### TFLite Inference Example

```python
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

# pip install tflite-runtime  (or use tensorflow.lite)
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="output/inference_model_fp32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input — TFLite model expects NHWC, ImageNet-normalized
input_height, input_width = input_details[0]["shape"][1:3]
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)

# Apply ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

# Add batch dimension: (1, H, W, 3)
image_array = image_tensor.permute(1, 2, 0).unsqueeze(0).contiguous().numpy().astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]["index"], image_array)
interpreter.invoke()

boxes_detail = next((detail for detail in output_details if "dets" in str(detail.get("name", ""))), None)
labels_detail = next((detail for detail in output_details if "labels" in str(detail.get("name", ""))), None)
if boxes_detail is None or labels_detail is None:
    raise ValueError(f"Expected TFLite outputs named dets and labels; got {output_details}")

boxes = interpreter.get_tensor(boxes_detail["index"])
labels = interpreter.get_tensor(labels_detail["index"])
```

## ExecuTorch Export

!!! warning "Experimental — Use with Caution"

    ExecuTorch export is **experimental**. The `executorch` package is under active development and its installation and API are subject to breaking changes between releases.

    **Known limitations:**

    - `dynamic_batch=True` is not supported: the runtime cannot resize RF-DETR's windowed-attention reshapes, so export one `.pte` per batch size instead.
    - The `"qnn"` backend requires a **source build** of ExecuTorch against the QAIRT SDK and cannot be installed via `pip`.
    - CoreML export runs in fp16; top-level detections are correct but raw tensor values will differ from the PyTorch fp32 model as expected for fp16 computation.

ExecuTorch is PyTorch's on-device inference runtime. Unlike ONNX export, the model is exported directly via `torch.export` to a portable `.pte` binary — no intermediate ONNX conversion step is involved.

### Prerequisites

```bash
pip install "rfdetr[executorch]"
```

### XNNPACK Backend (Portable CPU, fp32)

The `"xnnpack"` backend targets any CPU platform and runs in fp32. It is the recommended, portable backend and requires only the standard `rfdetr[executorch]` wheel. `backend` has no default — it must always be passed explicitly for `format="executorch"`.

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

This produces `output/rfdetr-seg-medium_xnnpack.pte` — the file is named after the model variant plus the backend (`{variant}_{backend}.pte`, or `{variant}_qnn_{soc}.pte` for the SoC-locked `qnn` backend), not a generic `inference_model_{backend}.pte`. The backend is always encoded because it determines which hardware/runtime can load the file.

### CoreML Backend (Apple Neural Engine, fp16)

!!! note "Not the same as native CoreML export"

    This is the ExecuTorch delegate — `format="executorch", backend="coreml"` — which produces a `.pte` file for the ExecuTorch runtime. It is distinct from `format="coreml"`, which produces a native `.mlpackage` directly (no ExecuTorch runtime involved); see [Native CoreML Export](#native-coreml-export-mlpackage) below.

The `"coreml"` backend targets Apple devices (iPhone, iPad, Mac) and runs in fp16 on the Neural Engine. It requires `coremltools`, which is **not** included in the `rfdetr[executorch]` extra — install it separately:

```bash
pip install coremltools
```

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="executorch", backend="coreml")
```

!!! note

    CoreML export uses fp16 arithmetic. Top-level detections (bounding boxes and class labels) are correct, but raw tensor values will differ from the PyTorch fp32 baseline at the fp16 precision level — this is expected behavior.

### QNN Backend (Qualcomm Snapdragon HTP, fp16)

The `"qnn"` backend targets the Qualcomm AI Engine (HTP) on Snapdragon SoCs and runs in fp16. It **requires a source build** of ExecuTorch against the QAIRT SDK and cannot be installed via `pip`.

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="executorch", backend="qnn", soc="SM8650")
```

The `soc` parameter is required for QNN and must be a `QcomChipset` name matching your target device. For example, `"SM8650"` targets the Snapdragon 8 Gen 3. This produces `output/rfdetr-medium_qnn_SM8650.pte` — the SoC is baked into the filename (not just the backend) since a QNN `.pte` is compiled ahead-of-time for one specific chip and will not run on another.

!!! warning

    QNN export is validated on-device but cannot be tested in CI (requires QAIRT SDK). Validate detections on your target Snapdragon device before deploying to production.

### ExecuTorch Limitations

- **`dynamic_batch=True` is not supported.** The ExecuTorch runtime cannot resize RF-DETR's windowed-attention reshapes for a variable batch size. Export one `.pte` file per batch size instead (e.g. `batch_size=1` for single-image inference).
- **QNN requires a source build.** The QNN backend is not available via the pip wheel; see the ExecuTorch documentation for source-build instructions against the QAIRT SDK.

### ExecuTorch Inference Example

!!! warning "torch/executorch ABI compatibility"

    Loading a `.pte` via `executorch.runtime` (below) requires a `torch` version whose ABI matches the `executorch` wheel you installed — `.pte` **export** itself does not need `executorch.runtime` and is unaffected. For `executorch==1.3.1`, pin `torch<2.13` (`pip install "torch<2.13"`); a newer `torch` release can silently break `executorch.runtime` with an `undefined symbol` / `dlopen` error at import time, since ExecuTorch's prebuilt wheels are compiled against whichever `torch` ABI existed at their release time.

!!! warning "The input tensor must be contiguous"

    The ExecuTorch runtime reads the input buffer as contiguous NCHW and ignores tensor strides. Preprocessing steps that permute axes — `np.transpose`, `Tensor.permute`, torchvision's `ToImage` — return a strided view rather than a copy, and such a view is misread as a scrambled image. Nothing errors: the model runs without error and returns plausible-shaped output, but every detection's score collapses below threshold. Finish preprocessing with `np.ascontiguousarray(...)` (or `Tensor.contiguous()`) before calling `execute`.

```python
import torch
from executorch.runtime import Runtime
from PIL import Image
import torchvision.transforms.functional as F

# Load the exported .pte program
runtime = Runtime.get()
method = runtime.load_program("output/rfdetr-medium_xnnpack.pte").load_method("forward")

# Prepare input — the .pte expects the same NCHW, ImageNet-normalized input as the ONNX export
input_height, input_width = 576, 576
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

image_array = image_tensor.unsqueeze(0).contiguous().numpy()  # add batch dimension: (1, 3, H, W)
input_tensor = torch.from_numpy(image_array).float()

# Run inference
outputs = method.execute([input_tensor])
boxes, labels = outputs[0], outputs[1]
```

## Native CoreML Export (`.mlpackage`)

!!! warning "Experimental — Use with Caution"

    Native CoreML export is **experimental and work-in-progress**. `dynamic_batch=True` is not supported — fixed shapes are required for reliable ANE / GPU scheduling. Export one `.mlpackage` per batch size instead.

!!! note "Not the same as the ExecuTorch CoreML backend"

    `format="coreml"` exports directly via `torch.export` + `coremltools` to a native `.mlpackage` (mlprogram, iOS 16+) — no ONNX and no ExecuTorch runtime involved. This is distinct from [`format="executorch", backend="coreml"`](#coreml-backend-apple-neural-engine-fp16), which produces a `.pte` file for the ExecuTorch runtime. Passing both `format="coreml"` and `backend="coreml"` together does not fall through to the ExecuTorch delegate — `backend` is ignored (with a warning) and the native `.mlpackage` path always runs.

RF-DETR's native CoreML export produces a `.mlpackage` you can drag directly into Xcode, with no ONNX intermediary and no ExecuTorch runtime dependency — the lowest-friction path for Apple-native (iOS / macOS) developers.

### Prerequisites

```bash
pip install "rfdetr[coreml]"
```

### Basic CoreML Export

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="coreml")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="coreml")
    ```

This produces `output/rfdetr-medium_fp32.mlpackage` — the file is named after the model variant plus the resolved precision (`{variant}_fp32.mlpackage` / `{variant}_fp16.mlpackage`), not a generic `inference_model_fp32.mlpackage`. The precision is always encoded, even at its default value, since fp16 vs fp32 materially changes the bundle.

### Compute Precision

CoreML export defaults to `FLOAT32` for tight CPU parity with eager PyTorch. Pass `coreml_precision="float16"` for a smaller, ANE-oriented bundle (expect larger numeric drift) — this also changes the output filename to `output/rfdetr-medium_fp16.mlpackage`:

```python
model.export(format="coreml", coreml_precision="float16")
```

!!! note

    Output tensor names in the saved `.mlpackage` spec are coremltools-inferred, not renamed to `dets`/`labels`/etc. — match outputs by **position**, in the same order as the ONNX `output_names` contract (`dets, labels` for detection; `dets, labels, masks` for segmentation).

### CoreML Inference Example

```python
import coremltools as ct
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

mlmodel = ct.models.MLModel("output/rfdetr-medium_fp32.mlpackage")

input_height, input_width = 576, 576
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

image_array = image_tensor.unsqueeze(0).numpy()  # add batch dimension: (1, 3, H, W)

# Outputs are positional (see the precision note above) — dets, labels, in that order.
outputs = list(mlmodel.predict({"input": image_array.astype(np.float32)}).values())
boxes, labels = outputs[0], outputs[1]
```

## Using the Exported Model

Once exported, you can use the ONNX model with various inference frameworks:

### ONNX Runtime

The exported graph returns **raw** tensors — `dets` (`pred_boxes`, normalized `cxcywh`) and `labels` (`pred_logits`, un-activated). Nothing is decoded inside the graph, so your inference code must apply sigmoid, drop the no-object background column, and convert box format yourself.

!!! warning "Match outputs by name, not by shape"

    RF-DETR always adds an implicit no-object class, so the logits tensor's last dimension is `num_classes + 1`. If `num_classes == 3`, that dimension is `4` — identical to the box tensor's last dimension (`4`, `cxcywh`). Disambiguating outputs by shape instead of by name (`"dets"` / `"labels"`) will silently swap boxes and logits at exactly `num_classes == 3`, producing garbage detections while every other `num_classes` value looks fine. Always match by name first.

```python
import onnxruntime as ort
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

# Load the ONNX model
session = ort.InferenceSession("output/inference_model.onnx")

# Prepare input image
input_height, input_width = session.get_inputs()[0].shape[2:4]
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)

# Normalize
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

# Convert to NCHW format
image_array = image_tensor.unsqueeze(0).numpy()

# Run inference
outputs = session.run(None, {"input": image_array})

# Match outputs by name — do NOT assume positional order or infer role from shape.
output_names = [out.name for out in session.get_outputs()]
boxes_idx = next((i for i, name in enumerate(output_names) if "dets" in name), None)
logits_idx = next((i for i, name in enumerate(output_names) if "labels" in name), None)
if boxes_idx is None or logits_idx is None:
    raise ValueError(f"Could not find expected outputs 'dets'/'labels'. Available outputs: {output_names}")

boxes_cwh = outputs[boxes_idx][0]  # (num_queries, 4) normalized cxcywh
# Drop the last logit column: RF-DETR appends a no-object slot (num_classes + 1 total).
logits = outputs[logits_idx][0, :, :-1]  # (num_queries, num_classes)

# RF-DETR uses per-class sigmoid (multi-label), not softmax.
scores_all = 1.0 / (1.0 + np.exp(-logits.clip(-88, 88)))
scores = scores_all.max(axis=-1)
class_ids = scores_all.argmax(axis=-1)

threshold = 0.5
keep = scores > threshold

# cxcywh (normalized) -> xyxy (pixel space)
cx, cy, bw, bh = boxes_cwh[keep].T
xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
xyxy *= np.array([image.width, image.height, image.width, image.height], dtype=np.float32)

boxes, labels, confidences = xyxy, class_ids[keep], scores[keep]
```

For a fuller reference implementation (name-based matching with a documented shape-based fallback), see `_run_inference` in [`src/rfdetr/export/_onnx/inference.py`](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/export/_onnx/inference.py).

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](deploy.md) for cloud-based inference and workflow integration

- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) for multi-backend inference (PyTorch, ONNX, TensorRT) with automatic backend selection

- Deploy TFLite models on mobile/edge devices with TensorFlow Lite

- Deploy ExecuTorch `.pte` models on mobile/edge devices with the ExecuTorch runtime

- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
