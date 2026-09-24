---
description: Export RF-DETR models to TFLite (FP32, FP16, INT8) for mobile and edge deployment.
---

# TFLite Export

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

## Prerequisites

```bash
pip install "rfdetr[tflite]"
```

## Basic TFLite Export (FP32)

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

## INT8 Quantization

`quantization="int8"` produces a **dynamic-range** INT8 model: weights are stored as INT8, activations stay in float, and the weight scales are derived from the weights themselves. No calibration data is required, and supplying it does not change the result — static/full-integer INT8, the mode that *would* need representative data, is intentionally unsupported because RF-DETR's transformer activations do not survive it.

Dynamic-range INT8 requires a float-capable runtime and is not suitable for integer-only accelerators such as the Coral Edge TPU or integer-only NPUs.

`calibration_data` accepts a directory of JPEG, PNG, BMP or WebP images, a path to an `.npy` file of shape `(N, H, W, 3)` (float32, values in `[0, 1]`), or a NumPy array in that format; `max_images` caps how many images are read from a directory. These arguments are not consumed when building the generated `.tflite` models. Omitting them is the normal path:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
model.export(format="tflite", quantization="int8", output_dir="output")
```

This writes `output/inference_model_dynamic_range_quant.tflite` alongside the FP32 and FP16 models. When GridSample ops are patched, the filename includes a `_gs_patched` infix: `output/inference_model_gs_patched_dynamic_range_quant.tflite` (the standard RF-DETR path).

## FP16 Export

FP16 models are always produced alongside FP32. You can explicitly request FP16 mode:

```python
model.export(format="tflite", quantization="fp16", output_dir="output")
```

## TFLite Output Files

The `onnx2tf` converter **always** produces both FP32 and FP16 TFLite files, regardless of the requested quantization mode. When `quantization="int8"` is specified, it additionally produces the INT8-quantized model.

| File                                         | Description                             |
| -------------------------------------------- | --------------------------------------- |
| `inference_model_fp32.tflite`                | FP32 model (always produced)            |
| `inference_model_fp16.tflite`                | FP16 model (always produced)            |
| `inference_model_dynamic_range_quant.tflite` | INT8 model (when `quantization="int8"`) |

!!! note

    Segmentation models produce TFLite files with three outputs: `dets` (bounding boxes), `labels` (class scores), and `masks` (per-instance segmentation masks). Keypoint models produce three outputs too, the third being `keypoints`.

!!! warning "RF-DETR's TFLite outputs are not named `dets` / `labels`"

    RF-DETR converts through `onnx2tf`'s SavedModel route, which renames every output: the `dets` / `labels` / `masks` / `keypoints` names visible in the `.onnx` file arrive as `StatefulPartitionedCall:0`, `StatefulPartitionedCall:1`, … in the `.tflite` file, and the signature def exposes them as `output_0`, `output_1`, … Only the *input* keeps a readable name (`serving_default_input:0`). Match outputs by rank and last dimension instead — boxes are the rank-3 tensor with last dim `4`, logits the other rank-3 tensor — and treat the name check as a best-effort first attempt.

    Segmentation masks and keypoints are both rank-4, so neither the name nor the rank tells them apart. The TFLite `_run_inference` reference helper safely defaults `rank4_output` to `None`, decoding a mask only from an output that names itself. For a name-stripped segmentation export, pass `rank4_output="masks"`; pass `"keypoints"` to suppress anonymous-mask decoding for a keypoint export.

## TFLite Inference Example

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

# onnx2tf renames the outputs, so the ONNX names are usually gone: fall back to rank and last dimension.
# The fallback cannot resolve num_classes == 3, where the logits' last dimension is 4 as well; it raises there.
boxes_detail = next((detail for detail in output_details if "dets" in str(detail.get("name", ""))), None)
labels_detail = next((detail for detail in output_details if "labels" in str(detail.get("name", ""))), None)
if boxes_detail is None or labels_detail is None:
    rank3 = [detail for detail in output_details if len(detail["shape"]) == 3]
    boxes_detail = next((detail for detail in rank3 if detail["shape"][-1] == 4), None)
    labels_detail = next((detail for detail in rank3 if detail["shape"][-1] != 4), None)
if boxes_detail is None or labels_detail is None:
    raise ValueError(f"Could not identify the dets/labels TFLite outputs; got {output_details}")

boxes = interpreter.get_tensor(boxes_detail["index"])
labels = interpreter.get_tensor(labels_detail["index"])
```
