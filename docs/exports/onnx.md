---
description: Run inference with exported RF-DETR ONNX models using ONNX Runtime.
---

# ONNX Inference

For the common prediction API, load the ONNX file with `RFDETRInference`. The exported graph itself returns raw tensors; the advanced ONNX Runtime example below shows how to run and decode those tensors directly.

## Predict with RFDETRInference

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/inference_model.onnx")
detections = model.predict("image.jpg", threshold=0.5)
```

This returns `supervision.Detections` for detection and segmentation models, or `supervision.KeyPoints` for keypoint models. See [Predict with RFDETRInference](basics.md#predict-with-rfdetrinference) for checkpoint inputs, metadata, batch behavior, and runtime settings.

## Advanced: Run ONNX Runtime directly

The exported graph returns **raw** tensors — `dets` (`pred_boxes`, normalized `cxcywh`) and `labels` (`pred_logits`, un-activated). The code below shows direct ONNX Runtime execution and NumPy decoding for low-level inspection. The private `_run_inference` helper remains a dependency-light numerical reference and shares raw execution with the `RFDETRInference` adapter; use `RFDETRInference` for normal image prediction.

!!! warning "Match outputs by name, not by shape"

    RF-DETR allocates `num_classes + 1` logit slots. If `num_classes == 3`, that dimension is `4` — identical to the box tensor's last dimension (`4`, `cxcywh`). Disambiguating outputs by shape instead of by name (`"dets"` / `"labels"`) will silently swap boxes and logits at exactly `num_classes == 3`, producing garbage detections while every other `num_classes` value looks fine. Always match by name first.

!!! warning "Choose the background slot from the checkpoint layout"

    The tensor width does not identify the background slot, and the layout depends on how categories were mapped during training, not simply on whether the checkpoint is fine-tuned. Checkpoints trained with contiguous 0-based category IDs — the common case for custom/Roboflow datasets — and active-first keypoint checkpoints use the final slot (index `-1`) as background. Checkpoints trained directly on sparse COCO category IDs — including the official pretrained weights — retain every slot with `background_class_id=None`, since a real foreground category (90 for official COCO) occupies the final slot. Legacy background-first keypoint checkpoints use slot `0`. The ONNX and TFLite `_run_inference` reference helpers expose this choice explicitly and default to `-1` for backward compatibility.

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
raw_logits = outputs[logits_idx][0]

# Select this from the checkpoint layout. Use None for official sparse-ID COCO
# checkpoints, -1 for contiguous-ID/active-first checkpoints, or 0 for legacy
# background-first keypoint checkpoints.
background_class_id = -1
class_slots = np.arange(raw_logits.shape[-1])
if background_class_id is None:
    logits = raw_logits
else:
    num_slots = raw_logits.shape[-1]
    if not -num_slots <= background_class_id < num_slots:
        raise ValueError(f"background_class_id must index one of {num_slots} exported class slots")
    background_class_id %= num_slots
    foreground_mask = class_slots != background_class_id
    logits = raw_logits[:, foreground_mask]
    class_slots = class_slots[foreground_mask]

# RF-DETR uses per-class sigmoid (multi-label), not softmax. This compact example keeps
# one top class per query; the reference decoders instead rank query/class pairs globally,
# so they can retain multiple above-threshold classes for one query.
scores_all = 1.0 / (1.0 + np.exp(-logits.clip(-88, 88)))
scores = scores_all.max(axis=-1)
class_ids = class_slots[scores_all.argmax(axis=-1)]

threshold = 0.5
keep = scores > threshold

# cxcywh (normalized) -> xyxy (pixel space)
cx, cy, bw, bh = boxes_cwh[keep].T
xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
xyxy *= np.array([image.width, image.height, image.width, image.height], dtype=np.float32)

boxes, labels, confidences = xyxy, class_ids[keep], scores[keep]
```

For a fuller reference implementation (name-based matching with a documented shape-based fallback), see `_run_inference` in [`src/rfdetr/export/_onnx/inference.py`](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/export/_onnx/inference.py).

## INT8 Quantization

`quantization="int8"` writes a static INT8 model — 8-bit weights *and* activations — beside the FP32 graph it was derived from:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
model.export(format="onnx", quantization="int8", calibration_data="calibration_images/")
# output/inference_model.onnx        (FP32, the source)
# output/inference_model_int8.onnx   (INT8, ~3x smaller)
```

`calibration_data` is **required** and must be representative of your deployment domain. Static quantization reads activation ranges from that data; out-of-domain or synthetic images produce a model that loads, runs, and is quietly wrong. Accepted forms are a directory of images (preprocessed exactly as `predict()` does, capped by `max_images`), a `.npy` file of shape `(N, C, H, W)` already normalized, or an equivalent array. A few dozen to a few hundred images from your validation split is the normal choice.

!!! warning "What gets quantized, and why it is not everything"

    Only `MatMul` and `Gemm` run in 8-bit. Attention-score multiplies, the detection heads, normalization, softmax and the surrounding elementwise math stay in float, and those ops' *outputs* are not quantized either.

    This is not conservatism. Measured on a 500-image COCO val2017 subset, ONNX Runtime's default op coverage — which quantizes `LayerNormalization`, `Mul`, `Div`, `Reshape` and every shape op as well — costs **9.65 mAP on Nano and 12.14 on Small**. Restricting to the matrix multiplies brings that back to **2.65 and 2.67**. It is also *faster*: the Q/DQ conversions around non-matmul ops cost more than the 8-bit kernels save, so quantizing less wins on both axes.

!!! note "Expect a measurable accuracy cost"

    On the same subset, INT8 cost 2.65 mAP (Nano) and 2.67 (Small) against each model's own FP32 baseline, for roughly 3x smaller files and ~1.4x faster CPU inference at batch 1. That is a real trade, not a rounding error — **measure on your own data and task before deploying**. Published INT8 RF-DETR models built with other tooling land closer to 0.5 mAP, so this gap is a property of the current configuration rather than of the architecture.

Quantization runs on CPU via ONNX Runtime, which the `rfdetr[onnx]` extra already installs.
