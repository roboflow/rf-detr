---
description: Run inference with exported RF-DETR ONNX models using ONNX Runtime.
---

# ONNX Inference

The exported graph returns **raw** tensors — `dets` (`pred_boxes`, normalized `cxcywh`) and `labels` (`pred_logits`, un-activated). Nothing is decoded inside the graph, so your inference code must apply sigmoid, exclude the checkpoint's background slot when it has one, and convert box format yourself.

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
