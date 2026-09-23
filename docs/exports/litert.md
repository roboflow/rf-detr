---
description: Export RF-DETR models to LiteRT (`.tflite`) straight from PyTorch using litert-torch.
---

# LiteRT Export

!!! warning "Experimental — Use with Caution"

    LiteRT export is **experimental**. `litert-torch` is pre-1.0 and its converter changes between releases; the `[litert]` extra pins the range this route was validated on (0.9.4).

    **Known limitations:**

    - Float32 only: `quantization` other than `None` / `"fp32"` raises `NotImplementedError` on this route (use `format="tflite"` for its FP16/INT8 modes, or quantize the exported file with [ai-edge-quantizer](https://github.com/google-ai-edge/ai-edge-quantizer)).
    - `dynamic_batch=True` is not supported: the `.tflite` bakes a fixed input shape, so export one file per batch size.
    - Keypoint models are not supported on litert-torch 0.9.4: its converter rejects the rank-4 `batch_matmul` that the keypoint head's `nn.Linear` lowers to.
    - The single exported graph includes the two-stage query selection (`TOPK_V2` / `GATHER_ND`), which the LiteRT GPU delegate has no kernels for, so the file runs on the CPU (XNNPACK) delegate. Running the detector on a phone GPU needs further graph rewrites and a two-graph split that this route does not do yet.

LiteRT (formerly TensorFlow Lite) is Google's on-device runtime. `format="litert"` hands the PyTorch model to [litert-torch](https://github.com/google-ai-edge/litert-torch), which captures it with `torch.export` and lowers it to a `.tflite` file directly — no ONNX and no TensorFlow step, unlike the [TFLite export](tflite.md) which converts ONNX → TensorFlow → TFLite with `onnx2tf`. Both routes produce a `.tflite` that the same `ai_edge_litert` interpreter runs; this one keeps PyTorch's NCHW layout and the deformable-attention sampling as litert-torch lowers it. On CPU the exported graphs track eager PyTorch closely. Measured with the pretrained Nano and Seg-Nano checkpoints on a real photo, the ten highest-confidence queries differ by at most about `1e-7` for boxes and `3e-5` for class logits and mask probabilities; across all 300 queries the maxima rise to about `6e-6` (boxes), `4e-4` (class logits) and `2e-3` (raw mask logits, about `3e-5` after sigmoid), because low-confidence proposals reorder slightly between backends. The `e2e_litert` test suite asserts looser bounds on the confident queries (boxes `1e-3`, logits `0.1`, mask probabilities `0.05`) as a regression gate; those bounds are not the measured precision.

## Prerequisites

```bash
pip install "rfdetr[litert]"
```

## Basic LiteRT Export

=== "Object Detection"

    ```python
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="litert", output_dir="output")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegSmall

    model = RFDETRSegSmall(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="litert", output_dir="output")
    ```

This writes one float32 file named after the model's variant, `output/<model-variant>.tflite` (for example `output/rfdetr-small.tflite`; `-backbone` is appended with `backbone_only=True`, and `output_name` overrides the stem). `shape=(H, W)` picks a custom resolution exactly as for the other formats.

## LiteRT Inference Example

The file has one input (NCHW float32, ImageNet-normalized like `predict()`) and positional outputs: boxes `[batch, 300, 4]` in normalized `cxcywh`, class logits `[batch, 300, num_classes]`, and — for segmentation models — mask logits as a third output. Output tensor names are litert-torch's own (`serving_default_output_<i>_output`), so match outputs by position, as for the CoreML and OpenVINO exports.

```python
import numpy as np
import torchvision.transforms.functional as F
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

interpreter = Interpreter(model_path="output/rfdetr-small.tflite")
interpreter.allocate_tensors()
(input_detail,) = interpreter.get_input_details()
_, _, height, width = input_detail["shape"]

# Same preprocessing as predict(): antialias-free bilinear resize, then ImageNet normalization
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [height, width], antialias=False)
image_tensor = F.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

interpreter.set_tensor(input_detail["index"], image_tensor.unsqueeze(0).numpy())
interpreter.invoke()
boxes, logits = (interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()[:2])
scores = 1 / (1 + np.exp(-logits))  # sigmoid; boxes are normalized cxcywh
```
