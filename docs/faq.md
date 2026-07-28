---
description: Frequently asked questions about RF-DETR — object count limits, model selection, class configuration, input resolution, batch size, export formats, and evaluation.
---

# Frequently Asked Questions

Answers to questions that come up often in [GitHub issues](https://github.com/roboflow/rf-detr/issues).

## How many objects can RF-DETR detect in one image?

By default up to `num_queries` distinct objects — 300 for Nano, Small, Medium, and Large. To go higher, raise both `num_queries` and `num_select` when you build the model:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(num_queries=600, num_select=600)
```

- `num_queries` — decoder object slots; the hard cap on distinct objects.
- `num_select` — predictions kept by post-processing. It must track `num_queries`, otherwise the output is capped at the lower value. Raising `num_select` alone returns duplicate boxes (same object, different label), not new objects.

Raising `num_queries` above a variant's default adds new rows to the query embeddings, so RF-DETR warns that part of the checkpoint will not load and those slots start untrained. **Fine-tune after raising `num_queries`.** More queries also cost more compute (decoder attention scales with the square of the query count), so keep the default unless your scenes really are that crowded.

## Which model size should I use?

Pick by the accuracy/latency trade-off in the [pre-trained checkpoints table](learn/run/detection.md#pre-trained-checkpoints): `RFDETRNano` through `RFDETR2XLarge`. `RFDETRSmall` is a good default. The older `RFDETRBase` is legacy — use `RFDETRSmall` instead.

## Do I need to set `num_classes` when fine-tuning or loading a checkpoint?

No. `RFDETR.from_checkpoint(...)` infers `num_classes` from the checkpoint's classification head shape, and training infers it from your dataset's annotations. Pass an explicit `num_classes=N` only when you need to pin it.

## Can I keep the COCO classes while training on my own classes?

No. Fine-tuning on a custom dataset retrains the classification head for that dataset's classes; the pretrained COCO classes are not retained. To detect both your classes and some COCO classes, include those COCO classes in your training data.

## What input resolutions are allowed?

`resolution` must be a positive integer divisible by `patch_size × num_windows` for the selected variant (for example, current detection checkpoints use a block size of 32). A non-divisible value raises `ValueError` indicating the required divisor. Input is square; each variant ships a sensible default resolution.

## Does my effective batch size have to be 16?

It is the recommended target, not a hard requirement. Effective batch size is `batch_size × grad_accum_steps × num_gpus`. On smaller GPUs, keep `batch_size` low and raise `grad_accum_steps` to reach the same effective size — for example `batch_size=4, grad_accum_steps=4` on a T4. See [Training Parameters](learn/train/training-parameters.md).

## What export formats are supported?

ONNX, TFLite (FP32/FP16/INT8), ExecuTorch (XNNPACK, CoreML, QNN), and native CoreML (`.mlpackage`). TensorRT runs via the exported ONNX model. Install the matching extra — `rfdetr[onnx]`, `rfdetr[tflite]`, `rfdetr[executorch]`, or `rfdetr[coreml]` — then call `model.export()`. Native CoreML (`format="coreml"`) is distinct from the ExecuTorch CoreML backend (`format="executorch", backend="coreml"`) — the former produces a `.mlpackage` directly, the latter a `.pte`. See [Export Model](learn/export.md).

## How do I evaluate a trained model and get mAP?

Call `model.evaluate(split="test")` (or `split="val"`). It returns a dictionary of metrics including mAP. See the [Custom Training API](learn/train/customization.md).
