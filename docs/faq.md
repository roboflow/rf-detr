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

## How do I load my own weights or a checkpoint from a specific path?

Pass `pretrain_weights=` when you build any model — it accepts a fine-tuned checkpoint, a pretrained backbone, or one of the published names:

```python
from rfdetr import RFDETRSmall

# A checkpoint anywhere on disk (absolute or relative path)
model = RFDETRSmall(pretrain_weights="/data/runs/exp1/checkpoint_best_total.pth")

# A published checkpoint by name — downloaded into the cache dir if missing
model = RFDETRSmall(pretrain_weights="rf-detr-small.pth")
```

Resolution rules:

- **Bare filename** (no directory, e.g. `rf-detr-small.pth`) → resolved to the model cache directory (see below) and downloaded there if not already present.
- **Path with a directory** (e.g. `./my.pth`, `/abs/my.pth`, `~/models/my.pth`) → used as-is.
- **`None`** → train from randomly initialized weights (no pretrained checkpoint).

## Where does RF-DETR store downloaded weights, and how do I change it?

Published checkpoints are cached in `~/.roboflow/models` by default. Override the location for all models by setting an environment variable — `RF_HOME` (canonical) or its alias `ROBOFLOW_HOME`:

```bash
export RF_HOME=/mnt/shared/models
# or, equivalently, use the alias instead:
export ROBOFLOW_HOME=/mnt/shared/models
```

If both are set, `RF_HOME` wins. This only affects where **bare-filename** weights are cached; an explicit path in `pretrain_weights=` is always honored as given.

## What input resolutions are allowed?

`resolution` must be a positive integer divisible by `patch_size × num_windows` for the selected variant (for example, current detection checkpoints use a block size of 32). A non-divisible value raises `ValueError` indicating the required divisor. Input is square; each variant ships a sensible default resolution.

## Does my effective batch size have to be 16?

No. 16 is a reasonable nominal target, not a requirement. The nominal effective batch is `batch_size × grad_accum_steps × num_gpus`.

Note that the defaults no longer produce 16: `grad_accum_steps` defaults to `1` (it was `4` in earlier versions), so `batch_size=4` alone gives a nominal effective batch of 4. If you were relying on the old defaults and want the previous behavior, set `grad_accum_steps=4` explicitly.

## Should I raise `batch_size` or `grad_accum_steps`?

Raise `batch_size` first, and only reach for `grad_accum_steps` when memory stops you.

Both increase the nominal effective batch, but they are not equivalent in cost or necessarily in optimization behavior. If hardware memory prevents the physical batch from reaching your target, increase `grad_accum_steps` to recover the nominal optimizer-step window. Gradient accumulation splits that window across smaller forward/backward passes, changes microbatch cadence, and a small physical batch tends to leave the GPU under-occupied. On one L4 training `rfdetr-small`, `batch_size=16, grad_accum_steps=1` ran about 27% faster per epoch than `batch_size=4, grad_accum_steps=4` at the same nominal effective batch of 16, with mAP equal within run-to-run noise. That is a single GPU and a single dataset, so take the direction rather than the exact number.

So: use `batch_size="auto"` on CUDA when the model, task, resolution, or available memory makes manual sizing uncertain. On CPU or MPS, use a concrete integer batch size. If you set it manually, increase the physical batch conservatively and use `grad_accum_steps` only when memory requires it. Keeping `batch_size × grad_accum_steps` constant preserves only the nominal effective-batch target; it does not guarantee the same optimization trajectory, even though the nominal images-per-optimizer-update target is unchanged. See [Training Parameters](learn/train/training-parameters.md).

## What export formats are supported?

ONNX, TFLite (FP32/FP16/INT8), ExecuTorch (XNNPACK, CoreML, QNN), and native CoreML (`.mlpackage`). TensorRT runs via the exported ONNX model. Install the matching extra — `rfdetr[onnx]`, `rfdetr[tflite]`, `rfdetr[executorch]`, or `rfdetr[coreml]` — then call `model.export()`. Native CoreML (`format="coreml"`) is distinct from the ExecuTorch CoreML backend (`format="executorch", backend="coreml"`) — the former produces a `.mlpackage` directly, the latter a `.pte`. See [Export Model](learn/export.md).

## How do I evaluate a trained model and get mAP?

Call `model.evaluate(split="test")` (or `split="val"`). It returns a dictionary of metrics including mAP. See the [Custom Training API](learn/train/customization.md).
