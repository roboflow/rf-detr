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

Pick by the accuracy/latency trade-off in the [pre-trained checkpoints table](learn/run/detection.md#pre-trained-checkpoints): `RFDETRAtto` through `RFDETR2XLarge`. `RFDETRSmall` is a good default. The older `RFDETRBase` is legacy — use `RFDETRSmall` instead. Nano to Large are Apache 2.0 and commercially usable out of the box; Atto, Femto, Pico, XLarge and 2XLarge need `rfdetr[plus]` (Atto, Femto and Pico need `rfdetr_plus>=1.2.0`) and use PML 1.0 (commercial use included on Roboflow; commercial self-hosting requires the Enterprise add-on). See [licensing](#can-i-use-rf-detr-in-a-commercial-product).

## Can I use RF-DETR in a commercial product?

Yes.

--8<-- "docs/snippets/license-rfdetr.md"

Coming from YOLO? The license terms differ:

--8<-- "docs/snippets/license-yolo-note.md"

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

Published checkpoints download from `https://repo.roboflow.com/rfdetr`. On a network that only allows listed hosts, allow `repo.roboflow.com`, or copy the files into the cache directory ahead of time: a cached file whose MD5 matches is never downloaded again.

## What input resolutions are allowed?

`resolution` must be a positive integer divisible by `patch_size × num_windows` for the selected variant (for example, current detection checkpoints use a block size of 32). A non-divisible value raises `ValueError` indicating the required divisor. Input is square; each variant ships a sensible default resolution.

## Does my effective batch size have to be 16?

No. 16 is a reasonable nominal target, not a requirement. The nominal effective batch is `batch_size × grad_accum_steps × num_gpus`.

Note that the defaults no longer produce 16: `grad_accum_steps` defaults to `1` (it was `4` in earlier versions), so `batch_size=4` alone gives a nominal effective batch of 4. If you were relying on the old defaults and want the previous behavior, set `grad_accum_steps=4` explicitly.

## Should I raise `batch_size` or `grad_accum_steps`?

Raise `batch_size` first, and only reach for `grad_accum_steps` when memory stops you.

Both increase the nominal effective batch, but they are not equivalent in cost or necessarily in optimization behavior. If hardware memory prevents the physical batch from reaching your target, increase `grad_accum_steps` to recover the nominal optimizer-step window. Gradient accumulation splits that window across smaller forward/backward passes, changes microbatch cadence, and a small physical batch tends to leave the GPU under-occupied. On one L4 training `rfdetr-small`, `batch_size=16, grad_accum_steps=1` ran about 27% faster per epoch than `batch_size=4, grad_accum_steps=4` at the same nominal effective batch of 16, with mAP equal within run-to-run noise. That is a single GPU and a single dataset, so take the direction rather than the exact number.

So: use `batch_size="auto"` on CUDA when the model, task, resolution, or available memory makes manual sizing uncertain. On CPU or MPS, use a concrete integer batch size. If you set it manually, increase the physical batch conservatively and use `grad_accum_steps` only when memory requires it. Keeping `batch_size × grad_accum_steps` constant preserves only the nominal effective-batch target; it does not guarantee the same optimization trajectory, even though the nominal images-per-optimizer-update target is unchanged. See [Training Parameters](learn/train/training-parameters.md).

## Does FP8 training work on Apple MPS, and do I need the CUDA extra?

FP8 training uses NVIDIA Transformer Engine and requires supported NVIDIA CUDA hardware. It does not run on Apple MPS. The `cuda` extra adds the PyTorch extension on Linux x86-64; ordinary CUDA and MPS training only need `rfdetr[train]`. FP8 remains opt-in through `amp_dtype="fp8"`.

See [Advanced FP8 setup](learn/train/advanced.md#fp8-training-on-nvidia-cuda) for installation, CUDA-major compatibility, and fixes for empty-metapackage, skipped-layer, and EMA extra-state messages.

## What export formats are supported?

ONNX, TFLite (FP32/FP16/INT8), LiteRT (`.tflite` straight from PyTorch via litert-torch), TensorRT (`.trt`), ExecuTorch (XNNPACK, CoreML, QNN), native CoreML (`.mlpackage`), and OpenVINO IR. Install the matching extra — `rfdetr[onnx]`, `rfdetr[tflite]`, `rfdetr[litert]`, `rfdetr[tensorrt]`, `rfdetr[executorch]`, `rfdetr[coreml]`, or `rfdetr[openvino]` — then call `model.export(format=...)`. `format="tensorrt"` (alias `"trt"`) exports ONNX and builds the engine in-process, so it is one call rather than a separate conversion step. Native CoreML (`format="coreml"`) is distinct from the ExecuTorch CoreML backend (`format="executorch", backend="coreml"`) — the former produces a `.mlpackage` directly, the latter a `.pte`. See [Export Model](exports/index.md).

## Why is an inductor predictor slower after I built several models in one process?

`model.inference(compile_backend="inductor")` (the default on Apple Silicon) compiles through `torch.compile`, and PyTorch Dynamo stops compiling a function after `torch._dynamo.config.recompile_limit` compiled variants (also named `cache_size_limit`, default 8). Every RF-DETR size runs the same `forward`, so each inductor predictor built in one process, per size, dtype and resolution, uses one variant. After the eighth, Dynamo logs `torch._dynamo hit config.recompile_limit (8)` and later predictors run several times slower, with no exception.

`RFDETR.inference()` now holds the limit at 64 (or your higher value) while it compiles and restores it afterwards, so building predictors through it is not affected. It also compiles with `dynamic=False`: with the limit lifted but dynamic shapes left on, Femto in fp16 failed to compile on Apple Silicon (`no 'buffer' resource location available`) after several other sizes had been built in the same process, and with `dynamic=False` the same sequence compiled. You can still hit the limit when you call `torch.compile` on RF-DETR modules yourself, or on an installed version without that change. If timings depend on the order in which you build predictors, look for the warning in the log, then raise the limit before building them, or build each predictor in its own process:

```python
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
```

On an Apple M4 Max, Atto with inductor in fp16 took 12.20 ms when built after Nano, Small, Medium and Large in fp32 and fp16 (eight variants), against 4.61 ms in the same process with the limit raised. With the limit set to 1, the second model took 10.6 to 12.8 ms instead of about 4.6 to 6.5 ms. Whether Dynamo runs the function eagerly once the limit is hit was not checked directly.

## How do I evaluate a trained model and get mAP?

Call `model.evaluate(split="test")` (or `split="val"`). It returns a dictionary of metrics including mAP. See the [Custom Training API](learn/train/customization.md).
