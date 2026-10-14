---
description: Load RF-DETR models and run image prediction through one native and exported inference API.
---

# Run Inference

`RFDETRInference` gives native models, checkpoints, and supported full exports one image-prediction API. It owns input conversion, preprocessing, one runtime call per image batch, result decoding, and Supervision outputs.

![Diagram showing inference source loading and ownership, followed by per-call image preprocessing, one runtime execution, shared decoding, and Supervision results.](../assets/inference-data-flow.svg)

## Choose a source

Pass a live model, a native checkpoint path, or a full exported artifact path. A live model is borrowed: the facade does not copy it, and later predictions read its current weights and optimization state. Configure its device on the model and omit `device`, `metadata`, and `runtime_options` for this source. This borrowing does not promise safety when training mutates the model concurrently with prediction.

```python
from rfdetr import RFDETRInference, RFDETRSmall

native_model = RFDETRSmall()
predictor = RFDETRInference(native_model)
```

A `.pth`, `.pt`, or `.ckpt` path loads and owns a native model; an exported path loads and owns its runtime. Checkpoint loading may require architecture information available from the checkpoint name or supplied to `RFDETR.from_checkpoint()`. For a CPU-safe checkpoint recipe, load explicitly on CPU, optimize without compilation, then wrap the live model:

```python
from rfdetr import RFDETR, RFDETRInference

native_model = RFDETR.from_checkpoint("checkpoint.pth", device="cpu")
native_model.inference(compile=False)
predictor = RFDETRInference(native_model)
results = predictor.predict("image.jpg")
```

For export files, install the runtime extra for the target format and keep any `.rfdetr.json` sidecar beside its artifact. ONNX stores inference metadata in the model; other exported formats use the sidecar. Older exports can receive missing inference semantics through `metadata=` as a JSON path or mapping. Sidecar metadata is checked against the artifact. `notes` do not replace task, classes, output, or preprocessing metadata. See [Export Basics](../exports/basics.md#predict-with-rfdetrinference) and its format guides for runtime installation and options.

Backbone-only artifacts contain features rather than predictions and cannot be loaded by this image-prediction API. LiteRT does not support full keypoint exports. The constructor accepts a `trust_checkpoint` option for checkpoints containing custom Python objects; enable it only for a checkpoint you trust.

## Predict images

`predict()` accepts a file path (`str` or `os.PathLike[str]`), an HTTP(S) URL string, a PIL image, a NumPy array, a PyTorch tensor, or a list or tuple of these inputs. A list or tuple is one batch and returns one result per image. The runtime receives that batch in one execution; the API does not split or pad it. Each item must meet the source's channel, shape, and batch constraints.

PIL images and files are converted to the model's expected channels. For a one-channel model, a single-channel PIL image or file is converted to grayscale (`L`); for three-channel models, non-RGB PIL images are converted to RGB. NumPy arrays retain their supplied layout: use `(H, W)` or `(H, W, 1)` for one channel and `(H, W, 3)` for RGB. Tensor inputs use channel-first `(C, H, W)` and must already have the model's channel count. Floating-point arrays and all tensor inputs must have pixel values in `[0, 1]`; PIL images and `uint8` NumPy arrays use `[0, 255]`. Tensors are image pixels before mean/std normalization, not already-preprocessed runtime batches.

With `include_source_image=True` (the default), predictions retain a writable `uint8` source image: single-channel PIL/file inputs converted to `L` use `(H, W)`, NumPy arrays keep `(H, W)` or `(H, W, 1)`, and single-channel tensor inputs become `(H, W, 1)` after conversion from channel-first layout. RGB source images use `(H, W, 3)`. `source_image` is stored in `Detections.metadata`; `KeyPoints` stores it per object in `data`. Convert grayscale to RGB explicitly before passing it to annotation code that expects three-channel color:

```python
import numpy as np
import supervision as sv
from PIL import Image

gray_predictor = RFDETRInference("output/grayscale.onnx")  # A one-channel detector.
with Image.open("gray.png") as image:
    gray = np.array(image.convert("L"))
detections = gray_predictor.predict(gray)
rgb = np.repeat(gray[..., None], 3, axis=2)
annotated = sv.BoxAnnotator().annotate(scene=rgb, detections=detections)
```

The output is `supervision.Detections` for detection and segmentation, or `supervision.KeyPoints` for keypoints. `class_names` returns the class names for the source. Prediction does not promise a particular accuracy or latency; those depend on the model, runtime, hardware, and settings.

## Inspect runtime constraints

`runtime_info` is a fresh dictionary describing the selected backend, device policy, task, and shape/batch constraints. The constructor's `device` defaults to `"auto"`; for checkpoint paths `"auto"` leaves device selection to native checkpoint loading, while an explicit device is forwarded. Export paths accept the device values supported by their runtime. An unsupported device is refused. The device field reports a policy or selected runtime device; vendor runtimes may choose a physical device internally. See the [export format guides](../exports/index.md) for format-specific device values.

| Field               | Meaning                                                    |
| ------------------- | ---------------------------------------------------------- |
| `backend`, `device` | Runtime and its device policy or selected device.          |
| `task`              | `detect`, `segment`, or `keypoints`.                       |
| `input_shape`       | Canonical NCHW shape; `None` marks a dynamic axis.         |
| `default_shape`     | Default image height and width as `(H, W)`.                |
| `batch_size`        | Exact required batch size, or `None` when not fixed.       |
| `max_batch_size`    | Batch upper bound, or `None` when unbounded by the source. |

Eager native models accept dynamic spatial dimensions and batch sizes. An optimized native model fixes its spatial dimensions; it reports an exact `batch_size` only after compilation. Exported artifacts report their own fixed or dynamic axes and batch limits. A fixed batch requires exactly that many images; a maximum permits smaller batches down to one. Invalid shapes or batches fail before runtime execution.

By default, preprocessing resizes to `default_shape`. Pass `shape=(height, width)` to override it where the source allows. Each dimension must be positive and divisible by `patch_size * num_windows`; fixed-shape artifacts require their exported dimensions. `patch_size` is an advanced validation override, not a way to change the backbone: it must be a positive integer matching the model's configured patch size. Omit it to use the configured value.

```python
from pathlib import Path

predictor = RFDETRInference("output/model.onnx")
print(predictor.class_names)
print(predictor.runtime_info)
results = predictor.predict(Path("image.jpg"), threshold=0.4)
```

The public prediction keywords are `threshold`, `shape`, `patch_size`, `include_source_image`, and `antialias`. Unknown keywords raise `TypeError` before model execution. Constructor-only settings are `device`, `metadata`, `trust_checkpoint`, and `runtime_options`; runtime options are export-format-specific and reject unknown keys. See the [runtime options table](../exports/basics.md#predict-with-rfdetrinference) and the [format guides](../exports/index.md).
