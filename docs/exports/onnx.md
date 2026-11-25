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

On the CPU provider `RFDETRInference` runs ONNX Runtime on every logical core the process may use, where ONNX Runtime alone would use only the physical cores; pass `runtime_options={"num_threads": N}` to choose another pool size.

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

Pass `int8_keep_mlp_float=True` to also keep GELU-fed weighted projections in float. This includes the second layer of backbone MLPs and can include projections in segmentation heads. GELU outputs can have outlier channels that determine the 8-bit step for the entire tensor. The option defaults to `False` and requires `quantization="int8"` for ONNX.

The default favors speed, and the option favors accuracy. It keeps more multiplies in float, so its export runs slower than the default one. Use it when the default INT8 export loses too much accuracy on your data. The larger models are the most likely to need it. The [CPU cookbook](../../cookbooks/export-cpu/) exports, times and scores both placements for one model.

```python
model.export(format="onnx", quantization="int8", calibration_data="calibration_images/", int8_keep_mlp_float=True)
```

!!! warning "What gets quantized, and why it is not everything"

    Only `MatMul` and `Gemm` run in 8-bit. Attention-score and attention-value multiplies, the detection heads, normalization, softmax and the surrounding elementwise math stay in float, and those ops' *outputs* are not quantized either.

    This is not conservatism. Measured on a 500-image COCO val2017 subset, ONNX Runtime's default op coverage — which quantizes `LayerNormalization`, `Mul`, `Div`, `Reshape` and every shape op as well — costs **9.65 mAP on Nano and 12.14 on Small**. Restricting to the matrix multiplies brings most of that back (the current cost on Small is in the note below). It is also *faster*: the Q/DQ conversions around non-matmul ops cost more than the 8-bit kernels save, so quantizing less wins on both axes.

!!! note "Expect a measurable accuracy cost"

    In the [x86 CPU run](#on-an-x86-cpu) of the CPU cookbook, the default INT8 export of `RFDETRSmall` cost 0.8 mAP against its FP32 baseline (55.7 against 56.5 on a 500-image val2017 subset) and `int8_keep_mlp_float=True` cost 0.4 (56.1), for files 3.3 and 2.1 times smaller and 1.37 and 1.15 times faster inference at batch 1. That is a real trade, not a rounding error — **measure on your own data and task before deploying**.

Quantization runs on CPU via ONNX Runtime, which the `rfdetr[onnx]` extra already installs.

The latency and accuracy of both INT8 placements against the fp32 model on an x86 CPU are in [On an x86 CPU](#on-an-x86-cpu) below.

## Measured Latency

Every latency is the end-to-end `predict()` call (preprocessing, forward, postprocessing) through `RFDETRInference`, `RFDETRSmall`. `FPS` is images per second. `mAP@50:95` is a seeded 500-image COCO val2017 subset, scored at batch 1 on the same artifact that was timed. The PyTorch baseline of each machine is on the [export overview](index.md#pytorch-baseline).

### On an NVIDIA L4

Colab NVIDIA L4, `rfdetr` 1.12.0.dev0, CUDA execution provider, 20 warmup + 100 timed runs per row, from the [CUDA cookbook](../../cookbooks/export-cuda/). On this provider `RFDETRInference` prepares each batch on the GPU and hands it to ONNX Runtime through IO binding, so the input never returns to the host.

| Runtime | Precision | Batch | end2end [ms] | FPS [img/s] | mAP@50:95 [%] | Size [MB] |
| ------- | --------- | ----- | ------------ | ----------- | ------------- | --------- |
| CUDA EP | fp32      | 1     | 8.4          | 119.2       | 56.5          | 127.5     |
| CUDA EP | fp32      | 4     | 30.1         | 132.9       | —             | 127.5     |
| CUDA EP | fp32      | 8     | 66.4         | 120.5       | —             | 127.5     |

ONNX Runtime on CUDA scales little with batch: batch 8 is slower per image than batch 4 (120.5 against 132.9 img/s). The batch 4 and 8 latencies are batch size divided by the measured img/s, rounded to one decimal. On this GPU a [TensorRT engine](tensorrt.md#latency-across-nvidia-gpus) is the faster route.

### On an x86 CPU

Colab x86-64 host, AMD EPYC 7B12 with 4 physical cores (8 logical), Linux 6.6, `RFDETRSmall`, `rfdetr` 1.13.0.dev0 from the `develop` archive (commit not recorded), PyTorch version not recorded, ONNX Runtime 1.22.1, run on 2026-11-24, 15 warmup + 50 timed runs per row, from the [CPU cookbook](../../cookbooks/export-cpu/). ONNX Runtime is pinned to the 4 physical cores. The INT8 models were calibrated on 100 val2017 images held out of the scored subset and are batch 1 only.

| Runtime | Precision | Quantization          | Batch | end2end [ms]   | FPS [img/s] | mAP@50:95 [%] | Size [MB] |
| ------- | --------- | --------------------- | ----- | -------------- | ----------- | ------------- | --------- |
| CPU EP  | fp32      | —                     | 1     | 335.9 ± 26.5   | 3.0         | 56.5          | 127.5     |
| CPU EP  | fp32      | —                     | 4     | 1059.7 ± 106.8 | 3.8         | —             | 127.5     |
| CPU EP  | INT8      | static QDQ            | 1     | 245.9 ± 21.5   | 4.1         | 55.7          | 39.1      |
| CPU EP  | INT8      | static QDQ, MLP float | 1     | 292.7 ± 21.6   | 3.4         | 56.1          | 60.3      |

`static QDQ` is the default INT8 export and `static QDQ, MLP float` is `int8_keep_mlp_float=True`. In this run:

- The default INT8 export takes 73% of the fp32 time and costs 0.8 mAP. The precise placement takes 87% of it and costs 0.4 mAP, so on `RFDETRSmall` it gives back half of the accuracy loss for a 19% slower call than the default.
- The default INT8 export runs at the speed of the [OpenVINO fp32 IR](openvino.md#measured-latency-on-an-x86-cpu) (246.8 ms) on the same host, which keeps the full accuracy.
- `RFDETRInference`'s ONNX path took 335.9 ms against 314.8 ± 62.4 ms for the same model through `inference-models` on the same ONNX Runtime, a 7% gap inside that package's noise. An earlier run had measured a 39% gap that was never explained.

The [latency cookbook](../../cookbooks/measure-latency/) times ONNX Runtime FP32 and INT8 on a Colab x86 CPU (AMD EPYC 7B12, 8 logical cores, ONNX Runtime 1.32.0, `rfdetr` 1.13.0.dev0) for every pretrained size, as the median of 200 images at batch 1, with the thread count set to the 8 logical cores through the `num_threads` runtime option. The tables are on the Run pages for [detection](../learn/run/detection.md#fastest-configuration-on-an-x86-cpu), [segmentation](../learn/run/segmentation.md#fastest-configuration-on-an-x86-cpu) and [keypoints](../learn/run/keypoints.md#fastest-configuration-on-an-x86-cpu), next to PyTorch TorchScript and OpenVINO. In them:

- ONNX INT8 takes 64% to 80% of the ONNX FP32 time at every size it was timed for.
- ONNX INT8 is among the fastest of the six configurations for 10 of the 16 models (the three PE sizes have no INT8 configuration, and Nano, Seg Nano and Seg Small are faster in another configuration). ONNX FP32 is never among them: it is slower than PyTorch TorchScript FP32 at every size.
- ONNX compact, timed for Atto, Femto and Pico only, takes 70% to 73% of the ONNX FP32 time, tied with PyTorch TorchScript FP32 on all three and with OpenVINO FP32 on Femto and Pico.
- No accuracy was scored for the latency cookbook's rows; the INT8 accuracy cost on `RFDETRSmall` is in the table above.

On Intel CPUs, [OpenVINO](openvino.md#measured-latency-on-an-x86-cpu) is also a measured route.

### Compact PE models on CPU

Install `pip install "rfdetr-plus[onnx]"` to export Atto, Femto or Pico with ONNX Runtime's CPU operators. Export a released detector or reload a fine-tuned checkpoint to retain its trained weights, class head and resolution.

```python
from rfdetr import RFDETRInference, from_checkpoint
from rfdetr_plus import RFDETRFemto

pretrained = RFDETRFemto(device="cpu")
pretrained.export_cpu_onnx("output/pretrained")

trained = from_checkpoint("checkpoint_best_total.pth", device="cpu")
artifact = trained.export_cpu_onnx("output/finetuned")
predictions = RFDETRInference(artifact, device="cpu").predict("image.jpg")
```

The compact ONNX file takes one normalized NCHW float image and returns normalized center-format boxes (`dets`) and class logits (`labels`). The prediction API handles preprocessing and postprocessing. Compare AP on your held-out images and latency on the deployment CPU against the standard FP32 export before choosing a format.

To enforce an 8 MB storage budget, pass `size_limit_bytes=8_000_000` to `export_cpu_onnx`. The complete uncompressed artifact must be strictly smaller. The released COCO checkpoints produced 7.13 MB for Atto, 11.77 MB for Femto and 13.10 MB for Pico in the [CPU export qualification](https://github.com/roboflow/rf-detr_plus/pull/83). A fine-tuned model can have a different artifact size.
