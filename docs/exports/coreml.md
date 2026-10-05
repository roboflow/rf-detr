---
description: Export RF-DETR models directly to native CoreML (`.mlpackage`) for Xcode / Apple-platform deployment.
---

# Native CoreML Export (`.mlpackage`)

!!! warning "Experimental — Use with Caution"

    Native CoreML export is **experimental and work-in-progress**. `dynamic_batch=True` is not supported — fixed shapes are required for reliable ANE / GPU scheduling. Export one `.mlpackage` per batch size instead.

!!! note "Not the same as the ExecuTorch CoreML backend"

    `format="coreml"` exports directly via `torch.export` + `coremltools` to a native `.mlpackage` (mlprogram, iOS 15+) — no ONNX and no ExecuTorch runtime involved. This is distinct from [`format="executorch", backend="coreml"`](executorch.md#coreml-backend-apple-neural-engine-fp16), which produces a `.pte` file for the ExecuTorch runtime. Passing both `format="coreml"` and `backend="coreml"` together does not fall through to the ExecuTorch delegate — `backend` is ignored (with a warning) and the native `.mlpackage` path always runs.

RF-DETR's native CoreML export produces a `.mlpackage` you can drag directly into Xcode, with no ONNX intermediary and no ExecuTorch runtime dependency — the lowest-friction path for Apple-native (iOS / macOS) developers.

## Prerequisites

```bash
pip install "rfdetr[coreml]"
```

## Basic CoreML Export

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

## Compute Precision

CoreML export defaults to `FLOAT32` for tight CPU parity with eager PyTorch. Pass `coreml_precision="float16"` for a bundle half the size that Core ML can schedule onto the Neural Engine — this also changes the output filename to `output/rfdetr-medium_fp16.mlpackage`:

```python
model.export(format="coreml", coreml_precision="float16")
```

!!! note

    Output tensor names in the saved `.mlpackage` spec are coremltools-inferred, not renamed to `dets`/`labels`/etc. — match outputs by **position**, in the same order as the ONNX `output_names` contract (`dets, labels` for detection; `dets, labels, masks` for segmentation; `dets, labels, keypoints` for keypoints).

!!! note "Raw tensors can differ more than the precision suggests"

    RF-DETR's two-stage encoder picks its queries with a `topk` over the encoder tokens' class scores. CoreML's fp32 arithmetic differs from eager PyTorch's in the last bits — measured at up to 1e-5 on a ranking score — so when two neighbouring scores sit closer together than that, the two can rank them in opposite order and run the decoder on a slightly different set of queries. Every raw output then shifts, by ~1e0 on logits, on an export that otherwise tracks eager to ~1e-4. This repo's own parity tests require a 1e-4 gap between neighbouring top-k scores before they compare raw tensors at all.

    This is a property of the ranking, not a conversion error, and it is not specific to `format="coreml"`: any runtime whose arithmetic differs from eager in the last bits can trip it, and fp16 — the ExecuTorch CoreML delegate, or `coreml_precision="float16"` — makes it more likely, not less. Detections comfortably above a confidence threshold survive it: on one pretrained `RFDETRSmall` image at threshold 0.5 (Apple M3 Pro, coremltools 9.0, default `ComputeUnit.ALL`) the two agree on every detection, within 1.1e-4 on scores and 0.005 px on boxes. A detection sitting *on* the threshold can still cross it, since a swap was measured to move post-processed scores by up to 1.2e-3. Compare **post-processed detections**, not raw tensors, when validating an export.

## CoreML Inference Example

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

# The input name is coremltools-inferred (currently "tensors"), so read it from the spec
# rather than hard-coding it.
input_name = mlmodel.get_spec().description.input[0].name

# Outputs are positional (see the precision note above) — dets, labels, in that order.
predictions = mlmodel.predict({input_name: image_array.astype(np.float32)})
outputs = [predictions[output.name] for output in mlmodel.get_spec().description.output]
boxes, labels = outputs[0], outputs[1]
```

## Neural Engine, Compute Units, and the Fallback Boundary

Core ML decides at **load time** which of the CPU, GPU and Apple Neural Engine (ANE) runs each part of the model. That choice is not stored in the `.mlpackage` — `MLModel` defaults to `ComputeUnit.ALL` every time it is loaded — so it is the caller's to make:

=== "Python"

    ```python
    import coremltools as ct

    mlmodel = ct.models.MLModel(
        "output/rfdetr-small_fp16.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    ```

=== "Swift"

    ```swift
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    let model = try MLModel(contentsOf: url, configuration: configuration)
    ```

**Precision decides whether the ANE is reachable at all.** The ANE has no fp32 path, so an fp32 bundle never touches it — `CPU_AND_NE` then measures the same as `CPU_ONLY`. Only `coreml_precision="float16"` puts the model on the ANE.

Single-image latency, pretrained `RFDETRNano` at 384x384, Apple M3 Pro (macOS 27.0.1, coremltools 9.0), batch 1, p50 in ms over 3 runs of 100 iterations after 10 warm-ups, 200 ms between timed passes:

| Precision | `CPU_ONLY` | `CPU_AND_NE` | `CPU_AND_GPU` | `ALL`    |
| --------- | ---------- | ------------ | ------------- | -------- |
| fp32      | 75.1       | 76.3         | **26.1**      | 27.0     |
| fp16      | 50.7       | 24.9         | 25.5          | **22.7** |

So: **fp32 belongs on the GPU, fp16 on the ANE or `ALL`.** Loading an fp16 bundle with `CPU_AND_NE` is about 2.0x faster than CPU and about as fast as the GPU on this machine (back to back, with no sleep between passes: 12.7 ms on the ANE, 12.1 ms on the GPU). Single-image latency is therefore no longer a reason to prefer the ANE over the GPU.

The ANE pays for that at load: compiling an fp16 RFDETRNano for it takes about 5 s on first load, against about 0.5 s for the GPU or CPU path (subsequent loads of the same bundle are cached by the system). For a process that runs a handful of images and exits, the GPU is the better trade.

**fp16 keeps the accuracy.** On COCO val2017 (all 5000 images, pretrained `RFDETRNano`, same decoding for both, Apple M3 Pro):

| Precision                   | mAP@[.5:.95] | mAP@.5 |
| --------------------------- | ------------ | ------ |
| eager PyTorch fp32          | 48.0         | 67.1   |
| Core ML fp32, `ALL`         | 48.0         | 67.1   |
| Core ML fp16, `CPU_AND_NE`  | 48.0         | 67.1   |
| Core ML fp16, `CPU_AND_GPU` | 48.0         | 67.1   |

The fp16 bundle is an iOS 15 program on purpose. An iOS 16 program runs `resample` on the Neural Engine in fp16, and that moved mAP down by about 3 points (45.1) while the same bundle was fine on the GPU or CPU; as an iOS 15 program that op stays off the ANE. Validate an fp16 bundle on your own data before shipping it.

**The fallback boundary, at fp16.** An fp16 RF-DETR graph is almost entirely ANE-eligible. The exceptions are the two-stage query selection — `topk`, and the `expand_dims`/`tile`/`gather_along_axis` that consume its indices — plus the deformable-attention sampling (`resample`) with the `cast` ops beside it, which Core ML runs on the CPU. Measured with `MLComputePlan` under `CPU_AND_NE`:

| Model           | Ops on ANE | Ops on CPU | Share of estimated work on the ANE |
| --------------- | ---------- | ---------- | ---------------------------------- |
| `RFDETRNano`    | 591        | 11         | 99.7%                              |
| `RFDETRSmall`   | 648        | 13         | 99.7%                              |
| `RFDETRMedium`  | 705        | 15         | 99.7%                              |
| `RFDETRSegNano` | 725        | 15         | 99.9%                              |

Those eleven to fifteen ops are the whole boundary, and they cost 0.1% to 0.3% of the model's estimated work.

At fp32 there is no boundary to speak of, because there is no ANE: the plan reports *every* op as ANE-unsupported, and the same RFDETRNano bundle runs all 597 ops on the GPU under `ALL` and all 597 on the CPU under `CPU_AND_NE`.

**`ALL` is not always the fastest choice.** With `ALL`, Core ML is free to put part of the graph on the GPU, and for `RFDETRSegNano` it does: the plan splits 79% ANE / 21% GPU, and the transfers between them cost real time — 44.2 ms under `ALL` against 38.8 ms under `CPU_AND_NE` (p50, same methodology as the table above). Detection models are unaffected; their plan is the same under both. Measure both on your target device rather than assuming the default is best.

The [ExecuTorch CoreML delegate](executorch.md#coreml-backend-apple-neural-engine-fp16) lowers RF-DETR to a single Core ML model inside the `.pte`, and that model does reach the Neural Engine. Its internal split is not measurable with `MLComputePlan`, which needs an `.mlpackage`, so the table above is not a statement about the `.pte`.
