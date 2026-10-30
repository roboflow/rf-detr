---
description: Export RF-DETR models to ExecuTorch (XNNPACK, CoreML, QNN) for on-device PyTorch inference.
---

# ExecuTorch Export

!!! warning "Experimental — Use with Caution"

    ExecuTorch export is **experimental**. The `executorch` package is under active development and its installation and API are subject to breaking changes between releases.

    **Known limitations:**

    - `dynamic_batch=True` is not supported: the runtime cannot resize RF-DETR's windowed-attention reshapes, so export one `.pte` per batch size instead.
    - The `"qnn"` backend requires a **source build** of ExecuTorch against the QAIRT SDK and cannot be installed via `pip`.
    - CoreML export runs in fp16; confident top-level detections carry over but raw tensor values will differ from the PyTorch fp32 model, both as expected for fp16 computation and through the query-ranking effect described under [Native CoreML Export](coreml.md#neural-engine-compute-units-and-the-fallback-boundary).

ExecuTorch is PyTorch's on-device inference runtime. Unlike ONNX export, the model is exported directly via `torch.export` to a portable `.pte` binary — no intermediate ONNX conversion step is involved.

## Prerequisites

```bash
pip install "rfdetr[executorch]"
```

## XNNPACK Backend (Portable CPU, fp32)

The `"xnnpack"` backend targets any CPU platform and runs in fp32. It is the recommended, portable backend and requires only the standard `rfdetr[executorch]` wheel. `backend` has no default — it must always be passed explicitly for `format="executorch"`.

On this backend the export rewrites attention without a mask as two batched matrix multiplications and a softmax, and computes the query, key and value slices of the decoder's packed `in_proj_weight` at export time. XNNPACK then runs the encoder and decoder attention, which `F.scaled_dot_product_attention` and `nn.MultiheadAttention` otherwise split into portable kernels between many small XNNPACK partitions. The outputs are the same to float32 rounding, and the median CPU latency is 33–54% lower (1.5–2.2x faster) in the measurements of the changelog entry for this change (ExecuTorch 1.5.1, 16 threads, Xeon and Apple M4 Max).

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="executorch", backend="xnnpack")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="executorch", backend="xnnpack")
    ```

This produces `output/rfdetr-seg-medium_xnnpack.pte` — the file is named after the model variant plus the backend (`{variant}_{backend}.pte`, or `{variant}_qnn_{soc}.pte` for the SoC-locked `qnn` backend), not a generic `inference_model_{backend}.pte`. The backend is always encoded because it determines which hardware/runtime can load the file.

## CoreML Backend (Apple Neural Engine, fp16)

!!! note "Not the same as native CoreML export"

    This is the ExecuTorch delegate — `format="executorch", backend="coreml"` — which produces a `.pte` file for the ExecuTorch runtime. It is distinct from `format="coreml"`, which produces a native `.mlpackage` directly (no ExecuTorch runtime involved); see [Native CoreML Export](coreml.md).

The `"coreml"` backend targets Apple devices (iPhone, iPad, Mac) and runs in fp16 on the Neural Engine. It requires `coremltools`, which is **not** included in the `rfdetr[executorch]` extra — install it separately:

```bash
pip install coremltools
```

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="executorch", backend="coreml")
```

RF-DETR's graph lowers to a **single** CoreML delegate — no operator is left behind on ExecuTorch's portable CPU kernels — so the `.pte` carries one Core ML model, and the Neural Engine is reachable through it. The compute units are not baked in; the app that loads the `.pte` chooses them, exactly as for a native `.mlpackage`.

Measured on a pretrained `RFDETRNano` (Apple M3 Pro, macOS 27.0.1, COCO val2017, all 5000 images), this delegate scores 48.0 mAP, the same as eager PyTorch and as a native fp16 `.mlpackage`. Both are iOS 15 (specification 6) Core ML programs holding fp16 weights, so the resample ops stay off the Neural Engine in each. The delegate does reach the Neural Engine, but its op split cannot be measured, because `MLComputePlan` needs an `.mlpackage` (see [Native CoreML Export](coreml.md#neural-engine-compute-units-and-the-fallback-boundary)).

!!! note

    CoreML export uses fp16 arithmetic. Top-level detections (bounding boxes and class labels) are correct, but raw tensor values will differ from the PyTorch fp32 baseline — at the fp16 precision level, and through the two-stage query ranking described under [Native CoreML Export](coreml.md#neural-engine-compute-units-and-the-fallback-boundary), which fp16 makes more likely to diverge rather than less. For how Core ML splits the model across the ANE, GPU and CPU, see [Neural Engine, compute units, and the fallback boundary](coreml.md#neural-engine-compute-units-and-the-fallback-boundary).

## QNN Backend (Qualcomm Snapdragon HTP, fp16)

The `"qnn"` backend targets the Qualcomm AI Engine (HTP) on Snapdragon SoCs and runs in fp16. It **requires a source build** of ExecuTorch against the QAIRT SDK and cannot be installed via `pip`.

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="executorch", backend="qnn", soc="SM8650")
```

The `soc` parameter is required for QNN and must be a `QcomChipset` name matching your target device. For example, `"SM8650"` targets the Snapdragon 8 Gen 3. This produces `output/rfdetr-medium_qnn_SM8650.pte` — the SoC is baked into the filename (not just the backend) since a QNN `.pte` is compiled ahead-of-time for one specific chip and will not run on another.

!!! warning

    QNN export is validated on-device but cannot be tested in CI (requires QAIRT SDK). Validate detections on your target Snapdragon device before deploying to production.

## ExecuTorch Limitations

- **`dynamic_batch=True` is not supported.** The ExecuTorch runtime cannot resize RF-DETR's windowed-attention reshapes for a variable batch size. Export one `.pte` file per batch size instead (e.g. `batch_size=1` for single-image inference).
- **QNN requires a source build.** The QNN backend is not available via the pip wheel; see the ExecuTorch documentation for source-build instructions against the QAIRT SDK.

## Measured Latency on a Laptop CPU

XNNPACK backend, Apple M4 Max CPU (arm64), `RFDETRSmall`, 5 warmup + 30 timed runs per row, `executorch==1.3.1`, one `.pte` per batch size, from the [mobile cookbook](../../cookbooks/export-mobile/). The PyTorch baseline of the same machine is on the [export overview](index.md#pytorch-baseline). For the CoreML backend see [below](#measured-latency-on-apple-silicon).

| Runtime | Precision | Quantization | Batch | forward [ms]  | end2end [ms]  | FPS [img/s] (end2end) | Memory [MB] |
| ------- | --------- | ------------ | ----- | ------------- | ------------- | --------------------- | ----------- |
| XNNPACK | fp32      | —            | 1     | 81.68 ± 3.31  | 80.74 ± 1.90  | 12.4                  | 140.2       |
| XNNPACK | fp32      | —            | 4     | 313.56 ± 3.24 | 318.27 ± 6.67 | 12.6                  | 1075.7      |

!!! warning "Earlier measurement, pending a rerun"

    These rows were measured before the cookbook pinned one thread count for every runtime. Treat each row as a same-machine ratio, never as a phone latency: a phone's own delegate (NNAPI, Hexagon, Core ML) is not exercised.

The XNNPACK program does not beat eager PyTorch on the machine that exported it (80.74 against 54.54 ms at batch 1): export formats exist for portability and on-device deployment without a Python runtime. The memory jump at batch 4 (1075.7 against 140.2 MB) is a full second program load in the same process, not a leak.

## Measured Latency on Apple Silicon

CoreML backend, Apple M4 Max, macOS 26.7.1, `RFDETRSmall`, 5 warmup + 30 timed runs per row, from the [Apple cookbook](../../cookbooks/export-apple/). Every latency is the end-to-end `predict()` call through `RFDETRInference`; `FPS` is images per second at that batch size. `mAP@50:95` is a seeded 500-image COCO val2017 subset, scored on the same batch-1 artifact that was timed. The backend exports fp16 only.

| Backend | Precision | b1 latency [ms] | Size [MB] | mAP@50:95 [%] | FPS b1 | FPS b4 | FPS b8 |
| ------- | --------- | --------------- | --------- | ------------- | ------ | ------ | ------ |
| CoreML  | fp16      | 24.4            | 58.2      | 56.5          | 41.0   | 49.1   | 83.0   |

It matches [native CoreML fp16](coreml.md#measured-latency-on-apple-silicon) at batch 1 (24.4 ms) and is slower than native CoreML fp32 (12.7 ms). Another job ran on the machine during this measurement, so the batch-8 deviation is large (± 16.3 ms).

## Predict with RFDETRInference

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/rfdetr-medium_xnnpack.pte")
detections = model.predict("image.jpg", threshold=0.5)
```

This loads the `.pte` with ExecuTorch and returns the shared Supervision result types. See [Predict with RFDETRInference](basics.md#predict-with-rfdetrinference) for checkpoint inputs, metadata, and batch behavior.

## Advanced: Use the ExecuTorch runtime directly

!!! warning "torch/executorch ABI compatibility"

    Loading a `.pte` via `executorch.runtime` (below) requires a `torch` version whose ABI matches the `executorch` wheel you installed — `.pte` **export** itself does not need `executorch.runtime` and is unaffected. For `executorch==1.3.1`, pin `torch<2.13` (`pip install "torch<2.13"`); a newer `torch` release can silently break `executorch.runtime` with an `undefined symbol` / `dlopen` error at import time, since ExecuTorch's prebuilt wheels are compiled against whichever `torch` ABI existed at their release time.

!!! warning "The input tensor must be contiguous"

    The ExecuTorch runtime reads the input buffer as contiguous NCHW and ignores tensor strides. Preprocessing steps that permute axes — `np.transpose`, `Tensor.permute`, torchvision's `ToImage` — return a strided view rather than a copy, and such a view is misread as a scrambled image. Nothing errors: the model runs without error and returns plausible-shaped output, but every detection's score collapses below threshold. Finish preprocessing with `np.ascontiguousarray(...)` (or `Tensor.contiguous()`) before calling `execute`.

```python
import torch
from executorch.runtime import Runtime
from PIL import Image
import torchvision.transforms.functional as F

# Load the exported .pte program
runtime = Runtime.get()
method = runtime.load_program("output/rfdetr-medium_xnnpack.pte").load_method("forward")

# Prepare input — the .pte expects the same NCHW, ImageNet-normalized input as the ONNX export
input_height, input_width = 576, 576
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [input_height, input_width], antialias=False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

image_array = image_tensor.unsqueeze(0).contiguous().numpy()  # add batch dimension: (1, 3, H, W)
input_tensor = torch.from_numpy(image_array).float()

# Run inference
outputs = method.execute([input_tensor])
boxes, labels = outputs[0], outputs[1]
```
