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

Measured on a pretrained `RFDETRNano` (Apple M3 Pro, macOS 27.0, COCO val2017, all 5000 images), this delegate keeps more accuracy than a native fp16 `.mlpackage`: 48.0 mAP against 45.1, at 14.0 ms against 20.8 ms. Both run their arithmetic in fp16, but the native export also stores the weights in fp16, and that is what costs the accuracy.

!!! note

    CoreML export uses fp16 arithmetic. Top-level detections (bounding boxes and class labels) are correct, but raw tensor values will differ from the PyTorch fp32 baseline — at the fp16 precision level, and through the two-stage query ranking described under [Native CoreML Export](coreml.md#neural-engine-compute-units-and-the-fallback-boundary), which fp16 makes more likely to diverge rather than less. For what fp16 costs in mAP, and for how Core ML splits the model across the ANE, GPU and CPU, see [Neural Engine, compute units, and the fallback boundary](coreml.md#neural-engine-compute-units-and-the-fallback-boundary).

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

## ExecuTorch Inference Example

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
