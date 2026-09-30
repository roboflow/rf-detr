---
description: Export RF-DETR models to OpenVINO IR for optimized inference on Intel CPUs, GPUs, and NPUs.
---

# OpenVINO IR Export

OpenVINO IR (Intermediate Representation) is a proprietary model format used by the OpenVINO Toolkit to optimize and deploy deep learning models.

## Prerequisites

```bash
pip install "rfdetr[openvino]"
```

## Basic OpenVINO Export

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="openvino", output_dir="output")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export(format="openvino", output_dir="output")
    ```

This produces two files (named after the model's variant):

- `output/<model-variant>.xml` - The model structure (Intermediate Representation)
- `output/<model-variant>.bin` - The model weights

## OpenVINO Export with Custom Resolution

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="openvino", shape=(608, 608))
```

## OpenVINO Export with Precision

OpenVINO export defaults to FP16 weight compression. Pass `openvino_precision="float32"` to keep the stored IR weights at full precision (larger file, no compression) — this controls IR *storage* precision only. The precision the device *computes* in is a runtime setting, chosen when the model is compiled (see the warning under [OpenVINO Inference Example](#openvino-inference-example)):

```python
model.export(format="openvino", openvino_precision="float32")
```

## Predict with RFDETRInference

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/rfdetr-medium.xml")
detections = model.predict("image.jpg", threshold=0.5)
```

This returns the shared Supervision result types. See [Predict with RFDETRInference](basics.md#predict-with-rfdetrinference) for checkpoint inputs, metadata, and batch behavior.

<span id="openvino-inference-example"></span>

## Advanced: Use the OpenVINO session directly

`OpenVINOInference` remains available for callers who need a raw OpenVINO session. It takes already-preprocessed NCHW tensors and returns raw output tensors. The common `RFDETRInference` API above handles preprocessing and decoding.

!!! warning "The input array must be float32 and contiguous"

    `infer()` validates this at the boundary and raises `ValueError` if violated, but a resize step that diverges from `predict()`'s own preprocessing (e.g. PIL's default `Image.resize()`, which resamples with bicubic) will still silently produce different — not obviously wrong — detections. Use `torchvision.transforms.functional.resize(..., antialias=False)` as below to match `predict()`'s antialias-free bilinear resize exactly.

```python
import torchvision.transforms.functional as F
from PIL import Image
from rfdetr.export.inference import OpenVINOInference

# Load the exported model; device is "AUTO", "CPU", "GPU" or "NPU"
model = OpenVINOInference("output/rfdetr-medium.xml", device="AUTO")

# Prepare input image (NCHW format, ImageNet normalized) — matches predict()'s own preprocessing
image = Image.open("image.jpg").convert("RGB")
image_tensor = F.to_tensor(image)
image_tensor = F.resize(image_tensor, [576, 576], antialias=False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_tensor = F.normalize(image_tensor, mean, std)

# Convert to NCHW format
image_array = image_tensor.unsqueeze(0).numpy()

# Run inference
outputs = model(image_array)
boxes, labels = outputs  # boxes: normalized cxcywh (center_x, center_y, width, height), not xywh
```

!!! warning "Execution precision: `OpenVINOInference` computes in float32 by default"

    OpenVINO's CPU plugin picks its own execution precision, whatever precision the IR was stored in, and on hardware with native half-precision support it can compute in f16 or bf16. In f16 RF-DETR's class logits collapse: on an Apple M4 Max, `RFDETRNano` went from 3 detections above 0.5 to none. The same collapse in bf16 on x86 is expected but has not been measured. `OpenVINOInference` therefore sets `INFERENCE_PRECISION_HINT` to `"f32"` unless told otherwise (on the NPU it sends no hint, because that plugin does not support f32; an explicit `inference_precision` is always passed through), which keeps the confident logits close to ONNX Runtime's (the CPU cookbook's `parity` check reports the actual drift). OpenVINO's own `EXECUTION_MODE_HINT=ACCURACY` is another way to ask for full precision. Pass `inference_precision=None` to keep the device default (faster on hardware that computes natively in reduced precision, at that accuracy cost), or another hint such as `"f16"` to set it explicitly. Further compile properties, such as `{"INFERENCE_NUM_THREADS": 4}`, go in `config`. Calling `openvino.Core().compile_model(...)` yourself leaves the device default in place, so pass `{"INFERENCE_PRECISION_HINT": "f32"}` there too.

!!! tip "Construct once, and use one instance per worker thread"

    Building an `OpenVINOInference` compiles the model, which is the expensive step — do it once and reuse the instance for every image. Pass `cache_dir="<dir>"` to reuse compiled kernels across process starts as well. Calls through a single instance are serialized by an internal lock, so sharing one instance across threads is safe but not faster; for parallel throughput give each worker thread its own instance.

## Benchmark OpenVINO Model

Use OpenVINO's `benchmark_app` tool to measure performance. Pass `-infer_precision f32` so it times the same execution precision `OpenVINOInference` runs by default; without it, `benchmark_app` times the device's reduced-precision default:

```bash
benchmark_app -m output/rfdetr-medium.xml -data_shape [1,3,576,576] -infer_precision f32
```

## OpenVINO Model Outputs

The exported OpenVINO IR model produces the following outputs:

- **Object Detection Models**:

    - Output 0: Bounding boxes `[batch, 300, 4]` — normalized `cxcywh` (center_x, center_y, width, height), not top-left `xywh`
    - Output 1: Class logits `[batch, 300, num_classes]`

- **Segmentation Models**:

    - Output 0: Bounding boxes `[batch, 300, 4]`
    - Output 1: Class logits `[batch, 300, num_classes]`
    - Output 2: Instance masks (if segmentation head is present)

- **Keypoint Models**:

    - Output 0: Bounding boxes `[batch, 300, 4]`
    - Output 1: Class logits `[batch, 300, num_classes]`
    - Output 2: Keypoints (if keypoint head is present)
