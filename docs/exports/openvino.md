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

OpenVINO export defaults to FP16 weight compression. Pass `openvino_precision="float32"` to keep the stored IR weights at full precision (larger file, no compression) — this controls IR *storage* precision only. The precision the device *computes* in is a runtime setting, chosen when the model is compiled (see the warning under [Advanced: Use the OpenVINO session directly](#advanced-use-the-openvino-session-directly)):

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

`runtime_options` takes `cache_dir`, `inference_precision` and `config`, with the meaning they have on the [OpenVINO session](#advanced-use-the-openvino-session-directly) below: reuse compiled kernels across process starts, set the precision the device computes in (`"f32"` by default, except on the NPU), and pass further compile properties.

```python
model = RFDETRInference(
    "output/rfdetr-medium.xml",
    device="cpu",
    runtime_options={"cache_dir": "output/openvino-cache", "config": {"INFERENCE_NUM_THREADS": 4}},
)
```

<span id="openvino-inference-example"></span>

## Advanced: Use the OpenVINO session directly

!!! warning "OpenVINOInference is deprecated"

    Constructing `OpenVINOInference` emits `FutureWarning`: it is deprecated since v1.12.0 and will be removed in v1.15.0. Use `RFDETRInference(model_path).predict(image)` for image inputs and Supervision results, and its `runtime_options` for `cache_dir`, `inference_precision` and `config`. The old class remains available for compatibility until then.

The compatibility class takes preprocessed NCHW tensors and returns raw outputs. It shares session loading and execution functions with the common adapter. The adapter calls these functions directly and does not construct the deprecated class.

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

!!! warning "Execution precision: OpenVINO computes in float32 by default through RF-DETR"

    OpenVINO's CPU plugin picks its own execution precision, whatever precision the IR was stored in, and on hardware with native half-precision support it can compute in f16 or bf16. In f16 RF-DETR's class logits collapse: on an Apple M4 Max, `RFDETRNano` went from 3 detections above 0.5 to none. The same collapse in bf16 on x86 is expected but has not been measured. `RFDETRInference` and the deprecated `OpenVINOInference` therefore set `INFERENCE_PRECISION_HINT` to `"f32"` unless told otherwise (on the NPU it sends no hint, because that plugin does not support f32; an explicit `inference_precision` is always passed through), which keeps the confident logits close to ONNX Runtime's (the CPU cookbook's `parity` check reports the actual drift). OpenVINO's own `EXECUTION_MODE_HINT=ACCURACY` is another way to ask for full precision. For `RFDETRInference`, pass `runtime_options={"inference_precision": None}` to keep the device default (faster on hardware that computes natively in reduced precision, at that accuracy cost), or another hint such as `"f16"` to set it explicitly. Further compile properties, such as `{"INFERENCE_NUM_THREADS": 4}`, go in `runtime_options["config"]`. The deprecated class accepts these settings as direct constructor keywords. Calling `openvino.Core().compile_model(...)` yourself leaves the device default in place, so pass `{"INFERENCE_PRECISION_HINT": "f32"}` there too.

!!! tip "Construct once, and use one instance per worker thread"

    Building an `OpenVINOInference` compiles the model, which is the expensive step — do it once and reuse the instance for every image. Pass `cache_dir="<dir>"` to reuse compiled kernels across process starts as well. Calls through a single instance are serialized by an internal lock, so sharing one instance across threads is safe but not faster; for parallel throughput give each worker thread its own instance.

## Benchmark OpenVINO Model

Use OpenVINO's `benchmark_app` tool to measure performance. Pass `-infer_precision f32` so it times the same execution precision `RFDETRInference` and the deprecated `OpenVINOInference` use by default; without it, `benchmark_app` times the device's reduced-precision default:

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

## INT8 Quantization

`quantization="int8"` compresses the IR with [NNCF](https://github.com/openvinotoolkit/nncf) before it is written, so the `.xml` / `.bin` pair on disk is already 8-bit:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
model.export(format="openvino", quantization="int8", calibration_data="calibration_images/")
```

NNCF is a separate install — `pip install nncf` — rather than part of the `rfdetr[openvino]` extra, since only this mode needs it and it carries a heavyweight dependency set an FP32 or FP16 export has no use for.

`calibration_data` is **required**: static quantization reads activation ranges from it, so it must be representative of your deployment domain. It accepts a directory of images (preprocessed exactly as `predict()` does, capped by `max_images`), a `.npy` file of shape `(N, C, H, W)` already normalized, or an equivalent array.

!!! note "`quantization` and `openvino_precision` are different knobs"

    `openvino_precision` controls how IR weights are *stored* (FP32 vs FP16 compression) and leaves arithmetic alone. `quantization="int8"` changes the arithmetic. Set them independently.

!!! warning "The transformer recipe matters"

    The export asks NNCF for `ModelType.TRANSFORMER` rather than its generic defaults. Measured on a 500-image COCO val2017 subset, plain `nncf.quantize(model, dataset)` cost **5.34 mAP on Nano and 6.18 on Small** — worse than a comparable ONNX Runtime configuration — because it quantizes normalization and the elementwise math around attention along with the matrix multiplies. The transformer recipe keeps those paths in float. Measure on your own data before deploying.

On the Colab x86 host of the [measured table below](#measured-latency-on-an-x86-cpu), the INT8 IR cost 0.4 mAP against the fp32 IR (56.1 against 56.5) and ran at the same speed (274.2 against 274.6 ms at batch 1), for a file 3.6 times smaller (32.3 against 115.0 MB).

## Measured Latency on an x86 CPU

Colab x86-64 host with 4 physical cores (8 logical), Linux 6.6, `RFDETRSmall`, `rfdetr` 1.12.0.dev0, ONNX Runtime 1.31.1, OpenVINO 2026.4.1, NNCF 3.4.0, 15 warmup + 50 timed runs per row, from the [CPU cookbook](../../cookbooks/export-cpu/). Every latency is the end-to-end `predict()` call (preprocessing, forward, postprocessing) through `RFDETRInference`; OpenVINO is pinned to the physical-core count. `FPS` is images per second. `mAP@50:95` is a seeded 500-image COCO val2017 subset, scored on the same batch-1 artifact that was timed; the INT8 artifacts were calibrated on 100 other val2017 images held out of that subset. The INT8 artifacts are batch 1 only.

| Runtime | Precision | Quantization | Batch | end2end [ms]    | FPS [img/s] | mAP@50:95 [%] | Size [MB] | Memory [MB] |
| ------- | --------- | ------------ | ----- | --------------- | ----------- | ------------- | --------- | ----------- |
| CPU     | fp32      | —            | 1     | 274.63 ± 23.23  | 3.6         | 56.5          | 115.0     | 317.7       |
| CPU     | fp16      | —            | 1     | 291.75 ± 25.47  | 3.4         | 56.5          | 58.2      | 212.9       |
| CPU     | INT8      | static NNCF  | 1     | 274.17 ± 28.65  | 3.6         | 56.1          | 32.3      | 131.2       |
| CPU     | fp32      | —            | 4     | 1045.49 ± 65.74 | 3.8         | —             | 115.0     | 637.5       |
| CPU     | fp16      | —            | 4     | 1037.53 ± 81.16 | 3.9         | —             | 58.2      | 552.4       |

What the table supports:

- The fp32 IR is 1.26 times faster than eager PyTorch on the same host (274.6 against 345.4 ms at batch 1, a gap of about three standard deviations; see the [PyTorch baseline](index.md#pytorch-baseline)) and 1.38 times faster than [ONNX Runtime fp32](onnx.md#on-an-x86-cpu) (377.9 ms).
- The fp16 IR (291.8 ms) is no faster than the fp32 IR (274.6 ms), within one standard deviation, and half the size. It saves disk and bandwidth, not arithmetic.
- The INT8 IR runs at the same speed as the fp32 IR, costs 0.4 mAP, and is 3.6 times smaller.
- Batching helps little on this CPU: 3.6 FPS at batch 1, 3.8 at batch 4 for fp32.
- Standard deviations are 4 to 10% of the mean, so gaps under about 25 ms at batch 1 are noise, not a ranking.

!!! note "Memory column"

    `Memory [MB]` is host resident-memory growth across constructing the runtime plus its first call, in one shared process. Read it as a rough per-row indication.
