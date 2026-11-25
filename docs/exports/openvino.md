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

OpenVINO export defaults to FP16 weight compression. Pass `openvino_precision="float32"` to keep the stored IR weights at full precision (larger file, no compression) — this controls IR *storage* precision only. The precision the device *computes* in is a runtime setting, chosen when the model is compiled (see the warning under [Runtime Options](#runtime-options)):

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

`runtime_options` takes `cache_dir`, `inference_precision`, `config` and `num_threads` (the CPU thread count, the same key the ONNX, TFLite and LiteRT loaders read; it sets `INFERENCE_NUM_THREADS`, so giving both is an error), as described under [Runtime Options](#runtime-options) below: reuse compiled kernels across process starts, set the precision the device computes in (`"f32"` by default, except on the NPU), and pass further compile properties.

```python
model = RFDETRInference(
    "output/rfdetr-medium.xml",
    device="cpu",
    runtime_options={"cache_dir": "output/openvino-cache", "config": {"INFERENCE_NUM_THREADS": 4}},
)
```

## Runtime Options

!!! warning "Execution precision: OpenVINO computes in float32 by default through RF-DETR"

    OpenVINO's CPU plugin picks its own execution precision, whatever precision the IR was stored in, and on hardware with native half-precision support it can compute in f16 or bf16. In f16 RF-DETR's class logits collapse: on an Apple M4 Max, `RFDETRNano` went from 3 detections above 0.5 to none. The same collapse in bf16 on x86 is expected but has not been measured. `RFDETRInference` therefore sets `INFERENCE_PRECISION_HINT` to `"f32"` unless told otherwise (on the NPU it sends no hint, because that plugin does not support f32; an explicit `inference_precision` is always passed through), which keeps the confident logits close to ONNX Runtime's (the CPU cookbook's `parity` check reports the actual drift). OpenVINO's own `EXECUTION_MODE_HINT=ACCURACY` is another way to ask for full precision. Pass `runtime_options={"inference_precision": None}` to keep the device default (faster on hardware that computes natively in reduced precision, at that accuracy cost), or another hint such as `"f16"` to set it explicitly. Further compile properties, such as `{"INFERENCE_NUM_THREADS": 4}`, go in `runtime_options["config"]`. Calling `openvino.Core().compile_model(...)` yourself leaves the device default in place, so pass `{"INFERENCE_PRECISION_HINT": "f32"}` there too.

!!! tip "Construct once, and use one instance per worker thread"

    Building an `RFDETRInference` for an OpenVINO model compiles it, which is the expensive step — do it once and reuse the instance for every image. Pass `runtime_options={"cache_dir": "<dir>"}` to reuse compiled kernels across process starts as well. Calls through a single instance are serialized by an internal lock, so sharing one instance across threads is safe but not faster; for parallel throughput give each worker thread its own instance.

## Benchmark OpenVINO Model

Use OpenVINO's `benchmark_app` tool to measure performance. Pass `-infer_precision f32` so it times the same execution precision `RFDETRInference` uses by default; without it, `benchmark_app` times the device's reduced-precision default:

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

On the Colab x86 host of the [measured table below](#measured-latency-on-an-x86-cpu), the INT8 IR cost 0.4 mAP against the fp32 IR (56.1 against 56.5) and was no faster (260.7 ± 29.8 against 246.8 ± 17.3 ms at batch 1), for a file 3.6 times smaller (32.3 against 115.0 MB).

## Measured Latency on an x86 CPU

Colab x86-64 host, AMD EPYC 7B12 with 4 physical cores (8 logical), Linux 6.6, `RFDETRSmall`, `rfdetr` 1.13.0.dev0 from the `develop` archive (commit not recorded), PyTorch version not recorded, ONNX Runtime 1.22.1, OpenVINO 2026.4.1, NNCF 3.4.0, run on 2026-11-24, 15 warmup + 50 timed runs per row, from the [CPU cookbook](../../cookbooks/export-cpu/). Every latency is the end-to-end `predict()` call (preprocessing, forward, postprocessing) through `RFDETRInference`; OpenVINO is pinned to the physical-core count. `FPS` is images per second. `mAP@50:95` is a seeded 500-image COCO val2017 subset, scored on the same batch-1 artifact that was timed; the INT8 artifacts were calibrated on 100 other val2017 images held out of that subset. The INT8 artifacts are batch 1 only.

| Runtime | Precision | Quantization | Batch | end2end [ms] | FPS [img/s] | mAP@50:95 [%] | Size [MB] | Memory [MB] |
| ------- | --------- | ------------ | ----- | ------------ | ----------- | ------------- | --------- | ----------- |
| CPU     | fp32      | —            | 1     | 246.8 ± 17.3 | 4.1         | 56.5          | 115.0     | 318.3       |
| CPU     | fp16      | —            | 1     | 268.1 ± 25.3 | 3.7         | 56.5          | 58.2      | 213.8       |
| CPU     | INT8      | static NNCF  | 1     | 260.7 ± 29.8 | 3.8         | 56.1          | 32.3      | 140.4       |
| CPU     | fp32      | —            | 4     | 947.7 ± 58.2 | 4.2         | —             | 115.0     | 633.8       |
| CPU     | fp16      | —            | 4     | 981.3 ± 77.4 | 4.1         | —             | 58.2      | 525.9       |

What the table supports:

- On this host the fp32 IR (246.8 ms) is the fastest full-accuracy route at batch 1. It is ahead of eager PyTorch (329.0 ms), its TorchScript `inference()` snapshot (318.5 ms) and [ONNX Runtime fp32](onnx.md#on-an-x86-cpu) (335.9 ms) by more than three standard deviations (see the [PyTorch baseline](index.md#pytorch-baseline)).
- The fp16 IR (268.1 ms) is no faster than the fp32 IR, within about one standard deviation of it, and half the size. It saves disk and bandwidth, not arithmetic.
- The INT8 IR (260.7 ms) is no faster than the fp32 IR on this host, though 3.6 times smaller and within 0.4 mAP. An earlier run of this cookbook on another Colab host had it 1.45 times slower than the fp32 IR (542.1 against 373.7 ms), so INT8 speed on a CPU depends on the host and is worth measuring on the machine you deploy to.
- Batching helps little on this CPU: 4.1 FPS at batch 1, 4.2 at batch 4 for fp32.
- Predicting a folder of 64 images gave no more than batch 1: 4.0 img/s for the fp32 IR, 4.2 for fp16 and 4.2 for INT8. The model, not the input side, is the limit on a CPU.
- Compare rows within this table, never against numbers from another run: the earlier run on another Colab host timed eager PyTorch at 401.6 ms against 329.0 ms here.

### Every Size on the Run Pages

The [latency cookbook](../../cookbooks/measure-latency/) times OpenVINO FP32, FP16 and INT8 for every pretrained size on a Colab x86 CPU (AMD EPYC 7B12, 8 logical cores, OpenVINO 2026.4.1, NNCF 3.4.0, `rfdetr` 1.13.0.dev0), as the median of 200 images at batch 1 after 50 warm-up calls. Its tables are on the Run pages for [detection](../learn/run/detection.md#fastest-configuration-on-an-x86-cpu), [segmentation](../learn/run/segmentation.md#fastest-configuration-on-an-x86-cpu) and [keypoints](../learn/run/keypoints.md#fastest-configuration-on-an-x86-cpu). They use a different protocol from the table above: OpenVINO is asked for all 8 logical cores with hyper-threading enabled, where the table above pins it to the 4 physical cores, and it takes the median over 200 different images where the table above averages 50 calls on one image, so compare numbers only within one set. For `RFDETRSmall` the cookbook measured OpenVINO FP32 at 258.05 ms, against 246.8 ms in the table above. In the cookbook:

- OpenVINO is among the fastest of the six configurations only at mid sizes (detection Nano to Large, segmentation Small to XLarge) and is never alone there.
- OpenVINO INT8 takes 0.86 to 1.01 times the FP32 time for detection, 0.92 to 1.09 times for segmentation and 0.97 times for keypoints, and FP16 takes 0.95 to 1.25 times, 1.00 to 1.04 times and 1.03 times. INT8 speed on a CPU depends on the host, so measure it on yours.

!!! note "Memory column"

    `Memory [MB]` is host resident-memory growth across constructing the runtime plus its first call, in one shared process. Read it as a rough per-row indication.
