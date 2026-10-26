---
description: Export RF-DETR models to a TensorRT engine from PyTorch for low-latency inference on NVIDIA GPUs.
---

# TensorRT Export

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
>
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference, or build a [portable engine](#portable-engines).

## Prerequisites

- Install the TensorRT extra: `pip install rfdetr[tensorrt]` (provides `tensorrt`, `polygraphy`, `onnx`, `onnxconverter-common` and `onnxruntime-gpu`; `onnx` and `onnxconverter-common` cast the ONNX graph to FP16 on TensorRT 11+, and onnxruntime calibrates [INT8](#int8) engines; no `trtexec` binary needed)
- A CUDA GPU (by default the engine is built for the local GPU architecture)
- Export an ONNX model first (for example: `output/inference_model.onnx`)

## Export Directly to TensorRT

Pass `format="tensorrt"` to `export()` to export ONNX and convert to a TensorRT engine in one step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt")
```

This exports `output/inference_model.onnx` first and then produces `output/inference_model_fp16.trt` (the `_fp16`/`_fp32`/`_int8` suffix always reflects the precision actually built — see `fp16` in [Export Parameters](advanced.md#export-parameters) — unless `output_name` is set).

The ONNX file is an intermediate conversion input and has no RF-DETR inference metadata. The final engine receives its own metadata sidecar. To load the intermediate ONNX directly with `RFDETRInference`, pass the missing task and interface semantics through `metadata=`; see the [metadata notes](basics.md#predict-with-rfdetrinference).

!!! note "Dynamic batch"

    Pass `dynamic_batch=True` together with `max_batch_size` to build one engine that accepts any batch from 1 to `max_batch_size`. The engine gets a single TensorRT optimization profile with `min=1`, `opt=batch_size` and `max=max_batch_size`, so `batch_size` should be the batch you serve most often; other sizes inside the range run, TensorRT just tunes its kernels for `opt`. Without `dynamic_batch` the engine accepts only the batch size baked into the intermediate ONNX graph.

    ```python
    model.export(format="tensorrt", dynamic_batch=True, batch_size=4, max_batch_size=16)
    ```

    **Why a single profile with `min=1`, not several.** The engine always builds with one TensorRT optimization profile spanning the full `1 .. max_batch_size` range, rather than several narrower profiles picked at runtime with `set_optimization_profile_async`. This is a deliberate trade-off, not a limitation: it keeps the export API and the runtime simple (one engine, one profile, no profile-selection logic in the caller), and the measured cost at the tuned `opt` batch is small (see the [changelog](https://github.com/roboflow/rf-detr/blob/main/CHANGELOG.md) for per-GPU numbers). A deployment that never serves batches below some floor — for example a DeepStream or Triton pipeline always fed a fixed batch of frames (see [#376](https://github.com/roboflow/rf-detr/issues/376)) — pays for optimizing kernels down to batch 1 even though it never uses them, foreclosing per-batch-band multi-profile support (`set_optimization_profile_async` plus several `Profile()` entries), which is TensorRT's own standard mitigation for the away-from-opt penalty. A `min_batch_size` (paired with `max_batch_size`) or a list of `opt_batch_sizes` each with its own profile may become configurable in a future release if a narrow-band deployment need arises; today, export one profile spanning the batches you plan to serve.

## INT8

Pass `quantization="int8"` and a directory of representative images to build an engine that runs most of the backbone encoder and decoder in INT8 and the rest in FP16:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt", quantization="int8", calibration_data="path/to/images", batch_size=8)
```

This writes `output/rfdetr-small_int8.trt` (next to the intermediate `output/rfdetr-small.onnx`). The export runs the first `max_images` (default `100`) calibration images by file name through the FP32 graph with onnxruntime on the CPU, preprocessed exactly as `predict()` does, and records each quantized tensor's largest absolute value. Those ranges decide the engine's accuracy, so use images from the deployment domain: out-of-domain images give an engine that loads, runs and is quietly less accurate. `calibration_data` also accepts a `.npy` path or an array shaped `(N, C, H, W)`, already normalized the way `predict()` normalizes; an integer array, raw pixels, is refused. `max_images` caps only a directory of images; an array or `.npy` file is used whole, so pass as many samples as you want calibrated. Calibration runs the graph at the export's `batch_size`, so its host memory grows with the batch.

### Results

The benchmark of [Section 8.1 of the CUDA cookbook](../../cookbooks/export-cuda/#81-benchmark-for-the-docs-tables) ran on two GPUs, an RTX 5070 and a Tesla T4. The RTX 5070 rows and the Tesla T4 segmentation and keypoint rows used TensorRT 11.4.0.106; the Tesla T4 detection rows used TensorRT 10.16.1.11. Every row below is from that benchmark.

!!! note "The Tesla T4 detection rows use another TensorRT version"

    The Tesla T4 detection rows were re-measured on TensorRT 10.16.1.11, with the detection models' patch embedding in INT8 (see the [placement rules](#placement-rules)). Every other row, including the RTX 5070 detection rows, was measured on TensorRT 11.4.0.106. The segmentation and keypoint rows are unchanged on both GPUs. A difference between the Tesla T4 detection rows and the other Tesla T4 rows is not an effect of the patch embedding alone: the TensorRT version differs too.

**Summary.** INT8 is faster than FP16 in every row on both GPUs and loses less than 1 AP point on each model's own task. On the Tesla T4 the engines of the segmentation and keypoint models are larger than the FP16 ones at batch 4 and 8.

|                                 | RTX 5070              | Tesla T4              |
| ------------------------------- | --------------------- | --------------------- |
| Speedup, plain call             | 1.06× to 1.42×        | 1.02× to 1.30×        |
| Speedup, CUDA graph             | 1.06× to 1.43×        | 1.05× to 1.31×        |
| AP change at batch 1            | −0.47 to −0.91 points | −0.37 to −0.98 points |
| AP change, share of the FP16 AP | 0.9% to 2.0%          | 0.8% to 2.2%          |
| Engine size, INT8 ÷ FP16        | 0.53× to 0.79×        | 0.64× to 2.19×        |

#### How it was measured

- **Machines:** an RTX 5070 (sm_120, native Linux) and a Tesla T4 (sm_75, Colab). The RTX 5070 and the Tesla T4 segmentation and keypoint rows ran TensorRT 11.4.0.106; the Tesla T4 detection rows ran TensorRT 10.16.1.11, the version the Colab runtime installed.
- **Engines:** a static FP16 and a static INT8 engine of every model at batch 1, 4 and 8. The INT8 engines are calibrated on the first 128 `train2017` images by file name.
- **Accuracy:** batch 1, all 5000 `val2017` images, decoded the way `predict()` decodes. Box AP and mask AP come from `instances_val2017`; box AP (person class only) and keypoint AP from `person_keypoints_val2017`.
- **Latency:** GPU time per image (CUDA events), the median of 15 rounds of 200 executions, FP16 and INT8 interleaved. A *plain call* is 200 back-to-back `execute_async_v3` calls with no synchronization in between. A *CUDA graph* replays one captured call 200 times.
- **Reproduce it:** [Section 8.1 of the CUDA cookbook](../../cookbooks/export-cuda/#81-benchmark-for-the-docs-tables) runs these exact steps on your GPU. It is written for Colab, takes hours for all nine models, and prints the GPU and the TensorRT version next to its table, because the numbers depend on them. The tables above used TensorRT 11.4.0.106, except the Tesla T4 detection rows (10.16.1.11), so compare the version it prints before you compare numbers.

The RTX 5070 differences are computed from the unrounded values, and the Tesla T4 ones from the values its notebook printed (engine sizes in whole MB), so recomputing them from the rounded cells can differ in the last digit.

#### At a glance

Batch 1, all 5000 `val2017` images. Δ is INT8 minus FP16 in COCO AP points. Speedup is the FP16 latency divided by the INT8 latency, for a plain call. Each task has its own table, because each is scored on a different AP. The Tesla T4 tables have no box AP for the segmentation and keypoint models: that run kept only the AP of each model's own task.

=== "RTX 5070"

    **Detection (box AP)**

    | Model            | Box AP, FP16 | Box AP, INT8 | Δ     | Speedup |
    | ---------------- | ------------ | ------------ | ----- | ------- |
    | Detection Nano   | 48.03        | 47.32        | −0.70 | 1.35×   |
    | Detection Small  | 52.82        | 52.14        | −0.67 | 1.34×   |
    | Detection Medium | 54.64        | 53.73        | −0.91 | 1.26×   |
    | Detection Large  | 56.53        | 55.92        | −0.62 | 1.15×   |

    **Segmentation (mask AP)**

    | Model      | Mask AP, FP16 | Mask AP, INT8 | Δ     | Box AP, FP16 | Box AP, INT8 | Δ box | Speedup |
    | ---------- | ------------- | ------------- | ----- | ------------ | ------------ | ----- | ------- |
    | Seg Nano   | 40.09         | 39.59         | −0.50 | 48.42        | 47.57        | −0.85 | 1.06×   |
    | Seg Small  | 42.95         | 42.13         | −0.82 | 51.10        | 49.97        | −1.13 | 1.22×   |
    | Seg Medium | 44.94         | 44.03         | −0.91 | 53.42        | 52.19        | −1.22 | 1.17×   |
    | Seg Large  | 46.44         | 45.97         | −0.47 | 54.91        | 54.00        | −0.91 | 1.09×   |

    **Keypoints (keypoint AP)**

    | Model            | Keypoint AP, FP16 | Keypoint AP, INT8 | Δ     | Box AP, FP16 | Box AP, INT8 | Δ box | Speedup |
    | ---------------- | ----------------- | ----------------- | ----- | ------------ | ------------ | ----- | ------- |
    | Keypoint Preview | 71.78             | 71.11             | −0.67 | 51.52        | 51.02        | −0.50 | 1.16×   |

=== "Tesla T4"

    **Detection (box AP)**

    | Model            | Box AP, FP16 | Box AP, INT8 | Δ     | Speedup |
    | ---------------- | ------------ | ------------ | ----- | ------- |
    | Detection Nano   | 48.05        | 47.37        | −0.68 | 1.11×   |
    | Detection Small  | 52.86        | 52.20        | −0.66 | 1.10×   |
    | Detection Medium | 54.64        | 53.84        | −0.80 | 1.11×   |
    | Detection Large  | 56.54        | 56.04        | −0.50 | 1.09×   |

    **Segmentation (mask AP)**

    | Model      | Mask AP, FP16 | Mask AP, INT8 | Δ     | Speedup |
    | ---------- | ------------- | ------------- | ----- | ------- |
    | Seg Nano   | 40.10         | 39.52         | −0.58 | 1.02×   |
    | Seg Small  | 42.97         | 42.19         | −0.78 | 1.09×   |
    | Seg Medium | 44.93         | 43.95         | −0.98 | 1.05×   |
    | Seg Large  | 46.41         | 46.04         | −0.37 | 1.11×   |

    **Keypoints (keypoint AP)**

    | Model            | Keypoint AP, FP16 | Keypoint AP, INT8 | Δ     | Speedup |
    | ---------------- | ----------------- | ----------------- | ----- | ------- |
    | Keypoint Preview | 71.75             | 71.09             | −0.66 | 1.12×   |

#### Latency per image

Latency is per image: the time of one call divided by its batch. Speedup is the FP16 latency divided by the INT8 latency, in the same mode, for the same model and batch.

=== "RTX 5070, plain call"

    **Detection**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Detection Nano   | 1     | 0.933 ms | 0.693 ms | 1.35×   |
    | Detection Nano   | 4     | 0.574 ms | 0.420 ms | 1.37×   |
    | Detection Nano   | 8     | 0.514 ms | 0.362 ms | 1.42×   |
    | Detection Small  | 1     | 1.404 ms | 1.048 ms | 1.34×   |
    | Detection Small  | 4     | 1.005 ms | 0.743 ms | 1.35×   |
    | Detection Small  | 8     | 0.927 ms | 0.677 ms | 1.37×   |
    | Detection Medium | 1     | 1.612 ms | 1.278 ms | 1.26×   |
    | Detection Medium | 4     | 1.193 ms | 0.904 ms | 1.32×   |
    | Detection Medium | 8     | 1.141 ms | 0.872 ms | 1.31×   |
    | Detection Large  | 1     | 2.030 ms | 1.765 ms | 1.15×   |
    | Detection Large  | 4     | 1.742 ms | 1.459 ms | 1.19×   |
    | Detection Large  | 8     | 1.754 ms | 1.438 ms | 1.22×   |

    **Segmentation**

    | Model      | Batch | FP16     | INT8     | Speedup |
    | ---------- | ----- | -------- | -------- | ------- |
    | Seg Nano   | 1     | 1.255 ms | 1.189 ms | 1.06×   |
    | Seg Nano   | 4     | 0.838 ms | 0.731 ms | 1.15×   |
    | Seg Nano   | 8     | 0.801 ms | 0.686 ms | 1.17×   |
    | Seg Small  | 1     | 1.597 ms | 1.307 ms | 1.22×   |
    | Seg Small  | 4     | 1.236 ms | 0.994 ms | 1.24×   |
    | Seg Small  | 8     | 1.251 ms | 1.028 ms | 1.22×   |
    | Seg Medium | 1     | 1.941 ms | 1.654 ms | 1.17×   |
    | Seg Medium | 4     | 1.613 ms | 1.348 ms | 1.20×   |
    | Seg Medium | 8     | 1.722 ms | 1.469 ms | 1.17×   |
    | Seg Large  | 1     | 2.394 ms | 2.192 ms | 1.09×   |
    | Seg Large  | 4     | 2.292 ms | 2.051 ms | 1.12×   |
    | Seg Large  | 8     | 2.418 ms | 2.160 ms | 1.12×   |

    **Keypoints**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Keypoint Preview | 1     | 3.222 ms | 2.783 ms | 1.16×   |
    | Keypoint Preview | 4     | 2.632 ms | 2.225 ms | 1.18×   |
    | Keypoint Preview | 8     | 2.597 ms | 2.205 ms | 1.18×   |

=== "RTX 5070, CUDA graph"

    **Detection**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Detection Nano   | 1     | 0.882 ms | 0.629 ms | 1.40×   |
    | Detection Nano   | 4     | 0.561 ms | 0.407 ms | 1.38×   |
    | Detection Nano   | 8     | 0.506 ms | 0.355 ms | 1.43×   |
    | Detection Small  | 1     | 1.370 ms | 1.025 ms | 1.34×   |
    | Detection Small  | 4     | 0.990 ms | 0.729 ms | 1.36×   |
    | Detection Small  | 8     | 0.916 ms | 0.668 ms | 1.37×   |
    | Detection Medium | 1     | 1.584 ms | 1.246 ms | 1.27×   |
    | Detection Medium | 4     | 1.179 ms | 0.891 ms | 1.32×   |
    | Detection Medium | 8     | 1.129 ms | 0.862 ms | 1.31×   |
    | Detection Large  | 1     | 1.982 ms | 1.721 ms | 1.15×   |
    | Detection Large  | 4     | 1.714 ms | 1.436 ms | 1.19×   |
    | Detection Large  | 8     | 1.741 ms | 1.427 ms | 1.22×   |

    **Segmentation**

    | Model      | Batch | FP16     | INT8     | Speedup |
    | ---------- | ----- | -------- | -------- | ------- |
    | Seg Nano   | 1     | 1.234 ms | 1.166 ms | 1.06×   |
    | Seg Nano   | 4     | 0.833 ms | 0.725 ms | 1.15×   |
    | Seg Nano   | 8     | 0.798 ms | 0.683 ms | 1.17×   |
    | Seg Small  | 1     | 1.576 ms | 1.283 ms | 1.23×   |
    | Seg Small  | 4     | 1.231 ms | 0.993 ms | 1.24×   |
    | Seg Small  | 8     | 1.247 ms | 1.024 ms | 1.22×   |
    | Seg Medium | 1     | 1.915 ms | 1.625 ms | 1.18×   |
    | Seg Medium | 4     | 1.607 ms | 1.342 ms | 1.20×   |
    | Seg Medium | 8     | 1.719 ms | 1.465 ms | 1.17×   |
    | Seg Large  | 1     | 2.366 ms | 2.167 ms | 1.09×   |
    | Seg Large  | 4     | 2.286 ms | 2.045 ms | 1.12×   |
    | Seg Large  | 8     | 2.415 ms | 2.157 ms | 1.12×   |

    **Keypoints**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Keypoint Preview | 1     | 3.204 ms | 2.764 ms | 1.16×   |
    | Keypoint Preview | 4     | 2.625 ms | 2.218 ms | 1.18×   |
    | Keypoint Preview | 8     | 2.590 ms | 2.189 ms | 1.18×   |

=== "Tesla T4, plain call"

    **Detection**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Detection Nano   | 1     | 3.267 ms  | 2.953 ms  | 1.11×   |
    | Detection Nano   | 4     | 2.670 ms  | 2.268 ms  | 1.18×   |
    | Detection Nano   | 8     | 2.892 ms  | 2.228 ms  | 1.30×   |
    | Detection Small  | 1     | 5.314 ms  | 4.816 ms  | 1.10×   |
    | Detection Small  | 4     | 5.334 ms  | 4.207 ms  | 1.27×   |
    | Detection Small  | 8     | 5.482 ms  | 4.403 ms  | 1.24×   |
    | Detection Medium | 1     | 6.706 ms  | 6.037 ms  | 1.11×   |
    | Detection Medium | 4     | 6.608 ms  | 5.516 ms  | 1.20×   |
    | Detection Medium | 8     | 7.187 ms  | 6.073 ms  | 1.18×   |
    | Detection Large  | 1     | 10.496 ms | 9.635 ms  | 1.09×   |
    | Detection Large  | 4     | 10.587 ms | 9.777 ms  | 1.08×   |
    | Detection Large  | 8     | 11.379 ms | 10.106 ms | 1.13×   |

    **Segmentation**

    | Model      | Batch | FP16      | INT8      | Speedup |
    | ---------- | ----- | --------- | --------- | ------- |
    | Seg Nano   | 1     | 5.205 ms  | 5.097 ms  | 1.02×   |
    | Seg Nano   | 4     | 4.661 ms  | 4.313 ms  | 1.08×   |
    | Seg Nano   | 8     | 5.049 ms  | 4.461 ms  | 1.13×   |
    | Seg Small  | 1     | 6.921 ms  | 6.374 ms  | 1.09×   |
    | Seg Small  | 4     | 6.900 ms  | 5.648 ms  | 1.22×   |
    | Seg Small  | 8     | 7.202 ms  | 5.955 ms  | 1.21×   |
    | Seg Medium | 1     | 9.532 ms  | 9.035 ms  | 1.05×   |
    | Seg Medium | 4     | 9.636 ms  | 8.174 ms  | 1.18×   |
    | Seg Medium | 8     | 10.113 ms | 8.795 ms  | 1.15×   |
    | Seg Large  | 1     | 14.112 ms | 12.707 ms | 1.11×   |
    | Seg Large  | 4     | 14.274 ms | 12.678 ms | 1.13×   |
    | Seg Large  | 8     | 14.306 ms | 13.131 ms | 1.09×   |

    **Keypoints**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Keypoint Preview | 1     | 15.345 ms | 13.708 ms | 1.12×   |
    | Keypoint Preview | 4     | 16.083 ms | 14.101 ms | 1.14×   |
    | Keypoint Preview | 8     | 16.737 ms | 14.599 ms | 1.15×   |

=== "Tesla T4, CUDA graph"

    **Detection**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Detection Nano   | 1     | 3.193 ms  | 2.748 ms  | 1.16×   |
    | Detection Nano   | 4     | 2.664 ms  | 2.221 ms  | 1.20×   |
    | Detection Nano   | 8     | 2.903 ms  | 2.212 ms  | 1.31×   |
    | Detection Small  | 1     | 5.335 ms  | 4.721 ms  | 1.13×   |
    | Detection Small  | 4     | 5.272 ms  | 4.140 ms  | 1.27×   |
    | Detection Small  | 8     | 5.476 ms  | 4.379 ms  | 1.25×   |
    | Detection Medium | 1     | 6.711 ms  | 5.939 ms  | 1.13×   |
    | Detection Medium | 4     | 6.611 ms  | 5.491 ms  | 1.20×   |
    | Detection Medium | 8     | 7.158 ms  | 6.041 ms  | 1.18×   |
    | Detection Large  | 1     | 10.562 ms | 9.533 ms  | 1.11×   |
    | Detection Large  | 4     | 10.518 ms | 9.760 ms  | 1.08×   |
    | Detection Large  | 8     | 11.370 ms | 10.112 ms | 1.12×   |

    **Segmentation**

    | Model      | Batch | FP16      | INT8      | Speedup |
    | ---------- | ----- | --------- | --------- | ------- |
    | Seg Nano   | 1     | 5.239 ms  | 4.987 ms  | 1.05×   |
    | Seg Nano   | 4     | 4.598 ms  | 4.228 ms  | 1.09×   |
    | Seg Nano   | 8     | 5.006 ms  | 4.453 ms  | 1.12×   |
    | Seg Small  | 1     | 6.965 ms  | 6.416 ms  | 1.09×   |
    | Seg Small  | 4     | 6.914 ms  | 5.563 ms  | 1.24×   |
    | Seg Small  | 8     | 7.223 ms  | 5.962 ms  | 1.21×   |
    | Seg Medium | 1     | 9.592 ms  | 8.889 ms  | 1.08×   |
    | Seg Medium | 4     | 9.601 ms  | 8.083 ms  | 1.19×   |
    | Seg Medium | 8     | 10.112 ms | 8.822 ms  | 1.15×   |
    | Seg Large  | 1     | 14.050 ms | 12.703 ms | 1.11×   |
    | Seg Large  | 4     | 14.262 ms | 12.644 ms | 1.13×   |
    | Seg Large  | 8     | 14.300 ms | 13.143 ms | 1.09×   |

    **Keypoints**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Keypoint Preview | 1     | 15.050 ms | 13.492 ms | 1.12×   |
    | Keypoint Preview | 4     | 16.066 ms | 14.098 ms | 1.14×   |
    | Keypoint Preview | 8     | 16.733 ms | 14.594 ms | 1.15×   |

On the RTX 5070, the models whose backbone attention runs in INT8 (detection Nano, Small and Medium, Seg Small and Seg Medium; see the placement rules below) gain 1.18× to 1.40× at batch 1 with a CUDA graph. The ones that keep it in FP16 (detection Large, Seg Nano, Seg Large and Keypoint Preview) gain 1.06× to 1.16×. For detection the gain at batch 8 is 0.03× to 0.07× above the gain at batch 1 (Nano dips at batch 4 and Medium peaks there); for Seg Nano it grows by 0.11×; for Seg Small, Seg Medium, Seg Large and Keypoint Preview it changes by at most 0.03× between batch 1, 4 and 8. On the Tesla T4 that split does not hold: with a CUDA graph at batch 1, Seg Small (1.09×) and Seg Medium (1.08×), which run INT8 attention, gain no more than Seg Large (1.11×) and Keypoint Preview (1.12×), which keep it in FP16. The cause was not measured. On both GPUs the plain call and the CUDA graph give almost the same speedup: they differ by at most 0.05× on the RTX 5070 and on the Tesla T4.

#### Engine size

!!! warning "On the Tesla T4 the segmentation and keypoint INT8 engines grow with the batch much faster than the FP16 one"

    On the Tesla T4 the INT8 engine of every segmentation and keypoint model is larger than the FP16 one at batch 4 and 8, 1.42× to 2.19× at batch 8: Seg Large at batch 8 is 193 MB against 94 MB, and Keypoint Preview 248 MB against 113 MB. At batch 1 it is larger for Keypoint Preview (83 MB against 82 MB). Plan the batch you serve when you build, and check the size of each engine before you deploy a large batch. The cause has not been diagnosed. The Tesla T4 detection engines, measured on TensorRT 10.16.1.11, do not grow like this: the INT8 engine is 0.64× to 0.75× of the FP16 one at every batch. In the same session the engines built without the INT8 patch embedding did not grow either (Nano 37.9 MB at batch 1 and at batch 8), so the patch embedding is not shown to be the cause, and the earlier detection rows, which did grow on TensorRT 11.4, were not repeated on 11.4. On the RTX 5070 the INT8 engine is smaller than the FP16 one in every row.

The size of an engine is not a function of the batch alone. On the RTX 5070 the same model varies by up to 10 MB between batch 1, 4 and 8, and not always in step with the batch, and the size depends on the GPU and the TensorRT version. `INT8 ÷ FP16` is below 1 when the INT8 engine is the smaller one.

=== "RTX 5070"

    | Model            | Batch | FP16  | INT8  | INT8 ÷ FP16 |
    | ---------------- | ----- | ----- | ----- | ----------- |
    | Detection Nano   | 1     | 62 MB | 38 MB | 0.61×       |
    | Detection Nano   | 4     | 66 MB | 38 MB | 0.57×       |
    | Detection Nano   | 8     | 72 MB | 39 MB | 0.53×       |
    | Detection Small  | 1     | 66 MB | 41 MB | 0.61×       |
    | Detection Small  | 4     | 74 MB | 41 MB | 0.55×       |
    | Detection Small  | 8     | 68 MB | 42 MB | 0.61×       |
    | Detection Medium | 1     | 71 MB | 43 MB | 0.61×       |
    | Detection Medium | 4     | 70 MB | 44 MB | 0.63×       |
    | Detection Medium | 8     | 72 MB | 44 MB | 0.61×       |
    | Detection Large  | 1     | 68 MB | 49 MB | 0.72×       |
    | Detection Large  | 4     | 72 MB | 49 MB | 0.69×       |
    | Detection Large  | 8     | 76 MB | 50 MB | 0.65×       |
    | Seg Nano         | 1     | 69 MB | 51 MB | 0.74×       |
    | Seg Nano         | 4     | 67 MB | 51 MB | 0.77×       |
    | Seg Nano         | 8     | 70 MB | 53 MB | 0.75×       |
    | Seg Small        | 1     | 69 MB | 46 MB | 0.67×       |
    | Seg Small        | 4     | 69 MB | 47 MB | 0.67×       |
    | Seg Small        | 8     | 71 MB | 48 MB | 0.68×       |
    | Seg Medium       | 1     | 72 MB | 49 MB | 0.68×       |
    | Seg Medium       | 4     | 74 MB | 50 MB | 0.68×       |
    | Seg Medium       | 8     | 75 MB | 52 MB | 0.69×       |
    | Seg Large        | 1     | 73 MB | 55 MB | 0.75×       |
    | Seg Large        | 4     | 75 MB | 56 MB | 0.75×       |
    | Seg Large        | 8     | 77 MB | 58 MB | 0.76×       |
    | Keypoint Preview | 1     | 80 MB | 62 MB | 0.77×       |
    | Keypoint Preview | 4     | 85 MB | 66 MB | 0.78×       |
    | Keypoint Preview | 8     | 90 MB | 71 MB | 0.79×       |

=== "Tesla T4"

    | Model            | Batch | FP16   | INT8   | INT8 ÷ FP16 |
    | ---------------- | ----- | ------ | ------ | ----------- |
    | Detection Nano   | 1     | 58 MB  | 37 MB  | 0.64×       |
    | Detection Nano   | 4     | 57 MB  | 37 MB  | 0.65×       |
    | Detection Nano   | 8     | 56 MB  | 38 MB  | 0.68×       |
    | Detection Small  | 1     | 61 MB  | 40 MB  | 0.66×       |
    | Detection Small  | 4     | 59 MB  | 40 MB  | 0.68×       |
    | Detection Small  | 8     | 59 MB  | 40 MB  | 0.68×       |
    | Detection Medium | 1     | 64 MB  | 42 MB  | 0.66×       |
    | Detection Medium | 4     | 62 MB  | 43 MB  | 0.69×       |
    | Detection Medium | 8     | 62 MB  | 43 MB  | 0.69×       |
    | Detection Large  | 1     | 64 MB  | 48 MB  | 0.75×       |
    | Detection Large  | 4     | 63 MB  | 46 MB  | 0.73×       |
    | Detection Large  | 8     | 63 MB  | 46 MB  | 0.73×       |
    | Seg Nano         | 1     | 69 MB  | 56 MB  | 0.81×       |
    | Seg Nano         | 4     | 70 MB  | 74 MB  | 1.06×       |
    | Seg Nano         | 8     | 73 MB  | 104 MB | 1.42×       |
    | Seg Small        | 1     | 67 MB  | 55 MB  | 0.82×       |
    | Seg Small        | 4     | 71 MB  | 86 MB  | 1.21×       |
    | Seg Small        | 8     | 79 MB  | 127 MB | 1.61×       |
    | Seg Medium       | 1     | 71 MB  | 60 MB  | 0.85×       |
    | Seg Medium       | 4     | 77 MB  | 100 MB | 1.30×       |
    | Seg Medium       | 8     | 86 MB  | 152 MB | 1.77×       |
    | Seg Large        | 1     | 73 MB  | 70 MB  | 0.96×       |
    | Seg Large        | 4     | 82 MB  | 122 MB | 1.49×       |
    | Seg Large        | 8     | 94 MB  | 193 MB | 2.05×       |
    | Keypoint Preview | 1     | 82 MB  | 83 MB  | 1.01×       |
    | Keypoint Preview | 4     | 95 MB  | 152 MB | 1.60×       |
    | Keypoint Preview | 8     | 113 MB | 248 MB | 2.19×       |

### Placement rules

The INT8 placement follows what was measured, not "quantize every matrix multiply":

- **INT8:** the patch embedding of the detection models, every backbone MLP, the windowed backbone blocks' attention with its projections (up to 325 tokens; see below), and the decoder's cross-attention, feed-forward and reference-point layers. Every layer left unquantized stays FP16.
- **FP16:** the projector, the two-stage proposal head and the detection heads (quantizing them cost accuracy and bought nothing), the mask and keypoint heads (not measured), plus the attention blocks below.
- **The patch embedding runs in INT8 as a matrix multiply, for the detection models.** TensorRT cannot run the 3-channel patch convolution in INT8 without padding its input to 32 channels and storing the added zero channels in the engine as a constant that grows with the batch, which made the engine larger and the INT8 speedup smaller. So every convolution whose input channels are not a multiple of 32 stays FP16, and the export rewrites the 16-pixel patch embedding of the Nano to Large detection models as a matrix multiply over the 768 values of each patch, which has nothing to pad. It is quantized like the other backbone layers. The segmentation and keypoint models cut 12-pixel patches of 432 values, which is not a multiple of 32, so their patch embedding stays an FP16 convolution, as does any convolution that is not a non-overlapping patch embedding.
- **Attention is all INT8 or all FP16.** A block with head size 16, 32 or 64 and at most 325 tokens gets INT8 projections and INT8 attention matrix multiplies, which TensorRT fuses into one INT8 attention kernel: the windowed backbone blocks of detection Nano (145 tokens), Small (257) and Medium (325), and of Seg Small (257) and Seg Medium (325). 325 tokens is the largest window where INT8 attention was measured to pay; on Large's 485-token windows it was no faster and less accurate, so Large keeps them FP16. Windows of 326 to 484 tokens are not measured and stay FP16. The global blocks (580 tokens on Nano) and the decoder self-attention keep projections and attention in FP16 too. A windowed block has `(resolution / patch_size / num_windows)² + 1` tokens, so a higher export resolution can move a model's attention to FP16; the export logs how many backbone attention blocks stay FP16.

### Requirements and limits

- Detection, segmentation and keypoint models. The mask head and the keypoint embedding stay FP16 (quantizing them was not measured), and so does the keypoint decoder's self-attention between each detection and its keypoints (`kp_inst_self_attn`); the keypoint decoder's cross-attention and feed-forward layers run in INT8.
- The attention rule above applies unchanged. Only `RFDETRSegSmall` (257 tokens per window) and `RFDETRSegMedium` (325) run INT8 attention, in the 9 windowed blocks of their 12 backbone blocks (the other 3 attend globally). `RFDETRSegNano` (one 677-token window per block), `RFDETRSegLarge` (442) and `RFDETRKeypointPreview` (577) keep all attention in FP16, so only their backbone MLPs and the decoder's cross-attention, feed-forward and reference-point layers run in INT8.
- `RFDETRSegXLarge` and `RFDETRSeg2XLarge` (`rfdetr_plus`) are accepted but were not measured. `backbone_only=True` is refused with `ValueError`.
- A static batch (`dynamic_batch=True` is refused), `fp16=True`, and TensorRT 10 or newer. Every table above was measured on TensorRT 11.4 except the Tesla T4 detection rows, which were measured on 10.16.1.11. The detection placement without the patch-embedding matrix multiply was also measured on 10.16 and 11.3; the matrix multiply itself on 11.4 (RTX 5070) and on 10.16 (Tesla T4); the segmentation and keypoint models only on 11.4; the other releases from 10 on are accepted but untested. The batch is fixed when you build the engine.
- Only an RTX 5070 (sm_120) and a Tesla T4 (sm_75) were measured, and only without `trt_hardware_compatibility`, `trt_version_compatible` or a shared `trt_timing_cache`. These settings are accepted with `quantization="int8"`, but engines built with them were not measured, and a portable engine may not get the fused INT8 attention kernel the speedup relies on.
- INT8 attention relies on TensorRT's fused INT8 attention kernel, which NVIDIA lists for sm_75 to sm_90, sm_120 and sm_121. On other GPUs (for example sm_100, B200) it would run unfused, so measure on your GPU before deploying.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. By default it is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases; see [Portable Engines](#portable-engines) for the options that widen that.

    If you plan to run inference with [`inference-models`](basics.md#run-inference-with-inference-models) (a separate deployment option), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

## Predict with RFDETRInference

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/inference_model_fp16.trt")
detections = model.predict("image.jpg", threshold=0.5)
```

The engine must be compatible with the target GPU and TensorRT version. See [Predict with RFDETRInference](basics.md#predict-with-rfdetrinference) for checkpoint inputs, metadata, and batch behavior.

`RFDETRInference` launches the engine with `execute_v2` by default, which needs no [`pycuda`](https://pypi.org/project/pycuda/). `runtime_options` takes three settings, each a `bool` that defaults to `False`:

- `cuda_graph`: capture the engine's launch into a CUDA graph on the first call and replay it on later calls. Only an engine whose optimization profile fixes every input shape qualifies; an engine exported with `dynamic_batch=True` is refused when it is loaded. We have not measured the speed-up through `RFDETRInference`.
- `engine_host_code_allowed`: load an engine that carries host code; see [Portable Engines](#portable-engines).
- `verbose`: log TensorRT at VERBOSE rather than INFO.

```python
model = RFDETRInference("output/inference_model_fp16.trt", runtime_options={"cuda_graph": True})
```

!!! warning "TRTInference is deprecated"

    Constructing `TRTInference` emits `FutureWarning`: it is deprecated since v1.12.0 and will be removed in v1.15.0. Use `RFDETRInference(engine_path).predict(image)` for image inputs and Supervision results, and its `runtime_options` for `cuda_graph`, `engine_host_code_allowed` and `verbose`. The old class remains available for compatibility until then.

Both facades use the same session loading and execution functions. The shared adapter calls these functions directly and does not construct the deprecated class.

## Engine Description File

A `.trt` file has no slot for user metadata, so a consumer that does not import the model, such as a C++ service, [Triton](https://developer.nvidia.com/triton-inference-server) or [DeepStream](https://developer.nvidia.com/deepstream-sdk), cannot learn what the engine expects from the file itself. Pass `trt_metadata=True` to write `<engine>.json` next to it, with the same name and a `.json` suffix:

```python
import hashlib
import json
from pathlib import Path

engine = model.export(format="tensorrt", trt_metadata=True)
# engine -> output/rfdetr-medium_fp16.trt, described by output/rfdetr-medium_fp16.json

with open(Path(engine).with_suffix(".json")) as description_file:
    description = json.load(description_file)
engine_bytes = Path(engine).read_bytes()  # deserialize these bytes, not a second read of the file
if hashlib.sha256(engine_bytes).hexdigest() != description["engine"]["sha256"]:
    raise RuntimeError(f"{engine} is not the engine this description was written for")
height, width = description["input"]["height"], description["input"]["width"]
```

| Key              | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version` | Version of this layout, `1`. It changes when a key is removed, renamed or changes meaning. Adding a key does not change it, so a reader ignores the keys it does not know.                                                                                                                                                                                                                                                                                                                                                                                              |
| `rfdetr_version` | The `rfdetr` version that exported the engine, or `null` when run from a source tree that is not installed.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `variant`        | The model variant name, such as `rfdetr-nano`, or `null` when there is none.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `backbone_only`  | Whether the engine is a backbone-only export, whose outputs are feature maps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `engine`         | The engine file this description file was written for: its `size` in bytes and `sha256`, the SHA-256 hex digest of its bytes. Compare them with the `.trt` you load, as in the example above.                                                                                                                                                                                                                                                                                                                                                                           |
| `input`          | The input tensor: `name`, `layout` (`NCHW`), `dtype`, `height`, `width`, `channels`, `channel_order` (`"RGB"` for three channels, otherwise `null`), `normalization` and `resize`. Pixel values are divided by `normalization.scale`, then `(x - mean) / std` is applied per channel; `resize` is bilinear with half-pixel centers and no antialiasing, and its `aspect_ratio` is `"stretch"`: the whole image is resized to `height` x `width`, with no letterbox and no padding, so the boxes are normalized to the image and scale by its original width and height. |
| `outputs`        | The output tensors in the order the engine returns them, each with `name` and `dtype`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch`          | `{"dynamic": false, "size": N}` for a static engine, or `{"dynamic": true, "min": 1, "opt": N, "max": M}` for one built with `dynamic_batch=True`.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `build`          | How the engine was built: `precision` (the one actually built, which can differ from `fp16` on a lean TensorRT wheel, and is `int8` for a `quantization="int8"` engine; on TensorRT before 11 it records that the FP16 builder flag was set, so FP16 is allowed but not forced, while on TensorRT 11 and later it records an fp16 cast graph, so read `tensorrt_version` to tell which applies), `opset`, `tensorrt_version` and `gpu` (`name` and `compute_capability`, or `null` when PyTorch has no CUDA).                                                           |
| `notes`          | The `notes` passed to `export()`: a string as it is, any other value JSON-encoded into a string (decode it once more), or `null`.                                                                                                                                                                                                                                                                                                                                                                                                                                       |

All inputs and outputs are `float32`, also for an `fp16` engine. The normalization values are the ones `RFDETR.predict()` applies. The `resize` block records the convention of the export runtime and of `RFDETR.predict()` by default; a checkpoint trained with an antialiased resize may expect `antialias=True`, which is the `antialias` argument of `RFDETR.predict()`, and the file does not record that.

!!! note "What the file does not say"

    It does not record class names or which logit slot is the background class. How a slot maps to a class depends on the checkpoint (official COCO weights use sparse COCO category ids, fine-tuned models use contiguous ids), and `RFDETR.predict()` decides it internally. For `dets` and `labels` outputs, `dets` holds normalized `cx, cy, w, h` boxes and `labels` holds raw class logits, which `RFDETR.predict()` turns into scores with a per-class sigmoid (not a softmax) before it keeps the highest-scoring (query, class) pairs above the confidence threshold; see [`decode_detections`](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/export/_runtime/decode.py) for how the repository turns them into detections.

The file is written after the engine and replaces an older description of the same name in one step, so a reader never sees half a file; if a `.json` of that name that an RF-DETR export did not write is already there, `export()` raises `FileExistsError` before building the engine, and you rename or remove that file or pick another `output_name`. On Linux and macOS it gets the engine's permission bits and, where you are allowed to set it, the engine's group, so a service that can read the engine can read the description too. Only `RFDETR.export` (or calling the exporter on a prepared graph) writes it; `TensorRTExporter.build_engine` on an `.onnx` file has no graph to describe and does not, and with `TensorRTConfig(metadata=True)` it warns that the setting has no effect there. If the file can't be written, `export()` raises `OSError` after the engine is built, and the message says so.

The engine and its description file are still two files, replaced one after the other. A program that reads them while an export is running, or after two exports of the same name ran at the same time, can find an engine next to the description file of another build. That is what `engine` is for: compare its `sha256` with the bytes you load, as the example above does, before you trust the rest.

The engine's file name doesn't include its batch profile or input size, so a later export of the same model overwrites the engine and can leave an old `.json` next to it. When a build doesn't write a description file (`export()` without `trt_metadata=True`, or `build_engine`) and one with the engine's name is already there, you get a warning that it may no longer match. The warning repeats on every rebuild of that engine name for as long as the old description file stays in place. The file is never deleted, because this build didn't write it; a program that checks its `engine.sha256` finds out whether it still matches.

## Portable Engines

By default an engine only runs on the kind of GPU and the TensorRT version that built it. Two options ask TensorRT for more portable engines. Both are off by default, and the default build doesn't change. An engine built with either option gets its own file name, like `<variant>_fp16_ampere_plus.trt`, so it doesn't overwrite a default engine in the same folder (unless you set `output_name`, which is used as is).

```python
# Ask for an engine that NVIDIA Ampere GPUs (compute capability 8.x) and newer can run.
model.export(format="tensorrt", trt_hardware_compatibility="ampere_plus")

# Ask for an engine that other releases of the same TensorRT major version may load.
model.export(format="tensorrt", trt_version_compatible=True)
```

`trt_hardware_compatibility` is `"ampere_plus"` or `"same_compute_capability"`. The first targets Ampere and every newer GPU, and has to be built on one of them; the second targets GPUs with the same compute capability as the one you build on. TensorRT records the level in the engine. We only had one GPU, so we haven't tested running such an engine on a second one. NVIDIA doesn't support hardware compatibility on Jetson (JetPack) or DriveOS ([engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)). If your TensorRT doesn't have the level, you get a `ValueError` before the export runs the model.

`trt_version_compatible=True` asks for an engine that other releases of the same TensorRT major version may load. It worked between TensorRT 11.2 and 11.3, in both directions. It didn't work across major versions (10 and 11), or between TensorRT 10.13 and 10.16, so test your pair of releases before you rely on it. The build needs TensorRT's lean runtime, which is a separate package: install the lean wheel that matches your `tensorrt` wheel, for example [`tensorrt-lean-cu13-libs`](https://pypi.org/project/tensorrt-lean-cu13-libs/) next to [`tensorrt-cu13-libs`](https://pypi.org/project/tensorrt-cu13-libs/) (the version may differ by a `.post` suffix), or use the library from the TensorRT archive or system package. Without it the export raises `ImportError` naming the library, before it runs the model.

The options have a cost, so measure on your own model. In our FP16 tests, `ampere_plus` made the build about 3.5 times slower, the engine about 60% larger and inference about 10% slower. `same_compute_capability` cost nothing we could measure. `trt_version_compatible` added about 105 MB to an `RFDETRNano` engine on TensorRT 11 (on TensorRT 10.16 the engine was the size of a default one). The [changelog](https://github.com/roboflow/rf-detr/blob/main/CHANGELOG.md) has the numbers.

An engine built by TensorRT 11 with `trt_version_compatible=True` contains host code, and TensorRT only loads it if you say you trust the file. `RFDETRInference` takes `engine_host_code_allowed` in `runtime_options` for that, and the deprecated `TRTInference` and `rfdetr.export.benchmark` take it as a keyword; it's off by default:

```python
from rfdetr import RFDETRInference

engine = model.export(format="tensorrt", trt_version_compatible=True)
runtime = RFDETRInference(engine, runtime_options={"engine_host_code_allowed": True})
```

Only turn it on for a file you built yourself or otherwise trust.

## Reuse the Timing Cache

A large share of an engine build goes into timing candidate kernels for each layer. TensorRT can save those timings in a cache file, and a later build that uses the file skips the timing it already did. Pass `trt_timing_cache` to keep one across exports:

```python
model.export(format="tensorrt", trt_timing_cache="output/rfdetr.cache")
```

The first export creates the file (and its folder). Later exports load it and write the merged timings back. It pays off when you rebuild the same model at the same precision and batch profile, on the same GPU and TensorRT version, for example after fine-tuning: the timings depend on the layers' shapes, not on the weights. In our tests, a cache from the pretrained model saved about as much time for a model fine-tuned to 3 classes, whose class head has a different shape. A different resolution changes every layer's shape, and we haven't measured it. A cache from a matching dynamic-batch build (batch 1-8) helps too: it cut the build time by 74-76% in fp16 and 42-43% in fp32. We did not measure reusing a cache between a static and a dynamic batch profile, or across precisions.

A cache TensorRT can't use doesn't stop the export. For a file written by another TensorRT major version, or an empty or corrupt one, TensorRT logs a serialization error, builds as if there were no cache, and the file is replaced with the new timings. A timing cache is specific to the GPU, the CUDA version and the TensorRT version, so keep one cache file per GPU and per software stack. TensorRT's API reference says a cache whose recorded CUDA device properties differ from the current environment is reported as a failure; we had one GPU and could not test that. We tested TensorRT 10.16 and 11.3 with Polygraphy 0.53.4, while the supported range is `tensorrt>=8.6.1` and the Polygraphy version is not pinned, so behavior on other versions is untested.

A cache that can't be written does stop the export, though: Polygraphy writes the timing cache before it returns the engine, so if that write fails (a full disk, say) the export raises and the built engine is not saved. Fix the cause and export again; the engine can simply be rebuilt.

A bad path is refused before the engine is built: a directory, or a path ending in a separator, raises `ValueError`, and a location that can't be created or written raises `OSError`. `~` is expanded, and a relative path is relative to the working directory, not to `output_dir`.

Polygraphy keeps an empty `<file>.lock` next to the cache and locks it while it reads or writes the cache, so the folder has to be writable. We did not test two exports sharing one cache file at the same time. Without `trt_timing_cache`, no file is read or written.

## Advanced: Convert an existing ONNX file with the Python API

Use this only to convert an **already-exported** `.onnx` file without re-running the model export. To go straight from a checkpoint to an engine, use [`format="tensorrt"`](#export-directly-to-tensorrt) above.

!!! warning "Internal API"

    `rfdetr.export._tensorrt.exporter` is a private module — the leading underscore means it carries no stability guarantee and may move or change signature in any release. `RFDETR.export(format="tensorrt")` is the supported entry point; use the class below only when you need to convert an already-exported `.onnx` file.

```python
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter

exporter = TensorRTExporter(TensorRTConfig(fp16=True))
engine_path = exporter.build_engine("output/inference_model.onnx")
# -> "output/inference_model_fp16.trt"
```

`TensorRTExporter.build_engine` builds the engine in-process via the TensorRT Python API (no `trtexec` subprocess) and returns the path to the generated `.trt` engine file. Precision and progress logging come from the `TensorRTConfig` the exporter is constructed with — pass `TensorRTConfig(output_name="my-engine")` to write `output/my-engine.trt` verbatim instead. `TensorRTConfig(timing_cache="output/rfdetr.cache")` reuses a timing cache here too; see [Reuse the Timing Cache](#reuse-the-timing-cache).

An `.onnx` file exported with `dynamic_batch=True` needs `dynamic_batch=True` here as well, plus `max_batch_size`, the largest batch the engine accepts. `opt_batch_size` (default `1`) is the batch its kernels are tuned for. Without `dynamic_batch=True`, `build_engine` raises `ValueError` rather than building an engine that accepts batch 1 only.

```python
exporter = TensorRTExporter(TensorRTConfig(fp16=True, dynamic_batch=True, max_batch_size=16, opt_batch_size=4))
engine_path = exporter.build_engine("output/inference_model.onnx")
```

The portability options work here too, as `TensorRTConfig(hardware_compatibility="ampere_plus")` and `TensorRTConfig(version_compatible=True)`; see [Portable Engines](#portable-engines).
