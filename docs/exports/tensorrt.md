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

This writes `output/rfdetr-small_int8.trt` (next to the intermediate `output/rfdetr-small.onnx`). The export runs the first `max_images` (default `100`) calibration images by file name through the FP32 graph with onnxruntime on the CPU, preprocessed exactly as `predict()` does, and records each quantized tensor's largest absolute value. For a detection model it runs the images twice more: first to measure, per channel, how wide each backbone LayerNorm output is, so that the widest channels can be moved into the weights that read them before the ranges are recorded, and last to measure how much INT8 shifts the mean output of each backbone MLP layer, which is then subtracted from the layer's bias (see the [placement rules](#placement-rules)). Those ranges decide the engine's accuracy, so use images from the deployment domain: out-of-domain images give an engine that loads, runs and is quietly less accurate. `calibration_data` also accepts a `.npy` path or an array shaped `(N, C, H, W)`, already normalized the way `predict()` normalizes; an integer array, raw pixels, is refused. `max_images` caps only a directory of images; an array or `.npy` file is used whole, so pass as many samples as you want calibrated. Calibration runs the graph at the export's `batch_size`, so its host memory grows with the batch.

### Results

The benchmark of [Section 8.1 of the CUDA cookbook](../../cookbooks/export-cuda/#81-benchmark-for-the-docs-tables) ran on two GPUs, an RTX 5070 and a Tesla T4, both on TensorRT 11.4.0.106. Every row below is from that benchmark.

**Summary.** INT8 is faster than FP16 in every row on both GPUs and loses 1.01 AP points or less on each model's own task. The INT8 engine is smaller than the FP16 one in every row on both GPUs.

|                                 | RTX 5070              | Tesla T4              |
| ------------------------------- | --------------------- | --------------------- |
| Speedup, plain call             | 1.10× to 1.42×        | 1.04× to 1.40×        |
| Speedup, CUDA graph             | 1.11× to 1.42×        | 1.08× to 1.41×        |
| AP change at batch 1            | −0.32 to −1.01 points | −0.27 to −0.97 points |
| AP change, share of the FP16 AP | 0.6% to 2.3%          | 0.5% to 2.2%          |
| Engine size, INT8 ÷ FP16        | 0.53× to 0.76×        | 0.41× to 0.74×        |

#### How it was measured

- **Machines:** an RTX 5070 (sm_120, native Linux) and a Tesla T4 (sm_75, Colab). Both ran TensorRT 11.4.0.106 (on the Colab runtime, the `tensorrt-cu12` build).
- **Engines:** a static FP16 and a static INT8 engine of every model at batch 1, 4 and 8. The INT8 engines are calibrated on the first 128 `train2017` images by file name. The segmentation and keypoint INT8 engines of both GPUs run the patch embedding as a matrix multiply that stays FP16 (see the [placement rules](#placement-rules)); the FP16 export is unchanged, and the segmentation and keypoint FP16 rows of both GPUs were re-measured in the same runs as the INT8 ones.
- **Accuracy:** batch 1, all 5000 `val2017` images, decoded the way `predict()` decodes. Box AP and mask AP come from `instances_val2017`; box AP (person class only) and keypoint AP from `person_keypoints_val2017`.
- **Latency:** GPU time per image (CUDA events), the median of 15 rounds of 200 executions, FP16 and INT8 interleaved. A *plain call* is 200 back-to-back `execute_async_v3` calls with no synchronization in between. A *CUDA graph* replays one captured call 200 times.
- **Reproduce it:** [Section 8.1 of the CUDA cookbook](../../cookbooks/export-cuda/#81-benchmark-for-the-docs-tables) runs these exact steps on your GPU. It is written for Colab, takes hours for all nine models, and prints the GPU and the TensorRT version next to its table, because the numbers depend on them. The tables above used TensorRT 11.4.0.106, so compare the version it prints before you compare numbers.

The RTX 5070 differences and the Tesla T4 segmentation and keypoint ones are computed from the unrounded values, and the Tesla T4 detection ones from the values its notebook printed (engine sizes in whole MB), so recomputing them from the rounded cells can differ in the last digit.

#### At a glance

Batch 1, all 5000 `val2017` images. Δ is INT8 minus FP16 in COCO AP points. Speedup is the FP16 latency divided by the INT8 latency, for a plain call. Each task has its own table, because each is scored on a different AP.

=== "RTX 5070"

    **Detection (box AP)**

    | Model            | Box AP, FP16 | Box AP, INT8 | Δ     | Speedup |
    | ---------------- | ------------ | ------------ | ----- | ------- |
    | Detection Nano   | 48.02        | 47.52        | −0.50 | 1.36×   |
    | Detection Small  | 52.82        | 52.35        | −0.47 | 1.33×   |
    | Detection Medium | 54.66        | 54.00        | −0.66 | 1.24×   |
    | Detection Large  | 56.53        | 56.21        | −0.32 | 1.14×   |

    **Segmentation (mask AP)**

    | Model      | Mask AP, FP16 | Mask AP, INT8 | Δ     | Box AP, FP16 | Box AP, INT8 | Δ box | Speedup |
    | ---------- | ------------- | ------------- | ----- | ------------ | ------------ | ----- | ------- |
    | Seg Nano   | 40.09         | 39.48         | −0.61 | 48.42        | 47.63        | −0.79 | 1.10×   |
    | Seg Small  | 42.94         | 41.98         | −0.97 | 51.09        | 49.85        | −1.24 | 1.23×   |
    | Seg Medium | 44.95         | 43.93         | −1.01 | 53.42        | 52.14        | −1.27 | 1.17×   |
    | Seg Large  | 46.44         | 46.05         | −0.39 | 54.91        | 54.16        | −0.75 | 1.11×   |

    **Keypoints (keypoint AP)**

    | Model            | Keypoint AP, FP16 | Keypoint AP, INT8 | Δ     | Box AP, FP16 | Box AP, INT8 | Δ box | Speedup |
    | ---------------- | ----------------- | ----------------- | ----- | ------------ | ------------ | ----- | ------- |
    | Keypoint Preview | 71.77             | 71.06             | −0.71 | 51.52        | 50.96        | −0.56 | 1.16×   |

=== "Tesla T4"

    **Detection (box AP)**

    | Model            | Box AP, FP16 | Box AP, INT8 | Δ     | Speedup |
    | ---------------- | ------------ | ------------ | ----- | ------- |
    | Detection Nano   | 48.07        | 47.57        | −0.50 | 1.25×   |
    | Detection Small  | 52.80        | 52.44        | −0.36 | 1.14×   |
    | Detection Medium | 54.67        | 54.02        | −0.65 | 1.15×   |
    | Detection Large  | 56.53        | 56.26        | −0.27 | 1.12×   |

    **Segmentation (mask AP)**

    | Model      | Mask AP, FP16 | Mask AP, INT8 | Δ     | Box AP, FP16 | Box AP, INT8 | Δ box | Speedup |
    | ---------- | ------------- | ------------- | ----- | ------------ | ------------ | ----- | ------- |
    | Seg Nano   | 40.07         | 39.50         | −0.58 | 48.39        | 47.59        | −0.80 | 1.04×   |
    | Seg Small  | 42.89         | 42.22         | −0.67 | 50.97        | 50.07        | −0.90 | 1.11×   |
    | Seg Medium | 45.00         | 44.03         | −0.97 | 53.48        | 52.26        | −1.23 | 1.08×   |
    | Seg Large  | 46.38         | 46.03         | −0.35 | 54.93        | 54.15        | −0.78 | 1.11×   |

    **Keypoints (keypoint AP)**

    | Model            | Keypoint AP, FP16 | Keypoint AP, INT8 | Δ     | Box AP, FP16 | Box AP, INT8 | Δ box | Speedup |
    | ---------------- | ----------------- | ----------------- | ----- | ------------ | ------------ | ----- | ------- |
    | Keypoint Preview | 71.72             | 71.11             | −0.61 | 51.51        | 50.95        | −0.56 | 1.11×   |

#### Latency per image

Latency is per image: the time of one call divided by its batch. Speedup is the FP16 latency divided by the INT8 latency, in the same mode, for the same model and batch.

=== "RTX 5070, plain call"

    **Detection**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Detection Nano   | 1     | 0.909 ms | 0.668 ms | 1.36×   |
    | Detection Nano   | 4     | 0.575 ms | 0.421 ms | 1.37×   |
    | Detection Nano   | 8     | 0.513 ms | 0.361 ms | 1.42×   |
    | Detection Small  | 1     | 1.412 ms | 1.064 ms | 1.33×   |
    | Detection Small  | 4     | 1.003 ms | 0.741 ms | 1.35×   |
    | Detection Small  | 8     | 0.925 ms | 0.672 ms | 1.38×   |
    | Detection Medium | 1     | 1.619 ms | 1.301 ms | 1.24×   |
    | Detection Medium | 4     | 1.193 ms | 0.904 ms | 1.32×   |
    | Detection Medium | 8     | 1.149 ms | 0.870 ms | 1.32×   |
    | Detection Large  | 1     | 2.009 ms | 1.762 ms | 1.14×   |
    | Detection Large  | 4     | 1.743 ms | 1.457 ms | 1.20×   |
    | Detection Large  | 8     | 1.756 ms | 1.433 ms | 1.23×   |

    **Segmentation**

    | Model      | Batch | FP16     | INT8     | Speedup |
    | ---------- | ----- | -------- | -------- | ------- |
    | Seg Nano   | 1     | 1.271 ms | 1.155 ms | 1.10×   |
    | Seg Nano   | 4     | 0.845 ms | 0.733 ms | 1.15×   |
    | Seg Nano   | 8     | 0.801 ms | 0.683 ms | 1.17×   |
    | Seg Small  | 1     | 1.613 ms | 1.312 ms | 1.23×   |
    | Seg Small  | 4     | 1.240 ms | 1.001 ms | 1.24×   |
    | Seg Small  | 8     | 1.254 ms | 1.021 ms | 1.23×   |
    | Seg Medium | 1     | 1.939 ms | 1.652 ms | 1.17×   |
    | Seg Medium | 4     | 1.623 ms | 1.353 ms | 1.20×   |
    | Seg Medium | 8     | 1.733 ms | 1.475 ms | 1.18×   |
    | Seg Large  | 1     | 2.396 ms | 2.167 ms | 1.11×   |
    | Seg Large  | 4     | 2.301 ms | 2.046 ms | 1.12×   |
    | Seg Large  | 8     | 2.429 ms | 2.158 ms | 1.13×   |

    **Keypoints**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Keypoint Preview | 1     | 3.222 ms | 2.784 ms | 1.16×   |
    | Keypoint Preview | 4     | 2.651 ms | 2.223 ms | 1.19×   |
    | Keypoint Preview | 8     | 2.598 ms | 2.167 ms | 1.20×   |

=== "RTX 5070, CUDA graph"

    **Detection**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Detection Nano   | 1     | 0.883 ms | 0.628 ms | 1.41×   |
    | Detection Nano   | 4     | 0.563 ms | 0.408 ms | 1.38×   |
    | Detection Nano   | 8     | 0.506 ms | 0.355 ms | 1.42×   |
    | Detection Small  | 1     | 1.369 ms | 1.030 ms | 1.33×   |
    | Detection Small  | 4     | 0.986 ms | 0.725 ms | 1.36×   |
    | Detection Small  | 8     | 0.917 ms | 0.665 ms | 1.38×   |
    | Detection Medium | 1     | 1.580 ms | 1.262 ms | 1.25×   |
    | Detection Medium | 4     | 1.177 ms | 0.889 ms | 1.32×   |
    | Detection Medium | 8     | 1.140 ms | 0.870 ms | 1.31×   |
    | Detection Large  | 1     | 1.961 ms | 1.714 ms | 1.14×   |
    | Detection Large  | 4     | 1.713 ms | 1.430 ms | 1.20×   |
    | Detection Large  | 8     | 1.742 ms | 1.420 ms | 1.23×   |

    **Segmentation**

    | Model      | Batch | FP16     | INT8     | Speedup |
    | ---------- | ----- | -------- | -------- | ------- |
    | Seg Nano   | 1     | 1.227 ms | 1.098 ms | 1.12×   |
    | Seg Nano   | 4     | 0.826 ms | 0.714 ms | 1.16×   |
    | Seg Nano   | 8     | 0.795 ms | 0.676 ms | 1.18×   |
    | Seg Small  | 1     | 1.569 ms | 1.272 ms | 1.23×   |
    | Seg Small  | 4     | 1.223 ms | 0.979 ms | 1.25×   |
    | Seg Small  | 8     | 1.249 ms | 1.017 ms | 1.23×   |
    | Seg Medium | 1     | 1.895 ms | 1.608 ms | 1.18×   |
    | Seg Medium | 4     | 1.598 ms | 1.330 ms | 1.20×   |
    | Seg Medium | 8     | 1.719 ms | 1.463 ms | 1.17×   |
    | Seg Large  | 1     | 2.350 ms | 2.120 ms | 1.11×   |
    | Seg Large  | 4     | 2.279 ms | 2.023 ms | 1.13×   |
    | Seg Large  | 8     | 2.414 ms | 2.143 ms | 1.13×   |

    **Keypoints**

    | Model            | Batch | FP16     | INT8     | Speedup |
    | ---------------- | ----- | -------- | -------- | ------- |
    | Keypoint Preview | 1     | 3.177 ms | 2.734 ms | 1.16×   |
    | Keypoint Preview | 4     | 2.619 ms | 2.194 ms | 1.19×   |
    | Keypoint Preview | 8     | 2.581 ms | 2.155 ms | 1.20×   |

=== "Tesla T4, plain call"

    **Detection**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Detection Nano   | 1     | 3.587 ms  | 2.864 ms  | 1.25×   |
    | Detection Nano   | 4     | 2.968 ms  | 2.197 ms  | 1.35×   |
    | Detection Nano   | 8     | 3.357 ms  | 2.400 ms  | 1.40×   |
    | Detection Small  | 1     | 5.500 ms  | 4.814 ms  | 1.14×   |
    | Detection Small  | 4     | 5.933 ms  | 4.259 ms  | 1.39×   |
    | Detection Small  | 8     | 6.328 ms  | 4.854 ms  | 1.30×   |
    | Detection Medium | 1     | 6.868 ms  | 5.998 ms  | 1.15×   |
    | Detection Medium | 4     | 7.580 ms  | 5.754 ms  | 1.32×   |
    | Detection Medium | 8     | 8.532 ms  | 6.558 ms  | 1.30×   |
    | Detection Large  | 1     | 10.688 ms | 9.543 ms  | 1.12×   |
    | Detection Large  | 4     | 12.610 ms | 10.282 ms | 1.23×   |
    | Detection Large  | 8     | 13.414 ms | 11.040 ms | 1.22×   |

    **Segmentation**

    | Model      | Batch | FP16      | INT8      | Speedup |
    | ---------- | ----- | --------- | --------- | ------- |
    | Seg Nano   | 1     | 5.290 ms  | 5.091 ms  | 1.04×   |
    | Seg Nano   | 4     | 5.029 ms  | 4.440 ms  | 1.13×   |
    | Seg Nano   | 8     | 5.233 ms  | 4.842 ms  | 1.08×   |
    | Seg Small  | 1     | 7.150 ms  | 6.442 ms  | 1.11×   |
    | Seg Small  | 4     | 7.568 ms  | 5.885 ms  | 1.29×   |
    | Seg Small  | 8     | 7.593 ms  | 6.411 ms  | 1.18×   |
    | Seg Medium | 1     | 9.883 ms  | 9.190 ms  | 1.08×   |
    | Seg Medium | 4     | 10.104 ms | 8.733 ms  | 1.16×   |
    | Seg Medium | 8     | 10.949 ms | 9.582 ms  | 1.14×   |
    | Seg Large  | 1     | 14.360 ms | 12.981 ms | 1.11×   |
    | Seg Large  | 4     | 15.147 ms | 13.659 ms | 1.11×   |
    | Seg Large  | 8     | 15.437 ms | 14.171 ms | 1.09×   |

    **Keypoints**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Keypoint Preview | 1     | 15.637 ms | 14.112 ms | 1.11×   |
    | Keypoint Preview | 4     | 17.333 ms | 15.178 ms | 1.14×   |
    | Keypoint Preview | 8     | 18.077 ms | 16.013 ms | 1.13×   |

=== "Tesla T4, CUDA graph"

    **Detection**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Detection Nano   | 1     | 3.653 ms  | 2.923 ms  | 1.25×   |
    | Detection Nano   | 4     | 2.898 ms  | 2.171 ms  | 1.33×   |
    | Detection Nano   | 8     | 3.378 ms  | 2.403 ms  | 1.41×   |
    | Detection Small  | 1     | 5.485 ms  | 4.684 ms  | 1.17×   |
    | Detection Small  | 4     | 5.932 ms  | 4.230 ms  | 1.40×   |
    | Detection Small  | 8     | 6.390 ms  | 4.885 ms  | 1.31×   |
    | Detection Medium | 1     | 6.964 ms  | 5.895 ms  | 1.18×   |
    | Detection Medium | 4     | 7.618 ms  | 5.803 ms  | 1.31×   |
    | Detection Medium | 8     | 8.535 ms  | 6.559 ms  | 1.30×   |
    | Detection Large  | 1     | 10.795 ms | 9.418 ms  | 1.15×   |
    | Detection Large  | 4     | 12.638 ms | 10.447 ms | 1.21×   |
    | Detection Large  | 8     | 13.392 ms | 10.995 ms | 1.22×   |

    **Segmentation**

    | Model      | Batch | FP16      | INT8      | Speedup |
    | ---------- | ----- | --------- | --------- | ------- |
    | Seg Nano   | 1     | 5.642 ms  | 5.244 ms  | 1.08×   |
    | Seg Nano   | 4     | 5.018 ms  | 4.391 ms  | 1.14×   |
    | Seg Nano   | 8     | 5.219 ms  | 4.833 ms  | 1.08×   |
    | Seg Small  | 1     | 7.186 ms  | 6.508 ms  | 1.10×   |
    | Seg Small  | 4     | 7.567 ms  | 5.816 ms  | 1.30×   |
    | Seg Small  | 8     | 7.596 ms  | 6.396 ms  | 1.19×   |
    | Seg Medium | 1     | 9.885 ms  | 9.044 ms  | 1.09×   |
    | Seg Medium | 4     | 10.092 ms | 8.688 ms  | 1.16×   |
    | Seg Medium | 8     | 10.949 ms | 9.589 ms  | 1.14×   |
    | Seg Large  | 1     | 14.309 ms | 12.982 ms | 1.10×   |
    | Seg Large  | 4     | 15.147 ms | 13.611 ms | 1.11×   |
    | Seg Large  | 8     | 15.399 ms | 14.170 ms | 1.09×   |

    **Keypoints**

    | Model            | Batch | FP16      | INT8      | Speedup |
    | ---------------- | ----- | --------- | --------- | ------- |
    | Keypoint Preview | 1     | 15.303 ms | 13.921 ms | 1.10×   |
    | Keypoint Preview | 4     | 17.325 ms | 15.177 ms | 1.14×   |
    | Keypoint Preview | 8     | 18.104 ms | 16.013 ms | 1.13×   |

On the RTX 5070, the models whose backbone attention runs in INT8 (detection Nano, Small and Medium, Seg Small and Seg Medium; see the placement rules below) gain 1.18× to 1.41× at batch 1 with a CUDA graph. The ones that keep it in FP16 (detection Large, Seg Nano, Seg Large and Keypoint Preview) gain 1.11× to 1.16×. For detection the gain at batch 8 is 0.02× to 0.08× above the gain at batch 1 (Nano dips at batch 4 and Medium peaks there); for Seg Nano it grows by 0.06×; for Seg Small, Seg Medium, Seg Large and Keypoint Preview it changes by at most 0.04× between batch 1, 4 and 8. On the Tesla T4 that split does not hold for segmentation: with a CUDA graph at batch 1, Seg Small (1.10×) and Seg Medium (1.09×), which run INT8 attention, gain about the same as Seg Large (1.10×) and Keypoint Preview (1.10×), which keep it in FP16. The cause was not measured. For detection on the Tesla T4 the split is small: at batch 1 with a CUDA graph Nano gains 1.25×, while Small (1.17×) and Medium (1.18×) are within 0.03× of Large (1.15×). On both GPUs the plain call and the CUDA graph give almost the same speedup: they differ by at most 0.05× on the RTX 5070 and on the Tesla T4.

#### Engine size

On the Tesla T4 the segmentation and keypoint INT8 engines no longer grow with the batch. Each stays within 3 MB from batch 1 to batch 8 (Keypoint Preview 60, 59 and 62 MB; Seg Small 44 MB at all three), while the FP16 engine of the same model grows by 7 to 32 MB (Keypoint Preview 81 to 113 MB), so the INT8 engine is 0.52× to 0.74× of the FP16 one. Before the patch embedding became a matrix multiply, the INT8 engines of the same five models on the same GPU and TensorRT version were larger than the FP16 ones at batch 4 and 8, up to 193 MB against 94 MB for Seg Large and 248 MB against 113 MB for Keypoint Preview at batch 8. The growth beyond the FP16 engine's from batch 1 to 8 matched the 3-channel image padded to 32 channels in FP16, `7 × 29 × height × width × 2` bytes, to within 5 MB (arithmetic on the earlier table, not a reading of an engine). The patch embedding was the only change to the export, but the earlier rows came from another Colab session, so the comparison is not a controlled A/B: it makes the patch embedding the likely cause of the growth, not a proven one. The FP16 export is unchanged, yet the FP16 engine sizes differ by up to 3 MB from the earlier table (Seg Nano at batch 1: 69 MB then, 66 MB now), which we attribute to the two sessions. The FP16 engine's growth with the batch was not investigated. The Tesla T4 detection INT8 engines are 0.41× to 0.69× of the FP16 one at every batch and stay between 37 and 48 MB from batch 1 to batch 8, while the FP16 one grows with the batch (Large: 70 MB at batch 1, 111 MB at batch 8). On the RTX 5070 the INT8 engine is smaller than the FP16 one in every row. The RTX 5070 segmentation and keypoint INT8 engines are 0 to 5 MB smaller than in the table before the matrix multiply, and the saving does not fall as the batch grows (Keypoint Preview at batch 8: 71 MB then, 66 MB now); that table came from another machine with an RTX 5070, so this is not a controlled A/B either.

The size of an engine is not a function of the batch alone. On the RTX 5070 the same model varies by up to 11 MB between batch 1, 4 and 8, and not always in step with the batch, and the size depends on the GPU and the TensorRT version. `INT8 ÷ FP16` is below 1 when the INT8 engine is the smaller one.

=== "RTX 5070"

    | Model            | Batch | FP16  | INT8  | INT8 ÷ FP16 |
    | ---------------- | ----- | ----- | ----- | ----------- |
    | Detection Nano   | 1     | 62 MB | 38 MB | 0.61×       |
    | Detection Nano   | 4     | 66 MB | 38 MB | 0.57×       |
    | Detection Nano   | 8     | 72 MB | 39 MB | 0.53×       |
    | Detection Small  | 1     | 66 MB | 41 MB | 0.61×       |
    | Detection Small  | 4     | 74 MB | 41 MB | 0.55×       |
    | Detection Small  | 8     | 68 MB | 41 MB | 0.61×       |
    | Detection Medium | 1     | 71 MB | 43 MB | 0.61×       |
    | Detection Medium | 4     | 70 MB | 44 MB | 0.63×       |
    | Detection Medium | 8     | 72 MB | 44 MB | 0.61×       |
    | Detection Large  | 1     | 68 MB | 49 MB | 0.72×       |
    | Detection Large  | 4     | 72 MB | 49 MB | 0.69×       |
    | Detection Large  | 8     | 76 MB | 50 MB | 0.65×       |
    | Seg Nano         | 1     | 69 MB | 51 MB | 0.73×       |
    | Seg Nano         | 4     | 67 MB | 50 MB | 0.76×       |
    | Seg Nano         | 8     | 70 MB | 51 MB | 0.72×       |
    | Seg Small        | 1     | 69 MB | 46 MB | 0.66×       |
    | Seg Small        | 4     | 69 MB | 45 MB | 0.65×       |
    | Seg Small        | 8     | 71 MB | 46 MB | 0.65×       |
    | Seg Medium       | 1     | 72 MB | 49 MB | 0.67×       |
    | Seg Medium       | 4     | 74 MB | 49 MB | 0.66×       |
    | Seg Medium       | 8     | 75 MB | 49 MB | 0.65×       |
    | Seg Large        | 1     | 73 MB | 54 MB | 0.74×       |
    | Seg Large        | 4     | 75 MB | 54 MB | 0.72×       |
    | Seg Large        | 8     | 77 MB | 54 MB | 0.70×       |
    | Keypoint Preview | 1     | 81 MB | 61 MB | 0.76×       |
    | Keypoint Preview | 4     | 85 MB | 64 MB | 0.75×       |
    | Keypoint Preview | 8     | 90 MB | 66 MB | 0.73×       |

=== "Tesla T4"

    | Model            | Batch | FP16   | INT8  | INT8 ÷ FP16 |
    | ---------------- | ----- | ------ | ----- | ----------- |
    | Detection Nano   | 1     | 61 MB  | 37 MB | 0.61×       |
    | Detection Nano   | 4     | 64 MB  | 38 MB | 0.59×       |
    | Detection Nano   | 8     | 70 MB  | 38 MB | 0.54×       |
    | Detection Small  | 1     | 65 MB  | 40 MB | 0.62×       |
    | Detection Small  | 4     | 72 MB  | 40 MB | 0.56×       |
    | Detection Small  | 8     | 85 MB  | 40 MB | 0.47×       |
    | Detection Medium | 1     | 68 MB  | 42 MB | 0.62×       |
    | Detection Medium | 4     | 79 MB  | 43 MB | 0.54×       |
    | Detection Medium | 8     | 94 MB  | 43 MB | 0.46×       |
    | Detection Large  | 1     | 70 MB  | 48 MB | 0.69×       |
    | Detection Large  | 4     | 87 MB  | 46 MB | 0.53×       |
    | Detection Large  | 8     | 111 MB | 46 MB | 0.41×       |
    | Seg Nano         | 1     | 66 MB  | 49 MB | 0.74×       |
    | Seg Nano         | 4     | 70 MB  | 47 MB | 0.67×       |
    | Seg Nano         | 8     | 73 MB  | 47 MB | 0.64×       |
    | Seg Small        | 1     | 67 MB  | 44 MB | 0.65×       |
    | Seg Small        | 4     | 71 MB  | 44 MB | 0.62×       |
    | Seg Small        | 8     | 80 MB  | 44 MB | 0.56×       |
    | Seg Medium       | 1     | 71 MB  | 47 MB | 0.66×       |
    | Seg Medium       | 4     | 77 MB  | 47 MB | 0.61×       |
    | Seg Medium       | 8     | 87 MB  | 45 MB | 0.52×       |
    | Seg Large        | 1     | 72 MB  | 52 MB | 0.72×       |
    | Seg Large        | 4     | 80 MB  | 50 MB | 0.63×       |
    | Seg Large        | 8     | 94 MB  | 50 MB | 0.54×       |
    | Keypoint Preview | 1     | 81 MB  | 60 MB | 0.74×       |
    | Keypoint Preview | 4     | 95 MB  | 59 MB | 0.63×       |
    | Keypoint Preview | 8     | 113 MB | 62 MB | 0.55×       |

### Placement rules

The INT8 placement follows what was measured, not "quantize every matrix multiply":

- **INT8:** the patch embedding of the detection models with 16-pixel patches (Nano to Large), every backbone MLP, the windowed backbone blocks' attention with its projections (up to 325 tokens; see below), and the decoder's cross-attention, feed-forward and reference-point layers. Every layer left unquantized stays FP16.
- **FP16:** the projector, the two-stage proposal head and the detection heads (quantizing them cost accuracy and bought nothing), the mask head (quantized as a whole it made the engine slower on Seg Nano, Seg Small and Seg Large, and its matrix multiplies alone were no faster) and the keypoint heads (quantizing the keypoint embedding of Keypoint Preview was no faster and lowered its keypoint AP), plus the attention blocks below. Quantizing the projector was no faster on Seg Nano, Seg Small, Seg Large or Keypoint Preview either.
- **The patch embedding is a matrix multiply; the detection models run it in INT8.** TensorRT cannot run the 3-channel patch convolution in INT8 without padding its input to 32 channels and storing the added zero channels in the engine as a constant that grows with the batch, which made the engine larger and the INT8 speedup smaller (an FP16 convolution can be padded too: the Tesla T4 segmentation and keypoint INT8 engines, whose patch embedding was an FP16 convolution, grew with the batch by about the image padded to 32 channels, inferred from the engine sizes, not read from an engine; with the matrix multiply they no longer grow, see [Engine size](#engine-size)). So every convolution whose input channels are not a multiple of 32 stays FP16, and the export rewrites the patch embedding as a matrix multiply, which has nothing to pad. The Nano to Large detection models cut 16-pixel patches of 768 values, a multiple of 32, and the matrix multiply is quantized like the other backbone layers. The segmentation and keypoint models cut 12-pixel patches of 432 values, which is not a multiple of 32: their matrix multiply stays FP16, because as INT8 it ran no faster than the FP16 one and lowered the accuracy on Seg Small and Keypoint Preview. Any convolution that is not a non-overlapping patch embedding stays an FP16 convolution.
- **The detection models' backbone norms are smoothed (SmoothQuant).** A backbone LayerNorm output is quantized with one INT8 step for the whole tensor, and its widest channels, several times wider than its median channel, set that step for every channel. Following [SmoothQuant](https://arxiv.org/abs/2211.10438), the export divides each channel of the norm's scale and bias by a factor measured on the calibration images, `sqrt(activation range / weight range)` normalized over the channels, and multiplies the matching rows of the projections that read the norm by it. The weights are quantized per output channel and take the range easily. The model computes the same outputs, and the engine has no extra layer.
- **The detection models' backbone MLPs get their bias corrected.** Rounding a layer's input to INT8 shifts the mean of its output, most of all for the one-sided GELU output the second MLP projection reads. The export measures that shift per output channel on the calibration images and subtracts it from the layer's bias. Again only constants change. Together the two steps cut the box AP lost at batch 1 on the RTX 5070 from 0.66, 0.71, 0.94 and 0.60 points to 0.50, 0.47, 0.66 and 0.32 for detection Nano, Small, Medium and Large, with the same latency and engine size (Section 8.1 of the CUDA cookbook, run in one session on this export and on the one before it, commit `5b22dfc0`; TensorRT 11.4). On the Tesla T4 (TensorRT 11.4, this export only) the loss is 0.50, 0.36, 0.65 and 0.27 points; the T4 run of the export before it measured 0.68, 0.66, 0.80 and 0.50, but on TensorRT 10.16.1.11, so that difference is not shown to come from the two steps alone. On Nano, correcting the attention and decoder layers' biases too measured no better. A new layer on the GELU path was the alternative and is not safe: on TensorRT 11.4 even a no-op subtraction there changed how TensorRT fuses the first MLP projection and cost more AP than the whole INT8 plan does.
- **Attention is all INT8 or all FP16.** A block with head size 16, 32 or 64 and at most 325 tokens gets INT8 projections and INT8 attention matrix multiplies, which TensorRT fuses into one INT8 attention kernel: the windowed backbone blocks of detection Nano (145 tokens), Small (257) and Medium (325), and of Seg Small (257) and Seg Medium (325). 325 tokens is the largest window where INT8 attention was measured to pay; on Large's 485-token windows it was no faster and less accurate, so Large keeps them FP16. On Seg Large's 442-token windows it was a little faster and smaller but cost more accuracy than that is worth, so they stay FP16; on Keypoint Preview's 577 and Seg Nano's 677 tokens, above the fused kernel's 512, it was much slower. Other windows of 326 to 484 tokens are not measured and stay FP16. The global blocks (580 tokens on Nano) and the decoder self-attention keep projections and attention in FP16 too. A windowed block has `(resolution / patch_size / num_windows)² + 1` tokens, so a higher export resolution can move a model's attention to FP16; the export logs how many backbone attention blocks stay FP16.

### Requirements and limits

- Detection, segmentation and keypoint models. The smoothing and the bias correction apply to the detection models only: the segmentation and keypoint models share the backbone but were not measured with them, so their engines are built as before. The mask head and the keypoint embedding stay FP16 (the mask head was measured: quantized as a whole it made the engine slower, and its matrix multiplies alone were no faster; the keypoint embedding was measured on Keypoint Preview: no faster, and a lower keypoint AP), and so does the keypoint decoder's self-attention between each detection and its keypoints (`kp_inst_self_attn`); the keypoint decoder's cross-attention and feed-forward layers run in INT8.
- The attention rule above applies unchanged. Only `RFDETRSegSmall` (257 tokens per window) and `RFDETRSegMedium` (325) run INT8 attention, in the 9 windowed blocks of their 12 backbone blocks (the other 3 attend globally). `RFDETRSegNano` (one 677-token window per block), `RFDETRSegLarge` (442) and `RFDETRKeypointPreview` (577) keep all attention in FP16, so only their backbone MLPs and the decoder's cross-attention, feed-forward and reference-point layers run in INT8.
- `RFDETRSegXLarge` and `RFDETRSeg2XLarge` (`rfdetr_plus`) are accepted but were not measured. The detection models of `rfdetr_plus` get the norm smoothing and the bias correction wherever their graph has the layers these rules look for, also unmeasured. `backbone_only=True` is refused with `ValueError`.
- A static batch (`dynamic_batch=True` is refused), `fp16=True`, and TensorRT 10 or newer. Every table above was measured on TensorRT 11.4. The detection placement without the patch-embedding matrix multiply was also measured on 10.16 and 11.3; the matrix multiply itself on 11.4 (RTX 5070 and Tesla T4) and on 10.16 (Tesla T4); the norm smoothing and the MLP bias correction only on 11.4 (RTX 5070 and Tesla T4); the segmentation and keypoint matrix multiply, which stays FP16, on 11.4 on the RTX 5070 and the Tesla T4 for all five models (the tables above) and, as a separate check of whether it should run in INT8, on the RTX 5070 for Seg Nano, Seg Small and Keypoint Preview. Other patch sizes whose flattened patch is not a multiple of 32 values, such as 14 pixels (588 values), take the same FP16 matrix multiply and were not measured either. The segmentation and keypoint models were measured only on 11.4; the other releases from 10 on are accepted but untested. The batch is fixed when you build the engine.
- Only an RTX 5070 (sm_120) and a Tesla T4 (sm_75) were measured, and only without `trt_hardware_compatibility`, `trt_version_compatible` or a shared `trt_timing_cache`. These settings are accepted with `quantization="int8"`, but engines built with them were not measured, and a portable engine may not get the fused INT8 attention kernel the speedup relies on.
- INT8 attention relies on TensorRT's fused INT8 attention kernel, which NVIDIA lists for sm_75 to sm_90, sm_120 and sm_121. On other GPUs (for example sm_100, B200) it would run unfused, so measure on your GPU before deploying.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact for raw TensorRT deployment. By default it is locked to the GPU architecture and TensorRT version of the machine that built it, so it is not portable across different GPUs or TensorRT releases; see [Portable Engines](#portable-engines) for the options that widen that.

    If you plan to run inference with [`inference-models`](basics.md#run-inference-with-inference-models) (a separate deployment option), do **not** pass `format="tensorrt"` — `inference-models` builds and manages its own TensorRT engine internally and does not consume this file. Export a plain ONNX model instead and let `inference-models` handle the backend.

## Measured Latency on an NVIDIA L4

Batch 1, 4 and 8 on a Colab NVIDIA L4, `RFDETRSmall`, `rfdetr` 1.12.0.dev0, TensorRT 10.16.1.11, 20 warmup + 100 timed runs per row, from the [CUDA cookbook](../../cookbooks/export-cuda/). Every latency is the end-to-end `predict()` call (preprocessing, forward, postprocessing) through `RFDETRInference`, except the two rows from the separate `inference-models` package, which selects and loads its own prebuilt TensorRT engine and runs its own call path, not the `.trt` file this page exports. The `Package` column says which package ran each row: `rfdetr` rows are this package (`model.export()` plus `RFDETRInference`), `inference-models` rows are not. `FPS` is images per second. `mAP@50:95` is a seeded 500-image COCO val2017 subset, scored at batch 1 on the same artifact that was timed; the `inference-models` rows are not scored. `Memory [MB]` is device-memory growth; the batch-4 and batch-8 rows can read `0.0` because the engine sizes its output buffers for the batch-8 profile maximum when it is loaded. The ONNX Runtime rows from the same run are on the [ONNX page](onnx.md#measured-latency), the PyTorch baseline on the [export overview](index.md#pytorch-baseline).

| Package            | Runtime        | Precision | Batch | end2end [ms] | FPS [img/s] | mAP@50:95 [%] | Size [MB] | Memory [MB] |
| ------------------ | -------------- | --------- | ----- | ------------ | ----------- | ------------- | --------- | ----------- |
| `inference-models` | managed engine | auto      | 1     | 3.45 ± 0.15  | 289.8       | —             | —         | 119.5       |
| `inference-models` | managed engine | fp32      | 1     | 6.67 ± 0.10  | 149.9       | —             | —         | 228.6       |
| `rfdetr`           | .trt engine    | fp16      | 1     | 5.88 ± 0.34  | 169.9       | 56.5          | 63.0      | 182.5       |
| `inference-models` | managed engine | auto      | 4     | 11.15 ± 0.35 | 358.7       | —             | —         | 0.0         |
| `inference-models` | managed engine | fp32      | 4     | 25.31 ± 0.52 | 158.1       | —             | —         | 0.0         |
| `rfdetr`           | .trt engine    | fp16      | 4     | 17.58 ± 0.48 | 227.5       | —             | 63.0      | 0.0         |
| `inference-models` | managed engine | auto      | 8     | 20.49 ± 0.75 | 390.4       | —             | —         | 25.2        |
| `inference-models` | managed engine | fp32      | 8     | 49.86 ± 0.65 | 160.5       | —             | —         | 0.0         |
| `rfdetr`           | .trt engine    | fp16      | 8     | 33.46 ± 0.64 | 239.1       | —             | 63.0      | 25.2        |

What the table supports:

- The `.trt` engine through `RFDETRInference` takes 5.88 ms end to end at batch 1, 4.0 times faster than eager PyTorch (23.46 ms) and 2.6 times faster than ONNX Runtime on CUDA (15.45 ms), at the same mAP (56.5).
- The managed `inference-models` engine is faster still (3.45 ms, 289.8 FPS at batch 1, 390.4 FPS at batch 8) and scales best with batch; forcing it to fp32 doubles the time (6.67 ms).
- The `.trt` engine scales with batch: 169.9, 227.5 and 239.1 FPS at batch 1, 4 and 8.
- The fp16 `.trt` engine scores the same mAP as eager PyTorch fp32 (56.5), so fp16 costs no accuracy on the subset.

!!! note "Not comparable with other L4 figures"

    These times include preprocessing and postprocessing on the host. They are not the engine-only times of the INT8 benchmark above, and not the 8.3 ms TensorRT and 12.9 ms ONNX figures on Roboflow's model page, whose timing scope is not stated.

A quick INT8 check on the same L4 (`RFDETRSmall`, static batch-1 engines, 128 calibration images held out of the scored images, end-to-end `predict()`) gave 56.0 mAP@50:95 for INT8 against 56.5 for FP16, 5.19 against 5.33 ms (1.03 times), and a 39.6 MB engine against 61.8 MB. End to end the gain is small because preprocessing and postprocessing dominate a call this short. Timed on the engine alone with a CUDA graph, as in the tables above, INT8 was 1.12 times faster at batch 1 and 1.32 times at batch 8 (1.748 against 1.953 ms per image at batch 1) for 0.09 box AP. This is the quick default of Section 8.1 (one model, 500 images, `val2017` calibration), not the full benchmark behind the RTX 5070 and Tesla T4 tables, so it is not added to them.

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
