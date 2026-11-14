---
description: Export RF-DETR models to a TensorRT engine from PyTorch for low-latency inference on NVIDIA GPUs.
---

# TensorRT Export

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

!!! important

    Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference, or build a [portable engine](#portable-engines).

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

    Pass `dynamic_batch=True` together with `max_batch_size` to build one engine that accepts any batch from 1 to `max_batch_size`. The engine gets a single optimization profile with `min=1`, `opt=batch_size` and `max=max_batch_size`, so set `batch_size` to the batch you serve most often: other sizes in the range run, but TensorRT tunes its kernels for `opt`. Without `dynamic_batch` the engine accepts only the batch size baked into the ONNX graph.

    ```python
    model.export(format="tensorrt", dynamic_batch=True, batch_size=4, max_batch_size=16)
    ```

    One profile spanning `1 .. max_batch_size` is a deliberate trade-off: it keeps the export API and the runtime simple (no profile selection in the caller) at a small measured cost at `opt`. A deployment that never serves small batches still pays for kernels tuned down to batch 1; per-band multi-profile support (`min_batch_size`, several `opt_batch_sizes`) may follow if a need appears (see [#376](https://github.com/roboflow/rf-detr/issues/376)).

## INT8

Pass `quantization="int8"` and a directory of representative images to build an engine that runs most of the backbone encoder and decoder in INT8 and the rest in FP16:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="tensorrt", quantization="int8", calibration_data="path/to/images", batch_size=8)
```

This writes `output/rfdetr-small_int8.trt` (next to the intermediate `output/rfdetr-small.onnx`). The export runs the first `max_images` (default `100`) calibration images by file name through the FP32 graph with onnxruntime on the CPU, preprocessed exactly as `predict()` does, and records each quantized tensor's largest absolute value. For a detection model it makes two more passes: one to measure how wide each backbone LayerNorm output is per channel (so the widest channels can be folded into the weights that read them) and one to measure how much INT8 shifts each backbone MLP layer's mean output (subtracted from the layer's bias); see the [placement rules](#placement-rules).

Those ranges decide the engine's accuracy, so use images from the deployment domain: out-of-domain images give an engine that loads, runs and is quietly less accurate. `calibration_data` also accepts a `.npy` path or an array shaped `(N, C, H, W)`, already normalized the way `predict()` normalizes (an integer array of raw pixels is refused). `max_images` caps only a directory of images; an array or `.npy` file is used whole. Calibration runs the graph at the export's `batch_size`, so its host memory grows with the batch.

### Results

The numbers come from [Section 8.1 of the CUDA cookbook](../../cookbooks/export-cuda/#81-benchmark-for-the-docs-tables), run on a Tesla T4 (Colab) with TensorRT 11.4.0.106. The three tabs hold every model.

<!-- COPIES: these tables come from Section 8.1 of docs/cookbooks/export-cuda.ipynb. Any range quoted from it elsewhere (INT8 summary text, docs/exports/index.md) must match it. Update them together. -->

- **AP** is scored at batch 1 on all 5000 `val2017` images, decoded the way `predict()` decodes. Detection uses box AP, segmentation mask AP, Keypoint Preview keypoint AP. Δ is INT8 minus FP16 in AP points.
- **Latency** is GPU time per image at batch 1 with a CUDA graph (CUDA events, median of 15 rounds of 200 executions, FP16 and INT8 interleaved). It is engine-only time, not the end-to-end time of the [latency section](#latency-across-nvidia-gpus) below. Speedup is FP16 ÷ INT8.
- **Not the Benchmarks page:** the T4 FP16 figures here (Nano 3.65 ms, Medium 6.96, Large 10.80) are about 1.6× the model-only T4 figures of the [Benchmarks page](../learn/benchmarks.md) (2.3, 4.4, 6.8 ms). That page uses TensorRT 10.4 and a 200 ms pause between forward passes; these runs use TensorRT 11.4 and back-to-back executions on a Colab T4. Which of the two differences causes the gap was not isolated, so compare within a page.
- **API:** these tables do **not** go through `RFDETRInference`. Latency calls the raw TensorRT engine (`execute_async_v3`, replayed as a CUDA graph) from Section 8.1 of the cookbook, so it excludes preprocessing, postprocessing and `RFDETRInference` overhead. The run predates the current `RFDETRInference` and was not repeated through it. Treat the speedup ratios and AP deltas as the finding, and the absolute milliseconds as engine-only figures for that machine. The end-to-end `RFDETRInference` figures are in [Latency Across NVIDIA GPUs](#latency-across-nvidia-gpus), which has no INT8 row.
- **Size** is the engine file at batch 1. INT8 engines are calibrated on the first 128 `train2017` images by file name.

=== "Detection"

    | Model  | Box AP, FP16 | Box AP, INT8 | Δ     | FP16 ms | INT8 ms | Speedup | Size, FP16 → INT8 |
    | ------ | ------------ | ------------ | ----- | ------- | ------- | ------- | ----------------- |
    | Nano   | 48.07        | 47.57        | −0.50 | 3.653   | 2.923   | 1.25×   | 61 → 37 MB        |
    | Small  | 52.80        | 52.44        | −0.36 | 5.485   | 4.684   | 1.17×   | 65 → 40 MB        |
    | Medium | 54.67        | 54.02        | −0.65 | 6.964   | 5.895   | 1.18×   | 68 → 42 MB        |
    | Large  | 56.53        | 56.26        | −0.27 | 10.795  | 9.418   | 1.15×   | 70 → 48 MB        |

=== "Segmentation"

    | Model  | Mask AP, FP16 | Mask AP, INT8 | Δ     | FP16 ms | INT8 ms | Speedup | Size, FP16 → INT8 |
    | ------ | ------------- | ------------- | ----- | ------- | ------- | ------- | ----------------- |
    | Nano   | 40.07         | 39.50         | −0.58 | 5.642   | 5.244   | 1.08×   | 66 → 49 MB        |
    | Small  | 42.89         | 42.22         | −0.67 | 7.186   | 6.508   | 1.10×   | 67 → 44 MB        |
    | Medium | 45.00         | 44.03         | −0.97 | 9.885   | 9.044   | 1.09×   | 71 → 47 MB        |
    | Large  | 46.38         | 46.03         | −0.35 | 14.309  | 12.982  | 1.10×   | 72 → 52 MB        |

=== "Keypoints"

    | Model   | Keypoint AP, FP16 | Keypoint AP, INT8 | Δ     | FP16 ms | INT8 ms | Speedup | Size, FP16 → INT8 |
    | ------- | ----------------- | ----------------- | ----- | ------- | ------- | ------- | ----------------- |
    | Preview | 71.72             | 71.11             | −0.61 | 15.303  | 13.921  | 1.10×   | 81 → 60 MB        |

What the table supports:

- INT8 is faster than FP16 for every model (1.08× to 1.25× at batch 1), loses at most 0.97 AP points, and the engine is 0.61× to 0.74× the FP16 size.
- The gain is largest for detection Nano, Small and Medium (1.17× to 1.25×), where the backbone attention runs in INT8. Segmentation gains 1.08× to 1.10×; the cause of the weaker segmentation gain was not measured.
- Batches 4 and 8 give similar speedups (up to 1.41×), and a plain call (no CUDA graph) is within 0.05× of the graph figure. The cookbook prints all of them.
- The FP16 engine grows with the batch on the T4 (Detection Large: 70 MB at batch 1, 111 MB at batch 8) while the INT8 engine stays within a few MB. Engine size also depends on the GPU and TensorRT version, so use these as ratios.

### Placement rules

The INT8 placement follows what was measured, not "quantize every matrix multiply":

- **INT8:** the patch embedding of the detection models with 16-pixel patches (Nano to Large), every backbone MLP, the windowed backbone blocks' attention with its projections (up to 325 tokens), and the decoder's cross-attention, feed-forward and reference-point layers. Every layer left unquantized stays FP16.
- **FP16:** the projector, the two-stage proposal head, the detection heads, the mask head and the keypoint heads (quantizing them cost accuracy or speed and bought nothing), plus the attention blocks below.
- **The patch embedding is a matrix multiply.** TensorRT cannot run the 3-channel patch convolution in INT8 without padding its input to 32 channels and storing the zeros in the engine, growing it with the batch. So every convolution whose input channels are not a multiple of 32 stays FP16, and the export rewrites the patch embedding as a matrix multiply, which has nothing to pad. The detection models' 768-value patches are a multiple of 32 and run INT8. The segmentation and keypoint models' 432-value patches are not, so their matrix multiply stays FP16 (as INT8 it was no faster and less accurate). Other patch sizes that are not a multiple of 32, such as 14 pixels (588 values), take the same FP16 path and were not measured.
- **Detection backbone norms are smoothed ([SmoothQuant](https://arxiv.org/abs/2211.10438)).** One INT8 step per LayerNorm output is set by its widest channels. The export divides each channel of the norm's scale and bias by a factor measured on the calibration images and multiplies the matching rows of the projections that read it. The outputs are the same and the engine has no extra layer.
- **Detection backbone MLP biases are corrected.** Rounding a layer's input to INT8 shifts the mean of its output, most of all for the one-sided GELU output the second MLP projection reads. The export measures that shift per output channel and subtracts it from the bias. On an RTX 5070 (a measurement not repeated on the table above) the two steps cut the box AP lost at batch 1 from 0.66, 0.71, 0.94 and 0.60 points to 0.50, 0.47, 0.66 and 0.32 (Nano to Large), at the same latency and size. Keep any new op off the GELU → second-projection path: on TensorRT 11.4 even a no-op subtraction there changed the fusion and cost more AP than the whole INT8 plan.
- **Attention is all INT8 or all FP16.** A block with head size 16, 32 or 64 and at most 325 tokens gets INT8 projections and INT8 attention matrix multiplies, fused into one INT8 attention kernel. That covers the windowed blocks of detection Nano (145 tokens), Small (257) and Medium (325), and Seg Small (257) and Seg Medium (325). Detection Large (485), Seg Large (442), Keypoint Preview (577) and Seg Nano (677) keep attention in FP16: INT8 was no faster or less accurate at 442 to 485 tokens and much slower above the fused kernel's 512. Windows of 326 to 484 tokens that are not listed were not measured and stay FP16, as do global blocks and the decoder self-attention. A windowed block has `(resolution / patch_size / num_windows)² + 1` tokens, so a higher export resolution can move a model's attention to FP16; the export logs how many backbone attention blocks stay FP16.

### Requirements and limits

- Detection, segmentation and keypoint models. The smoothing and bias correction apply to detection models only. `RFDETRSegXLarge`, `RFDETRSeg2XLarge` and the `rfdetr_plus` detection models are accepted but were not measured. `backbone_only=True` is refused with `ValueError`.
- A static batch (`dynamic_batch=True` is refused), `fp16=True`, and TensorRT 10 or newer. The batch is fixed when you build the engine.
- All tables above were measured on TensorRT 11.4. The detection placement without the patch-embedding matrix multiply was also checked on 10.16 and 11.3; the segmentation and keypoint models only on 11.4, and other releases from 10 on are accepted but untested.
- Only a Tesla T4 (sm_75) was measured for engine-only speed (the end-to-end INT8 rows of [Latency Across NVIDIA GPUs](#latency-across-nvidia-gpus) also cover the L4, A100 and RTX PRO 6000), and only without `trt_hardware_compatibility`, `trt_version_compatible` or a shared `trt_timing_cache`. Those settings are accepted with `quantization="int8"`, but a portable engine may not get the fused INT8 attention kernel the speedup relies on.
- The fused INT8 attention kernel is listed by NVIDIA for sm_75 to sm_90, sm_120 and sm_121. On other GPUs (for example sm_100, B200) attention would run unfused, so measure before deploying.

!!! note "Who consumes the `.trt` engine?"

    The `.trt` engine produced by `format="tensorrt"` is a standalone artifact, loaded with `RFDETRInference` or raw TensorRT. By default it is locked to the GPU architecture and TensorRT version of the machine that built it; see [Portable Engines](#portable-engines) for options that widen that.

## Latency Across NVIDIA GPUs

The same end-to-end measurement for the four detection sizes on four GPUs, from one run per GPU, timed with the same `RFDETRInference` calls as the [CUDA cookbook](../../cookbooks/export-cuda/), which times one size per run, set by `MODEL_SIZE`. Its Section 8 subsection *Batch 1, 4 and 8, end to end* reproduces the `.trt` FP16 and INT8 rows, with one static engine per batch size. All ran `rfdetr` 1.12.0.dev0, PyTorch 2.15.0 with CUDA 13.2 and TensorRT 10.16.1.11, with 20 warmup and 100 timed runs per row. Every row times one decoded RGB NumPy array through preprocessing, forward, postprocessing and conversion to `sv.Detections`, and passes `include_source_image=False`.

The [latency cookbook](../../cookbooks/measure-latency/) times every pretrained size on the same four GPUs, as the median of 500 images with TensorRT 11.4.0.106, at batch 1 only. Its tables are on the Run pages for [detection](../learn/run/detection.md#fastest-configuration-on-nvidia-gpus), [segmentation](../learn/run/segmentation.md#fastest-configuration-on-nvidia-gpus) and [keypoints](../learn/run/keypoints.md#fastest-configuration-on-nvidia-gpus). They do not time the managed engine or batch 8, and they use a different protocol and TensorRT version from the tables below, so compare numbers only within one set.

--8<-- "docs/snippets/latency-routes.md"

- **Latency only.** No accuracy was scored; the INT8 accuracy is in [Results](#results), whose latency is engine-only time and so not comparable with these tables.
- Each cell is `latency (throughput)`, for example `5.12 ms (334 img/s)`: latency in ms at **batch 1**, throughput in img/s at **batch 8**. Lower latency is better, higher throughput is better, so the two numbers move in opposite directions.

<!-- COPIES: the four GPU tabs are mirrored by docs/exports/index.md, "Fastest route across NVIDIA GPUs" (the fastest of the graph, FP16, INT8 and managed routes per cell, ms at batch 1, img/s at batch 8; FP32 is never the fastest) and the "NVIDIA L4" row of "Measured Performance by Hardware" (Small, .trt INT8). Update all of them together. -->

=== "Tesla T4"

    | Route                          | Nano                | Small               | Medium              | Large               |
    | ------------------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
    | `inference()` graph, FP16      | 5.36 ms (276 img/s) | 7.66 ms (160 img/s) | 9.47 ms (119 img/s) | 13.15 ms (82 img/s) |
    | `.trt` engine, FP32            | 12.34 ms (89 img/s) | 25.62 ms (39 img/s) | 33.27 ms (28 img/s) | 56.03 ms (18 img/s) |
    | `.trt` engine, FP16            | 5.12 ms (334 img/s) | 6.21 ms (197 img/s) | 7.20 ms (149 img/s) | 10.86 ms (98 img/s) |
    | `.trt` engine, INT8            | 4.90 ms (386 img/s) | 6.10 ms (223 img/s) | 7.02 ms (165 img/s) | 11.45 ms (80 img/s) |
    | Managed engine, auto precision | 5.20 ms (271 img/s) | 5.89 ms (191 img/s) | 7.35 ms (153 img/s) | 10.22 ms (99 img/s) |

=== "NVIDIA L4"

    | Route                          | Nano                | Small               | Medium              | Large               |
    | ------------------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
    | `inference()` graph, FP16      | 2.78 ms (596 img/s) | 3.60 ms (324 img/s) | 4.52 ms (256 img/s) | 5.64 ms (170 img/s) |
    | `.trt` engine, FP32            | 6.32 ms (217 img/s) | 9.79 ms (100 img/s) | 13.49 ms (72 img/s) | 20.55 ms (43 img/s) |
    | `.trt` engine, FP16            | 3.25 ms (724 img/s) | 3.57 ms (432 img/s) | 4.13 ms (349 img/s) | 5.14 ms (227 img/s) |
    | `.trt` engine, INT8            | 2.92 ms (809 img/s) | 3.45 ms (516 img/s) | 4.10 ms (406 img/s) | 4.72 ms (272 img/s) |
    | Managed engine, auto precision | 3.22 ms (466 img/s) | 3.70 ms (355 img/s) | 4.23 ms (286 img/s) | 5.18 ms (216 img/s) |

=== "A100"

    | Route                          | Nano                | Small               | Medium              | Large                |
    | ------------------------------ | ------------------- | ------------------- | ------------------- | -------------------- |
    | `inference()` graph, FP16      | 3.16 ms (877 img/s) | 3.62 ms (688 img/s) | 3.80 ms (590 img/s) | 4.27 ms (457 img/s)  |
    | `.trt` engine, FP32            | 5.72 ms (340 img/s) | 7.56 ms (212 img/s) | 9.13 ms (167 img/s) | 12.28 ms (111 img/s) |
    | `.trt` engine, FP16            | 3.29 ms (967 img/s) | 3.44 ms (783 img/s) | 4.22 ms (676 img/s) | 4.35 ms (558 img/s)  |
    | `.trt` engine, INT8            | 3.40 ms (924 img/s) | 3.69 ms (760 img/s) | 4.06 ms (674 img/s) | 4.34 ms (563 img/s)  |
    | Managed engine, auto precision | no engine           | no engine           | no engine           | no engine            |

=== "RTX PRO 6000"

    | Route                          | Nano                 | Small                | Medium               | Large                |
    | ------------------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
    | `inference()` graph, FP16      | 1.54 ms (1976 img/s) | 1.83 ms (1381 img/s) | 1.95 ms (1262 img/s) | 2.20 ms (984 img/s)  |
    | `.trt` engine, FP32            | 2.70 ms (1071 img/s) | 3.40 ms (607 img/s)  | 3.64 ms (481 img/s)  | 4.77 ms (282 img/s)  |
    | `.trt` engine, FP16            | 1.47 ms (2271 img/s) | 1.62 ms (1782 img/s) | 1.76 ms (1608 img/s) | 2.08 ms (1249 img/s) |
    | `.trt` engine, INT8            | 1.37 ms (2563 img/s) | 1.57 ms (1916 img/s) | 1.73 ms (1694 img/s) | 1.92 ms (1362 img/s) |
    | Managed engine, auto precision | no engine            | no engine            | no engine            | no engine            |

What the tables support:

- **At batch 8 a `.trt` engine is the fastest route on 15 of 16 cells:** INT8 on 12, FP16 on 3 (A100 Nano, Small and Medium). The managed engine leads only T4 Large, 99 against 98 img/s.
- **At batch 1 the routes are close and no route wins everywhere.** The fastest is the INT8 `.trt` engine on 9 of 16 cells, the CUDA graph on 4 (L4 Nano, A100 Nano, Medium and Large), the managed engine on 2 (T4 Small and Large) and the FP16 `.trt` engine on 1 (A100 Small). The CUDA graph is 1.2× to 1.3× slower than the FP16 `.trt` engine on T4 Small to Large.
- **Batch 1 does not scale with GPU size.** The RTX PRO 6000 is 2.2× to 2.5× faster than the L4 for the FP16 `.trt` engine (Nano: 1.47 against 3.25 ms), yet the A100 is slower than the L4 for Nano (3.16 against 2.78 ms, both CUDA graph). The likely cause is the host share of the call, which was not measured.
- **Static INT8 gains little end to end.** At **batch 1** the INT8 engine is within 11% of the FP16 one, and slower on A100 Nano and Small (3.40 against 3.29 ms, 3.69 against 3.44 ms) and on T4 Large (11.45 against 10.86 ms). The T4 engine-only gains above are 1.17× to 1.25× for Nano to Medium, against 1.02× to 1.05× end to end on the same GPU, because host preprocessing and postprocessing are the same for both engines. At **batch 8** the host share is smaller and INT8 is faster on every GPU except the A100 (Nano 924 against 967 img/s, Small 760 against 783, Medium level). Large keeps its attention in FP16 (see [Placement rules](#placement-rules)) but still runs its patch embedding and MLPs in INT8: 4.72 against 5.14 ms on the L4 and 1.92 against 2.08 ms on the RTX PRO 6000.
- **FP32 is not a serving option.** The FP32 `.trt` engine is 1.7× to 5.2× slower than FP16 at **batch 1** and 2.1× to 5.4× at **batch 8** across all four GPUs and sizes. Use it only as a numerical reference.
- **Pass an array, not a PIL image.** `predict()` first copies a PIL image into a NumPy array, so pass `np.array(image)` or a file path when latency matters. Most of the time above the engine itself is host preprocessing and postprocessing.

!!! note "Not comparable with other figures"

    These times include preprocessing and postprocessing on the host. They are not the engine-only times of the INT8 results above, and not the 8.3 ms TensorRT and 12.9 ms ONNX figures on Roboflow's model page, whose timing scope is not stated.

    The managed engine row needs `inference-models`: version 0.39.8 selected no engine for any size on the A100 and RTX PRO 6000 (`NoModelPackagesAvailableError`), so those cells are empty.

### Instance Segmentation

The same measurement for the four instance segmentation sizes (`RFDETRSeg*`) on the same four GPUs, one run per GPU with 20 warmup and 100 timed runs. The A100 here is the 40 GB part, and the RTX PRO 6000 ran PyTorch 2.14.1 with CUDA 13.0; the other GPUs ran PyTorch 2.15.0 with CUDA 13.2. Each call returns masks, so the times include decoding them on the host and are not comparable with the detection tables above. Latency only: no mask AP was scored. Cells and the INT8 speedup read as in the detection tabs above.

<!-- COPIES: none yet. If these tables are copied elsewhere, list the copy here and in the copy. -->

=== "Tesla T4"

    | Route                          | Nano                | Small               | Medium               | Large               |
    | ------------------------------ | ------------------- | ------------------- | -------------------- | ------------------- |
    | `inference()` graph, FP16      | 8.62 ms (132 img/s) | 11.01 ms (94 img/s) | 14.56 ms (66 img/s)  | 20.08 ms (51 img/s) |
    | `.trt` engine, FP32            | 23.37 ms (43 img/s) | 30.39 ms (35 img/s) | 42.24 ms (25 img/s)  | 61.13 ms (17 img/s) |
    | `.trt` engine, FP16            | 6.60 ms (191 img/s) | 8.29 ms (144 img/s) | 10.09 ms (109 img/s) | 13.45 ms (80 img/s) |
    | `.trt` engine, INT8            | 7.23 ms (178 img/s) | 8.70 ms (145 img/s) | 11.18 ms (101 img/s) | 14.50 ms (75 img/s) |
    | Managed engine, auto precision | 21.83 ms (57 img/s) | 22.10 ms (52 img/s) | 26.88 ms (42 img/s)  | 27.58 ms (36 img/s) |

=== "NVIDIA L4"

    | Route                          | Nano                | Small               | Medium              | Large               |
    | ------------------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
    | `inference()` graph, FP16      | 4.19 ms (322 img/s) | 4.74 ms (218 img/s) | 5.93 ms (163 img/s) | 7.16 ms (124 img/s) |
    | `.trt` engine, FP32            | 9.35 ms (104 img/s) | 11.29 ms (79 img/s) | 16.15 ms (57 img/s) | 23.06 ms (37 img/s) |
    | `.trt` engine, FP16            | 4.32 ms (367 img/s) | 4.51 ms (275 img/s) | 5.29 ms (212 img/s) | 6.27 ms (155 img/s) |
    | `.trt` engine, INT8            | 4.38 ms (363 img/s) | 4.50 ms (267 img/s) | 5.38 ms (201 img/s) | 6.39 ms (152 img/s) |
    | Managed engine, auto precision | 15.43 ms (71 img/s) | 16.45 ms (63 img/s) | 19.04 ms (57 img/s) | 20.91 ms (50 img/s) |

=== "A100"

    | Route                          | Nano                | Small               | Medium               | Large                |
    | ------------------------------ | ------------------- | ------------------- | -------------------- | -------------------- |
    | `inference()` graph, FP16      | 3.98 ms (463 img/s) | 4.31 ms (405 img/s) | 4.74 ms (363 img/s)  | 5.41 ms (294 img/s)  |
    | `.trt` engine, FP32            | 7.76 ms (213 img/s) | 8.85 ms (169 img/s) | 10.87 ms (134 img/s) | 13.51 ms (100 img/s) |
    | `.trt` engine, FP16            | 4.15 ms (518 img/s) | 4.26 ms (455 img/s) | 4.60 ms (431 img/s)  | 4.97 ms (370 img/s)  |
    | `.trt` engine, INT8            | 4.31 ms (487 img/s) | 4.59 ms (429 img/s) | 5.14 ms (391 img/s)  | 5.51 ms (338 img/s)  |
    | Managed engine, auto precision | no engine           | no engine           | no engine            | no engine            |

=== "RTX PRO 6000"

    | Route                          | Nano                 | Small                | Medium               | Large               |
    | ------------------------------ | -------------------- | -------------------- | -------------------- | ------------------- |
    | `inference()` graph, FP16      | 2.03 ms (1173 img/s) | 2.20 ms (961 img/s)  | 2.56 ms (859 img/s)  | 2.69 ms (687 img/s) |
    | `.trt` engine, FP32            | 3.67 ms (633 img/s)  | 3.96 ms (471 img/s)  | 4.23 ms (359 img/s)  | 5.22 ms (242 img/s) |
    | `.trt` engine, FP16            | 1.90 ms (1381 img/s) | 2.06 ms (1184 img/s) | 2.18 ms (1080 img/s) | 2.38 ms (838 img/s) |
    | `.trt` engine, INT8            | 1.99 ms (1368 img/s) | 1.91 ms (1231 img/s) | 2.08 ms (1031 img/s) | 2.29 ms (845 img/s) |
    | Managed engine, auto precision | no engine            | no engine            | no engine            | no engine           |

- **Static INT8 does not speed up segmentation end to end.** At **batch 1** the INT8 engine is within 10% of the FP16 one and slower on 12 of 16 cells. Only Seg Small and Seg Medium get INT8 attention; Seg Nano and Seg Large keep it FP16 (see [Placement rules](#placement-rules)). So use the FP16 engine.
- **The FP32 `.trt` engine is 1.9× to 4.6× slower than FP16** at **batch 1** and 2.2× to 4.7× at **batch 8**.
- **At batch 8 a `.trt` engine is the fastest route on every GPU and size.** FP16 leads on 13 of 16 rows; INT8 is ahead by at most 4% on T4 Small, RTX PRO 6000 Small and RTX PRO 6000 Large.
- **The managed `inference-models` engine is the slowest route.** On the T4 and L4 it is about 2× to 4× the `.trt` engine at **batch 1**.

## Predict with RFDETRInference

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/inference_model_fp16.trt")
detections = model.predict("image.jpg", threshold=0.5)
```

The engine must be compatible with the target GPU and TensorRT version. See [Predict with RFDETRInference](basics.md#predict-with-rfdetrinference) for checkpoint inputs, metadata, and batch behavior.

`RFDETRInference` launches the engine with `execute_v2` by default, which needs no [`pycuda`](https://pypi.org/project/pycuda/). `runtime_options` takes three settings, each a `bool` that defaults to `False`:

- `cuda_graph`: capture the engine's launch into a CUDA graph on the first call and replay it on later calls. Only an engine whose optimization profile fixes every input shape qualifies; an engine exported with `dynamic_batch=True` is refused when it is loaded. Through `RFDETRInference` on an L4 it cut the end-to-end batch-1 time of a static FP16 `RFDETRSmall` engine from 5.04 to 4.98 ms, because the launch is a small part of a call dominated by preprocessing and postprocessing.
- `engine_host_code_allowed`: load an engine that carries host code; see [Portable Engines](#portable-engines).
- `verbose`: log TensorRT at VERBOSE rather than INFO.

```python
model = RFDETRInference("output/inference_model_fp16.trt", runtime_options={"cuda_graph": True})
```

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

An engine built by TensorRT 11 with `trt_version_compatible=True` contains host code, and TensorRT only loads it if you say you trust the file. `RFDETRInference` takes `engine_host_code_allowed` in `runtime_options` for that; it is off by default:

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
