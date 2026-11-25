---
description: Run RF-DETR object detection on images, video, and streams. Nano to 2XLarge models with 2.3-17.2 ms latency and up to 60.1 AP on COCO.
---

# Run an RF-DETR Object Detection Model

RF-DETR is a real-time transformer architecture for object detection, built on a DINOv2 vision transformer backbone (a PE-Core-T backbone for Atto, Femto and Pico). The base models are trained on the Microsoft COCO dataset and achieve state-of-the-art accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR offers model sizes from Atto to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. Nano to Large are licensed under Apache 2.0, so they're commercially usable out of the box: you can ship them in closed-source, commercial products, including self-hosted deployments, without paying a license fee or open-sourcing your code. Atto, Femto, Pico, XLarge and 2XLarge (marked with △) are provided by the [`rfdetr_plus`](https://github.com/roboflow/rf-detr_plus) extension (`pip install rfdetr[plus]`) under the [Roboflow Platform Model License (PML 1.0)](https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE): commercial use is included when you deploy on Roboflow, and commercial self-hosting requires the Enterprise add-on.

<!-- COPIES: the three AP columns come from the results table of docs/cookbooks/coco-accuracy.ipynb (one run, provenance in the note below); params and resolution repeat the Detection table of docs/learn/benchmarks.md. -->

|      Size       | FP32 AP<sub>50:95</sub> | FP16 AP<sub>50:95</sub> | INT8 AP<sub>50:95</sub> | Params (M) | Resolution |  License   |
| :-------------: | :---------------------: | :---------------------: | :---------------------: | :--------: | :--------: | :--------: |
|   Atto (A) △    |          30.7           |          30.7           |         30.4 ‡          |    7.4     |  380x380   |  PML 1.0   |
|   Femto (F) △   |          37.6           |          37.6           |         37.4 ‡          |    7.9     |  384x384   |  PML 1.0   |
|   Pico (P) △    |          41.7           |          41.6           |         41.0 ‡          |    8.4     |  560x560   |  PML 1.0   |
|    Nano (N)     |          48.0           |          48.0           |          47.6           |    30.5    |  384x384   | Apache 2.0 |
|    Small (S)    |          52.8           |          52.8           |          52.3           |    32.1    |  512x512   | Apache 2.0 |
|   Medium (M)    |          54.7           |          54.6           |          54.0           |    33.7    |  576x576   | Apache 2.0 |
|    Large (L)    |          56.6           |          56.5           |          56.2           |    33.9    |  704x704   | Apache 2.0 |
|  XLarge (XL) △  |          58.7           |          58.7           |          58.3           |   126.4    |  700x700   |  PML 1.0   |
| 2XLarge (2XL) △ |          60.1           |          60.0           |          59.6           |   126.9    |  880x880   |  PML 1.0   |

> △ Requires the `rfdetr_plus` extension: `pip install rfdetr[plus]`.
>
> ‡ Atto, Femto and Pico were scored in a second run of `coco-accuracy.ipynb` limited to these three models, on the same GPU as the rest of the column (NVIDIA RTX PRO 6000 Blackwell Server Edition) on 2026-11-21, with `rfdetr` at commit `e4cadc4f`, `rfdetr_plus` 1.3.0, torch 2.14.1+cu130 and TensorRT 11.4.0.106, because their TensorRT INT8 placement changed: the MLPs of the first six PE-Core-T backbone blocks now stay FP16. The first INT8 placement for these sizes quantized those MLPs and lost 7.0, 8.7 and 8.5 AP against PyTorch FP32 (a development build, never released). With the placement that ships, INT8 is 0.3, 0.2 and 0.7 AP below PyTorch FP32 and 0.3, 0.2 and 0.6 AP below TensorRT FP16. It is at most 9% faster than FP16 end to end ([latency below](#fastest-configuration-on-nvidia-gpus)), so prefer FP16 unless you need INT8 for another reason.

--8<-- "docs/snippets/reproduced-coco-ap.md"

## Fastest Configuration on NVIDIA GPUs

The configuration with the lowest end-to-end latency for each size on four NVIDIA GPUs, batch 1. Run the [latency cookbook](../../../cookbooks/measure-latency/) on your own GPU to find yours.

<!-- COPIES: every number comes from the results of docs/cookbooks/measure-latency.ipynb (HARDWARE = "cuda", one run per GPU, provenance in docs/learn/run/latency.md). -->

=== "NVIDIA T4"

    |      Size       | PyTorch CUDA graph FP16 [ms] | TensorRT FP16 [ms] | TensorRT FP16 graph [ms] | TensorRT INT8 [ms] | TensorRT INT8 graph [ms] |
    | :-------------: | :--------------------------: | :----------------: | :----------------------: | :----------------: | :----------------------: |
    |   Atto (A) △    |         3.60 (4.24)          |    3.76 (4.17)     |       3.57 (4.22)        |    3.41 (3.91)     |     **3.23 (3.58)**      |
    |   Femto (F) △   |         4.04 (4.56)          |    3.84 (4.54)     |     **3.69 (4.04)**      |  **3.62 (3.99)**   |     **3.57 (3.90)**      |
    |   Pico (P) △    |         4.64 (4.98)          |  **4.12 (4.59)**   |       4.19 (4.54)        |  **4.03 (4.34)**   |     **3.92 (4.21)**      |
    |    Nano (N)     |         5.17 (5.90)          |    4.78 (5.61)     |       4.91 (5.42)        |  **4.61 (5.59)**   |     **4.44 (4.90)**      |
    |    Small (S)    |         7.99 (8.38)          |    6.88 (7.24)     |       7.03 (7.85)        |  **6.32 (6.99)**   |     **6.33 (6.65)**      |
    |   Medium (M)    |        10.04 (10.46)         |    8.50 (8.84)     |       8.83 (9.48)        |  **7.51 (7.89)**   |     **7.50 (8.05)**      |
    |    Large (L)    |        14.91 (15.37)         |   12.41 (13.07)    |      12.74 (13.24)       | **11.11 (11.77)**  |    **11.21 (11.74)**     |
    |  XLarge (XL) △  |        27.25 (28.77)         |   24.12 (26.20)    |      24.56 (26.54)       | **19.83 (20.46)**  |    **20.08 (20.69)**     |
    | 2XLarge (2XL) △ |        39.69 (41.26)         |   38.18 (40.39)    |      35.50 (37.99)       | **28.10 (29.72)**  |    **29.47 (31.13)**     |

=== "NVIDIA L4"

    |      Size       | PyTorch CUDA graph FP16 [ms] | TensorRT FP16 [ms] | TensorRT FP16 graph [ms] | TensorRT INT8 [ms] | TensorRT INT8 graph [ms] |
    | :-------------: | :--------------------------: | :----------------: | :----------------------: | :----------------: | :----------------------: |
    |   Atto (A) △    |       **2.44 (2.78)**        |    3.06 (3.40)     |       2.90 (3.19)        |    3.12 (3.47)     |       2.85 (3.10)        |
    |   Femto (F) △   |       **2.80 (3.10)**        |    3.21 (3.51)     |       2.98 (3.20)        |    3.38 (4.32)     |       3.02 (3.53)        |
    |   Pico (P) △    |       **3.00 (3.20)**        |    3.32 (3.59)     |       3.16 (3.85)        |    3.47 (3.81)     |     **3.11 (3.56)**      |
    |    Nano (N)     |       **3.20 (3.42)**        |    3.87 (4.14)     |       3.54 (3.89)        |    3.55 (4.13)     |       3.45 (4.06)        |
    |    Small (S)    |       **4.01 (4.33)**        |    4.23 (4.51)     |     **4.03 (4.31)**      |    4.12 (4.48)     |     **3.90 (4.21)**      |
    |   Medium (M)    |         4.81 (5.09)          |    4.82 (5.35)     |     **4.56 (5.09)**      |    4.74 (5.09)     |     **4.43 (4.87)**      |
    |    Large (L)    |         6.01 (6.30)          |    5.89 (6.28)     |       5.65 (6.01)        |  **5.27 (5.75)**   |     **5.10 (5.54)**      |
    |  XLarge (XL) △  |        11.02 (11.41)         |    9.90 (10.37)    |       9.87 (10.20)       |  **8.53 (8.89)**   |     **8.34 (8.72)**      |
    | 2XLarge (2XL) △ |        14.80 (15.39)         |   12.67 (13.12)    |      12.56 (12.96)       | **10.07 (10.88)**  |     **9.95 (10.54)**     |

=== "NVIDIA A100"

    |      Size       | PyTorch CUDA graph FP16 [ms] | TensorRT FP16 [ms] | TensorRT FP16 graph [ms] | TensorRT INT8 [ms] | TensorRT INT8 graph [ms] |
    | :-------------: | :--------------------------: | :----------------: | :----------------------: | :----------------: | :----------------------: |
    |   Atto (A) △    |       **3.08 (3.35)**        |    3.40 (3.80)     |     **3.23 (3.72)**      |    3.41 (4.14)     |     **3.08 (3.65)**      |
    |   Femto (F) △   |       **3.25 (3.65)**        |    3.38 (3.76)     |     **3.11 (3.39)**      |    3.65 (4.59)     |     **3.12 (3.38)**      |
    |   Pico (P) △    |         3.72 (4.00)          |    3.67 (4.05)     |     **3.34 (3.59)**      |    3.63 (4.15)     |     **3.27 (3.55)**      |
    |    Nano (N)     |       **3.35 (3.92)**        |    3.74 (4.17)     |     **3.44 (3.73)**      |    3.68 (4.10)     |       3.54 (4.18)        |
    |    Small (S)    |       **3.81 (4.43)**        |    4.03 (4.49)     |     **3.67 (3.97)**      |    4.00 (4.61)     |     **3.68 (4.16)**      |
    |   Medium (M)    |       **4.06 (4.64)**        |    4.38 (5.02)     |     **4.06 (4.38)**      |    4.31 (4.85)     |     **3.93 (4.33)**      |
    |    Large (L)    |         4.64 (5.04)          |    4.81 (5.36)     |     **4.38 (4.84)**      |    4.82 (5.54)     |       4.68 (4.98)        |
    |  XLarge (XL) △  |         6.18 (6.50)          |    6.13 (6.76)     |     **5.75 (6.01)**      |    5.99 (6.38)     |     **5.66 (6.04)**      |
    | 2XLarge (2XL) △ |         6.90 (7.13)          |    6.67 (7.03)     |     **6.32 (6.89)**      |  **6.38 (6.78)**   |     **6.10 (6.44)**      |

=== "NVIDIA RTX PRO 6000"

    |      Size       | PyTorch CUDA graph FP16 [ms] | TensorRT FP16 [ms] | TensorRT FP16 graph [ms] | TensorRT INT8 [ms] | TensorRT INT8 graph [ms] |
    | :-------------: | :--------------------------: | :----------------: | :----------------------: | :----------------: | :----------------------: |
    |   Atto (A) △    |         1.53 (1.60)          |    1.46 (1.53)     |     **1.27 (1.35)**      |    1.44 (1.51)     |     **1.25 (1.32)**      |
    |   Femto (F) △   |         1.73 (1.78)          |    1.54 (1.62)     |     **1.34 (1.41)**      |    1.55 (1.62)     |     **1.29 (1.36)**      |
    |   Pico (P) △    |         1.86 (1.90)          |    1.71 (1.83)     |       1.46 (1.53)        |    1.68 (1.77)     |     **1.39 (1.46)**      |
    |    Nano (N)     |         1.73 (1.82)          |    1.72 (1.79)     |       1.54 (1.62)        |    1.69 (1.77)     |     **1.43 (1.51)**      |
    |    Small (S)    |         2.07 (2.16)          |    1.94 (2.00)     |       1.73 (1.80)        |    1.87 (1.99)     |     **1.63 (1.70)**      |
    |   Medium (M)    |         2.23 (2.33)          |    2.07 (2.14)     |       1.84 (1.90)        |    1.96 (2.02)     |     **1.73 (1.81)**      |
    |    Large (L)    |         2.52 (2.58)          |    2.30 (2.42)     |       2.09 (2.15)        |    2.19 (2.26)     |     **1.96 (2.02)**      |
    |  XLarge (XL) △  |         3.63 (3.70)          |    3.13 (3.19)     |       2.93 (3.10)        |    2.98 (3.13)     |     **2.78 (2.84)**      |
    | 2XLarge (2XL) △ |         4.27 (4.35)          |    3.77 (3.84)     |       3.54 (3.61)        |    3.43 (3.51)     |     **3.19 (3.27)**      |

Cells are median ms (95th percentile), lower is better. Bold marks the fastest median of each row and every median within 5% of it, and those are tied. [How the numbers were measured](latency.md).

- For Atto, Femto and Pico the best INT8 engine takes 0.91 to 1.01 times the time of the best FP16 engine, so INT8 and FP16 are tied on most GPUs. INT8 is clearly ahead only for Atto on the T4 (0.91) and Pico on the RTX PRO 6000 (0.95). On the L4, PyTorch CUDA graph FP16 is the fastest configuration for Atto and Femto.
- The Atto, Femto and Pico rows are from a second run on 2026-11-21, after the INT8 placement of these sizes changed; all five configurations were timed again in it. The A100 of that run is the 80 GB part, the other sizes were timed on the 40 GB part. See [How the numbers were measured](latency.md#nvidia-gpus).
- From Small up an INT8 engine is among the fastest on every GPU, except at Large on the A100, where FP16 graph alone is.
- The 95th percentile stays within 1.22 times the median, except for Femto on the L4 (1.28 times) and on the A100 (1.26 times).

## Fastest Configuration on Apple Silicon

The configuration with the lowest end-to-end latency for each size on an Apple M4 Max, batch 1. Run the [latency cookbook](../../../cookbooks/measure-latency/) on your own device to find yours.

<!-- COPIES: every number comes from the results table of docs/cookbooks/measure-latency.ipynb (HARDWARE = "apple", one run, provenance in docs/learn/run/latency.md). -->

|      Size       | PyTorch inductor FP32 [ms] | PyTorch inductor FP16 [ms] | PyTorch TorchScript FP16 [ms] | CoreML FP32 [ms] | CoreML FP16 [ms] | CoreML GPU FP32 [ms] | CoreML GPU FP16 [ms] |
| :-------------: | :------------------------: | :------------------------: | :---------------------------: | :--------------: | :--------------: | :------------------: | :------------------: |
|   Atto (A) △    |        4.92 (5.38)         |        4.75 (5.20)         |         9.98 (11.54)          |   7.82 (9.53)    | **3.81 (4.03)**  |     8.07 (9.55)      |     6.93 (9.58)      |
|   Femto (F) △   |      **4.74 (5.06)**       |      **4.60 (5.07)**       |         9.89 (10.79)          |   8.30 (9.51)    |  24.51 (25.41)   |     7.15 (9.41)      |     8.16 (9.43)      |
|   Pico (P) △    |      **4.99 (5.17)**       |      **4.93 (5.20)**       |         9.84 (10.36)          |   8.92 (9.41)    |   8.51 (9.41)    |     8.56 (9.49)      |     8.68 (9.50)      |
|    Nano (N)     |      **7.48 (8.07)**       |      **7.22 (8.26)**       |         10.04 (11.86)         |   9.74 (11.51)   |  12.64 (14.58)   |     9.75 (11.16)     |     9.62 (11.09)     |
|    Small (S)    |       11.65 (13.10)        |     **10.82 (13.32)**      |         12.71 (15.89)         |  12.17 (16.73)   |  25.19 (29.14)   |    12.33 (14.55)     |  **11.28 (13.56)**   |
|   Medium (M)    |       15.98 (18.37)        |       14.71 (20.54)        |         15.58 (18.30)         |  15.11 (20.66)   |  33.75 (36.55)   |    14.73 (16.95)     |  **13.24 (15.34)**   |
|    Large (L)    |       21.52 (23.59)        |       18.19 (20.13)        |         18.55 (19.14)         |  17.80 (18.24)   |  54.89 (56.16)   |    18.67 (20.25)     |  **16.90 (18.65)**   |
|  XLarge (XL) △  |       36.25 (39.79)        |       33.31 (33.83)        |         34.43 (34.83)         |  38.46 (39.06)   |  91.62 (92.86)   |    36.93 (38.08)     |  **31.29 (31.80)**   |
| 2XLarge (2XL) △ |       49.95 (51.37)        |     **41.74 (42.00)**      |       **43.51 (43.80)**       |  50.36 (52.05)   | 128.73 (129.12)  |    50.01 (52.11)     |  **41.74 (42.47)**   |

Cells are median ms (95th percentile), lower is better. Bold marks the fastest median of each row and every median within 5% of it, and those are tied. [How the numbers were measured](latency.md).

Atto is fastest with CoreML FP16 alone, 25% ahead of PyTorch inductor FP16 (3.81 and 4.75 ms). Femto to Small are fastest with PyTorch inductor FP16 (level with inductor FP32 up to Nano and with CoreML GPU FP16 at Small), Medium to XLarge with CoreML GPU FP16, and 2XLarge is a three-way tie.

Femto's CoreML FP16 time (24.51 ms) is not noise: the Neural Engine compiler fails on its FP16 package under the default compute units, as the [CoreML page](../../exports/coreml.md#measured-latency-on-apple-silicon) explains. CoreML GPU FP16 avoids that plan and takes 8.16 ms.

## Fastest Configuration on an x86 CPU

The configuration with the lowest end-to-end latency for each size on a Colab x86 CPU (AMD EPYC 7B12, 8 logical cores), batch 1. Run the [latency cookbook](../../../cookbooks/measure-latency/) on your own CPU to find yours.

<!-- COPIES: every number comes from the results of docs/cookbooks/measure-latency.ipynb (HARDWARE = "cpu"; Atto, Femto and Pico from the run of 2026-11-20, every other size from the run of 2026-11-22 and 2026-11-23; provenance in docs/learn/run/latency.md). -->

|      Size       | PyTorch TorchScript FP32 [ms] |  ONNX FP32 [ms]   |    ONNX INT8 [ms]     |  ONNX compact [ms]  | OpenVINO FP32 [ms]  | OpenVINO FP16 [ms]  | OpenVINO INT8 [ms]  |
| :-------------: | :---------------------------: | :---------------: | :-------------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: |
|   Atto (A) △    |       **50.19 (79.63)**       |  74.52 (108.92)   |           —           |  **52.10 (86.36)**  |   63.91 (103.90)    |    68.74 (99.52)    |          —          |
|   Femto (F) △   |      **76.45 (135.71)**       |  106.03 (137.89)  |           —           | **76.05 (109.08)**  | **76.05 (133.16)**  | **75.90 (120.54)**  |          —          |
|   Pico (P) △    |      **105.63 (141.22)**      |  142.46 (173.60)  |           —           | **104.56 (138.39)** | **100.89 (164.15)** |   113.34 (172.46)   |          —          |
|    Nano (N)     |      **158.08 (245.97)**      |  249.21 (281.89)  |    169.75 (209.25)    |          —          |   167.17 (257.91)   |   179.72 (278.98)   | **152.08 (259.56)** |
|    Small (S)    |        273.47 (569.05)        |  407.39 (443.08)  |  **259.15 (301.92)**  |          —          | **258.05 (399.15)** |   321.81 (474.43)   | **261.56 (458.74)** |
|   Medium (M)    |      **355.16 (950.70)**      |  505.60 (547.70)  |  **342.88 (384.67)**  |          —          |   371.45 (567.63)   | **351.54 (529.80)** | **339.10 (568.12)** |
|    Large (L)    |     **529.01 (1222.42)**      |  676.97 (756.59)  |  **513.09 (574.21)**  |          —          |   555.97 (776.07)   |   568.16 (813.45)   | **530.42 (839.05)** |
|  XLarge (XL) △  |       1177.53 (2183.39)       | 1379.60 (1527.71) | **933.85 (1037.51)**  |          —          |  1245.84 (1656.02)  |  1288.71 (1708.34)  |  1135.08 (1730.73)  |
| 2XLarge (2XL) △ |       1662.25 (2719.25)       | 1823.72 (1968.23) | **1333.04 (1493.49)** |          —          |  1893.48 (2653.58)  |  2053.03 (2612.76)  |  1626.12 (2354.66)  |

Cells are median ms (95th percentile), lower is better. Bold marks the fastest median of each row and every median within 5% of it, and those are tied. [How the numbers were measured](latency.md). Atto, Femto and Pico have no INT8 cell, and ONNX compact exists only for them. Their row is from a different run than the other sizes, see [How the numbers were measured](latency.md#x86-cpu). Run on an AMD EPYC 7B12 with 8 logical cores, `rfdetr` 1.13.0.dev0 (commit `add6c3e4`), `rfdetr_plus` 1.3.0, PyTorch 2.14.1 (CPU build), ONNX Runtime 1.32.0, OpenVINO 2026.4.1 and NNCF 3.4.0, from 2026-11-22 21:41 to 2026-11-23 03:47 UTC. The Atto, Femto and Pico rows were run on 2026-11-20 on the same CPU type, with `rfdetr` commit `f063c178`, `rfdetr_plus` 1.3.0 and the same runtime versions.

- Atto, Femto and Pico have no single fastest configuration. On Atto, TorchScript FP32 and ONNX compact are tied. On Femto, TorchScript FP32, ONNX compact and both OpenVINO precisions are tied. On Pico, OpenVINO FP32, ONNX compact and TorchScript FP32 are tied. Medium is a four-way tie.
- ONNX compact takes 27% to 30% less time than ONNX FP32 on all three sizes, so it closes the gap that ONNX FP32 has to TorchScript FP32, but it does not beat TorchScript FP32 or OpenVINO FP32. It scored 30.35, 37.45 and 41.45 AP on all 5000 val2017 images, 0.32, 0.16 and 0.21 below PyTorch FP32 on the same CPU.
- From Small up ONNX INT8 is fastest or tied, and alone at XLarge and 2XLarge, 21.5% and 22.0% ahead of the next configuration (OpenVINO INT8) and 26.1% and 24.7% ahead of TorchScript FP32. At Nano it is 11.6% behind the fastest (OpenVINO INT8).
- OpenVINO INT8 takes 0.86 to 1.01 times the OpenVINO FP32 time from Nano to 2XLarge, and OpenVINO FP16 0.95 to 1.25 times. The 95th percentile reaches 2.68 times the median (PyTorch TorchScript FP32, Medium), so this one shared-host run cannot order configurations inside the 5% band.

!!! warning "FP16 and INT8 can lower accuracy"

    Most of the fastest configurations above run in FP16 or INT8, and accuracy was scored for only some of them. Check it on your own data before you rely on one; [the scored configurations and sources](latency.md#fp16-and-int8-can-lower-accuracy).

## Run on an Image

Perform inference on an image with the `rfdetr` package. To use a different model size, select the corresponding class from the table above.

```python
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSmall()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

!!! note "Using COCO classes vs. fine-tuned model classes"

    `COCO_CLASSES` works for COCO-pretrained models (80 COCO classes, indexed 0-79). For fine-tuned models, use `detections.data["class_name"]` instead — it resolves class names from the checkpoint and works for both COCO and custom datasets.

`predict()` resizes without antialiasing by default (`antialias=False`), which matches checkpoints trained with the default CPU augmentation backend when `rfdetr[augment]` is installed (Albumentations). Pass `antialias=True` for checkpoints trained with torchvision resizing: the Kornia/GPU augmentation backend, the CPU backend without `rfdetr[augment]`, or older RF-DETR releases. Antialiasing costs a little extra preprocessing time per image. `RFDETRInference` uses the artifact metadata's `resize_antialias` by default for exported models; pass an explicit `antialias=True` or `antialias=False` to override it. The lower-level `rfdetr.export` runtime preprocessing helpers always resize without antialiasing. On MPS with PyTorch older than 2.7, `antialias=True` resizes on the CPU because MPS has no antialiased resize kernel there.

For repeated inference with the `rfdetr` package on CUDA, a fixed batch size, and a fixed resolution, the direct CUDA Graph backend records the TorchScript forward once and replays it. Capture allocates a graph-private memory pool holding the captured forward's intermediates plus the static input/output buffers; it persists for the graph's lifetime and scales with batch size, resolution and model, so this backend uses more device memory than the TorchScript backend. The backend was measured on detection RF-DETR Nano at batch size 1; segmentation and keypoint models use the same path but are unmeasured, and replay/clone memory cost grows with masks and batch size. Outputs are cloned before they leave the graph, so predictions returned by an earlier call are not overwritten by the next one. It is the default on CUDA; pass `compile_backend="torchscript"` to trace without capturing:

```python
predictor = model.inference(compile_backend="cudagraph", batch_size=1, dtype="float16")
detections = predictor.predict("image.jpg")
```

An operator that CUDA Graphs cannot capture makes `inference()` raise a `RuntimeError`. A failed capture can leave CUDA random-number state unusable, so restart the process before choosing another backend.

PyTorch Inductor is another opt-in backend for long-running inference. Cold compilation can have a higher one-time setup cost, but Inductor can apply broader graph optimizations and later processes may reuse its disk cache. Both examples require a compatible CUDA device, operators, and installed PyTorch version; `dtype="float16"` also requires FP16 support:

```python
predictor = model.inference(compile_backend="inductor", batch_size=1, dtype="float16")
detections = predictor.predict("image.jpg")
```

The native factory returns a predictor that owns one independent copy of the current model weights. The source model remains unchanged; deleting the predictor releases its runtime when no other references remain. Use the same pattern for CPU inference without compilation:

```python
predictor = model.inference(compile=False, dtype="float16")
detections = predictor.predict("image.jpg")
```

## Set a confidence threshold for each class

One `threshold` holds every class to the same bar, but a trained model rarely finds all classes equally well: for some classes the scores of correct detections and of false positives overlap more than for others. Pass a `ClassThresholds` instead, and `predict()` keeps a detection when its score exceeds the threshold of its own class. Name the classes that need their own bar; every other class takes `default`:

```python
from rfdetr import ClassThresholds, RFDETRSmall

model = RFDETRSmall()

thresholds = ClassThresholds({"person": 0.35, "traffic light": 0.6}, default=0.5)

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=thresholds)
```

- A name that is not in `model.class_names` raises a `ValueError` that lists the valid names, before the model runs. A NaN raises a `ValueError` when the `ClassThresholds` is built; use `float("inf")` to drop a class.
- `ClassThresholds` also takes one threshold per entry of `model.class_names`, in the same order, as a list, tuple, [NumPy](https://pypi.org/project/numpy/) array or tensor: `ClassThresholds([0.4, 0.55, 0.45])`. It must hold exactly `len(model.class_names)` entries.
- `threshold` takes a single number or a `ClassThresholds`. A plain list or `dict` raises a `TypeError` that points to `ClassThresholds`, since a `dict` alone says nothing about the classes it leaves out.
- The comparison is strict, like the single-number form, so a `ClassThresholds` that gives every class `t` keeps exactly the detections of the listed classes that `t` keeps.
- A `ClassThresholds` drops detections that belong to no entry of `model.class_names`: the no-object slot of a fine-tuned model and the unused category IDs of a pretrained COCO model. A single number keeps them.
- Segmentation and keypoint models take the same argument. For keypoint models the thresholds apply to `key_points.detection_confidence`.

The [per-class confidence thresholds cookbook](../../../cookbooks/per-class-confidence-thresholds/) shows how to choose the values from held-out images, and measures when a list of thresholds beats a single one and when it does not.

## Steadier scores with shift test-time augmentation

The backbone reads the image in square patches (16 pixels for most models, 12 for the segmentation models). When an object moves by a fraction of a patch, its box barely changes, but its confidence score can swing enough to cross the threshold ([#1599](https://github.com/roboflow/rf-detr/issues/1599)). `shift_tta=N` predicts the image `N` times, each time moved by another `1/N` of a patch, and averages the matching detections:

```python
detections = model.predict("image.jpg", threshold=0.5, shift_tta=4)
```

The animation shows sequential execution on CPU and MPS.

![Two pages showing sequential shift views. Page 1: a plain predict() runs the network once on the whole input image, which holds a person and a small dog, and returns a bounding box with a score for each: person 0.91 and dog 0.70, both kept. Page 2: with shift_tta=4 the same input is moved down and right by 0, a quarter, a half and three quarters of a patch on a canvas one patch larger, so nothing is cropped. Each snapshot goes through the same network once and returns its own boxes and scores. The network finds the person in every pass but misses the dog in one. The combine panel then plays the steps in order: the boxes of each pass slide back by subtracting their offset, boxes that share a class and have an IoU above 0.6 are grouped, each group is averaged into one box, and its score is built from one quarter of every pass score, so a missed pass adds nothing and counts as 0. The person gets 0.91 and the dog 0.54, and both are kept. Scores are illustrative.](../../assets/shift-tta.svg){ loading=lazy }

- On CPU and MPS, each shift runs its own forward pass. CUDA can batch shifts on the same canvas and split them when the forward pixel budget limits how many views fit together. Two shifts require less work than four.
- Detections match across passes when they have the same class and a box IoU above 0.6. The fused box is the mean box, and the fused score is the mean score over all passes, with a pass that missed the object counting as 0. No pass is held to `threshold` on its own: a pass that scores the object low still counts at its own score, and `threshold` applies only to the averaged score. Averaged scores can sit a little lower than single-pass ones; pick the threshold with `shift_tta` on.
- Segmentation masks keep the pixels that at least half of the matched masks cover. Keypoint models do not support `shift_tta`, and the value is checked before any image is read.
- `RFDETRInference.predict(..., tta=TTAConfig())` opts into zoomed views. Set `flip=True` to request mirroring and supply `zoom` explicitly to combine it with zoomed views. [Test-Time Augmentation](../tta.md) measures what each recipe adds and costs on COCO val2017.

On RF-DETR-N fine-tuned on the `deeppcb` dataset of RF100-VL, moving each test image by 0 to 16 pixels changed the keep-or-drop decision for 21.0% of the defects with one pass, 7.4% with `shift_tta=2` and 3.7% with `shift_tta=4`, each at its own best-F1 threshold. Test mAP50-95 was 0.702, 0.699 and 0.708.

## Run on video, webcam, or RTSP stream

!!! tip "Built-in video and stream sources"

    `RFDETRInference(model).predict()` reads video files, video URLs, webcams, RTSP streams, and YouTube directly and can yield results lazily with `stream=True`. See [Run on Videos and Streams](videos-and-streams.md). The examples below keep full control of decoding and display with OpenCV.

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video source: <SOURCE_VIDEO_PATH>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    WEBCAM_INDEX = 0
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open webcam: {WEBCAM_INDEX}")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "stream"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    video_capture = cv2.VideoCapture("<RTSP_STREAM_URL>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open RTSP stream: <RTSP_STREAM_URL>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
