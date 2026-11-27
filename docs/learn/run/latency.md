---
description: How the RF-DETR latency tables were measured. End-to-end batch-1 timing on Apple Silicon, NVIDIA GPUs and an x86 CPU, how ties are marked, the hardware, configurations and package versions of each run, and what the numbers do not say about accuracy.
---

# How RF-DETR Latency Is Measured

Every "Fastest Configuration" table on the Run pages for [detection](detection.md), [segmentation](segmentation.md) and [keypoints](keypoints.md) comes from the [latency cookbook](../../../cookbooks/measure-latency/), one run per hardware. This page states what is timed, how the tables mark ties, and which hardware, configurations and package versions each run used.

## Reading the Tables

- **Cells:** each cell is the median latency in milliseconds, with the 95th percentile of the same calls in brackets. Lower is better.
- **Bold:** the fastest median of each row and every median within 5% of it.
- **Ties:** several bold cells in a row are tied. A gap under 5% to the fastest configuration counts as noise, because repeated runs of the largest models on the M4 Max differed by 5 to 7%, so these numbers cannot say which of the tied configurations is faster. Choose among them by accuracy and tooling. Each NVIDIA GPU and the CPU host were measured once, so their own run-to-run spread is not known and the same 5% is used.

## What Is Timed

The end-to-end `RFDETRInference.predict` call (preprocessing, forward pass and postprocessing) at batch 1, timed once for each distinct COCO val2017 image (500 on Apple Silicon and NVIDIA, 200 on the CPU, where a call is far slower) after 20 warm-up calls for the Apple run and 50 for the NVIDIA and CPU runs. Every export is a static batch-1 artifact.

## Apple Silicon

- **Configurations:** PyTorch with inductor in FP32 and FP16, PyTorch with TorchScript in FP16, and CoreML in FP32 and FP16 with two sets of compute units. PyTorch TorchScript is the traced model that `model.inference()` builds with `compile_backend="torchscript"`. CoreML uses the default `device="auto"`, which lets Core ML plan work on the Neural Engine, and CoreML GPU loads the same package with `device="cpu_and_gpu"`. No INT8 configuration is measured on Apple Silicon, because the Core ML export has no INT8 option.
- **Run:** Apple M4 Max (16-core CPU, 40-core GPU), macOS 26.7.1, measured on 2026-11-13, with the versions in the [table below](#package-versions). Every column and every size comes from one run of the cookbook, and no other work was started on the machine while it ran.
- **Inductor:** PyTorch inductor predictors are built by `model.inference()`, which holds Dynamo's recompile limit above the default of 8 and compiles with `dynamic=False`. Without that, inductor predictors built after the eighth in one process run several times slower or fail to compile, as the [FAQ](../../faq.md#why-is-an-inductor-predictor-slower-after-i-built-several-models-in-one-process) explains.

## NVIDIA GPUs

- **Configurations:** PyTorch CUDA graph FP16 is `model.inference()` with fp16 and a captured CUDA graph. TensorRT FP16 and INT8 are static engines from `RFDETR.export(format="tensorrt")`, loaded by `RFDETRInference`, and *graph* replays the same engine as a CUDA graph through the `cuda_graph` runtime option. The INT8 engines are calibrated on 128 train2017 images. The TensorRT INT8 engines of Atto, Femto and Pico keep the MLPs of the first six backbone blocks in FP16, and their accuracy is scored on the [Detection page](detection.md#pre-trained-checkpoints).
- **Run:** NVIDIA T4, NVIDIA L4, NVIDIA A100-SXM4-40GB and NVIDIA RTX PRO 6000 Blackwell Server Edition, one run of the cookbook per GPU on 2026-11-13, each with the versions in the [table below](#package-versions). Every engine was built on the GPU that timed it. Atto, Femto and Pico were timed again in a second run of the cookbook limited to those three models, on 2026-11-21 with `rfdetr` at commit `e4cadc4f`, because their INT8 placement changed after the first run; all five configurations were timed again in it, and its A100 is an NVIDIA A100-SXM4-80GB.
- **Host:** the host CPU was not recorded, and a call includes host work, so a small model on a fast GPU can be limited by the host.

## x86 CPU

- **Configurations:** PyTorch TorchScript in FP32 (`model.inference()` with `compile_backend="torchscript"`), ONNX Runtime in FP32 and INT8, the compact CPU ONNX export of `rfdetr_plus` 1.3 (`export_cpu_onnx`, for Atto, Femto and Pico only) and OpenVINO in FP32, FP16 and INT8, each exported with `RFDETR.export` and loaded by `RFDETRInference` on the CPU. Every runtime is asked for all 8 logical cores: PyTorch through `torch.set_num_threads`, ONNX Runtime and OpenVINO through the `num_threads` runtime option, with OpenVINO hyper-threading enabled. The INT8 artifacts are calibrated on 128 train2017 images. Atto, Femto and Pico have no INT8 configuration, for the reason given for NVIDIA above.
- **Run:** a Colab x86 host with an AMD EPYC 7B12 and 8 logical cores, with the versions in the [table below](#package-versions). Nano to 2XLarge, every segmentation size and keypoints were timed in one continuous run of the cookbook from 2026-11-22 21:41 to 2026-11-23 03:47 UTC, with `rfdetr` at commit `add6c3e496b4b9546ceee25183aba802f2ed629a`, which keeps the attention-value multiply of the ONNX INT8 export in float. Atto, Femto and Pico were timed on 2026-11-20, which also times the compact ONNX export. The 2026-11-20 run records the `rfdetr` commit `f063c17875c9ae51ea0fda9c992c993ef8a74db7` and re-timed every configuration of the three sizes in the same session as the compact export. Two earlier sessions on the same host type the same day gave different absolute times (PyTorch TorchScript FP32 on Pico: 135, 98 and 106 ms), so compare configurations within a session, not across sessions. The host is shared, and the 95th percentile reaches 2.68 times the median in the worst cell (TorchScript FP32 on detection Medium).

## Package Versions

| Run                                                                     | `rfdetr`                        | `rfdetr_plus` | PyTorch            | Runtime packages                                                 |
| ----------------------------------------------------------------------- | ------------------------------- | ------------- | ------------------ | ---------------------------------------------------------------- |
| Apple M4 Max                                                            | 1.13.0.dev0                     | 1.2.0         | 2.14.0             | coremltools 9.0                                                  |
| NVIDIA T4, L4, A100 and RTX PRO 6000                                    | 1.13.0.dev0                     | 1.2.0         | 2.14.1 (CUDA 13.0) | TensorRT 11.4.0.106                                              |
| NVIDIA T4, L4, A100 (80 GB) and RTX PRO 6000, Atto, Femto and Pico only | 1.13.0.dev0 (commit `e4cadc4f`) | 1.3.0         | 2.14.1 (CUDA 13.0) | TensorRT 11.4.0.106                                              |
| x86 CPU (AMD EPYC 7B12), Nano to 2XLarge, segmentation and keypoints    | 1.13.0.dev0 (commit `add6c3e4`) | 1.3.0         | 2.14.1 (CPU build) | ONNX Runtime 1.32.0, OpenVINO 2026.4.1, NNCF 3.4.0, psutil 5.9.5 |
| x86 CPU (AMD EPYC 7B12), Atto, Femto and Pico only                      | 1.13.0.dev0 (commit `f063c178`) | 1.3.0         | 2.14.1 (CPU build) | ONNX Runtime 1.32.0, OpenVINO 2026.4.1, NNCF 3.4.0, psutil 7.2.2 |

The cookbook writes the installed versions of these packages into every result row, and installs the extras for its hardware: `plus,coreml`, `plus,tensorrt` or `plus,onnx,openvino` (CPU runs add `nncf` and `psutil`). The x86 run of 2026-11-22 records its commit in every row. The NVIDIA runs and the other CPU run installed `rfdetr` 1.13.0.dev0 from the `docs/measure-latency-cookbook` branch archive, so no commit hash was recorded, and the Python version was recorded only for the CPU run (3.13). To reproduce a number, pin the packages above in a fresh environment and rerun the cookbook for your hardware.

## FP16 and INT8 Can Lower Accuracy

Most of the fastest configurations run in FP16 or INT8. The FP16 and INT8 AP in the Pre-trained Checkpoints tables of the [detection](detection.md#pre-trained-checkpoints), [segmentation](segmentation.md#pre-trained-checkpoints) and [keypoints](keypoints.md#pre-trained-checkpoints) pages was scored on TensorRT engines, where FP16 stays within 0.1 AP of FP32 for every size and INT8 costs 0.2 to 0.9 AP for the sizes timed in INT8, so read the INT8 column before you choose an INT8 engine. The AP was not scored for PyTorch CUDA graph FP16, or for PyTorch and CoreML FP16 on Apple Silicon, so check the accuracy on your own data before you rely on one of those configurations.

The same holds on the CPU: ONNX INT8 is among the fastest configurations for 10 of the 16 models (the three PE sizes have no INT8 configuration, and Nano, Seg Nano and Seg Small are faster in another configuration), and no AP was scored for those CPU rows. The ONNX compact export of Atto, Femto and Pico was scored on all 5000 val2017 images: 30.35, 37.45 and 41.45 AP, 0.32, 0.16 and 0.21 below PyTorch FP32 on the same CPU. The [ONNX page](../../exports/onnx.md) lists what stays in float, but its accuracy cost has not been scored on this CPU, so measure it on your own data before you choose it.
