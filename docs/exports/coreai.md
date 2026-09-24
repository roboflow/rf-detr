---
description: Export RF-DETR models to Apple Core AI (`.aimodel`) for iOS / iPadOS / macOS 27+ deployment.
---

# Apple Core AI Export (`.aimodel`)

!!! warning "Experimental — Use with Caution"

    Core AI export is **experimental and work-in-progress**. The `.aimodel` runs on iOS, iPadOS and macOS 27 or later, and `dynamic_batch=True` is not supported: export one `.aimodel` per batch size instead.

[Core AI](https://developer.apple.com/documentation/coreai) is Apple's on-device inference framework from iOS, iPadOS and macOS 27. `format="coreai"` traces the model with `torch.export` and converts it with [coreai-torch](https://github.com/apple/coreai-torch) into an `.aimodel` asset — no ONNX step. Core AI decides at load time whether the CPU, the GPU or the Neural Engine runs it.

## Prerequisites

```bash
pip install "rfdetr[coreai]"
```

`coreai-torch` supports Python 3.11 to 3.13 and installs on macOS (Apple silicon) and Linux x86-64, so an `.aimodel` can be exported on either. Running it needs the Core AI runtime of iOS, iPadOS or macOS 27.

## Basic Core AI Export

```python
from rfdetr import RFDETRNano

model = RFDETRNano(pretrain_weights="<path/to/checkpoint.pth>")

model.export(format="coreai")
```

This produces `output/rfdetr-nano_fp32.aimodel`. Segmentation and keypoint models export the same way. Pass `coreai_precision="float16"` for a half-size `rfdetr-nano_fp16.aimodel` whose input and outputs are float16 as well.

The asset keeps the contract of the other formats: one fixed `[batch, 3, H, W]` input, resized without antialiasing and ImageNet-normalized, as in the [ONNX Inference](onnx.md) example. Unlike CoreML, the tensors keep their names — `input`, then `dets` and `labels`, plus `masks` or `keypoints` — and any `notes` are stored in the asset metadata under `rfdetr_notes`.

## Core AI Inference Example

=== "Python"

    ```python
    import asyncio

    import coreai.runtime as rt
    import numpy as np


    async def run(image: np.ndarray) -> dict[str, np.ndarray]:
        model = await rt.AIModel.load("output/rfdetr-nano_fp32.aimodel", rt.SpecializationOptions.default())
        outputs = await model.load_function("main")({"input": rt.NDArray(image)})
        return {name: outputs[name].numpy() for name in ("dets", "labels")}


    # image: (1, 3, H, W) float32, preprocessed as in the ONNX Inference example
    outputs = asyncio.run(run(image))
    ```

=== "Swift"

    ```swift
    import CoreAI

    let model = try await AIModel(contentsOf: url)  // SpecializationOptions.default
    let main = try model.loadFunction(named: "main")!
    var outputs = try await main.run(inputs: ["input": input])  // input: NDArray [1, 3, H, W]
    let dets = outputs.remove("dets")!.ndArray!
    let labels = outputs.remove("labels")!.ndArray!
    ```

## Precision, Compute Units and Latency

**Start with float32.** With the default specialization Core AI runs a float32 `.aimodel` on the GPU, where it matches eager PyTorch detection for detection. Single-image latency of pretrained models with public test images (batch 1; M5 Pro Mac: macOS 27.0, Python runtime, median of 100 runs after 10 warm-ups; M4 iPad Air and A15 iPhone 13: 27.0, native Swift runtime in a release-profile app, median of three runs of 20 after 2 warm-ups):

| Model, precision          | Core AI default | Core AI CPU | CoreML `ALL` | CoreML `CPU_ONLY` |
| ------------------------- | --------------- | ----------- | ------------ | ----------------- |
| `RFDETRNano` fp32, Mac    | 7.4 ms          | 30.6 ms     | 7.7 ms       | 28.0 ms           |
| `RFDETRNano` fp16, Mac    | 3.6 ms          | 20.0 ms     | 3.4 ms       | 14.2 ms           |
| `RFDETRMedium` fp32, Mac  | 15.7 ms         | 68.4 ms     | 16.0 ms      | 64.8 ms           |
| `RFDETRSegNano` fp32, Mac | 11.1 ms         | 47.4 ms     | 10.8 ms      | 45.1 ms           |
| `RFDETRNano` fp32, iPad   | 18.2 ms         | —           | 18.6 ms      | —                 |
| `RFDETRNano` fp16, iPad   | 24.1 ms         | 20.2 ms     | —            | —                 |
| `RFDETRNano` fp32, iPhone | 52.3 ms         | —           | 44.2 ms      | —                 |
| `RFDETRNano` fp16, iPhone | 30.5 ms         | 32.6 ms     | —            | —                 |

On Macs and M-series iPads Core AI and CoreML run RF-DETR at the same speed; choose by the framework your application targets. On iOS and iPadOS the default specialization places a float16 `.aimodel` on the Neural Engine. Whether that pays off depends on the chip:

- On the M4 iPad it is slower than the float32 GPU path.
- On the A15 iPhone 13 it is the fastest option, 1.4× faster than CoreML float32.

The first load compiles the asset for the Neural Engine (5 to 9 s on these devices; cached afterwards), so measure on your target devices before choosing float16.

!!! note "float16 and the Neural Engine"

    On the Neural Engine a float16 `topk` returns corrupt indices, which would make RF-DETR's two-stage query selection gather the wrong encoder tokens and detect nothing (a float16 failure with the same symptom is reported in [apple/coreai-torch#115](https://github.com/apple/coreai-torch/issues/115)). The exporter therefore runs that one `topk` in float32; the rest of a float16 graph stays float16. With it, float16 `RFDETRNano` on the Neural Engine of an M5 Pro Mac scores 47.97 box AP on COCO val2017, against 48.02 for float32.

!!! warning "Keypoint models: do not run float16 on the Neural Engine"

    A float16 `RFDETRKeypointPreview` `.aimodel` terminates the process when Core AI runs it on the Neural Engine: the first inference aborts inside MPSGraph (`ANERegion.mm:414: ANE inference operation failed`), and no error reaches the caller. iOS and iPadOS choose the Neural Engine for float16 by default. Measured on macOS 27.0 (26A428), M5 Pro, with a Neural Engine preference; the same asset is correct with `SpecializationOptions.cpu_only()` or a GPU preference, and float32 is correct on every compute unit. Export keypoint models in float32, which is the default.

## How the Conversion Works

`coreai-torch` has no lowering for `aten.grid_sampler_2d`, which the deformable attention uses, so the exporter decomposes it into gathers. Its in-bounds masks use float arithmetic rather than a comparison-to-bool chain, which the Core AI runtime can mishandle ([apple/coreai-torch#11](https://github.com/apple/coreai-torch/issues/11)). The approach follows the RF-DETR port in the community [coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo).
