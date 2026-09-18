---
description: Advanced RF-DETR training with resume, early stopping, multi-GPU DDP, gradient checkpointing, and memory optimization for large models.
---

# Advanced Training

This page covers advanced training topics including resuming training, early stopping, multi-GPU training, and memory optimization techniques.

!!! tip "PTL API for deeper customisation"

    All examples on this page use the `RFDETR.train()` high-level API. For custom callbacks, non-default loggers, and fine-grained distributed training control, see the [Custom Training API](customization.md) guide.

## FP8 Training on NVIDIA CUDA

FP8 is an opt-in training mode for supported NVIDIA GPUs. RF-DETR uses Lightning's built-in Transformer Engine precision plugin; no custom accelerator is needed. Eligible layers use FP8 computation with BF16 weights. Installing the `cuda` extra alone does not change training precision: `amp_dtype` still defaults to `"auto"`. The optional Transformer Engine-aware CUDA-graph path is separate from Lightning's precision plugin and is described in [CUDA Graph Training](#cuda-graph-training).

### Install the CUDA extra

The `cuda` extra installs `transformer-engine[pytorch]>=2.19,<3` on Linux x86-64. The upstream framework extra is named `pytorch`, not `torch`. Transformer Engine does not support Apple MPS; the dependency is skipped on macOS and other platforms outside this extra's platform marker. Ordinary CUDA and MPS training do not need this extra.

1. Install CUDA-enabled PyTorch for your driver and CUDA environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

2. Install the [Transformer Engine prerequisites](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html), including CUDA toolkit headers, cuDNN, and the compiler needed to build its PyTorch extension. A GPU driver alone is insufficient. NVIDIA documents CUDA 12.8 or later for Blackwell.

3. Install RF-DETR's training and CUDA extras into that same environment:

    ```bash
    python -m pip install --progress-bar on --no-build-isolation 'rfdetr[train,cuda]'
    python -m pip install --progress-bar on 'jedi>=0.18'
    python -m pip check
    ```

    With `uv`, use `uv pip install --no-build-isolation 'rfdetr[train,cuda]'`. From a local RF-DETR checkout, replace `rfdetr[train,cuda]` with `.[train,cuda]`. Restart the notebook kernel after installing or replacing compiled dependencies.

!!! warning "Match the CUDA core library to your environment"

    Successful package resolution does not validate CUDA toolkit, driver, or PyTorch ABI compatibility. Our historical dependency-resolution check selected `transformer-engine-cu13` for Transformer Engine 2.18.0; do not treat that old lock outcome as a promise for every environment, and do not assume a CUDA 12 PyTorch build selects CUDA 12 in that older resolver path. The tested Transformer Engine 2.19 PyTorch installer derives its core CUDA major from `torch.version.cuda`, but you must still verify the resolved package, toolkit, driver, cuDNN, and ABI together. Adding a second core library alone does not establish compatibility with the PyTorch extension.

The Transformer Engine-aware FP8 graph path was tested with Transformer Engine 2.19.0 and uses the current named-argument API of [`make_graphed_callables`](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html#transformer_engine.pytorch.make_graphed_callables). Install it through RF-DETR's existing `cuda` extra; the tested 2.19 PyTorch installer derives its core CUDA major from `torch.version.cuda`, so this guide adds no separate core wheel or CUDA-version override. The PyTorch extension is compiled against the active CUDA toolkit, cuDNN, driver, and C++ ABI; a wheel resolving successfully does not prove that those components match. Follow [NVIDIA's Transformer Engine prerequisites](https://docs.nvidia.com/deeplearning/transformer-engine/installation.html), use the same environment as RF-DETR, and restart the kernel after installing compiled dependencies.

### Verify and enable FP8

Check the environment before starting a long run. Importing the PyTorch extension verifies more than importing the top-level metapackage:

```python
from importlib.metadata import version

import torch
import transformer_engine.pytorch

print("Transformer Engine:", version("transformer-engine"))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

These checks establish import and device availability, not FP8 kernel compatibility. Run a short training job that reaches validation and training shutdown before committing to a longer run:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
model.train(
    dataset_dir="path/to/small/coco-format-dataset",
    output_dir="output/fp8-smoke",
    epochs=1,
    batch_size=4,
    amp_dtype="fp8",
    use_ema=True,
    run_test=True,
)
```

FP8 requires an explicit integer `batch_size`; `batch_size="auto"` is rejected because its probe does not exercise Transformer Engine layers. Choose a smaller micro-batch and adjust `grad_accum_steps` manually when needed.

FP8 requires model AMP to be enabled. CPU, MPS, TPU/XLA, FSDP, and DeepSpeed combinations are rejected; use DDP for supported multi-GPU FP8 training.

### Troubleshooting FP8

| Symptom                                                              | Meaning and action                                                                                                                                                                                                                             |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Missing `transformer_engine` requirement or empty metapackage        | Install the `cuda` extra in the active Python environment. The PyTorch extension is required; the bare metapackage is insufficient.                                                                                                            |
| Missing CUDA headers, undefined symbols, or extension import failure | Verify toolkit, cuDNN, driver, and PyTorch/Transformer Engine build compatibility using NVIDIA's installation guide.                                                                                                                           |
| Missing JAX extension warning                                        | RF-DETR uses the PyTorch extension. JAX support is not required for this training path; verify `import transformer_engine.pytorch` succeeds.                                                                                                   |
| Linear-layer dimensions are not divisible by 8 and 16                | Lightning skips incompatible layers. This warning is nonfatal and means FP8 coverage is partial. Do not change class counts or output shapes just to suppress it.                                                                              |
| Model summary says Transformer Engine precision is unsupported       | The displayed parameter-memory estimate is a fallback estimate, not measured GPU memory.                                                                                                                                                       |
| EMA tensor dtype/device mismatch during updates                      | Update RF-DETR to a revision that initializes EMA after Lightning precision conversion. This was an integration bug, not an FP8 hardware limitation.                                                                                           |
| Pickled Transformer Engine extra-state refusal at training shutdown  | Update RF-DETR to a revision containing the EMA extra-state fix. EMA transfers exclude serialized `_extra_state` while retaining weights, buffers, and counters. Do not enable `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE` as a routine workaround. |

EMA checkpoint transfers intentionally omit FP8 scaling history; they do not promise bitwise continuation of that history. Full trainer checkpoints and regular non-EMA checkpoints have separate loading paths. Validate the specific resume, inference, and export workflow you need before relying on it for a long experiment.

### Measure the benefit

Compare `amp_dtype="fp8"` with `amp_dtype="bf16"` on the same GPU, model, dataset, resolution, physical batch size, gradient accumulation, and EMA settings. Use separate output directories, exclude warm-up from step timing, and include validation/checkpoint overhead when comparing whole epochs. Record peak GPU memory and validation accuracy as well as throughput. Smaller models or input-bound workloads may see no speedup or a slowdown; faster FP8 matrix operations do not guarantee a faster epoch.

## Resume Training

You can resume training from a previously saved full checkpoint by passing the path to `last.ckpt` using the `resume` argument. This is useful when training is interrupted or you want to continue fine-tuning an already partially trained model.

The training loop will automatically load:

- Model weights
- Optimizer state
- Learning rate scheduler state
- Training epoch number

!!! warning "Lightweight checkpoints resume without optimizer/scheduler state"

    The above applies to the trainer's own full checkpoints (`last.ckpt`, `checkpoint_<epoch>.ckpt`). The best-model tracker also writes four lighter `.pth` files — `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, `checkpoint_best_total.pth`, `last_ema.pth` — that intentionally omit optimizer/scheduler state to stay small. New files with matching configured callbacks can restore callback state (EMA and early stopping). Best-score tracking additionally requires `output_dir` to be the exact directory where the checkpoint was written. Files created before callback-state persistence (or with an empty callback section) restart callback state. The optimizer and LR scheduler always start cold. `resume=` logs the applicable warning; pass a full trainer checkpoint instead if you need optimizer/scheduler continuity.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        resume="output/last.ckpt",
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        resume="output/last.ckpt",
    )
    ```

!!! tip "Resume vs Pretrain Weights"

    - Use `resume="last.ckpt"` to continue training with optimizer state
    - Use `pretrain_weights="checkpoint_best_total.pth"` when initializing a model to start fresh training from those weights

---

## Early Stopping

Early stopping monitors the validation task metric selected by `best_model_metric` and halts training if improvements remain below a threshold for a set number of epochs. With the default `best_model_metric="map"`, detection models use box mAP, segmentation models use mask mAP, and keypoint models use COCO keypoint AP. With `best_model_metric="mar"`, detection and segmentation models use box mAR and keypoint models use keypoint mAR; mAR for detection and segmentation is evaluated using the configured `eval_max_dets` limit, while keypoint mAR uses fixed COCO `maxDets=20`.

### Basic Usage

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        early_stopping=True,
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        early_stopping=True,
    )
    ```

### Configuration Options

| Parameter                  | Default | Description                                                        |
| -------------------------- | ------- | ------------------------------------------------------------------ |
| `early_stopping_patience`  | 10      | Number of epochs without improvement before stopping               |
| `early_stopping_min_delta` | 0.001   | Minimum metric change to count as improvement                      |
| `early_stopping_use_ema`   | False   | Use EMA model metrics for comparisons                              |
| `best_model_metric`        | "map"   | Metric family for best checkpoint / early stopping: "map" or "mar" |

### Advanced Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=200,
    early_stopping=True,
    early_stopping_patience=15,  # Wait 15 epochs before stopping
    early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
    early_stopping_use_ema=True,  # Track EMA model performance
)
```

### How It Works

1. After each epoch, the validation task metric is computed
2. If the metric improves by at least `min_delta`, the patience counter resets
3. If the metric doesn't improve, the patience counter increments
4. When patience counter reaches `patience`, training stops
5. The best checkpoint is already saved as `checkpoint_best_total.pth`

```
Epoch 10: <selected-metric> = 0.450 (best: 0.450) - counter: 0
Epoch 11: <selected-metric> = 0.455 (best: 0.455) - counter: 0 (improved)
Epoch 12: <selected-metric> = 0.454 (best: 0.455) - counter: 1 (no improvement)
Epoch 13: <selected-metric> = 0.453 (best: 0.455) - counter: 2
...
Epoch 22: <selected-metric> = 0.452 (best: 0.455) - counter: 10 → STOP
```

---

## Multi-GPU Training

RF-DETR's training stack is built on PyTorch Lightning, so multi-GPU and multi-node training use the Lightning `Trainer` strategies directly. You can start multi-GPU runs through the high-level API or by using the Lightning primitives explicitly.

### Using RFDETR.train() with multiple GPUs

Create a training script and launch it with `torchrun`:

```python
# train.py
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,  # per-GPU batch size
    grad_accum_steps=1,
    lr=1e-4,
    output_dir="output",
    devices="auto",  # required — see note below
)
```

```bash
torchrun --nproc_per_node=4 train.py
```

!!! warning "Pass `devices=` explicitly"

    `build_trainer()` defaults to `devices=1`. Without overriding this, training silently runs on a single GPU even when `torchrun` launches multiple processes.

    Pass `devices="auto"` to use all GPUs visible to the process, or pass an explicit integer (e.g. `devices=4`). These values are forwarded to `build_trainer` via `**trainer_kwargs`:

    ```python
    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=1,
        lr=1e-4,
        output_dir="output",
        devices="auto",  # or devices=4
    )
    ```

### Batch Size with Multiple GPUs

When using multiple GPUs, your effective batch size is multiplied by the number of GPUs:

```
effective_batch_size = batch_size × grad_accum_steps × num_gpus
```

**Example configurations for effective batch size of 16:**

| GPUs | `batch_size` | `grad_accum_steps` | Effective |
| ---- | ------------ | ------------------ | --------- |
| 1    | 4            | 4                  | 16        |
| 2    | 4            | 2                  | 16        |
| 4    | 4            | 1                  | 16        |
| 8    | 2            | 1                  | 16        |

!!! warning "Adjust for GPU count"

    When switching between single and multi-GPU training, remember to adjust `batch_size` and `grad_accum_steps` to maintain the same effective batch size.

### Multi-Node Training

For training across multiple machines, pass the standard `torchrun` flags:

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train.py
```

Run this command on each node, changing `--node_rank` accordingly.

### Keypoint / Pose models

Keypoint models (`RFDETRKeypointPreview`) train under `DistributedDataParallel` on multiple GPUs and multiple nodes exactly like detection models — build a script and launch it with `torchrun`, setting `devices=` (e.g. `"auto"` or an integer like `8`):

```python
# train_pose.py
from rfdetr import RFDETRKeypointPreview

model = RFDETRKeypointPreview()

model.train(
    dataset_dir="path/to/keypoint-dataset",
    epochs=100,
    batch_size=2,  # per-GPU batch size
    grad_accum_steps=1,  # see note below for multi-GPU accumulation behavior
    lr=1e-4,
    output_dir="output",
    devices="auto",  # or devices=8
)
```

```bash
torchrun --nproc_per_node=8 train_pose.py
```

!!! note "Gradient accumulation on multi-GPU keypoint training"

    Keypoint models use **manual optimization** so the per-step box-count loss normalization is computed over the full accumulated batch. Intermediate microbatches accumulate gradients locally on each rank. The backward pass that closes the accumulation window synchronizes the full accumulated gradient before the optimizer step, avoiding redundant DDP reductions while preserving full-effective-batch normalization.

    Sharded strategies (FSDP / DeepSpeed) are **not** supported for keypoint models — use `ddp` (or `strategy="auto"` with `devices > 1`).

### Advanced multi-GPU options (PTL API)

For fine-grained control over strategy, sync batch norm, precision, and other distributed settings, use the Lightning API directly.

→ **[Multi-GPU with the PTL API](customization.md#multi-gpu-training)**

---

## Custom Augmentations

RF-DETR uses torchvision-native default augmentations during training. Passing a non-empty `aug_config` switches to one of two optional backends, selected by `augmentation_backend`:

- **CPU (default when `aug_config` is set):** [Albumentations](https://albumentations.ai/) integration, with access to over 70 image transformations optimized for object detection.
- **GPU (`augmentation_backend="kornia"` or `"auto"` with CUDA):** [Kornia](https://kornia.readthedocs.io/) integration, applying augmentations on-batch on the GPU instead of per-sample on CPU workers.

Both optional backends share the same `aug_config` dictionary format. See [Augmentation Backend Values](augmentations.md#augmentation-backend-values) for the full set of accepted `augmentation_backend` strings, including `"torchvision"` to force the default pipeline regardless of what's installed. Install the optional augmentation extra before using custom `aug_config` dictionaries or the built-in presets:

```bash
pip install "rfdetr[train,augment]"
```

→ **[Complete Augmentation Guide](augmentations.md)** - Configuration examples, best practices, troubleshooting, and advanced topics.

### Quick Start

Pass an `aug_config` dictionary to `model.train()`. Each key is an Albumentations transform name; the value is a dict of keyword arguments for that transform:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
    aug_config={
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.5},
        "Rotate": {"limit": 45, "p": 0.5},
    },
)
```

Use a built-in preset by importing it from `rfdetr.datasets.aug_configs`:

```python
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model.train(dataset_dir="path/to/dataset", aug_config=AUG_AGGRESSIVE)
```

To disable all augmentations, pass an empty dict:

```python
model.train(dataset_dir="path/to/dataset", aug_config={})
```

`aug_config` controls only the augmentation stack (Albumentations on CPU, or the equivalent Kornia pipeline when `augmentation_backend="kornia"`/`"auto"`). The training resize pipeline's independent resize → crop → resize branch (Option B) is controlled separately by `scale_jitter`:

```python
# Keep aug_config's default augmentation stack, but disable random crop/scale jitter
model.train(dataset_dir="path/to/dataset", scale_jitter=False)
```

`scale_jitter` defaults to `True`. Set it to `False` to use direct resize only — no random crop, so annotations near image borders are never clipped.

---

## CUDA Graph Training

For a long-running detection job on one NVIDIA GPU, enable CUDA graph replay on the model constructor:

```python
from rfdetr import RFDETRNano

model = RFDETRNano(cuda_graphs=True)
model.train(dataset_dir="path/to/dataset")
```

An INFO line at train start confirms replay is enabled, and each captured input shape logs its own INFO line; without one of these the run is eager (and the reason was logged as a warning). The first batch at each input signature runs CUDA graph warm-up and capture. Later batches with the same batch size, resolution, dtype, device, and autocast mode replay the captured forward and backward. BF16 capture can cache one graph per signature, so multi-scale training retains one graph and memory pool per resolution. Graph pools are private CUDA allocations; compare peak memory and the largest usable batch size as well as step time on your own workload.

There are two capture routes:

- **BF16/eager route:** `cuda_graphs=True` with `compile=False` supports single-GPU detection with BF16 (`amp_dtype="bf16"` or `"auto"` resolving to BF16). Segmentation, keypoints, distributed training, gradient checkpointing, CPU, MPS, FP16, FP32, and unsupported trainer combinations remain eager and log a warning. BF16 graph replay supports gradient accumulation; the runner preserves gradients already accumulated by earlier microbatches.
- **FP8/Transformer Engine route:** with a tested-compatible Transformer Engine release (2.19.0 was tested), use `cuda_graphs=True`, `compile=False`, and `amp_dtype="fp8"`. This route calls Transformer Engine's native `make_graphed_callables` with the active Lightning FP8 recipe, disables quantized-parameter caching, and clones returned parameter gradients. It is intentionally limited to single-GPU detection, `grad_accum_steps=1`, no gradient checkpointing, `multi_scale=False` and `square_resize_div_64=True`. The last setting keeps the external image boundary on a fixed square shape; aspect-preserving batches with `square_resize_div_64=False` stay eager. It captures one fixed batch/resolution signature; a later shape change raises. Unsupported shape/accumulation combinations stay eager with a warning; an incompatible Transformer Engine version or precision-plugin misconfiguration stops training with an error instead.

`cuda_graphs=True` may be combined with `compile=True` (see [Combining CUDA graphs with compilation](#combining-cuda-graphs-with-compilation)), but that is the Inductor route. Ordinary FP8 with `compile=False` still uses Lightning's normal Transformer Engine plugin. When all three flags — `cuda_graphs=True`, `compile=True`, and `amp_dtype="fp8"` — are selected, RF-DETR warns and keeps the run compile-only; it does not wrap the compiled module in the Transformer Engine graph helper. FP8 with `compile=True` but `cuda_graphs=False` uses ordinary compiled FP8 training without this graph-routing warning. A capture failure stops training with the original exception attached: an invalidated capture can leave CUDA state unsafe for further operations. Restart the process before retrying with graphs disabled; catching the exception and continuing in the same process is not a supported fallback.

Use the Transformer Engine-aware route explicitly. `TrainConfig.multi_scale` defaults to `"per-batch"`, so it must be explicitly set to `False` (or `"off"`) here (along with `square_resize_div_64=True`) — otherwise the run silently falls back to eager mode, with only a log warning as the signal:

```python
from rfdetr import RFDETRNano

model = RFDETRNano(cuda_graphs=True, compile=False)
model.train(
    dataset_dir="path/to/dataset",
    amp_dtype="fp8",
    batch_size=4,
    grad_accum_steps=1,
    multi_scale=False,
    square_resize_div_64=True,
)
```

CUDA graphs and `compile=True` remove different costs, and they can be enabled together. The two measurements immediately below compare CUDA graphs with eager training; the matrix in [Combining CUDA graphs with compilation](#combining-cuda-graphs-with-compilation) compares all four combinations on one GPU. Compare steady-state throughput, startup time, and peak memory with identical data, batch size, precision, and resolutions before choosing. Compilation keeps positional-embedding interpolation eager to avoid PyTorch's symbolic antialiased-bicubic backward failure; its interpolation math and gradients are unchanged. This is a compatibility mitigation, not an upstream compiler fix.

As one BF16 reference point, RF-DETR Nano on an NVIDIA L4 with BF16, batch 4, deterministic synthetic detection batches, and a fixed 8-resolution multi-scale set (`expanded_scales=False`) reduced the median Lightning training batch from 149.8 ms to 101.7 ms across five runs — a 32.4% median reduction in the five paired per-run measurements. Peak allocated memory increased from 2,395 MiB to 3,466 MiB, and peak reserved memory increased from 2,754 MiB to 14,550 MiB for those 8 captured graph pools. `TrainConfig`'s own default is `expanded_scales=True`, which resolves to 11 multi-scale resolutions for this model and was not benchmarked; expect proportionally more graph pools and higher peak reserved memory under that default. The speed and memory cost depend on the model, shapes, batch size, GPU, and PyTorch version; this option is not a good fit when memory already limits the batch size.

For the Transformer Engine route, a synthetic RF-DETR Nano run at 384 px and batch 4 on an RTX PRO 6000 Blackwell used 20 warm-up and 50 measured steps: eager averaged 78.075 ms per step versus 40.066 ms with Transformer Engine-aware graphs (about 1.95x throughput), with one graph capture serving 70 calls. This is a single fixed-shape experiment, not COCO parity or accuracy evidence, and does not establish a large-batch gain. Repeat numerical-parity and performance checks on your target GPU before treating it as a production baseline.

!!! note "FP8 CUDA-graph capture/replay tests are CI-blind today"

    `tests/training/test_cuda_graph_te.py` covers this route, but the repository's GPU CI runner is a Tesla T4 (compute capability 7.5). The test's FP8-hardware gate checks `torch.cuda.get_device_capability(0) >= (8, 9)` before anything imports Transformer Engine, so on a T4 the whole module skips rather than exercising the capture path. These tests are validated only by local or manual runs on Ada/Hopper-class or newer hardware until an 8.9+ GPU is available in CI.

A second reference point, on real data, shows the other end of the range. RF-DETR Nano on an NVIDIA RTX PRO 6000 (Blackwell, 96 GB) with BF16, batch 64, resolution 384, `multi_scale=False`, COCO train2017 through the Albumentations CPU pipeline on Colab: eager and `cuda_graphs=True` both ran at 3.54 it/s (about 8 min 40 s per epoch, one captured graph, no fallback), while `compile=True` finished the same epoch in 7 min 10 s, about 17% faster. Nothing was wrong with the capture; the two options remove different costs.

### Choosing between CUDA graphs and compilation

- **CUDA graphs** remove kernel-launch gaps and CPU dispatch between the model's kernels. They pay when each kernel is short, so the GPU idles between launches: small batch size, small model, low resolution, or a fast GPU driven by a slow CPU. They do nothing for the kernels themselves. When the batch is large enough that every kernel runs for a long time, replay measures the same as eager — that is expected, not a capture failure.
- **`compile=True`** (Inductor) fuses elementwise operations and reduces memory traffic, so the kernels themselves get cheaper. That gain does not depend on the batch size, but compilation adds startup time and, with `dynamic=True`, one compiled graph per run.
- Rule of thumb from the two measurements above: at batch 4 with a mid-range GPU (L4) graphs gave 32%; at batch 64 on a high-end GPU they gave 0% and compilation gave 17%. Prefer graphs when the effective batch per step is in the single digits or the progress bar shows the GPU far from saturated; prefer compilation for large batches. In between, run 200 steps of each with identical settings and compare the steady-state it/s — startup and capture time distort the first epoch.
- For large batches, combining the two buys nothing over `compile=True` alone; for small batches it stacks. The section below has the numbers.

### Combining CUDA graphs with compilation

Set both flags and RF-DETR hands CUDA graph replay to Inductor's CUDA graph trees (the mechanism behind `torch.compile(mode="reduce-overhead")`, passed here as the `triton.cudagraphs` compile option so it coexists with RF-DETR's other Inductor settings). The compiled forward and backward kernels are recorded once per input shape and replayed afterwards; the eager graph runner used by `cuda_graphs=True` alone stays off, and an INFO line at construction says so.

```python
from rfdetr import RFDETRNano

model = RFDETRNano(compile=True, cuda_graphs=True)
model.train(dataset_dir="path/to/dataset")
```

Measured on one NVIDIA RTX PRO 6000 (Blackwell) with RF-DETR Nano, BF16, resolution 384, `multi_scale=False`, synthetic detection batches, torch 2.11, 100 timed steps after warm-up (GPU busy is the fraction of wall time with a kernel running, from a profiler trace):

| batch [img] | eager [img/s] | `cuda_graphs` [img/s] | `compile` [img/s] | both [img/s]  | both vs `compile` [×] | GPU busy eager → both [-] |
| ----------- | ------------- | --------------------- | ----------------- | ------------- | --------------------- | ------------------------- |
| 4           | 78.7          | 95.3 (1.21×)          | 96.8 (1.23×)      | 116.0 (1.47×) | **1.20×**             | 0.32 → 0.42               |
| 64          | 246.8         | 249.7 (1.01×)         | 322.8 (1.31×)     | 325.7 (1.32×) | 1.01×                 | 0.85 → 0.83               |

At batch 4 the two gains stack: compilation alone leaves the launch gaps in place (GPU busy stays at 0.32), and graph replay of the compiled kernels removes them. At batch 64 the compiled kernels already run back to back, so replay adds under 1%, inside run-to-run noise; the remaining 15–20% of each step is spent outside the model (matcher, loss, data loading) and neither option reaches it. The same run on an A100 gave 1.70× over `compile` at batch 4 and the same parity at batch 64, so the small-batch gain depends on how much launch overhead the host adds.

Scope and cost:

- Validated for single-GPU detection training without gradient accumulation. Segmentation, keypoints, gradient checkpointing, `grad_accum_steps > 1`, and multi-device or multi-node runs fall back to `compile=True` alone with a warning. Gradient accumulation is the hard limit: cudagraph trees allocate the backward's gradient outputs inside the graph pool, so a `.grad` adopted by the first microbatch is overwritten by the next replay.
- The fallback is decided from `TrainConfig` when the model is built. If the trainer later resolves gradient accumulation (for example through `trainer_kwargs`) or more than one process (`devices="auto"` on a multi-GPU host), training stops at train start with a `RuntimeError` naming the conflict, because the model is already compiled with graph replay by then.
- The combined path skips the eager runner's BF16-only gate; BF16 is the measured path. FP8 remains compile-only with a warning, by design; it does not use the Transformer Engine graph wrapper.
- The benchmark ran training steps only (`use_ema=False`, no validation). With the default `use_ema=True` validation runs the eager EMA copy; with `use_ema=False` or `eval_base_model=True` the compiled model also records an eval-mode graph. Neither validation nor EMA was exercised under the combined path.
- With `multi_scale="per-batch"` (or `"per-sample"`), `dynamic=True` compilation records one graph per resolution (each with its own memory pool) and PyTorch warns after `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit` (8) distinct shapes. Benchmark `multi_scale=False` first, then confirm memory with the multi-scale set you train on.
- Compilation start-up dominates short runs: 5–7 minutes on the RTX PRO 6000 and 13–15 minutes on an A100 for Nano before the first timed step. `TORCH_LOGS=cudagraphs` prints the graph partitions Inductor records, including any operator it refused to capture.

---

## Memory Optimization

### Gradient Checkpointing

For large models or high resolutions, enable gradient checkpointing to trade compute for memory.

!!! warning "Constructor parameter — not a `train()` parameter"

    `gradient_checkpointing` is a `ModelConfig` field and must be passed to the **model constructor**, not to `train()`. Passing it to `train()` will raise a `ValidationError` because `TrainConfig` has `extra="forbid"`.

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(gradient_checkpointing=True)

model.train(
    dataset_dir="path/to/dataset",
    batch_size=2,  # May be able to increase with checkpointing
)
```

This re-computes activations during the backward pass instead of storing them, reducing memory usage by ~30-40% at the cost of ~20% slower training.

### Memory-Efficient Configurations

| Memory Level      | Configuration                                                                          |
| ----------------- | -------------------------------------------------------------------------------------- |
| Very Low (8GB)    | `batch_size=1`, `grad_accum_steps=16`, `gradient_checkpointing=True`, `resolution=576` |
| Low (12GB)        | `batch_size=2`, `grad_accum_steps=8`, `gradient_checkpointing=True`                    |
| Medium (16GB)     | `batch_size=4`, `grad_accum_steps=4`                                                   |
| High (24GB)       | `batch_size=8`, `grad_accum_steps=2`                                                   |
| Very High (40GB+) | `batch_size=16`, `grad_accum_steps=1`, `resolution=768`                                |

---

## Training Tips

### Learning Rate Tuning

- **Fine-tuning from COCO weights (default):** Use default learning rates (`lr=1e-4`, `lr_encoder=1.5e-4`)
- **Small dataset (\<1000 images):** Consider lower `lr` (e.g., `5e-5`) to prevent overfitting
- **Large dataset (>10000 images):** May benefit from higher `lr` (e.g., `2e-4`)

### Epoch Count

| Dataset Size      | Recommended Epochs |
| ----------------- | ------------------ |
| < 500 images      | 100-200            |
| 500-2000 images   | 50-100             |
| 2000-10000 images | 30-50              |
| > 10000 images    | 20-30              |

Use early stopping to automatically determine the optimal stopping point.

### Data Augmentation

RF-DETR applies built-in augmentations during training:

- Random resizing
- Random cropping
- Horizontal flipping

These defaults are implemented with torchvision and don't require manual setup. Color jitter and other advanced transforms are available through the optional Albumentations presets and custom `aug_config` dictionaries.

---

## Troubleshooting

### Out of Memory (OOM)

If you encounter CUDA out of memory errors:

1. Reduce `batch_size`
2. Enable `gradient_checkpointing=True` (pass to the model constructor, not `train()`)
3. Reduce `resolution`
4. Increase `grad_accum_steps` to maintain effective batch size

### Training Too Slow

1. Increase `batch_size` (if memory allows)
2. Use multiple GPUs with DDP
3. Ensure you're using GPU (check `device="cuda"`)
4. Consider using a smaller model (e.g., `RFDETRSmall` instead of `RFDETRLarge`)

### Loss Not Decreasing

1. Check that your dataset is correctly formatted
2. Verify annotations are correct (bounding boxes in correct format)
3. Try reducing the learning rate
4. Check for class imbalance in your dataset
