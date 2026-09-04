---
description: Advanced RF-DETR training with resume, early stopping, multi-GPU DDP, gradient checkpointing, and memory optimization for large models.
---

# Advanced Training

This page covers advanced training topics including resuming training, early stopping, multi-GPU training, and memory optimization techniques.

!!! tip "PTL API for deeper customisation"

    All examples on this page use the `RFDETR.train()` high-level API. For custom callbacks, non-default loggers, and fine-grained distributed training control, see the [Custom Training API](customization.md) guide.

## FP8 Training on NVIDIA CUDA

FP8 is an opt-in training mode for supported NVIDIA GPUs. RF-DETR uses Lightning's built-in Transformer Engine precision plugin; no custom accelerator is needed. Eligible layers use FP8 computation with BF16 weights. Installing the `cuda` extra alone does not change training precision: `amp_dtype` still defaults to `"auto"`.

### Install the CUDA extra

The `cuda` extra installs `transformer-engine[pytorch]>=2.18,<3` on Linux x86-64. The upstream framework extra is named `pytorch`, not `torch`. Transformer Engine does not support Apple MPS; the dependency is skipped on macOS and other platforms outside this extra's platform marker. Ordinary CUDA and MPS training do not need this extra.

1. Install CUDA-enabled PyTorch for your driver and CUDA environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

2. Install the [Transformer Engine prerequisites](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html), including CUDA toolkit headers, cuDNN, and the compiler needed to build its PyTorch extension. A GPU driver alone is insufficient. NVIDIA documents CUDA 12.8 or later for Blackwell.

3. Install RF-DETR's training and CUDA extras into that same environment:

    ```bash
    python -m pip install --no-build-isolation 'rfdetr[train,cuda]'
    ```

    With `uv`, use `uv pip install --no-build-isolation 'rfdetr[train,cuda]'`. From a local RF-DETR checkout, replace `rfdetr[train,cuda]` with `.[train,cuda]`. Restart the notebook kernel after installing or replacing compiled dependencies.

!!! warning "Match the CUDA core library to your environment"

    Successful package resolution does not validate CUDA toolkit, driver, or PyTorch ABI compatibility. Our dependency-resolution check selected `transformer-engine-cu13` for Transformer Engine 2.18.0; do not assume the extra selects CUDA 12 merely because your PyTorch reports CUDA 12.8. Check the resolved packages against NVIDIA's installation instructions, including its `core_cu12` / `core_cu13` selection and source-build guidance. Adding a second core library alone does not establish compatibility with the PyTorch extension.

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
)
```

FP8 requires model AMP to be enabled. CPU, MPS, FSDP, and DeepSpeed combinations are rejected; use DDP for supported multi-GPU FP8 training.

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
    grad_accum_steps=1,  # recommended on multi-GPU — see note below
    lr=1e-4,
    output_dir="output",
    devices="auto",  # or devices=8
)
```

```bash
torchrun --nproc_per_node=8 train_pose.py
```

!!! note "Prefer `grad_accum_steps=1` on multi-GPU for keypoints"

    Keypoint models use **manual optimization** so the per-step box-count loss normalization is computed over the full accumulated batch. As a result, gradients synchronize on **every** microbatch rather than only at the end of an accumulation window. Training with `grad_accum_steps > 1` on multiple GPUs is still numerically correct, but performs one `all_reduce` per microbatch (i.e. `grad_accum_steps`× the necessary communication). For best throughput, scale with more GPUs / a larger per-GPU `batch_size` and keep `grad_accum_steps=1`.

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
