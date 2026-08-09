---
description: Advanced RF-DETR training with resume, early stopping, multi-GPU DDP, gradient checkpointing, and memory optimization for large models.
---

# Advanced Training

This page covers advanced training topics including resuming training, early stopping, multi-GPU training, and memory optimization techniques.

!!! tip "PTL API for deeper customisation"

    All examples on this page use the `RFDETR.train()` high-level API. For custom callbacks, non-default loggers, and fine-grained distributed training control, see the [Custom Training API](customization.md) guide.

## Resume Training

You can resume training from a previously saved checkpoint by passing the path to the `checkpoint.pth` file using the `resume` argument. This is useful when training is interrupted or you want to continue fine-tuning an already partially trained model.

The training loop will automatically load:

- Model weights
- Optimizer state
- Learning rate scheduler state
- Training epoch number

!!! warning "Lightweight checkpoints resume without optimizer/scheduler state"

    The above applies to the trainer's own full checkpoints (`last.ckpt`, `checkpoint_<epoch>.ckpt`). The best-model
    tracker also writes four lighter `.pth` files — `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`,
    `checkpoint_best_total.pth`, `last_ema.pth` — that intentionally omit optimizer/scheduler state to stay small.
    Resuming from one of these restores model weights, epoch count, and callback state (best-score tracking, EMA,
    early stopping), but the optimizer and LR scheduler always start cold. `resume=` logs a warning when it detects
    one of these files; pass a full trainer checkpoint instead if you need optimizer/scheduler continuity.

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
        resume="output/checkpoint.pth",
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
        resume="output/checkpoint.pth",
    )
    ```

!!! tip "Resume vs Pretrain Weights"

    - Use `resume="checkpoint.pth"` to continue training with optimizer state
    - Use `pretrain_weights="checkpoint_best_total.pth"` when initializing a model to start fresh training from those weights

---

## Early Stopping

Early stopping monitors the validation task metric and halts training if improvements remain below a threshold for a
set number of epochs. Detection and segmentation models use box mAP; keypoint preview models use COCO keypoint AP.

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

| Parameter                  | Default | Description                                          |
| -------------------------- | ------- | ---------------------------------------------------- |
| `early_stopping_patience`  | 10      | Number of epochs without improvement before stopping |
| `early_stopping_min_delta` | 0.001   | Minimum metric change to count as improvement        |
| `early_stopping_use_ema`   | False   | Use EMA model metrics for comparisons                |

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
Epoch 10: mAP = 0.450 (best: 0.450) - counter: 0
Epoch 11: mAP = 0.455 (best: 0.455) - counter: 0 (improved)
Epoch 12: mAP = 0.454 (best: 0.455) - counter: 1 (no improvement)
Epoch 13: mAP = 0.453 (best: 0.455) - counter: 2
...
Epoch 22: mAP = 0.452 (best: 0.455) - counter: 10 → STOP
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

    `build_trainer()` defaults to `devices=1`. Without overriding this, training silently
    runs on a single GPU even when `torchrun` launches multiple processes.

    Pass `devices="auto"` to use all GPUs visible to the process, or pass an explicit
    integer (e.g. `devices=4`). These values are forwarded to `build_trainer` via
    `**trainer_kwargs`:

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

Keypoint models (`RFDETRKeypointPreview`) train under `DistributedDataParallel` on multiple GPUs and
multiple nodes exactly like detection models — build a script and launch it with `torchrun`, setting
`devices=` (e.g. `"auto"` or an integer like `8`):

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

    Keypoint models use **manual optimization** so the per-step box-count loss normalization is
    computed over the full accumulated batch. As a result, gradients synchronize on **every**
    microbatch rather than only at the end of an accumulation window. Training with
    `grad_accum_steps > 1` on multiple GPUs is still numerically correct, but performs one
    `all_reduce` per microbatch (i.e. `grad_accum_steps`× the necessary communication). For best
    throughput, scale with more GPUs / a larger per-GPU `batch_size` and keep `grad_accum_steps=1`.

    Sharded strategies (FSDP / DeepSpeed) are **not** supported for keypoint models — use `ddp`
    (or `strategy="auto"` with `devices > 1`).

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

`aug_config` controls only the augmentation stack (Albumentations on CPU, or the
equivalent Kornia pipeline when `augmentation_backend="kornia"`/`"auto"`). The training
resize pipeline's independent resize → crop → resize branch (Option B) is controlled
separately by `scale_jitter`:

```python
# Keep aug_config's default augmentation stack, but disable random crop/scale jitter
model.train(dataset_dir="path/to/dataset", scale_jitter=False)
```

`scale_jitter` defaults to `True`. Set it to `False` to use direct resize only —
no random crop, so annotations near image borders are never clipped.

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
