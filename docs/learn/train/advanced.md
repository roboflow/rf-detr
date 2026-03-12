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

Early stopping monitors validation mAP and halts training if improvements remain below a threshold for a set number of epochs. This prevents wasted computation once the model has converged.

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
| `early_stopping_min_delta` | 0.001   | Minimum mAP change to count as improvement           |
| `early_stopping_use_ema`   | False   | Use EMA model's mAP for comparisons                  |

### Advanced Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=200,
    early_stopping=True,
    early_stopping_patience=15,  # Wait 15 epochs before stopping
    early_stopping_min_delta=0.005,  # Require 0.5% mAP improvement
    early_stopping_use_ema=True,  # Track EMA model performance
)
```

### How It Works

1. After each epoch, validation mAP is computed
2. If mAP improves by at least `min_delta`, the patience counter resets
3. If mAP doesn't improve, the patience counter increments
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

### Advanced multi-GPU options (PTL API)

For fine-grained control over strategy, sync batch norm, precision, and other distributed settings, use the Lightning API directly.

→ **[Multi-GPU with the PTL API](customization.md#multi-gpu-training)**

---

## Custom Augmentations

RF-DETR supports advanced data augmentations using the [Albumentations](https://albumentations.ai/) library, providing access to over 70 different image transformations optimized for object detection.

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

Use a built-in preset by importing it from `rfdetr.datasets.aug_config`:

```python
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model.train(dataset_dir="path/to/dataset", aug_config=AUG_AGGRESSIVE)
```

To disable all augmentations, pass an empty dict:

```python
model.train(dataset_dir="path/to/dataset", aug_config={})
```

---

## Memory Optimization

### Gradient Checkpointing

For large models or high resolutions, enable gradient checkpointing to trade compute for memory:

```python
model.train(
    dataset_dir="path/to/dataset",
    gradient_checkpointing=True,
    batch_size=2,  # May be able to increase with checkpointing
)
```

This re-computes activations during the backward pass instead of storing them, reducing memory usage by ~30-40% at the cost of ~20% slower training.

### Memory-Efficient Configurations

| Memory Level      | Configuration                                                                          |
| ----------------- | -------------------------------------------------------------------------------------- |
| Very Low (8GB)    | `batch_size=1`, `grad_accum_steps=16`, `gradient_checkpointing=True`, `resolution=560` |
| Low (12GB)        | `batch_size=2`, `grad_accum_steps=8`, `gradient_checkpointing=True`                    |
| Medium (16GB)     | `batch_size=4`, `grad_accum_steps=4`                                                   |
| High (24GB)       | `batch_size=8`, `grad_accum_steps=2`                                                   |
| Very High (40GB+) | `batch_size=16`, `grad_accum_steps=1`, `resolution=784`                                |

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
- Color jittering
- Horizontal flipping

These are automatically configured and don't require manual setup.

---

## Troubleshooting

### Out of Memory (OOM)

If you encounter CUDA out of memory errors:

1. Reduce `batch_size`
2. Enable `gradient_checkpointing=True`
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
