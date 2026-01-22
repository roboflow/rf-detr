# Training Parameters

This page provides a complete reference of all parameters available when training RF-DETR models.

## Basic Example

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output"
)
```

## Core Parameters

These are the essential parameters for training:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_dir` | `str` | Required | Path to your dataset directory. RF-DETR auto-detects if it's in COCO or YOLO format. See [Dataset Formats](dataset-formats.md). |
| `output_dir` | `str` | `"output"` | Directory where training artifacts (checkpoints, logs) are saved. |
| `epochs` | `int` | `100` | Number of full passes over the training dataset. |
| `batch_size` | `int` | `4` | Number of samples processed per iteration. Higher values require more GPU memory. |
| `grad_accum_steps` | `int` | `4` | Accumulates gradients over multiple mini-batches. Use with `batch_size` to achieve effective batch size. |
| `resume` | `str` | `None` | Path to a saved checkpoint to continue training. Restores model weights, optimizer state, and scheduler. |

### Understanding Batch Size

The **effective batch size** is calculated as:

```
effective_batch_size = batch_size × grad_accum_steps × num_gpus
```

Recommended configurations for different GPUs (targeting effective batch size of 16):

| GPU | VRAM | `batch_size` | `grad_accum_steps` |
|-----|------|--------------|-------------------|
| A100 | 40-80GB | 16 | 1 |
| RTX 4090 | 24GB | 8 | 2 |
| RTX 3090 | 24GB | 8 | 2 |
| T4 | 16GB | 4 | 4 |
| RTX 3070 | 8GB | 2 | 8 |

## Learning Rate Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `1e-4` | Learning rate for most parts of the model. |
| `lr_encoder` | `float` | `1.5e-4` | Learning rate specifically for the backbone encoder. Useful for fine-tuning the encoder at a slower pace than the rest of the model. |

!!! tip "Learning rate tips"
    - Start with the default values for fine-tuning
    - If the model doesn't converge, try reducing `lr` by half
    - For training from scratch (not recommended), you may need higher learning rates

## Resolution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | `int` | Model-dependent | Input image resolution. Higher values can improve accuracy but require more memory. **Must be divisible by 56.** |

Common resolution values:

| Resolution | Memory Usage | Use Case |
|------------|--------------|----------|
| 560 | Low | Small objects, limited GPU memory |
| 672 | Medium | Balanced (default for many models) |
| 784 | High | High accuracy requirements |
| 896 | Very High | Maximum quality (requires large GPU) |

## Regularization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_decay` | `float` | `1e-4` | L2 regularization coefficient. Helps prevent overfitting by penalizing large weights. |

## Hardware Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"cuda"` | Device to run training on. Options: `"cuda"`, `"cpu"`, `"mps"` (Apple Silicon). |
| `gradient_checkpointing` | `bool` | `False` | Re-computes parts of the forward pass during backpropagation to reduce memory usage. Lowers memory needs but increases training time. |

## EMA (Exponential Moving Average)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ema` | `bool` | `True` | Enables Exponential Moving Average of weights. Produces a smoothed checkpoint that often improves final performance. |

!!! info "What is EMA?"
    EMA maintains a moving average of the model weights throughout training. This smoothed version often generalizes better than the raw weights and is commonly used for the final model.

## Checkpoint Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_interval` | `int` | `10` | Frequency (in epochs) at which model checkpoints are saved. More frequent saves provide better coverage but consume more storage. |
| `resume` | `str` | `None` | Path to a saved checkpoint to continue training. Restores model weights, optimizer state, and scheduler. |

### Checkpoint Files

During training, multiple checkpoints are saved:

| File | Description |
|------|-------------|
| `checkpoint.pth` | Most recent checkpoint (for resuming) |
| `checkpoint_<N>.pth` | Periodic checkpoint at epoch N |
| `checkpoint_best_ema.pth` | Best validation performance (EMA weights) |
| `checkpoint_best_regular.pth` | Best validation performance (raw weights) |
| `checkpoint_best_total.pth` | Final best model for inference |

## Early Stopping Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping` | `bool` | `False` | Enable early stopping based on validation mAP. |
| `early_stopping_patience` | `int` | `10` | Number of epochs without improvement before stopping. |
| `early_stopping_min_delta` | `float` | `0.001` | Minimum change in mAP to qualify as an improvement. |
| `early_stopping_use_ema` | `bool` | `False` | Whether to track improvements using EMA model metrics. |

### Early Stopping Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=200,
    batch_size=4,
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.005
)
```

This configuration will:
- Train for up to 200 epochs
- Stop early if mAP doesn't improve by at least 0.005 for 15 consecutive epochs

## Logging Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensorboard` | `bool` | `True` | Enable TensorBoard logging. Requires `pip install "rfdetr[metrics]"`. |
| `wandb` | `bool` | `False` | Enable Weights & Biases logging. Requires `pip install "rfdetr[metrics]"`. |
| `project` | `str` | `None` | Project name for W&B logging. |
| `run` | `str` | `None` | Run name for W&B logging. If not specified, W&B assigns a random name. |

### Logging Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    tensorboard=True,
    wandb=True,
    project="my-detection-project",
    run="experiment-001"
)
```

## Complete Parameter Reference

Below is a summary table of all training parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_dir` | str | Required | Path to COCO or YOLO formatted dataset with train/valid/test splits. |
| `output_dir` | str | "output" | Directory for checkpoints, logs, and other training artifacts. |
| `epochs` | int | 100 | Number of full passes over the dataset. |
| `batch_size` | int | 4 | Samples per iteration. Balance with `grad_accum_steps`. |
| `grad_accum_steps` | int | 4 | Gradient accumulation steps for effective larger batch sizes. |
| `lr` | float | 1e-4 | Learning rate for the model (excluding encoder). |
| `lr_encoder` | float | 1.5e-4 | Learning rate for the backbone encoder. |
| `resolution` | int | Model-specific | Input image size (must be divisible by 56). |
| `weight_decay` | float | 1e-4 | L2 regularization coefficient. |
| `device` | str | "cuda" | Training device: cuda, cpu, or mps. |
| `use_ema` | bool | True | Enable Exponential Moving Average of weights. |
| `gradient_checkpointing` | bool | False | Trade compute for memory during backprop. |
| `checkpoint_interval` | int | 10 | Save checkpoint every N epochs. |
| `resume` | str | None | Path to checkpoint for resuming training. |
| `tensorboard` | bool | True | Enable TensorBoard logging. |
| `wandb` | bool | False | Enable Weights & Biases logging. |
| `project` | str | None | W&B project name. |
| `run` | str | None | W&B run name. |
| `early_stopping` | bool | False | Enable early stopping. |
| `early_stopping_patience` | int | 10 | Epochs without improvement before stopping. |
| `early_stopping_min_delta` | float | 0.001 | Minimum mAP change to qualify as improvement. |
| `early_stopping_use_ema` | bool | False | Use EMA model for early stopping metrics. |
