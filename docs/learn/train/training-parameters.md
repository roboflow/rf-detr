---
description: Complete RF-DETR training parameter reference. Learning rate, batch size, EMA, early stopping, resolution, and hardware configuration.
---

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
    output_dir="output",
)
```

## Core Parameters

These are the essential parameters for training:

| Parameter          | Type            | Default    | Description                                                                                                                                                       |
| ------------------ | --------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_dir`      | `str`           | Required   | Path to your dataset directory. RF-DETR auto-detects if it's in COCO or YOLO format. See [Dataset Formats](dataset-formats.md).                                   |
| `output_dir`       | `str`           | `"output"` | Directory where training artifacts (checkpoints, logs) are saved.                                                                                                 |
| `epochs`           | `int`           | `100`      | Number of full passes over the training dataset.                                                                                                                  |
| `batch_size`       | `int or "auto"` | `4`        | Number of samples processed per iteration. Higher values require more GPU memory. Set to `"auto"` to probe the GPU for the largest safe batch size automatically. |
| `grad_accum_steps` | `int`           | `4`        | Accumulates gradients over multiple mini-batches. Use with `batch_size` to achieve effective batch size.                                                          |
| `resume`           | `str`           | `None`     | Path to a saved checkpoint to continue training. Restores model weights, optimizer state, and scheduler.                                                          |

### Understanding Batch Size

The **effective batch size** is calculated as:

```
effective_batch_size = batch_size × grad_accum_steps × num_gpus
```

Recommended configurations for different GPUs (targeting effective batch size of 16):

| GPU      | VRAM    | `batch_size` | `grad_accum_steps` |
| -------- | ------- | ------------ | ------------------ |
| A100     | 40-80GB | 16           | 1                  |
| RTX 4090 | 24GB    | 8            | 2                  |
| RTX 3090 | 24GB    | 8            | 2                  |
| T4       | 16GB    | 4            | 4                  |
| RTX 3070 | 8GB     | 2            | 8                  |

## Learning Rate Parameters

| Parameter    | Type    | Default  | Description                                                                                                                                                          |
| ------------ | ------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lr`         | `float` | `1e-4`   | Learning rate for most parts of the model.                                                                                                                           |
| `lr_encoder` | `float` | `1.5e-4` | Learning rate specifically for the backbone encoder. Can be set lower than `lr` if you want to fine-tune the encoder more conservatively than the rest of the model. |

!!! tip "Learning rate tips"

    - Start with the default values for fine-tuning
    - If the model doesn't converge, try reducing `lr` by half
    - For training from scratch (not recommended), you may need higher learning rates

## Resolution Parameters

| Parameter    | Type  | Default         | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------ | ----- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolution` | `int` | Model-dependent | Input image resolution. Higher values can improve accuracy but require more memory. Each model has its own valid block size: current standard detection checkpoints use multiples of 32, current segmentation checkpoints use multiples of 24 (most variants) or 12 (`RFDETRSegNano`), and the definitive rule is that the resolution must be divisible by `patch_size * num_windows` for the selected model. |

Common resolution values for currently documented checkpoints:

- Detection: `384`, `512`, `576`, `704`
- Segmentation: `312`, `384`, `432`, `504`, `624`, `768`

For example, `RFDETRSegXLarge` uses `624x624`, which is valid because `624` is divisible by `24`.

## Regularization Parameters

| Parameter      | Type    | Default | Description                                                                           |
| -------------- | ------- | ------- | ------------------------------------------------------------------------------------- |
| `weight_decay` | `float` | `1e-4`  | L2 regularization coefficient. Helps prevent overfitting by penalizing large weights. |

## Hardware Parameters

| Parameter                | Type   | Default | Description                                                                                                                                                                                                                                    |
| ------------------------ | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device`                 | `str`  | `None`  | Device to run training on. `None` means auto-detected by PyTorch Lightning. Options: `"cuda"`, `"cpu"`, `"mps"` (Apple Silicon).                                                                                                               |
| `gradient_checkpointing` | `bool` | `False` | **Constructor-only parameter** — pass to the model constructor (`RFDETRMedium(gradient_checkpointing=True)`), not to `train()`. Re-computes activations during backprop to reduce memory usage by ~30-40% at the cost of ~20% slower training. |

## EMA (Exponential Moving Average)

| Parameter | Type   | Default | Description                                                                                                          |
| --------- | ------ | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `use_ema` | `bool` | `True`  | Enables Exponential Moving Average of weights. Produces a smoothed checkpoint that often improves final performance. |

!!! info "What is EMA?"

    EMA maintains a moving average of the model weights throughout training. This smoothed version often generalizes better than the raw weights and is commonly used for the final model.

## Checkpoint Parameters

| Parameter             | Type  | Default | Description                                                                                                                            |
| --------------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `checkpoint_interval` | `int` | `10`    | Frequency (in epochs) at which model checkpoints are saved. More frequent saves provide better coverage but consume more storage.      |
| `skip_best_epochs`    | `int` | `0`     | Ignore the first N epochs when tracking best checkpoints and early-stopping patience. Useful when fine-tuning from a prior checkpoint. |

### Checkpoint Files

During training, multiple checkpoints are saved:

| File                          | Description                               |
| ----------------------------- | ----------------------------------------- |
| `checkpoint.pth`              | Most recent checkpoint (for resuming)     |
| `checkpoint_<N>.pth`          | Periodic checkpoint at epoch N            |
| `checkpoint_best_ema.pth`     | Best validation performance (EMA weights) |
| `checkpoint_best_regular.pth` | Best validation performance (raw weights) |
| `checkpoint_best_total.pth`   | Final best model for inference            |

Best validation performance uses the task metric for the model family: box mAP for detection/segmentation and COCO
keypoint AP for keypoint preview.

## Early Stopping Parameters

| Parameter                  | Type    | Default | Description                                                                              |
| -------------------------- | ------- | ------- | ---------------------------------------------------------------------------------------- |
| `early_stopping`           | `bool`  | `False` | Enable early stopping based on the validation task metric.                               |
| `early_stopping_patience`  | `int`   | `10`    | Number of epochs without improvement before stopping.                                    |
| `early_stopping_min_delta` | `float` | `0.001` | Minimum metric change to qualify as an improvement.                                      |
| `early_stopping_use_ema`   | `bool`  | `False` | Whether to track improvements using EMA model metrics.                                   |
| `skip_best_epochs`         | `int`   | `0`     | Ignore the first N epochs (0..N-1) for best-model selection and early-stopping patience. |

### Early Stopping Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=200,
    batch_size=4,
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.005,
    skip_best_epochs=3,
)
```

This configuration will:

- Train for up to 200 epochs
- Ignore epochs 0-2 for best-checkpoint tracking and patience counting
- Stop early if the validation metric doesn't improve by at least 0.005 for 15 consecutive epochs

!!! note "Transfer learning with `pretrain_weights`"

    When fine-tuning from `pretrain_weights`, the pretrained model's epoch-0 validation metric can be artificially high
    relative to the training trajectory on the new dataset. This causes `checkpoint_best_total.pth` to always contain
    the untrained pretrained weights and may trigger early stopping prematurely. Use `skip_best_epochs` to defer
    best-checkpoint selection and patience counting until the model has had time to adapt.

## Logging Parameters

| Parameter     | Type   | Default | Description                                                                                                                                                                                                 |
| ------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensorboard` | `bool` | `True`  | Enable TensorBoard logging. Requires `pip install "rfdetr[loggers]"`. If the `tensorboard` package is not installed, training continues with a `UserWarning` and TensorBoard output is silently suppressed. |
| `wandb`       | `bool` | `False` | Enable Weights & Biases logging. Requires `pip install "rfdetr[loggers]"`.                                                                                                                                  |
| `project`     | `str`  | `None`  | Project name for W&B logging.                                                                                                                                                                               |
| `run`         | `str`  | `None`  | Run name for W&B logging. If not specified, W&B assigns a random name.                                                                                                                                      |

### Logging Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    tensorboard=True,
    wandb=True,
    project="my-detection-project",
    run="experiment-001",
)
```

## Evaluation Parameters

| Parameter               | Type                | Default | Description                                                                                                        |
| ----------------------- | ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `eval_max_dets`         | `int`               | `500`   | Maximum number of detections per image considered during COCO evaluation. Lower values speed up evaluation.        |
| `eval_interval`         | `int`               | `1`     | Run COCO evaluation every N epochs. Set to a higher value to reduce evaluation overhead during long training runs. |
| `log_per_class_metrics` | `bool`              | `True`  | Log per-class AP metrics to the console and loggers. Disable to reduce log verbosity when there are many classes.  |
| `progress_bar`          | str \| bool \| None | `None`  | Progress bar style: `"tqdm"`, `"rich"`, or `None`. Legacy booleans are still accepted.                             |

## Keypoint Preview Parameters

These parameters apply when training `RFDETRKeypointPreview` on COCO keypoint annotations or Ultralytics YOLO pose labels.

| Parameter                     | Type                  | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------- | --------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_keypoints_per_class`     | `list[int]`           | `[17]`  | **Constructor parameter** — pass to `RFDETRKeypointPreview(num_keypoints_per_class=...)`. Keypoint schema by model label slot. A zero entry marks a detection-only class slot; legacy checkpoints may use a background-first `[0, 17]` schema.                                                                                                                                                                                                                                                            |
| `keypoint_flip_pairs`         | `list[int]`           | `[]`    | Flat left/right keypoint index pairs used to swap joints after horizontal-flip augmentation. YOLO `flip_idx` metadata is a permutation; RF-DETR converts it to this pair-list form during automatic schema inference when possible — it extracts only symmetric mutual pairs where `flip_idx[i] == j` and `flip_idx[j] == i`. Asymmetric entries and self-mapped keypoints (`flip_idx[i] == i`) are silently omitted; supply `keypoint_flip_pairs` explicitly when your `flip_idx` includes such entries. |
| `keypoint_l1_loss_coef`       | `float`               | `1.0`   | Weight for keypoint coordinate L1 loss in keypoint preview training.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `keypoint_findable_loss_coef` | `float`               | `1.0`   | Weight for keypoint findable/objectness loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `keypoint_visible_loss_coef`  | `float`               | `1.0`   | Weight for keypoint visibility loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `keypoint_nll_loss_coef`      | `float`               | `1.0`   | Weight for keypoint negative-log-likelihood loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `keypoint_oks_sigmas`         | `list[float] \| None` | `None`  | Per-keypoint OKS sigma values used for COCO AP evaluation. When `None`, 17-keypoint person datasets use the evaluator's standard COCO sigmas and custom keypoint counts use RF-DETR's uniform custom fallback. Pass explicit values, such as schema-inferred sigmas, when you need a specific custom OKS policy.                                                                                                                                                                                          |

!!! note "OKS sigma values: flat vs per-keypoint"

    `infer_coco_keypoint_schema` and `infer_yolo_keypoint_schema` return a flat sigma of 0.1 for all inferred keypoints, and the keypoint demos pass those values explicitly for custom datasets. If `keypoint_oks_sigmas=None`, COCO person-keypoint evaluation uses the standard 17-keypoint COCO sigmas, while non-17 custom keypoint counts use RF-DETR's uniform custom fallback. Flat custom sigmas are not directly comparable to official COCO benchmark numbers.

## Advanced Parameters

The parameters below are available for fine-grained control over training behaviour. Most users can leave these at their defaults.

### Scheduler and Regularization

| Parameter       | Type    | Default  | Description                                                                                                 |
| --------------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| `lr_scheduler`  | `str`   | `"step"` | Learning rate scheduler type. Options: `"step"` (step decay at `lr_drop`) or `"cosine"` (cosine annealing). |
| `lr_min_factor` | `float` | `0.0`    | Floor for the cosine scheduler, expressed as a fraction of the initial LR. Ignored when using `"step"`.     |
| `warmup_epochs` | `float` | `0.0`    | Number of epochs for linear learning rate warmup at the start of training.                                  |
| `drop_path`     | `float` | `0.0`    | Stochastic depth drop-path rate applied to the backbone. Higher values add more regularization.             |

### Runtime and Accelerator

| Parameter           | Type   | Default  | Description                                                                                      |
| ------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------ |
| `accelerator`       | `str`  | `"auto"` | PyTorch Lightning accelerator selection. `"auto"` picks GPU if available, then MPS, then CPU.    |
| `seed`              | `int`  | `None`   | Global random seed for reproducibility. `None` means no fixed seed is set.                       |
| `fp16_eval`         | `bool` | `False`  | Run evaluation passes in FP16 precision. Reduces memory usage but may lower numerical precision. |
| `compute_val_loss`  | `bool` | `True`   | Compute and log the detection loss on the validation set each epoch.                             |
| `compute_test_loss` | `bool` | `True`   | Compute and log the detection loss during the final test run.                                    |

### DataLoader Tuning

| Parameter            | Type   | Default | Description                                                                                               |
| -------------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------- |
| `pin_memory`         | `bool` | `None`  | Pin host memory in the DataLoader for faster GPU transfers. `None` defers to PyTorch Lightning's default. |
| `persistent_workers` | `bool` | `None`  | Keep DataLoader worker processes alive between epochs. `None` defers to PyTorch Lightning's default.      |
| `prefetch_factor`    | `int`  | `None`  | Number of batches to prefetch per DataLoader worker. `None` uses PyTorch's built-in default.              |

## Complete Parameter Reference

Below is a summary table of all training parameters:

| Parameter                  | Type                | Default        | Description                                                                                                                           |
| -------------------------- | ------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_dir`              | str                 | Required       | Path to COCO or YOLO formatted dataset with train/valid/test splits.                                                                  |
| `output_dir`               | str                 | "output"       | Directory for checkpoints, logs, and other training artifacts.                                                                        |
| `epochs`                   | int                 | 100            | Number of full passes over the dataset.                                                                                               |
| `batch_size`               | int or "auto"       | 4              | Samples per iteration. Set to `"auto"` to let RF-DETR probe the GPU for the largest safe batch size. Balance with `grad_accum_steps`. |
| `grad_accum_steps`         | int                 | 4              | Gradient accumulation steps for effective larger batch sizes.                                                                         |
| `lr`                       | float               | 1e-4           | Learning rate for the model (excluding encoder).                                                                                      |
| `lr_encoder`               | float               | 1.5e-4         | Learning rate for the backbone encoder.                                                                                               |
| `resolution`               | int                 | Model-specific | Input image size (must be divisible by the selected model's `patch_size * num_windows`).                                              |
| `weight_decay`             | float               | 1e-4           | L2 regularization coefficient.                                                                                                        |
| `device`                   | str                 | "cuda"         | Training device: cuda, cpu, or mps.                                                                                                   |
| `use_ema`                  | bool                | True           | Enable Exponential Moving Average of weights.                                                                                         |
| `gradient_checkpointing`   | bool                | False          | Trade compute for memory during backprop.                                                                                             |
| `checkpoint_interval`      | int                 | 10             | Save checkpoint every N epochs.                                                                                                       |
| `resume`                   | str                 | None           | Path to checkpoint for resuming training.                                                                                             |
| `tensorboard`              | bool                | True           | Enable TensorBoard logging.                                                                                                           |
| `wandb`                    | bool                | False          | Enable Weights & Biases logging.                                                                                                      |
| `project`                  | str                 | None           | W&B project name.                                                                                                                     |
| `run`                      | str                 | None           | W&B run name.                                                                                                                         |
| `early_stopping`           | bool                | False          | Enable early stopping.                                                                                                                |
| `early_stopping_patience`  | int                 | 10             | Epochs without improvement before stopping.                                                                                           |
| `early_stopping_min_delta` | float               | 0.001          | Minimum validation metric change to qualify as improvement.                                                                           |
| `early_stopping_use_ema`   | bool                | False          | Use EMA model for early stopping metrics.                                                                                             |
| `eval_max_dets`            | int                 | 500            | Maximum detections per image considered during COCO evaluation.                                                                       |
| `eval_interval`            | int                 | 1              | Run COCO evaluation every N epochs.                                                                                                   |
| `log_per_class_metrics`    | bool                | True           | Log per-class AP metrics to the console and loggers.                                                                                  |
| `progress_bar`             | str \| bool \| None | None           | Progress bar style: `"tqdm"`, `"rich"`, or `None`. Legacy booleans are still accepted.                                                |
| `accelerator`              | str                 | "auto"         | PyTorch Lightning accelerator. "auto" selects GPU/MPS/CPU automatically.                                                              |
| `seed`                     | int                 | None           | Random seed for reproducibility. None means no fixed seed.                                                                            |
| `lr_scheduler`             | str                 | "step"         | Learning rate scheduler type: "step" or "cosine".                                                                                     |
| `lr_min_factor`            | float               | 0.0            | Minimum LR as a fraction of the initial LR (cosine scheduler floor).                                                                  |
| `warmup_epochs`            | float               | 0.0            | Number of linear warmup epochs at the start of training.                                                                              |
| `drop_path`                | float               | 0.0            | Stochastic depth drop-path rate for the backbone.                                                                                     |
| `compute_val_loss`         | bool                | True           | Compute and log loss during validation.                                                                                               |
| `compute_test_loss`        | bool                | True           | Compute and log loss during the test run.                                                                                             |
| `fp16_eval`                | bool                | False          | Run evaluation in FP16 precision to reduce memory usage.                                                                              |
| `pin_memory`               | bool                | None           | Pin DataLoader memory. None defers to PyTorch Lightning's default.                                                                    |
| `persistent_workers`       | bool                | None           | Keep DataLoader workers alive between epochs. None uses PTL default.                                                                  |
| `prefetch_factor`          | int                 | None           | Number of batches prefetched per worker. None uses PyTorch default.                                                                   |
