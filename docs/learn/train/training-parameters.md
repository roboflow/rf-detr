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
    batch_size="auto",
    lr=1e-4,
    output_dir="output",
)
```

`batch_size="auto"` requires a CUDA-capable GPU because it probes CUDA memory. For CPU or MPS training, provide a concrete integer batch size and use `grad_accum_steps` if memory limits the physical batch.

## Core Parameters

These are the essential parameters for training:

| Parameter                          | Type            | Default    | Description                                                                                                                                                                                                                                                            |
| ---------------------------------- | --------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_dir`                      | `str`           | Required   | Path to your dataset directory. RF-DETR auto-detects if it's in COCO or YOLO format. See [Dataset Formats](dataset-formats.md).                                                                                                                                        |
| `output_dir`                       | `str`           | `"output"` | Directory where training artifacts (checkpoints, logs) are saved.                                                                                                                                                                                                      |
| `epochs`                           | `int`           | `100`      | Number of full passes over the training dataset.                                                                                                                                                                                                                       |
| `batch_size`                       | `int or "auto"` | `4`        | Number of samples processed per iteration. Higher values require more GPU memory. Set to `"auto"` to probe the GPU for the largest safe batch size automatically.                                                                                                      |
| `grad_accum_steps`                 | `int`           | `1`        | Accumulates gradients over several mini-batches before each optimizer step. Opt-in: raise it only when memory caps `batch_size` below the nominal effective-batch target you want.                                                                                     |
| `eval_batch_size`                  | `int or None`   | `None`     | Batch size for the validation, test and predict dataloaders. `None` inherits `batch_size`. `no_grad` avoids autograd activation storage, but in-fit evaluation still shares device memory with the model and optimizer state and needs memory for its forward outputs. |
| `auto_batch_target_effective`      | `int`           | `16`       | Only used when `batch_size="auto"`. Global nominal effective-batch target; the probe derives a per-device target before scaling by `devices * num_nodes`.                                                                                                              |
| `auto_batch_max_targets_per_image` | `int`           | `100`      | Only used when `batch_size="auto"`. Synthetic target count per image the probe uses to simulate worst-case matcher/loss memory.                                                                                                                                        |
| `auto_batch_ema_headroom`          | `float`         | `0.7`      | Only used when `batch_size="auto"` and `use_ema=True`. Fraction of the probed batch size reserved as headroom for the EMA model's extra memory use. Must be in `(0, 1]`.                                                                                               |
| `resume`                           | `str`           | `None`     | Path to a saved checkpoint to continue training. Full `.ckpt` files restore model, optimizer, and scheduler state; lightweight best `.pth` files restart optimizer and scheduler state.                                                                                |

### Understanding Batch Size

The **nominal effective batch size** is calculated as:

```
effective_batch_size = batch_size × grad_accum_steps × num_gpus
```

**Set `batch_size` first, `grad_accum_steps` second.** Raise `batch_size` only as far as the selected model, task, resolution, and GPU allow, and leave `grad_accum_steps` at its default of `1` when possible. If hardware memory prevents the physical batch from reaching your nominal target, increase `grad_accum_steps` to recover that target's optimizer-step window. This is not an optimization-equivalent substitute: it splits the window across smaller forward/backward passes, changes microbatch cadence, and a small physical batch tends to leave the GPU under-occupied. On one L4 training `rfdetr-small`, `batch_size=16, grad_accum_steps=1` ran about 27% faster per epoch than `batch_size=4, grad_accum_steps=4` at the same nominal effective batch of 16, with mAP equal within run-to-run noise. That figure is one GPU and one dataset, so treat the direction as the lesson, not the number.

When you trade between the two, use the product `batch_size × grad_accum_steps` as a nominal effective-batch target. Keeping that product constant does not guarantee identical optimization behavior: changing `batch_size` changes the forward/backward microbatch cadence even when the nominal images-per-optimizer-update target is unchanged. Start with `batch_size="auto"` on CUDA when the available memory or workload is uncertain; on CPU or MPS, choose a conservative integer batch size instead.

Configurations reaching a nominal effective batch size of 16 on illustrative hardware:

| GPU      | VRAM    | `batch_size` | `grad_accum_steps` |
| -------- | ------- | ------------ | ------------------ |
| A100     | 40-80GB | 16           | 1                  |
| RTX 4090 | 24GB    | 8            | 2                  |
| RTX 3090 | 24GB    | 8            | 2                  |
| T4       | 16GB    | 4            | 4                  |
| RTX 3070 | 8GB     | 2            | 8                  |

These are illustrative starting configurations from the documented model/workload setup, not capacity guarantees. Actual safe values depend on the model, task, resolution, augmentations, optimizer state, and other workloads on the device. Use `batch_size="auto"` on CUDA or validate a conservative integer batch size before increasing it.

!!! note "The default nominal effective batch is 4, not 16"

    `grad_accum_steps` defaults to `1` (it was `4` in earlier versions), so the defaults give a nominal effective batch of `batch_size × 1 = 4`. If you were relying on the old defaults and want the previous nominal target of 16, set `grad_accum_steps=4` explicitly, then validate the resulting training behavior. Runs using `batch_size="auto"` are unaffected by this default change: the probe sets both values itself.

## Learning Rate Parameters

| Parameter          | Type              | Default   | Description                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------ | ----------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lr`               | `float`           | `1e-4`    | Learning rate for most parts of the model.                                                                                                                                                                                                                                                                                                                                                                          |
| `lr_encoder`       | `float`           | `1.5e-4`  | Learning rate specifically for the backbone encoder. Can be set lower than `lr` if you want to fine-tune the encoder more conservatively than the rest of the model.                                                                                                                                                                                                                                                |
| `optimizer`        | `str \| Callable` | `"adamw"` | Optimizer as a native `torch.optim` short name, dotted import path, or callable. Managed short names (native `torch.optim` only, e.g. `"adamw"`, `"sgd"`) have RF-DETR inject `lr`/`weight_decay`; a dotted import path (`"torch.optim.AdamW"`, `"pytorch_optimizer.Lion"`) or callable is built from `optimizer_kwargs` / its own bound arguments only. See [Custom optimizer](customization.md#custom-optimizer). |
| `optimizer_kwargs` | `dict`            | `{}`      | Keyword arguments for the optimizer constructor. Managed short names reserve `params`/`lr`/`weight_decay`/`fused`; explicit import paths take them here; ignored (with a warning) for callables.                                                                                                                                                                                                                    |

!!! tip "Learning rate tips"

    - Start with the default values for fine-tuning
    - If the model doesn't converge, try reducing `lr` by half
    - For training from scratch (not recommended), you may need higher learning rates

### Custom Optimizer Example

```python
model.train(
    dataset_dir="path/to/dataset",
    optimizer="pytorch_optimizer.Lion",  # third-party optimizer by import path (install it yourself)
    optimizer_kwargs={"weight_decouple": True},
)
```

Bare short names resolve to native `torch.optim` optimizers; any other optimizer is given by full dotted import path or a callable, always preserving RF-DETR's parameter groups and layer-wise learning rates.

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

| Parameter       | Type   | Default | Description                                                                                                                                                                                                                                                                                                            |
| --------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_ema`       | `bool` | `True`  | Enables Exponential Moving Average of weights. Produces a smoothed checkpoint that often improves final performance.                                                                                                                                                                                                   |
| `eval_ema_only` | `bool` | `False` | Validation-only: forward through the EMA model instead of the base model, halving per-batch validation compute. Requires `use_ema=True`. The base model is never evaluated that epoch, so `val/mAP_*` stays unpopulated; the real per-epoch score is logged under `val/ema_mAP_*` instead — see Evaluation Parameters. |

!!! info "What is EMA?"

    EMA maintains a moving average of the model weights throughout training. This smoothed version often generalizes better than the raw weights and is commonly used for the final model.

## Checkpoint Parameters

| Parameter             | Type  | Default | Description                                                                                                                            |
| --------------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `checkpoint_interval` | `int` | `10`    | Frequency (in epochs) at which model checkpoints are saved. More frequent saves provide better coverage but consume more storage.      |
| `skip_best_epochs`    | `int` | `0`     | Ignore the first N epochs when tracking best checkpoints and early-stopping patience. Useful when fine-tuning from a prior checkpoint. |

### Checkpoint Files

During training, multiple checkpoints are saved:

| File                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `last.ckpt`                   | Most recent full checkpoint (for resuming)                   |
| `checkpoint_<epoch>.ckpt`     | Periodic full checkpoint at an epoch                         |
| `checkpoint_best_ema.pth`     | Best EMA weights; lightweight callback state when available  |
| `checkpoint_best_regular.pth` | Best raw weights; lightweight callback state when available  |
| `checkpoint_best_total.pth`   | Final best model; lightweight callback state when available  |
| `last_ema.pth`                | Final EMA weights; lightweight callback state when available |

Best validation performance uses the task metric for the model family (`best_model_metric="map"`, the default): box mAP for detection, mask mAP for segmentation, and COCO keypoint AP for keypoint models. Set `best_model_metric="mar"` to rank checkpoints by mAR instead: detection and segmentation use box mAR, while keypoint models use keypoint mAR. mAR for detection and segmentation is evaluated using the configured `eval_max_dets` limit; keypoint mAR uses fixed COCO `maxDets=20`.

## Early Stopping Parameters

| Parameter                  | Type                   | Default | Description                                                                              |
| -------------------------- | ---------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `early_stopping`           | `bool`                 | `False` | Enable early stopping based on the validation task metric.                               |
| `early_stopping_patience`  | `int`                  | `10`    | Number of epochs without improvement before stopping.                                    |
| `early_stopping_min_delta` | `float`                | `0.001` | Minimum metric change to qualify as an improvement.                                      |
| `early_stopping_use_ema`   | `bool`                 | `False` | Whether to track improvements using EMA model metrics.                                   |
| `best_model_metric`        | `Literal["map","mar"]` | `"map"` | Metric family for best-checkpoint selection and early stopping — mAP or mAR.             |
| `skip_best_epochs`         | `int`                  | `0`     | Ignore the first N epochs (0..N-1) for best-model selection and early-stopping patience. |

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

    When fine-tuning from `pretrain_weights`, the pretrained model's epoch-0 validation metric can be artificially high relative to the training trajectory on the new dataset. This causes `checkpoint_best_total.pth` to always contain the untrained pretrained weights and may trigger early stopping prematurely. Use `skip_best_epochs` to defer best-checkpoint selection and patience counting until the model has had time to adapt.

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

| Parameter                    | Type                | Default | Description                                                                                                                                                                                                                                                          |
| ---------------------------- | ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval_max_dets`              | `int`               | `500`   | Maximum detections per image for detection/segmentation COCO AP and AR evaluation. Keypoint AP/AR uses fixed COCO `maxDets=20`; lower values speed up detection/segmentation evaluation.                                                                             |
| `eval_interval`              | `int`               | `1`     | Skip the whole COCO validation loop (forward pass, metric compute, EMA forward) on epochs that aren't a multiple of N, to reduce evaluation overhead during long training runs. The final epoch always validates regardless of this setting.                         |
| `log_per_class_metrics`      | `bool`              | `False` | Log per-class AP metrics to the console and loggers. Enable it to also run the underlying per-class `torchmetrics` computation. Aggregate mAP/mAR and F1 metrics remain available either way.                                                                        |
| `eval_ema_only`              | `bool`              | `False` | Forward through the EMA model only during validation, skipping the duplicate base-model pass. Requires `use_ema=True`. See [EMA](#ema-exponential-moving-average).                                                                                                   |
| `eval_masks_head_resolution` | `bool`              | `False` | Segmentation only. Skip upsampling predicted masks to full image resolution during validation, comparing at the mask head's native (lower) resolution instead. `val/segm_mAP` is then not comparable to a full-resolution run. No effect on `RFDETR.predict` output. |
| `progress_bar`               | str \| bool \| None | `None`  | Progress bar style: `"tqdm"`, `"rich"`, or `None`. Legacy booleans are still accepted. `"rich"` leaves each completed epoch's bar in the terminal history instead of overwriting it.                                                                                 |

### Validation performance

- `log_per_class_metrics=False` is the default. It retains aggregate mAP/mAR and F1/precision/recall while omitting per-class rows and their underlying per-class metric computation. Set it to `True` when per-class reporting is needed.
- `compute_val_loss="auto"` is the default. It computes `val/loss` only when a configured scheduler, checkpoint, or early-stopping callback monitors that key. Set it to `True` to always log validation loss or `False` to disable it; `False` is rejected when a configured consumer monitors `val/loss`.
- `eval_ema_only=True` requires `use_ema=True` and evaluates only EMA weights. Regular `val/mAP_*` keys are absent, EMA values use `val/ema_*`, and best-checkpoint or early-stopping routing follows the available EMA metric. `val/F1` remains under its regular key.
- `eval_interval` controls validation frequency, not the cost of a validation epoch: non-evaluation epochs skip the complete validation loop, while the final epoch always evaluates.
- Lowering `eval_max_dets` can reduce detection/segmentation evaluation work, but it also changes AP and AR semantics. Keypoint evaluation keeps COCO `maxDets=20`.

## Keypoint Preview Parameters

These parameters apply when training `RFDETRKeypointPreview` on COCO keypoint annotations or Ultralytics YOLO pose labels.

| Parameter                     | Type                  | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------- | --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_keypoints_per_class`     | `list[int]`           | `[17]`  | **Constructor parameter** — pass to `RFDETRKeypointPreview(num_keypoints_per_class=...)`. Keypoint schema by model label slot. A zero entry marks a detection-only class slot; legacy checkpoints may use a background-first `[0, 17]` schema.                                                                                                                                                                                                                                                                                                                                              |
| `keypoint_flip_pairs`         | `list[int]`           | `[]`    | Flat left/right keypoint index pairs used to swap joints after horizontal-flip augmentation. YOLO `flip_idx` metadata is a permutation; RF-DETR converts it to this pair-list form during automatic schema inference when possible — it extracts only symmetric mutual pairs where `flip_idx[i] == j` and `flip_idx[j] == i`. Asymmetric entries and self-mapped keypoints (`flip_idx[i] == i`) are silently omitted; supply `keypoint_flip_pairs` explicitly when your `flip_idx` includes such entries. See the note below for what an empty list means for horizontal-flip augmentation. |
| `keypoint_l1_loss_coef`       | `float`               | `1.0`   | Weight for keypoint coordinate L1 loss in keypoint preview training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `keypoint_findable_loss_coef` | `float`               | `1.0`   | Weight for keypoint findable/objectness loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `keypoint_visible_loss_coef`  | `float`               | `1.0`   | Weight for keypoint visibility loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `keypoint_nll_loss_coef`      | `float`               | `1.0`   | Weight for keypoint negative-log-likelihood loss.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `keypoint_oks_sigmas`         | `list[float] \| None` | `None`  | Per-keypoint OKS sigma values used for COCO AP evaluation. When `None`, 17-keypoint person datasets use the evaluator's standard COCO sigmas and custom keypoint counts use RF-DETR's uniform custom fallback. Pass explicit values, such as schema-inferred sigmas, when you need a specific custom OKS policy.                                                                                                                                                                                                                                                                            |

!!! warning "`keypoint_flip_pairs`: `None` vs `[]` vs a populated list"

    This value is tri-state, and the state — not just the value — controls whether horizontal-flip augmentations run at all, on both the default torchvision-native pipeline (`aug_config=None`) and a custom Albumentations `aug_config`:

    - `None` marks a detection-only pipeline. Horizontal-flip augmentations (torchvision's default flip, or `HorizontalFlip`/`Flip`/`D4` in your `aug_config`) are always kept, since there are no keypoint annotations that a flip could invalidate.
    - `[]` on a keypoint pipeline means no flip pairs are defined. RF-DETR then drops horizontal-flip augmentations rather than flip an image without knowing which keypoints to swap — this is intentional annotation-safety behavior, not a bug, but easy to trip over if you set `keypoint_flip_pairs=[]` yourself without expecting the augmentation to disappear.
    - A populated list on a keypoint pipeline supplies the actual left/right index pairs, so horizontal-flip augmentations run and swap the paired keypoints.

    The current pydantic default for `keypoint_flip_pairs` is `[]`, matching the field definition in `src/rfdetr/config.py`. See `_build_torchvision_pipeline` in `src/rfdetr/datasets/coco.py` for the default-backend gating check, and `AlbumentationsWrapper.from_config` in `src/rfdetr/datasets/transforms.py` for the Albumentations-backend equivalent — both use the same `keypoint_flip_pairs is not None and not keypoint_flip_pairs` check.

!!! note "OKS sigma values: flat vs per-keypoint"

    `infer_coco_keypoint_schema` and `infer_yolo_keypoint_schema` return a flat sigma of 0.1 for all inferred keypoints, and the keypoint demos pass those values explicitly for custom datasets. If `keypoint_oks_sigmas=None`, COCO person-keypoint evaluation uses the standard 17-keypoint COCO sigmas, while non-17 custom keypoint counts use RF-DETR's uniform custom fallback. Flat custom sigmas are not directly comparable to official COCO benchmark numbers.

## Advanced Parameters

The parameters below are available for fine-grained control over training behaviour. Most users can leave these at their defaults.

### Scheduler and Regularization

| Parameter               | Type              | Default      | Description                                                                                                                               |
| ----------------------- | ----------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `lr_scheduler`          | `str \| Callable` | `"step"`     | Scheduler preset (`"step"`/`"cosine"`), dotted import path, or callable. See [Custom LR scheduler](customization.md#custom-lr-scheduler). |
| `lr_scheduler_kwargs`   | `dict`            | `{}`         | Keyword arguments forwarded to an explicit scheduler; also carries `lr_drop` / `min_factor` for the managed presets.                      |
| `lr_scheduler_interval` | `str`             | `"step"`     | Stepping cadence for explicit schedulers: `"step"` (per optimizer step) or `"epoch"`. Managed presets always step per step.               |
| `lr_scheduler_monitor`  | `str`             | `"val/loss"` | Metric fed to `ReduceLROnPlateau` (stepped once per epoch).                                                                               |
| `lr_min_factor`         | `float`           | `0.0`        | **Deprecated** — pass `lr_scheduler_kwargs={"min_factor": ...}` instead. Cosine-preset floor, as a fraction of the initial LR.            |
| `lr_drop`               | `int`             | `100`        | **Deprecated** — pass `lr_scheduler_kwargs={"lr_drop": ...}` instead. Epoch at which the `"step"` preset drops the LR by 10x.             |
| `optimizer`             | `str \| Callable` | `"adamw"`    | Optimizer name, dotted import path, or callable. See [Custom optimizer](customization.md#custom-optimizer).                               |
| `optimizer_kwargs`      | `dict`            | `{}`         | Keyword arguments forwarded to the optimizer constructor; ignored (with a warning) for callables.                                         |
| `warmup_epochs`         | `float`           | `0.0`        | Epochs of linear LR warmup. For explicit schedulers this prepends a `SequentialLR` warmup ramp (skipped for `ReduceLROnPlateau`).         |
| `drop_path`             | `float`           | `0.0`        | Stochastic depth drop-path rate applied to the backbone. Higher values add more regularization.                                           |

### Runtime and Accelerator

| Parameter              | Type             | Default  | Description                                                                                                                                                              |
| ---------------------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `accelerator`          | `str`            | `"auto"` | PyTorch Lightning accelerator selection. `"auto"` picks GPU if available, then MPS, then CPU.                                                                            |
| `seed`                 | `int`            | `None`   | Global random seed for reproducibility. `None` means no fixed seed is set.                                                                                               |
| `fp16_eval`            | `bool`           | `False`  | Run evaluation passes in FP16 precision. Reduces memory usage but may lower numerical precision.                                                                         |
| `compute_val_loss`     | `bool \| "auto"` | `"auto"` | Compute and log validation loss only when a configured consumer monitors `val/loss`. Set `True` to always compute it or `False` to disable it.                           |
| `compute_test_loss`    | `bool`           | `True`   | Compute and log the detection loss during the final test run.                                                                                                            |
| `num_sanity_val_steps` | `int`            | `0`      | PyTorch Lightning sanity-check validation batches run before training starts. `0` disables it (the default); increase to catch val-path errors before a full epoch runs. |

### DataLoader Tuning

| Parameter            | Type   | Default | Description                                                                                               |
| -------------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------- |
| `pin_memory`         | `bool` | `None`  | Pin host memory in the DataLoader for faster GPU transfers. `None` defers to PyTorch Lightning's default. |
| `persistent_workers` | `bool` | `None`  | Keep DataLoader worker processes alive between epochs. `None` defers to PyTorch Lightning's default.      |
| `prefetch_factor`    | `int`  | `None`  | Number of batches to prefetch per DataLoader worker. `None` uses PyTorch's built-in default.              |

## Complete Parameter Reference

Below is a summary table of all training parameters:

| Parameter                    | Type                   | Default        | Description                                                                                                                                                                                                               |
| ---------------------------- | ---------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_dir`                | str                    | Required       | Path to COCO or YOLO formatted dataset with train/valid/test splits.                                                                                                                                                      |
| `output_dir`                 | str                    | "output"       | Directory for checkpoints, logs, and other training artifacts.                                                                                                                                                            |
| `epochs`                     | int                    | 100            | Number of full passes over the dataset.                                                                                                                                                                                   |
| `batch_size`                 | int or "auto"          | 4              | Samples per iteration. Set to `"auto"` to let RF-DETR probe the GPU for the largest safe batch size. Balance with `grad_accum_steps`.                                                                                     |
| `grad_accum_steps`           | int                    | 1              | Gradient accumulation steps. Opt-in; raise only when memory caps `batch_size`.                                                                                                                                            |
| `eval_batch_size`            | int or None            | None           | Batch size for validation, test and predict dataloaders. `None` inherits `batch_size`; `no_grad` avoids autograd activation storage, but in-fit evaluation still shares device memory with the model and optimizer state. |
| `lr`                         | float                  | 1e-4           | Learning rate for the model (excluding encoder).                                                                                                                                                                          |
| `lr_encoder`                 | float                  | 1.5e-4         | Learning rate for the backbone encoder.                                                                                                                                                                                   |
| `resolution`                 | int                    | Model-specific | Input image size (must be divisible by the selected model's `patch_size * num_windows`).                                                                                                                                  |
| `weight_decay`               | float                  | 1e-4           | L2 regularization coefficient.                                                                                                                                                                                            |
| `device`                     | str                    | "cuda"         | Training device: cuda, cpu, or mps.                                                                                                                                                                                       |
| `use_ema`                    | bool                   | True           | Enable Exponential Moving Average of weights.                                                                                                                                                                             |
| `gradient_checkpointing`     | bool                   | False          | Trade compute for memory during backprop.                                                                                                                                                                                 |
| `checkpoint_interval`        | int                    | 10             | Save checkpoint every N epochs.                                                                                                                                                                                           |
| `resume`                     | str                    | None           | Path to checkpoint for resuming training.                                                                                                                                                                                 |
| `tensorboard`                | bool                   | True           | Enable TensorBoard logging.                                                                                                                                                                                               |
| `wandb`                      | bool                   | False          | Enable Weights & Biases logging.                                                                                                                                                                                          |
| `project`                    | str                    | None           | W&B project name.                                                                                                                                                                                                         |
| `run`                        | str                    | None           | W&B run name.                                                                                                                                                                                                             |
| `early_stopping`             | bool                   | False          | Enable early stopping.                                                                                                                                                                                                    |
| `early_stopping_patience`    | int                    | 10             | Epochs without improvement before stopping.                                                                                                                                                                               |
| `early_stopping_min_delta`   | float                  | 0.001          | Minimum validation metric change to qualify as improvement.                                                                                                                                                               |
| `early_stopping_use_ema`     | bool                   | False          | Use EMA model for early stopping metrics.                                                                                                                                                                                 |
| `best_model_metric`          | `Literal["map","mar"]` | "map"          | Metric family for best-checkpoint selection and early stopping — mAP or mAR.                                                                                                                                              |
| `eval_max_dets`              | int                    | 500            | Maximum detections per image for detection/segmentation COCO AP and AR evaluation. Keypoint AP/AR uses fixed COCO `maxDets=20`.                                                                                           |
| `eval_interval`              | int                    | 1              | Skip the whole validation loop on epochs not a multiple of N; final epoch always validates.                                                                                                                               |
| `log_per_class_metrics`      | bool                   | False          | Log per-class AP metrics; enable to run the underlying per-class compute.                                                                                                                                                 |
| `eval_ema_only`              | bool                   | False          | Forward through the EMA model only during validation. Requires use_ema=True.                                                                                                                                              |
| `eval_masks_head_resolution` | bool                   | False          | Segmentation only. Compare masks at native (lower) resolution instead of upsampling; not comparable across runs.                                                                                                          |
| `progress_bar`               | str \| bool \| None    | None           | Progress bar style: `"tqdm"`, `"rich"`, or `None`. Legacy booleans are still accepted.                                                                                                                                    |
| `accelerator`                | str                    | "auto"         | PyTorch Lightning accelerator. "auto" selects GPU/MPS/CPU automatically.                                                                                                                                                  |
| `seed`                       | int                    | None           | Random seed for reproducibility. None means no fixed seed.                                                                                                                                                                |
| `lr_scheduler`               | str \| Callable        | "step"         | Scheduler preset ("step"/"cosine"), dotted import path, or callable.                                                                                                                                                      |
| `lr_scheduler_kwargs`        | dict                   | {}             | Keyword arguments for an explicit scheduler; also carries lr_drop / min_factor for the managed presets.                                                                                                                   |
| `lr_scheduler_interval`      | str                    | "step"         | Explicit-scheduler stepping cadence: "step" or "epoch".                                                                                                                                                                   |
| `lr_scheduler_monitor`       | str                    | "val/loss"     | Metric fed to ReduceLROnPlateau.                                                                                                                                                                                          |
| `lr_min_factor`              | float                  | 0.0            | Deprecated — use lr_scheduler_kwargs["min_factor"]. Cosine-preset floor as a fraction of the initial LR.                                                                                                                  |
| `lr_drop`                    | int                    | 100            | Deprecated — use lr_scheduler_kwargs["lr_drop"]. Epoch at which the "step" preset drops the LR by 10x.                                                                                                                    |
| `warmup_epochs`              | float                  | 0.0            | Number of linear warmup epochs at the start of training.                                                                                                                                                                  |
| `drop_path`                  | float                  | 0.0            | Stochastic depth drop-path rate for the backbone.                                                                                                                                                                         |
| `compute_val_loss`           | bool \| "auto"         | "auto"         | Compute validation loss only for a configured `val/loss` consumer; `True` forces it and `False` disables it.                                                                                                              |
| `compute_test_loss`          | bool                   | True           | Compute and log loss during the test run.                                                                                                                                                                                 |
| `num_sanity_val_steps`       | int                    | 0              | PTL sanity-check validation batches run before training starts. 0 disables it; increase to catch val-path errors early.                                                                                                   |
| `fp16_eval`                  | bool                   | False          | Run evaluation in FP16 precision to reduce memory usage.                                                                                                                                                                  |
| `pin_memory`                 | bool                   | None           | Pin DataLoader memory. None defers to PyTorch Lightning's default.                                                                                                                                                        |
| `persistent_workers`         | bool                   | None           | Keep DataLoader workers alive between epochs. None uses PTL default.                                                                                                                                                      |
| `prefetch_factor`            | int                    | None           | Number of batches prefetched per worker. None uses PyTorch default.                                                                                                                                                       |
