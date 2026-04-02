# Custom Training API

The high-level `RFDETR.train()` method is the quickest path to fine-tuning, but the underlying training primitives are fully public and are the **recommended path for any customisation**: custom callbacks, alternative loggers, mixed-precision overrides, multi-GPU strategies, or integration with external training frameworks.

!!! tip "Quickstart vs. customisation"

    If you want to start training with minimal code, use `model.train()` — it sets up and runs the full PTL stack automatically. Come here when you need to take direct control over any part of that stack.

## How `RFDETR.train()` relates to PTL

When you call `model.train(...)`, three things happen internally:

```python
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

module = RFDETRModelModule(model_config, train_config)
datamodule = RFDETRDataModule(model_config, train_config)
trainer = build_trainer(train_config, model_config)
trainer.fit(module, datamodule, ckpt_path=train_config.resume or None)
```

Each of these objects is a standard PTL class. You can construct them directly, modify them, swap out callbacks, or replace the trainer entirely.

---

## RFDETRModelModule

`RFDETRModelModule` is a `pytorch_lightning.LightningModule`. It owns the model weights, the criterion, the postprocessor, and the optimizer/scheduler configuration.

```python
from rfdetr.config import (
    RFDETRMediumConfig,
    TrainConfig,
)  # config classes live in rfdetr.config, not the top-level rfdetr namespace
from rfdetr.training import RFDETRModelModule

model_config = RFDETRMediumConfig(num_classes=10)
train_config = TrainConfig(
    dataset_dir="path/to/dataset",
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
)

module = RFDETRModelModule(model_config, train_config)
```

### Lifecycle hooks

| Hook                       | Behaviour                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| `on_fit_start`             | Seeds RNGs when `train_config.seed` is set.                                                     |
| `on_train_batch_start`     | Applies multi-scale random resize when `train_config.multi_scale=True`.                         |
| `transfer_batch_to_device` | Moves `NestedTensor` batches to the target device.                                              |
| `training_step`            | Computes loss, divides by `accumulate_grad_batches`, and logs `train/loss` and per-term losses. |
| `validation_step`          | Runs forward pass and postprocessing; returns `{results, targets}` for `COCOEvalCallback`.      |
| `test_step`                | Same as `validation_step`, logs under `test/`.                                                  |
| `predict_step`             | Runs inference-only forward pass and returns postprocessed detections.                          |
| `configure_optimizers`     | Builds AdamW with layer-wise LR decay and a LambdaLR scheduler (cosine or step).                |
| `on_load_checkpoint`       | Auto-converts legacy `.pth` checkpoints to PTL format.                                          |

### Accessing the underlying model

The raw `nn.Module` is `module.model`. After training completes, `RFDETR.train()` syncs it back onto `self.model.model` so `predict()` and `export()` continue to work.

---

## RFDETRDataModule

`RFDETRDataModule` is a `pytorch_lightning.LightningDataModule`. It builds train/val/test datasets and wraps them in `DataLoader` objects.

```python
from rfdetr.training import RFDETRDataModule

datamodule = RFDETRDataModule(model_config, train_config)
```

### Stages

| Stage        | Datasets built                             |
| ------------ | ------------------------------------------ |
| `"fit"`      | `train` + `val`                            |
| `"validate"` | `val` only                                 |
| `"test"`     | `test` (or `val` for COCO-format datasets) |

The `setup(stage)` method is lazy — each split is built at most once, even if called multiple times.

### class_names property

```python
datamodule.setup("fit")
print(datamodule.class_names)  # e.g. ["cat", "dog", "person"]
```

Returns sorted category names from the COCO annotation file of the first available split, or `None` if the dataset has not been set up yet.

---

## build_trainer

`build_trainer` assembles a `pytorch_lightning.Trainer` with the full RF-DETR callback and logger stack. All `TrainConfig` fields are wired automatically.

```python
from rfdetr.training import build_trainer

trainer = build_trainer(train_config, model_config)
```

### What build_trainer configures

| Concern               | Source                                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Max epochs            | `train_config.epochs`                                                                                                                                      |
| Gradient accumulation | `train_config.grad_accum_steps`                                                                                                                            |
| Gradient clipping     | `train_config.clip_max_norm` (default `0.1`)                                                                                                               |
| Mixed precision       | Resolved from `model_config.amp` and device capability (`bf16-mixed` on Ampere+, `16-mixed` otherwise)                                                     |
| Accelerator           | `train_config.accelerator` (default `"auto"`)                                                                                                              |
| Strategy              | Pass `strategy=` as a `**trainer_kwarg` to `build_trainer`. `TrainConfig` has no `strategy` field — setting it on `TrainConfig` will raise a `ValueError`. |
| Sync batch norm       | `train_config.sync_bn`                                                                                                                                     |
| Progress bar          | `train_config.progress_bar`                                                                                                                                |
| Loggers               | CSVLogger always; TensorBoard, WandB, MLflow when their `train_config` flags are `True`                                                                    |
| Callbacks             | `RFDETREMACallback`, `DropPathCallback`, `COCOEvalCallback`, `BestModelCallback`, `RFDETREarlyStopping` (conditional)                                      |

### Overriding PTL Trainer kwargs

Pass any keyword argument accepted by `pytorch_lightning.Trainer` via `**trainer_kwargs`. These override the built configuration:

```python
trainer = build_trainer(
    train_config,
    model_config,
    fast_dev_run=2,  # run 2 batches per epoch for a smoke test
    accumulate_grad_batches=8,  # override TrainConfig.grad_accum_steps
    log_every_n_steps=10,
)
```

---

## Running the training loop

### Full training run

```python
from rfdetr.config import (
    RFDETRMediumConfig,
    TrainConfig,
)  # config classes live in rfdetr.config, not the top-level rfdetr namespace
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

model_config = RFDETRMediumConfig(num_classes=10)
train_config = TrainConfig(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output",
)

module = RFDETRModelModule(model_config, train_config)
datamodule = RFDETRDataModule(model_config, train_config)
trainer = build_trainer(train_config, model_config)

trainer.fit(module, datamodule)
```

### Resume from checkpoint

Pass the checkpoint path to `trainer.fit` via `ckpt_path`. The path can be a PTL `.ckpt` file or a legacy RF-DETR `.pth` file — `RFDETRModelModule.on_load_checkpoint` converts either format automatically.

```python
trainer.fit(module, datamodule, ckpt_path="output/last.ckpt")
# or a legacy checkpoint:
trainer.fit(module, datamodule, ckpt_path="output/checkpoint.pth")
```

> **Note:** When `checkpoint_interval=1`, no `last.ckpt` is written. Use `checkpoint_{epoch}.ckpt` (e.g. `output/checkpoint_epoch=4.ckpt`) to resume instead.

If you need to persist a converted checkpoint on disk (for example to inspect it, share it, or use it outside of PTL), convert it explicitly before passing it to `trainer.fit`:

```python
from rfdetr.training import convert_legacy_checkpoint

convert_legacy_checkpoint("old_checkpoint.pth", "new_checkpoint.ckpt")
trainer.fit(module, datamodule, ckpt_path="new_checkpoint.ckpt")
```

`convert_legacy_checkpoint` reads a pre-PTL `.pth` file produced by the legacy `engine.py` training loop and writes a PTL-compatible `.ckpt` file. Use it when migrating saved checkpoints to the PTL format rather than relying on on-the-fly conversion at load time.

### Validation only

```python
trainer.validate(module, datamodule)
```

Runs one full validation pass and logs `val/mAP_50_95`, `val/mAP_50`, `val/F1`, and per-class AP metrics to all active loggers.

### Inference with the data pipeline

```python
predictions = trainer.predict(module, dataloaders=datamodule.val_dataloader())
```

Calls `module.predict_step` on every batch and returns a list of postprocessed detection results. Pass any `DataLoader` instance — `datamodule.val_dataloader()`, `datamodule.test_dataloader()`, or a custom loader — as the `dataloaders` argument. This is useful for offline evaluation or generating submission files.

!!! note "predict_dataloader not implemented"

    `RFDETRDataModule` does not define a `predict_dataloader()` method, so `trainer.predict(module, datamodule)` will raise an error. Always pass a dataloader explicitly via the `dataloaders=` argument.

---

## Multi-GPU training

`build_trainer` configures PyTorch Lightning's `Trainer` directly, so all PTL strategies work out of the box.

### Data Parallel (DDP) — recommended

Set `train_config.accelerator = "auto"` and pass `strategy="ddp"` to `build_trainer`, then launch with `torchrun`:

!!! note "`devices` must be overridden for multi-GPU runs"

    `build_trainer` defaults to `devices=1`. To use all available GPUs, pass `devices="auto"` (or an explicit count) as a `**trainer_kwarg`:

    ```python
    trainer = build_trainer(train_config, model_config, strategy="ddp", devices="auto")
    ```

    Without this override, `torchrun` will spawn multiple processes but each process will only see one device, defeating the purpose of the multi-GPU launch.

```bash
torchrun --nproc_per_node=4 train.py
```

where `train.py` contains:

```python
from rfdetr.config import (
    RFDETRMediumConfig,
    TrainConfig,
)  # config classes live in rfdetr.config, not the top-level rfdetr namespace
from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

model_config = RFDETRMediumConfig(num_classes=10)
train_config = TrainConfig(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,  # per-GPU batch size
    grad_accum_steps=1,  # reduce when using more GPUs
    output_dir="output",
    sync_bn=True,  # sync batch norms across GPUs
)

module = RFDETRModelModule(model_config, train_config)
datamodule = RFDETRDataModule(model_config, train_config)
trainer = build_trainer(train_config, model_config, strategy="ddp", devices="auto")

trainer.fit(module, datamodule)
```

!!! warning "EMA is not compatible with FSDP or DeepSpeed"

    `build_trainer` automatically disables `RFDETREMACallback` when `strategy` contains `"fsdp"` or `"deepspeed"`, and emits a `UserWarning`. Use `strategy="ddp"` or `strategy="auto"` to keep EMA active.

### Effective batch size

```
effective_batch_size = batch_size × grad_accum_steps × num_gpus
```

Maintain an effective batch size of 16 regardless of GPU count:

| GPUs | `batch_size` | `grad_accum_steps` | Effective |
| ---- | ------------ | ------------------ | --------- |
| 1    | 4            | 4                  | 16        |
| 2    | 4            | 2                  | 16        |
| 4    | 4            | 1                  | 16        |
| 8    | 2            | 1                  | 16        |

---

## Custom callbacks

`build_trainer` builds the default callback stack. To add your own callbacks alongside the built-in ones, pass them via `trainer_kwargs`:

```python
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from rfdetr.training import build_trainer

extra_callbacks = [
    LearningRateMonitor(logging_interval="step"),
    ModelSummary(max_depth=3),
]

trainer = build_trainer(
    train_config,
    model_config,
    callbacks=extra_callbacks,  # replaces the default callback list entirely
)
```

!!! warning "Replacing vs. extending callbacks"

    Passing `callbacks=` to `build_trainer` via `trainer_kwargs` **replaces** the entire default callback list built inside `build_trainer` (EMA, COCO eval, best-model checkpointing, etc.). To extend rather than replace, build the extra callbacks separately and merge them after calling `build_trainer`:

    ```python
    trainer = build_trainer(train_config, model_config)
    trainer.callbacks.extend(
        [
            LearningRateMonitor(logging_interval="step"),
        ]
    )
    trainer.fit(module, datamodule)
    ```

### Built-in callbacks

| Class                 | Purpose                                                                                     | Enabled when                                            |
| --------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `RFDETREMACallback`   | Maintains an EMA copy of model weights                                                      | `train_config.use_ema=True` and strategy is not sharded |
| `DropPathCallback`    | Anneals drop-path rate over training                                                        | `train_config.drop_path > 0`                            |
| `COCOEvalCallback`    | Computes mAP and F1 after each validation epoch                                             | Always                                                  |
| `BestModelCallback`   | Saves `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, `checkpoint_best_total.pth` | Always                                                  |
| `RFDETREarlyStopping` | Stops training when validation mAP stops improving                                          | `train_config.early_stopping=True`                      |

---

## Custom loggers

`build_trainer` adds loggers based on `TrainConfig` flags. To attach a logger not supported by `TrainConfig` (for example a custom Neptune or Comet logger), build it yourself and pass it alongside the defaults:

```python
from pytorch_lightning.loggers import NeptuneLogger  # hypothetical
from rfdetr.training import build_trainer

trainer = build_trainer(train_config, model_config)
trainer.loggers.append(NeptuneLogger(project="my-workspace/rf-detr"))
trainer.fit(module, datamodule)
```

All logged keys (`train/loss`, `val/mAP_50_95`, `val/F1`, `val/ema_mAP_50_95`, etc.) are written to every active logger in the list.

---

## Logged metrics reference

| Key                  | When logged            | Description                                               |
| -------------------- | ---------------------- | --------------------------------------------------------- |
| `train/loss`         | Every step / epoch     | Total weighted training loss                              |
| `train/<term>`       | Every step / epoch     | Individual loss terms (e.g. `train/loss_bbox`)            |
| `val/loss`           | Each epoch             | Validation loss (if `train_config.compute_val_loss=True`) |
| `val/mAP_50_95`      | Each eval epoch        | COCO box mAP@[.50:.05:.95]                                |
| `val/mAP_50`         | Each eval epoch        | COCO box mAP@.50                                          |
| `val/mAP_75`         | Each eval epoch        | COCO box mAP@.75                                          |
| `val/mAR`            | Each eval epoch        | COCO mean average recall                                  |
| `val/ema_mAP_50_95`  | Each eval epoch        | EMA-model mAP@[.50:.05:.95] (if EMA active)               |
| `val/F1`             | Each eval epoch        | Macro F1 at best confidence threshold                     |
| `val/precision`      | Each eval epoch        | Precision at best F1 threshold                            |
| `val/recall`         | Each eval epoch        | Recall at best F1 threshold                               |
| `val/AP/<class>`     | Each eval epoch        | Per-class AP (if `log_per_class_metrics=True`)            |
| `val/segm_mAP_50_95` | Each eval epoch        | Segmentation mAP (segmentation models only)               |
| `val/segm_mAP_50`    | Each eval epoch        | Segmentation mAP@.50 (segmentation models only)           |
| `test/*`             | After `trainer.test()` | Mirror of `val/*` keys                                    |

---

## See also

- [RFDETR.train() — high-level API](../index.md#quick-start) — the one-liner training path
- [Training parameters](../training-parameters.md) — all `TrainConfig` fields
- [Training loggers](../loggers.md) — TensorBoard, WandB, MLflow setup
- [Advanced training](../advanced.md) — checkpointing, early stopping, memory optimisation
- [PTL primitives API reference](../../reference/training.md) — full docstring reference
