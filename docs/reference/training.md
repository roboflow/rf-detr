---
description: RF-DETR Lightning training API reference for RFDETRModelModule, RFDETRDataModule, build_trainer, callbacks, and training primitives.
---

# Training API Reference

This page documents the training primitives that power RF-DETR. For a narrative guide with runnable examples, see [Custom Training API](../learn/train/customization.md).

## RFDETRModelModule

::: rfdetr.training.module_model.RFDETRModelModule
    options:
      show_source: false
      members:
        - __init__
        - on_fit_start
        - on_train_batch_start
        - transfer_batch_to_device
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers
        - clip_gradients
        - on_load_checkpoint
        - reinitialize_detection_head

---

## RFDETRDataModule

::: rfdetr.training.module_data.RFDETRDataModule
    options:
      show_source: false
      members:
        - __init__
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - class_names

---

## build_trainer

::: rfdetr.training.trainer.build_trainer
    options:
      show_source: false

---

## Callbacks

### RFDETREMACallback

::: rfdetr.training.callbacks.ema.RFDETREMACallback
    options:
      show_source: false
      members:
        - __init__

### BestModelCallback

::: rfdetr.training.callbacks.best_model.BestModelCallback
    options:
      show_source: false
      members:
        - __init__

### RFDETREarlyStopping

::: rfdetr.training.callbacks.best_model.RFDETREarlyStopping
    options:
      show_source: false
      members:
        - __init__

### DropPathCallback

::: rfdetr.training.callbacks.drop_schedule.DropPathCallback
    options:
      show_source: false
      members:
        - __init__

### COCOEvalCallback

::: rfdetr.training.callbacks.coco_eval.COCOEvalCallback
    options:
      show_source: false
      members:
        - __init__

### GPUMemoryTQDMProgressBar

::: rfdetr.training.callbacks.gpu_memory_progress_bar.GPUMemoryTQDMProgressBar
    options:
      show_source: false
      members:
        - get_metrics
      inherited_members:
        - get_metrics

### GPUMemoryRichProgressBar

::: rfdetr.training.callbacks.gpu_memory_progress_bar.GPUMemoryRichProgressBar
    options:
      show_source: false
      members:
        - get_metrics
      inherited_members:
        - get_metrics

---

## RFDETRCli

!!! info "CLI requires the `train` and `cli` extras"

    ```bash
    pip install "rfdetr[train,cli]"
    ```

    The `rfdetr` console script and its `--config` / `--print_config` flags are provided by `jsonargparse`, which is included in the `cli` extra.

`RFDETRCli` is the command-line entry point for RF-DETR. It wraps
`RFDETRModelModule` and `RFDETRDataModule` under a single `rfdetr` command and
auto-generates four subcommands from the PyTorch Lightning CLI machinery:

```bash
rfdetr fit      --config configs/rfdetr_base.yaml
rfdetr validate --config configs/rfdetr_base.yaml --ckpt_path output/rfdetr_base/last.ckpt
rfdetr test     --config configs/rfdetr_base.yaml --ckpt_path output/rfdetr_base/last.ckpt
rfdetr predict  --config configs/rfdetr_base.yaml --ckpt_path output/rfdetr_base/last.ckpt
```

Both `model_config` and `train_config` are specified once; `RFDETRCli`
automatically links them to the datamodule so you do not need to repeat the
same arguments under `--data.*`.

`rfdetr fit` builds its trainer with `build_trainer`, as `RFDETR.train()` does.
`train_config` sets the epochs, precision, devices, gradient accumulation and
clipping, the EMA, COCO metric, checkpoint and best-model callbacks, and the
loggers, and everything is written to its `output_dir`. A `--trainer.*`
argument overrides the value built from `train_config`. So does a value in a
config file's `trainer:` section or in a `--trainer` file, unless it equals
Lightning's default: the `config.yaml` that `fit` saves lists every trainer
argument, and running from it again keeps the values from `train_config`.
Arguments apply in order, so put `--trainer.*` after `--config`, as the
configs' usage lines do. `--trainer.callbacks` adds callbacks to the RF-DETR
ones. Change the number of epochs with `--model.train_config.init_args.epochs`:
`--trainer.max_epochs` or `--trainer.max_steps` ends the run but leaves the
learning-rate and drop-path schedules sized for `epochs`, and warns.

`validate`, `test` and `predict` build the evaluation-only trainer that
`RFDETR.evaluate()` uses. `validate` and `test` score the weights saved in
`--ckpt_path`. `last.ckpt`, which `fit` writes unless `checkpoint_interval` is
1, holds the trained weights without the EMA average that validation during
`fit` reports; `checkpoint_best_total.pth` holds the best weights, for
`rfdetr.from_checkpoint`. `predict` passes its results to the trainer's
callbacks only, so it prints and saves nothing unless you add a
`BasePredictionWriter` with `--trainer.callbacks`; in Python,
`rfdetr.from_checkpoint(path).predict(...)` returns detections.

A `fit` with `--ckpt_path`, or with `resume` in `train_config` as
`RFDETR.train()` takes it, resumes the run and appends to the `metrics.csv`
already in `output_dir`. Resuming from `last.ckpt` restores the optimizer and
learning-rate scheduler too; the best-model `.pth` files hold the weights and
epoch only, so those two restart cold.

::: rfdetr.training.cli.RFDETRCli
    options:
      show_source: false
      members:
        - __init__
        - instantiate_trainer
