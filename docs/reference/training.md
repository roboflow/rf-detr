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

---

## RFDETRCli

`RFDETRCli` is the command-line entry point for RF-DETR. It wraps
`RFDETRModelModule` and `RFDETRDataModule` under a single `rfdetr` command and
auto-generates four subcommands from the PyTorch Lightning CLI machinery:

```bash
rfdetr fit      --config configs/rfdetr_base.yaml
rfdetr validate --ckpt_path output/best.ckpt
rfdetr test     --ckpt_path output/best.ckpt
rfdetr predict  --ckpt_path output/best.ckpt
```

Both `model_config` and `train_config` are specified once; `RFDETRCli`
automatically links them to the datamodule so you do not need to repeat the
same arguments under `--data.*`.

::: rfdetr.training.cli.RFDETRCli
    options:
      show_source: false
      members:
        - __init__
