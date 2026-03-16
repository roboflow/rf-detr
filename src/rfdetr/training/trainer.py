# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Trainer factory — assembles a PTL Trainer from RF-DETR configs."""

import warnings
from typing import Any

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.training.callbacks import (
    BestModelCallback,
    DropPathCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.utilities.logger import get_logger

_logger = get_logger()


def build_trainer(
    train_config: TrainConfig,
    model_config: ModelConfig,
    *,
    accelerator: str | None = None,
    **trainer_kwargs: Any,
) -> Trainer:
    """Assemble a PTL ``Trainer`` with the full RF-DETR callback and logger stack.

    Resolves training precision from ``model_config.amp`` and device capability,
    guards EMA against sharded strategies, wires conditional loggers, and applies
    promoted training knobs (gradient clipping, sync_batchnorm, strategy).

    Args:
        train_config: Training hyperparameter configuration.
        model_config: Architecture configuration (used for precision and segmentation).
        accelerator: PTL accelerator string (e.g. ``"auto"``, ``"cpu"``, ``"gpu"``).
            Defaults to ``None`` which reads from ``train_config.accelerator``
            (itself defaulting to ``"auto"``).
            Pass ``"cpu"`` to override auto-detection (e.g. when the caller
            explicitly requests CPU training via ``device="cpu"``).
        **trainer_kwargs: Extra keyword arguments forwarded verbatim to
            ``pytorch_lightning.Trainer``.  Use this to pass PTL-native flags
            that are not exposed through ``TrainConfig``, for example::

                build_trainer(tc, mc, fast_dev_run=2)

            Any key present in both ``trainer_kwargs`` and the built config dict
            will be overridden by the value in ``trainer_kwargs``.

    Returns:
        A configured ``pytorch_lightning.Trainer`` instance.
    """
    tc = train_config
    if accelerator is None:
        accelerator = tc.accelerator

    # --- Precision resolution ---
    def _resolve_precision() -> str:
        if not model_config.amp:
            return "32-true"
        if torch.cuda.is_available():
            # Ampere+ GPUs support bf16-mixed which is scaler-free —
            # no GradScaler.scale/unscale/update overhead per optimizer step.
            # BF16 is safe for fine-tuning (pretrained weights loaded by default).
            # Training from random init with very small LR may underflow; callers
            # can override via trainer_kwargs(precision="16-mixed") if needed.
            if torch.cuda.is_bf16_supported():
                return "bf16-mixed"
            return "16-mixed"
        if torch.backends.mps.is_available():
            return "16-mixed"
        return "32-true"

    # --- Strategy + EMA sharding guard ---
    strategy = getattr(tc, "strategy", "auto")
    sharded = any(s in str(strategy).lower() for s in ("fsdp", "deepspeed"))
    enable_ema = bool(tc.use_ema) and not sharded
    if tc.use_ema and sharded:
        warnings.warn(
            f"EMA disabled: RFDETREMACallback is not compatible with sharded strategies "
            f"(strategy={strategy!r}). Set use_ema=False to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    # --- Build callbacks ---
    callbacks = []

    if enable_ema:
        callbacks.append(
            RFDETREMACallback(
                decay=tc.ema_decay,
                tau=tc.ema_tau,
                update_interval_steps=tc.ema_update_interval,
            )
        )

    # Drop-path / dropout scheduling (vit_encoder_num_layers defaults to 12).
    if tc.drop_path > 0.0:
        callbacks.append(DropPathCallback(drop_path=tc.drop_path))

    # COCO mAP + F1 evaluation.
    callbacks.append(
        COCOEvalCallback(
            max_dets=tc.eval_max_dets,
            segmentation=model_config.segmentation_head,
            eval_interval=tc.eval_interval,
            log_per_class_metrics=tc.log_per_class_metrics,
        )
    )

    # Best-model checkpointing — monitor EMA metric only when EMA is active.
    callbacks.append(
        BestModelCallback(
            output_dir=tc.output_dir,
            monitor_ema="val/ema_mAP_50_95" if enable_ema else None,
            run_test=tc.run_test,
        )
    )

    # Optional early stopping.
    if tc.early_stopping:
        callbacks.append(
            RFDETREarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                use_ema=tc.early_stopping_use_ema,
            )
        )

    # --- Build loggers ---
    # Each logger is guarded by a try/except because tensorboard, wandb, and mlflow
    # are optional dependencies (installed via the [metrics] extra).  A missing dep
    # emits a UserWarning instead of crashing.
    # CSVLogger is always enabled — no extra package required.
    # Produces metrics.csv in output_dir so there is always a log file.
    loggers: list = [CSVLogger(save_dir=tc.output_dir, name="", version="")]

    if tc.tensorboard:
        try:
            loggers.append(
                TensorBoardLogger(
                    save_dir=tc.output_dir,
                    name="",
                    version="",
                )
            )
        except ModuleNotFoundError as exc:
            _logger.warning("TensorBoard logging disabled: %s. Install with: pip install tensorboard", exc)

    if tc.wandb:
        try:
            loggers.append(
                WandbLogger(
                    name=tc.run,
                    project=tc.project,
                    save_dir=tc.output_dir,
                )
            )
        except ModuleNotFoundError as exc:
            _logger.warning("WandB logging disabled: %s. Install with: pip install wandb", exc)

    if tc.mlflow:
        try:
            loggers.append(
                MLFlowLogger(
                    experiment_name=tc.project or "rfdetr",
                    run_name=tc.run,
                    save_dir=tc.output_dir,
                )
            )
        except ModuleNotFoundError as exc:
            _logger.warning("MLflow logging disabled: %s. Install with: pip install mlflow", exc)

    if tc.clearml:
        warnings.warn(
            "ClearML logging is not supported via a native PTL logger in this version."
            " Metrics will not be logged to ClearML. Use the ClearML SDK callback directly"
            " or wait for a dedicated ClearML PTL logger integration.",
            UserWarning,
            stacklevel=2,
        )

    # --- Promoted config fields (T4-2 added these to TrainConfig) ---
    clip_max_norm: float = tc.clip_max_norm
    sync_bn: bool = tc.sync_bn

    trainer_config: dict[str, Any] = {
        "max_epochs": tc.epochs,
        "accelerator": accelerator,
        "devices": 1,
        "strategy": strategy,
        "precision": _resolve_precision(),
        "accumulate_grad_batches": tc.grad_accum_steps,
        "gradient_clip_val": clip_max_norm,
        "sync_batchnorm": sync_bn,
        "callbacks": callbacks,
        "logger": loggers if loggers else False,
        "enable_progress_bar": tc.progress_bar,
        "default_root_dir": tc.output_dir,
        "log_every_n_steps": 50,
        "deterministic": False,
    }
    trainer_config.update(trainer_kwargs)
    return Trainer(**trainer_config)
