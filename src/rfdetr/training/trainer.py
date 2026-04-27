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
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy as _DDPStrategy

# _MultiProcessingLauncher is a private PTL API (leading underscore) that may change
# in minor PTL releases within the >=2.6,<3 range.  No public equivalent exists in
# PTL 2.x.  Monitor PTL changelogs when bumping the lower bound.
try:
    from pytorch_lightning.strategies.launchers.multiprocessing import _MultiProcessingLauncher
except ImportError:  # pragma: no cover - exercised in unit tests via monkeypatch
    _MultiProcessingLauncher = None  # type: ignore[assignment]

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


# ---------------------------------------------------------------------------
# Notebook-safe spawn-based DDP
# ---------------------------------------------------------------------------
# ``ddp_notebook`` maps to fork-based DDP which is fundamentally unsafe:
# PyTorch's OpenMP thread pool (created during model construction) cannot
# survive fork() — the worker threads become zombie handles, causing
# "Invalid thread pool!" SIGABRT when the autograd engine initialises in
# the forked child.
#
# PTL considers ``start_method="spawn"`` incompatible with interactive
# environments and raises ``MisconfigurationException`` if used in Jupyter.
# However, PTL's own ``_wrapping_function`` is the entry-point for spawned
# children — no ``if __name__ == "__main__"`` guard is required — so spawn
# is perfectly safe here.
#
# Classes MUST live at module level (not inside a function) so that Python's
# pickle can serialise them for the spawned child processes.


if _MultiProcessingLauncher is not None:

    class _InteractiveSpawnLauncher(_MultiProcessingLauncher):
        """Spawn launcher that reports itself as interactive-compatible."""

        @property
        def is_interactive_compatible(self) -> bool:  # type: ignore[override]
            return True

else:
    _InteractiveSpawnLauncher = None


class _NotebookSpawnDDPStrategy(_DDPStrategy):
    """Spawn-based DDP strategy that works inside Jupyter / Kaggle notebooks."""

    def _configure_launcher(self) -> None:
        if self.cluster_environment is None:
            raise RuntimeError(
                "_NotebookSpawnDDPStrategy requires a cluster environment; "
                "ensure the strategy is initialised through PTL's Trainer."
            )
        if _InteractiveSpawnLauncher is None:
            raise RuntimeError(
                "Notebook spawn strategy requires "
                "pytorch_lightning.strategies.launchers.multiprocessing._MultiProcessingLauncher. "
                "Your installed PyTorch Lightning version changed this private API; "
                "pin/upgrade PTL to a compatible version in the supported >=2.6,<3 range."
            )
        self._launcher = _InteractiveSpawnLauncher(self, start_method=self._start_method)


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
        # Ampere+ GPUs support bf16-mixed which is scaler-free —
        # no GradScaler.scale/unscale/update overhead per optimizer step.
        # BF16 is safe for fine-tuning (pretrained weights loaded by default).
        # Training from random init with very small LR may underflow; callers
        # can override via trainer_kwargs(precision="16-mixed") if needed.
        #
        # Note: torch.cuda.is_available() and torch.cuda.is_bf16_supported() both
        # create a CUDA driver context in the parent process.  This is intentional
        # and safe for the multi-process launch modes we rely on here because we
        # avoid fork-based launching in notebook contexts (see
        # _NotebookSpawnDDPStrategy above), and spawn/subprocess-based launchers
        # start child processes with a fresh CUDA state regardless of what the
        # parent has initialised. If a fork-based path is ever added, this
        # precision check must be moved into the child process.
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "bf16-mixed"
            return "16-mixed"
        if torch.backends.mps.is_available():
            return "16-mixed"
        return "32-true"

    # --- Strategy + EMA sharding guard ---
    strategy = tc.strategy

    # Transparently replace fork-based DDP with spawn-based DDP — see the
    # module-level comment block above _InteractiveSpawnLauncher for rationale.
    if strategy in ("ddp_notebook", "ddp_spawn"):
        strategy = _NotebookSpawnDDPStrategy(start_method="spawn", find_unused_parameters=True)
        _logger.info(
            "%s → spawn-based DDP to avoid OpenMP thread pool corruption after fork.",
            tc.strategy,
        )
    elif strategy == "ddp" and model_config.segmentation_head:
        # The segmentation head's sparse_forward() returns dict intermediates and
        # leaves some parameters unused on certain forward steps, causing DDP to
        # raise "It looks like your LightningModule has parameters that were not
        # used in producing the loss" with plain ddp.  Enabling
        # find_unused_parameters lets DDP traverse the autograd graph after each
        # backward pass to detect which parameters contributed to the loss.
        strategy = _DDPStrategy(find_unused_parameters=True)
        _logger.info(
            "segmentation_head=True with strategy='ddp' → DDPStrategy(find_unused_parameters=True).",
        )
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

    if tc.progress_bar == "rich":
        callbacks.append(RichProgressBar(theme=RichProgressBarTheme(metrics_format=".3e")))
    elif tc.progress_bar == "tqdm":
        callbacks.append(TQDMProgressBar())

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

    # Latest resume checkpoint — overwritten every epoch.
    # Skip when checkpoint_interval == 1 to avoid duplicate ModelCheckpoint state_key.
    if tc.checkpoint_interval != 1:
        callbacks.append(
            ModelCheckpoint(
                dirpath=tc.output_dir,
                filename="last",
                every_n_epochs=1,
                save_top_k=1,
                enable_version_counter=False,
                auto_insert_metric_name=False,
                verbose=False,
            )
        )

    # Interval archive checkpoints — kept for the full run.
    callbacks.append(
        ModelCheckpoint(
            dirpath=tc.output_dir,
            filename="checkpoint_{epoch}",
            every_n_epochs=tc.checkpoint_interval,
            save_top_k=-1,
            enable_version_counter=False,
            auto_insert_metric_name=False,
            verbose=False,
        )
    )

    # Best-model checkpointing — monitor EMA metric only when EMA is active.
    callbacks.append(
        BestModelCallback(
            output_dir=tc.output_dir,
            monitor_ema="val/ema_mAP_50_95" if enable_ema else None,
            run_test=tc.run_test,
            skip_best_epochs=tc.skip_best_epochs,
        )
    )

    # Optional early stopping.
    if tc.early_stopping:
        callbacks.append(
            RFDETREarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                use_ema=tc.early_stopping_use_ema,
                skip_best_epochs=tc.skip_best_epochs,
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
        raise NotImplementedError("ClearML logging is not yet supported. Remove clearml=True from TrainConfig.")

    # --- Promoted config fields (T4-2 added these to TrainConfig) ---
    clip_max_norm: float = tc.clip_max_norm
    sync_bn: bool = tc.sync_bn

    trainer_config: dict[str, Any] = {
        "max_epochs": tc.epochs,
        "accelerator": accelerator,
        "devices": tc.devices,
        "num_nodes": tc.num_nodes,
        "strategy": strategy,
        "precision": _resolve_precision(),
        "accumulate_grad_batches": tc.grad_accum_steps,
        "gradient_clip_val": clip_max_norm,
        "sync_batchnorm": sync_bn,
        "callbacks": callbacks,
        "logger": loggers if loggers else False,
        "enable_progress_bar": tc.progress_bar is not None,
        "default_root_dir": tc.output_dir,
        "log_every_n_steps": 50,
        "deterministic": False,
    }
    trainer_config.update(trainer_kwargs)
    return Trainer(**trainer_config)
