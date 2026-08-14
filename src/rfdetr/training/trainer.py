# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Trainer factory — assembles a PTL Trainer from RF-DETR configs."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy as _DDPStrategy

# _MultiProcessingLauncher is a private PTL API (leading underscore) that may change
# in minor PTL releases within the >=2.6,<3 range.  No public equivalent exists in
# PTL 2.x.  Monitor PTL changelogs when bumping the lower bound.
try:
    from pytorch_lightning.strategies.launchers.multiprocessing import _MultiProcessingLauncher
except ImportError:  # pragma: no cover - exercised in unit tests via monkeypatch
    _MultiProcessingLauncher = None  # type: ignore[assignment,misc]

from rfdetr.config import KeypointTrainConfig, ModelConfig, TrainConfig
from rfdetr.training.callbacks import (
    BestModelCallback,
    DropPathCallback,
    GPUMemoryRichProgressBar,
    GPUMemoryTQDMProgressBar,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.utilities.logger import get_logger

_logger = get_logger()


def _try_import_tensorboard_summary_writer() -> None:
    """Probe the full tensorboard import chain to surface numpy/tensorflow incompatibilities early.

    When tensorboard is installed alongside a numpy-2.0-incompatible tensorflow, importing
    ``torch.utils.tensorboard`` raises ``AttributeError`` at module level (e.g. ``np.float_`` was
    removed in NumPy 2.0).  Calling this function inside the logger-construction try/except lets
    ``build_trainer`` degrade gracefully to CSV-only logging instead of crashing mid-training.

    Raises:
        ImportError: If the ``tensorboard`` package is absent.
        AttributeError: If ``torch.utils.tensorboard`` fails to import due to a NumPy 2.0 /
            tensorflow incompatibility.
    """
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401


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


_InteractiveSpawnLauncher: type[Any] | None = None

if _MultiProcessingLauncher is not None:

    class _InteractiveSpawnLauncherImpl(_MultiProcessingLauncher):
        """Spawn launcher that reports itself as interactive-compatible."""

        @property
        def is_interactive_compatible(self) -> bool:
            return True

    _InteractiveSpawnLauncher = _InteractiveSpawnLauncherImpl


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
        if self._start_method == "popen":
            raise RuntimeError(
                "_NotebookSpawnDDPStrategy does not support start_method='popen'; "
                "it is always constructed with start_method='spawn' in build_trainer()."
            )
        self._launcher = _InteractiveSpawnLauncher(self, start_method=self._start_method)


def _normalize_xla_precision(precision: str) -> Literal["32-true", "16-true", "bf16-true"]:
    """Normalize resolved precision strings to a valid XLAPrecision literal.

    Args:
        precision: Precision string produced by the local resolver, e.g. ``"16-mixed"``.

    Returns:
        One of ``"32-true"``, ``"16-true"``, or ``"bf16-true"`` suitable for XLA plugin creation.

    Raises:
        ValueError: If the resolved precision is not supported by ``XLAPrecision``.
    """
    if precision == "32-true":
        return "32-true"
    if precision == "16-true":
        return "16-true"
    if precision == "bf16-true":
        return "bf16-true"
    raise ValueError(
        f"Unexpected precision value for XLAPrecision: {precision!r}; expected '32-true', '16-true', or 'bf16-true'."
    )


def _is_distributed_strategy_requested(strategy: str) -> bool:
    """Return whether a TrainConfig strategy string requests distributed execution."""
    strategy_name = strategy.lower()
    return any(token in strategy_name for token in ("ddp", "fsdp", "deepspeed"))


def _is_sharded_strategy(strategy: object) -> bool:
    """Return whether *strategy* is a sharded distributed strategy.

    Detects FSDP, FSDP2 (``ModelParallelStrategy``) and DeepSpeed given either a config
    string (e.g. ``"fsdp"``, ``"deepspeed"``) or an instantiated ``Strategy`` object. Object
    detection uses ``isinstance`` because ``str(ModelParallelStrategy())`` contains neither the
    ``"fsdp"`` nor ``"deepspeed"`` token, so a substring test alone would misclassify FSDP2 as
    non-sharded and let it reach the unvalidated manual-optimization path.

    Args:
        strategy: A strategy string or an instantiated PyTorch Lightning ``Strategy`` object.

    Returns:
        ``True`` if *strategy* requests a sharded strategy, ``False`` otherwise.

    Examples:
        >>> _is_sharded_strategy("fsdp")
        True
        >>> _is_sharded_strategy("ddp")
        False
    """
    if isinstance(strategy, str):
        return any(token in strategy.lower() for token in ("fsdp", "deepspeed"))
    try:
        from pytorch_lightning.strategies import (
            DeepSpeedStrategy,
            FSDPStrategy,
            ModelParallelStrategy,
        )
    except ImportError:
        sharded_types: tuple[type, ...] = ()
    else:
        sharded_types = (FSDPStrategy, DeepSpeedStrategy, ModelParallelStrategy)
    if sharded_types and isinstance(strategy, sharded_types):
        return True
    return any(token in str(strategy).lower() for token in ("fsdp", "deepspeed", "modelparallel"))


def _accelerator_has_multiple_auto_devices(accelerator: str | None) -> bool:
    """Return whether PTL auto/all device resolution can select multiple devices."""
    accelerator_name = (accelerator or "auto").strip().lower()
    if accelerator_name in ("auto", "cuda", "gpu"):
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    return False


def _requests_multiple_devices(devices: int | str, accelerator: str | None = None) -> bool:
    """Return whether the configured devices value explicitly requests multiple devices."""
    if isinstance(devices, int):
        if devices == -1:
            return _accelerator_has_multiple_auto_devices(accelerator)
        return devices > 1
    devices_name = devices.strip().lower()
    if devices_name in ("auto", "-1"):
        return _accelerator_has_multiple_auto_devices(accelerator)
    if devices_name.isdigit():
        return int(devices_name) > 1
    if "," in devices_name:
        return len([entry for entry in devices_name.split(",") if entry.strip()]) > 1
    return False


def _preserve_csv_history_across_resume(csv_logger: CSVLogger, output_dir: str | Path) -> None:
    """Stop CSVLogger from silently deleting metrics.csv history when a run resumes.

    ``build_trainer`` constructs a brand-new ``CSVLogger(version="")`` every time training starts or
    resumes, always pointed at the same ``output_dir``. The first access to ``CSVLogger.experiment``
    triggers PTL's ``_ExperimentWriter._check_log_dir_exists``, which deletes any pre-existing
    ``metrics.csv`` in that directory (see ``lightning.fabric.loggers.csv_logs``). On a fresh run there
    is nothing to delete, but on a resumed run this wipes every row logged before the resume. Snapshot
    the file before that deletion happens, restore it immediately after, and seed the writer's column
    cache to match so the next ``save()`` call appends rather than starting over.

    Only call this for a resumed run (a truthy tc.resume). This matches the
    public Trainer.fit(..., ckpt_path=config.resume or None) normalization.
    Reusing output_dir for a fresh run, including an empty resume value, must
    still let CSVLogger reset metrics.csv — appending fresh-run history onto
    unrelated prior-run history would silently corrupt the log.

    Args:
        csv_logger: The just-constructed ``CSVLogger`` for this run, not yet attached to a ``Trainer``.
        output_dir: The training run's output directory; must match ``csv_logger``'s log location
            (``name=""``, ``version=""``).

    Regression fix for :issue:`1321`.
    """
    metrics_path = Path(output_dir) / "metrics.csv"
    if not metrics_path.is_file():
        return
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return

    experiment = csv_logger.experiment  # Triggers PTL's delete-on-init; restored right after.
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    experiment.metrics_keys = sorted(fieldnames)


# TrainConfig.best_model_metric -> (keypoint_key, segmentation_key, detection_key), each
# relative to "val/" / "val/ema_". See _append_training_callbacks for how they are combined
# with the task branch (has_keypoints / model_config.segmentation_head).
_BEST_MODEL_MONITOR_KEYS: dict[str, tuple[str, str, str]] = {
    "map": ("keypoint_map_50_95", "segm_mAP_50_95", "mAP_50_95"),
    "mar": ("keypoint_mAR", "mAR", "mAR"),
}


def _append_training_callbacks(
    callbacks: list[Callback],
    loggers: list[Any],
    tc: TrainConfig,
    model_config: ModelConfig,
    *,
    enable_ema: bool,
    has_keypoints: bool,
) -> None:
    """Append the training-only callbacks and loggers to *callbacks* / *loggers* in place.

    Called by :func:`build_trainer` only when ``include_training_callbacks=True``: EMA, drop-path, checkpointing,
    best-model selection, early stopping, and the configured loggers (CSVLogger always; TensorBoard/WandB/MLflow
    optionally). Extracted from ``build_trainer`` to keep the training-vs-eval mode split to a single call site
    rather than a re-indented function body.

    Args:
        callbacks: The trainer's callback list, appended to in place.
        loggers: The trainer's logger list, appended to in place.
        tc: Training hyperparameter configuration.
        model_config: Architecture configuration (used to select the best-model monitor metric).
        enable_ema: Whether EMA is active (already resolved against sharded-strategy compatibility).
        has_keypoints: Whether the model uses the keypoint head (selects the monitor metric).
    """
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

    # Metric key per task, selected by TrainConfig.best_model_metric. "mar" reuses the box-level
    # val/mAR for both detection and segmentation (torchmetrics does not expose a separate mask
    # mAR), and the OKS-based val/keypoint_mAR for the keypoint task.
    kp_key, segm_key, det_key = _BEST_MODEL_MONITOR_KEYS[tc.best_model_metric]
    if has_keypoints:
        monitor_regular = f"val/{kp_key}"
        early_stopping_monitor_ema = f"val/ema_{kp_key}"
    elif model_config.segmentation_head:
        monitor_regular = f"val/{segm_key}"
        early_stopping_monitor_ema = f"val/ema_{segm_key}"
    else:
        monitor_regular = f"val/{det_key}"
        early_stopping_monitor_ema = f"val/ema_{det_key}"
    monitor_ema = early_stopping_monitor_ema if enable_ema else None

    best_model_smooth_alpha = tc.smooth_alpha

    # Best-model checkpointing — monitor EMA metric only when EMA is active and emitted.
    # PTL _reorder_callbacks moves all Checkpoint subclasses (including BestModelCallback)
    # to the end of the callback list; RFDETREarlyStopping (not a Checkpoint subclass) always
    # fires BEFORE BestModelCallback on every on_validation_end, regardless of append order.
    # The try/finally restore in BestModelCallback.on_validation_end guarantees EarlyStopping
    # always reads the raw (un-smoothed) metric value.
    callbacks.append(
        BestModelCallback(
            output_dir=str(tc.output_dir),
            monitor_regular=monitor_regular,
            monitor_ema=monitor_ema,
            run_test=tc.run_test,
            skip_best_epochs=tc.skip_best_epochs,
            smooth_alpha=best_model_smooth_alpha,
        )
    )

    # Optional early stopping.
    if tc.early_stopping:
        callbacks.append(
            RFDETREarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                use_ema=tc.early_stopping_use_ema,
                monitor_regular=monitor_regular,
                monitor_ema=early_stopping_monitor_ema,
                skip_best_epochs=tc.skip_best_epochs,
            )
        )

    # --- Build loggers ---
    # Each logger is guarded by a try/except because tensorboard, wandb, and mlflow
    # are optional dependencies (installed via the [loggers] extra).  A missing dep
    # emits a UserWarning instead of crashing.
    # CSVLogger is always enabled — no extra package required.
    # Produces metrics.csv in output_dir so there is always a log file.
    csv_logger = CSVLogger(save_dir=tc.output_dir, name="", version="")
    if tc.resume:
        _preserve_csv_history_across_resume(csv_logger, tc.output_dir)
    loggers.append(csv_logger)

    if tc.tensorboard:
        try:
            _try_import_tensorboard_summary_writer()
            loggers.append(
                TensorBoardLogger(
                    save_dir=tc.output_dir,
                    name="",
                    version="",
                )
            )
        except (ImportError, AttributeError) as exc:
            _logger.warning(
                "TensorBoard logging disabled: %s. "
                "If using NumPy 2.x, ensure your TensorBoard installation is NumPy 2.0 compatible "
                "(the failure can originate from tensorboard.compat.tensorflow_stub). "
                "Install TensorBoard with: pip install tensorboard",
                exc,
            )

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
                    save_dir=str(tc.output_dir),
                )
            )
        except ModuleNotFoundError as exc:
            _logger.warning("MLflow logging disabled: %s. Install with: pip install mlflow", exc)

    if tc.clearml:
        raise NotImplementedError("ClearML logging is not yet supported. Remove clearml=True from TrainConfig.")


class _ForceLastEpochValidationCallback(Callback):
    """Force a validation run on the final training epoch when ``eval_interval`` would skip it.

    ``check_val_every_n_epoch=N`` makes Lightning skip the whole validation loop on non-eval epochs (the compute saving
    behind ``eval_interval``), but Lightning has no "always validate the last epoch" switch while RF-DETR guarantees
    last-epoch metrics (``COCOEvalCallback`` treats the final epoch as an eval epoch). The fit loop reads
    ``trainer.check_val_every_n_epoch`` live on every epoch, so resetting it to 1 at the start of the final epoch re-
    enables validation exactly there.
    """

    def on_train_epoch_start(self, trainer: Trainer, pl_module: "LightningModule") -> None:
        """Reset ``check_val_every_n_epoch`` to 1 once the final epoch starts.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The module being trained (unused).
        """
        max_epochs = trainer.max_epochs
        if isinstance(max_epochs, int) and max_epochs > 0 and trainer.current_epoch >= max_epochs - 1:
            trainer.check_val_every_n_epoch = 1


def build_trainer(
    train_config: TrainConfig,
    model_config: ModelConfig,
    *,
    accelerator: str | None = None,
    include_training_callbacks: bool = True,
    **trainer_kwargs: Any,
) -> Trainer:
    """Assemble a PTL ``Trainer`` with the full RF-DETR callback and logger stack.

    Resolves training precision from ``model_config.amp`` and device capability, guards EMA against sharded strategies,
    wires conditional loggers, and applies promoted training knobs (sync_batchnorm, strategy).

    Args:
        train_config: Training hyperparameter configuration.
        model_config: Architecture configuration. Used for precision resolution
            (``model_config.amp``) and to guard against unsupported distributed
            configurations for keypoint models.
        accelerator: PTL accelerator string (e.g. ``"auto"``, ``"cpu"``, ``"gpu"``).
            Defaults to ``None`` which reads from ``train_config.accelerator`` (itself defaulting to ``"auto"``). Pass
            ``"cpu"`` to override auto-detection (e.g. when the caller explicitly requests CPU training via
            ``device="cpu"``).
        include_training_callbacks: When ``True`` (default) the full training stack is wired (EMA, drop-path,
            checkpointing, best-model selection, early stopping) along with the configured loggers. When ``False`` an
            evaluation-only trainer is built that keeps just the metric callback (and the progress bar): no
            checkpoints or logs are written. Used by :meth:`rfdetr.detr.RFDETR.evaluate`.
        **trainer_kwargs: Extra keyword arguments forwarded to ``pytorch_lightning.Trainer``. Use this to pass
            PTL-native flags that are not exposed through ``TrainConfig``, for example::

                build_trainer(tc, mc, fast_dev_run=2)

            Most keys present in both ``trainer_kwargs`` and the built config dict are overridden by the value in
            ``trainer_kwargs``. Detection and segmentation models forward ``accumulate_grad_batches`` from
            ``train_config.grad_accum_steps`` and ``gradient_clip_val`` from ``train_config.clip_max_norm`` to the
            Trainer normally. Keypoint models force ``accumulate_grad_batches=1`` and ``gradient_clip_val=None``
            because ``RFDETRModelModule`` owns both operations under manual optimization; passing those keys for a
            keypoint config raises a ``UserWarning`` to make the override explicit.

    Note:
        Two process-wide side effects: (1) unconditionally calls
        ``torch.set_float32_matmul_precision("high")``, which persists after this function
        returns and overrides any caller-set precision (e.g. ``"highest"``) with no opt-out —
        mirrors the import-time guard in ``rfdetr.detr`` so the Lightning CLI path
        (``rfdetr fit``) gets the same TF32 behavior as the python API path. (2) sets
        ``check_val_every_n_epoch=tc.eval_interval``, so ``eval_interval`` now gates the whole
        validation loop (forward pass, metric compute, EMA forward), not just result logging;
        a ``_ForceLastEpochValidationCallback`` still guarantees the final epoch always
        validates even when ``epochs`` is not a multiple of ``eval_interval``.

        When ``accelerator`` resolves to ``"xla"`` or ``"tpu"``, the resolved precision is
        set via an ``XLAPrecision`` plugin instead of a ``precision`` argument — PTL's
        ``XLAStrategy`` only accepts the plugin and raises ``TypeError`` for standard precision
        strings. The XLA plugin is appended to caller-supplied plugins.

    Returns:
        A configured ``pytorch_lightning.Trainer`` instance.
    """
    tc = train_config
    if accelerator is None:
        accelerator = tc.accelerator
    # XLAStrategy's precision_plugin setter only accepts the XLAPrecision plugin
    # (Literal["32-true", "16-true", "bf16-true"]); passing precision="bf16-mixed" raises
    # TypeError. Detected here so trainer_config assembly can translate the resolved precision
    # into the required XLA plugin after applying caller-provided Trainer arguments.
    xla_accelerator = str(accelerator).lower() in ("xla", "tpu")

    # TF32 matmul for fp32 residual matmuls on Ampere+.  ``rfdetr.detr`` sets this at import
    # time for the python API path, but the Lightning CLI path (``rfdetr fit``) never imports
    # that module — build_trainer is on every training entry path, so set it here as well.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # defensive parity with the rfdetr.detr import-time guard
        _logger.debug("torch.set_float32_matmul_precision('high') failed", exc_info=True)

    # --- Precision resolution ---
    def _resolve_precision() -> str:
        if not model_config.amp:
            if tc.amp_dtype != "auto":
                warnings.warn(
                    f"amp_dtype={tc.amp_dtype!r} has no effect when model_config.amp=False.",
                    UserWarning,
                    stacklevel=2,
                )
            return "32-true"
        # CPU accelerator: bf16 autocast on macOS CPU (Apple Silicon) is ~13x slower
        # than fp32 due to missing native bfloat16 kernels — no benefit, high cost.
        if accelerator == "cpu":
            return "32-true"
        # ``train_config.amp_dtype`` (a train() kwarg) lets callers pin the autocast dtype (see issue #1132):
        #   "auto" — bf16 on bf16-capable CUDA, fp16 otherwise (historical default);
        #   "fp16" — force "16-mixed" (e.g. deployment targets without bf16 support);
        #   "bf16" — force "bf16-mixed", falling back to fp16 with a warning when unsupported.
        # Unrecognised values are coerced to "auto" (with a warning) by TrainConfig validation.
        amp_dtype = tc.amp_dtype
        # Ampere+ GPUs support bf16-mixed which is scaler-free —
        # no GradScaler.scale/unscale/update overhead per optimizer step.
        # BF16 is safe for fine-tuning (pretrained weights loaded by default).
        # Training from random init with very small LR may underflow; pass
        # ``amp_dtype="fp16"`` if needed.
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
            if amp_dtype == "fp16":
                return "16-mixed"
            if amp_dtype == "bf16":
                if torch.cuda.is_bf16_supported():
                    return "bf16-mixed"
                _logger.warning(
                    "amp_dtype='bf16' was requested but this CUDA device does not support bfloat16; "
                    "falling back to fp16 ('16-mixed')."
                )
                warnings.warn(
                    "amp_dtype='bf16' was requested but this CUDA device does not support bfloat16; "
                    "falling back to fp16 ('16-mixed').",
                    UserWarning,
                    stacklevel=2,
                )
                return "16-mixed"
            # amp_dtype == "auto"
            return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        if torch.backends.mps.is_available():
            if amp_dtype == "bf16":
                _logger.warning(
                    "amp_dtype='bf16' is not applied on MPS; RF-DETR uses fp16 ('16-mixed') for MPS autocast."
                )
                warnings.warn(
                    "amp_dtype='bf16' is not applied on MPS; RF-DETR uses fp16 ('16-mixed') for MPS autocast.",
                    UserWarning,
                    stacklevel=2,
                )
            return "16-mixed"
        return "32-true"

    # --- Strategy + EMA sharding guard ---
    strategy = trainer_kwargs.get("strategy", tc.strategy)
    devices = trainer_kwargs.get("devices", tc.devices)
    num_nodes = trainer_kwargs.get("num_nodes", tc.num_nodes)
    strategy_name = strategy.strip().lower() if isinstance(strategy, str) else None
    has_keypoints = bool(model_config.use_grouppose_keypoints)
    if isinstance(tc, KeypointTrainConfig) != has_keypoints:
        raise ValueError(
            f"Config/model mismatch: isinstance(tc, KeypointTrainConfig)={isinstance(tc, KeypointTrainConfig)} "
            f"but model_config.use_grouppose_keypoints={model_config.use_grouppose_keypoints}. "
            "Pass KeypointTrainConfig for keypoint models and TrainConfig for detection models."
        )
    distributed_requested = (
        _is_distributed_strategy_requested(str(strategy))
        or num_nodes > 1
        or _requests_multiple_devices(devices, accelerator)
    )
    if has_keypoints and distributed_requested:
        # Keypoint models train with manual optimization (see RFDETRModelModule) and a
        # graph-connected keypoint loss (see models/heads/keypoints.py), so every rank's
        # keypoint-head parameters always receive a gradient and DistributedDataParallel's
        # reducer stays in sync. Combined with find_unused_parameters=True (set below),
        # DDP / ddp_spawn / ddp_notebook and multi-node DDP are supported.
        #
        # Sharded strategies (FSDP / DeepSpeed) shard optimizer state and gradients in ways
        # the manual-optimization + dynamic per-step loss-normalization path has not been
        # validated against, so those remain unsupported for keypoint models.
        if _is_sharded_strategy(strategy):
            raise NotImplementedError(
                "Keypoint training does not support sharded distributed strategies "
                f"(strategy={strategy!r}). Use DistributedDataParallel instead, e.g. strategy='ddp' "
                "(or strategy='auto' with devices>1), which is supported for keypoint models."
            )
        if isinstance(strategy, _DDPStrategy):
            # A supplied DDPStrategy object bypasses the string-strategy
            # find_unused_parameters wrap below. Keypoint models can leave parameters
            # unused on some steps (two-stage encoder / group-DETR branches), which plain
            # DDP with find_unused_parameters=False rejects mid-training, so require it.
            ddp_kwargs = getattr(strategy, "_ddp_kwargs", {})
            if not ddp_kwargs.get("find_unused_parameters", False):
                raise ValueError(
                    "Keypoint training under a supplied DDPStrategy requires "
                    "find_unused_parameters=True; construct it as "
                    "DDPStrategy(find_unused_parameters=True). Keypoint models can leave "
                    "parameters unused on some steps, which plain DDP rejects."
                )
        _logger.info(
            "Keypoint model + distributed execution (strategy=%r, devices=%r, num_nodes=%r) → "
            "DDP with manual optimization. For best throughput on multi-GPU keep grad_accum_steps=1: "
            "the manual-optimization path synchronizes gradients on every microbatch, so "
            "grad_accum_steps>1 is correct but performs redundant all-reduces.",
            strategy,
            devices,
            num_nodes,
        )

    # Transparently replace fork-based DDP with spawn-based DDP — see the
    # module-level comment block above _InteractiveSpawnLauncher for rationale.
    if strategy_name in ("ddp_notebook", "ddp_spawn"):
        strategy = _NotebookSpawnDDPStrategy(start_method="spawn", find_unused_parameters=True)
        _logger.info(
            "%s → spawn-based DDP to avoid OpenMP thread pool corruption after fork.",
            strategy_name,
        )
    elif strategy_name == "ddp" or (strategy_name == "auto" and distributed_requested):
        # DETR-family architectures can leave parameters unused on certain forward
        # steps under DDP, causing "It looks like your LightningModule has parameters
        # that were not used in producing the loss".  Sources include:
        #   - segmentation_head.sparse_forward() returning dict intermediates;
        #   - two-stage encoder query groups (group_detr ModuleLists) where per-group
        #     matcher assignment can leave groups without targets on low-annotation
        #     batches (issue #1093);
        #   - conditional auxiliary-loss branches.
        # Enabling find_unused_parameters lets DDP traverse the autograd graph after
        # each backward pass to identify which parameters contributed to the loss.
        # To opt out (e.g. configs with two_stage=False that never hit unused params),
        # pass strategy=DDPStrategy(find_unused_parameters=False) via trainer_kwargs.
        strategy = _DDPStrategy(find_unused_parameters=True)
        if strategy_name == "auto":
            _logger.info(
                "strategy='auto' with distributed execution → DDPStrategy(find_unused_parameters=True).",
            )
        else:
            _logger.info(
                "strategy='ddp' → DDPStrategy(find_unused_parameters=True).",
            )
    sharded = _is_sharded_strategy(strategy)
    enable_ema = bool(tc.use_ema) and not sharded
    if tc.use_ema and sharded:
        warnings.warn(
            f"EMA disabled: RFDETREMACallback is not compatible with sharded strategies "
            f"(strategy={strategy!r}). Set use_ema=False to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    # --- Build callbacks ---
    callbacks: list[Callback] = []

    if tc.progress_bar == "rich":
        callbacks.append(
            GPUMemoryRichProgressBar(
                refresh_rate=5,
                leave=True,
                theme=RichProgressBarTheme(metrics_format=".3e"),
            )
        )
    elif tc.progress_bar == "tqdm":
        callbacks.append(GPUMemoryTQDMProgressBar(refresh_rate=5))

    # Training-only callbacks and loggers.  Evaluation-only trainers
    # (``include_training_callbacks=False``, used by :meth:`rfdetr.detr.RFDETR.evaluate`) keep just the
    # progress bar and the ``COCOEvalCallback`` appended below, so they run no EMA / drop-path / best-model /
    # early-stopping and write no checkpoints or logs to ``output_dir``.  ``COCOEvalCallback`` writes its
    # metrics in the ``*_epoch_end`` hooks while ``BestModelCallback`` / ``RFDETREarlyStopping`` read them in
    # the later ``on_validation_end`` hook, so appending it after these callbacks does not change behaviour.
    loggers: list[Any] = []
    if include_training_callbacks:
        _append_training_callbacks(
            callbacks, loggers, tc, model_config, enable_ema=enable_ema, has_keypoints=has_keypoints
        )

    # COCO mAP + F1 — the metric engine shared by training-time validation and standalone evaluate().
    callbacks.append(
        COCOEvalCallback(
            max_dets=tc.eval_max_dets,
            segmentation=model_config.segmentation_head,
            eval_interval=tc.eval_interval,
            log_per_class_metrics=tc.log_per_class_metrics,
            keypoint_oks_sigmas=tc.keypoint_oks_sigmas,
            eval_ema_only=tc.eval_ema_only,
        )
    )

    # eval_interval must skip the whole validation loop, not just metric logging: without
    # check_val_every_n_epoch Lightning runs (and COCOEvalCallback discards) a full validation
    # pass — including the EMA forward — on every non-eval epoch.  The final epoch is always
    # validated (COCOEvalCallback treats it as an eval epoch), which Lightning's modulus check
    # alone does not guarantee when epochs % eval_interval != 0.
    if tc.eval_interval > 1:
        callbacks.append(_ForceLastEpochValidationCallback())

    # --- Promoted config fields (T4-2 added these to TrainConfig) ---
    clip_max_norm: float = tc.clip_max_norm
    sync_bn: bool = tc.sync_bn

    # Manual optimization (currently scoped to keypoint models) owns gradient accumulation
    # and clipping inside ``RFDETRModelModule._step_optimizer`` so the box-count denominator
    # spans the full effective batch.  Detection and segmentation models keep Lightning's
    # automatic optimization, which means ``accumulate_grad_batches`` and ``gradient_clip_val``
    # must flow through to the Trainer as usual for them.
    manual_optimization = has_keypoints
    if manual_optimization:
        accumulate_grad_batches: int = 1
        gradient_clip_val: float | None = None
    else:
        accumulate_grad_batches = tc.grad_accum_steps
        gradient_clip_val = clip_max_norm

    trainer_config: dict[str, Any] = {
        "max_epochs": tc.epochs,
        "accelerator": accelerator,
        "devices": tc.devices,
        "num_nodes": tc.num_nodes,
        "strategy": strategy,
        "accumulate_grad_batches": accumulate_grad_batches,
        "gradient_clip_val": gradient_clip_val,
        "sync_batchnorm": sync_bn,
        "callbacks": callbacks,
        "logger": loggers if loggers else False,
        # Disable PTL's implicit default ModelCheckpoint in eval mode so evaluation writes nothing to output_dir.
        "enable_checkpointing": include_training_callbacks,
        "enable_progress_bar": tc.progress_bar is not None,
        "default_root_dir": tc.output_dir,
        "log_every_n_steps": 50,
        "deterministic": False,
        "check_val_every_n_epoch": tc.eval_interval,
    }
    if not xla_accelerator:
        trainer_config["precision"] = _resolve_precision()
    trainer_config.update(trainer_kwargs)
    if xla_accelerator:
        from pytorch_lightning.plugins import XLAPrecision

        # XLAStrategy rejects precision= strings. Preserve caller plugins, then append the
        # one mandatory XLA precision plugin after kwargs have been applied.
        plugins = trainer_config.pop("plugins", [])
        if plugins is None:
            plugins = []
        elif not isinstance(plugins, (list, tuple)):
            plugins = [plugins]
        trainer_config.pop("precision", None)
        xla_precision = _normalize_xla_precision(_resolve_precision().replace("-mixed", "-true"))
        trainer_config["plugins"] = [*plugins, XLAPrecision(xla_precision)]
    trainer_config["strategy"] = strategy
    if manual_optimization:
        # Re-apply manual-optimization invariants so a caller-supplied trainer_kwargs
        # value cannot silently re-enable Lightning-owned accumulation or clipping while
        # the module is doing its own.  Warn loudly so the override is visible — silent
        # coercion has historically masked subtle gradient-scaling bugs on this code path.
        for key in ("accumulate_grad_batches", "gradient_clip_val"):
            if key in trainer_kwargs:
                effective = "1" if key == "accumulate_grad_batches" else "None"
                alt = "grad_accum_steps" if key == "accumulate_grad_batches" else "clip_max_norm"
                warnings.warn(
                    f"build_trainer() ignored trainer_kwargs[{key!r}]={trainer_kwargs[key]!r} for a keypoint "
                    f"model. The model will train with {key}={effective} regardless of the value passed here "
                    f"because RFDETRModelModule owns gradient accumulation and clipping under manual "
                    f"optimization. To change the effective value, set TrainConfig.{alt} instead.",
                    UserWarning,
                    stacklevel=2,
                )
        trainer_config["accumulate_grad_batches"] = 1
        # gradient_clip_val=None here does NOT disable gradient clipping — clipping is
        # performed inside RFDETRModelModule._step_optimizer using train_config.clip_max_norm
        # (see src/rfdetr/training/module_model.py).  Under manual optimization the module
        # owns the clipping step; passing None to the PTL Trainer simply prevents PTL from
        # doing a second redundant clip on top of the module's own.
        trainer_config["gradient_clip_val"] = None
    return Trainer(**trainer_config)
