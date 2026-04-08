# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Best-model checkpointing and early stopping callbacks for RF-DETR Lightning training."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import __version__ as ptl_version
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import get_version
from rfdetr.utilities.state_dict import _make_fit_loop_state, strip_checkpoint

logger = get_logger()


class BestModelCallback(ModelCheckpoint):
    """Track best validation mAP and save best checkpoints during training.

    Extends :class:`pytorch_lightning.callbacks.ModelCheckpoint` to save
    stripped ``{model, args, epoch}`` ``.pth`` files (instead of full ``.ckpt``
    files) and to track a separate EMA checkpoint in parallel.

    At the end of training the overall winner (regular vs EMA, strict ``>`` for
    EMA) is copied to ``checkpoint_best_total.pth`` and optimizer/scheduler
    state is stripped via :func:`rfdetr.util.misc.strip_checkpoint`.

    Checkpoints are only updated on validation epochs where the monitor metric
    is actually logged.  On non-eval epochs (when ``eval_interval > 1`` causes
    COCO evaluation to be skipped) the callback is a no-op.

    Args:
        output_dir: Directory where checkpoint files are written.
        monitor_regular: Metric key for the regular model mAP.
        monitor_ema: Metric key for the EMA model mAP.  ``None`` disables
            EMA tracking.
        run_test: If ``True``, run ``trainer.test()`` on the best model at
            the end of training.
    """

    FILE_EXTENSION = ".pth"

    def __init__(
        self,
        output_dir: str,
        monitor_regular: str = "val/mAP_50_95",
        monitor_ema: str | None = None,
        run_test: bool = True,
    ) -> None:
        super().__init__(
            dirpath=output_dir,
            filename="checkpoint_best_regular",
            monitor=monitor_regular,
            mode="max",
            save_top_k=1,
            save_on_train_epoch_end=False,
            verbose=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        self._monitor_ema = monitor_ema
        self._run_test = run_test
        self._best_ema: float = 0.0
        self._output_dir = Path(output_dir)
        # Stash current pl_module so _save_checkpoint (no pl_module param) can access it.
        self._current_pl_module: LightningModule | None = None

    @staticmethod
    def _build_checkpoint_payload(
        model_state_dict: dict[str, torch.Tensor],
        args_dict: object,
        trainer: Trainer,
        model_name: str | None = None,
    ) -> dict[str, object]:
        """Build a PTL-compatible RF-DETR checkpoint payload.

        Args:
            model_state_dict: Model weights with raw (non-prefixed) keys.
            args_dict: Serialized training args/config payload.
            trainer: Active Lightning trainer providing epoch/step counters.
            model_name: Name of the model class (e.g. ``"RFDETRLarge"``).

        Returns:
            Checkpoint dictionary that supports ``Trainer.fit(ckpt_path=...)``
            while intentionally omitting optimizer/scheduler states.
        """
        payload: dict[str, object] = {
            "model": model_state_dict,
            "args": args_dict,
            "epoch": trainer.current_epoch,
            # PTL-compatible keys so trainer.fit(ckpt_path=...) works directly.
            "state_dict": {f"model.{k}": v for k, v in model_state_dict.items()},
            "global_step": trainer.global_step,
            "pytorch-lightning_version": ptl_version,
            "loops": {"fit_loop": _make_fit_loop_state(trainer.current_epoch)},
            # Keep keys present with empty values so PTL resume paths that
            # expect them can proceed without loading optimizer state.
            "optimizer_states": [],
            "lr_schedulers": [],
        }
        # Only write model_name when resolved — omit the key entirely when None
        # so old-format and unresolved checkpoints are indistinguishable.
        if model_name is not None:
            payload["model_name"] = model_name
        # Record the rfdetr package version for provenance / compatibility hints.
        # Omit the key when the version cannot be resolved (e.g. editable install
        # without package metadata) so old-format checkpoints are indistinguishable.
        version = get_version()
        if version is not None:
            payload["rfdetr_version"] = version
        return payload

    @staticmethod
    def _get_ema_model_state_dict(
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> dict[str, torch.Tensor]:
        """Resolve EMA model weights from the active EMA callback.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.

        Returns:
            EMA model state dict when available, otherwise the live model state dict.
        """
        for callback in trainer.callbacks:
            getter = getattr(callback, "get_ema_model_state_dict", None)
            if callable(getter):
                state_dict = getter()
                if state_dict is not None:
                    return state_dict
                break
        logger.warning(
            "EMA metric improved but EMA callback weights were unavailable; saving current model weights as fallback."
        )
        _orig = getattr(pl_module.model, "_orig_mod", None)
        raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
        return raw.state_dict()

    @staticmethod
    def _resolve_model_name(pl_module: LightningModule) -> str | None:
        """Resolve checkpoint model_name from model_config or config type.

        The CLI/PTL path does not call ``RFDETR.train()``, so
        ``model_config.model_name`` may be unset. In that case, infer the model
        class from concrete config names like ``RFDETRSmallConfig``.

        Note:
            The ``DeprecatedConfig`` ``RuntimeError`` guard is only reachable
            from the CLI/PTL path. ``RFDETR.train()`` pre-populates
            ``model_config.model_name`` before saving any checkpoint, so the
            config type-name branch (and therefore the ``DeprecatedConfig``
            guard) is never reached when training is started via
            ``RFDETR.train()``.
        """
        model_config = getattr(pl_module, "model_config", None)
        configured_name = getattr(model_config, "model_name", None) if model_config is not None else None
        if isinstance(configured_name, str):
            normalized_name = configured_name.strip()
            if normalized_name:
                return normalized_name

        config_type_name = type(model_config).__name__ if model_config is not None else ""

        if config_type_name.endswith("DeprecatedConfig"):
            raise RuntimeError(
                f"Deprecated model config '{config_type_name}' is no longer supported. "
                "Re-train your model using a current model variant."
            )
        if config_type_name.startswith("RFDETR") and config_type_name.endswith("Config"):
            return config_type_name.removesuffix("Config")
        return None

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        """Save stripped ``.pth`` format instead of a full ``.ckpt``.

        Skips on non-main processes.  Intentionally does NOT call
        ``trainer.save_checkpoint()`` — we only want ``{model, args, epoch}``.

        Args:
            trainer: The Lightning Trainer instance.
            filepath: Destination path (ends in ``.pth`` via ``FILE_EXTENSION``).
        """
        if not trainer.is_global_zero:
            return
        pl_module = self._current_pl_module
        if pl_module is None:
            raise RuntimeError(
                f"BestModelCallback._save_checkpoint called with filepath={filepath!r} "
                f"at epoch={trainer.current_epoch} but pl_module was not set."
            )
        pth_path = Path(filepath)
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        # Validation metrics are produced with EMA weights when the EMA callback
        # is active, so save the same weight source to keep metric/checkpoint
        # consistency for the monitored "regular" key.
        # Unwrap torch.compile's OptimizedModule (_orig_mod) so checkpoints always
        # contain plain keys — non-compiled consumers (sync-back, compat.evaluate) can load them.
        if self._monitor_ema is not None:
            model_state_dict = self._get_ema_model_state_dict(trainer, pl_module)
        else:
            _orig = getattr(pl_module.model, "_orig_mod", None)
            raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
            model_state_dict = raw.state_dict()
        # Enrich train_config with dataset class names so reloaded checkpoints
        # return the correct labels, not COCO defaults (#509).
        train_config = pl_module.train_config
        dataset_class_names = getattr(trainer.datamodule, "class_names", None)
        if (
            dataset_class_names is not None
            and hasattr(train_config, "model_copy")
            and getattr(train_config, "class_names", None) is None
        ):
            train_config = train_config.model_copy(update={"class_names": dataset_class_names})
        args_dict = train_config.model_dump() if hasattr(train_config, "model_dump") else train_config
        model_name = self._resolve_model_name(pl_module)
        torch.save(
            self._build_checkpoint_payload(model_state_dict, args_dict, trainer, model_name=model_name), pth_path
        )
        self._last_global_step_saved = trainer.global_step
        logger.info("Best regular mAP saved to %s (epoch %d)", pth_path, trainer.current_epoch)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save best regular/EMA checkpoints when validation mAP improves.

        Delegates regular-model checkpoint management to the
        :class:`~pytorch_lightning.callbacks.ModelCheckpoint` parent (handles
        improvement detection, fast_dev_run/sanity guards, ``best_model_path``
        and ``best_model_score`` bookkeeping).  EMA is tracked independently.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        # Stash for use inside _save_checkpoint (which has no pl_module param).
        self._current_pl_module = pl_module
        # Guard: only run checkpoint logic when the monitored metric was actually
        # logged this epoch (non-eval epochs with eval_interval > 1 skip COCO eval
        # so the key is absent from callback_metrics).
        if self.monitor not in trainer.callback_metrics:
            return
        super().on_validation_end(trainer, pl_module)

        # EMA model — custom tracking on top of parent.
        if self._monitor_ema is None or not trainer.is_global_zero:
            return
        ema_val = trainer.callback_metrics.get(self._monitor_ema, torch.tensor(0.0)).item()
        if ema_val > self._best_ema:
            self._best_ema = ema_val
            self._output_dir.mkdir(parents=True, exist_ok=True)
            ema_state_dict = self._get_ema_model_state_dict(trainer, pl_module)
            # Enrich train_config with dataset class names so reloaded checkpoints
            # return the correct labels, not COCO defaults (#509).
            ema_train_config = pl_module.train_config
            dataset_class_names = getattr(trainer.datamodule, "class_names", None)
            if (
                dataset_class_names is not None
                and hasattr(ema_train_config, "model_copy")
                and getattr(ema_train_config, "class_names", None) is None
            ):
                ema_train_config = ema_train_config.model_copy(update={"class_names": dataset_class_names})
            ema_args_dict = (
                ema_train_config.model_dump() if hasattr(ema_train_config, "model_dump") else ema_train_config
            )
            ema_model_name = self._resolve_model_name(pl_module)
            torch.save(
                self._build_checkpoint_payload(ema_state_dict, ema_args_dict, trainer, model_name=ema_model_name),
                self._output_dir / "checkpoint_best_ema.pth",
            )
            logger.info(
                "Best EMA mAP improved to %.4f (epoch %d)",
                ema_val,
                trainer.current_epoch,
            )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Select the overall best model and optionally run test evaluation.

        Copies the winner (regular vs EMA, strict ``>`` for EMA) to
        ``checkpoint_best_total.pth``, strips optimizer/scheduler state, then
        optionally runs ``trainer.test()``.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        if not trainer.is_global_zero:
            return

        best_regular = self.best_model_score.item() if self.best_model_score is not None else 0.0
        regular_path = Path(self.best_model_path) if self.best_model_path else None
        ema_path = self._output_dir / "checkpoint_best_ema.pth"
        total_path = self._output_dir / "checkpoint_best_total.pth"

        # Strict > for EMA to win (matches legacy behaviour).
        best_is_ema = self._best_ema > best_regular
        best_path = ema_path if (best_is_ema and ema_path.exists()) else regular_path

        if best_path and best_path.exists():
            shutil.copy2(best_path, total_path)
            strip_checkpoint(total_path)
            logger.info(
                "Best total checkpoint saved from %s (regular=%.4f, ema=%.4f)",
                "EMA" if best_is_ema else "regular",
                best_regular,
                self._best_ema,
            )

        if self._run_test:
            # Only call trainer.test() when the module actually defines test_step().
            cls_test_step = getattr(type(pl_module), "test_step", None)
            has_test_step = cls_test_step is not None and cls_test_step is not LightningModule.test_step
            if has_test_step:
                # Load best weights before test — mirrors legacy main.py:602-609.
                if total_path.exists():
                    ckpt = torch.load(total_path, map_location="cpu", weights_only=False)
                    # Checkpoints always store plain keys; load into the unwrapped module
                    # so compiled (OptimizedModule) and non-compiled models both work.
                    _orig = getattr(pl_module.model, "_orig_mod", None)
                    raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
                    raw.load_state_dict(ckpt["model"], strict=True)
                    logger.info("Loaded best weights from %s for test evaluation.", total_path)
                trainer.test(pl_module, datamodule=trainer.datamodule, verbose=False)


class RFDETREarlyStopping(EarlyStopping):
    """Early stopping callback monitoring validation mAP for RF-DETR.

    Extends :class:`pytorch_lightning.callbacks.EarlyStopping` with dual-metric
    monitoring: by default it monitors ``max(regular_mAP, ema_mAP)`` (legacy
    behaviour); set ``use_ema=True`` to monitor the EMA metric exclusively.

    The effective metric is injected into ``trainer.callback_metrics`` under a
    synthetic key before delegating to the parent's stopping logic, so all parent
    features are available for free: ``state_dict``/``load_state_dict`` for
    checkpoint resumption, NaN/inf guard via ``check_finite``, and
    ``stopping_threshold``/``divergence_threshold``.

    Early stopping evaluates only on validation epochs where the monitored
    metrics are logged; non-eval epochs (``eval_interval > 1``) are skipped
    automatically.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum mAP improvement to reset the patience counter.
        use_ema: When ``True`` and both regular and EMA metrics are available,
            monitor only the EMA metric.  When ``False``, monitor
            ``max(regular, ema)``.
        monitor_regular: Metric key for the regular model mAP.
        monitor_ema: Metric key for the EMA model mAP.
        verbose: If ``True``, log early stopping status each epoch.
    """

    _SYNTHETIC_MONITOR: str = "__rfdetr_effective_map__"

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        use_ema: bool = False,
        monitor_regular: str = "val/mAP_50_95",
        monitor_ema: str = "val/ema_mAP_50_95",
        verbose: bool = True,
    ) -> None:
        super().__init__(
            monitor=self._SYNTHETIC_MONITOR,
            mode="max",
            patience=patience,
            min_delta=min_delta,
            check_on_train_epoch_end=False,
            verbose=verbose,
            check_finite=True,
            strict=False,  # We inject the key ourselves; don't crash if temporarily absent.
            log_rank_zero_only=True,
        )
        self._monitor_regular = monitor_regular
        self._monitor_ema = monitor_ema
        self._use_ema = use_ema

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute effective mAP and delegate to parent stopping logic.

        Computes ``ema_mAP`` or ``max(regular_mAP, ema_mAP)`` depending on
        ``use_ema``, injects the result under the synthetic monitor key, then
        calls :meth:`EarlyStopping.on_validation_end` which handles patience,
        ``trainer.should_stop``, logging, and ``state_dict`` persistence.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        metrics = trainer.callback_metrics
        regular_tensor = metrics.get(self._monitor_regular)
        ema_tensor = metrics.get(self._monitor_ema)

        regular_val: float | None = regular_tensor.item() if regular_tensor is not None else None
        ema_val: float | None = ema_tensor.item() if ema_tensor is not None else None

        if regular_val is None and ema_val is None:
            return  # No metrics available — skip (matches legacy noop behaviour).

        if self._use_ema and ema_val is not None:
            effective = ema_val
        elif regular_val is not None and ema_val is not None:
            effective = max(regular_val, ema_val)
        elif ema_val is not None:
            effective = ema_val
        else:
            effective = regular_val  # type: ignore[assignment]

        trainer.callback_metrics[self._SYNTHETIC_MONITOR] = torch.tensor(effective)
        super().on_validation_end(trainer, pl_module)
