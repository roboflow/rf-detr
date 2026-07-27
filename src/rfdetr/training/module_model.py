# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningModule for RF-DETR training and validation."""

from __future__ import annotations

import importlib
import inspect
import math
import random
import warnings
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias (see AGENTS.md)
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import (
    ModelConfig,
    TrainConfig,
    _is_managed_optimizer_name,
    _is_managed_scheduler_name,
    _resolve_native_optimizer,
)
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import apply_lora, interpolate_position_embeddings, load_pretrain_weights
from rfdetr.training.callbacks.coco_eval import _get_ema_inner_module
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_OptimizerFactory = Callable[..., torch.optim.Optimizer]

_TRAIN_PROGRESS_LOSS_ALIASES: dict[str, str] = {
    "loss_ce": "loss_cls",
    "loss_bbox": "loss_box",
    "loss_giou": "loss_giou",
    "loss_mask_ce": "mask_ce",
    "loss_mask_dice": "mask_dice",
    "loss_keypoints_l1": "kp_l1",
    "loss_keypoints_findable": "kp_find",
    "loss_keypoints_visible": "kp_vis",
    "loss_keypoints_nll": "kp_nll",
}


def _is_builtin_fused_adamw(optimizer: object) -> bool:
    """Return whether the config selects RF-DETR's built-in (managed, fused) AdamW path.

    Args:
        optimizer: The ``TrainConfig.optimizer`` value (string or callable).

    Returns:
        ``True`` only for the built-in ``"adamw"`` short name.

    Examples:
        >>> _is_builtin_fused_adamw("adamw")
        True
        >>> _is_builtin_fused_adamw("torch.optim.AdamW")
        False
    """
    return isinstance(optimizer, str) and "." not in optimizer and optimizer.strip().lower() == "adamw"


_FUSED_IGNORED_MSG = (
    "fused_optimizer=True is ignored for optimizer=%r; the fused AdamW kernel only applies to the "
    "built-in optimizer='adamw' path."
)


def _import_optimizer_class(dotted_path: str) -> _OptimizerFactory:
    """Import an optimizer class or factory from a dotted path.

    Args:
        dotted_path: Fully-qualified path such as ``"torch.optim.AdamW"`` or
            ``"pytorch_optimizer.Lion"``.

    Returns:
        The imported optimizer class or factory.

    Raises:
        ValueError: If the module or attribute cannot be imported.
    """
    module_path, _, attribute = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"optimizer {dotted_path!r} is not a valid dotted import path.")
    try:
        module = importlib.import_module(module_path)
        return cast(_OptimizerFactory, getattr(module, attribute))
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not import optimizer {dotted_path!r}: {exc}. "
            "Use a fully-qualified path to an importable optimizer class, e.g. 'torch.optim.AdamW'."
        ) from exc


def _instantiate_explicit_optimizer(
    optimizer_class: _OptimizerFactory,
    optimizer_name: str,
    param_dicts: list[dict[str, Any]],
    optimizer_kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    """Instantiate an explicitly-selected optimizer from param groups and kwargs only.

    Explicit optimizers (dotted import paths and callables) receive the RF-DETR
    parameter groups — which already carry per-group learning rates — plus the
    user's ``optimizer_kwargs`` verbatim. RF-DETR injects no ``lr`` or
    ``weight_decay`` of its own.

    Args:
        optimizer_class: Optimizer class or factory to instantiate.
        optimizer_name: Name used in error messages.
        param_dicts: RF-DETR parameter groups with layer-wise learning rates.
        optimizer_kwargs: Keyword arguments forwarded verbatim to the constructor.

    Returns:
        Instantiated optimizer.

    Raises:
        TypeError | ValueError: Re-raised with an RF-DETR-specific hint on failure.
    """
    try:
        return optimizer_class(param_dicts, **optimizer_kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Failed to initialize optimizer {optimizer_name!r}: {exc}. "
            "Explicit optimizers (dotted paths and callables) are built from the RF-DETR parameter "
            "groups plus your `optimizer_kwargs` only, with no lr/weight_decay injected. Check that the "
            "class accepts these arguments, or pass a callable/functools.partial needing only `params`."
        ) from exc


def _optimizer_accepts_kwarg(optimizer_class: _OptimizerFactory, name: str) -> bool:
    """Return whether an optimizer constructor accepts a given keyword argument.

    Args:
        optimizer_class: Optimizer class or factory to inspect.
        name: Keyword-argument name to look for.

    Returns:
        ``True`` when the constructor declares ``name`` or accepts arbitrary
        keyword arguments, or when its signature cannot be introspected (e.g. a
        C-implemented constructor) — in which case the constructor validates the
        call itself.

    Examples:
        >>> _optimizer_accepts_kwarg(torch.optim.SGD, "weight_decay")
        True
    """
    try:
        signature = inspect.signature(optimizer_class)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    return name in signature.parameters


def _instantiate_optimizer(
    optimizer_class: _OptimizerFactory,
    optimizer_name: str,
    param_dicts: list[dict[str, Any]],
    train_config: TrainConfig,
) -> torch.optim.Optimizer:
    """Instantiate an optimizer class with RF-DETR optimizer arguments.

    ``weight_decay`` is injected only when the optimizer constructor accepts it,
    so optimizers with a different regularization API are not forced to fail.

    Args:
        optimizer_class: Optimizer class or factory to instantiate.
        optimizer_name: Name used in error messages.
        param_dicts: RF-DETR parameter groups with layer-wise learning rates.
        train_config: Training config with base optimizer hyperparameters.

    Returns:
        Instantiated optimizer.

    Raises:
        TypeError | ValueError: Re-raised with an RF-DETR-specific hint when the
            optimizer constructor rejects the supplied arguments.
    """
    init_kwargs: dict[str, Any] = {"lr": train_config.lr}
    if _optimizer_accepts_kwarg(optimizer_class, "weight_decay"):
        init_kwargs["weight_decay"] = train_config.weight_decay
    init_kwargs.update(train_config.optimizer_kwargs)
    try:
        return optimizer_class(param_dicts, **init_kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Failed to initialize optimizer {optimizer_name!r}: {exc}. "
            "For managed torch.optim short names, RF-DETR passes `params`, `lr`, and (when supported) "
            "`weight_decay`, then your `optimizer_kwargs`; this usually means an unsupported entry in "
            "`optimizer_kwargs`."
        ) from exc


_SchedulerFactory = Callable[..., LRScheduler | ReduceLROnPlateau]


def _import_scheduler_class(dotted_path: str) -> _SchedulerFactory:
    """Import an LR-scheduler class or factory from a dotted path.

    Args:
        dotted_path: Fully-qualified path such as ``"torch.optim.lr_scheduler.StepLR"`` or
            ``"pytorch_optimizer.CosineAnnealingWarmupRestarts"``.

    Returns:
        The imported scheduler class or factory.

    Raises:
        ValueError: If the module or attribute cannot be imported.
    """
    module_path, _, attribute = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"lr_scheduler {dotted_path!r} is not a valid dotted import path.")
    try:
        module = importlib.import_module(module_path)
        return cast(_SchedulerFactory, getattr(module, attribute))
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not import lr_scheduler {dotted_path!r}: {exc}. "
            "Use a fully-qualified path to an importable scheduler class, e.g. 'torch.optim.lr_scheduler.StepLR'."
        ) from exc


def _instantiate_explicit_scheduler(
    scheduler_factory: _SchedulerFactory,
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    scheduler_kwargs: dict[str, Any],
) -> LRScheduler | ReduceLROnPlateau:
    """Instantiate an explicitly-selected LR scheduler from the optimizer and kwargs only.

    Explicit schedulers (dotted import paths and callables) receive the built optimizer plus the
    user's ``lr_scheduler_kwargs`` verbatim. RF-DETR injects no ``total_steps`` / ``T_max`` of its
    own — the managed ``"step"`` / ``"cosine"`` presets remain the runtime-aware option.

    Args:
        scheduler_factory: Scheduler class or factory to instantiate.
        scheduler_name: Name used in error messages.
        optimizer: The optimizer the scheduler drives.
        scheduler_kwargs: Keyword arguments forwarded verbatim to the constructor.

    Returns:
        Instantiated scheduler.

    Raises:
        TypeError | ValueError: Re-raised with an RF-DETR-specific hint on failure.
    """
    try:
        return scheduler_factory(optimizer, **scheduler_kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"Failed to initialize lr_scheduler {scheduler_name!r}: {exc}. "
            "Explicit schedulers (dotted paths and callables) are built from the optimizer plus your "
            "`lr_scheduler_kwargs` only, with no total_steps/T_max injected. Check that the class accepts these "
            "arguments, or pass a callable/functools.partial needing only the optimizer."
        ) from exc


def _build_managed_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    total_steps: int,
    steps_per_epoch: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the managed ``"step"`` / ``"cosine"`` scheduler as a warmup-aware ``LambdaLR``.

    Preserves RF-DETR's built-in schedule: a linear warmup ramp over ``warmup_steps`` followed by
    either cosine annealing down to ``min_factor`` or a 10x step decay after ``lr_drop`` epochs. The
    ``min_factor`` and ``lr_drop`` values are read from ``lr_scheduler_kwargs`` first (the current API),
    falling back to the deprecated ``lr_min_factor`` / ``lr_drop`` fields.

    Args:
        optimizer: The optimizer the scheduler drives.
        train_config: Training config carrying the preset name and schedule knobs.
        total_steps: Total optimizer steps over the whole run.
        steps_per_epoch: Optimizer steps per epoch.
        warmup_steps: Number of optimizer steps in the linear warmup ramp.

    Returns:
        A ``LambdaLR`` implementing the managed schedule.
    """
    kwargs = train_config.lr_scheduler_kwargs
    # Managed presets are always strings (guaranteed by the _is_managed_scheduler_name branch at the call site).
    preset = cast(str, train_config.lr_scheduler).strip().lower()
    min_factor = float(kwargs.get("min_factor", train_config.lr_min_factor))
    lr_drop = int(kwargs.get("lr_drop", train_config.lr_drop))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if preset == "cosine":
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
        # Step decay: drop by 10x after lr_drop epochs.
        if current_step < lr_drop * steps_per_epoch:
            return 1.0
        return 0.1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _wrap_with_warmup(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Prepend a linear warmup ramp to an explicit scheduler via ``SequentialLR``.

    The warmup ramps the LR from ``1 / warmup_steps`` of its base value up to full over
    ``warmup_steps`` optimizer steps, then hands control to ``scheduler``. ``ReduceLROnPlateau`` is
    metric-driven and cannot be composed this way — callers must skip wrapping it.

    Args:
        scheduler: The explicit scheduler to run after warmup.
        optimizer: The optimizer both schedulers drive.
        warmup_steps: Number of optimizer steps in the linear warmup ramp.

    Returns:
        A ``SequentialLR`` chaining the warmup ramp and ``scheduler``.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(1, warmup_steps),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler],
        milestones=[warmup_steps],
    )


class RFDETRModelModule(LightningModule):
    """LightningModule wrapping the RF-DETR model and training loop.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        # Manual optimization is enabled only for keypoint models so that the box-count
        # normalizer can be accumulated across grad-accum microbatches. Detection and
        # segmentation use Lightning's automatic optimization (PTL handles accumulation,
        # AMP, and gradient clipping), which keeps their step semantics unchanged from
        # the pre-fix/scaling behaviour.
        self._use_manual_optimization: bool = bool(getattr(model_config, "use_grouppose_keypoints", False))
        self.automatic_optimization = not self._use_manual_optimization
        # LR-scheduler stepping cadence resolved in configure_optimizers(); read by the manual-optimization
        # step loop and the epoch-end hook. Defaults keep pre-configure behaviour (per-step stepping).
        self._lr_scheduler_interval: str = "step"
        self._lr_scheduler_monitor: str | None = None
        self._accumulated_box_normalizer: Tensor | None = None
        # Allow partial state-dict loading when resuming from a .pth checkpoint
        # (which contains only model weights, not criterion/postprocess state).
        self.strict_loading = False

        # Model, criterion, and postprocessor.
        self.model = build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            # Canonical loader handles PE interpolation, PTL .ckpt normalisation,
            # per-group query slicing, class-name extraction, partial-load warnings,
            # and writes any auto-aligned ``num_classes`` back onto ``model_config``.
            load_pretrain_weights(self.model, self.model_config)
            if model_config.use_grouppose_keypoints:
                # Older model shims may omit the keypoint reset hook; call it only when implemented.
                reset_keypoint_gaussian_parameters = getattr(self.model, "reset_keypoint_gaussian_parameters", None)
                if callable(reset_keypoint_gaussian_parameters):
                    reset_keypoint_gaussian_parameters()
                    logger.info(
                        "Reset keypoint Gaussian precision outputs to unit values after pretrained weight load."
                    )
        if model_config.backbone_lora:
            apply_lora(self.model)

        # Build criterion/postprocessors after potential num_classes alignment so
        # they are constructed with a config that matches the current model head.
        self.criterion, self.postprocess = build_criterion_from_config(self.model_config, self.train_config)

        # torch.compile is opt-in: set model_config.compile=True to enable.
        # Only enabled on CUDA; MPS and CPU do not benefit from compilation.
        # Use the fork-safe DEVICE constant instead of torch.cuda.is_available(),
        # which creates a CUDA driver context that breaks fork-based DDP.
        from rfdetr.config import DEVICE

        accelerator = str(train_config.accelerator).lower()
        uses_cuda_accelerator = accelerator in {"auto", "gpu", "cuda"}
        compile_enabled = (
            model_config.compile and DEVICE == "cuda" and uses_cuda_accelerator and not train_config.multi_scale
        )
        if model_config.compile and train_config.multi_scale:
            logger.info("Disabling torch.compile because multi_scale=True introduces dynamic input shapes.")
        if compile_enabled:
            # dynamic=True: one compiled graph handles all multi-scale input sizes instead
            # of recompiling per (H, W) pair. suppress_errors=True: if inductor can't
            # compile a subgraph (e.g. bicubic backward with symbolic shapes), it falls
            # back to eager mode for that subgraph rather than crashing.
            # capture_scalar_outputs=True: include Tensor.item() calls
            # (gen_encoder_output_proposals / ms_deform_attn use spatial-shape .item()
            # as Python slice indices). Safe with dynamic=True because item() results
            # are backed symbols derived from input shapes — not unbacked symbols that
            # would cause PendingUnbackedSymbolNotFound (which only occurs without dynamic).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            # OptimizedModule forwards attribute access to the wrapped LWDETR via
            # __getattr__ at runtime, so self.model keeps working everywhere it's used below.
            self.model = torch.compile(self.model, dynamic=True)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Seed RNGs at fit start when ``TrainConfig.seed`` is set.

        This avoids hidden global side-effects in ``build_trainer`` while still preserving deterministic training
        behaviour for actual fit runs.
        """
        if self.train_config.seed is not None:
            seed_everything(self.train_config.seed + self.global_rank, workers=True)

    def on_train_start(self) -> None:
        """Normalize restored fused-optimizer state before the first training step.

        Lightning restores optimizer state after ``on_fit_start``.  Fused AdamW is strict about the dtype, device, and
        layout of its moment tensors, so resuming from a checkpoint can fail if Lightning rehydrates those tensors in a
        layout that no longer matches the live parameters.  Recasting same-shaped floating-point tensors here keeps the
        resumed optimizer compatible without discarding the saved momentum state.
        """
        if not self._use_fused_optimizer:
            return

        try:
            optimizers = self.optimizers(use_pl_optimizer=False)
        except RuntimeError:
            return
        if optimizers is None:
            return
        if isinstance(optimizers, list):
            optimizer_list = optimizers
        else:
            optimizer_list = [optimizers]

        normalized_tensors = 0
        for optimizer in optimizer_list:
            if not isinstance(optimizer, torch.optim.Optimizer):
                optimizer = getattr(optimizer, "optimizer", optimizer)
            if isinstance(optimizer, torch.optim.Optimizer):
                normalized_tensors += self._normalize_optimizer_state(optimizer)
        if normalized_tensors and getattr(self.trainer, "is_global_zero", True):
            logger.info(
                "Normalized %d restored fused AdamW state tensors after checkpoint resume.",
                normalized_tensors,
            )

    def on_train_batch_start(self, batch: tuple[Any, Any], batch_idx: int) -> None:
        """Apply optional multi-scale resize to the incoming batch.

        Modifications to ``batch`` (in-place on ``NestedTensor``) are visible in ``training_step`` because they share
        the same object.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Index of the current batch within the epoch.
        """
        tc = self.train_config
        mc = self.model_config

        if tc.multi_scale and not tc.do_random_resize_via_padding:
            samples, _ = batch
            scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)
            step = self.trainer.global_step
            # Use a step-local generator so the scale choice is deterministic and DDP-consistent
            # without reseeding the process-global RNG on every batch.
            scale = random.Random(step).choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = (
                    F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                )

    def on_train_epoch_start(self) -> None:
        """Reset the accumulated box normalizer at the start of every training epoch.

        Lightning may reuse the module across epochs without calling ``_step_optimizer`` at the boundary (for example
        when an epoch ends mid-accumulation window with a non-divisible batch count). Clearing the accumulator here
        guarantees the manual-optimization path always starts each epoch from a known state, so the first microbatch's
        gradients are scaled by its own box count and not by a stale previous-epoch denominator.

        This is a no-op for non-keypoint models because they use Lightning's automatic optimization path and never
        populate ``self._accumulated_box_normalizer``.

        Note: on finite datasets the final-batch fallback in ``_should_step_optimizer`` always flushes a partial
        trailing window, so this reset is the only change needed.  On IterableDatasets (infinite
        ``num_training_batches``) a partial window may survive epoch end with un-stepped gradients; those are
        discarded here and the optimizer is zeroed so the first microbatch of the new epoch starts from a clean state.
        """
        if self._accumulated_box_normalizer is not None:
            # Discard any partial accumulation window that survived the epoch boundary
            # (only possible for IterableDatasets where num_training_batches is infinite).
            try:
                opts = self.optimizers()
                for opt in opts if isinstance(opts, list) else [opts]:
                    opt.zero_grad()
            except RuntimeError:
                pass  # Not attached to Trainer (unit-test context); nothing to zero.
        self._accumulated_box_normalizer = None

    def training_step(self, batch: tuple[Any, Any], batch_idx: int) -> Tensor | dict[str, Any]:
        """Compute loss for one training step and log metrics.

        PTL handles AMP (``precision``) without a manual ``GradScaler``. Keypoint models perform manual optimization so
        box-count loss normalization is based on the full accumulated effective batch rather than each microbatch
        independently; detection and segmentation models keep Lightning's automatic optimization path.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss tensor by default. When ``compute_train_metrics=True``,
            returns a Lightning-compatible dict containing ``loss`` plus
            detached postprocessed predictions for train mAP logging.
        """
        samples, targets = batch
        batch_size = len(targets)
        outputs = self.model(samples, targets)
        if self._use_manual_optimization:
            loss_dict, raw_loss, normalizer = self._compute_train_losses(outputs, targets)
            loss_for_backward = self._scale_loss_for_accumulation(raw_loss, normalizer)
        else:
            loss_dict = self.criterion(outputs, targets)
            loss_for_backward = None
        weight_dict = self.criterion.weight_dict
        loss: Tensor = torch.stack([loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]).sum()
        # Automatic optimization path: divide by accumulate_grad_batches so the accumulated
        # gradient matches a single large batch, matching the legacy engine.  PTL accumulates
        # full-scale gradients by default; dividing here keeps the effective LR identical.
        accumulate_grad_batches = max(1, int(self.trainer.accumulate_grad_batches))
        loss_for_return = loss if self._use_manual_optimization else loss / accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self._log_train_progress_metrics(loss, loss_dict, batch_size=batch_size)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        # Optimizer may have multiple param groups with different LRs (e.g., backbone/decoder).
        # Preserve the first group's LR for backward compatibility, but also log the
        # min/max across all groups so the progress bar reflects the full schedule.
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            base_lr = group_lrs[0]
            min_lr = min(group_lrs)
            max_lr = max(group_lrs)
            self.log("train/lr", base_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_min", min_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_max", max_lr, prog_bar=False, on_step=True, on_epoch=False)
        if self._use_manual_optimization:
            # loss_for_backward is only None in the automatic-optimization branch above,
            # which is mutually exclusive with _use_manual_optimization.
            assert loss_for_backward is not None
            self.manual_backward(loss_for_backward)
            if self._should_step_optimizer(batch_idx):
                self._step_optimizer(optimizer)
        if self.train_config.compute_train_metrics:
            with torch.no_grad():
                orig_sizes = torch.stack([t["orig_size"] for t in targets])
                # Slice to group-0 queries only — mirrors the eval-mode path in
                # lwdetr.py that trims refpoint_embed to [:num_queries]. Without
                # this, training mode emits group_detr×num_queries queries (e.g.
                # 13×300=3900) and postprocess top-k selection draws from all
                # groups, producing OKS/mAP values ~50× below true accuracy.
                nq = self.model_config.num_queries
                # Only include tensor-valued keys — pred_masks is a dict in
                # train mode (sparse_forward) and postprocess cannot handle it.
                inference_outputs = {
                    k: v[:, :nq] if v.ndim >= 2 else v
                    for k, v in outputs.items()
                    if k in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints") and isinstance(v, Tensor)
                }
                results = self.postprocess(inference_outputs, orig_sizes)
            return {
                "loss": loss_for_return.detach() if self._use_manual_optimization else loss_for_return,
                "results": self._detach_results(results),
                "targets": targets,
            }
        return loss_for_return.detach() if self._use_manual_optimization else loss_for_return

    def _compute_train_losses(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """Compute normalized losses for logging and raw weighted loss for backward.

        Args:
            outputs: Model output dictionary.
            targets: Target dictionaries for the current batch.

        Returns:
            A tuple of normalized loss dictionary, unnormalized weighted loss numerator, and box normalizer.
        """
        weight_dict = self.criterion.weight_dict
        if not getattr(self.criterion, "supports_loss_normalizer_override", False):
            raise ValueError(
                f"{type(self.criterion).__name__}.supports_loss_normalizer_override is False; "
                "manual optimization (keypoint models) requires a criterion that accepts a "
                "num_boxes keyword argument. Set supports_loss_normalizer_override = True on "
                "your criterion subclass and implement the num_boxes parameter in forward()."
            )
        normalizer = self.criterion.num_boxes_for_targets(outputs, targets)
        numerator_loss_dict = self.criterion(outputs, targets, num_boxes=torch.ones_like(normalizer))
        # Keys in weight_dict are loss terms whose criterion implementation divides by num_boxes
        # (so passing num_boxes=1.0 yields raw numerators that we divide by normalizer here).
        # Keys outside weight_dict (e.g. "class_error", "cardinality_error") are diagnostics
        # that do NOT divide by num_boxes internally — they are passed through unchanged.
        # If a future loss term divides by num_boxes AND is omitted from weight_dict, its
        # logged value will be on a different scale than the keypoint path; verify when adding
        # new criterion terms.
        loss_dict = {
            key: value / normalizer if key in weight_dict else value for key, value in numerator_loss_dict.items()
        }
        raw_loss = sum(numerator_loss_dict[k] * weight_dict[k] for k in numerator_loss_dict if k in weight_dict)
        return loss_dict, raw_loss, normalizer

    def _scale_loss_for_accumulation(
        self,
        raw_loss: Tensor,
        normalizer: Tensor,
    ) -> Tensor:
        """Scale the current numerator loss by the accumulated box denominator.

        Args:
            raw_loss: Current microbatch weighted loss numerator.
            normalizer: Current microbatch box denominator.

        Returns:
            Loss scalar to pass to ``manual_backward``.
        """
        normalizer = normalizer.detach()
        previous_normalizer = self._accumulated_box_normalizer
        accumulated_normalizer = normalizer if previous_normalizer is None else previous_normalizer + normalizer
        if previous_normalizer is not None:
            self._rescale_accumulated_gradients(previous_normalizer / accumulated_normalizer)
        self._accumulated_box_normalizer = accumulated_normalizer.detach()
        return raw_loss / accumulated_normalizer

    def _rescale_accumulated_gradients(self, scale: Tensor) -> None:
        """Rescale gradients already accumulated in the current optimizer window.

        Args:
            scale: Multiplicative factor that converts previous gradients from the old denominator to the new one.
        """
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        """Return whether the current batch closes an optimizer accumulation window.

        The optimizer steps when either:

        - The current batch closes a complete ``grad_accum_steps`` window
          (``(batch_idx + 1) % grad_accum_steps == 0``), or
        - This is the final batch of the epoch and a partial accumulation window
          is still open, so the trailing microbatches are not silently dropped.

        Lightning's ``Trainer.num_training_batches`` may be reported as ``float('inf')``
        for iterable / streaming datasets where the epoch length is unknown. In that case
        only the modulo path can ever close the window — the final-batch fallback is
        skipped because ``batch_idx + 1`` can never reach infinity.

        Args:
            batch_idx: Batch index within the epoch.

        Returns:
            ``True`` when the optimizer should step after this batch.
        """
        accum_steps = max(1, int(self.train_config.grad_accum_steps))
        if (batch_idx + 1) % accum_steps == 0:
            return True
        num_training_batches = getattr(self.trainer, "num_training_batches", None)
        return (
            isinstance(num_training_batches, (int, float))
            and math.isfinite(num_training_batches)
            and batch_idx + 1 >= num_training_batches
        )

    def _step_optimizer(self, optimizer: torch.optim.Optimizer | LightningOptimizer) -> None:
        """Clip gradients, step optimizer and scheduler, then reset accumulation state.

        Args:
            optimizer: Optimizer returned by Lightning.
        """
        trainer_gradient_clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if trainer_gradient_clip_val is None:
            gradient_clip_val = self.train_config.clip_max_norm
        elif isinstance(trainer_gradient_clip_val, (int, float)):
            gradient_clip_val = trainer_gradient_clip_val
        else:
            gradient_clip_val = None
        gradient_clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", None)
        if not isinstance(gradient_clip_algorithm, str):
            gradient_clip_algorithm = None
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
        optimizer.step()
        optimizer.zero_grad()
        self._step_lr_scheduler()
        self._accumulated_box_normalizer = None

    def _current_lr_scheduler(self) -> LRScheduler | ReduceLROnPlateau | None:
        """Return the single configured LR scheduler, or ``None`` when none is available.

        Returns:
            The scheduler object (unwrapping Lightning's single-element list), or ``None`` when no
            scheduler is configured yet or Lightning is between fit stages.
        """
        try:
            scheduler = self.lr_schedulers()
        except (AttributeError, RuntimeError):
            return None
        if isinstance(scheduler, list):
            return scheduler[0] if scheduler else None
        return scheduler

    def _step_lr_scheduler(self) -> None:
        """Step step-interval schedulers once per optimizer step (manual-optimization path).

        Epoch-interval schedulers and metric-driven ``ReduceLROnPlateau`` are stepped at epoch boundaries by
        ``on_train_epoch_end`` / ``on_validation_epoch_end`` instead, so they are skipped here.
        """
        if self._lr_scheduler_interval != "step":
            return
        scheduler = self._current_lr_scheduler()
        if scheduler is None or isinstance(scheduler, ReduceLROnPlateau):
            return
        scheduler.step()

    def on_train_epoch_end(self) -> None:
        """Step epoch-interval (non-plateau) schedulers on the manual-optimization path.

        The automatic-optimization path leaves scheduler stepping entirely to Lightning; only the manual keypoint loop
        steps schedulers itself.
        """
        if self.automatic_optimization or self._lr_scheduler_interval != "epoch":
            return
        scheduler = self._current_lr_scheduler()
        if scheduler is None or isinstance(scheduler, ReduceLROnPlateau):
            return
        scheduler.step()

    def on_validation_epoch_end(self) -> None:
        """Step ``ReduceLROnPlateau`` from the monitored metric on the manual-optimization path.

        The automatic-optimization path lets Lightning feed the monitored metric; the manual keypoint loop must read it
        from ``trainer.callback_metrics`` and step the scheduler itself. The pre-training sanity-check validation is
        skipped so plateau patience/cooldown bookkeeping is not seeded from the untrained model.
        """
        if self.automatic_optimization or self.trainer.sanity_checking:
            return
        scheduler = self._current_lr_scheduler()
        if not isinstance(scheduler, ReduceLROnPlateau):
            return
        monitor = self._lr_scheduler_monitor or "val/loss"
        metric = self.trainer.callback_metrics.get(monitor)
        if metric is None:
            # Warn-and-continue would let the LR never reduce while training silently proceeds. Fail loud instead,
            # mirroring Lightning's strict-monitor behavior on the automatic-optimization path.
            raise RuntimeError(
                f"ReduceLROnPlateau monitor {monitor!r} was not found in callback_metrics, so the learning rate "
                "would never be reduced. Ensure the monitored metric is logged every validation epoch (e.g. set "
                "compute_val_loss=True for the default 'val/loss' monitor), or set lr_scheduler_monitor to a metric "
                "that is produced."
            )
        scheduler.step(metric)

    @staticmethod
    def _detach_results(results: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        """Detach postprocessed result tensors before handing them to callbacks.

        Args:
            results: Per-image postprocessed prediction dictionaries.

        Returns:
            Per-image dictionaries with tensor values detached from the graph.
        """
        return [
            {key: value.detach() if torch.is_tensor(value) else value for key, value in result.items()}
            for result in results
        ]

    def _log_train_progress_metrics(
        self,
        loss: Tensor,
        loss_dict: dict[str, Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log compact per-step convergence metrics for the progress bar only.

        Args:
            loss: Unscaled aggregate training loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        # When ``train_log_on_step`` is True, the ``train/loss`` call in ``training_step``
        # logs with ``on_step=True, on_epoch=True``; Lightning forks that into
        # ``train/loss_step`` + ``train/loss_epoch``, and ``train/loss_step`` already
        # provides the live per-step progress-bar view. Emitting this separate ``loss``
        # scalar in that case just duplicates it, so only log it on the default
        # ``train_log_on_step=False`` path.
        if not bool(self.train_config.train_log_on_step):
            self.log(
                "loss",
                loss,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        for loss_name, progress_name in _TRAIN_PROGRESS_LOSS_ALIASES.items():
            value = loss_dict.get(loss_name)
            if value is None:
                continue
            self.log(
                progress_name,
                value,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

    def _log_val_loss_metrics(
        self,
        loss: Tensor,
        loss_dict: dict[str, Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log aggregate and component validation losses.

        Args:
            loss: Aggregate weighted validation loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _resolve_eval_model(self) -> Any:
        """Return the model to forward through for validation.

        When ``TrainConfig.eval_ema_only`` is set, validation forwards through the EMA-averaged
        weights directly instead of the base model — this replaces the second, duplicate
        base+EMA forward pass ``COCOEvalCallback`` would otherwise also run every validation
        batch (see issue 416). Falls back to the base model when EMA is enabled but not yet
        warmed up (e.g. the very first validation epoch, before ``RFDETREMACallback.setup``
        has built its averaged model), or when this module isn't attached to a ``Trainer`` at
        all (``LightningModule.trainer`` raises ``RuntimeError`` rather than returning ``None``
        when unattached — e.g. ``validation_step`` called directly outside ``Trainer.fit``/
        ``Trainer.validate``).

        Uses the same ``_get_ema_inner_module`` helper as ``COCOEvalCallback`` (see
        ``coco_eval.py``) so both consumers resolve the EMA-averaged detection net through one
        shared code path instead of independently duck-typing ``RFDETREMACallback``.

        Returns:
            The base model, or the EMA-averaged inner module's underlying detection net when
            ``eval_ema_only`` is active and available.
        """
        if not self.train_config.eval_ema_only:
            return self.model
        try:
            callbacks = getattr(self.trainer, "callbacks", [])
        except RuntimeError:
            return self.model
        for callback in callbacks:
            ema_inner = _get_ema_inner_module(callback)
            if ema_inner is not None:
                return ema_inner.model
        return self.model

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one validation step.

        Returns raw results and targets so ``COCOEvalCallback`` can accumulate them across the epoch via
        ``on_validation_batch_end``.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the validation epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self._resolve_eval_model()(samples)
        if self.train_config.compute_val_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self._log_val_loss_metrics(loss, loss_dict, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    @property
    def _fused_adamw_env_eligible(self) -> bool:
        """Return whether the runtime would enable fused AdamW, ignoring optimizer choice.

        Captures only the hardware/precision preconditions (BF16 on CUDA), so the
        custom-optimizer path can tell whether a dropped ``fused_optimizer=True``
        would actually have mattered.

        Returns:
            ``True`` when fused AdamW is requested and the runtime supports it.
        """
        return (
            self.model_config.fused_optimizer
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and str(self.trainer.precision) in {"bf16-mixed", "bf16", "bf16-true"}
        )

    @property
    def _use_fused_optimizer(self) -> bool:
        """Return whether fused AdamW should be used for the current training configuration.

        Fused AdamW is only safe when the trainer's actual precision is a BF16 variant.  Checking GPU capability alone
        (``is_bf16_supported()``) is
        insufficient: on Ampere+ hardware that flag is always ``True`` even when
        the trainer is configured for ``32-true``, which causes a ``params, grads, exp_avgs, and exp_avg_sqs must have
        same dtype, device, and layout`` crash in DDP because gradient bucket views have non-matching strides in FP32.
        It additionally requires the built-in ``optimizer="adamw"`` selection: fused state normalization and gradient
        clipping are AdamW-specific and must not fire for custom optimizers.

        Returns:
            ``True`` when fused AdamW is requested, safe, and the built-in AdamW optimizer is selected.

        Examples:
            >>> from unittest.mock import patch
            >>> module = RFDETRModelModule.__new__(RFDETRModelModule)
            >>> module.model_config = type("Cfg", (), {"fused_optimizer": True})()
            >>> with patch("torch.cuda.is_available", return_value=False):
            ...     module._use_fused_optimizer
            False
        """
        if not self._fused_adamw_env_eligible:
            return False
        return _is_builtin_fused_adamw(self.train_config.optimizer)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Build the configured optimizer with layer-wise LR decay and scheduler.

        Uses ``trainer.estimated_stepping_batches`` for total step count so cosine annealing covers the full training
        run regardless of dataset size or accumulation settings.
        ``optimizer="adamw"`` keeps RF-DETR's fused torch AdamW path;
        other names can be loaded from ``pytorch-optimizer``.

        Returns:
            PTL optimizer config dict with optimizer and step-interval scheduler.
        """
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        # Unwrap torch.compile's OptimizedModule so get_param_dict sees the
        # original module's named_parameters() — compiled wrapper can cause
        # name-prefix mismatches that put the same tensor in multiple groups.
        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        param_dicts = [param_group for param_group in param_dicts if param_group["params"].requires_grad]

        optimizer_cfg = tc.optimizer
        optimizer: torch.optim.Optimizer
        if _is_builtin_fused_adamw(optimizer_cfg):
            # Built-in managed fused AdamW path (unchanged behavior).
            try:
                optimizer = torch.optim.AdamW(
                    param_dicts,
                    lr=tc.lr,
                    weight_decay=tc.weight_decay,
                    fused=self._use_fused_optimizer,
                    **tc.optimizer_kwargs,
                )
            except TypeError as exc:
                raise TypeError(
                    f"Failed to initialize optimizer 'adamw': {exc}. "
                    "Check optimizer_kwargs for arguments supported by torch.optim.AdamW."
                ) from exc
        else:
            if self._fused_adamw_env_eligible:
                logger.warning(_FUSED_IGNORED_MSG, optimizer_cfg)
            if not isinstance(optimizer_cfg, str):
                # Explicit callable / functools.partial: called with param groups only.
                callable_name = getattr(optimizer_cfg, "__qualname__", None) or repr(optimizer_cfg)
                optimizer = _instantiate_explicit_optimizer(optimizer_cfg, callable_name, param_dicts, {})
            elif _is_managed_optimizer_name(optimizer_cfg):
                # Managed native torch.optim short name (lr + signature-aware weight_decay injected).
                native_class: _OptimizerFactory = _resolve_native_optimizer(optimizer_cfg)
                optimizer = _instantiate_optimizer(native_class, optimizer_cfg, param_dicts, tc)
            else:
                # Explicit dotted import path: constructed from optimizer_kwargs only.
                optimizer_class = _import_optimizer_class(optimizer_cfg)
                optimizer = _instantiate_explicit_optimizer(
                    optimizer_class, optimizer_cfg, param_dicts, tc.optimizer_kwargs
                )

        # ``trainer.estimated_stepping_batches`` is reported in *microbatch* units when
        # the keypoint path runs with ``Trainer(accumulate_grad_batches=1)`` and manages
        # accumulation manually. ``LambdaLR.step()`` is called once per optimizer-step
        # (i.e. every ``grad_accum_steps`` microbatches), so the schedule must be sized
        # in optimizer-step units rather than microbatches; otherwise warmup and cosine
        # decay finish ``grad_accum_steps``× too early. Detection / segmentation models
        # still rely on Lightning's automatic optimization, where PTL already accounts
        # for ``accumulate_grad_batches`` inside ``estimated_stepping_batches`` and the
        # division below is a no-op (``grad_accum_steps`` would be 1 in that path).
        grad_accum_steps = max(1, int(tc.grad_accum_steps))
        microbatches = int(self.trainer.estimated_stepping_batches)
        # _should_step_optimizer steps the final partial window at epoch end, so the true
        # number of optimizer steps is ceil(microbatches / grad_accum_steps).  Using floor
        # would undercount when the epoch is not evenly divisible, causing warmup / cosine
        # schedules to finish one step earlier than the last actual step fires.
        total_steps = (
            max(1, math.ceil(microbatches / grad_accum_steps)) if self._use_manual_optimization else microbatches
        )
        steps_per_epoch = max(1, total_steps // tc.epochs)
        warmup_steps = int(steps_per_epoch * tc.warmup_epochs)

        scheduler_cfg = tc.lr_scheduler
        scheduler: LRScheduler | ReduceLROnPlateau
        interval = "step"
        monitor: str | None = None
        if _is_managed_scheduler_name(scheduler_cfg):
            # Managed "step" / "cosine" preset — warmup + total-step sizing baked into a LambdaLR (unchanged behavior).
            scheduler = _build_managed_scheduler(optimizer, tc, total_steps, steps_per_epoch, warmup_steps)
        else:
            if not isinstance(scheduler_cfg, str):
                # Explicit callable / functools.partial: built from the optimizer only (kwargs baked in).
                scheduler_name = getattr(scheduler_cfg, "__qualname__", None) or repr(scheduler_cfg)
                scheduler = _instantiate_explicit_scheduler(scheduler_cfg, scheduler_name, optimizer, {})
            else:
                # Explicit dotted import path: constructed from lr_scheduler_kwargs only.
                scheduler_class = _import_scheduler_class(scheduler_cfg)
                scheduler = _instantiate_explicit_scheduler(
                    scheduler_class, scheduler_cfg, optimizer, tc.lr_scheduler_kwargs
                )
            interval = tc.lr_scheduler_interval
            if isinstance(scheduler, ReduceLROnPlateau):
                monitor = tc.lr_scheduler_monitor
                # The monitored metric (e.g. val/loss) is only available per epoch, so plateau always steps
                # on the epoch boundary regardless of the configured interval.
                interval = "epoch"
                if warmup_steps > 0:
                    logger.warning(
                        "warmup_epochs=%s is ignored for ReduceLROnPlateau; a metric-driven scheduler cannot "
                        "be composed with a linear warmup ramp.",
                        tc.warmup_epochs,
                    )
            else:
                # Auto-wrap explicit schedulers with a linear warmup ramp. The wrap is stepped at the same cadence as
                # the scheduler, so size it in the scheduler's own units: optimizer steps for "step", epochs for "epoch"
                # (otherwise a step-sized ramp stepped once per epoch would stretch across the whole run).
                if interval == "step":
                    warmup_units = warmup_steps
                else:
                    # Epoch cadence: the ramp is stepped once per epoch. ceil keeps a fractional warmup_epochs from
                    # truncating to zero (a silently dropped warmup). A single-epoch ramp has start_factor == 1.0,
                    # i.e. a flat no-op that looks like warmup but isn't, so a gradual epoch-granular warmup needs
                    # >= 2 epochs; warn and skip rather than emit a degenerate ramp.
                    warmup_units = math.ceil(tc.warmup_epochs)
                    if tc.warmup_epochs > 0 and warmup_units < 2:
                        logger.warning(
                            "warmup_epochs=%s with lr_scheduler_interval='epoch' cannot form a gradual warmup ramp "
                            "(epoch-granular warmup needs >= 2 epochs); skipping warmup. Use "
                            "lr_scheduler_interval='step' for sub-epoch warmup, or set warmup_epochs >= 2.",
                            tc.warmup_epochs,
                        )
                        warmup_units = 0
                if warmup_units > 0:
                    scheduler = _wrap_with_warmup(scheduler, optimizer, warmup_units)

        self._lr_scheduler_interval = interval
        self._lr_scheduler_monitor = monitor

        lr_scheduler_config: LRSchedulerConfigType = {"scheduler": scheduler, "interval": interval}
        if monitor is not None:
            lr_scheduler_config["monitor"] = monitor
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Override PTL gradient clipping to support fused AdamW.

        PTL's AMP precision plugin refuses to clip gradients when the optimizer declares it handles unscaling internally
        (fused=True).  When fused is active we are on BF16 (no GradScaler) so ``clip_grad_norm_`` is correct.  For the
        non-fused path (FP16 + GradScaler or FP32) we delegate to ``super()`` to preserve scaler-aware unscaling.

        Args:
            optimizer: The current optimizer.
            gradient_clip_val: Maximum gradient norm.
            gradient_clip_algorithm: Clipping algorithm; forwarded to super()
                for the non-fused path.
        """
        if self._use_fused_optimizer:
            if gradient_clip_val and gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        else:
            # PTL's own type stub only declares Optimizer here, but LightningOptimizer dynamically
            # multiply-inherits from the wrapped optimizer's class, so it satisfies this at runtime too.
            super().clip_gradients(
                optimizer,  # type: ignore[arg-type]
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    @staticmethod
    def _normalize_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
        """Cast restored floating-point optimizer state tensors to match live parameters.

        Args:
            optimizer: AdamW optimizer whose state may have been rehydrated with a mismatched dtype or layout.

        Returns:
            Number of state tensors that were reallocated to match the current parameter layout.
        """
        normalized = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param)
                if not state:
                    continue
                for key, value in list(state.items()):
                    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                        continue
                    if value.shape != param.shape:
                        continue
                    if value.device == param.device and value.dtype == param.dtype and value.stride() == param.stride():
                        continue
                    restored = torch.empty_like(param)
                    restored.copy_(value.to(device=param.device, dtype=param.dtype))
                    state[key] = restored
                    normalized += 1
        return normalized

    def test_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one test step.

        Mirrors :meth:`validation_step` so ``COCOEvalCallback`` can accumulate results via ``on_test_batch_end`` when
        ``trainer.test()`` is called (e.g. from :class:`~rfdetr.training.callbacks.BestModelCallback` at end of
        training).

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the test epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_test_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("test/loss", loss, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def predict_step(self, batch: tuple[Any, Any], batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run inference on a preprocessed batch and return postprocessed results.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index.
            dataloader_idx: Index of the predict dataloader.

        Returns:
            Postprocessed detection results from ``PostProcess``.
        """
        samples, targets = batch
        with torch.no_grad():
            outputs = self.model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        return self.postprocess(outputs, orig_sizes)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Auto-detect legacy formats and reconcile PE shapes at checkpoint load time.

        PTL calls this hook before applying ``checkpoint["state_dict"]`` to the module.  Three normalisation steps are
        applied in order:

        1. **Raw legacy format** — a ``*.pth`` file loaded directly by
           ``Trainer`` (e.g. via ``ckpt_path=``).  Recognised by the presence of ``"model"`` without ``"state_dict"``.
           The state dict is rewritten in-place with the ``"model."`` prefix so PTL can apply it normally.

        2. **Positional-embedding interpolation** — when the checkpoint was
           saved at a different image resolution than the current model, the DINOv2 ``position_embeddings`` tensor shape
           will mismatch. :func:`~rfdetr.models.weights.interpolate_position_embeddings` is called to bicubic-resize the
           PE to ``model_config.positional_encoding_size`` before PTL applies the state dict.  Regression fix for
           :issue:`998`.

        3. **Converted format** — a file produced by
           :func:`~rfdetr.training.checkpoint.convert_legacy_checkpoint` that already has ``"state_dict"`` but also
           carries ``"legacy_ema_state_dict"``.  The EMA weights are stashed on ``self._pending_legacy_ema_state`` for
           optional restoration by :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback`.

        Note:
            This hook only fires on ``Trainer(ckpt_path=...)`` resume paths. Fresh-train bootstrap from a
            ``pretrain_weights`` checkpoint runs through :func:`~rfdetr.models.weights.load_pretrain_weights` during
            ``__init__`` instead — that helper performs its own PTL ``.ckpt`` normalisation (``state_dict`` → ``model``
            key, ``_orig_mod`` strip) and PE interpolation, so the two code paths intentionally do not share state.

        Args:
            checkpoint: Checkpoint dict passed in by PTL (mutated in-place).
        """
        # Raw legacy .pth: no "state_dict" key — build it from "model".
        if "model" in checkpoint and "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {"model." + k: v for k, v in checkpoint["model"].items()}

        # Interpolate DINOv2 positional embeddings when the checkpoint was saved
        # at a different resolution than the current model.  PTL applies
        # checkpoint["state_dict"] immediately after this hook, so the shapes
        # must already match at this point.  Regression: #998.
        if "state_dict" in checkpoint:
            interpolate_position_embeddings(
                checkpoint["state_dict"],
                self.model_config.positional_encoding_size,
            )

        # Stash legacy EMA weights for RFDETREMACallback.setup(), which restores
        # them into AveragedModel when resuming from converted legacy checkpoints.
        if "legacy_ema_state_dict" in checkpoint:
            self._pending_legacy_ema_state = checkpoint["legacy_ema_state_dict"]
            warnings.warn(
                "Checkpoint contains legacy EMA weights (`legacy_ema_state_dict`). "
                "Add RFDETREMACallback to your trainer callbacks to restore them; "
                "without it the stashed weights will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a new class count.

        Args:
            num_classes: New number of classes (excluding background).
        """
        self.model.reinitialize_detection_head(num_classes)
