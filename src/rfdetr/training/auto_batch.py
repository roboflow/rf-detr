# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Auto-batch probing: find a safe micro-batch size before training.

Probe assumptions (worst-case so training does not OOM):
- Resolution: When multi_scale is True we use the maximum of the multi-scale
  augmentation scales (same as compute_multi_scale_scales). Otherwise we use model resolution. This ensures the step
  uses the max resolution seen in training.
- Targets: Memory grows with number of targets per image. We use
  auto_batch_max_targets_per_image (config) to synthesize that many targets per image so the probe reflects worst-case
  matcher and loss memory.
- EMA: When use_ema is True, an EMA copy of the model is kept in memory. We
  apply auto_batch_ema_headroom (e.g. 0.7) to the probed batch size so the effective safe batch leaves room for the EMA
  model.
- Optimizer state: AdamW allocates exp_avg/exp_avg_sq lazily, on its first
  step() rather than at construction, each the same size as the trainable parameters -- so the combined state is 2x
  that size, 3x with amsgrad. The probe runs a shadow AdamW step (see _build_shadow_optimizer) so that memory is part
  of the estimate instead of first appearing on training's own first optimizer step. Only AdamW-shaped optimizers are
  modeled this way (see _is_adamw_shaped); anything else gets the runtime warning in resolve_auto_batch_config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.models import build_criterion_from_config
from rfdetr.training.module_model import _is_builtin_fused_adamw
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()


@dataclass(frozen=True)
class AutoBatchResult:
    """Result of auto-batch probing: safe micro-batch size and recommended grad accumulation.

    Attributes:
        safe_micro_batch: Per-device batch size that fits in memory for one train step.
        recommended_grad_accum_steps: Steps to accumulate to reach target effective batch.
        effective_batch_size: safe_micro_batch * recommended_grad_accum_steps.
        device_name: Human-readable GPU name used for probing.
    """

    safe_micro_batch: int
    recommended_grad_accum_steps: int
    effective_batch_size: int
    device_name: str


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _make_synthetic_batch(
    micro_batch_size: int,
    resolution: int,
    device: torch.device,
    num_classes: int,
    segmentation_head: bool = False,
    max_targets_per_image: int = 1,
    num_channels: int = 3,
) -> tuple[NestedTensor, list[dict[str, torch.Tensor]]]:
    """Build a minimal (samples, targets) batch for probing.

    Uses max_targets_per_image targets per image so memory reflects worst-case matcher and loss. When segmentation_head
    is True, each target dict includes "masks" of shape (max_targets_per_image, resolution, resolution).
    """
    tensors = torch.randn(micro_batch_size, num_channels, resolution, resolution, device=device)
    mask = torch.zeros(micro_batch_size, resolution, resolution, dtype=torch.bool, device=device)
    samples = NestedTensor(tensors, mask)

    max_label = max(0, num_classes - 1)
    n = max(1, max_targets_per_image)
    targets: list[dict[str, torch.Tensor]] = []
    for idx in range(micro_batch_size):
        # Replicate one box/label n times so matcher and loss see n targets per image.
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=device).expand(n, 4)
        labels = torch.tensor([min(1, max_label)], dtype=torch.int64, device=device).expand(n)
        iscrowd = torch.zeros(n, dtype=torch.int64, device=device)
        area = torch.full((n,), 0.04, dtype=torch.float32, device=device)
        t: dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx, dtype=torch.int64, device=device),
            "orig_size": torch.tensor([resolution, resolution], dtype=torch.int64, device=device),
            "size": torch.tensor([resolution, resolution], dtype=torch.int64, device=device),
            "iscrowd": iscrowd,
            "area": area,
        }
        if segmentation_head:
            t["masks"] = torch.zeros(n, resolution, resolution, dtype=torch.bool, device=device)
        targets.append(t)
    return samples, targets


def _is_adamw_shaped(optimizer_cfg: object) -> bool:
    """Return whether a configured optimizer allocates the same per-parameter state as ``torch.optim.AdamW``.

    Deliberately broader than ``module_model._is_builtin_fused_adamw``, which answers a different question (does this
    config select RF-DETR's managed fused-kernel path?) and must stay narrow. For a memory estimate the only thing that
    matters is whether the real optimizer keeps ``exp_avg``/``exp_avg_sq`` per parameter, which a dotted path importing
    ``torch.optim.AdamW`` does just as much as the built-in ``"adamw"`` short name.

    Args:
        optimizer_cfg: The ``TrainConfig.optimizer`` value; a callable's state layout is unknowable here, so it
            reports ``False``.

    Returns:
        ``True`` for the ``"adamw"`` short name (case-insensitive) and for a dotted path naming ``torch.optim.AdamW``.

    Examples:
        >>> _is_adamw_shaped("AdamW")
        True
        >>> _is_adamw_shaped("torch.optim.AdamW")
        True
        >>> _is_adamw_shaped("torch.optim.SGD")
        False
    """
    if not isinstance(optimizer_cfg, str):
        return False
    name = optimizer_cfg.strip()
    if "." in name:
        # A dotted value is an import path, so it has to match one exactly: capitalization is part of the attribute
        # name, and "torch.optim.adamw" alone names the module rather than the class.
        return name in {"torch.optim.AdamW", "torch.optim.adamw.AdamW"}
    return name.lower() == "adamw"


def _build_shadow_optimizer(
    trainable_params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    optimizer_kwargs: dict[str, Any] | None = None,
    fused: bool | None = None,
) -> tuple[torch.optim.Optimizer, list[torch.Tensor]]:
    """Build an AdamW optimizer over throwaway tensors shaped like ``trainable_params``, for probing optimizer-state
    memory without ever writing to the real model's weights.

    ``optimizer.step()`` needs to run against *some* set of parameters to allocate
    AdamW's per-parameter state (``exp_avg``/``exp_avg_sq``), but running it against the real model's own
    parameters would apply a real (garbage, synthetic-data-derived) gradient update to weights that
    may already hold a loaded pretrained checkpoint -- corrupting them before training even starts.
    The shadow tensors this returns are separate GPU allocations of the same shape/dtype, so the
    optimizer state they accumulate costs the same memory as the real optimizer eventually would,
    without the update ever touching a real parameter.

    That separateness is also what this design costs: the shadow copies are allocated alongside the
    model's own already-resident parameters, so the probe holds one extra copy of the trainable
    parameters that real training never holds. On the non-fused path (amp off, fp16, or
    ``fused_optimizer=False``) there is a second, transient divergence: the shadow optimizer steps a
    single parameter group, so ``_foreach_sqrt`` builds one whole-model-sized temporary, where real
    training's per-layer-group parameter dicts (``get_param_dict``, ``module_model.py``) keep that
    temporary per-group. Both make the probe stricter than the run it sizes rather than more
    permissive, and the transient one does not arise at all whenever the probe runs fused -- the
    common bf16-on-CUDA default.

    Args:
        trainable_params: The real model parameters being probed for (only used for shape/dtype/device).
        lr: Learning rate for the shadow optimizer (does not affect state tensor size).
        weight_decay: Weight decay for the shadow optimizer (does not affect state tensor size).
        optimizer_kwargs: Extra AdamW kwargs from ``train_config.optimizer_kwargs`` (e.g. ``amsgrad=True``,
            which adds a third ``max_exp_avg_sq`` state buffer). Only meaningful when the real training run
            uses an AdamW-shaped optimizer (see ``_is_adamw_shaped``) -- pass ``None`` otherwise, since these
            kwargs are AdamW-specific and would raise for a different optimizer class. Keys given here override
            ``lr``/``weight_decay``/``fused`` below: on the dotted-path config (``"torch.optim.AdamW"``)
            ``configure_optimizers`` builds the real optimizer from this dict alone, injecting no lr or
            weight_decay of its own (``module_model.py``), so this dict is where that run's real values live.
        fused: Whether to build a fused AdamW, matching ``RFDETRModelModule._use_fused_optimizer`` for the real
            optimizer (``module_model.py``). The fused and foreach/single-tensor implementations differ in
            step-time temporary memory, so leaving this unset (``None``, torch's own default selection) can
            make the probe's memory profile diverge from the real optimizer's -- in either direction.

    Returns:
        The shadow optimizer and the list of shadow tensors it steps, in the same order as
        ``trainable_params`` so gradients can be paired up by index before each ``step()``.

    Raises:
        TypeError: If ``optimizer_kwargs`` carries an argument ``torch.optim.AdamW`` does not accept.
    """
    # No requires_grad: nothing here is ever backpropagated through. torch.optim only asks its params to be
    # leaves (zeros_like already returns one) and step() runs under no_grad, so the flag would buy nothing.
    shadow_params = [torch.zeros_like(p) for p in trainable_params]
    # Merge rather than splat both sets side by side: a dotted-path config keeps its lr/weight_decay/fused inside
    # optimizer_kwargs (see the Args note above), so passing them separately would raise "got multiple values for
    # keyword argument". The caller's dict wins, since that is what the real optimizer will be built with.
    adamw_kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, "fused": fused}
    adamw_kwargs.update(optimizer_kwargs or {})
    try:
        optimizer = torch.optim.AdamW(shadow_params, **adamw_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to initialize the auto-batch probe's shadow 'adamw' optimizer: {exc}. "
            "Check optimizer_kwargs for arguments supported by torch.optim.AdamW."
        ) from exc
    return optimizer, shadow_params


def _probe_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    micro_batch_size: int,
    resolution: int,
    device: torch.device,
    num_classes: int,
    amp: bool,
    *,
    trainable_params: list[torch.nn.Parameter],
    shadow_optimizer: torch.optim.Optimizer,
    shadow_params: list[torch.Tensor],
    segmentation_head: bool = False,
    max_targets_per_image: int = 1,
    num_channels: int = 3,
    autocast_dtype: torch.dtype | None = None,
    warn_missing_grads: bool = False,
) -> bool:
    """Run one forward + loss + backward + shadow optimizer step; return True if successful, False on OOM.

    The shadow step matters for the memory estimate, not just for realism: AdamW (and most stateful optimizers) allocate
    their per-parameter state (``exp_avg``/``exp_avg_sq``, each the same size as the parameter itself) lazily, on the
    first ``step()`` call -- forward+backward alone never touches that allocation.
    ``shadow_optimizer``/``shadow_params`` are built once by the caller and reused across every probe iteration, so this
    first call is the only one that pays the allocation; every later iteration (larger candidate batch sizes) competes
    for GPU memory against that already-resident state, the same way a real second-and-later training step would. The
    shadow tensors, not the real model parameters, receive the update, so this never mutates the model being probed.

    ``warn_missing_grads`` asks this step to report parameters the synthetic batch never reached: AdamW allocates state
    only for parameters that carry a gradient, so those are absent from the estimate. The caller sets it for the first
    step only (every later iteration would recount the same parameters), and the report is emitted only if that step
    goes on to succeed.

    Args:
        model: The model to probe; already in train mode, and left untouched apart from its gradients.
        criterion: The loss criterion (must match model output and target format).
        micro_batch_size: Per-device batch size to attempt in this single step.
        resolution: Input spatial size (square) for the synthetic images.
        device: CUDA device the synthetic batch is allocated on.
        num_classes: Number of classes, used to pick synthetic target labels.
        amp: Whether to run the forward under autocast.
        trainable_params: The model's ``requires_grad`` parameters, in the order ``shadow_params`` was built from --
            index i of one must correspond to index i of the other, which the aliasing loop asserts.
        shadow_optimizer: The AdamW built over ``shadow_params``; stepped here so its state is allocated and stays
            resident for later iterations.
        shadow_params: Throwaway tensors mirroring ``trainable_params``, which take the update in their place.
        segmentation_head: If True, synthetic targets include "masks" so loss_masks is exercised.
        max_targets_per_image: Number of synthetic targets per image (worst-case matcher and loss memory).
        num_channels: Number of input image channels for the synthetic images.
        autocast_dtype: Explicit autocast dtype; ``None`` leaves torch's own default for the device.
        warn_missing_grads: Whether to report parameters that received no gradient (see above). Off by default so
            only the caller's chosen step reports.

    Returns:
        True if the step ran to completion, False if it hit a CUDA OOM (the signal that this ``micro_batch_size``
        does not fit). Any other RuntimeError propagates.
    """
    try:
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        samples, targets = _make_synthetic_batch(
            micro_batch_size=micro_batch_size,
            resolution=resolution,
            device=device,
            num_classes=num_classes,
            segmentation_head=segmentation_head,
            max_targets_per_image=max_targets_per_image,
            num_channels=num_channels,
        )

        autocast_kwargs: dict[str, Any] = {"device_type": "cuda", "enabled": amp}
        if autocast_dtype is not None:
            autocast_kwargs["dtype"] = autocast_dtype
        with torch.autocast(**autocast_kwargs):
            outputs = model(samples, targets)
            loss_dict = cast("dict[str, torch.Tensor]", criterion(outputs, targets))
            weight_dict = cast("dict[str, float]", criterion.weight_dict)
            weighted_losses = [loss_dict[name] * weight_dict[name] for name in loss_dict if name in weight_dict]
            if not weighted_losses:
                raise RuntimeError(
                    "auto-batch probe could not build weighted losses: no overlap between criterion loss_dict and "
                    "weight_dict keys.",
                )
            loss = torch.stack(weighted_losses).sum()

        if not torch.isfinite(loss):
            raise RuntimeError("auto-batch probe produced a non-finite training loss.")

        torch.autograd.backward(loss)
        # Alias (not copy) each real parameter's freshly populated .grad onto its shadow counterpart:
        # AdamW.step() only reads .grad, so sharing the tensor is enough to drive a real state update
        # without allocating a second copy of every gradient.
        for shadow_p, real_p in zip(shadow_params, trainable_params, strict=True):
            shadow_p.grad = real_p.grad
        missing_grads = sum(1 for real_p in trainable_params if real_p.grad is None) if warn_missing_grads else 0
        try:
            shadow_optimizer.step()
        finally:
            # Drop the alias whether step() succeeded or raised (e.g. OOM inside AdamW's own state
            # allocation): shadow_p.grad is the only remaining reference to real_p's gradient tensor
            # once model.zero_grad() below clears real_p.grad, so leaving it set on an OOM would keep
            # that tensor resident -- unreachable by empty_cache() -- through every later probe
            # iteration for the rest of this search, not just the next one.
            for shadow_p in shadow_params:
                shadow_p.grad = None
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        if missing_grads:
            logger.warning(
                "[auto-batch] %d/%d trainable params received no gradient from the synthetic probe batch; their "
                "optimizer state is not modeled, so the returned batch size may be too permissive.",
                missing_grads,
                len(trainable_params),
            )
        return True
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            torch.cuda.empty_cache()
            return False
        raise


def probe_max_micro_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    resolution: int,
    device: torch.device,
    num_classes: int,
    amp: bool,
    segmentation_head: bool = False,
    max_targets_per_image: int = 1,
    safety_margin: float = 0.9,
    max_micro_batch: int = 128,
    num_channels: int = 3,
    autocast_dtype: torch.dtype | None = None,
    optimizer_lr: float = 1e-4,
    optimizer_weight_decay: float = 1e-4,
    optimizer_kwargs: dict[str, Any] | None = None,
    optimizer_fused: bool | None = None,
) -> int:
    """Find the largest per-device batch size that fits in memory for one train step.

    Uses exponential search (1, 2, 4, ...) up to the first failure, then binary search between the last successful size
    and the first failure to get the exact maximum. The returned value is floor(max_ok * safety_margin), so
    safety_margin in (0, 1] scales down the result for headroom (e.g. 0.9 keeps 10% margin).

    Every probe iteration also runs a shadow AdamW step (see ``_build_shadow_optimizer``): forward and
    backward alone never trigger the ``exp_avg``/``exp_avg_sq`` state AdamW allocates lazily on its
    first real ``step()`` -- each buffer the same size as the trainable parameters themselves, so the
    combined state is 2x that size (3x with ``amsgrad``) -- so a probe that skipped this would report a
    batch size that fits forward+backward but can still OOM training on its first real optimizer step.

    Args:
        model: The model to probe (will be set to train mode).
        criterion: The loss criterion (must match model output and target format).
        resolution: Input spatial size (square).
        device: CUDA device to run on.
        num_classes: Number of classes (for synthetic targets).
        amp: Whether to use autocast for the forward.
        segmentation_head: If True, synthetic targets include "masks" for loss_masks.
        max_targets_per_image: Number of synthetic targets per image (worst-case memory).
        safety_margin: Fraction of max batch to return (0 < safety_margin <= 1).
        max_micro_batch: Cap on batch size to try.
        num_channels: Number of input image channels (for synthetic probe images).
        optimizer_lr: Learning rate for the shadow AdamW optimizer used to size its state (does not
            affect the state tensors' shape/size, only included so the shadow optimizer is
            constructible without a training config on hand). Not derived from ``train_config``:
            ``resolve_auto_batch_config`` leaves this and ``optimizer_weight_decay`` at their defaults,
            since neither changes the memory the probe measures.
        optimizer_weight_decay: Weight decay for the shadow AdamW optimizer (same caveat as ``optimizer_lr``).
        optimizer_kwargs: Extra AdamW kwargs from ``train_config.optimizer_kwargs`` (e.g. ``amsgrad=True``).
            Pass only when the real run uses the built-in ``"adamw"`` path -- see ``_build_shadow_optimizer``.
        optimizer_fused: Whether the real training run will use fused AdamW (mirrors
            ``RFDETRModelModule._use_fused_optimizer``). Forwarded to ``_build_shadow_optimizer`` so the shadow
            optimizer's step-time temporary memory matches the real one's, since fused and foreach/single-tensor
            AdamW allocate different amounts of scratch memory during ``step()``.

    Returns:
        Safe micro-batch size (>= 1).

    Raises:
        RuntimeError: If device is not CUDA or if micro_batch_size=1 already fails (OOM).
        ValueError: If max_micro_batch < 1 or safety_margin not in (0, 1].
    """
    if device.type != "cuda":
        raise RuntimeError("auto-batch probing currently supports CUDA only.")
    if max_micro_batch < 1:
        raise ValueError("max_micro_batch must be >= 1.")
    if not (0 < safety_margin <= 1.0):
        raise ValueError("safety_margin must be in (0, 1].")

    model_training = model.training
    criterion_training = criterion.training

    shadow_optimizer: torch.optim.Optimizer | None = None
    shadow_params: list[torch.Tensor] | None = None
    try:
        model.train()
        criterion.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "auto-batch probing requires at least one trainable parameter; every model parameter has "
                "requires_grad=False."
            )
        try:
            shadow_optimizer, shadow_params = _build_shadow_optimizer(
                trainable_params, optimizer_lr, optimizer_weight_decay, optimizer_kwargs, optimizer_fused
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            # Without this the first thing a user sees is a bare CUDA OOM from inside torch.zeros_like,
            # with nothing tying it to auto-batch probing or naming a lever to pull.
            raise RuntimeError(
                "auto-batch probe ran out of memory allocating the shadow optimizer's parameter copies, before "
                "any micro_batch_size was tried; that allocation needs one extra copy of the trainable "
                "parameters. Try freeing GPU memory, lowering resolution, or enabling gradient_checkpointing."
            ) from exc

        # Armed until one step actually completes: only a successful step proves its gradients were the
        # real ones, and every step after it would recount the same parameters.
        warn_missing_grads = True

        def _probe(candidate_size: int) -> bool:
            nonlocal warn_missing_grads
            ok = _probe_step(
                model,
                criterion,
                candidate_size,
                resolution,
                device,
                num_classes,
                amp,
                trainable_params=trainable_params,
                shadow_optimizer=shadow_optimizer,
                shadow_params=shadow_params,
                segmentation_head=segmentation_head,
                max_targets_per_image=max_targets_per_image,
                num_channels=num_channels,
                autocast_dtype=autocast_dtype,
                warn_missing_grads=warn_missing_grads,
            )
            if ok:
                warn_missing_grads = False
            return ok

        # shadow_optimizer.step() allocates AdamW's exp_avg/exp_avg_sq state lazily, on its first-ever
        # call. That first call is necessarily made without the state resident yet -- unlike every real
        # training step from the second one onward, which always runs forward+backward with that state
        # already occupying memory. So a lone success at micro_batch_size=1 only proves the batch fits
        # *before* the state exists; it says nothing about whether it still fits once the state is
        # resident, which matters because the exponential search below can terminate immediately (if
        # candidate=2 already fails) without ever re-probing batch=1 under that steady-state condition.
        # Probe batch=1 twice up front so its own "safe" verdict is measured the same way every other
        # candidate's is: with the state already resident.
        if not _probe(1):
            raise RuntimeError(
                "auto-batch probe failed at micro_batch_size=1. "
                "Try lowering resolution or enabling gradient_checkpointing."
            )
        if not _probe(1):
            raise RuntimeError(
                "auto-batch probe failed at micro_batch_size=1 once optimizer state is resident. "
                "Try lowering resolution or enabling gradient_checkpointing."
            )

        lower_ok = 1
        candidate = 2
        upper_fail = None

        while candidate <= max_micro_batch:
            if _probe(candidate):
                lower_ok = candidate
                candidate *= 2
            else:
                upper_fail = candidate
                break

        if upper_fail is None:
            upper_fail = max_micro_batch + 1

        lo = lower_ok + 1
        hi = min(upper_fail - 1, max_micro_batch)
        while lo <= hi:
            mid = (lo + hi) // 2
            if _probe(mid):
                lower_ok = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # safe_micro_batch <= lower_ok always, since safety_margin <= 1.0.
        safe_micro_batch = max(1, math.floor(lower_ok * safety_margin))
        return safe_micro_batch
    finally:
        model.train(model_training)
        criterion.train(criterion_training)
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        # Drop the shadow state's last references before empty_cache(), while this frame is still alive
        # and would otherwise keep both names bound: only already-unreferenced memory can be returned to
        # the driver, so releasing them after the call would free nothing this call gets to see.
        # The guards are not about `del` legality -- deleting a None-bound name is fine -- but about the
        # closure: `_probe` captures both names, so an unconditional `del` reads as an undefined
        # reference to it (ruff F821), and rebinding to None instead costs mypy's narrowing at the same
        # spot. Conditional deletes are the one form both accept; keep them conditional.
        if shadow_optimizer is not None:
            del shadow_optimizer
        if shadow_params is not None:
            del shadow_params
        torch.cuda.empty_cache()


def recommend_grad_accum_steps(safe_micro_batch: int, target_effective_batch: int) -> int:
    """Recommend gradient accumulation steps to reach target effective batch size.

    Args:
        safe_micro_batch: Per-step batch size that fits in memory.
        target_effective_batch: Desired effective batch (micro_batch * accum_steps).

    Returns:
        ceil(target_effective_batch / safe_micro_batch), at least 1.

    Raises:
        ValueError: If either argument is < 1.
    """
    if safe_micro_batch < 1:
        raise ValueError("safe_micro_batch must be >= 1.")
    if target_effective_batch < 1:
        raise ValueError("target_effective_batch must be >= 1.")
    return max(1, math.ceil(target_effective_batch / safe_micro_batch))


def resolve_auto_batch_config(
    model_context: Any,
    model_config: ModelConfig,
    train_config: TrainConfig,
    safety_margin: float = 0.9,
    max_micro_batch: int = 128,
) -> AutoBatchResult:
    """Resolve batch_size='auto' into concrete batch_size and grad_accum_steps using a probe.

    Expects model_context to have attributes: .device (torch.device) and .model (nn.Module). Runs probe_max_micro_batch
    on the current model/criterion, then recommend_grad_accum_steps using train_config.auto_batch_target_effective. Logs
    device, segmentation flag, resolution, and the chosen values; also logs that the probe is train-step-only and that
    eval/test may use more memory.

    The optimizer-state memory the probe accounts for (see probe_max_micro_batch) is always modeled as AdamW's, since
    that is the only optimizer the shadow step builds. When train_config.optimizer selects something that is not
    AdamW-shaped (e.g. "sgd", a pytorch-optimizer name, or a custom callable -- all supported by configure_optimizers;
    see _is_adamw_shaped for what still counts as AdamW), this logs a warning: the real optimizer's state memory may
    differ from AdamW's, so the resulting batch size can be too conservative or too permissive. Also re-derives whether
    the real run will use fused AdamW (see
    RFDETRModelModule._use_fused_optimizer) so the shadow optimizer's step-time temporary memory matches it --
    fused and foreach/single-tensor AdamW have different scratch-memory profiles during step().

    Args:
        model_context: Object with .device and .model (e.g. RFDETR.model from get_model()).
        model_config: Architecture config (resolution, num_classes, amp, segmentation_head).
        train_config: Training config (auto_batch_target_effective); batch_size should be "auto".
        safety_margin: Fraction of max batch to use (passed to probe_max_micro_batch).
        max_micro_batch: Upper bound on batch size to try (passed to probe_max_micro_batch).

    Returns:
        AutoBatchResult with safe_micro_batch, recommended_grad_accum_steps, effective_batch_size, and device_name.

    Raises:
        RuntimeError: If CUDA is not available or model_context.device is not CUDA.
    """
    device = model_context.device
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("batch_size='auto' requires a CUDA device for probing in v1.")

    # Use max multi-scale resolution when multi_scale is True so probe reflects worst-case.
    multi_scale = getattr(train_config, "multi_scale", False)
    do_random_resize = getattr(train_config, "do_random_resize_via_padding", False)
    if multi_scale and not do_random_resize:
        expanded_scales = getattr(train_config, "expanded_scales", True)
        patch_size = getattr(model_config, "patch_size", 14)
        num_windows = getattr(model_config, "num_windows", 4)
        scales = compute_multi_scale_scales(
            model_config.resolution,
            expanded_scales,
            patch_size,
            num_windows,
        )
        probe_resolution = max(scales) if scales else model_config.resolution
    else:
        probe_resolution = model_config.resolution

    max_targets_per_image = getattr(train_config, "auto_batch_max_targets_per_image", 100)

    optimizer_cfg = getattr(train_config, "optimizer", "adamw")
    # Whether the real optimizer's *state* is AdamW-shaped -- a broader question than which optimizer
    # gets RF-DETR's fused kernel (derived separately below from _is_builtin_fused_adamw), so this uses
    # its own local predicate: "torch.optim.AdamW" allocates exactly the state the shadow models, while
    # never being eligible for the managed fused path.
    is_adamw_shaped = _is_adamw_shaped(optimizer_cfg)
    if is_adamw_shaped:
        # Same kwargs configure_optimizers passes to the real AdamW (module_model.py); forwarding
        # them here matters because some (e.g. amsgrad=True) add a third per-parameter state buffer
        # that the shadow optimizer would otherwise silently miss.
        shadow_optimizer_kwargs = getattr(train_config, "optimizer_kwargs", {})
    else:
        shadow_optimizer_kwargs = None
        logger.warning(
            "[auto-batch] optimizer=%r does not allocate AdamW-shaped state; the probe only models AdamW's "
            "exp_avg/exp_avg_sq optimizer-state memory (see _build_shadow_optimizer), so this "
            "batch-size estimate may be too conservative or too permissive for the optimizer "
            "actually configured for training.",
            optimizer_cfg,
        )

    criterion, _ = build_criterion_from_config(model_config, train_config)
    criterion = criterion.to(device)

    amp_enabled = bool(model_config.amp)
    amp_dtype_str = getattr(train_config, "amp_dtype", "auto")
    if amp_enabled:
        if amp_dtype_str == "fp16":
            probe_autocast_dtype: torch.dtype | None = torch.float16
        else:
            # "bf16" or "auto" — both use bf16 on capable hardware, fp16 as fallback
            probe_autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        probe_autocast_dtype = None

    # Mirrors RFDETRModelModule._use_fused_optimizer (module_model.py), which build_trainer's
    # real AdamW reads at construction time -- but that Lightning module doesn't exist yet at
    # probe time, so re-derive its two conditions instead of calling it directly: (1) the
    # built-in "adamw" path, which is _is_builtin_fused_adamw's own narrow question and stays on
    # that predicate rather than the broader state-shape one above -- only the short name reaches
    # the managed fused kernel, so a dotted-path config must not be silently upgraded to fused --
    # and (2) the resolved precision is a bf16 variant, which for this function's CUDA-only
    # autocast resolution above is exactly the case where probe_autocast_dtype came out to
    # torch.bfloat16 (see trainer.py's _resolve_precision: bf16-mixed iff CUDA + bf16-capable +
    # amp_dtype in {"auto", "bf16"}, the same inputs probe_autocast_dtype was derived from).
    use_fused_optimizer = (
        _is_builtin_fused_adamw(optimizer_cfg)
        and bool(getattr(model_config, "fused_optimizer", True))
        and probe_autocast_dtype is torch.bfloat16
    )

    safe_micro_batch = probe_max_micro_batch(
        model=model_context.model,
        criterion=criterion,
        resolution=probe_resolution,
        device=device,
        num_classes=model_config.num_classes,
        amp=amp_enabled,
        segmentation_head=model_config.segmentation_head,
        max_targets_per_image=max_targets_per_image,
        safety_margin=safety_margin,
        max_micro_batch=max_micro_batch,
        num_channels=getattr(model_config, "num_channels", 3),
        autocast_dtype=probe_autocast_dtype,
        optimizer_kwargs=shadow_optimizer_kwargs,
        optimizer_fused=use_fused_optimizer,
    )

    use_ema = getattr(train_config, "use_ema", False)
    if use_ema:
        headroom = getattr(train_config, "auto_batch_ema_headroom", 0.7)
        safe_micro_batch = max(1, math.floor(safe_micro_batch * headroom))
        logger.info("[auto-batch] Applied EMA headroom (%.2f): safe_micro_batch=%s", headroom, safe_micro_batch)

    # Infer world size from the PTL device syntax so a global target remains stable when the
    # trainer selects multiple CUDA devices from an automatic or comma-separated specification.
    devices = getattr(train_config, "devices", None)
    num_nodes = getattr(train_config, "num_nodes", 1)
    if isinstance(devices, int) and isinstance(num_nodes, int):
        device_count = torch.cuda.device_count() if devices == -1 and torch.cuda.is_available() else devices
        world_size = max(1, device_count * num_nodes)
    elif isinstance(devices, str) and isinstance(num_nodes, int):
        devices_name = devices.strip().lower()
        if devices_name in ("auto", "-1"):
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        elif devices_name.isdigit():
            device_count = int(devices_name)
        elif "," in devices_name:
            device_count = len([entry for entry in devices_name.split(",") if entry.strip()])
        else:
            device_count = 1
        world_size = max(1, device_count * num_nodes)
    else:
        world_size = 1

    # Interpret auto_batch_target_effective as a global target and derive a per-device target
    target_effective_global = train_config.auto_batch_target_effective
    if world_size > 1:
        target_effective_per_device = max(1, math.ceil(target_effective_global / world_size))
    else:
        target_effective_per_device = target_effective_global

    grad_accum_steps = recommend_grad_accum_steps(safe_micro_batch, target_effective_per_device)
    effective_batch_size_per_device = safe_micro_batch * grad_accum_steps
    global_effective_batch_size = effective_batch_size_per_device * world_size
    device_name = torch.cuda.get_device_name(device)

    logger.info(
        "[auto-batch] device=%s world_size=%s segmentation=%s probe_resolution=%s max_targets_per_image=%s",
        device_name,
        world_size,
        model_config.segmentation_head,
        probe_resolution,
        max_targets_per_image,
    )
    logger.info(
        "[auto-batch] safe_micro_batch=%s grad_accum_steps=%s effective_batch_per_device=%s global_effective_batch=%s",
        safe_micro_batch,
        grad_accum_steps,
        effective_batch_size_per_device,
        global_effective_batch_size,
    )
    logger.info("[auto-batch] This probe estimates train-step-safe micro-batch only.")
    logger.info("[auto-batch] Validation/test (especially segmentation mask eval) may require more memory.")

    return AutoBatchResult(
        safe_micro_batch=safe_micro_batch,
        recommended_grad_accum_steps=grad_accum_steps,
        effective_batch_size=effective_batch_size_per_device,
        device_name=device_name,
    )
