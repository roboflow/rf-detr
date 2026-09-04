# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exponential Moving Average callback compatible with ``ModelEma``."""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

if TYPE_CHECKING:
    from rfdetr.training.module_model import RFDETRModelModule


class RFDETREMACallback(Callback):
    """Exponential Moving Average with optional tau-based warm-up.

    Drop-in replacement for ``rfdetr.util.utils.ModelEma`` implemented as a plain Lightning callback around
    :class:`torch.optim.swa_utils.AveragedModel`. The ``_avg_fn`` reproduces the exact same formula as ``ModelEma``
    (1-indexed ``updates`` counter, optional ``tau`` warm-up).

    Args:
        decay: Base EMA decay factor. Corresponds to ``TrainConfig.ema_decay``.
        tau: Warm-up time constant (in optimizer steps). When > 0 the
            effective decay ramps from 0 towards *decay* following ``decay * (1 - exp(-updates / tau))``. Corresponds to
            ``TrainConfig.ema_tau``.
        use_buffers: Whether buffers are averaged in addition to parameters.
        update_interval_steps: Update EMA every N optimizer steps.

    Attributes:
        suppress_test_swap: When ``True`` the test-epoch hooks skip the EMA weight swap.  Set (and restored) by
            :class:`~rfdetr.training.callbacks.best_model.BestModelCallback` around its fit-end ``trainer.test()``
            run, which has already loaded the best checkpoint weights into the module — swapping in the final EMA
            weights there would make the reported ``test/*`` metrics reflect the wrong model.  Standalone
            ``trainer.test()`` runs keep the default ``False`` and evaluate EMA weights as before.
    """

    def __init__(
        self,
        decay: float = 0.993,
        tau: int = 100,
        use_buffers: bool = True,
        update_interval_steps: int = 1,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._tau = tau
        self._use_buffers = use_buffers
        self._update_interval_steps = max(1, int(update_interval_steps))
        self.suppress_test_swap = False

        self._average_model: AveragedModel | None = None
        self._latest_update_step = 0
        self._swapped_state_dict: dict[str, Tensor] | None = None
        self._pending_average_state_dict: dict[str, Any] | None = None

    # Retained as the per-tensor fallback for non-floating-point groups (see
    # _multi_avg_fn) — no longer the registered AveragedModel avg_fn.
    def _avg_fn(
        self,
        averaged_param: Tensor,
        model_param: Tensor,
        num_averaged: Tensor | int,
    ) -> Tensor:
        """Compute the EMA update for a single parameter tensor.

        Matches the ``ModelEma`` formula where ``updates`` is 1-indexed: PTL's ``num_averaged`` starts at 0 (incremented
        *after* calling ``avg_fn``), so ``updates = num_averaged + 1`` reproduces the same sequence of effective decay
        values.

        Args:
            averaged_param: Current EMA parameter value.
            model_param: Corresponding live model parameter value.
            num_averaged: Number of models averaged so far (0-indexed). ``AveragedModel`` always passes this as a
                0-dim tensor; the ``int`` branch only matches the declared ``torch.optim.swa_utils`` signature.

        Returns:
            Updated EMA parameter tensor.
        """
        num_averaged_value = num_averaged.item() if isinstance(num_averaged, Tensor) else num_averaged
        effective_decay = self._effective_decay(int(num_averaged_value))
        return averaged_param * effective_decay + model_param * (1.0 - effective_decay)

    def _effective_decay(self, num_averaged: int) -> float:
        """Return the effective decay for the given 0-indexed average counter.

        Args:
            num_averaged: Number of models averaged so far (0-indexed).

        Returns:
            Effective decay after the optional tau warm-up ramp.
        """
        updates = num_averaged + 1  # match ModelEma 1-indexed counter
        if self._tau > 0:
            return self._decay * (1 - math.exp(-updates / self._tau))
        return self._decay

    def _multi_avg_fn(
        self,
        averaged_params: tuple[Tensor, ...] | list[Tensor],
        model_params: tuple[Tensor, ...] | list[Tensor],
        num_averaged: Tensor | int,
    ) -> None:
        """Update a (device, dtype) group of EMA tensors in-place via foreach kernels.

        ``AveragedModel.update_parameters`` routes to this grouped path when ``multi_avg_fn`` is set, replacing the
        per-tensor ``avg_fn`` loop that performed one ``num_averaged.item()`` GPU→CPU sync *per tensor* per step with a
        single sync per group. The float path applies ``ema * decay + model * (1 - decay)``, numerically equivalent
        within floating-point tolerance to ``_avg_fn`` (``torch._foreach_add_(..., alpha=)`` may lower to an FMA
        instruction, so the result can differ from separate mul-then-add by ~1 ULP); non-floating-point groups (e.g.
        integer buffers when averaging buffers) fall back to the per-tensor formula to preserve its cast semantics.

        Args:
            averaged_params: EMA tensors of one device/dtype group, updated in-place.
            model_params: Matching live model tensors.
            num_averaged: Number of models averaged so far (0-indexed); passed by ``AveragedModel`` as a 0-dim tensor.
        """
        num_averaged_value = int(num_averaged.item()) if isinstance(num_averaged, Tensor) else int(num_averaged)
        effective_decay = self._effective_decay(num_averaged_value)
        if not averaged_params:
            return
        if not averaged_params[0].is_floating_point():
            for averaged_param, model_param in zip(averaged_params, model_params):
                averaged_param.copy_(self._avg_fn(averaged_param, model_param, num_averaged_value))
            return
        # Two non-atomic in-place ops: a failure between them (e.g. a future dtype/shape
        # mismatch) would leave averaged_params scaled-but-not-added, with no rollback.
        # Accepted risk — AveragedModel pairs matching tensors, so this cannot occur today.
        torch._foreach_mul_(averaged_params, effective_decay)
        torch._foreach_add_(averaged_params, model_params, alpha=1.0 - effective_decay)

    def _restore_pending_average_state(self, pl_module: LightningModule) -> None:
        """Restore a callback or legacy EMA state after constructing the averaged model.

        Current callback state takes precedence over the legacy model-only state
        when a checkpoint supplies both formats.

        Args:
            pl_module: The live module that may hold a stashed legacy EMA state.
        """
        if self._average_model is None:
            return
        if self._pending_average_state_dict is not None:
            self._average_model.load_state_dict(self._pending_average_state_dict)
            self._pending_average_state_dict = None
            return
        if not hasattr(pl_module, "_pending_legacy_ema_state"):
            return

        legacy_ema_state = pl_module._pending_legacy_ema_state
        if isinstance(legacy_ema_state, dict):
            average_module = cast("RFDETRModelModule", self._average_model.module)
            incompatible = average_module.model.load_state_dict(legacy_ema_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                warnings.warn(
                    "Legacy EMA checkpoint loaded with non-exact key match; "
                    f"missing={len(incompatible.missing_keys)} "
                    f"unexpected={len(incompatible.unexpected_keys)}.",
                    UserWarning,
                    stacklevel=2,
                )
        delattr(pl_module, "_pending_legacy_ema_state")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialise EMA after Lightning applies precision conversion and device placement.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        device = pl_module.device
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected a torch.device from the Lightning module, got {type(device).__name__}.")

        self._average_model = AveragedModel(
            model=pl_module,
            device=device,
            use_buffers=self._use_buffers,
            multi_avg_fn=self._multi_avg_fn,
        )
        # The averaged model is inference-only; PTL never calls .eval() on it
        # because it is not registered as a Lightning module.  Without this,
        # dropout layers stay in training mode and produce ~random outputs.
        self._average_model.eval()

        self._restore_pending_average_state(pl_module)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply resumed EMA state that arrived after ``on_fit_start()``.

        Standard strategies restore callback and module checkpoint state before
        ``on_fit_start()``, so that hook restores the EMA state immediately after
        constructing its precision-matched averaged model. Strategies that restore
        after their setup may deliver callback or legacy EMA state later; this hook
        is the first point after restoration where the same state can be applied.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        # Lightweight checkpoints deliberately restart optimizer-loop progress at
        # zero. An absolute saved guard would otherwise skip that many new batches.
        if trainer.global_step < self._latest_update_step:
            self._latest_update_step = trainer.global_step
        self._restore_pending_average_state(pl_module)

    def should_update(
        self,
        step_idx: int | None = None,
        epoch_idx: int | None = None,
    ) -> bool:
        """Return whether either trigger index is present.

        ``epoch_idx`` remains part of the callback interface for backwards compatibility and still counts as a
        trigger when supplied. The callback now invokes this method only from ``on_train_batch_end`` with ``step_idx``;
        it no longer dispatches an epoch-end EMA update, which previously double-counted the last step of each epoch
        and bypassed ``update_interval_steps``.

        Args:
            step_idx: Index of the last optimizer step, or ``None``.
            epoch_idx: Index of the last epoch, or ``None``. Retained for backwards API compatibility.

        Returns:
            ``True`` when either trigger index is not ``None``.
        """
        return step_idx is not None or epoch_idx is not None

    def _swap_models(self, pl_module: LightningModule) -> None:
        """Swap live model weights with averaged EMA weights."""
        if self._average_model is None:
            return
        if self._swapped_state_dict is None:
            self._swapped_state_dict = deepcopy(pl_module.state_dict())
            pl_module.load_state_dict(self._average_model.module.state_dict(), strict=True)
            return
        pl_module.load_state_dict(self._swapped_state_dict, strict=True)
        self._swapped_state_dict = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after optimizer steps."""
        if self._average_model is None:
            return
        step_idx = trainer.global_step - 1
        if trainer.global_step <= self._latest_update_step:
            return

        self._latest_update_step = trainer.global_step
        should_update_step = trainer.global_step % self._update_interval_steps == 0
        if should_update_step and self.should_update(step_idx=step_idx):
            self._average_model.update_parameters(pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate tests using averaged EMA weights unless the swap is suppressed."""
        if self.suppress_test_swap:
            return
        self._swap_models(pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore live weights after test evaluation unless the swap is suppressed."""
        if self.suppress_test_swap:
            return
        self._swap_models(pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Leave the module in EMA state after training finishes."""
        if self._average_model is not None:
            pl_module.load_state_dict(self._average_model.module.state_dict(), strict=True)
        self._swapped_state_dict = None

    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        state: dict[str, Any] = {
            "latest_update_step": self._latest_update_step,
        }
        if self._average_model is not None:
            state["average_model_state_dict"] = self._average_model.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore callback state from checkpoints."""
        self._latest_update_step = state_dict.get("latest_update_step", 0)
        self._pending_average_state_dict = state_dict.get("average_model_state_dict")

    def get_ema_model_state_dict(self) -> dict[str, Tensor] | None:
        """Expose EMA model weights for external checkpoint callbacks."""
        if self._average_model is None or not hasattr(self._average_model.module, "model"):
            return None
        average_module = cast("RFDETRModelModule", self._average_model.module)
        return {k: v.detach().clone() for k, v in average_module.model.state_dict().items()}
