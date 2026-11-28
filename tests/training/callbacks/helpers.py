# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared test helpers for the rfdetr.training.callbacks test suite.

Plain mock builders, data factories and tiny Lightning modules (not pytest fixtures) shared by several callback test
modules. Import with an absolute import::

    from tests.training.callbacks.helpers import _cpu_module, _make_pl_module, _make_trainer

The ``COCOEvalCallback`` tests use ``_make_trainer`` and ``_make_pl_module``. The ``BestModelCallback`` tests need a
trainer with ``callback_metrics`` and a module whose state dict can be saved, provided by ``_make_best_model_trainer``
and ``_make_best_model_pl_module``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor

# ---------------------------------------------------------------------------
# COCOEvalCallback helpers
# ---------------------------------------------------------------------------


def _make_pl_module() -> MagicMock:
    """Return a minimal mock LightningModule.

    Examples:
        >>> isinstance(_make_pl_module(), MagicMock)
        True
    """
    return MagicMock(name="pl_module")


def _make_trainer(datamodule=None, callbacks: list[object] | None = None) -> MagicMock:
    """Return a minimal mock Trainer with an optional DataModule.

    Examples:
        >>> trainer = _make_trainer(datamodule="datamodule")
        >>> trainer.datamodule, trainer.callbacks
        ('datamodule', [])
    """
    trainer = MagicMock(name="trainer")
    trainer.datamodule = datamodule
    trainer.callbacks = callbacks or []
    return trainer


def _detection_preds(n: int = 0) -> list[dict]:
    """Return a list with one per-image prediction dict.

    Examples:
        >>> _detection_preds(2)[0]["boxes"].shape
        torch.Size([2, 4])
    """
    return [
        {
            "boxes": torch.zeros(n, 4),
            "scores": torch.zeros(n),
            "labels": torch.zeros(n, dtype=torch.long),
        }
    ]


def _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1, label=1) -> list[dict]:
    """Return a single-image target dict with one box in normalised CxCyWH.

    Examples:
        >>> _detection_targets(label=3)[0]["labels"].tolist()
        [3]
    """
    return [
        {
            "boxes": torch.tensor([[cx, cy, w, h]]),
            "labels": torch.tensor([label]),
            "orig_size": torch.tensor([100, 200]),  # H=100, W=200
        }
    ]


def _minimal_metrics(pfx: str = "", max_dets: int = 500) -> dict:
    """Return a minimal torchmetrics-style metrics dict.

    Examples:
        >>> sorted(_minimal_metrics("ema_", max_dets=100))
        ['ema_map', 'ema_map_50', 'ema_map_75', 'ema_mar_100']
    """
    return {
        f"{pfx}map": torch.tensor(0.4),
        f"{pfx}map_50": torch.tensor(0.6),
        f"{pfx}map_75": torch.tensor(0.3),
        f"{pfx}mar_{max_dets}": torch.tensor(0.5),
    }


def _ema_callback(ema_underlying: MagicMock | None = None) -> MagicMock:
    """Return a mock that ``_get_ema_callback`` recognises (has ``get_ema_model_state_dict``).

    With ``ema_underlying`` the mock's averaged model wraps it, so the EMA inner module's ``.model`` is that object.

    Examples:
        >>> underlying = MagicMock(name="ema_underlying_model")
        >>> _ema_callback(underlying)._average_model.module.model is underlying
        True
        >>> callable(_ema_callback().get_ema_model_state_dict)
        True
    """
    cb = MagicMock(name="ema_callback")
    cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
    if ema_underlying is not None:
        cb._average_model = SimpleNamespace(module=SimpleNamespace(model=ema_underlying))
    return cb


def _cpu_module() -> MagicMock:
    """Mock LightningModule whose ``device`` is a real string so ``metric.to(device)`` works.

    Examples:
        >>> _cpu_module().device
        'cpu'
    """
    module = MagicMock(name="pl_module")
    module.device = "cpu"
    return module


def _matching_entry(total_gt: int) -> dict:
    """Return a per-class matching-accumulator entry carrying ``total_gt`` instances.

    Examples:
        >>> _matching_entry(3)["total_gt"]
        3
    """
    return {
        "scores": np.array([0.9], dtype=np.float32),
        "matches": np.array([1], dtype=np.int64),
        "ignore": np.array([False]),
        "total_gt": total_gt,
    }


def _metrics_with_per_class(aps: dict[int, float]) -> dict:
    """Return a metrics dict carrying per-class AP for the given category ids.

    Examples:
        >>> _metrics_with_per_class({0: 0.6, 1: 0.0})["classes"].tolist()
        [0, 1]
    """
    metrics = _minimal_metrics()
    metrics["classes"] = torch.tensor(sorted(aps))
    metrics["map_per_class"] = torch.tensor([aps[c] for c in sorted(aps)])
    metrics["mar_500_per_class"] = torch.tensor([0.5 for _ in aps])
    return metrics


# ---------------------------------------------------------------------------
# BestModelCallback helpers
# ---------------------------------------------------------------------------


def _make_best_model_trainer(
    metrics: dict[str, float],
    current_epoch: int = 1,
    is_global_zero: bool = True,
    callbacks: list[object] | None = None,
    sanity_checking: bool = False,
) -> MagicMock:
    """Create a minimal mock Trainer with controllable callback_metrics.

    Sets the attributes required by ModelCheckpoint and EarlyStopping skip-guards so that callbacks run normally in unit
    tests.

    Args:
        metrics: Validation metric names mapped to values, exposed as ``trainer.callback_metrics``.
        current_epoch: Value for ``trainer.current_epoch``.
        is_global_zero: Value for ``trainer.is_global_zero``.
        callbacks: Value for ``trainer.callbacks``; defaults to an empty list.
        sanity_checking: Value for ``trainer.sanity_checking`` — True mimics PyTorch Lightning's pre-training
            sanity-check pass (see #1348).

    Returns:
        A mock standing in for a :class:`~pytorch_lightning.Trainer`.

    Examples:
        >>> trainer = _make_best_model_trainer({"val/mAP_50_95": 0.5}, current_epoch=2, sanity_checking=True)
        >>> trainer.current_epoch, trainer.sanity_checking
        (2, True)
        >>> trainer.callback_metrics["val/mAP_50_95"].item()
        0.5
    """
    trainer = MagicMock()
    trainer.callback_metrics = {k: torch.tensor(v) for k, v in metrics.items()}
    trainer.current_epoch = current_epoch
    trainer.is_global_zero = is_global_zero
    trainer.callbacks = callbacks or []
    trainer.should_stop = False
    # Required by ModelCheckpoint._should_skip_saving_checkpoint
    trainer.fast_dev_run = False
    trainer.state.fn = TrainerFn.FITTING
    trainer.sanity_checking = sanity_checking
    trainer.global_step = 1  # int; differs from _last_global_step_saved=0
    # Required by EarlyStopping._log_info (world_size > 1 check)
    trainer.world_size = 1
    # Required by ModelCheckpoint.check_monitor_top_k and EarlyStopping (DDP reduce)
    trainer.strategy.reduce_boolean_decision.side_effect = lambda x, **kwargs: x
    # Required by BestModelCallback.on_fit_end, which broadcasts the run_test decision from the main process.
    trainer.strategy.broadcast.side_effect = lambda obj, src=0: obj
    # Prevent MagicMock auto-attribute from triggering class_names enrichment.
    trainer.datamodule.class_names = None
    return trainer


def _make_best_model_pl_module() -> MagicMock:
    """Create a minimal mock RFDETRModule with state_dict and train_config.

    Examples:
        >>> _make_best_model_pl_module().train_config
        {'lr': 0.001}
    """
    pl_module = MagicMock()
    pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
    # Use a real dict so torch.save can pickle it (MagicMock is not picklable).
    pl_module.train_config = {"lr": 0.001}
    return pl_module


_REGULAR_AND_EMA_CASES = [
    pytest.param({}, {"val/mAP_50_95": 0.5}, "checkpoint_best_regular.pth", id="regular"),
    pytest.param(
        {"monitor_ema": "val/ema_mAP_50_95"},
        {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.6},
        "checkpoint_best_ema.pth",
        id="ema",
    ),
]


class _ModuleWithTestStep(LightningModule):
    """Real module defining ``test_step`` so BestModelCallback.on_fit_end considers running ``trainer.test()``."""

    def test_step(self, batch: object, batch_idx: int) -> None:
        """No-op test step."""


def _make_test_step_module(
    state_dict: dict[str, object] | None = None, missing_keys: list[str] | None = None
) -> _ModuleWithTestStep:
    """Build a real ``test_step`` module around a mocked model that reloads with the given ``missing_keys``.

    A real ``_IncompatibleKeys`` is returned from ``load_state_dict`` because a MagicMock's truthy result would fail the
    ``strict=False`` reload check.

    Args:
        state_dict: What the model's ``state_dict()`` returns; one zero tensor ``w`` by default.
        missing_keys: Keys the reload reports as missing; none by default.

    Returns:
        The module with ``model``, ``train_config`` and the mocked reload set.

    Examples:
        >>> module = _make_test_step_module(missing_keys=["w"])
        >>> module.model.load_state_dict({}, strict=False).missing_keys
        ['w']
    """
    module = _ModuleWithTestStep()
    module.model = MagicMock()
    module.model.state_dict.return_value = state_dict if state_dict is not None else {"w": torch.zeros(1)}
    module.model.load_state_dict.return_value = torch.nn.modules.module._IncompatibleKeys(missing_keys or [], [])
    module.train_config = {"lr": 0.001}
    return module


class _EmaMetricModule(LightningModule):
    """LightningModule that logs independently controlled ``val/mAP_50_95`` and ``val/ema_mAP_50_95`` values.

    Used to drive :class:`BestModelCallback`'s EMA-checkpoint bookkeeping (``checkpoint_best_ema.pth``) with a
    real, trainable model, so :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback` produces genuinely
    different EMA weights per :class:`~pytorch_lightning.Trainer` phase.
    """

    def __init__(self, regular_value: float, ema_value: float) -> None:
        """Initialize with the fixed regular/EMA metric values to log every epoch.

        Args:
            regular_value: Value logged as ``val/mAP_50_95`` on every validation epoch.
            ema_value: Value logged as ``val/ema_mAP_50_95`` on every validation epoch.
        """
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_config = {"lr": 0.01}
        self._regular_value = regular_value
        self._ema_value = ema_value

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute MSE loss for one training batch.

        Args:
            batch: ``(x, y)`` tensors from the ``TensorDataset`` loader.
            batch_idx: Index of the batch within the current epoch (unused).

        Returns:
            Scalar MSE loss.
        """
        del batch_idx
        x, y = batch
        pred = self.model(x)
        return torch.nn.functional.mse_loss(pred, y)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Log the fixed regular and EMA metric values, ignoring the actual batch contents.

        Args:
            batch: Unused; only the epoch-level log calls matter for these tests.
            batch_idx: Unused.
        """
        del batch, batch_idx
        self.log("val/mAP_50_95", torch.tensor(self._regular_value), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/ema_mAP_50_95", torch.tensor(self._ema_value), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a plain SGD optimizer sufficient to drive ``Trainer.fit()``."""
        return torch.optim.SGD(self.model.parameters(), lr=0.01)
