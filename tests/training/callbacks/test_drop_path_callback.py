# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for :class:`rfdetr.training.callbacks.drop_schedule.DropPathCallback`."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rfdetr.training.callbacks.drop_schedule import DropPathCallback
from rfdetr.training.drop_schedule import drop_scheduler


@pytest.fixture
def trainer() -> MagicMock:
    """Mock Trainer at ``global_step`` 0 with 50 estimated stepping batches."""
    trainer = MagicMock()
    trainer.global_step = 0
    trainer.estimated_stepping_batches = 50
    return trainer


@pytest.fixture
def pl_module() -> MagicMock:
    """Mock RFDETRModule whose ``train_config.epochs`` is 5."""
    pl_module = MagicMock()
    pl_module.train_config.epochs = 5
    return pl_module


class TestDropPathCallbackInit:
    """Constructor defaults."""

    def test_default_args(self) -> None:
        """Default rates are zero, vit_encoder_num_layers is 12 and no schedule is built yet."""
        cb = DropPathCallback()
        assert cb._drop_path == 0.0
        assert cb._dropout == 0.0
        assert cb._vit_encoder_num_layers == 12
        assert cb._dp_schedule is None
        assert cb._do_schedule is None


class TestOnTrainStart:
    """Schedule arrays built in ``on_train_start``."""

    @pytest.mark.parametrize(
        ("kwargs", "attr", "rate"),
        [
            pytest.param({"drop_path": 0.3}, "_dp_schedule", 0.3, id="drop_path"),
            pytest.param({"dropout": 0.1}, "_do_schedule", 0.1, id="dropout"),
        ],
    )
    def test_schedule_matches_drop_scheduler_standard(
        self, trainer: MagicMock, pl_module: MagicMock, kwargs: dict, attr: str, rate: float
    ) -> None:
        """Each configured rate yields a schedule equal to ``drop_scheduler`` in standard mode."""
        cb = DropPathCallback(**kwargs)

        cb.on_train_start(trainer, pl_module)

        assert getattr(cb, attr) is not None
        np.testing.assert_array_equal(getattr(cb, attr), drop_scheduler(rate, 5, 10))

    def test_no_dp_schedule_when_rate_zero(self, trainer: MagicMock, pl_module: MagicMock) -> None:
        """drop_path=0.0 leaves ``_dp_schedule`` as None."""
        cb = DropPathCallback(drop_path=0.0)

        cb.on_train_start(trainer, pl_module)

        assert cb._dp_schedule is None

    @pytest.mark.parametrize("mode", ["early", "late"])
    def test_dp_schedule_cutoff_modes(self, trainer: MagicMock, pl_module: MagicMock, mode: str) -> None:
        """Early and late modes: rates at step 0 and step 30 match ``drop_scheduler``."""
        cb = DropPathCallback(drop_path=0.3, cutoff_epoch=2, mode=mode)

        cb.on_train_start(trainer, pl_module)

        expected = drop_scheduler(0.3, 5, 10, 2, mode)
        assert cb._dp_schedule is not None
        assert cb._dp_schedule[0] == expected[0]
        assert cb._dp_schedule[30] == expected[30]


class TestOnTrainBatchStart:
    """Model update calls in ``on_train_batch_start``."""

    @pytest.mark.parametrize("step", [0, 5, 9])
    def test_update_drop_path_uses_schedule_value_at_step(
        self, trainer: MagicMock, pl_module: MagicMock, step: int
    ) -> None:
        """``update_drop_path`` receives the pre-computed schedule value for the current step."""
        cb = DropPathCallback(drop_path=0.3, vit_encoder_num_layers=6)
        cb.on_train_start(trainer, pl_module)

        trainer.global_step = step
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        assert cb._dp_schedule is not None
        pl_module.model.update_drop_path.assert_called_once_with(cb._dp_schedule[step], 6)

    def test_update_dropout_uses_schedule_value_at_step_zero(self, trainer: MagicMock, pl_module: MagicMock) -> None:
        """``update_dropout`` receives the schedule value at step 0."""
        cb = DropPathCallback(dropout=0.1)
        cb.on_train_start(trainer, pl_module)

        trainer.global_step = 0
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        assert cb._do_schedule is not None
        pl_module.model.update_dropout.assert_called_once_with(cb._do_schedule[0])

    def test_no_update_when_step_out_of_bounds(self, trainer: MagicMock, pl_module: MagicMock) -> None:
        """No model updates when ``global_step`` exceeds schedule length."""
        cb = DropPathCallback(drop_path=0.3, dropout=0.1)
        cb.on_train_start(trainer, pl_module)

        trainer.global_step = 9999
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        pl_module.model.update_drop_path.assert_not_called()
        pl_module.model.update_dropout.assert_not_called()


class TestDropSchedulerValidation:
    """``drop_scheduler`` rejects invalid inputs."""

    @pytest.mark.parametrize("cutoff_epoch", [6, -1])
    def test_raises_for_invalid_cutoff_epoch(self, cutoff_epoch: int) -> None:
        """ValueError is raised when cutoff_epoch is outside [0, epochs]."""
        with pytest.raises(ValueError, match="cutoff_epoch must be in"):
            drop_scheduler(0.3, 5, 10, cutoff_epoch=cutoff_epoch, mode="early")

    @pytest.mark.parametrize(
        ("epochs", "niter_per_ep", "match"),
        [
            pytest.param(0, 10, "epochs must be >= 1", id="epochs_zero"),
            pytest.param(5, 0, "niter_per_ep must be >= 1", id="niter_per_ep_zero"),
        ],
    )
    def test_raises_for_invalid_epoch_counts(self, epochs: int, niter_per_ep: int, match: str) -> None:
        """ValueError is raised when epochs or niter_per_ep is less than 1."""
        with pytest.raises(ValueError, match=match):
            drop_scheduler(0.3, epochs, niter_per_ep)


class TestDropSchedulerBoundary:
    """``drop_scheduler`` with cutoff_epoch at the inclusive boundaries 0 and epochs."""

    @pytest.mark.parametrize(
        ("cutoff_epoch", "mode", "expected_first", "expected_last"),
        [
            pytest.param(0, "early", 0.0, 0.0, id="early_cutoff_zero_all_zeros"),
            pytest.param(5, "early", 0.3, 0.3, id="early_cutoff_full_all_rate"),
            pytest.param(0, "late", 0.3, 0.3, id="late_cutoff_zero_all_rate"),
            pytest.param(5, "late", 0.0, 0.0, id="late_cutoff_full_all_zeros"),
        ],
    )
    def test_boundary_cutoff_epoch(
        self, cutoff_epoch: int, mode: str, expected_first: float, expected_last: float
    ) -> None:
        """Boundary cutoff_epoch values (0 and epochs) produce the expected first and last rates."""
        schedule = drop_scheduler(0.3, 5, 10, cutoff_epoch=cutoff_epoch, mode=mode)
        assert schedule[0] == expected_first
        assert schedule[-1] == expected_last


class TestDropSchedulerLinear:
    """``drop_scheduler`` with schedule='linear' in early mode."""

    @pytest.fixture
    def schedule(self) -> np.ndarray:
        """Linear early schedule: rate 0.3, 5 epochs of 10 iterations, cutoff at epoch 2."""
        return drop_scheduler(0.3, 5, 10, cutoff_epoch=2, mode="early", schedule="linear")

    def test_starts_at_drop_rate(self, schedule: np.ndarray) -> None:
        """First value equals drop_rate."""
        assert schedule[0] == pytest.approx(0.3, abs=1e-9)

    def test_early_phase_ends_at_zero(self, schedule: np.ndarray) -> None:
        """Last value of the early phase equals 0."""
        assert schedule[19] == pytest.approx(0.0, abs=1e-9)

    def test_late_phase_is_zero(self, schedule: np.ndarray) -> None:
        """All values after cutoff_epoch are zero."""
        np.testing.assert_array_equal(schedule[20:], 0.0)

    def test_decreases_monotonically(self, schedule: np.ndarray) -> None:
        """Values decrease monotonically during the early phase."""
        assert np.all(np.diff(schedule[:20]) <= 0)

    def test_same_shape_as_constant(self, schedule: np.ndarray) -> None:
        """Output has the same length as schedule='constant'."""
        constant = drop_scheduler(0.3, 5, 10, cutoff_epoch=2, mode="early", schedule="constant")
        assert schedule.shape == constant.shape
