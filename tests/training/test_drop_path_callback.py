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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_trainer(global_step: int = 0, estimated_stepping_batches: int = 50) -> MagicMock:
    """Create a minimal mock Trainer with controllable step metadata."""
    trainer = MagicMock()
    trainer.global_step = global_step
    trainer.estimated_stepping_batches = estimated_stepping_batches
    return trainer


def _make_mock_pl_module(epochs: int = 5) -> MagicMock:
    """Create a minimal mock RFDETRModule with ``train_config.epochs``."""
    pl_module = MagicMock()
    pl_module.train_config.epochs = epochs
    return pl_module


# ---------------------------------------------------------------------------
# TestDropPathCallbackInit
# ---------------------------------------------------------------------------


class TestDropPathCallbackInit:
    """Verify constructor defaults."""

    def test_default_args(self) -> None:
        """Default rates are zero and vit_encoder_num_layers is 12."""
        cb = DropPathCallback()
        assert cb._drop_path == 0.0
        assert cb._dropout == 0.0
        assert cb._vit_encoder_num_layers == 12
        assert cb._dp_schedule is None
        assert cb._do_schedule is None


# ---------------------------------------------------------------------------
# TestOnTrainStart
# ---------------------------------------------------------------------------


class TestOnTrainStart:
    """Verify schedule arrays built in ``on_train_start``."""

    def test_dp_schedule_matches_drop_scheduler_standard(self) -> None:
        """drop_path schedule matches ``drop_scheduler`` for standard mode."""
        cb = DropPathCallback(drop_path=0.3)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        expected = drop_scheduler(0.3, 5, 10)
        assert cb._dp_schedule is not None
        np.testing.assert_array_equal(cb._dp_schedule, expected)

    def test_do_schedule_matches_drop_scheduler_standard(self) -> None:
        """dropout schedule matches ``drop_scheduler`` for standard mode."""
        cb = DropPathCallback(dropout=0.1)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        expected = drop_scheduler(0.1, 5, 10)
        assert cb._do_schedule is not None
        np.testing.assert_array_equal(cb._do_schedule, expected)

    def test_no_dp_schedule_when_rate_zero(self) -> None:
        """drop_path=0.0 leaves ``_dp_schedule`` as None."""
        cb = DropPathCallback(drop_path=0.0)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        assert cb._dp_schedule is None

    def test_dp_schedule_early_mode(self) -> None:
        """Early mode: rates at step 0 and step 30 match ``drop_scheduler``."""
        cb = DropPathCallback(drop_path=0.3, cutoff_epoch=2, mode="early")
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        expected = drop_scheduler(0.3, 5, 10, 2, "early")
        assert cb._dp_schedule is not None
        assert cb._dp_schedule[0] == expected[0]
        assert cb._dp_schedule[30] == expected[30]

    def test_dp_schedule_late_mode(self) -> None:
        """Late mode: rates at step 0 and step 30 match ``drop_scheduler``."""
        cb = DropPathCallback(drop_path=0.3, cutoff_epoch=2, mode="late")
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        expected = drop_scheduler(0.3, 5, 10, 2, "late")
        assert cb._dp_schedule is not None
        assert cb._dp_schedule[0] == expected[0]
        assert cb._dp_schedule[30] == expected[30]


# ---------------------------------------------------------------------------
# TestOnTrainBatchStart
# ---------------------------------------------------------------------------


class TestOnTrainBatchStart:
    """Verify model update calls in ``on_train_batch_start``."""

    def test_update_drop_path_called_with_correct_rate(self) -> None:
        """``update_drop_path`` is called with the schedule value at step 0."""
        cb = DropPathCallback(drop_path=0.3, vit_encoder_num_layers=6)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        trainer.global_step = 0
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        assert cb._dp_schedule is not None
        pl_module.model.update_drop_path.assert_called_once_with(cb._dp_schedule[0], 6)

    def test_update_dropout_called_with_correct_rate(self) -> None:
        """``update_dropout`` is called with the schedule value at step 0."""
        cb = DropPathCallback(dropout=0.1)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        trainer.global_step = 0
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        assert cb._do_schedule is not None
        pl_module.model.update_dropout.assert_called_once_with(cb._do_schedule[0])

    def test_no_update_when_step_out_of_bounds(self) -> None:
        """No model updates when ``global_step`` exceeds schedule length."""
        cb = DropPathCallback(drop_path=0.3, dropout=0.1)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        trainer.global_step = 9999
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        pl_module.model.update_drop_path.assert_not_called()
        pl_module.model.update_dropout.assert_not_called()

    @pytest.mark.parametrize(
        "step",
        [
            pytest.param(0, id="first_step"),
            pytest.param(5, id="mid_step"),
            pytest.param(9, id="last_of_first_epoch"),
        ],
    )
    def test_drop_rates_at_multiple_steps_match_schedule(self, step: int) -> None:
        """Each step uses the correct value from the pre-computed schedule."""
        cb = DropPathCallback(drop_path=0.3, vit_encoder_num_layers=6)
        trainer = _make_mock_trainer(estimated_stepping_batches=50)
        pl_module = _make_mock_pl_module(epochs=5)

        cb.on_train_start(trainer, pl_module)

        trainer.global_step = step
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)

        assert cb._dp_schedule is not None
        pl_module.model.update_drop_path.assert_called_once_with(cb._dp_schedule[step], 6)
