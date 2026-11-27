# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the gated-mAP fallback warnings and the console header of ``COCOEvalCallback``."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest_mock import MockerFixture

from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from tests.training.callbacks.helpers import (
    _cpu_module,
    _make_trainer,
    _matching_entry,
    _metrics_with_per_class,
)

_WARNING = "rfdetr.training.callbacks.coco_eval.logger.warning"


def _compute(
    cb: COCOEvalCallback, merged: dict, metrics: dict, trainer: MagicMock | None = None
) -> tuple[MagicMock, MagicMock]:
    """Run one base-track ``_compute_and_log`` with the metric and the table printer stubbed out.

    Examples:
        >>> cb = COCOEvalCallback(min_class_support=2)
        >>> trainer, print_tables = _compute(cb, {0: _matching_entry(5)}, _metrics_with_per_class({0: 0.5}))
        >>> print_tables.call_args.args[2]["gated applied"]
        1.0
    """
    trainer = trainer if trainer is not None else _make_trainer()
    module = _cpu_module()
    cb.setup(trainer, module, stage="fit")
    trainer.callback_metrics = {}
    cb.map_metric = MagicMock(name="map_metric")
    cb.map_metric.has_updates = True
    cb.map_metric.compute.return_value = metrics
    with (
        patch.object(cb, "_print_metrics_tables") as print_tables,
        patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value=merged),
    ):
        cb._compute_and_log(trainer, module, "val")
    return trainer, print_tables


def _messages(warning: MagicMock) -> list[str]:
    """Return the formatted text of every logger warning recorded on ``warning``.

    Examples:
        >>> warning = MagicMock()
        >>> _ = warning("%s: n=%d", "val", 3)
        >>> _messages(warning)
        ['val: n=3']
    """
    return [call.args[0] % call.args[1:] for call in warning.call_args_list]


class TestNoEligibleClassWarning:
    """An empty eligible set is warned about once per split per fit, on rank zero, naming mAP and F1."""

    def test_warns_once_per_split_across_epochs(self, mocker: MockerFixture) -> None:
        """Two evaluations of the same split produce one warning, not one per epoch and track."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10_000)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        metrics = _metrics_with_per_class({0: 0.6, 1: 0.0})

        _compute(cb, merged, metrics)
        _compute(cb, merged, metrics)

        assert len(_messages(warning)) == 1

    def test_message_names_gated_map_and_f1_sweep(self, mocker: MockerFixture) -> None:
        """The single warning covers both quantities the floor gates."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10_000)

        _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: 0.6}))

        (text,) = _messages(warning)
        assert "mAP_50_95_gated" in text
        assert "F1 sweep" in text

    def test_on_fit_start_rearms_the_warning(self, mocker: MockerFixture) -> None:
        """A new fit warns again."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10_000)
        merged = {0: _matching_entry(100)}
        metrics = _metrics_with_per_class({0: 0.6})

        _compute(cb, merged, metrics)
        cb.on_fit_start(_make_trainer(), _cpu_module())
        _compute(cb, merged, metrics)

        assert len(_messages(warning)) == 2

    def test_silent_off_rank_zero(self, mocker: MockerFixture) -> None:
        """A non-zero rank logs nothing for an empty eligible set."""
        warning = mocker.patch(_WARNING)
        mocker.patch("rfdetr.training.callbacks.coco_eval.is_main_process", return_value=False)
        trainer = _make_trainer()
        trainer.is_global_zero = False
        cb = COCOEvalCallback(min_class_support=10_000)

        _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: 0.6}), trainer)

        assert not _messages(warning)

    def test_no_warning_when_a_class_qualifies(self, mocker: MockerFixture) -> None:
        """An eligible class means neither fallback message fires."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10)

        _, print_tables = _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: 0.6}))

        assert not _messages(warning)
        assert print_tables.call_args.args[2]["gated applied"] == 1.0


class TestNoValidPerClassAp:
    """Eligible classes without a valid AP (no detections) fall back with their own message and flag."""

    def test_sentinel_aps_use_the_short_message_and_mark_gate_unapplied(self, mocker: MockerFixture) -> None:
        """Zero detections give every class AP -1: not an 'empty eligible set', and the gate did not apply."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10)
        metrics = _metrics_with_per_class({0: -1.0, 1: -1.0})

        trainer, print_tables = _compute(cb, {0: _matching_entry(100), 1: _matching_entry(50)}, metrics)

        (text,) = _messages(warning)
        assert "no valid per-class" in text
        assert "min_class_support" not in text
        assert print_tables.call_args.args[2]["gated applied"] == 0.0
        assert float(trainer.callback_metrics["val/mAP_50_95_gated"]) == pytest.approx(0.4)

    def test_sentinel_message_is_silent_off_rank_zero(self, mocker: MockerFixture) -> None:
        """The all-sentinel message is rank-zero only."""
        warning = mocker.patch(_WARNING)
        mocker.patch("rfdetr.training.callbacks.coco_eval.is_main_process", return_value=False)
        cb = COCOEvalCallback(min_class_support=10)

        _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: -1.0}))

        assert not _messages(warning)

    def test_applied_flag_is_reported_through_gated_map(self) -> None:
        """``_gated_map`` keeps returning a float and reports whether the gate applied through ``applied``."""
        cb = COCOEvalCallback(min_class_support=10)
        applied: list[bool] = []
        metrics: dict[str, Any] = _metrics_with_per_class({0: 0.6, 1: 0.0})

        value = cb._gated_map(metrics, "", {0}, "val", applied)

        assert value == pytest.approx(0.6)
        assert applied == [True]
        assert isinstance(cb._gated_map(metrics, "", {0}, "val"), float)
