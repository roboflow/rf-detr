# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for MetricKeypointOKS."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rfdetr.evaluation.keypoint_oks import MetricKeypointOKS


def _make_coco_gt() -> MagicMock:
    """Return a minimal COCO ground-truth mock."""
    return MagicMock(name="coco_gt")


def _make_predictions(image_id: int = 1, num_dets: int = 1, num_keypoints: int = 3) -> dict:
    """Return a single-image prediction dict."""
    return {
        image_id: {
            "boxes": torch.zeros(num_dets, 4),
            "scores": torch.ones(num_dets),
            "labels": torch.zeros(num_dets, dtype=torch.long),
            "keypoints": torch.zeros(num_dets, num_keypoints, 3),
        }
    }


def _make_evaluator_mock(stats: list[float]) -> MagicMock:
    """Return a CocoEvaluator mock returning the given stats array."""
    evaluator = MagicMock(name="evaluator")
    evaluator.coco_eval = {"keypoints": MagicMock(stats=np.array(stats, dtype=np.float32))}
    return evaluator


class TestHasUpdates:
    """has_updates reflects whether predictions are accumulated."""

    def test_false_on_construction(self) -> None:
        """Fresh metric reports no updates."""
        metric = MetricKeypointOKS(_make_coco_gt())
        assert metric.has_updates is False

    def test_true_after_update(self) -> None:
        """has_updates becomes True after any update() call."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({1: {}})
        assert metric.has_updates is True

    def test_false_after_reset(self) -> None:
        """has_updates returns False after reset() clears accumulated preds."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({1: {}})
        metric.reset()
        assert metric.has_updates is False


class TestReset:
    """Reset() clears all accumulated predictions."""

    def test_clears_accumulated_predictions(self) -> None:
        """Reset() empties internal _preds dict."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        metric.update(_make_predictions(image_id=2))
        metric.reset()
        assert metric._preds == {}

    def test_idempotent_on_empty_state(self) -> None:
        """Reset() on empty metric does not raise."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.reset()
        assert metric.has_updates is False


class TestUpdate:
    """Update() accumulates predictions across multiple calls."""

    def test_merges_image_ids_across_batches(self) -> None:
        """Multiple update() calls accumulate distinct image_ids."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        metric.update(_make_predictions(image_id=2))
        assert 1 in metric._preds
        assert 2 in metric._preds

    def test_overwrites_existing_image_id(self) -> None:
        """Updating the same image_id twice keeps the latest prediction."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({1: {"scores": torch.tensor([0.9])}})
        metric.update({1: {"scores": torch.tensor([0.5])}})
        assert float(metric._preds[1]["scores"][0]) == pytest.approx(0.5)

    def test_empty_prediction_dict_marks_image(self) -> None:
        """Empty dict for image_id registers the image as having no detections."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({42: {}})
        assert 42 in metric._preds
        assert metric._preds[42] == {}


class TestCompute:
    """Compute() delegates to CocoEvaluator and returns correct stat dict."""

    def test_returns_correct_stat_keys(self) -> None:
        """Compute() returns dict with map, map_50, map_75, mar keys."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            result = metric.compute()
        assert set(result.keys()) == {"map", "map_50", "map_75", "mar"}

    def test_maps_stats_indices_to_dict_keys(self) -> None:
        """Compute() maps stats[0,1,2,5] to map, map_50, map_75, mar."""
        evaluator = _make_evaluator_mock([0.42, 0.72, 0.31, -1.0, -1.0, 0.55])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            result = metric.compute()
        assert result["map"] == pytest.approx(0.42)
        assert result["map_50"] == pytest.approx(0.72)
        assert result["map_75"] == pytest.approx(0.31)
        assert result["mar"] == pytest.approx(0.55)

    def test_returns_minus_one_for_short_stats(self) -> None:
        """Compute() returns -1.0 for any stat index beyond the stats array length."""
        evaluator = _make_evaluator_mock([0.3])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            result = metric.compute()
        assert result["map"] == pytest.approx(0.3)
        assert result["map_50"] == pytest.approx(-1.0)
        assert result["map_75"] == pytest.approx(-1.0)
        assert result["mar"] == pytest.approx(-1.0)

    def test_calls_synchronize_and_accumulate(self) -> None:
        """Compute() calls synchronize_between_processes() and accumulate() on the evaluator."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        evaluator.synchronize_between_processes.assert_called_once()
        evaluator.accumulate.assert_called_once()

    def test_constructs_evaluator_with_metric_params(self) -> None:
        """Compute() passes max_dets and keypoint_oks_sigmas to CocoEvaluator."""
        coco_gt = _make_coco_gt()
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6])
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[0.05, 0.1], max_dets=100)
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator) as cls:
            metric.compute()
        cls.assert_called_once_with(
            coco_gt,
            ["keypoints"],
            max_dets=100,
            keypoint_oks_sigmas=[0.05, 0.1],
            log_summary=False,
        )

    def test_forwards_accumulated_predictions_to_evaluator(self) -> None:
        """Compute() replays all accumulated predictions into the CocoEvaluator."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6])
        metric = MetricKeypointOKS(_make_coco_gt())
        preds = _make_predictions(image_id=7)
        metric.update(preds)
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        evaluator.update.assert_called_once()
        passed_preds = evaluator.update.call_args.args[0]
        assert 7 in passed_preds

    def test_skips_evaluator_update_when_no_predictions(self) -> None:
        """Compute() does not call evaluator.update() when no predictions accumulated."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        evaluator.update.assert_not_called()

    @pytest.mark.parametrize(
        "sigmas",
        [
            pytest.param(None, id="no_sigmas"),
            pytest.param([0.05] * 17, id="17kp_sigmas"),
            pytest.param([0.05] * 4, id="4kp_sigmas"),
        ],
    )
    def test_compute_accepts_arbitrary_keypoint_counts(self, sigmas: list[float] | None) -> None:
        """Compute() passes any keypoint_oks_sigmas length to CocoEvaluator without restriction."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6])
        metric = MetricKeypointOKS(_make_coco_gt(), keypoint_oks_sigmas=sigmas)
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator) as cls:
            metric.compute()
        assert cls.call_args.kwargs["keypoint_oks_sigmas"] == sigmas
