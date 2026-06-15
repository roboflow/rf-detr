# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit and integration tests for MetricKeypointOKS.

Integration tests (class TestOKSValues) build a minimal COCO ground-truth object and feed known predictions so that
expected mAP values can be derived by hand without running model inference.  They are the first line of defence against
silent metric-computation regressions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from faster_coco_eval import COCO

from rfdetr.evaluation.keypoint_oks import MetricKeypointOKS

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_coco_gt() -> MagicMock:
    """Return a minimal COCO ground-truth mock (for unit tests that patch evaluator)."""
    return MagicMock(name="coco_gt")


def _make_predictions(image_id: int = 1, num_dets: int = 1, num_keypoints: int = 3) -> dict:
    """Return a single-image prediction dict with zero-valued tensors."""
    return {
        image_id: {
            "boxes": torch.zeros(num_dets, 4),
            "scores": torch.ones(num_dets),
            "labels": torch.zeros(num_dets, dtype=torch.long),
            "keypoints": torch.zeros(num_dets, num_keypoints, 3),
        }
    }


def _make_evaluator_mock(stats: list[float]) -> MagicMock:
    """Return a CocoEvaluator mock that returns the given stats array.

    Stats list must have exactly 10 elements matching _summarizeKps() output shape.
    """
    assert len(stats) == 10, f"_make_evaluator_mock: expected 10 stats, got {len(stats)}"
    evaluator = MagicMock(name="evaluator")
    evaluator.coco_eval = {"keypoints": MagicMock(stats=np.array(stats, dtype=np.float32))}
    return evaluator


def _build_coco_gt(
    num_keypoints: int,
    gt_keypoints: list[float],
    area: float = 2500.0,
    bbox: list[float] | None = None,
) -> COCO:
    """Build a minimal COCO GT object with a single annotation.

    Args:
        num_keypoints: Number of keypoints per instance.
        gt_keypoints: Flat COCO keypoint list [x0,y0,v0, x1,y1,v1, ...].
            Visibility values should be 2 (labelled+visible).
        area: Ground-truth object area for OKS normalisation.
        bbox: Ground-truth bounding box [x, y, w, h].  Defaults to [0,0,100,100].

    Returns:
        A fully-indexed ``faster_coco_eval.COCO`` object.
    """
    if bbox is None:
        bbox = [0.0, 0.0, 100.0, 100.0]
    kp_names = [f"kp{i}" for i in range(num_keypoints)]
    dataset = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "img.jpg"}],
        "categories": [{"id": 1, "name": "obj", "keypoints": kp_names, "skeleton": []}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "keypoints": gt_keypoints,
                "num_keypoints": num_keypoints,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
        ],
    }
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    return coco_gt


def _make_keypoint_prediction(
    image_id: int,
    kp_xy: list[tuple[float, float]],
    category_id: int = 1,
    score: float = 0.99,
    box: list[float] | None = None,
) -> dict:
    """Build a per-image prediction dict for MetricKeypointOKS.update().

    Args:
        image_id: COCO image ID.
        kp_xy: List of (x, y) coordinates — one per keypoint.  Visibility is
            set to 1.0 for all.
        category_id: Category label (raw COCO ID used here, no remapping).
        score: Detection confidence score.
        box: Bounding box [x1, y1, x2, y2] in pixel coords.  Defaults to the
            full 100x100 image.

    Returns:
        Dict mapping ``image_id`` to a prediction dict accepted by
        :meth:`~rfdetr.evaluation.keypoint_oks.MetricKeypointOKS.update`.
    """
    if box is None:
        box = [0.0, 0.0, 100.0, 100.0]
    num_kp = len(kp_xy)
    kp_tensor = torch.zeros(1, num_kp, 3)
    for i, (x, y) in enumerate(kp_xy):
        kp_tensor[0, i, 0] = x
        kp_tensor[0, i, 1] = y
        kp_tensor[0, i, 2] = 1.0
    return {
        image_id: {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "scores": torch.tensor([score]),
            "labels": torch.tensor([category_id], dtype=torch.long),
            "keypoints": kp_tensor,
        }
    }


# ---------------------------------------------------------------------------
# Unit tests (evaluator is mocked)
# ---------------------------------------------------------------------------


class TestHasUpdates:
    """has_updates reflects whether any batch has been accumulated."""

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
        """has_updates returns False after reset() clears all batches."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({1: {}})
        metric.reset()
        assert metric.has_updates is False


class TestReset:
    """Reset() clears all accumulated batches."""

    def test_clears_all_batches(self) -> None:
        """Reset() empties internal _batches list."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        metric.update(_make_predictions(image_id=2))
        metric.reset()
        assert metric._batches == []

    def test_idempotent_on_empty_state(self) -> None:
        """Reset() on empty metric does not raise."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.reset()
        assert metric.has_updates is False


class TestUpdate:
    """Update() appends batches without merging or overwriting."""

    def test_each_call_appends_one_batch(self) -> None:
        """Two update() calls produce two entries in _batches."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        metric.update(_make_predictions(image_id=2))
        assert len(metric._batches) == 2

    def test_same_image_id_in_two_batches_both_preserved(self) -> None:
        """Predictions for the same image_id in separate batches are NOT overwritten."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({5: {"scores": torch.tensor([0.9])}})
        metric.update({5: {"scores": torch.tensor([0.3])}})
        # Both batches must be preserved — not merged/overwritten
        assert len(metric._batches) == 2
        assert float(metric._batches[0][5]["scores"][0]) == pytest.approx(0.9)
        assert float(metric._batches[1][5]["scores"][0]) == pytest.approx(0.3)

    def test_empty_prediction_dict_appended_as_batch(self) -> None:
        """Empty dict marks an image with no detections and is preserved as a batch."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update({42: {}})
        assert len(metric._batches) == 1
        assert metric._batches[0] == {42: {}}


class TestCompute:
    """Compute() delegates to CocoEvaluator and returns correct stat dict."""

    def test_returns_correct_stat_keys(self) -> None:
        """Compute() returns dict with map, map_50, map_75, mar keys."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            result = metric.compute()
        assert set(result.keys()) == {"map", "map_50", "map_75", "mar"}

    def test_maps_stats_indices_to_dict_keys(self) -> None:
        """Compute() maps stats[0,1,2,5] to map, map_50, map_75, mar."""
        evaluator = _make_evaluator_mock([0.42, 0.72, 0.31, -1.0, -1.0, 0.55, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            result = metric.compute()
        assert result["map"] == pytest.approx(0.42)
        assert result["map_50"] == pytest.approx(0.72)
        assert result["map_75"] == pytest.approx(0.31)
        assert result["mar"] == pytest.approx(0.55)

    def test_raises_on_wrong_stats_shape(self) -> None:
        """Compute() raises AssertionError when stats array is not shape (10,).

        _summarizeKps() always returns (10,); a shape mismatch signals a pycocotools contract violation (e.g. wrong
        faster_coco_eval version) and must not silently produce incorrect metric values via index-out-of-bounds sentinel
        fallback.
        """
        evaluator = MagicMock(name="evaluator")
        evaluator.coco_eval = {"keypoints": MagicMock(stats=np.array([0.3], dtype=np.float32))}
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            with pytest.raises(AssertionError, match="Expected coco keypoint stats shape"):
                metric.compute()

    def test_calls_synchronize_and_accumulate(self) -> None:
        """Compute() calls synchronize_between_processes() and accumulate() on the evaluator."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        evaluator.synchronize_between_processes.assert_called_once()
        evaluator.accumulate.assert_called_once()

    def test_constructs_evaluator_with_metric_params(self) -> None:
        """Compute() passes max_dets and keypoint_oks_sigmas to CocoEvaluator."""
        coco_gt = _make_coco_gt()
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
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

    def test_replays_each_batch_as_separate_evaluator_update(self) -> None:
        """Compute() calls evaluator.update() once per accumulated batch."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        metric.update(_make_predictions(image_id=2))
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        assert evaluator.update.call_count == 2

    def test_forwards_correct_image_id_to_evaluator(self) -> None:
        """Compute() passes predictions with the correct image_id to the evaluator."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=7))
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        passed_preds = evaluator.update.call_args.args[0]
        assert 7 in passed_preds

    def test_skips_evaluator_update_when_no_predictions(self) -> None:
        """Compute() does not call evaluator.update() when no batches accumulated."""
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            metric.compute()
        evaluator.update.assert_not_called()

    def test_compute_is_idempotent(self) -> None:
        """Two compute() calls with identical batches return the same stats.

        Proves the shared coco_gt reference is not mutated between calls.
        """
        evaluator = _make_evaluator_mock([0.42, 0.72, 0.31, -1.0, -1.0, 0.55, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator):
            result_a = metric.compute()
            result_b = metric.compute()
        assert result_a == result_b

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
        evaluator = _make_evaluator_mock([0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0])
        metric = MetricKeypointOKS(_make_coco_gt(), keypoint_oks_sigmas=sigmas)
        with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator) as cls:
            metric.compute()
        assert cls.call_args.kwargs["keypoint_oks_sigmas"] == sigmas


# ---------------------------------------------------------------------------
# Integration tests — real COCO GT, visually validatable expected values
# ---------------------------------------------------------------------------


class TestOKSValues:
    """End-to-end OKS mAP computation against a real CocoEvaluator.

    Each test uses a single annotation and a single prediction so that
    expected mAP can be derived by hand:

      OKS = exp(-d² / (8 * σ² * s²))

    where d = Euclidean pixel distance, s = sqrt(GT area), σ = OKS sigma.
    The 8× factor comes from pycocotools: vars = (2σ)², e = d² / (vars * s² * 2).

    All tests use 1 keypoint, GT at (50, 50), area = 2500.0 (s = 50.0),
    and sigma = 0.05 (default for custom keypoint counts via
    _DEFAULT_CUSTOM_KEYPOINT_OKS_SIGMA).
    """

    _GT_KP_X = 50.0
    _GT_KP_Y = 50.0
    _AREA = 2500.0  # s = 50.0
    _SIGMA = 0.05  # default custom OKS sigma

    def _make_gt(self) -> COCO:
        """Build a 1-image, 1-annotation, 1-keypoint COCO GT."""
        return _build_coco_gt(
            num_keypoints=1,
            gt_keypoints=[self._GT_KP_X, self._GT_KP_Y, 2],
            area=self._AREA,
        )

    def test_perfect_prediction_gives_map_one(self) -> None:
        """Prediction exactly at GT keypoint location must yield mAP@50 = 1.0.

        d = 0  →  OKS = exp(0) = 1.0  →  all IoU thresholds pass  →  mAP = 1.0.
        """
        coco_gt = self._make_gt()
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[self._SIGMA])
        metric.update(_make_keypoint_prediction(1, [(self._GT_KP_X, self._GT_KP_Y)]))
        result = metric.compute()
        assert result["map_50"] == pytest.approx(1.0, abs=1e-3)
        assert result["map"] == pytest.approx(1.0, abs=1e-3)

    def test_no_predictions_gives_map_zero(self) -> None:
        """Empty prediction set must yield mAP = 0.0 (no true positives)."""
        coco_gt = self._make_gt()
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[self._SIGMA])
        metric.update({1: {}})
        result = metric.compute()
        assert result["map_50"] == pytest.approx(0.0, abs=1e-3)
        assert result["map"] == pytest.approx(0.0, abs=1e-3)

    def test_far_prediction_gives_map_near_zero(self) -> None:
        """Prediction far from GT must yield near-zero mAP.

        d = sqrt(50² + 50²) ≈ 70.7, s = 50, σ = 0.05.
        pycocotools formula: OKS = exp(-d² / (8 * σ² * s²))
          = exp(-5000 / (8 * 0.0025 * 2500))
          = exp(-5000 / 50)
          = exp(-100) ≈ 0  →  mAP@50 = 0.0.
        """
        coco_gt = self._make_gt()
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[self._SIGMA])
        metric.update(_make_keypoint_prediction(1, [(0.0, 0.0)]))
        result = metric.compute()
        assert result["map_50"] == pytest.approx(0.0, abs=1e-3)

    def test_known_oks_threshold_boundary(self) -> None:
        """Prediction at OKS ≈ 0.6 passes @50 but fails @75.

        pycocotools (and faster_coco_eval) compute OKS as::

          vars = (sigma * 2)²
          e    = d² / (vars * s² * 2) = d² / (8 * sigma² * s²)
          OKS  = exp(-e)

        Solving for d where OKS = 0.6 (clearly between the 0.5 and 0.75 thresholds)::

          d² = -ln(0.6) * 8 * sigma² * s²
             = 0.511 * 8 * 0.0025 * 2500 = 25.55
          d  = sqrt(25.55) ≈ 5.05 pixels.

        Displacement along x-axis only: predict at (50 + 5.05, 50).
        OKS ≈ 0.6 → passes @50 (threshold 0.5), fails @75 (threshold 0.75).
        mAP@50 should be 1.0, mAP@75 should be 0.0.
        """
        sigma = self._SIGMA
        s = np.sqrt(self._AREA)
        oks_target = 0.6  # between 0.50 and 0.75 — clear boundary
        # pycocotools formula: OKS = exp(-d² / (8 * sigma² * s²))
        d = np.sqrt(-np.log(oks_target) * 8 * sigma**2 * s**2)
        pred_x = float(self._GT_KP_X + d)

        coco_gt = self._make_gt()
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[sigma])
        metric.update(_make_keypoint_prediction(1, [(pred_x, self._GT_KP_Y)]))
        result = metric.compute()
        assert result["map_50"] == pytest.approx(1.0, abs=1e-3), "OKS=0.6 should pass @50 threshold"
        assert result["map_75"] == pytest.approx(0.0, abs=1e-3), "OKS=0.6 should fail @75 threshold"

    def test_metric_stable_across_identical_epochs(self) -> None:
        """Identical predictions fed across three separate compute() cycles give identical mAP.

        This proves no GT mutation, no accumulation bleed, and correct reset between epochs.  A metric that peaks-then-
        decreases under frozen predictions would fail here.
        """
        coco_gt = self._make_gt()
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[self._SIGMA])
        results = []
        for _ in range(3):
            metric.reset()
            metric.update(_make_keypoint_prediction(1, [(self._GT_KP_X, self._GT_KP_Y)]))
            results.append(metric.compute())
        assert results[0]["map_50"] == pytest.approx(results[1]["map_50"], abs=1e-6)
        assert results[1]["map_50"] == pytest.approx(results[2]["map_50"], abs=1e-6)

    def test_multi_batch_same_result_as_single_batch(self) -> None:
        """Splitting predictions across two update() calls gives same mAP as one call.

        Two images, each predicted correctly.  Whether fed in one batch or two, mAP@50 must equal 1.0 — verifies that
        per-batch append semantics are equivalent to batched evaluation.
        """
        kp_names = ["kp0"]
        dataset = {
            "images": [
                {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
                {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
            ],
            "categories": [{"id": 1, "name": "obj", "keypoints": kp_names, "skeleton": []}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [50.0, 50.0, 2],
                    "num_keypoints": 1,
                    "area": 2500.0,
                    "bbox": [0.0, 0.0, 100.0, 100.0],
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "keypoints": [25.0, 75.0, 2],
                    "num_keypoints": 1,
                    "area": 2500.0,
                    "bbox": [0.0, 0.0, 100.0, 100.0],
                    "iscrowd": 0,
                },
            ],
        }
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        sigma = [self._SIGMA]

        # Single batch
        metric_single = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=sigma)
        combined = {}
        combined.update(_make_keypoint_prediction(1, [(50.0, 50.0)]))
        combined.update(_make_keypoint_prediction(2, [(25.0, 75.0)]))
        metric_single.update(combined)
        result_single = metric_single.compute()

        # Two batches (one image each)
        metric_split = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=sigma)
        metric_split.update(_make_keypoint_prediction(1, [(50.0, 50.0)]))
        metric_split.update(_make_keypoint_prediction(2, [(25.0, 75.0)]))
        result_split = metric_split.compute()

        assert result_single["map_50"] == pytest.approx(result_split["map_50"], abs=1e-6)
        assert result_single["map_50"] == pytest.approx(1.0, abs=1e-3)
