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

import contextlib
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from faster_coco_eval import COCO

from rfdetr.evaluation.coco_eval import CocoEvaluator
from rfdetr.evaluation.keypoint_oks import MetricKeypointOKS
from tests._markers import _requires_pycocotools

__doctest_requires__ = {"TestGroupedKeypointStats._pycocotools_mean": ["pycocotools"]}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_coco_gt() -> MagicMock:
    """Return a minimal COCO ground-truth mock (for unit tests that patch evaluator).

    Examples:
        >>> gt = _make_coco_gt()
        >>> gt._mock_name
        'coco_gt'
    """
    return MagicMock(name="coco_gt")


def _make_predictions(image_id: int = 1, num_dets: int = 1, num_keypoints: int = 3) -> dict:
    """Return a single-image prediction dict with zero-valued tensors.

    Examples:
        >>> preds = _make_predictions(image_id=5, num_dets=2, num_keypoints=3)
        >>> list(preds.keys())
        [5]
        >>> preds[5]["boxes"].shape
        torch.Size([2, 4])
        >>> preds[5]["keypoints"].shape
        torch.Size([2, 3, 3])
    """
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

    Examples:
        >>> mock = _make_evaluator_mock([float(i) for i in range(10)])
        >>> mock.coco_eval["keypoints"].stats.tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
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

    Examples:
        >>> gt = _build_coco_gt(2, [10.0, 20.0, 2.0, 30.0, 40.0, 2.0])
        >>> list(gt.imgs.keys())
        [1]
        >>> len(gt.anns)
        1
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

    Examples:
        >>> pred = _make_keypoint_prediction(3, [(10.0, 20.0), (30.0, 40.0)])
        >>> list(pred.keys())
        [3]
        >>> pred[3]["keypoints"].shape
        torch.Size([1, 2, 3])
        >>> round(float(pred[3]["scores"][0]), 2)
        0.99
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


_DEFAULT_STATS = [0.5, 0.7, 0.4, -1.0, -1.0, 0.6, -1.0, -1.0, -1.0, -1.0]


@pytest.fixture
def evaluator_cls(mocker):
    """Patch ``CocoEvaluator`` in keypoint_oks and return the class mock; its instance reports ``_DEFAULT_STATS``."""
    return mocker.patch(
        "rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=_make_evaluator_mock(_DEFAULT_STATS)
    )


class TestCompute:
    """Compute() delegates to CocoEvaluator and returns correct stat dict."""

    def test_returns_correct_stat_keys(self, evaluator_cls) -> None:
        """Compute() returns dict with map, map_50, map_75, mar keys."""
        metric = MetricKeypointOKS(_make_coco_gt())
        result = metric.compute()
        assert set(result.keys()) == {"map", "map_50", "map_75", "mar"}

    def test_maps_stats_indices_to_dict_keys(self, mocker) -> None:
        """Compute() maps stats[0,1,2,5] to map, map_50, map_75, mar."""
        evaluator = _make_evaluator_mock([0.42, 0.72, 0.31, -1.0, -1.0, 0.55, -1.0, -1.0, -1.0, -1.0])
        mocker.patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator)
        metric = MetricKeypointOKS(_make_coco_gt())
        result = metric.compute()
        assert result["map"] == pytest.approx(0.42)
        assert result["map_50"] == pytest.approx(0.72)
        assert result["map_75"] == pytest.approx(0.31)
        assert result["mar"] == pytest.approx(0.55)

    def test_raises_on_wrong_stats_shape(self, mocker) -> None:
        """Compute() raises AssertionError when stats array is not shape (10,).

        _summarizeKps() always returns (10,); a shape mismatch signals a pycocotools contract violation (e.g. wrong
        faster_coco_eval version) and must not silently produce incorrect metric values via index-out-of-bounds sentinel
        fallback.
        """
        evaluator = mocker.MagicMock(name="evaluator")
        evaluator.coco_eval = {"keypoints": mocker.MagicMock(stats=np.array([0.3], dtype=np.float32))}
        mocker.patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator)
        metric = MetricKeypointOKS(_make_coco_gt())
        with pytest.raises(AssertionError, match="Expected coco keypoint stats shape"):
            metric.compute()

    def test_calls_synchronize_and_accumulate(self, evaluator_cls) -> None:
        """Compute() calls synchronize_between_processes() and accumulate() on the evaluator."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.compute()
        evaluator_cls.return_value.synchronize_between_processes.assert_called_once()
        evaluator_cls.return_value.accumulate.assert_called_once()

    def test_constructs_evaluator_with_metric_params(self, evaluator_cls) -> None:
        """Compute() passes max_dets and keypoint_oks_sigmas to CocoEvaluator."""
        coco_gt = _make_coco_gt()
        metric = MetricKeypointOKS(coco_gt, keypoint_oks_sigmas=[0.05, 0.1], max_dets=100)
        metric.compute()
        evaluator_cls.assert_called_once_with(
            coco_gt,
            ["keypoints"],
            max_dets=100,
            keypoint_oks_sigmas=[0.05, 0.1],
            log_summary=False,
        )

    def test_replays_each_batch_as_separate_evaluator_update(self, evaluator_cls) -> None:
        """Compute() calls evaluator.update() once per accumulated batch."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
        metric.update(_make_predictions(image_id=2))
        metric.compute()
        assert evaluator_cls.return_value.update.call_count == 2

    def test_forwards_correct_image_id_to_evaluator(self, evaluator_cls) -> None:
        """Compute() passes predictions with the correct image_id to the evaluator."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=7))
        metric.compute()
        assert 7 in evaluator_cls.return_value.update.call_args.args[0]

    def test_skips_evaluator_update_when_no_predictions(self, evaluator_cls) -> None:
        """Compute() does not call evaluator.update() when no batches accumulated."""
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.compute()
        evaluator_cls.return_value.update.assert_not_called()

    def test_compute_is_idempotent(self, mocker) -> None:
        """Two compute() calls with identical batches return the same stats.

        Proves the shared coco_gt reference is not mutated between calls.
        """
        evaluator = _make_evaluator_mock([0.42, 0.72, 0.31, -1.0, -1.0, 0.55, -1.0, -1.0, -1.0, -1.0])
        mocker.patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=evaluator)
        metric = MetricKeypointOKS(_make_coco_gt())
        metric.update(_make_predictions(image_id=1))
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
    def test_compute_accepts_arbitrary_keypoint_counts(self, evaluator_cls, sigmas: list[float] | None) -> None:
        """Compute() passes any keypoint_oks_sigmas length to CocoEvaluator without restriction."""
        metric = MetricKeypointOKS(_make_coco_gt(), keypoint_oks_sigmas=sigmas)
        metric.compute()
        assert evaluator_cls.call_args.kwargs["keypoint_oks_sigmas"] == sigmas


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


class TestGroupedKeypointStats:
    """Mixed keypoint counts: every stat must be the mean over the categories COCO would score for that stat.

    In most cases category 1 gets a perfect detection (AP 1.0) and the last category gets one with its keypoints far
    away (AP 0.0). COCO leaves out categories with no scorable ground truth, so those cases come to 0.5.
    """

    #: Keys ``MetricKeypointOKS.compute`` returns.
    _METRIC_KEYS = ("map", "map_50", "map_75", "mar")

    @staticmethod
    def _annotation(
        annotation_id: int, image_id: int, category_id: int, count: int, side: int = 100, **overrides: int
    ) -> dict:
        """Return one ground-truth annotation with a square box, whose ``side`` sets the COCO area range.

        Examples:
            >>> annotation = TestGroupedKeypointStats._annotation(1, 2, 3, 2, side=60, iscrowd=1)
            >>> annotation["area"], annotation["iscrowd"], annotation["keypoints"]
            (3600, 1, [110, 110, 2, 120, 120, 2])
        """
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [100, 100, side, side],
            "area": side * side,
            "iscrowd": 0,
            "keypoints": [value for index in range(count) for value in (110 + 10 * index, 110 + 10 * index, 2)],
            "num_keypoints": count,
            **overrides,
        }

    def _zero_keypoint_annotation(self, annotation_id: int, image_id: int, category_id: int, count: int) -> dict:
        """Return an annotation with no visible keypoints, which COCO ignores.

        Examples:
            >>> annotation = TestGroupedKeypointStats()._zero_keypoint_annotation(1, 2, 3, 1)
            >>> annotation["num_keypoints"], annotation["keypoints"]
            (0, [0, 0, 0])
        """
        annotation = self._annotation(annotation_id, image_id, category_id, count, num_keypoints=0)
        annotation["keypoints"] = [0, 0, 0] * count
        return annotation

    @staticmethod
    def _coco_gt(category_counts: dict[int, int], annotations: list[dict]) -> COCO:
        """Return a COCO ground truth with images 1 to 4, for categories given as ``{id: keypoint count}``.

        Examples:
            >>> coco_gt = TestGroupedKeypointStats._coco_gt({1: 3, 5: 4}, [])
            >>> sorted(coco_gt.cats), len(coco_gt.imgs)
            ([1, 5], 4)
        """
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": image_id, "width": 500, "height": 500} for image_id in range(1, 5)],
            "categories": [
                {"id": category_id, "name": str(category_id), "keypoints": [f"k{point}" for point in range(count)]}
                for category_id, count in category_counts.items()
            ],
            "annotations": annotations,
        }
        coco_gt.createIndex()
        return coco_gt

    @staticmethod
    def _prediction(image_id: int, label: int, count: int, start: int, width: int) -> dict:
        """Return one detection for ``image_id`` with keypoints on a diagonal, padded to ``width`` points.

        Examples:
            >>> TestGroupedKeypointStats._prediction(1, 1, 3, 110, 4)[1]["keypoints"].shape
            torch.Size([1, 4, 3])
        """
        diagonal = [start + 10.0 * index for index in range(count)]
        points = list(zip(diagonal, diagonal)) + [(0.0, 0.0)] * (width - count)
        return _make_keypoint_prediction(
            image_id, points, category_id=label, score=0.9, box=[100.0, 100.0, 200.0, 200.0]
        )

    def _metric_result(self, category_counts: dict[int, int], annotations: list[dict], predictions: list[dict]) -> dict:
        """Return what ``MetricKeypointOKS`` computes for these predictions.

        Examples:
            >>> test = TestGroupedKeypointStats()
            >>> annotations = [test._annotation(1, 1, 1, 3)]
            >>> test._metric_result({1: 3}, annotations, [test._prediction(1, 1, 3, 110, 3)])["map"]
            1.0
        """
        metric = MetricKeypointOKS(
            self._coco_gt(category_counts, annotations), keypoint_oks_sigmas=[0.05] * max(category_counts.values())
        )
        for prediction in predictions:
            metric.update(prediction)
        return metric.compute()

    def _evaluator(self, category_counts: dict[int, int], annotations: list[dict]) -> CocoEvaluator:
        """Return a ``CocoEvaluator`` for these categories and annotations.

        Examples:
            >>> TestGroupedKeypointStats()._evaluator({1: 3}, []).iou_types
            ['keypoints']
        """
        return CocoEvaluator(
            self._coco_gt(category_counts, annotations),
            ["keypoints"],
            keypoint_oks_sigmas=[0.05] * max(category_counts.values()),
            log_summary=False,
        )

    def _stats(
        self,
        category_counts: dict[int, int],
        annotations: list[dict],
        predictions: list[dict],
        image_ids: list[int],
    ) -> np.ndarray:
        """Return all keypoint stats from one ``CocoEvaluator`` over ``image_ids``.

        Examples:
            >>> test = TestGroupedKeypointStats()
            >>> annotations = [test._annotation(1, 1, 1, 3)]
            >>> test._stats({1: 3}, annotations, [test._prediction(1, 1, 3, 110, 3)], [1]).round(3).tolist()
            [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
        """
        evaluator = self._evaluator(category_counts, annotations)
        evaluator.update(dict.fromkeys(image_ids, {}))
        for prediction in predictions:
            evaluator.update(prediction)
        evaluator.accumulate()
        return np.asarray(evaluator.coco_eval["keypoints"].stats, dtype=np.float64)

    def _pycocotools_mean(
        self,
        category_counts: dict[int, int],
        annotations: list[dict],
        predictions: list[dict],
        image_ids: list[int],
    ) -> np.ndarray:
        """Return each stat's mean over per-category pycocotools evaluations, skipping the ``-1`` values they report.

        One COCO evaluation can't score mixed keypoint counts, so this independent mean is the reference for grouped
        stats.

        Examples:
            >>> test = TestGroupedKeypointStats()
            >>> annotations = [test._annotation(1, 1, 1, 2), test._annotation(2, 2, 2, 3)]
            >>> predictions = [test._prediction(1, 1, 2, 110, 3), test._prediction(2, 2, 3, 310, 3)]
            >>> test._pycocotools_mean({1: 2, 2: 3}, annotations, predictions, [1, 2]).round(3).tolist()
            [0.5, 0.5, 0.5, -1.0, 0.5, 0.5, 0.5, 0.5, -1.0, 0.5]
        """
        import pycocotools.coco as pycoco
        from pycocotools.cocoeval import COCOeval

        evaluator = self._evaluator(category_counts, annotations)
        results = [result for prediction in predictions for result in evaluator.prepare(prediction, "keypoints")]
        per_category = []
        with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints its progress and summary
            for category_id, count in category_counts.items():
                ground_truth = pycoco.COCO()
                ground_truth.dataset = self._coco_gt(
                    {category_id: count},
                    [annotation for annotation in annotations if annotation["category_id"] == category_id],
                ).dataset
                ground_truth.createIndex()
                category_results = [result for result in results if result["category_id"] == category_id]
                if category_results:
                    detections = ground_truth.loadRes(category_results)
                else:  # loadRes rejects an empty list
                    detections = pycoco.COCO()
                    detections.dataset = {**ground_truth.dataset, "annotations": []}
                    detections.createIndex()
                coco_eval = COCOeval(ground_truth, detections, "keypoints")
                coco_eval.params.kpt_oks_sigmas = np.full(count, 0.05)
                coco_eval.params.imgIds = image_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                per_category.append(coco_eval.stats)
        return np.ma.masked_less(per_category, 0).mean(axis=0).filled(-1.0)

    @pytest.mark.parametrize(
        "category_counts",
        [
            pytest.param((3, 4), id="no_empty_category"),
            pytest.param((3, 3, 3), id="one_group_with_empty_category"),
            pytest.param((3, 3, 4), id="empty_category_beside_perfect_one"),
            pytest.param((3, 4, 4), id="empty_category_beside_missed_one"),
        ],
    )
    def test_mean_ignores_categories_without_ground_truth(self, category_counts: tuple[int, ...]) -> None:
        """Only the two categories with ground truth count, whichever group an empty category falls in."""
        last_id = len(category_counts)
        first_count, last_count = category_counts[0], category_counts[-1]
        width = max(category_counts)

        result = self._metric_result(
            dict(enumerate(category_counts, start=1)),
            [self._annotation(1, 1, 1, first_count), self._annotation(2, 2, last_id, last_count)],
            [self._prediction(1, 1, first_count, 110, width), self._prediction(2, last_id, last_count, 310, width)],
        )

        assert result == pytest.approx(dict.fromkeys(self._METRIC_KEYS, 0.5), abs=1e-3)

    @pytest.mark.parametrize("unscorable", ["crowd", "zero_keypoints", "image_not_evaluated"])
    def test_mean_ignores_categories_with_only_unscorable_ground_truth(self, unscorable: str) -> None:
        """Category 2 has annotations, but none COCO can score, so it must not count beside the perfect category 1."""
        annotation = {
            "crowd": self._annotation(3, 1, 2, 3, iscrowd=1),
            "zero_keypoints": self._zero_keypoint_annotation(3, 1, 2, 3),
            "image_not_evaluated": self._annotation(3, 3, 2, 3),
        }[unscorable]

        result = self._metric_result(
            {1: 3, 2: 3, 3: 4},
            [self._annotation(1, 1, 1, 3), self._annotation(2, 2, 3, 4), annotation],
            [self._prediction(1, 1, 3, 110, 4), self._prediction(2, 3, 4, 310, 4)],
        )

        assert result == pytest.approx(dict.fromkeys(self._METRIC_KEYS, 0.5), abs=1e-3)

    def test_mean_ignores_group_with_no_ground_truth(self) -> None:
        """A keypoint-count group with no annotations adds nothing, though its arrays hold no categories at all."""
        result = self._metric_result(
            {1: 3, 2: 3, 3: 4},
            [self._annotation(1, 1, 1, 3), self._annotation(2, 2, 2, 3)],
            [self._prediction(1, 1, 3, 110, 4), self._prediction(2, 2, 3, 310, 4)],
        )

        assert result == pytest.approx(dict.fromkeys(self._METRIC_KEYS, 0.5), abs=1e-3)

    def test_mean_counts_category_with_ground_truth_and_no_detection(self) -> None:
        """A category COCO can score but with no detection holds AP 0.0, and that 0.0 belongs in the mean."""
        result = self._metric_result(
            {1: 3, 2: 3, 3: 4},
            [self._annotation(1, 1, 1, 3), self._annotation(2, 2, 3, 4), self._annotation(3, 1, 2, 3)],
            [self._prediction(1, 1, 3, 110, 4), self._prediction(2, 3, 4, 310, 4)],
        )

        assert result == pytest.approx(dict.fromkeys(self._METRIC_KEYS, 1 / 3), abs=1e-3)

    def test_area_stats_count_only_categories_in_that_area_range(self) -> None:
        """Perfect medium and large 3-keypoint categories beside a missed large 4-keypoint one."""
        stats = self._stats(
            {1: 3, 2: 3, 3: 4},
            [self._annotation(1, 1, 1, 3, side=60), self._annotation(2, 2, 2, 3), self._annotation(3, 3, 3, 4)],
            [self._prediction(1, 1, 3, 110, 4), self._prediction(2, 2, 3, 110, 4), self._prediction(3, 3, 4, 310, 4)],
            [1, 2, 3],
        )

        all_area = [2 / 3, 2 / 3, 2 / 3]
        assert stats.tolist() == pytest.approx([*all_area, 1.0, 0.5, *all_area, 1.0, 0.5], abs=1e-3)

    def test_stat_no_group_has_stays_unavailable(self) -> None:
        """With only large ground truth no group has a medium value, so the medium stats stay ``-1``."""
        stats = self._stats(
            {1: 3, 2: 4},
            [self._annotation(1, 1, 1, 3), self._annotation(2, 2, 2, 4)],
            [self._prediction(1, 1, 3, 110, 4), self._prediction(2, 2, 4, 310, 4)],
            [1, 2],
        )

        assert stats.tolist() == pytest.approx([0.5, 0.5, 0.5, -1.0, 0.5, 0.5, 0.5, 0.5, -1.0, 0.5], abs=1e-3)

    def test_every_stat_stays_unavailable_with_no_annotations_in_any_group(self) -> None:
        """With no annotations the merged arrays hold no categories, which must still summarize to ``-1``."""
        stats = self._stats({1: 3, 2: 4}, [], [self._prediction(1, 1, 3, 110, 4)], [1])

        assert stats.tolist() == [-1.0] * 10

    def test_accumulate_again_summarizes_only_the_latest_groups(self) -> None:
        """A second ``accumulate`` replaces the first call's groups, so their arrays stay out of the merged stats."""
        evaluator = self._evaluator({1: 3, 2: 3, 3: 4}, [self._annotation(1, 1, 1, 3), self._annotation(2, 2, 3, 4)])
        evaluator.update(self._prediction(1, 1, 3, 110, 4))
        evaluator.update({2: {}})
        evaluator.accumulate()

        evaluator.update(self._prediction(2, 3, 4, 110, 4))
        evaluator.accumulate()

        assert evaluator.coco_eval["keypoints"].stats[0] == pytest.approx(1.0)

    def test_summary_leaves_group_evaluators_untouched(self) -> None:
        """The merged summary works on a copy, so each group keeps its own arrays and stats."""
        evaluator = self._evaluator({1: 3, 2: 4}, [self._annotation(1, 1, 1, 3), self._annotation(2, 2, 2, 4)])
        evaluator.update(self._prediction(1, 1, 3, 110, 4))
        evaluator.update(self._prediction(2, 2, 4, 310, 4))

        evaluator.accumulate()

        groups = evaluator.coco_eval["keypoints"].evals
        assert [(group.eval["precision"].shape[2], float(group.stats[0])) for group in groups] == [(1, 1.0), (1, 0.0)]

    @_requires_pycocotools
    def test_every_stat_matches_pycocotools_per_category_mean(self) -> None:
        """Every stat equals pycocotools' per-category mean, with all the kinds of unscorable category side by side."""
        # The three groups hold categories in a different ratio for each of the all, medium and large area ranges.
        category_counts = {1: 3, 2: 3, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 3}
        scored = [  # image, category, box side (60 is medium, 100 is large, 20 is small), keypoint start
            (1, 1, 60, 110),
            (2, 1, 60, 114),
            (1, 2, 100, 112),
            (2, 2, 100, 118),
            (3, 2, 100, 310),
            (1, 5, 60, 112),
            (2, 5, 100, 110),
            (3, 5, 100, 118),
            (1, 8, 60, 114),
            (2, 8, 100, 122),
            (3, 9, 20, 110),
            (1, 10, 60, 116),
            (1, 11, 100, 116),
        ]
        annotations = [
            self._annotation(annotation_id, image_id, category_id, category_counts[category_id], side=side)
            for annotation_id, (image_id, category_id, side, _) in enumerate(scored, start=1)
        ]
        annotations += [
            self._annotation(100, 1, 3, 3, iscrowd=1),  # category 3 has only a crowd annotation
            self._zero_keypoint_annotation(101, 2, 4, 3),  # category 4 has only a zero-keypoint annotation
            self._annotation(102, 4, 7, 4),  # category 7 is only on image 4, which isn't evaluated, and 6 has nothing
        ]
        predictions = [
            self._prediction(image_id, category_id, category_counts[category_id], start, 5)
            for image_id, category_id, _, start in scored
        ]
        image_ids = [1, 2, 3]

        stats = self._stats(category_counts, annotations, predictions, image_ids)
        reference = self._pycocotools_mean(category_counts, annotations, predictions, image_ids)

        assert len(stats) == 10
        assert (reference > 0).all()
        assert stats.tolist() == pytest.approx(reference.tolist(), abs=1e-6)
