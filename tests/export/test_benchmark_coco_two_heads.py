# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCO keypoint AP of a model with both a segmentation and a keypoint head.

``sv.KeyPoints`` has no mask field, so such a model returns ``sv.Detections`` holding its ``(N, K, 3)`` keypoints in
``data["keypoints"]``. Everything runs on the tiny synthetic COCO files of ``test_benchmark_coco``; no network access
and no real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.export._benchmark import evaluate_coco_ap, evaluate_coco_predictions
from tests.export.test_benchmark_coco import _IMAGE_SIZES, _gt_keypoints, _gt_segments, _write_task_coco


def _gt_masks_and_keypoints(image: Image.Image) -> sv.Detections:
    """Return the ground-truth box, mask and person keypoints the way a model with both heads predicts them.

    Examples:
        >>> detections = _gt_masks_and_keypoints(Image.new("RGB", (64, 64)))
        >>> detections.mask.shape, detections.data["keypoints"].shape, detections.class_id.tolist()
        ((1, 64, 64), (1, 17, 3), [1])
    """
    detections = _gt_segments(image)
    points = _gt_keypoints(image)
    detections.class_id = points.class_id
    detections.data["keypoints"] = np.concatenate([points.xy, points.keypoint_confidence[..., None]], axis=-1)
    return detections


class TestTwoHeadKeypointAp:
    """``task="keypoints"`` scores ``sv.Detections`` that carry keypoints in their data."""

    def test_perfect_predictions_score_full_keypoint_ap(self, tmp_path: Path) -> None:
        """Exact boxes and keypoints score full box and keypoint AP."""
        subset = _write_task_coco(tmp_path, "keypoints")
        result = evaluate_coco_ap(_gt_masks_and_keypoints, subset, "keypoints", progress=False)
        assert (result.box_ap, result.box_ap50, result.task_ap, result.task_ap50) == pytest.approx((1.0, 1.0, 1.0, 1.0))

    def test_stored_predictions_score_as_the_run_did(self, tmp_path: Path) -> None:
        """Scoring stored two-head results gives the AP the live run gives."""
        subset = _write_task_coco(tmp_path, "keypoints")
        stored = {image_id: _gt_masks_and_keypoints(Image.new("RGB", size)) for image_id, size in _IMAGE_SIZES.items()}
        live = evaluate_coco_ap(_gt_masks_and_keypoints, subset, "keypoints", progress=False)
        assert evaluate_coco_predictions(stored, subset, "keypoints") == live

    def test_empty_result_scores_zero(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A two-head result with no rows counts as a missed image, not as a type mismatch."""
        output = sv.Detections.empty()
        output.data["keypoints"] = np.zeros((0, 17, 3), dtype=np.float32)
        evaluator = mocker.patch("rfdetr.evaluation.coco_eval.CocoEvaluator", autospec=True)
        result = evaluate_coco_ap(
            mocker.Mock(return_value=output), _write_task_coco(tmp_path, "keypoints"), "keypoints", progress=False
        )
        assert (result.box_ap, result.task_ap, result.n_images) == (0.0, 0.0, 3)
        evaluator.assert_not_called()

    def test_keypoints_without_confidence_are_refused(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Keypoints without a confidence channel are refused instead of being scored."""
        output = _gt_masks_and_keypoints(Image.new("RGB", (64, 64)))
        output.data["keypoints"] = output.data["keypoints"][..., :2]
        with pytest.raises(ValueError, match=r"\(N, K, 3\)"):
            evaluate_coco_ap(
                mocker.Mock(return_value=output), _write_task_coco(tmp_path, "keypoints"), "keypoints", progress=False
            )

    def test_detections_without_keypoints_are_refused(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Detections from a segmentation-only model still do not match the keypoint task."""
        output = _gt_segments(Image.new("RGB", (64, 64)))
        with pytest.raises(ValueError, match="task"):
            evaluate_coco_ap(
                mocker.Mock(return_value=output), _write_task_coco(tmp_path, "keypoints"), "keypoints", progress=False
            )
