# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the GT/prediction visualization saver's empty-box handling."""

from __future__ import annotations

from pathlib import Path

from rfdetr.visualize.data import save_gt_predictions_visualization


class TestSaveGtPredictionsVisualizationEmptyBoxes:
    """An empty ``gt_boxes`` or ``pred_boxes`` list must be a no-op for that side, not a crash."""

    def test_empty_gt_boxes_does_not_raise(self, tmp_path: Path) -> None:
        """A scenario with no ground-truth boxes still saves a visualization for the predictions.

        Regression test for an ``IndexError``: an empty ``gt_boxes`` list produced a 1-D ``np.array([])`` that
        ``xywh_to_xyxy`` then indexed as 2-D.
        """
        save_gt_predictions_visualization(
            scenario_name="empty_gt",
            image_width=32,
            image_height=32,
            gt_boxes=[],
            gt_class_ids=[],
            pred_boxes=[[4.0, 4.0, 8.0, 8.0]],
            pred_class_ids=[1],
            pred_confidences=[0.9],
            pred_ious=[None],
            save_dir=tmp_path,
        )

        assert (tmp_path / "empty_gt.png").exists()

    def test_empty_pred_boxes_does_not_raise(self, tmp_path: Path) -> None:
        """A scenario with no predictions still saves a visualization for the ground truth.

        Regression test for the same ``IndexError`` on the prediction side of the function.
        """
        save_gt_predictions_visualization(
            scenario_name="empty_pred",
            image_width=32,
            image_height=32,
            gt_boxes=[[4.0, 4.0, 8.0, 8.0]],
            gt_class_ids=[1],
            pred_boxes=[],
            pred_class_ids=[],
            pred_confidences=[],
            pred_ious=[],
            save_dir=tmp_path,
        )

        assert (tmp_path / "empty_pred.png").exists()

    def test_both_empty_does_not_raise(self, tmp_path: Path) -> None:
        """A scenario with neither GT nor predictions still saves a (blank) visualization."""
        save_gt_predictions_visualization(
            scenario_name="both_empty",
            image_width=32,
            image_height=32,
            gt_boxes=[],
            gt_class_ids=[],
            pred_boxes=[],
            pred_class_ids=[],
            pred_confidences=[],
            pred_ious=[],
            save_dir=tmp_path,
        )

        assert (tmp_path / "both_empty.png").exists()
