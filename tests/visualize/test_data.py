# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the RF-DETR ground-truth/prediction visualization helper."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from rfdetr.visualize.data import save_gt_predictions_visualization


class TestSaveGtPredictionsVisualization:
    """Tests for `save_gt_predictions_visualization`."""

    def test_saves_png_with_gt_and_pred_boxes(self, tmp_path: Path) -> None:
        """A scenario with both ground-truth and prediction boxes saves a PNG named after it.

        This is the function's primary contract: given at least one box on each side, it renders and persists
        `<scenario_name>.png` under `save_dir` without raising.
        """
        save_gt_predictions_visualization(
            scenario_name="scenario_a",
            image_width=8,
            image_height=8,
            gt_boxes=[[1.0, 1.0, 2.0, 2.0]],
            gt_class_ids=[1],
            pred_boxes=[[2.0, 2.0, 2.0, 2.0]],
            pred_class_ids=[1],
            pred_confidences=[0.9],
            pred_ious=[0.5],
            save_dir=tmp_path,
        )

        assert (tmp_path / "scenario_a.png").is_file()

    def test_creates_missing_save_directory(self, tmp_path: Path) -> None:
        """A `save_dir` that does not exist yet is created rather than raising.

        Mirrors a fresh training run's first visualization call, where the target directory has not been created by any
        earlier step.
        """
        save_dir = tmp_path / "viz"

        save_gt_predictions_visualization(
            scenario_name="scenario_b",
            image_width=4,
            image_height=4,
            gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
            gt_class_ids=[1],
            pred_boxes=[[0.0, 0.0, 1.0, 1.0]],
            pred_class_ids=[1],
            pred_confidences=[0.5],
            pred_ious=[None],
            save_dir=save_dir,
        )

        assert save_dir.is_dir()
        assert (save_dir / "scenario_b.png").is_file()

    def test_output_image_includes_top_padding(self, tmp_path: Path) -> None:
        """Saved image height equals the requested height plus the fixed 60px label strip.

        The top padding reserves room for labels above the boxes; a regression here would silently crop or misplace
        every annotation.
        """
        save_gt_predictions_visualization(
            scenario_name="scenario_c",
            image_width=10,
            image_height=20,
            gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
            gt_class_ids=[1],
            pred_boxes=[[0.0, 0.0, 1.0, 1.0]],
            pred_class_ids=[1],
            pred_confidences=[0.5],
            pred_ious=[None],
            save_dir=tmp_path,
        )

        with Image.open(tmp_path / "scenario_c.png") as image:
            assert image.size == (10, 80)

    @pytest.mark.parametrize(
        ("pred_iou", "expected_label"),
        [
            pytest.param(0.512, "c2\nconf=0.876\niou=0.512", id="iou-known"),
            pytest.param(None, "c2\nconf=0.876", id="iou-unknown"),
        ],
    )
    def test_pred_label_reflects_iou_availability(
        self, tmp_path: Path, pred_iou: float | None, expected_label: str
    ) -> None:
        """Prediction labels append the IoU segment only when an IoU value is known.

        An unmatched false-positive prediction carries `iou=None`; the label loop must drop the IoU segment in that case
        instead of formatting `None` into the string.
        """
        with (
            mock.patch("rfdetr.visualize.data.BoxAnnotator"),
            mock.patch("rfdetr.visualize.data.LabelAnnotator") as mock_label_annotator,
        ):
            save_gt_predictions_visualization(
                scenario_name="scenario_d",
                image_width=6,
                image_height=6,
                gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
                gt_class_ids=[1],
                pred_boxes=[[0.0, 0.0, 2.0, 2.0]],
                pred_class_ids=[2],
                pred_confidences=[0.876],
                pred_ious=[pred_iou],
                save_dir=tmp_path,
            )

        pred_labels = mock_label_annotator.return_value.annotate.call_args_list[-1].kwargs["labels"]
        assert pred_labels == [expected_label]

    def test_gt_and_pred_boxes_offset_by_top_padding(self, tmp_path: Path) -> None:
        """Both GT and prediction boxes are shifted down by the 60px top padding before annotation.

        Verifies the xywh->xyxy conversion sees the offset y-coordinate, and that each side's class ids (and the
        prediction's confidence) reach the annotator unchanged.
        """
        with (
            mock.patch("rfdetr.visualize.data.BoxAnnotator") as mock_box_annotator,
            mock.patch("rfdetr.visualize.data.LabelAnnotator"),
        ):
            save_gt_predictions_visualization(
                scenario_name="scenario_e",
                image_width=8,
                image_height=6,
                gt_boxes=[[0.0, 0.0, 4.0, 4.0]],
                gt_class_ids=[1],
                pred_boxes=[[1.0, 1.0, 2.0, 2.0]],
                pred_class_ids=[2],
                pred_confidences=[0.75],
                pred_ious=[None],
                save_dir=tmp_path,
            )

        gt_call, pred_call = mock_box_annotator.return_value.annotate.call_args_list
        np.testing.assert_array_equal(gt_call.kwargs["detections"].xyxy, np.array([[0.0, 60.0, 4.0, 64.0]]))
        np.testing.assert_array_equal(gt_call.kwargs["detections"].class_id, np.array([1]))
        np.testing.assert_array_equal(pred_call.kwargs["detections"].xyxy, np.array([[1.0, 61.0, 3.0, 63.0]]))
        np.testing.assert_array_equal(pred_call.kwargs["detections"].class_id, np.array([2]))
        np.testing.assert_array_equal(pred_call.kwargs["detections"].confidence, np.array([0.75]))
