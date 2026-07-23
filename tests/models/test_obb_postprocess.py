# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math

import pytest
import torch

from rfdetr.models.postprocess import PostProcess


class TestPostProcessOriented:
    def test_oriented_output_keys(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = pp(outputs, target_sizes)
        assert len(results) == 1
        assert set(results[0]) == {"scores", "labels", "boxes_obb", "corners", "boxes"}

    def test_oriented_obb_shape(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = pp(outputs, target_sizes)
        assert results[0]["boxes_obb"].shape == (5, 5)
        assert results[0]["corners"].shape == (5, 4, 2)

    def test_oriented_scales_spatial_dims(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.full((1, 10, 5), 0.5),
        }
        target_sizes = torch.tensor([[100, 200]])
        results = pp(outputs, target_sizes)
        obb = results[0]["boxes_obb"]
        assert torch.allclose(obb[0, 0], torch.tensor(100.0), atol=1.0)
        assert torch.allclose(obb[0, 1], torch.tensor(50.0), atol=1.0)

    def test_standard_postprocess_unchanged(self) -> None:
        pp = PostProcess(num_select=5, oriented=False)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 4),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = pp(outputs, target_sizes)
        assert "boxes" in results[0]
        assert "boxes_obb" not in results[0]

    def test_batch_support(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(3, 10, 3),
            "pred_boxes": torch.rand(3, 10, 5),
        }
        target_sizes = torch.tensor([[480, 640], [320, 320], [600, 800]])
        results = pp(outputs, target_sizes)
        assert len(results) == 3

    def test_each_image_scales_by_its_own_target_size(self) -> None:
        """Per-image scale factors must not leak across the batch."""
        pp = PostProcess(num_select=1, oriented=True)
        outputs = {
            "pred_logits": torch.zeros(2, 1, 3),
            "pred_boxes": torch.full((2, 1, 5), 0.5),
        }
        results = pp(outputs, torch.tensor([[100, 200], [400, 800]]))
        assert results[0]["boxes_obb"][0, 0].item() == pytest.approx(100.0, abs=1.0)
        assert results[1]["boxes_obb"][0, 0].item() == pytest.approx(400.0, abs=1.0)


class TestPostProcessOrientedEnvelope:
    """The ``boxes`` xyxy envelope feeds the COCO eval callback and torchmetrics."""

    def test_envelope_matches_corner_extremes(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        results = pp(outputs, torch.tensor([[480, 640]]))

        corners, boxes = results[0]["corners"], results[0]["boxes"]
        expected = torch.stack(
            [
                corners[..., 0].min(dim=-1).values,
                corners[..., 1].min(dim=-1).values,
                corners[..., 0].max(dim=-1).values,
                corners[..., 1].max(dim=-1).values,
            ],
            dim=-1,
        )
        assert torch.allclose(boxes, expected, atol=1e-5)

    def test_rotated_box_envelope_matches_projection_formula(self) -> None:
        """A rotated box projects to w*|cos| + h*|sin| on x, and w*|sin| + h*|cos| on y.

        Reusing the raw w/h instead of the corner extremes would give 40x10 here rather than the correct 35.36x35.36,
        mis-stating IoU against axis-aligned ground truth in both directions.
        """
        pp = PostProcess(num_select=1, oriented=True)
        angle = math.pi / 4
        boxes = torch.zeros(1, 1, 5)
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.1, angle])
        results = pp({"pred_logits": torch.zeros(1, 1, 3), "pred_boxes": boxes}, torch.tensor([[100, 100]]))

        box = results[0]["boxes"][0]
        w_px, h_px = 40.0, 10.0
        expected_w = w_px * abs(math.cos(angle)) + h_px * abs(math.sin(angle))
        expected_h = w_px * abs(math.sin(angle)) + h_px * abs(math.cos(angle))
        assert (box[2] - box[0]).item() == pytest.approx(expected_w, abs=1e-3)
        assert (box[3] - box[1]).item() == pytest.approx(expected_h, abs=1e-3)

    def test_axis_aligned_envelope_equals_box_dimensions(self) -> None:
        """At angle 0 the envelope is exactly the box."""
        pp = PostProcess(num_select=1, oriented=True)
        boxes = torch.zeros(1, 1, 5)
        boxes[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.2, 0.0])
        results = pp({"pred_logits": torch.zeros(1, 1, 3), "pred_boxes": boxes}, torch.tensor([[100, 100]]))

        box = results[0]["boxes"][0]
        assert box.tolist() == pytest.approx([30.0, 40.0, 70.0, 60.0], abs=1e-3)
