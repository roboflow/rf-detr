# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for PostProcess box clamping behaviour."""

import torch

from rfdetr.models.postprocess import PostProcess


class TestGatherAndScaleBoxes:
    """Tests for :meth:`PostProcess._gather_and_scale_boxes`."""

    def test_clamps_boxes_to_image_bounds(self):
        """Boxes that extrapolate beyond [0, 1] in normalized space are
        clamped to pixel-space image dimensions after scaling."""
        # Three synthetic boxes in cxcywh normalized coords:
        #   [0] cx=0.01, w=0.10 → x1 = (0.01 - 0.05) * 640 = -25.6  ← negative
        #   [1] cx=0.99, w=0.10 → x2 = (0.99 + 0.05) * 640 = 665.6  ← overflow
        #   [2] cx=0.50, w=0.20 → fully in-bounds
        out_bbox = torch.tensor(
            [
                [
                    [0.01, 0.01, 0.10, 0.10],  # negative x1, y1 after scale
                    [0.99, 0.99, 0.10, 0.10],  # x2 > img_w, y2 > img_h after scale
                    [0.50, 0.50, 0.20, 0.20],  # in-bounds control
                ]
            ]
        )  # shape (B=1, Q=3, 4)

        topk_boxes = torch.tensor([[0, 1, 2]])  # select all three
        target_sizes = torch.tensor([[480, 640]])  # (h, w)

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        img_h, img_w = 480, 640

        # All coords must be >= 0
        assert (boxes >= 0).all(), f"Negative coords present: {boxes[boxes < 0]}"
        # x1, x2 must be <= image width
        assert (boxes[..., 0] <= img_w).all()
        assert (boxes[..., 2] <= img_w).all()
        # y1, y2 must be <= image height
        assert (boxes[..., 1] <= img_h).all()
        assert (boxes[..., 3] <= img_h).all()

    def test_in_bounds_boxes_unchanged(self):
        """Boxes already within image bounds are not altered by clamping."""
        out_bbox = torch.tensor(
            [
                [
                    [0.30, 0.30, 0.20, 0.20],
                    [0.70, 0.60, 0.30, 0.40],
                ]
            ]
        )

        topk_boxes = torch.tensor([[0, 1]])
        target_sizes = torch.tensor([[480, 640]])

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        # Manually computed expected values (no clamping needed)
        expected = torch.tensor(
            [
                [
                    [128.0, 96.0, 256.0, 192.0],  # cx=0.30,cy=0.30,w=0.20,h=0.20
                    [352.0, 192.0, 544.0, 384.0],  # cx=0.70,cy=0.60,w=0.30,h=0.40
                ]
            ]
        )

        assert torch.allclose(boxes, expected, atol=1e-4), (
            f"In-bounds boxes were altered.\nExpected:\n{expected}\nGot:\n{boxes}"
        )

    def test_multiple_images_in_batch(self):
        """Clamping works correctly across a batch of mixed image sizes."""
        out_bbox = torch.tensor(
            [
                [
                    [0.01, 0.50, 0.10, 0.20],  # image 0: negative x1
                ],
                [
                    [0.99, 0.50, 0.10, 0.20],  # image 1: x2 overflow
                ],
            ]
        )

        topk_boxes = torch.tensor([[0], [0]])
        target_sizes = torch.tensor(
            [
                [300, 400],  # image 0: 400×300
                [600, 800],  # image 1: 800×600
            ]
        )

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        # Image 0: all coords must be in [0, 400]×[0, 300]
        assert (boxes[0, :, 0] >= 0).all() and (boxes[0, :, 0] <= 400).all()
        assert (boxes[0, :, 1] >= 0).all() and (boxes[0, :, 1] <= 300).all()
        assert (boxes[0, :, 2] >= 0).all() and (boxes[0, :, 2] <= 400).all()
        assert (boxes[0, :, 3] >= 0).all() and (boxes[0, :, 3] <= 300).all()

        # Image 1: all coords must be in [0, 800]×[0, 600]
        assert (boxes[1, :, 0] >= 0).all() and (boxes[1, :, 0] <= 800).all()
        assert (boxes[1, :, 1] >= 0).all() and (boxes[1, :, 1] <= 600).all()
        assert (boxes[1, :, 2] >= 0).all() and (boxes[1, :, 2] <= 800).all()
        assert (boxes[1, :, 3] >= 0).all() and (boxes[1, :, 3] <= 600).all()
