# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
from unittest.mock import MagicMock

import torch

from rfdetr.models.heads.detection import DetectionHead
from rfdetr.models.lwdetr import LWDETR


def _make_oriented_lwdetr(num_classes: int = 91) -> LWDETR:
    """Construct a minimal oriented LWDETR for testing."""
    hidden_dim = 8
    backbone = MagicMock()
    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.decoder = MagicMock()
    transformer.decoder.bbox_embed = None
    return LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=4,
        group_detr=1,
        oriented=True,
    )


class TestDetectionHeadOriented:
    def test_standard_head_output_shape(self) -> None:
        head = DetectionHead(hidden_dim=16, num_classes=10)
        hs = torch.randn(2, 5, 16)
        cls_out, coord_out = head(hs)
        assert cls_out.shape == (2, 5, 10)
        assert coord_out.shape == (2, 5, 4)

    def test_oriented_head_output_shape(self) -> None:
        head = DetectionHead(hidden_dim=16, num_classes=10, oriented=True)
        hs = torch.randn(2, 5, 16)
        cls_out, coord_out = head(hs)
        assert cls_out.shape == (2, 5, 10)
        assert coord_out.shape == (2, 5, 5)

    def test_oriented_angle_range(self) -> None:
        head = DetectionHead(hidden_dim=16, num_classes=10, oriented=True)
        hs = torch.randn(2, 5, 16)
        _, coord_out = head(hs)
        angles = coord_out[..., 4]
        assert (angles >= 0).all()
        assert (angles < math.pi + 0.01).all()

    def test_oriented_has_angle_embed(self) -> None:
        head = DetectionHead(hidden_dim=16, num_classes=10, oriented=True)
        assert head.angle_embed is not None

    def test_standard_has_no_angle_embed(self) -> None:
        head = DetectionHead(hidden_dim=16, num_classes=10)
        assert head.angle_embed is None

    def test_oriented_gradients_flow(self) -> None:
        head = DetectionHead(hidden_dim=16, num_classes=10, oriented=True)
        hs = torch.randn(2, 5, 16, requires_grad=True)
        _, coord_out = head(hs)
        loss = coord_out.sum()
        loss.backward()
        assert hs.grad is not None
        assert torch.isfinite(hs.grad).all()


class TestLWDETROriented:
    def test_oriented_model_has_angle_embed(self) -> None:
        model = _make_oriented_lwdetr()
        assert model.angle_embed is not None
        assert model.oriented is True

    def test_non_oriented_model_has_no_angle_embed(self) -> None:
        hidden_dim = 8
        backbone = MagicMock()
        transformer = MagicMock()
        transformer.d_model = hidden_dim
        transformer.decoder = MagicMock()
        transformer.decoder.bbox_embed = None
        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=91,
            num_queries=4,
            group_detr=1,
            oriented=False,
        )
        assert model.angle_embed is None
        assert model.oriented is False
