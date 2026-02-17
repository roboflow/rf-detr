# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch
from torch import nn


class TestLWDETRDepthHead:
    def test_depth_head_creates_depth_embed(self):
        """LWDETR with depth_head=True should have depth_embed attribute."""
        from rfdetr.models.lwdetr import LWDETR, MLP

        # Create minimal mock backbone and transformer
        backbone = _MockBackbone(256)
        transformer = _MockTransformer(256)
        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=3,
            num_queries=10,
            depth_head=True,
            z_max=120.0,
        )
        assert hasattr(model, "depth_embed"), "depth_embed head should exist"
        assert isinstance(model.depth_embed, MLP)

    def test_no_depth_head_by_default(self):
        """LWDETR with depth_head=False should NOT have depth_embed."""
        from rfdetr.models.lwdetr import LWDETR

        backbone = _MockBackbone(256)
        transformer = _MockTransformer(256)
        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=3,
            num_queries=10,
            depth_head=False,
        )
        assert not hasattr(model, "depth_embed"), (
            "depth_embed should not exist when depth_head=False"
        )


class _MockBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dummy = nn.Linear(dim, dim)

    def forward(self, x):
        return [], []


class _MockTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_model = dim
        self.decoder = _MockDecoder()

    def forward(self, *args, **kwargs):
        return None, None, None, None


class _MockDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_embed = None
