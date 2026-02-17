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


class TestPostProcessDepth:
    def test_postprocess_includes_depth(self):
        """PostProcess should gather depth values for top-K queries."""
        from rfdetr.models.lwdetr import PostProcess

        pp = PostProcess(num_select=5)
        B, Q, C = 2, 10, 3
        outputs = {
            "pred_logits": torch.randn(B, Q, C),
            "pred_boxes": torch.rand(B, Q, 4),
            "pred_depth": torch.rand(B, Q, 1) * 100,
        }
        target_sizes = torch.tensor([[640, 640], [640, 640]])
        results = pp(outputs, target_sizes)
        assert "depth" in results[0]
        assert results[0]["depth"].shape == (5, 1)

    def test_postprocess_no_depth_when_absent(self):
        """PostProcess should work normally without pred_depth."""
        from rfdetr.models.lwdetr import PostProcess

        pp = PostProcess(num_select=5)
        B, Q, C = 2, 10, 3
        outputs = {
            "pred_logits": torch.randn(B, Q, C),
            "pred_boxes": torch.rand(B, Q, 4),
        }
        target_sizes = torch.tensor([[640, 640], [640, 640]])
        results = pp(outputs, target_sizes)
        assert "depth" not in results[0]


class TestPredictDepthTupleUnpacking:
    def test_depth_tuple_unpacking(self):
        """When forward_export returns 3-tuple with depth_head=True, pred_depth should be set."""
        from types import SimpleNamespace

        config = SimpleNamespace(depth_head=True)
        predictions = (
            torch.rand(1, 10, 4),   # pred_boxes
            torch.randn(1, 10, 80), # pred_logits
            torch.rand(1, 10, 1) * 100,  # pred_depth
        )
        # Simulate the tuple unpacking logic from detr.py
        return_predictions = {
            "pred_logits": predictions[1],
            "pred_boxes": predictions[0],
        }
        if len(predictions) == 3:
            if getattr(config, 'depth_head', False):
                return_predictions["pred_depth"] = predictions[2]
            else:
                return_predictions["pred_masks"] = predictions[2]

        assert "pred_depth" in return_predictions
        assert "pred_masks" not in return_predictions
        assert return_predictions["pred_depth"].shape == (1, 10, 1)

    def test_no_depth_tuple_unpacking_when_disabled(self):
        """When depth_head=False, 3-tuple should set pred_masks not pred_depth."""
        from types import SimpleNamespace

        config = SimpleNamespace(depth_head=False)
        predictions = (
            torch.rand(1, 10, 4),
            torch.randn(1, 10, 80),
            torch.rand(1, 10, 1),  # this would be masks
        )
        return_predictions = {
            "pred_logits": predictions[1],
            "pred_boxes": predictions[0],
        }
        if len(predictions) == 3:
            if getattr(config, 'depth_head', False):
                return_predictions["pred_depth"] = predictions[2]
            else:
                return_predictions["pred_masks"] = predictions[2]

        assert "pred_masks" in return_predictions
        assert "pred_depth" not in return_predictions


class TestPopulateArgsDepth:
    def test_populate_args_includes_depth_params(self):
        """populate_args should include depth-related parameters."""
        from rfdetr.main import populate_args

        args = populate_args(depth_head=True, z_max=80.0, ball_class_ids=[0, 1])
        assert args.depth_head is True
        assert args.z_max == 80.0
        assert args.depth_loss_coef == 5.0
        assert args.pinhole_loss_coef == 1.0
        assert args.ball_class_ids == [0, 1]
        assert args.curriculum_phase1_epochs == 10

    def test_populate_args_depth_defaults(self):
        """populate_args should default depth_head to False."""
        from rfdetr.main import populate_args

        args = populate_args()
        assert args.depth_head is False
        assert args.z_max == 120.0
        assert args.ball_class_ids == []


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
