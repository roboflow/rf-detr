# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Tests for the keypoint/pose estimation implementation.
"""

import pytest
import torch


class TestKeypointHead:
    """Tests for the KeypointHead module."""

    def test_keypoint_head_import(self):
        """Test that KeypointHead can be imported."""
        from rfdetr.models.keypoint_head import KeypointHead

        assert KeypointHead is not None

    def test_keypoint_head_init(self):
        """Test KeypointHead initialization."""
        from rfdetr.models.keypoint_head import KeypointHead

        head = KeypointHead(hidden_dim=256, num_keypoints=17, num_layers=3)
        assert head.num_keypoints == 17
        assert head.hidden_dim == 256

    def test_keypoint_head_forward(self):
        """Test KeypointHead forward pass."""
        from rfdetr.models.keypoint_head import KeypointHead

        head = KeypointHead(hidden_dim=256, num_keypoints=17, num_layers=3)

        batch_size = 2
        num_queries = 300
        hidden_dim = 256

        query_features = [torch.randn(batch_size, num_queries, hidden_dim)]
        outputs = head(query_features)

        assert len(outputs) == 1
        assert outputs[0].shape == (batch_size, num_queries, 17, 3)

    def test_keypoint_head_with_reference_boxes(self):
        """Test KeypointHead with reference boxes for relative prediction."""
        from rfdetr.models.keypoint_head import KeypointHead

        head = KeypointHead(hidden_dim=256, num_keypoints=17, num_layers=3)

        batch_size = 2
        num_queries = 300
        hidden_dim = 256

        query_features = [torch.randn(batch_size, num_queries, hidden_dim)]
        reference_boxes = torch.rand(batch_size, num_queries, 4)

        outputs = head(query_features, reference_boxes=reference_boxes)

        assert len(outputs) == 1
        assert outputs[0].shape == (batch_size, num_queries, 17, 3)
        assert outputs[0][..., :2].min() >= 0.0
        assert outputs[0][..., :2].max() <= 1.0

    def test_keypoint_head_custom_keypoints(self):
        """Test KeypointHead with custom number of keypoints."""
        from rfdetr.models.keypoint_head import KeypointHead

        num_keypoints = 5
        head = KeypointHead(hidden_dim=128, num_keypoints=num_keypoints, num_layers=2)

        query_features = [torch.randn(1, 100, 128)]
        outputs = head(query_features)

        assert outputs[0].shape == (1, 100, num_keypoints, 3)

    def test_keypoint_head_multiple_layers(self):
        """Test KeypointHead with multiple decoder layers."""
        from rfdetr.models.keypoint_head import KeypointHead

        head = KeypointHead(hidden_dim=256, num_keypoints=17, num_layers=3)

        query_features = [
            torch.randn(2, 300, 256),
            torch.randn(2, 300, 256),
            torch.randn(2, 300, 256),
        ]
        outputs = head(query_features)

        assert len(outputs) == 3
        for out in outputs:
            assert out.shape == (2, 300, 17, 3)


class TestKeypointConstants:
    """Tests for COCO keypoint constants."""

    def test_coco_keypoint_names(self):
        """Test COCO keypoint names are correctly defined."""
        from rfdetr.models.keypoint_head import COCO_KEYPOINT_NAMES

        assert len(COCO_KEYPOINT_NAMES) == 17
        assert COCO_KEYPOINT_NAMES[0] == "nose"
        assert "left_shoulder" in COCO_KEYPOINT_NAMES
        assert "right_ankle" in COCO_KEYPOINT_NAMES

    def test_coco_skeleton(self):
        """Test COCO skeleton connections are valid."""
        from rfdetr.models.keypoint_head import COCO_KEYPOINT_NAMES, COCO_SKELETON

        for connection in COCO_SKELETON:
            assert len(connection) == 2
            assert 0 <= connection[0] < len(COCO_KEYPOINT_NAMES)
            assert 0 <= connection[1] < len(COCO_KEYPOINT_NAMES)

    def test_coco_sigmas(self):
        """Test COCO keypoint sigmas are valid."""
        from rfdetr.models.keypoint_head import COCO_KEYPOINT_SIGMAS

        assert len(COCO_KEYPOINT_SIGMAS) == 17
        for sigma in COCO_KEYPOINT_SIGMAS:
            assert 0 < sigma < 1

    def test_coco_flip_pairs(self):
        """Test COCO flip pairs are valid and symmetric."""
        from rfdetr.models.keypoint_head import (
            COCO_KEYPOINT_FLIP_PAIRS,
            COCO_KEYPOINT_NAMES,
        )

        for left_idx, right_idx in COCO_KEYPOINT_FLIP_PAIRS:
            left_name = COCO_KEYPOINT_NAMES[left_idx]
            right_name = COCO_KEYPOINT_NAMES[right_idx]
            assert "left" in left_name
            assert "right" in right_name


class TestKeypointConfig:
    """Tests for keypoint configuration classes."""

    def test_rfdetr_pose_config(self):
        """Test RFDETRPoseConfig default values."""
        from rfdetr.config import RFDETRPoseConfig

        config = RFDETRPoseConfig()
        assert config.keypoint_head is True
        assert config.num_keypoints == 17
        assert len(config.keypoint_names) == 17
        assert config.skeleton is not None
        assert config.num_classes == 1  # Person class for pose

    def test_keypoint_train_config(self):
        """Test KeypointTrainConfig default values."""
        from rfdetr.config import KeypointTrainConfig

        config = KeypointTrainConfig(dataset_dir="/tmp/test")
        assert config.keypoint_head is True
        assert config.num_keypoints == 17
        assert config.keypoint_loss_coef == 5.0
        assert config.keypoint_visibility_loss_coef == 2.0
        assert config.keypoint_oks_loss_coef == 2.0

    def test_model_config_keypoint_fields(self):
        """Test that ModelConfig has keypoint fields."""
        from rfdetr.config import RFDETRBaseConfig

        config = RFDETRBaseConfig()
        assert config.keypoint_head is False
        assert config.num_keypoints == 17


class TestRFDETRPose:
    """Tests for RFDETRPose class."""

    def test_rfdetr_pose_import(self):
        """Test that RFDETRPose can be imported."""
        from rfdetr import RFDETRPose

        assert RFDETRPose is not None

    def test_rfdetr_pose_config(self):
        """Test RFDETRPose uses correct configs."""
        from rfdetr import RFDETRPose
        from rfdetr.config import KeypointTrainConfig, RFDETRPoseConfig

        pose = RFDETRPose.__new__(RFDETRPose)
        pose.model_config = RFDETRPoseConfig()
        model_config = pose.get_model_config()
        train_config = pose.get_train_config(dataset_dir="/tmp")

        assert isinstance(model_config, RFDETRPoseConfig)
        assert isinstance(train_config, KeypointTrainConfig)
        assert model_config.keypoint_head is True
        assert train_config.keypoint_head is True


class TestKeypointLoss:
    """Tests for keypoint loss functions."""

    def test_loss_keypoints_exists(self):
        """Test that loss_keypoints method exists in SetCriterion."""
        from rfdetr.models.criterion import SetCriterion

        assert hasattr(SetCriterion, "loss_keypoints")

    def test_loss_map_includes_keypoints(self):
        """Test that keypoints is in the loss map."""
        from rfdetr.models.criterion import SetCriterion

        # Create a minimal criterion to check loss_map
        criterion = SetCriterion.__new__(SetCriterion)
        criterion.losses = ["keypoints"]

        loss_map = {
            "labels": "loss_labels",
            "boxes": "loss_boxes",
            "masks": "loss_masks",
            "keypoints": "loss_keypoints",
        }
        assert "keypoints" in loss_map


class TestKeypointPostProcess:
    """Tests for keypoint post-processing."""

    def test_postprocess_handles_keypoints(self):
        """Test that PostProcess handles keypoint outputs."""
        from rfdetr.models.postprocess import PostProcess

        batch_size = 2
        num_queries = 300
        num_classes = 1
        num_keypoints = 17

        outputs = {
            "pred_logits": torch.randn(batch_size, num_queries, num_classes),
            "pred_boxes": torch.rand(batch_size, num_queries, 4),
            "pred_keypoints": torch.rand(batch_size, num_queries, num_keypoints, 3),
        }
        target_sizes = torch.tensor([[480, 640], [480, 640]])

        postprocess = PostProcess(num_select=100)
        results = postprocess(outputs, target_sizes)

        assert len(results) == batch_size
        for result in results:
            assert "scores" in result
            assert "labels" in result
            assert "boxes" in result
            assert "keypoints" in result
            assert result["keypoints"].shape[-1] == 3  # x, y, visibility

    def test_postprocess_without_keypoints(self):
        """Test that PostProcess works normally without keypoints."""
        from rfdetr.models.postprocess import PostProcess

        outputs = {
            "pred_logits": torch.randn(1, 100, 2),
            "pred_boxes": torch.rand(1, 100, 4),
        }
        target_sizes = torch.tensor([[480, 640]])

        postprocess = PostProcess(num_select=10)
        results = postprocess(outputs, target_sizes)

        assert len(results) == 1
        assert "keypoints" not in results[0]


class TestKeypointInference:
    """Tests for keypoint inference pipeline."""

    def test_keypoints_output_structure(self):
        """Test that keypoint output has correct structure."""
        from rfdetr.models.postprocess import PostProcess

        postprocess = PostProcess(num_select=10)

        outputs = {
            "pred_logits": torch.randn(1, 300, 1),
            "pred_boxes": torch.rand(1, 300, 4),
            "pred_keypoints": torch.rand(1, 300, 17, 3),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = postprocess(outputs, target_sizes)

        assert len(results) == 1
        result = results[0]

        assert "keypoints" in result
        kpts = result["keypoints"]

        # Shape should be [num_select, num_keypoints, 3]
        assert kpts.shape == (10, 17, 3)

        # visibility should be sigmoid (0 or 2 after thresholding)
        assert kpts[..., 2].min() >= 0.0

    def test_keypoints_visibility_sigmoid(self):
        """Test that visibility values are sigmoids in [0, 1]."""
        from rfdetr.models.postprocess import PostProcess

        postprocess = PostProcess(num_select=5)

        outputs = {
            "pred_logits": torch.randn(1, 100, 1),
            "pred_boxes": torch.rand(1, 100, 4),
            "pred_keypoints": torch.zeros(1, 100, 17, 3),
        }
        outputs["pred_keypoints"][..., 2] = torch.randn(1, 100, 17) * 5

        target_sizes = torch.tensor([[480, 640]])
        results = postprocess(outputs, target_sizes)

        # Check raw confidence is in [0, 1]
        vis_conf = results[0]["keypoints_confidence"]
        assert vis_conf.min() >= 0.0
        assert vis_conf.max() <= 1.0

    def test_keypoints_coordinate_scaling(self):
        """Test that keypoint coordinates are properly scaled to image size."""
        from rfdetr.models.postprocess import PostProcess

        postprocess = PostProcess(num_select=1)

        outputs = {
            "pred_logits": torch.tensor([[[10.0]] * 100]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * 100]),
            "pred_keypoints": torch.zeros(1, 100, 17, 3),
        }
        # Set first keypoint to center (0.5, 0.5) with high visibility
        outputs["pred_keypoints"][0, :, 0, :] = torch.tensor([0.5, 0.5, 5.0])

        target_sizes = torch.tensor([[480, 640]])  # H, W
        results = postprocess(outputs, target_sizes)

        kpts = results[0]["keypoints"]

        # First keypoint x should be scaled to ~320 (0.5 * 640)
        # First keypoint y should be scaled to ~240 (0.5 * 480)
        assert abs(kpts[0, 0, 0].item() - 320) < 1.0
        assert abs(kpts[0, 0, 1].item() - 240) < 1.0


class TestKeypointIntegration:
    """Integration tests for keypoint functionality."""

    def test_keypoint_head_gradient_flow(self):
        """Test that gradients flow through KeypointHead."""
        from rfdetr.models.keypoint_head import KeypointHead

        head = KeypointHead(hidden_dim=256, num_keypoints=17, num_layers=3)

        query_features = [torch.randn(2, 100, 256)]
        outputs = head(query_features)

        loss = outputs[0].sum()
        loss.backward()

        has_grad = False
        for param in head.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in KeypointHead parameters"

    def test_keypoint_output_format(self):
        """Test complete keypoint output format through PostProcess."""
        from rfdetr.models.postprocess import PostProcess

        postprocess = PostProcess(num_select=10)

        outputs = {
            "pred_logits": torch.randn(1, 300, 1),
            "pred_boxes": torch.rand(1, 300, 4),
            "pred_keypoints": torch.rand(1, 300, 17, 3),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = postprocess(outputs, target_sizes)

        assert len(results) == 1
        result = results[0]

        assert "scores" in result
        assert "labels" in result
        assert "boxes" in result
        assert "keypoints" in result

        assert result["scores"].shape[0] == 10
        assert result["boxes"].shape == (10, 4)
        assert result["keypoints"].shape == (10, 17, 3)

        kpts = result["keypoints"]
        # Visibility in COCO format (0 or 2)
        unique_vis = kpts[..., 2].unique()
        for v in unique_vis:
            assert v.item() in (0.0, 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
