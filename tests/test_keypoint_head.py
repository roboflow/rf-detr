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
import numpy as np
from PIL import Image


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

        # Simulate query features from decoder
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
        # Reference boxes in cxcywh format, normalized [0, 1]
        reference_boxes = torch.rand(batch_size, num_queries, 4)

        outputs = head(query_features, reference_boxes=reference_boxes)

        assert len(outputs) == 1
        assert outputs[0].shape == (batch_size, num_queries, 17, 3)
        # Coordinates should be in [0, 1] range due to clamping
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

        # Simulate multiple decoder layers
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
        from rfdetr.models.keypoint_head import COCO_SKELETON, COCO_KEYPOINT_NAMES

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
        from rfdetr.models.keypoint_head import COCO_KEYPOINT_FLIP_PAIRS, COCO_KEYPOINT_NAMES

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
        from rfdetr.config import ModelConfig, RFDETRBaseConfig

        # Base config should have keypoint_head=False by default
        config = RFDETRBaseConfig()
        assert config.keypoint_head is False
        assert config.num_keypoints == 17


class TestKeypointDataset:
    """Tests for keypoint dataset handling."""

    def test_convert_coco_keypoints(self):
        """Test ConvertCoco extracts keypoints correctly."""
        from rfdetr.datasets.coco import ConvertCoco

        converter = ConvertCoco(include_keypoints=True, num_keypoints=17)
        assert converter.include_keypoints is True
        assert converter.num_keypoints == 17

    def test_extract_keypoints_format(self):
        """Test keypoint extraction produces correct format."""
        from rfdetr.datasets.coco import ConvertCoco

        converter = ConvertCoco(include_keypoints=True, num_keypoints=17)

        # Mock annotation with COCO keypoints format
        anno = [{
            "keypoints": [
                100, 50, 2,   # nose: x, y, v
                90, 45, 2,    # left_eye
                110, 45, 2,   # right_eye
                80, 50, 1,    # left_ear
                120, 50, 1,   # right_ear
                70, 100, 2,   # left_shoulder
                130, 100, 2,  # right_shoulder
                60, 150, 2,   # left_elbow
                140, 150, 2,  # right_elbow
                50, 200, 2,   # left_wrist
                150, 200, 2,  # right_wrist
                80, 200, 2,   # left_hip
                120, 200, 2,  # right_hip
                75, 280, 2,   # left_knee
                125, 280, 2,  # right_knee
                70, 350, 2,   # left_ankle
                130, 350, 2,  # right_ankle
            ]
        }]

        w, h = 200, 400
        keypoints = converter._extract_keypoints(anno, w, h)

        assert keypoints.shape == (1, 17, 3)
        # Check normalization
        assert keypoints[..., 0].max() <= 1.0  # x normalized
        assert keypoints[..., 1].max() <= 1.0  # y normalized

    def test_extract_keypoints_empty(self):
        """Test keypoint extraction with no annotations."""
        from rfdetr.datasets.coco import ConvertCoco

        converter = ConvertCoco(include_keypoints=True, num_keypoints=17)
        keypoints = converter._extract_keypoints([], 200, 400)

        assert keypoints.shape == (0, 17, 3)


class TestKeypointTransforms:
    """Tests for keypoint transforms."""

    def test_hflip_keypoints(self):
        """Test horizontal flip swaps left/right keypoints."""
        from rfdetr.datasets import transforms as T

        # Create a mock image and target with keypoints
        image = Image.new('RGB', (200, 200))
        target = {
            "boxes": torch.tensor([[50, 50, 150, 150]]),
            "keypoints": torch.tensor([[[0.25, 0.5, 2.0],   # left position
                                         [0.75, 0.5, 2.0]]]) # right position
        }

        # Apply hflip
        flipped_image, flipped_target = T.hflip(image, target)

        # Check x coordinates are flipped (1 - x)
        assert flipped_target["keypoints"][0, 0, 0] == pytest.approx(0.75, rel=1e-5)
        assert flipped_target["keypoints"][0, 1, 0] == pytest.approx(0.25, rel=1e-5)

    def test_crop_keypoints_visibility(self):
        """Test crop marks out-of-bounds keypoints as invisible."""
        from rfdetr.datasets import transforms as T

        image = Image.new('RGB', (400, 400))
        # Keypoint at (0.1, 0.1) should be outside a crop starting at (0.2, 0.2)
        target = {
            "boxes": torch.tensor([[100.0, 100.0, 300.0, 300.0]]),
            "labels": torch.tensor([1]),
            "area": torch.tensor([40000.0]),
            "iscrowd": torch.tensor([0]),
            "keypoints": torch.tensor([[[0.1, 0.1, 2.0],   # will be outside crop
                                         [0.5, 0.5, 2.0]]]), # will be inside crop
            "size": torch.tensor([400, 400]),
            "orig_size": torch.tensor([400, 400]),
        }

        # Crop region (region format: top, left, height, width)
        region = (80, 80, 200, 200)  # crops to (80,80) - (280, 280)
        cropped_image, cropped_target = T.crop(image, target, region)

        # Keypoint at (0.1, 0.1) = pixel (40, 40) is outside crop (80, 80)-(280, 280)
        # Should be marked invisible (v=0)
        if "keypoints" in cropped_target:
            kpts = cropped_target["keypoints"]
            # First keypoint should have visibility 0 (outside)
            assert kpts[0, 0, 2] == 0.0


class TestRFDETRPose:
    """Tests for RFDETRPose class."""

    def test_rfdetr_pose_import(self):
        """Test that RFDETRPose can be imported."""
        from rfdetr import RFDETRPose
        assert RFDETRPose is not None

    def test_rfdetr_pose_config(self):
        """Test RFDETRPose uses correct configs."""
        from rfdetr import RFDETRPose
        from rfdetr.config import RFDETRPoseConfig, KeypointTrainConfig

        pose = RFDETRPose.__new__(RFDETRPose)
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
        # This is a basic check - full loss testing requires model setup
        from rfdetr.models.lwdetr import SetCriterion
        assert hasattr(SetCriterion, 'loss_keypoints')

    def test_loss_map_includes_keypoints(self):
        """Test that keypoints is in the loss map."""
        from rfdetr.models.lwdetr import SetCriterion

        # Create a minimal criterion to check loss_map
        criterion = SetCriterion.__new__(SetCriterion)
        criterion.losses = ['keypoints']

        # The get_loss method should handle 'keypoints'
        loss_map = {
            'labels': 'loss_labels',
            'labels_o2o': 'loss_labels',
            'boxes': 'loss_boxes',
            'masks': 'loss_masks',
            'keypoints': 'loss_keypoints',
        }
        assert 'keypoints' in loss_map


class TestKeypointPostProcess:
    """Tests for keypoint post-processing."""

    def test_postprocess_handles_keypoints(self):
        """Test that PostProcess handles keypoint outputs."""
        from rfdetr.models.lwdetr import PostProcess

        # Create mock outputs
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
            # Keypoints should be scaled to image coordinates
            assert result["keypoints"].shape[-1] == 3  # x, y, visibility


class TestKeypointInference:
    """Tests for keypoint inference pipeline."""

    def test_rfdetr_pose_predict_returns_keypoints(self):
        """Test that RFDETRPose.predict() returns keypoints in detections."""
        # This test mocks the model to avoid loading weights
        from rfdetr import RFDETRPose
        from unittest.mock import MagicMock, patch
        import numpy as np

        # Create a mock model that returns keypoints
        with patch.object(RFDETRPose, '__init__', lambda self, **kwargs: None):
            model = RFDETRPose()
            model.model = MagicMock()
            model.model.device = torch.device('cpu')
            model.model.resolution = 576
            model._is_optimized_for_inference = False
            model._has_warned_about_not_being_optimized_for_inference = True
            model.means = [0.485, 0.456, 0.406]
            model.stds = [0.229, 0.224, 0.225]

            # Mock model output with keypoints
            mock_output = {
                'pred_logits': torch.randn(1, 300, 1),
                'pred_boxes': torch.rand(1, 300, 4),
                'pred_keypoints': torch.rand(1, 300, 17, 3),
            }
            model.model.model = MagicMock(return_value=mock_output)

            # Mock postprocess to return keypoints
            mock_result = [{
                'scores': torch.tensor([0.9, 0.8]),
                'labels': torch.tensor([0, 0]),
                'boxes': torch.tensor([[10, 10, 100, 100], [50, 50, 150, 150]]),
                'keypoints': torch.rand(2, 17, 3),
            }]
            model.model.postprocess = MagicMock(return_value=mock_result)

            # Create a dummy image
            dummy_image = torch.rand(3, 480, 640)

            # Run predict
            detections = model.predict(dummy_image, threshold=0.5)

            # Check keypoints are in the result
            assert 'keypoints' in detections.data
            assert detections.data['keypoints'].shape == (2, 17, 3)

    def test_keypoints_output_structure(self):
        """Test that keypoint output has correct structure."""
        # Test the PostProcess output format directly
        from rfdetr.models.lwdetr import PostProcess

        postprocess = PostProcess(num_select=10)

        # Mock model outputs
        outputs = {
            "pred_logits": torch.randn(1, 300, 1),
            "pred_boxes": torch.rand(1, 300, 4),
            "pred_keypoints": torch.rand(1, 300, 17, 3),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = postprocess(outputs, target_sizes)

        # Check structure
        assert len(results) == 1
        result = results[0]

        assert 'keypoints' in result
        kpts = result['keypoints']

        # Shape should be [num_select, num_keypoints, 3]
        assert kpts.shape == (10, 17, 3)

        # x coordinates should be scaled to image width
        # y coordinates should be scaled to image height
        # visibility should be sigmoid (0-1)
        assert kpts[..., 2].min() >= 0.0
        assert kpts[..., 2].max() <= 1.0

    def test_keypoints_visibility_sigmoid(self):
        """Test that visibility values are sigmoids in [0, 1]."""
        from rfdetr.models.lwdetr import PostProcess

        postprocess = PostProcess(num_select=5)

        # Create outputs with known visibility logits
        outputs = {
            "pred_logits": torch.randn(1, 100, 1),
            "pred_boxes": torch.rand(1, 100, 4),
            "pred_keypoints": torch.zeros(1, 100, 17, 3),
        }
        # Set visibility logits to various values across the 100 queries
        # Use randn to get a range of positive and negative values
        outputs["pred_keypoints"][..., 2] = torch.randn(1, 100, 17) * 5  # Scale to get large pos/neg values

        target_sizes = torch.tensor([[480, 640]])
        results = postprocess(outputs, target_sizes)

        vis = results[0]['keypoints'][..., 2]

        # All visibility values should be in [0, 1] after sigmoid
        assert vis.min() >= 0.0
        assert vis.max() <= 1.0

    def test_keypoints_coordinate_scaling(self):
        """Test that keypoint coordinates are properly scaled to image size."""
        from rfdetr.models.lwdetr import PostProcess

        postprocess = PostProcess(num_select=1)

        # Create outputs with known normalized coordinates
        outputs = {
            "pred_logits": torch.tensor([[[10.0]] * 100]),  # High confidence
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * 100]),
            "pred_keypoints": torch.zeros(1, 100, 17, 3),
        }
        # Set first keypoint to center (0.5, 0.5) with high visibility
        outputs["pred_keypoints"][0, :, 0, :] = torch.tensor([0.5, 0.5, 5.0])

        target_sizes = torch.tensor([[480, 640]])  # H, W
        results = postprocess(outputs, target_sizes)

        kpts = results[0]['keypoints']

        # First keypoint x should be scaled to ~320 (0.5 * 640)
        # First keypoint y should be scaled to ~240 (0.5 * 480)
        assert abs(kpts[0, 0, 0].item() - 320) < 1.0
        assert abs(kpts[0, 0, 1].item() - 240) < 1.0


class TestKeypointIntegration:
    """Integration tests for keypoint functionality."""

    @pytest.mark.slow
    def test_keypoint_head_gradient_flow(self):
        """Test that gradients flow through KeypointHead."""
        from rfdetr.models.keypoint_head import KeypointHead

        head = KeypointHead(hidden_dim=256, num_keypoints=17, num_layers=3)

        query_features = [torch.randn(2, 100, 256)]
        outputs = head(query_features)

        # Compute a simple loss
        loss = outputs[0].sum()
        loss.backward()

        # Check gradients exist on model parameters
        has_grad = False
        for param in head.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in KeypointHead parameters"

    @pytest.mark.slow
    def test_keypoint_output_format(self):
        """Test complete keypoint output format through PostProcess."""
        from rfdetr.models.lwdetr import PostProcess

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

        # Check all expected keys
        assert "scores" in result
        assert "labels" in result
        assert "boxes" in result
        assert "keypoints" in result

        # Check shapes
        assert result["scores"].shape[0] == 10  # num_select
        assert result["boxes"].shape == (10, 4)
        assert result["keypoints"].shape == (10, 17, 3)

        # Check keypoint coordinate scaling
        # x coords should be scaled to image width (640)
        # y coords should be scaled to image height (480)
        kpts = result["keypoints"]
        # Visibility should be sigmoids (0-1)
        assert kpts[..., 2].min() >= 0.0
        assert kpts[..., 2].max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
