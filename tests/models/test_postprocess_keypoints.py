# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for keypoint decoding in PostProcess."""

import torch

from rfdetr.models.postprocess import PostProcess


def test_postprocess_keypoints_shape_and_scores() -> None:
    """PostProcess should emit keypoints and raw precision parameters for top detections."""
    postprocess = PostProcess(num_select=2, num_keypoints_per_class=[17])
    outputs = {
        "pred_logits": torch.tensor([[[10.0, -10.0], [9.0, -10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.4, 0.6, 0.2, 0.3]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 2, 17, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, :, :, 0] = 0.5
    outputs["pred_keypoints"][0, :, :, 1] = 0.25
    outputs["pred_keypoints"][0, :, :, 2] = 3.0
    outputs["pred_keypoints"][0, :, :, 4] = 0.25
    outputs["pred_keypoints"][0, :, :, 5] = 0.5
    outputs["pred_keypoints"][0, :, :, 6] = -0.25

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)
    keypoints = results[0]["keypoints"]
    keypoint_precision = results[0]["keypoint_precision_cholesky"]

    assert keypoints.shape == (2, 17, 3)
    assert torch.allclose(keypoints[:, :, 0], torch.full((2, 17), 100.0))
    assert torch.allclose(keypoints[:, :, 1], torch.full((2, 17), 25.0))
    assert torch.all((keypoints[:, :, 2] > 0) & (keypoints[:, :, 2] < 1))
    assert keypoint_precision.shape == (2, 17, 3)
    torch.testing.assert_close(keypoint_precision[:, :, 0], torch.full((2, 17), 0.25))
    torch.testing.assert_close(keypoint_precision[:, :, 1], torch.full((2, 17), 0.5))
    torch.testing.assert_close(keypoint_precision[:, :, 2], torch.full((2, 17), -0.25))


def test_postprocess_keypoints_class_filtering() -> None:
    """Class-specific keypoint slots should be selected from padded per-class keypoint tensors."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[2, 1])
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 4, 8), dtype=torch.float32),
    }
    # class 0 slots: [0, 1], class 1 slots: [2, 3]
    outputs["pred_keypoints"][0, 0, 2, 0] = 0.25
    outputs["pred_keypoints"][0, 0, 2, 1] = 0.4
    outputs["pred_keypoints"][0, 0, 2, 2] = 2.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)
    keypoints = results[0]["keypoints"]
    keypoint_precision = results[0]["keypoint_precision_cholesky"]

    assert keypoints.shape == (1, 2, 3)
    assert torch.allclose(keypoints[0, 0, 0], torch.tensor(50.0))
    assert torch.allclose(keypoints[0, 0, 1], torch.tensor(40.0))
    assert 0.0 < keypoints[0, 0, 2].item() < 1.0
    torch.testing.assert_close(keypoints[0, 1], torch.zeros(3))
    torch.testing.assert_close(keypoint_precision[0, 1], torch.full((3,), float("nan")), equal_nan=True)
