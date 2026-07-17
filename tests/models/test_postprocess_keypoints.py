# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for keypoint decoding in PostProcess."""

import pytest
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


def test_postprocess_keypoints_trace_alpha_rescores_active_keypoints_only() -> None:
    """Trace fusion should use active keypoints for the predicted class and ignore padded slots."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[2, 1], trace_alpha=1.0)
    outputs = {
        "pred_logits": torch.tensor([[[-10.0, 0.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 4, 8), dtype=torch.float32),
    }
    # class 1 has one active slot at flat index 2 and one padded inactive slot at flat index 3.
    outputs["pred_keypoints"][0, 0, 2, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 2, 4] = 0.0
    outputs["pred_keypoints"][0, 0, 2, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 2, 6] = 0.0
    outputs["pred_keypoints"][0, 0, 3, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 3, 4] = -2.0
    outputs["pred_keypoints"][0, 0, 3, 6] = -2.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    expected_score = torch.tensor([0.2], dtype=torch.float32)
    torch.testing.assert_close(results[0]["scores"], expected_score, rtol=1e-4, atol=1e-6)


def test_postprocess_keypoints_trace_alpha_normalizes_large_fused_scores() -> None:
    """Trace-fused keypoint scores should be bounded after empirical normalization."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[1], trace_alpha=1.0)
    outputs = {
        "pred_logits": torch.tensor([[[10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 1, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, 0, 0, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 0, 4] = 2.0
    outputs["pred_keypoints"][0, 0, 0, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 0, 6] = 2.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    original_score = torch.sigmoid(torch.tensor([10.0], dtype=torch.float32))
    mean_trace = torch.tensor([2.0], dtype=torch.float32) * torch.exp(torch.tensor([-4.0], dtype=torch.float32))
    fused_score = original_score * mean_trace.pow(-1.0)
    expected_score = fused_score / (1.0 + fused_score)
    assert fused_score.item() > 1.0
    assert 0.0 < results[0]["scores"].item() < 1.0
    torch.testing.assert_close(results[0]["scores"], expected_score, rtol=1e-4, atol=1e-6)


def test_postprocess_keypoints_trace_alpha_clamps_overflowing_fused_scores() -> None:
    """Trace fusion should stay finite and strictly below 1.0 when the raw fused score overflows."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[1], trace_alpha=1.0)
    outputs = {
        "pred_logits": torch.tensor([[[0.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 1, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, 0, 0, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 0, 4] = 50.0
    outputs["pred_keypoints"][0, 0, 0, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 0, 6] = 50.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    score = results[0]["scores"]
    expected_score = torch.nextafter(torch.ones_like(score), torch.zeros_like(score))
    assert torch.isfinite(score).all()
    assert 0.0 < score.item() < 1.0
    torch.testing.assert_close(score, expected_score, rtol=0.0, atol=0.0)


def test_postprocess_keypoints_trace_alpha_uses_log_space_for_extreme_trace() -> None:
    """Trace fusion should stay finite for extreme covariance terms."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[1])
    outputs = {
        "pred_logits": torch.tensor([[[0.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 1, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, 0, 0, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 0, 4] = -50.0
    outputs["pred_keypoints"][0, 0, 0, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 0, 6] = 0.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    expected_score = torch.tensor([0.5], dtype=torch.float32) * torch.exp(torch.tensor([-20.0], dtype=torch.float32))
    torch.testing.assert_close(results[0]["scores"], expected_score, rtol=1e-4, atol=1e-12)


def test_postprocess_validate_outputs_raises_when_masks_and_keypoints_both_present() -> None:
    """PostProcess should raise ValueError when both pred_masks and pred_keypoints are present."""
    postprocess = PostProcess(num_select=10)
    outputs = {
        "pred_logits": torch.zeros((1, 2, 2)),
        "pred_boxes": torch.zeros((1, 2, 4)),
        "pred_masks": torch.zeros((1, 2, 4, 4)),
        "pred_keypoints": torch.zeros((1, 2, 17, 8)),
    }
    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)

    with pytest.raises(ValueError, match="cannot be used together"):
        postprocess(outputs, target_sizes)
