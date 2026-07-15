# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for rfdetr.models.math utility functions."""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.math import accuracy, interpolate, inverse_sigmoid


class TestInterpolate:
    """Verify interpolate() delegates to F.interpolate across torchvision versions."""

    def test_resizes_to_target_size(self) -> None:
        """Interpolate() upsamples a 4-D tensor to the requested spatial size."""
        x = torch.randn(2, 3, 4, 4)

        out = interpolate(x, size=[8, 8], mode="bilinear", align_corners=False)

        assert out.shape == (2, 3, 8, 8)

    def test_handles_empty_batch(self) -> None:
        """Interpolate() supports an empty batch dimension without error."""
        x = torch.randn(0, 3, 4, 4)

        out = interpolate(x, size=[8, 8], mode="nearest")

        assert out.shape == (0, 3, 8, 8)


class TestAccuracy:
    """Verify accuracy() computes precision@k correctly."""

    def test_top1_perfect_batch(self) -> None:
        """All predictions correct returns top-1 accuracy of 100.0."""
        output = torch.tensor([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        target = torch.tensor([1, 0, 1])
        result = accuracy(output, target, topk=(1,))
        assert len(result) == 1
        assert result[0].item() == pytest.approx(100.0)

    def test_top1_zero_accuracy(self) -> None:
        """All predictions wrong returns top-1 accuracy of 0.0."""
        output = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        target = torch.tensor([1, 0])
        result = accuracy(output, target, topk=(1,))
        assert result[0].item() == pytest.approx(0.0)

    def test_topk_returns_list_of_correct_length(self) -> None:
        """Topk=(1, 5) returns a list of length 2."""
        output = torch.randn(10, 10)
        target = torch.zeros(10, dtype=torch.long)
        result = accuracy(output, target, topk=(1, 5))
        assert len(result) == 2

    def test_empty_target_returns_single_zero_regardless_of_topk(self) -> None:
        """Empty target returns list of length 1 with value 0 regardless of topk length."""
        output = torch.zeros(0, 5)
        target = torch.zeros(0, dtype=torch.long)
        result = accuracy(output, target, topk=(1, 5))
        assert len(result) == 1
        assert result[0].item() == pytest.approx(0.0)


class TestInverseSigmoid:
    """Verify inverse_sigmoid() computes the logit function correctly."""

    def test_identity_at_half(self) -> None:
        """inverse_sigmoid(0.5) equals 0.0 since sigmoid(0.0) = 0.5."""
        x = torch.tensor([0.5])
        result = inverse_sigmoid(x)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_clamping_at_zero_is_finite(self) -> None:
        """inverse_sigmoid(0.0) is finite due to eps clamping."""
        x = torch.tensor([0.0])
        result = inverse_sigmoid(x)
        assert torch.isfinite(result).all()

    def test_clamping_at_one_is_finite(self) -> None:
        """inverse_sigmoid(1.0) is finite due to eps clamping."""
        x = torch.tensor([1.0])
        result = inverse_sigmoid(x)
        assert torch.isfinite(result).all()

    def test_output_shape_matches_input(self) -> None:
        """Output shape matches input shape for a multi-dimensional tensor."""
        x = torch.rand(3, 4)
        result = inverse_sigmoid(x)
        assert result.shape == x.shape

    def test_gradient_flows_for_non_saturated_input(self) -> None:
        """Gradients are non-zero for a non-saturated input value."""
        x = torch.tensor([0.3], requires_grad=True)
        inverse_sigmoid(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().item() > 0.0
