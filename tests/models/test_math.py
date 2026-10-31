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

    def test_upsamples_to_target_size(self) -> None:
        """Interpolate() upsamples a 4-D tensor to the requested spatial size."""
        x = torch.randn(2, 3, 4, 4)

        out = interpolate(x, size=[8, 8], mode="bilinear", align_corners=False)

        assert out.shape == (2, 3, 8, 8)

    def test_empty_batch_keeps_zero_batch_dim(self) -> None:
        """Interpolate() supports an empty batch dimension without error."""
        x = torch.randn(0, 3, 4, 4)

        out = interpolate(x, size=[8, 8], mode="nearest")

        assert out.shape == (0, 3, 8, 8)


class TestAccuracy:
    """Verify accuracy() computes precision@k correctly."""

    @pytest.mark.parametrize(
        ("output", "target", "expected"),
        [
            pytest.param([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]], [1, 0, 1], 100.0, id="all_correct"),
            pytest.param([[10.0, 0.0], [0.0, 10.0]], [1, 0], 0.0, id="all_wrong"),
        ],
    )
    def test_top1_accuracy(self, output: list[list[float]], target: list[int], expected: float) -> None:
        """Top-1 accuracy is 100.0 when every prediction is correct and 0.0 when none is."""
        result = accuracy(torch.tensor(output), torch.tensor(target), topk=(1,))

        assert len(result) == 1
        assert result[0].item() == pytest.approx(expected)

    def test_topk_pair_returns_two_values(self) -> None:
        """Topk=(1, 5) returns a list of length 2."""
        output = torch.randn(10, 10)
        target = torch.zeros(10, dtype=torch.long)
        result = accuracy(output, target, topk=(1, 5))
        assert len(result) == 2

    def test_empty_target_returns_single_zero(self) -> None:
        """Empty target returns list of length 1 with value 0 regardless of topk length."""
        output = torch.zeros(0, 5)
        target = torch.zeros(0, dtype=torch.long)
        result = accuracy(output, target, topk=(1, 5))
        assert len(result) == 1
        assert result[0].item() == pytest.approx(0.0)


class TestInverseSigmoid:
    """Verify inverse_sigmoid() computes the logit function correctly."""

    def test_half_maps_to_zero(self) -> None:
        """inverse_sigmoid(0.5) equals 0.0 since sigmoid(0.0) = 0.5."""
        x = torch.tensor([0.5])
        result = inverse_sigmoid(x)
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_saturated_input_is_finite(self, value: float) -> None:
        """inverse_sigmoid of 0.0 or 1.0 is finite thanks to eps clamping."""
        result = inverse_sigmoid(torch.tensor([value]))

        assert torch.isfinite(result).all()

    def test_preserves_input_shape(self) -> None:
        """Output shape matches input shape for a multi-dimensional tensor."""
        x = torch.rand(3, 4)
        result = inverse_sigmoid(x)
        assert result.shape == x.shape

    def test_gradient_nonzero_for_non_saturated_input(self) -> None:
        """Gradients are non-zero for a non-saturated input value."""
        x = torch.tensor([0.3], requires_grad=True)
        inverse_sigmoid(x).sum().backward()
        assert x.grad is not None
        assert x.grad.abs().item() > 0.0
