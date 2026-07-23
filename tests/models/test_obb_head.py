# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.models.lwdetr import LWDETR


def _make_lwdetr(*, oriented: bool, num_classes: int = 91) -> LWDETR:
    """Construct a minimal LWDETR for testing the angle head."""
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
        oriented=oriented,
    )


def _predict_angle(model: LWDETR, hs: torch.Tensor) -> torch.Tensor:
    """Apply the angle projection exactly as ``LWDETR.forward`` does.

    Mirrors lwdetr.py:532 and lwdetr.py:654, which share this expression.
    """
    assert model.angle_embed is not None
    return model.angle_embed(hs).sigmoid() * math.pi


class TestLWDETRAngleHead:
    """The angle head lives inline in LWDETR, not in a separate head module."""

    @pytest.mark.parametrize(
        ("oriented", "expected"),
        [pytest.param(True, True, id="oriented"), pytest.param(False, False, id="axis-aligned")],
    )
    def test_angle_embed_present_only_when_oriented(self, oriented: bool, expected: bool) -> None:
        model = _make_lwdetr(oriented=oriented)
        assert (model.angle_embed is not None) is expected
        assert model.oriented is expected

    def test_angle_embed_is_zero_initialised(self) -> None:
        """The final layer is zero-init'd (lwdetr.py:213-215) so training starts neutral."""
        model = _make_lwdetr(oriented=True)
        assert model.angle_embed is not None
        assert torch.count_nonzero(model.angle_embed.layers[-1].weight) == 0
        assert torch.count_nonzero(model.angle_embed.layers[-1].bias) == 0

    def test_initial_angle_is_pi_over_two(self) -> None:
        """Zero-init means sigmoid(0)*pi, i.e. every box starts at 90 degrees.

        This is a consequence of the zero-init above, not an arbitrary constant: a non-zero init would bias every query
        toward some other orientation.
        """
        model = _make_lwdetr(oriented=True)
        angle = _predict_angle(model, torch.randn(2, 5, 8))
        assert torch.allclose(angle, torch.full_like(angle, math.pi / 2), atol=1e-6)

    def test_angle_output_shape(self) -> None:
        model = _make_lwdetr(oriented=True)
        assert _predict_angle(model, torch.randn(2, 5, 8)).shape == (2, 5, 1)

    def test_angle_stays_in_range_for_extreme_features(self) -> None:
        """The sigmoid()*pi projection bounds the angle whatever the feature magnitude."""
        model = _make_lwdetr(oriented=True)
        torch.nn.init.normal_(model.angle_embed.layers[-1].weight, std=10.0)  # type: ignore[union-attr]
        angle = _predict_angle(model, torch.randn(4, 6, 8) * 100)
        assert bool((angle >= 0).all())
        assert bool((angle <= math.pi).all())

    def test_angle_gradients_reach_the_features(self) -> None:
        model = _make_lwdetr(oriented=True)
        torch.nn.init.normal_(model.angle_embed.layers[-1].weight, std=0.1)  # type: ignore[union-attr]
        hs = torch.randn(2, 5, 8, requires_grad=True)

        _predict_angle(model, hs).sum().backward()

        assert hs.grad is not None
        assert torch.isfinite(hs.grad).all()
        assert torch.count_nonzero(hs.grad) > 0
