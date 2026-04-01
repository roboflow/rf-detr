# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regression tests for _resize_linear() and LWDETR.reinitialize_detection_head().

These tests guard against the out_features staleness bug where in-place .data
mutation did not update nn.Linear.out_features, causing ONNX export to emit
stale (pre-fine-tuning) class counts.
"""

from unittest.mock import MagicMock

import torch
from torch import nn

from rfdetr.models.lwdetr import LWDETR, _resize_linear


def _make_minimal_lwdetr(num_classes: int = 91, two_stage: bool = False) -> LWDETR:
    """Construct the smallest viable LWDETR without loading pretrained weights.

    Uses a MagicMock backbone and transformer with hidden_dim=4 so the model
    can be constructed in milliseconds without any network I/O.

    Args:
        num_classes: Initial number of output classes passed to LWDETR.
        two_stage: Whether to enable two-stage mode (creates enc_out_class_embed).

    Returns:
        An LWDETR instance with hidden_dim=4, num_queries=2, group_detr=1.

    Examples:
        >>> model = _make_minimal_lwdetr(num_classes=91)
        >>> isinstance(model, LWDETR)
        True
    """
    hidden_dim = 4
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
        num_queries=2,
        group_detr=1,
        two_stage=two_stage,
    )


class TestResizeLinear:
    """Unit tests for _resize_linear() — verifies out_features, weight shape, and bias shape."""

    def test_shrink_out_features(self) -> None:
        """Shrink: out_features equals the requested smaller class count."""
        result = _resize_linear(nn.Linear(256, 91), 8)
        assert result.out_features == 8, f"Expected out_features=8, got {result.out_features}"
        assert result.weight.shape == (8, 256), f"Expected weight (8, 256), got {result.weight.shape}"
        assert result.bias is not None
        assert result.bias.shape == (8,), f"Expected bias (8,), got {result.bias.shape}"

    def test_expand_out_features(self) -> None:
        """Expand: out_features equals the requested larger class count via tiling."""
        result = _resize_linear(nn.Linear(256, 10), 25)
        assert result.out_features == 25, f"Expected out_features=25, got {result.out_features}"
        assert result.weight.shape == (25, 256), f"Expected weight (25, 256), got {result.weight.shape}"
        assert result.bias is not None
        assert result.bias.shape == (25,), f"Expected bias (25,), got {result.bias.shape}"

    def test_same_size_preserves_values(self) -> None:
        """Same size: shapes and weight/bias values are preserved exactly."""
        linear = nn.Linear(256, 91)
        result = _resize_linear(linear, 91)
        assert result.out_features == 91
        assert result.weight.shape == (91, 256)
        assert result.bias is not None
        assert result.bias.shape == (91,)
        assert torch.allclose(result.weight.data, linear.weight.data)
        assert torch.allclose(result.bias.data, linear.bias.data)

    def test_no_bias_returns_no_bias(self) -> None:
        """bias=False input: returned module has bias=None and out_features is correct."""
        linear = nn.Linear(256, 91, bias=False)
        result = _resize_linear(linear, 8)
        assert result.out_features == 8, f"Expected out_features=8, got {result.out_features}"
        assert result.bias is None, "Expected bias=None for bias=False input"


class TestReinitializeDetectionHead:
    """Integration tests for LWDETR.reinitialize_detection_head().

    Uses a minimal LWDETR (hidden_dim=4, no real backbone) to verify that
    out_features is updated on the replaced nn.Linear modules — the core
    invariant required for correct ONNX export.
    """

    def test_updates_class_embed_out_features(self) -> None:
        """class_embed.out_features must reflect num_classes after reinitialize.

        The `num_outputs_including_background` argument represents the total number
        of classifier outputs (foreground classes plus background).
        """
        num_outputs_including_background = 8
        model = _make_minimal_lwdetr(num_classes=91)
        model.reinitialize_detection_head(num_outputs_including_background)
        assert model.class_embed.out_features == num_outputs_including_background, (
            f"Expected class_embed.out_features={num_outputs_including_background}, "
            f"got {model.class_embed.out_features}"
        )
        assert model.class_embed.weight.shape == (num_outputs_including_background, 4), (
            f"Expected weight ({num_outputs_including_background}, 4), got {model.class_embed.weight.shape}"
        )

    def test_two_stage_updates_enc_out_class_embed(self) -> None:
        """enc_out_class_embed entries must also have updated out_features in two-stage mode.

        The `num_outputs_including_background` argument represents the total number
        of classifier outputs (foreground classes plus background).
        """
        num_outputs_including_background = 8
        model = _make_minimal_lwdetr(num_classes=91, two_stage=True)
        model.reinitialize_detection_head(num_outputs_including_background)
        enc_embeds = model.transformer.enc_out_class_embed
        assert len(enc_embeds) > 0, "enc_out_class_embed should be non-empty in two-stage mode"
        for i, embed in enumerate(enc_embeds):
            assert embed.out_features == num_outputs_including_background, (
                f"enc_out_class_embed[{i}].out_features={embed.out_features}, "
                f"expected {num_outputs_including_background}"
            )
            assert embed.weight.shape == (num_outputs_including_background, 4), (
                f"enc_out_class_embed[{i}].weight.shape={embed.weight.shape}, "
                f"expected ({num_outputs_including_background}, 4)"
            )
