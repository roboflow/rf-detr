# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for _resize_linear(), LWDETR.reinitialize_detection_head(), and _aggregate_keypoint_class_logits().

These tests guard against the out_features staleness bug where in-place .data mutation did not update
nn.Linear.out_features, causing ONNX export to emit stale (pre-fine-tuning) class counts.

Also covers the spurious "Keypoint class-logit boost has N classes but detection head has M" warning that fired when
num_keypoints_per_class exactly covered all foreground classes (correct configuration) but the comparison was against
class_embed.out_features which includes the background slot (+1).
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from rfdetr.models.lwdetr import LWDETR, _resize_linear


def _make_minimal_lwdetr(num_classes: int = 91, two_stage: bool = False) -> LWDETR:
    """Construct the smallest viable LWDETR without loading pretrained weights.

    Uses a MagicMock backbone and transformer with hidden_dim=4 so the model can be constructed in milliseconds without
    any network I/O.

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


def _make_keypoint_lwdetr(num_classes: int, num_keypoints_per_class: list[int]) -> LWDETR:
    """Construct a minimal keypoint-capable LWDETR with detection head resized to num_classes+1.

    Mirrors what happens after loading a pretrained checkpoint and fine-tuning to num_classes
    foreground categories: reinitialize_detection_head is called with num_classes+1 (includes
    background), so class_embed.out_features == num_classes+1 in the returned model.

    Args:
        num_classes: Number of foreground detection classes.
        num_keypoints_per_class: Keypoint count per foreground class.

    Returns:
        An LWDETR with use_grouppose_keypoints=True and class_embed.out_features==num_classes+1.

    Examples:
        >>> model = _make_keypoint_lwdetr(num_classes=2, num_keypoints_per_class=[17, 4])
        >>> model.class_embed.out_features
        3
    """
    hidden_dim = 4
    backbone = MagicMock()
    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.decoder = MagicMock()
    transformer.decoder.bbox_embed = None
    transformer.decoder.num_keypoints_per_class = num_keypoints_per_class
    transformer.decoder.keypoint_class_mask = torch.zeros(1, 1, dtype=torch.bool)
    transformer.num_keypoints_per_class = num_keypoints_per_class
    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=2,
        group_detr=1,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=num_keypoints_per_class,
    )
    # Simulate post-checkpoint-load state: detection head includes background slot.
    model.reinitialize_detection_head(num_classes + 1)
    return model


def _keypoint_tensor(num_keypoints_per_class: list[int], batch: int = 1, seq: int = 1) -> torch.Tensor:
    """Build a zero keypoint prediction tensor with the shape expected by _aggregate_keypoint_class_logits.

    The second-to-last dimension must equal num_kp_classes * max_kp (padded layout).

    Args:
        num_keypoints_per_class: Keypoint schema for the model.
        batch: Batch size dimension.
        seq: Sequence (query) dimension.

    Returns:
        Zero tensor of shape (batch, seq, num_kp_classes * max_kp, 8).

    Examples:
        >>> t = _keypoint_tensor([17, 4])
        >>> t.shape
        torch.Size([1, 1, 34, 8])
    """
    num_kp_classes = len(num_keypoints_per_class)
    max_kp = max(num_keypoints_per_class) if any(num_keypoints_per_class) else 1
    total_padded = num_kp_classes * max_kp
    return torch.zeros(batch, seq, total_padded, 8)


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
        """Bias=False input: returned module has bias=None and out_features is correct."""
        linear = nn.Linear(256, 91, bias=False)
        result = _resize_linear(linear, 8)
        assert result.out_features == 8, f"Expected out_features=8, got {result.out_features}"
        assert result.bias is None, "Expected bias=None for bias=False input"


class TestReinitializeDetectionHead:
    """Integration tests for LWDETR.reinitialize_detection_head().

    Uses a minimal LWDETR (hidden_dim=4, no real backbone) to verify that out_features is updated on the replaced
    nn.Linear modules — the core invariant required for correct ONNX export.
    """

    def test_updates_class_embed_out_features(self) -> None:
        """class_embed.out_features must reflect num_classes after reinitialize.

        The `num_outputs_including_background` argument represents the total number of classifier outputs (foreground
        classes plus background).
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

        The `num_outputs_including_background` argument represents the total number of classifier outputs (foreground
        classes plus background).
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


class TestAggregateKeypointClassLogits:
    """Regression tests for LWDETR._aggregate_keypoint_class_logits().

    Guards against a spurious warning that fired when num_keypoints_per_class exactly covered all
    foreground classes: class_embed.out_features includes background (+1), so len(schema)==num_classes
    always satisfied schema_len < detection_num_classes, triggering the warning incorrectly.

    Uses _kp_zero_pad_warned as a proxy for whether the warning fired — the rf-detr logger uses
    propagate=False which prevents standard caplog capture.
    """

    @pytest.mark.parametrize(
        "num_classes,num_keypoints_per_class",
        [
            pytest.param(1, [17], id="coco-person-1class"),
            pytest.param(2, [17, 4], id="basketball-2class"),
            pytest.param(3, [17, 4, 0], id="3class-schema-covers-all"),
        ],
    )
    def test_no_warning_when_schema_covers_all_foreground_classes(
        self,
        num_classes: int,
        num_keypoints_per_class: list[int],
    ) -> None:
        """No warning when num_keypoints_per_class covers exactly all foreground detection classes.

        Regression: the comparison used class_embed.out_features (num_classes+1) instead of
        num_classes, so a fully correct schema always triggered the spurious mismatch warning.
        """
        model = _make_keypoint_lwdetr(num_classes=num_classes, num_keypoints_per_class=num_keypoints_per_class)
        fake_kp = _keypoint_tensor(num_keypoints_per_class)

        model._aggregate_keypoint_class_logits(fake_kp)

        assert not model._kp_zero_pad_warned, (
            f"Spurious warning fired for schema={num_keypoints_per_class} with num_classes={num_classes}"
        )

    def test_warning_fires_when_schema_shorter_than_foreground_classes(self) -> None:
        """Warning fires when schema covers fewer classes than foreground detection classes.

        Scenario: 3 foreground classes but schema only covers 1 (e.g. only person has keypoints).
        The two uncovered foreground classes receive zero boost — a real mismatch worth warning about.
        """
        model = _make_keypoint_lwdetr(num_classes=3, num_keypoints_per_class=[17])
        fake_kp = _keypoint_tensor([17])

        model._aggregate_keypoint_class_logits(fake_kp)

        assert model._kp_zero_pad_warned, "Expected warning flag set for schema shorter than foreground class count"

    def test_output_shape_matches_detection_head(self) -> None:
        """Output shape is (batch, seq, detection_num_classes) regardless of schema length."""
        num_classes = 2
        schema = [17, 4]
        model = _make_keypoint_lwdetr(num_classes=num_classes, num_keypoints_per_class=schema)
        batch, seq = 2, 10
        fake_kp = _keypoint_tensor(schema, batch=batch, seq=seq)

        out = model._aggregate_keypoint_class_logits(fake_kp)

        assert out.shape == (batch, seq, num_classes + 1), (
            f"Expected shape {(batch, seq, num_classes + 1)}, got {out.shape}"
        )
