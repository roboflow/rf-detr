# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for dual-projector backbone joiner routing."""

from __future__ import annotations

import torch
from torch import nn

from rfdetr.models.backbone import Joiner
from rfdetr.utilities.tensors import NestedTensor


class _FakeBackbone(nn.Module):
    """Backbone shim used to validate Joiner contract changes."""

    def __init__(
        self,
        features: list[NestedTensor],
        cross_attention_features: list[object] | None,
    ) -> None:
        super().__init__()
        self._features = features
        self._cross_attention_features = cross_attention_features

    def forward(self, tensor: torch.Tensor | NestedTensor):
        if isinstance(tensor, torch.Tensor):
            feats = [f.tensors for f in self._features]
            masks = [f.mask for f in self._features]
            return feats, masks, self._cross_attention_features
        return self._features, self._cross_attention_features


class _FakePositionEncoding(nn.Module):
    """Tiny callable that behaves like a position encoder."""

    def forward(self, nested_tensor: NestedTensor | torch.Tensor, align_dim_orders: bool = False) -> torch.Tensor:
        if isinstance(nested_tensor, NestedTensor):
            base = nested_tensor.tensors
        else:
            base = nested_tensor
        if base.dim() == 3:
            base = base[:, None]
        return torch.zeros((base.shape[0], 1, base.shape[-2], base.shape[-1]), dtype=base.dtype, device=base.device)


def _feature(shape: tuple[int, ...], batch_size: int = 2) -> NestedTensor:
    channels, height, width = shape
    return NestedTensor(
        tensors=torch.ones((batch_size, channels, height, width), dtype=torch.float32),
        mask=torch.zeros((batch_size, height, width), dtype=torch.bool),
    )


def _input_tensor(batch_size: int = 2) -> tuple[NestedTensor, torch.Tensor]:
    return (
        NestedTensor(
            tensors=torch.ones((batch_size, 3, 16, 16), dtype=torch.float32),
            mask=torch.zeros((batch_size, 16, 16), dtype=torch.bool),
        ),
        torch.ones((batch_size, 3, 16, 16), dtype=torch.float32),
    )


def test_joiner_dual_projector_disabled_contract() -> None:
    """Joiner should forward one feature stream and a ``None`` cross-attention stream when disabled."""
    features = [_feature((256, 16, 16))]
    joiner = Joiner(_FakeBackbone(features, None), _FakePositionEncoding())

    input_tensor, image = _input_tensor()

    _, _, cross_attention = joiner(input_tensor)
    assert cross_attention is None
    assert len(joiner(input_tensor)[0]) == 1

    exported = joiner.forward_export(image)
    assert exported[3] is None
    assert len(exported[0]) == 1
    assert exported[2][0].shape == (2, 16, 16)


def test_joiner_dual_projector_enabled_contract() -> None:
    """Joiner should forward cross-attention features in parallel with feature features when enabled."""
    features = [_feature((256, 16, 16)), _feature((256, 8, 8))]
    cross_attention_features = [_feature((256, 16, 16)), _feature((256, 8, 8))]
    joiner = Joiner(_FakeBackbone(features, cross_attention_features), _FakePositionEncoding())

    input_tensor, _ = _input_tensor()

    feature_tensors, _, cross_attention = joiner(input_tensor)
    assert len(feature_tensors) == len(cross_attention)
    assert all(f.tensors.shape == c.tensors.shape for f, c in zip(feature_tensors, cross_attention))
    assert all(f.mask is not None for f in cross_attention)


def test_joiner_forward_export_contract() -> None:
    """Exported joiner contracts should remain 4-tuples and preserve cross-attention stream arity."""
    exported_features = [torch.ones(2, 256, 16, 16), torch.ones(2, 256, 8, 8)]
    exported_masks = [torch.zeros(2, 16, 16, dtype=torch.bool), torch.zeros(2, 8, 8, dtype=torch.bool)]
    export_backbone = _FakeBackbone(
        [NestedTensor(t, mask) for t, mask in zip(exported_features, exported_masks)],
        [torch.ones(2, 256, 16, 16), torch.ones(2, 256, 8, 8)],
    )
    joiner = Joiner(export_backbone, _FakePositionEncoding())

    outputs = joiner.forward_export(torch.ones(2, 3, 16, 16))
    feats_out, masks_out, poss, cross_attention = outputs

    assert len(feats_out) == len(exported_features)
    assert len(masks_out) == len(exported_masks)
    assert feats_out[0].shape == exported_features[0].shape
    assert masks_out[0].shape == exported_masks[0].shape
    assert len(outputs) == 4
    assert poss[0].shape == exported_features[0][:, :1, :, :].shape
    assert isinstance(cross_attention, list)
    assert all(isinstance(feature, torch.Tensor) for feature in cross_attention)
