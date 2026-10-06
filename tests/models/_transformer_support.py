# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Helpers shared by the decoder-rewrite tests (``test_transformer.py`` and ``test_transformer_rewrites.py``)."""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.transformer import TransformerDecoderLayer

#: Patch target of the shared eager-CUDA gate every decoder rewrite consults.
EAGER_CUDA = "rfdetr.models.transformer._eager_cuda"


def gate_open(tensor: torch.Tensor) -> bool:
    """Stand in for the eager-CUDA gate with one that is always open, so CPU tensors take the CUDA routing.

    Use it as ``@patch(EAGER_CUDA, new=gate_open)``. The rewrites' ops are device independent, which is what lets a
    CPU test follow the routing the gate decides.

    Examples:
        >>> gate_open(torch.zeros(1))
        True
    """
    return True


def decoder_layer(
    dropout: float = 0.0, *, d_model: int = 16, heads: int = 4, group_detr: int = 3
) -> TransformerDecoderLayer:
    """Build a small decoder layer in training mode, three groups of 4-head 16-wide attention by default.

    Args:
        dropout: Dropout probability of the layer.
        d_model: Embedding width; must be divisible by ``heads``.
        heads: Head count of both the self- and the cross-attention.
        group_detr: Number of query groups.

    Examples:
        >>> decoder_layer().group_detr
        3
        >>> decoder_layer(d_model=12, heads=1, group_detr=1).self_attn.num_heads
        1
    """
    return TransformerDecoderLayer(
        d_model=d_model,
        sa_nhead=heads,
        ca_nhead=heads,
        dim_feedforward=32,
        dropout=dropout,
        group_detr=group_detr,
        num_feature_levels=2,
    ).train()


def record_apply_calls(monkeypatch: pytest.MonkeyPatch, function: type[torch.autograd.Function]) -> list[tuple]:
    """Spy on ``function.apply``: record every call's arguments and still run the real autograd function.

    Args:
        monkeypatch: The test's monkeypatch fixture, which restores ``apply`` afterwards.
        function: The autograd function to spy on.

    Returns:
        The list that receives one argument tuple per ``apply`` call.

    Examples:
        >>> class Twice(torch.autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, x):
        ...         return x * 2
        >>> with pytest.MonkeyPatch.context() as patch:
        ...     calls = record_apply_calls(patch, Twice)
        ...     Twice.apply(torch.ones(1))
        ...     len(calls)
        tensor([2.])
        1
    """
    calls: list[tuple] = []
    real_apply = function.apply
    monkeypatch.setattr(function, "apply", lambda *args: (calls.append(args), real_apply(*args))[1])
    return calls
