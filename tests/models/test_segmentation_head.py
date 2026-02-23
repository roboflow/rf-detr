# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from contextlib import contextmanager

import torch

from rfdetr.models.segmentation_head import DepthwiseConvBlock


class _EchoDepthwise(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x


def test_depthwise_conv_runs_with_cudnn_disabled(monkeypatch) -> None:
    """Depthwise conv should execute with cuDNN disabled for compatibility."""
    block = DepthwiseConvBlock(dim=8)
    fallback_dwconv = _EchoDepthwise()
    block.dwconv = fallback_dwconv

    fallback_context_calls = 0

    @contextmanager
    def _fake_cudnn_flags(*, enabled: bool):
        nonlocal fallback_context_calls
        assert enabled is False
        fallback_context_calls += 1
        yield

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_cudnn_flags)

    x = torch.randn(1, 8, 4, 4)
    y = block(x)

    assert y.shape == x.shape
    assert fallback_dwconv.calls == 1
    assert fallback_context_calls == 1
