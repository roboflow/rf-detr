# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from contextlib import contextmanager

import torch

from rfdetr.models.segmentation_head import DepthwiseConvBlock


def test_depthwise_conv_disables_cudnn(monkeypatch) -> None:
    """Depthwise conv should execute with cuDNN disabled for compatibility."""
    block = DepthwiseConvBlock(dim=8)
    active_enabled: bool | None = None

    class _MockDepthwiseConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert active_enabled is False
            return x

    fallback_dwconv = _MockDepthwiseConv()
    block.dwconv = fallback_dwconv

    fallback_context_calls = 0
    enabled_value: bool | None = None

    @contextmanager
    def _fake_cudnn_flags(*, enabled: bool):
        nonlocal active_enabled, enabled_value, fallback_context_calls
        previous = active_enabled
        active_enabled = enabled
        enabled_value = enabled
        fallback_context_calls += 1
        try:
            yield
        finally:
            active_enabled = previous

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_cudnn_flags)

    x = torch.randn(1, 8, 4, 4)
    y = block(x)

    assert y.shape == x.shape
    assert fallback_dwconv.calls == 1
    assert fallback_context_calls == 1
    assert enabled_value is False
