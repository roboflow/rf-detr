# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from contextlib import contextmanager

import torch

from rfdetr.models.segmentation_head import DepthwiseConvBlock


class _FailingThenPassingDepthwise(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("GET was unable to find an engine to execute this computation")
        return x


def test_depthwise_conv_retries_with_cudnn_disabled_on_get_engine_error(monkeypatch) -> None:
    """Retry depthwise conv with cuDNN disabled when engine selection fails."""
    block = DepthwiseConvBlock(dim=8)
    fallback_dwconv = _FailingThenPassingDepthwise()
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
    assert fallback_dwconv.calls == 2
    assert fallback_context_calls == 1
