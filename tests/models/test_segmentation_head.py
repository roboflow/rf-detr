# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from contextlib import contextmanager

import pytest
import torch

from rfdetr.models.heads.segmentation import DepthwiseConvBlock


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="gpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available",
                ),
            ],
        ),
    ],
)
def test_depthwise_conv_block_forward(device: str) -> None:
    """DepthwiseConvBlock forward pass produces correct output shape without error."""
    block = DepthwiseConvBlock(dim=8).to(device)
    x = torch.randn(1, 8, 4, 4, device=device)
    y = block(x)
    assert y.shape == x.shape


def test_depthwise_conv_block_always_disables_cudnn(monkeypatch) -> None:
    """Depthwise conv should execute with cuDNN disabled for compatibility."""
    block = DepthwiseConvBlock(dim=8)
    cudnn_enabled = True

    class _MockDepthwiseConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert not cudnn_enabled
            return x

    fallback_dwconv = _MockDepthwiseConv()
    block.dwconv = fallback_dwconv

    fallback_context_calls = 0
    enabled_calls: list[bool] = []

    @contextmanager
    def _fake_cudnn_flags(*, enabled: bool):
        nonlocal cudnn_enabled, fallback_context_calls
        previous = cudnn_enabled
        cudnn_enabled = enabled
        enabled_calls.append(enabled)
        fallback_context_calls += 1
        try:
            yield
        finally:
            cudnn_enabled = previous

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_cudnn_flags)

    assert cudnn_enabled
    x = torch.randn(1, 8, 4, 4)
    y = block(x)
    assert cudnn_enabled

    assert y.shape == x.shape
    assert fallback_dwconv.calls == 1
    assert fallback_context_calls == 1
    assert enabled_calls == [False]
