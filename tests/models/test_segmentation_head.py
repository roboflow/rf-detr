# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for DepthwiseConvBlock and _DepthwiseConvWithoutCuDNN (segmentation head)."""

from contextlib import contextmanager

import pytest
import torch

from rfdetr.models.heads.segmentation import DepthwiseConvBlock


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)


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


def test_depthwise_conv_forward_disables_cudnn(monkeypatch) -> None:
    """Depthwise conv should execute with cuDNN disabled during forward."""
    block = DepthwiseConvBlock(dim=8)
    enabled_calls: list[bool] = []
    original_flags = torch.backends.cudnn.flags

    @contextmanager
    def _tracking_flags(*, enabled: bool):
        enabled_calls.append(enabled)
        with original_flags(enabled=enabled):
            yield

    monkeypatch.setattr(torch.backends.cudnn, "flags", _tracking_flags)

    x = torch.randn(1, 8, 4, 4)
    y = block(x)
    assert y.shape == x.shape
    assert enabled_calls, "torch.backends.cudnn.flags was never called"
    assert all(not e for e in enabled_calls)


def test_depthwise_conv_backward_disables_cudnn(monkeypatch) -> None:
    """Backward pass must also run with cuDNN disabled (issue #731).

    The previous fix (PR #728) only wrapped the forward pass in a context
    manager.  The backward kernels ran with cuDNN re-enabled, causing
    RuntimeError on T4/P100 GPUs.
    """
    block = DepthwiseConvBlock(dim=8)
    enabled_calls: list[bool] = []
    original_flags = torch.backends.cudnn.flags

    @contextmanager
    def _tracking_flags(*, enabled: bool):
        enabled_calls.append(enabled)
        with original_flags(enabled=enabled):
            yield

    monkeypatch.setattr(torch.backends.cudnn, "flags", _tracking_flags)

    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    y = block(x)
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    # cuDNN must be disabled for both forward and backward
    assert len(enabled_calls) >= 2
    assert all(not e for e in enabled_calls)


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
def test_depthwise_conv_backward_produces_correct_gradients(device: str) -> None:
    """Backward pass through DepthwiseConvBlock produces valid gradients."""
    block = DepthwiseConvBlock(dim=8).to(device)
    x = torch.randn(1, 8, 4, 4, device=device, requires_grad=True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_depthwise_conv_gradients_match_reference() -> None:
    """Custom autograd Function gradients match nn.Conv2d gradients.

    Verifies that _DepthwiseConvWithoutCuDNN produces the same gradients as
    a standard nn.Conv2d forward+backward (run with cuDNN disabled globally).
    """
    torch.manual_seed(42)
    dim = 8
    block = DepthwiseConvBlock(dim=dim)

    # Reference: run standard nn.Conv2d with cuDNN globally disabled
    x_ref = torch.randn(1, dim, 4, 4, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False):
        y_ref = block.dwconv(x_ref)
    y_ref.sum().backward()

    x_ref_grad = x_ref.grad.clone()
    weight_ref_grad = block.dwconv.weight.grad.clone()
    bias_ref_grad = block.dwconv.bias.grad.clone()

    # Our implementation via _depthwise_conv.  zero_grad() so that the second
    # backward does not accumulate into weight.grad from the first run.
    block.zero_grad()
    x_test = x_ref.detach().clone().requires_grad_(True)
    y_test = block._depthwise_conv(x_test)
    y_test.sum().backward()

    assert torch.allclose(y_ref, y_test, atol=1e-6)
    assert torch.allclose(x_ref_grad, x_test.grad, atol=1e-6)
    assert torch.allclose(weight_ref_grad, block.dwconv.weight.grad, atol=1e-6)
    assert torch.allclose(bias_ref_grad, block.dwconv.bias.grad, atol=1e-6)


def test_depthwise_conv_backward_fp16_grad_output() -> None:
    """Backward must not crash when grad_output is fp16 (AMP 16-mixed on T4/P100).

    On T4/P100, trainer resolves amp=True to '16-mixed'.  In that mode the
    backward receives fp16 grad_output while the saved weight stays fp32.
    Without explicit dtype casting, conv2d_input raises:
        RuntimeError: expected scalar type Half but found Float
    """
    dim = 8
    block = DepthwiseConvBlock(dim=dim)
    x = torch.randn(1, dim, 4, 4, requires_grad=True)

    # Simulate 16-mixed backward: forward in fp32, grad_output arrives as fp16
    y = block._depthwise_conv(x)
    grad_output = torch.ones_like(y, dtype=torch.float16)
    y.backward(grad_output)

    assert x.grad is not None
    assert x.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all()


def test_depthwise_conv_no_cudnn_bias_none() -> None:
    """_DepthwiseConvWithoutCuDNN forward and backward work correctly with bias=None.

    Exercises the ctx.has_bias=False branch in forward and the grad_bias=None
    return in backward — never reached via DepthwiseConvBlock (always has bias).
    """
    from rfdetr.models.heads.segmentation import _DepthwiseConvWithoutCuDNN

    dim = 8
    weight = torch.randn(dim, 1, 3, 3, requires_grad=True)
    x = torch.randn(1, dim, 4, 4, requires_grad=True)
    y = _DepthwiseConvWithoutCuDNN.apply(x, weight, None, (1, 1), (1, 1), (1, 1), dim)
    y_ref = torch.nn.functional.conv2d(x.detach(), weight.detach(), None, stride=1, padding=1, dilation=1, groups=dim)
    assert torch.allclose(y.detach(), y_ref, atol=1e-6)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape
    assert torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("layer_scale_init_value", [0, 1e-6], ids=["no_gamma", "with_gamma"])
def test_depthwise_conv_block_layer_scale(layer_scale_init_value: float) -> None:
    """DepthwiseConvBlock with and without layer scaling produces valid output and gradients.

    Exercises the gamma=None (layer_scale_init_value=0) and gamma!=None
    (layer_scale_init_value>0) branches in DepthwiseConvBlock.forward().
    """
    block = DepthwiseConvBlock(dim=8, layer_scale_init_value=layer_scale_init_value)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    if layer_scale_init_value > 0:
        assert block.gamma is not None
        assert block.gamma.grad is not None
