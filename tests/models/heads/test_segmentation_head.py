# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for DepthwiseConvBlock, _DepthwiseConvWithoutCuDNN, and SegmentationHead."""

from contextlib import contextmanager
from unittest import mock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models.heads.segmentation import DepthwiseConvBlock, SegmentationHead, point_sample
from rfdetr.utilities.tensors import _nearest_grid_sample


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

    The previous fix (PR #728) only wrapped the forward pass in a context manager.  The backward kernels ran with cuDNN
    re-enabled, causing RuntimeError on T4/P100 GPUs.
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

    Verifies that _DepthwiseConvWithoutCuDNN produces the same gradients as a standard nn.Conv2d forward+backward (run
    with cuDNN disabled globally).
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

    On T4/P100, trainer resolves amp=True to '16-mixed'.  In that mode the backward receives fp16 grad_output while the
    saved weight stays fp32. Without explicit dtype casting, conv2d_input raises:
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


def test_depthwise_conv_backward_bf16_activation_keeps_grads_fp32() -> None:
    """grad_input and weight.grad must be fp32 when saved activation x is bf16 (issue #959).

    Under bf16-mixed AMP, the activation x entering _DepthwiseConvWithoutCuDNN is bf16 while weight stays fp32.  The old
    code cast grad_input back to x.dtype (bf16), propagating bf16 gradients to fp32 backbone parameters so that
    param.grad.dtype became bf16.  Fused AdamW then crashed with 'params, grads, exp_avgs, and exp_avg_sqs must have
    same dtype, device, and layout' (see issue #959).  The fix keeps grad_input in weight.dtype (fp32).

    This test drives the backward directly with a bf16 saved activation to reproduce the dtype that is present at
    training time without requiring a GPU.
    """
    import types

    from rfdetr.models.heads.segmentation import _DepthwiseConvWithoutCuDNN

    dim = 8
    weight = torch.randn(dim, 1, 3, 3, requires_grad=True)  # fp32 parameter (never cast by AMP)
    x_bf16 = torch.randn(1, dim, 4, 4, dtype=torch.bfloat16)  # bf16 activation (cast by AMP forward)
    grad_output = torch.ones(1, dim, 4, 4, dtype=torch.bfloat16)  # bf16 grad (from bf16 backward)

    # Build a minimal context mirroring what ctx would contain after the AMP forward pass.
    ctx = types.SimpleNamespace()
    ctx.saved_tensors = (x_bf16, weight)
    ctx.has_bias = False
    ctx.stride = (1, 1)
    ctx.padding = (1, 1)
    ctx.dilation = (1, 1)
    ctx.groups = dim
    ctx.needs_input_grad = [True, True, False, False, False, False, False]

    grad_input, grad_weight, *_ = _DepthwiseConvWithoutCuDNN.backward(ctx, grad_output)

    assert grad_input is not None, "grad_input should not be None when needs_input_grad[0] is True"
    assert grad_input.dtype == torch.float32, (
        f"grad_input is {grad_input.dtype} — bf16 grad_input propagates to fp32 backbone params "
        "and crashes fused AdamW (issue #959)"
    )
    assert grad_weight is not None, "grad_weight should not be None when needs_input_grad[1] is True"
    assert grad_weight.dtype == torch.float32, (
        f"grad_weight is {grad_weight.dtype} — weight grad must stay fp32 to match param dtype"
    )


def test_depthwise_conv_no_cudnn_bias_none() -> None:
    """_DepthwiseConvWithoutCuDNN forward and backward work correctly with bias=None.

    Exercises the ctx.has_bias=False branch in forward and the grad_bias=None return in backward — never reached via
    DepthwiseConvBlock (always has bias).
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


@pytest.mark.parametrize(
    "layer_scale_init_value", [pytest.param(0, id="no_gamma"), pytest.param(1e-6, id="with_gamma")]
)
def test_depthwise_conv_block_layer_scale(layer_scale_init_value: float) -> None:
    """DepthwiseConvBlock with and without layer scaling produces valid output and gradients.

    Exercises the gamma=None (layer_scale_init_value=0) and gamma!=None (layer_scale_init_value>0) branches in
    DepthwiseConvBlock.forward().
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


class TestSegmentationHeadSkipBlocksAppliesProjection:
    """``skip_blocks=True`` must run ``spatial_features_proj`` on the spatial features, exactly like the
    ``skip_blocks=False`` branch and ``forward_export()`` both already do — the two-stage encoder-only mask path
    (``skip_blocks=True``) is not a distinct architecture, just a shorter path through the same head, so it must not
    skip a learned layer the other paths apply."""

    @staticmethod
    def _build_head(bottleneck_ratio: int = 1) -> SegmentationHead:
        """Build a tiny head whose spatial_features_proj is a real, non-identity layer.

        A non-``None`` bottleneck ratio makes ``spatial_features_proj`` a real, randomly initialized convolution rather
        than ``nn.Identity()``, so skipping it is numerically observable. Ratios greater than one also verify the
        channel-reducing contract shared by the spatial and query projections.

        Examples:
            >>> head = TestSegmentationHeadSkipBlocksAppliesProjection._build_head()
            >>> isinstance(head.spatial_features_proj, torch.nn.Conv2d)
            True
        """
        return SegmentationHead(in_dim=4, num_blocks=1, bottleneck_ratio=bottleneck_ratio, downsample_ratio=1)

    def test_forward_skip_blocks_applies_spatial_features_proj(self) -> None:
        """forward(skip_blocks=True) must project spatial_features before the mask einsum."""
        head = self._build_head()
        spatial_features = torch.randn(1, 4, 4, 4)
        query_features = torch.randn(1, 2, 4)
        image_size = (4, 4)

        with torch.no_grad():
            resized = F.interpolate(spatial_features, size=image_size, mode="bilinear", align_corners=False)
            expected_proj = head.spatial_features_proj(resized)
            expected_qf = head.query_features_proj(head.query_features_block(query_features))
            expected = torch.einsum("bchw,bnc->bnhw", expected_proj, expected_qf) + head.bias

            actual = head.forward(spatial_features, [query_features], image_size, skip_blocks=True)[0]

        torch.testing.assert_close(actual, expected)

    def test_sparse_forward_skip_blocks_applies_spatial_features_proj(self) -> None:
        """sparse_forward(skip_blocks=True) must return the already-projected spatial_features in its dict output."""
        head = self._build_head()
        spatial_features = torch.randn(1, 4, 4, 4)
        query_features = torch.randn(1, 2, 4)
        image_size = (4, 4)

        with torch.no_grad():
            resized = F.interpolate(spatial_features, size=image_size, mode="bilinear", align_corners=False)
            expected_proj = head.spatial_features_proj(resized)

            actual = head.sparse_forward(spatial_features, [query_features], image_size, skip_blocks=True)[0]

        torch.testing.assert_close(actual["spatial_features"], expected_proj)

    @pytest.mark.parametrize(
        "bottleneck_ratio",
        [pytest.param(1, id="same_channels"), pytest.param(2, id="projected_channels")],
    )
    def test_forward_export_skip_blocks_matches_training_path(self, bottleneck_ratio: int) -> None:
        """Exported encoder-only masks must apply the same spatial projection as the training path."""
        head = self._build_head(bottleneck_ratio)
        spatial_features = torch.randn(1, 4, 4, 4)
        query_features = [torch.randn(1, 2, 4)]
        image_size = (4, 4)

        with torch.no_grad():
            expected = head.forward(spatial_features, query_features, image_size, skip_blocks=True)[0]
            head.export()
            actual = head.forward(spatial_features, query_features, image_size, skip_blocks=True)[0]

        assert actual.shape == (1, 2, 4, 4)
        torch.testing.assert_close(actual, expected)


class TestSegmentationHeadSkipBlocksFalseUnaffected:
    """Regression guard for the ``skip_blocks=False`` branch, untouched by this fix.

    ``SegmentationHead`` had no test coverage at all before this PR (neither branch), so this class
    covers the branch this fix does not modify, alongside ``TestSegmentationHeadSkipBlocksAppliesProjection``
    for the branch it does.
    """

    def test_forward_skip_blocks_false_applies_spatial_features_proj(self) -> None:
        """forward(skip_blocks=False) already projects spatial_features per decoder layer."""
        head = SegmentationHead(in_dim=4, num_blocks=2, bottleneck_ratio=1, downsample_ratio=1)
        spatial_features = torch.randn(1, 4, 4, 4)
        query_features = [torch.randn(1, 2, 4), torch.randn(1, 2, 4)]
        image_size = (4, 4)

        with torch.no_grad():
            resized = F.interpolate(spatial_features, size=image_size, mode="bilinear", align_corners=False)
            expected_logits = []
            block_input = resized
            for block, qf in zip(head.blocks, query_features):
                block_input = block(block_input)
                sf_proj = head.spatial_features_proj(block_input)
                qf_proj = head.query_features_proj(head.query_features_block(qf))
                expected_logits.append(torch.einsum("bchw,bnc->bnhw", sf_proj, qf_proj) + head.bias)

            actual_logits = head.forward(spatial_features, query_features, image_size, skip_blocks=False)

        assert len(actual_logits) == len(expected_logits)
        for actual, expected in zip(actual_logits, expected_logits):
            torch.testing.assert_close(actual, expected)


class TestPointSampleNearestRouting:
    """``mode="nearest"`` must use the backend-agnostic gather path, not ``F.grid_sample``.

    On XLA ``F.grid_sample`` lowers to an ``aten::grid_sampler_2d`` host fallback, which is what
    ``SetCriterion.loss_masks`` hits when it samples ground-truth mask labels.
    """

    @pytest.mark.parametrize("padding_mode", ["zeros", "border"])
    def test_nearest_routes_through_the_gather_helper(self, padding_mode: str) -> None:
        """Without the routing change this call reaches ``F.grid_sample`` instead."""
        input = torch.randn(1, 2, 6, 6)
        point_coords = torch.rand(1, 5, 2)

        with mock.patch(
            "rfdetr.models.heads.segmentation._nearest_grid_sample",
            wraps=_nearest_grid_sample,
        ) as spy:
            point_sample(input, point_coords, mode="nearest", padding_mode=padding_mode)

        assert spy.call_count == 1

    def test_nearest_output_matches_grid_sample(self) -> None:
        """Routing must not change values: the helper delegates to F.grid_sample off MPS/XLA."""
        input = torch.randn(1, 3, 7, 7)
        point_coords = torch.rand(1, 9, 2)

        actual = point_sample(input, point_coords, mode="nearest")
        grid = (2.0 * point_coords - 1.0).unsqueeze(2)
        expected = F.grid_sample(input, grid, mode="nearest", padding_mode="border", align_corners=False).squeeze(3)

        assert torch.equal(actual, expected)

    def test_unsupported_mode_still_delegates(self) -> None:
        """A mode the gather path does not implement must keep reaching F.grid_sample."""
        input = torch.randn(1, 1, 5, 5)
        point_coords = torch.rand(1, 4, 2)

        with mock.patch("rfdetr.models.heads.segmentation.F.grid_sample", wraps=F.grid_sample) as spy:
            point_sample(input, point_coords, mode="bicubic")

        assert spy.call_count == 1
