# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the decoder layer's grouped self-attention, fused linear + ReLU feed-forward, and the plain-module-call
gate that keeps observed or subclassed modules on their ``Module.__call__`` path."""

from collections.abc import Callable
from functools import partial

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from pytest_mock import MockerFixture
from torch import nn
from torch.utils.hooks import RemovableHandle

from rfdetr.models.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    _AddInDtype,
    _InterleavedSinCos,
    _is_tracing,
    _LinearReLU,
    _module_call_is_plain,
)
from tests._markers import _cuda_marks, _requires_cuda
from tests.models.transformer.helpers import decoder_layer, record_apply_calls


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def test_decoder_grouping_reuses_query_tensor_as_key() -> None:
    """Grouped self-attention should preserve layout while reusing query as key."""

    class _RecordingSelfAttention(nn.Module):
        """Fake self-attention that records whether ``query`` and ``key`` are the same object."""

        def __init__(self) -> None:
            super().__init__()
            self.query_is_key = False
            self.query: torch.Tensor | None = None
            self.key: torch.Tensor | None = None
            self.value: torch.Tensor | None = None

        def forward(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
        ) -> tuple[torch.Tensor, None]:
            """Record attention inputs, then return zeros shaped like ``query``."""
            self.query_is_key = query is key
            self.query = query.detach().clone()
            self.key = key.detach().clone()
            self.value = value.detach().clone()
            return torch.zeros_like(query), None

    class _ZeroCrossAttention(nn.Module):
        """Fake cross-attention that ignores its inputs and returns zeros shaped like ``query``."""

        def forward(self, query: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Return a zero tensor shaped like ``query``."""
            return torch.zeros_like(query)

    layer = TransformerDecoderLayer(
        d_model=16,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=32,
        dropout=0,
        group_detr=3,
        num_feature_levels=2,
    )
    self_attn = _RecordingSelfAttention()
    layer.self_attn = self_attn
    layer.cross_attn = _ZeroCrossAttention()

    tgt = torch.arange(2 * 12 * 16, dtype=torch.float32).reshape(2, 12, 16)
    memory = torch.zeros(2, 20, 16)
    query_pos = torch.full_like(tgt, 0.5)

    layer.forward_post(tgt=tgt, memory=memory, query_pos=query_pos)

    assert self_attn.query_is_key
    assert self_attn.query is not None
    assert self_attn.key is not None
    assert self_attn.value is not None

    expected_query = torch.cat((tgt + query_pos).split(4, dim=1), dim=0)
    expected_value = torch.cat(tgt.split(4, dim=1), dim=0)
    assert self_attn.query.shape == (6, 4, 16)
    torch.testing.assert_close(self_attn.query, expected_query)
    torch.testing.assert_close(self_attn.key, expected_query)
    torch.testing.assert_close(self_attn.value, expected_value)


def _cpu_addmm_activation_available(dtype: torch.dtype) -> bool:
    """Return whether ``torch._addmm_activation`` runs on CPU in ``dtype`` on the installed torch.

    ``_LinearReLU`` is built on this private op, whose CPU kernels are not part of torch's documented surface, so the
    CPU cases of its test are collected only where the op exists.

    Examples:
        >>> isinstance(_cpu_addmm_activation_available(torch.float32), bool)
        True
    """
    try:
        torch._addmm_activation(
            torch.zeros(2, dtype=dtype), torch.zeros(2, 2, dtype=dtype), torch.zeros(2, 2, dtype=dtype)
        )
    except (AttributeError, RuntimeError, NotImplementedError):
        return False
    return True


#: Every hook ``Module.__call__`` runs around ``forward``, as ``(module, hook) -> handle``: the four per-module
#: kinds and the four registered for all modules through ``torch.nn.modules.module``.
_MODULE_HOOK_REGISTRARS: dict[str, Callable[[nn.Module, Callable[..., object]], RemovableHandle]] = {
    "forward_hook": lambda module, hook: module.register_forward_hook(hook),
    "forward_pre_hook": lambda module, hook: module.register_forward_pre_hook(hook),
    "backward_hook": lambda module, hook: module.register_full_backward_hook(hook),
    "backward_pre_hook": lambda module, hook: module.register_full_backward_pre_hook(hook),
    "global_forward_hook": lambda _module, hook: nn.modules.module.register_module_forward_hook(hook),
    "global_forward_pre_hook": lambda _module, hook: nn.modules.module.register_module_forward_pre_hook(hook),
    "global_backward_hook": lambda _module, hook: nn.modules.module.register_module_full_backward_hook(hook),
    "global_backward_pre_hook": lambda _module, hook: nn.modules.module.register_module_full_backward_pre_hook(hook),
}


def _module_path_self_attention(
    layer: TransformerDecoderLayer, tgt: torch.Tensor, query_pos: torch.Tensor
) -> torch.Tensor:
    """The ``forward_post`` self-attention block as written around the ``nn.MultiheadAttention`` call.

    Examples:
        >>> _module_path_self_attention(decoder_layer(), torch.zeros(2, 12, 16), torch.zeros(2, 12, 16)).shape
        torch.Size([2, 12, 16])
    """
    bs, num_queries, _ = tgt.shape
    q = layer.with_pos_embed(tgt, query_pos)
    q = torch.cat(q.split(num_queries // layer.group_detr, dim=1), dim=0)
    v = torch.cat(tgt.split(num_queries // layer.group_detr, dim=1), dim=0)
    tgt2 = layer.self_attn(q, q, v, need_weights=False)[0]
    return torch.cat(tgt2.split(bs, dim=0), dim=1)


class _MultiheadAttentionSubclass(nn.MultiheadAttention):
    """Test double that must keep the generic multi-head-attention call path."""

    pass


class _LinearSubclass(nn.Linear):
    """Test double that must keep the generic linear call path."""

    pass


class TestGroupedSelfAttention:
    """Group-DETR self-attention fast path: parity, eligibility and CUDA-only routing."""

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.usefixtures("highest_float32_matmul_precision")
    @pytest.mark.parametrize(
        ("group_detr", "batch", "queries_per_group", "heads", "d_model"),
        [
            pytest.param(3, 2, 4, 4, 16, id="baseline"),
            pytest.param(1, 2, 4, 4, 16, id="single-group"),
            pytest.param(3, 1, 4, 4, 16, id="single-image"),
            pytest.param(3, 2, 1, 4, 16, id="one-query-per-group"),
            pytest.param(3, 2, 4, 1, 16, id="single-head"),
            pytest.param(3, 2, 4, 4, 12, id="width-not-a-multiple-of-8"),
            pytest.param(13, 2, 5, 8, 32, id="many-groups-odd-queries"),
            pytest.param(1, 1, 1, 1, 8, id="all-singletons"),
        ],
    )
    def test_is_bitwise_the_module_path(
        self, group_detr: int, batch: int, queries_per_group: int, heads: int, d_model: int
    ) -> None:
        """CUDA regrouping reproduces the module path bitwise in the forward pass and input gradients.

        Weight gradients sum the same rows in a different order, so their comparison allows FP32 rounding.

        The group count, batch size, queries per group, head count and width each take their singleton or an odd value
        in turn, since the regrouping views are where a boundary shape would go wrong.
        """
        torch.manual_seed(0)
        layer = decoder_layer(group_detr=group_detr, heads=heads, d_model=d_model).cuda()
        num_queries = group_detr * queries_per_group
        tgt = torch.randn(batch, num_queries, d_model, device="cuda", requires_grad=True)
        query_pos = torch.randn(batch, num_queries, d_model, device="cuda", requires_grad=True)
        grad_out = torch.randn(batch, num_queries, d_model, device="cuda")

        expected = _module_path_self_attention(layer, tgt, query_pos)
        expected.backward(grad_out)
        expected_input_grads = [tgt.grad, query_pos.grad]
        expected_param_grads = [p.grad for p in layer.self_attn.parameters()]
        tgt.grad = query_pos.grad = None
        layer.zero_grad(set_to_none=True)

        actual = layer._grouped_self_attention(tgt, query_pos)
        actual.backward(grad_out)
        actual_input_grads = [tgt.grad, query_pos.grad]
        actual_param_grads = [p.grad for p in layer.self_attn.parameters()]

        assert torch.equal(actual, expected)
        for expected_grad, actual_grad in zip(expected_input_grads, actual_input_grads, strict=True):
            assert expected_grad is not None and actual_grad is not None
            assert torch.equal(actual_grad, expected_grad)
        for expected_grad, actual_grad in zip(expected_param_grads, actual_param_grads, strict=True):
            assert expected_grad is not None and actual_grad is not None
            # Reassociating a sum of ``batch * num_queries`` fp32 terms moves it by rounding proportional to the
            # gradient's own magnitude, which grows with the row count, so the bound scales with the largest entry.
            atol = 1e-6 + 1e-6 * float(expected_grad.abs().max())
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=atol)

    def test_uses_a_view_for_the_regrouping(self) -> None:
        """Regrouping is batch-major: image 1's queries attend only within image 1, so swapping images swaps outputs.

        A group-major view would scramble rows across images.
        """
        layer = decoder_layer()
        tgt = torch.randn(2, 12, 16)
        query_pos = torch.zeros(2, 12, 16)
        tgt_swapped = tgt.flip(0)

        with torch.no_grad():
            out = layer._grouped_self_attention(tgt, query_pos)
            out_swapped = layer._grouped_self_attention(tgt_swapped, query_pos)

        assert torch.equal(out_swapped, out.flip(0))

    @pytest.fixture
    def cuda_layer(self) -> tuple[TransformerDecoderLayer, torch.Tensor]:
        """A CUDA decoder layer and a divisible ``tgt`` for which the explicit path is eligible (positive control).

        On CPU every fallback case is ineligible for its device alone, so a CPU run would pass whether or not the
        case's own guard exists; the control is asserted before any test changes the layer.

        Examples:
            >>> cuda_layer  # doctest: +SKIP
        """
        layer = decoder_layer().cuda()
        tgt = torch.randn(2, 12, 16, device="cuda")
        assert layer._grouped_self_attention_eligible(tgt, None, None) is True
        return layer, tgt

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda layer, mocker: layer.eval(), id="eval"),
            pytest.param(lambda layer, mocker: setattr(layer.self_attn, "dropout", 0.1), id="dropout"),
            pytest.param(
                lambda layer, mocker: setattr(layer.self_attn, "forward", layer.self_attn.forward),
                id="instance_forward",
            ),
            pytest.param(
                lambda layer, mocker: setattr(layer.self_attn, "_compiled_call_impl", mocker.Mock()),
                id="compiled_call_impl",
            ),
        ],
    )
    def test_eligibility_falls_back_to_the_module_call_for_layer_state(
        self,
        mutate: Callable[[TransformerDecoderLayer, MockerFixture], object],
        cuda_layer: tuple[TransformerDecoderLayer, torch.Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Eval mode, attention dropout and a per-instance call override keep the generic ``self_attn(...)`` call."""
        layer, tgt = cuda_layer
        mutate(layer, mocker)

        assert layer._grouped_self_attention_eligible(tgt, None, None) is False

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "make_self_attn",
        [
            pytest.param(partial(_MultiheadAttentionSubclass, 16, 4, batch_first=True), id="subclass"),
            pytest.param(partial(nn.MultiheadAttention, 16, 4, bias=False, batch_first=True), id="no_in_proj_bias"),
            pytest.param(partial(nn.MultiheadAttention, 16, 4, add_bias_kv=True, batch_first=True), id="add_bias_kv"),
            pytest.param(
                partial(nn.MultiheadAttention, 16, 4, add_zero_attn=True, batch_first=True), id="add_zero_attn"
            ),
            pytest.param(partial(nn.MultiheadAttention, 16, 4, batch_first=False), id="sequence_first"),
            pytest.param(partial(nn.MultiheadAttention, 16, 4, kdim=8, vdim=8, batch_first=True), id="distinct_kdim"),
        ],
    )
    def test_eligibility_falls_back_to_the_module_call_for_replaced_self_attn(
        self,
        make_self_attn: Callable[[], nn.MultiheadAttention],
        cuda_layer: tuple[TransformerDecoderLayer, torch.Tensor],
    ) -> None:
        """A subclass or a ``MultiheadAttention`` configuration the explicit path cannot reproduce keeps the call."""
        layer, tgt = cuda_layer
        layer.self_attn = make_self_attn().cuda()

        assert layer._grouped_self_attention_eligible(tgt, None, None) is False

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "make_inputs",
        [
            pytest.param(
                lambda: (
                    torch.randn(2, 12, 16, device="cuda"),
                    torch.zeros(4, 4, device="cuda", dtype=torch.bool),
                    None,
                ),
                id="tgt_mask",
            ),
            pytest.param(
                lambda: (
                    torch.randn(2, 12, 16, device="cuda"),
                    None,
                    torch.zeros(6, 4, device="cuda", dtype=torch.bool),
                ),
                id="key_padding_mask",
            ),
            pytest.param(lambda: (torch.randn(2, 11, 16, device="cuda"), None, None), id="indivisible_queries"),
        ],
    )
    def test_eligibility_falls_back_to_the_module_call_for_inputs(
        self,
        make_inputs: Callable[[], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
        cuda_layer: tuple[TransformerDecoderLayer, torch.Tensor],
    ) -> None:
        """Attention masks and a query count not divisible by ``group_detr`` keep the generic ``self_attn(...)``
        call."""
        layer, _ = cuda_layer
        tgt, tgt_mask, key_padding_mask = make_inputs()

        assert layer._grouped_self_attention_eligible(tgt, tgt_mask, key_padding_mask) is False

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "hook",
        [
            "forward_hook",
            "forward_pre_hook",
            "backward_hook",
            "backward_pre_hook",
            "global_forward_hook",
            "global_backward_hook",
        ],
    )
    def test_eligibility_falls_back_to_the_module_call_for_hooks(
        self,
        hook: str,
        cuda_layer: tuple[TransformerDecoderLayer, torch.Tensor],
        request: pytest.FixtureRequest,
        mocker: MockerFixture,
    ) -> None:
        """A per-module or global, forward or backward hook on ``self_attn`` keeps the generic call."""
        layer, tgt = cuda_layer
        request.addfinalizer(_MODULE_HOOK_REGISTRARS[hook](layer.self_attn, mocker.Mock(return_value=None)).remove)

        assert layer._grouped_self_attention_eligible(tgt, None, None) is False

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize("predicate", ["is_compiling", "_is_tracing"])
    def test_eligibility_falls_back_to_the_module_call_while_compiling_or_tracing(
        self, predicate: str, cuda_layer: tuple[TransformerDecoderLayer, torch.Tensor], mocker: MockerFixture
    ) -> None:
        """``torch.compile`` and ``torch.jit.trace`` graphs keep the generic ``self_attn(...)`` call."""
        layer, tgt = cuda_layer
        mocker.patch(f"rfdetr.models.transformer.{predicate}", return_value=True)

        assert layer._grouped_self_attention_eligible(tgt, None, None) is False

    def test_is_cuda_only(self) -> None:
        """The layout savings are a CUDA measurement; CPU keeps the module call."""
        layer = decoder_layer()

        assert layer._grouped_self_attention_eligible(torch.randn(2, 12, 16), None, None) is False

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.usefixtures("highest_float32_matmul_precision")
    @pytest.mark.parametrize(
        "autocast_dtype",
        [
            pytest.param(None, id="fp32"),
            pytest.param(torch.bfloat16, id="bf16"),
            pytest.param(torch.float16, id="fp16"),
        ],
    )
    def test_forward_post_grouped_path_matches_module_path_on_cuda(
        self, autocast_dtype: torch.dtype | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On CUDA in training the explicit path is taken, and ``forward_post`` returns the module path's bits.

        The oracle turns off every eager rewrite of the layer (grouped self-attention, the folded positional add and the
        fused FFN epilogue) and the gradients of every layer parameter, not only the self-attention ones, are compared,
        so ``linear1``, ``linear2``, the norms and ``cross_attn`` are checked against the two-op graph under autocast
        too.

        Shapes follow RF-DETR Nano's decoder (256-wide, 8 heads, 13 groups of 300 queries) at batch 2, so the kernels
        exercised are the ones training runs. The gradients are compared with tolerances: the weight gradients reduce
        over the tokens in a different row order, and the flash-attention and deformable-attention backward kernels are
        not deterministic on their own (fp32 to rounding, bf16 and fp16 to one ulp of the largest entry).
        """
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn as _MSDeformAttn

        torch.manual_seed(0)
        layer = TransformerDecoderLayer(
            d_model=256, sa_nhead=8, ca_nhead=16, dim_feedforward=2048, dropout=0.0, group_detr=13, num_feature_levels=1
        ).cuda()
        layer.train()
        assert type(layer.cross_attn) is _MSDeformAttn
        tgt = torch.randn(2, 13 * 300, 256, device="cuda")
        memory = torch.randn(2, 24 * 24, 256, device="cuda")
        query_pos = torch.randn(2, 13 * 300, 256, device="cuda")
        reference_points = torch.rand(2, 13 * 300, 1, 4, device="cuda")
        spatial_shapes = torch.tensor([[24, 24]], device="cuda")
        level_start_index = torch.tensor([0], device="cuda")
        grad_out = torch.randn(2, 13 * 300, 256, device="cuda")

        compute_dtype = torch.float32 if autocast_dtype is None else autocast_dtype

        def run(use_explicit_path: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
            if not use_explicit_path:
                monkeypatch.setattr(
                    TransformerDecoderLayer, "_grouped_self_attention_eligible", lambda *_args, **_kwargs: False
                )
                monkeypatch.setattr(
                    TransformerDecoderLayer, "_pos_embed_for_linear", TransformerDecoderLayer.with_pos_embed
                )
                monkeypatch.setattr(
                    TransformerDecoderLayer, "_ffn_hidden", lambda self, tgt: self.activation(self.linear1(tgt))
                )
            else:
                monkeypatch.undo()
            layer.zero_grad(set_to_none=True)
            tgt_in = tgt.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=compute_dtype, enabled=autocast_dtype is not None):
                out = layer.forward_post(
                    tgt_in,
                    memory.to(compute_dtype),
                    query_pos=query_pos.to(compute_dtype),
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    spatial_shapes_hw=[(24, 24)],
                )
            assert isinstance(out, torch.Tensor)
            out.backward(grad_out.to(out.dtype))
            assert tgt_in.grad is not None
            grads = [tgt_in.grad.clone(), *(p.grad.clone() for p in layer.parameters() if p.grad is not None)]
            return out.detach(), grads

        assert layer._grouped_self_attention_eligible(tgt, None, None) is True
        expected, expected_grads = run(use_explicit_path=False)
        actual, actual_grads = run(use_explicit_path=True)

        assert torch.equal(actual, expected)
        ulp = {None: 2**-16, torch.bfloat16: 2**-7, torch.float16: 2**-10}[autocast_dtype]
        for expected_grad, actual_grad in zip(expected_grads, actual_grads, strict=True):
            atol = float(expected_grad.abs().max()) * ulp
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=atol)


#: Skip a CPU ``_LinearReLU`` case whose dtype has no CPU ``torch._addmm_activation`` kernel on the installed torch.
_requires_cpu_addmm_activation_fp32 = pytest.mark.skipif(
    not _cpu_addmm_activation_available(torch.float32), reason="torch._addmm_activation has no CPU float32 kernel"
)


_requires_cpu_addmm_activation_bf16 = pytest.mark.skipif(
    not _cpu_addmm_activation_available(torch.bfloat16), reason="torch._addmm_activation has no CPU bfloat16 kernel"
)


_requires_cpu_addmm_activation_fp16 = pytest.mark.skipif(
    not _cpu_addmm_activation_available(torch.float16), reason="torch._addmm_activation has no CPU float16 kernel"
)


#: ``(device, dtype)`` cases of ``_LinearReLU``: CUDA always, and CPU in every dtype for which
#: the private ``torch._addmm_activation`` op exists on the installed torch, so a CPU job gets a signal on the op's
#: drift across the supported torch range without failing where the op has no CPU kernel.
_LINEAR_RELU_CASES = [
    pytest.param(device, dtype, id=f"{device}-{name}", marks=_cuda_marks if device == "cuda" else cpu_marker)
    for device in ("cpu", "cuda")
    for name, dtype, cpu_marker in (
        ("fp32", torch.float32, _requires_cpu_addmm_activation_fp32),
        ("bf16", torch.bfloat16, _requires_cpu_addmm_activation_bf16),
        ("fp16", torch.float16, _requires_cpu_addmm_activation_fp16),
    )
]


class TestLinearReLU:
    """Fused linear + ReLU autograd function against relu(linear)."""

    @pytest.mark.usefixtures("float32_matmul_precision", "mkldnn_disabled")
    @pytest.mark.parametrize("dim_feedforward", [64, 63])
    @pytest.mark.parametrize(("device", "dtype"), _LINEAR_RELU_CASES)
    def test_is_bitwise_relu_of_linear_forward_and_backward(
        self, device: str, dtype: torch.dtype, dim_feedforward: int
    ) -> None:
        """The epilogue ReLU rounds once then clamps, which commutes with the clamp; the backward is autograd's.

        The fp32 cases run under both float32 matmul precisions (production sets ``"high"``, TF32 GEMMs), and the odd
        ``dim_feedforward`` is not a multiple of 8, the alignment cuBLASLt's fused epilogue prefers.
        """
        torch.manual_seed(0)
        x = (torch.randn(3, 40, 32, device=device) * 3).to(dtype).requires_grad_(True)
        weight = (torch.randn(dim_feedforward, 32, device=device) * 0.2).to(dtype).requires_grad_(True)
        bias = torch.randn(dim_feedforward, device=device).to(dtype).requires_grad_(True)
        grad_out = torch.randn(3, 40, dim_feedforward, device=device).to(dtype)

        expected = F.relu(F.linear(x, weight, bias))
        expected.backward(grad_out)
        expected_grads = [t.grad for t in (x, weight, bias)]
        x.grad = weight.grad = bias.grad = None

        actual = _LinearReLU.apply(x, weight, bias)
        actual.backward(grad_out)

        assert actual.dtype == dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
        for expected_grad, tensor in zip(expected_grads, (x, weight, bias), strict=True):
            assert expected_grad is not None and tensor.grad is not None
            assert torch.equal(tensor.grad, expected_grad)

    @pytest.mark.gpu
    @_requires_cuda
    def test_matches_relu_of_linear_for_nan_inf_and_negative_zero_on_cuda(self) -> None:
        """The epilogue ReLU treats NaN, infinities and the sign of zero like ``relu(linear)``.

        Probe of what the ``_LinearReLU`` docstring leaves unverified: row 0 is NaN, row 1 mixes +-inf, row 2 is all
        zero with a ``-0.0`` bias (so the pre-activation is -0.0). NaN is compared by mask, not payload; every other
        entry is compared through its integer bit view, which tells -0.0 from +0.0 and +inf from -inf.
        """
        x = torch.randn(4, 32, device="cuda")
        x[0] = float("nan")
        x[1, :2] = torch.tensor([float("inf"), float("-inf")], device="cuda")
        x[2] = 0.0
        weight = torch.randn(64, 32, device="cuda") * 0.2
        bias = torch.randn(64, device="cuda")
        bias[:] = -0.0

        expected = F.relu(F.linear(x, weight, bias))
        actual = _LinearReLU.apply(x, weight, bias)

        finite = ~expected.isnan()
        assert torch.equal(actual.isnan(), expected.isnan())
        assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))


class TestFfnHidden:
    """Decoder FFN hidden activation: fused CUDA path and its fallbacks."""

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.usefixtures("float32_matmul_precision")
    @pytest.mark.parametrize(
        "mode", [pytest.param("train", id="train"), pytest.param("inference", id="eval-inference-mode")]
    )
    @pytest.mark.parametrize(
        "autocast_dtype",
        [
            pytest.param(None, id="fp32"),
            pytest.param(torch.bfloat16, id="bf16"),
            pytest.param(torch.float16, id="fp16"),
        ],
    )
    def test_is_bitwise_the_two_op_path_on_cuda(
        self, autocast_dtype: torch.dtype | None, mode: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_ffn_hidden`` must return the bits of ``activation(linear1(tgt))`` in the dtype that path produces.

        ``_ffn_hidden`` is not tied to training, so ``predict()`` on CUDA takes it too: the ``inference`` mode runs it
        in ``eval()`` under ``torch.inference_mode``. The spy proves the fused path ran instead of a silent fallback.
        """
        layer = decoder_layer().cuda()
        tgt = torch.randn(2, 12, 16, device="cuda")
        if mode == "inference":
            layer.eval()
        fused_calls = record_apply_calls(monkeypatch, _LinearReLU)

        with (
            torch.inference_mode(mode == "inference"),
            torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None),
        ):
            expected = layer.activation(layer.linear1(tgt))
            actual = layer._ffn_hidden(tgt)

        assert len(fused_calls) == 1
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)

    @pytest.fixture
    def cuda_ffn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[TransformerDecoderLayer, torch.Tensor, list[tuple[object, ...]]]:
        """A CUDA decoder layer, its input and the fused-call spy, after a positive control proved the fused path.

        On CPU each fallback case would fall back for its device alone and pass whether or not its own guard exists.

        Examples:
            >>> cuda_ffn  # doctest: +SKIP
        """
        layer = decoder_layer().cuda()
        tgt = torch.randn(2, 12, 16, device="cuda")
        fused_calls = record_apply_calls(monkeypatch, _LinearReLU)
        layer._ffn_hidden(tgt)
        assert len(fused_calls) == 1
        fused_calls.clear()
        return layer, tgt, fused_calls

    def test_falls_back_to_the_two_op_path_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CPU keeps ``activation(linear1(tgt))``: the fused epilogue is a CUDA path."""
        layer = decoder_layer()
        tgt = torch.randn(2, 12, 16)
        fused_calls = record_apply_calls(monkeypatch, _LinearReLU)

        out = layer._ffn_hidden(tgt)

        assert fused_calls == []
        assert torch.equal(out, layer.activation(layer.linear1(tgt)))

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda layer, mocker: setattr(layer, "activation", F.gelu), id="gelu"),
            pytest.param(
                lambda layer, mocker: setattr(layer.linear1, "forward", layer.linear1.forward), id="instance_forward"
            ),
            pytest.param(
                lambda layer, mocker: setattr(
                    layer.linear1, "_compiled_call_impl", mocker.Mock(side_effect=layer.linear1._call_impl)
                ),
                id="compiled_call_impl",
            ),
        ],
    )
    def test_falls_back_to_the_two_op_path_for_layer_state(
        self,
        mutate: Callable[[TransformerDecoderLayer, MockerFixture], object],
        cuda_ffn: tuple[TransformerDecoderLayer, torch.Tensor, list[tuple[object, ...]]],
        mocker: MockerFixture,
    ) -> None:
        """A non-ReLU activation or a per-instance call override on ``linear1`` keeps ``activation(linear1(tgt))``."""
        layer, tgt, fused_calls = cuda_ffn
        mutate(layer, mocker)

        out = layer._ffn_hidden(tgt)

        assert fused_calls == []
        assert torch.equal(out, layer.activation(layer.linear1(tgt)))

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "make_linear",
        [
            pytest.param(partial(_LinearSubclass, 16, 32), id="linear_subclass"),
            pytest.param(partial(nn.Linear, 16, 32, bias=False), id="no_bias"),
        ],
    )
    def test_falls_back_to_the_two_op_path_for_replaced_linear(
        self,
        make_linear: Callable[[], nn.Linear],
        cuda_ffn: tuple[TransformerDecoderLayer, torch.Tensor, list[tuple[object, ...]]],
    ) -> None:
        """A ``Linear`` subclass or a bias-free ``Linear`` as ``linear1`` keeps ``activation(linear1(tgt))``."""
        layer, tgt, fused_calls = cuda_ffn
        layer.linear1 = make_linear().cuda()

        out = layer._ffn_hidden(tgt)

        assert fused_calls == []
        assert torch.equal(out, layer.activation(layer.linear1(tgt)))

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "hook",
        [
            "forward_hook",
            "forward_pre_hook",
            "backward_hook",
            "backward_pre_hook",
            "global_forward_hook",
            "global_backward_hook",
        ],
    )
    def test_falls_back_to_the_two_op_path_for_hooks(
        self,
        hook: str,
        cuda_ffn: tuple[TransformerDecoderLayer, torch.Tensor, list[tuple[object, ...]]],
        request: pytest.FixtureRequest,
        mocker: MockerFixture,
    ) -> None:
        """A per-module or global, forward or backward hook on ``linear1`` keeps ``activation(linear1(tgt))``."""
        layer, tgt, fused_calls = cuda_ffn
        request.addfinalizer(_MODULE_HOOK_REGISTRARS[hook](layer.linear1, mocker.Mock(return_value=None)).remove)

        out = layer._ffn_hidden(tgt)

        assert fused_calls == []
        assert torch.equal(out, layer.activation(layer.linear1(tgt)))

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize("predicate", ["is_compiling", "_is_tracing"])
    def test_falls_back_to_the_two_op_path_while_compiling_or_tracing(
        self,
        predicate: str,
        cuda_ffn: tuple[TransformerDecoderLayer, torch.Tensor, list[tuple[object, ...]]],
        mocker: MockerFixture,
    ) -> None:
        """``torch.compile`` and ``torch.jit.trace`` graphs keep ``activation(linear1(tgt))``."""
        layer, tgt, fused_calls = cuda_ffn
        mocker.patch(f"rfdetr.models.transformer.{predicate}", return_value=True)

        out = layer._ffn_hidden(tgt)

        assert fused_calls == []
        assert torch.equal(out, layer.activation(layer.linear1(tgt)))


class TestObservedModulesKeepTheModuleCall:
    """Hooks and instance overrides on decoder submodules keep the full-precision module call."""

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize("target", ["ref_point_head", "first_linear"])
    @pytest.mark.parametrize("feature", ["forward_pre_hook", "instance_forward"])
    def test_ref_point_head_observers_keep_the_full_precision_embedding_input(
        self, target: str, feature: str, mocker: MockerFixture
    ) -> None:
        """An observer on the embedding consumer must receive the fp32 input the original module path exposed.

        The first call is a positive control proving the unobserved production path writes the sine embedding directly
        in bf16. The second call installs a real forward-pre hook on either consumer boundary and requires both the
        write and the hook-visible input to stay fp32.
        """
        decoder = TransformerDecoder(decoder_layer(), num_layers=1, d_model=16, lite_refpoint_refine=True).cuda()
        tgt = torch.randn(2, 12, 16, device="cuda")
        memory = torch.randn(2, 20, 16, device="cuda")
        refpoints = torch.randn(2, 12, 4, device="cuda")
        spatial_shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
        level_start_index = torch.tensor([0, 16], device="cuda")
        valid_ratios = torch.ones(2, 2, 2, device="cuda")
        written_dtypes: list[torch.dtype] = []
        real_apply = _InterleavedSinCos.apply
        mocker.patch.object(
            _InterleavedSinCos,
            "apply",
            side_effect=lambda angle, dtype: (written_dtypes.append(dtype), real_apply(angle, dtype))[1],
        )

        def run() -> None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                decoder(
                    tgt,
                    memory,
                    refpoints_unsigmoid=refpoints,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_hw=[(4, 4), (2, 2)],
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                )

        run()
        assert written_dtypes == [torch.bfloat16]
        written_dtypes.clear()

        watched = decoder.ref_point_head if target == "ref_point_head" else decoder.ref_point_head.layers[0]
        observed_dtypes: list[torch.dtype] = []
        forward_mock = None
        if feature == "forward_pre_hook":
            watched.register_forward_pre_hook(lambda _module, args: observed_dtypes.append(args[0].dtype))
        else:
            forward_mock = mocker.Mock(side_effect=watched.forward)
            watched.forward = forward_mock  # type: ignore[method-assign]
        run()
        if forward_mock is not None:
            observed_dtypes.append(forward_mock.call_args.args[0].dtype)

        assert written_dtypes == [torch.float32]
        assert observed_dtypes == [torch.float32]

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize("target", ["cross_attn", "sampling_offsets", "attention_weights"])
    @pytest.mark.parametrize("feature", ["forward_pre_hook", "instance_forward"])
    def test_cross_attention_observers_keep_the_full_precision_query(
        self, target: str, feature: str, mocker: MockerFixture
    ) -> None:
        """Cross-attention and its query-consuming linears must observe the original fp32 positional sum.

        A positive control first proves that both the self- and cross-attention positional adds take the fused path.
        After installing a real hook, only self-attention may keep the folded add and the watched cross-attention
        boundary must receive fp32.
        """
        layer = decoder_layer().cuda()
        tgt = torch.randn(2, 12, 16, device="cuda")
        memory = torch.randn(2, 20, 16, device="cuda")
        query_pos = torch.randn(2, 12, 16, device="cuda")
        reference_points = torch.rand(2, 12, 2, 4, device="cuda")
        spatial_shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
        level_start_index = torch.tensor([0, 16], device="cuda")
        folded_adds: list[object] = []
        real_apply = _AddInDtype.apply
        mocker.patch.object(
            _AddInDtype, "apply", side_effect=lambda *args: (folded_adds.append(args), real_apply(*args))[1]
        )

        def run() -> None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                layer.forward_post(
                    tgt,
                    memory,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_hw=[(4, 4), (2, 2)],
                    level_start_index=level_start_index,
                )

        run()
        assert len(folded_adds) == 2
        folded_adds.clear()

        watched = layer.cross_attn if target == "cross_attn" else getattr(layer.cross_attn, target)
        observed_dtypes: list[torch.dtype] = []
        forward_mock = None
        if feature == "forward_pre_hook":
            watched.register_forward_pre_hook(lambda _module, args: observed_dtypes.append(args[0].dtype))
        else:
            forward_mock = mocker.Mock(side_effect=watched.forward)
            watched.forward = forward_mock  # type: ignore[method-assign]
        run()
        if forward_mock is not None:
            observed_dtypes.append(forward_mock.call_args.args[0].dtype)

        assert len(folded_adds) == 1
        assert observed_dtypes == [torch.float32]

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "feature",
        [
            "forward_hook",
            "backward_hook",
            "backward_pre_hook",
            "global_forward_hook",
            "global_backward_hook",
            "compiled_call_impl",
        ],
    )
    @pytest.mark.parametrize("target", ["linear1", "self_attn"])
    def test_forward_post_keeps_the_module_call_that_a_feature_observes(
        self, target: str, feature: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
    ) -> None:
        """On CUDA in training ``forward_post`` still calls a module whose call something observes.

        The grouped self-attention and the fused FFN epilogue read a module's parameters instead of calling it, so a
        hook (per module or global, forward or backward) or a ``compile()`` wrapper would silently stop firing. The
        oracle is the feature's own effect during a real ``forward_post`` plus backward; a control run without the
        feature first proves that both fast paths are reached, so the fallback cannot pass merely because they never
        were.
        """
        layer = decoder_layer().cuda()
        watched = getattr(layer, target)
        fused: list[object] = []
        grouped: list[object] = []
        real_apply = _LinearReLU.apply
        real_grouped = TransformerDecoderLayer._grouped_self_attention
        monkeypatch.setattr(_LinearReLU, "apply", lambda *args: (fused.append(args), real_apply(*args))[1])
        monkeypatch.setattr(
            TransformerDecoderLayer,
            "_grouped_self_attention",
            lambda *args: (grouped.append(args), real_grouped(*args))[1],
        )
        memory = torch.randn(2, 4 * 4 + 2 * 2, 16, device="cuda")
        reference_points = torch.rand(2, 12, 2, 4, device="cuda")
        spatial_shapes = torch.tensor([[4, 4], [2, 2]], device="cuda")
        level_start_index = torch.tensor([0, 16], device="cuda")
        fired: list[object] = []

        def run() -> None:
            tgt = torch.randn(2, 12, 16, device="cuda", requires_grad=True)
            out = layer.forward_post(
                tgt,
                memory,
                query_pos=torch.randn(2, 12, 16, device="cuda"),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                spatial_shapes_hw=[(4, 4), (2, 2)],
            )
            assert isinstance(out, torch.Tensor)
            out.sum().backward()

        run()
        assert (len(fused), len(grouped), fired) == (1, 1, [])
        fused.clear()
        grouped.clear()

        def record_call(*args: object, **kwargs: object) -> object:
            fired.append(watched)
            return watched._call_impl(*args, **kwargs)

        def record_hook(module: nn.Module, *_: object) -> None:
            # Global hooks see every module call in the layer; only the watched module's counts.
            if module is watched:
                fired.append(module)

        if feature == "compiled_call_impl":
            watched._compiled_call_impl = record_call
        else:
            request.addfinalizer(_MODULE_HOOK_REGISTRARS[feature](watched, record_hook).remove)
        run()

        assert fired, f"the {feature} on {target} never fired"
        is_global = feature.startswith("global_")
        assert len(fused) == (0 if target == "linear1" or is_global else 1)
        assert len(grouped) == (0 if target == "self_attn" or is_global else 1)


class TestPlainModuleCallGate:
    """Plain-call predicate, tracing canary and the positional-embedding add that depends on them."""

    @pytest.mark.parametrize("case", [*_MODULE_HOOK_REGISTRARS, "instance_forward", "compiled_call_impl"])
    def test_is_false_when_the_call_is_observed_or_overridden(
        self, case: str, request: pytest.FixtureRequest, mocker: MockerFixture
    ) -> None:
        """Everything ``Module.__call__`` can do beyond ``forward`` makes the call non-plain.

        The predicate is device independent, so this runs on CPU. A per-module feature spares an unrelated module (the
        negative counterexample) while a global hook applies to every module. The plain control comes first so that no
        case passes because the predicate is always ``False``.
        """
        observed, bystander = nn.Linear(4, 4), nn.Linear(4, 4)
        assert _module_call_is_plain(observed, bystander) is True
        if case == "instance_forward":
            observed.forward = mocker.Mock(side_effect=observed.forward)  # type: ignore[method-assign]
        elif case == "compiled_call_impl":
            observed._compiled_call_impl = mocker.Mock()
        else:
            request.addfinalizer(_MODULE_HOOK_REGISTRARS[case](observed, mocker.Mock(return_value=None)).remove)

        assert _module_call_is_plain(observed, bystander) is False
        assert _module_call_is_plain(bystander) is (not case.startswith("global_"))

    def test_pos_embed_for_linear_is_the_plain_add_outside_cuda_autocast(self) -> None:
        """Without CUDA autocast there is no cast to fold, so the helper is exactly ``with_pos_embed``."""
        layer = decoder_layer()
        tensor = torch.randn(2, 12, 16)
        pos = torch.randn(2, 12, 16)

        assert layer._pos_embed_for_linear(tensor, None) is tensor
        assert torch.equal(layer._pos_embed_for_linear(tensor, pos), tensor + pos)
        assert layer._pos_embed_for_linear(tensor, pos).dtype == torch.float32

    @pytest.mark.gpu
    @_requires_cuda
    def test_pos_embed_for_linear_emits_the_autocast_dtype_under_cuda_autocast(self) -> None:
        """Under bf16 autocast the positional add is produced in bf16 in one step, bitwise the cast sum."""
        layer = decoder_layer().cuda()
        tensor = torch.randn(2, 12, 16, device="cuda")
        pos = torch.randn(2, 12, 16, device="cuda").to(torch.bfloat16)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused = layer._pos_embed_for_linear(tensor, pos)
        layer.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            in_eval = layer._pos_embed_for_linear(tensor, pos)

        assert fused.dtype == torch.bfloat16
        assert torch.equal(fused, (tensor + pos).to(torch.bfloat16))
        assert in_eval.dtype == torch.float32
        assert torch.equal(in_eval, tensor + pos)

    def test_is_tracing_true_only_while_jit_trace_records(self) -> None:
        """``_is_tracing`` reports a real ``torch.jit.trace`` recording, not just the monkeypatched stand-in.

        The fallback tests for the eager CUDA rewrites switch tracing on by replacing ``_is_tracing`` with a constant,
        which keeps passing if the real predicate stops reporting a trace (a torch change, or a swapped-in call). This
        CPU canary ties that stand-in to the real thing: the predicate is ``False`` in plain eager execution and
        ``True`` while a function is being traced.
        """
        seen: list[bool] = []

        def probe(x: torch.Tensor) -> torch.Tensor:
            """Record ``_is_tracing()`` on every call and return ``x + 1`` so the traced graph has a real op."""
            seen.append(_is_tracing())
            return x + 1

        outside = _is_tracing()
        torch.jit.trace(probe, (torch.zeros(1),), check_trace=False)

        assert outside is False
        assert seen == [True]
