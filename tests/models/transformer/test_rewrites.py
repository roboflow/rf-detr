# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Eligibility gates and custom autograd functions of the eager CUDA decoder rewrites in ``models/transformer.py``.

Every rewrite must either reproduce the ops it replaces or step aside for the previous ops. These tests pin the cases
where it must step aside, the autograd contract of the functions it runs instead, and that the production Nano model
takes every rewrite with loss parity, including under CUDA-graph replay.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model, build_model_from_config
from rfdetr.models.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    _AddInDtype,
    _CastThenExpand,
    _eager_cuda,
    _InterleavedSinCos,
    _LinearReLU,
    _module_call_is_plain,
    gen_sineembed_for_position,
)
from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner
from rfdetr.utilities.tensors import NestedTensor
from tests._markers import _requires_cuda
from tests.models.transformer.helpers import (
    EAGER_CUDA,
    decoder_layer,
    gate_open,
    record_apply_calls,
    single_level_two_stage_transformer,
)

#: Every ``_global*hook*`` name ``torch.nn.modules.module`` defines (torch 2.14). Only the forward, forward-pre,
#: backward and backward-pre registries act inside ``Module.__call__``; the rest are registration hooks or metadata of
#: those four. A name outside this set is a registry ``_module_call_is_plain`` was not written against.
_KNOWN_GLOBAL_HOOK_NAMES = frozenset(
    {
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
        "_global_buffer_registration_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks",
        "_global_is_full_backward_hook",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
    }
)
_GLOBAL_CALL_HOOK_REGISTRIES = [
    "_global_forward_hooks",
    "_global_forward_pre_hooks",
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
]
_MODULE_CALL_HOOK_REGISTRIES = ["_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"]


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


@pytest.fixture
def eager_cuda_gate_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the shared eager-CUDA gate (``_eager_cuda``) so CPU tensors follow the CUDA routing.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects ``monkeypatch``):

        >>> eager_cuda_gate_open  # doctest: +SKIP
    """
    monkeypatch.setattr(EAGER_CUDA, gate_open)


def _linear_relu_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return leaf ``(x, weight, bias)`` tensors for ``_LinearReLU``, shaped like a small decoder FFN call.

    Examples:
        >>> x, weight, bias = _linear_relu_inputs(torch.float64)
        >>> tuple(x.shape), tuple(weight.shape), tuple(bias.shape), x.dtype
        ((2, 5, 4), (6, 4), (6,), torch.float64)
    """
    x = torch.randn(2, 5, 4, dtype=dtype, requires_grad=True)
    weight = torch.randn(6, 4, dtype=dtype, requires_grad=True)
    bias = torch.randn(6, dtype=dtype, requires_grad=True)
    return x, weight, bias


class TestLinearReLUAutograd:
    """``_LinearReLU``'s hand-written backward must behave like autograd's for ``relu(linear(x))``."""

    @pytest.mark.parametrize(
        "check",
        [
            pytest.param(torch.autograd.gradcheck, id="first-order"),
            pytest.param(torch.autograd.gradgradcheck, id="second-order"),
        ],
    )
    def test_gradients_match_finite_differences(self, check: Callable[..., bool]) -> None:
        """First- and second-order gradients of the fused FFN match finite differences in fp64 on CPU.

        ``create_graph=True`` callers (gradient penalties, higher-order optimisers) differentiate the backward itself. A
        saved tensor that is neither an input nor an output of the function is cut off from that graph, which silently
        drops the matching terms instead of raising, so the second-order case is as correct as for ``relu(linear(x))``.
        """
        assert check(_LinearReLU.apply, _linear_relu_inputs(torch.float64))

    def test_weight_gradient_is_contiguous(self) -> None:
        """The weight gradient has the parameter's own layout, as autograd's ``F.linear`` backward gives it.

        A transposed (non-contiguous) gradient makes ``AccumulateGrad`` copy it into the parameter layout every step.
        """
        x, weight, bias = _linear_relu_inputs(torch.float32)

        (grad_weight,) = torch.autograd.grad(_LinearReLU.apply(x, weight, bias).sum(), weight)

        assert grad_weight.is_contiguous()

    def test_weight_gradient_is_bitwise_autograds(self) -> None:
        """The weight gradient is bitwise the one autograd computes for ``relu(linear(x))`` on CPU.

        The backward spells out ``AddmmBackward0``'s weight GEMM (``grad.t().mm(rows)``), operand order included.
        """
        x, weight, bias = _linear_relu_inputs(torch.float32)

        (fused,) = torch.autograd.grad(_LinearReLU.apply(x, weight, bias).sum(), weight)
        (reference,) = torch.autograd.grad(F.relu(F.linear(x, weight, bias)).sum(), weight)

        assert torch.equal(fused, reference)

    def test_backward_skips_gradients_no_input_needs(self) -> None:
        """A frozen ``linear1`` (weight and bias without grad) costs no weight GEMM and no bias reduce.

        The backward is called with its real context (the output's ``grad_fn``), whose ``needs_input_grad`` marks only
        ``x``; the skipped gradients come back as ``None``.
        """
        x = torch.randn(2, 5, 4, requires_grad=True)
        result = _LinearReLU.apply(x, torch.randn(6, 4), torch.randn(6))

        grads = _LinearReLU.backward(result.grad_fn, torch.ones_like(result))

        assert grads[1:] == (None, None)

    def test_torch_ships_the_private_addmm_activation_op(self) -> None:
        """``torch._addmm_activation`` exists on the installed torch, so op drift fails here instead of skipping tests.

        ``_LinearReLU`` is built on this private op, and the CPU cases of its bitwise test skip when the op has no CPU
        kernel. A torch release that renames or drops the op would silently turn those skips into the whole signal, so
        this canary fails loudly instead.
        """
        assert hasattr(torch, "_addmm_activation"), (
            "torch._addmm_activation is gone: _LinearReLU (models/transformer.py) is built on it and must be reworked"
        )


@pytest.mark.gpu
@_requires_cuda
class TestNonFp32SourcesKeepThePreviousOps:
    """Autocast casts only fp32 matmul inputs, so a cast folded into a non-fp32 source would change the ops."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
    def test_pos_embed_for_linear_is_the_plain_add(self, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-fp32 sum under bf16 autocast is not emitted in the compute dtype.

        Autocast leaves an fp64 matmul input in fp64, so writing the sum in bf16 would hand the projections a dtype
        their fp64 weights reject; an fp16 source has no measured bitwise argument. Both keep the plain add.
        """
        applied = record_apply_calls(monkeypatch, _AddInDtype)
        layer = decoder_layer().cuda()
        tensor = torch.randn(2, 6, 16, device="cuda", dtype=dtype)
        pos = torch.randn(2, 6, 16, device="cuda", dtype=dtype)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            layer._pos_embed_for_linear(tensor, pos)

        assert applied == []

    def test_ffn_hidden_keeps_the_two_ops_for_fp64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An fp64 FFN under bf16 autocast runs ``relu(linear1(x))`` in fp64, not the epilogue in bf16.

        Casting fp64 operands to bf16 would feed ``linear2``'s fp64 weight a bf16 activation, an error the plain path
        never raises because autocast does not cast fp64.
        """
        applied = record_apply_calls(monkeypatch, _LinearReLU)
        layer = decoder_layer().cuda().double()
        tgt = torch.randn(2, 6, 16, device="cuda", dtype=torch.float64)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            layer._ffn_hidden(tgt)

        assert applied == []


class TestEagerCuda:
    """The shared gate every eager CUDA rewrite takes: a CUDA input outside compile and tracing."""

    def test_is_true_for_a_cuda_input_in_eager_execution(self) -> None:
        """A CUDA input in plain eager execution takes the rewrites.

        The predicate reads nothing but ``is_cuda`` from the tensor, so a stand-in exercises it without a GPU.
        """
        assert _eager_cuda(Mock(spec=torch.Tensor, is_cuda=True)) is True

    @pytest.mark.parametrize("mode", ["is_compiling", "_is_tracing"])
    def test_is_false_under_compile_or_tracing(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """``torch.compile`` and tracing keep the plain ops even for a CUDA input.

        Inductor fuses the plain ops itself and the exporters expect no custom autograd function, so both modes must
        reach the previous ops through the one shared gate.
        """
        monkeypatch.setattr(f"rfdetr.models.transformer.{mode}", lambda: True)

        assert _eager_cuda(Mock(spec=torch.Tensor, is_cuda=True)) is False

    def test_is_false_on_a_rocm_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ROCm (HIP) build keeps the previous ops although its AMD tensors report ``is_cuda``.

        The rewrites' speed and bitwise parity were measured on NVIDIA CUDA only.
        """
        monkeypatch.setattr("rfdetr.models.transformer._IS_HIP", True)

        assert _eager_cuda(Mock(spec=torch.Tensor, is_cuda=True)) is False

    def test_is_false_inside_a_torch_func_transform(self) -> None:
        """Code running under ``torch.func.vmap`` (or ``grad``/``jvp``) keeps the previous ops.

        The rewrites' autograd functions define no ``setup_context``, which ``torch.func`` requires, so taking one
        inside a transform raises where the plain ops work.
        """
        seen: list[bool] = []
        cuda_input = Mock(spec=torch.Tensor, is_cuda=True)

        torch.func.vmap(lambda row: (seen.append(_eager_cuda(cuda_input)), row)[1])(torch.ones(3, 2))

        assert seen == [False]


class TestSineEmbeddingEligibility:
    """The interleaved sine write is taken only where it reproduces the plain ops, here with the CUDA gate forced open.

    Forcing ``_eager_cuda`` lets CPU exercise the routing; the rewrite's ops are device independent.
    """

    @pytest.mark.usefixtures("eager_cuda_gate_open")
    def test_even_dim_takes_the_interleaved_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control: with the gate open an even ``dim`` takes the interleaved write, so the odd case below is real."""
        applied = record_apply_calls(monkeypatch, _InterleavedSinCos)

        gen_sineembed_for_position(torch.rand(2, 8, 4), dim=6)

        assert len(applied) == 1

    @pytest.mark.usefixtures("eager_cuda_gate_open")
    def test_odd_dim_keeps_the_plain_ops_and_their_error(self) -> None:
        """An odd ``dim`` raises the plain path's error instead of returning a wider embedding.

        The interleaved write builds ``ceil(dim / 2)`` frequencies, so an odd ``dim`` silently produced a different
        width on CUDA than the ``RuntimeError`` CPU raises for the same call.
        """
        with pytest.raises(RuntimeError):
            gen_sineembed_for_position(torch.rand(2, 8, 4), dim=5)


class _WrappedTensor(torch.Tensor):
    """A ``torch.Tensor`` subclass standing in for a quantized or otherwise wrapped weight (torchao-style).

    Examples:
        >>> type(torch.zeros(1).as_subclass(_WrappedTensor)).__name__
        '_WrappedTensor'
    """


def _subclass_parameter(param: torch.Tensor) -> nn.Parameter:
    """Wrap ``param``'s values in a tensor subclass, the way a quantization library swaps a weight.

    ``nn.Parameter`` keeps a ``torch.Tensor`` subclass's own type, so the result passes ``isinstance`` checks but is not
    an exact ``nn.Parameter``.

    Examples:
        >>> wrapped = _subclass_parameter(nn.Linear(2, 2).weight)
        >>> type(wrapped).__name__, isinstance(wrapped, nn.Parameter)
        ('_WrappedTensor', True)
    """
    return nn.Parameter(param.detach().clone().as_subclass(_WrappedTensor))


class TestWeightSubclassesKeepTheModuleOps:
    """Rewrites that read parameters instead of calling the module require exact ``nn.Parameter`` tensors.

    A tensor subclass dispatches ``F.linear`` itself but may implement none of ``mm``, slicing or ``_addmm_activation``,
    so it must keep the previous ops. The CUDA gate is forced open to run the routing on CPU.
    """

    @pytest.mark.usefixtures("eager_cuda_gate_open")
    def test_ffn_with_plain_parameters_takes_the_epilogue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control: with the gate open, plain ``linear1`` parameters take the fused epilogue."""
        applied = record_apply_calls(monkeypatch, _LinearReLU)
        layer = decoder_layer()

        layer._ffn_hidden(torch.randn(2, 6, 16))

        assert len(applied) == 1

    @pytest.mark.usefixtures("eager_cuda_gate_open")
    @pytest.mark.parametrize("name", ["weight", "bias"])
    def test_ffn_with_a_subclass_parameter_keeps_the_two_ops(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrapped ``linear1`` weight or bias keeps ``relu(linear1(x))``."""
        applied = record_apply_calls(monkeypatch, _LinearReLU)
        layer = decoder_layer()
        setattr(layer.linear1, name, _subclass_parameter(getattr(layer.linear1, name)))

        layer._ffn_hidden(torch.randn(2, 6, 16))

        assert applied == []

    @pytest.mark.usefixtures("eager_cuda_gate_open")
    def test_self_attention_with_plain_projections_is_eligible(self) -> None:
        """Control: with the gate open, plain in-projection parameters are eligible for the grouped path."""
        layer = decoder_layer()

        assert layer._grouped_self_attention_eligible(torch.randn(2, 12, 16), None, None) is True

    @pytest.mark.usefixtures("eager_cuda_gate_open")
    @pytest.mark.parametrize("name", ["in_proj_weight", "in_proj_bias"])
    def test_self_attention_with_a_subclass_projection_is_ineligible(self, name: str) -> None:
        """A wrapped packed in-projection keeps the ``self_attn`` module call."""
        layer = decoder_layer()
        setattr(layer.self_attn, name, _subclass_parameter(getattr(layer.self_attn, name)))

        assert layer._grouped_self_attention_eligible(torch.randn(2, 12, 16), None, None) is False


class _ScaledPosDecoderLayer(TransformerDecoderLayer):
    """A decoder layer subclass whose ``with_pos_embed`` doubles the positional embedding.

    Examples:
        >>> layer = _ScaledPosDecoderLayer(d_model=16, sa_nhead=4, ca_nhead=4, dim_feedforward=32)
        >>> layer.with_pos_embed(torch.zeros(1), torch.ones(1))
        tensor([2.])
    """

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Return ``tensor + 2 * pos``, or ``tensor`` when ``pos`` is ``None``."""
        return tensor if pos is None else tensor + 2 * pos


class TestPosEmbedForLinearHonoursOverrides:
    """The fused positional add stands in for ``with_pos_embed``, so an override of that method must win."""

    def test_delegates_to_a_subclass_override(self) -> None:
        """A subclass's ``with_pos_embed`` defines the sum the folded add would otherwise compute.

        Without delegation the override applied on CPU, in eval and under compile, but not in CUDA eager training.
        """
        layer = _ScaledPosDecoderLayer(
            d_model=16, sa_nhead=4, ca_nhead=4, dim_feedforward=32, group_detr=3, num_feature_levels=2
        ).train()
        tensor, pos = torch.randn(2, 6, 16), torch.randn(2, 6, 16)

        assert torch.equal(layer._pos_embed_for_linear(tensor, pos), tensor + 2 * pos)

    def test_delegates_to_an_instance_override(self) -> None:
        """A ``with_pos_embed`` assigned on the instance is honoured too, which a class check alone would miss."""
        layer = decoder_layer()
        layer.with_pos_embed = lambda tensor, pos: tensor + 2 * pos  # type: ignore[method-assign,assignment]
        tensor, pos = torch.randn(2, 6, 16), torch.randn(2, 6, 16)

        assert torch.equal(layer._pos_embed_for_linear(tensor, pos), tensor + 2 * pos)


def test_replaced_ref_point_head_without_layers_runs_the_forward() -> None:
    """A replacement head with no ``.layers`` (here a plain ``nn.Linear``) still runs, as it did before the rewrite.

    Reading ``ref_point_head.layers[0]`` ahead of the ``MLP`` type check raised ``AttributeError`` on every device.
    """
    transformer = single_level_two_stage_transformer(16, 3, num_classes=5).eval()
    transformer.decoder.ref_point_head = nn.Linear(32, 16)
    srcs, pos_embeds = [torch.randn(2, 16, 4, 4)], [torch.randn(2, 16, 4, 4)]
    masks = [torch.zeros(2, 4, 4, dtype=torch.bool)]

    hidden_states = transformer(srcs, masks, pos_embeds, torch.rand(3, 4), torch.randn(3, 16))[0]

    assert hidden_states.shape[-1] == 16


class TestModuleCallIsPlainFailsClosed:
    """The plain-call predicate reads private torch state; anything missing must keep the module call."""

    def test_a_plain_module_call_is_plain(self) -> None:
        """An unhooked, uncompiled ``nn.Linear`` is plain on the installed torch.

        Because the predicate fails closed, a torch release that renames one of the attributes it reads turns this into
        ``False``: the rewrites silently stop being taken, and this canary is what fails.
        """
        assert _module_call_is_plain(nn.Linear(2, 2)) is True

    def test_torch_defines_no_unknown_global_hook_registry(self) -> None:
        """Every ``_global*hook*`` name in ``torch.nn.modules.module`` is one the predicate was written against.

        ``hasattr`` catches a renamed or dropped registry, not an added one; a new global registry that
        ``Module.__call__`` consults would be ignored, so a torch bump that adds one must fail here.
        """
        found = {name for name in dir(nn.modules.module) if name.startswith("_global") and "hook" in name}

        assert found <= _KNOWN_GLOBAL_HOOK_NAMES

    @pytest.mark.parametrize("name", _GLOBAL_CALL_HOOK_REGISTRIES)
    def test_is_false_without_a_global_hook_registry(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing global registry makes the call non-plain instead of reading as empty."""
        linear = nn.Linear(2, 2)
        monkeypatch.delattr(nn.modules.module, name)

        assert _module_call_is_plain(linear) is False

    @pytest.mark.parametrize("name", _MODULE_CALL_HOOK_REGISTRIES)
    def test_is_false_without_a_module_hook_registry(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A module lacking one of its own hook registries is not plain."""
        linear = nn.Linear(2, 2)
        monkeypatch.delattr(linear, name)

        assert _module_call_is_plain(linear) is False

    def test_is_false_without_the_compile_wrapper_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``_compiled_call_impl`` the predicate cannot tell whether ``compile()`` wraps the call."""
        linear = nn.Linear(2, 2)
        monkeypatch.delattr(nn.Module, "_compiled_call_impl")

        assert _module_call_is_plain(linear) is False


@dataclass
class _RoutedCalls:
    """Arguments the eager-rewrite entry points received while the CUDA gate was forced open on CPU."""

    add: list[tuple[object, ...]] = field(default_factory=list)
    ffn: list[tuple[object, ...]] = field(default_factory=list)
    sine_dtypes: list[torch.dtype | None] = field(default_factory=list)


@pytest.fixture
def bf16_gate(monkeypatch: pytest.MonkeyPatch, forced_eager_cuda_gate: None) -> _RoutedCalls:
    """Force the eager-CUDA gate open with a bf16 autocast dtype on CPU and record what the rewrites receive.

    There is no autocast on CPU, so the spies assert routing only: each returns the fp32 value the plain ops give, which
    keeps the fp32 consumers downstream of a bf16-folded rewrite valid, and ``_LinearReLU.apply`` is never run in bf16
    because the CPU kernel of ``torch._addmm_activation`` for it depends on the torch build.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects ``monkeypatch``):

        >>> bf16_gate  # doctest: +SKIP
        _RoutedCalls(add=[], ffn=[], sine_dtypes=[])
    """
    calls = _RoutedCalls()
    real_sine = gen_sineembed_for_position

    def add_spy(tensor: torch.Tensor, pos: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        calls.add.append((tensor, pos, dtype))
        return tensor + pos

    def ffn_spy(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        calls.ffn.append((x, weight, bias))
        return F.relu(F.linear(x.float(), weight.float(), bias.float()))

    def sine_spy(pos_tensor: torch.Tensor, dim: int = 128, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        calls.sine_dtypes.append(out_dtype)
        return real_sine(pos_tensor, dim)

    monkeypatch.setattr(_AddInDtype, "apply", add_spy)
    monkeypatch.setattr(_LinearReLU, "apply", ffn_spy)
    monkeypatch.setattr("rfdetr.models.transformer.gen_sineembed_for_position", sine_spy)
    return calls


def _subclassed(module: nn.Module) -> nn.Module:
    """Turn ``module`` into an instance of an empty subclass of its own type, keeping its parameters and children.

    Examples:
        >>> linear = nn.Linear(2, 2)
        >>> type(_subclassed(linear)) is nn.Linear, isinstance(linear, nn.Linear)
        (False, True)
    """
    module.__class__ = type(f"_Sub{type(module).__name__}", (type(module),), {})
    return module


def _run_forward_post(layer: TransformerDecoderLayer) -> torch.Tensor:
    """Run one ``forward_post`` of ``layer`` on a small CPU batch with two feature levels.

    Examples:
        >>> tuple(_run_forward_post(decoder_layer()).shape)
        (2, 12, 16)
    """
    output = layer.forward_post(
        torch.randn(2, 12, 16),
        torch.randn(2, 20, 16),
        query_pos=torch.randn(2, 12, 16),
        reference_points=torch.rand(2, 12, 2, 4),
        spatial_shapes=torch.tensor([[4, 4], [2, 2]]),
        spatial_shapes_hw=[(4, 4), (2, 2)],
        level_start_index=torch.tensor([0, 16]),
    )
    assert isinstance(output, torch.Tensor)
    return output


class _PassThroughLayer(nn.Module):
    """A decoder layer that returns its input, so a decoder run exercises only the reference-point handling.

    Examples:
        >>> _PassThroughLayer()(torch.ones(1), torch.zeros(1), query_pos=None)
        tensor([1.])
    """

    def forward(self, output: torch.Tensor, memory: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Return ``output`` unchanged."""
        return output


def _run_decoder_with_pass_through_layer(modify_head: Callable[[nn.Module], nn.Module] | None = None) -> torch.Tensor:
    """Run a one-layer decoder whose layer is a pass-through, optionally after ``modify_head(ref_point_head)``.

    Args:
        modify_head: Applied to ``ref_point_head`` (and its result installed as the head), or ``None`` to keep it.

    Examples:
        >>> tuple(_run_decoder_with_pass_through_layer().shape)
        (1, 2, 12, 16)
    """
    decoder = TransformerDecoder(decoder_layer(), num_layers=1, d_model=16, lite_refpoint_refine=True)
    decoder.layers = nn.ModuleList([_PassThroughLayer()])
    if modify_head is not None:
        decoder.ref_point_head = modify_head(decoder.ref_point_head)
    result = decoder(
        torch.randn(2, 12, 16),
        torch.randn(2, 20, 16),
        refpoints_unsigmoid=torch.randn(2, 12, 4),
        spatial_shapes=torch.tensor([[4, 4], [2, 2]]),
        spatial_shapes_hw=[(4, 4), (2, 2)],
        level_start_index=torch.tensor([0, 16]),
        valid_ratios=torch.ones(2, 2, 2),
    )
    assert isinstance(result, torch.Tensor)
    return result


def _first_layer_subclassed(head: nn.Module) -> nn.Module:
    """Return ``head`` after turning its first ``nn.Linear`` into an instance of a subclass.

    Examples:
        >>> head = nn.Module()
        >>> head.layers = nn.ModuleList([nn.Linear(2, 2)])
        >>> type(_first_layer_subclassed(head).layers[0]) is nn.Linear
        False
    """
    _subclassed(head.layers[0])
    return head


class TestAutocastFoldsAreTakenOnlyForPlainModules:
    """The cast-folding rewrites need the exact modules the layer builds; an ``isinstance`` check would admit
    subclasses.

    The CUDA gate is forced open with a bf16 autocast dtype so CPU runs the routing. Each negative case has a positive
    control that takes the fold, so none can pass merely because the fold is never reached, and an ``isinstance`` in
    place of a ``type(...) is`` check in the gate fails the negative case.
    """

    def test_pos_embed_for_linear_folds_the_add_into_the_autocast_dtype(self, bf16_gate: _RoutedCalls) -> None:
        """Control: a training layer's fp32 positional sum is emitted in the autocast dtype by ``_AddInDtype``."""
        layer = decoder_layer()

        layer._pos_embed_for_linear(torch.randn(2, 12, 16), torch.randn(2, 12, 16))

        assert [call[2] for call in bf16_gate.add] == [torch.bfloat16]

    def test_ffn_hidden_folds_the_epilogue_in_the_autocast_dtype(self, bf16_gate: _RoutedCalls) -> None:
        """Control: a plain ``linear1`` takes the ReLU epilogue with its operands cast to the autocast dtype."""
        layer = decoder_layer()

        layer._ffn_hidden(torch.randn(2, 12, 16))

        assert [{operand.dtype for operand in call} for call in bf16_gate.ffn] == [{torch.bfloat16}]

    def test_ffn_hidden_keeps_the_two_ops_for_a_linear_subclass(self, bf16_gate: _RoutedCalls) -> None:
        """A ``linear1`` that is a subclass of ``nn.Linear`` may override its call, so it keeps ``relu(linear1(x))``."""
        layer = decoder_layer()
        _subclassed(layer.linear1)

        layer._ffn_hidden(torch.randn(2, 12, 16))

        assert bf16_gate.ffn == []

    def test_cross_attention_query_is_folded_for_the_plain_module(self, bf16_gate: _RoutedCalls) -> None:
        """Control: self- and cross-attention both take the folded positional add, so the cases below are real."""
        _run_forward_post(decoder_layer())

        assert [call[2] for call in bf16_gate.add] == [torch.bfloat16, torch.bfloat16]

    @pytest.mark.parametrize("path", ["cross_attn", "cross_attn.sampling_offsets", "cross_attn.attention_weights"])
    def test_cross_attention_query_keeps_the_plain_add_for_a_subclass(self, path: str, bf16_gate: _RoutedCalls) -> None:
        """A subclassed ``MSDeformAttn`` or query head may observe the query, so only self-attention keeps the fold."""
        layer = decoder_layer()
        _subclassed(layer.get_submodule(path))

        _run_forward_post(layer)

        assert len(bf16_gate.add) == 1

    def test_sine_embedding_is_written_in_the_autocast_dtype_for_the_plain_head(self, bf16_gate: _RoutedCalls) -> None:
        """Control: with a plain ``ref_point_head`` the fp32 reference embedding is requested in the autocast dtype."""
        _run_decoder_with_pass_through_layer()

        assert bf16_gate.sine_dtypes == [torch.bfloat16]

    @pytest.mark.parametrize(
        "modify_head", [pytest.param(_subclassed, id="mlp"), pytest.param(_first_layer_subclassed, id="first-linear")]
    )
    def test_sine_embedding_keeps_its_dtype_for_a_head_subclass(
        self, modify_head: Callable[[nn.Module], nn.Module], bf16_gate: _RoutedCalls
    ) -> None:
        """A subclassed ``MLP`` or first ``nn.Linear`` may observe its input, so it receives the fp32 embedding."""
        _run_decoder_with_pass_through_layer(modify_head)

        assert bf16_gate.sine_dtypes == [None]


@pytest.mark.parametrize(
    "pos_in_compute_dtype", [pytest.param(False, id="fp32-pos"), pytest.param(True, id="compute-pos")]
)
@pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")])
def test_add_in_dtype_is_bitwise_the_add_then_cast_graph(dtype: torch.dtype, pos_in_compute_dtype: bool) -> None:
    """One kernel, same bits: forward equals ``(a + b).to(dtype)``, backward equals that graph's gradients."""
    pos_dtype = dtype if pos_in_compute_dtype else torch.float32
    tensor = torch.randn(2, 30, 8, dtype=torch.float32, requires_grad=True)
    pos = torch.randn(2, 30, 8).to(pos_dtype).requires_grad_(True)
    grad_out = torch.randn(2, 30, 8).to(dtype)

    expected = (tensor + pos).to(dtype)
    expected.backward(grad_out)
    expected_grads = (tensor.grad, pos.grad)
    tensor.grad = pos.grad = None

    actual = _AddInDtype.apply(tensor, pos, dtype)
    actual.backward(grad_out)

    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    assert tensor.grad is not None and pos.grad is not None
    assert expected_grads[0] is not None and expected_grads[1] is not None
    assert tensor.grad.dtype == torch.float32 and pos.grad.dtype == pos_dtype
    assert torch.equal(tensor.grad, expected_grads[0])
    assert torch.equal(pos.grad, expected_grads[1])


class _AutocastRewriteBlock(nn.Module):
    """Smallest module that runs the two autocast-only rewrites, :class:`_AddInDtype` and :class:`_CastThenExpand`.

    A fp32 activation is added to a bf16 projection and rounded once to bf16, and the same fp32 activation is cast once
    to bf16 and expanded over ``groups``; ``rewrites=False`` computes both with the plain two-op graphs they replace.
    The forward takes ``NestedTensor`` and ``targets`` like ``LWDETR`` so ``CudaGraphTrainingRunner`` can capture it.

    Examples:
        >>> block = _AutocastRewriteBlock()
        >>> samples = NestedTensor(torch.zeros(2, 4, 8), torch.zeros(2, 4, dtype=torch.bool))
        >>> with torch.autocast("cpu", dtype=torch.bfloat16):
        ...     block(samples)["pred"].shape
        torch.Size([3, 2, 4, 8])
    """

    def __init__(self, width: int = 8, groups: int = 3, rewrites: bool = True) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.rand(width) + 0.5)
        self.pos_proj = nn.Linear(width, width)
        self.mix = nn.Linear(width, width)
        self.groups = groups
        self.rewrites = rewrites

    def forward(self, samples: NestedTensor, targets: object = None) -> dict[str, torch.Tensor]:
        """Return the mixed bf16 sum plus the group-expanded bf16 copy of the scaled activation."""
        del targets
        x = samples.tensors
        activation = x * self.scale
        pos = self.pos_proj(x)
        if self.rewrites:
            summed = _AddInDtype.apply(activation, pos, torch.bfloat16)
            memory = _CastThenExpand.apply(activation, self.groups, torch.bfloat16)
        else:
            summed = (activation + pos).to(torch.bfloat16)
            memory = activation.unsqueeze(0).expand(self.groups, *activation.shape).to(torch.bfloat16)
        return {"pred": self.mix(summed).unsqueeze(0) + memory}


def _assert_within_norm_bound(actual: torch.Tensor, expected: torch.Tensor, rel_tol: float, name: str) -> None:
    """Assert ``||actual - expected||`` is at most ``rel_tol`` of ``||expected||`` (plus a tiny absolute floor).

    Unlike an element-wise tolerance this separates kernel rounding noise on a large entry from a wrong value on a
    small one: a stale or aliased tensor is off by a large fraction of the norm, rounding noise is not.

    Examples:
        >>> _assert_within_norm_bound(torch.ones(4), torch.ones(4) * 1.001, 1e-2, "ones")
        >>> _assert_within_norm_bound(torch.ones(4), torch.zeros(4), 1e-2, "ones")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        AssertionError: ones: ||diff|| 2.000e+00 vs ||expected|| 0.000e+00...
    """
    difference_norm = (actual.float() - expected.float()).norm()
    expected_norm = expected.float().norm()
    assert difference_norm <= rel_tol * expected_norm + 1e-9, (
        f"{name}: ||diff|| {difference_norm:.3e} vs ||expected|| {expected_norm:.3e}"
    )


class TestEagerRewritesOnTheProductionModel:
    """Eager decoder rewrites on the production Nano model: routing, loss parity and CUDA-graph replay."""

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        ("training", "autocast_dtype", "captured"),
        [
            pytest.param(True, torch.bfloat16, False, id="train-bf16"),
            pytest.param(True, torch.float16, False, id="train-fp16"),
            pytest.param(True, None, False, id="train-fp32"),
            pytest.param(False, torch.bfloat16, False, id="eval-bf16"),
            pytest.param(True, torch.bfloat16, True, id="train-bf16-cuda-graph"),
        ],
    )
    def test_every_decoder_rewrite_is_taken_on_cuda(
        self, training: bool, autocast_dtype: torch.dtype | None, captured: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each rewrite runs on the route ``LWDETR.forward`` takes, in exactly the modes it is meant for.

        The rewrites are bitwise the ops they replace, so no output comparison can tell them from their fallbacks:
        unwiring one leaves every parity test green and silently gives the speedup back. Delegating spies on a real Nano
        built by ``build_model`` observe each one at its owner. The FFN epilogue and the sine-embedding write are not
        tied to training and also run in ``eval()``; the grouped self-attention, the fused positional adds and the
        cast-once memory are training-only, and the last two need autocast (there is no cast to fold without it).

        The ``captured`` case drives the model through ``CudaGraphTrainingRunner`` (``cuda_graphs=True``) under bf16
        autocast: a host-side index tensor or synchronisation in any rewrite fails the capture. The capture tests in
        ``tests/training/optim/test_cuda_graph_step.py`` run without autocast and never reach the two autocast-only
        rewrites.
        """
        from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner

        calls: dict[str, list[tuple[object, ...]]] = {key: [] for key in ("sine", "add", "ffn", "attn", "memory")}

        def count(owner: type, attr: str, key: str) -> None:
            real = getattr(owner, attr)

            def spy(*args: object, **kwargs: object) -> object:
                calls[key].append(args)
                return real(*args, **kwargs)

            monkeypatch.setattr(owner, attr, spy)

        count(_InterleavedSinCos, "apply", "sine")
        count(_AddInDtype, "apply", "add")
        count(_LinearReLU, "apply", "ffn")
        count(TransformerDecoderLayer, "_grouped_self_attention", "attn")
        count(_CastThenExpand, "apply", "memory")
        torch.manual_seed(0)
        ns = _namespace_from_configs(
            RFDETRNanoConfig(num_classes=7, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
        )
        model = build_model(ns).cuda().train(training)
        transformer = model.transformer
        assert transformer._two_stage_batching_eligible()
        layers = len(transformer.decoder.layers)
        samples = NestedTensor(
            torch.randn(2, 3, 256, 256, device="cuda"), torch.zeros(2, 256, 256, dtype=torch.bool, device="cuda")
        )

        with (
            torch.set_grad_enabled(training),
            torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None),
        ):
            outputs = CudaGraphTrainingRunner(model)(samples) if captured else model(samples)
        if captured:
            loss = outputs["pred_logits"].float().sum()
            loss.backward()
            assert torch.isfinite(loss)

        folds_a_cast = training and autocast_dtype is not None
        per_pass = {
            "sine": 1 if transformer.decoder.lite_refpoint_refine else layers,
            "ffn": layers,
            "attn": layers if training else 0,
            "add": 2 * layers if folds_a_cast else 0,
            "memory": 1 if folds_a_cast else 0,
        }
        # A CUDA graph runs the module for its warm-up passes and the capture, so the counts are a whole number of
        # passes.
        passes = len(calls["ffn"]) // layers
        assert passes >= 1 and (captured or passes == 1)
        assert {key: len(taken) for key, taken in calls.items()} == {key: passes * n for key, n in per_pass.items()}
        # ``ref_point_head`` is the embedding's only consumer, so under autocast it is written in the compute dtype.
        assert {args[1] for args in calls["sine"]} == {autocast_dtype or torch.float32}

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        ("autocast_dtype", "grad_rtol"),
        [pytest.param(None, 1e-4, id="fp32"), pytest.param(torch.bfloat16, 1e-2, id="bf16")],
    )
    def test_loss_terms_are_bitwise_and_gradients_match_with_rewrites_on_and_off(
        self, autocast_dtype: torch.dtype | None, grad_rtol: float, monkeypatch: pytest.MonkeyPatch, tf32_disabled: None
    ) -> None:
        """A real Nano forward plus criterion gives the same outputs and loss terms with the eager rewrites on or off.

        The rewrites are bitwise the ops they replace in the forward, so outputs and every loss term must be equal bit
        for bit. Gradients are compared with a scale-aware norm bound instead, because the rewrites reorder
        weight-gradient sums: ``grad_rtol`` of the reference gradient's norm (a wrong gradient is off by far more), with
        a looser bf16 constant because bf16 rounds each sum to eight bits. Autocast is a case because the folded
        positional add and the cast-once memory only run under it.

        Every rewrite lives in the decoder, so each parameter outside the ViT trunk (``backbone.0.encoder.``) gets its
        own bound. In bf16 a bitwise difference entering the trunk reshuffles the rounding of its twelve layers (CI saw
        about ``grad_rtol`` of the norm in the first layers), and its attention key biases have an analytically zero
        gradient (softmax ignores a shift shared by all keys), so their stored value is rounding residue. The trunk is
        therefore checked as one aggregate norm, which still catches a wrong gradient entering it. The reference is the
        rewrites-off bf16 graph, not fp32: the two-stage ``topk`` and Hungarian matching are discrete, so bf16 and fp32
        forwards can pair queries differently. ``_sineembed_interleaved`` still computes the angles on the rewrites-off
        side; only its sin/cos kernel is replaced. This test needs a GPU and was written without one.
        """
        torch.manual_seed(0)
        model_config = RFDETRNanoConfig(pretrain_weights=None, num_classes=3, device="cuda")
        train_config = TrainConfig(dataset_dir="unused", drop_path=0.0)
        model = build_model_from_config(model_config, train_config).cuda().train()
        criterion, _ = build_criterion_from_config(model_config, train_config)
        criterion = criterion.cuda()
        samples = NestedTensor(
            torch.randn(2, 3, 256, 256, device="cuda"), torch.zeros(2, 256, 256, dtype=torch.bool, device="cuda")
        )
        targets = [
            {"labels": torch.tensor([1], device="cuda"), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device="cuda")},
            {
                "labels": torch.tensor([0, 2], device="cuda"),
                "boxes": torch.tensor([[0.3, 0.4, 0.2, 0.2], [0.7, 0.6, 0.3, 0.1]], device="cuda"),
            },
        ]
        fused_calls: list[object] = []
        real_apply = _LinearReLU.apply

        def run(rewrites_on: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            with monkeypatch.context() as patch:
                if rewrites_on:
                    patch.setattr(_LinearReLU, "apply", lambda *args: (fused_calls.append(args), real_apply(*args))[1])
                else:
                    patch.setattr(
                        TransformerDecoderLayer, "_grouped_self_attention_eligible", lambda *_args, **_kwargs: False
                    )
                    patch.setattr(
                        TransformerDecoderLayer, "_pos_embed_for_linear", TransformerDecoderLayer.with_pos_embed
                    )
                    patch.setattr(
                        TransformerDecoderLayer, "_ffn_hidden", lambda self, tgt: self.activation(self.linear1(tgt))
                    )
                    patch.setattr(
                        _CastThenExpand,
                        "apply",
                        lambda x, groups, dtype: x.unsqueeze(0).expand(groups, *x.shape).to(dtype),
                    )
                    patch.setattr(
                        _InterleavedSinCos,
                        "apply",
                        lambda angle, dtype: torch.stack((angle.sin(), angle.cos()), -1).to(dtype),
                    )
                model.zero_grad(set_to_none=True)
                torch.manual_seed(1)
                with torch.autocast("cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None):
                    outputs = model(samples, targets)
                losses = criterion(outputs, targets)
                total = sum(losses[key] * weight for key, weight in criterion.weight_dict.items() if key in losses)
                total.backward()
                terms = {"pred_logits": outputs["pred_logits"].detach(), "pred_boxes": outputs["pred_boxes"].detach()}
                terms.update({key: value.detach() for key, value in losses.items()})
                terms["total"] = total.detach()
                return terms, {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        expected_terms, expected_grads = run(rewrites_on=False)
        assert fused_calls == []
        actual_terms, actual_grads = run(rewrites_on=True)

        assert fused_calls, "the rewrites never ran, so the comparison would not test them"
        assert actual_terms.keys() == expected_terms.keys()
        for key, expected_term in expected_terms.items():
            assert torch.equal(actual_terms[key], expected_term), key
        # A NaN norm compares false against any bound, so a non-finite gradient would otherwise pass unnoticed.
        non_finite = sorted(
            f"{label}: {name}"
            for label, grads in (("rewrites off", expected_grads), ("rewrites on", actual_grads))
            for name, grad in grads.items()
            if not torch.isfinite(grad).all()
        )
        assert not non_finite, f"non-finite gradients: {non_finite}"
        assert actual_grads.keys() == expected_grads.keys()
        violations: list[tuple[float, str, float, float]] = []
        trunk_difference_norms: list[torch.Tensor] = []
        trunk_reference_norms: list[torch.Tensor] = []
        for name, reference in expected_grads.items():
            difference_norm = (actual_grads[name] - reference).norm()
            if name.startswith("backbone.0.encoder."):
                trunk_difference_norms.append(difference_norm)
                trunk_reference_norms.append(reference.norm())
                continue
            bound = grad_rtol * reference.norm() + 1e-9
            if difference_norm > bound:
                violations.append(
                    ((difference_norm / bound).item(), name, difference_norm.item(), reference.norm().item())
                )
        assert trunk_difference_norms, "no ViT trunk gradient found, so the aggregate bound would check nothing"
        trunk_difference = torch.stack(trunk_difference_norms).norm()
        trunk_reference = torch.stack(trunk_reference_norms).norm()
        trunk_bound = grad_rtol * trunk_reference
        if trunk_difference > trunk_bound:
            violations.append(
                (
                    (trunk_difference / trunk_bound).item(),
                    "backbone.0.encoder.* (aggregate)",
                    trunk_difference.item(),
                    trunk_reference.item(),
                )
            )
        # Report every violating parameter: stopping at the first one hides the decoder gradients behind the backbone's.
        assert not violations, "gradient mismatches, worst first (ratio = ||diff|| / bound):\n" + "\n".join(
            f"{ratio:.3f} {name}: ||diff|| {diff:.3e} vs ||reference|| {ref:.3e}"
            for ratio, name, diff, ref in sorted(violations, reverse=True)
        )

    @pytest.mark.gpu
    @_requires_cuda
    def test_autocast_only_rewrites_match_the_two_op_graph_under_cuda_graph_replay(self) -> None:
        """Captured and replayed on new inputs, ``_AddInDtype`` and ``_CastThenExpand`` keep the two-op graph's values.

        The capture tests in ``tests/training/optim/test_cuda_graph_step.py`` run without autocast, and the production-
        model capture case only checks that the loss is finite. Here a minimal module around the two autocast-only
        rewrites is captured once and replayed twice on different inputs, each time against an eager run of the plain
        two-op graph; a stale saved tensor or an aliased output buffer shows as an error of order one. bf16 capture and
        eager pick different kernels, so the bound is a few percent of the norm rather than the fp32 ``1e-4`` of the
        full-Nano fp32 test. Needs a GPU; written without one.
        """
        torch.manual_seed(0)
        reference = _AutocastRewriteBlock(rewrites=False).cuda().train()
        graphed = copy.deepcopy(reference)
        graphed.rewrites = True
        runner = CudaGraphTrainingRunner(graphed)

        def run_once() -> None:
            samples = NestedTensor(
                torch.randn(2, 8, 8, device="cuda"), torch.zeros(2, 8, dtype=torch.bool, device="cuda")
            )
            reference.zero_grad(set_to_none=True)
            graphed.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                expected = reference(samples)["pred"]
                actual = runner(samples)["pred"]
            expected.float().square().sum().backward()
            actual.float().square().sum().backward()
            _assert_within_norm_bound(actual, expected, 3e-2, "pred")
            for (name, parameter), expected_parameter in zip(graphed.named_parameters(), reference.parameters()):
                assert parameter.grad is not None and expected_parameter.grad is not None, name
                _assert_within_norm_bound(parameter.grad, expected_parameter.grad, 3e-2, f"grad of {name}")

        run_once()  # captures
        run_once()  # replays on new values
        run_once()

        assert len(runner._graphed_cache) == 1
