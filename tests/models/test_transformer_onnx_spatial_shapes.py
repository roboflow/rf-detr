# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX-export regression tests for the shared Transformer's spatial_shapes path.

These guard the fix that builds ``spatial_shapes`` from symbolic feature-map shapes (``Shape`` -> ``Concat``) instead of
``torch.empty`` + in-place index assignment. The latter (added in #871 to keep the trace symbolic for dynamic-batch
export) emitted a ``ScatterND`` that fed a shape tensor, which TensorRT rejects ("IScatterLayer cannot be used to
compute a shape tensor"). The constant-baking ``torch.as_tensor`` alternative avoids the ScatterND but regresses the
symbolic trace back to a baked constant.

Under ``torch.compile`` neither form is used: ``spatial_shapes`` is stacked from one 0-d tensor per size, which keeps
the sizes symbolic so a single graph serves every multi-scale resolution.

The Transformer is shared by detection, segmentation and keypoint models, so a single low-level export here covers the
spatial_shapes path for all of them.
"""

import inspect
import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn

from rfdetr.models.transformer import Transformer
from tests._markers import _requires_onnx, _requires_onnx_and_onnxruntime

if TYPE_CHECKING:
    import onnx

# onnx is imported lazily inside the fixtures that build an ONNX graph, never at module scope: a module-level import
# would fail collection of the torch.compile/TorchScript-trace tests, which must still run in CI without the [onnx]
# extra (e.g. ci-tests-cpu.yml's "train,augment,cli,visual" set). Only the ONNX-dependent tests carry skipif markers.


# CI guard: torch._shape_as_tensor is a private ATen API used on the live forward path in
# Transformer.forward(). If a future PyTorch upgrade removes it, this assertion fails
# loudly in CI before users hit an AttributeError at inference time.
assert hasattr(torch, "_shape_as_tensor"), (
    "torch._shape_as_tensor not found — update spatial_shapes construction in "
    "Transformer.forward() for the new PyTorch API."
)

# dynamo kwarg was added in PyTorch 2.1; our minimum is >=2.2.0, so it is always present.
# Check via inspect so that a future signature removal surfaces here rather than as a
# confusing TypeError inside torch.onnx.export.
_DYNAMO_KWARG: dict[str, bool] = (
    {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
)


class _TransformerExportWrapper(nn.Module):
    """Wrap Transformer.forward with flat tensor args for 2-level ONNX export.

    ``torch.onnx.export`` (TorchScript/non-dynamo path) cannot trace Python list arguments; this wrapper flattens the
    list args of ``Transformer.forward`` into positional tensor arguments.
    """

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        s0: torch.Tensor,
        s1: torch.Tensor,
        p0: torch.Tensor,
        p1: torch.Tensor,
        m0: torch.Tensor,
        m1: torch.Tensor,
        refpoint_embed: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Run Transformer with two feature levels and return the first decoder output.

        Args:
            s0: First-level feature map of shape ``(B, C, H0, W0)``.
            s1: Second-level feature map of shape ``(B, C, H1, W1)``.
            p0: Positional embeddings for level 0, same shape as ``s0``.
            p1: Positional embeddings for level 1, same shape as ``s1``.
            m0: Boolean padding mask for level 0 of shape ``(B, H0, W0)``.
            m1: Boolean padding mask for level 1 of shape ``(B, H1, W1)``.
            refpoint_embed: Reference point embeddings of shape ``(num_queries, 4)``.
            query_feat: Query feature embeddings of shape ``(num_queries, C)``.

        Returns:
            First intermediate decoder output tensor.
        """
        outputs = self.transformer([s0, s1], [m0, m1], [p0, p1], refpoint_embed, query_feat, cross_attn_srcs=None)
        return outputs[0]


class _TransformerExportWrapper1Level(nn.Module):
    """Wrap Transformer.forward with flat tensor args for 1-level ONNX export.

    Production RF-DETR models (Base, Small, Nano, Medium, Large) use a single feature level
    (``projector_scale=["P4"]``). This wrapper exercises that path.
    """

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        s0: torch.Tensor,
        p0: torch.Tensor,
        m0: torch.Tensor,
        refpoint_embed: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Run Transformer with one feature level and return the first decoder output.

        Args:
            s0: Feature map of shape ``(B, C, H0, W0)``.
            p0: Positional embeddings for level 0, same shape as ``s0``.
            m0: Boolean padding mask for level 0 of shape ``(B, H0, W0)``.
            refpoint_embed: Reference point embeddings of shape ``(num_queries, 4)``.
            query_feat: Query feature embeddings of shape ``(num_queries, C)``.

        Returns:
            First intermediate decoder output tensor.
        """
        outputs = self.transformer([s0], [m0], [p0], refpoint_embed, query_feat, cross_attn_srcs=None)
        return outputs[0]


# ---------------------------------------------------------------------------
# Session-scoped fixtures — build and export once per test session
# ---------------------------------------------------------------------------


def _build_eval_transformer(num_feature_levels: int) -> Transformer:
    """Build an eval-mode one-layer ``Transformer`` (``d_model=16``, 6 queries) with the given feature-level count.

    Examples:
        >>> transformer = _build_eval_transformer(num_feature_levels=2)
        >>> transformer.num_feature_levels, transformer.training
        (2, False)
    """
    transformer = Transformer(
        d_model=16,
        num_queries=6,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=num_feature_levels,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        use_grouppose_keypoints=False,
    )
    return transformer.eval()


@pytest.fixture(scope="session")
def transformer_wrapper_2lvl() -> _TransformerExportWrapper:
    """Build a 2-level Transformer and wrap it for ONNX export.

    Returns:
        Eval-mode ``_TransformerExportWrapper`` with ``d_model=16``.
    """
    return _TransformerExportWrapper(_build_eval_transformer(num_feature_levels=2)).eval()


@pytest.fixture(scope="session")
def transformer_wrapper_1lvl() -> _TransformerExportWrapper1Level:
    """Build a 1-level Transformer and wrap it for ONNX export.

    Returns:
        Eval-mode ``_TransformerExportWrapper1Level`` with ``d_model=16``.
    """
    return _TransformerExportWrapper1Level(_build_eval_transformer(num_feature_levels=1)).eval()


@pytest.fixture(scope="session")
def example_inputs_2lvl() -> tuple[torch.Tensor, ...]:
    """Return fixed 2-level example input tensors for ``_TransformerExportWrapper``.

    Level 0: spatial size 4×4. Level 1: spatial size 2×2. Batch size 1, ``d_model=16``.

    Returns:
        8-tuple ``(s0, s1, p0, p1, m0, m1, refpoint_embed, query_feat)``.
    """
    return (
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 16, 2, 2),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 16, 2, 2),
        torch.zeros(1, 4, 4, dtype=torch.bool),
        torch.zeros(1, 2, 2, dtype=torch.bool),
        torch.rand(6, 4),
        torch.randn(6, 16),
    )


@pytest.fixture(scope="session")
def example_inputs_1lvl() -> tuple[torch.Tensor, ...]:
    """Return fixed 1-level example input tensors for ``_TransformerExportWrapper1Level``.

    Level 0: spatial size 4×4. Batch size 1, ``d_model=16``.

    Returns:
        5-tuple ``(s0, p0, m0, refpoint_embed, query_feat)``.
    """
    return (
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 16, 4, 4),
        torch.zeros(1, 4, 4, dtype=torch.bool),
        torch.rand(6, 4),
        torch.randn(6, 16),
    )


@pytest.fixture(scope="session")
def exported_static_onnx(
    tmp_path_factory: pytest.TempPathFactory,
    transformer_wrapper_2lvl: _TransformerExportWrapper,
    example_inputs_2lvl: tuple[torch.Tensor, ...],
) -> "onnx.ModelProto":
    """Export the 2-level Transformer to ONNX (static batch) and return the loaded proto.

    Returns:
        Loaded ``onnx.ModelProto`` for structural graph assertions.
    """
    import onnx

    out = tmp_path_factory.mktemp("onnx_static") / "transformer.onnx"
    torch.onnx.export(
        transformer_wrapper_2lvl,
        example_inputs_2lvl,
        str(out),
        input_names=["s0", "s1", "p0", "p1", "m0", "m1", "refpoint_embed", "query_feat"],
        output_names=["hs"],
        opset_version=17,
        **_DYNAMO_KWARG,
    )
    return onnx.load(str(out))


@pytest.fixture(scope="session")
def exported_dynamic_onnx_bytes(
    transformer_wrapper_2lvl: _TransformerExportWrapper,
    example_inputs_2lvl: tuple[torch.Tensor, ...],
) -> bytes:
    """Export the 2-level Transformer with dynamic batch axis and return ONNX bytes.

    Returns:
        Serialized ONNX model bytes for onnxruntime inference with variable batch sizes.
    """
    buf = io.BytesIO()
    torch.onnx.export(
        transformer_wrapper_2lvl,
        example_inputs_2lvl,
        buf,
        input_names=["s0", "s1", "p0", "p1", "m0", "m1", "refpoint_embed", "query_feat"],
        output_names=["hs"],
        dynamic_axes={
            "s0": {0: "batch"},
            "s1": {0: "batch"},
            "p0": {0: "batch"},
            "p1": {0: "batch"},
            "m0": {0: "batch"},
            "m1": {0: "batch"},
        },
        opset_version=17,
        **_DYNAMO_KWARG,
    )
    return buf.getvalue()


@pytest.fixture(scope="session")
def exported_1lvl_onnx(
    tmp_path_factory: pytest.TempPathFactory,
    transformer_wrapper_1lvl: _TransformerExportWrapper1Level,
    example_inputs_1lvl: tuple[torch.Tensor, ...],
) -> "onnx.ModelProto":
    """Export the 1-level Transformer to ONNX and return the loaded proto.

    Returns:
        Loaded ``onnx.ModelProto`` for structural graph assertions.
    """
    import onnx

    out = tmp_path_factory.mktemp("onnx_1lvl") / "transformer_1lvl.onnx"
    torch.onnx.export(
        transformer_wrapper_1lvl,
        example_inputs_1lvl,
        str(out),
        input_names=["s0", "p0", "m0", "refpoint_embed", "query_feat"],
        output_names=["hs"],
        opset_version=17,
        **_DYNAMO_KWARG,
    )
    return onnx.load(str(out))


# ---------------------------------------------------------------------------
# Tests: structural ONNX graph assertions (2-level static export)
# ---------------------------------------------------------------------------


_ONNX_LEVEL_FIXTURES = [
    pytest.param("exported_static_onnx", id="2-level"),
    pytest.param("exported_1lvl_onnx", id="1-level"),
]


@_requires_onnx
@pytest.mark.parametrize("onnx_fixture", _ONNX_LEVEL_FIXTURES)
def test_export_has_no_scatternd(onnx_fixture: str, request: pytest.FixtureRequest) -> None:
    """The exported Transformer must not contain a ScatterND (TRT shape-tensor killer), at 1 and 2 feature levels.

    ``torch.stack`` on a one-element list (the single-level case) must still produce a Shape-derived result.
    """
    op_types = [n.op_type for n in request.getfixturevalue(onnx_fixture).graph.node]
    assert "ScatterND" not in op_types, (
        "ScatterND reintroduced in Transformer export — spatial_shapes is no longer "
        "built from symbolic Shape ops; this breaks TensorRT engine building."
    )


@_requires_onnx
@pytest.mark.parametrize("onnx_fixture", _ONNX_LEVEL_FIXTURES)
def test_export_is_shape_derived(onnx_fixture: str, request: pytest.FixtureRequest) -> None:
    """Sanity-check that the exported Transformer graph contains Shape ops, at 1 and 2 feature levels.

    Note: ``spatial_shapes`` itself traces as a ``Constant`` node in TorchScript ONNX export — the tracer records the
    concrete H,W values at trace time, not ``Shape`` ops. The ``Shape`` ops present here originate from other model
    computations (e.g. batch-size extraction). The true regression guards are ``test_export_has_no_scatternd`` (no
    ScatterND) and ``test_dynamic_batch_inference`` (runtime correctness at variable batch size).
    """
    op_types = [n.op_type for n in request.getfixturevalue(onnx_fixture).graph.node]
    assert "Shape" in op_types, "No Shape op in Transformer ONNX graph — unexpected graph structure change."


# ---------------------------------------------------------------------------
# Tests: dynamic-batch export (2-level)
# ---------------------------------------------------------------------------


@_requires_onnx_and_onnxruntime
@pytest.mark.parametrize("batch_size", [pytest.param(1, id="batch1"), pytest.param(2, id="batch2")])
def test_dynamic_batch_inference(
    exported_dynamic_onnx_bytes: bytes,
    batch_size: int,
) -> None:
    """Dynamic-batch Transformer ONNX must run onnxruntime inference at batch != trace batch.

    Regression guard: a baked batch constant in ``spatial_shapes`` or any upstream tensor
    would cause shape mismatches at runtime for any batch size other than the trace batch (1).
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(exported_dynamic_onnx_bytes, providers=["CPUExecutionProvider"])
    # The TorchScript tracer may constant-fold positional embeddings (p0, p1) into the
    # graph; query the actual session inputs rather than assuming all 8 are present.
    actual_names = {inp.name for inp in session.get_inputs()}
    candidate_feeds = {
        "s0": np.random.randn(batch_size, 16, 4, 4).astype(np.float32),
        "s1": np.random.randn(batch_size, 16, 2, 2).astype(np.float32),
        "p0": np.random.randn(batch_size, 16, 4, 4).astype(np.float32),
        "p1": np.random.randn(batch_size, 16, 2, 2).astype(np.float32),
        "m0": np.zeros((batch_size, 4, 4), dtype=bool),
        "m1": np.zeros((batch_size, 2, 2), dtype=bool),
        "refpoint_embed": np.random.rand(6, 4).astype(np.float32),
        "query_feat": np.random.randn(6, 16).astype(np.float32),
    }
    feeds = {k: v for k, v in candidate_feeds.items() if k in actual_names}
    (hs,) = session.run(None, feeds)
    assert hs.shape[1] == batch_size, f"Expected batch dim=={batch_size} at index 1, got shape {hs.shape}"


# ---------------------------------------------------------------------------
# Tests: level_start_index numerical correctness
# ---------------------------------------------------------------------------


def test_level_start_index_correctness_two_levels() -> None:
    """spatial_shapes construction must extract correct (H, W) for non-square feature maps.

    Uses H0=8, W0=6 and H1=4, W1=3 — non-square and H≠W so a transposed [2:4] slice would produce wrong values. Verifies
    that the ``torch.stack(_shape_as_tensor)`` formula in ``Transformer.forward()`` yields
    ``spatial_shapes=[[8,6],[4,3]]`` and ``level_start_index=[0, 48]`` (cumulative H*W: 8*6=48, then 48+4*3=60 but index
    stops at level boundaries → [0, 48]).
    """
    s0 = torch.randn(1, 16, 8, 6)
    s1 = torch.randn(1, 16, 4, 3)
    srcs = [s0, s1]

    spatial_shapes = torch.stack([torch._shape_as_tensor(src)[2:4] for src in srcs]).to(
        device=s0.device, dtype=torch.long
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

    assert torch.equal(spatial_shapes, torch.tensor([[8, 6], [4, 3]], dtype=torch.long)), (
        f"spatial_shapes mismatch: got {spatial_shapes.tolist()}"
    )
    assert torch.equal(level_start_index, torch.tensor([0, 48], dtype=torch.long)), (
        f"level_start_index expected [0, 48], got {level_start_index.tolist()}"
    )


# ---------------------------------------------------------------------------
# Tests: torch.compile stacks 0-d tensors per size (Dynamo polyfills _shape_as_tensor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wrapper_fixture", "inputs_fixture", "compile_predicate"),
    [
        pytest.param("transformer_wrapper_1lvl", "example_inputs_1lvl", "compiler", id="compiler-1lvl"),
        pytest.param("transformer_wrapper_2lvl", "example_inputs_2lvl", "compiler", id="compiler-2lvl"),
        pytest.param("transformer_wrapper_1lvl", "example_inputs_1lvl", "dynamo", id="dynamo-1lvl"),
        pytest.param("transformer_wrapper_2lvl", "example_inputs_2lvl", "dynamo", id="dynamo-2lvl"),
    ],
)
def test_spatial_shapes_survives_dynamo_shape_as_tensor_polyfill(
    wrapper_fixture: str,
    inputs_fixture: str,
    compile_predicate: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """``Transformer.forward`` must not call ``torch._shape_as_tensor`` while compiling.

    Dynamo polyfills ``torch._shape_as_tensor`` to return a ``torch.Size`` rather than a tensor, so
    ``torch.stack([...])`` raises ``TypeError: expected Tensor as element 0 in argument 0, but got torch.Size`` and the
    whole compile aborts (``suppress_errors=True`` does not catch it: the ``TypeError`` comes from user code, not from
    Dynamo). This reproduces that polyfill and the ``is_compiling()`` state on CPU, without paying for a real compile in
    CI, and checks the output still matches the eager one. PyTorch 2.2 lacks
    ``torch.compiler.is_compiling``, so its legacy ``torch._dynamo.is_compiling`` predicate is covered too. Both
    feature-level counts are covered because the two branches build ``spatial_shapes`` from differently shaped
    sources.

    Args:
        wrapper_fixture: Name of the Transformer wrapper fixture to exercise.
        inputs_fixture: Name of the matching example-input fixture.
        compile_predicate: Compile-state API to emulate (public ``torch.compiler`` or legacy Dynamo).
        request: Pytest fixture request used to resolve the two names.
        monkeypatch: Pytest helper that isolates the selected compile-state API.
    """
    wrapper = request.getfixturevalue(wrapper_fixture)
    inputs = request.getfixturevalue(inputs_fixture)
    with torch.no_grad():
        eager = wrapper(*inputs)

    if compile_predicate == "compiler":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True, raising=False)
    else:
        monkeypatch.delattr(torch.compiler, "is_compiling", raising=False)
        monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)

    mocker.patch.object(torch, "_shape_as_tensor", new=lambda t: torch.Size(t.shape))
    with torch.no_grad():
        compiling = wrapper(*inputs)
    assert torch.equal(eager, compiling), "is_compiling() branch changed Transformer output"


def test_spatial_shapes_compile_guard_is_false_under_torchscript_trace() -> None:
    """``is_compiling()`` must stay ``False`` under ``torch.jit.trace``.

    The ``_shape_as_tensor`` form exists because TensorRT accepts the Constant it produces while a
    ``ScatterND``-producing form is rejected (#1155). If ``is_compiling()`` were true during the TorchScript trace the
    export path would silently switch formulation.
    """
    seen: list[bool] = []

    class _Probe(nn.Module):
        """Minimal module that records ``is_compiling()`` on every forward call."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Record the ``is_compiling()`` state seen during the trace.

            Args:
                x: Any tensor; only its identity matters here.

            Returns:
                ``x + 1``, so the traced graph has a real op.
            """
            seen.append(bool(getattr(torch.compiler, "is_compiling", lambda: False)()))
            return x + 1

    probe = _Probe().eval()
    torch.jit.trace(probe, (torch.randn(1, 3, 4, 4),), check_trace=False)
    assert seen and not any(seen), "torch.compiler.is_compiling() was true during torch.jit.trace"
