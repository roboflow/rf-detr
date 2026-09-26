# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the in-process TensorRT engine builder (`TensorRTExporter.build_engine`).

The unit tests monkeypatch the polygraphy entry points so they run without TensorRT, a GPU, or `polygraphy` installed.
``TestBenchmarkBuildEngine`` covers the sibling builder in ``rfdetr.export._tensorrt.inference``, which drives the raw
TensorRT builder API rather than polygraphy but shares the same precision-strategy decision. The end-to-end class
(``@pytest.mark.e2e_tensorrt``, GPU + ``rfdetr[tensorrt]``, opt-in) builds a real engine from an exported RF-DETR ONNX
and checks runtime parity — mirroring the CoreML and ExecuTorch export suites.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import inference as tensorrt_inference
from rfdetr.export._tensorrt.exporter import (
    _IS_FP16_CASTER_AVAILABLE,
    _IS_TENSORRT_AVAILABLE,
    TensorRTConfig,
    TensorRTExporter,
)
from tests.export.conftest import (
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)

tensorrt_only = pytest.mark.skipif(not _IS_TENSORRT_AVAILABLE, reason="tensorrt not installed")
fp16_caster_only = pytest.mark.skipif(not _IS_FP16_CASTER_AVAILABLE, reason="onnx/onnxconverter-common not installed")

if _IS_FP16_CASTER_AVAILABLE:
    import onnx
    from onnx import helper

# A class-level skipif does not cover a module-level doctest, so gate every live helper doctest that
# touches onnx on the packages its body needs. Without this they raise NameError where the caster is absent.
__doctest_requires__ = {
    (
        "_opset17_model",
        "_float32_model_with_cast",
        "_float32_model_with_dynamic_batch",
        "_float32_model_with_topk",
        "_model_with_a_consumed_output",
        "_model_with_an_initializer_output",
        "_model_with_an_input_that_is_also_an_output",
        "_model_with_a_capturing_subgraph",
        "_model_with_a_preexisting_fp16_name",
    ): ["onnx", "onnxconverter_common"],
}

# A FP32 TensorRT engine still fuses/reorders kernels relative to eager PyTorch, so it diverges more
# than the XNNPACK CPU path (~1e-5). The bound tolerates kernel-level numerical differences while still
# failing on a structural regression (outputs collapse by >=1e-1). Recalibrate once real GPU numbers are
# observed in the tensorrt-parity CI job.
_TENSORRT_MAX_ABS_DIFF = 1e-2

# FP16 carries a 10-bit mantissa, so its relative resolution is 2^-11 ~= 4.9e-4. On RF-DETR's logit
# outputs (O(10) in magnitude) a single rounding is already ~5e-3, and the error compounds across the
# depth of the network because a whole-graph cast leaves no FP32 fallback for sensitive layers. The
# review of this change measured 0.11-0.16 max-abs on ``labels`` running this same cast graph under
# onnxruntime-CPU, so the bound sits above that. It is sized to catch a structural failure — NaN/Inf
# (which fails the `<` comparison outright), outputs collapsing, or a silently FP32 engine being
# reported as FP16 — not to certify detection accuracy, which needs a COCO mAP delta on real weights.
# Recalibrate once real GPU numbers are observed in the tensorrt-parity CI job.
_TENSORRT_FP16_MAX_ABS_DIFF = 3e-1


def _patch_polygraphy_chain(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub the polygraphy build chain and return the dict that captures ``CreateConfig`` kwargs.

    The loader parses every path into a network with one fixed-batch input, and TensorRT is reported available, so
    the build runs the same way whether or not the host has ``polygraphy``/``tensorrt`` installed.

    Args:
        monkeypatch: Fixture used to replace the polygraphy entry points on the module under test.

    Returns:
        Dict populated with the keyword arguments ``build_engine`` passes to ``CreateConfig``.

    Examples:
        >>> with pytest.MonkeyPatch.context() as monkeypatch:
        ...     config_kwargs = _patch_polygraphy_chain(monkeypatch)
        ...     _ = tensorrt_export.CreateConfig(fp16=True)
        >>> config_kwargs
        {'fp16': True}
    """
    config_kwargs: dict = {}

    def _create_config(*, fp16: bool) -> str:
        # Signature-bound (not **kwargs) so a real ``CreateConfig`` keyword rename in ``_compile`` fails
        # this stub with a TypeError instead of silently swallowing it.
        config_kwargs["fp16"] = fp16
        return "config"

    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(
        tensorrt_export, "network_from_onnx_path", lambda path: ("builder", _FakeNetwork(_STATIC_INPUT), "parser")
    )
    monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
    monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: "engine")
    monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
    return config_kwargs


def _patch_polygraphy_build_capture(monkeypatch: pytest.MonkeyPatch, network: _FakeNetwork | None = None) -> dict:
    """Stub the polygraphy build chain and return the dict that captures the arguments the build receives.

    Args:
        monkeypatch: Fixture used to replace the polygraphy entry points on the module under test.
        network: The parsed-network stand-in the loader hands back, or ``None`` for one fixed-batch input.

    Returns:
        Dict populated with the ``source`` path the loader parsed — which is what reveals *which graph* the
        engine was built from — and the ``network`` and ``config`` ``build_engine`` hands to ``engine_from_network``.

    Examples:
        >>> with pytest.MonkeyPatch.context() as monkeypatch:
        ...     build_args = _patch_polygraphy_build_capture(monkeypatch)
        ...     _ = tensorrt_export.network_from_onnx_path("model.onnx")
        >>> build_args
        {'source': 'model.onnx'}
    """
    build_args: dict = {}

    parsed_network = network if network is not None else _FakeNetwork(_STATIC_INPUT)

    def _network_from_onnx_path(path: str) -> tuple:
        build_args["source"] = path
        return ("builder", parsed_network, "parser")

    def _engine_from_network(network, config):
        build_args["network"] = network
        build_args["config"] = config
        return "engine"

    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _network_from_onnx_path)
    monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
    return build_args


def _fake_tensorrt(version: str, *, has_fp16_flag: bool) -> types.ModuleType:
    """Build a stand-in ``tensorrt`` module reporting *version* and optionally lacking ``BuilderFlag.FP16``.

    Lets the version-dependent branches in ``build_engine`` be exercised on a machine with no TensorRT
    at all, including the TensorRT 11 shape where the flag was removed from the API.

    Args:
        version: Value to expose as ``tensorrt.__version__``.
        has_fp16_flag: Whether ``BuilderFlag`` should carry an ``FP16`` member.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "tensorrt", ...)``.

    Examples:
        >>> module = _fake_tensorrt("11.2.1.2", has_fp16_flag=False)
        >>> module.__version__
        '11.2.1.2'
        >>> hasattr(module.BuilderFlag, "FP16")
        False
        >>> hasattr(_fake_tensorrt("10.16.1.11", has_fp16_flag=True).BuilderFlag, "FP16")
        True
    """
    module = types.ModuleType("tensorrt")
    module.__version__ = version

    class BuilderFlag:
        INT8 = 0

    if has_fp16_flag:
        BuilderFlag.FP16 = 1
    module.BuilderFlag = BuilderFlag
    return module


def _fake_polygraphy_trt() -> types.ModuleType:
    """Build a stand-in ``polygraphy.backend.trt`` exposing the five names the exporter imports from it.

    Polygraphy imports ``tensorrt`` only when one of these is called, so importing them succeeds on a host without
    TensorRT; this stand-in reproduces that on hosts where polygraphy is not installed at all.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "polygraphy.backend.trt", ...)``.

    Examples:
        >>> sorted(name for name in vars(_fake_polygraphy_trt()) if not name.startswith("__"))
        ['CreateConfig', 'Profile', 'engine_from_network', 'network_from_onnx_path', 'save_engine']
    """
    module = types.ModuleType("polygraphy.backend.trt")
    for name in ("CreateConfig", "Profile", "engine_from_network", "network_from_onnx_path", "save_engine"):
        setattr(module, name, object())
    return module


def _unexpected_cast(onnx_path: str) -> str:
    """Stand in for ``_cast_onnx_to_fp16`` on paths that must never cast, failing loudly if called.

    Args:
        onnx_path: Path the caller tried to cast.

    Raises:
        AssertionError: Always.

    Examples:
        >>> _unexpected_cast("/tmp/model.onnx")
        Traceback (most recent call last):
        AssertionError: _cast_onnx_to_fp16 must not be called for /tmp/model.onnx
    """
    raise AssertionError(f"_cast_onnx_to_fp16 must not be called for {onnx_path}")


def _fake_tensorrt_without_builder_flag(version: str) -> types.ModuleType:
    """Build a stand-in ``tensorrt`` module that has no ``BuilderFlag`` attribute at all.

    Some lean and vendored wheels omit the symbol rather than just its ``FP16`` member, which used to
    raise ``AttributeError`` out of the strategy probe instead of resolving to a strategy.

    Args:
        version: Value to expose as ``tensorrt.__version__``.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "tensorrt", ...)``.

    Examples:
        >>> hasattr(_fake_tensorrt_without_builder_flag("10.16.1.11"), "BuilderFlag")
        False
    """
    module = types.ModuleType("tensorrt")
    module.__version__ = version
    return module


def _opset17_model(graph: "onnx.GraphProto") -> "onnx.ModelProto":
    """Wrap *graph* in a model pinned to the opset and IR version the fp16 caster is exercised against.

    Args:
        graph: Graph to wrap.

    Returns:
        A ``ModelProto`` importing opset 17 at IR version 8.

    Examples:
        >>> model = _opset17_model(_float32_model_with_cast().graph)
        >>> model.opset_import[0].version, model.ir_version
        (17, 8)
    """
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _float32_model_with_cast() -> "onnx.ModelProto":
    """Build a tiny float32 model that starts with an explicit ``Cast(to=FLOAT)``, as RF-DETR exports do.

    Returns:
        A valid float32 ONNX model with one ``Cast`` and one ``Conv``.

    Examples:
        >>> model = _float32_model_with_cast()
        >>> [node.op_type for node in model.graph.node]
        ['Cast', 'Conv']
    """
    weight = np.zeros((2, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input"], ["casted"], to=onnx.TensorProto.FLOAT, name="leading_cast"),
            helper.make_node("Conv", ["casted", "weight"], ["output"], pads=[1, 1, 1, 1], name="conv"),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 8, 8])],
        [helper.make_tensor("weight", onnx.TensorProto.FLOAT, weight.shape, weight.tobytes(), raw=True)],
    )
    return _opset17_model(graph)


def _float32_model_with_dynamic_batch() -> "onnx.ModelProto":
    """Build a tiny float32 model whose input carries a symbolic ``"batch"`` dim_param, as a
    ``dynamic_batch`` ONNX export does.

    Returns:
        A valid float32 ONNX model with one ``Relu`` and a symbolic batch axis on both input and output.

    Examples:
        >>> model = _float32_model_with_dynamic_batch()
        >>> model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        'batch'
    """
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"], name="relu")],
        "dynamic_batch",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", 3, 8, 8])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ["batch", 3, 8, 8])],
    )
    return _opset17_model(graph)


def _float32_model_with_topk() -> "onnx.ModelProto":
    """Build a tiny float32 model ending in ``TopK``, the block-listed op RF-DETR's query selection uses.

    ``onnxconverter-common`` protects a block-listed op by leaving it FP32 behind boundary casts, so this
    is the graph shape where a careless ``Cast`` retarget would silently undo that protection. The ``k``
    input is an INT64 initializer, as every real RF-DETR export has.

    Returns:
        A valid float32 ONNX model whose ``TopK`` is fed through a ``Mul``.

    Examples:
        >>> model = _float32_model_with_topk()
        >>> [node.op_type for node in model.graph.node]
        ['Cast', 'Mul', 'TopK']
    """
    weight = np.arange(10, dtype=np.float32).reshape(1, 10)
    k = np.array([3], dtype=np.int64)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input"], ["casted"], to=onnx.TensorProto.FLOAT, name="leading_cast"),
            helper.make_node("Mul", ["casted", "weight"], ["scores"], name="scale"),
            helper.make_node("TopK", ["scores", "k"], ["values", "indices"], axis=1, name="topk"),
        ],
        "tiny_topk",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])],
        [
            helper.make_tensor_value_info("values", onnx.TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [1, 3]),
        ],
        [
            helper.make_tensor("weight", onnx.TensorProto.FLOAT, weight.shape, weight.tobytes(), raw=True),
            helper.make_tensor("k", onnx.TensorProto.INT64, k.shape, k.tobytes(), raw=True),
        ],
    )
    return _opset17_model(graph)


def _model_with_a_consumed_output() -> "onnx.ModelProto":
    """Build a model where ``mid`` is both a graph output and the input of a later node.

    Restoring the FP32 output contract has to rewire that later consumer onto the renamed inner tensor;
    appending the boundary ``Cast`` while the consumer still reads ``mid`` leaves it reading a tensor no
    earlier node produces.

    Returns:
        A valid float32 ONNX model with two graph outputs, one of them consumed internally.

    Examples:
        >>> model = _model_with_a_consumed_output()
        >>> [value.name for value in model.graph.output]
        ['mid', 'output']
    """
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["mid"], name="first"),
            helper.make_node("Relu", ["mid"], ["output"], name="second"),
        ],
        "consumed_output",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info("mid", onnx.TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4]),
        ],
    )
    return _opset17_model(graph)


def _model_with_an_initializer_output() -> "onnx.ModelProto":
    """Build a model whose ``const`` graph output is defined by an initializer rather than by a node.

    Returns:
        A valid float32 ONNX model exposing an initializer directly as a graph output.

    Examples:
        >>> model = _model_with_an_initializer_output()
        >>> [initializer.name for initializer in model.graph.initializer]
        ['const']
    """
    const = np.ones((1, 4), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"], name="relu")],
        "initializer_output",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("const", onnx.TensorProto.FLOAT, [1, 4]),
        ],
        [helper.make_tensor("const", onnx.TensorProto.FLOAT, const.shape, const.tobytes(), raw=True)],
    )
    return _opset17_model(graph)


def _model_with_an_input_that_is_also_an_output() -> "onnx.ModelProto":
    """Build a model where ``passthrough`` is declared as both a graph input and a graph output.

    The input side already restores such a tensor to FP32, so adding an output boundary cast for it
    defines the name a second time and breaks single static assignment.

    Returns:
        A valid float32 ONNX model with a pass-through tensor.

    Examples:
        >>> model = _model_with_an_input_that_is_also_an_output()
        >>> sorted({value.name for value in model.graph.input} & {value.name for value in model.graph.output})
        ['passthrough']
    """
    graph = helper.make_graph(
        [helper.make_node("Relu", ["passthrough"], ["output"], name="relu")],
        "passthrough",
        [helper.make_tensor_value_info("passthrough", onnx.TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("passthrough", onnx.TensorProto.FLOAT, [1, 4]),
        ],
    )
    return _opset17_model(graph)


def _model_with_a_capturing_subgraph() -> "onnx.ModelProto":
    """Build a model whose ``If`` branch captures the outer graph input and casts it inside the body.

    ``convert_float_to_float16`` converts subgraphs too, so both the retarget and the boundary rename
    have to follow a captured tensor inward or the branch keeps reading the restored FP32 value.

    Returns:
        A valid float32 ONNX model with an ``If`` whose ``then`` body holds a ``Cast(to=FLOAT)``.

    Examples:
        >>> model = _model_with_a_capturing_subgraph()
        >>> [node.op_type for node in model.graph.node]
        ['Squeeze', 'If']
    """
    cond = np.array([True], dtype=bool)
    then_body = helper.make_graph(
        [
            helper.make_node("Cast", ["input"], ["then_cast"], to=onnx.TensorProto.FLOAT, name="then_cast_node"),
            helper.make_node("Relu", ["then_cast"], ["branch_out"], name="then_relu"),
        ],
        "then_body",
        [],
        [helper.make_tensor_value_info("branch_out", onnx.TensorProto.FLOAT, [1, 4])],
    )
    else_body = helper.make_graph(
        [helper.make_node("Neg", ["input"], ["branch_out"], name="else_neg")],
        "else_body",
        [],
        [helper.make_tensor_value_info("branch_out", onnx.TensorProto.FLOAT, [1, 4])],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Squeeze", ["cond"], ["cond_scalar"], name="squeeze_cond"),
            helper.make_node(
                "If", ["cond_scalar"], ["output"], then_branch=then_body, else_branch=else_body, name="branch"
            ),
        ],
        "capturing_subgraph",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor("cond", onnx.TensorProto.BOOL, cond.shape, cond.tobytes(), raw=True)],
    )
    return _opset17_model(graph)


def _model_with_a_preexisting_fp16_name() -> "onnx.ModelProto":
    """Build a model that already binds ``input_fp16``, the name the input boundary cast wants to generate.

    Returns:
        A valid float32 ONNX model whose first node output is called ``input_fp16``.

    Examples:
        >>> model = _model_with_a_preexisting_fp16_name()
        >>> list(model.graph.node[0].output)
        ['input_fp16']
    """
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["input"], ["input_fp16"], name="first"),
            helper.make_node("Relu", ["input_fp16"], ["output"], name="second"),
        ],
        "preexisting_fp16_name",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])],
    )
    return _opset17_model(graph)


class TestBuildEngineDryRun:
    """Dry-run derives the ``.trt`` path without touching the polygraphy build chain."""

    @pytest.mark.parametrize(
        ("onnx_path", "expected_engine"),
        [
            pytest.param("/output/rfdetr.onnx", "/output/rfdetr_fp16.trt", id="plain-path"),
            pytest.param("/path with spaces/model.onnx", "/path with spaces/model_fp16.trt", id="path-with-spaces"),
            pytest.param("/model;rm -rf /.onnx", "/model;rm -rf /.onnx_fp16.trt", id="shell-metachar"),
            pytest.param(
                "/data/my.onnx.backup/model.onnx",
                "/data/my.onnx.backup/model_fp16.trt",
                id="earlier-onnx-in-dir",
            ),
            pytest.param(
                "/output/model_v1.onnx.old.onnx",
                "/output/model_v1.onnx.old_fp16.trt",
                id="double-onnx-in-filename",
            ),
            pytest.param(
                "/output/model_without_extension",
                "/output/model_without_extension_fp16.trt",
                id="no-onnx-extension",
            ),
        ],
    )
    def test_derives_trt_path(self, onnx_path: str, expected_engine: str) -> None:
        """Only the final suffix is swapped to ``_fp16.trt``; earlier ``.onnx`` segments are never corrupted."""
        result = TensorRTExporter(TensorRTConfig()).build_engine(onnx_path, dry_run=True)

        assert result == expected_engine

    def test_does_not_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry-run must return the engine path without invoking the polygraphy build chain."""
        called: list[str] = []
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda *a, **k: called.append("built"))

        result = TensorRTExporter(TensorRTConfig()).build_engine("/tmp/model.onnx", dry_run=True)

        assert result == "/tmp/model_fp16.trt"
        assert not called, "dry_run must not invoke the polygraphy build chain"

    def test_output_name_overrides_and_suppresses_precision_suffix(self) -> None:
        """``output_name`` names the engine verbatim, in the ONNX's directory, with no ``_fp16``/``_fp32`` suffix."""
        exporter = TensorRTExporter(TensorRTConfig(output_name="my-engine"))

        result = exporter.build_engine("/output/rfdetr-medium.onnx", dry_run=True)

        assert result == "/output/my-engine.trt"

    def test_output_name_argument_overrides_the_configured_one(self) -> None:
        """An explicit ``output_name`` wins over the configured one — the channel a backbone export names through.

        ``_convert`` passes the backbone-marked ONNX stem this way so the engine keeps its ``-backbone`` marker instead
        of being named after the user's plain ``output_name``.
        """
        exporter = TensorRTExporter(TensorRTConfig(output_name="custom"))

        result = exporter.build_engine("/output/custom-backbone.onnx", dry_run=True, output_name="custom-backbone")

        assert result == "/output/custom-backbone.trt"

    def test_output_name_preserves_windows_directory_separators(self) -> None:
        """A Windows-style ``onnx_path`` keeps its backslash directory prefix verbatim (no ``os.sep`` rewrite)."""
        exporter = TensorRTExporter(TensorRTConfig(output_name="my-engine"))

        result = exporter.build_engine(r"C:\out\m.onnx", dry_run=True)

        assert result == r"C:\out\my-engine.trt"


class TestTensorRTAvailability:
    """TensorRT counts as available only when ``tensorrt`` itself is installed, and its absence is reported first."""

    @pytest.mark.parametrize(
        ("polygraphy_installed", "tensorrt_layout", "available"),
        [
            pytest.param(True, "package", True, id="both-installed"),
            pytest.param(True, "absent", False, id="polygraphy-only"),
            pytest.param(True, "bare-directory", False, id="bare-tensorrt-directory"),
            pytest.param(False, "package", False, id="tensorrt-only"),
        ],
    )
    def test_flag_requires_the_tensorrt_package(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        polygraphy_installed: bool,
        tensorrt_layout: str,
        available: bool,
    ) -> None:
        """Polygraphy alone, which ``rfdetr[onnx]`` installs, must not mark TensorRT as available.

        Polygraphy imports ``tensorrt`` lazily, so its own import succeeds without it. A bare ``tensorrt/`` directory,
        such as an export folder named after the format, is importable as a namespace package but is no install either.
        The exporter source is executed under a private module name, with only *tmp_path* searched for ``tensorrt``, so
        a real TensorRT on the host cannot answer and the real module keeps its identity for the other tests.
        """
        if tensorrt_layout != "absent":
            (tmp_path / "tensorrt").mkdir()
        if tensorrt_layout == "package":
            (tmp_path / "tensorrt" / "__init__.py").touch()
        for name in ("polygraphy", "polygraphy.backend"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        # ``None`` in sys.modules is how the import system spells "cannot be imported".
        monkeypatch.setitem(
            sys.modules, "polygraphy.backend.trt", _fake_polygraphy_trt() if polygraphy_installed else None
        )
        monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        spec = importlib.util.spec_from_file_location("_tensorrt_exporter_probe", tensorrt_export.__file__)
        probe = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, probe)

        spec.loader.exec_module(probe)

        assert probe._IS_TENSORRT_AVAILABLE is available

    def test_missing_tensorrt_raises_the_install_hint(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Building from an existing ONNX file without TensorRT raises an actionable ImportError, not a polygraphy
        one."""
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            TensorRTExporter(TensorRTConfig()).build_engine(str(tmp_path / "model.onnx"))

    def test_dry_run_needs_no_tensorrt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``dry_run`` is documented as needing no TensorRT, so the availability check stays behind it."""
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)

        result = TensorRTExporter(TensorRTConfig()).build_engine(str(tmp_path / "model.onnx"), dry_run=True)

        assert result == str(tmp_path / "model_fp16.trt")


class TestBuildEngineWiring:
    """``build_engine`` wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""

    @pytest.mark.parametrize("fp16", [pytest.param(True, id="fp16"), pytest.param(False, id="fp32")])
    def test_invokes_polygraphy_and_saves_trt(self, monkeypatch: pytest.MonkeyPatch, fp16: bool) -> None:
        """build_engine wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""
        config_kwargs: dict = {}
        build_args: dict = {}
        saved: dict = {}
        network = _FakeNetwork(_STATIC_INPUT)

        # Pin a weakly typed TensorRT: this asserts the builder-flag wiring, and without the pin the
        # assertions would flip on a host that really has TensorRT >= 11 installed.
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr(
            tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config-sentinel"
        )

        def _network_from_onnx_path(path: str) -> tuple:
            build_args["source"] = path
            return ("builder", network, "parser")

        def _engine_from_network(network, config):
            build_args["network"] = network
            build_args["config"] = config
            return "engine-sentinel"

        def _save_engine(engine, path):
            saved["engine"] = engine
            saved["path"] = path

        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _network_from_onnx_path)
        monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
        monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)

        result = TensorRTExporter(TensorRTConfig(fp16=fp16)).build_engine("model.onnx")
        expected_path = f"model_{'fp16' if fp16 else 'fp32'}.trt"

        assert result == expected_path
        assert config_kwargs == {"fp16": fp16}
        assert build_args == {
            "source": "model.onnx",
            "network": ("builder", network, "parser"),
            "config": "config-sentinel",
        }
        assert saved == {"engine": "engine-sentinel", "path": expected_path}


class _FakeNetworkInput:
    """One parsed network input: a name and a TensorRT-style shape (``-1`` marks the dynamic batch axis)."""

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape


class _FakeNetwork:
    """Stand-in for a parsed ``trt.INetworkDefinition`` exposing only ``num_inputs`` / ``get_input``."""

    def __init__(self, *inputs: _FakeNetworkInput) -> None:
        self._inputs = inputs

    @property
    def num_inputs(self) -> int:
        return len(self._inputs)

    def get_input(self, index: int) -> _FakeNetworkInput:
        return self._inputs[index]


#: The one input an RF-DETR graph exported without ``dynamic_batch`` has: every axis fixed at trace time.
_STATIC_INPUT = _FakeNetworkInput("input", (1, 3, 384, 384))


class _FakeProfile:
    """Stand-in for ``polygraphy.backend.trt.Profile`` recording every ``add`` call."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, tuple[int, ...]]] = {}

    def add(self, name: str, min: tuple[int, ...], opt: tuple[int, ...], max: tuple[int, ...]) -> _FakeProfile:
        self.entries[name] = {"min": min, "opt": opt, "max": max}
        return self


def _patch_dynamic_polygraphy_chain(monkeypatch: pytest.MonkeyPatch, network: _FakeNetwork) -> dict:
    """Stub the polygraphy chain for a dynamic-batch build and capture what reaches ``CreateConfig`` / the build.

    ``network_from_onnx_path`` (polygraphy's immediately-evaluated form) returns ``(builder, network, parser)``
    around *network*, which is what the exporter inspects to read input shapes before building.

    Args:
        monkeypatch: Fixture used to replace the polygraphy entry points on the module under test.
        network: The parsed-network stand-in the loader hands back.

    Returns:
        Dict with the ``CreateConfig`` kwargs under ``"config"`` and the tuple passed to ``engine_from_network``
        under ``"network"``.

    Examples:
        >>> network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384)))
        >>> with pytest.MonkeyPatch.context() as monkeypatch:
        ...     captured = _patch_dynamic_polygraphy_chain(monkeypatch, network)
        ...     _ = tensorrt_export.CreateConfig(fp16=False, profiles=["profile"])
        >>> captured
        {'config': {'fp16': False, 'profiles': ['profile']}}
    """
    captured: dict = {}

    def _engine_from_network(parsed, config):
        captured["network"] = parsed
        return "engine"

    def _create_config(*, fp16: bool, profiles: list) -> str:
        # Signature-bound (not **kwargs) so a real ``CreateConfig`` keyword rename in ``_compile`` fails
        # this stub with a TypeError instead of silently swallowing it.
        captured["config"] = {"fp16": fp16, "profiles": profiles}
        return "config"

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("builder", network, "parser"))
    monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
    monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
    return captured


class TestDynamicBatchConfig:
    """``dynamic_batch`` on TensorRT needs the profile bounds, and the registry advertises the capability."""

    def test_registry_advertises_dynamic_batch(self) -> None:
        """The pre-import guard lets ``dynamic_batch=True`` through for TensorRT."""
        from rfdetr.export.base import reject_unsupported_dynamic_batch

        reject_unsupported_dynamic_batch("tensorrt", dynamic_batch=True)

    def test_requires_max_batch_size(self) -> None:
        """A dynamic request without an upper bound cannot build a profile and is refused before any work."""
        with pytest.raises(ValueError, match="max_batch_size"):
            TensorRTExporter(TensorRTConfig(dynamic_batch=True))

    @pytest.mark.parametrize("opt_batch_size,max_batch_size", [(4, 2), (0, 4)])
    def test_rejects_inconsistent_bounds(self, opt_batch_size: int, max_batch_size: int) -> None:
        """The profile must satisfy ``1 <= opt <= max``."""
        with pytest.raises(ValueError, match="1 <= batch_size <= max_batch_size"):
            TensorRTExporter(
                TensorRTConfig(dynamic_batch=True, opt_batch_size=opt_batch_size, max_batch_size=max_batch_size)
            )

    @pytest.mark.parametrize(
        "opt_batch_size,max_batch_size",
        [
            pytest.param(4.5, 8, id="float-batch-size"),
            pytest.param(float("nan"), 8, id="nan-batch-size"),
            pytest.param(4, 8.0, id="float-max-batch-size"),
            pytest.param(4, float("nan"), id="nan-max-batch-size"),
            pytest.param(True, 8, id="bool-batch-size"),
            pytest.param(4, True, id="bool-max-batch-size"),
        ],
    )
    def test_rejects_non_integer_bounds(self, opt_batch_size: object, max_batch_size: object) -> None:
        """A non-``int`` bound (``float``, ``nan``, or ``bool``) is refused before any work, not deep in the build.

        ``nan`` compares false against every ``<``/``>=`` bound, so the pre-existing numeric checks silently let it
        through; a plain ``float`` does too, since Python allows ``4.5 < 1`` and ``8.0 < 4`` comparisons. Both used to
        fail only after a full DINOv2 forward pass and an ONNX export.
        """
        with pytest.raises(ValueError, match="must be integers"):
            TensorRTExporter(
                TensorRTConfig(dynamic_batch=True, opt_batch_size=opt_batch_size, max_batch_size=max_batch_size)
            )

    def test_allows_a_degenerate_profile_where_opt_equals_max(self) -> None:
        """``opt_batch_size == max_batch_size`` is a legal, if degenerate, profile and must not be rejected.

        ``_check_capabilities`` only rejects ``max_batch_size < opt_batch_size``, so the equal-bounds edge is permitted
        by construction; this pins that down explicitly instead of leaving it implied.
        """
        TensorRTExporter(TensorRTConfig(dynamic_batch=True, opt_batch_size=4, max_batch_size=4))

    def test_rejects_a_negative_max_batch_size(self) -> None:
        """A negative ``max_batch_size`` is refused.

        Matched on the shared ``"max_batch_size"`` substring rather than the full ``1 <= batch_size <= max_batch_size``
        bound message, which the existing bounds check in ``_check_capabilities`` raises.
        """
        with pytest.raises(ValueError, match="max_batch_size"):
            TensorRTExporter(TensorRTConfig(dynamic_batch=True, max_batch_size=-1))

    def test_static_request_ignores_the_bounds(self) -> None:
        """Without ``dynamic_batch`` the profile fields are inert, so a missing ``max_batch_size`` is fine."""
        TensorRTExporter(TensorRTConfig(dynamic_batch=False, opt_batch_size=8))

    def test_export_keywords_reach_the_configuration(self) -> None:
        """``RFDETR.export(batch_size=..., max_batch_size=...)`` lands on ``opt_batch_size`` / ``max_batch_size``."""
        config = TensorRTExporter.build_config(
            output_dir=Path("out"), dynamic_batch=True, batch_size=4, max_batch_size=16
        )
        assert (config.opt_batch_size, config.max_batch_size) == (4, 16)


class TestBuildEngineDynamicBatch:
    """A dynamic-batch build hands polygraphy one optimization profile spanning batch 1 through ``max_batch_size``.

    A graph whose batch axis disagrees with the request is refused in either direction, before anything is built.
    """

    def test_profile_spans_one_to_max_on_the_dynamic_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Min/opt/max keep the traced spatial shape and vary only the batch axis."""
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384)))
        captured = _patch_dynamic_polygraphy_chain(monkeypatch, network)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=True, opt_batch_size=4, max_batch_size=16))

        exporter.build_engine("/tmp/model.onnx")

        (profile,) = captured["config"]["profiles"]
        assert profile.entries == {
            "input": {"min": (1, 3, 384, 384), "opt": (4, 3, 384, 384), "max": (16, 3, 384, 384)}
        }
        assert captured["config"]["fp16"] is False

    def test_parsed_network_is_built_rather_than_reparsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tuple the loader produced for shape inspection is what ``engine_from_network`` receives."""
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384)))
        captured = _patch_dynamic_polygraphy_chain(monkeypatch, network)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=True, max_batch_size=8))

        exporter.build_engine("/tmp/model.onnx")

        assert captured["network"] == ("builder", network, "parser")

    @pytest.mark.parametrize("static_shape", [pytest.param((1, 2), id="fixed-batch"), pytest.param((), id="rank-0")])
    def test_static_inputs_are_left_out_of_the_profile(
        self, monkeypatch: pytest.MonkeyPatch, static_shape: tuple[int, ...]
    ) -> None:
        """Only inputs with a dynamic batch axis get a profile entry; a rank-0 input has no batch axis at all."""
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384)), _FakeNetworkInput("aux", static_shape))
        captured = _patch_dynamic_polygraphy_chain(monkeypatch, network)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=True, max_batch_size=8))

        exporter.build_engine("/tmp/model.onnx")

        (profile,) = captured["config"]["profiles"]
        assert set(profile.entries) == {"input"}

    def test_static_graph_under_dynamic_request_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An ONNX graph with no dynamic batch axis cannot honour the request and says so."""
        network = _FakeNetwork(_FakeNetworkInput("input", (1, 3, 384, 384)))
        _patch_dynamic_polygraphy_chain(monkeypatch, network)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=True, max_batch_size=8))

        with pytest.raises(ValueError, match="no network input has a dynamic batch axis"):
            exporter.build_engine("/tmp/model.onnx")

    @pytest.mark.parametrize(
        ("inputs", "named"),
        [
            pytest.param((_FakeNetworkInput("input", (-1, 3, 384, 384)),), "['input']", id="dynamic-input"),
            pytest.param(
                (_FakeNetworkInput("input", (1, 3, 384, 384)), _FakeNetworkInput("aux", (-1, 4))),
                "['aux']",
                id="one-of-two-inputs-dynamic",
            ),
        ],
    )
    def test_dynamic_graph_under_static_request_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch, inputs: tuple[_FakeNetworkInput, ...], named: str
    ) -> None:
        """Without a profile, polygraphy fixes a dynamic batch axis to 1, so the engine would accept batch 1 only.

        The error names exactly the inputs that carry a dynamic batch axis.
        """
        _patch_polygraphy_build_capture(monkeypatch, _FakeNetwork(*inputs))

        with pytest.raises(ValueError, match=rf"dynamic batch axis on {re.escape(named)}"):
            TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

    def test_dynamic_graph_under_static_request_is_not_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refusal comes before the builder runs, so no batch-1 engine is left behind."""
        build_args = _patch_polygraphy_build_capture(
            monkeypatch, _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384)))
        )

        with pytest.raises(ValueError):
            TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

        assert "network" not in build_args

    def test_static_request_error_names_the_callers_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """On a strongly typed TensorRT the network is parsed from an fp16 copy, but the user only knows their own
        file."""
        source = tmp_path / "model.onnx"
        _patch_polygraphy_build_capture(monkeypatch, _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384))))
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(
            tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(tmp_path / "model.fp16-abcd1234.onnx")
        )

        with pytest.raises(ValueError, match=re.escape(f"'{source}'")):
            TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(source))

    def test_static_request_builds_a_graph_with_a_scalar_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rank-0 input has no batch axis to inspect; the batch-axis check must skip it, not index into it."""
        network = _FakeNetwork(_STATIC_INPUT, _FakeNetworkInput("score_threshold", ()))
        build_args = _patch_polygraphy_build_capture(monkeypatch, network)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

        assert build_args["network"] == ("builder", network, "parser")

    @pytest.mark.parametrize(
        ("dynamic_batch", "input_shape"),
        [
            pytest.param(True, (1, 3, 384, 384), id="dynamic-request-on-static-graph"),
            pytest.param(False, (-1, 3, 384, 384), id="static-request-on-dynamic-graph"),
        ],
    )
    def test_rejected_graph_releases_the_parsed_network(
        self, monkeypatch: pytest.MonkeyPatch, dynamic_batch: bool, input_shape: tuple[int, ...]
    ) -> None:
        """A graph refused for its batch axis must release the parsed builder/network/parser rather than leak them.

        The refusal comes between `network_from_onnx_path` (which owns TensorRT resources) and `engine_from_network`
        (which only takes ownership of them on success). Regression test for that gap, in both directions a request and
        a graph can disagree.
        """
        released: list[str] = []

        class _RefCountedSentinel:
            def __init__(self, name: str) -> None:
                self._name = name

            def __del__(self) -> None:
                released.append(self._name)

        network = _FakeNetwork(_FakeNetworkInput("input", input_shape))

        def _fake_network_from_onnx_path(path: str) -> tuple:
            # A fresh tuple per call, so the only reference once returned lives in `_compile`'s frame -- the
            # production `del` on the error path is what must drop it, not a reference this closure retains.
            return (_RefCountedSentinel("builder"), network, _RefCountedSentinel("parser"))

        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _fake_network_from_onnx_path)
        monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
        # The refusal comes before these are reached; they are stubbed so that a build which wrongly goes ahead fails
        # on the missing ValueError rather than inside the real (or absent) polygraphy.
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda *args, **kwargs: "engine")
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
        monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=dynamic_batch, max_batch_size=8))

        # Bound so the traceback, and with it every frame of the failed build, is still alive at the assertion, as it
        # is for a caller that holds on to the error (or an IPython session keeping the last one). Dropped at once,
        # the frame would free `parsed` whether or not the production code releases it, and the test could not fail.
        with pytest.raises(ValueError, match="dynamic batch axis") as rejection:
            exporter.build_engine("model.onnx")

        assert set(released) == {"builder", "parser"}, f"parsed resources outlive the held {rejection.value!r}"

    def test_static_build_passes_no_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``dynamic_batch`` the builder configuration carries no ``profiles`` key at all."""
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        config_kwargs = _patch_polygraphy_chain(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("/tmp/model.onnx")

        assert "profiles" not in config_kwargs

    def test_strongly_typed_fp16_still_builds_the_profile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``fp16=True`` + ``dynamic_batch=True`` is the documented default call.

        Every other case in this class pins ``fp16=False``, so the cast-graph branch (TensorRT >= 11) had no coverage of
        the profile surviving alongside it. It must parse the profile from the cast source, not silently drop it.
        """
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384)))
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        captured: dict = {}

        def _network_from_onnx_path(path: str) -> tuple:
            captured["source"] = path
            return ("builder", network, "parser")

        def _create_config(*, fp16: bool, profiles: list) -> str:
            captured["config"] = {"fp16": fp16, "profiles": profiles}
            return "config"

        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(cast_path))
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _network_from_onnx_path)
        monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
        monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda parsed, config: "engine")
        monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, dynamic_batch=True, opt_batch_size=2, max_batch_size=8))

        exporter.build_engine(str(tmp_path / "model.onnx"))

        (profile,) = captured["config"]["profiles"]
        expected = {"input": {"min": (1, 3, 384, 384), "opt": (2, 3, 384, 384), "max": (8, 3, 384, 384)}}
        assert profile.entries == expected
        assert captured["config"]["fp16"] is False, "strong typing takes precision from the cast graph, not the flag"
        assert captured["source"] == str(cast_path)


@pytest.fixture
def fp16_cast_graph(tmp_path: Path) -> "onnx.GraphProto":
    """Graph of a tiny float32 model after a real round-trip through ``_cast_onnx_to_fp16``."""
    source = tmp_path / "tiny.onnx"
    onnx.save(_float32_model_with_cast(), source)
    return onnx.load(tensorrt_export._cast_onnx_to_fp16(str(source))).graph


class TestTensorRTMajor:
    """``_tensorrt_major`` reads the leading integer off a TensorRT version string."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("11.2.1.2", 11),
            ("10.16.1.11", 10),
            ("8.6.1", 8),
            ("12", 12),
            ("unknown", None),
            pytest.param("", None, id="empty"),
        ],
    )
    def test_parses_major(self, version: str, expected: int | None) -> None:
        """A numeric leading component is returned as an int; anything else yields None."""
        assert tensorrt_export._tensorrt_major(version) == expected


class TestBuildEngineWeaklyTyped:
    """TensorRT < 11 still exposes the FP16 builder flag: that path must stay exactly as it was."""

    @pytest.mark.parametrize("version", ["10.16.1.11", "9.0.0", "8.6.1"])
    def test_sets_the_builder_flag(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """Weak typing takes precision from the flag, so it must still be requested."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(version, has_fp16_flag=True))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx")

        assert config_kwargs == {"fp16": True}

    def test_builds_from_the_original_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No graph rewriting on a weakly typed build — the float32 ONNX is handed over untouched."""
        build_args = _patch_polygraphy_build_capture(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("model.onnx")

        assert build_args["source"] == "model.onnx"

    def test_engine_name_reports_fp16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Naming is unchanged from before the strong-typing branch existed."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))

        assert TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx") == "/tmp/model_fp16.trt"

    def test_fp32_request_keeps_the_fp32_engine_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``fp16=False`` on a weakly typed build names the engine ``_fp32`` as it always has."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("/tmp/model.onnx") == "/tmp/model_fp32.trt"

    def test_fp32_request_does_not_set_the_builder_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``fp16=False`` must reach the builder as an FP32 config, not merely as an FP32 filename."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("/tmp/model.onnx")

        assert config_kwargs == {"fp16": False}


class TestBuildEngineLeanWheelFallback:
    """A *weakly typed* TensorRT lacking the FP16 flag is a lean wheel: fall back to FP32."""

    @pytest.mark.parametrize("version", ["10.16.1.11", "8.6.1", "unknown"])
    def test_engine_name_reports_fp32(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """Without a graph-level alternative on TensorRT < 11, an FP32 engine beats failing the export."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(version, has_fp16_flag=False))

        assert TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx") == "/tmp/model_fp32.trt"

    @pytest.mark.parametrize("version", ["10.16.1.11", "8.6.1", "unknown"])
    def test_downgrades_the_builder_config_to_fp32(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """The downgrade has to reach the builder too, or the name and the engine disagree."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(version, has_fp16_flag=False))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx")

        assert config_kwargs == {"fp16": False}

    def test_does_not_cast_the_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The FP32 fallback must not invoke the fp16 graph caster."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx") == "/tmp/model_fp32.trt"

    def test_a_missing_builder_flag_symbol_still_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard: a wheel omitting ``BuilderFlag`` entirely used to raise ``AttributeError``.

        Scenario: a lean TensorRT < 11 exposes no ``BuilderFlag`` attribute at all, not merely a
        ``BuilderFlag`` without ``FP16``. Probing the flag before checking the attribute exists turned
        that wheel into a crash instead of the documented FP32 fallback.
        """
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt_without_builder_flag("10.16.1.11"))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx") == "/tmp/model_fp32.trt"


class TestBuildEngineStrongTyping:
    """On TensorRT >= 11 the FP16 flag is gone by design; precision comes from the ONNX graph."""

    def test_builds_from_the_cast_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Regression guard for #1453: the network must load from the cast graph, not the float32 one.

        Scenario: TensorRT 11 takes precision from the graph, so an FP16 request that still parses the
        original float32 ONNX yields an FP32 engine while every label downstream calls it FP16.
        """
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        build_args = _patch_polygraphy_build_capture(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(cast_path))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert build_args["source"] == str(cast_path)

    def test_engine_name_reports_fp16(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Regression guard for #1453: the engine really is FP16, so the filename must not say ``_fp32``."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(tmp_path / "model.fp16.onnx"))

        result = TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert result == str(tmp_path / "model_fp16.trt")

    def test_builder_flag_is_not_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Polygraphy aborts if asked for a flag TensorRT 11 removed, so the config must request FP32."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(tmp_path / "model.fp16.onnx"))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert config_kwargs == {"fp16": False}

    def test_a_surviving_fp16_flag_still_takes_the_cast_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression guard: strong typing, not the absent flag, is what selects the cast path.

        Scenario: a TensorRT 11 build that still exposes a deprecated ``BuilderFlag.FP16``. Deciding
        flag-first sent it down the weakly typed branch, which on a strongly typed builder produces an
        FP32 engine from an uncast graph — the very mislabelled precision this path exists to prevent.
        """
        build_args = _patch_polygraphy_build_capture(monkeypatch)
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(cast_path))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert build_args["source"] == str(cast_path)

    def test_a_surviving_fp16_flag_is_not_requested(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A strongly typed builder reads precision off the graph, so the flag must stay unrequested."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(tmp_path / "model.fp16.onnx"))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert config_kwargs == {"fp16": False}

    def test_fp32_request_does_not_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``fp16=False`` must take the plain FP32 path with no graph rewriting."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("/tmp/model.onnx") == "/tmp/model_fp32.trt"

    def test_raises_when_caster_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the caster there is no way to honour the request, so fail loudly."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"tensorrt<11"):
            TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx")

    def test_a_missing_caster_does_not_fall_through_to_an_fp32_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The caster-absent ImportError must abort the build, not downgrade it.

        Scenario: TensorRT 11 with no ``onnx``/``onnxconverter-common`` to cast the graph. Raising but
        still building would hand back an FP32 engine that anyone benchmarking reports as FP16 latency.
        """
        build_args = _patch_polygraphy_build_capture(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)

        with pytest.raises(ImportError):
            TensorRTExporter(TensorRTConfig(fp16=True)).build_engine("/tmp/model.onnx")

        assert not build_args, "an FP16 request must not fall through to an FP32 build"


class TestBuildEngineCastArtifactCleanup:
    """The cast graph is a build intermediate, so ``.trt`` stays the only file the export leaves."""

    def _build_with_cast(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, save_fails: bool) -> Path:
        """Run ``build_engine`` on the strongly typed path against a real cast file on disk.

        Args:
            monkeypatch: Fixture used to stub the polygraphy chain and the fake ``tensorrt``.
            tmp_path: Directory the stand-in cast graph is written to.
            save_fails: Whether ``save_engine`` should raise, simulating a failed build.

        Returns:
            Path the stand-in cast graph was written to, for an existence assertion.

        Examples:
            Needs live ``monkeypatch`` and ``tmp_path`` fixtures, so it cannot run standalone.

            >>> TestBuildEngineCastArtifactCleanup()._build_with_cast(mp, tmp, save_fails=False)
            ... # doctest: +SKIP
        """
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        cast_path.write_bytes(b"cast-graph")

        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr(
            tensorrt_export, "network_from_onnx_path", lambda path: ("builder", _FakeNetwork(_STATIC_INPUT), "parser")
        )
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: "engine")

        def _save_engine(engine, path):
            if save_fails:
                raise RuntimeError("builder ran out of workspace")

        monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(cast_path))
        return cast_path

    def test_cast_graph_is_removed_after_a_successful_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Users are told the export writes a ``.trt``; a stray ``.fp16.onnx`` beside it is a surprise."""
        cast_path = self._build_with_cast(monkeypatch, tmp_path, save_fails=False)

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert not cast_path.exists()

    def test_cast_graph_is_removed_after_a_failed_build(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A half-finished build must not leave the intermediate behind either."""
        cast_path = self._build_with_cast(monkeypatch, tmp_path, save_fails=True)

        with pytest.raises(RuntimeError):
            TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert not cast_path.exists()

    def test_missing_cast_graph_does_not_mask_the_build_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cleanup of an already-absent file must not raise over the real failure."""
        cast_path = self._build_with_cast(monkeypatch, tmp_path, save_fails=True)
        cast_path.unlink()

        with pytest.raises(RuntimeError, match="workspace"):
            TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))


@fp16_caster_only
class TestCastOnnxToFp16:
    """Graph-level fp16 conversion: weights become FP16 while graph I/O stays FP32."""

    def test_weights_become_fp16(self, fp16_cast_graph: onnx.GraphProto) -> None:
        """No float32 initializer may survive the cast, or the engine is not really FP16.

        Asserted as an absence rather than as an exact dtype set: a real RF-DETR graph also carries
        INT64 shape tensors, so pinning the set to ``{FLOAT16}`` would only ever hold for a toy fixture.
        """
        dtypes = {initializer.data_type for initializer in fp16_cast_graph.initializer}

        assert onnx.TensorProto.FLOAT not in dtypes

    @pytest.mark.parametrize("collection", ["input", "output"])
    def test_graph_io_stays_fp32(self, fp16_cast_graph: onnx.GraphProto, collection: str) -> None:
        """Callers feed and read float32 on a weakly typed FP16 engine; keep that contract."""
        dtypes = {value.type.tensor_type.elem_type for value in getattr(fp16_cast_graph, collection)}

        assert dtypes == {onnx.TensorProto.FLOAT}

    def test_preexisting_float_casts_are_retargeted(self, fp16_cast_graph: onnx.GraphProto) -> None:
        """A leftover ``Cast(to=FLOAT)`` feeding an FP16 consumer makes TensorRT reject the graph."""
        body = [node for node in fp16_cast_graph.node if not node.name.startswith("Cast_")]
        targets = [
            attribute.i
            for node in body
            if node.op_type == "Cast"
            for attribute in node.attribute
            if attribute.name == "to"
        ]

        assert onnx.TensorProto.FLOAT not in targets

    def test_model_is_valid(self, fp16_cast_graph: onnx.GraphProto) -> None:
        """The rewritten graph must still pass ONNX's own checker."""
        onnx.checker.check_model(helper.make_model(fp16_cast_graph, opset_imports=[helper.make_opsetid("", 17)]))

    def test_dynamic_batch_axis_survives_the_cast(self, tmp_path: Path) -> None:
        """A symbolic batch ``dim_param`` must survive the cast, not just the input's ``elem_type``.

        ``dynamic_batch`` exports carry a ``"batch"`` dim_param on the graph's batch axis instead of a
        fixed dim_value; ``_restore_fp32_inputs`` only rewrites ``elem_type`` in place and never touches
        ``shape.dim``, but that is worth asserting directly rather than trusting by omission -- a fp16=True
        + dynamic_batch=True build is the documented default call and would silently lose the profile's
        dynamic axis if this ever regressed.
        """
        source = tmp_path / "dynamic.onnx"
        onnx.save(_float32_model_with_dynamic_batch(), source)

        cast_model = onnx.load(tensorrt_export._cast_onnx_to_fp16(str(source)))

        input_dim = cast_model.graph.input[0].type.tensor_type.shape.dim[0]
        assert input_dim.dim_param == "batch"

    def test_raises_without_caster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ImportError must name both remedies: install the extra, or pin an older TensorRT."""
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            tensorrt_export._cast_onnx_to_fp16("/tmp/model.onnx")

    def test_does_not_claim_a_preexisting_file(self, tmp_path: Path) -> None:
        """``build_engine`` deletes what this returns, so it must never claim a file it did not create.

        The squatter sits at ``tiny.fp16.onnx``, the name a deterministic implementation would derive
        from the source stem. Asserting on the directory listing rather than on the squatter's bytes is
        what makes this fail if the naming ever becomes derived: a returned path that already existed is
        a file the caller is about to delete on someone else's behalf.
        """
        source = tmp_path / "tiny.onnx"
        onnx.save(_float32_model_with_cast(), source)
        (tmp_path / "tiny.fp16.onnx").write_bytes(b"not ours")
        before = set(tmp_path.iterdir())

        cast_path = Path(tensorrt_export._cast_onnx_to_fp16(str(source)))

        assert cast_path not in before

    def test_concurrent_casts_get_separate_files(self, tmp_path: Path) -> None:
        """Two builds from one source model must not hand each other's graph to the parser."""
        source = tmp_path / "tiny.onnx"
        onnx.save(_float32_model_with_cast(), source)

        first = tensorrt_export._cast_onnx_to_fp16(str(source))
        second = tensorrt_export._cast_onnx_to_fp16(str(source))

        assert first != second

    def test_cast_lands_beside_the_source_model(self, tmp_path: Path) -> None:
        """The intermediate rivals the model in size, and /tmp is often a size-capped tmpfs."""
        source = tmp_path / "tiny.onnx"
        onnx.save(_float32_model_with_cast(), source)

        cast_path = tensorrt_export._cast_onnx_to_fp16(str(source))

        assert Path(cast_path).parent == tmp_path


@pytest.fixture
def fp16_cast_graph_with_topk(tmp_path: Path) -> "onnx.GraphProto":
    """Graph of a tiny float32 ``TopK`` model after a real round-trip through ``_cast_onnx_to_fp16``."""
    source = tmp_path / "tiny_topk.onnx"
    onnx.save(_float32_model_with_topk(), source)
    return onnx.load(tensorrt_export._cast_onnx_to_fp16(str(source))).graph


@fp16_caster_only
class TestCastOnnxToFp16WithBlockListedOps:
    """``onnxconverter-common`` keeps precision-sensitive ops FP32; the rewrite must not undo that."""

    def test_the_guard_cast_into_a_block_listed_op_stays_fp32(self, fp16_cast_graph_with_topk: onnx.GraphProto) -> None:
        """Retargeting the cast that protects a block-listed op would silently run it in FP16.

        Scenario: RF-DETR's query selection uses ``TopK``, which the converter block-lists and fences
        with FP32 boundary casts. ``_retarget_float_casts`` flips ``Cast(to=FLOAT)`` wherever the output
        is declared FLOAT16; catching a block-list guard cast here removes the protection outright, and
        nothing downstream would report it.
        """
        guard = next(node for node in fp16_cast_graph_with_topk.node if node.output[0] == "topk_input_cast_0")
        targets = [attribute.i for attribute in guard.attribute if attribute.name == "to"]

        assert targets == [onnx.TensorProto.FLOAT]

    def test_an_int64_initializer_is_left_alone(self, fp16_cast_graph_with_topk: onnx.GraphProto) -> None:
        """``TopK``'s ``k`` is INT64; casting integer tensors would make the graph unparsable."""
        dtypes = {initializer.name: initializer.data_type for initializer in fp16_cast_graph_with_topk.initializer}

        assert dtypes["k"] == onnx.TensorProto.INT64

    def test_model_is_valid(self, fp16_cast_graph_with_topk: onnx.GraphProto) -> None:
        """A graph mixing FP16 body and FP32 block-list islands must still pass ONNX's own checker."""
        model = helper.make_model(fp16_cast_graph_with_topk, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        onnx.checker.check_model(model, full_check=True)


@fp16_caster_only
class TestCastOnnxToFp16GraphShapes:
    """Graph shapes whose boundary rewrite previously produced a structurally invalid fp16 model."""

    @pytest.mark.parametrize(
        "build_model",
        [
            pytest.param(_model_with_a_consumed_output, id="output-also-consumed-internally"),
            pytest.param(_model_with_an_initializer_output, id="output-defined-by-an-initializer"),
            pytest.param(_model_with_an_input_that_is_also_an_output, id="input-that-is-also-an-output"),
            pytest.param(_model_with_a_capturing_subgraph, id="subgraph-capturing-an-outer-tensor"),
            pytest.param(_model_with_a_preexisting_fp16_name, id="preexisting-fp16-tensor-name"),
        ],
    )
    def test_cast_model_passes_full_check(self, tmp_path: Path, build_model) -> None:
        """Each shape once yielded a graph TensorRT's parser would reject; the checker is the guard.

        Scenario: restoring FP32 graph I/O renames tensors and inserts boundary casts. Each of these
        shapes breaks a different assumption that rewrite used to make — a renamed output whose other
        consumers were left behind, a definition held by an initializer rather than a node, a tensor
        declared as both input and output, a name captured inside an ``If`` body, and a graph that
        already binds the generated ``_fp16`` name. ``full_check=True`` is required: the subgraph case
        surfaces only through strict type inference, not through plain structural validation.
        """
        source = tmp_path / "tiny.onnx"
        onnx.save(build_model(), source)

        cast_path = tensorrt_export._cast_onnx_to_fp16(str(source))

        onnx.checker.check_model(onnx.load(cast_path), full_check=True)


def _fake_benchmark_tensorrt(
    version: str,
    *,
    has_fp16_flag: bool,
    has_explicit_batch: bool = True,
    parse_succeeds: bool = True,
) -> types.ModuleType:
    """Extend ``_fake_tensorrt`` with the builder stack ``TRTInference.build_engine`` drives directly.

    That method uses the raw TensorRT API rather than polygraphy, so it needs a ``Builder``, an
    ``OnnxParser`` and a builder config, each usable as a context manager. Every call the method makes
    is recorded on ``module.record`` so a test can assert on the flags and the graph it actually used.

    Args:
        version: Value to expose as ``tensorrt.__version__``.
        has_fp16_flag: Whether ``BuilderFlag`` should carry an ``FP16`` member.
        has_explicit_batch: Whether ``NetworkDefinitionCreationFlag`` should carry ``EXPLICIT_BATCH``.
        parse_succeeds: Whether ``OnnxParser.parse`` reports success.

    Returns:
        A module object suitable for ``monkeypatch.setattr(inference, "trt", ...)``.

    Examples:
        >>> module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, has_explicit_batch=False)
        >>> hasattr(module.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
        False
        >>> module.record["flags_set"]
        []
    """
    module = _fake_tensorrt(version, has_fp16_flag=has_fp16_flag)
    record: dict = {"flags_set": [], "network_flags": None, "parsed": None, "written": None}
    module.record = record

    class _Closeable:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class _Config(_Closeable):
        def set_memory_pool_limit(self, pool, size):
            record["workspace"] = size

        def set_flag(self, flag):
            record["flags_set"].append(flag)

    class _Builder(_Closeable):
        def __init__(self, logger):
            record["logger"] = logger

        def create_network(self, flags):
            record["network_flags"] = flags
            return _Closeable()

        def create_builder_config(self):
            return _Config()

        def build_serialized_network(self, network, config):
            return b"serialized-engine"

    class _Parser(_Closeable):
        num_errors = 1

        def __init__(self, network, logger):
            pass

        def parse(self, payload):
            record["parsed"] = payload
            return parse_succeeds

        def get_error(self, index):
            return f"parser error {index}"

    class MemoryPoolType:
        WORKSPACE = 0

    class NetworkDefinitionCreationFlag:
        pass

    if has_explicit_batch:
        NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0

    module.Builder = _Builder
    module.OnnxParser = _Parser
    module.MemoryPoolType = MemoryPoolType
    module.NetworkDefinitionCreationFlag = NetworkDefinitionCreationFlag
    return module


def _run_benchmark_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trt_module: types.ModuleType, cast_path: Path | None
) -> object:
    """Drive ``TRTInference.build_engine`` unbound against *trt_module*, with a real ONNX file on disk.

    ``TRTInference.__init__`` needs a GPU and a serialized engine, so the method is called unbound
    against a stand-in carrying only the ``logger`` attribute it touches — the pattern the method's own
    docstring documents.

    Args:
        monkeypatch: Fixture used to replace ``inference.trt`` and the fp16 graph caster.
        tmp_path: Directory the source model and the engine are written to.
        trt_module: Stand-in ``tensorrt`` module from :func:`_fake_benchmark_tensorrt`.
        cast_path: File the stubbed caster returns, or ``None`` to leave the caster untouched.

    Returns:
        Whatever ``build_engine`` returned — the serialized engine, or ``None`` when parsing failed.

    Examples:
        Needs live ``monkeypatch`` and ``tmp_path`` fixtures, so it cannot run standalone.

        >>> _run_benchmark_build(mp, tmp, _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False), None)
        ... # doctest: +SKIP
    """
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx-bytes")
    monkeypatch.setattr(tensorrt_inference, "trt", trt_module)
    if cast_path is not None:
        cast_path.write_bytes(b"cast-graph")
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(cast_path))
    return tensorrt_inference.TRTInference.build_engine(
        types.SimpleNamespace(logger="trt-logger"), str(source), str(tmp_path / "model.trt")
    )


class TestBenchmarkBuildEngine:
    """``TRTInference.build_engine`` always requests FP16, so it runs the same strategy decision."""

    def test_strongly_typed_parses_the_cast_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """TensorRT 11 takes precision from the graph, so the parser must be fed the cast copy."""
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, tmp_path / "model.fp16-abcd1234.onnx")

        assert trt_module.record["parsed"] == b"cast-graph"

    def test_strongly_typed_does_not_set_the_fp16_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Requesting a flag TensorRT 11 removed is what broke this method in the first place."""
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, tmp_path / "model.fp16-abcd1234.onnx")

        assert trt_module.record["flags_set"] == []

    def test_weakly_typed_sets_the_fp16_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """TensorRT < 11 takes precision from the builder flag, so it must still be requested."""
        trt_module = _fake_benchmark_tensorrt("10.16.1.11", has_fp16_flag=True)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, None)

        assert trt_module.record["flags_set"] == [trt_module.BuilderFlag.FP16]

    def test_weakly_typed_parses_the_original_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No graph rewriting on a weakly typed build — the caller's own ONNX is parsed untouched."""
        trt_module = _fake_benchmark_tensorrt("10.16.1.11", has_fp16_flag=True)
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, None)

        assert trt_module.record["parsed"] == b"onnx-bytes"

    def test_a_lean_wheel_falls_back_without_the_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A weakly typed wheel with no FP16 flag has no FP16 route, so it benchmarks FP32 instead."""
        trt_module = _fake_benchmark_tensorrt("10.16.1.11", has_fp16_flag=False)
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, None)

        assert trt_module.record["flags_set"] == []

    def test_explicit_batch_flag_is_set_when_available(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """On TensorRT < 11 explicit batch is opt-in, so the network must be created with its bit set."""
        trt_module = _fake_benchmark_tensorrt("10.16.1.11", has_fp16_flag=True)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, None)

        assert trt_module.record["network_flags"] == 1 << int(trt_module.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    def test_an_absent_explicit_batch_flag_yields_an_empty_flag_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression guard: TensorRT 11 removed ``EXPLICIT_BATCH`` because it is the only mode there.

        Scenario: a strongly typed TensorRT whose ``NetworkDefinitionCreationFlag`` has no
        ``EXPLICIT_BATCH`` member. Reading it unconditionally raises ``AttributeError`` — and it is read
        after the cast graph is written, so the crash would also leak that intermediate.
        """
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, has_explicit_batch=False)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, tmp_path / "model.fp16-abcd1234.onnx")

        assert trt_module.record["network_flags"] == 0

    def test_cast_graph_is_removed_after_a_successful_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The cast copy is a build intermediate; leaving it beside the user's model is a surprise."""
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"

        _run_benchmark_build(
            monkeypatch, tmp_path, _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False), cast_path
        )

        assert not cast_path.exists()

    def test_cast_graph_is_removed_after_a_failed_parse(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A parse failure returns early rather than raising, and must still not leak the intermediate."""
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, parse_succeeds=False)

        _run_benchmark_build(monkeypatch, tmp_path, trt_module, cast_path)

        assert not cast_path.exists()

    def test_a_failed_parse_returns_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An unparsable ONNX yields ``None`` rather than an engine the caller would go on to use."""
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, parse_succeeds=False)

        result = _run_benchmark_build(monkeypatch, tmp_path, trt_module, tmp_path / "model.fp16-abcd1234.onnx")

        assert result is None


def _distinct_batch(batch: int, resolution: int) -> torch.Tensor:
    """Stack *batch* structured inputs with different per-image scaling, so batch positions are not interchangeable.

    ``_structured_parity_input`` repeats one sample across the batch; a dynamic-batch engine that mixed up or
    duplicated batch positions would still pass on that. Scaling each image differently makes every position
    distinguishable.

    Args:
        batch: Number of images to stack.
        resolution: Square spatial size of each image.

    Returns:
        Contiguous float tensor shaped ``(batch, 3, resolution, resolution)``.

    Examples:
        >>> t = _distinct_batch(3, 8)
        >>> t.shape
        torch.Size([3, 3, 8, 8])
        >>> bool(torch.equal(t[0], t[1]))
        False
    """
    base = _structured_parity_input(1, 3, resolution, resolution)
    return torch.cat([base * (1.0 + 0.15 * index) for index in range(batch)], dim=0).contiguous()


@tensorrt_only
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestTensorRTEndToEnd:
    """Real ONNX -> TensorRT engine build + runtime parity on GPU (requires ``rfdetr[tensorrt]`` and CUDA)."""

    @pytest.fixture(scope="class")
    def trt_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
        """Export RFDETRNano to ONNX, build a FP32 ``.trt`` engine, and reuse it across the parity checks."""
        from rfdetr import RFDETRNano

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt")
        detector = RFDETRNano(pretrain_weights=None)
        onnx_path = detector.export(output_dir=str(out_dir), format="onnx", verbose=False)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False))
        engine_path = exporter.build_engine(str(onnx_path))

        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        return model, example, Path(engine_path)

    def test_engine_file_written(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """build_engine must produce a non-empty ``.trt`` engine from the exported ONNX."""
        _, _, engine_path = trt_engine
        assert engine_path.is_file()
        assert engine_path.suffix == ".trt"
        assert engine_path.stat().st_size > 0

    def test_runtime_output_matches_pytorch(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """The TensorRT engine's outputs (dets, labels) must match eager PyTorch within FP32 tolerance."""
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, example, engine_path = trt_engine
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.asarray(outputs[name], dtype=np.float32)) for name in output_names]

        diffs = max_abs_output_diffs(eager_tensors, trt_tensors, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_MAX_ABS_DIFF, (
            f"TensorRT outputs diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_MAX_ABS_DIFF})"
        )

    @pytest.fixture(scope="class")
    def trt_fp16_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
        """Export RFDETRNano to ONNX, build an FP16 ``.trt`` engine, and reuse it across the parity checks.

        Mirrors ``trt_engine`` with ``fp16=True``: on a strongly typed TensorRT that routes through the whole-graph FP16
        cast, which is the path with no numerical evidence behind it otherwise.
        """
        from rfdetr import RFDETRNano

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt_fp16")
        detector = RFDETRNano(pretrain_weights=None)
        onnx_path = detector.export(output_dir=str(out_dir), format="onnx", verbose=False)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, verbose=False))
        engine_path = exporter.build_engine(str(onnx_path))

        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        return model, example, Path(engine_path)

    def test_fp16_engine_file_written(self, trt_fp16_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """An FP16 request must produce an engine named ``_fp16``, not a silently downgraded ``_fp32``."""
        _, _, engine_path = trt_fp16_engine

        assert engine_path.stem.endswith("_fp16")

    def test_fp16_runtime_output_matches_pytorch(
        self, trt_fp16_engine: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """The FP16 engine's outputs must stay within the FP16 bound of eager PyTorch.

        Scenario: the FP16 path casts the whole graph, so every non-block-listed op runs in FP16 with no
        per-layer FP32 fallback. This is the only check in the repo that evaluates a number from an FP16
        engine — engine file size proves weight storage, not detections. Values are compared sorted
        rather than positionally: the graph ends in ``TopK`` query selection, and on this randomly
        initialised fixture the logits sit in a narrow band where FP16 rounding can reorder near-tied
        queries. A positional diff would then explode to O(1) on a numerically healthy engine, so the
        comparison is deliberately order-insensitive. NaN or Inf still fails, because neither compares
        less than the bound.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, example, engine_path = trt_fp16_engine
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.asarray(outputs[name], dtype=np.float32)) for name in output_names]
        sorted_eager = [torch.sort(tensor.flatten()).values for tensor in eager_tensors]
        sorted_trt = [torch.sort(tensor.flatten()).values for tensor in trt_tensors]

        diffs = max_abs_output_diffs(sorted_eager, sorted_trt, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_FP16_MAX_ABS_DIFF, (
            f"FP16 TensorRT outputs diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_FP16_MAX_ABS_DIFF})"
        )

    @pytest.fixture(scope="class")
    def trt_dynamic_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, int, Path]:
        """Export RFDETRNano with a dynamic batch axis and build one FP32 engine spanning batch 1 through 4.

        Built through ``RFDETR.export(format="tensorrt", ...)`` rather than ``build_engine`` directly, so the
        ``batch_size`` / ``max_batch_size`` keywords are exercised end to end.
        """
        from rfdetr import RFDETRNano

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt_dynamic")
        detector = RFDETRNano(pretrain_weights=None)
        engine_path = detector.export(
            output_dir=str(out_dir),
            format="tensorrt",
            fp16=False,
            dynamic_batch=True,
            batch_size=2,
            max_batch_size=4,
            verbose=False,
        )

        model = detector.model.model.to("cpu").eval()
        model.export()
        return model, int(detector.model.resolution), Path(engine_path)

    @pytest.mark.parametrize("batch", [1, 3, 4])
    def test_dynamic_engine_matches_pytorch_at_each_batch(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], batch: int
    ) -> None:
        """One engine must serve every batch inside its profile, each image matching eager PyTorch.

        The batch repeats the same structured image the static parity tests use: on this randomly initialised
        fixture the two-stage top-k sits on near-ties for other inputs, where a rank swap turns a healthy engine
        into an O(1) positional diff (see ``test_fp16_runtime_output_matches_pytorch``). Whether distinct images
        stay independent inside a batch is ``test_dynamic_engine_batch_positions_are_independent``'s job.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, resolution, engine_path = trt_dynamic_engine
        example = _structured_parity_input(batch, 3, resolution, resolution)
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.array(outputs[name], dtype=np.float32)) for name in output_names]

        diffs = max_abs_output_diffs(eager_tensors, trt_tensors, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_MAX_ABS_DIFF, (
            f"dynamic TensorRT outputs at batch {batch} diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_MAX_ABS_DIFF})"
        )

    def test_dynamic_engine_batch_positions_are_independent(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """Four distinct images run as one batch must each equal the same image run alone through the same engine.

        ``TrtRunner.infer`` reuses its host output buffers between calls, so every result is copied out before the next
        call.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        _, resolution, engine_path = trt_dynamic_engine
        example = np.ascontiguousarray(_distinct_batch(4, resolution).numpy())
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            batched = {name: np.array(value) for name, value in runner.infer(feed_dict={"input": example}).items()}
            alone = [
                {name: np.array(value) for name, value in runner.infer(feed_dict={"input": example[i : i + 1]}).items()}
                for i in range(4)
            ]

        for name in ("dets", "labels"):
            for index in range(4):
                diff = float(np.abs(batched[name][index] - alone[index][name][0]).max())
                assert diff < 1e-4, f"{name} for image {index} differs between batch 4 and batch 1: {diff}"

    def test_dynamic_engine_rejects_a_batch_beyond_the_profile(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A batch above ``max_batch_size`` is outside the profile and must not silently run.

        Polygraphy's ``TrtRunner.infer`` reports an out-of-profile shape by having ``G_LOGGER.critical`` raise a
        ``PolygraphyException`` naming the failed ``set_input_shape`` call -- narrower than a bare ``Exception``, which
        would also swallow an unrelated crash (OOM, a driver error) as a false pass.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        from polygraphy.exception import PolygraphyException

        _, resolution, engine_path = trt_dynamic_engine
        feed = {"input": np.ascontiguousarray(_distinct_batch(5, resolution).numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner, pytest.raises(PolygraphyException, match="failed to set shape"):
            runner.infer(feed_dict=feed)

    def test_trt_inference_helper_serves_the_dynamic_engine(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """``TRTInference`` allocates at the profile's max batch, trims outputs to the batch run, and agrees with
        polygraphy."""
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        _, resolution, engine_path = trt_dynamic_engine
        example = _distinct_batch(3, resolution)
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            reference = {
                name: np.array(value)
                for name, value in runner.infer(feed_dict={"input": np.ascontiguousarray(example.numpy())}).items()
            }

        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        assert runtime.bindings["input"].shape[0] == 4
        outputs = runtime({"input": example.to("cuda:0")})

        for name in ("dets", "labels"):
            got = outputs[name].detach().float().cpu().numpy()
            assert got.shape == reference[name].shape
            diff = float(np.abs(got - reference[name]).max())
            assert diff < 1e-4, f"TRTInference {name} differs from polygraphy on the same engine: {diff}"

    def test_trt_inference_helper_refuses_a_batch_beyond_the_profile(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A real context returns ``False`` from ``set_input_shape`` for batch 5; the helper must raise, not run."""
        _, resolution, engine_path = trt_dynamic_engine
        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)

        with pytest.raises(ValueError, match="outside the engine's optimization profile"):
            runtime({"input": _distinct_batch(5, resolution).to("cuda:0")})
