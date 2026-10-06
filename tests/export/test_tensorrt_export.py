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

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints

import numpy as np
import pytest
import torch

from rfdetr.detr import RFDETR
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import inference as tensorrt_inference
from rfdetr.export._tensorrt.exporter import (
    _IS_FP16_CASTER_AVAILABLE,
    _IS_POLYGRAPHY_AVAILABLE,
    _IS_TENSORRT_AVAILABLE,
    TensorRTConfig,
    TensorRTExporter,
)
from rfdetr.export.prepare import ExportGraph
from tests.export.conftest import (
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)

tensorrt_only = pytest.mark.skipif(
    not (_IS_TENSORRT_AVAILABLE and _IS_POLYGRAPHY_AVAILABLE), reason="tensorrt/polygraphy not installed"
)
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


#: Stand-in for the engine polygraphy builds. The exporter only serializes it, once, and writes those bytes.
_FAKE_ENGINE = types.SimpleNamespace(serialize=lambda: b"engine")


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
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr(
        tensorrt_export, "network_from_onnx_path", lambda path: ("builder", _FakeNetwork(_STATIC_INPUT), "parser")
    )
    monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
    monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: _FAKE_ENGINE)
    monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
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
        return _FAKE_ENGINE

    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _network_from_onnx_path)
    monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
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
    """Build a stand-in ``polygraphy.backend.trt`` exposing the four names the exporter imports from it.

    Polygraphy imports ``tensorrt`` only when one of these is called, so importing them succeeds on a host without
    TensorRT; this stand-in reproduces that on hosts where polygraphy is not installed at all.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "polygraphy.backend.trt", ...)``.

    Examples:
        >>> sorted(name for name in vars(_fake_polygraphy_trt()) if not name.startswith("__"))
        ['CreateConfig', 'Profile', 'engine_from_network', 'network_from_onnx_path']
    """
    module = types.ModuleType("polygraphy.backend.trt")
    for name in ("CreateConfig", "Profile", "engine_from_network", "network_from_onnx_path"):
        setattr(module, name, object())
    return module


def _unexpected_cast(onnx_path: str, **_: object) -> str:
    """Stand in for ``_cast_onnx_to_fp16`` on paths that must never cast, failing loudly if called.

    Args:
        onnx_path: Path the caller tried to cast.
        **_: The keyword arguments the real cast takes (``dynamic_batch``), accepted and ignored so a call lands on
            the assertion below rather than on a ``TypeError`` that reads like a signature mismatch.

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
    """``tensorrt`` and ``polygraphy`` are probed apart, and a build refuses the host missing either."""

    @pytest.mark.parametrize(
        ("polygraphy_installed", "tensorrt_layout", "expected_flags"),
        [
            pytest.param(True, "package", (True, True), id="both-installed"),
            pytest.param(True, "absent", (False, True), id="polygraphy-only"),
            pytest.param(True, "bare-directory", (False, True), id="bare-tensorrt-directory"),
            pytest.param(False, "package", (True, False), id="tensorrt-only"),
        ],
    )
    def test_each_package_is_reported_separately(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        polygraphy_installed: bool,
        tensorrt_layout: str,
        expected_flags: tuple[bool, bool],
    ) -> None:
        """Each flag answers for its own package, so a refusal can name the one that is actually missing.

        Polygraphy imports ``tensorrt`` lazily, so its own import succeeds without it — and ``rfdetr[onnx]`` installs it
        alone. A bare ``tensorrt/`` directory, such as an export folder named after the format, is importable as a
        namespace package but is no install either. The exporter source is executed under a private module name, with
        only *tmp_path* searched for ``tensorrt``, so a real TensorRT on the host cannot answer and the real module
        keeps its identity for the other tests.
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
        # The engine is written with polygraphy's own file writer, the one other name the exporter imports.
        polygraphy_util = types.ModuleType("polygraphy.util")
        polygraphy_util.save_file = object()
        monkeypatch.setitem(sys.modules, "polygraphy.util", polygraphy_util if polygraphy_installed else None)
        monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        spec = importlib.util.spec_from_file_location("_tensorrt_exporter_probe", tensorrt_export.__file__)
        probe = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, probe)

        spec.loader.exec_module(probe)

        assert (probe._IS_TENSORRT_AVAILABLE, probe._IS_POLYGRAPHY_AVAILABLE) == expected_flags

    def test_missing_tensorrt_raises_the_install_hint(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Building from an existing ONNX file without TensorRT raises an actionable ImportError, not a polygraphy
        one."""
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            TensorRTExporter(TensorRTConfig()).build_engine(str(tmp_path / "model.onnx"))

    def test_the_refusal_names_only_the_missing_package(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A host with TensorRT but no polygraphy is told about polygraphy alone.

        ``rfdetr[onnx]`` installs polygraphy without TensorRT, and a lean TensorRT install is the mirror case, so a
        message naming both packages sends the reader hunting for an install that is already there.
        """
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"^TensorRT export requires 'polygraphy',"):
            TensorRTExporter(TensorRTConfig()).build_engine(str(tmp_path / "model.onnx"))

    def test_check_dependencies_reports_a_missing_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The hook ``RFDETR.export()`` calls before preparing the graph refuses the host without an exporter.

        It is a classmethod because the facade reaches it from the resolved exporter class, and it must answer without
        constructing anything or importing TensorRT.
        """
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            TensorRTExporter.check_dependencies()

    def test_dry_run_needs_no_tensorrt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``dry_run`` is documented as needing no TensorRT, so the availability check stays behind it."""
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)

        result = TensorRTExporter(TensorRTConfig()).build_engine(str(tmp_path / "model.onnx"), dry_run=True)

        assert result == str(tmp_path / "model_fp16.trt")


def _minimal_export_graph() -> ExportGraph:
    """Build the smallest static `ExportGraph` `TensorRTExporter._convert()` needs, without a real RF-DETR model.

    Examples:
        >>> graph = _minimal_export_graph()
        >>> graph.backbone_only
        False
        >>> graph.input_tensors.shape
        torch.Size([1, 3, 8, 8])
    """
    return ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 3, 8, 8),
        input_names=("input",),
        output_names=("dets",),
        dynamic_axes=None,
        shape=(8, 8),
        backbone_only=False,
    )


class TestConvertDependencyGuard:
    """`TensorRTExporter._convert()` (not just `build_engine()`) must fail actionably without TensorRT.

    Regression coverage for the gap the challenger flagged (L7): the existing `RFDETR.export()` E2E tests in
    `test_export.py` monkeypatch `OnnxExporter._convert` and `_build` together, so a missing-TensorRT failure they
    exercise is coupled to `RFDETR.export()`'s device-move/deepcopy plumbing. Calling `_convert()` directly here
    decouples the two.
    """

    def test_missing_polygraphy_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """`_convert()` must raise its actionable ImportError before the ONNX export stage runs.

        `_convert()` calls `_require_tensorrt()` first and only then exports to ONNX (see `TensorRTExporter._convert`),
        so a host missing `tensorrt`/`polygraphy` never pays for the ONNX stage before being refused. This asserts the
        ONNX stage does *not* run.
        """
        onnx_path = str(tmp_path / "model.onnx")
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or onnx_path,
        )
        graph = _minimal_export_graph()

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            TensorRTExporter(TensorRTConfig())._convert(graph)

        assert onnx_calls == [], "expected the ONNX stage to be skipped — _require_tensorrt() runs ahead of it"


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
        monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
        monkeypatch.setattr(
            tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config-sentinel"
        )

        def _network_from_onnx_path(path: str) -> tuple:
            build_args["source"] = path
            return ("builder", network, "parser")

        def _engine_from_network(network, config):
            build_args["network"] = network
            build_args["config"] = config
            return types.SimpleNamespace(serialize=lambda: b"engine-sentinel")

        def _save_file(contents, dest, description=None):
            saved["contents"] = contents
            saved["dest"] = dest

        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _network_from_onnx_path)
        monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
        monkeypatch.setattr(tensorrt_export, "save_file", _save_file)

        result = TensorRTExporter(TensorRTConfig(fp16=fp16)).build_engine("model.onnx")
        expected_path = f"model_{'fp16' if fp16 else 'fp32'}.trt"

        assert result == expected_path
        assert config_kwargs == {"fp16": fp16}
        assert build_args == {
            "source": "model.onnx",
            "network": ("builder", network, "parser"),
            "config": "config-sentinel",
        }
        assert saved == {"contents": b"engine-sentinel", "dest": expected_path}


@dataclass(frozen=True)
class _FakeNetworkInput:
    """One parsed network input: a name and a TensorRT-style shape (``-1`` marks the dynamic batch axis)."""

    name: str
    shape: tuple[int, ...]


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
        return _FAKE_ENGINE

    def _create_config(*, fp16: bool, profiles: list) -> str:
        # Signature-bound (not **kwargs) so a real ``CreateConfig`` keyword rename in ``_compile`` fails
        # this stub with a TypeError instead of silently swallowing it.
        captured["config"] = {"fp16": fp16, "profiles": profiles}
        return "config"

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("builder", network, "parser"))
    monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
    monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
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
        """An ONNX graph with no dynamic batch axis cannot honour the request, and the refusal quotes that graph.

        The opposite mismatch names the file it read, so this one does too: whichever direction the disagreement runs,
        the reader is told which ``.onnx`` to re-export rather than only what the flag said.
        """
        network = _FakeNetwork(_FakeNetworkInput("input", (1, 3, 384, 384)))
        _patch_dynamic_polygraphy_chain(monkeypatch, network)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=True, max_batch_size=8))

        with pytest.raises(ValueError, match=r"^'/tmp/model\.onnx' has no input with a dynamic batch axis"):
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
            tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(tmp_path / "model.fp16-abcd1234.onnx")
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
        a graph can disagree. The network is tracked alongside the builder and parser because it, unlike them, is also
        a parameter of the frames that raise: dropping the parsed tuple alone leaves it alive in their traceback.
        """
        released: list[str] = []

        class _RefCountedSentinel:
            def __init__(self, name: str) -> None:
                self._name = name

            def __del__(self) -> None:
                released.append(self._name)

        class _RefCountedNetwork(_FakeNetwork):
            def __del__(self) -> None:
                released.append("network")

        def _fake_network_from_onnx_path(path: str) -> tuple:
            # A fresh tuple -- and a fresh network -- per call, so the only references once returned live in the
            # frames of the failed build. The production releases are what must drop them, not a reference this
            # closure, or the test body, retains: either would free them whatever the production code does.
            return (
                _RefCountedSentinel("builder"),
                _RefCountedNetwork(_FakeNetworkInput("input", input_shape)),
                _RefCountedSentinel("parser"),
            )

        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _fake_network_from_onnx_path)
        monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
        # The refusal comes before these are reached; they are stubbed so that a build which wrongly goes ahead fails
        # on the missing ValueError rather than inside the real (or absent) polygraphy.
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda *args, **kwargs: _FAKE_ENGINE)
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
        monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, dynamic_batch=dynamic_batch, max_batch_size=8))

        # Bound so the traceback, and with it every frame of the failed build, is still alive at the assertion, as it
        # is for a caller that holds on to the error (or an IPython session keeping the last one). Dropped at once,
        # the frame would free `parsed` whether or not the production code releases it, and the test could not fail.
        with pytest.raises(ValueError, match="dynamic batch axis") as rejection:
            exporter.build_engine("model.onnx")

        assert set(released) == {"builder", "network", "parser"}, (
            f"parsed resources outlive the held {rejection.value!r}"
        )

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
        monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(cast_path))
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", _network_from_onnx_path)
        monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
        monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda parsed, config: _FAKE_ENGINE)
        monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, dynamic_batch=True, opt_batch_size=2, max_batch_size=8))

        exporter.build_engine(str(tmp_path / "model.onnx"))

        (profile,) = captured["config"]["profiles"]
        expected = {"input": {"min": (1, 3, 384, 384), "opt": (2, 3, 384, 384), "max": (8, 3, 384, 384)}}
        assert profile.entries == expected
        assert captured["config"]["fp16"] is False, "strong typing takes precision from the cast graph, not the flag"
        assert captured["source"] == str(cast_path)


def _patch_polygraphy_chain_recording(monkeypatch: pytest.MonkeyPatch, network: _FakeNetwork | None = None) -> dict:
    """Stub the polygraphy chain and record every keyword the build hands to ``CreateConfig`` and the engine build.

    The other stubs bind their signature to the exact keywords ``_compile`` passes today, so a renamed keyword fails
    loudly. The keywords a feature adds are optional, so this one accepts any keyword and lets the test assert on
    exactly which were passed, and which were not.

    Args:
        monkeypatch: Fixture used to replace the polygraphy entry points on the module under test.
        network: The parsed-network stand-in the loader hands back, or ``None`` for one fixed-batch input.

    Returns:
        Dict with the ``CreateConfig`` keywords under ``"config"`` and the keywords ``engine_from_network`` received
        besides the parsed network and the configuration under ``"build"``.

    Examples:
        >>> with pytest.MonkeyPatch.context() as monkeypatch:
        ...     captured = _patch_polygraphy_chain_recording(monkeypatch)
        ...     _ = tensorrt_export.CreateConfig(fp16=True, option=1)
        ...     _ = tensorrt_export.engine_from_network("network", config="config", build_option=2)
        >>> captured
        {'config': {'fp16': True, 'option': 1}, 'build': {'build_option': 2}}
    """
    captured: dict = {"config": {}, "build": {}}
    parsed = network if network is not None else _FakeNetwork(_STATIC_INPUT)

    def _create_config(**kwargs: object) -> str:
        captured["config"].update(kwargs)
        return "config"

    def _engine_from_network(_parsed: object, config: object, **kwargs: object) -> str:
        captured["build"].update(kwargs)
        return types.SimpleNamespace(serialize=lambda: b"engine")

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("builder", parsed, "parser"))
    monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
    monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
    return captured


class TestTimingCache:
    """``timing_cache`` reaches Polygraphy's config and engine build, and nothing is passed unless it is set."""

    def test_no_timing_cache_keyword_is_passed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``timing_cache`` the build calls Polygraphy exactly as it always did."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

        assert captured["config"] == {"fp16": False}
        assert captured["build"] == {}

    @pytest.mark.parametrize("dynamic_batch", [False, True])
    def test_an_existing_cache_is_loaded_and_saved_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dynamic_batch: bool
    ) -> None:
        """A cache file that exists seeds the build, and the same file receives the merged result.

        The exporter probes the file before the build to refuse an unwritable cache, and that probe must leave the
        timings a user already collected untouched; emptying them would silently throw the warm cache away.
        """
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384) if dynamic_batch else (1, 3, 384, 384)))
        captured = _patch_polygraphy_chain_recording(monkeypatch, network)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        config = TensorRTConfig(fp16=False, dynamic_batch=dynamic_batch, max_batch_size=4, timing_cache=cache)

        TensorRTExporter(config).build_engine("model.onnx")

        assert captured["config"]["load_timing_cache"] == str(cache)
        assert captured["build"] == {"save_timing_cache": str(cache)}
        assert cache.read_bytes() == b"previous timings"

    def test_a_missing_cache_is_saved_but_not_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The first use has no file yet: loading it would only make Polygraphy warn about a missing cache."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "not-yet.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert "load_timing_cache" not in captured["config"]
        assert captured["build"] == {"save_timing_cache": str(cache)}

    @pytest.mark.parametrize("dynamic_batch", [False, True])
    def test_the_keywords_handed_to_polygraphy_exist_in_its_api(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dynamic_batch: bool
    ) -> None:
        """Every keyword the exporter passes with a timing cache is one the installed Polygraphy accepts.

        The recording stub takes any keyword, so a keyword Polygraphy renamed or removed would leave the CPU suite green
        while only a GPU run broke. The keywords the exporter really passes are recorded for a fixed-batch and a
        dynamic-batch build (the latter adds a profile next to the cache), then checked against the signatures of the
        real ``CreateConfig`` and ``engine_from_network``. Skipped where Polygraphy does not import.
        """
        trt_backend = pytest.importorskip("polygraphy.backend.trt", exc_type=ImportError)
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384) if dynamic_batch else (1, 3, 384, 384)))
        captured = _patch_polygraphy_chain_recording(monkeypatch, network)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        config = TensorRTConfig(fp16=False, dynamic_batch=dynamic_batch, max_batch_size=4, timing_cache=cache)

        TensorRTExporter(config).build_engine("model.onnx")

        # ``CreateConfig`` forwards most of its options to a base class through ``**kwargs``, so collect them all.
        create_config_keywords = {
            name
            for cls in trt_backend.CreateConfig.__mro__
            if cls is not object and "__init__" in vars(cls)
            for name in inspect.signature(vars(cls)["__init__"]).parameters
        }
        build_keywords = set(inspect.signature(trt_backend.engine_from_network).parameters)
        assert "load_timing_cache" in captured["config"], "the exporter no longer loads the cache it was given"
        assert set(captured["config"]) <= create_config_keywords, "CreateConfig does not accept every keyword passed"
        assert set(captured["build"]) <= build_keywords, "engine_from_network does not accept every keyword passed"

    @pytest.mark.parametrize(
        "value", [pytest.param("", id="empty"), pytest.param(b"engine.cache", id="bytes"), 3, True]
    )
    def test_a_value_that_is_not_a_path_is_refused(self, value: object) -> None:
        """A non-path value is refused when the exporter is built, before any work on the model, and says why."""
        with pytest.raises(ValueError, match="trt_timing_cache must be a non-empty file path"):
            TensorRTExporter(TensorRTConfig(timing_cache=value))

    def test_a_missing_directory_is_created(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Polygraphy opens the lock file beside the cache before it creates the directory, so the export does it."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "not" / "yet" / "engine.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert cache.parent.is_dir()
        assert captured["build"] == {"save_timing_cache": str(cache)}

    def test_a_directory_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A directory cannot hold the cache, and Polygraphy would only say so after the whole build."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=tmp_path))

        with pytest.raises(ValueError, match=r"trt_timing_cache.*directory"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    def test_a_location_that_cannot_be_created_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A file where a directory is needed cannot be worked around by any user, on any operating system."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=blocker / "engine.cache"))

        with pytest.raises(OSError, match="trt_timing_cache"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    def test_a_path_ending_in_a_separator_is_refused_when_the_exporter_is_built(self, tmp_path: Path) -> None:
        """A path ending in a separator is refused when the exporter is built, before any work on the model.

        ``out/cache/`` names a directory, and that is plain from the value alone, so no filesystem access is needed.
        """
        with pytest.raises(ValueError, match=r"trt_timing_cache.*names a directory"):
            TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=str(tmp_path / "cache") + os.sep))

    @pytest.mark.skipif(os.name == "nt", reason="creating a symbolic link needs extra privileges on Windows")
    def test_a_link_to_a_missing_file_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A symbolic link to a missing file is refused before the build, not after it.

        Polygraphy would create the file at the link's target after the build, and a missing target folder would lose
        the timings.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        link = tmp_path / "engine.cache"
        link.symlink_to(tmp_path / "unmounted" / "engine.cache")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=link))

        with pytest.raises(OSError, match=r"trt_timing_cache.*symbolic link"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_a_read_only_cache_file_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The build would save into the file after the engine exists, so a file it cannot write is refused now."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        cache.chmod(0o444)
        if os.access(cache, os.W_OK):
            pytest.skip("this user can write a read-only file (root)")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache))

        with pytest.raises(OSError, match="trt_timing_cache"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_refusing_a_read_only_cache_file_leaves_no_lock_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A refused request must not leave behind a lock file that would then belong to the user who was refused."""
        _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        cache.chmod(0o444)
        if os.access(cache, os.W_OK):
            pytest.skip("this user can write a read-only file (root)")

        with pytest.raises(OSError, match="trt_timing_cache"):
            TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert sorted(path.name for path in tmp_path.iterdir()) == ["engine.cache"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_a_read_only_directory_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The most common bad location: a folder the user cannot write, with no cache in it yet."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        folder = tmp_path / "shared"
        folder.mkdir()
        folder.chmod(0o555)
        if os.access(folder, os.W_OK):
            pytest.skip("this user can write a read-only directory (root)")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=folder / "engine.cache"))

        try:
            with pytest.raises(OSError, match="trt_timing_cache"):
                exporter.build_engine("model.onnx")
        finally:
            folder.chmod(0o755)

        assert captured == {"config": {}, "build": {}}

    def test_the_lock_file_polygraphy_needs_is_created_beside_the_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Polygraphy opens ``<file>.lock`` before it writes the cache, so the export opens it first."""
        _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert (tmp_path / "engine.cache.lock").is_file()

    def test_a_dry_run_touches_nothing(self, tmp_path: Path) -> None:
        """A dry run builds nothing, so it creates neither the cache directory nor a lock file."""
        cache = tmp_path / "not" / "yet" / "engine.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx", dry_run=True)

        assert not (tmp_path / "not").exists()

    def test_a_bad_location_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_convert`` checks the location first, so a doomed request does not wait for the ONNX stage."""
        _patch_polygraphy_chain_recording(monkeypatch)
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or str(tmp_path / "model.onnx"),
        )

        with pytest.raises(ValueError, match="directory"):
            TensorRTExporter(TensorRTConfig(timing_cache=tmp_path))._convert(_minimal_export_graph())

        assert onnx_calls == []

    def test_the_home_directory_is_expanded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A shell expands ``~`` but Python does not; left alone it would create a directory named ``~``."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        # Should the expansion break, the relative "~/..." must land in the test's folder, not the repository.
        monkeypatch.chdir(tmp_path)

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache="~/engine.cache")).build_engine("model.onnx")

        assert Path(captured["build"]["save_timing_cache"]) == tmp_path / "engine.cache"

    def test_a_relative_path_is_relative_to_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The most natural call, ``trt_timing_cache="rfdetr.cache"``, puts the cache where the command runs."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.chdir(tmp_path)

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache="engine.cache")).build_engine("out/model.onnx")

        assert (Path(captured["build"]["save_timing_cache"]), (tmp_path / "engine.cache.lock").is_file()) == (
            tmp_path / "engine.cache",
            True,
        )


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


def _tensorrt_with_hardware_levels(version: str, **levels: str) -> types.ModuleType:
    """Build a stand-in ``tensorrt`` module whose ``HardwareCompatibilityLevel`` has exactly the given members.

    Each member is a distinct string, so a wrong mapping from the ``export`` keyword to the enum member fails an
    equality check instead of passing by coincidence.

    Args:
        version: Value to expose as ``tensorrt.__version__``.
        **levels: Enum member name to the sentinel value it carries.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "tensorrt", ...)``.

    Examples:
        >>> module = _tensorrt_with_hardware_levels("10.16.1.11", AMPERE_PLUS="ampere-plus-level")
        >>> module.HardwareCompatibilityLevel.AMPERE_PLUS
        'ampere-plus-level'
        >>> hasattr(module.HardwareCompatibilityLevel, "SAME_COMPUTE_CAPABILITY")
        False
    """
    module = _fake_tensorrt(version, has_fp16_flag=True)
    module.HardwareCompatibilityLevel = types.SimpleNamespace(**levels)
    return module


def _literal_spellings(annotation: object) -> set[str]:
    """Return the string values of a ``Literal[...] | None`` annotation, ignoring the ``None``.

    Args:
        annotation: A resolved annotation such as ``Literal["a", "b"] | None``.

    Returns:
        The values of the ``Literal`` member.

    Examples:
        >>> from typing import Literal
        >>> sorted(_literal_spellings(Literal["a", "b"] | None))
        ['a', 'b']
    """
    return {value for member in get_args(annotation) if member is not type(None) for value in get_args(member)}


def _cannot_load(name: str) -> None:
    """Stand in for the dynamic loader failing to find the library *name*.

    Args:
        name: The library file name the loader was asked for.

    Raises:
        OSError: Always, naming the library.

    Examples:
        >>> _cannot_load("libnvinfer_lean.so.11")
        Traceback (most recent call last):
        ...
        OSError: libnvinfer_lean.so.11: cannot open shared object file
    """
    raise OSError(f"{name}: cannot open shared object file")


class TestPortableEngines:
    """The compatibility settings reach Polygraphy's config, and nothing is passed unless one is set."""

    @pytest.fixture(autouse=True)
    def lean_runtime(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Pretend the lean runtime library loads, and record the library names TensorRT was asked for.

        The pip package is replaced too: importing the real one while ``CDLL`` is faked would leave it in
        ``sys.modules`` without its libraries, and a real version-compatible build later in the same session would then
        be refused.
        """
        loaded: list[str] = []
        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", loaded.append)
        monkeypatch.setitem(sys.modules, "tensorrt_lean_libs", types.ModuleType("tensorrt_lean_libs"))
        return loaded

    def test_no_compatibility_keyword_is_passed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without either setting the build calls Polygraphy exactly as it always did."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

        assert captured["config"] == {"fp16": False}

    @pytest.mark.parametrize(
        ("level", "expected"), [("ampere_plus", "ampere-plus-level"), ("same_compute_capability", "same-cc-level")]
    )
    def test_hardware_compatibility_selects_the_matching_enum_member(
        self, monkeypatch: pytest.MonkeyPatch, level: str, expected: str
    ) -> None:
        """Each keyword value maps to its own ``HardwareCompatibilityLevel`` member."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "tensorrt",
            _tensorrt_with_hardware_levels(
                "11.3.0.99", AMPERE_PLUS="ampere-plus-level", SAME_COMPUTE_CAPABILITY="same-cc-level"
            ),
        )

        TensorRTExporter(TensorRTConfig(fp16=False, hardware_compatibility=level)).build_engine("model.onnx")

        assert captured["config"]["hardware_compatibility_level"] == expected

    def test_version_compatible_reaches_the_builder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``version_compatible=True`` is passed to ``CreateConfig``; the default passes no such keyword."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        assert captured["config"] == {"fp16": False, "version_compatible": True}

    def test_the_installed_polygraphy_create_config_accepts_the_portability_keywords(self) -> None:
        """The real ``CreateConfig`` still takes the two keywords the exporter passes for portable engines.

        The unit tests above record whatever keyword the exporter passes, so a Polygraphy release that renamed one would
        go unnoticed there; this reads the installed signature instead.
        """
        create_config = pytest.importorskip("polygraphy.backend.trt").CreateConfig

        parameters = inspect.signature(create_config).parameters

        assert {"version_compatible", "hardware_compatibility_level"} <= parameters.keys()

    @pytest.mark.parametrize("dynamic_batch", [False, True])
    def test_both_settings_reach_the_builder_next_to_the_profile(
        self, monkeypatch: pytest.MonkeyPatch, dynamic_batch: bool
    ) -> None:
        """The compatibility keywords travel with the batch profile, for a static and a dynamic build alike."""
        shape = (-1, 3, 384, 384) if dynamic_batch else (1, 3, 384, 384)
        captured = _patch_polygraphy_chain_recording(monkeypatch, _FakeNetwork(_FakeNetworkInput("input", shape)))
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )
        config = TensorRTConfig(
            fp16=False,
            dynamic_batch=dynamic_batch,
            max_batch_size=4,
            hardware_compatibility="ampere_plus",
            version_compatible=True,
        )

        TensorRTExporter(config).build_engine("model.onnx")

        assert captured["config"]["hardware_compatibility_level"] == "ampere-plus-level"
        assert captured["config"]["version_compatible"] is True
        assert ("profiles" in captured["config"]) is dynamic_batch

    def test_a_tensorrt_without_the_level_refuses_before_building(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A TensorRT that lacks the requested level names it and its own version, and builds nothing."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("10.16.1.11", AMPERE_PLUS="ampere-plus-level")
        )
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, hardware_compatibility="same_compute_capability"))

        with pytest.raises(ValueError, match=r"same_compute_capability.*TensorRT 10\.16\.1\.11"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    def test_ampere_plus_is_refused_on_a_pre_ampere_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A GPU older than Ampere cannot build ``ampere_plus``; the refusal names its compute capability.

        TensorRT itself fails only inside the build, after the forward pass and the ONNX export; the check reads the
        current CUDA device first, so a Turing card (7.5) is turned away before any of that.
        """
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

        with pytest.raises(ValueError, match=r"ampere_plus.*compute capability 7\.5"):
            TensorRTExporter(TensorRTConfig(hardware_compatibility="ampere_plus")).check_environment()

    @pytest.mark.parametrize(
        ("level", "capability"),
        [
            pytest.param("ampere_plus", (8, 0), id="ampere_plus-on-ampere"),
            pytest.param("ampere_plus", (12, 0), id="ampere_plus-on-blackwell"),
            pytest.param("same_compute_capability", (7, 5), id="same_compute_capability-on-turing"),
        ],
    )
    def test_the_gpu_check_passes_where_the_level_can_be_built(
        self, monkeypatch: pytest.MonkeyPatch, level: str, capability: tuple[int, int]
    ) -> None:
        """Ampere and newer pass ``ampere_plus``, and ``same_compute_capability`` has no minimum GPU at all."""
        monkeypatch.setitem(
            sys.modules,
            "tensorrt",
            _tensorrt_with_hardware_levels(
                "11.3.0.99", AMPERE_PLUS="ampere-plus-level", SAME_COMPUTE_CAPABILITY="same-cc-level"
            ),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)

        TensorRTExporter(TensorRTConfig(hardware_compatibility=level)).check_environment()

    def test_ampere_plus_is_not_checked_against_a_gpu_on_a_host_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a visible CUDA device there is no capability to compare, so the request is let through."""
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "get_device_capability", _cannot_load)

        TensorRTExporter(TensorRTConfig(hardware_compatibility="ampere_plus")).check_environment()

    @pytest.mark.parametrize("level", [pytest.param("", id="empty"), "AMPERE_PLUS", "ampere", "none", 3, True])
    def test_an_unknown_hardware_level_is_refused(self, level: object) -> None:
        """Only the documented spellings are accepted, and the refusal lists them, before any work on the model."""
        with pytest.raises(ValueError, match=r"hardware_compatibility.*ampere_plus.*same_compute_capability"):
            TensorRTExporter(TensorRTConfig(hardware_compatibility=level))

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_version_compatible_must_be_a_bool(self, value: object) -> None:
        """A truthy non-``bool`` would enable the mode by accident, so it is refused."""
        with pytest.raises(ValueError, match="version_compatible"):
            TensorRTExporter(TensorRTConfig(version_compatible=value))

    @pytest.mark.parametrize(
        ("major", "platform", "expected"),
        [
            (10, "linux", "libnvinfer_lean.so.10"),
            (10, "win32", "nvinfer_lean_10.dll"),
            (8, "win32", "nvinfer_lean.dll"),
        ],
    )
    def test_the_lean_library_name_follows_the_platform_and_major_version(
        self, major: int, platform: str, expected: str
    ) -> None:
        """The runtime library is named by TensorRT's major version, and differently on Windows (without it on 8.6)."""
        assert tensorrt_export._lean_library_name(major, platform) == expected

    def test_on_windows_the_lean_library_is_looked_up_on_path(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str]
    ) -> None:
        """A bare DLL name is not searched on PATH, where a TensorRT zip install puts the library; the full path is."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setattr(tensorrt_export.sys, "platform", "win32")
        found = "C:/TensorRT/lib/nvinfer_lean_10.dll"
        monkeypatch.setattr(tensorrt_export.ctypes.util, "find_library", lambda name: found)

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        assert lean_runtime == [found]

    @pytest.mark.parametrize("level", [None, "ampere_plus"])
    def test_the_lean_runtime_is_not_probed_unless_version_compatible_is_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str], level: str | None
    ) -> None:
        """Only a version-compatible build embeds the lean runtime; a hardware-compatible one must not need it."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )

        TensorRTExporter(TensorRTConfig(fp16=False, hardware_compatibility=level)).build_engine("model.onnx")

        assert lean_runtime == []

    def test_an_unparseable_tensorrt_version_skips_the_lean_probe(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str]
    ) -> None:
        """The library is named by the major version; without one there is no name to look for, and the build goes
        on."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("unknown", has_fp16_flag=True))

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        assert lean_runtime == []

    def test_version_compatibility_warns_before_the_forward_pass_where_it_was_not_seen_to_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A TensorRT 10 request is warned about from ``check_environment``, which runs before the model does.

        It was verified between TensorRT 11 releases only; warning from the build instead would reach the user only
        after the forward pass and the ONNX export had already been paid for.
        """
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))

        with pytest.warns(UserWarning, match=r"verified between TensorRT 11.*TensorRT 10\.16\.1\.11"):
            TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).check_environment()

    def test_version_compatibility_does_not_warn_on_tensorrt_11(
        self, monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Between TensorRT 11 releases the mode was seen to work, so the request goes through without a warning."""
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.3.0.99", has_fp16_flag=True))

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).check_environment()

        assert [str(w.message) for w in recwarn if "verified between TensorRT 11" in str(w.message)] == []

    def test_the_lean_runtime_package_is_imported_before_the_library_is_probed(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str]
    ) -> None:
        """The pip package loads its libraries when imported, so it has to come first or the probe would miss it."""
        _patch_polygraphy_chain_recording(monkeypatch)
        events: list[str] = []
        real_import = tensorrt_export.importlib.import_module

        def _import(name: str, *args: object, **kwargs: object) -> object:
            events.append(f"import {name}")
            return real_import(name, *args, **kwargs) if name != "tensorrt_lean_libs" else types.ModuleType(name)

        monkeypatch.setattr(tensorrt_export.importlib, "import_module", _import)
        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", lambda name: events.append(f"load {name}"))

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        expected = tensorrt_export._lean_library_name(10, sys.platform)
        # Other imports may pass through the patched ``import_module``, so only the two relevant events are compared.
        relevant = [event for event in events if event in ("import tensorrt_lean_libs", f"load {expected}")]
        assert relevant == ["import tensorrt_lean_libs", f"load {expected}"]

    def test_a_missing_lean_runtime_is_refused_with_the_install_hint_and_builds_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the library TensorRT fails with a bare "Invalid Engine"; the refusal names the package instead."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", _cannot_load)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True))

        with pytest.raises(ImportError, match=r"lean runtime.*tensorrt-lean-cu\*-libs") as refusal:
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}
        assert isinstance(refusal.value.__cause__, OSError), "the loader's own error stays attached"

    def test_a_missing_lean_runtime_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``Exporter.__call__`` checks first, so the user is not refused only after the ONNX stage has run."""
        _patch_polygraphy_chain_recording(monkeypatch)
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or str(tmp_path / "model.onnx"),
        )

        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", _cannot_load)

        with pytest.raises(ImportError, match="lean runtime"):
            TensorRTExporter(TensorRTConfig(version_compatible=True))(_minimal_export_graph())

        assert onnx_calls == []

    def test_a_missing_hardware_level_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A TensorRT without the requested level is found out first, not after the ONNX stage has run."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("10.16.1.11", AMPERE_PLUS="ampere-plus-level")
        )
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or str(tmp_path / "model.onnx"),
        )
        config = TensorRTConfig(hardware_compatibility="same_compute_capability")

        with pytest.raises(ValueError, match="same_compute_capability"):
            TensorRTExporter(config)(_minimal_export_graph())

        assert onnx_calls == []

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda exporter: exporter(_minimal_export_graph()), id="call"),
            pytest.param(lambda exporter: exporter.build_engine("model.onnx"), id="build_engine"),
        ],
    )
    def test_a_host_without_tensorrt_is_named_before_the_portability_check(
        self, monkeypatch: pytest.MonkeyPatch, run: Callable[[TensorRTExporter], object]
    ) -> None:
        """The packages are checked before the configuration, whose lean probe would import the missing TensorRT."""
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "tensorrt", None)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            run(TensorRTExporter(TensorRTConfig(version_compatible=True)))

    @pytest.mark.parametrize(
        ("options", "suffix"),
        [
            pytest.param({}, "", id="default"),
            pytest.param({"hardware_compatibility": "ampere_plus"}, "_ampere_plus", id="ampere-plus"),
            pytest.param(
                {"hardware_compatibility": "same_compute_capability"}, "_same_compute_capability", id="same-cc"
            ),
            pytest.param({"version_compatible": True}, "_version_compatible", id="version-compatible"),
            pytest.param(
                {"hardware_compatibility": "ampere_plus", "version_compatible": True},
                "_ampere_plus_version_compatible",
                id="both",
            ),
        ],
    )
    def test_a_portable_engine_gets_its_own_file_name(self, options: dict, suffix: str) -> None:
        """A portable engine must not overwrite (or be overwritten by) a default one built in the same directory.

        The default name stays exactly what it was; each option that is on adds a detail to it.
        """
        path = TensorRTExporter(TensorRTConfig(fp16=True, **options)).build_engine("out/model.onnx", dry_run=True)

        assert path == f"out/model_fp16{suffix}.trt"

    def test_an_fp32_portable_engine_keeps_its_detail_too(self) -> None:
        """The detail follows the precision, whichever it is."""
        config = TensorRTConfig(fp16=False, hardware_compatibility="ampere_plus")

        assert TensorRTExporter(config).build_engine("out/model.onnx", dry_run=True) == "out/model_fp32_ampere_plus.trt"

    def test_a_missing_level_is_refused_before_the_fp16_graph_is_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``build_engine`` checks the level first, so a doomed FP16 build on a strongly typed TensorRT casts
        nothing."""
        _patch_polygraphy_chain_recording(monkeypatch)
        strongly_typed = _fake_tensorrt("11.3.0.99", has_fp16_flag=False)
        strongly_typed.HardwareCompatibilityLevel = types.SimpleNamespace(AMPERE_PLUS="ampere-plus-level")
        monkeypatch.setitem(sys.modules, "tensorrt", strongly_typed)
        casts: list[str] = []
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: casts.append(path) or path)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, hardware_compatibility="same_compute_capability"))

        with pytest.raises(ValueError, match="same_compute_capability"):
            exporter.build_engine("model.onnx")

        assert casts == []

    def test_a_custom_output_name_is_used_verbatim_for_a_portable_engine(self) -> None:
        """``output_name`` already says what the file is, so no detail is appended to it."""
        config = TensorRTConfig(fp16=True, version_compatible=True, output_name="mine")

        assert TensorRTExporter(config).build_engine("out/model.onnx", dry_run=True) == "out/mine.trt"

    def test_the_export_keyword_lists_the_levels_the_validator_accepts(self) -> None:
        """``RFDETR.export`` types the levels by hand; they must not drift from the spellings the exporter accepts."""
        spellings = set(tensorrt_export._HARDWARE_COMPATIBILITY_LEVELS)

        assert _literal_spellings(get_type_hints(RFDETR.export)["trt_hardware_compatibility"]) == spellings

    def test_the_settings_are_read_from_the_export_keywords(self) -> None:
        """``RFDETR.export``'s ``trt_``-prefixed keywords reach the configuration."""
        config = TensorRTExporter.build_config(trt_hardware_compatibility="ampere_plus", trt_version_compatible=True)

        assert (config.hardware_compatibility, config.version_compatible) == ("ampere_plus", True)

    def test_the_settings_are_off_by_default(self) -> None:
        """Without the keywords the configuration asks for no portability."""
        config = TensorRTExporter.build_config()

        assert (config.hardware_compatibility, config.version_compatible) == (None, False)


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
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(cast_path))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert build_args["source"] == str(cast_path)

    def test_engine_name_reports_fp16(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Regression guard for #1453: the engine really is FP16, so the filename must not say ``_fp32``."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(tmp_path / "model.fp16.onnx"))

        result = TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert result == str(tmp_path / "model_fp16.trt")

    def test_builder_flag_is_not_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Polygraphy aborts if asked for a flag TensorRT 11 removed, so the config must request FP32."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(tmp_path / "model.fp16.onnx"))

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
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(cast_path))

        TensorRTExporter(TensorRTConfig(fp16=True)).build_engine(str(tmp_path / "model.onnx"))

        assert build_args["source"] == str(cast_path)

    def test_a_surviving_fp16_flag_is_not_requested(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A strongly typed builder reads precision off the graph, so the flag must stay unrequested."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(tmp_path / "model.fp16.onnx"))

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
            save_fails: Whether ``save_file`` should raise, simulating a failed build.

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
        monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
        monkeypatch.setattr(
            tensorrt_export, "network_from_onnx_path", lambda path: ("builder", _FakeNetwork(_STATIC_INPUT), "parser")
        )
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: _FAKE_ENGINE)

        def _save_file(contents, dest, description=None):
            if save_fails:
                raise RuntimeError("builder ran out of workspace")

        monkeypatch.setattr(tensorrt_export, "save_file", _save_file)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(cast_path))
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

    def test_a_static_request_refuses_a_dynamic_graph_before_writing(self, tmp_path: Path) -> None:
        """A graph the engine request contradicts is refused before the cast writes anything.

        On a strongly typed TensorRT (the default ``fp16=True`` path) the parsed network is the first thing that used to
        notice this mismatch — after the cast had converted every weight and written a second copy of the model the size
        of the original. The file on disk already answers the question, so the refusal comes first and the directory
        stays clean.
        """
        source = tmp_path / "dynamic.onnx"
        onnx.save(_float32_model_with_dynamic_batch(), source)

        with pytest.raises(ValueError, match="dynamic batch axis"):
            tensorrt_export._cast_onnx_to_fp16(str(source), dynamic_batch=False)

        assert list(tmp_path.glob("*.fp16-*.onnx")) == []

    def test_a_dynamic_request_casts_a_dynamic_graph(self, tmp_path: Path) -> None:
        """The refusal is about the two disagreeing, so the documented ``dynamic_batch=True`` build still casts."""
        source = tmp_path / "dynamic.onnx"
        onnx.save(_float32_model_with_dynamic_batch(), source)

        cast_path = tensorrt_export._cast_onnx_to_fp16(str(source), dynamic_batch=True)

        assert Path(cast_path).is_file()

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
    build_succeeds: bool = True,
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
        build_succeeds: Whether ``Builder.build_serialized_network`` returns an engine; TensorRT reports a failed
            build (for instance a dynamic-batch network with no optimization profile) by returning ``None``.

    Returns:
        A module object suitable for ``monkeypatch.setattr(inference, "trt", ...)``.

    Examples:
        >>> module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, has_explicit_batch=False)
        >>> hasattr(module.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
        False
        >>> module.record["flags_set"]
        []
        >>> failing = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, build_succeeds=False)
        >>> failing.Builder("logger").build_serialized_network(None, None) is None
        True
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

        def build_serialized_network(self, network: object, config: object) -> bytes | None:
            return b"serialized-engine" if build_succeeds else None

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
        Whatever ``build_engine`` returned — the serialized engine. Raises ``RuntimeError`` when parsing or
        building fails; it never returns ``None``.

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
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: str(cast_path))
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
        """A parse failure raises rather than returning silently, and must still not leak the intermediate."""
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, parse_succeeds=False)

        with pytest.raises(RuntimeError, match="could not parse the ONNX file"):
            _run_benchmark_build(monkeypatch, tmp_path, trt_module, cast_path)

        assert not cast_path.exists()

    def test_a_failed_parse_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An unparsable ONNX raises ``RuntimeError`` rather than returning ``None`` for the caller to go on and use.

        Before this, a parse failure logged and returned ``None`` -- the sibling build failure already raised, so a
        caller checking only for an exception would proceed with ``None`` as if it were an engine.
        """
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, parse_succeeds=False)

        with pytest.raises(RuntimeError, match="could not parse the ONNX file"):
            _run_benchmark_build(monkeypatch, tmp_path, trt_module, tmp_path / "model.fp16-abcd1234.onnx")

    def test_a_failed_build_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A build TensorRT refused is an error, not a ``TypeError`` from ``f.write(None)``.

        The error names the caller's own model: on a strongly typed TensorRT the parser read the cast intermediate,
        whose ``model.fp16-<suffix>.onnx`` name the caller never chose.
        """
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, build_succeeds=False)

        with pytest.raises(RuntimeError, match=r"could not build an engine from '[^']*model\.onnx'"):
            _run_benchmark_build(monkeypatch, tmp_path, trt_module, tmp_path / "model.fp16-abcd1234.onnx")

    @pytest.mark.parametrize("previous_engine", [None, b"engine from an earlier build"])
    def test_a_failed_build_leaves_the_engine_path_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, previous_engine: bytes | None
    ) -> None:
        """Nothing is written for a failed build: no empty ``.trt`` appears, and an earlier engine is not truncated.

        Scenario: a dynamic-batch ONNX, which this builder cannot build because it declares no optimization profile.
        The target used to be opened for writing before the result was checked, so a failure left a zero-byte file
        where a previous engine may have been.
        """
        engine_path = tmp_path / "model.trt"
        if previous_engine is not None:
            engine_path.write_bytes(previous_engine)
        trt_module = _fake_benchmark_tensorrt("10.16.1.11", has_fp16_flag=True, build_succeeds=False)

        with pytest.raises(RuntimeError):
            _run_benchmark_build(monkeypatch, tmp_path, trt_module, None)

        assert (engine_path.read_bytes() if engine_path.exists() else None) == previous_engine

    def test_cast_graph_is_removed_after_a_failed_build(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The error raised for a failed build still unwinds the cast intermediate beside the user's model."""
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        trt_module = _fake_benchmark_tensorrt("11.2.1.2", has_fp16_flag=False, build_succeeds=False)

        with pytest.raises(RuntimeError):
            _run_benchmark_build(monkeypatch, tmp_path, trt_module, cast_path)

        assert not cast_path.exists()


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

    @pytest.fixture(
        scope="class",
        params=[
            pytest.param((False, False), id="static-fp32"),
            pytest.param((True, False), id="dynamic-fp32"),
            pytest.param((False, True), id="static-fp16"),
            pytest.param((True, True), id="dynamic-fp16"),
        ],
    )
    def trt_sidecar_engine(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
        """Export RFDETRNano to a TensorRT engine with ``trt_metadata=True`` and deserialize the engine it wrote.

        Returns:
            A namespace with the parsed ``sidecar``, the ``engine_path``, the deserialized ``engine`` (and the
            ``runtime`` that must outlive it), the model's ``resolution``, and the ``dynamic_batch`` and ``fp16``
            settings of this parameter.
        """
        import tensorrt as trt

        from rfdetr import RFDETRNano

        dynamic_batch, fp16 = request.param
        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt_sidecar")
        detector = RFDETRNano(pretrain_weights=None)
        engine_path = detector.export(
            output_dir=str(out_dir),
            format="tensorrt",
            fp16=fp16,
            verbose=False,
            trt_metadata=True,
            dynamic_batch=dynamic_batch,
            batch_size=2,
            max_batch_size=4 if dynamic_batch else None,
        )
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        return types.SimpleNamespace(
            sidecar=json.loads(engine_path.with_suffix(".json").read_text()),
            engine_path=engine_path,
            engine=engine,
            runtime=runtime,
            resolution=int(detector.model.resolution),
            dynamic_batch=dynamic_batch,
            fp16=fp16,
        )

    @staticmethod
    def _tensor_names(engine: Any) -> tuple[list[str], list[str]]:
        """Return the deserialized engine's input tensor names and output tensor names, in binding order.

        Examples:
            Needs a real engine on a GPU, so this is documentation only (not a doctest):

            >>> TestTensorRTEndToEnd._tensor_names(engine)  # doctest: +SKIP
            (['input'], ['dets', 'labels'])
        """
        import tensorrt as trt

        modes = {name: engine.get_tensor_mode(name) for name in engine}
        return (
            [name for name, mode in modes.items() if mode == trt.TensorIOMode.INPUT],
            [name for name, mode in modes.items() if mode == trt.TensorIOMode.OUTPUT],
        )

    def test_the_sidecar_input_is_the_engines_input(self, trt_sidecar_engine: Any) -> None:
        """The input's name and the spatial size match what the engine expects (a static engine's shape, a dynamic one's
        profile)."""
        sidecar, engine = trt_sidecar_engine.sidecar, trt_sidecar_engine.engine
        (engine_input,), _ = self._tensor_names(engine)

        assert sidecar["input"]["name"] == engine_input
        resolution = trt_sidecar_engine.resolution
        assert (sidecar["input"]["height"], sidecar["input"]["width"]) == (resolution, resolution)
        assert tuple(engine.get_tensor_shape(engine_input)[1:]) == (3, resolution, resolution)

    def test_the_sidecar_outputs_are_the_engines_outputs_in_order_and_float32(self, trt_sidecar_engine: Any) -> None:
        """Names and order match the engine's outputs, and every tensor is FP32, also for an FP16 build."""
        import tensorrt as trt

        engine = trt_sidecar_engine.engine
        (engine_input,), engine_outputs = self._tensor_names(engine)

        assert [output["name"] for output in trt_sidecar_engine.sidecar["outputs"]] == engine_outputs
        assert {engine.get_tensor_dtype(name) for name in (engine_input, *engine_outputs)} == {trt.float32}

    def test_the_sidecar_batch_is_the_engines_batch(self, trt_sidecar_engine: Any) -> None:
        """A dynamic engine's min/opt/max equal its optimization profile; a static one's size is its batch axis."""
        sidecar, engine = trt_sidecar_engine.sidecar, trt_sidecar_engine.engine
        (engine_input,), _ = self._tensor_names(engine)

        if trt_sidecar_engine.dynamic_batch:
            profile = engine.get_tensor_profile_shape(engine_input, 0)
            assert sidecar["batch"] == {"dynamic": True, "min": 1, "opt": 2, "max": 4}
            assert [shape[0] for shape in profile] == [1, 2, 4]
        else:
            assert sidecar["batch"] == {"dynamic": False, "size": 2}
            assert engine.get_tensor_shape(engine_input)[0] == 2

    def test_the_sidecar_identifies_the_engine_file(self, trt_sidecar_engine: Any) -> None:
        """The recorded size and SHA-256 are those of the ``.trt`` the export returned, so a consumer's check passes."""
        engine_bytes = trt_sidecar_engine.engine_path.read_bytes()

        assert trt_sidecar_engine.sidecar["engine"] == {
            "size": len(engine_bytes),
            "sha256": hashlib.sha256(engine_bytes).hexdigest(),
        }

    def test_the_sidecar_records_the_build_that_ran(self, trt_sidecar_engine: Any) -> None:
        """The precision requested, the TensorRT version and the GPU the test itself sees.

        The last two are read from the same sources the exporter reads, so they check the plumbing, not TensorRT. The
        precision equals the request because a full TensorRT wheel never takes the lean-wheel fallback.
        """
        import tensorrt as trt

        build = trt_sidecar_engine.sidecar["build"]
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())

        assert build["precision"] == ("fp16" if trt_sidecar_engine.fp16 else "fp32")
        assert build["tensorrt_version"] == trt.__version__
        assert build["gpu"] == {"name": properties.name, "compute_capability": f"{properties.major}.{properties.minor}"}

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

    @staticmethod
    def _polygraphy_outputs(engine_path: str | Path, example: torch.Tensor) -> dict:
        """Run one engine through Polygraphy on *example* and return its outputs as NumPy arrays.

        Polygraphy's loader always allows host code, so this also opens a version-compatible engine.

        Args:
            engine_path: The serialized engine.
            example: The input batch, on any device.

        Returns:
            Each output's name mapped to its values.

        Examples:
            Needs a real engine on a GPU, so this is documentation only (not a doctest):

            >>> TestTensorRTEndToEnd._polygraphy_outputs("model.trt", example)  # doctest: +SKIP
            {'dets': array(...), 'labels': array(...)}
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        with TrtRunner(EngineFromBytes(BytesFromPath(str(engine_path)))) as runner:
            return {name: np.array(value) for name, value in runner.infer(feed_dict=feed).items()}

    @pytest.mark.parametrize("level", ["ampere_plus", "same_compute_capability"])
    def test_hardware_compatible_engine_records_its_level_and_matches_the_default_engine(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], level: str
    ) -> None:
        """A hardware-compatible engine is tagged with the requested level and computes what the default one does.

        ``AMPERE_PLUS`` needs compute capability 8.0 or newer, so it always skips on a T4 (7.5). That skip is the only
        one the TensorRT CI job whitelists. Loading an engine on a different GPU or TensorRT release than built it is
        not CI-verified: this test builds and loads on one device with one release.
        """
        import numpy as np
        import tensorrt as trt

        if not hasattr(trt.HardwareCompatibilityLevel, level.upper()):
            pytest.skip(f"this TensorRT has no HardwareCompatibilityLevel.{level.upper()}")
        if level == "ampere_plus" and torch.cuda.get_device_capability() < (8, 0):
            pytest.skip("AMPERE_PLUS engines need compute capability 8.0 or newer")
        _, example, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        config = TensorRTConfig(fp16=False, verbose=False, hardware_compatibility=level)

        portable_path = TensorRTExporter(config).build_engine(str(onnx_path), output_name=f"hardware-{level}")

        with open(portable_path, "rb") as engine_file:
            engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(engine_file.read())
        assert engine.hardware_compatibility_level == getattr(trt.HardwareCompatibilityLevel, level.upper())
        default, portable = (self._polygraphy_outputs(path, example) for path in (engine_path, portable_path))
        diffs = {name: float(np.abs(default[name] - portable[name]).max()) for name in ("dets", "labels")}
        assert max(diffs.values()) < _TENSORRT_MAX_ABS_DIFF, (
            f"{level} engine differs from the default engine's: {diffs}"
        )

    @pytest.fixture(scope="class")
    def trt_version_compatible_engine(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> Path:
        """Build a version-compatible engine from the FP32 engine's ONNX, or skip when TensorRT has no lean runtime.

        The lean runtime is a separate package (``tensorrt-lean-cu*-libs``) that the ``rfdetr[tensorrt]`` extra does not
        install, so this is skipped wherever it is missing. CI's TensorRT job installs it and fails if it does not load.
        """
        _, _, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False, version_compatible=True))
        try:
            exporter.check_environment()
        except ImportError as error:
            pytest.skip(str(error))
        return Path(exporter.build_engine(str(onnx_path), output_name="version-compatible"))

    def test_a_version_compatible_engine_is_refused_by_trt_inference_without_host_code_on_tensorrt_11(
        self, trt_version_compatible_engine: Path
    ) -> None:
        """TensorRT 11 will not deserialize the host code such an engine carries unless the caller says it is trusted.

        TensorRT 10.16 loads its own version-compatible engine without the opt-in, so the refusal is asserted only where
        it exists.
        """
        import tensorrt as trt

        if int(trt.__version__.split(".")[0]) < 11:
            pytest.skip("TensorRT 10 loads a version-compatible engine without the host-code opt-in")

        with pytest.raises(RuntimeError, match="engine_host_code_allowed=True"):
            tensorrt_inference.TRTInference(str(trt_version_compatible_engine), device="cuda:0", sync_mode=True)

    def test_a_version_compatible_engine_computes_what_the_default_engine_does_once_host_code_is_allowed(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], trt_version_compatible_engine: Path
    ) -> None:
        """With the opt-in ``TRTInference`` runs it, and its outputs agree with the default engine's."""
        import numpy as np

        _, example, engine_path = trt_engine
        runtime = tensorrt_inference.TRTInference(
            str(trt_version_compatible_engine), device="cuda:0", sync_mode=True, engine_host_code_allowed=True
        )
        outputs = runtime({"input": example.to("cuda:0")})

        default = self._polygraphy_outputs(engine_path, example)
        diffs = {
            name: float(np.abs(outputs[name].detach().float().cpu().numpy() - default[name]).max())
            for name in ("dets", "labels")
        }
        assert max(diffs.values()) < _TENSORRT_MAX_ABS_DIFF, (
            f"version-compatible engine differs from the default's: {diffs}"
        )

    @pytest.mark.parametrize("device", ["cuda:0", "cuda"])
    def test_trt_inference_helper_serves_the_dynamic_engine(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], device: str
    ) -> None:
        """``TRTInference`` allocates at the profile's max batch, trims outputs to the batch run, and agrees with
        polygraphy.

        A bare ``"cuda"`` resolves to the current device, so its ``cuda:0`` input must be accepted as the engine's own.
        """
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

        runtime = tensorrt_inference.TRTInference(str(engine_path), device=device, sync_mode=True)
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

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
    def test_trt_inference_runs_on_a_non_default_device(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A runtime constructed for ``cuda:1`` gives the same output as one on ``cuda:0``, not a device no-op.

        Every other device test in this suite runs on a single-GPU host and only asserts which device the
        constructor *asked* torch to make current -- a hardware no-op would pass those too. This is the one
        real-placement check: it needs a second GPU, so it is skipped everywhere except a multi-GPU runner.
        """
        _, resolution, engine_path = trt_dynamic_engine
        example = _distinct_batch(1, resolution)

        reference = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        reference_out = reference({"input": example.to("cuda:0")})

        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:1", sync_mode=True)
        assert runtime.engine_device == torch.device("cuda", 1)
        outputs = runtime({"input": example.to("cuda:1")})

        for name in ("dets", "labels"):
            got = outputs[name].detach().float().cpu().numpy()
            expected = reference_out[name].detach().float().cpu().numpy()
            diff = float(np.abs(got - expected).max())
            assert diff < 1e-4, f"TRTInference on cuda:1 {name} differs from cuda:0: {diff}"

    def test_trt_inference_refuses_a_channels_last_input(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A ``channels_last`` input is refused before launch rather than read as dense NCHW memory.

        Bound by pointer, it took pretrained RF-DETR Nano from 13 detections above 0.5 to none on a real image.
        """
        _, resolution, engine_path = trt_dynamic_engine
        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        example = _distinct_batch(2, resolution).to("cuda:0", memory_format=torch.channels_last)

        with pytest.raises(ValueError, match="not contiguous"):
            runtime({"input": example})

    def test_trt_inference_reports_a_truncated_engine(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], tmp_path: Path
    ) -> None:
        """Real TensorRT returns ``None`` for a truncated engine rather than raising; the runtime reports it as such."""
        _, _, engine_path = trt_dynamic_engine
        truncated = tmp_path / "truncated.trt"
        with open(engine_path, "rb") as engine_file:
            truncated.write_bytes(engine_file.read(4096))

        with pytest.raises(RuntimeError, match="Rebuild"):
            tensorrt_inference.TRTInference(str(truncated), device="cuda:0", sync_mode=True)

    def test_benchmark_build_engine_keeps_the_previous_engine_on_a_failed_build(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], tmp_path: Path
    ) -> None:
        """Real TensorRT refuses a dynamic-batch ONNX in ``TRTInference.build_engine``, which declares no profile.

        It reports that by returning ``None``; the engine already at the target path must survive the failure.
        """
        import tensorrt as trt

        _, _, engine_path = trt_dynamic_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        target = tmp_path / "benchmark.trt"
        target.write_bytes(b"engine from an earlier build")
        runtime_stand_in = types.SimpleNamespace(logger=trt.Logger(trt.Logger.ERROR))

        with pytest.raises(RuntimeError, match="could not build an engine"):
            tensorrt_inference.TRTInference.build_engine(runtime_stand_in, str(onnx_path), str(target))

        assert target.read_bytes() == b"engine from an earlier build"

    def test_warm_timing_cache_builds_an_equivalent_engine(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first build writes the timing cache, and a build that reuses it produces the cold build's outputs.

        Reusing measured timings changes which kernels a build may skip timing, never what the network computes, so on
        an FP32 engine the two engines must agree to numerical noise.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        _, example, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        cache = tmp_path / "engine.cache"
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False, timing_cache=cache))
        # The real ``CreateConfig`` receives the keywords, so Polygraphy itself vouches for their names.
        real_create_config = tensorrt_export.CreateConfig
        create_config_calls: list[dict[str, object]] = []

        def recording_create_config(**kwargs: object) -> object:
            """Record the keywords and build the real configuration."""
            create_config_calls.append(kwargs)
            return real_create_config(**kwargs)

        monkeypatch.setattr(tensorrt_export, "CreateConfig", recording_create_config)

        cold_path = exporter.build_engine(str(onnx_path), output_name="cold-with-cache")
        assert cache.is_file() and cache.stat().st_size > 0, "the first build must write the timing cache"
        warm_path = exporter.build_engine(str(onnx_path), output_name="warm-with-cache")
        assert ["load_timing_cache" in call for call in create_config_calls] == [False, True]
        assert create_config_calls[1]["load_timing_cache"] == str(cache)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        outputs = []
        for path in (cold_path, warm_path):
            with TrtRunner(EngineFromBytes(BytesFromPath(path))) as runner:
                outputs.append({name: np.array(value) for name, value in runner.infer(feed_dict=feed).items()})
        diffs = {name: float(np.abs(outputs[0][name] - outputs[1][name]).max()) for name in ("dets", "labels")}
        assert max(diffs.values()) < 1e-4, f"the warm-cache engine differs from the cold build's: {diffs}"

    @pytest.fixture(scope="class")
    def populated_timing_cache(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], tmp_path_factory: pytest.TempPathFactory
    ) -> bytes:
        """The timing cache a real build writes, built once for the tests that damage it."""
        _, _, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        cache = tmp_path_factory.mktemp("timing-cache") / "engine.cache"
        config = TensorRTConfig(fp16=False, verbose=False, timing_cache=cache)

        TensorRTExporter(config).build_engine(str(onnx_path), output_name="populate-timing-cache")

        return cache.read_bytes()

    @pytest.mark.parametrize("damage", ["empty", "garbage", "truncated"])
    def test_a_damaged_timing_cache_is_replaced_and_does_not_fail_the_build(
        self,
        trt_engine: tuple[torch.nn.Module, torch.Tensor, Path],
        populated_timing_cache: bytes,
        tmp_path: Path,
        damage: str,
    ) -> None:
        """A cache TensorRT cannot read does not stop the build, and the build writes its own timings over it.

        The documented behaviour of ``trt_timing_cache`` for an empty, a garbage and a truncated file (a real cache cut
        in half). TensorRT logs a serialization error for the last two; the export still succeeds.
        """
        _, _, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        damaged = {
            "empty": b"",
            "garbage": b"this is not a timing cache" * 40,
            "truncated": populated_timing_cache[: len(populated_timing_cache) // 2],
        }[damage]
        cache = tmp_path / "engine.cache"
        cache.write_bytes(damaged)

        built = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False, timing_cache=cache)).build_engine(
            str(onnx_path), output_name=f"after-a-{damage}-cache"
        )

        assert Path(built).stat().st_size > 0
        assert len(cache.read_bytes()) > len(damaged), "the build must write a fuller timing cache over the damaged one"


class TestTimingCacheSafety:
    """The timing cache Polygraphy is handed is one well-defined regular file, and a bad one is refused up front."""

    @pytest.mark.skipif(os.name == "nt", reason="creating a symbolic link needs extra privileges on Windows")
    def test_a_link_to_an_existing_cache_hands_polygraphy_its_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A link and the file it points to load, save and lock the same cache file.

        Polygraphy locks ``<path>.lock`` beside the path it is given. Handed the link itself, an export through the link
        and another through the target would each lock their own file and race the same cache's read-merge-write.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        link = tmp_path / "link.cache"
        link.symlink_to(cache)

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=link)).build_engine("model.onnx")

        assert (captured["config"]["load_timing_cache"], captured["build"], sorted(os.listdir(tmp_path))) == (
            str(cache),
            {"save_timing_cache": str(cache)},
            ["engine.cache", "engine.cache.lock", "link.cache"],
        )

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes need os.mkfifo, which only POSIX has")
    def test_a_named_pipe_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A FIFO cannot hold the cache, and it is refused before the build without ever being opened.

        ``os.path.isfile`` is false for a FIFO, so it used to pass as a cache that does not exist yet; Polygraphy's save
        into it would then block, with no reader, after the whole build.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        fifo = tmp_path / "engine.cache"
        os.mkfifo(fifo)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=fifo))

        with pytest.raises(ValueError, match="trt_timing_cache must name a regular file"):
            exporter.build_engine("model.onnx")

        assert (captured, sorted(os.listdir(tmp_path))) == ({"config": {}, "build": {}}, ["engine.cache"])

    def test_a_refused_location_keeps_its_operating_system_error_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The refusal names the setting and still raises the errno's own ``OSError`` subclass.

        A file stands where the cache's directory has to be created, which fails with ``FileExistsError`` on every
        operating system; a caller catching that subclass must not lose it to a plain ``OSError`` re-wrap.
        """
        _patch_polygraphy_chain_recording(monkeypatch)
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=blocker / "engine.cache"))

        with pytest.raises(FileExistsError, match="trt_timing_cache"):
            exporter.build_engine("model.onnx")

    def test_a_path_with_a_nul_byte_is_refused_when_the_exporter_is_built(self) -> None:
        """No file can carry a NUL byte in its name, so the value is refused before any work on the model.

        Left to the filesystem, the NUL byte surfaced as a bare ``ValueError('embedded null byte')`` that named no
        setting.
        """
        with pytest.raises(ValueError, match="trt_timing_cache must not contain a NUL byte"):
            TensorRTExporter(TensorRTConfig(fp16=False, timing_cache="engine\x00.cache"))

    @pytest.mark.parametrize(
        ("cache_name", "has_fp16_flag"),
        [
            pytest.param("model.onnx", True, id="onnx-model"),
            pytest.param("model_fp16.trt", True, id="fp16-engine"),
            pytest.param("model_fp32.trt", False, id="fp32-fallback-engine"),
        ],
    )
    def test_a_cache_that_is_the_model_or_the_engine_is_refused_before_anything_is_created(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_name: str, has_fp16_flag: bool
    ) -> None:
        """A cache aliasing the build's ONNX model or its engine is refused before the build, and before its lock.

        Saved over the ONNX model, the cache would destroy the export's own input; saved where the engine goes, it would
        be overwritten by the engine on every run. A TensorRT without the FP16 builder flag falls back to an FP32
        engine, so the comparison has to use the engine's final name, not the requested one.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=has_fp16_flag))
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"exported model")
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, timing_cache=tmp_path / cache_name))

        with pytest.raises(ValueError, match="trt_timing_cache .* is the same file as"):
            exporter.build_engine(str(onnx_path))

        assert (captured, sorted(os.listdir(tmp_path))) == ({"config": {}, "build": {}}, ["model.onnx"])
