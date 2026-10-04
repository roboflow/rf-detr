# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``export(format="tensorrt", quantization="int8")``.

The request checks and the build wiring run without TensorRT. The quantization plan and the Q/DQ rewrite are pure ONNX
work and run on CPU with ``onnx``, on small synthetic graphs that mirror RF-DETR's naming, plus one exact-count check on
a real exported Nano graph. Calibration needs onnxruntime and runs in the ``onnx`` integration job. The end-to-end class
builds real FP16 and INT8 engines on a GPU and compares their detections on a photo.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rfdetr.export._onnx import exporter as onnx_export
from rfdetr.export._runtime import calibration
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import quantize
from rfdetr.export._tensorrt.exporter import (
    _IS_POLYGRAPHY_AVAILABLE,
    _IS_TENSORRT_AVAILABLE,
    TensorRTConfig,
    TensorRTExporter,
)
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

#: Node-name prefix of a block the plan treats as backbone encoder (INT8 attention allowed).
BACKBONE = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/"
#: Node-name prefix of a block the plan treats as decoder (INT8 linears, FP16 attention).
DECODER = "/transformer/decoder/layers.0/"
#: Node-name prefix outside every quantized region (a detection head).
HEAD = "/class_embed/"

tensorrt_only = pytest.mark.skipif(
    not (_IS_TENSORRT_AVAILABLE and _IS_POLYGRAPHY_AVAILABLE), reason="tensorrt/polygraphy not installed"
)


def _weight(name: str, rows: int, cols: int, dtype: type = np.float32, seed: int = 0) -> Any:
    """Return a seeded random ``rows x cols`` weight initializer.

    Examples:
        >>> _weight("w", 2, 3).dims
        [2, 3]
    """
    values = np.random.default_rng(seed).standard_normal((rows, cols)) * 0.1
    return numpy_helper.from_array(values.astype(dtype), name)


def _attention_and_mlp(
    attention_prefix: str,
    *,
    sequence: int,
    head_size: int,
    mlp_prefix: str = BACKBONE,
    second_mlp_prefix: str | None = None,
    activation: str = "Relu",
    dtype: type = np.float32,
    scaled_scores: bool = False,
    second_reader: bool = False,
) -> Any:
    """Build ``x -> attention block -> MLP`` with RF-DETR-style node names, as a ``ModelProto``.

    The attention block projects ``x`` to q/k/v, splits two heads, runs ``MatMul -> Softmax -> MatMul``, projects back;
    the MLP is ``fc1 -> activation -> fc2``. ``activation="Erf"`` writes GELU in its traced erf form, ``"Gelu"`` as the
    opset-20 op.

    Args:
        attention_prefix: Node-name prefix of the attention block.
        sequence: Tokens per sequence.
        head_size: Channels per head (two heads).
        mlp_prefix: Node-name prefix of ``fc1``.
        second_mlp_prefix: Node-name prefix of ``fc2``; defaults to *mlp_prefix*.
        activation: ``"Relu"``, ``"Erf"`` or ``"Gelu"``.
        dtype: Float type of the graph (``np.float32`` or ``np.float16``).
        scaled_scores: Divide the attention scores by a constant before the ``Softmax`` (eager attention's form).
        second_reader: Give the merged attention output a second reader besides the output projection.

    Returns:
        A checked ``ModelProto``.

    Examples:
        >>> model = _attention_and_mlp(BACKBONE, sequence=4, head_size=16)
        >>> sum(node.op_type == "Softmax" for node in model.graph.node)
        1
    """
    width = 2 * head_size
    elem = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    second_mlp_prefix = second_mlp_prefix or mlp_prefix
    inits = [_weight(f"w{n}", width, width, dtype, seed) for seed, n in enumerate("qkvo")]
    inits += [_weight("w1", width, 2 * width, dtype, 5), _weight("w2", 2 * width, width, dtype, 6)]
    inits += [
        numpy_helper.from_array(np.array(v, dtype), n) for n, v in (("sqrt2", 1.4142), ("one", 1.0), ("half", 0.5))
    ]
    inits += [
        numpy_helper.from_array(np.array([1, sequence, 2, head_size], np.int64), "heads"),
        numpy_helper.from_array(np.array([1, sequence, width], np.int64), "merged"),
    ]
    p = attention_prefix
    nodes = []
    for name in "qkv":
        nodes += [
            helper.make_node("MatMul", ["x", f"w{name}"], [f"{name}0"], f"{p}{name}/MatMul"),
            helper.make_node("Reshape", [f"{name}0", "heads"], [f"{name}1"], f"{p}{name}/Reshape"),
        ]
    nodes += [
        helper.make_node("Transpose", ["q1"], ["q2"], f"{p}q/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k1"], ["k2"], f"{p}k/Transpose", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["v1"], ["v2"], f"{p}v/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q2", "k2"], ["s0" if scaled_scores else "s"], f"{p}scores/MatMul"),
        *([helper.make_node("Div", ["s0", "sqrt2"], ["s"], f"{p}scale/Div")] if scaled_scores else []),
        helper.make_node("Softmax", ["s"], ["p"], f"{p}Softmax", axis=-1),
        helper.make_node("MatMul", ["p", "v2"], ["c"], f"{p}context/MatMul"),
        helper.make_node("Transpose", ["c"], ["c1"], f"{p}merge/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["c1", "merged"], ["c2"], f"{p}merge/Reshape"),
        helper.make_node("MatMul", ["c2", "wo"], ["a"], f"{p}o/MatMul"),
        helper.make_node("MatMul", ["a", "w1"], ["h"], f"{mlp_prefix}fc1/MatMul"),
    ]
    if activation == "Erf":
        nodes += [
            helper.make_node("Div", ["h", "sqrt2"], ["h1"], f"{mlp_prefix}act/Div"),
            helper.make_node("Erf", ["h1"], ["h2"], f"{mlp_prefix}act/Erf"),
            helper.make_node("Add", ["h2", "one"], ["h3"], f"{mlp_prefix}act/Add"),
            helper.make_node("Mul", ["h", "h3"], ["h4"], f"{mlp_prefix}act/Mul"),
            helper.make_node("Mul", ["h4", "half"], ["g"], f"{mlp_prefix}act/Mul_1"),
        ]
    else:
        nodes.append(helper.make_node(activation, ["h"], ["g"], f"{mlp_prefix}act/{activation}"))
    nodes.append(helper.make_node("MatMul", ["g", "w2"], ["y"], f"{second_mlp_prefix}fc2/MatMul"))
    outputs = [helper.make_tensor_value_info("y", elem, [1, sequence, width])]
    if second_reader:
        nodes.append(helper.make_node("Identity", ["c2"], ["attention_out"], f"{p}tap/Identity"))
        outputs.append(helper.make_tensor_value_info("attention_out", elem, [1, sequence, width]))
    graph = helper.make_graph(
        nodes, "attention_mlp", [helper.make_tensor_value_info("x", elem, [1, sequence, width])], outputs, inits
    )
    opset = 20 if activation == "Gelu" else 17
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model, full_check=True)
    return model


def _export_graph(*extra_outputs: str) -> ExportGraph:
    """Return a static-batch detector :class:`ExportGraph` with ``dets``/``labels`` plus *extra_outputs*.

    Examples:
        >>> _export_graph("masks").output_names
        ('dets', 'labels', 'masks')
    """
    return ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 3, 8, 8),
        input_names=("input",),
        output_names=("dets", "labels", *extra_outputs),
        dynamic_axes=None,
        shape=(8, 8),
        backbone_only=False,
    )


def _refuse_host(cls: type) -> None:
    """Stand in for ``_require_int8_host`` on a host that cannot build INT8.

    Examples:
        >>> _refuse_host(object)
        Traceback (most recent call last):
        ...
        ImportError: host refused
    """
    raise ImportError("host refused")


def _names(plan: quantize.Int8Plan) -> list[str]:
    """Return the last two path segments of each weighted node, e.g. ``"fc1/MatMul"``.

    Examples:
        >>> _names(quantize.Int8Plan(weighted=("/a/b/fc1/MatMul",), activations=()))
        ['fc1/MatMul']
    """
    return ["/".join(name.split("/")[-2:]) for name in plan.weighted]


class TestInt8Request:
    """``quantization="int8"`` is validated from the configuration alone, before any forward pass."""

    @pytest.mark.parametrize("quantization", ["fp16", "fp32", "int4", "INT8"])
    def test_unknown_mode_is_refused(self, quantization: str) -> None:
        with pytest.raises(ValueError, match="quantization=None or 'int8'"):
            TensorRTExporter(TensorRTConfig(quantization=quantization))  # type: ignore[arg-type]

    def test_int8_needs_calibration_data(self) -> None:
        with pytest.raises(ValueError, match="needs calibration_data"):
            TensorRTExporter(TensorRTConfig(quantization="int8"))

    def test_calibration_data_without_int8_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="only read with quantization='int8'"):
            TensorRTExporter(TensorRTConfig(calibration_data=str(tmp_path)))

    @pytest.mark.parametrize(
        ("settings", "reason"),
        [
            pytest.param({"fp16": False}, "fp16=False", id="fp32"),
            pytest.param({"dynamic_batch": True, "max_batch_size": 4}, "static batch", id="dynamic-batch"),
            pytest.param({"backbone_only": True}, "backbone_only", id="backbone-only"),
            pytest.param({"max_images": 0}, "positive integer", id="no-images"),
            pytest.param({"max_images": True}, "positive integer", id="bool-images"),
            pytest.param({"max_images": 2.5}, "positive integer", id="float-images"),
            pytest.param({"max_images": np.bool_(True)}, "positive integer", id="numpy-bool-images"),
        ],
    )
    def test_unsupported_combination_is_refused(self, tmp_path: Path, settings: dict, reason: str) -> None:
        config = TensorRTConfig(quantization="int8", calibration_data=str(tmp_path), **settings)
        with pytest.raises(ValueError, match=reason):
            TensorRTExporter(config)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(["a.jpg"], id="list"),
            pytest.param(torch.zeros(1, 3, 8, 8), id="tensor"),
            pytest.param(np.zeros((3, 8, 8), np.float32), id="rank-3"),
            pytest.param(np.zeros((1, 3, 8, 8), np.uint8), id="uint8"),
            pytest.param(np.zeros((0, 3, 8, 8), np.float32), id="empty"),
        ],
    )
    def test_unusable_calibration_data_is_refused(self, data: object) -> None:
        with pytest.raises(ValueError, match="calibration_data"):
            TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=data))  # type: ignore[arg-type]

    @pytest.mark.parametrize("max_images", [np.int64(5), 5])
    def test_integer_max_images_is_accepted(self, tmp_path: Path, max_images: int | np.integer) -> None:
        TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=str(tmp_path), max_images=max_images))

    def test_export_keywords_reach_the_configuration(self, tmp_path: Path) -> None:
        config = TensorRTExporter.build_config(quantization="int8", calibration_data=str(tmp_path), max_images=7)
        assert (config.quantization, config.calibration_data, config.max_images) == ("int8", str(tmp_path), 7)

    @pytest.mark.parametrize(
        ("onnx_path", "output_name", "engine"),
        [
            pytest.param("out/model.onnx", None, "out/model_int8.trt", id="plain"),
            pytest.param("model.onnx", None, "model_int8.trt", id="no-directory"),
            pytest.param("out\\model.onnx", None, "out\\model_int8.trt", id="windows-separator"),
            pytest.param("out/model.v1.onnx", None, "out/model.v1_int8.trt", id="dotted-stem"),
            pytest.param("out/model.onnx", "custom", "out/custom.trt", id="custom-name"),
            pytest.param("out/model.onnx", "custom.trt", "out/custom.trt", id="custom-name-with-extension"),
        ],
    )
    def test_engine_is_named_int8(self, tmp_path: Path, onnx_path: str, output_name: str | None, engine: str) -> None:
        config = TensorRTConfig(quantization="int8", calibration_data=str(tmp_path), output_name=output_name)
        assert TensorRTExporter(config).build_engine(onnx_path, dry_run=True) == engine

    @pytest.mark.parametrize("fp16", [True, False])
    def test_precision_suffix_follows_the_request(self, tmp_path: Path, fp16: bool) -> None:
        config = TensorRTConfig(quantization=None, fp16=fp16)
        expected = "model_fp16.trt" if fp16 else "model_fp32.trt"
        assert TensorRTExporter(config).build_engine(str(tmp_path / "model.onnx"), dry_run=True).endswith(expected)


class TestInt8Host:
    """The host must have onnxruntime and TensorRT >= 10; both are checked before the ONNX export."""

    def test_missing_onnxruntime_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", False)
        with pytest.raises(ImportError, match="onnxruntime"):
            TensorRTExporter._require_int8_host()

    @pytest.mark.parametrize("version", ["8.6.1", "9.3.0.post12.dev1", "unknown"])
    def test_old_tensorrt_is_refused(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", True)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__=version))
        with pytest.raises(ImportError, match="TensorRT 10 or newer"):
            TensorRTExporter._require_int8_host()

    @pytest.mark.parametrize("version", ["10.0.1.6", "11.3.0.99"])
    def test_supported_tensorrt_is_accepted(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", True)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__=version))
        TensorRTExporter._require_int8_host()

    def test_missing_fp16_caster_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="10.16.1.11"))
        with pytest.raises(ImportError, match="onnxconverter-common"):
            TensorRTExporter._require_int8_host()

    @pytest.mark.parametrize("form", ["str", "path", "empty"])
    def test_missing_calibration_path_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, form: str
    ) -> None:
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: pytest.fail("ONNX export ran"))
        missing = {"str": str(tmp_path / "missing"), "path": tmp_path / "missing", "empty": ""}[form]
        config = TensorRTConfig(quantization="int8", calibration_data=missing)
        with pytest.raises(ValueError, match="does not exist"):
            TensorRTExporter(config)._convert(_export_graph())

    @pytest.mark.parametrize(
        ("name", "content", "message"),
        [
            pytest.param("raw.npy", np.zeros((1, 3, 8, 8), np.uint8), "preprocessed", id="uint8-npy"),
            pytest.param("image.npy", np.zeros((3, 8, 8), np.float32), "preprocessed", id="rank-3-npy"),
            pytest.param("empty.npy", np.zeros((0, 3, 8, 8), np.float32), "at least one image", id="no-image-npy"),
            pytest.param("broken.npy", b"", "could not read", id="unreadable-npy"),
            pytest.param("notes.txt", b"not an array", "file to be a .npy array", id="not-npy"),
        ],
    )
    def test_unusable_calibration_file_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, content: np.ndarray | bytes, message: str
    ) -> None:
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: pytest.fail("ONNX export ran"))
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            np.save(path, content)
        with pytest.raises(ValueError, match=message):
            TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=path))._convert(_export_graph())

    def test_preprocessed_npy_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "images.npy"
        np.save(path, np.zeros((2, 3, 8, 8), np.float32))
        TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=path))._require_calibration_path()

    def test_build_engine_refuses_a_missing_calibration_path_before_calibrating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", lambda *_, **__: pytest.fail("calibration ran"))
        exporter = TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=str(tmp_path / "missing")))
        with pytest.raises(ValueError, match="does not exist"):
            exporter.build_engine(str(tmp_path / "model.onnx"))

    def test_convert_checks_the_host_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(_refuse_host))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: pytest.fail("ONNX export ran"))
        exporter = TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=str(tmp_path)))
        with pytest.raises(ImportError, match="host refused"):
            exporter._convert(_export_graph())

    def test_build_engine_checks_the_host_before_calibrating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(_refuse_host))
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", lambda *_, **__: pytest.fail("calibration ran"))
        exporter = TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=str(tmp_path)))
        with pytest.raises(ImportError, match="host refused"):
            exporter.build_engine(str(tmp_path / "model.onnx"))

    @pytest.mark.parametrize("extra", ["masks", "keypoints"])
    def test_non_detection_model_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: str
    ) -> None:
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: pytest.fail("ONNX export ran"))
        exporter = TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=str(tmp_path)))
        with pytest.raises(NotImplementedError, match="detection models only"):
            exporter._convert(_export_graph(extra))


class TestInt8BuildWiring:
    """An INT8 build parses the quantized graph as a strongly typed network, whatever the TensorRT major."""

    def test_builds_the_quantized_graph_strongly_typed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: dict[str, Any] = {}

        @contextlib.contextmanager
        def fake_source(onnx_path: str, **kwargs: Any) -> Iterator[str]:
            calls["source"] = (onnx_path, kwargs)
            yield str(tmp_path / "quantized.onnx")

        def fake_parse(path: str, **kwargs: Any) -> tuple[None, object, None]:
            calls["parse"] = (path, kwargs)
            return (None, types.SimpleNamespace(num_inputs=0), None)

        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", fake_source)
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", fake_parse)
        monkeypatch.setattr(tensorrt_export, "_dynamic_batch_inputs", lambda network: {})
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: ("config", kwargs))
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda parsed, config: ("engine", config))
        monkeypatch.setattr(
            tensorrt_export, "save_engine", lambda engine, path: calls.setdefault("saved", (engine, path))
        )

        config = TensorRTConfig(quantization="int8", calibration_data=str(tmp_path), max_images=3, verbose=False)
        engine_path = TensorRTExporter(config).build_engine(str(tmp_path / "model.onnx"))

        assert calls["source"] == (
            str(tmp_path / "model.onnx"),
            {"calibration_data": str(tmp_path), "max_images": 3, "dynamic_batch": False},
        )
        assert calls["parse"] == (str(tmp_path / "quantized.onnx"), {"strongly_typed": True})
        assert calls["saved"] == (("engine", ("config", {"fp16": False})), engine_path)
        assert engine_path == str(tmp_path / "model_int8.trt")


class TestPlanInt8:
    """Which nodes get Q/DQ: the rules in :mod:`rfdetr.export._tensorrt.quantize`, on synthetic graphs."""

    def test_fusable_backbone_attention_is_quantized_whole(self) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        assert _names(plan) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul", "fc1/MatMul", "fc2/MatMul"]
        batched = sorted(
            (name.split("/")[-2], index)
            for name, index in plan.activations
            if "/scores/" in name or "/context/" in name
        )
        assert batched == [("context", 0), ("context", 1), ("scores", 0), ("scores", 1)]

    @pytest.mark.parametrize(("sequence", "head_size"), [(8, 8), (8, 128), (326, 16)])
    def test_attention_outside_the_int8_shapes_stays_float(self, sequence: int, head_size: int) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=sequence, head_size=head_size))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]
        assert [name for name, _ in plan.activations] == list(plan.weighted)

    def test_backbone_attention_left_float_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=326, head_size=16))
        assert [m for m in messages if "stay FP16" in m] != []

    def test_decoder_attention_left_float_is_not_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(_attention_and_mlp(DECODER, sequence=326, head_size=16))
        assert [m for m in messages if "stay FP16" in m] == []

    def test_attention_left_float_for_its_projections_is_not_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # FP16 because its output projection is outside the quantized regions, not because the kernel cannot fuse it.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        for node in model.graph.node:
            if node.name.endswith("o/MatMul"):
                node.name = f"{HEAD}o/MatMul"
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(model)
        assert [m for m in messages if "stay FP16" in m] == []

    def test_attention_sharing_a_name_stays_float(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        for node in model.graph.node:
            if node.name.endswith("context/MatMul"):
                node.name = f"{BACKBONE}scores/MatMul"
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    def test_token_limit_is_quantized_whole(self) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=325, head_size=32))
        assert sum("/scores/" in name for name, _ in plan.activations) == 2

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Mul", "Sub", "Where"])
    def test_scaled_or_masked_scores_are_still_recognised_as_attention(self, op: str) -> None:
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16, scaled_scores=True)
        next(node for node in model.graph.node if node.name.endswith("scale/Div")).op_type = op
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize("op", ["Cast", "Identity"])
    def test_probabilities_through_a_passthrough_are_still_recognised_as_attention(self, op: str) -> None:
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16)
        nodes = list(model.graph.node)
        index = next(i for i, node in enumerate(nodes) if node.op_type == "Softmax")
        nodes[index].output[0] = "p32"
        nodes.insert(index + 1, helper.make_node(op, ["p32"], ["p"], f"{DECODER}probs/{op}"))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    def test_unrecognised_attention_in_a_region_is_refused(self) -> None:
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16, scaled_scores=True)
        next(node for node in model.graph.node if node.name.endswith("scale/Div")).op_type = "Max"
        with pytest.raises(ValueError, match="not a recognised attention block"):
            quantize.plan_int8(model)

    def test_unrecognised_attention_named_outside_the_regions_is_refused(self) -> None:
        # Its projections sit in a quantized region, so a rename of the Softmax must not let them slip through.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, scaled_scores=True)
        next(node for node in model.graph.node if node.name.endswith("scale/Div")).op_type = "Max"
        next(node for node in model.graph.node if node.op_type == "Softmax").name = f"{HEAD}Softmax"
        with pytest.raises(ValueError, match="not a recognised attention block"):
            quantize.plan_int8(model)

    def test_float_attention_with_an_unfollowable_output_is_refused(self) -> None:
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16, second_reader=True)
        with pytest.raises(ValueError, match="output projection"):
            quantize.plan_int8(model)

    def test_fusable_attention_with_a_projection_outside_the_regions_stays_float(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        for node in model.graph.node:
            if node.name.endswith("o/MatMul"):
                node.name = f"{HEAD}o/MatMul"
        plan = quantize.plan_int8(model)
        assert (_names(plan), [n for n, _ in plan.activations if "/scores/" in n]) == (["fc1/MatMul", "fc2/MatMul"], [])

    def test_gelu_chain_drops_every_projection_feeding_a_float_consumer(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=8, activation="Gelu")
        graph = model.graph
        graph.initializer.append(_weight("w3", 32, 16, seed=7))
        graph.node.extend(
            [
                helper.make_node("Gelu", ["y"], ["y_gelu"], f"{BACKBONE}act2/Gelu"),
                helper.make_node("MatMul", ["y_gelu", "w3"], ["head"], f"{HEAD}out/MatMul"),
            ]
        )
        graph.output.append(helper.make_tensor_value_info("head", TensorProto.FLOAT, [1, 8, 16]))
        with pytest.raises(ValueError, match="RF-DETR detector exports only"):
            quantize.plan_int8(model)

    def test_decoder_attention_stays_float(self) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(DECODER, sequence=8, head_size=16))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize(
        ("channels", "group", "quantized"),
        [
            pytest.param(3, 1, False, id="rgb"),
            pytest.param(48, 1, False, id="48-channels"),
            pytest.param(32, 32, True, id="depthwise-32"),
        ],
    )
    def test_convolution_is_quantized_only_on_a_multiple_of_32_channels(
        self, channels: int, group: int, quantized: bool
    ) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["image", "first"], ["features"], f"{BACKBONE}patch/Conv", group=group),
                helper.make_node("Conv", ["features", "mix"], ["mixed"], f"{BACKBONE}mix/Conv"),
            ],
            "convs",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, channels, 8, 8])],
            [helper.make_tensor_value_info("mixed", TensorProto.FLOAT, [1, 16, 8, 8])],
            [
                numpy_helper.from_array(np.ones((32, channels // group, 1, 1), np.float32), "first"),
                numpy_helper.from_array(np.ones((16, 32, 1, 1), np.float32), "mix"),
            ],
        )
        plan = quantize.plan_int8(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
        assert ("patch/Conv" in _names(plan)) == quantized

    def test_heads_are_left_float(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, second_mlp_prefix=HEAD)
        assert "fc2/MatMul" not in _names(quantize.plan_int8(model))

    @pytest.mark.parametrize("activation", ["Erf", "Gelu"])
    def test_gelu_projection_waits_for_its_consumer(self, activation: str) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, second_mlp_prefix=HEAD, activation=activation)
        assert _names(quantize.plan_int8(model)) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"]

    def test_gelu_projection_waits_for_a_consumer_the_plan_cannot_name(self) -> None:
        # An unnamed fc2 cannot be planned, but it is still the FP16 consumer fc1's GELU feeds.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, activation="Erf")
        next(node for node in model.graph.node if node.name.endswith("fc2/MatMul")).name = ""
        assert _names(quantize.plan_int8(model)) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"]

    def test_gelu_projection_waits_for_a_consumer_behind_another_op(self) -> None:
        # Only a GELU's direct readers were followed: an Identity before the unplannable fc2 hid it.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, activation="Erf")
        nodes = list(model.graph.node)
        fc2 = next(node for node in nodes if node.name.endswith("fc2/MatMul"))
        fc2.name, fc2.input[0] = "", "g_copy"
        index = nodes.index(fc2)
        nodes.insert(index, helper.make_node("Identity", ["g"], ["g_copy"], f"{BACKBONE}tap/Identity"))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        assert _names(quantize.plan_int8(model)) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"]

    def test_relu_projection_does_not_wait(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, second_mlp_prefix=HEAD, activation="Relu")
        assert _names(quantize.plan_int8(model))[-1] == "fc1/MatMul"

    def test_graph_without_quantizable_layers_is_refused(self) -> None:
        model = _attention_and_mlp(HEAD, sequence=8, head_size=16, mlp_prefix=HEAD)
        with pytest.raises(ValueError, match="RF-DETR detector exports only"):
            quantize.plan_int8(model)

    def test_layer_inside_a_subgraph_is_not_planned(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        inner = helper.make_node("MatMul", ["c2", "wo"], ["inner"], f"{BACKBONE}inner/MatMul")
        branch = helper.make_graph([inner], "branch", [], [helper.make_tensor_value_info("inner", 1, [1, 8, 32])])
        model.graph.initializer.append(numpy_helper.from_array(np.array(True), "flag"))
        model.graph.node.append(
            helper.make_node("If", ["flag"], ["picked"], "if", then_branch=branch, else_branch=branch)
        )
        model.graph.output.append(helper.make_tensor_value_info("picked", 1, [1, 8, 32]))
        assert "inner/MatMul" not in _names(quantize.plan_int8(model))

    def test_nodes_sharing_a_name_are_skipped(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=8)
        for node in model.graph.node:
            if node.name.endswith("fc2/MatMul"):
                node.name = f"{BACKBONE}fc1/MatMul"
        with pytest.raises(ValueError, match="RF-DETR detector exports only"):
            quantize.plan_int8(model)


def _node(op_type: str, inputs: list[str], output: str, name: str = "", **attributes: Any) -> Any:
    """Return a one-output ``NodeProto`` called *name*; an empty name stands for an unnamed layer.

    Examples:
        >>> _node("Add", ["a", "b"], "c", "sum").output[0]
        'c'
    """
    return helper.make_node(op_type, inputs, [output], name, **attributes)


def _weights(*names: str, dims: tuple[int, ...] = (4, 4)) -> dict[str, Any]:
    """Return a constant-weight map holding a zero initializer of *dims* for each of *names*.

    Examples:
        >>> sorted(_weights("w", "v", dims=(2, 3)))
        ['v', 'w']
    """
    return {name: numpy_helper.from_array(np.zeros(dims, np.float32), name) for name in names}


def _readers(nodes: list[Any]) -> dict[str, list[Any]]:
    """Map every tensor name to the nodes that read it, as ``plan_int8`` builds it.

    Examples:
        >>> add = _node("Add", ["a", "b"], "c")
        >>> [node.op_type for node in _readers([add])["a"]]
        ['Add']
    """
    readers: dict[str, list[Any]] = {}
    for node in nodes:
        for tensor in node.input:
            readers.setdefault(tensor, []).append(node)
    return readers


def _producers(nodes: list[Any]) -> dict[str, Any]:
    """Map every tensor name to the node that writes it.

    Examples:
        >>> _producers([_node("Add", ["a", "b"], "c", "sum")])["c"].name
        'sum'
    """
    return {output: node for node in nodes for output in node.output}


class TestWeightLookup:
    """``_weight_name`` says which nodes the plan can refer to; ``_holds_weight`` says which ones read a constant."""

    #: Nodes as (node, weight dims). ``w`` is the only constant; ``act`` is an activation.
    CASES = [
        pytest.param(_node("MatMul", ["x", "w"], "y", "fc"), (4, 4), True, True, id="matmul"),
        pytest.param(_node("Gemm", ["x", "w"], "y", "fc"), (4, 4), True, True, id="gemm"),
        pytest.param(_node("Gemm", ["x", "w"], "y", "fc", transB=1), (4, 4), True, True, id="gemm-trans-b"),
        pytest.param(_node("Gemm", ["x", "w"], "y", "fc", transA=1), (4, 4), False, True, id="gemm-trans-a"),
        pytest.param(_node("Conv", ["x", "w"], "y", "conv"), (4, 3, 1, 1), True, True, id="conv"),
        pytest.param(_node("Conv", ["x", "w"], "y", "conv"), (4, 4), False, True, id="conv-with-a-matrix"),
        pytest.param(_node("MatMul", ["x", "w"], "y", "fc"), (2, 4, 4), False, True, id="matmul-with-a-batch"),
        pytest.param(_node("MatMul", ["x", "w"], "y", ""), (4, 4), False, True, id="unnamed"),
        pytest.param(_node("MatMul", ["x", "act"], "y", "bmm"), (4, 4), False, False, id="two-activations"),
        pytest.param(_node("Add", ["x", "w"], "y", "add"), (4, 4), False, False, id="not-a-weight-op"),
        pytest.param(_node("MatMul", ["x"], "y", "fc"), (4, 4), False, False, id="one-input"),
    ]

    @pytest.mark.parametrize(("node", "dims", "plannable", "_reads"), CASES)
    def test_weight_name_is_the_weight_only_for_a_plannable_node(
        self, node: Any, dims: tuple[int, ...], plannable: bool, _reads: bool
    ) -> None:
        assert (quantize._weight_name(node, _weights("w", dims=dims)) == "w") is plannable

    @pytest.mark.parametrize(("node", "dims", "_plannable", "reads"), CASES)
    def test_holds_weight_ignores_everything_but_the_constant(
        self, node: Any, dims: tuple[int, ...], _plannable: bool, reads: bool
    ) -> None:
        assert quantize._holds_weight(node, _weights("w", dims=dims)) is reads

    def test_weight_reached_through_an_identity_is_constant(self) -> None:
        graph = helper.make_graph(
            [_node("Identity", ["w"], "w_alias"), _node("Constant", [], "c", value=_weights("c")["c"])],
            "g",
            [],
            [],
            [_weights("w")["w"]],
        )
        assert sorted(quantize._constant_weights(graph)) == ["c", "w", "w_alias"]


class TestGraphWalkers:
    """The helpers that find an attention block's neighbours, on graphs of a few nodes."""

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Mul", "Sub", "Where"])
    def test_scores_are_found_behind_scaling_and_masking(self, op: str) -> None:
        nodes = [
            _node("MatMul", ["q", "k"], "s0", "scores"),
            _node(op, ["s0", "mask"], "s", "scale"),
            _node("Softmax", ["s"], "p", "softmax"),
        ]
        assert quantize._scores_matmul(nodes[2], _producers(nodes), {}).name == "scores"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([_node("Softmax", ["s"], "p", "softmax")], id="no-producer"),
            pytest.param(
                [_node("MatMul", ["q", "w"], "s", "projection"), _node("Softmax", ["s"], "p", "softmax")],
                id="weight-multiply",
            ),
            pytest.param(
                [
                    _node("Relu", ["s0"], "s", "act"),
                    _node("MatMul", ["q", "k"], "s0", "scores"),
                    _node("Softmax", ["s"], "p", "softmax"),
                ],
                id="blocked-by-another-op",
            ),
        ],
    )
    def test_scores_are_not_found_where_they_are_not_a_multiply_of_activations(self, nodes: list[Any]) -> None:
        softmax = next(node for node in nodes if node.op_type == "Softmax")
        assert quantize._scores_matmul(softmax, _producers(nodes), _weights("w")) is None

    def test_scores_are_found_when_a_node_is_reached_twice(self) -> None:
        nodes = [
            _node("MatMul", ["q", "k"], "s0", "scores"),
            _node("Add", ["s0", "s0"], "s", "twice"),
            _node("Softmax", ["s"], "p", "softmax"),
        ]
        assert quantize._scores_matmul(nodes[2], _producers(nodes), {}).name == "scores"

    @pytest.mark.parametrize("passthrough", [[], ["Cast"], ["Identity"], ["Cast", "Identity"]])
    def test_context_is_found_behind_casts(self, passthrough: list[str]) -> None:
        nodes, tensor = [_node("Softmax", ["s"], "p0", "softmax")], "p0"
        for index, op in enumerate(passthrough):
            nodes.append(_node(op, [tensor], f"p{index + 1}", f"pass{index}"))
            tensor = f"p{index + 1}"
        nodes.append(_node("MatMul", [tensor, "v"], "c", "context"))
        assert quantize._context_matmul(nodes[0], _readers(nodes)).name == "context"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([_node("Softmax", ["s"], "p", "softmax")], id="no-reader"),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Relu", ["p"], "r", "act"),
                    _node("MatMul", ["r", "v"], "c", "context"),
                ],
                id="blocked-by-another-op",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("MatMul", ["p", "v"], "c", "context"),
                    _node("Identity", ["p"], "tap", "tap"),
                ],
                id="two-readers",
            ),
            pytest.param(
                [_node("Softmax", ["s"], "p", "softmax"), _node("MatMul", ["v", "p"], "c", "context")],
                id="probabilities-as-the-second-operand",
            ),
        ],
    )
    def test_context_is_not_found_unless_probabilities_feed_one_multiply(self, nodes: list[Any]) -> None:
        assert quantize._context_matmul(nodes[0], _readers(nodes)) is None

    @pytest.mark.parametrize("follower", ["Cast", "Expand", "Identity", "Reshape", "Squeeze", "Transpose", "Unsqueeze"])
    def test_softmax_reaches_a_multiply_through_shape_ops(self, follower: str) -> None:
        nodes = [
            _node("Softmax", ["s"], "p", "softmax"),
            _node(follower, ["p"], "p1", "follow"),
            _node("MatMul", ["p1", "v"], "c", "context"),
        ]
        assert quantize._feeds_activation_matmul(nodes[0], _readers(nodes), {}) is True

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([_node("Softmax", ["s"], "p", "softmax")], id="no-reader"),
            pytest.param(
                [_node("Softmax", ["s"], "p", "softmax"), _node("MatMul", ["p", "w"], "c", "projection")],
                id="weight-multiply",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Mul", ["p", "p"], "m", "mul"),
                    _node("MatMul", ["m", "v"], "c", "context"),
                ],
                id="blocked-by-another-op",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Identity", ["p"], "a", "a"),
                    _node("Identity", ["p"], "b", "b"),
                    _node("Mul", ["a", "b"], "m", "mul"),
                ],
                id="reader-reached-twice",
            ),
        ],
    )
    def test_softmax_that_never_reaches_a_multiply_of_activations_is_not_attention(self, nodes: list[Any]) -> None:
        assert quantize._feeds_activation_matmul(nodes[0], _readers(nodes), _weights("w")) is False

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Identity", "Mul", "Transpose"])
    def test_projection_is_found_behind_each_passthrough_op(self, op: str) -> None:
        nodes = [_node("MatMul", ["x", "w"], "q0", "q_proj"), _node(op, ["q0", "bias"], "q", "after")]
        assert quantize._trace_projection("q", _producers(nodes), _weights("w", "bias")) == "q_proj"

    @pytest.mark.parametrize("op", ["Reshape", "Squeeze", "Unsqueeze"])
    def test_projection_is_found_through_the_data_input_of_a_shape_op(self, op: str) -> None:
        # The shape operand comes from another projection and must not be followed.
        nodes = [
            _node("MatMul", ["x", "w"], "q0", "q_proj"),
            _node("MatMul", ["x", "w"], "shape", "shape_proj"),
            _node(op, ["q0", "shape"], "q", "after"),
        ]
        assert quantize._trace_projection("q", _producers(nodes), _weights("w")) == "q_proj"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([], id="graph-input"),
            pytest.param([_node("Shape", ["x"], "q", "shape")], id="shape-arithmetic"),
            pytest.param([_node("Relu", ["q0"], "q", "act"), _node("MatMul", ["x", "w"], "q0", "q_proj")], id="act"),
        ],
    )
    def test_projection_is_not_found_behind_other_ops(self, nodes: list[Any]) -> None:
        assert quantize._trace_projection("q", _producers(nodes), _weights("w")) is None

    @pytest.mark.parametrize("hops", [0, 1, 3])
    def test_output_projection_is_found_within_three_shape_ops(self, hops: int) -> None:
        nodes, tensor = [], "ctx"
        for index in range(hops):
            nodes.append(_node("Transpose" if index % 2 else "Reshape", [tensor, "shape"], f"t{index}", f"hop{index}"))
            tensor = f"t{index}"
        nodes.append(_node("MatMul", [tensor, "w"], "y", "out_proj"))
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w", "shape")) == "out_proj"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([], id="no-reader"),
            pytest.param(
                [_node("MatMul", ["ctx", "w"], "a", "p1"), _node("MatMul", ["ctx", "w"], "b", "p2")], id="two-readers"
            ),
            pytest.param([_node("Relu", ["ctx"], "a", "act"), _node("MatMul", ["a", "w"], "y", "p")], id="other-op"),
            pytest.param([_node("MatMul", ["x", "ctx"], "y", "ctx_as_weight")], id="context-is-the-second-operand"),
        ],
    )
    def test_output_projection_is_not_found_unless_one_weight_reads_the_context(self, nodes: list[Any]) -> None:
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w")) is None

    def test_output_projection_is_not_found_beyond_three_shape_ops(self) -> None:
        nodes = [_node("Reshape", [f"t{i}", "shape"], f"t{i + 1}", f"hop{i}") for i in range(4)]
        nodes[0].input[0] = "ctx"
        nodes.append(_node("MatMul", ["t4", "w"], "y", "out_proj"))
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w", "shape")) is None


class TestGeluConsumers:
    """``_feeds_gelu_into`` lists what reads a GELU output, so a projection can wait for a quantized consumer."""

    @staticmethod
    def _chain(activation: str, *, after: list[str] | None = None, consumer_name: str = "fc2") -> tuple[list[Any], Any]:
        """Return ``fc1 -> GELU -> [ops in *after*] -> consumer`` as a node list and the ``fc1`` node.

        Examples:
            >>> nodes, fc1 = TestGeluConsumers._chain("Gelu", after=["Identity"])
            >>> [node.op_type for node in nodes]
            ['MatMul', 'Gelu', 'Identity', 'MatMul']
        """
        nodes = [_node("MatMul", ["x", "w1"], "h", "fc1")]
        if activation == "Erf":
            nodes += [
                _node("Div", ["h", "sqrt2"], "h1", "div"),
                _node("Erf", ["h1"], "h2", "erf"),
                _node("Add", ["h2", "one"], "h3", "add"),
                _node("Mul", ["h", "h3"], "h4", "mul"),
                _node("Mul", ["h4", "half"], "g", "mul_half"),
            ]
        else:
            nodes.append(_node(activation, ["h"], "g", "act"))
        tensor = "g"
        for index, op in enumerate(after or []):
            nodes.append(_node(op, [tensor], f"g{index}", f"after{index}"))
            tensor = f"g{index}"
        nodes.append(_node("MatMul", [tensor, "w2"], "y", consumer_name))
        return nodes, nodes[0]

    @pytest.mark.parametrize("activation", ["Erf", "Gelu"])
    def test_both_gelu_forms_lead_to_the_second_projection(self, activation: str) -> None:
        nodes, fc1 = self._chain(activation)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}

    def test_relu_is_not_a_gelu(self) -> None:
        nodes, fc1 = self._chain("Relu")
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == set()

    @pytest.mark.parametrize("after", [["Identity"], ["Cast", "Reshape"]])
    def test_any_op_between_the_gelu_and_its_consumer_counts_as_a_consumer(self, after: list[str]) -> None:
        nodes, fc1 = self._chain("Gelu", after=after)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"after0"}

    @pytest.mark.parametrize("consumer_name", ["", "fc2"])
    def test_a_consumer_is_listed_by_name_even_when_it_cannot_be_planned(self, consumer_name: str) -> None:
        nodes, fc1 = self._chain("Gelu", consumer_name=consumer_name)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {consumer_name}

    def test_a_consumer_reading_the_gelu_twice_is_listed_once(self) -> None:
        nodes, fc1 = self._chain("Gelu")
        nodes[-1].input[0] = "g"
        nodes[-1].input.append("g")
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}

    def test_the_branch_that_skips_the_gelu_is_not_a_consumer(self) -> None:
        nodes, fc1 = self._chain("Gelu")
        nodes.append(_node("MatMul", ["h", "w2"], "skip", "skip_fc"))
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}


class TestAttentionShapes:
    """INT8 attention needs a head size TensorRT's fused kernel accepts and a window of at most 325 tokens."""

    @pytest.mark.parametrize("head_size", [16, 32, 64])
    @pytest.mark.parametrize("sequence", [1, 8, 324, 325])
    def test_supported_shape_is_quantized_whole(self, sequence: int, head_size: int) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=sequence, head_size=head_size))
        assert (sum("/scores/" in n for n, _ in plan.activations), "o/MatMul" in _names(plan)) == (2, True)

    @pytest.mark.parametrize("head_size", [8, 24, 48, 128])
    def test_unsupported_head_size_stays_float(self, head_size: int) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=head_size))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize("sequence", [326, 400, 485, 580])
    def test_long_window_stays_float(self, sequence: int) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=sequence, head_size=16))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]

    def test_graph_without_shapes_counts_as_not_fusable(self) -> None:
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        del model.graph.input[:]
        model.graph.input.append(helper.make_empty_tensor_value_info("x"))
        for node in model.graph.node:
            if node.op_type == "Reshape":
                node.input[1] = "unknown_shape"  # a runtime shape: inference cannot see the head layout
        plan = quantize.plan_int8(model)
        assert [n for n, _ in plan.activations if "/scores/" in n] == []

    def test_symbolic_dimension_is_unknown(self) -> None:
        info = helper.make_tensor_value_info("t", TensorProto.FLOAT, ["n", 4])
        assert quantize._dims({"t": info}, "t") == (None, 4)

    def test_value_info_without_a_shape_is_unknown(self) -> None:
        assert quantize._dims({"t": helper.make_empty_tensor_value_info("t")}, "t") is None


def _quantized_with(mutate: Any) -> tuple[Any, quantize.Int8Plan]:
    """Plan on the FP32 attention-and-MLP graph, then quantize its FP16 twin, after *mutate* edited both.

    Examples:
        >>> model, plan = _quantized_with(lambda m: None)
        >>> onnx.checker.check_model(model, full_check=True)
    """
    fp32 = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
    fp16 = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
    mutate(fp32)
    mutate(fp16)
    plan = quantize.plan_int8(fp32)
    quantize.insert_qdq(fp16, plan, dict.fromkeys(plan.activations, 2.0))
    return fp16, plan


class TestInsertQdq:
    """The rewritten FP16 graph is valid opset-19 ONNX with FP16 scales and per-channel INT8 weights."""

    @pytest.fixture
    def quantized(self) -> tuple[Any, quantize.Int8Plan]:
        """Plan on the FP32 graph, then quantize its FP16 twin, as ``int8_source_graph`` does.

        Examples:
            A pytest fixture cannot be called directly.

            >>> model, plan = quantized()  # doctest: +SKIP
        """
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        return model, plan

    @pytest.fixture
    def ranged(self) -> tuple[Any, quantize.Int8Plan, dict[str, float]]:
        """Like ``quantized``, but each activation tensor has its own range; also returns tensor -> range.

        Examples:
            A pytest fixture cannot be called directly.

            >>> model, plan, by_tensor = ranged()  # doctest: +SKIP
        """
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        tensors = sorted({quantize._input_tensor(model.graph, *key) for key in plan.activations})
        by_tensor = {tensor: 0.5 * (index + 1) for index, tensor in enumerate(tensors)}
        ranges = {key: by_tensor[quantize._input_tensor(model.graph, *key)] for key in plan.activations}
        quantize.insert_qdq(model, plan, ranges)
        return model, plan, by_tensor

    def test_weight_shared_by_two_layers_is_quantized_for_each(self) -> None:
        def share(model: Any) -> None:
            next(n for n in model.graph.node if n.name.endswith("k/MatMul")).input[1] = "wq"

        model, plan = _quantized_with(share)
        by_name = {n.name: n for n in model.graph.node}
        readers = {by_name[f"{BACKBONE}{layer}/MatMul"].input[1] for layer in "qk"}
        assert len(readers) == 2 and all(r.endswith("_dequantized") for r in readers)
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize("tensor", ["c2", "wq", "x"])
    def test_quantized_tensor_that_is_also_a_graph_output_stays_an_output(self, tensor: str) -> None:
        shapes = {"c2": [1, 8, 32], "x": [1, 8, 32], "wq": [32, 32]}

        def expose(model: Any) -> None:
            elem = model.graph.output[0].type.tensor_type.elem_type
            model.graph.output.append(helper.make_tensor_value_info(tensor, elem, shapes[tensor]))

        model, _ = _quantized_with(expose)
        assert tensor in [o.name for o in model.graph.output]
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_tensor_read_inside_a_subgraph_keeps_its_original_binding(self) -> None:
        def capture(model: Any) -> None:
            tap = helper.make_node("Identity", ["c2"], ["inner"], "inner/Identity")
            branch = helper.make_graph([tap], "branch", [], [helper.make_tensor_value_info("inner", 10, [1, 8, 32])])
            model.graph.initializer.append(numpy_helper.from_array(np.array(True), "flag"))
            model.graph.node.append(
                helper.make_node("If", ["flag"], ["picked"], "if", then_branch=branch, else_branch=branch)
            )
            model.graph.output.append(helper.make_tensor_value_info("picked", 10, [1, 8, 32]))

        model, _ = _quantized_with(capture)
        if_node = next(n for n in model.graph.node if n.op_type == "If")
        assert if_node.attribute[0].g.node[0].input[0] == "c2"
        onnx.checker.check_model(model, full_check=True)

    def test_every_planned_input_reads_a_dequantize(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        model, plan, _ = ranged
        producer = {output: node for node in model.graph.node for output in node.output}
        by_name = {node.name: node for node in model.graph.node}
        read = [producer[by_name[name].input[index]].op_type for name, index in plan.activations]
        read += [producer[by_name[name].input[1]].op_type for name in plan.weighted]
        assert set(read) == {"DequantizeLinear"}

    def test_activation_scale_is_its_range_over_127(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        model, plan, by_tensor = ranged
        producer = {output: node for node in model.graph.node for output in node.output}
        by_name = {node.name: node for node in model.graph.node}
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        found = {}
        for name, index in plan.activations:
            dequantize = producer[by_name[name].input[index]]
            quantize_node = producer[dequantize.input[0]]
            found[quantize_node.input[0]] = (float(inits[quantize_node.input[1]]), float(inits[dequantize.input[1]]))
        expected = {t: (float(np.float16(r / 127)),) * 2 for t, r in by_tensor.items()}
        assert found == expected

    def test_activation_zero_point_is_int8_zero(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        model, _, _ = ranged
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        points = {
            node.input[2] for node in model.graph.node if node.op_type == "QuantizeLinear" and len(node.input) > 2
        }
        assert {(inits[name].dtype, int(inits[name])) for name in points} == {(np.dtype(np.int8), 0)}

    def test_convolution_weight_is_quantized_per_output_channel(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Conv", ["features", "mix"], ["mixed"], f"{BACKBONE}mix/Conv")],
            "conv",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 32, 4, 4])],
            [helper.make_tensor_value_info("mixed", TensorProto.FLOAT, [1, 16, 4, 4])],
            [numpy_helper.from_array(np.ones((16, 32, 1, 1), np.float32), "mix")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        plan = quantize.plan_int8(model)
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 1.0))
        inits = {init.name: init for init in model.graph.initializer}
        weight_dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits)
        assert ([a.i for a in weight_dq.attribute if a.name == "axis"], list(inits[weight_dq.input[1]].dims)) == (
            [0],
            [16],
        )

    def test_result_is_valid_onnx(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        model, _ = quantized
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_opset_is_lifted(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        model, _ = quantized
        assert [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")] == [19]

    def test_one_quantize_per_activation_tensor(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        model, _ = quantized
        quantized_inputs = [node.input[0] for node in model.graph.node if node.op_type == "QuantizeLinear"]
        # x feeds q, k and v through one pair; then the four batched-multiply inputs, o's input, fc1's and fc2's.
        assert sorted(quantized_inputs) == sorted(["x", "q2", "k2", "p", "v2", "c2", "a", "g"])

    def test_scales_are_fp16(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        model, _ = quantized
        inits = {init.name: init for init in model.graph.initializer}
        scales = {node.input[1] for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")}
        assert {inits[name].data_type for name in scales} == {TensorProto.FLOAT16}

    def test_weights_are_int8_per_output_channel(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        model, plan = quantized
        inits = {init.name: init for init in model.graph.initializer}
        weight_dq = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits]
        assert len(weight_dq) == len(plan.weighted)
        assert {(inits[n.input[0]].data_type, n.attribute[0].i) for n in weight_dq} == {(TensorProto.INT8, 1)}

    def test_dequantized_weight_matches_the_original(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        model, _ = quantized
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0].startswith("w1"))
        scales = inits[dq.input[1]].astype(np.float32)  # one per output channel (axis 1)
        restored = inits[dq.input[0]].astype(np.float32) * scales
        original = numpy_helper.to_array(_weight("w1", 32, 64, np.float16, 5)).astype(np.float32)
        # Rounded against the stored FP16 scale, every weight is within half a step of its original value.
        assert float((np.abs(restored - original) / scales).max()) <= 0.5 + 1e-3

    def test_scales_stay_positive_in_fp16(self) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        weight = next(init for init in model.graph.initializer if init.name == "w1")
        values = numpy_helper.to_array(weight).copy()
        values[:, 0] = 2e-6  # a near-dead output channel
        weight.CopyFrom(numpy_helper.from_array(values, "w1"))
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 0.0))
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        scales = [inits[n.input[1]] for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        # Not merely positive: a subnormal FP16 scale is as unusable as zero, so the floor is the smallest normal value.
        assert min(float(np.min(scale)) for scale in scales) >= np.finfo(np.float16).tiny

    def test_initializer_read_only_inside_a_subgraph_survives(self) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        model.graph.initializer.append(numpy_helper.from_array(np.ones(32, np.float16), "bias"))
        branch = helper.make_graph(
            [helper.make_node("Add", ["y", "bias"], ["z"], "inner/Add")],
            "branch",
            [],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT16, [1, 8, 32])],
        )
        model.graph.input.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, []))
        model.graph.node.append(helper.make_node("If", ["cond"], ["out"], "if", then_branch=branch, else_branch=branch))
        model.graph.output.append(helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 8, 32]))
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize(("transposed", "axis"), [(0, 1), (1, 0)])
    def test_gemm_weight_axis_follows_trans_b(self, transposed: int, axis: int) -> None:
        shape = (4, 8) if transposed else (8, 4)
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["x", "w"], ["y"], f"{DECODER}linear/Gemm", transB=transposed)],
            "gemm",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 4])],
            [_weight("w", *shape, np.float16)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        plan = quantize.Int8Plan(weighted=(f"{DECODER}linear/Gemm",), activations=((f"{DECODER}linear/Gemm", 0),))
        quantize.insert_qdq(model, plan, {(f"{DECODER}linear/Gemm", 0): 1.0})
        onnx.checker.check_model(model, full_check=True)
        weight_dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] != "x_int8")
        assert weight_dq.attribute[0].i == axis

    def test_generated_names_avoid_existing_tensors(self) -> None:
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        # Claim the names the rewrite would pick first, as a graph output and an unused initializer.
        model.graph.output.append(helper.make_tensor_value_info("x_int8", TensorProto.FLOAT16, [1, 8, 32]))
        model.graph.node.append(helper.make_node("Identity", ["y"], ["x_int8"], "claim"))
        model.graph.initializer.append(numpy_helper.from_array(np.array(0, np.int8), "int8_zero_point"))
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        onnx.checker.check_model(model, full_check=True)
        outputs = [o for n in model.graph.node for o in n.output]
        assert len(outputs) == len(set(outputs))


class TestCalibrationGraph:
    """The probe graph that measures ranges stays valid ONNX whatever names the model already uses."""

    def test_probe_names_avoid_initializer_outputs(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"], "relu")],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("x__absmax", TensorProto.FLOAT, []),
            ],
            [numpy_helper.from_array(np.array(1.0, np.float32), "x__absmax")],
        )
        probe = quantize._calibration_graph(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), ["x"]
        )
        onnx.checker.check_model(probe, full_check=True)


#: The reductions whose ``axes`` moved from an attribute to an input in opset 18; spelled out so a shrunken set fails.
REDUCTIONS = [
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
]


class TestLiftOpset:
    """FP16 scales need opset 19; lifting a 17 graph keeps every op's meaning or refuses."""

    def test_reduction_inside_a_subgraph_moves_axes_to_an_input(self) -> None:
        reduce = helper.make_node("ReduceMax", ["x"], ["r"], "inner/ReduceMax", axes=[1], keepdims=0)
        branch = helper.make_graph([reduce], "branch", [], [helper.make_tensor_value_info("r", TensorProto.FLOAT, [2])])
        if_node = helper.make_node("If", ["cond"], ["y"], "if", then_branch=branch, else_branch=branch)
        graph = helper.make_graph(
            [if_node],
            "lift",
            [
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        quantize._lift_opset(model)
        onnx.checker.check_model(model, full_check=True)
        inner = model.graph.node[0].attribute[0].g.node[0]
        assert (len(inner.input), [a.name for a in inner.attribute]) == (2, ["keepdims"])

    @staticmethod
    def _reduction_model(op: str, opset: int, *, axes: list[int] | None, taken: str | None = None) -> Any:
        """Return ``x -> op -> y`` at *opset*, with an ``axes`` attribute unless *axes* is ``None``.

        Examples:
            >>> model = TestLiftOpset._reduction_model("ReduceMax", 17, axes=[1])
            >>> (model.opset_import[0].version, model.graph.node[0].attribute[0].name)
            (17, 'axes')
        """
        attributes = {"keepdims": 0} | ({"axes": axes} if axes is not None else {})
        node = helper.make_node(op, ["x"], ["y"], "reduce", **attributes)
        graph = helper.make_graph(
            [node],
            "reduce",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.zeros(1, np.int64), taken)] if taken else [],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    @pytest.mark.parametrize("opset", [17, 18])
    @pytest.mark.parametrize("op", REDUCTIONS)
    def test_every_reduction_moves_its_axes_to_an_input(self, op: str, opset: int) -> None:
        model = self._reduction_model(op, opset, axes=[1])
        quantize._lift_opset(model)
        node = model.graph.node[0]
        assert (model.opset_import[0].version, len(node.input), "axes" in {a.name for a in node.attribute}) == (
            19,
            2,
            False,
        )

    @pytest.mark.parametrize("op", REDUCTIONS)
    def test_reduction_without_axes_is_left_alone(self, op: str) -> None:
        model = self._reduction_model(op, 17, axes=None)
        quantize._lift_opset(model)
        assert (len(model.graph.node[0].input), len(model.graph.initializer)) == (1, 0)

    def test_axes_input_does_not_reuse_a_taken_name(self) -> None:
        model = self._reduction_model("ReduceMax", 17, axes=[1], taken="reduce_axes")
        quantize._lift_opset(model)
        new = model.graph.node[0].input[1]
        assert new != "reduce_axes" and sorted(i.name for i in model.graph.initializer) == sorted([new, "reduce_axes"])

    @pytest.mark.parametrize(
        ("inputs", "expected"),
        [pytest.param(["x"], [2], id="unsized"), pytest.param(["x", "sizes"], [], id="sized")],
    )
    def test_split_without_sizes_gets_its_output_count(self, inputs: list[str], expected: list[int]) -> None:
        split = helper.make_node("Split", inputs, ["a", "b"], "split", axis=0)
        sizes = [numpy_helper.from_array(np.array([1, 1], np.int64), "sizes")]
        graph = helper.make_graph(
            [split],
            "split",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in "ab"],
            sizes if "sizes" in inputs else [],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        quantize._lift_opset(model)
        onnx.checker.check_model(model, full_check=True)
        assert [a.i for a in model.graph.node[0].attribute if a.name == "num_outputs"] == expected

    def test_graph_already_at_opset_19_is_left_alone(self) -> None:
        reduce = helper.make_node("ReduceMax", ["x"], ["r"], "reduce", axes=[1], keepdims=0)
        graph = helper.make_graph(
            [reduce],
            "lift",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("r", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        before = model.SerializeToString()
        quantize._lift_opset(model)
        assert model.SerializeToString() == before

    def test_other_domains_are_left_alone(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("CustomOp", ["x"], ["y"], "custom", domain="com.example")],
            "custom",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.example", 1)]
        )
        quantize._lift_opset(model)
        assert sorted((e.domain, e.version) for e in model.opset_import) == [("", 19), ("com.example", 1)]

    def test_op_whose_meaning_changed_is_refused(self) -> None:
        pad = helper.make_node("Pad", ["x", "pads"], ["y"], "pad")
        graph = helper.make_graph(
            [pad],
            "pad",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
            [numpy_helper.from_array(np.array([1, 1], np.int64), "pads")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with pytest.raises(ValueError, match="'pad' \\(Pad\\) changed meaning"):
            quantize._lift_opset(model)


class TestInt8SourceGraph:
    """A graph INT8 cannot start from is refused before any calibration, with advice an INT8 request can follow."""

    @pytest.mark.parametrize(
        ("rewrite", "message"),
        [
            pytest.param("qdq", "already explicitly quantized", id="quantized"),
            pytest.param("qdq-subgraph", "already explicitly quantized", id="quantized-subgraph"),
            pytest.param("fp16", "not a float32 export", id="fp16"),
            pytest.param("fp16-weights", "not a float32 export", id="fp16-weights-fp32-io"),
            pytest.param("dynamic", "static batch", id="dynamic-batch"),
            pytest.param("pad", "changed meaning", id="unliftable"),
            pytest.param("masks", "detection models only", id="segmentation"),
        ],
    )
    def test_unquantizable_source_is_refused(self, tmp_path: Path, rewrite: str, message: str) -> None:
        dtype = np.float16 if rewrite == "fp16" else np.float32
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=dtype)
        if rewrite == "qdq":
            plan = quantize.plan_int8(model)
            quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        elif rewrite == "qdq-subgraph":
            inner = helper.make_graph(
                [
                    helper.make_node("QuantizeLinear", ["y", "s", "z"], ["yq"], "inner/Q"),
                    helper.make_node("DequantizeLinear", ["yq", "s", "z"], ["w"], "inner/DQ"),
                ],
                "branch",
                [],
                [helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 8, 32])],
                [
                    numpy_helper.from_array(np.array(1.0, np.float32), "s"),
                    numpy_helper.from_array(np.array(0, np.int8), "z"),
                ],
            )
            model.graph.input.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, []))
            model.graph.node.append(
                helper.make_node("If", ["cond"], ["out"], "if", then_branch=inner, else_branch=inner)
            )
            model.graph.output.append(helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 32]))
        elif rewrite == "fp16-weights":
            for init in model.graph.initializer:
                if init.data_type == TensorProto.FLOAT:
                    init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
        elif rewrite == "dynamic":
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        elif rewrite == "pad":
            model.graph.initializer.append(numpy_helper.from_array(np.zeros(6, np.int64), "pads"))
            model.graph.node.append(helper.make_node("Pad", [model.graph.input[0].name, "pads"], ["labels"], "pad"))
        if rewrite in ("pad", "masks"):
            # A detector's outputs, so that only the rewrite under test can be what is refused.
            model.graph.node.append(helper.make_node("Identity", ["y"], ["dets"], "dets"))
            outputs = ["dets", "labels"] + (["masks"] if rewrite == "masks" else [])
            if rewrite == "masks":
                model.graph.node.append(helper.make_node("Identity", ["y"], ["labels"], "labels"))
                model.graph.node.append(helper.make_node("Identity", ["y"], ["masks"], "masks"))
            del model.graph.output[:]
            model.graph.output.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs)
        path = tmp_path / "source.onnx"
        onnx.save(model, path)
        with pytest.raises(ValueError, match=message):
            with quantize.int8_source_graph(
                str(path), calibration_data=str(tmp_path), max_images=1, dynamic_batch=False
            ):
                pass


@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestCalibration:
    """Ranges come from the FP32 graph under onnxruntime (CPU here); selected by the ``onnx`` integration job."""

    @pytest.fixture
    def image_graph(self, tmp_path: Path) -> Path:
        """A non-square ``(2, 3, 4, 6)``-input graph with two backbone Convs, saved to disk.

        The first reads the 3-channel image and stays FP16; the second reads 32 channels and is quantized. The outputs
        carry a detector's names.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path = image_graph(tmp_path)  # doctest: +SKIP
        """
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["input", "kernel"], ["features"], f"{BACKBONE}patch/Conv"),
                helper.make_node("Conv", ["features", "mix"], ["mixed"], f"{BACKBONE}mix/Conv"),
                helper.make_node("Relu", ["mixed"], ["dets"], "relu"),
                helper.make_node("Identity", ["mixed"], ["labels"], "labels"),
            ],
            "image",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 6])],
            [
                helper.make_tensor_value_info("dets", TensorProto.FLOAT, [2, 2, 4, 6]),
                helper.make_tensor_value_info("labels", TensorProto.FLOAT, [2, 2, 4, 6]),
            ],
            [
                numpy_helper.from_array(np.ones((32, 3, 1, 1), np.float32), "kernel"),
                numpy_helper.from_array(np.ones((2, 32, 1, 1), np.float32), "mix"),
            ],
        )
        path = tmp_path / "image.onnx"
        # Pinned: onnx 1.21 writes IR version 14 by default, which the onnxruntime CI installs (IR 13 at most) refuses.
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)
        return path

    @pytest.mark.parametrize(
        "values",
        [
            # Stopping after the first group would miss the largest magnitude in the padded final group, and keeping
            # only the last group would miss it in the first.
            pytest.param((2.0, 3.0, -5.0), id="largest-in-the-last-group"),
            pytest.param((-5.0, 2.0, 3.0), id="largest-in-the-first-group"),
        ],
    )
    def test_ranges_are_the_absolute_maximum_over_every_image(
        self, image_graph: Path, values: tuple[float, ...]
    ) -> None:
        pytest.importorskip("onnxruntime")
        images = [np.full((1, 3, 4, 6), value, np.float32) for value in values]
        ranges = quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input", "features"], images)
        assert ranges == pytest.approx({"input": 5.0, "features": 15.0})

    def test_calibration_data_is_closed_when_calibration_fails(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path
    ) -> None:
        pytest.importorskip("onnxruntime")
        generators: list[Iterator[np.ndarray]] = []
        real = calibration.calibration_batches

        def keep(*args: Any, **kwargs: Any) -> Iterator[np.ndarray]:
            """Return the real batch generator and remember it."""
            generators.append(real(*args, **kwargs))
            return generators[-1]

        monkeypatch.setattr(calibration, "calibration_batches", keep)
        images = np.zeros((2, 3, 4, 6), np.float32)
        images[1, 0, 1, 2] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            with quantize.int8_source_graph(
                str(image_graph), calibration_data=images, max_images=2, dynamic_batch=False
            ):
                pass
        assert [inspect.getgeneratorstate(g) for g in generators] == [inspect.GEN_CLOSED]

    def test_no_image_is_refused(self, image_graph: Path) -> None:
        pytest.importorskip("onnxruntime")
        with pytest.raises(ValueError, match="no image"):
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [])

    def test_calibration_runs_on_the_cpu(self, monkeypatch: pytest.MonkeyPatch, image_graph: Path) -> None:
        ort = pytest.importorskip("onnxruntime")
        requested: list[object] = []

        def record(*_: object, providers: object = None, **__: object) -> None:
            requested.append(providers)
            raise RuntimeError("recorded")

        monkeypatch.setattr(ort, "InferenceSession", record)
        with pytest.raises(RuntimeError, match="recorded"):
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [np.zeros((1, 3, 4, 6))])
        assert requested == [["CPUExecutionProvider"]]

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_calibration_image_is_refused(self, image_graph: Path, value: float) -> None:
        pytest.importorskip("onnxruntime")
        # Not the first element: onnxruntime's ReduceMax skips a NaN anywhere else and returns a finite range.
        images = np.zeros((2, 3, 4, 6), np.float32)
        images[1, 0, 1, 2] = value
        with pytest.raises(ValueError, match="NaN or infinite"):
            with quantize.int8_source_graph(
                str(image_graph), calibration_data=images, max_images=2, dynamic_batch=False
            ):
                pass

    def test_range_too_large_for_an_fp16_scale_is_refused(self, image_graph: Path) -> None:
        pytest.importorskip("onnxruntime")
        images = np.full((2, 3, 4, 6), 1e8, np.float32)  # finite, but 1e8 / 127 overflows FP16
        with pytest.raises(ValueError, match="does not fit an FP16 scale"):
            with quantize.int8_source_graph(
                str(image_graph), calibration_data=images, max_images=2, dynamic_batch=False
            ):
                pass

    def test_probe_is_removed_when_the_session_fails(self, monkeypatch: pytest.MonkeyPatch, image_graph: Path) -> None:
        ort = pytest.importorskip("onnxruntime")

        def refuse(*_: object, **__: object) -> None:
            raise RuntimeError("session failed")

        monkeypatch.setattr(ort, "InferenceSession", refuse)
        with pytest.raises(RuntimeError, match="session failed"):
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [np.zeros((1, 3, 4, 6))])
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    def test_intermediates_are_removed_when_the_rewrite_fails(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path
    ) -> None:
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxconverter_common")

        def fail(*_: object, **__: object) -> None:
            raise RuntimeError("rewrite failed")

        monkeypatch.setattr(quantize, "insert_qdq", fail)
        calibration = np.zeros((2, 3, 4, 6), np.float32)
        with pytest.raises(RuntimeError, match="rewrite failed"):
            with quantize.int8_source_graph(
                str(image_graph), calibration_data=calibration, max_images=10, dynamic_batch=False
            ):
                pass
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    def test_max_images_caps_a_calibration_directory(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        images = tmp_path / "images"
        images.mkdir()
        for index in range(3):
            Image.new("RGB", (6, 4)).save(images / f"{index}.png")
        seen: list[int] = []

        def count(onnx_path: str, model: Any, tensors: list[str], batches: Any) -> dict[str, float]:
            seen.append(len(list(batches)))
            raise RuntimeError("stop after counting")

        monkeypatch.setattr(quantize, "calibrate_ranges", count)
        with pytest.raises(RuntimeError, match="stop after counting"):
            with quantize.int8_source_graph(
                str(image_graph), calibration_data=str(images), max_images=2, dynamic_batch=False
            ):
                pass
        assert seen == [2]

    def test_source_graph_is_quantized_and_cleaned_up(self, image_graph: Path) -> None:
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxconverter_common")
        calibration = np.random.default_rng(0).standard_normal((3, 3, 4, 6)).astype(np.float32)
        with quantize.int8_source_graph(
            str(image_graph), calibration_data=calibration, max_images=10, dynamic_batch=False
        ) as path:
            model = onnx.load(path)
            assert path != str(image_graph)
            onnx.checker.check_model(model, full_check=True)
            assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 1
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]


class TestNanoInt8Plan:
    """Exact counts on a real RFDETRNano export pin the plan to the architecture it was measured on."""

    @pytest.fixture(scope="class")
    def nano_onnx(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Export an untrained RFDETRNano to ONNX on CPU, once for the class.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path = nano_onnx(tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano
        from rfdetr.utilities.reproducibility import seed_all

        seed_all(7)
        out_dir = tmp_path_factory.mktemp("nano_int8_plan")
        return Path(
            RFDETRNano(pretrain_weights=None, device="cpu").export(
                output_dir=str(out_dir), format="onnx", verbose=False
            )
        )

    @pytest.fixture(scope="class")
    def plan(self, nano_onnx: Path) -> quantize.Int8Plan:
        """The INT8 plan of the exported Nano graph.

        Examples:
            A pytest fixture cannot be called directly.

            >>> plan = plan(nano_onnx)  # doctest: +SKIP
        """
        return quantize.plan_int8(onnx.load(str(nano_onnx)))

    def test_attention_blocks_left_fp16_are_counted(self, monkeypatch: pytest.MonkeyPatch, nano_onnx: Path) -> None:
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(onnx.load(str(nano_onnx)))
        assert [m.split(";")[0] for m in messages if "stay FP16" in m] == [
            "INT8 attention in 9 of 12 backbone attention blocks"
        ]

    def test_rewritten_graph_is_valid_onnx(self, nano_onnx: Path, plan: quantize.Int8Plan) -> None:
        pytest.importorskip("onnxconverter_common")
        from rfdetr.export._tensorrt.exporter import fp16_source_graph

        with fp16_source_graph(str(nano_onnx)) as fp16_path:
            model = onnx.load(fp16_path)
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 1.0))
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_counts(self, plan: quantize.Int8Plan) -> None:
        # Backbone: 9 windowed blocks x (q, k, v, o, fc1, fc2) + 3 global blocks x (fc1, fc2) = 60; the patch embedding
        # reads the 3-channel image and stays FP16.
        # Decoder: 2 layers x (value, offsets, attention weights, output, linear1, linear2) + reference-point MLP = 14.
        # Activations: one per weighted node + 4 per INT8 attention block (9 windowed blocks).
        assert (len(plan.weighted), len(plan.activations)) == (74, 74 + 36)

    def test_global_blocks_keep_attention_in_fp16(self, plan: quantize.Int8Plan) -> None:
        global_attention = [n for n in plan.weighted if any(f"layer.{i}/attention/" in n for i in (3, 6, 9))]
        assert global_attention == []

    def test_decoder_self_attention_is_left_float(self, plan: quantize.Int8Plan) -> None:
        assert [n for n in plan.weighted if "/self_attn/" in n] == []

    def test_every_mlp_is_quantized_in_pairs(self, plan: quantize.Int8Plan) -> None:
        fc1 = {n.replace("fc1", "fc") for n in plan.weighted if "/mlp/fc1/" in n}
        fc2 = {n.replace("fc2", "fc") for n in plan.weighted if "/mlp/fc2/" in n}
        assert (len(fc1), fc1) == (12, fc2)


def _assert_confident_detections_match(fp16_engine: Path, int8_engine: Path, example: np.ndarray, batch: int) -> None:
    """Run both engines on *batch* copies of *example*; every confident FP16 detection needs an INT8 match.

    A match is an INT8 box with IoU above 0.85 and a score within 0.1 of the FP16 one, and at least three FP16
    detections must be confident for the check to mean anything. The INT8 engine must also give every row of the batch
    the same answer.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(np.repeat(example, batch, axis=0)).cuda()}
    results = []
    for path in (fp16_engine, int8_engine):
        outputs = TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs)
        results.append((outputs["dets"].float().cpu(), outputs["labels"].float().sigmoid().max(dim=-1).values.cpu()))
    (fp16_boxes, fp16_scores), (int8_boxes, int8_scores) = results
    assert torch.isfinite(int8_boxes).all() and torch.isfinite(int8_scores).all()
    assert torch.allclose(int8_boxes, int8_boxes[:1].expand_as(int8_boxes), atol=2e-2), "rows of one batch differ"
    for row in range(batch):
        confident = fp16_scores[row] > 0.5
        assert int(confident.sum()) >= 3, "the photo must give the FP16 engine several confident people"
        # (confident FP16 boxes, all INT8 boxes); several INT8 queries can cover one person, so a match is the closest
        # score among the INT8 boxes that overlap, not the score of the single best-overlapping box.
        overlaps = box_iou(box_cxcywh_to_xyxy(fp16_boxes[row][confident]), box_cxcywh_to_xyxy(int8_boxes[row]))[0]
        gaps = (int8_scores[row][None, :] - fp16_scores[row][confident][:, None]).abs()
        gaps = gaps.masked_fill(overlaps <= 0.85, float("inf")).min(dim=1).values
        unmatched = [
            (round(float(o), 3), round(float(g), 3)) for o, g in zip(overlaps.max(dim=1).values, gaps) if g >= 0.1
        ]
        assert unmatched == [], (
            f"row {row}: confident FP16 detections without an INT8 match (best IoU, score gap): {unmatched}"
        )


@tensorrt_only
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained RFDETRNano agree on a photo's confident detections."""

    @pytest.fixture(scope="class")
    def photo(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
        """Crops and flips of supervision's PEOPLE_WALKING photo for calibration, and the photo as a model input.

        Examples:
            A pytest fixture, and it needs a network download.

            >>> calibration_dir, example = photo(tmp_path_factory)  # doctest: +SKIP
        """
        from PIL import Image, ImageOps
        from supervision.assets import ImageAssets, download_assets

        from rfdetr import RFDETRNano
        from rfdetr.export._runtime.preprocess import preprocess_to_nchw

        asset_dir = tmp_path_factory.mktemp("int8_assets")
        cwd = Path.cwd()
        os.chdir(asset_dir)
        try:
            photo = Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
        finally:
            os.chdir(cwd)
        calibration = tmp_path_factory.mktemp("int8_calibration")
        with Image.open(photo) as image:
            image = image.convert("RGB")
            width, height = image.size
            for index, box in enumerate(
                [(0, 0, width, height), (0, 0, width // 2, height), (width // 2, 0, width, height)]
            ):
                crop = image.crop(box)
                crop.save(calibration / f"{index}.jpg")
                ImageOps.mirror(crop).save(calibration / f"{index}_flipped.jpg")
            resolution = int(RFDETRNano().model.resolution)
            example = preprocess_to_nchw(image, height=resolution, width=resolution)
        return calibration, example

    @pytest.fixture(scope="class")
    def engines(
        self, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray]:
        """Export FP16 and INT8 engines at batch 1, calibrating on the photo's crops and flips.

        Examples:
            A pytest fixture, and it needs a GPU and TensorRT.

            >>> fp16_engine, int8_engine, example = engines(photo, tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp("int8_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        return Path(fp16), Path(int8), example

    def test_engine_is_written_without_leftovers(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        _, int8, _ = engines
        assert int8.name == "rfdetr-nano_int8.trt"
        assert sorted(p.name for p in int8.parent.iterdir()) == ["rfdetr-nano.onnx", "rfdetr-nano_int8.trt"]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        fp16_path, int8_path, example = engines
        _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)

    @pytest.mark.parametrize("source", ["npy-path", "array"])
    def test_preprocessed_calibration_gives_a_matching_engine(
        self,
        source: str,
        photo: tuple[Path, np.ndarray],
        engines: tuple[Path, Path, np.ndarray],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        from rfdetr import RFDETRNano
        from rfdetr.export._runtime.calibration import calibration_batches

        calibration, example = photo
        fp16_path, _, _ = engines
        size = example.shape[-1]
        samples = np.concatenate(list(calibration_batches(calibration, height=size, width=size)))
        data: Any = samples
        if source == "npy-path":
            data = tmp_path_factory.mktemp("int8_samples") / "calibration.npy"
            np.save(data, samples)
        int8 = RFDETRNano().export(
            output_dir=str(tmp_path_factory.mktemp(f"int8_{source}")),
            format="tensorrt",
            quantization="int8",
            calibration_data=data,
            verbose=False,
        )
        assert Path(int8).name == "rfdetr-nano_int8.trt"
        _assert_confident_detections_match(fp16_path, Path(int8), example, batch=1)

    @pytest.mark.parametrize("batch", [2, 4])
    def test_static_batch_engine_matches_fp16_on_every_row(
        self, batch: int, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp(f"int8_batch_{batch}")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", batch_size=batch, verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            batch_size=batch,
            verbose=False,
        )
        _assert_confident_detections_match(Path(fp16), Path(int8), example, batch=batch)
