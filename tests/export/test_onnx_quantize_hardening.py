# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Hardening tests for static INT8 ONNX quantization (:mod:`rfdetr.export._onnx.quantize`)."""

import shutil
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx not installed; skip ONNX quantization tests")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from rfdetr.export._onnx.quantize import nodes_to_exclude, quantize_int8  # noqa: E402
from rfdetr.export._runtime import calibration  # noqa: E402


def _fail_after_partial_write(*args: Any, **kwargs: Any) -> None:
    """Stand in for an ORT stage that writes part of its output file, then dies.

    Both ``quant_pre_process`` and ``quantize_static`` take their output path as the second positional argument.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     try:
        ...         _fail_after_partial_write("in.onnx", f"{tmp}/out.onnx")
        ...     except RuntimeError:
        ...         print(Path(tmp, "out.onnx").read_bytes())
        b'partial'
    """
    Path(args[1]).write_bytes(b"partial")
    raise RuntimeError("simulated ONNX Runtime failure")


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stand-in ``onnxruntime.quantization``: pre-processing copies the graph, quantizing writes a stub.

    Lets the file-handling contract of :func:`quantize_int8` run without ONNX Runtime; it swaps the real package out
    even where one is installed, so every host exercises the same code path.

    Examples:
        A pytest fixture -- it needs ``monkeypatch`` to install the modules, so the example is not run:

        >>> fake_ort(monkeypatch).quantize_static  # doctest: +SKIP
    """
    root = types.ModuleType("onnxruntime")
    quantization = types.ModuleType("onnxruntime.quantization")
    shape_inference = types.ModuleType("onnxruntime.quantization.shape_inference")
    quantization.QuantFormat = types.SimpleNamespace(QDQ="QDQ")
    quantization.QuantType = types.SimpleNamespace(QInt8="QInt8")
    quantization.CalibrationMethod = types.SimpleNamespace(MinMax="MinMax")
    quantization.CalibrationDataReader = object
    quantization.quantize_static = lambda model_input, model_output, reader, **kwargs: Path(model_output).write_bytes(
        b"int8"
    )
    shape_inference.quant_pre_process = lambda source, target, **kwargs: shutil.copyfile(source, target)
    root.quantization = quantization
    quantization.shape_inference = shape_inference
    for module in (root, quantization, shape_inference):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return quantization


@pytest.fixture
def source_model(tmp_path: Path) -> Path:
    """An FP32 ``model.onnx`` with a static ``(1, 3, 8, 8)`` input, alone in its directory.

    Examples:
        A pytest fixture -- it needs ``tmp_path``, so the example is not run:

        >>> source_model(tmp_path).name  # doctest: +SKIP
        'model.onnx'
    """
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["out"], name="identity")],
        "model",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), str(path))
    return path


_CALIBRATION = np.zeros((2, 3, 8, 8), dtype=np.float32)


@pytest.mark.usefixtures("fake_ort")
class TestCalibrationSettings:
    """How :func:`quantize_int8` configures ONNX Runtime's calibration."""

    def test_pins_minmax_calibration(self, source_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``quantize_static`` is asked for min/max ranges explicitly, not whatever ORT's default becomes.

        The accuracy figures this mode is documented with were measured with min/max calibration; inheriting the default
        would let an ORT upgrade change the model silently.
        """
        received: dict[str, Any] = {}
        monkeypatch.setattr(
            "onnxruntime.quantization.quantize_static",
            lambda model_input, model_output, reader, **kwargs: (
                received.update(kwargs),
                Path(model_output).write_bytes(b"int8"),
            ),
        )
        quantize_int8(source_model, _CALIBRATION)
        assert received["calibrate_method"] == "MinMax"

    def test_checks_calibration_channels_against_the_graph(self, source_model: Path) -> None:
        """The ONNX quantizer passes the graph's channel count (input dim 1) to the calibration-array check.

        A single-channel array for a three-channel graph must fail up front, naming the shape the graph expects.
        """
        with pytest.raises(ValueError, match=r"\(N, 3, 8, 8\)"):
            quantize_int8(source_model, np.zeros((2, 1, 8, 8), dtype=np.float32))

    def test_warns_below_the_calibration_floor(self, source_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two calibration samples trigger the small-calibration-set warning on the ONNX path.

        The floor lives in the shared calibration module; this pins that the ONNX quantizer actually consults it.
        """
        warnings: list[str] = []
        monkeypatch.setattr(calibration.logger, "warning", warnings.append)
        quantize_int8(source_model, _CALIBRATION)
        assert len(warnings) == 1


def _identity_model(dims: list[int | str]) -> object:
    """Build an ``Identity`` model whose ``images`` input has *dims* (a string is a symbolic dimension).

    Args:
        dims: Input shape, e.g. ``[2, 3, 8, 8]`` or ``["batch", 3, 8, 8]``.

    Returns:
        An ``onnx.ModelProto``.

    Examples:
        >>> model = _identity_model(["batch", 3, 8, 8])
        >>> model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        'batch'
    """
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["out"], name="identity")],
        "model",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, dims)],
    )
    return helper.make_model(graph)


@pytest.mark.usefixtures("fake_ort")
class TestStaticBatch:
    """How calibration samples are fed to a graph traced at a fixed batch size."""

    @pytest.mark.parametrize(
        ("dims", "expected"),
        [
            pytest.param([1, 3, 8, 8], [(1, 3, 8, 8)] * 3, id="static-batch-1"),
            pytest.param([2, 3, 8, 8], [(2, 3, 8, 8)] * 2, id="static-batch-2-padded"),
            pytest.param(["batch", 3, 8, 8], [(1, 3, 8, 8)] * 3, id="symbolic-batch"),
        ],
    )
    def test_feeds_match_the_input_batch_dimension(
        self, dims: list[int | str], expected: list[tuple[int, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three samples reach ORT as feeds shaped like the graph input: stacked and padded for a static batch of 2.

        ORT rejects a ``(1, C, H, W)`` feed for a graph whose batch dimension is fixed at 2, so single-sample batches
        made static quantization impossible for any export traced with ``batch_size > 1``.
        """
        source = tmp_path / "model.onnx"
        onnx.save(_identity_model(dims), str(source))
        shapes: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            "onnxruntime.quantization.quantize_static",
            lambda model_input, model_output, reader, **kwargs: (
                shapes.extend(feed["images"].shape for feed in iter(reader.get_next, None)),
                Path(model_output).write_bytes(b"int8"),
            ),
        )
        quantize_int8(source, np.zeros((3, 3, 8, 8), dtype=np.float32))
        assert shapes == expected

    def test_rejects_symbolic_spatial_dimensions(self, tmp_path: Path) -> None:
        """A graph with a symbolic height and width is refused up front instead of calibrating at ``0x0``.

        The dimensions are read as integers; a symbolic one reads as 0 and would otherwise flow into preprocessing.
        """
        source = tmp_path / "model.onnx"
        onnx.save(_identity_model([1, 3, "height", "width"]), str(source))
        with pytest.raises(ValueError, match="static"):
            quantize_int8(source, np.zeros((2, 3, 8, 8), dtype=np.float32))


@pytest.mark.usefixtures("fake_ort")
class TestScratchFiles:
    """What :func:`quantize_int8` leaves on disk, on success and on failure."""

    def test_success_leaves_only_source_and_target(self, source_model: Path) -> None:
        """A successful run leaves the source and the ``_int8`` model, and no pre-processed or scratch file.

        The pre-processed graph used to be written beside the source; a scratch directory keeps it out of the export
        folder entirely.
        """
        quantize_int8(source_model, _CALIBRATION)
        assert sorted(path.name for path in source_model.parent.iterdir()) == ["model.onnx", "model_int8.onnx"]

    @pytest.mark.parametrize(
        "stage",
        ["onnxruntime.quantization.quantize_static", "onnxruntime.quantization.shape_inference.quant_pre_process"],
    )
    def test_failure_leaves_no_partial_artifact(
        self, stage: str, source_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stage dying mid-write leaves only the source behind -- no half-written ``_int8`` or ``_prep`` file.

        A truncated ``model_int8.onnx`` would load in a later run or be shipped by a script that only checks the file
        exists.
        """
        monkeypatch.setattr(stage, _fail_after_partial_write)
        with pytest.raises(RuntimeError, match="simulated"):
            quantize_int8(source_model, _CALIBRATION)
        assert [path.name for path in source_model.parent.iterdir()] == ["model.onnx"]

    def test_failure_keeps_previous_int8_model(self, source_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed re-run leaves an earlier ``_int8`` model byte-for-byte intact.

        The target is only replaced once the new model is complete, so a failure never destroys the last good one.
        """
        target = source_model.with_name("model_int8.onnx")
        target.write_bytes(b"previous")
        monkeypatch.setattr("onnxruntime.quantization.quantize_static", _fail_after_partial_write)
        with pytest.raises(RuntimeError, match="simulated"):
            quantize_int8(source_model, _CALIBRATION)
        assert target.read_bytes() == b"previous"


def _score_graph(key: str = "k", scaling: tuple[str, str] | None = None) -> object:
    """Build an attention block whose score ``MatMul`` reaches its ``Softmax`` through an optional scaling op.

    The block's own output runs through three ``Relu`` hops after the value multiply, so the head sweep never reaches
    ``scores`` and only the ``Softmax`` walk can exclude it.

    Args:
        key: Right-hand operand of ``scores``: ``"k"`` is a graph input (activation), ``"w"`` an initializer (weight).
        scaling: ``(op_type, operand)`` inserted between ``scores`` and the ``Softmax``; ``operand`` is ``"scale"``
            (initializer), ``"const"`` (``Constant`` node output) or ``"divisor"`` (graph input). ``None`` connects
            ``scores`` to the ``Softmax`` directly.

    Returns:
        An ``onnx.GraphProto``.

    Examples:
        >>> graph = _score_graph(scaling=("Div", "scale"))
        >>> [node.op_type for node in graph.node][:4]
        ['Constant', 'MatMul', 'Div', 'Softmax']
    """
    scalar = numpy_helper.from_array(np.array(2.0, dtype=np.float32))
    nodes = [
        helper.make_node("Constant", [], ["const"], name="const_node", value=scalar),
        helper.make_node("MatMul", ["q", key], ["score"], name="scores"),
    ]
    softmax_input = "score"
    if scaling is not None:
        op_type, operand = scaling
        nodes.append(helper.make_node(op_type, ["score", operand], ["scaled"], name="scaling"))
        softmax_input = "scaled"
    nodes += [
        helper.make_node("Softmax", [softmax_input], ["weights"], name="softmax"),
        helper.make_node("MatMul", ["weights", "value"], ["context"], name="values"),
        helper.make_node("Relu", ["context"], ["r1"], name="relu1"),
        helper.make_node("Relu", ["r1"], ["r2"], name="relu2"),
        helper.make_node("Relu", ["r2"], ["out"], name="relu3"),
    ]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [4, 4]) for name in ("q", "k", "value", "divisor")]
    outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [4, 4])]
    initializers = [
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="scale"),
        numpy_helper.from_array(np.eye(4, dtype=np.float32), name="w"),
    ]
    return helper.make_graph(nodes, "scores", inputs, outputs, initializer=initializers)


def _head_graph() -> object:
    """Build three graph outputs shaped like detection heads, each probing one property of the head walk.

    * ``boxes``: a three-layer bbox MLP (``layers.0``-``layers.2``) whose last layer sits four elementwise hops from
      the output (refine ``Add``, ``Sigmoid``, ``Unsqueeze``, ``Identity``) -- deeper than a fixed three-hop sweep.
    * ``features``: a ``MatMul`` behind a ``Conv``.
    * ``far``: a ``MatMul`` behind eleven ``Identity`` hops, past the walk's hop cap.

    Returns:
        An ``onnx.GraphProto``.

    Examples:
        >>> graph = _head_graph()
        >>> [output.name for output in graph.output]
        ['boxes', 'features', 'far']
    """
    nodes = [
        helper.make_node("Gemm", ["x", "w0"], ["h0"], name="layers.0"),
        helper.make_node("Relu", ["h0"], ["a0"], name="relu0"),
        helper.make_node("Gemm", ["a0", "w1"], ["h1"], name="layers.1"),
        helper.make_node("Relu", ["h1"], ["a1"], name="relu1"),
        helper.make_node("Gemm", ["a1", "w2"], ["delta"], name="layers.2"),
        helper.make_node("Add", ["delta", "reference"], ["refined"], name="refine"),
        helper.make_node("Sigmoid", ["refined"], ["coords"], name="sigmoid"),
        helper.make_node("Unsqueeze", ["coords", "axes"], ["expanded"], name="unsqueeze"),
        helper.make_node("Identity", ["expanded"], ["boxes"], name="boxes_identity"),
        helper.make_node("MatMul", ["x", "w3"], ["pre_conv"], name="behind_conv"),
        helper.make_node("Conv", ["pre_conv", "kernel"], ["conv_out"], name="conv"),
        helper.make_node("Relu", ["conv_out"], ["features"], name="relu_features"),
        helper.make_node("MatMul", ["x", "w4"], ["far_0"], name="far_matmul"),
    ]
    nodes += [helper.make_node("Identity", [f"far_{i}"], [f"far_{i + 1}"], name=f"far_id{i}") for i in range(10)]
    nodes.append(helper.make_node("Identity", ["far_10"], ["far"], name="far_out"))
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("boxes", "features", "far")]
    return helper.make_graph(nodes, "heads", inputs, outputs)


class TestHeadWalk:
    """Which head multiplies the walk back from the graph outputs holds in float."""

    @pytest.mark.parametrize(
        ("name", "excluded"),
        [
            pytest.param("layers.2", True, id="last-mlp-layer-beyond-three-hops"),
            pytest.param("layers.1", False, id="walk-stops-at-first-matmul"),
            pytest.param("layers.0", False, id="first-mlp-layer-stays-quantized"),
            pytest.param("refine", False, id="elementwise-ops-not-listed"),
            pytest.param("behind_conv", False, id="conv-ends-the-walk"),
            pytest.param("far_matmul", False, id="hop-cap-bounds-the-walk"),
        ],
    )
    def test_head_exclusion(self, name: str, excluded: bool) -> None:
        """Only the last ``MatMul``/``Gemm`` of a head is excluded, however deep behind elementwise ops it sits.

        A bbox MLP's final layer sits behind refinement arithmetic and a sigmoid; a fixed node-count sweep missed it
        while listing nodes that are never quantized anyway. The walk must also stop at the first multiply, at a
        ``Conv``, and at its hop cap, so the backbone stays quantizable.
        """
        assert (name in nodes_to_exclude(_head_graph())) is excluded


class TestAttentionScoreWalk:
    """Which multiplies the walk back from a ``Softmax`` holds in float."""

    @pytest.mark.parametrize(
        "scaling",
        [
            pytest.param(None, id="direct"),
            pytest.param(("Div", "scale"), id="div-by-initializer"),
            pytest.param(("Mul", "const"), id="mul-by-constant-node"),
            pytest.param(("Add", "scale"), id="add-constant-bias"),
        ],
    )
    def test_excludes_score_matmul_through_constant_scaling(self, scaling: tuple[str, str] | None) -> None:
        """``Q @ K^T`` stays float whether it feeds the ``Softmax`` directly or through constant scaling.

        The eager attention path divides the scores by ``sqrt(d)`` before the ``Softmax``; checking only the direct
        producer missed exactly that shape, while the direct case must keep being excluded as before.
        """
        assert "scores" in nodes_to_exclude(_score_graph(scaling=scaling))

    def test_does_not_walk_through_scaling_by_an_activation(self) -> None:
        """A ``Div`` whose divisor is itself an activation ends the walk, so the multiply stays quantizable.

        Only constant scaling is a known attention shape; walking through arbitrary math would widen the float set
        without evidence that those nodes are attention scores.
        """
        assert "scores" not in nodes_to_exclude(_score_graph(scaling=("Div", "divisor")))

    def test_keeps_weight_matmul_feeding_softmax_quantizable(self) -> None:
        """A multiply by an initializer that feeds a ``Softmax`` is a linear layer, not an attention score.

        Excluding it would keep an ordinary projection (e.g. deformable-attention weights) in float for no accuracy
        reason.
        """
        assert "scores" not in nodes_to_exclude(_score_graph(key="w"))
