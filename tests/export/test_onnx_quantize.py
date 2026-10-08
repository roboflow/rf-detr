# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for static INT8 quantization of ONNX exports (:mod:`rfdetr.export._onnx.quantize`)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._onnx.quantize import VALID_QUANTIZATIONS, nodes_to_exclude, quantize_int8
from rfdetr.export._runtime.calibration import calibration_batches
from rfdetr.export._tensorrt.exporter import TensorRTConfig
from rfdetr.export._tflite.exporter import TFLiteConfig

_IS_ONNX_INSTALLED = importlib.util.find_spec("onnx") is not None
_IS_ONNXRUNTIME_INSTALLED = importlib.util.find_spec("onnxruntime") is not None

onnx_only = pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed; skip ONNX quantization tests")
onnx_runtime_only = pytest.mark.skipif(
    not (_IS_ONNX_INSTALLED and _IS_ONNXRUNTIME_INSTALLED),
    reason="onnx/onnxruntime not installed; skip end-to-end quantization",
)

# The helper doctests build real ONNX graphs, so they need the package a class-level skipif cannot gate.
__doctest_requires__ = {
    (
        "_weights",
        "_deep_attention_model",
        "_dequantized_inputs",
        "_graph",
        "_matmul_chain",
        "_relu_tail",
        "_attention_graph",
    ): ["onnx"],
}


def _weights(name: str, shape: tuple[int, ...]) -> object:
    """Build a deterministic float32 initializer.

    Args:
        name: Initializer name.
        shape: Tensor shape.

    Returns:
        An ``onnx.TensorProto`` holding cosine-patterned values (no RNG, so reruns are identical).

    Examples:
        >>> _weights("w", (2, 2)).name
        'w'
    """
    from onnx import numpy_helper

    values = (np.cos(np.arange(int(np.prod(shape)))) * 0.5).reshape(shape).astype(np.float32)
    return numpy_helper.from_array(values, name)


def _deep_attention_model() -> object:
    """Build a model whose layers sit at very different distances from the output.

    ``projection`` (initializer weights) is separated from the output by many non-matmul nodes, so it is the layer
    quantization is meant to reach. ``scores`` multiplies two dynamic tensors and feeds a ``Softmax``; ``head``
    produces the graph output. ``scores`` consumes ``q`` and its transpose only, so no *quantizable* multiply shares
    ``q`` with it, which keeps "did ``scores`` stay float" a question about ``scores`` alone.

    Returns:
        An ``onnx.ModelProto`` taking a ``[1, 3, 4, 4]`` input.

    Examples:
        >>> model = _deep_attention_model()
        >>> len(model.graph.node) >= 6
        True
    """
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("MatMul", ["x", "w_proj"], ["q"], name="projection"),
        helper.make_node("Transpose", ["q"], ["k"], perm=[0, 1, 3, 2], name="transpose"),
        helper.make_node("MatMul", ["q", "k"], ["score"], name="scores"),
        helper.make_node("Softmax", ["score"], ["weights"], axis=-1, name="softmax"),
        helper.make_node("Relu", ["q"], ["v"], name="value_relu"),
        helper.make_node("MatMul", ["weights", "v"], ["context"], name="values"),
        helper.make_node("Relu", ["context"], ["r1"], name="r1"),
        helper.make_node("Add", ["r1", "b1"], ["a1"], name="a1"),
        helper.make_node("Relu", ["a1"], ["r2"], name="r2"),
        helper.make_node("Add", ["r2", "b2"], ["a2"], name="a2"),
        helper.make_node("Relu", ["a2"], ["r3"], name="r3"),
        helper.make_node("Add", ["r3", "b3"], ["a3"], name="a3"),
        helper.make_node("Relu", ["a3"], ["r4"], name="r4"),
        helper.make_node("MatMul", ["r4", "w_head"], ["out"], name="head"),
    ]
    initializers = [
        _weights("w_proj", (4, 4)),
        _weights("w_head", (4, 4)),
        _weights("b1", (4,)),
        _weights("b2", (4,)),
        _weights("b3", (4,)),
    ]
    graph = helper.make_graph(
        nodes,
        "deep_attention",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 4, 4])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # a version ONNX Runtime accepts regardless of which onnx release built the model
    return model


def _dequantized_inputs(model: object, node_name: str) -> list[bool]:
    """Report, per input of *node_name*, whether a ``DequantizeLinear`` produces it.

    A matrix multiply that was quantized reads every input through a ``DequantizeLinear``; one left in float reads at
    least one input directly.

    Args:
        model: An ``onnx.ModelProto``.
        node_name: Name of the node to inspect.

    Returns:
        One flag per input, in input order.

    Examples:
        >>> from onnx import helper
        >>> node = helper.make_node("Add", ["a", "b"], ["c"], name="add")
        >>> dq = helper.make_node("DequantizeLinear", ["qa", "s"], ["a"], name="dq")
        >>> graph = helper.make_graph([dq, node], "g", [], [])
        >>> _dequantized_inputs(helper.make_model(graph), "add")
        [True, False]
    """
    produced_by = {output: node.op_type for node in model.graph.node for output in node.output}
    node = next(node for node in model.graph.node if node.name == node_name)
    return [produced_by.get(name) == "DequantizeLinear" for name in node.input]


def _graph(node_groups: list[list[object]], inputs: list[str], outputs: list[str]) -> object:
    """Assemble a graph from groups of nodes, for tests that only read its structure.

    Args:
        node_groups: Lists of nodes, concatenated in order (each group must already be topologically sorted).
        inputs: Names of the graph inputs.
        outputs: Names of the graph outputs.

    Returns:
        An ``onnx.GraphProto``.

    Examples:
        >>> from onnx import helper
        >>> node = helper.make_node("Relu", ["x"], ["y"], name="r")
        >>> [n.name for n in _graph([[node]], ["x"], ["y"]).node]
        ['r']
    """
    from onnx import TensorProto, helper

    return helper.make_graph(
        [node for group in node_groups for node in group],
        "g",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in inputs],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1]) for name in outputs],
    )


def _matmul_chain(prefix: str, length: int) -> list[object]:
    """Build ``length`` chained ``MatMul`` nodes ``{prefix}0 ..

    {prefix}{length-1}`` reading ``x`` first.
        Args:
            prefix: Node-name prefix; node ``i`` outputs ``{prefix}{i}_out``.
            length: Number of nodes.

        Returns:
            The nodes, in topological order.

        Examples:
            >>> [n.name for n in _matmul_chain("m", 3)]
            ['m0', 'm1', 'm2']
    """
    from onnx import helper

    return [
        helper.make_node(
            "MatMul",
            ["x" if index == 0 else f"{prefix}{index - 1}_out", "w"],
            [f"{prefix}{index}_out"],
            name=f"{prefix}{index}",
        )
        for index in range(length)
    ]


def _relu_tail(source: str, length: int, prefix: str) -> tuple[list[object], str]:
    """Build a run of ``length`` ``Relu`` nodes after tensor *source*, as filler distance before a graph output.

    Args:
        source: Tensor the run starts from.
        length: Number of nodes.
        prefix: Node-name prefix.

    Returns:
        The nodes and the name of the run's final output tensor.

    Examples:
        >>> nodes, end = _relu_tail("a", 2, "t")
        >>> [n.name for n in nodes], end
        (['t0', 't1'], 't1_out')
    """
    from onnx import helper

    nodes = [
        helper.make_node(
            "Relu",
            [source if index == 0 else f"{prefix}{index - 1}_out"],
            [f"{prefix}{index}_out"],
            name=f"{prefix}{index}",
        )
        for index in range(length)
    ]
    return nodes, f"{prefix}{length - 1}_out"


def _attention_graph() -> object:
    """Build a graph shaped like one attention block followed by a head.

    ``projection`` sits deeper than the head sweep reaches and feeds no ``Softmax``, so it stands for the bulk of the
    network: the layers that are supposed to end up in 8-bit.

    Returns:
        An ``onnx.GraphProto`` with a deep ``MatMul``, a score ``MatMul`` feeding a ``Softmax``, a value ``MatMul``,
        and a head ``Gemm`` producing the graph output.

    Examples:
        >>> graph = _attention_graph()
        >>> sorted(node.name for node in graph.node)
        ['head', 'projection', 'scores', 'softmax', 'values']
    """
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("MatMul", ["x", "w_in"], ["query"], name="projection"),
        helper.make_node("MatMul", ["query", "key"], ["score"], name="scores"),
        helper.make_node("Softmax", ["score"], ["weights"], name="softmax"),
        helper.make_node("MatMul", ["weights", "value"], ["context"], name="values"),
        helper.make_node("Gemm", ["context", "proj"], ["logits"], name="head"),
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, 4])]
    return helper.make_graph(nodes, "attention", inputs, outputs)


@onnx_only
class TestNodeSelection:
    """Which nodes are held back from quantization."""

    def test_excludes_matmul_feeding_softmax(self) -> None:
        assert "scores" in nodes_to_exclude(_attention_graph())

    def test_excludes_head_producing_graph_output(self) -> None:
        assert "head" in nodes_to_exclude(_attention_graph())

    def test_keeps_deep_projection_quantizable(self) -> None:
        # The point of the exclusion list is to be narrow: a layer that neither feeds a Softmax nor sits near an
        # output must still be quantized, or the mode buys nothing.
        assert "projection" not in nodes_to_exclude(_attention_graph())

    def test_head_walk_stops_at_the_first_matmul_of_a_path(self) -> None:
        # "values" feeds the head Gemm; the walk from the output ends at that Gemm, so a multiply behind it is
        # not part of the head and stays quantizable.
        assert "values" not in nodes_to_exclude(_attention_graph())

    def test_excludes_output_end_of_a_matmul_chain_but_not_its_start(self) -> None:
        """In a long chain of multiplies only a run at the output end is held back; the start stays quantizable."""
        excluded = nodes_to_exclude(_graph([_matmul_chain("m", 12)], ["x"], ["m11_out"]))
        assert "m11" in excluded and "m0" not in excluded

    def test_excluded_chain_nodes_form_one_run_ending_at_the_output(self) -> None:
        """Whatever the head region's size, it is contiguous: no gap between excluded chain nodes."""
        chain_length = 12
        excluded = set(nodes_to_exclude(_graph([_matmul_chain("m", chain_length)], ["x"], ["m11_out"])))
        indices = sorted(int(name.removeprefix("m")) for name in excluded)
        assert indices == list(range(indices[0], chain_length))

    def test_excludes_both_branches_of_a_diamond_and_lists_the_shared_trunk_once(self) -> None:
        """A multiply reachable from the output along two paths is listed once; both branch multiplies are held back."""
        from onnx import helper

        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["trunk_out"], name="trunk"),
            helper.make_node("MatMul", ["trunk_out", "w"], ["left_out"], name="left"),
            helper.make_node("MatMul", ["trunk_out", "w"], ["right_out"], name="right"),
            helper.make_node("Add", ["left_out", "right_out"], ["out"], name="join"),
        ]
        excluded = nodes_to_exclude(_graph([nodes], ["x"], ["out"]))
        assert {"left", "right"} <= set(excluded) and len(excluded) == len(set(excluded))

    def test_excludes_every_producer_of_a_multi_output_graph_in_sorted_order(self) -> None:
        """Each graph output's producing multiply is held back, and names come back sorted, not in graph order.

        The nodes are declared ``z_head`` first, so a result in discovery order would not equal the sorted list.
        """
        from onnx import helper

        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["logits"], name="z_head"),
            helper.make_node("MatMul", ["x", "w"], ["boxes"], name="a_head"),
        ]
        assert nodes_to_exclude(_graph([nodes], ["x"], ["logits", "boxes"])) == ["a_head", "z_head"]

    def test_softmax_without_a_producer_is_tolerated_and_excludes_nothing_far_away(self) -> None:
        """A Softmax reading a graph input has no producer to exclude; a distant multiply is still quantizable."""
        from onnx import helper

        far = [helper.make_node("MatMul", ["x", "w"], ["far_out"], name="far")]
        far_tail, far_end = _relu_tail("far_out", 11, "far_tail")  # beyond the bounded head walk
        attention = [helper.make_node("Softmax", ["external_scores"], ["probs"], name="attention")]
        graph = _graph([far, far_tail, attention], ["x", "external_scores"], [far_end, "probs"])
        assert "far" not in nodes_to_exclude(graph)

    def test_softmax_fed_by_a_non_matmul_does_not_exclude_that_producer(self) -> None:
        """Only multiplies feeding a Softmax are attention scores; an elementwise producer is left alone."""
        from onnx import helper

        nodes = [helper.make_node("Relu", ["x"], ["pre"], name="pre_softmax")]
        nodes.append(helper.make_node("Softmax", ["pre"], ["probs"], name="softmax"))
        tail, tail_end = _relu_tail("probs", 8, "tail")
        assert "pre_softmax" not in nodes_to_exclude(_graph([nodes, tail], ["x"], [tail_end]))

    def test_returns_each_name_once_in_sorted_order_for_the_attention_graph(self) -> None:
        """The attention graph's result is sorted and duplicate-free."""
        excluded = nodes_to_exclude(_attention_graph())
        assert excluded == sorted(excluded) and len(excluded) == len(set(excluded))


class TestCalibrationBatches:
    """Turning user-supplied calibration data into model inputs."""

    def test_array_yields_one_batch_per_sample(self) -> None:
        batches = list(calibration_batches(np.zeros((3, 3, 8, 8), dtype=np.float32), height=8, width=8))
        assert len(batches) == 3

    def test_array_batches_carry_a_leading_batch_dimension(self) -> None:
        batches = list(calibration_batches(np.zeros((2, 3, 8, 8), dtype=np.float32), height=8, width=8))
        assert batches[0].shape == (1, 3, 8, 8)

    def test_rejects_array_with_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="rank 4"):
            list(calibration_batches(np.zeros((3, 8, 8), dtype=np.float32), height=8, width=8))

    def test_rejects_array_with_mismatched_resolution(self) -> None:
        with pytest.raises(ValueError, match="expects 16x16"):
            list(calibration_batches(np.zeros((1, 3, 8, 8), dtype=np.float32), height=16, width=16))

    def test_rejects_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            list(calibration_batches(tmp_path / "absent", height=8, width=8))

    def test_rejects_non_npy_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "data.txt"
        text_file.write_text("not an array")
        with pytest.raises(ValueError, match=r"\.npy"):
            list(calibration_batches(text_file, height=8, width=8))

    def test_rejects_directory_without_images(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No calibration images"):
            list(calibration_batches(tmp_path, height=8, width=8))

    def test_reads_npy_file(self, tmp_path: Path) -> None:
        array_path = tmp_path / "calib.npy"
        np.save(array_path, np.zeros((2, 3, 8, 8), dtype=np.float32))
        assert len(list(calibration_batches(array_path, height=8, width=8))) == 2

    def test_preprocesses_images_from_a_directory(self, tmp_path: Path) -> None:
        from PIL import Image

        for name in ("a.jpg", "b.jpg"):
            Image.new("RGB", (32, 24)).save(tmp_path / name)
        batches = list(calibration_batches(tmp_path, height=8, width=8))
        assert [batch.shape for batch in batches] == [(1, 3, 8, 8), (1, 3, 8, 8)]

    def test_directory_reading_honours_max_images(self, tmp_path: Path) -> None:
        from PIL import Image

        for index in range(4):
            Image.new("RGB", (32, 24)).save(tmp_path / f"{index}.jpg")
        assert len(list(calibration_batches(tmp_path, height=8, width=8, max_images=2))) == 2


class TestConfigValidation:
    """Refusals that happen before any work on the model."""

    def test_int8_is_a_valid_mode(self) -> None:
        assert "int8" in VALID_QUANTIZATIONS

    def test_rejects_unknown_quantization_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization="fp16"))

    def test_rejects_int8_without_calibration_data(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires calibration_data"):
            OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization="int8"))

    def test_accepts_int8_with_calibration_data(self, tmp_path: Path) -> None:
        samples = np.zeros((1, 3, 8, 8), dtype=np.float32)
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization="int8", calibration_data=samples))
        assert exporter.config.quantization == "int8"

    @pytest.mark.parametrize("quantization", [None, "fp32"])
    def test_accepts_float_modes_without_calibration_data(self, quantization: str | None, tmp_path: Path) -> None:
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization=quantization))
        assert exporter.config.quantization == quantization


class TestSettingPlumbing:
    """The keywords ``RFDETR.export`` forwards to this format."""

    def test_quantization_settings_reach_the_config(self) -> None:
        config = OnnxExporter.build_config(quantization="int8", calibration_data="images/", max_images=32)
        assert (config.quantization, config.calibration_data, config.max_images) == ("int8", "images/", 32)

    @pytest.mark.parametrize(
        "two_stage_config",
        [
            pytest.param(
                TFLiteConfig(quantization="int8", calibration_data=np.zeros((1, 3, 8, 8), dtype=np.float32)),
                id="tflite-int8-with-calibration",
            ),
            pytest.param(TensorRTConfig(), id="tensorrt"),
        ],
    )
    def test_two_stage_onnx_stage_does_not_inherit_quantization(self, two_stage_config: object) -> None:
        """The intermediate ONNX stage of TFLite/TensorRT carries no INT8 request or calibration data.

        TFLite's own ``quantization="int8"`` is dynamic-range and means something else; leaking it into the ONNX stage
        would make that stage quantize statically (or demand calibration data) as a side effect of a different format.
        """
        derived = two_stage_config.onnx_stage()
        assert (derived.quantization, derived.calibration_data) == (None, None)

    @pytest.mark.parametrize(
        "two_stage_config",
        [
            pytest.param(
                TFLiteConfig(quantization="int8", calibration_data=np.zeros((1, 3, 8, 8), dtype=np.float32)),
                id="tflite-int8-with-calibration",
            ),
            pytest.param(TensorRTConfig(), id="tensorrt"),
        ],
    )
    def test_derived_onnx_stage_builds_an_exporter(self, two_stage_config: object) -> None:
        """An exporter built from a derived stage config does not refuse for missing calibration data."""
        exporter = OnnxExporter(two_stage_config.onnx_stage())
        assert exporter.config.quantization is None


@onnx_runtime_only
class TestQuantizeInt8EndToEnd:
    """``quantize_int8`` run against a real ONNX Runtime on a small synthetic graph.

    Unmarked on purpose: ``e2e_onnx`` tests are by convention also ``integration`` and so skipped by the default CPU CI
    job, but this graph is a few nodes and the runtime is in the ``tests`` group, so it can run there.
    """

    @pytest.fixture
    def onnxruntime(self) -> object:
        """The ``onnxruntime`` module."""
        import onnxruntime

        return onnxruntime

    @pytest.fixture
    def run(self, onnxruntime: object, tmp_path: Path) -> tuple[Path, Path, np.ndarray, object]:
        """Quantize the synthetic model; return source path, INT8 path, calibration data and the loaded INT8 model."""
        import onnx

        source = tmp_path / "model.onnx"
        onnx.save(_deep_attention_model(), str(source))
        data = np.stack([np.sin(np.arange(48).reshape(3, 4, 4) * (index + 1)) for index in range(4)]).astype(np.float32)
        target = quantize_int8(source, data)
        return source, target, data, onnx.load(str(target))

    def test_writes_int8_model_beside_source(self, run: tuple[Path, Path, np.ndarray, object]) -> None:
        """The quantized copy lands next to the source with an ``_int8`` suffix."""
        source, target, _, _ = run
        assert target == source.with_name("model_int8.onnx") and target.is_file()

    def test_leaves_only_source_and_int8_files_behind(self, run: tuple[Path, Path, np.ndarray, object]) -> None:
        """No scratch file from the preprocessing step survives in the output directory."""
        source, target, _, _ = run
        assert sorted(path.name for path in source.parent.iterdir()) == sorted([source.name, target.name])

    def test_leaves_source_graph_unquantized(self, run: tuple[Path, Path, np.ndarray, object]) -> None:
        """The FP32 source stays a float graph: it is the accuracy baseline."""
        import onnx

        source, _, _, _ = run
        op_types = {node.op_type for node in onnx.load(str(source)).graph.node}
        assert "QuantizeLinear" not in op_types and "DequantizeLinear" not in op_types

    def test_quantizes_initializer_weight_matmul_far_from_output(
        self, run: tuple[Path, Path, np.ndarray, object]
    ) -> None:
        """A multiply with initializer weights far from any output and any Softmax gets QDQ nodes on both inputs."""
        _, _, _, model = run
        assert _dequantized_inputs(model, "projection") == [True, True]

    def test_keeps_attention_score_matmul_in_float(self, run: tuple[Path, Path, np.ndarray, object]) -> None:
        """The multiply feeding the ``Softmax`` is not quantized, even though both its inputs are dynamic."""
        _, _, _, model = run
        assert not any(_dequantized_inputs(model, "scores"))

    def test_keeps_head_matmul_in_float(self, run: tuple[Path, Path, np.ndarray, object]) -> None:
        """The multiply producing the graph output keeps its raw initializer weight (no ``DequantizeLinear``)."""
        _, _, _, model = run
        assert _dequantized_inputs(model, "head")[1] is False

    def test_quantized_model_loads_and_runs_in_onnxruntime(
        self, onnxruntime: object, run: tuple[Path, Path, np.ndarray, object]
    ) -> None:
        """The INT8 model builds a CPU session (per-tensor weights keep the dynamic multiply loadable) and runs."""
        _, target, data, _ = run
        session = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        (output,) = session.run(None, {"x": data[:1]})
        assert output.shape == (1, 3, 4, 4) and np.isfinite(output).all()
