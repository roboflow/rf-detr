# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the INT8 graph rewrite: graph walkers, Q/DQ insertion, opset lifting and the patch-embedding ``MatMul``.

They run on CPU with ``onnx``, on small synthetic graphs that mirror RF-DETR's naming.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import quantize
from tests._markers import _requires_onnx_and_onnxruntime, _requires_onnxconverter_common, _requires_onnxruntime
from tests.export.tensorrt.helpers import (
    BACKBONE,
    DECODER,
    HEAD,
    _attention_and_mlp,
    _names,
    _outputs_before_and_after_lift,
    _weight,
)

# Module-level on purpose: parametrized constants and graph builders touch ONNX while the module is collected, so a
# guarded import would raise ``NameError`` instead of skipping.
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


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
        """_weight_name returns the weight only for a named, untransposed-A weight layer."""
        assert (quantize._weight_name(node, _weights("w", dims=dims)) == "w") is plannable

    @pytest.mark.parametrize(("node", "dims", "_plannable", "reads"), CASES)
    def test_holds_weight_ignores_everything_but_the_constant(
        self, node: Any, dims: tuple[int, ...], _plannable: bool, reads: bool
    ) -> None:
        """_holds_weight is true for any weight op that reads the constant, named or not."""
        assert quantize._holds_weight(node, _weights("w", dims=dims)) is reads

    def test_weight_reached_through_an_identity_is_constant(self) -> None:
        """An Identity or Constant output counts as a constant weight."""
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
        """The scores multiply is found behind scale, mask and cast ops."""
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
        """No scores multiply is found for a missing producer, a weight multiply or a blocking op."""
        softmax = next(node for node in nodes if node.op_type == "Softmax")
        assert quantize._scores_matmul(softmax, _producers(nodes), _weights("w")) is None

    def test_scores_are_found_when_a_node_is_reached_twice(self) -> None:
        """A node reached by two paths does not break the scores search."""
        nodes = [
            _node("MatMul", ["q", "k"], "s0", "scores"),
            _node("Add", ["s0", "s0"], "s", "twice"),
            _node("Softmax", ["s"], "p", "softmax"),
        ]
        assert quantize._scores_matmul(nodes[2], _producers(nodes), {}).name == "scores"

    @pytest.mark.parametrize("passthrough", [[], ["Cast"], ["Identity"], ["Cast", "Identity"]])
    def test_context_is_found_behind_casts(self, passthrough: list[str]) -> None:
        """The context multiply is found behind Cast and Identity ops."""
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
        """No context multiply is found unless the probabilities feed exactly one first operand."""
        assert quantize._context_matmul(nodes[0], _readers(nodes)) is None

    def test_projection_is_traced_through_unnamed_nodes(self) -> None:
        """A projection is traced through unnamed Reshape and Transpose nodes."""
        weights = {"w": numpy_helper.from_array(np.zeros((4, 4), np.float32), "w")}
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["q"], "q_proj"),
            helper.make_node("Reshape", ["q", "shape"], ["q_heads"]),
            helper.make_node("Transpose", ["q_heads"], ["q_t"]),
        ]
        producers = {output: node for node in nodes for output in node.output}
        assert quantize._trace_projection("q_t", producers, weights) == "q_proj"

    @pytest.mark.parametrize(
        "follower",
        ["Abs", "Add", "Cast", "Clip", "Concat", "Div", "Dropout", "Identity", "Mul", "Relu", "Reshape", "Slice", "Sub"]
        + ["Transpose", "Where"],
    )
    def test_softmax_reaches_a_multiply_through_any_op(self, follower: str) -> None:
        """A Softmax reaches an activation multiply through any follower op."""
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
                    _node("MatMul", ["p", "w"], "m", "projection"),
                    _node("MatMul", ["m", "v"], "c", "later_block"),
                ],
                id="blocked-by-a-projection",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Gemm", ["p", "w"], "m", "gemm_projection"),
                    _node("MatMul", ["m", "v"], "c", "later_block"),
                ],
                id="blocked-by-a-gemm",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Conv", ["p", "w"], "m", "conv_projection"),
                    _node("MatMul", ["m", "v"], "c", "later_block"),
                ],
                id="blocked-by-a-convolution",
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
        """A Softmax that reaches no activation multiply is not attention."""
        assert quantize._feeds_activation_matmul(nodes[0], _readers(nodes), _weights("w")) is False

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Identity", "Mul", "Transpose"])
    def test_projection_is_found_behind_each_passthrough_op(self, op: str) -> None:
        """A projection is found behind each pass-through op."""
        nodes = [_node("MatMul", ["x", "w"], "q0", "q_proj"), _node(op, ["q0", "bias"], "q", "after")]
        assert quantize._trace_projection("q", _producers(nodes), _weights("w", "bias")) == "q_proj"

    @pytest.mark.parametrize("op", ["Reshape", "Squeeze", "Unsqueeze"])
    def test_projection_is_found_through_the_data_input_of_a_shape_op(self, op: str) -> None:
        """A projection is traced through the data input of a shape op, not its shape operand."""
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
        """No projection is found behind a graph input, shape arithmetic or an activation."""
        assert quantize._trace_projection("q", _producers(nodes), _weights("w")) is None

    @pytest.mark.parametrize("hops", [0, 1, 3])
    def test_output_projection_is_found_within_three_shape_ops(self, hops: int) -> None:
        """The output projection is found within three shape ops of the context."""
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
        """No output projection is found unless exactly one weight multiply reads the context."""
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w")) is None

    def test_output_projection_is_not_found_beyond_three_shape_ops(self) -> None:
        """The output projection search stops after three shape ops."""
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
        """Erf and Gelu forms both lead from fc1 to the second projection."""
        nodes, fc1 = self._chain(activation)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}

    def test_relu_is_not_a_gelu(self) -> None:
        """A ReLU activation has no GELU consumers."""
        nodes, fc1 = self._chain("Relu")
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == set()

    @pytest.mark.parametrize("after", [["Identity"], ["Cast", "Reshape"]])
    def test_any_op_between_the_gelu_and_its_consumer_counts_as_a_consumer(self, after: list[str]) -> None:
        """The first op after the GELU counts as its consumer."""
        nodes, fc1 = self._chain("Gelu", after=after)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"after0"}

    @pytest.mark.parametrize("consumer_name", ["", "fc2"])
    def test_a_consumer_is_listed_by_name_even_when_it_cannot_be_planned(self, consumer_name: str) -> None:
        """A consumer is listed by its name, empty included."""
        nodes, fc1 = self._chain("Gelu", consumer_name=consumer_name)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {consumer_name}

    def test_a_consumer_reading_the_gelu_twice_is_listed_once(self) -> None:
        """A consumer reading the GELU output twice is listed once."""
        nodes, fc1 = self._chain("Gelu")
        nodes[-1].input[0] = "g"
        nodes[-1].input.append("g")
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}

    def test_the_branch_that_skips_the_gelu_is_not_a_consumer(self) -> None:
        """A reader of the pre-GELU tensor is not a GELU consumer."""
        nodes, fc1 = self._chain("Gelu")
        nodes.append(_node("MatMul", ["h", "w2"], "skip", "skip_fc"))
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}


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
        """A weight read by two layers gets its own quantize pair per layer."""

        def share(model: Any) -> None:
            next(n for n in model.graph.node if n.name.endswith("k/MatMul")).input[1] = "wq"

        model, plan = _quantized_with(share)
        by_name = {n.name: n for n in model.graph.node}
        readers = {by_name[f"{BACKBONE}{layer}/MatMul"].input[1] for layer in "qk"}
        assert len(readers) == 2 and all(r.endswith("_dequantized") for r in readers)
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize("tensor", ["c2", "wq", "x"])
    def test_quantized_tensor_that_is_also_a_graph_output_stays_an_output(self, tensor: str) -> None:
        """A quantized tensor that is also a graph output stays an output of valid ONNX."""
        shapes = {"c2": [1, 8, 32], "x": [1, 8, 32], "wq": [32, 32]}

        def expose(model: Any) -> None:
            elem = model.graph.output[0].type.tensor_type.elem_type
            model.graph.output.append(helper.make_tensor_value_info(tensor, elem, shapes[tensor]))

        model, _ = _quantized_with(expose)
        assert tensor in [o.name for o in model.graph.output]
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_tensor_read_inside_a_subgraph_keeps_its_original_binding(self) -> None:
        """A tensor read inside a subgraph keeps its original name there."""

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
        """Every planned activation and weight input is read through a DequantizeLinear."""
        model, plan, _ = ranged
        producer = {output: node for node in model.graph.node for output in node.output}
        by_name = {node.name: node for node in model.graph.node}
        read = [producer[by_name[name].input[index]].op_type for name, index in plan.activations]
        read += [producer[by_name[name].input[1]].op_type for name in plan.weighted]
        assert set(read) == {"DequantizeLinear"}

    def test_activation_scale_is_its_range_over_127(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        """An activation scale is its measured range over 127, stored in FP16."""
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
        """Activation zero points are INT8 zero."""
        model, _, _ = ranged
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        points = {
            node.input[2] for node in model.graph.node if node.op_type == "QuantizeLinear" and len(node.input) > 2
        }
        assert {(inits[name].dtype, int(inits[name])) for name in points} == {(np.dtype(np.int8), 0)}

    def test_convolution_weight_is_quantized_per_output_channel(self) -> None:
        """A convolution weight is quantized per output channel on axis 0."""
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
        """The rewritten graph passes the ONNX checker and strict shape inference."""
        model, _ = quantized
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_opset_is_lifted(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """The rewritten graph is at opset 19."""
        model, _ = quantized
        assert [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")] == [19]

    def test_one_quantize_per_activation_tensor(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Each distinct activation tensor is quantized once and shared by its readers."""
        model, _ = quantized
        quantized_inputs = [node.input[0] for node in model.graph.node if node.op_type == "QuantizeLinear"]
        # x feeds q, k and v through one pair; then the four batched-multiply inputs, o's input, fc1's and fc2's.
        assert sorted(quantized_inputs) == sorted(["x", "q2", "k2", "p", "v2", "c2", "a", "g"])

    def test_scales_are_fp16(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Every quantize and dequantize scale is FP16."""
        model, _ = quantized
        inits = {init.name: init for init in model.graph.initializer}
        scales = {node.input[1] for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")}
        assert {inits[name].data_type for name in scales} == {TensorProto.FLOAT16}

    def test_weights_are_int8_per_output_channel(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Weights are stored INT8 and dequantized per output channel on axis 1."""
        model, plan = quantized
        inits = {init.name: init for init in model.graph.initializer}
        weight_dq = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits]
        assert len(weight_dq) == len(plan.weighted)
        assert {(inits[n.input[0]].data_type, n.attribute[0].i) for n in weight_dq} == {(TensorProto.INT8, 1)}

    def test_weight_zero_point_is_int8_zero(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Weight zero points are INT8 zero."""
        model, _ = quantized
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        weight_dq = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits]
        points = [inits[n.input[2]] for n in weight_dq]
        assert {(point.dtype, int(np.abs(point).max())) for point in points} == {(np.dtype(np.int8), 0)}

    def test_dequantized_weight_matches_the_original(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """A dequantized weight is within half a quantization step of the original."""
        model, _ = quantized
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0].startswith("w1"))
        scales = inits[dq.input[1]].astype(np.float32)  # one per output channel (axis 1)
        restored = inits[dq.input[0]].astype(np.float32) * scales
        original = numpy_helper.to_array(_weight("w1", 32, 64, np.float16, 5)).astype(np.float32)
        # Rounded against the stored FP16 scale, every weight is within half a step of its original value.
        assert float((np.abs(restored - original) / scales).max()) <= 0.5 + 1e-3

    def test_scales_stay_positive_in_fp16(self) -> None:
        """Scales never fall below the smallest normal FP16 value, even for a near-dead channel."""
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
        """An initializer read only inside a subgraph survives the rewrite."""
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
        """A Gemm weight is quantized along the output-channel axis chosen by transB."""
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
        """Names the rewrite generates do not collide with existing tensors or initializers."""
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
        """Probe output names avoid names already used by graph outputs and initializers."""
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
        """Lifting to opset 19 moves a subgraph reduction's axes attribute to an input."""
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
        """Every reduction op moves its axes attribute to an input when lifted to opset 19."""
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
        """A reduction without an axes attribute is left untouched by the lift."""
        model = self._reduction_model(op, 17, axes=None)
        quantize._lift_opset(model)
        assert (len(model.graph.node[0].input), len(model.graph.initializer)) == (1, 0)

    def test_axes_input_does_not_reuse_a_taken_name(self) -> None:
        """The new axes initializer takes a fresh name when the obvious one is taken."""
        model = self._reduction_model("ReduceMax", 17, axes=[1], taken="reduce_axes")
        quantize._lift_opset(model)
        new = model.graph.node[0].input[1]
        assert new != "reduce_axes" and sorted(i.name for i in model.graph.initializer) == sorted([new, "reduce_axes"])

    @pytest.mark.parametrize(
        ("inputs", "expected"),
        [pytest.param(["x"], [2], id="unsized"), pytest.param(["x", "sizes"], [], id="sized")],
    )
    def test_split_without_sizes_gets_its_output_count(self, inputs: list[str], expected: list[int]) -> None:
        """A Split without sizes gets num_outputs on lift; one with sizes does not."""
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
        """A graph already at opset 19 is not modified."""
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
        """Only the default domain is lifted; custom domains keep their version."""
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
        """An op whose meaning changed between opsets is refused instead of silently lifted."""
        pad = helper.make_node("Pad", ["x", "pads"], ["y"], "pad")
        graph = helper.make_graph(
            [pad],
            "pad",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
            [numpy_helper.from_array(np.array([1, 1], np.int64), "pads")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with pytest.raises(ValueError) as exc_info:
            quantize._lift_opset(model)
        assert "'pad' (Pad) changed meaning" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("op", "opset", "refused"),
        [
            ("Resize", 10, True),
            ("Resize", 12, True),
            ("Resize", 13, False),
            ("ScatterND", 13, True),
            ("ScatterND", 15, True),
            ("ScatterND", 16, False),
        ],
    )
    def test_resize_and_scatter_nd_are_lifted_only_from_the_opset_they_were_checked_from(
        self, op: str, opset: int, refused: bool
    ) -> None:
        """``Resize`` (before opset 13) and ``ScatterND`` (before 16) did mean something else than at opset 19.

        Their opset-17 behavior was checked against opset 19, so only a graph at or above the opset from which that
        holds may have them lifted; a hand-made graph below it is refused instead of failing late in the runtime.
        """
        node = helper.make_node(op, ["x", "a", "b"], ["y"], "node")
        graph = helper.make_graph(
            [node],
            "old",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        if refused:
            with pytest.raises(ValueError, match=f"\\({op}\\) changed meaning"):
                quantize._lift_opset(model)
        else:
            quantize._lift_opset(model)
            assert [imported.version for imported in model.opset_import if imported.domain == ""] == [
                quantize._FP16_QDQ_OPSET
            ]

    def test_unknown_op_is_refused(self) -> None:
        """An op ONNX does not define is refused with a ValueError naming the node, not onnx's own SchemaError.

        ``int8_source_graph`` promises a ValueError for a graph it cannot lift; onnx's schema lookup raises its own
        error type, which is no ValueError.
        """
        graph = helper.make_graph(
            [helper.make_node("NotAnOp", ["x"], ["y"], "mystery")],
            "unknown",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with pytest.raises(ValueError) as exc_info:
            quantize._lift_opset(model)
        assert "'mystery' (NotAnOp) is not an ONNX op" in str(exc_info.value)

    @_requires_onnx_and_onnxruntime
    @pytest.mark.parametrize("coordinate_mode", ["half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric"])
    @pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
    def test_resize_gives_the_same_values_at_opset_19(self, mode: str, coordinate_mode: str) -> None:
        """Resize outputs match before and after the lift for every mode and coordinate mode."""
        node = helper.make_node(
            "Resize", ["x", "", "", "sizes"], ["y"], "resize", mode=mode, coordinate_transformation_mode=coordinate_mode
        )
        graph = helper.make_graph(
            [node],
            "resize",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 7, 9])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 20, 31])],
            [numpy_helper.from_array(np.array([1, 2, 20, 31], np.int64), "sizes")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x = np.random.default_rng(0).standard_normal((1, 2, 7, 9)).astype(np.float32)
        before, after = _outputs_before_and_after_lift(model, {"x": x})
        np.testing.assert_allclose(after[0], before[0], atol=1e-6)

    @_requires_onnx_and_onnxruntime
    @pytest.mark.parametrize("reduction", ["none", "add", "mul"])
    def test_scatter_nd_gives_the_same_values_at_opset_19(self, reduction: str) -> None:
        """ScatterND outputs match before and after the lift for every reduction."""
        node = helper.make_node("ScatterND", ["data", "indices", "updates"], ["y"], "scatter", reduction=reduction)
        graph = helper.make_graph(
            [node],
            "scatter",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 3])],
            [
                numpy_helper.from_array(np.array([[1], [3], [1]], np.int64), "indices"),
                numpy_helper.from_array(np.arange(1, 10, dtype=np.float32).reshape(3, 3), "updates"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
        data = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
        before, after = _outputs_before_and_after_lift(model, {"data": data})
        np.testing.assert_allclose(after[0], before[0], atol=1e-6)


def _patch_embedding(
    *,
    channels: int = 3,
    patch: int = 16,
    size: int = 32,
    out_channels: int = 8,
    bias: bool = True,
    empty_bias: bool = False,
    prefix: str = BACKBONE,
    batch: int | str = 1,
    height: int | str | None = None,
    width: int | None = None,
    kernel: tuple[int, int] | None = None,
    **attributes: Any,
) -> Any:
    """Build ``image -> Conv -> features`` as a patch embedding: kernel and stride *kernel*, no padding.

    Args:
        channels: Input channels of the image.
        patch: Kernel size and stride, in pixels, when *kernel* is not given.
        size: Image height and width, unless *height* or *width* is given.
        out_channels: Output channels of the kernel.
        bias: Give the ``Conv`` a bias.
        empty_bias: Name the bias input with an empty string, how ONNX spells an omitted optional input.
        prefix: Node-name prefix of the ``Conv``.
        batch: Batch size of the image; a string makes it symbolic.
        height: Overrides the declared image height; a string makes it symbolic.
        width: Overrides the declared image width.
        kernel: ``(rows, columns)`` of a rectangular kernel and stride.
        **attributes: ``Conv`` attributes that replace the patch-embedding ones.

    Returns:
        A checked-shape ``ModelProto`` whose output ``features`` is the ``Conv``'s output.

    Examples:
        >>> model = _patch_embedding(channels=2, patch=4, size=8)
        >>> [node.op_type for node in model.graph.node], model.graph.output[0].name
        (['Conv'], 'features')
    """
    rng = np.random.default_rng(0)
    rows, columns = kernel or (patch, patch)
    high, wide = height or size, width or size
    inits = [
        numpy_helper.from_array(rng.standard_normal((out_channels, channels, rows, columns)).astype(np.float32), "w")
    ]
    inputs = ["image", "w"]
    if bias:
        inits.append(numpy_helper.from_array(rng.standard_normal(out_channels).astype(np.float32), "b"))
        inputs.append("b")
    elif empty_bias:
        inputs.append("")
    conv = {"kernel_shape": [rows, columns], "strides": [rows, columns], **attributes}
    grid = ["out_height" if isinstance(high, str) else high // rows, wide // columns]
    graph = helper.make_graph(
        [helper.make_node("Conv", inputs, ["features"], f"{prefix}patch/Conv", **conv)],
        "patch_embedding",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [batch, channels, high, wide])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [batch, out_channels, *grid])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _with_detector_outputs(model: Any) -> Any:
    """Make *model*'s ``features`` the source of ``dets`` and ``labels`` and return it, so that INT8 accepts the graph.

    Examples:
        >>> model = _with_detector_outputs(_patch_embedding(channels=2, patch=4, size=8))
        >>> [output.name for output in model.graph.output]
        ['dets', 'labels']
    """
    model.graph.node.extend([helper.make_node("Identity", ["features"], [name], name) for name in ("dets", "labels")])
    del model.graph.output[:]
    model.graph.output.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("dets", "labels")
    )
    return model


class TestPatchEmbeddingAsMatmul:
    """The 3-channel patch-embedding ``Conv`` becomes a ``MatMul`` that INT8 can run, with the same output."""

    @_requires_onnxruntime
    @pytest.mark.parametrize(
        ("batch", "channels", "patch", "height", "width"),
        [
            pytest.param(1, 3, 16, 32, 32, id="rgb-16-pixel-patches"),
            pytest.param(1, 3, 12, 24, 24, id="rgb-12-pixel-patches"),
            pytest.param(2, 3, 16, 32, 48, id="batch-2-non-square"),
            pytest.param(3, 2, 4, 8, 16, id="batch-3-two-channels-non-square"),
            pytest.param(1, 1, 8, 16, 16, id="grayscale"),
        ],
    )
    @pytest.mark.parametrize("bias", [True, False])
    def test_features_match_the_convolution(
        self, batch: int, channels: int, patch: int, height: int, width: int, bias: bool
    ) -> None:
        """The MatMul rewrite produces the same features as the convolution, with and without bias."""
        import onnxruntime as ort

        model = _patch_embedding(channels=channels, patch=patch, height=height, width=width, batch=batch, bias=bias)
        image = np.random.default_rng(1).standard_normal((batch, channels, height, width)).astype(np.float32)
        expected = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"image": image}
        )[0]
        assert quantize._patch_embedding_as_matmul(model) == 1
        actual = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"image": image}
        )[0]
        # The convolution and the matrix multiply add up a patch's float32 products (768 in the first case) in a
        # different order: macOS differed by 4e-5 on outputs of about 40. A wrong permutation, kernel layout or bias is
        # off by whole units.
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize(
        ("channels", "patch", "multiple_of", "rewritten"),
        [
            pytest.param(3, 12, 1, 1, id="432-values-by-default"),
            pytest.param(3, 12, 32, 0, id="432-values-not-a-multiple-of-32"),
            pytest.param(3, 16, 32, 1, id="768-values-a-multiple-of-32"),
            pytest.param(2, 4, 32, 1, id="32-values-of-two-channels"),
            pytest.param(4, 2, 32, 0, id="16-values-of-four-channels"),
        ],
    )
    def test_multiple_of_rewrites_only_the_patches_it_divides(
        self, channels: int, patch: int, multiple_of: int, rewritten: int
    ) -> None:
        """Only patches whose value count is a multiple of multiple_of are rewritten."""
        model = _patch_embedding(channels=channels, patch=patch, size=2 * patch)
        assert quantize._patch_embedding_as_matmul(model, multiple_of=multiple_of) == rewritten

    @pytest.mark.parametrize("multiple_of", [0, -32])
    def test_multiple_of_below_one_is_refused(self, multiple_of: int) -> None:
        """A multiple_of below one is refused and leaves the model untouched."""
        model = _patch_embedding()
        before = model.SerializeToString()
        with pytest.raises(ValueError) as exc_info:
            quantize._patch_embedding_as_matmul(model, multiple_of=multiple_of)
        assert "multiple_of" in str(exc_info.value)
        assert model.SerializeToString() == before

    def test_convolution_is_replaced_by_the_matmul_the_plan_quantizes(self) -> None:
        """The patch Conv becomes a MatMul that plan_int8 selects for quantization."""
        model = _patch_embedding()
        assert quantize._patch_embedding_as_matmul(model) == 1
        assert [node.op_type for node in model.graph.node].count("Conv") == 0
        assert _names(quantize.plan_int8(model)) == ["patch/MatMul"]

    def test_the_matmul_weight_is_the_flattened_kernel(self) -> None:
        """The MatMul weight is the kernel flattened to (out, C*H*W) and transposed."""
        model = _patch_embedding(channels=3, patch=16, size=32, out_channels=8)
        kernel = numpy_helper.to_array(model.graph.initializer[0])
        quantize._patch_embedding_as_matmul(model)
        matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
        weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        np.testing.assert_array_equal(weights[matmul.input[1]], kernel.reshape(8, 3 * 16 * 16).T)

    def test_the_output_keeps_its_name(self) -> None:
        """The rewritten subgraph still produces the original output tensor name."""
        model = _patch_embedding()
        quantize._patch_embedding_as_matmul(model)
        assert model.graph.node[-1].output[0] == "features"

    @pytest.mark.parametrize("bias", [True, False])
    def test_the_rewritten_graph_is_valid_onnx(self, bias: bool) -> None:
        """The rewritten graph passes the ONNX checker and strict shape inference."""
        model = _patch_embedding(bias=bias)
        quantize._patch_embedding_as_matmul(model)
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_an_empty_bias_name_is_no_bias(self) -> None:
        """An empty bias input name means no bias and leaves no empty input behind."""
        model = _patch_embedding(bias=False, empty_bias=True)
        assert quantize._patch_embedding_as_matmul(model) == 1
        assert "" not in [name for node in model.graph.node for name in node.input]
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_the_kernel_keeps_its_dtype(self, dtype: type) -> None:
        """The MatMul weight keeps the kernel's FP32 or FP16 dtype."""
        model = _patch_embedding()
        for init in model.graph.initializer:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(dtype), init.name))
        quantize._patch_embedding_as_matmul(model)
        matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
        weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        assert weights[matmul.input[1]].dtype == dtype

    def test_a_kernel_reached_through_an_identity_is_found(self) -> None:
        """A kernel reached through an Identity alias is still rewritten."""
        model = _patch_embedding()
        conv = model.graph.node[0]
        conv.input[1] = "w_alias"
        model.graph.node.insert(0, helper.make_node("Identity", ["w"], ["w_alias"], "alias"))
        assert quantize._patch_embedding_as_matmul(model) == 1

    def test_every_replaced_kernel_is_dropped(self) -> None:
        """Every kernel initializer whose Conv was replaced is dropped."""
        model = _patch_embedding()
        model.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(model.graph.initializer[0]), "w2"))
        model.graph.node.append(
            helper.make_node(
                "Conv", ["image", "w2", "b"], ["more"], f"{BACKBONE}more/Conv", kernel_shape=[16, 16], strides=[16, 16]
            )
        )
        assert quantize._patch_embedding_as_matmul(model) == 2
        assert {"w", "w2"}.isdisjoint(init.name for init in model.graph.initializer)

    @pytest.mark.parametrize("reader", ["another-node", "graph-output", "if-branch"])
    def test_a_kernel_something_else_still_reads_is_kept(self, reader: str) -> None:
        """A kernel still read by another node, output or branch stays in the graph."""
        model = _patch_embedding()
        kernel_info = helper.make_tensor_value_info("w_copy", TensorProto.FLOAT, [8, 3, 16, 16])
        if reader == "another-node":
            model.graph.node.append(helper.make_node("Identity", ["w"], ["w_copy"], "copy"))
            model.graph.output.append(kernel_info)
        elif reader == "graph-output":
            model.graph.output.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 16, 16]))
        else:
            # A branch that captures the kernel from the outer scope reads it, though no top-level node does.
            branch = helper.make_graph(
                [helper.make_node("Identity", ["w"], ["w_copy"], "inner")], "branch", [], [kernel_info]
            )
            model.graph.initializer.append(numpy_helper.from_array(np.array(True), "flag"))
            model.graph.node.append(
                helper.make_node("If", ["flag"], ["w_copy"], "if", then_branch=branch, else_branch=branch)
            )
            model.graph.output.append(kernel_info)
        assert quantize._patch_embedding_as_matmul(model) == 1
        assert "w" in [init.name for init in model.graph.initializer]

    def test_an_initializer_that_is_a_graph_input_is_never_dropped(self) -> None:
        """An initializer that is also a graph input is never dropped as unread."""
        # Dropping it would turn a default value into an input the caller must now supply.
        graph = helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"], "relu")],
            "g",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("old", TensorProto.FLOAT, [1]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            [numpy_helper.from_array(np.zeros(1, np.float32), name) for name in ("old", "unused")],
        )
        quantize._drop_unread_initializers(graph, ["old", "unused"])
        assert [init.name for init in graph.initializer] == ["old"]

    def test_generated_names_never_reuse_a_name_in_the_graph(self) -> None:
        """Generated tensor and node names never collide with names already in the graph."""
        model = _patch_embedding()
        taken = f"{BACKBONE}patch/cut"
        model.graph.node.append(helper.make_node("Identity", ["image"], [taken], f"{BACKBONE}patch/Cut"))
        model.graph.output.append(helper.make_tensor_value_info(taken, TensorProto.FLOAT, [1, 3, 32, 32]))
        quantize._patch_embedding_as_matmul(model)
        produced = [name for node in model.graph.node for name in node.output]
        assert len(produced) == len(set(produced))
        assert len({node.name for node in model.graph.node}) == len(model.graph.node)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param({"strides": [2, 2]}, id="overlapping-patches"),
            pytest.param({"pads": [1, 1, 1, 1]}, id="padded"),
            pytest.param({"dilations": [2, 2]}, id="dilated"),
            pytest.param({"auto_pad": "SAME_UPPER"}, id="auto-pad"),
            pytest.param({"kernel": (16, 8)}, id="rectangular-kernel"),
            pytest.param({"channels": 32, "patch": 4, "size": 8}, id="channels-TensorRT-does-not-pad"),
            pytest.param({"height": 36}, id="kernel-does-not-tile-the-height"),
            pytest.param({"width": 36}, id="kernel-does-not-tile-the-width"),
            pytest.param({"height": "height"}, id="symbolic-height"),
            pytest.param({"batch": "batch"}, id="symbolic-batch"),
            pytest.param({"size": 0}, id="empty-image"),
            pytest.param({"prefix": HEAD}, id="outside-the-backbone"),
            pytest.param({"prefix": DECODER}, id="in-the-decoder"),
        ],
    )
    def test_other_convolutions_are_left_alone(self, case: dict[str, Any]) -> None:
        """Convolutions that are not a plain backbone patch embedding are left untouched."""
        model = _patch_embedding(**case)
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_a_grouped_convolution_is_left_alone(self) -> None:
        """A grouped convolution is not rewritten even when its patch size would allow it."""
        # Two input channels per group and 4-pixel patches make 32 values per patch: only the grouping rules it out.
        model = _patch_embedding(channels=4, patch=4, size=8, out_channels=4, group=2)
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((4, 2, 4, 4), np.float32), "w"))
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_an_existing_matmul_name_is_never_reused(self) -> None:
        """A convolution is left alone when its MatMul name is already taken."""
        model = _patch_embedding()
        model.graph.node.append(helper.make_node("Identity", ["features"], ["copy"], f"{BACKBONE}patch/MatMul"))
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_a_duplicated_convolution_name_is_left_alone(self) -> None:
        """Convolutions sharing a node name cannot be told apart and are left alone."""
        model = _patch_embedding()
        model.graph.node.append(
            helper.make_node("Conv", ["image", "w"], ["other"], f"{BACKBONE}patch/Conv", kernel_shape=[16, 16])
        )
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_two_convolutions_in_one_scope_get_one_matmul(self) -> None:
        """Two convolutions mapping to one MatMul name yield a single rewrite and unique names."""
        # ``patch/Conv`` and ``patch/Conv_1`` would both become ``patch/MatMul``; the second must not take the name.
        model = _patch_embedding()
        model.graph.node.append(
            helper.make_node(
                "Conv",
                ["image", "w", "b"],
                ["other"],
                f"{BACKBONE}patch/Conv_1",
                kernel_shape=[16, 16],
                strides=[16, 16],
            )
        )
        assert quantize._patch_embedding_as_matmul(model) == 1
        names = [node.name for node in model.graph.node]
        assert len(names) == len(set(names))

    def test_a_rejected_convolution_does_not_stop_the_next_one(self) -> None:
        """A convolution that cannot be rewritten does not stop later ones from being rewritten."""
        model = _patch_embedding()
        model.graph.input.append(helper.make_tensor_value_info("narrow", TensorProto.FLOAT, [1, 3, 36, 32]))
        model.graph.node.insert(
            0,
            helper.make_node(
                "Conv", ["narrow", "w"], ["tall"], f"{BACKBONE}tall/Conv", kernel_shape=[16, 16], strides=[16, 16]
            ),
        )
        assert quantize._patch_embedding_as_matmul(model) == 1

    @pytest.mark.parametrize("initializer", [False, True])
    def test_a_kernel_that_can_be_replaced_at_run_time_is_left_alone(self, initializer: bool) -> None:
        """A kernel exposed as a graph input is not constant and its convolution is left alone."""
        # A graph input without an initializer is no constant kernel; one with an initializer can be overridden.
        model = _patch_embedding()
        model.graph.input.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 16, 16]))
        if not initializer:
            kept = [init for init in model.graph.initializer if init.name != "w"]
            model.graph.ClearField("initializer")
            model.graph.initializer.extend(kept)
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before


@_requires_onnx_and_onnxruntime
@_requires_onnxconverter_common
class TestPatchEmbeddingInInt8Source:
    """``int8_source_graph`` plans, calibrates and quantizes the patch embedding as the matrix multiply it becomes."""

    @pytest.fixture
    def patch_graph(self, tmp_path: Path) -> Path:
        """A 16-pixel patch embedding over a ``(1, 3, 16, 16)`` image, saved to disk with a detector's output names.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path = patch_graph(tmp_path)  # doctest: +SKIP
        """
        path = tmp_path / "patch.onnx"
        onnx.save(_with_detector_outputs(_patch_embedding(size=16)), path)
        return path

    def test_patch_embedding_is_quantized_as_a_matrix_multiply(self, patch_graph: Path) -> None:
        """int8_source_graph quantizes the patch embedding as a MatMul with Q/DQ on both inputs."""
        images = np.random.default_rng(0).standard_normal((3, 3, 16, 16)).astype(np.float32)
        with quantize.int8_source_graph(
            str(patch_graph),
            calibration_data=images,
            max_images=3,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            graph = onnx.load(int8_path).graph
        producers = {output: node for node in graph.node for output in node.output}
        (matmul,) = [node for node in graph.node if node.name == f"{BACKBONE}patch/MatMul"]
        activation = producers[matmul.input[0]]
        assert [node.op_type for node in graph.node].count("Conv") == 0
        assert activation.op_type == "DequantizeLinear"
        assert producers[activation.input[0]].op_type == "QuantizeLinear"
        assert producers[matmul.input[1]].op_type == "DequantizeLinear"

    def test_the_planned_patch_is_the_one_quantized_when_two_share_a_parent(self, tmp_path: Path) -> None:
        """With two patch embeddings under one parent, the planned one is the one quantized."""
        # The 432-value patch (skipped by the plan) comes first; the 192-value one is planned. Both would be named
        # ``patch/MatMul``, so the FP16 copy must rewrite the planned patch first or the plan quantizes the wrong layer.
        model = _patch_embedding(patch=12, size=24)
        second = _patch_embedding(patch=8, size=24)
        for initializer in second.graph.initializer:
            initializer.name += "_second"
        conv = second.graph.node[0]
        conv.name = f"{BACKBONE}patch/Conv_1"
        conv.input[1:] = [f"{name}_second" for name in conv.input[1:]]
        conv.output[0] = "features_second"
        model.graph.initializer.extend(second.graph.initializer)
        model.graph.node.append(conv)
        model.graph.node.extend(
            helper.make_node("Identity", [source], [name], name)
            for source, name in (("features", "dets"), ("features_second", "labels"))
        )
        del model.graph.output[:]
        model.graph.output.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("dets", "labels")
        )
        path = tmp_path / "two_patches.onnx"
        onnx.save(model, path)
        with quantize.int8_source_graph(
            str(path),
            calibration_data=np.zeros((3, 3, 24, 24), np.float32),
            max_images=3,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            graph = onnx.load(int8_path).graph
        producers = {output: node for node in graph.node for output in node.output}
        weights = {initializer.name: initializer for initializer in graph.initializer}
        (matmul,) = [node for node in graph.node if node.op_type == "MatMul" and "patch/" in node.name]
        (conv,) = [node for node in graph.node if node.op_type == "Conv"]
        quantized_weight = producers[matmul.input[1]]
        assert quantized_weight.op_type == "DequantizeLinear"
        assert tuple(weights[quantized_weight.input[0]].dims) == (192, 8)
        assert tuple(weights[conv.input[1]].dims) == (8, 3, 12, 12)

    def test_the_fp16_copy_is_lifted_before_the_patch_embedding_is_rewritten(self, tmp_path: Path) -> None:
        """An opset-13 source is lifted before the rewrite so the INT8 graph ends at opset 19."""
        # At opset 13 the rewrite's ``Add`` would be refused by the lift that follows it, after calibration has run.
        model = _with_detector_outputs(_patch_embedding(size=16))
        model.opset_import[0].version = 13
        path = tmp_path / "opset13.onnx"
        onnx.save(model, path)
        with quantize.int8_source_graph(
            str(path),
            calibration_data=np.zeros((3, 3, 16, 16), np.float32),
            max_images=3,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            quantized = onnx.load(int8_path)
        assert quantized.opset_import[0].version == 19

    @pytest.mark.parametrize("copy", ["overlapping-patches", "renamed"])
    def test_fp16_copy_without_a_planned_node_is_refused(self, patch_graph: Path, tmp_path: Path, copy: str) -> None:
        """An FP16 copy lacking a planned node is refused with a ValueError, not a KeyError."""
        # Overlapping patches leave nothing to rewrite; a renamed ``Conv`` is rewritten under another ``MatMul`` name.
        # Either way the node the plan quantizes is missing, which would otherwise surface as a ``KeyError``.
        other = (
            _patch_embedding(size=16, strides=[8, 8]) if copy == "overlapping-patches" else _patch_embedding(size=16)
        )
        if copy == "renamed":
            other.graph.node[0].name = f"{BACKBONE}other/Conv"
        onnx.save(_with_detector_outputs(other), tmp_path / "other.onnx")

        @contextlib.contextmanager
        def other_fp16(*_: Any, **__: Any) -> Iterator[str]:
            yield str(tmp_path / "other.onnx")

        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(patch_graph),
                calibration_data=np.zeros((3, 3, 16, 16), np.float32),
                max_images=3,
                dynamic_batch=False,
                fp16_graph=other_fp16,
            ):
                pass
        assert "lacks 1 node" in str(exc_info.value)
