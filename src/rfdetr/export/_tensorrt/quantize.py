# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Explicit INT8 quantization of the FP16 graph a TensorRT engine is built from.

TensorRT 11 has no INT8 builder flag and no calibrator: an INT8 engine is a strongly typed build of a graph that
carries its own ``QuantizeLinear``/``DequantizeLinear`` pairs. Where those pairs sit decides whether the engine is
faster than FP16 at all, so the placement here is not "quantize every matrix multiply". It follows what was measured
on RF-DETR Nano (RTX 5070, TensorRT 11.3 and 10.16), and for the attention token limit on Small to Large (TensorRT
11.3); issue #1024:

* **FP16 scales on the FP16 graph.** Q/DQ is inserted into the graph :func:`_cast_onnx_to_fp16` produces, with FP16
  scales, so every region left unquantized stays FP16. FP32 scales make each ``DequantizeLinear`` return FP32 and drag
  the rest of the graph to FP32, which is slower than FP16 before any INT8 kernel runs.
* **Weight-bearing ``MatMul``/``Gemm``/``Conv`` inputs**, in the backbone encoder and the decoder (plus the batched
  multiplies of a block whose attention runs in INT8, below). The output of
  each quantized operation stays in floating point, so TensorRT fuses the bias, LayerScale and residual add into the
  INT8 kernel's epilogue and the quantize step into the preceding LayerNorm or GELU. The projector, the two-stage
  proposal head and the detection heads stay FP16: quantizing them costs accuracy and buys nothing.
* **No convolution whose input channels TensorRT would pad.** TensorRT ran the INT8 patch embedding on the 3-channel
  image padded to 32 channels and stored the 29 added zero channels in the engine as a ``B x 29 x H x W`` FP16 constant,
  read on every run: at batch 8 the engine was 110 MB instead of 51 MB and INT8 1.21x instead of 1.37x faster than
  FP16. Only that convolution was measured; any convolution whose input channels are not a multiple of 32 is kept FP16
  as a precaution. (The FP16 convolution pads too, to 4 or 8 channels picked per build, so both engines still store 1
  or 5 zero channels per image.)
* **Attention is all INT8 or all FP16.** An attention block that TensorRT's INT8 fused-attention kernel accepts and
  that was measured to run faster for it (head size 16, 32 or 64 and at most 325 tokens: the windowed backbone blocks
  of Nano, Small and Medium) gets Q/DQ on both batched matrix multiplies as well as on its q/k/v/output projections,
  which keeps attention fused in INT8. Any other block (Large's windowed blocks, the global backbone blocks, the decoder
  self-attention) keeps its projections and batched multiplies in FP16: quantizing
  only the projections makes TensorRT emit FP32 projections plus reshape kernels in front of the fused FP16 kernel, and
  a quantized output projection behind FP16 attention makes TensorRT 11 fuse an INT8 output into that kernel, which
  returns wrong values.
* **A GELU never feeds an FP16 consumer from an INT8 multiply.** TensorRT 10.16 and 11.3 return NaN for an INT8
  ``MatMul`` -> bias -> GELU -> FP16 consumer chain, so an MLP's first projection is quantized only when its second one
  is.

Activation ranges come from calibration images run through the FP32 graph on onnxruntime's CPU provider: the absolute
maximum of each quantized tensor, which measured better than percentile, MSE and entropy clipping on this model.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import warn_if_too_few_samples
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: The only quantization mode a TensorRT export accepts besides ``None`` (FP16 or FP32, chosen with ``fp16``).
INT8 = "int8"

#: Largest magnitude a symmetric INT8 value takes; TensorRT requires a zero point of 0.
_INT8_MAX = 127

#: Opset from which ``QuantizeLinear``/``DequantizeLinear`` accept FP16 inputs and scales.
_FP16_QDQ_OPSET = 19

#: Smallest positive normal FP16 value, the floor of every scale. A smaller scale rounds to 0 in FP16 (or lands in the
#: subnormal range), and TensorRT refuses a zero scale at parse time ("Scale coefficients must all be positive").
_MIN_FP16_SCALE = float(np.finfo(np.float16).tiny)

#: Largest finite FP16 value; a range whose scale exceeds it would be stored as ``inf``.
_MAX_FP16 = float(np.finfo(np.float16).max)

#: Reductions whose ``axes`` moved from an attribute to an input in opset 18. Lifting a graph past that rewrites the
#: attribute into an input, which keeps the meaning exactly.
_AXES_ATTRIBUTE_REDUCTIONS = frozenset(
    {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSumSquare",
    }
)

#: Ops whose schema changed between opset 17 and :data:`_FP16_QDQ_OPSET` only by accepting more types (float8, strings)
#: or a new attribute whose default keeps the old behavior, so lifting the opset import alone keeps their meaning.
_TYPE_ONLY_CHANGES = frozenset(
    {"Cast", "CastLike", "Constant", "Equal", "Identity", "If", "Loop", "Reshape", "Scan", "Shape", "Size"}
)

#: Head sizes TensorRT's INT8 fused-attention kernel accepts (TensorRT 11.3 "Attention fusion", SM75-90, SM120-121).
_FUSED_INT8_HEAD_SIZES = frozenset({16, 32, 64})

#: Longest query/key sequence that gets INT8 attention. TensorRT's INT8 fused-attention kernel accepts up to 512; this
#: is the largest window measured to pay (Medium: 5% faster from a CUDA graph, about equal with a plain call, -0.28 AP).
#: Large's 485-token windows ran no faster than with FP16 attention and lost 1.0 AP; 326-484 is not measured (RTX 5070,
#: TensorRT 11.3, COCO val2017).
_INT8_ATTENTION_MAX_TOKENS = 325

#: Node-name prefixes of the regions whose weight-bearing operations are quantized: backbone encoder and decoder.
_QUANTIZED_REGIONS = ("/backbone/backbone.0/encoder/", "/transformer/decoder/")

#: Node-name prefix of the region whose attention may run as INT8 fused attention. The decoder self-attention fits the
#: kernel's size limits too, but TensorRT does not fuse it in INT8 and the unfused INT8 path is slower than FP16.
_INT8_ATTENTION_REGION = "/backbone/backbone.0/encoder/"

#: Ops a projection's output passes through on its way into a batched multiply: bias, reshape to heads, scaling.
_PROJECTION_PASSTHROUGH = frozenset(
    {"Add", "Cast", "Div", "Identity", "Mul", "Reshape", "Squeeze", "Transpose", "Unsqueeze"}
)

#: Ops between a first MLP projection and its activation-path successor, including both GELU forms.
_GELU_PATH = frozenset({"Add", "Div", "Erf", "Gelu", "Mul"})

#: TensorRT padded the INT8 patch embedding's input channels to a multiple of this and stored the padding in the engine.
_INT8_CONV_CHANNEL_MULTIPLE = 32

#: Ops that make a path a GELU.
_GELU_OPS = frozenset({"Erf", "Gelu"})

#: Ops that may sit between the attention-score multiply and its ``Softmax``: scaling and masking.
_SCORE_PASSTHROUGH = frozenset({"Add", "Cast", "Div", "Mul", "Sub", "Where"})

#: Ops that may sit between a ``Softmax`` and the multiply that reads its probabilities (eager attention casts them).
_PROBABILITY_PASSTHROUGH = frozenset({"Cast", "Identity"})


@dataclass(frozen=True)
class Int8Plan:
    """Which nodes of a graph get INT8 Q/DQ.

    Node names are stable between the FP32 export and its FP16 cast, so one plan, computed on the FP32 graph where
    shapes are known, applies to both: ranges are calibrated on the FP32 graph and Q/DQ is inserted into the FP16 one.

    Attributes:
        weighted: Weight-bearing ``MatMul``/``Gemm``/``Conv`` nodes whose weight is stored as INT8.
        activations: ``(node name, input index)`` pairs whose activation input gets a Q/DQ pair. Includes input 0 of
            every node in *weighted*, and both inputs of the batched multiplies of INT8 attention blocks.

    Examples:
        >>> Int8Plan(weighted=("fc",), activations=(("fc", 0),)).activations
        (('fc', 0),)
    """

    weighted: tuple[str, ...]
    activations: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _Attention:
    """One ``MatMul -> Softmax -> MatMul`` attention block and the projections around it."""

    scores: Any
    context: Any
    projections: frozenset[str]
    output_projection: str | None
    int8: bool


def _constant_weights(graph: Any) -> dict[str, Any]:
    """Map every tensor of *graph* that holds a constant to its ``TensorProto``.

    Besides initializers this follows ``Identity`` nodes, which the exporter emits when two layers start from identical
    weights (an untrained model's zero-initialized deformable-attention projections, say) and it stores the tensor
    once, and ``Constant`` nodes.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> graph = helper.make_graph(
        ...     [helper.make_node("Identity", ["w"], ["w_alias"])], "g", [], [],
        ...     [numpy_helper.from_array(np.zeros(2, np.float32), "w")])
        >>> sorted(_constant_weights(graph))
        ['w', 'w_alias']
    """
    constants = {init.name: init for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
        elif node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value":
            constants[node.output[0]] = node.attribute[0].t
    return constants


def _weight_name(node: Any, initializers: Mapping[str, Any]) -> str | None:
    """Return the name of *node*'s constant weight when INT8 per-channel weights apply to it, else ``None``.

    Only a 2-D ``MatMul``/``Gemm`` weight or a 4-D ``Conv`` kernel qualifies, and only for a node with a non-empty
    name, since a plan refers to nodes by name. A ``Gemm`` that transposes its activation input is left alone.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {"w": numpy_helper.from_array(np.zeros((4, 2), np.float32), "w")}
        >>> _weight_name(helper.make_node("MatMul", ["x", "w"], ["y"], "fc"), weights)
        'w'
        >>> _weight_name(helper.make_node("MatMul", ["x", "z"], ["y"], "bmm"), weights) is None
        True
    """
    if not node.name or node.op_type not in ("MatMul", "Gemm", "Conv") or len(node.input) < 2:
        return None
    weight = initializers.get(node.input[1])
    if weight is None:
        return None
    if node.op_type == "Gemm" and any(a.name == "transA" and a.i for a in node.attribute):
        return None
    return str(node.input[1]) if len(weight.dims) == (4 if node.op_type == "Conv" else 2) else None


def _holds_weight(node: Any, initializers: Mapping[str, Any]) -> bool:
    """Return whether *node* is a ``MatMul``/``Gemm``/``Conv`` with a constant weight, however it is named.

    Unlike :func:`_weight_name` this does not ask whether the node can be planned, so an unnamed layer, or a ``Gemm``
    that transposes its activation, still counts as a consumer of the tensor it reads.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {"w": numpy_helper.from_array(np.zeros((4, 2), np.float32), "w")}
        >>> _holds_weight(helper.make_node("MatMul", ["x", "w"], ["y"], ""), weights)
        True
        >>> _holds_weight(helper.make_node("MatMul", ["x", "z"], ["y"], "bmm"), weights)
        False
    """
    return node.op_type in ("MatMul", "Gemm", "Conv") and len(node.input) > 1 and node.input[1] in initializers


def _pads_input_channels(node: Any, initializers: Mapping[str, Any]) -> bool:
    """Return whether *node* is a ``Conv`` whose input channels TensorRT would pad to run it in INT8.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {"rgb": numpy_helper.from_array(np.zeros((8, 3, 4, 4), np.float32), "rgb")}
        >>> _pads_input_channels(helper.make_node("Conv", ["image", "rgb"], ["y"], "patch"), weights)
        True
    """
    if node.op_type != "Conv":
        return False
    groups = int(next((a.i for a in node.attribute if a.name == "group"), 1))
    channels = int(initializers[node.input[1]].dims[1]) * groups
    return channels % _INT8_CONV_CHANNEL_MULTIPLE != 0


def _dims(shapes: Mapping[str, Any], tensor: str) -> tuple[int | None, ...] | None:
    """Return the static dimensions shape inference recorded for *tensor* (``None`` per symbolic axis).

    Examples:
        >>> from onnx import TensorProto, helper
        >>> info = helper.make_tensor_value_info("t", TensorProto.FLOAT, [2, "n", 64])
        >>> _dims({"t": info}, "t")
        (2, None, 64)
        >>> _dims({}, "t") is None
        True
    """
    info = shapes.get(tensor)
    if info is None or not info.type.tensor_type.HasField("shape"):
        return None
    return tuple(d.dim_value if d.HasField("dim_value") else None for d in info.type.tensor_type.shape.dim)


def _trace_projection(start: str, producers: Mapping[str, Any], initializers: Mapping[str, Any]) -> str | None:
    """Walk upstream from a batched-multiply input to the weight-bearing projection that produced it.

    Follows only the shape and scaling ops a projection passes through on its way into attention, so a path through
    shape arithmetic (``Shape`` -> ``Sqrt``) or another block's output ends without a match.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {"w": numpy_helper.from_array(np.zeros((4, 4), np.float32), "w")}
        >>> nodes = [helper.make_node("MatMul", ["x", "w"], ["q"], "q_proj"),
        ...          helper.make_node("Reshape", ["q", "shape"], ["q_heads"], "split")]
        >>> _trace_projection("q_heads", {o: n for n in nodes for o in n.output}, weights)
        'q_proj'
    """
    frontier, seen = [start], set()
    while frontier:
        tensor = frontier.pop()
        node = producers.get(tensor)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if _weight_name(node, initializers):
            return str(node.name)
        if node.op_type not in _PROJECTION_PASSTHROUGH:
            continue
        data_inputs = node.input[:1] if node.op_type in ("Reshape", "Squeeze", "Unsqueeze") else node.input
        frontier.extend(name for name in data_inputs if name and name not in initializers)
    return None


def _output_projection(context: Any, consumers: Mapping[str, list[Any]], initializers: Mapping[str, Any]) -> str | None:
    """Walk downstream from an attention block's output to the projection that reads it.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {"w": numpy_helper.from_array(np.zeros((4, 4), np.float32), "w")}
        >>> context = helper.make_node("MatMul", ["p", "v"], ["ctx"], "context")
        >>> nodes = [helper.make_node("Transpose", ["ctx"], ["merged"], "merge"),
        ...          helper.make_node("MatMul", ["merged", "w"], ["y"], "out_proj")]
        >>> _output_projection(context, {"ctx": [nodes[0]], "merged": [nodes[1]]}, weights)
        'out_proj'
    """
    tensor = context.output[0]
    for _ in range(4):
        readers = consumers.get(tensor, [])
        if len(readers) != 1:
            return None
        reader = readers[0]
        if _weight_name(reader, initializers) and reader.input[0] == tensor:
            return str(reader.name)
        if reader.op_type not in ("Reshape", "Transpose"):
            return None
        tensor = reader.output[0]
    return None


def _scores_matmul(softmax: Any, producers: Mapping[str, Any], initializers: Mapping[str, Any]) -> Any | None:
    """Walk up from a ``Softmax`` through scaling and masking ops to the multiply that computed the scores.

    Examples:
        >>> from onnx import helper
        >>> nodes = [helper.make_node("MatMul", ["q", "k"], ["s0"], "scores"),
        ...          helper.make_node("Div", ["s0", "d"], ["s"], "scale"),
        ...          helper.make_node("Softmax", ["s"], ["p"], "softmax")]
        >>> _scores_matmul(nodes[2], {o: n for n in nodes for o in n.output}, {}).name
        'scores'
    """
    # Breadth first, inputs in order: the scores are the nearest multiply, and the first operand of a mask ``Add``.
    frontier, seen = [softmax.input[0]], set()
    while frontier:
        node = producers.get(frontier.pop(0))
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if node.op_type == "MatMul" and not any(name in initializers for name in node.input):
            return node
        if node.op_type in _SCORE_PASSTHROUGH:
            frontier.extend(name for name in node.input if name and name not in initializers)
    return None


def _context_matmul(softmax: Any, consumers: Mapping[str, list[Any]]) -> Any | None:
    """Return the ``MatMul`` that multiplies *softmax*'s probabilities by the values, through casts, or ``None``.

    Examples:
        >>> from onnx import helper
        >>> nodes = [helper.make_node("Softmax", ["s"], ["p"], "softmax"),
        ...          helper.make_node("Cast", ["p"], ["p16"], "cast", to=10),
        ...          helper.make_node("MatMul", ["p16", "v"], ["c"], "context")]
        >>> _context_matmul(nodes[0], {"p": [nodes[1]], "p16": [nodes[2]]}).name
        'context'
    """
    tensor = softmax.output[0]
    while len(readers := consumers.get(tensor, [])) == 1:
        reader = readers[0]
        if reader.op_type == "MatMul" and reader.input[0] == tensor:
            return reader
        if reader.op_type not in _PROBABILITY_PASSTHROUGH:
            return None
        tensor = reader.output[0]
    return None


def _feeds_activation_matmul(softmax: Any, consumers: Mapping[str, list[Any]], initializers: Mapping[str, Any]) -> bool:
    """Return whether *softmax*'s output reaches a ``MatMul`` whose operands are both activations, as attention's does.

    The walk follows every reader of the probabilities, whatever the op (scaling, masking, ``Where``, ``Dropout``,
    shape ops), and stops only at an operation that carries a weight: a projection ends the attention path, and
    anything past it is another block's business.

    Examples:
        >>> from onnx import helper
        >>> nodes = [helper.make_node("Softmax", ["s"], ["p"], "softmax"),
        ...          helper.make_node("MatMul", ["p", "v"], ["c"], "context")]
        >>> _feeds_activation_matmul(nodes[0], {"p": [nodes[1]]}, {}), _feeds_activation_matmul(nodes[0], {}, {})
        (True, False)
    """
    frontier, seen = [softmax.output[0]], set()
    while frontier:
        for reader in consumers.get(frontier.pop(), []):
            if id(reader) in seen:
                continue
            seen.add(id(reader))
            weighted = any(name in initializers for name in reader.input)
            if reader.op_type == "MatMul" and not weighted:
                return True
            if not (weighted and reader.op_type in ("Conv", "Gemm", "MatMul")):
                frontier.extend(reader.output)
    return False


def _attention_blocks(model: Any) -> list[_Attention]:
    """Find every ``MatMul -> Softmax -> MatMul`` attention block of *model*'s top-level graph.

    Only the top-level graph is read: a ``Softmax`` inside an ``If``/``Loop`` subgraph is not seen, and RF-DETR's
    exports contain none.

    A block can run its attention in INT8 when shape inference proves its head size is one of
    :data:`_FUSED_INT8_HEAD_SIZES` and both sequence lengths are at most :data:`_INT8_ATTENTION_MAX_TOKENS`; an unknown
    dimension counts as not.

    Args:
        model: FP32 ``ModelProto`` with static spatial shapes.

    Returns:
        One entry per attention block, with the names of the projections found around it.

    Raises:
        ValueError: If a top-level ``Softmax`` feeds an activation-by-activation multiply but is not
            recognised as a block, so its projections could not be kept out of INT8 with certainty.
    """
    import onnx

    graph = model.graph
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: v for v in (*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output)}
    initializers = _constant_weights(graph)
    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    blocks = []
    for softmax in (node for node in graph.node if node.op_type == "Softmax"):
        scores = _scores_matmul(softmax, producers, initializers)
        context = _context_matmul(softmax, consumers)
        if scores is None or context is None or any(name in initializers for name in (*scores.input, context.input[1])):
            # Whatever its name: a renamed Softmax must not hide an attention block from the projection rules.
            if _feeds_activation_matmul(softmax, consumers, initializers):
                raise ValueError(
                    f"Cannot place INT8 safely: the Softmax {softmax.name!r} feeds a matrix multiply of two "
                    "activations but is not a recognised attention block. INT8 TensorRT export applies to RF-DETR "
                    "detector exports only."
                )
            continue
        sources = (scores.input[0], scores.input[1], context.input[1])
        projections = {_trace_projection(name, producers, initializers) for name in sources}
        output_projection = _output_projection(context, consumers, initializers)
        projections.add(output_projection)
        score_dims, query_dims = _dims(shapes, scores.output[0]), _dims(shapes, scores.input[0])
        int8 = (
            score_dims is not None
            and query_dims is not None
            and query_dims[-1] in _FUSED_INT8_HEAD_SIZES
            and all(d is not None and d <= _INT8_ATTENTION_MAX_TOKENS for d in score_dims[-2:])
        )
        blocks.append(_Attention(scores, context, frozenset(p for p in projections if p), output_projection, int8))
    return blocks


def _feeds_gelu_into(node: Any, consumers: Mapping[str, list[Any]], initializers: Mapping[str, Any]) -> set[str]:
    """Return the weight-bearing nodes that read *node*'s output through a GELU (empty when no GELU follows it).

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {n: numpy_helper.from_array(np.zeros((4, 4), np.float32), n) for n in ("w1", "w2")}
        >>> nodes = [helper.make_node("MatMul", ["x", "w1"], ["h"], "fc1"),
        ...          helper.make_node("Gelu", ["h"], ["g"], "act"),
        ...          helper.make_node("MatMul", ["g", "w2"], ["y"], "fc2")]
        >>> uses = {"h": [nodes[1]], "g": [nodes[2]]}
        >>> _feeds_gelu_into(nodes[0], uses, weights)
        {'fc2'}
    """
    # Each frontier entry remembers whether its path already went through a GELU op.
    frontier, seen, found = [(node.output[0], False)], set(), set()
    while frontier:
        tensor, through_gelu = frontier.pop()
        for reader in consumers.get(tensor, []):
            if (id(reader), through_gelu) in seen:
                continue
            seen.add((id(reader), through_gelu))
            if _holds_weight(reader, initializers):
                if through_gelu:
                    found.add(str(reader.name))
            elif reader.op_type in _GELU_PATH:
                frontier.extend((name, through_gelu or reader.op_type in _GELU_OPS) for name in reader.output)
            elif through_gelu:
                found.add(str(reader.name))  # any other reader of a GELU output runs in FP16
    return found


def plan_int8(model: Any) -> Int8Plan:
    """Decide which nodes of an FP32 RF-DETR graph get INT8 Q/DQ (see the module docstring for the rules).

    Args:
        model: The FP32 ``ModelProto`` the engine is built from.

    Returns:
        The plan, in graph order.

    Raises:
        ValueError: If the graph has nothing to quantize, which means it is not an RF-DETR detector export; if an FP16
            attention block's output projection cannot be identified; or if a top-level ``Softmax`` feeds a
            multiply of two activations without being a recognised attention block.
    """
    graph = model.graph
    initializers = _constant_weights(graph)
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    names = [node.name for node in graph.node]
    duplicated = {name for name, count in Counter(names).items() if count > 1}

    weighted = {
        node.name
        for node in graph.node
        if node.name.startswith(_QUANTIZED_REGIONS)
        and node.name not in duplicated
        and _weight_name(node, initializers)
        and not _pads_input_channels(node, initializers)
    }
    attention_inputs: list[tuple[str, int]] = []
    backbone_blocks = fused = 0
    unfused: list[str] = []  # backbone blocks whose attention stays FP16 because of its shape
    for block in _attention_blocks(model):
        in_region = block.scores.name.startswith(_INT8_ATTENTION_REGION)
        backbone_blocks += in_region
        unnamed = {block.scores.name, block.context.name} & duplicated
        if block.int8 and in_region and not unnamed and block.projections <= weighted:
            fused += 1
            attention_inputs += [(block.scores.name, 0), (block.scores.name, 1)]
            attention_inputs += [(block.context.name, 0), (block.context.name, 1)]
            continue
        if block.output_projection is None:
            # A quantized projection behind FP16 attention is what TensorRT 11 computes wrongly, and an output that
            # cannot be followed to its projection cannot be kept out of INT8 with certainty.
            raise ValueError(
                f"Cannot place INT8 safely: the output projection of the FP16 attention at {block.scores.name!r} "
                "could not be identified. INT8 TensorRT export applies to RF-DETR detector exports only."
            )
        weighted -= block.projections
        if in_region and not block.int8:
            unfused.append(block.scores.name)
    if unfused:
        logger.info(
            f"INT8 attention in {fused} of {backbone_blocks} backbone attention blocks; "
            f"{len(unfused)} stay FP16 by shape (e.g. {unfused[0]!r}): INT8 attention needs a head size in "
            f"{sorted(_FUSED_INT8_HEAD_SIZES)} and at most {_INT8_ATTENTION_MAX_TOKENS} tokens"
        )

    # A projection feeding an FP16 consumer through a GELU comes back as NaN from TensorRT; dropping one projection can
    # expose another, so repeat until nothing changes.
    by_name = {node.name: node for node in graph.node}
    changed = True
    while changed:
        changed = False
        for name in sorted(weighted):
            if not _feeds_gelu_into(by_name[name], consumers, initializers) <= weighted:
                weighted.discard(name)
                changed = True

    if not weighted:
        raise ValueError(
            "Found no backbone encoder or decoder layer to quantize in this graph; INT8 TensorRT export applies to "
            "RF-DETR detector exports only."
        )
    order = {name: index for index, name in enumerate(names)}
    activations = sorted(
        {(name, 0) for name in weighted} | set(attention_inputs), key=lambda key: (order[key[0]], key[1])
    )
    return Int8Plan(weighted=tuple(sorted(weighted, key=order.__getitem__)), activations=tuple(activations))


def _input_tensor(graph: Any, node_name: str, index: int) -> str:
    """Return the tensor feeding input *index* of the node called *node_name* in *graph*."""
    node = next(node for node in graph.node if node.name == node_name)
    return str(node.input[index])


def _graph_batches(batches: Iterable[NDArray[np.float32]], batch: int | None) -> Iterator[NDArray[np.float32]]:
    """Group single-image calibration batches into the batch size a static graph was exported at.

    A short final group is padded by repeating its last image: calibration keeps each tensor's absolute maximum, which
    a repeated image cannot change.

    Examples:
        >>> singles = [np.full((1, 1, 1, 1), v, np.float32) for v in range(3)]
        >>> [b[:, 0, 0, 0].tolist() for b in _graph_batches(singles, 2)]
        [[0.0, 1.0], [2.0, 2.0]]
    """
    group: list[NDArray[np.float32]] = []
    for sample in batches:
        group.append(sample)
        if len(group) == (batch or 1):
            yield np.concatenate(group)
            group = []
    if group:
        group += [group[-1]] * ((batch or 1) - len(group))
        yield np.concatenate(group)


def _calibration_graph(model: Any, tensors: list[str]) -> Any:
    """Return a copy of *model* that also outputs ``max(abs(t))`` for each of *tensors*.

    Reducing inside the graph returns one scalar per tensor rather than every activation.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
        ...     [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
        >>> probe = _calibration_graph(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), ["x"])
        >>> [output.name for output in probe.graph.output]
        ['y', 'x__absmax']
    """
    import onnx
    from onnx import TensorProto, helper

    # Imported here: exporter.py imports this module at module scope.
    from rfdetr.export._tensorrt.exporter import _tensor_names

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    taken = _tensor_names(probe.graph)
    for tensor in tensors:
        absolute, maximum = f"{tensor}__abs", f"{tensor}__absmax"
        while absolute in taken or maximum in taken:
            absolute, maximum = f"_{absolute}", f"_{maximum}"
        taken.update((absolute, maximum))
        probe.graph.node.append(helper.make_node("Abs", [tensor], [absolute]))
        probe.graph.node.append(helper.make_node("ReduceMax", [absolute], [maximum], keepdims=0))
        probe.graph.output.append(helper.make_tensor_value_info(maximum, TensorProto.FLOAT, []))
    return probe


def calibrate_ranges(
    onnx_path: str, model: Any, tensors: list[str], batches: Iterable[NDArray[np.float32]]
) -> dict[str, float]:
    """Run calibration batches through the FP32 graph and return each tensor's absolute maximum.

    Args:
        onnx_path: The FP32 ``.onnx`` file *model* was loaded from; the probe graph is written to a temporary file
            beside it.
        model: The loaded FP32 ``ModelProto``.
        tensors: Tensor names to measure.
        batches: Preprocessed ``(1, C, H, W)`` images.

    Returns:
        Tensor name to the largest absolute value seen.

    Raises:
        ValueError: If *batches* is empty or holds a NaN or infinite value.

    Logs a warning, after the last batch, when fewer than ``MIN_CALIBRATION_SAMPLES`` images were seen.
    """
    import onnx
    import onnxruntime as ort

    stem = os.path.basename(os.path.splitext(onnx_path)[0])
    handle, probe_path = tempfile.mkstemp(
        prefix=f"{stem}.calibration-", suffix=".onnx", dir=os.path.dirname(onnx_path) or "."
    )
    os.close(handle)
    try:
        onnx.save(_calibration_graph(model, tensors), probe_path)
        # The CPU, not CUDA: at batch 32 the FP32 graph needs about 9 GB of GPU memory under onnxruntime, more than a
        # 12 GB card has left beside the export, while the CPU calibrates 128 images in about 15 s.
        session = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
        feed_name = session.get_inputs()[0].name
        dimension = session.get_inputs()[0].shape[0]
        # The probe outputs follow the model's own, one per tensor in order (names may carry a collision prefix).
        outputs = [o.name for o in session.get_outputs()][-len(tensors) :]
        ranges: NDArray[np.float64] = np.zeros(len(tensors), dtype=np.float64)
        count = images = 0

        def tally(source: Iterable[NDArray[np.float32]]) -> Iterator[NDArray[np.float32]]:
            """Pass *source* through, adding up the images it holds (the groups below repeat the last one to pad)."""
            nonlocal images
            for image in source:
                images += len(image)
                yield image

        for group in _graph_batches(tally(batches), dimension if isinstance(dimension, int) else None):
            # onnxruntime's ReduceMax skips a NaN, so a NaN image would calibrate to a finite but wrong range.
            if not np.isfinite(group).all():
                raise ValueError(f"Calibration batch {count + 1} holds NaN or infinite values.")
            ranges = np.maximum(ranges, np.asarray(session.run(outputs, {feed_name: group}), dtype=np.float64))
            count += 1
        if count == 0:
            raise ValueError("Calibration data held no image.")
        warn_if_too_few_samples(images)
        logger.info(f"Calibrated {len(tensors)} INT8 activation ranges over {count} batch(es)")
        return dict(zip(tensors, ranges.tolist()))
    finally:
        with contextlib.suppress(OSError):
            Path(probe_path).unlink(missing_ok=True)


def _quantized_weight(weight: NDArray[Any], axis: int) -> tuple[NDArray[np.int8], NDArray[np.float16]]:
    """Quantize *weight* to symmetric INT8 with one FP16 scale per index of *axis*.

    Values are rounded against the FP16 scale the graph stores, so dequantizing reproduces them exactly; a scale never
    drops below :data:`_MIN_FP16_SCALE`, whatever the channel holds (a dead channel quantizes to zeros).

    Examples:
        >>> values, scales = _quantized_weight(np.array([[1.0, -2.0], [0.5, 1.0]], np.float32), axis=1)
        >>> values.tolist(), (scales.astype(np.float32) * 127).round(2).tolist()
        ([[127, -127], [64, 64]], [1.0, 2.0])
        >>> float(_quantized_weight(np.zeros((2, 1), np.float32), axis=1)[1][0]) > 0
        True
    """
    reduce = tuple(i for i in range(weight.ndim) if i != axis)
    scales = np.maximum(np.abs(weight.astype(np.float32)).max(axis=reduce) / _INT8_MAX, _MIN_FP16_SCALE)
    scales16 = scales.astype(np.float16)
    shape = [1] * weight.ndim
    shape[axis] = -1
    divisor = scales16.astype(np.float32).reshape(shape)
    values = np.clip(np.round(weight.astype(np.float32) / divisor), -_INT8_MAX, _INT8_MAX)
    return values.astype(np.int8), scales16


def _weight_axis(node: Any) -> int:
    """Return the output-channel axis of *node*'s weight: ``0`` for ``Conv`` and a ``transB`` ``Gemm``, else ``1``.

    Examples:
        >>> from onnx import helper
        >>> _weight_axis(helper.make_node("Gemm", ["a", "w"], ["y"], transB=1))
        0
        >>> _weight_axis(helper.make_node("MatMul", ["a", "w"], ["y"]))
        1
    """
    if node.op_type == "Conv":
        return 0
    if node.op_type == "Gemm" and any(a.name == "transB" and a.i for a in node.attribute):
        return 0
    return 1


def _lift_opset(model: Any) -> None:
    """Raise *model*'s default-domain opset to :data:`_FP16_QDQ_OPSET`, in place, keeping every op's meaning.

    FP16 quantization scales need opset 19, while RF-DETR exports at 17 by default. Between the two, reductions moved
    ``axes`` from an attribute to an input and ``Split`` gained ``num_outputs``; both are rewritten to the new form. Any
    other op whose schema changed in that range is refused rather than lifted unchecked.

    Args:
        model: ``ModelProto`` to lift; left untouched when its opset is already high enough.

    Raises:
        ValueError: If the graph holds an op whose meaning the lift does not know how to keep.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> node = helper.make_node("ReduceMax", ["x"], ["y"], "max", axes=[1], keepdims=0)
        >>> graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
        >>> model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        >>> _lift_opset(model)
        >>> model.opset_import[0].version, list(model.graph.node[0].input)
        (19, ['x', 'max_axes'])
    """
    from onnx import defs, helper

    # Imported here: exporter.py imports this module at module scope.
    from rfdetr.export._tensorrt.exporter import _iter_graphs, _tensor_names

    entries = [entry for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    current = max((entry.version for entry in entries), default=_FP16_QDQ_OPSET)
    if current >= _FP16_QDQ_OPSET:
        return
    taken = _tensor_names(model.graph)
    for graph in _iter_graphs(model.graph):
        for node in graph.node:
            if node.domain not in ("", "ai.onnx"):
                continue
            if node.op_type in _AXES_ATTRIBUTE_REDUCTIONS:
                _move_axes_to_input(node, graph, taken)
            elif node.op_type == "Split":
                # Opset 18 wants the output count spelled out when no `split` sizes are given; sizes keep the meaning.
                unsized = len(node.input) < 2 or not node.input[1]
                if unsized and not any(attribute.name == "num_outputs" for attribute in node.attribute):
                    node.attribute.append(helper.make_attribute("num_outputs", len(node.output)))
            elif node.op_type not in _TYPE_ONLY_CHANGES and (
                defs.get_schema(node.op_type, current).since_version
                != defs.get_schema(node.op_type, _FP16_QDQ_OPSET).since_version
            ):
                raise ValueError(
                    f"Cannot quantize this graph for TensorRT: node {node.name!r} ({node.op_type}) changed meaning "
                    f"between opset {current} and {_FP16_QDQ_OPSET}, which FP16 quantization scales need."
                )
    for entry in entries:
        entry.version = _FP16_QDQ_OPSET


def _move_axes_to_input(node: Any, graph: Any, taken: set[str]) -> None:
    """Rewrite a reduction's ``axes`` attribute (opset < 18) into the ``axes`` input opset 18 expects.

    Args:
        node: The reduction node, edited in place.
        graph: The graph holding *node*; the axes constant is added to its initializers.
        taken: Tensor names already bound, extended with the new constant's name.

    Examples:
        >>> from onnx import helper
        >>> node = helper.make_node("ReduceMean", ["x"], ["y"], "mean", axes=[-1])
        >>> graph = helper.make_graph([node], "g", [], [])
        >>> _move_axes_to_input(node, graph, {"x", "y"})
        >>> list(node.input), graph.initializer[0].name
        (['x', 'mean_axes'], 'mean_axes')
    """
    from onnx import numpy_helper

    # Imported here: exporter.py imports this module at module scope.
    from rfdetr.export._tensorrt.exporter import _unique_name

    axes = next((attribute for attribute in node.attribute if attribute.name == "axes"), None)
    if axes is None:
        return
    name = _unique_name(f"{node.name or node.op_type}_axes", taken)
    graph.initializer.append(numpy_helper.from_array(np.array(list(axes.ints), np.int64), name))
    node.attribute.remove(axes)
    node.input.append(name)


def insert_qdq(model: Any, plan: Int8Plan, ranges: Mapping[tuple[str, int], float]) -> None:
    """Insert *plan*'s INT8 Q/DQ pairs into the FP16 *model*, in place.

    Weights become INT8 initializers read through a per-output-channel ``DequantizeLinear``; each planned activation
    input gets one ``QuantizeLinear``/``DequantizeLinear`` pair, shared by every planned node reading the same tensor.
    Every scale is FP16, so the dequantized values stay FP16, and the opset is raised to the one that allows it.

    Args:
        model: The FP16 ``ModelProto`` (output of :func:`_cast_onnx_to_fp16`).
        plan: Nodes to quantize, by name.
        ranges: Absolute maximum per planned ``(node name, input index)``.

    Raises:
        ValueError: If the graph cannot be lifted to the opset FP16 scales need.
    """
    from onnx import helper, numpy_helper

    # Imported here: exporter.py imports this module at module scope.
    from rfdetr.export._tensorrt.exporter import _iter_graphs, _tensor_names, _unique_name

    _lift_opset(model)
    graph = model.graph
    taken = _tensor_names(graph)
    initializers = _constant_weights(graph)
    by_name = {node.name: node for node in graph.node}
    inserted: dict[str, list[Any]] = {}
    shared: dict[str, str] = {}
    zero = _unique_name("int8_zero_point", taken)
    new_initializers = [numpy_helper.from_array(np.array(0, np.int8), zero)]

    # Each shared pair goes in front of the first node, in this graph's own order, that reads it.
    position = {node.name: index for index, node in enumerate(graph.node)}
    for node_name, index in sorted(plan.activations, key=lambda key: (position[key[0]], key[1])):
        node = by_name[node_name]
        tensor = node.input[index]
        if tensor not in shared:
            scale = _unique_name(f"{tensor}_int8_scale", taken)
            quantized, dequantized = _unique_name(f"{tensor}_int8", taken), _unique_name(f"{tensor}_dequantized", taken)
            step = max(float(ranges[(node_name, index)]) / _INT8_MAX, _MIN_FP16_SCALE)
            new_initializers.append(numpy_helper.from_array(np.array(step, np.float16), scale))
            inserted.setdefault(node_name, []).extend(
                [
                    helper.make_node(
                        "QuantizeLinear", [tensor, scale, zero], [quantized], _unique_name(f"{quantized}_Q", taken)
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [quantized, scale, zero],
                        [dequantized],
                        _unique_name(f"{dequantized}_DQ", taken),
                    ),
                ]
            )
            shared[tensor] = dequantized
        node.input[index] = shared[tensor]

    for node_name in plan.weighted:
        node = by_name[node_name]
        axis = _weight_axis(node)
        values, scales = _quantized_weight(numpy_helper.to_array(initializers[node.input[1]]), axis)
        weight, scale = _unique_name(f"{node.input[1]}_int8", taken), _unique_name(f"{node.input[1]}_int8_scale", taken)
        point, dequantized = _unique_name(f"{weight}_zero_point", taken), _unique_name(f"{weight}_dequantized", taken)
        new_initializers += [
            numpy_helper.from_array(values, weight),
            numpy_helper.from_array(scales, scale),
            numpy_helper.from_array(np.zeros(scales.shape, np.int8), point),
        ]
        inserted.setdefault(node_name, []).append(
            helper.make_node(
                "DequantizeLinear",
                [weight, scale, point],
                [dequantized],
                _unique_name(f"{dequantized}_DQ", taken),
                axis=axis,
            )
        )
        node.input[1] = dequantized

    ordered: list[Any] = []
    for node in graph.node:
        ordered.extend(inserted.get(node.name, ()))
        ordered.append(node)
    del graph.node[:]
    graph.node.extend(ordered)
    graph.initializer.extend(new_initializers)
    # An initializer captured by an If/Loop body is read there, not by a top-level node.
    still_read = {name for nested in _iter_graphs(graph) for node in nested.node for name in node.input}
    still_read |= {output.name for output in graph.output}
    kept = [init for init in graph.initializer if init.name in still_read]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def _refuse_unquantizable_source(model: Any, onnx_path: str) -> None:
    """Refuse a graph INT8 quantization cannot start from: quantized, not float32, dynamic batch, or not a detector.

    Each would otherwise fail late or point the wrong way: a quantized graph plans nothing to quantize, and an FP16
    graph is calibrated in full before the FP16 cast refuses it with advice (``fp16=False``) that an INT8 request
    cannot follow.

    Args:
        model: The loaded source graph.
        onnx_path: The caller's file, named in the error.

    Raises:
        ValueError: If the graph holds Q/DQ nodes, FP16 inputs or weights, or a dynamic batch axis, or its outputs are
            not exactly a detector's ``dets`` and ``labels`` (a segmentation, keypoint or backbone-only export).

    Examples:
        >>> from onnx import TensorProto, helper
        >>> graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
        ...     [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1])])
        >>> _refuse_unquantizable_source(helper.make_model(graph), "m.onnx")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: 'm.onnx' is not a float32 export; ...
    """
    from onnx import TensorProto

    # Imported here: exporter.py imports this module at module scope.
    from rfdetr.export._tensorrt.exporter import (
        _INT8_OUTPUT_NAMES,
        _QUANTIZATION_OP_TYPES,
        _iter_graphs,
        _onnx_dynamic_batch_inputs,
    )

    if any(node.op_type in _QUANTIZATION_OP_TYPES for graph in _iter_graphs(model.graph) for node in graph.node):
        raise ValueError(
            f"'{onnx_path}' is already explicitly quantized, and quantization='int8' quantizes a float32 export. "
            "Build this graph as it is with quantization=None and fp16=False."
        )
    inputs = (value.type.tensor_type.elem_type for value in model.graph.input)
    if TensorProto.FLOAT16 in inputs or any(init.data_type == TensorProto.FLOAT16 for init in model.graph.initializer):
        raise ValueError(
            f"'{onnx_path}' is not a float32 export; quantization='int8' calibrates and quantizes the float32 graph. "
            "Build from the float32 .onnx that RFDETR.export writes."
        )
    if _onnx_dynamic_batch_inputs(model.graph):
        raise ValueError(
            f"'{onnx_path}' has a dynamic batch axis, and an INT8 engine is built for a static batch. Re-export the "
            "ONNX graph with a fixed batch_size and dynamic_batch=False."
        )
    outputs = tuple(output.name for output in model.graph.output)
    if outputs != _INT8_OUTPUT_NAMES:
        raise ValueError(
            f"'{onnx_path}' outputs {list(outputs)}, and quantization='int8' is measured for detection models only, "
            f"whose outputs are {list(_INT8_OUTPUT_NAMES)}. Build this graph with quantization=None."
        )


@contextlib.contextmanager
def int8_source_graph(
    onnx_path: str,
    *,
    calibration_data: str | Path | NDArray[Any],
    max_images: int,
    dynamic_batch: bool | None,
) -> Iterator[str]:
    """Provide an INT8-quantized FP16 copy of *onnx_path* to build from, deleting it when the block exits.

    Args:
        onnx_path: The FP32 ``.onnx`` export.
        calibration_data: Directory of images, ``.npy`` path or array, as accepted by
            :func:`~rfdetr.export._runtime.calibration.calibration_batches`.
        max_images: Maximum images read from a *calibration_data* directory.
        dynamic_batch: The batch request the engine is built for, forwarded to the FP16 cast.

    Yields:
        Path to the quantized graph, valid only inside the ``with`` block.

    Raises:
        ValueError: If the graph is already quantized, is not float32, has a dynamic batch axis, cannot be lifted to
            opset 19, is not a quantizable RF-DETR detector (its outputs must be exactly ``dets`` and ``labels``), or
            holds attention INT8 cannot be placed around; or if the calibration data is unusable, holds a NaN or
            infinite value, or gives a range that is not finite or does not fit an FP16 scale; or if a calibration
            image cannot be identified as an image (the message names the file). An image Pillow identifies but
            cannot decode (a truncated file) raises Pillow's own ``OSError``, which does not name the file.
    """
    import onnx

    from rfdetr.export._runtime.calibration import calibration_batches

    # Imported here: exporter.py imports this module at module scope.
    from rfdetr.export._tensorrt.exporter import fp16_source_graph

    source = onnx.load(onnx_path)
    _refuse_unquantizable_source(source, onnx_path)
    # Lifting the FP32 source too keeps its meaning and makes an unliftable graph fail here, before calibration.
    _lift_opset(source)
    plan = plan_int8(source)
    logger.info(
        f"INT8 plan: {len(plan.weighted)} weighted layer(s), {len(plan.activations)} quantized activation input(s)"
    )
    dims = source.graph.input[0].type.tensor_type.shape.dim
    channels, height, width = (d.dim_value for d in dims[1:4])
    keys = list(plan.activations)
    tensors = sorted({_input_tensor(source.graph, name, index) for name, index in keys})
    batches = calibration_batches(
        calibration_data, height=height, width=width, channels=channels, max_images=max_images
    )
    with contextlib.closing(batches):  # releases a memory-mapped ``.npy`` even when calibration raises
        by_tensor = calibrate_ranges(onnx_path, source, tensors, batches)
    unusable = sorted(t for t, r in by_tensor.items() if not np.isfinite(r) or r / _INT8_MAX > _MAX_FP16)
    if unusable:
        raise ValueError(
            f"Calibration gave {len(unusable)} tensor(s) a range that is not finite or does not fit an FP16 scale, "
            f"e.g. {unusable[0]!r} = {by_tensor[unusable[0]]}. Check that calibration_data is normalized as predict() "
            "normalizes."
        )
    ranges = {key: by_tensor[_input_tensor(source.graph, *key)] for key in keys}
    del source

    with fp16_source_graph(onnx_path, dynamic_batch=dynamic_batch) as fp16_path:
        model = onnx.load(fp16_path)
    insert_qdq(model, plan, ranges)
    stem = os.path.basename(os.path.splitext(onnx_path)[0])
    handle, int8_path = tempfile.mkstemp(prefix=f"{stem}.int8-", suffix=".onnx", dir=os.path.dirname(onnx_path) or ".")
    os.close(handle)
    try:
        onnx.save(model, int8_path)
        del model
        yield int8_path
    finally:
        with contextlib.suppress(OSError):
            Path(int8_path).unlink(missing_ok=True)
