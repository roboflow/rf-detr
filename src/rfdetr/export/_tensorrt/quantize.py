# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Explicit INT8 quantization of the FP16 graph a TensorRT engine is built from.

TensorRT 11 has no INT8 builder flag and no calibrator: an INT8 engine is a strongly typed build of a graph that
carries its own ``QuantizeLinear``/``DequantizeLinear`` pairs. Where those pairs sit decides whether the engine is
faster than FP16 at all, so the placement here is not "quantize every matrix multiply". It follows what was measured
on RF-DETR Nano (RTX 5070, TensorRT 11.3 and 10.16), for the attention token limit on Small to Large (TensorRT
11.3), for the segmentation and keypoint models on an RTX 5070 with TensorRT 11.4, and for the patch-embedding matrix
multiply on Nano to Large, Seg Nano, Seg Small and Keypoint Preview with TensorRT 11.4 (Nano to Large also 10.16 on a
Tesla T4). The FP16 matrix multiply of the segmentation and keypoint models' patch embedding was built for Seg Nano
to Seg Large and Keypoint Preview on an RTX 5070 and a Tesla T4 with TensorRT 11.4; issue #1024:

* **FP16 scales on the FP16 graph.** Q/DQ is inserted into the graph :func:`_cast_onnx_to_fp16` produces, with FP16
  scales, so every region left unquantized stays FP16. FP32 scales make each ``DequantizeLinear`` return FP32 and drag
  the rest of the graph to FP32, which is slower than FP16 before any INT8 kernel runs.
* **Weight-bearing ``MatMul``/``Gemm``/``Conv`` inputs**, in the backbone encoder and the decoder (plus the batched
  multiplies of a block whose attention runs in INT8, below). The output of
  each quantized operation stays in floating point, so TensorRT fuses the bias, LayerScale and residual add into the
  INT8 kernel's epilogue and the quantize step into the preceding LayerNorm or GELU. The projector, the two-stage
  proposal head and the detection heads stay FP16: quantizing them costs accuracy and buys nothing (on Seg Nano,
  Seg Small, Seg Large and Keypoint Preview the projector was no faster in INT8 either). The mask head stays FP16
  too: quantized as a whole (Seg Nano, Seg Small, Seg Large) it made the engine slower, and its matrix multiplies alone
  were no faster. The keypoint heads sit outside the quantized regions and stay FP16: quantizing the keypoint
  embedding of Keypoint Preview was no faster and lowered its keypoint AP.
* **The patch embedding is a matrix multiply, quantized only when its patch is a multiple of 32 values.** TensorRT ran
  the INT8 patch embedding, a ``Conv`` over the 3-channel image, on the image padded to 32 channels and stored the 29
  added zero channels in the engine as a ``B x 29 x H x W`` FP16 constant, read on every run: at batch 8 the engine was
  about twice as large and INT8 clearly less far ahead of FP16. Only that convolution was measured; any convolution
  whose input channels are not a multiple of 32 is kept FP16 as a precaution. An FP16 convolution can pad too: the Tesla
  T4 segmentation and keypoint INT8 engines, whose patch embedding was an FP16 convolution, grew with the batch by about
  the image padded to 32 channels (inferred from the engine sizes, not read from an engine).
  :func:`_patch_embedding_as_matmul` therefore turns each backbone patch embedding it can (see the function) into a
  matrix multiply over ``3 * 16 * 16`` values per patch (``3 * 12 * 12`` for the segmentation and keypoint models),
  which has no channels to pad. The detection models' 768 values take INT8 with the rest of the backbone. The 432
  values of the 12-pixel patches are not a multiple of 32: on Seg Small and Keypoint Preview their INT8 matrix
  multiply ran no faster than the FP16 one and lowered the AP, so it stays FP16.
* **Attention is all INT8 or all FP16.** An attention block that TensorRT's INT8 fused-attention kernel accepts and
  that was measured to pay for it (faster at a small cost in AP; head size 16, 32 or 64 and at most 325 tokens: the
  windowed backbone blocks of detection Nano, Small and Medium and of segmentation Small and Medium) gets Q/DQ on both
  batched matrix multiplies as well as on its q/k/v/output projections, which keeps attention fused in INT8. Any other
  block (the windowed blocks of detection Large, Seg Nano, Seg Large and Keypoint Preview, the global backbone blocks,
  the decoder self-attention and the keypoint decoder's ``kp_inst_self_attn``) keeps its projections and batched
  multiplies in FP16: quantizing only the projections makes TensorRT emit FP32 projections plus reshape kernels in
  front of the fused FP16 kernel, and a quantized output projection behind FP16 attention makes TensorRT 11 fuse an
  INT8 output into that kernel, which returns wrong values.
* **A GELU never feeds an FP16 consumer from an INT8 multiply.** TensorRT 10.16 and 11.3 return NaN for an INT8
  ``MatMul`` -> bias -> GELU -> FP16 consumer chain, so an MLP's first projection is quantized only when its second one
  is.
* **A detection model's backbone norms are smoothed.** A backbone ``LayerNormalization`` output is quantized per
  tensor, and its widest channels, several times wider than its median one, set the one INT8 step of every channel.
  SmoothQuant (:func:`measure_smoothing_scales`) divides each channel of the norm's scale and bias by a per-channel
  factor and multiplies the matching rows of the projections that read the norm by it, moving range from the
  activation into the weights, which are quantized per output channel. Every output is unchanged but for rounding, and
  no op is added.
* **A detection model's backbone MLPs get their bias corrected.** Rounding an activation to INT8 shifts its mean per
  channel, most of all for the one-sided GELU output an MLP's second projection reads, and each projection passes that
  shift on to its output channels. :func:`measure_mlp_bias_shifts` measures the shift on the calibration images
  (``E[Q(x) Q(W)] - E[x W]`` per output channel, with the engine's scales and per-channel INT8 weights, up to one FP16
  rounding of a smoothed weight) and the shift is subtracted from the layer's bias. Again only constants change; a new
  op on the GELU path would not be safe: a no-op ``Sub`` there moved ``fc1``'s GELU and quantize out of the GEMM's own
  epilogue and cost Nano more box AP than the whole INT8 plan does (TensorRT 11.4).
  Together the two lowered the box AP lost on COCO val2017 for Nano to Large on every calibration set measured (RTX
  5070, TensorRT 11.4). On Nano, correcting the attention and decoder layers' biases too, correcting each layer for
  the drift of the layers before it, and fitting a gain per channel measured no better. The segmentation and keypoint
  models share the backbone but were not measured with either, so their norms and biases are left alone.

Activation ranges come from calibration images run through the FP32 graph on onnxruntime's CPU provider: the absolute
maximum of each quantized tensor, which measured better than percentile, MSE and entropy clipping on this model.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import warn_if_too_few_samples
from rfdetr.export._tensorrt._graph import (
    _QUANTIZATION_OP_TYPES,
    _iter_graphs,
    _onnx_dynamic_batch_inputs,
    _tensor_names,
    _unique_name,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: The only quantization mode a TensorRT export accepts besides ``None`` (FP16 or FP32, chosen with ``fp16``).
INT8 = "int8"

#: The graph outputs an INT8 export is measured for: a detector's, plus a segmentation model's mask logits or a keypoint
#: model's keypoints. The heads that produce them sit outside the quantized regions and stay FP16. Any other set (a
#: backbone-only export, a graph with both outputs) is refused.
_INT8_OUTPUT_NAMES = (("dets", "labels"), ("dets", "labels", "masks"), ("dets", "labels", "keypoints"))

#: A detection model's graph outputs: only these models get their backbone norms smoothed and their backbone MLP biases
#: corrected, as the segmentation and keypoint models were not measured with either.
_DETECTION_OUTPUT_NAMES = _INT8_OUTPUT_NAMES[0]

#: SmoothQuant's migration strength: the share of each LayerNorm output channel's range moved into the weights that read
#: it (0.5 balances the two). Measured on Nano at 0.5 and 0.75, which gave the same AP; 0.5 is what the bias correction
#: was measured on top of.
_SMOOTHING_ALPHA = 0.5

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
#: or a new attribute whose default keeps the old behavior (``Cast``, ``CastLike``: ``saturate``), so lifting the opset
#: import alone keeps their meaning.
_TYPE_ONLY_CHANGES = frozenset(
    {"Cast", "CastLike", "Constant", "Equal", "Identity", "If", "Loop", "Reshape", "Scan", "Shape", "Size"}
)

#: Ops whose schema changed between these opsets and :data:`_FP16_QDQ_OPSET` only by a new attribute whose default keeps
#: the old behavior (``Resize``: ``antialias``, ``axes``, ``keep_aspect_ratio_policy``), a new value of an attribute
#: (``Resize``: the ``half_pixel_symmetric`` coordinate mode, which an older graph cannot carry; ``ScatterND``:
#: ``reduction`` ``max`` and ``min``) or more types. A graph at a lower opset is checked as any other op: before opset
#: 11 ``Resize`` took ``scales`` as its second input, opset 11 inserted ``roi`` ahead of it, and opsets 11 and 12 still
#: accept the ``tf_half_pixel_for_nn`` coordinate mode that opset 13 removed.
_UNCHANGED_SINCE = {"Resize": 13, "ScatterND": 16}

#: Head sizes TensorRT's INT8 fused-attention kernel accepts (TensorRT 11.3 "Attention fusion", SM75-90, SM120-121).
_FUSED_INT8_HEAD_SIZES = frozenset({16, 32, 64})

#: Longest query/key sequence that gets INT8 attention. TensorRT's INT8 fused-attention kernel accepts up to 512; this
#: is the largest window measured to pay (Medium: a little faster from a CUDA graph, about equal with a plain call, at a
#: small cost in AP). Large's 485-token windows ran no faster than with FP16 attention and lost more AP. Seg Large's
#: 442-token windows ran a little faster and lost more AP than the speedup justifies; Keypoint Preview's 577 and Seg
#: Nano's 677 tokens are above the kernel's 512 and ran much slower. Other lengths from 326 to 484 are not measured (RTX
#: 5070, TensorRT 11.3 and 11.4, COCO val2017).
_INT8_ATTENTION_MAX_TOKENS = 325

#: Node-name prefixes of the regions whose weight-bearing operations are quantized: backbone encoder and decoder. They
#: are the scope names the TorchScript ONNX exporter gives nodes (``dynamo=False`` in ``export/_onnx/exporter.py``)
#: under the current module attribute paths; another exporter or a renamed module matches no layer, and the planner
#: refuses.
_QUANTIZED_REGIONS = ("/backbone/backbone.0/encoder/", "/transformer/decoder/")

#: Node-name prefix of the backbone encoder, the first of the quantized regions: the patch embedding sits in it.
_BACKBONE_ENCODER = _QUANTIZED_REGIONS[0]

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
#: A patch embedding is quantized as a matrix multiply only when its flattened patch is a multiple of this; one that is
#: not (the segmentation and keypoint models' 432 values) is a matrix multiply that stays FP16.
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


def _consumers(graph: Any) -> dict[str, list[Any]]:
    """Map every tensor that a node of *graph* reads to those nodes, in graph order.

    Only the top-level nodes are read: a tensor read inside an ``If``/``Loop`` subgraph is not listed for that read. A
    node that reads one tensor through two of its inputs is listed twice.

    Examples:
        >>> from onnx import helper
        >>> graph = helper.make_graph(
        ...     [helper.make_node("Relu", ["x"], ["y"], "relu"), helper.make_node("Add", ["x", "y"], ["z"], "add")],
        ...     "g", [], [])
        >>> {name: [node.name for node in nodes] for name, nodes in _consumers(graph).items()}
        {'x': ['relu', 'add'], 'y': ['add']}
    """
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


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


class _PatchEmbedding(NamedTuple):
    """The shape of a patch embedding: the image it reads, the patches it cuts and the channels it produces."""

    batch: int
    channels: int
    height: int
    width: int
    patch: int
    out_channels: int


def _patch_kernel(node: Any, initializers: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Return ``(patch, channels, out_channels)`` when *node*'s kernel cuts non-overlapping patches, else ``None``.

    A patch embedding is an undilated ``Conv`` whose square kernel equals its stride and which has no padding. Whether
    the kernel spans every input channel (it does not in a grouped convolution) is for :func:`_patch_image`, which sees
    the image.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> weights = {"w": numpy_helper.from_array(np.zeros((8, 2, 4, 4), np.float32), "w")}
        >>> conv = helper.make_node("Conv", ["image", "w"], ["y"], "patch", kernel_shape=[4, 4], strides=[4, 4])
        >>> _patch_kernel(conv, weights)
        (4, 2, 8)
        >>> _patch_kernel(helper.make_node("Conv", ["image", "w"], ["y"], "patch"), weights) is None
        True
    """
    from onnx import helper

    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    out_channels, channels, *kernel = (int(d) for d in initializers[node.input[1]].dims)
    patch = kernel[0]
    plain = (
        kernel[0] == kernel[1]
        and list(attributes.get("strides", [1, 1])) == kernel
        and list(attributes.get("dilations", [1, 1])) == [1, 1]
        and not any(attributes.get("pads", [0]))
        and attributes.get("auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET")
    )
    if not plain:
        return None
    return patch, channels, out_channels


def _patch_image(dims: tuple[int | None, ...] | None, kernel: tuple[int, int, int]) -> _PatchEmbedding | None:
    """Return the patch embedding that *kernel* makes of an image of shape *dims*, or ``None`` when it does not fit.

    The image must be ``(batch, channels, height, width)`` with every size known and non-empty, the channels must be
    the ones the kernel spans, and the patches must tile the height and the width exactly.

    Examples:
        >>> _patch_image((1, 2, 8, 12), (4, 2, 8))
        _PatchEmbedding(batch=1, channels=2, height=8, width=12, patch=4, out_channels=8)
        >>> _patch_image((1, 2, 8, 10), (4, 2, 8)) is None, _patch_image((1, 2, None, 8), (4, 2, 8)) is None
        (True, True)
    """
    patch, channels, out_channels = kernel
    known = [d for d in dims or () if d]  # drops a symbolic (None) and an empty (0) dimension
    if len(known) != 4 or known[1] != channels or known[2] % patch or known[3] % patch:
        return None
    return _PatchEmbedding(known[0], channels, known[2], known[3], patch, out_channels)


def _patch_matmul_nodes(
    conv: Any,
    matmul_name: str,
    patches: _PatchEmbedding,
    graph: Any,
    weight: NDArray[Any],
    taken: set[str],
    node_names: set[str],
) -> list[Any]:
    """Build the nodes that compute *conv*'s output as a ``MatMul`` over flattened patches.

    The new constants are appended to *graph*'s initializers. The patches are cut with a ``Reshape`` and a
    ``Transpose`` into rows of ``(channel, row, column)`` values, which is the order a ``Conv`` kernel flattens in, so
    the ``MatMul`` against the transposed, flattened kernel is the same sum. The last node writes *conv*'s own output
    tensor, so everything downstream (a graph output included) is untouched.

    Args:
        conv: The ``Conv`` being replaced.
        matmul_name: Name of the new ``MatMul``, which the INT8 plan finds by its prefix.
        patches: The shape :func:`_patch_image` found for *conv*.
        graph: The graph holding *conv*.
        weight: *conv*'s kernel, ``(out_channels, channels, patch, patch)``, in the dtype the graph computes in.
        taken: Tensor names already bound in the graph, extended with the new ones.
        node_names: Node names already in the graph, extended with the new ones.

    Returns:
        The replacement nodes, in order.
    """
    from onnx import helper, numpy_helper

    batch, channels, height, width, patch, out_channels = patches
    stem = matmul_name.rsplit("/", 1)[0]
    rows, cols = (height // patch) * (width // patch), channels * patch * patch
    tensors = {
        key: _unique_name(f"{stem}/{key}", taken)
        for key in (
            "cut_shape",
            "cut",
            "ordered",
            "rows_shape",
            "rows",
            "matrix",
            "product",
            "biased",
            "image_shape",
            "image",
        )
    }
    constants = {
        "cut_shape": [batch, channels, height // patch, patch, width // patch, patch],
        "rows_shape": [batch, rows, cols],
        "image_shape": [batch, height // patch, width // patch, out_channels],
    }
    for key, dims in constants.items():
        graph.initializer.append(numpy_helper.from_array(np.array(dims, np.int64), tensors[key]))
    graph.initializer.append(
        numpy_helper.from_array(np.ascontiguousarray(weight.reshape(out_channels, -1).T), tensors["matrix"])
    )

    def node(op_type: str, inputs: list[str], output: str, name: str, **attributes: Any) -> Any:
        """Build one replacement node called ``<stem>/<name>``, which no other node in the graph has."""
        return helper.make_node(op_type, inputs, [output], _unique_name(f"{stem}/{name}", node_names), **attributes)

    nodes = [
        node("Reshape", [conv.input[0], tensors["cut_shape"]], tensors["cut"], "Cut"),
        node("Transpose", [tensors["cut"]], tensors["ordered"], "Order", perm=[0, 2, 4, 1, 3, 5]),
        node("Reshape", [tensors["ordered"], tensors["rows_shape"]], tensors["rows"], "Rows"),
        helper.make_node("MatMul", [tensors["rows"], tensors["matrix"]], [tensors["product"]], matmul_name),
    ]
    node_names.add(matmul_name)
    result = tensors["product"]
    if len(conv.input) > 2 and conv.input[2]:
        result = tensors["biased"]
        nodes.append(node("Add", [tensors["product"], conv.input[2]], result, "Bias"))
    nodes.append(node("Reshape", [result, tensors["image_shape"]], tensors["image"], "Image"))
    nodes.append(node("Transpose", [tensors["image"]], conv.output[0], "Channels", perm=[0, 3, 1, 2]))
    return nodes


def _drop_unread_initializers(graph: Any, names: Iterable[str]) -> None:
    """Remove the initializers called *names* that no node, subgraph, graph input or graph output refers to any more.

    Examples:
        >>> from onnx import TensorProto, helper, numpy_helper
        >>> graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [], [],
        ...     [numpy_helper.from_array(np.zeros(1, np.float32), name) for name in ("old", "x")])
        >>> _drop_unread_initializers(graph, ["old", "x"])
        >>> [init.name for init in graph.initializer]
        ['x']
    """
    read = {name for nested in _iter_graphs(graph) for node in nested.node for name in node.input}
    read |= {value.name for value in (*graph.input, *graph.output)}
    for initializer in [init for init in graph.initializer if init.name in set(names) - read]:
        graph.initializer.remove(initializer)


def _patch_embedding_as_matmul(model: Any, *, multiple_of: int = 1) -> int:
    """Rewrite the backbone's patch-embedding ``Conv`` as a ``MatMul`` over flattened patches, in place.

    TensorRT pads a convolution's 3 input channels (to 32 for an INT8 convolution; an FP16 one may pad too, inferred
    from Tesla T4 engine sizes) and stores the padding in the engine as a constant that grows with the batch (see
    :func:`_pads_input_channels`). As a matrix multiply over ``channels * patch * patch`` values per patch (768 for an
    RGB image and 16-pixel patches, 432 for 12-pixel ones) it has no channels to pad, and an INT8 build can quantize it
    like the other backbone layers. The two compute the same sum.

    Only a ``Conv`` inside the backbone encoder that :func:`_pads_input_channels` would flag and that
    :func:`_patch_kernel` and :func:`_patch_image` accept is rewritten, unless its ``<parent>/MatMul`` name is already
    taken, in which case it is left as it is. The shape inference that finds the image size
    runs only when such a ``Conv`` exists. A kernel that is also a graph input is left alone, since it can be replaced
    at run time. The kernel of a replaced ``Conv`` is dropped when nothing else reads it (one reached through an
    ``Identity`` alias is kept). The rewrite is deterministic for a given name, so applying it to the FP32 graph that is
    planned and calibrated and to the FP16 graph that is quantized yields the same ``MatMul`` name.

    Args:
        model: ``ModelProto`` to rewrite; left untouched when it holds no such layer.
        multiple_of: At least 1. Rewrite only a patch whose flattened ``channels * patch * patch`` values are a
            multiple of this. The INT8 plan passes :data:`_INT8_CONV_CHANNEL_MULTIPLE`, which keeps the 768 values of a
            16-pixel patch and leaves out the 432 of a 12-pixel one: on Seg Small and Keypoint Preview that INT8 matrix
            multiply ran no faster than the FP16 one and lowered the AP. The default rewrites every patch embedding.

    Returns:
        The number of layers rewritten.

    Raises:
        ValueError: If *multiple_of* is below 1.
    """
    import onnx
    from onnx import numpy_helper

    if multiple_of < 1:
        raise ValueError(f"multiple_of must be at least 1, got {multiple_of}.")
    graph = model.graph
    initializers = _constant_weights(graph)
    counts = Counter(node.name for node in graph.node)
    overridable = {value.name for value in graph.input}  # a kernel listed here can be replaced at run time
    candidates = [
        (node, kernel)
        for node in graph.node
        if node.op_type == "Conv"
        and node.name.startswith(_BACKBONE_ENCODER)
        and counts[node.name] == 1
        and node.input[1] not in overridable
        and _weight_name(node, initializers)
        and _pads_input_channels(node, initializers)
        and (kernel := _patch_kernel(node, initializers))
        and (kernel[1] * kernel[0] ** 2) % multiple_of == 0
    ]
    if not candidates:
        return 0
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: v for v in (*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output)}
    taken, node_names = _tensor_names(graph), set(counts)
    replacements: dict[str, list[Any]] = {}
    for conv, kernel in candidates:
        patches = _patch_image(_dims(shapes, conv.input[0]), kernel)
        matmul_name = f"{conv.name.rsplit('/', 1)[0]}/MatMul"
        if patches is None or matmul_name in node_names:
            continue
        weight = numpy_helper.to_array(initializers[conv.input[1]])
        replacements[conv.name] = _patch_matmul_nodes(conv, matmul_name, patches, graph, weight, taken, node_names)
    if replacements:
        kernels = [conv.input[1] for conv, _ in candidates if conv.name in replacements]
        ordered = [new for node in graph.node for new in replacements.get(node.name, [node])]
        del graph.node[:]
        graph.node.extend(ordered)
        _drop_unread_initializers(graph, kernels)
    return len(replacements)


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
    consumers = _consumers(graph)

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
                    "model exports only (detection, segmentation or keypoint)."
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
        ValueError: If the graph has nothing to quantize, which means it is not an RF-DETR model export; if an FP16
            attention block's output projection cannot be identified; or if a top-level ``Softmax`` feeds a
            multiply of two activations without being a recognised attention block.
    """
    graph = model.graph
    initializers = _constant_weights(graph)
    consumers = _consumers(graph)
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
                "could not be identified. INT8 TensorRT export applies to RF-DETR model exports only (detection, "
                "segmentation or keypoint)."
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
            "RF-DETR model exports only (detection, segmentation or keypoint). Layers are found by the node-name "
            f"prefixes {list(_QUANTIZED_REGIONS)} that the TorchScript ONNX exporter (dynamo=False) gives them."
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


def _graph_batches(
    batches: Iterable[NDArray[np.float32]], batch: int | None
) -> Iterator[tuple[NDArray[np.float32], int]]:
    """Group single-image calibration batches into the batch size a static graph was exported at.

    A short final group is padded by repeating its last image, and each group comes with the number of real images it
    holds, first in the group. A maximum is not changed by a repeated image; a mean must leave the copies out.

    Examples:
        >>> singles = [np.full((1, 1, 1, 1), v, np.float32) for v in range(3)]
        >>> [(b[:, 0, 0, 0].tolist(), real) for b, real in _graph_batches(singles, 2)]
        [([0.0, 1.0], 2), ([2.0, 2.0], 1)]
    """
    group: list[NDArray[np.float32]] = []
    for sample in batches:
        group.append(sample)
        if len(group) == (batch or 1):
            yield np.concatenate(group), len(group)
            group = []
    if group:
        real = len(group)
        group += [group[-1]] * ((batch or 1) - real)
        yield np.concatenate(group), real


def _calibration_graph(model: Any, tensors: list[str]) -> Any:
    """Return a copy of *model* that also outputs ``max(abs(t))`` for each of *tensors*.

    Reducing inside the graph returns one scalar per tensor rather than every activation. A NaN anywhere in a tensor is
    measured as ``inf``, so a range check that refuses non-finite ranges sees it.

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
    from onnx import TensorProto, helper, numpy_helper

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    taken = _tensor_names(probe.graph)
    infinity = _unique_name("calibration_infinity", taken)
    probe.graph.node.append(
        helper.make_node("Constant", [], [infinity], value=numpy_helper.from_array(np.array(np.inf, np.float32)))
    )
    for tensor in tensors:
        guarded = _absolute_with_nan_as_infinity(probe.graph, tensor, tensor, infinity, taken)
        maximum = _unique_name(f"{tensor}__absmax", taken)
        probe.graph.node.append(helper.make_node("ReduceMax", [guarded], [maximum], keepdims=0))
        probe.graph.output.append(helper.make_tensor_value_info(maximum, TensorProto.FLOAT, []))
    return probe


def _absolute_with_nan_as_infinity(graph: Any, tensor: str, stem: str, infinity: str, taken: set[str]) -> str:
    """Append the absolute value of *tensor*, with every NaN made ``+inf``, to *graph*; return the result's name.

    onnxruntime's ``ReduceMax`` skips a NaN anywhere but the first element, so a maximum taken over this result reports
    a NaN as an infinite value, which a refusal of non-finite values sees.

    Args:
        graph: The probe ``GraphProto`` to append the nodes to.
        tensor: The tensor to read.
        stem: The prefix of the new tensor names.
        infinity: A scalar ``+inf`` FP32 tensor of *graph*.
        taken: The names already used in *graph*; the new names are added to it.

    Returns:
        The name of the guarded absolute value.

    Examples:
        >>> from onnx import helper
        >>> graph = helper.make_graph([], "g", [], [])
        >>> _absolute_with_nan_as_infinity(graph, "x", "x", "inf", {"x", "inf"}), [n.op_type for n in graph.node]
        ('x__guarded', ['Abs', 'IsNaN', 'Where'])
    """
    from onnx import helper

    absolute, nan, guarded = (_unique_name(f"{stem}__{suffix}", taken) for suffix in ("abs", "isnan", "guarded"))
    graph.node.extend(
        [
            helper.make_node("Abs", [tensor], [absolute]),
            helper.make_node("IsNaN", [tensor], [nan]),
            helper.make_node("Where", [nan, infinity, absolute], [guarded]),
        ]
    )
    return guarded


@contextlib.contextmanager
def _probe_session(onnx_path: str, probe: Any) -> Iterator[Any]:
    """Yield an onnxruntime CPU session of *probe*, saved to a temporary file beside *onnx_path* and deleted after.

    The CPU, not CUDA: at batch 32 the FP32 graph needs about 9 GB of GPU memory under onnxruntime, more than a 12 GB
    card has left beside the export, while the CPU runs one pass over 128 images in about 15 s. Priority-based order:
    under the default one, the probe's added reductions raised peak host memory far above the plain graph's on an
    RF-DETR Nano export, for the same ranges.

    Args:
        onnx_path: The FP32 ``.onnx`` file the probe was made from.
        probe: The probe ``ModelProto``.

    Yields:
        The session.
    """
    import onnx
    import onnxruntime as ort

    stem = os.path.basename(os.path.splitext(onnx_path)[0])
    handle, probe_path = tempfile.mkstemp(
        prefix=f"{stem}.calibration-", suffix=".onnx", dir=os.path.dirname(onnx_path) or "."
    )
    os.close(handle)
    try:
        onnx.save(probe, probe_path)
        options = ort.SessionOptions()
        options.execution_order = ort.ExecutionOrder.PRIORITY_BASED
        yield ort.InferenceSession(probe_path, sess_options=options, providers=["CPUExecutionProvider"])
    finally:
        try:
            Path(probe_path).unlink(missing_ok=True)
        except OSError as error:
            logger.warning(f"Could not delete the calibration probe graph {probe_path!r}: {error}")


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
        ValueError: If *tensors* or *batches* is empty, or *batches* holds a NaN or infinite value.

    Logs a warning, after the last batch, when fewer than ``MIN_CALIBRATION_SAMPLES`` images were seen.
    """
    if not tensors:
        # onnxruntime treats an empty output list as every output, so the slice below would measure the wrong tensors.
        raise ValueError("Calibration was given no tensor to measure.")
    probe = _calibration_graph(model, tensors)
    # The probe outputs follow the model's own, one per tensor in order (names may carry a collision prefix).
    outputs = [output.name for output in probe.graph.output][-len(tensors) :]
    ranges: NDArray[np.float64] = np.zeros(len(tensors), dtype=np.float64)
    count = images = 0
    for values, real in _run_probe(onnx_path, probe, outputs, batches):
        # A copy padding the last group repeats an image, which cannot change a maximum.
        ranges = np.maximum(ranges, np.asarray(values, dtype=np.float64))
        count += 1
        images += real
    warn_if_too_few_samples(images)
    logger.info(f"Calibrated {len(tensors)} INT8 activation ranges over {count} batch(es)")
    return dict(zip(tensors, ranges.tolist()))


def _run_probe(
    onnx_path: str, probe: Any, names: list[str], batches: Iterable[NDArray[np.float32]]
) -> Iterator[tuple[list[NDArray[np.float64]], int]]:
    """Run *batches* through *probe*, yielding its *names* outputs for each group with the number of real images in it.

    The images are grouped into the probe's static batch as calibration groups them, so the last group may end in
    copies of its last image (see :func:`_graph_batches`); an output with one row per image keeps only its first rows
    for the real ones.

    Args:
        onnx_path: The FP32 ``.onnx`` file the probe was made from; the probe is written to a temporary file beside it.
        probe: The probe ``ModelProto``.
        names: The probe outputs to read.
        batches: Preprocessed ``(1, C, H, W)`` images.

    Yields:
        The values of *names*, in order, and the number of real images in the group.

    Raises:
        ValueError: If *batches* is empty or holds a NaN or infinite value, which is refused before the group holding it
            runs, on whichever pass over the images reads it first.
    """
    images = 0
    with _probe_session(onnx_path, probe) as session:
        feed_name = session.get_inputs()[0].name
        dimension = session.get_inputs()[0].shape[0]
        groups = _graph_batches(batches, dimension if isinstance(dimension, int) else None)
        for number, (group, real) in enumerate(groups, start=1):
            # Refused here rather than as a non-finite result afterwards, so the message names the batch.
            if not np.isfinite(group).all():
                raise ValueError(f"Calibration batch {number} holds NaN or infinite values.")
            yield [np.asarray(value, dtype=np.float64) for value in session.run(names, {feed_name: group})], real
            images += real
    if images == 0:
        raise ValueError("Calibration data held no image.")


def _smoothable_norms(model: Any, plan: Int8Plan) -> dict[str, tuple[str, ...]]:
    """Map each backbone ``LayerNormalization`` whose output only planned-style projections read to those projections.

    A norm qualifies when it is in the backbone encoder, normalizes the last axis, has a constant scale and bias with
    one value per channel, and every reader of its output is a ``MatMul`` that multiplies it by a constant 2-D weight,
    at least one of them quantized by *plan*. Scaling such a norm's channels down and the matching weight rows up leaves
    every output unchanged (:func:`_smooth_norms`). A norm read by any other op, such as the final backbone norm, or
    whose output is a graph output or is read inside a subgraph, is left out.

    Args:
        model: The FP32 ``ModelProto`` *plan* was made from.
        plan: Its INT8 plan.

    Returns:
        Norm name to the names of the projections reading it, in graph order.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> block = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/"
        >>> nodes = [helper.make_node("LayerNormalization", ["x", "g", "b"], ["n"], f"{block}norm2/LayerNormalization"),
        ...          helper.make_node("MatMul", ["n", "w"], ["y"], f"{block}mlp/fc1/MatMul")]
        >>> weights = [numpy_helper.from_array(np.ones(shape, np.float32), name)
        ...            for name, shape in (("g", (2,)), ("b", (2,)), ("w", (2, 3)))]
        >>> model = helper.make_model(helper.make_graph(nodes, "g", [], [], weights))
        >>> plan = Int8Plan(weighted=(f"{block}mlp/fc1/MatMul",), activations=())
        >>> [name.rsplit("/", 2)[-2] for name in _smoothable_norms(model, plan)]
        ['norm2']
    """
    graph = model.graph
    constants = _constant_weights(graph)
    consumers = _consumers(graph)
    names = Counter(node.name for node in graph.node)
    planned = set(plan.weighted)
    # Smoothing changes the norm's output, so only the projections that undo it may see that output.
    exposed = {value.name for value in graph.output}
    exposed |= {name for nested in list(_iter_graphs(graph))[1:] for node in nested.node for name in node.input}
    norms = {}
    for node in graph.node:
        if node.op_type != "LayerNormalization" or not node.name.startswith(_BACKBONE_ENCODER) or names[node.name] != 1:
            continue
        if node.output[0] in exposed:
            continue
        axis = next((attribute.i for attribute in node.attribute if attribute.name == "axis"), -1)
        readers = consumers.get(node.output[0], [])
        if axis != -1 or len(node.input) != 3 or any(n not in constants for n in node.input[1:]):
            continue
        channels = [int(d) for d in constants[node.input[1]].dims]
        if len(channels) != 1 or [int(d) for d in constants[node.input[2]].dims] != channels:
            continue
        output = str(node.output[0])
        if all(
            _projects(reader, output, channels[0], constants) and names[reader.name] == 1 for reader in readers
        ) and any(reader.name in planned for reader in readers):
            norms[str(node.name)] = tuple(str(reader.name) for reader in readers)
    return norms


def _projects(reader: Any, tensor: str, channels: int, constants: Mapping[str, Any]) -> bool:
    """Return whether *reader* is a ``MatMul`` of *tensor* by a constant 2-D weight with one row per channel of it.

    Such a reader undoes a per-channel scale of *tensor* when its weight rows are scaled back (:func:`_smooth_norms`).

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> constants = {"w": numpy_helper.from_array(np.zeros((4, 2), np.float32), "w")}
        >>> _projects(helper.make_node("MatMul", ["n", "w"], ["y"]), "n", 4, constants)
        True
        >>> _projects(helper.make_node("MatMul", ["w", "n"], ["y"]), "n", 4, constants)
        False
        >>> _projects(helper.make_node("MatMul", ["m", "w"], ["y"]), "n", 4, constants)
        False
        >>> _projects(helper.make_node("MatMul", ["n", "w"], ["y"]), "n", 3, constants)
        False
    """
    if reader.op_type != "MatMul" or len(reader.input) != 2 or reader.input[0] != tensor:
        return False
    weight = constants.get(reader.input[1])
    return weight is not None and len(weight.dims) == 2 and int(weight.dims[0]) == channels


def _unfit_smoothed_term(labels: list[str], terms: list[NDArray[np.float64]]) -> str | None:
    """Say which smoothed constant of a norm does not fit FP16 and why, or return ``None`` when every one fits.

    A term is left out of FP16 for one of two causes, worded apart: its largest magnitude reaches
    :data:`_MAX_FP16`, or it is not finite (a NaN compares False, so it counts as not fitting too).

    Args:
        labels: One description per term, naming the constant it replaces.
        terms: The smoothed constants, in the order of *labels*.

    Returns:
        The first term that does not fit and its cause, or ``None``.

    Examples:
        >>> _unfit_smoothed_term(["bias 'b'"], [np.array([1.0, -2.0])]) is None
        True
        >>> _unfit_smoothed_term(["bias 'b'"], [np.array([1e6])])
        "its smoothed bias 'b' reaches 1e+06, past the FP16 maximum 6.55e+04"
        >>> _unfit_smoothed_term(["bias 'b'"], [np.array([np.nan])])
        "its smoothed bias 'b' is not finite"
    """
    for label, term in zip(labels, terms, strict=True):
        peak = float(np.abs(term).max())
        if peak < _MAX_FP16:
            continue
        if np.isfinite(peak):
            return f"its smoothed {label} reaches {peak:.4g}, past the FP16 maximum {_MAX_FP16:.4g}"
        return f"its smoothed {label} is not finite"
    return None


def measure_smoothing_scales(
    onnx_path: str, model: Any, plan: Int8Plan, batches: Iterable[NDArray[np.float32]]
) -> dict[str, NDArray[np.float64]]:
    """Measure SmoothQuant's per-channel scale for each norm of :func:`_smoothable_norms`.

    SmoothQuant (Xiao et al., 2023) divides a channel of the norm's output by ``s`` and multiplies the matching rows of
    the projections that read it by ``s``, which moves range from the activation, quantized per tensor, into the
    weights, quantized per output channel. With ``a`` the channel's absolute maximum over the calibration images run
    through the FP32 graph and ``w`` the largest weight of its rows, ``s = a ** alpha / w ** (1 - alpha)`` with alpha
    :data:`_SMOOTHING_ALPHA`, divided by the geometric mean of the norm's scales so that the norm keeps its overall
    size. A channel whose ``a`` or ``w`` is 0 keeps ``s = 1``, and a norm whose scaled scale, bias or weights would not
    fit FP16 is left out, logged at debug level with the constant that does not fit and why.

    Args:
        onnx_path: The FP32 ``.onnx`` file *model* was loaded from; the probe graph is written to a temporary file
            beside it.
        model: The loaded FP32 ``ModelProto``, lifted to the opset FP16 scales need and rewritten as it was for *plan*.
        plan: The INT8 plan of *model*.
        batches: Preprocessed ``(1, C, H, W)`` images.

    Returns:
        Norm name to its channel scales; empty when no norm qualifies.

    Raises:
        ValueError: If *batches* is empty or holds a NaN or infinite value, or a norm's output is not finite on it (a
            NaN computed inside the graph included).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    norms = _smoothable_norms(model, plan)
    if not norms:
        return {}
    constants = _constant_weights(model.graph)
    by_name = {
        node.name: node
        for node in model.graph.node
        if node.name in norms or any(node.name in r for r in norms.values())
    }
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    taken = _tensor_names(probe.graph)
    rows_axis, infinity = (_unique_name(name, taken) for name in ("channel_maxima_rows", "channel_maxima_infinity"))
    probe.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0], np.int64), rows_axis),
            numpy_helper.from_array(np.array(np.inf, np.float32), infinity),
        ]
    )
    names = []
    for norm in norms:
        tensor, channels = by_name[norm].output[0], int(constants[by_name[norm].input[1]].dims[0])
        shape, rows = (_unique_name(f"{tensor}__{suffix}", taken) for suffix in ("rows_shape", "rows"))
        probe.graph.initializer.append(numpy_helper.from_array(np.array([-1, channels], np.int64), shape))
        probe.graph.node.append(helper.make_node("Reshape", [tensor, shape], [rows]))
        guarded = _absolute_with_nan_as_infinity(probe.graph, rows, tensor, infinity, taken)
        maximum = _unique_name(f"{tensor}__channel_max", taken)
        probe.graph.node.append(helper.make_node("ReduceMax", [guarded, rows_axis], [maximum], keepdims=0))
        probe.graph.output.append(helper.make_tensor_value_info(maximum, TensorProto.FLOAT, [channels]))
        names.append(maximum)
    maxima: list[NDArray[np.float64]] = []
    for values, _ in _run_probe(onnx_path, probe, names, batches):
        # A copy padding the last group repeats an image, which cannot change a maximum.
        maxima = values if not maxima else [np.maximum(total, value) for total, value in zip(maxima, values)]
    unusable = [norm for norm, activation in zip(norms, maxima) if not np.isfinite(activation).all()]
    if unusable:
        # Refused here: a norm left out of the smoothing for it would only be refused by the calibration pass after.
        raise ValueError(
            f"Calibration gave {len(unusable)} backbone norm output(s) a range that is not finite (the graph "
            f"computed a NaN or infinite activation), e.g. {unusable[0]!r}. Check that calibration_data is normalized "
            "as predict() normalizes and that the model's outputs on it are finite."
        )
    scales = {}
    for (norm, readers), activation in zip(norms.items(), maxima):
        weights = [numpy_helper.to_array(constants[by_name[reader].input[1]]).astype(np.float64) for reader in readers]
        weight = np.max([np.abs(w).max(axis=1) for w in weights], axis=0)
        alive = (activation > 0) & (weight > 0)
        scale = np.ones_like(activation)
        scale[alive] = activation[alive] ** _SMOOTHING_ALPHA / weight[alive] ** (1 - _SMOOTHING_ALPHA)
        scale /= np.exp(np.log(scale).mean())
        node = by_name[norm]
        terms = [numpy_helper.to_array(constants[node.input[i]]).astype(np.float64) / scale for i in (1, 2)]
        terms += [w * scale[:, None] for w in weights]
        labels = [f"scale {node.input[1]!r}", f"bias {node.input[2]!r}"]
        labels += [f"weight {by_name[reader].input[1]!r} of {reader!r}" for reader in readers]
        cause = _unfit_smoothed_term(labels, terms)
        if cause is None:
            scales[norm] = scale
        else:
            logger.debug(f"Left backbone LayerNorm {norm!r} unsmoothed for INT8: {cause}.")
    return scales


def _smooth_norms(model: Any, scales: Mapping[str, NDArray[np.float64]]) -> None:
    """Divide each norm's output channels by its scales and multiply the rows of the projections reading it, in place.

    The norm's scale and bias and each projection's weight become new initializers of their own dtype, so a constant
    that other nodes read as well (the exporter stores identical constants once) keeps its value for them, and one that
    nothing reads any more is removed. The graph's outputs are unchanged but for rounding.

    Args:
        model: An FP32 or FP16 ``ModelProto``, with the norm and projection names of the graph *scales* was measured on.
        scales: Norm name to its channel scales (see :func:`measure_smoothing_scales`).

    Raises:
        ValueError: If *model* has no single norm of one of the names, a norm has no constant scale and bias, or a
            reader of a norm is not a projection multiplying it by a constant 2-D weight with a row per channel.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> nodes = [helper.make_node("LayerNormalization", ["x", "g", "b"], ["n"], "norm"),
        ...          helper.make_node("MatMul", ["n", "w"], ["y"], "proj")]
        >>> weights = [numpy_helper.from_array(np.full(shape, 2.0, np.float32), name)
        ...            for name, shape in (("g", (2,)), ("b", (2,)), ("w", (2, 1)))]
        >>> model = helper.make_model(helper.make_graph(nodes, "g", [], [], weights))
        >>> _smooth_norms(model, {"norm": np.array([2.0, 0.5])})
        >>> {init.name: numpy_helper.to_array(init).ravel().tolist() for init in model.graph.initializer}
        {'g_smoothed': [1.0, 4.0], 'b_smoothed': [1.0, 4.0], 'w_smoothed': [4.0, 1.0]}
    """
    from onnx import numpy_helper

    graph = model.graph
    constants = _constant_weights(graph)
    counts = Counter(node.name for node in graph.node)
    consumers = _consumers(graph)
    by_name = {node.name: node for node in graph.node if node.name in scales and counts[node.name] == 1}
    taken = _tensor_names(graph)

    replaced: list[str] = []

    def replace(node: Any, index: int, values: NDArray[np.float64], dtype: Any) -> None:
        """Point input *index* of *node* at a new initializer holding *values* as *dtype*."""
        name = _unique_name(f"{node.input[index]}_smoothed", taken)
        graph.initializer.append(numpy_helper.from_array(values.astype(dtype), name))
        replaced.append(str(node.input[index]))
        node.input[index] = name

    for name, scale in scales.items():
        norm = by_name.get(name)
        output = str(norm.output[0]) if norm is not None else ""
        readers = consumers.get(output, []) if norm is not None else []
        projections = all(_projects(reader, output, len(scale), constants) for reader in readers)
        if (
            norm is None
            or norm.op_type != "LayerNormalization"
            or len(norm.input) != 3
            or not set(norm.input[1:]) <= set(constants)
            or not readers
            or not projections
        ):
            raise ValueError(
                f"The graph has no LayerNormalization {name!r} with constant scale, bias and projection weights to "
                "smooth for INT8, which the FP16 cast cannot cause."
            )
        for index in (1, 2):
            values = numpy_helper.to_array(constants[norm.input[index]])
            replace(norm, index, values.astype(np.float64) / scale, values.dtype)
        for reader in readers:
            values = numpy_helper.to_array(constants[reader.input[1]])
            replace(reader, 1, values.astype(np.float64) * scale[:, None], values.dtype)
    _drop_unread_initializers(graph, replaced)


def _mlp_biases(model: Any, plan: Int8Plan) -> dict[str, str]:
    """Map each planned layer of a backbone MLP to the ``Add`` that adds its bias, for the layers whose bias is found.

    A backbone MLP is a planned ``MatMul`` of the backbone encoder whose output reaches planned layers through a GELU,
    together with those layers. A layer's bias ``Add`` is the only reader of its output and adds a constant (an
    initializer, or one reached through an ``Identity``) with one value per output channel; a layer without one is left
    out.

    Args:
        model: The FP32 ``ModelProto`` *plan* was made from.
        plan: Its INT8 plan.

    Returns:
        Layer name to the name of its bias ``Add``, in graph order.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> f = lambda *shape: np.zeros(shape, np.float32)
        >>> mlp = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/mlp/"
        >>> nodes = [helper.make_node("MatMul", ["x", "w1"], ["h"], f"{mlp}fc1/MatMul"),
        ...          helper.make_node("Add", ["h", "b1"], ["hb"], f"{mlp}fc1/Add"),
        ...          helper.make_node("Gelu", ["hb"], ["g"], f"{mlp}act/Gelu", domain=""),
        ...          helper.make_node("MatMul", ["g", "w2"], ["y0"], f"{mlp}fc2/MatMul"),
        ...          helper.make_node("Add", ["y0", "b2"], ["y"], f"{mlp}fc2/Add")]
        >>> weights = [numpy_helper.from_array(v, n) for n, v in
        ...            (("w1", f(2, 4)), ("b1", f(4)), ("w2", f(4, 2)), ("b2", f(2)))]
        >>> graph = helper.make_graph(nodes, "g", [], [], weights)
        >>> plan = Int8Plan(weighted=(f"{mlp}fc1/MatMul", f"{mlp}fc2/MatMul"), activations=())
        >>> [add.rsplit("/", 2)[-2] for add in _mlp_biases(helper.make_model(graph), plan).values()]
        ['fc1', 'fc2']
    """
    graph = model.graph
    constants = _constant_weights(graph)
    consumers = _consumers(graph)
    names = Counter(node.name for node in graph.node)
    planned = set(plan.weighted)
    by_name = {node.name: node for node in graph.node if node.name in planned}
    layers: set[str] = set()
    for name in plan.weighted:
        node = by_name[name]
        second = _feeds_gelu_into(node, consumers, constants)
        if node.op_type == "MatMul" and name.startswith(_BACKBONE_ENCODER) and second and second <= planned:
            layers |= {name, *second}
    biases = {}
    for name in (name for name in plan.weighted if name in layers):
        node = by_name[name]
        adds = consumers.get(node.output[0], [])
        if node.op_type != "MatMul" or len(adds) != 1 or adds[0].op_type != "Add" or names[adds[0].name] != 1:
            continue
        bias = _bias_input(adds[0], node.output[0], constants)
        if bias is not None and list(constants[adds[0].input[bias]].dims) == [int(constants[node.input[1]].dims[1])]:
            biases[name] = str(adds[0].name)
    return biases


def _bias_input(add: Any, product: str, constants: Mapping[str, Any]) -> int | None:
    """Return the index of the constant input that the bias ``Add`` *add* adds to *product*, or ``None``.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> constants = {"b": numpy_helper.from_array(np.zeros(2, np.float32), "b")}
        >>> _bias_input(helper.make_node("Add", ["b", "y0"], ["y"], "add"), "y0", constants)
        0
        >>> _bias_input(helper.make_node("Add", ["y0", "z"], ["y"], "add"), "y0", constants) is None
        True
    """
    if len(add.input) != 2:
        return None
    index = 1 - list(add.input).index(product)
    return index if add.input[index] in constants else None


def _channel_means_graph(
    model: Any, tensors: Mapping[str, tuple[int, float]], batch: int
) -> tuple[Any, dict[str, tuple[str, str]]]:
    """Return a copy of *model* that also outputs, per image and channel, the mean of each tensor and its INT8 rounding.

    Each tensor of *tensors* is read as *batch* images, each a run of rows of its last axis, its ``channels``: the
    tensor's leading axis must list the images one after another, as RF-DETR's backbone does when it splits each image
    into windows. The mean of every channel is taken over the rows of each image, as is the mean of the tensor
    quantized to INT8 with the given step and dequantized again, so each output has one row per image. *model* must be
    at opset 18 or later, where ``ReduceMean`` takes its axes as an input.

    Args:
        model: The FP32 ``ModelProto``.
        tensors: Tensor name to ``(channels, step)``.
        batch: The number of images the graph runs at once.

    Returns:
        The probe, and per tensor the names of its two mean outputs (plain, then INT8).

    Examples:
        >>> from onnx import TensorProto, helper
        >>> graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
        ...     [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 2])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])])
        >>> model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        >>> probe, outputs = _channel_means_graph(model, {"y": (2, 0.5)}, batch=3)
        >>> outputs, [output.name for output in probe.graph.output]
        ({'y': ('y__mean', 'y__int8_mean')}, ['y', 'y__mean', 'y__int8_mean'])
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    taken = _tensor_names(probe.graph)
    rows_axis, zero = _unique_name("channel_means_rows", taken), _unique_name("channel_means_zero_point", taken)
    probe.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1], np.int64), rows_axis),
            numpy_helper.from_array(np.array(0, np.int8), zero),
        ]
    )
    outputs = {}
    for tensor, (channels, step) in tensors.items():
        shape, rows, scale, quantized, dequantized, mean, int8_mean = (
            _unique_name(f"{tensor}__{suffix}", taken)
            for suffix in ("rows_shape", "rows", "int8_step", "int8", "dequantized", "mean", "int8_mean")
        )
        probe.graph.initializer.extend(
            [
                numpy_helper.from_array(np.array([batch, -1, channels], np.int64), shape),
                numpy_helper.from_array(np.array(step, np.float32), scale),
            ]
        )
        probe.graph.node.extend(
            [
                helper.make_node("Reshape", [tensor, shape], [rows]),
                helper.make_node("ReduceMean", [rows, rows_axis], [mean], keepdims=0),
                helper.make_node("QuantizeLinear", [rows, scale, zero], [quantized]),
                helper.make_node("DequantizeLinear", [quantized, scale, zero], [dequantized]),
                helper.make_node("ReduceMean", [dequantized, rows_axis], [int8_mean], keepdims=0),
            ]
        )
        for name in (mean, int8_mean):
            probe.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, channels]))
        outputs[tensor] = (mean, int8_mean)
    return probe, outputs


def measure_mlp_bias_shifts(
    onnx_path: str,
    model: Any,
    plan: Int8Plan,
    ranges: Mapping[tuple[str, int], float],
    batches: Iterable[NDArray[np.float32]],
) -> dict[str, NDArray[np.float64]]:
    """Measure the shift INT8 adds to the mean output of each backbone MLP layer, per output channel.

    An INT8 layer ``y = x W + b`` computes ``Q(x) Q(W) + b``, where ``Q(x)`` rounds the input to its INT8 step and
    ``Q(W)`` is the per-channel INT8 rounding of the weight cast to FP16. That matches the weight :func:`insert_qdq`
    stores for the FP16 copy up to one FP16 rounding of a smoothed weight: the copy scales the weight it already cast to
    FP16 and rounds it again, ``fp16(fp16(W) s)``, while this casts the smoothed FP32 weight once, ``fp16(W s)``. Over
    the calibration images run through the FP32 graph its mean output moves by ``E[Q(x)] Q(W) - E[x] W``, the means
    taken per input channel over every token of every image, and subtracting that shift from ``b`` gives the INT8 layer
    the FP32 layer's mean output. Each layer is measured on FP32 inputs, independently of the INT8 error of the layers
    before it. A static batch larger than the images left over is padded with copies of the last image, as calibration
    pads it; the means are taken per image and the copies left out, so every image weighs the same at any batch. The
    layers are those of :func:`_mlp_biases`.

    Args:
        onnx_path: The FP32 ``.onnx`` file *model* was loaded from; the probe graph is written to a temporary file
            beside it.
        model: The loaded FP32 ``ModelProto``, lifted to the opset FP16 scales need and rewritten as it was for *plan*.
        plan: The INT8 plan of *model*.
        ranges: Calibrated absolute maximum per planned ``(node name, input index)``.
        batches: Preprocessed ``(1, C, H, W)`` images, the ones *ranges* were calibrated on.

    Returns:
        Name of each layer's bias ``Add`` to the shift to subtract from its bias; empty when no backbone MLP layer has a
        bias to correct.

    Raises:
        ValueError: If *batches* is empty.
    """
    from onnx import numpy_helper

    layers = _mlp_biases(model, plan)
    if not layers:
        return {}
    weights = _constant_weights(model.graph)
    by_name = {node.name: node for node in model.graph.node if node.name in layers}
    tensors = {
        by_name[name].input[0]: (
            int(weights[by_name[name].input[1]].dims[0]),
            float(_activation_step(ranges[(name, 0)])),
        )
        for name in layers
    }
    # The batch _graph_batches groups the images into: the static one, or a single image for a dynamic batch.
    batch = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value or 1
    probe, outputs = _channel_means_graph(model, tensors, batch)
    names = [name for pair in outputs.values() for name in pair]
    sums: list[NDArray[np.float64]] = []
    images = 0
    for measured, real in _run_probe(onnx_path, probe, names, batches):
        # One row per image, so the copies that pad the last group are left out.
        group = [value[:real].sum(axis=0) for value in measured]
        sums = group if not sums else [total + value for total, value in zip(sums, group)]
        images += real
    means = dict(zip(names, (total / images for total in sums)))
    shifts = {}
    for name, add in layers.items():
        node = by_name[name]
        weight = numpy_helper.to_array(weights[node.input[1]]).astype(np.float64)
        # The weight insert_qdq quantizes in the FP16 copy, up to one FP16 rounding of a smoothed weight: the copy holds
        # fp16(fp16(W) * s), this is fp16(W * s).
        values, scales = _quantized_weight(weight.astype(np.float16), axis=1)
        mean, int8_mean = (means[output] for output in outputs[node.input[0]])
        shifts[add] = int8_mean @ (values.astype(np.float64) * scales.astype(np.float64)) - mean @ weight
    return shifts


def _correct_biases(model: Any, shifts: Mapping[str, NDArray[np.float64]]) -> None:
    """Give each bias ``Add`` named in *shifts* its own copy of its bias minus the shift, in place.

    The ``Add`` reads a new initializer of the bias's dtype, so a bias that other nodes read as well (the exporter
    stores identical biases once) or that is a graph output keeps its value for them, and one that nothing reads any
    more is removed.

    Args:
        model: The FP16 ``ModelProto`` the engine is built from.
        shifts: Name of a bias ``Add`` to the shift to subtract (see :func:`measure_mlp_bias_shifts`).

    Raises:
        ValueError: If *model* has no single node of one of the names, or that node adds no constant.

    Examples:
        >>> from onnx import helper, numpy_helper
        >>> bias = numpy_helper.from_array(np.array([1.0, 2.0], np.float16), "b")
        >>> add = helper.make_node("Add", ["y0", "b"], ["y"], "add")
        >>> model = helper.make_model(helper.make_graph([add], "g", [], [], [bias]))
        >>> _correct_biases(model, {"add": np.array([0.5, -1.0])})
        >>> list(model.graph.node[0].input), numpy_helper.to_array(model.graph.initializer[-1]).tolist()
        (['y0', 'b_int8_corrected'], [0.5, 3.0])
    """
    from onnx import numpy_helper

    graph = model.graph
    constants = _constant_weights(graph)
    counts = Counter(node.name for node in graph.node)
    by_name = {node.name: node for node in graph.node if node.name in shifts and counts[node.name] == 1}
    taken = _tensor_names(graph)
    replaced: list[str] = []
    for name, shift in shifts.items():
        node = by_name.get(name)
        index = next((i for i, tensor in enumerate(node.input) if tensor in constants), None) if node else None
        if node is None or node.op_type != "Add" or index is None:
            raise ValueError(
                f"The FP16 copy has no bias Add {name!r} for the INT8 bias correction, which the FP16 cast cannot "
                "cause."
            )
        bias = numpy_helper.to_array(constants[node.input[index]])
        corrected = _unique_name(f"{node.input[index]}_int8_corrected", taken)
        graph.initializer.append(
            numpy_helper.from_array((bias.astype(np.float64) - shift).astype(bias.dtype), corrected)
        )
        replaced.append(str(node.input[index]))
        node.input[index] = corrected
    _drop_unread_initializers(graph, replaced)


def _activation_step(value: float) -> np.float16:
    """Return the FP16 INT8 step of an activation whose absolute maximum is *value*, at least :data:`_MIN_FP16_SCALE`.

    Examples:
        >>> float(_activation_step(12.7)), float(_activation_step(0.0)) > 0
        (0.0999755859375, True)
    """
    return np.float16(max(float(value) / _INT8_MAX, _MIN_FP16_SCALE))


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
    other op whose schema changed in that range is refused rather than lifted unchecked, except the ops of
    :data:`_TYPE_ONLY_CHANGES` and, from the opset in :data:`_UNCHANGED_SINCE`, ``Resize`` and ``ScatterND``.

    Args:
        model: ``ModelProto`` to lift; left untouched when its opset is already high enough.

    Raises:
        ValueError: If the graph holds an op whose meaning the lift does not know how to keep, or one ONNX does not
            define.

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
            elif node.op_type not in _TYPE_ONLY_CHANGES and current < _UNCHANGED_SINCE.get(
                node.op_type, _FP16_QDQ_OPSET
            ):
                try:
                    changed = (
                        defs.get_schema(node.op_type, current).since_version
                        != defs.get_schema(node.op_type, _FP16_QDQ_OPSET).since_version
                    )
                except defs.SchemaError as error:  # not a ValueError
                    raise ValueError(
                        f"Cannot quantize this graph for TensorRT: node {node.name!r} ({node.op_type}) is not an ONNX "
                        f"op at opset {current} or {_FP16_QDQ_OPSET}."
                    ) from error
                if changed:
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
            new_initializers.append(
                numpy_helper.from_array(np.array(_activation_step(ranges[(node_name, index)])), scale)
            )
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
    """Refuse a graph INT8 cannot start from: quantized, not float32, not a static image, or not a supported model.

    Each would otherwise fail late or point the wrong way: a quantized graph plans nothing to quantize, and an FP16
    graph is calibrated in full before the FP16 cast refuses it with advice (``fp16=False``) that an INT8 request
    cannot follow.

    Args:
        model: The loaded source graph.
        onnx_path: The caller's file, named in the error.

    Raises:
        ValueError: If the graph holds Q/DQ nodes, FP16 inputs or weights, or a dynamic batch axis, lists an
            initializer as a graph input too, or its first input is not rank 4 with fixed, positive channels, height
            and width.
        NotImplementedError: If its outputs are not a detector's ``dets`` and ``labels``, optionally followed by
            ``masks`` or ``keypoints`` (so a backbone-only export or any other set is refused), the type
            :meth:`TensorRTExporter._convert` raises for the same set.

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
    # A weight also listed as a graph input can be replaced at run time; quantizing bakes it in as an INT8 constant.
    overridable = sorted({value.name for value in model.graph.input} & {init.name for init in model.graph.initializer})
    if overridable:
        raise ValueError(
            f"'{onnx_path}' lists {len(overridable)} initializer(s) as graph inputs too, e.g. {overridable[0]!r}, so "
            "they can be overridden at run time, while quantization='int8' bakes weights into the graph. Re-export "
            "the ONNX graph with keep_initializers_as_inputs=False."
        )
    outputs = tuple(output.name for output in model.graph.output)
    if outputs not in _INT8_OUTPUT_NAMES:
        # The type _convert raises for the same model before its ONNX export, so build_engine callers catch one type.
        raise NotImplementedError(
            f"'{onnx_path}' outputs {list(outputs)}, and quantization='int8' is measured for detection, segmentation "
            f"and keypoint models only, whose outputs are {[list(names) for names in _INT8_OUTPUT_NAMES]}. Build this "
            "graph with quantization=None."
        )
    # Calibration preprocesses images to the input's own channels, height and width, so each must be a known size.
    image = model.graph.input[0]
    dims = image.type.tensor_type.shape.dim
    if len(dims) != 4 or any(dim.WhichOneof("value") != "dim_value" or dim.dim_value <= 0 for dim in dims[1:]):
        shape = [dim.dim_value if dim.WhichOneof("value") == "dim_value" else dim.dim_param or "?" for dim in dims]
        raise ValueError(
            f"'{onnx_path}' input {image.name!r} has shape {shape}, and quantization='int8' calibrates a "
            "(batch, channels, height, width) image input whose channels, height and width are fixed and positive. "
            "Build from the .onnx that RFDETR.export writes."
        )


@contextlib.contextmanager
def int8_source_graph(
    onnx_path: str,
    *,
    calibration_data: str | Path | NDArray[Any],
    max_images: int,
    dynamic_batch: bool | None,
    fp16_graph: Callable[..., contextlib.AbstractContextManager[str]],
) -> Iterator[str]:
    """Provide an INT8-quantized FP16 copy of *onnx_path* to build from, deleting it when the block exits.

    A detection model's calibration images are read three times: for SmoothQuant's per-channel scales of the backbone
    norms (:func:`measure_smoothing_scales`), applied to both graphs before the ranges are calibrated, for the
    activation ranges, and for the mean shift INT8 adds to the backbone MLP layers, which is subtracted from their
    biases (:func:`measure_mlp_bias_shifts`). The other models' images are read once, for the ranges.

    Args:
        onnx_path: The FP32 ``.onnx`` export.
        calibration_data: Directory of images, ``.npy`` path or array, as accepted by
            :func:`~rfdetr.export._runtime.calibration.calibration_batches`.
        max_images: Maximum images read from a *calibration_data* directory.
        dynamic_batch: The batch request the engine is built for, forwarded to the FP16 cast.
        fp16_graph: The FP16 cast, called as ``fp16_graph(onnx_path, dynamic_batch=dynamic_batch)`` once calibration
            is done (the exporter passes :func:`~rfdetr.export._tensorrt.exporter.fp16_source_graph`). Passed in
            rather than imported so this module never imports the exporter, which imports it.

    Yields:
        Path to the quantized graph, valid only inside the ``with`` block.

    Raises:
        ValueError: If the graph is already quantized, is not float32, has a dynamic batch axis, an initializer that
            is also a graph input, or an input that is not a static rank-4 image, cannot be lifted to opset 19, has no
            backbone or decoder layer to quantize, or holds attention INT8 cannot be placed around or a NaN or
            infinite weight it would quantize (checked before calibration); or if the calibration data is unusable,
            holds a NaN or infinite value, or gives a range that is not finite (a NaN computed inside the graph
            included) or does not fit an FP16 scale; or if a calibration
            image cannot be identified as an image (the message names the file); or if *fp16_graph* returns a copy
            that lacks a node the plan quantizes, a norm the smoothing scales (or reads it with other nodes than
            the projections it scales), or a bias ``Add`` the correction changes, which the real FP16 cast cannot
            do. An image Pillow identifies but cannot decode (a truncated file) raises Pillow's own ``OSError``,
            which does not name the file.
        NotImplementedError: If the graph's outputs are not a detector's ``dets`` and ``labels``, optionally followed
            by ``masks`` or ``keypoints`` (so a backbone-only export or any other set is refused), the type
            :meth:`TensorRTExporter._convert` raises for the same set.
    """
    import onnx
    from onnx import numpy_helper

    from rfdetr.export._runtime.calibration import calibration_batches

    source = onnx.load(onnx_path)
    _refuse_unquantizable_source(source, onnx_path)
    # Lifting the FP32 source too keeps its meaning and makes an unliftable graph fail here, before calibration.
    _lift_opset(source)
    # Before the plan, so that a patch embedding INT8 can quantize is planned and calibrated as the matrix multiply it
    # becomes. One whose flattened patch is not a multiple of 32 values stays a convolution here, so it is not
    # quantized.
    _patch_embedding_as_matmul(source, multiple_of=_INT8_CONV_CHANNEL_MULTIPLE)
    plan = plan_int8(source)
    # A NaN or infinite weight gets a NaN or infinite FP16 scale that only TensorRT would notice, after calibration.
    weights = _constant_weights(source.graph)
    planned = set(plan.weighted)
    weight_names = sorted({node.input[1] for node in source.graph.node if node.name in planned})
    non_finite = [name for name in weight_names if not np.isfinite(numpy_helper.to_array(weights[name])).all()]
    if non_finite:
        raise ValueError(
            f"'{onnx_path}' holds {len(non_finite)} weight(s) with NaN or infinite values, e.g. {non_finite[0]!r}, "
            "which INT8 cannot quantize. Export from a checkpoint whose weights are all finite."
        )
    logger.info(
        f"INT8 plan: {len(plan.weighted)} weighted layer(s), {len(plan.activations)} quantized activation input(s)"
    )
    dims = source.graph.input[0].type.tensor_type.shape.dim
    channels, height, width = (d.dim_value for d in dims[1:4])
    keys = list(plan.activations)
    tensors = sorted({_input_tensor(source.graph, name, index) for name, index in keys})

    def images(*, quiet: bool = False) -> contextlib.closing[Generator[NDArray[np.float32], None, None]]:
        """A fresh pass over the calibration images, closed even when the pass raises (a memory-mapped ``.npy``).

        *quiet* skips the per-sample range warnings, which the first pass over the images has already logged.
        """
        return contextlib.closing(
            calibration_batches(
                calibration_data, height=height, width=width, channels=channels, max_images=max_images, quiet=quiet
            )
        )

    detection = tuple(output.name for output in source.graph.output) == _DETECTION_OUTPUT_NAMES
    scales: dict[str, NDArray[np.float64]] = {}
    if detection:
        # Before calibration, so that the ranges are those of the smoothed norms.
        with images() as batches:
            scales = measure_smoothing_scales(onnx_path, source, plan, batches)
        _smooth_norms(source, scales)
        logger.info(f"Smoothed {len(scales)} backbone LayerNorm output(s) for INT8")
    with images(quiet=detection) as batches:  # the first pass unless the smoothing pass came before
        by_tensor = calibrate_ranges(onnx_path, source, tensors, batches)
    unusable = sorted(t for t, r in by_tensor.items() if not np.isfinite(r) or r / _INT8_MAX > _MAX_FP16)
    if unusable:
        raise ValueError(
            f"Calibration gave {len(unusable)} tensor(s) a range that is not finite (the graph computed a NaN or "
            f"infinite activation) or does not fit an FP16 scale, e.g. {unusable[0]!r} = {by_tensor[unusable[0]]}. "
            "Check that calibration_data is normalized as predict() normalizes and that the model's outputs on it "
            "are finite."
        )
    ranges = {key: by_tensor[_input_tensor(source.graph, *key)] for key in keys}
    shifts: dict[str, NDArray[np.float64]] = {}
    if detection:
        # Measured at the steps just calibrated, on the smoothed graph.
        with images(quiet=True) as batches:
            shifts = measure_mlp_bias_shifts(onnx_path, source, plan, ranges, batches)
        logger.info(f"Corrected the bias of {len(shifts)} backbone MLP layer(s) for the mean shift of INT8")
    del source

    with fp16_graph(onnx_path, dynamic_batch=dynamic_batch) as fp16_path:
        model = onnx.load(fp16_path)
    _lift_opset(model)  # as the FP32 source was, before the rewrite adds its nodes
    # Every patch embedding becomes a matrix multiply here: the planned ones need the node names the plan holds, and
    # none whose MatMul name is free is left as a 3-channel convolution for TensorRT to pad. The planned ones go first,
    # as in the FP32 source, so that a patch the plan skipped cannot take a planned name when two patches share a
    # parent; the one that loses the name stays a convolution. A patch the plan skipped stays FP16.
    _patch_embedding_as_matmul(model, multiple_of=_INT8_CONV_CHANNEL_MULTIPLE)
    _patch_embedding_as_matmul(model)
    missing = sorted({name for name, _ in plan.activations} - {node.name for node in model.graph.node})
    if missing:
        raise ValueError(
            f"The FP16 copy of '{onnx_path}' lacks {len(missing)} node(s) the INT8 plan quantizes, e.g. {missing[0]!r} "
            "(a planned patch embedding is rewritten in both graphs), so the plan cannot be applied to it."
        )
    _smooth_norms(model, scales)
    _correct_biases(model, shifts)
    insert_qdq(model, plan, ranges)
    stem = os.path.basename(os.path.splitext(onnx_path)[0])
    handle, int8_path = tempfile.mkstemp(prefix=f"{stem}.int8-", suffix=".onnx", dir=os.path.dirname(onnx_path) or ".")
    os.close(handle)
    try:
        onnx.save(model, int8_path)
        del model
        yield int8_path
    finally:
        try:
            Path(int8_path).unlink(missing_ok=True)
        except OSError as error:
            logger.warning(f"Could not delete the INT8 intermediate graph {int8_path!r}: {error}")
