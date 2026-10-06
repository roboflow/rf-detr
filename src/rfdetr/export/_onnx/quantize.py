# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Static INT8 post-training quantization for an exported ONNX graph.

``quantization="int8"`` on ``format="onnx"`` rewrites the traced graph into a QDQ model whose matrix multiplies run in
8-bit. What is *not* quantized is the point of this module.

ONNX Runtime's default op coverage quantizes everything it recognises, which on RF-DETR means ``LayerNormalization``,
``Mul``, ``Div``, ``Reshape`` and every shape op as well as the matrix multiplies. Measured on a 500-image COCO
val2017 subset, that costs 9.65 mAP on Nano and 12.14 on Small -- the transformer does not tolerate 8-bit activations
on those paths. Restricting quantization to ``MatMul``/``Gemm``, excluding the attention-score multiplies and the
detection heads, and leaving those ops' *outputs* in float brought Nano back to within 2.65 mAP; that measurement used
an earlier head rule (every node within three hops of an output), and this module now excludes the last
``MatMul``/``Gemm`` on each path back from an output instead -- see :func:`nodes_to_exclude`. It is also the graph
shape a published INT8 RF-DETR Base uses to hold roughly 53 mAP, so the remaining gap is a calibration and
boundary-placement question rather than an architectural limit.

Three decisions follow from that and are deliberately not configurable:

* **Only ``MatMul`` and ``Gemm``.** Everything else stays float. Quantizing more is both less accurate and *slower*
  here, because the Q/DQ conversions around non-matmul ops cost more than the 8-bit kernels save.
* **Outputs stay float** (``OpTypesToExcludeOutputQuantization``). ORT otherwise quantizes each matmul's output too,
  so the ops reading it see an 8-bit-rounded tensor.
* **Per-tensor weights, not per-channel.** With ``per_channel=True`` the quantized RF-DETR graph failed at ONNX
  Runtime session creation on a node that has not yet been identified; per-tensor weights load and run. Per-channel
  stays off for every node until that one is pinned down and can be held back on its own.

Calibration data is required and must be representative: unlike TFLite's dynamic-range mode, static quantization
derives activation ranges from the data it is shown, so random or out-of-domain images produce a model that loads and
runs and is quietly wrong.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import calibration_batches, warn_if_too_few_samples
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Quantization modes ``format="onnx"`` accepts. ``None`` and ``"fp32"`` both mean "write the traced graph unchanged".
VALID_QUANTIZATIONS: frozenset[str | None] = frozenset({None, "fp32", "int8"})

#: The only ops quantized. See the module docstring for why the list is this short.
_QUANTIZED_OP_TYPES: tuple[str, ...] = ("MatMul", "Gemm")

#: Ops that end the walk back from a graph output. The walk excludes the ``MatMul``/``Gemm`` it stops at -- the last
#: linear layer of a head -- and nothing behind it; a ``Conv`` means it has left the heads for the backbone.
_HEAD_STOP_OP_TYPES: frozenset[str] = frozenset({*_QUANTIZED_OP_TYPES, "Conv"})

#: Elementwise ops allowed between an attention-score multiply and its ``Softmax`` -- the ``1/sqrt(d)`` scaling and a
#: constant bias. The walk back from a ``Softmax`` passes through one only when its other operand is a constant.
_SCORE_SCALING_OP_TYPES: frozenset[str] = frozenset({"Div", "Mul", "Add", "Sub"})

#: Upper bound on the hops any backward walk over the graph takes, so a malformed or cyclic graph cannot stall export.
_MAX_WALK_HOPS: int = 10


def _constant_names(graph: Any) -> set[str]:
    """Return the names of tensors whose value is fixed at export time.

    Args:
        graph: An ``onnx.GraphProto``.

    Returns:
        Initializer names plus the outputs of ``Constant`` nodes.
    """
    names = {initializer.name for initializer in graph.initializer}
    names.update(output for node in graph.node if node.op_type == "Constant" for output in node.output)
    return names


def _attention_score_matmuls(graph: Any, producer: dict[str, Any], constants: set[str]) -> set[str]:
    """Return the activation-by-activation multiplies whose result reaches a ``Softmax``.

    From each ``Softmax`` input the walk goes back through scaling ops (:data:`_SCORE_SCALING_OP_TYPES`) whose other
    operand is a constant, and stops at the first ``MatMul``/``Gemm``. That node is kept only when neither of its two
    data inputs is a constant -- the ``Q @ K^T`` shape. A weight multiply feeding a ``Softmax`` (a classifier, the
    deformable-attention weights) is an ordinary linear layer and stays quantizable.

    Args:
        graph: An ``onnx.GraphProto``.
        producer: Tensor name -> node producing it.
        constants: Names of constant tensors, from :func:`_constant_names`.

    Returns:
        Names of the attention-score multiplies.
    """
    excluded: set[str] = set()
    for node in graph.node:
        if node.op_type != "Softmax" or not node.input:
            continue
        tensor = node.input[0]
        for _ in range(_MAX_WALK_HOPS):
            source = producer.get(tensor)
            if source is None:
                break
            if source.op_type in _QUANTIZED_OP_TYPES:
                if source.name and not constants.intersection(source.input[:2]):
                    excluded.add(source.name)
                break
            dynamic = [name for name in source.input if name not in constants]
            if source.op_type not in _SCORE_SCALING_OP_TYPES or len(dynamic) != 1:
                break
            tensor = dynamic[0]
    return excluded


def _head_matmuls(graph: Any, producer: dict[str, Any]) -> set[str]:
    """Return the last ``MatMul``/``Gemm`` on each path back from a graph output.

    The walk goes back from every graph output through any op not in :data:`_HEAD_STOP_OP_TYPES` (activations, box
    refinement arithmetic, reshapes, concatenations) for at most :data:`_MAX_WALK_HOPS` hops. Each path ends at the
    first stop op; only a ``MatMul``/``Gemm`` terminal is excluded, so the layers behind it stay quantizable however
    many elementwise ops sit between the head and the output.

    Args:
        graph: An ``onnx.GraphProto``.
        producer: Tensor name -> node producing it.

    Returns:
        Names of the head multiplies.
    """
    excluded: set[str] = set()
    visited: set[str] = set()
    frontier = [output.name for output in graph.output]
    for _ in range(_MAX_WALK_HOPS):
        next_frontier: list[str] = []
        for tensor in frontier:
            node = producer.get(tensor)
            if node is None or tensor in visited:
                continue
            visited.add(tensor)
            if node.op_type not in _HEAD_STOP_OP_TYPES:
                next_frontier.extend(node.input)
            elif node.op_type in _QUANTIZED_OP_TYPES and node.name:
                excluded.add(node.name)
        frontier = next_frontier
    return excluded


def nodes_to_exclude(graph: Any) -> list[str]:
    """Return the names of nodes that must stay in float even among ``MatMul``/``Gemm``.

    Two groups, both found by walking the graph rather than matched by name, so a renamed or restructured export stays
    covered:

    * the attention-score multiplies -- an activation-by-activation ``MatMul`` reaching a ``Softmax`` directly or
      through constant scaling (``Q @ K^T / sqrt(d)``), where an 8-bit range quantizes the comparison the attention is
      about to make;
    * the last ``MatMul``/``Gemm`` on each path back from a graph output -- the detection heads' final layers, which
      emit box deltas and class logits directly, so quantizing them quantizes the answer. The walk passes through any
      number of elementwise ops (box refinement, sigmoid, reshapes) up to :data:`_MAX_WALK_HOPS` hops, and stops at a
      ``Conv``.

    Args:
        graph: An ``onnx.GraphProto``.

    Returns:
        Node names to pass to ``quantize_static(nodes_to_exclude=...)``, sorted and deduplicated.

    Examples:
        >>> import onnx
        >>> from onnx import helper, TensorProto
        >>> node = helper.make_node("MatMul", ["a", "b"], ["out"], name="head_matmul")
        >>> graph = helper.make_graph(
        ...     [node],
        ...     "g",
        ...     [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 1])],
        ...     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])],
        ... )
        >>> nodes_to_exclude(graph)
        ['head_matmul']
    """
    producer = {output: node for node in graph.node for output in node.output}
    excluded = _attention_score_matmuls(graph, producer, _constant_names(graph))
    excluded |= _head_matmuls(graph, producer)
    return sorted(excluded)


def _calibration_reader(batches: Sequence[NDArray[np.float32]], input_name: str) -> Any:
    """Wrap preprocessed *batches* in the reader interface ``quantize_static`` consumes.

    Args:
        batches: Preprocessed batches, each shaped like the graph input (see :func:`_group_samples`).
        input_name: Name of the graph's input tensor.

    Returns:
        An ``onnxruntime.quantization.CalibrationDataReader``.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
    """
    from onnxruntime.quantization import CalibrationDataReader  # optional dependency

    class _Reader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self._index = 0

        def get_next(self) -> dict[str, NDArray[np.float32]] | None:
            if self._index >= len(batches):
                return None
            batch = batches[self._index]
            self._index += 1
            return {input_name: batch}

        def rewind(self) -> None:
            self._index = 0

    return _Reader()


def _static_input_dims(graph_input: Any) -> tuple[int, int, int, int]:
    """Return the ``(N, C, H, W)`` the graph input was traced at, reading a symbolic batch dimension as 1.

    Args:
        graph_input: An ``onnx.ValueInfoProto`` -- the graph's image input.

    Returns:
        Batch size, channels, height and width.

    Raises:
        ValueError: If the input is not rank 4, or its channel or spatial dimensions are symbolic or not positive.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> _static_input_dims(helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 8, 8]))
        (1, 3, 8, 8)
    """
    dims = [int(dim.dim_value) for dim in graph_input.type.tensor_type.shape.dim]
    if len(dims) != 4 or min(dims[1:]) <= 0:
        raise ValueError(
            f"INT8 quantization needs a static (N, C, H, W) image input; {graph_input.name!r} has dims {dims} "
            "(0 marks a symbolic dimension). Export at a fixed channel count and resolution."
        )
    # dim_value is 0 for a symbolic (dynamic) batch: calibrate one sample at a time.
    return max(dims[0], 1), dims[1], dims[2], dims[3]


def _group_samples(samples: Sequence[NDArray[np.float32]], batch_size: int) -> list[NDArray[np.float32]]:
    """Stack single-sample ``(1, C, H, W)`` batches into ``(batch_size, C, H, W)`` feeds for a static-batch graph.

    Consecutive samples share a feed. The last feed is padded by repeating samples from the start of the set, so every
    feed matches the graph's fixed batch dimension and no sample is dropped.

    Args:
        samples: Single-sample batches in calibration order.
        batch_size: The graph input's static batch dimension.

    Returns:
        *samples* unchanged when *batch_size* is 1, otherwise ``ceil(len(samples) / batch_size)`` stacked feeds.

    Examples:
        >>> import numpy as np
        >>> samples = [np.full((1, 1, 1, 1), i, dtype=np.float32) for i in range(3)]
        >>> [feed[:, 0, 0, 0].tolist() for feed in _group_samples(samples, 2)]
        [[0.0, 1.0], [2.0, 0.0]]
    """
    if batch_size == 1:
        return list(samples)
    count = len(samples)
    return [
        np.concatenate([samples[(start + offset) % count] for offset in range(batch_size)], axis=0)
        for start in range(0, count, batch_size)
    ]


def _require_onnxruntime() -> None:
    """Refuse a host without ``onnxruntime``, which static quantization needs to run calibration.

    Raises:
        ImportError: If ``onnxruntime`` is not importable.
    """
    try:
        import onnxruntime  # noqa: F401  # probe only
    except ImportError as exc:
        raise ImportError(
            "quantization='int8' needs ONNX Runtime to collect activation ranges. Install it: "
            "`pip install onnxruntime` (or `pip install 'rfdetr[onnx]'`)."
        ) from exc


def quantize_int8(
    model_path: str | Path,
    calibration_data: str | Path | NDArray[Any],
    *,
    max_images: int = 100,
) -> Path:
    """Write a static INT8 QDQ copy of the ONNX graph at *model_path* and return its path.

    The source graph is left on disk untouched; the quantized model is written beside it with an ``_int8`` suffix.

    Args:
        model_path: The FP32 ``.onnx`` file to quantize.
        calibration_data: Directory of representative images, a ``.npy`` path, or a preprocessed array. Images are
            preprocessed exactly as inference does.
        max_images: Maximum images read from a *calibration_data* directory.

    Returns:
        Path of the written INT8 model.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        ValueError: If *calibration_data* yields no usable sample.

    Examples:
        Needs an exported graph and representative images, so this is documentation only (not a doctest):

        ```python
        quantize_int8("output/inference_model.onnx", "calibration_images/")
        # -> PosixPath('output/inference_model_int8.onnx')
        ```
    """
    _require_onnxruntime()

    import onnx  # heavy optional dependency, imported at call time like the rest of this format
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source = Path(model_path)
    target = source.with_name(f"{source.stem}_int8.onnx")

    # Scratch files live beside the source: the same filesystem keeps the final os.replace a rename, and a failed run
    # leaves neither the pre-processed graph nor a half-written target behind -- nor clobbers an earlier INT8 model.
    with tempfile.TemporaryDirectory(dir=source.parent, prefix=f".{source.stem}_quantize_") as scratch:
        prepared = Path(scratch) / f"{source.stem}_prep.onnx"
        staged = Path(scratch) / target.name

        # Shape inference and constant folding first: quantize_static refuses many graphs without it.
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)

        model = onnx.load(str(prepared))
        graph_input = model.graph.input[0]
        batch_size, channels, height, width = _static_input_dims(graph_input)
        excluded = nodes_to_exclude(model.graph)

        batches = list(
            calibration_batches(
                calibration_data,
                height=height,
                width=width,
                channels=channels,
                max_images=max_images,
            )
        )
        if not batches:
            raise ValueError("Calibration data produced no samples.")
        warn_if_too_few_samples(len(batches))
        logger.info(
            f"Quantizing {source.name} to INT8 from {len(batches)} calibration samples "
            f"({len(excluded)} attention/head nodes kept in float)"
        )

        quantize_static(
            str(prepared),
            str(staged),
            _calibration_reader(_group_samples(batches, batch_size), graph_input.name),
            quant_format=QuantFormat.QDQ,
            # Pinned rather than inherited: the accuracy figures in the module docstring were taken with min/max ranges.
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=list(_QUANTIZED_OP_TYPES),
            nodes_to_exclude=excluded,
            per_channel=False,  # see the module docstring: per-channel failed at session creation
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "OpTypesToExcludeOutputQuantization": list(_QUANTIZED_OP_TYPES),
            },
        )
        os.replace(staged, target)

    return target
