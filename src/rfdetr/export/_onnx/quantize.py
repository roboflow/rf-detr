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
detection heads, and leaving those ops' *outputs* in float brings Nano back to within 2.65 mAP, and is the
configuration this module applies. It is also the graph shape a published INT8 RF-DETR Base uses to hold roughly 53
mAP, so the remaining gap is a calibration and boundary-placement question rather than an architectural limit.

Three decisions follow from that and are deliberately not configurable:

* **Only ``MatMul`` and ``Gemm``.** Everything else stays float. Quantizing more is both less accurate and *slower*
  here, because the Q/DQ conversions around non-matmul ops cost more than the 8-bit kernels save.
* **Outputs stay float** (``OpTypesToExcludeOutputQuantization``). ORT otherwise quantizes each matmul's output too,
  so the ops reading it see an 8-bit-rounded tensor.
* **Per-tensor weights, not per-channel.** RF-DETR's decoder self-attention multiplies two dynamic tensors, and ORT
  emits a ``QLinearMatMul`` whose per-channel zero-point the CPU kernel rejects outright at session creation.

Calibration data is required and must be representative: unlike TFLite's dynamic-range mode, static quantization
derives activation ranges from the data it is shown, so random or out-of-domain images produce a model that loads and
runs and is quietly wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import calibration_batches
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Quantization modes ``format="onnx"`` accepts. ``None`` and ``"fp32"`` both mean "write the traced graph unchanged".
VALID_QUANTIZATIONS: frozenset[str | None] = frozenset({None, "fp32", "int8"})

#: The only ops quantized. See the module docstring for why the list is this short.
_QUANTIZED_OP_TYPES: tuple[str, ...] = ("MatMul", "Gemm")

#: How far back from a graph output a node is still considered part of a detection head.
_HEAD_DEPTH: int = 3


def nodes_to_exclude(graph: Any) -> list[str]:
    """Return the names of nodes that must stay in float even among ``MatMul``/``Gemm``.

    Two groups, both found by walking the graph rather than matched by name, so a renamed or restructured export stays
    covered:

    * the multiplies feeding a ``Softmax`` -- the attention scores, where an 8-bit range quantizes the comparison the
      attention is about to make;
    * the nodes within :data:`_HEAD_DEPTH` hops of a graph output -- the detection heads, which emit box coordinates
      and class logits directly, so quantizing them quantizes the answer.

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
    excluded: set[str] = set()

    for node in graph.node:
        if node.op_type != "Softmax":
            continue
        for name in node.input:
            source = producer.get(name)
            if source is not None and source.op_type in _QUANTIZED_OP_TYPES and source.name:
                excluded.add(source.name)

    frontier = {output.name for output in graph.output}
    for _ in range(_HEAD_DEPTH):
        next_frontier: set[str] = set()
        for name in frontier:
            node = producer.get(name)
            if node is None:
                continue
            if node.name:
                excluded.add(node.name)
            next_frontier.update(node.input)
        frontier = next_frontier

    return sorted(excluded)


def _calibration_reader(batches: Sequence[NDArray[np.float32]], input_name: str) -> Any:
    """Wrap preprocessed *batches* in the reader interface ``quantize_static`` consumes.

    Args:
        batches: Preprocessed single-image batches.
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
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source = Path(model_path)
    prepared = source.with_name(f"{source.stem}_prep.onnx")
    target = source.with_name(f"{source.stem}_int8.onnx")

    # Shape inference and constant folding first: quantize_static refuses many graphs without it.
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)

    try:
        model = onnx.load(str(prepared))
        graph_input = model.graph.input[0]
        dims = graph_input.type.tensor_type.shape.dim
        channels, height, width = (int(dims[i].dim_value) for i in (1, 2, 3))
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
        logger.info(
            f"Quantizing {source.name} to INT8 from {len(batches)} calibration samples "
            f"({len(excluded)} attention/head nodes kept in float)"
        )

        quantize_static(
            str(prepared),
            str(target),
            _calibration_reader(batches, graph_input.name),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=list(_QUANTIZED_OP_TYPES),
            nodes_to_exclude=excluded,
            per_channel=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "OpTypesToExcludeOutputQuantization": list(_QUANTIZED_OP_TYPES),
            },
        )
    finally:
        prepared.unlink(missing_ok=True)

    return target
