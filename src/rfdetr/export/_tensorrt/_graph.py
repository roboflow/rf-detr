# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Format-neutral ONNX graph helpers shared by the TensorRT FP16 cast and the INT8 quantizer.

Both :mod:`rfdetr.export._tensorrt.exporter` and :mod:`rfdetr.export._tensorrt.quantize` import from here, and this
module imports neither, so the two never need to import each other's privates.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from rfdetr.export.prepare import BATCH_AXIS

#: An explicitly quantized graph carries its precision in these nodes and their scale/zero-point
#: tensors, so a blanket fp16 cast contradicts it rather than converting it.
_QUANTIZATION_OP_TYPES = frozenset({"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"})


def _onnx_dynamic_batch_inputs(graph: Any) -> list[str]:
    """Collect the ONNX graph inputs whose batch axis is symbolic, in graph order.

    The file-level twin of :func:`~rfdetr.export._tensorrt.exporter._dynamic_batch_inputs`, which asks the same
    question of an already parsed TensorRT network. Reading it from the graph is what lets a request and a graph
    that disagree be refused before an fp16 cast rewrites every weight into a second copy of the model. An axis is
    symbolic when it carries a ``dim_param`` — how ``torch.onnx.export`` marks a dynamic dimension — or nothing at
    all, rather than a ``dim_value``.

    Args:
        graph: An ``onnx.GraphProto``.

    Returns:
        Each such input's name. A rank-0 input has no batch axis and is skipped rather than indexed into, as are
        inputs that are not tensors.

    Examples:
        Needs an ``onnx.GraphProto``, so this is documentation rather than a doctest:

        ```python
        _onnx_dynamic_batch_inputs(onnx.load("output/rfdetr-medium.onnx").graph)
        # -> ['input']
        ```
    """
    dynamic_inputs = []
    for value_info in graph.input:
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) > BATCH_AXIS and dims[BATCH_AXIS].WhichOneof("value") != "dim_value":
            dynamic_inputs.append(value_info.name)
    return dynamic_inputs


def _subgraphs(node: Any) -> Iterator[Any]:
    """Yield the graphs nested directly in a node's attributes (``If`` branches, ``Loop``/``Scan`` bodies).

    Args:
        node: ``NodeProto`` to inspect.

    Yields:
        Each ``GraphProto`` held by one of *node*'s attributes.

    Examples:
        Needs an ``onnx.NodeProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> list(_subgraphs(node))  # doctest: +SKIP
        []
    """
    for attribute in node.attribute:
        if attribute.HasField("g"):
            yield attribute.g
        yield from attribute.graphs


def _iter_graphs(graph: Any) -> Iterator[Any]:
    """Yield *graph* and every graph nested below it, depth first.

    Args:
        graph: ``GraphProto`` to walk.

    Yields:
        *graph* itself, then each subgraph reachable through its nodes' attributes.

    Examples:
        Needs an ``onnx.GraphProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> len(list(_iter_graphs(model.graph)))  # doctest: +SKIP
        1
    """
    yield graph
    for node in graph.node:
        for subgraph in _subgraphs(node):
            yield from _iter_graphs(subgraph)


def _tensor_names(graph: Any) -> set[str]:
    """Collect every tensor name bound anywhere in *graph*, subgraphs included.

    Args:
        graph: Graph to scan.

    Returns:
        Names claimed by inputs, outputs, initializers, ``value_info`` entries and node edges.

    Examples:
        Needs an ``onnx.GraphProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> sorted(_tensor_names(model.graph))  # doctest: +SKIP
        ['input', 'output']
    """
    names: set[str] = set()
    for nested in _iter_graphs(graph):
        names.update(value.name for value in nested.input)
        names.update(value.name for value in nested.output)
        names.update(value.name for value in nested.value_info)
        names.update(initializer.name for initializer in nested.initializer)
        for node in nested.node:
            names.update(node.input)
            names.update(node.output)
    return names


def _unique_name(base: str, taken: set[str]) -> str:
    """Derive a tensor name from *base* that no existing tensor claims, reserving it in *taken*.

    Args:
        base: Preferred name.
        taken: Names already bound in the graph; the returned name is added to it.

    Returns:
        *base* when it is free, otherwise *base* with the smallest numeric suffix that is.

    Examples:
        >>> _unique_name("dets_fp16", {"dets"})
        'dets_fp16'
        >>> _unique_name("dets_fp16", {"dets_fp16"})
        'dets_fp16_1'
    """
    candidate, suffix = base, 1
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate
