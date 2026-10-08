# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""TensorRT export helper: build a serialized engine from ONNX in-process.

The engine is built with the TensorRT Python API (via `polygraphy`), so no
``trtexec`` binary on ``PATH`` is required — only ``pip install rfdetr[tensorrt]``.

For TensorRT *inference*, use the ``inference-models`` library which provides
multi-backend RF-DETR support (PyTorch, ONNX, TensorRT) with automatic backend
selection::

    from inference_models import AutoModel

    model = AutoModel.from_pretrained("rfdetr-small")

See https://github.com/roboflow/inference/tree/main/inference_models for details.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import importlib
import operator
import os
import sys
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, get_args

import numpy as np
import torch
from numpy.typing import NDArray

from rfdetr.export._naming import resolve_export_stem
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._runtime.calibration_checks import check_calibration_data
from rfdetr.export._tensorrt.metadata import (
    build_engine_metadata,
    gpu_facts,
    is_engine_description,
    is_rfdetr_description,
    serialized_engine_facts,
    sidecar_path,
    write_engine_metadata,
)
from rfdetr.export._tensorrt.quantize import INT8, int8_source_graph
from rfdetr.export.base import ExportConfig, Exporter
from rfdetr.export.prepare import BATCH_AXIS, ExportGraph
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import is_installed

logger = get_logger()

#: The ``trt_hardware_compatibility`` spellings, each the lower-case name of a ``tensorrt.HardwareCompatibilityLevel``
#: member. ``NONE`` is left out: it is what leaving the setting unset already means.
HardwareCompatibility = Literal["ampere_plus", "same_compute_capability"]
#: The same spellings as a tuple, so validation reads the annotation instead of keeping a second list.
_HARDWARE_COMPATIBILITY_LEVELS: Final[tuple[str, ...]] = get_args(HardwareCompatibility)
#: Lowest compute capability ``"ampere_plus"`` can be built on: NVIDIA Ampere is 8.x.
_AMPERE_COMPUTE_CAPABILITY: Final[tuple[int, int]] = (8, 0)


#: Whether ``tensorrt`` itself is installed, probed without loading its CUDA libraries (see
#: :func:`~rfdetr.utilities.package.is_installed`). polygraphy imports it only once a build runs, so polygraphy
#: importing proves nothing about it either way.
_IS_TENSORRT_AVAILABLE = is_installed("tensorrt")

# polygraphy ships in the ``rfdetr[tensorrt]`` extra alongside ``tensorrt``, but it declares no requirements, so it
# also installs without it. Import it lazily at module scope (guarded) so importing this module never fails on hosts
# without TensorRT, and so tests can monkeypatch these names without polygraphy installed.
try:
    from polygraphy.backend.trt import (
        CreateConfig,
        Profile,
        engine_from_network,
        network_from_onnx_path,
    )
    from polygraphy.util import save_file

    #: Whether polygraphy, which drives the build, imported. Tracked apart from ``_IS_TENSORRT_AVAILABLE`` so a
    #: refusal can name whichever of the two packages is actually missing.
    _IS_POLYGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by TestTensorRTAvailability on a separately executed copy
    CreateConfig = None
    Profile = None
    engine_from_network = None
    network_from_onnx_path = None
    save_file = None

    _IS_POLYGRAPHY_AVAILABLE = False


# TensorRT 11 removed weak typing: ``BuilderFlag.FP16`` no longer exists and engine precision is
# taken from the ONNX graph's dtypes. Building FP16 there means casting the graph first, which needs
# ``onnx`` + ``onnxconverter-common`` (both in the ``rfdetr[tensorrt]`` extra). Only availability is
# resolved here; the modules themselves are imported inside the functions that use them, matching how
# ``export/_onnx/exporter.py`` handles the same optional dependency.
_IS_FP16_CASTER_AVAILABLE = all(is_installed(name) for name in ("onnx", "onnxconverter_common"))

# TensorRT majors at or above this are strongly typed, so an absent FP16 builder flag is by design
# rather than a sign of a lean/partial wheel.
_STRONG_TYPING_MAJOR = 11

# An explicitly quantized graph carries its precision in these nodes and their scale/zero-point
# tensors, so a blanket fp16 cast contradicts it rather than converting it.
_QUANTIZATION_OP_TYPES = frozenset({"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"})

#: Oldest TensorRT major an INT8 engine is built on. INT8 engines are strongly typed builds of an explicitly quantized
#: graph; 10.16 and 11.3 are the only releases this was measured on (issue #1024). Earlier majors are refused rather
#: than assumed to work; the 10.x releases before 10.16 are admitted but untested.
_INT8_MIN_TENSORRT_MAJOR = 10

#: The graph outputs an INT8 export is measured for: a detector. Segmentation and keypoint models add outputs and are
#: refused.
_INT8_OUTPUT_NAMES = ("dets", "labels")

#: Whether onnxruntime is installed: calibration runs the FP32 graph under it. The ``rfdetr[tensorrt]`` extra installs
#: it as onnxruntime-gpu.
_IS_ONNXRUNTIME_AVAILABLE = is_installed("onnxruntime")


class Fp16CastUnsupportedError(ValueError):
    """An ONNX graph cannot be cast to fp16 for a strongly typed TensorRT build.

    Subclasses ``ValueError`` so callers already guarding the conversion keep working, while the message stays in rfdetr
    terms instead of naming converter internals the caller has no way to reach.
    """


class Fp16Strategy(str, Enum):
    """How an FP16 engine can be obtained from the installed TensorRT.

    Examples:
        >>> Fp16Strategy("cast_graph") is Fp16Strategy.CAST_GRAPH
        True
    """

    #: Weakly typed builder: precision is requested with ``BuilderFlag.FP16``.
    BUILDER_FLAG = "builder_flag"
    #: Strongly typed builder (TensorRT >= 11): precision comes from an fp16 ONNX graph.
    CAST_GRAPH = "cast_graph"
    #: Lean/partial weakly typed wheel: no FP16 route at all, so the build falls back to FP32.
    UNAVAILABLE = "unavailable"


def _is_positive_integer(value: object) -> bool:
    """Return whether *value* is an integer of at least 1, NumPy integers included and booleans excluded.

    Examples:
        >>> _is_positive_integer(np.int64(5)), _is_positive_integer(True), _is_positive_integer(0)
        (True, False, False)
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return operator.index(value) >= 1  # type: ignore[arg-type]
    except TypeError:
        return False


def _describe(data: object) -> str:
    """Name *data*'s type, plus its dtype and shape for an array, for an error message.

    Examples:
        >>> _describe(np.zeros((2, 3), np.uint8)), _describe(["a.jpg"])
        ('uint8 array of shape (2, 3)', 'list')
    """
    if isinstance(data, np.ndarray):
        return f"{data.dtype} array of shape {data.shape}"
    return type(data).__name__


def _is_calibration_source(data: object) -> bool:
    """Return whether *data* has a form :func:`~rfdetr.export._runtime.calibration.calibration_batches` reads.

    A path must be non-empty (what it names is judged by ``check_calibration_data``); an array must already be
    preprocessed and hold at least one image, so an integer image array (raw pixels, not normalized) or an empty one
    is refused here.

    Examples:
        >>> _is_calibration_source("images/"), _is_calibration_source(np.zeros((1, 3, 8, 8), np.float32))
        (True, True)
        >>> _is_calibration_source(np.zeros((1, 3, 8, 8), np.uint8)), _is_calibration_source(["a.jpg"])
        (False, False)
        >>> _is_calibration_source("")
        False
    """
    if isinstance(data, (str, os.PathLike)):
        return bool(os.fspath(data))
    return (
        isinstance(data, np.ndarray) and data.ndim == 4 and data.shape[0] > 0 and np.issubdtype(data.dtype, np.floating)
    )


def _tensorrt_major(version: str) -> int | None:
    """Extract the major version number from a TensorRT version string.

    Args:
        version: Value of ``tensorrt.__version__``, e.g. ``"11.2.1.2"``.

    Returns:
        The leading integer, or ``None`` when *version* does not start with one (lean or
        vendored wheels sometimes report a non-numeric version).

    Examples:
        >>> _tensorrt_major("11.2.1.2")
        11
        >>> _tensorrt_major("10.16.1.11")
        10
        >>> _tensorrt_major("unknown") is None
        True
    """
    major, _, _ = version.partition(".")
    try:
        return int(major)
    except ValueError:
        return None


def _lean_library_name(major: int, platform: str) -> str:
    """Name the TensorRT lean runtime library a version-compatible build needs, for one major version and platform.

    Windows DLLs carry the major version from TensorRT 10 on; TensorRT 8.6 names it ``nvinfer_lean.dll``.

    Args:
        major: TensorRT major version.
        platform: ``sys.platform``.

    Returns:
        The library's file name, which is what the dynamic loader is asked for.

    Examples:
        >>> _lean_library_name(11, "linux")
        'libnvinfer_lean.so.11'
        >>> _lean_library_name(11, "win32")
        'nvinfer_lean_11.dll'
        >>> _lean_library_name(8, "win32")
        'nvinfer_lean.dll'
    """
    if platform == "win32":
        return f"nvinfer_lean_{major}.dll" if major >= 10 else "nvinfer_lean.dll"
    return f"libnvinfer_lean.so.{major}"


def resolve_fp16_strategy(trt_module: Any | None) -> tuple[Fp16Strategy, str]:
    """Decide how the installed TensorRT can produce an FP16 engine.

    The absent ``BuilderFlag.FP16`` behind this code path is a *symptom* of strong typing, so the major
    version is consulted first: a strongly typed TensorRT takes precision from the graph whether or not
    it still exposes a deprecated or no-op FP16 flag. Only on a weakly typed build does a missing flag
    mean a lean/partial wheel with no FP16 route at all.

    Args:
        trt_module: The imported ``tensorrt`` module, or ``None`` when it could not be imported.

    Returns:
        The strategy to use, paired with the version TensorRT reports (``"unknown"`` when it reports
        none).

    Examples:
        >>> from types import SimpleNamespace
        >>> strongly_typed = SimpleNamespace(__version__="11.2.1.2", BuilderFlag=SimpleNamespace())
        >>> resolve_fp16_strategy(strongly_typed)[0] is Fp16Strategy.CAST_GRAPH
        True
        >>> lean_wheel = SimpleNamespace(__version__="10.16.1.11", BuilderFlag=SimpleNamespace())
        >>> resolve_fp16_strategy(lean_wheel)[0] is Fp16Strategy.UNAVAILABLE
        True
    """
    if trt_module is None:
        # A missing or broken tensorrt import is surfaced by the caller's build chain, not diagnosed here.
        return Fp16Strategy.BUILDER_FLAG, "unknown"

    version = getattr(trt_module, "__version__", "unknown")
    major = _tensorrt_major(version)
    if major is not None and major >= _STRONG_TYPING_MAJOR:
        return Fp16Strategy.CAST_GRAPH, version
    if hasattr(getattr(trt_module, "BuilderFlag", None), "FP16"):
        return Fp16Strategy.BUILDER_FLAG, version
    return Fp16Strategy.UNAVAILABLE, version


def _dynamic_batch_inputs(network: Any) -> dict[str, tuple[int, ...]]:
    """Collect the network inputs whose batch axis is dynamic (``-1``), in input order.

    Args:
        network: A parsed TensorRT network, or anything exposing ``num_inputs`` and ``get_input(index)`` with a
            ``name`` and a ``shape``.

    Returns:
        Each such input's name mapped to its full shape. A rank-0 input has no batch axis and is never included.

    Examples:
        >>> from types import SimpleNamespace
        >>> inputs = [
        ...     SimpleNamespace(name="input", shape=(-1, 3, 384, 384)),
        ...     SimpleNamespace(name="orig_size", shape=(1, 2)),
        ...     SimpleNamespace(name="threshold", shape=()),
        ... ]
        >>> _dynamic_batch_inputs(SimpleNamespace(num_inputs=len(inputs), get_input=inputs.__getitem__))
        {'input': (-1, 3, 384, 384)}
    """
    dynamic_inputs = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) > BATCH_AXIS and shape[BATCH_AXIS] == -1:
            dynamic_inputs[tensor.name] = shape
    return dynamic_inputs


def _onnx_dynamic_batch_inputs(graph: Any) -> list[str]:
    """Collect the ONNX graph inputs whose batch axis is symbolic, in graph order.

    The file-level twin of :func:`_dynamic_batch_inputs`, which asks the same question of an already parsed TensorRT
    network. Reading it from the graph is what lets a request and a graph that disagree be refused before an fp16
    cast rewrites every weight into a second copy of the model. An axis is symbolic when it carries a ``dim_param``
    — how ``torch.onnx.export`` marks a dynamic dimension — or nothing at all, rather than a ``dim_value``.

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


def _reject_dynamic_graph_under_static_request(onnx_path: str, dynamic_inputs: Iterable[str]) -> None:
    """Refuse a graph whose batch axis is dynamic when the engine was not asked to carry one.

    Without an optimization profile, polygraphy fixes every dynamic dimension to 1 and only warns, so the build
    would hand back an engine that accepts batch 1 only. Both places that can notice the mismatch — the ONNX file
    before an fp16 cast, and the parsed network before the builder runs — raise through here, so the caller reads
    the same sentence whichever one got there first.

    Args:
        onnx_path: The caller's ``.onnx`` file, named in the error even when the mismatch was found on a cast copy.
        dynamic_inputs: Names of the inputs carrying a dynamic batch axis; empty when the graph carries none.

    Raises:
        ValueError: If *dynamic_inputs* names any input.

    Examples:
        >>> _reject_dynamic_graph_under_static_request("model.onnx", [])
        >>> try:
        ...     _reject_dynamic_graph_under_static_request("model.onnx", ["input"])
        ... except ValueError as error:
        ...     print(str(error).split(", but")[0])
        'model.onnx' has a dynamic batch axis on ['input']
    """
    named = list(dynamic_inputs)
    if not named:
        return
    raise ValueError(
        f"'{onnx_path}' has a dynamic batch axis on {named}, but dynamic_batch is off, so "
        "polygraphy would fix that axis to 1 and build an engine that accepts batch 1 only. Build it with "
        "TensorRTConfig(dynamic_batch=True, max_batch_size=<largest batch>), or export the ONNX graph without "
        "dynamic_batch for a fixed-batch engine."
    )


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


def _rename_tensor_uses(graph: Any, old: str, new: str) -> None:
    """Repoint every consumer of *old* at *new*, following captures into nested subgraphs.

    ``onnxconverter-common`` converts ``If``/``Loop``/``Scan`` bodies too, so a branch capturing an
    outer-scope tensor has to follow the rename or it keeps reading the restored FP32 boundary tensor.
    A subgraph binding its own tensor of that name shadows the outer one and is left alone.

    Args:
        graph: Graph whose node inputs are rewritten in place.
        old: Tensor name to stop consuming.
        new: Tensor name to consume instead.

    Examples:
        Needs an ``onnx.GraphProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _rename_tensor_uses(model.graph, "dets", "dets_fp16")  # doctest: +SKIP
    """
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for subgraph in _subgraphs(node):
            shadowed = any(value.name == old for value in subgraph.input) or any(
                initializer.name == old for initializer in subgraph.initializer
            )
            if not shadowed:
                _rename_tensor_uses(subgraph, old, new)


def _rebind_definition(graph: Any, name: str, inner: str) -> bool:
    """Rename whatever defines *name* to *inner*, freeing the original name for a boundary cast.

    Args:
        graph: Graph searched for the definition, mutated in place.
        name: Tensor name currently defined by a node output or an initializer.
        inner: Name that definition is moved to.

    Returns:
        Whether a definition was found — a graph output defined by nothing is left untouched rather
        than pointed at a name no node produces.

    Examples:
        Needs an ``onnx.GraphProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _rebind_definition(model.graph, "dets", "dets_fp16")  # doctest: +SKIP
        True
    """
    for node in graph.node:
        for index, output in enumerate(node.output):
            if output == name:
                node.output[index] = inner
                return True
    for initializer in graph.initializer:
        if initializer.name == name:
            initializer.name = inner
            return True
    return False


def _retarget_float_casts(graph: Any, declared: dict[str, int] | None = None) -> int:
    """Point pre-existing ``Cast(to=FLOAT)`` nodes at FLOAT16 after a graph-wide fp16 conversion.

    ``onnxconverter-common`` relabels tensors but leaves the ``to`` attribute of ``Cast`` nodes that
    were already in the source graph untouched. RF-DETR exports 33-35 such nodes, so the tensor stays
    float32 while its ``value_info`` claims float16 and TensorRT's strongly-typed parser rejects the
    graph at the first convolution. The converter rewrites ``If``/``Loop``/``Scan`` bodies as well, so
    nested graphs are walked too, with the enclosing declarations still in scope.

    Args:
        graph: Graph of an already-converted fp16 model, mutated in place.
        declared: Tensor types declared by enclosing graphs, passed down when recursing into a subgraph.

    Returns:
        Number of ``Cast`` nodes retargeted, subgraphs included.

    Examples:
        Needs a converted fp16 ``ModelProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _retarget_float_casts(model.graph)  # doctest: +SKIP
        33
    """
    from onnx import TensorProto

    in_scope = dict(declared or {})
    in_scope.update(
        {value.name: value.type.tensor_type.elem_type for value in list(graph.value_info) + list(graph.output)}
    )
    retargeted = 0
    for node in graph.node:
        if node.op_type != "Cast":
            for subgraph in _subgraphs(node):
                retargeted += _retarget_float_casts(subgraph, in_scope)
            continue
        for attribute in node.attribute:
            if (
                attribute.name == "to"
                and attribute.i == TensorProto.FLOAT
                and in_scope.get(node.output[0]) == TensorProto.FLOAT16
            ):
                attribute.i = TensorProto.FLOAT16
                retargeted += 1
    return retargeted


def _restore_fp32_inputs(graph: Any, taken: set[str]) -> int:
    """Put an FP32 -> FP16 cast behind every fp16 graph input, restoring the FP32 input contract.

    Args:
        graph: Graph of an already-converted fp16 model, mutated in place.
        taken: Tensor names already bound in the graph; generated names are uniquified against it.

    Returns:
        Number of boundary ``Cast`` nodes inserted.

    Examples:
        Needs a converted fp16 ``ModelProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _restore_fp32_inputs(model.graph, set())  # doctest: +SKIP
        1
    """
    from onnx import TensorProto, helper

    inserted = 0
    for tensor in graph.input:
        if tensor.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        inner = _unique_name(f"{tensor.name}_fp16", taken)
        _rename_tensor_uses(graph, tensor.name, inner)
        graph.node.insert(
            0, helper.make_node("Cast", [tensor.name], [inner], to=TensorProto.FLOAT16, name=f"Cast_{inner}_in")
        )
        tensor.type.tensor_type.elem_type = TensorProto.FLOAT
        inserted += 1
    return inserted


def _restore_fp32_outputs(graph: Any, taken: set[str]) -> int:
    """Put an FP16 -> FP32 cast in front of every fp16 graph output, restoring the FP32 output contract.

    Whatever defines the output is renamed and its remaining consumers follow it, so a tensor that is
    both a graph output and an internal input keeps reading the fp16 value while the boundary stays
    FP32. An output that is also a graph input needs no cast at all: the input side already restored
    the tensor itself, leaving only its output declaration to correct.

    Args:
        graph: Graph of an already-converted fp16 model, mutated in place.
        taken: Tensor names already bound in the graph; generated names are uniquified against it.

    Returns:
        Number of boundary ``Cast`` nodes inserted.

    Examples:
        Needs a converted fp16 ``ModelProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _restore_fp32_outputs(model.graph, set())  # doctest: +SKIP
        2
    """
    from onnx import TensorProto, helper

    graph_inputs = {value.name for value in graph.input}
    inserted = 0
    for tensor in graph.output:
        if tensor.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        if tensor.name in graph_inputs:
            tensor.type.tensor_type.elem_type = TensorProto.FLOAT
            continue
        inner = _unique_name(f"{tensor.name}_fp16", taken)
        if not _rebind_definition(graph, tensor.name, inner):
            continue
        _rename_tensor_uses(graph, tensor.name, inner)
        graph.node.append(
            helper.make_node("Cast", [inner], [tensor.name], to=TensorProto.FLOAT, name=f"Cast_{inner}_out")
        )
        tensor.type.tensor_type.elem_type = TensorProto.FLOAT
        inserted += 1
    return inserted


def _restore_fp32_io(graph: Any) -> int:
    """Re-establish FP32 graph inputs/outputs around an fp16 body by inserting boundary casts.

    A weakly-typed TensorRT FP16 engine keeps its I/O tensors FP32, so callers feed and read float32.
    Preserving that contract keeps the strongly-typed path a drop-in replacement. This is done here
    rather than via ``convert_float_to_float16(keep_io_types=True)`` because that option wires the
    FP32 graph input straight into an FP16 convolution without inserting a ``Cast``, which TensorRT
    rejects.

    Only the top-level graph carries the engine's I/O contract — a subgraph's inputs come from its
    owning ``If``/``Loop`` node rather than from the caller — so the boundary casts are top-level by
    construction; nested graphs are still followed wherever a renamed tensor is captured inside one.

    Args:
        graph: Graph of an already-converted fp16 model, mutated in place.

    Returns:
        Number of boundary ``Cast`` nodes inserted.

    Examples:
        Needs a converted fp16 ``ModelProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _restore_fp32_io(model.graph)  # doctest: +SKIP
        3
    """
    taken = _tensor_names(graph)
    inserted = _restore_fp32_inputs(graph, taken) + _restore_fp32_outputs(graph, taken)

    # The boundary tensors are FP32 again, but the conversion left value_info entries still declaring
    # them FLOAT16. graph.input/graph.output already carry the authoritative type, so drop the
    # contradicting duplicates rather than trying to correct them.
    stale = {t.name for t in list(graph.input) + list(graph.output)}
    keep = [value for value in graph.value_info if value.name not in stale]
    del graph.value_info[:]
    graph.value_info.extend(keep)

    return inserted


def _reject_uncastable_graph(graph: Any, onnx_path: str) -> None:
    """Fail early on a graph that a blanket fp16 cast would invalidate rather than convert.

    An explicitly quantized graph states its precision in its ``QuantizeLinear``/``DequantizeLinear``
    pairs, and ``onnxconverter-common`` neither blocks nor special-cases those ops, so it casts their
    float inputs like any other. Below opset 19 ``QuantizeLinear`` does not even accept a float16 input,
    making the result structurally invalid; above it the cast is legal but silently restates the
    quantization. Either way a strongly typed TensorRT wants the quantized graph as-is, so refusing here
    names the real cause instead of leaving it to a parser error pointing at the wrong node.

    Args:
        graph: Graph about to be converted.
        onnx_path: Source path, quoted in the error message.

    Raises:
        Fp16CastUnsupportedError: If the graph, or any graph nested in it, carries quantization nodes.

    Examples:
        Needs an ``onnx.GraphProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _reject_uncastable_graph(model.graph, "model.onnx")  # doctest: +SKIP
    """
    quantized = sorted(
        {
            node.op_type
            for nested in _iter_graphs(graph)
            for node in nested.node
            if node.op_type in _QUANTIZATION_OP_TYPES
        }
    )
    if quantized:
        raise Fp16CastUnsupportedError(
            f"'{onnx_path}' is an explicitly quantized graph ({', '.join(quantized)}); casting it to fp16 "
            "wholesale would contradict that quantization and produce a model TensorRT cannot parse. "
            "Build this engine with fp16=False -- a strongly typed TensorRT takes the quantized "
            "precision from the graph itself."
        )


def _cast_onnx_to_fp16(onnx_path: str, *, dynamic_batch: bool | None = None) -> str:
    """Write an fp16 copy of an ONNX model next to it, keeping FP32 graph inputs and outputs.

    The file is a build intermediate, not a deliverable: ``build_engine`` deletes it afterwards.
    Its name is unique rather than derived from *onnx_path*, so a build never overwrites -- and then
    deletes -- a same-named file it did not create, and concurrent builds from one source model
    cannot claim each other's graph. It is written beside the source model rather than under
    ``/tmp`` because it is the same order of size as the model and ``/tmp`` is often a tmpfs.

    Args:
        onnx_path: Path to the float32 ``.onnx`` model.
        dynamic_batch: Whether the engine this cast feeds was asked to carry a dynamic batch dimension, or ``None``
            when the caller has no such request to check the graph against (a benchmark build, say). ``False``
            refuses a graph whose batch axis is dynamic here, where the model is loaded anyway, rather than leaving
            it to the parsed network — by then the cast has converted every weight and written a second copy of the
            model for a build that could not have succeeded.

    Returns:
        Path to the newly written fp16 model.

    Raises:
        ImportError: If ``onnx``/``onnxconverter-common`` are not installed.
        Fp16CastUnsupportedError: If the graph cannot be cast to fp16 — it is explicitly quantized, or
            the converter rejects it (most often because the model already is fp16).
        ValueError: If *dynamic_batch* is ``False`` and the graph's batch axis is dynamic.

    Examples:
        >>> _cast_onnx_to_fp16("output/rfdetr-medium.onnx")  # doctest: +SKIP
        'output/rfdetr-medium.fp16-h7k2p9qw.onnx'
    """
    if not _IS_FP16_CASTER_AVAILABLE:
        raise ImportError(
            "Building an FP16 engine on TensorRT >= 11 requires casting the ONNX graph to FP16 first, "
            "because TensorRT 11 removed the FP16 builder flag and takes precision from the graph. "
            "Install the caster with: pip install rfdetr[tensorrt] "
            "(or pin an older TensorRT with: pip install 'tensorrt<11')."
        )

    import onnx
    from onnxconverter_common import float16

    model = onnx.load(onnx_path)
    _reject_uncastable_graph(model.graph, onnx_path)
    if dynamic_batch is False:
        # `None` means the caller has no request to check against, which is not the same as asking for a static
        # engine -- only the latter contradicts a dynamic batch axis, so only it refuses here.
        _reject_dynamic_graph_under_static_request(onnx_path, _onnx_dynamic_batch_inputs(model.graph))
    try:
        model = float16.convert_float_to_float16(model, keep_io_types=False)
    except ValueError as error:
        # The converter rejects an already-fp16 model by naming an internal keyword argument no rfdetr
        # caller can reach; restate it in rfdetr terms and keep the original as the cause.
        raise Fp16CastUnsupportedError(
            f"'{onnx_path}' could not be cast to fp16 (most often because it already is fp16). Point the "
            "export at a float32 ONNX model, or request an FP32 engine with fp16=False."
        ) from error
    retargeted = _retarget_float_casts(model.graph)
    inserted = _restore_fp32_io(model.graph)
    logger.debug(f"fp16 cast: retargeted {retargeted} Cast node(s), inserted {inserted} boundary cast(s)")

    stem = os.path.basename(os.path.splitext(onnx_path)[0])
    handle, fp16_path = tempfile.mkstemp(prefix=f"{stem}.fp16-", suffix=".onnx", dir=os.path.dirname(onnx_path) or ".")
    os.close(handle)
    try:
        onnx.save(model, fp16_path)
    except Exception:
        # The caller only learns the path on a successful return, so nothing else can clean this up.
        os.remove(fp16_path)
        raise
    return fp16_path


@contextlib.contextmanager
def fp16_source_graph(onnx_path: str, *, dynamic_batch: bool | None = None) -> Iterator[str]:
    """Provide an fp16 copy of *onnx_path* to build from, deleting it when the block exits.

    The copy is a build intermediate, so it goes whether the build succeeds or fails. Removing it is
    best effort on purpose: the file may already be gone, or still be held open by the parser (Windows
    raises ``PermissionError`` then), and neither may replace the build's own exception.

    Args:
        onnx_path: Path to the float32 ``.onnx`` model to build from.
        dynamic_batch: The batch request the engine is being built for, forwarded to :func:`_cast_onnx_to_fp16`
            so a graph that contradicts it is refused before the copy is written. ``None`` when the caller has no
            such request.

    Yields:
        Path to the fp16 copy, valid only inside the ``with`` block.

    Raises:
        ImportError: If ``onnx``/``onnxconverter-common`` are not installed.
        Fp16CastUnsupportedError: If the graph cannot be cast to fp16.
        ValueError: If *dynamic_batch* is ``False`` and the graph's batch axis is dynamic.

    Examples:
        >>> with fp16_source_graph("output/rfdetr-medium.onnx") as fp16_path:  # doctest: +SKIP
        ...     engine_from_network(network_from_onnx_path(fp16_path))
    """
    cast_path = _cast_onnx_to_fp16(onnx_path, dynamic_batch=dynamic_batch)
    try:
        yield cast_path
    finally:
        with contextlib.suppress(OSError):
            Path(cast_path).unlink(missing_ok=True)


def _is_same_file(first: str, second: str) -> bool:
    """Tell whether two paths name one file, whether or not either of them exists yet.

    Two existing files are compared by identity, so a hard or symbolic link to the other counts. Otherwise each path is
    compared by where it resolves, which is where it would be created.

    Args:
        first: One path.
        second: The other path.

    Returns:
        ``True`` if both paths name the same file.

    Examples:
        >>> _is_same_file("engine.cache", os.path.join(".", "engine.cache"))
        True
        >>> _is_same_file("engine.cache", "model.onnx")
        False
    """
    try:
        return os.path.samefile(first, second)
    except OSError:
        # At least one of them does not exist yet.
        return os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second))


@dataclass(frozen=True, slots=True)
class TensorRTConfig(ExportConfig):
    """Settings for ``format="tensorrt"``, which builds an engine from an ONNX export.

    Attributes:
        opset_version: ONNX opset the intermediate graph targets.
        fp16: Enable FP16 precision when building the engine. How this is achieved depends on the
            installed TensorRT: weakly typed builds (TensorRT < 11) set the FP16 builder flag, while
            strongly typed ones (TensorRT >= 11, which removed that flag) get an FP16 engine by casting
            the ONNX graph to FP16 first — the engine's own inputs and outputs stay FP32 either way.
            Only downgraded to FP32 (with a warning) on a lean/partial TensorRT < 11 wheel that does not
            expose the flag, where no graph-level alternative exists; the engine filename then reflects
            the precision actually built (except under :meth:`TensorRTExporter.build_engine`'s *dry_run*,
            where nothing is built or probed, so the requested value is used as-is).
        opt_batch_size: With ``dynamic_batch``, the batch size the engine's optimization profile is tuned for
            (TensorRT picks kernels for this shape; other sizes within the profile run but may be slower). Fed
            from :meth:`rfdetr.detr.RFDETR.export`'s ``batch_size``, the same value the ONNX graph is traced at.
        max_batch_size: With ``dynamic_batch``, the largest batch the engine accepts; the profile spans
            ``1 .. max_batch_size``. Required when ``dynamic_batch`` is set, ignored otherwise.
        quantization: ``None`` builds an FP16 or FP32 engine as ``fp16`` says. ``"int8"`` builds an engine that runs
            most of the backbone encoder and decoder in INT8 and the rest in FP16, from a graph quantized with ranges
            calibrated on *calibration_data* (placement rules in :mod:`rfdetr.export._tensorrt.quantize`). It needs
            ``fp16``, a static batch, a full detector (no ``backbone_only``, no segmentation or keypoint head) and
            TensorRT 10 or newer.
        calibration_data: Representative images for ``quantization="int8"``: a directory of images, a ``.npy`` file
            or an array shaped ``(N, C, H, W)`` already normalized as :meth:`~rfdetr.detr.RFDETR.predict` does.
            Required with ``"int8"``, refused without it.
        max_images: Most images read from a *calibration_data* directory.
        metadata: Also write ``<engine>.json`` beside the engine: its input size and normalization, output names,
            batch profile, the precision actually built, the TensorRT version and GPU it was built on, and the engine
            file's size and SHA-256. A consumer that does not import the model (C++, Triton, DeepStream) reads it to
            run the engine. Written by :meth:`rfdetr.detr.RFDETR.export` and by calling the exporter on a prepared
            graph; :meth:`build_engine` alone, which has no graph, does not write it and warns that the setting has no
            effect there. ``False`` (the default) writes no description.
        hardware_compatibility: Ask TensorRT for an engine that other GPUs may run too. ``"ampere_plus"`` targets
            NVIDIA Ampere GPUs (compute capability 8.x) and newer, and needs an Ampere or newer GPU to build;
            ``"same_compute_capability"`` targets GPUs that share the building GPU's compute capability. Not supported
            on Jetson (JetPack) or DriveOS. The engine may run slower than one built for a single GPU. ``None`` (the
            default) builds for the building GPU only.
        version_compatible: Ask TensorRT for an engine that other releases of the same TensorRT major version may
            load. It worked between TensorRT 11.2 and 11.3 in both directions; it did not load across major versions,
            nor between 10.13 and 10.16. The build needs TensorRT's lean runtime library, which is a separate package
            from ``tensorrt`` (``tensorrt-lean-cu*-libs``); without it the build is refused. An
            engine built by TensorRT 11 carries host code, so loading it needs ``engine_host_code_allowed=True`` on
            :class:`~rfdetr.export._tensorrt.inference.TRTInference`. ``False`` (the default) builds an engine that
            loads on the building TensorRT version only.
        timing_cache: File that keeps the timings TensorRT measured while choosing kernels. A build loads it when it
            exists and writes the merged timings back, so the next build of layers with the same shapes (weights may
            differ) at the same precision and batch profile, on the same GPU and TensorRT version, skips the search it
            already did. A cache written by another TensorRT major version, or an empty or damaged file, does not stop
            the build: TensorRT logs an error, builds as if there were no cache, and the file gets this build's
            timings. ``None`` (the default) leaves TensorRT's own per-build cache in charge and touches no file.
    """

    opset_version: int = 17
    fp16: bool = True
    opt_batch_size: int = 1
    max_batch_size: int | None = None
    quantization: Literal["int8"] | None = None
    calibration_data: str | Path | NDArray[Any] | None = field(default=None, compare=False)
    max_images: int = 100
    metadata: bool = False
    hardware_compatibility: HardwareCompatibility | None = None
    version_compatible: bool = False
    timing_cache: str | os.PathLike[str] | None = None

    def onnx_stage(self) -> OnnxConfig:
        """Return the configuration for the ONNX export this format builds from.

        Returns:
            An :class:`~rfdetr.export._onnx.exporter.OnnxConfig` carrying the settings the intermediate graph needs.

        Examples:
            >>> TensorRTConfig(fp16=False).onnx_stage().opset_version
            17
        """
        return OnnxConfig.derive(self, opset_version=self.opset_version)


@dataclass(frozen=True, slots=True)
class _BuiltEngine:
    """What one engine build produced: the facts :meth:`TensorRTExporter._convert` describes the engine with.

    Attributes:
        path: Path the engine was written to.
        fp16: Whether the engine was built at FP16, after the lean-wheel fallback, so not always what was requested.
            The engine's file name does not carry it when ``output_name`` is set.
        engine_facts: The size and SHA-256 of the engine the build serialized, from :func:`serialized_engine_facts`,
            when the build was asked for them; ``None`` otherwise.

    Examples:
        >>> _BuiltEngine(path="model_fp32.trt", fp16=False, engine_facts=None).path
        'model_fp32.trt'
    """

    path: str
    fp16: bool
    engine_facts: dict[str, Any] | None


class TensorRTExporter(Exporter[TensorRTConfig]):
    """Export to TensorRT by running an ONNX export first and compiling its output into an engine.

    Unlike the portable formats, the engine is compiled for the machine that builds it: by default it is tied to that
    kind of GPU and that TensorRT version and does not move to another host. ``hardware_compatibility`` and
    ``version_compatible`` ask TensorRT for an engine that other GPUs, or other releases of the same TensorRT major
    version, may load.

    With ``dynamic_batch`` the intermediate ONNX graph carries a dynamic batch axis and the engine is built with one
    optimization profile spanning batch ``1 .. max_batch_size`` (tuned for ``opt_batch_size``); without it the engine
    accepts only the traced batch size.

    With ``quantization="int8"`` the engine is built from an explicitly quantized copy of the FP16 graph, calibrated on
    ``calibration_data``, and named ``*_int8.trt``.

    Examples:
        Requires the optional ``tensorrt`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        TensorRTExporter(TensorRTConfig(variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small_fp16.trt')
        ```
    """

    config_class = TensorRTConfig
    setting_names = {
        "opset_version": "opset_version",
        "fp16": "fp16",
        "opt_batch_size": "batch_size",
        "max_batch_size": "max_batch_size",
        "quantization": "quantization",
        "calibration_data": "calibration_data",
        "max_images": "max_images",
        "metadata": "trt_metadata",
        "hardware_compatibility": "trt_hardware_compatibility",
        "version_compatible": "trt_version_compatible",
        "timing_cache": "trt_timing_cache",
    }
    format = "tensorrt"
    display_name = "TensorRT"
    supports_dynamic_batch = True
    supports_notes = True
    pip_extra = "tensorrt"

    def _check_capabilities(self) -> None:
        """Reject settings that cannot work, before any work on the model.

        Raises:
            ValueError: For a quantization request :meth:`_check_quantization` refuses; if ``dynamic_batch`` is set
                without ``max_batch_size``; with a ``batch_size`` or ``max_batch_size`` that is not a plain ``int``
                (``bool`` included, since ``bool`` is a subclass of ``int``); with ``max_batch_size <
                opt_batch_size`` or ``opt_batch_size < 1``; with a ``metadata`` that is not a ``bool``; with a
                ``hardware_compatibility`` other than ``None``, ``"ampere_plus"`` or ``"same_compute_capability"``;
                with a ``version_compatible`` that is not a ``bool``; or with a ``timing_cache`` that is not a
                non-empty file path, that contains a NUL byte, or that ends in a path separator.
        """
        super()._check_capabilities()
        self._check_quantization()
        if not isinstance(self.config.metadata, bool):
            raise ValueError(f"trt_metadata must be a bool, got {self.config.metadata!r}.")
        self._check_portability()
        self._check_timing_cache()
        if not self.config.dynamic_batch:
            return
        if self.config.max_batch_size is None:
            raise ValueError(
                "TensorRT export with dynamic_batch=True needs max_batch_size: the engine is built with one "
                "optimization profile spanning batch 1 .. max_batch_size (tuned for batch_size). Pass "
                "max_batch_size=<largest batch the engine must accept>."
            )
        # A float (or float('nan')) compares fine against int bounds below -- nan is neither < nor >= anything,
        # so it silently clears every check here and only fails deep inside the TensorRT build, after a full
        # DINOv2 forward pass and an ONNX export have already run.
        for name, value in (("batch_size", self.config.opt_batch_size), ("max_batch_size", self.config.max_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"TensorRT dynamic_batch profile bounds must be integers, got {name}={value!r}.")
        if self.config.opt_batch_size < 1 or self.config.max_batch_size < self.config.opt_batch_size:
            raise ValueError(
                f"TensorRT dynamic_batch profile must satisfy 1 <= batch_size <= max_batch_size, got "
                f"batch_size={self.config.opt_batch_size} and max_batch_size={self.config.max_batch_size}."
            )

    def _check_quantization(self) -> None:
        """Reject a quantization request this exporter cannot honour as asked.

        Raises:
            ValueError: If *quantization* is not ``None`` or ``"int8"``; if *calibration_data* is given without
                ``"int8"``; if ``"int8"`` comes without *calibration_data*, with ``fp16=False``, with
                ``dynamic_batch``, with ``backbone_only``, or with a *max_images* that is not a positive integer; or if
                *calibration_data* is neither a non-empty path nor a non-empty rank-4 floating-point array, or is a
                path that does not exist, a directory without an image, or a file that is not ``.npy``.
        """
        config = self.config
        if config.quantization not in (None, INT8):
            raise ValueError(
                f"TensorRT export accepts quantization=None or 'int8', got {config.quantization!r}. Choose an FP16 or "
                "FP32 engine with fp16=True/False instead."
            )
        if config.quantization is None:
            if config.calibration_data is not None:
                raise ValueError("calibration_data is only read with quantization='int8'; pass both or neither.")
            return
        refusals = (
            (config.calibration_data is None, "needs calibration_data: a directory of representative images"),
            (not config.fp16, "builds on the FP16 graph, so it cannot be combined with fp16=False"),
            (config.dynamic_batch, "is measured for a static batch only; drop dynamic_batch"),
            (config.backbone_only, "applies to the full detector, not a backbone_only export"),
            (
                not _is_positive_integer(config.max_images),
                f"needs max_images to be a positive integer, got {config.max_images!r}",
            ),
            (
                not _is_calibration_source(config.calibration_data),
                "needs calibration_data to be a directory, a .npy path, or a preprocessed (N, C, H, W) float array "
                "with at least one image, "
                f"got {_describe(config.calibration_data)}",
            ),
        )
        for refused, reason in refusals:
            if refused:
                raise ValueError(f"TensorRT quantization='int8' {reason}.")
        # What the path itself shows (missing, no image, not a .npy): the check ONNX and OpenVINO make at this point.
        assert config.calibration_data is not None  # refused above
        check_calibration_data(config.calibration_data)

    @classmethod
    def _require_int8_host(cls) -> None:
        """Refuse a host that cannot build an INT8 engine: TensorRT older than 10, or no onnxruntime or FP16 caster.

        Raises:
            ImportError: If onnxruntime or onnxconverter-common is missing, or the installed TensorRT is older than 10.
        """
        if not _IS_ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "INT8 TensorRT export calibrates with onnxruntime, which is not installed. "
                'Install with: pip install "rfdetr[tensorrt]"'
            )
        if not _IS_FP16_CASTER_AVAILABLE:
            raise ImportError(
                "INT8 TensorRT export quantizes the FP16-cast graph, which needs onnxconverter-common. "
                'Install with: pip install "rfdetr[tensorrt]"'
            )
        import tensorrt as trt_module

        major = _tensorrt_major(trt_module.__version__)
        if major is None or major < _INT8_MIN_TENSORRT_MAJOR:
            raise ImportError(
                f"INT8 TensorRT export needs TensorRT {_INT8_MIN_TENSORRT_MAJOR} or newer for its strongly typed "
                f"build; found {trt_module.__version__}. "
                f"Install with: pip install 'tensorrt>={_INT8_MIN_TENSORRT_MAJOR}'"
            )

    def _require_calibration_path(self) -> None:
        """Refuse a *calibration_data* path that cannot be calibrated on, before any graph work reads it.

        The path is judged again by :func:`~rfdetr.export._runtime.calibration_checks.check_calibration_data`, because
        it can change between the configuration and the build. A ``.npy`` file is then held to the rule an in-memory
        array meets in :meth:`_check_quantization`; it is opened memory-mapped, so only its header is read here.

        Raises:
            ValueError: If *calibration_data* is a path that names nothing on disk, a directory without an image, a
                file that is not ``.npy``, a ``.npy`` that cannot be read, or one that does not hold a non-empty
                rank-4 floating-point array.
        """
        data = self.config.calibration_data
        if not isinstance(data, (str, os.PathLike)):
            return
        check_calibration_data(data)
        path = Path(data)
        if path.is_dir():
            return
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (ValueError, EOFError, OSError) as error:
            raise ValueError(
                f"TensorRT quantization='int8' could not read {path.name!r} as a .npy array: {error}"
            ) from error
        # Released before raising, so the error's traceback does not keep the file mapped (Windows would lock it).
        usable, described = _is_calibration_source(array), _describe(array)
        del array
        if not usable:
            raise ValueError(
                f"TensorRT quantization='int8' needs {path.name!r} to hold a preprocessed (N, C, H, W) float array "
                f"with at least one image, got {described}."
            )

    def _check_portability(self) -> None:
        """Reject a portability setting with a spelling or type that would otherwise be read the wrong way.

        A truthy non-``bool`` for ``version_compatible`` (``"no"``, ``1``) would switch the mode on by accident, and an
        unknown hardware level would only fail deep inside the build, after the forward pass and the ONNX export.

        Raises:
            ValueError: If ``hardware_compatibility`` is not ``None`` or one of :data:`_HARDWARE_COMPATIBILITY_LEVELS`,
                or ``version_compatible`` is not a ``bool``.
        """
        level = self.config.hardware_compatibility
        if level is not None and not (isinstance(level, str) and level in _HARDWARE_COMPATIBILITY_LEVELS):
            allowed = ", ".join(map(repr, _HARDWARE_COMPATIBILITY_LEVELS))
            raise ValueError(f"trt_hardware_compatibility must be None or one of {allowed}, got {level!r}.")
        if not isinstance(self.config.version_compatible, bool):
            raise ValueError(f"trt_version_compatible must be a bool, got {self.config.version_compatible!r}.")

    def _check_timing_cache(self) -> None:
        """Reject a ``timing_cache`` that cannot name a file, before the export pays for a forward pass.

        Only the value is read here. Whether the location can be written needs the filesystem, so
        :meth:`_prepare_timing_cache` finds that out later, still before the engine is built.

        Raises:
            ValueError: If ``timing_cache`` is set and is not a non-empty ``str`` or ``os.PathLike`` of one, if it
                contains a NUL byte, or if it ends in a path separator, which names a directory.
        """
        cache = self.config.timing_cache
        if cache is None:
            return
        try:
            path: str | bytes | None = os.fspath(cache)
        except TypeError:
            path = None
        if not isinstance(path, str) or not path:
            raise ValueError(f"trt_timing_cache must be a non-empty file path (str or os.PathLike), got {cache!r}.")
        if "\x00" in path:
            # No file can be named this; the filesystem calls would raise a ValueError naming no setting at all.
            raise ValueError(f"trt_timing_cache must not contain a NUL byte, got {path!r}.")
        if not os.path.basename(path):
            raise ValueError(f"trt_timing_cache must name a file, but {path!r} names a directory.")

    @classmethod
    def check_dependencies(cls) -> None:
        """Refuse a host without TensorRT or ``onnx`` before the export prepares the graph.

        The TensorRT probe reads a flag resolved at import time from ``find_spec``, so asking costs nothing on a host
        that does have TensorRT — and everything it saves on one that does not: :meth:`_convert` is only reached once
        :func:`~rfdetr.export.prepare.prepare_export_graph` has run a full forward pass through the model. It comes
        first because the ``onnx`` check imports ``onnx``: a refused request must not leave it loaded ahead of
        TensorFlow, or a ``format="tflite"`` export later in the same process starts in the import order that hangs
        its conversion (see :func:`~rfdetr.export._backend.preload_tensorflow_before_onnx`). A missing ``onnx`` names
        this format's extra, which installs it along with TensorRT.

        Raises:
            ImportError: If ``tensorrt``, ``polygraphy`` or ``onnx`` is not installed.
        """
        cls._require_tensorrt()
        from rfdetr.export._backend import check_onnx_available

        check_onnx_available('Install with: pip install "rfdetr[tensorrt]"', stage="TensorRT export")

    def check_environment(self) -> None:
        """Refuse a request that this host cannot build, before the forward pass.

        That is a portability request the installed TensorRT cannot build, and an INT8 request on a host without
        onnxruntime, the FP16 caster or TensorRT 10, or whose ``calibration_data`` path no longer holds usable data.

        ``RFDETR.export`` and ``Exporter.__call__`` call it after :meth:`check_dependencies`; :meth:`build_engine`, a
        public entry point that bypasses both, calls it too. A request that passes on a TensorRT older than 11 also
        gets its ``version_compatible`` warning here, before the forward pass rather than after it.

        Raises:
            ImportError: If ``version_compatible`` is set and TensorRT's lean runtime library cannot be loaded, or if
                ``quantization="int8"`` is set and onnxruntime or onnxconverter-common is missing or TensorRT is older
                than 10.
            ValueError: If ``hardware_compatibility`` names a level this TensorRT does not have, or is
                ``"ampere_plus"`` while the current CUDA device is older than Ampere (compute capability below 8.0);
                or if ``quantization="int8"`` is set and ``calibration_data`` is a path that
                :meth:`_require_calibration_path` refuses.
        """
        if self.config.quantization == INT8:
            self._require_int8_host()
            self._require_calibration_path()
        self._require_lean_runtime()
        if self.config.hardware_compatibility is not None:
            self._hardware_compatibility_level(self.config.hardware_compatibility)
        if self.config.hardware_compatibility == "ampere_plus":
            self._require_ampere_or_newer_gpu()
        if self.config.version_compatible:
            self._warn_if_version_compatibility_is_unverified()

    def _convert(self, graph: ExportGraph) -> str:
        """Export to ONNX, build the engine from it, and return the engine's path.

        The timing-cache filesystem preflight runs here, after graph preparation and before the ONNX export and the
        build, and again in the public :meth:`build_engine`, which callers can reach without going through ``_convert``.

        Raises:
            ImportError: If ``tensorrt`` or ``polygraphy`` is not installed, or (for ``quantization="int8"``)
                onnxruntime or onnxconverter-common is missing or TensorRT is older than 10, before the ONNX export
                runs.
            ValueError: If ``quantization="int8"`` names a *calibration_data* path that has vanished since the
                configuration was checked: :meth:`check_environment` refuses it before the forward pass when
                ``Exporter.__call__`` runs it, but a direct call of ``_convert`` meets it only in :meth:`_build`, after
                the ONNX export.
            NotImplementedError: If ``quantization="int8"`` is asked of a segmentation or keypoint model.
            FileExistsError: If ``metadata`` is set and the description's path holds something this exporter did not
                write, before the engine is built.
            OSError: If ``metadata`` is set and the description cannot be written; the engine has been built by then.
            ValueError: If ``timing_cache`` is a directory or another non-regular file, before the ONNX export runs.
            OSError: If ``timing_cache`` cannot be written, before the ONNX export runs.
        """
        # Exporter.__call__ has already run check_dependencies; this repeats its TensorRT half for a caller of _convert
        # itself, which would otherwise learn of a missing TensorRT only from _build, after the ONNX export.
        self._require_tensorrt()
        if self.config.quantization == INT8:
            # Exporter.__call__ has already run check_environment (host and calibration path); a caller of _convert
            # itself would otherwise learn of a missing onnxruntime only from _build, after the ONNX export.
            self._require_int8_host()
            if tuple(graph.output_names) != _INT8_OUTPUT_NAMES:
                raise NotImplementedError(
                    f"TensorRT quantization='int8' is measured for detection models only; this model also outputs "
                    f"{list(graph.output_names)[2:]}. Export it with quantization=None."
                )
        # A cache location that cannot be written would only be found out after the engine is built.
        self._prepare_timing_cache()
        onnx_path = OnnxExporter(self.config.onnx_stage())(graph)
        # A backbone-only export already carries the "-backbone" marker in the ONNX stem; reuse that stem so a
        # custom output_name does not silently produce an engine indistinguishable from a full-detector one.
        output_name = onnx_path.stem if graph.backbone_only and self.config.output_name else self.config.output_name
        if self.config.metadata:
            self._refuse_to_replace_a_foreign_file(str(onnx_path), output_name)
        logger.info("Converting ONNX model to TensorRT engine")
        built = self._build(str(onnx_path), output_name=output_name, digest=self.config.metadata)
        # The build records the engine's size and digest exactly when metadata asks for a description.
        if built.engine_facts is not None:
            self._write_metadata(graph, built.path, fp16=built.fp16, engine_facts=built.engine_facts)
        return built.path

    def _refuse_to_replace_a_foreign_file(self, onnx_path: str, output_name: str | None) -> None:
        """Refuse a description that would replace a file this exporter did not write, before the engine is built.

        ``.json`` is a generic extension, so ``<engine>.json`` may be a label map or a deployment manifest of the user's
        own, and the description would replace it without a word. Asked here rather than at the write, so the refusal
        does not come after a build that can take minutes. The path is the one the requested precision gives; a lean
        TensorRT wheel that falls back from FP16 to FP32 writes beside an ``_fp32`` engine instead, which this does not
        look at.

        Args:
            onnx_path: The ONNX file the engine is built from.
            output_name: The engine's file name without extension, as :meth:`_build` receives it.

        Raises:
            FileExistsError: If anything other than a description an RF-DETR export wrote is at the description's
                path, a directory or a dangling link included.
        """
        target = sidecar_path(self._engine_path(onnx_path, fp16_used=self.config.fp16, output_name=output_name))
        if os.path.lexists(target) and not is_rfdetr_description(target):
            raise FileExistsError(
                f"{target} already exists and was not written by an RF-DETR TensorRT export, so trt_metadata=True "
                "would replace it with the engine's description. Rename or remove it, or pick another output_name; "
                "the TensorRT engine was not built."
            )

    def _write_metadata(
        self, graph: ExportGraph, engine_path: str, *, fp16: bool, engine_facts: dict[str, Any]
    ) -> None:
        """Write the ``<engine>.json`` sidecar for the engine :meth:`_build` just wrote.

        Args:
            graph: The prepared graph the engine was built from.
            engine_path: Path of the engine.
            fp16: Whether the engine was built at FP16, after the lean-wheel fallback; an INT8 request is recorded as
                ``"int8"`` whatever this says.
            engine_facts: The size and SHA-256 of the engine the build serialized.

        Raises:
            OSError: If the file cannot be written. The message says the engine itself was built, because this runs
                after a build that can take minutes. The failed write's errno and file names are kept, and with them
                its subclass (a refused permission is still a ``PermissionError``); the failure is chained as the cause.
        """
        import tensorrt

        document = build_engine_metadata(
            self.config,
            graph,
            engine=engine_facts,
            precision=self._precision(fp16),
            tensorrt_version=tensorrt.__version__,
            gpu=gpu_facts(),
        )
        try:
            sidecar = write_engine_metadata(engine_path, document)
        except OSError as error:
            # strerror rather than str(error): the error raised below prints its own errno and file names around this.
            reason = error.strerror or str(error)
            message = f"The engine was written to {engine_path}, but its description could not be: {reason}."
            earlier = sidecar_path(engine_path)
            if is_engine_description(earlier):
                message += f" The earlier {earlier} was written for a previous engine and may not describe this one."
            if error.errno is None:
                raise OSError(message) from error
            # OSError picks its subclass from the errno (EACCES gives PermissionError), so a caller handling the
            # filesystem's reason still recognizes it. add_note would keep the original object, but needs Python 3.11.
            raise OSError(error.errno, message, error.filename, None, error.filename2) from error
        logger.info(f"Wrote the engine description: {sidecar}")

    @staticmethod
    def _warn_about_stale_description(engine_path: str) -> None:
        """Warn when a description written for an earlier engine of the same name sits beside the new one.

        The engine's file name does not encode its batch profile or shape, so a second export overwrites the first and
        leaves the first one's ``.json`` describing an engine that no longer exists. It is not deleted: the export did
        not write it.

        Args:
            engine_path: Path of the engine that was just built.
        """
        sidecar = sidecar_path(engine_path)
        if is_engine_description(sidecar):
            logger.warning(
                f"{sidecar} was written for an earlier engine with the same name and may not describe the one just "
                "built. Export with trt_metadata=True to write a new one, or delete it."
            )

    def build_engine(self, onnx_path: str, *, dry_run: bool = False, output_name: str | None = None) -> str:
        """Build a serialized TensorRT engine from an already-exported ONNX model, in-process.

        Uses the TensorRT Python API through ``polygraphy`` — no ``trtexec`` subprocess. Workspace size is left to
        the TensorRT default (it auto-sizes to the available device memory), which meets or exceeds the historical
        4 GiB cap. Precision and progress logging come from the exporter's configuration (``fp16``, ``verbose``).

        An ``fp16=True`` request never silently yields an FP32 engine on a strongly typed TensorRT; see
        :attr:`TensorRTConfig.fp16` for how each TensorRT generation is handled.

        This method never writes the ``<engine>.json`` description; only :meth:`_convert` does, with ``metadata``.
        Called directly with ``metadata`` set, it warns that the setting has no effect here. A description that an
        earlier export wrote beside an engine of the same name is reported with a warning too, because this build
        replaces that engine. It is not deleted.
        With a ``timing_cache`` the build loads the file when it exists and writes the merged timings back to it. Its
        directory is created first, and a path that cannot hold the cache is refused before anything is built. It is
        checked only when a build runs, so ``dry_run=True`` creates nothing.

        Args:
            onnx_path: Path to the source ``.onnx`` file. Its stem (typically the model variant name, e.g.
                ``"rfdetr-medium"``) is reused for the engine filename unless an output name is given.
            dry_run: Log the intended build and return the engine path without building anything (no TensorRT /
                GPU required).
            output_name: Full filename override (without extension), or ``None`` to fall back to the
                configuration's ``output_name``. Takes precedence over the ONNX stem and suppresses the
                ``_fp16``/``_fp32``/``_int8`` suffix — the engine is named ``{output_name}.trt`` verbatim, written
                alongside *onnx_path*. :meth:`_convert` passes the backbone-marked ONNX stem through here.

        Returns:
            Path to the generated ``.trt`` engine file.

        Raises:
            ImportError: If ``polygraphy``/``tensorrt`` are not installed, if ``fp16`` is requested on a
                strongly typed TensorRT without ``onnx``/``onnxconverter-common`` available to cast the graph, if
                ``quantization="int8"`` is requested without onnxruntime or onnxconverter-common, or on a TensorRT
                older than 10, or if ``version_compatible`` is set and TensorRT's lean runtime library cannot be
                loaded.
            Fp16CastUnsupportedError: If ``fp16`` is requested on a strongly typed TensorRT for a graph that
                cannot be cast to fp16 (already fp16, or explicitly quantized).
            ValueError: If the graph's batch axis disagrees with ``dynamic_batch``: a dynamic batch axis without
                ``dynamic_batch`` (the engine would accept batch 1 only), or ``dynamic_batch`` on a graph that has
                none; or if ``hardware_compatibility`` names a level the installed TensorRT does not have; or if
                ``timing_cache`` names a directory or another non-regular file, or the same file as *onnx_path* or
                the engine being written. For ``quantization="int8"``, also if the graph is already quantized or FP16,
                has a dynamic batch axis, cannot be lifted to opset 19, is not an RF-DETR detector export (a
                segmentation, keypoint or backbone-only graph included), or holds attention INT8 cannot be placed
                around; or if the calibration data is missing, unusable, holds a NaN or infinite value, or gives a
                range that does not fit FP16.
            OSError: If ``timing_cache`` cannot be written.

        Examples:
            The build logs its progress, so this is documentation rather than a doctest:

            ```python
            TensorRTExporter(TensorRTConfig()).build_engine("output/rfdetr-medium.onnx", dry_run=True)
            # -> 'output/rfdetr-medium_fp16.trt'
            ```
        """
        name = output_name if output_name is not None else self.config.output_name
        if self.config.metadata:
            # A documented fallback, so a warning rather than a refusal: only _convert has the graph to describe.
            warnings.warn(
                "build_engine writes no engine description: it builds from an .onnx file and has no graph to describe, "
                "so metadata=True has no effect here. RFDETR.export(format='tensorrt', trt_metadata=True) writes one.",
                UserWarning,
                stacklevel=2,
            )

        if dry_run:
            fp16 = self.config.fp16
            engine_path = self._engine_path(onnx_path, fp16_used=fp16, output_name=name)
            logger.info(f"[dry-run] Would build TensorRT {self._precision(fp16)} engine: {onnx_path} -> {engine_path}")
            return engine_path

        return self._build(onnx_path, output_name=name, digest=False).path

    def _build(self, onnx_path: str, *, output_name: str | None, digest: bool) -> _BuiltEngine:
        """Build the engine and report what was built: where it went, its precision and, if asked, its digest.

        Args:
            onnx_path: Path to the source ``.onnx`` file.
            output_name: The engine's file name without extension, or ``None`` to derive it from the ONNX stem and the
                precision (see :meth:`_engine_path`).
            digest: Whether the size and SHA-256 of the serialized engine are wanted, which :meth:`_convert` asks for
                when it writes the description next. Without them, a description an earlier export left beside an
                engine of this name is reported with a warning instead, because this build replaced that engine.

        Returns:
            The engine's path, the precision it was actually built with, and its size and digest when asked for.

        Raises:
            ImportError: If ``polygraphy``/``tensorrt`` are not installed; and as :meth:`build_engine` documents.
            Fp16CastUnsupportedError: As :meth:`build_engine` documents.
            ValueError: As :meth:`build_engine` documents.

        Examples:
            Needs TensorRT and a GPU, so this is documentation rather than a doctest:

            ```python
            TensorRTExporter(TensorRTConfig(fp16=False))._build("output/m.onnx", output_name=None, digest=True)
            # -> _BuiltEngine(path='output/m_fp32.trt', fp16=False, engine_facts={'size': ..., 'sha256': ...})
            ```
        """
        self._require_tensorrt()
        self.check_environment()

        fp16 = self.config.fp16
        if self.config.quantization == INT8:
            # Strongly typed whatever the TensorRT major: precision comes from the quantized FP16 graph.
            strategy, trt_version = Fp16Strategy.CAST_GRAPH, ""
        else:
            strategy, trt_version = self._fp16_strategy() if fp16 else (Fp16Strategy.BUILDER_FLAG, "unknown")

        if strategy is Fp16Strategy.UNAVAILABLE:
            # Lean/partial wheel on a weakly typed TensorRT: the flag is genuinely unavailable and
            # there is no graph-level alternative, so fall back rather than failing the export.
            logger.warning(
                "TensorRT %s does not expose the FP16 builder flag; building an FP32 engine instead. "
                "Pass fp16=False to silence this warning.",
                trt_version,
            )
            fp16 = False

        engine_path = self._engine_path(onnx_path, fp16_used=fp16, output_name=output_name)
        # Only once the engine's final name is known, so a cache that is the ONNX model or the engine itself is refused
        # before anything, the lock file included, is created.
        timing_cache = self._prepare_timing_cache(artifacts=(onnx_path, engine_path))
        serialized = self._compile(
            onnx_path, engine_path, fp16=fp16, strategy=strategy, trt_version=trt_version, timing_cache=timing_cache
        )
        if not digest:
            # The build just replaced any engine of this name; a description written for that one no longer fits.
            self._warn_about_stale_description(engine_path)
            return _BuiltEngine(path=engine_path, fp16=fp16, engine_facts=None)
        # The bytes this build serialized, not the file: another export of the same name writes that file in place
        # and may already have, so a digest read from disk could vouch for the other export's engine.
        return _BuiltEngine(path=engine_path, fp16=fp16, engine_facts=serialized_engine_facts(serialized))
        return engine_path

    def _engine_path(self, onnx_path: str, *, fp16_used: bool, output_name: str | None) -> str:
        """Derive the ``.trt`` path the engine is written to, beside *onnx_path*.

        Args:
            onnx_path: Path to the source ``.onnx`` file, whose directory prefix and stem the engine inherits.
            fp16_used: Whether the float precision being built is FP16; an INT8 request (``_int8``) overrides it.
            output_name: Full filename override (without extension), or ``None`` to derive the name from the ONNX
                stem plus a precision suffix, and a suffix for each portability option that is on.

        Returns:
            Path to the ``.trt`` file the engine is written to.
        """
        if output_name:
            # Delegate output_name sanitize to the shared resolver so the custom-name stem is derived
            # identically to the ONNX/CoreML/ExecuTorch backends (single source of truth for basename +
            # extension stripping); TensorRT still owns its own path prefix and precision suffix below.
            stem = resolve_export_stem(None, output_name)[0]
            # Preserve onnx_path's directory prefix verbatim rather than rebuilding it via
            # os.path.dirname + os.path.join, which inject os.sep (a backslash on Windows) regardless
            # of onnx_path's own separator style and mis-parse a foreign-separator path. The sibling
            # suffix branch below deliberately avoids pathlib/os.path for the same reason.
            sep_idx = max(onnx_path.rfind("/"), onnx_path.rfind("\\"))
            prefix = onnx_path[: sep_idx + 1] if sep_idx != -1 else ""
            return f"{prefix}{stem}.trt"
        # Precision materially changes the engine (fp16 vs fp32 accuracy/speed), so it is always
        # encoded — unless a custom name was requested. Swapping only the final suffix (rather than
        # rebuilding the whole path) keeps any earlier ".onnx"-like segment intact and never aliases
        # the input path; a string-level split (not pathlib) preserves separators verbatim (pathlib
        # rewrites "/" to "\\" on Windows).
        onnx_stem = os.path.splitext(onnx_path)[0]
        return f"{onnx_stem}_{self._precision(fp16_used)}{self._portability_suffix()}.trt"

    def _precision(self, fp16_used: bool) -> str:
        """Name the engine's precision as its file suffix does: ``int8`` for an INT8 request, else ``fp16``/``fp32``.

        Examples:
            >>> TensorRTExporter(TensorRTConfig())._precision(fp16_used=False)
            'fp32'
        """
        return INT8 if self.config.quantization == INT8 else ("fp16" if fp16_used else "fp32")

    def _portability_suffix(self) -> str:
        """Return the detail that names a portable engine, or ``""`` for a default one.

        A portable engine differs from a default one built in the same directory in size and, at FP16, in speed, so it
        must not share its file name. The default name is unchanged: each option that is on adds one detail.
        """
        details: list[str] = []
        if self.config.hardware_compatibility is not None:
            details.append(self.config.hardware_compatibility)
        if self.config.version_compatible:
            details.append("version_compatible")
        return "".join(f"_{detail}" for detail in details)

    @classmethod
    def _require_tensorrt(cls) -> None:
        """Fail early when the ``rfdetr[tensorrt]`` extra is missing.

        The message names only the package that is actually absent: polygraphy installs without ``tensorrt``, so a host
        can be missing either one, and naming both would send the reader looking for an install that is there.

        Raises:
            ImportError: If ``tensorrt`` or ``polygraphy`` is not installed.
        """
        missing = [
            name
            for name, installed in (("tensorrt", _IS_TENSORRT_AVAILABLE), ("polygraphy", _IS_POLYGRAPHY_AVAILABLE))
            if not installed
        ]
        if missing:
            named = " and ".join(f"'{name}'" for name in missing)
            raise ImportError(
                f"TensorRT export requires {named}, which this environment does not have. "
                "Install with: pip install rfdetr[tensorrt]"
            )

    def _require_lean_runtime(self) -> None:
        """Refuse ``version_compatible`` when this TensorRT cannot load its lean runtime library.

        The builder loads the lean runtime for a version-compatible engine, and it is a separate package from
        ``tensorrt`` (``tensorrt-lean-cu*-libs`` on PyPI). Without it TensorRT 10.16 and 11.3 log "Unable to load
        library" and Polygraphy reports only ``Invalid Engine``, so the missing library is named here instead. Nothing
        happens without ``version_compatible``.

        Raises:
            ImportError: If the lean runtime library cannot be loaded.
        """
        if not self.config.version_compatible:
            return
        import tensorrt

        major = _tensorrt_major(tensorrt.__version__)
        if major is None:
            return
        # The pip package loads its libraries when it is imported; an install from a TensorRT archive or system package
        # has none to import and relies on the library path, so a missing package alone proves nothing.
        with contextlib.suppress(ImportError):
            importlib.import_module("tensorrt_lean_libs")
        name = _lean_library_name(major, sys.platform)
        # Windows looks a bare DLL name up in the default directories only, not on PATH, where a TensorRT zip install
        # puts it; resolve it there. A DLL the pip package already loaded is found by its name.
        location = (ctypes.util.find_library(name) or name) if sys.platform == "win32" else name
        try:
            # Loaded into the process so the builder finds it by name when it asks for the lean runtime.
            ctypes.CDLL(location)
        except OSError as error:
            raise ImportError(
                "trt_version_compatible=True needs TensorRT's lean runtime library, and "
                f"TensorRT {tensorrt.__version__} could not load {name}. Install the lean runtime that matches it: the "
                "`tensorrt-lean-cu*-libs` wheel with the same CUDA suffix and version as your `tensorrt-cu*-libs` "
                "wheel, or the lean library from the TensorRT archive or system package."
            ) from error

    def _fp16_strategy(self) -> tuple[Fp16Strategy, str]:
        """Resolve how the installed TensorRT can produce the requested FP16 engine.

        Returns:
            The strategy from :func:`resolve_fp16_strategy`, paired with the version TensorRT reports. A missing
            ``tensorrt`` was already refused by :meth:`_require_tensorrt`; one that is installed but fails to import
            is left to the polygraphy build chain to surface.
        """
        try:
            import tensorrt as trt_module
        except ImportError:
            trt_module = None
        return resolve_fp16_strategy(trt_module)

    def _compile(
        self,
        onnx_path: str,
        engine_path: str,
        *,
        fp16: bool,
        strategy: Fp16Strategy,
        trt_version: str,
        timing_cache: str | None,
    ) -> Any:
        """Build the engine through polygraphy and serialize it to *engine_path*.

        Args:
            onnx_path: Path to the source ``.onnx`` file.
            engine_path: Path the serialized engine is written to.
            fp16: The precision the engine is built with, after the FP16 availability probe.
            strategy: How FP16 is obtained from the installed TensorRT (see :func:`resolve_fp16_strategy`).
            trt_version: The version TensorRT reports, for logging.
            timing_cache: The resolved cache path from :meth:`_prepare_timing_cache`, loaded and saved by the build,
                or ``None`` without a cache.

        Returns:
            The serialized engine: the bytes written to *engine_path*, which :meth:`_build` hashes when asked for a
            digest.
        """
        # The precision the engine ends up with and the flag handed to the builder are not the same thing
        # under strong typing: TensorRT >= 11 has no FP16 flag, and reads precision off the graph instead.
        builder_fp16 = fp16

        with contextlib.ExitStack() as cleanup:
            # Only the builder reads the cast intermediate; onnx_path keeps naming the caller's own model.
            build_source = onnx_path

            int8 = self.config.quantization == INT8
            if int8:
                # The quantized graph carries its own precision: INT8 where it holds Q/DQ pairs, FP16 elsewhere.
                assert self.config.calibration_data is not None, "_check_quantization requires it for int8"
                build_source = cleanup.enter_context(
                    int8_source_graph(
                        onnx_path,
                        calibration_data=self.config.calibration_data,
                        max_images=self.config.max_images,
                        dynamic_batch=self.config.dynamic_batch,
                    )
                )
                builder_fp16 = False
                logger.info("Building the INT8 engine from an explicitly quantized FP16 graph")
            elif strategy is Fp16Strategy.CAST_GRAPH:
                # Strongly typed: precision comes from the graph, so cast it and let the builder infer.
                # Raises rather than quietly downgrading -- an FP32 engine returned for an FP16 request
                # is reported as an FP16 latency by anyone benchmarking it. The batch request goes along so the
                # cast refuses a graph that contradicts it before converting every weight into a second copy of
                # the model; _build_config still refuses the parsed network for callers that arrive another way.
                build_source = cleanup.enter_context(
                    fp16_source_graph(onnx_path, dynamic_batch=self.config.dynamic_batch)
                )
                builder_fp16 = False
                logger.info(f"TensorRT {trt_version} is strongly typed; building the FP16 engine from a cast graph")
                logger.debug(f"fp16 cast graph: {build_source}")

            if self.config.verbose:
                logger.info(f"Building TensorRT {self._precision(fp16)} engine from {onnx_path}")

            # The builder configuration depends on the network's input shapes, so the parsed (builder, network, parser)
            # tuple is inspected first and then handed on, rather than letting engine_from_network parse it again.
            # Only the INT8 path asks for a strongly typed network explicitly: TensorRT 10 is weakly typed by default.
            parsed = (
                network_from_onnx_path(build_source, strongly_typed=True)
                if int8
                else network_from_onnx_path(build_source)
            )
            try:
                build_config = self._build_config(parsed[1], onnx_path, fp16=builder_fp16, timing_cache=timing_cache)
            except Exception:
                # _build_config refuses a graph whose batch axis disagrees with the request before engine_from_network
                # ever takes ownership of `parsed`. Only that call frees the parsed builder/network/parser on success,
                # so release them here explicitly rather than leaking them on this error path; adding `parsed` to
                # `cleanup` unconditionally would double-close it once engine_from_network also releases it. This
                # frees them only because no frame the traceback keeps alive still names the network -- _build_config
                # drops its own parameter before it can raise.
                del parsed
                raise
            # Only handed over when configured, so the default build calls Polygraphy exactly as it always did.
            build_options = {} if timing_cache is None else {"save_timing_cache": timing_cache}
            engine = engine_from_network(parsed, config=build_config, **build_options)
            # Serialized once: the same bytes are written here and hashed for a description. save_file opens the path
            # for writing in place, as polygraphy's save_engine does, so an existing engine keeps its owner and mode.
            serialized = engine.serialize()
            save_file(contents=serialized, dest=engine_path, description="engine")

        logger.info(f"Successfully built TensorRT engine: {engine_path}")
        return serialized

    def _build_config(self, network: Any, onnx_path: str, *, fp16: bool, timing_cache: str | None) -> Any:
        """Create the builder configuration, with a batch profile when ``dynamic_batch`` is set and none otherwise.

        Without a profile, polygraphy fixes every dynamic dimension to 1 and only warns, so a graph with a dynamic
        batch axis is refused here rather than quietly built into an engine that accepts batch 1 only.

        Args:
            network: The parsed TensorRT network. Read once, for its input shapes, and released before any refusal
                below — see the comment on that release.
            onnx_path: The caller's ``.onnx`` file, named in errors even when *network* was parsed from an fp16 copy.
            fp16: The FP16 builder flag.
            timing_cache: The resolved cache path from :meth:`_prepare_timing_cache`, or ``None`` without a cache.

        Returns:
            A polygraphy ``CreateConfig``.

        Raises:
            ValueError: If the graph's batch axis disagrees with ``dynamic_batch``, in either direction.
        """
        dynamic_inputs = _dynamic_batch_inputs(network)
        # Every refusal below (and every one _batch_profile raises) travels back to _compile's error path, where
        # `del parsed` can only free the parsed builder/network/parser if no frame the traceback keeps alive still
        # names them. This frame is one of those, and one scan of the input shapes is all it -- or the profile --
        # needs, so the parameter goes now rather than pinning TensorRT resources until the caller drops the error.
        del network
        options = {**self._portability_options(), **self._timing_cache_options(timing_cache)}
        if self.config.dynamic_batch:
            return CreateConfig(fp16=fp16, profiles=[self._batch_profile(onnx_path, dynamic_inputs)], **options)
        # The fp16 cast path refuses this from the ONNX file, before it writes anything; this is the last-resort
        # guard for the builds that never cast -- a weakly typed TensorRT, or an FP32 request.
        _reject_dynamic_graph_under_static_request(onnx_path, dynamic_inputs)
        return CreateConfig(fp16=fp16, **options)

    def _portability_options(self) -> dict[str, Any]:
        """Return the ``CreateConfig`` keywords for the portability settings that are switched on.

        Returns:
            Only the keywords the configuration asks for, so the default build passes none of them.

        Raises:
            ValueError: If the installed TensorRT has no hardware compatibility level of the requested name.
        """
        options: dict[str, Any] = {}
        level = self.config.hardware_compatibility
        if level is not None:
            options["hardware_compatibility_level"] = self._hardware_compatibility_level(level)
        if self.config.version_compatible:
            options["version_compatible"] = True
        return options

    @staticmethod
    def _warn_if_version_compatibility_is_unverified() -> None:
        """Warn when the installed TensorRT is one on which version compatibility was not seen to work.

        NVIDIA documents the direction as forward only: an engine built by an older release of a major version loads on
        the same or a newer release of it. That was seen between TensorRT 11 releases. On TensorRT 10.16 the engine had
        the size of a default one and loaded neither on 10.13 (the older, unsupported direction) nor on 11.3, so the
        request is honoured but its effect is not something to rely on. Attributed to the line that calls this helper,
        so the warning shows once even though :meth:`check_environment` runs up to three times per export.
        """
        import tensorrt as trt

        major = _tensorrt_major(trt.__version__)
        if major is not None and major < 11:
            warnings.warn(
                f"trt_version_compatible has only been verified between TensorRT 11 releases. NVIDIA supports loading "
                f"a version-compatible engine on the same or a newer release of the major version that built it; an "
                f"engine built by TensorRT {trt.__version__} may not load even there, so test your pair of releases "
                "before you rely on it.",
                UserWarning,
                stacklevel=2,
            )

    @staticmethod
    def _require_ampere_or_newer_gpu() -> None:
        """Refuse ``"ampere_plus"`` on a GPU older than Ampere, which TensorRT cannot build that level on.

        TensorRT builds on the current CUDA device, so that is the one checked. A host without a visible CUDA device is
        let through: there is nothing to compare, and the build reports the missing device itself.

        Raises:
            ValueError: If the current CUDA device has a compute capability below 8.0.
        """
        if not torch.cuda.is_available():
            return
        capability = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
        if capability < _AMPERE_COMPUTE_CAPABILITY:
            raise ValueError(
                "trt_hardware_compatibility='ampere_plus' must be built on an NVIDIA Ampere or newer GPU (compute "
                f"capability 8.0 or higher), but the current CUDA device has compute capability "
                f"{capability[0]}.{capability[1]}. Build on an Ampere or newer GPU, or pick "
                "'same_compute_capability'."
            )

    @staticmethod
    def _hardware_compatibility_level(level: str) -> Any:
        """Look up the ``tensorrt.HardwareCompatibilityLevel`` member named by *level* in the installed TensorRT.

        Args:
            level: One of :data:`_HARDWARE_COMPATIBILITY_LEVELS`.

        Returns:
            The enum member.

        Raises:
            ValueError: If this TensorRT has no such level. ``AMPERE_PLUS`` arrived with TensorRT 8.6, and
                ``SAME_COMPUTE_CAPABILITY`` later than that, so each is detected on its own.
        """
        import tensorrt as trt

        member = getattr(getattr(trt, "HardwareCompatibilityLevel", None), level.upper(), None)
        if member is None:
            raise ValueError(
                f"trt_hardware_compatibility={level!r} is not available in TensorRT {trt.__version__}; "
                "upgrade TensorRT or pick another level."
            )
        return member

    def _timing_cache_path(self) -> str | None:
        """Return the configured timing-cache file as a ``str`` with ``~`` expanded, or ``None`` when there is none."""
        cache = self.config.timing_cache
        return None if cache is None else os.path.expanduser(os.fspath(cache))

    def _prepare_timing_cache(self, *, artifacts: tuple[str, ...] = ()) -> str | None:
        """Make sure the timing cache can be written, before the export pays for an engine build.

        Polygraphy saves the cache only after the engine is built, and it opens ``<file>.lock`` before it creates a
        missing directory, so a bad location would cost the whole build and the engine with it. The directory is
        created here, and the files Polygraphy will open (the cache when it exists, then the lock) are opened for
        writing now, so an inaccessible location fails before the build, whatever the operating system and the user. A
        symbolic link to a file that does not exist is refused: Polygraphy would create the file at the link's target,
        and that cannot be checked without creating it.

        The path is resolved through symbolic links once, here, and the build hands that resolved path to Polygraphy
        for both loading and saving. Polygraphy locks ``<path>.lock`` beside whatever path it is given, so a link and
        its target would otherwise lock two different files and let two exports race the same cache's
        read-merge-write. Error messages keep naming the path as it was configured.

        Args:
            artifacts: Files the build reads or writes besides the cache (the ONNX model and the engine). A cache that
                is one of them would overwrite it, or be overwritten by it, so it is refused before anything is created.

        Returns:
            The resolved cache path the build loads, saves and locks, or ``None`` when no cache is configured.

        Raises:
            ValueError: If the configured path is a directory, or another existing file that is not a regular one (a
                FIFO or a device), or the same file as one of *artifacts*.
            OSError: If the path is a symbolic link to a missing file, if its directory cannot be created, or if the
                cache or its lock file cannot be opened for writing. A failure that carries an errno keeps its
                subclass (``PermissionError``, ``FileExistsError``, ...).
        """
        path = self._timing_cache_path()
        if path is None:
            return None
        if os.path.isdir(path):
            raise ValueError(f"trt_timing_cache must name a file, but {path!r} is a directory.")
        if os.path.islink(path) and not os.path.exists(path):
            raise OSError(
                f"trt_timing_cache {path!r} is a symbolic link to {os.readlink(path)!r}, which does not exist. Create "
                "that file first, or pass the path it should have."
            )
        if os.path.exists(path) and not os.path.isfile(path):
            # A FIFO or a device is not a file the cache can be saved into: with no reader, Polygraphy's save into a
            # FIFO would block once the whole build is done.
            raise ValueError(f"trt_timing_cache must name a regular file, but {path!r} is not one.")
        # Resolved only after the link check above: a resolved path is never itself a link, so resolving first would
        # silently disable that check.
        resolved = os.path.realpath(path)
        clash = next((artifact for artifact in artifacts if _is_same_file(resolved, artifact)), None)
        if clash is not None:
            raise ValueError(
                f"trt_timing_cache {path!r} is the same file as {clash!r}, which this build also reads or writes. Pass "
                "a separate file for the timing cache."
            )
        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            # The cache before the lock, so that refusing a read-only cache leaves no new lock file behind.
            if os.path.isfile(resolved):
                with open(resolved, "r+b"):
                    pass
            with open(f"{resolved}.lock", "ab"):
                pass
        except OSError as error:
            message = f"trt_timing_cache {path!r} cannot be written"
            if error.errno is None:
                raise OSError(f"{message}: {error}") from error
            # Rebuilt from the errno, so the subclass a caller may catch (PermissionError, ...) survives the context.
            raise OSError(error.errno, f"{message}: {error.strerror}", error.filename) from error
        return resolved

    @staticmethod
    def _timing_cache_options(path: str | None) -> dict[str, str]:
        """Return the ``CreateConfig`` keyword that seeds the build with the timing cache, when there is one to load.

        Empty without a configured cache, and also while the file does not exist yet: a first use is expected to
        find nothing, and asking Polygraphy to load it would only make it warn about a missing cache.

        Args:
            path: The resolved cache path from :meth:`_prepare_timing_cache`, or ``None`` without a cache.

        Returns:
            ``{"load_timing_cache": path}`` for an existing cache file, otherwise ``{}``.
        """
        if path is None:
            return {}
        if not os.path.isfile(path):
            logger.info(f"TensorRT timing cache {path} does not exist yet; this build creates it")
            return {}
        logger.info(f"Reusing the TensorRT timing cache {path}")
        return {"load_timing_cache": path}

    def _batch_profile(self, onnx_path: str, dynamic_inputs: Mapping[str, tuple[int, ...]]) -> Any:
        """Build the batch optimization profile for every dynamic input.

        The profile spans batch 1 to ``max_batch_size`` and is tuned for ``opt_batch_size``; the spatial dimensions
        stay fixed at what the graph was traced at.

        Takes the already-scanned shapes rather than the network itself: shapes are all a profile needs, and a
        parameter naming the network would keep it alive in this frame's traceback when the refusal below fires,
        defeating :meth:`_compile`'s release of the parsed resources.

        Args:
            onnx_path: The caller's ``.onnx`` file, quoted in the refusal below so it opens the same way the static
                request's refusal does — naming the graph the reader has to fix.
            dynamic_inputs: :func:`_dynamic_batch_inputs` of the parsed network — each dynamic-batch input's name
                mapped to its full shape, empty when the graph was exported without a dynamic batch axis.

        Returns:
            A polygraphy ``Profile`` with one entry per dynamic-batch input.

        Raises:
            ValueError: If no input carries a dynamic batch axis, which means the ONNX graph was exported without
                ``dynamic_batch`` and a profile would be meaningless.
        """
        opt = self.config.opt_batch_size
        max_batch = self.config.max_batch_size
        if not dynamic_inputs:
            raise ValueError(
                f"'{onnx_path}' has no input with a dynamic batch axis, but dynamic_batch=True was requested; export "
                "the ONNX graph with dynamic_batch=True first."
            )
        profile = Profile()
        for name, shape in dynamic_inputs.items():
            # BATCH_AXIS is the only dynamic axis, so everything past it is the fixed shape each bound repeats.
            trailing = shape[BATCH_AXIS + 1 :]
            profile.add(name, min=(1, *trailing), opt=(opt, *trailing), max=(max_batch, *trailing))
        logger.info(
            f"Building TensorRT engine with a batch profile min=1 opt={opt} max={max_batch} on {list(dynamic_inputs)}"
        )
        return profile
