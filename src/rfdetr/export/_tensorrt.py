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
import importlib.util
import os
import tempfile
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from rfdetr.export._naming import resolve_export_stem
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# polygraphy ships in the ``rfdetr[tensorrt]`` extra alongside ``tensorrt``. Import it
# lazily at module scope (guarded) so importing this module never fails on hosts
# without TensorRT, and so tests can monkeypatch these names without polygraphy
# installed.
try:
    from polygraphy.backend.trt import (
        CreateConfig,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )

    _IS_TENSORRT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via the guard in build_engine
    CreateConfig = None
    engine_from_network = None
    network_from_onnx_path = None
    save_engine = None

    _IS_TENSORRT_AVAILABLE = False

# TensorRT 11 removed weak typing: ``BuilderFlag.FP16`` no longer exists and engine precision is
# taken from the ONNX graph's dtypes. Building FP16 there means casting the graph first, which needs
# ``onnx`` + ``onnxconverter-common`` (both in the ``rfdetr[tensorrt]`` extra). Only availability is
# resolved here; the modules themselves are imported inside the functions that use them, matching how
# ``export/_onnx/exporter.py`` handles the same optional dependency.
_IS_FP16_CASTER_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxconverter_common"))

# TensorRT majors at or above this are strongly typed, so an absent FP16 builder flag is by design
# rather than a sign of a lean/partial wheel.
_STRONG_TYPING_MAJOR = 11

# An explicitly quantized graph carries its precision in these nodes and their scale/zero-point
# tensors, so a blanket fp16 cast contradicts it rather than converting it.
_QUANTIZATION_OP_TYPES = frozenset({"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"})


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


def _cast_onnx_to_fp16(onnx_path: str) -> str:
    """Write an fp16 copy of an ONNX model next to it, keeping FP32 graph inputs and outputs.

    The file is a build intermediate, not a deliverable: ``build_engine`` deletes it afterwards.
    Its name is unique rather than derived from *onnx_path*, so a build never overwrites -- and then
    deletes -- a same-named file it did not create, and concurrent builds from one source model
    cannot claim each other's graph. It is written beside the source model rather than under
    ``/tmp`` because it is the same order of size as the model and ``/tmp`` is often a tmpfs.

    Args:
        onnx_path: Path to the float32 ``.onnx`` model.

    Returns:
        Path to the newly written fp16 model.

    Raises:
        ImportError: If ``onnx``/``onnxconverter-common`` are not installed.
        Fp16CastUnsupportedError: If the graph cannot be cast to fp16 — it is explicitly quantized, or
            the converter rejects it (most often because the model already is fp16).

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
def fp16_source_graph(onnx_path: str) -> Iterator[str]:
    """Provide an fp16 copy of *onnx_path* to build from, deleting it when the block exits.

    The copy is a build intermediate, so it goes whether the build succeeds or fails. Removing it is
    best effort on purpose: the file may already be gone, or still be held open by the parser (Windows
    raises ``PermissionError`` then), and neither may replace the build's own exception.

    Args:
        onnx_path: Path to the float32 ``.onnx`` model to build from.

    Yields:
        Path to the fp16 copy, valid only inside the ``with`` block.

    Raises:
        ImportError: If ``onnx``/``onnxconverter-common`` are not installed.
        Fp16CastUnsupportedError: If the graph cannot be cast to fp16.

    Examples:
        >>> with fp16_source_graph("output/rfdetr-medium.onnx") as fp16_path:  # doctest: +SKIP
        ...     engine_from_network(network_from_onnx_path(fp16_path))
    """
    cast_path = _cast_onnx_to_fp16(onnx_path)
    try:
        yield cast_path
    finally:
        with contextlib.suppress(OSError):
            Path(cast_path).unlink(missing_ok=True)


def build_engine(
    onnx_path: str,
    *,
    fp16: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
    output_name: str | None = None,
) -> str:
    """Build a serialized TensorRT engine from an ONNX model, in-process.

    Uses the TensorRT Python API through ``polygraphy`` — no ``trtexec`` subprocess.
    Workspace size is left to the TensorRT default (it auto-sizes to the available
    device memory), which meets or exceeds the historical 4 GiB cap.

    An ``fp16=True`` request never silently yields an FP32 engine on a strongly typed TensorRT; see
    *fp16* below for how each TensorRT generation is handled.

    Args:
        onnx_path: Path to the source ``.onnx`` file. Its stem (typically the model variant name,
            e.g. ``"rfdetr-medium"``) is reused for the engine filename unless *output_name* is given.
        fp16: Enable FP16 precision when building the engine. How this is achieved depends on the
            installed TensorRT: weakly typed builds (TensorRT < 11) set the FP16 builder flag, while
            strongly typed ones (TensorRT >= 11, which removed that flag) get an FP16 engine by casting
            the ONNX graph to FP16 first — the engine's own inputs and outputs stay FP32 either way.
            Only downgraded to FP32 (with a warning) on a lean/partial TensorRT < 11 wheel that does not
            expose the flag, where no graph-level alternative exists; the engine filename then reflects
            the precision actually built (except under *dry_run*, where nothing is built or probed, so
            the requested value is used as-is).
        verbose: Emit extra progress logging.
        dry_run: Log the intended build and return the engine path without
            building anything (no TensorRT / GPU required).
        output_name: Full filename override (without extension). Takes precedence over the ONNX
            stem and suppresses the ``_fp16``/``_fp32`` suffix — the engine is named
            ``{output_name}.trt`` verbatim, written alongside *onnx_path*.

    Returns:
        Path to the generated ``.trt`` engine file.

    Raises:
        ImportError: If ``polygraphy``/``tensorrt`` are not installed, or if *fp16* is requested on a
            strongly typed TensorRT without ``onnx``/``onnxconverter-common`` available to cast the graph.
        Fp16CastUnsupportedError: If *fp16* is requested on a strongly typed TensorRT for a graph that
            cannot be cast to fp16 (already fp16, or explicitly quantized).

    Examples:
        >>> build_engine("output/rfdetr-medium.onnx", dry_run=True)  # doctest: +SKIP
        'output/rfdetr-medium_fp16.trt'
    """
    onnx_stem = os.path.splitext(onnx_path)[0]

    def _engine_path(*, fp16_used: bool) -> str:
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
        return f"{onnx_stem}_{'fp16' if fp16_used else 'fp32'}.trt"

    engine_path = _engine_path(fp16_used=fp16)

    if dry_run:
        logger.info(f"[dry-run] Would build TensorRT engine (fp16={fp16}): {onnx_path} -> {engine_path}")
        return engine_path

    if engine_from_network is None:
        raise ImportError("TensorRT export requires the 'tensorrt' extra. Install with: pip install rfdetr[tensorrt]")

    # The precision the engine ends up with and the flag handed to the builder are not the same thing
    # under strong typing: TensorRT >= 11 has no FP16 flag, and reads precision off the graph instead.
    builder_fp16 = fp16
    strategy, trt_version = Fp16Strategy.BUILDER_FLAG, "unknown"
    if fp16:
        try:
            import tensorrt as trt_module
        except ImportError:
            trt_module = None
        strategy, trt_version = resolve_fp16_strategy(trt_module)

    with contextlib.ExitStack() as cleanup:
        # Only the builder reads the cast intermediate; onnx_path keeps naming the caller's own model.
        build_source = onnx_path

        if strategy is Fp16Strategy.CAST_GRAPH:
            # Strongly typed: precision comes from the graph, so cast it and let the builder infer.
            # Raises rather than quietly downgrading -- an FP32 engine returned for an FP16 request
            # is reported as an FP16 latency by anyone benchmarking it.
            build_source = cleanup.enter_context(fp16_source_graph(onnx_path))
            builder_fp16 = False
            logger.info(f"TensorRT {trt_version} is strongly typed; building the FP16 engine from a cast graph")
            logger.debug(f"fp16 cast graph: {build_source}")
        elif strategy is Fp16Strategy.UNAVAILABLE:
            # Lean/partial wheel on a weakly typed TensorRT: the flag is genuinely unavailable and
            # there is no graph-level alternative, so fall back rather than failing the export.
            logger.warning(
                "TensorRT %s does not expose the FP16 builder flag; building an FP32 engine instead. "
                "Pass fp16=False to silence this warning.",
                trt_version,
            )
            fp16 = False
            builder_fp16 = False
            engine_path = _engine_path(fp16_used=fp16)

        if verbose:
            logger.info(f"Building TensorRT engine (fp16={fp16}) from {onnx_path}")

        engine = engine_from_network(
            network_from_onnx_path(build_source),
            config=CreateConfig(fp16=builder_fp16),
        )
        save_engine(engine, path=engine_path)

    logger.info(f"Successfully built TensorRT engine: {engine_path}")
    return engine_path
