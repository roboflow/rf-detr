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

import importlib.util
import os
import tempfile
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


def _retarget_float_casts(graph: Any) -> int:
    """Point pre-existing ``Cast(to=FLOAT)`` nodes at FLOAT16 after a graph-wide fp16 conversion.

    ``onnxconverter-common`` relabels tensors but leaves the ``to`` attribute of ``Cast`` nodes that
    were already in the source graph untouched. RF-DETR exports 33-35 such nodes, so the tensor stays
    float32 while its ``value_info`` claims float16 and TensorRT's strongly-typed parser rejects the
    graph at the first convolution.

    Args:
        graph: Graph of an already-converted fp16 model, mutated in place.

    Returns:
        Number of ``Cast`` nodes retargeted.

    Examples:
        Needs a converted fp16 ``ModelProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _retarget_float_casts(model.graph)  # doctest: +SKIP
        33
    """
    from onnx import TensorProto

    declared = {value.name: value.type.tensor_type.elem_type for value in list(graph.value_info) + list(graph.output)}
    retargeted = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if (
                attribute.name == "to"
                and attribute.i == TensorProto.FLOAT
                and declared.get(node.output[0]) == TensorProto.FLOAT16
            ):
                attribute.i = TensorProto.FLOAT16
                retargeted += 1
    return retargeted


def _restore_fp32_io(graph: Any) -> int:
    """Re-establish FP32 graph inputs/outputs around an fp16 body by inserting boundary casts.

    A weakly-typed TensorRT FP16 engine keeps its I/O tensors FP32, so callers feed and read float32.
    Preserving that contract keeps the strongly-typed path a drop-in replacement. This is done here
    rather than via ``convert_float_to_float16(keep_io_types=True)`` because that option wires the
    FP32 graph input straight into an FP16 convolution without inserting a ``Cast``, which TensorRT
    rejects.

    Args:
        graph: Graph of an already-converted fp16 model, mutated in place.

    Returns:
        Number of boundary ``Cast`` nodes inserted.

    Examples:
        Needs a converted fp16 ``ModelProto``; see ``TestCastOnnxToFp16`` for real invocations.

        >>> _restore_fp32_io(model.graph)  # doctest: +SKIP
        3
    """
    from onnx import TensorProto, helper

    inserted = 0

    for tensor in graph.input:
        if tensor.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        inner = f"{tensor.name}_fp16"
        for node in graph.node:
            for index, name in enumerate(node.input):
                if name == tensor.name:
                    node.input[index] = inner
        graph.node.insert(
            0, helper.make_node("Cast", [tensor.name], [inner], to=TensorProto.FLOAT16, name=f"Cast_{inner}_in")
        )
        tensor.type.tensor_type.elem_type = TensorProto.FLOAT
        inserted += 1

    for tensor in graph.output:
        if tensor.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        inner = f"{tensor.name}_fp16"
        for node in graph.node:
            for index, name in enumerate(node.output):
                if name == tensor.name:
                    node.output[index] = inner
        graph.node.append(
            helper.make_node("Cast", [inner], [tensor.name], to=TensorProto.FLOAT, name=f"Cast_{inner}_out")
        )
        tensor.type.tensor_type.elem_type = TensorProto.FLOAT
        inserted += 1

    # The boundary tensors are FP32 again, but the conversion left value_info entries still declaring
    # them FLOAT16. graph.input/graph.output already carry the authoritative type, so drop the
    # contradicting duplicates rather than trying to correct them.
    stale = {t.name for t in list(graph.input) + list(graph.output)}
    keep = [value for value in graph.value_info if value.name not in stale]
    del graph.value_info[:]
    graph.value_info.extend(keep)

    return inserted


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

    model = float16.convert_float_to_float16(onnx.load(onnx_path), keep_io_types=False)
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
    # Set only on the strongly typed path: the cast copy is ours to delete once the engine exists.
    cast_onnx_path: str | None = None

    if fp16:
        # Two different situations present identically as a missing FP16 builder flag, and they need
        # opposite handling, so disambiguate on the TensorRT major version rather than the flag alone.
        try:
            import tensorrt as trt

            trt_version = getattr(trt, "__version__", "unknown")
            has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")
        except ImportError:
            trt_version = "unknown"
            has_fp16_flag = True  # a missing/broken tensorrt import is surfaced by the build chain below

        if not has_fp16_flag:
            major = _tensorrt_major(trt_version)
            if major is not None and major >= _STRONG_TYPING_MAJOR:
                # Strongly typed: precision comes from the graph, so cast it and let the builder infer.
                # Raises rather than quietly downgrading -- an FP32 engine returned for an FP16 request
                # is reported as an FP16 latency by anyone benchmarking it.
                onnx_path = cast_onnx_path = _cast_onnx_to_fp16(onnx_path)
                builder_fp16 = False
                logger.info(f"TensorRT {trt_version} is strongly typed; building the FP16 engine from a cast graph")
                logger.debug(f"fp16 cast graph: {onnx_path}")
            else:
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

    try:
        engine = engine_from_network(
            network_from_onnx_path(onnx_path),
            config=CreateConfig(fp16=builder_fp16),
        )
        save_engine(engine, path=engine_path)
    finally:
        # Runs on failure too: a failed build should not leave the cast graph behind either.
        if cast_onnx_path is not None and os.path.exists(cast_onnx_path):
            os.remove(cast_onnx_path)

    logger.info(f"Successfully built TensorRT engine: {engine_path}")
    return engine_path
