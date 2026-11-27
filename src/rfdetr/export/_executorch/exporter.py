# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PyTorch -> ExecuTorch (``.pte``) conversion via ``torch.export``.

ExecuTorch is PyTorch's on-device runtime for mobile and edge hardware.  Unlike the TFLite path (which routes through
ONNX), ExecuTorch consumes a :func:`torch.export.export` graph directly, lowers it to the Edge dialect, and delegates
supported subgraphs to a hardware backend:

* **XNNPACK** (default) -- the portable CPU backend.  Works on every platform ExecuTorch supports (Android, iOS, Linux,
  macOS), needs no extra SDK, runs in fp32; validated end-to-end against eager.
* **CoreML** -- Apple's Neural Engine / GPU / CPU backend for iOS and macOS.  Requires ``coremltools`` at export time
  and an Apple device to run.  Defaults to fp16, so raw outputs differ noticeably from eager PyTorch while detections
  remain correct -- typical of CoreML exports (see Ultralytics YOLO CoreML, same behaviour); validate by detections,
  not tensor parity.  Export mode reshapes the deformable-attention sampling tensors to rank 5 so they satisfy CoreML's
  ``rank <= 5`` limit (the rank-5 form is numerically identical -- XNNPACK fp32 matches eager to ~1e-5).

* **Qualcomm QNN** -- the Snapdragon HTP/NPU backend (fp16).  Lowers cleanly and delegates the bulk of the network
  (including the deformable-attention ``grid_sample``) to the HTP; the two-stage selection ops (``topk``/``max.dim``)
  and a couple of attention-mask ops run on CPU (the HTP computes wrong indices for ``topk``/``max.dim`` in fp16 --
  see :data:`_QNN_CPU_FALLBACK_OPS`).  Its delegate is not in the ``executorch`` wheel: it requires an ExecuTorch
  source build against the Qualcomm AI Engine Direct (QAIRT) SDK, and runs only on a Snapdragon device.  Uses a
  dedicated lowering path (:func:`_lower_qnn`) rather than the generic partitioner.
* **TensorRT** -- NVIDIA GPUs, through Torch-TensorRT.  For the models checked (detection Nano to Large,
  segmentation Nano, keypoint preview) TensorRT converts every op, so the ``.pte`` holds one TensorRT engine
  (fp32) and is called with CPU tensors, like the other backends.  The engine is built on the exporting GPU and
  runs only on GPUs and TensorRT releases it is compatible with.  Requires the ``executorch-tensorrt`` extra and a
  CUDA GPU at export time.  Uses a dedicated lowering path (:func:`_lower_tensorrt`).

Vulkan is not exposed. It was evaluated end-to-end and rejected because it fails at three independent levels:

1. It does not lower. Two ExecuTorch bugs block it -- a partitioner crash on the deformable-attention multi-output
   ``split`` (avoidable by rewriting our own model code), and an unavoidable one: the conv+BatchNorm fusion pass
   crashes on *any* CNN with BatchNorm (reproducible in a 3-line model), which we could only sidestep by invasively
   pre-folding BatchNorm in a way that changes every other backend's graph.
2. It would not accelerate anything even if it lowered. The decoder's core ops (``grid_sample`` -- the heart of
   deformable attention -- plus ``topk``/``max``/masking) have no Vulkan kernels, so they fall back to CPU and the
   program fragments into ~37 CPU<->GPU subgraphs with the whole transformer left on CPU: no meaningful speedup.
3. The file it produces crashes anyway. Even past both bugs, the lowered ``.pte`` crashes at load inside the Vulkan
   backend's constant prepack (a convolution weight is serialized as a runtime tensor instead of a ``TensorRef``) --
   reproduced on the from-source ExecuTorch v1.3.1 runtime, so it never runs on any Vulkan device.

XNNPACK covers portable CPU; QNN and CoreML cover the mobile accelerators; TensorRT covers NVIDIA GPUs.

RF-DETR's detection graph exports cleanly once the deformable-attention modules are switched into their export-friendly
path (handled by :meth:`model.export`, which the caller invokes before reaching this exporter).  This module only
performs the conversion; numerical parity against eager PyTorch is checked by the test suite (mirroring the ONNX and
TFLite exporters, which likewise convert without an in-library validation step).

Note:
    The produced ``.pte`` expects the same input normalization as the ONNX export: ImageNet mean/std
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``).  The caller is responsible for applying this
    normalization at inference time.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from rfdetr.export._executorch.xnnpack import decompose_attention, fold_constants
from rfdetr.export._naming import append_backbone_marker, resolve_export_stem
from rfdetr.export._runtime.metadata import ExportMetadata, positional_metadata
from rfdetr.export.base import ExportConfig, Exporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import is_installed

logger = get_logger()

# Backends accepted by :class:`ExecuTorchExporter`.  XNNPACK (portable CPU, fp32) and CoreML (Apple devices, fp16) are
# validated end-to-end.  Qualcomm QNN (Snapdragon HTP/NPU, fp16) lowers cleanly and delegates
# the bulk of the network to the HTP -- but its delegate ships neither in the ``executorch`` wheel nor runs off a
# Snapdragon device, so it requires a source build of ExecuTorch against the Qualcomm AI Engine Direct SDK.  TensorRT
# (NVIDIA GPUs, fp32) comes from Torch-TensorRT, a separate package with its own extra.  Vulkan is omitted (blocked by
# two upstream ExecuTorch bugs + missing ops; see triage notes).
_VALID_BACKENDS: frozenset[str] = frozenset({"xnnpack", "coreml", "qnn", "tensorrt"})

# Backends whose ahead-of-time compilation bakes a target SoC into the ``.pte`` (see :func:`_lower_qnn`), so a SoC
# must be supplied at export time.  XNNPACK (portable CPU) and CoreML (Apple's runtime selects the device at load)
# need none.  Public :meth:`rfdetr.detr.RFDETR.export` consults this set (via
# :func:`rfdetr.export._backend._resolve_export_backend`) to decide whether its ``soc`` argument is required for the
# requested ``backend``.
_SOC_BACKENDS: frozenset[str] = frozenset({"qnn"})

# Public aliases so detr.py can import these without crossing a private-name boundary.
VALID_BACKENDS: frozenset[str] = _VALID_BACKENDS
SOC_BACKENDS: frozenset[str] = _SOC_BACKENDS

# SoC assumed when a SoC-locked backend is exported without one.  ``_resolve_export_backend`` requires the public
# :meth:`rfdetr.detr.RFDETR.export` caller to pass ``soc`` for those backends, so this only backs a direct
# construction of the exporter.  "SM8650" is Snapdragon 8 Gen 3.
_DEFAULT_SOC = "SM8650"

_INSTALL_HINT = (
    "ExecuTorch export requires the `executorch` package. Install it with: pip install rfdetr[executorch] "
    "(or, on Linux with Python 3.11 or newer, rfdetr[executorch-tensorrt] for backend='tensorrt')"
)
_COREML_HINT = "CoreML export requires `coremltools`. Install it with: pip install coremltools"
_TENSORRT_HINT = (
    "ExecuTorch TensorRT export requires Torch-TensorRT with ExecuTorch support (Linux, Python 3.11 or newer). "
    "Install it with: pip install rfdetr[executorch-tensorrt]"
)
# The delegate package `torch-tensorrt[executorch]` installs next to the ExecuTorch release it was built with, and the
# oldest ExecuTorch the `executorch-tensorrt` extra was validated with (pyproject.toml); the plain `executorch` extra
# pins an older one, so the two extras do not share an environment.
_TENSORRT_RUNTIME_MODULE = "torch_tensorrt_executorch_runtime"
_TENSORRT_MIN_EXECUTORCH: tuple[int, int] = (1, 6)
_TENSORRT_PARTIAL_CONVERSION_HINT = (
    "Torch-TensorRT could not convert every op of this model to TensorRT ({reason}). An ExecuTorch TensorRT export "
    "must hold the whole graph in one TensorRT engine, so a model it converts only in part is refused instead of "
    "written as a hybrid program. Detection Nano to Large, segmentation Nano and the keypoint preview model converted "
    "fully when checked; other models may not. For such a model, export with backend='xnnpack' (portable CPU) "
    "instead, or with format='tensorrt' for a native TensorRT engine."
)
_MATCHING_TORCH_HINT = (
    "Install the torch release your executorch version was built against: torch 2.12 for executorch 1.3, "
    "2.13 for 1.4, 2.14 for 1.5 and 2.15 for 1.6, e.g. `pip install 'torch>=2.14,<2.15'` for executorch 1.5."
)
_RUNTIME_ABI_HINT = (
    "executorch's compiled runtime extension (executorch.runtime) failed to load against the installed "
    "torch {torch_version} ({exc}). This is a known torch/executorch ABI-compatibility gap: each executorch "
    "release is compiled against one torch minor release, and another torch release can break that ABI. "
    "This does not affect .pte export itself (export() does not use executorch.runtime), only loading/running "
    "a .pte locally via ExecuTorch's Python runtime. " + _MATCHING_TORCH_HINT
)
_RUNTIME_OLD_TORCH_HINT = (
    "executorch {executorch_version} is installed with torch {torch_version}. executorch 1.5 is compiled against "
    "torch 2.14 and declares no torch requirement, so an older torch installs next to it, and with torch older "
    "than 2.13 every Method.execute call fails with 'tensor does not have a device'. " + _MATCHING_TORCH_HINT
)
_QNN_HINT = (
    "Qualcomm QNN export requires the ExecuTorch QNN backend, which is NOT in the `executorch` pip wheel. "
    "Build ExecuTorch from source against the Qualcomm AI Engine Direct (QAIRT) SDK "
    "(EXECUTORCH_BUILD_QNN=ON, QNN_SDK_ROOT set); see ExecuTorch docs/backends-qualcomm."
)
# Ops forced to run on CPU, for two distinct reasons:
#   * bitwise_not / lt.Scalar: no QNN node visitor in ExecuTorch 1.3.1 (attention mask).
#   * topk / max.dim: numerical correctness. These are the index-returning ops of the two-stage query
#     selection (max over classes -> top-300 proposals). On the HTP fp16 path they compute WRONG
#     indices, so the wrong 300 proposals are selected and every detection collapses (confidences ~0).
#     The problem is the integer index selection, not value precision: the CPU-fallback ops run in fp32
#     (the .pte has no fp16 tensors at the program level -- fp16 is internal to the HTP delegate, and its
#     boundary outputs are fp32), so they do exactly what eager does. Their inputs, however, still carry
#     the backbone+decoder's fp16 rounding, so the selected outputs differ from eager at fp16 level --
#     expected and harmless (the CPU ops propagate that error, they don't add to it). Verified by
#     layer-wise bisection on-device: pre-selection proposals match eager to
#     rel_rmse 3e-4 (fp16-accurate, not bit-exact); with HTP topk/max the post-selection reference points
#     diverge by ~0.9, and forcing just these two ops to CPU restores the top detections to sub-pixel
#     agreement. Everything else (incl. deformable-attention grid_sample) still delegates to the HTP.
_QNN_CPU_FALLBACK_OPS: frozenset[str] = frozenset(
    {"aten.bitwise_not.default", "aten.lt.Scalar", "aten.topk.default", "aten.max.dim"}
)


def _check_executorch_available(*, require_runtime: bool = False) -> None:
    """Verify that the ``executorch`` package is importable and meets the minimum version requirement.

    Args:
        require_runtime: Also verify that ``executorch.runtime`` (the on-host ``.pte`` loader/runner)
            imports cleanly. The export itself never touches ``executorch.runtime``, so
            this defaults to ``False`` and does not gate export capability. Set ``True`` only when the
            caller intends to load and run a ``.pte`` locally (e.g. a parity/verification check) — a
            torch/executorch ABI mismatch breaks only this submodule, not AOT export.

    Raises:
        ImportError: If ``executorch`` is not installed, is older than 1.3, or (when
            ``require_runtime=True``) its compiled runtime extension is ABI-incompatible with the
            installed ``torch`` version, or is executorch 1.5 next to a torch older than 2.13.
    """
    try:
        import executorch  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    import importlib.metadata

    try:
        _et_version = importlib.metadata.version("executorch")
        _major, _minor = (int(x) for x in _et_version.split(".")[:2])
        if (_major, _minor) < (1, 3):
            raise ImportError(
                f"executorch {_et_version} is installed but rfdetr[executorch] requires >=1.3 "
                "(private APIs used for QNN lowering changed in 1.3). "
                "Upgrade: pip install 'rfdetr[executorch]'"
            )
    except importlib.metadata.PackageNotFoundError:
        pass  # Source-build install — skip version check.

    if require_runtime:
        try:
            from executorch.runtime import Runtime  # noqa: F401
        except ImportError as exc:
            try:
                _torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                _torch_version = "unknown"
            raise ImportError(_RUNTIME_ABI_HINT.format(torch_version=_torch_version, exc=exc)) from exc
        # This pair imports cleanly and fails only at execute, so it is refused here with the fix.
        try:
            _et_version, _torch_version = importlib.metadata.version("executorch"), importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            return  # Source-build install — skip version check.
        _torch_release = tuple(int(part) for part in _torch_version.split("+")[0].split(".")[:2])
        if _et_version.startswith("1.5.") and _torch_release < (2, 13):
            raise ImportError(
                _RUNTIME_OLD_TORCH_HINT.format(executorch_version=_et_version, torch_version=_torch_version)
            )


def _check_tensorrt_executorch_integration() -> None:
    """Verify that Torch-TensorRT was installed with its ExecuTorch integration, not just on its own.

    Torch-TensorRT imports without it: ``torch-tensorrt[executorch]`` adds the delegate package that loads a TensorRT
    ``.pte`` and the ExecuTorch release it was built with. A host with Torch-TensorRT beside the plain ``executorch``
    extra has neither, so the export is refused here instead of failing, or yielding a ``.pte`` that cannot load, later.

    Raises:
        ImportError: If the delegate package is not installed, or the installed ``executorch`` is older than the release
            the ``executorch-tensorrt`` extra was validated with (see :data:`_TENSORRT_MIN_EXECUTORCH`).
    """
    if not is_installed(_TENSORRT_RUNTIME_MODULE):
        raise ImportError(_TENSORRT_HINT)
    try:
        version = importlib.metadata.version("executorch")
    except importlib.metadata.PackageNotFoundError:
        return  # Source-build install -- skip version check.
    if tuple(int(part) for part in version.split(".")[:2]) < _TENSORRT_MIN_EXECUTORCH:
        minimum = ".".join(str(part) for part in _TENSORRT_MIN_EXECUTORCH)
        raise ImportError(
            f"{_TENSORRT_HINT} Found executorch {version}, but the ExecuTorch TensorRT export was validated with "
            f"executorch {minimum}; rfdetr[executorch] pins an older release, so the two extras cannot share an "
            "environment."
        )


def _build_partitioner(backend: str) -> list[Any]:
    """Build the ExecuTorch partitioner list for *backend*.

    Args:
        backend: Lowercased backend name; see :data:`_VALID_BACKENDS`.

    Returns:
        A list of partitioner instances to pass to ``to_edge_transform_and_lower``.

    Raises:
        ImportError: If the backend's ExecuTorch extension is not installed.
    """
    if backend == "xnnpack":
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        return [XnnpackPartitioner()]
    if backend == "coreml":
        try:
            from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
        except ImportError as exc:
            raise ImportError(_COREML_HINT) from exc
        # No compile specs on purpose. Passing ExecuTorch's own
        # CoreMLBackend.generate_compile_specs(compute_precision=FLOAT16, minimum_deployment_target=iOS16) was
        # measured to be worse than the partitioner's own defaults on a pretrained RFDETRNano (Apple M3 Pro,
        # macOS 27.0.1, COCO val2017, all 5000 images): 48.05 -> 45.06 mAP. #1495 also recorded 14.0 -> 18.4 ms p50
        # for that `.pte`; it was not re-measured and is not reconciled with the native `.mlpackage` measurements,
        # where iOS16 is the faster program on the Neural Engine (10.8 vs 12.7 ms p50 back to back). The default path
        # already runs the delegate in fp16 and already reaches the Neural Engine (verified from the `com.apple.ane`
        # unified log). The two differ only in the deployment target: with no deployment-target compile spec,
        # ExecuTorch passes minimum_deployment_target=None and coremltools defaults an mlprogram to its fixed lowest
        # target, iOS15 (spec 6) (executorch 1.3.1, coremltools 9.0). That is the program the native CoreML export
        # now also produces, while iOS16 puts `resample` on the Neural Engine in fp16, which is what costs the
        # accuracy (#1024). With no compute-unit spec, ExecuTorch's CoreMLBackend converts with ct.ComputeUnit.ALL
        # (`compute_unit_from_compile_specs` fallback, applied in its `ct.convert` call at lowering time).
        return [CoreMLPartitioner()]
    # QNN and TensorRT use their own lowering functions; this raise is reached only if a new
    # backend is added to _VALID_BACKENDS without a corresponding branch above.
    raise ValueError(f"Unsupported ExecuTorch backend {backend!r}.")


def _lower_qnn(model: nn.Module, input_tensors: torch.Tensor, *, soc_model: str) -> Any:
    """Lower an RF-DETR model to a Qualcomm QNN (HTP, fp16) ExecuTorch program.

    QNN uses a dedicated lowering entry point (``to_edge_transform_and_lower_to_qnn``) rather than the generic
    partitioner path used by XNNPACK/CoreML, so it is handled separately.  The bulk of the network -- including the
    DINOv2 backbone and the deformable-attention ``grid_sample`` -- delegates to the Snapdragon HTP; the two-stage
    selection ops (``topk``/``max.dim``) and a couple of attention-mask ops run on CPU (see
    :data:`_QNN_CPU_FALLBACK_OPS`).

    Any op the QNN partitioner cannot evaluate support for -- one with no HTP visitor at all, or one whose visitor
    chokes on an unusual signature (e.g. the keypoint head's weightless ``LayerNorm``, where ET 1.3.1's
    ``op_layer_norm`` dereferences a ``None`` weight node) -- is treated as unsupported and left on CPU rather than
    aborting the whole lowering.  This keeps QNN export robust across the detection, segmentation, and keypoint heads.

    Args:
        model: RF-DETR module in export mode, on CPU.
        input_tensors: Example input ``(batch, channels, height, width)``; its shape is baked in.
        soc_model: Target Snapdragon SoC, a ``QcomChipset`` name (e.g. ``"SM8650"`` for Snapdragon 8 Gen 3).

    Returns:
        An ExecuTorch program manager (``.buffer`` holds the ``.pte`` bytes).

    Raises:
        ImportError: If the ExecuTorch QNN backend is not available (needs a source build against the QNN SDK).
        ValueError: If *soc_model* is not a known ``QcomChipset``.
    """
    try:
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnOperatorSupport
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
            to_edge_transform_and_lower_to_qnn,
        )
    except ImportError as exc:
        raise ImportError(_QNN_HINT) from exc

    soc = getattr(QcomChipset, soc_model.upper(), None)
    if soc is None:
        choices = sorted(n for n in dir(QcomChipset) if n[:2] in ("SM", "SA", "SS") and n[2:3].isdigit())
        raise ValueError(f"Unknown QNN SoC {soc_model!r}. Choose a QcomChipset name, e.g. {choices[:6]}.")

    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compiler_specs = generate_qnn_executorch_compiler_spec(soc_model=soc, backend_options=backend_options)

    # ExecuTorch 1.3.1's to_edge_transform_and_lower_to_qnn exports with strict=True, which lifts RF-DETR's
    # spatial-shape constant into the `transformer` submodule and then trips an un-lift bug (the same one the
    # XNNPACK/CoreML path avoids by exporting non-strict; see ExecuTorchExporter._lower_generic). Force the internal
    # torch.export to strict=False -- the captured graph is identical -- restoring the original afterwards.
    import torch.export as _torch_export

    _original_export = _torch_export.export

    def _nonstrict_export(*args: Any, **kwargs: Any) -> Any:
        kwargs["strict"] = False
        return _original_export(*args, **kwargs)

    # ET 1.3.1's QnnOperatorSupport.is_node_supported can raise instead of returning False: an op with no HTP
    # visitor raises KeyError, and a visitor fed an unusual signature (e.g. a weightless LayerNorm -> op_layer_norm
    # dereferences a None weight node) raises AttributeError. Either aborts the whole lowering. Treat any such
    # failure as "unsupported" so the node falls back to CPU and the rest of the graph still delegates.
    _original_is_node_supported = QnnOperatorSupport.is_node_supported

    def _safe_is_node_supported(self: Any, *args: Any, **kwargs: Any) -> bool:
        try:
            return bool(_original_is_node_supported(self, *args, **kwargs))
        except (KeyError, AttributeError) as exc:
            # ET 1.3.1: op with no HTP visitor raises KeyError; weightless LayerNorm raises
            # AttributeError (op_layer_norm dereferences a None weight). Treat as unsupported.
            logger.debug("QNN op support check suppressed (CPU fallback): %s", exc)
            return False

    try:
        _torch_export.export = _nonstrict_export
        QnnOperatorSupport.is_node_supported = _safe_is_node_supported
        edge_program = to_edge_transform_and_lower_to_qnn(
            model, (input_tensors,), compiler_specs, skip_node_op_set=set(_QNN_CPU_FALLBACK_OPS)
        )
    finally:
        _torch_export.export = _original_export
        QnnOperatorSupport.is_node_supported = _original_is_node_supported
    return edge_program.to_executorch()


def _lower_tensorrt(model: nn.Module, input_tensors: torch.Tensor) -> Any:
    """Lower an RF-DETR model to an ExecuTorch program whose graph runs as a TensorRT engine.

    Torch-TensorRT compiles the exported graph into TensorRT engines on the current CUDA device, and its ExecuTorch
    exporter turns each engine into a TensorRT delegate call.  That device is the GPU the engine is built for:
    :meth:`rfdetr.detr.RFDETR.export` makes the CUDA device of the model being exported current around the export, and
    a direct call builds on ``torch.cuda.current_device()``.  For the models checked (detection Nano to Large,
    segmentation Nano, keypoint preview) TensorRT converts every op, so the program is one delegate call plus the
    copies that move the input to the GPU and the outputs back to the CPU.  The compilation requires that full
    conversion (``require_full_compilation=True``): a model TensorRT converts only in part fails the export instead of
    yielding a hybrid program.

    Args:
        model: RF-DETR module in export mode, on CPU; it is moved to the current CUDA device.
        input_tensors: Example input ``(batch, channels, height, width)``; its shape is baked in.

    Returns:
        An ExecuTorch program manager (``.buffer`` holds the ``.pte`` bytes).

    Raises:
        ImportError: If Torch-TensorRT is not installed.
        RuntimeError: If Torch-TensorRT cannot convert every op of the model; the message names the models known to
            convert fully and the alternatives.
    """
    try:
        import torch_tensorrt.executorch
    except ModuleNotFoundError as exc:
        # An installed Torch-TensorRT that fails to load raises its own error unchanged.
        if exc.name not in {"torch_tensorrt", "torch_tensorrt.executorch", "executorch"}:
            raise
        raise ImportError(_TENSORRT_HINT) from exc

    example = input_tensors.cuda()
    # Non-strict capture, as for the other backends; parity and the one-engine structure were measured this way.
    exported_program = torch.export.export(model.cuda(), (example,), strict=False)
    try:
        # require_full_compilation turns a partial conversion into an error: without it Torch-TensorRT returns a hybrid
        # graph, and the .pte would run the unconverted ops outside TensorRT without a word.
        trt_module = torch_tensorrt.dynamo.compile(
            exported_program, arg_inputs=[example], require_full_compilation=True
        )
    except AssertionError as exc:
        # Torch-TensorRT 2.15's partitioners raise AssertionError("require_full_compilation=True was specified, but
        # ..."). Any other assertion is not about how much of the graph converted and keeps its own message.
        if "require_full_compilation" not in str(exc):
            raise
        raise RuntimeError(_TENSORRT_PARTIAL_CONVERSION_HINT.format(reason=exc)) from exc
    return torch_tensorrt.executorch.export(trt_module).to_executorch()


@dataclass(frozen=True, slots=True)
class ExecutorchConfig(ExportConfig):
    """Settings for ``format="executorch"``.

    Attributes:
        backend: Delegation backend the ``.pte`` is lowered for.
        soc: Target SoC, required by backends that compile ahead-of-time for one chip.
    """

    backend: Literal["xnnpack", "coreml", "qnn", "tensorrt"] = "xnnpack"
    soc: str | None = None


class ExecuTorchExporter(Exporter[ExecutorchConfig]):
    """Lower a prepared graph to an ExecuTorch ``.pte`` for the configured delegation backend.

    The conversion runs in four steps, each its own method, after :meth:`check_dependencies`: create the output
    directory, resolve the artifact filename, lower the graph (QNN and TensorRT through their own
    entry points, XNNPACK and CoreML through the generic partitioner), then write the ``.pte`` bytes.

    The graph arrives already switched into its export-friendly forward and on CPU: the base
    :class:`~rfdetr.export.base.Exporter` and the public :meth:`rfdetr.detr.RFDETR.export` entry point handle both
    before :meth:`_convert` runs.

    The artifact is named ``{stem}_{backend}.pte`` -- ``{stem}_qnn_{soc}.pte`` for the SoC-locked ``qnn`` backend --
    because the backend, and the SoC it compiles for, decides which hardware and runtime can load the file. A
    configured ``output_name`` names the file verbatim and suppresses that token; a backbone-only graph carries a
    trailing ``-backbone`` marker so it never overwrites a full-detector export of the same name.

    Examples:
        Requires the optional ``executorch`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        ExecuTorchExporter(ExecutorchConfig(backend="xnnpack", variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small_xnnpack.pte')
        ```
    """

    config_class = ExecutorchConfig
    setting_names = {"backend": "backend", "soc": "soc"}
    format = "executorch"
    display_name = "ExecuTorch"
    dynamic_batch_reason = "(see the ExecuTorch exporter for details). Export one .pte per batch size instead."
    experimental = True
    pip_extra = "executorch"
    # torch.export keeps the batch dim symbolic (verified: range stays [1, N] through export), but ExecuTorch 1.3.1
    # cannot carry it through lowering, for two independent reasons:
    #   1. to_edge's IR-validity check (EdgeOpArgValidator) calls len() on a Tensor[]-returning op's first result
    #      (RF-DETR's deformable-attention value.split(...)), which guards the batch to the example value and bakes a
    #      fixed-batch .pte.
    #   2. even with that check disabled, the windowed-attention reshapes (B <-> B*num_windows**2) lower to
    #      view_copy/et_view ops the runtime cannot resize for a dynamic batch (check_view_copy_args fails).
    # The resulting .pte runs only at the traced batch and silently mis-computes others, so it is refused rather than
    # shipped. (QNN additionally compiles a fixed-shape, SoC-locked binary.) Revisit on an ExecuTorch release that
    # fixes the verifier specialization and dynamic view resize.
    notes_reason = "ExecuTorch .pte has no metadata slot"

    @classmethod
    def _format_settings(cls, settings: Mapping[str, Any]) -> dict[str, Any]:
        """Read the delegation backend and SoC, refusing an unresolved backend rather than defaulting to one.

        ``_resolve_export_backend`` always resolves a backend for this format, so ``None`` here means that resolver
        was bypassed. Silently falling back to ``"xnnpack"`` would turn that into a quietly-wrong artifact instead
        of an error.

        Args:
            settings: The keyword arguments ``RFDETR.export`` was called with.

        Returns:
            The ``backend`` and ``soc`` keyword arguments for :class:`ExecutorchConfig`.

        Raises:
            ValueError: If no backend was resolved for this format.
        """
        backend = settings.get("backend")
        if backend is None:
            raise ValueError("format='executorch' requires a backend; none was resolved.")
        # The resolver already validated it against _VALID_BACKENDS; narrow the static type to the Literal the
        # configuration declares.
        return {"backend": cast("Literal['xnnpack', 'coreml', 'qnn', 'tensorrt']", backend), "soc": settings.get("soc")}

    def _check_capabilities(self) -> None:
        """Refuse an unsupported delegation backend when the exporter is built.

        Raises:
            ValueError: If the configured backend is not one ExecuTorch export supports.
        """
        super()._check_capabilities()
        backend_name = self.config.backend.lower()
        if backend_name not in _VALID_BACKENDS:
            raise ValueError(
                f"Unsupported ExecuTorch backend {backend_name!r}. Choose from: {sorted(_VALID_BACKENDS)}."
            )

    def _resolve_backend(self) -> str:
        """Return the configured delegation backend in lowercase, one of :data:`_VALID_BACKENDS`."""
        return self.config.backend.lower()

    def _resolve_soc(self) -> str:
        """Return the target SoC for a backend that compiles ahead-of-time for one chip.

        :func:`rfdetr.export._backend._resolve_export_backend` requires the public entry point to supply a ``soc`` for
        those backends and leaves it ``None`` for every other one, so the fallback below only ever backs a directly
        constructed exporter.

        Returns:
            The configured SoC, or :data:`_DEFAULT_SOC` when none was configured.
        """
        return self.config.soc if self.config.soc is not None else _DEFAULT_SOC

    @classmethod
    def check_dependencies(cls) -> None:
        """Verify ``executorch``, and the ``executorch.exir`` entry point the lowering goes through.

        Raises:
            ImportError: If ``executorch`` is missing, older than 1.3, or ships without ``executorch.exir``.
        """
        _check_executorch_available()
        # Probed here rather than at the lowering step: `executorch` can import while `executorch.exir` -- the entry
        # point the delegated path lowers through -- does not, and that has always failed before anything is written.
        from executorch.exir import to_edge_transform_and_lower  # noqa: F401

    def check_environment(self) -> None:
        """Refuse a TensorRT export without Torch-TensorRT's ExecuTorch integration or a CUDA device.

        Torch-TensorRT ships outside ``executorch`` and builds its engine on the current CUDA device, so the
        ``tensorrt`` backend cannot export without both. The other backends lower on the CPU and need nothing here.

        Raises:
            ImportError: If the backend is ``tensorrt`` and Torch-TensorRT, its ExecuTorch delegate package, or an
                ``executorch`` release new enough for it is not installed.
            RuntimeError: If the backend is ``tensorrt`` and no CUDA device is available.
        """
        if self._resolve_backend() != "tensorrt":
            return
        if not is_installed("torch_tensorrt"):
            raise ImportError(_TENSORRT_HINT)
        _check_tensorrt_executorch_integration()
        if not torch.cuda.is_available():
            raise RuntimeError("ExecuTorch TensorRT export requires an available CUDA device.")

    def _prepare_output_dir(self) -> Path:
        """Create and return the configured output directory.

        Returns:
            The output directory, created if it did not already exist.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_output_file(self, output_dir: Path, backend: str, *, backbone_only: bool) -> Path:
        """Return the path the ``.pte`` file is written to.

        Args:
            output_dir: Directory the artifact is written into.
            backend: Resolved delegation backend name.
            backbone_only: Whether the graph is a backbone-only export.

        Returns:
            Full path of the ``.pte`` file, inside *output_dir*.
        """
        stem, is_custom = resolve_export_stem(
            self.config.variant_name,
            self.config.output_name,
            default="backbone_model" if backbone_only else "inference_model",
        )
        if is_custom:
            export_name = stem
        else:
            # backend (+ SoC for qnn) determines what can load the file -- always encode it.
            backend_token = f"qnn_{self._resolve_soc()}" if backend == "qnn" else backend
            export_name = f"{stem}_{backend_token}"
        export_name = append_backbone_marker(
            export_name,
            backbone_only=backbone_only,
            named=bool(self.config.variant_name or self.config.output_name),
        )
        return output_dir / f"{export_name}.pte"

    def _lower_generic(self, model: nn.Module, input_tensors: torch.Tensor, backend: str) -> Any:
        """Lower *model* through ``torch.export`` and ExecuTorch's generic partitioner path (XNNPACK / CoreML).

        Args:
            model: The module to trace, in export mode, eval mode, and on CPU.
            input_tensors: Example input ``(batch, channels, height, width)``; its shape is baked in.
            backend: Resolved delegation backend name, selecting the partitioner.

        Returns:
            An ExecuTorch program manager (``.buffer`` holds the ``.pte`` bytes).

        Raises:
            ImportError: If the backend's ExecuTorch extension, ``AddmmToLinearTransform`` or, for XNNPACK,
                ``executorch.exir.passes.constant_prop_pass`` is unavailable.
        """
        from executorch.exir import to_edge_transform_and_lower

        # strict=False (non-strict capture). strict=True is the forward-looking default and captures this
        # model fine, but the resulting program lifts a small spatial-shape constant (spatial_shapes /
        # level_start_index) into the `transformer` submodule, and ExecuTorch 1.3.1's un-lift step
        # (torch/export/_unlift.py) cannot resolve a submodule-qualified lift_fresh_copy constant ->
        # lowering fails. Non-strict keeps those inline and lowers cleanly with verified parity. Revisit
        # strict=True when that upstream torch.export <-> ExecuTorch interaction is fixed.
        exported_program = torch.export.export(model, (input_tensors,), strict=False)
        if backend == "xnnpack":
            # Unmasked attention and constant weight slices keep the encoder and decoder inside the delegate (the
            # CHANGELOG entry for PR #1601 has the measurements). Not applied to CoreML, which was not evaluated.
            exported_program = fold_constants(decompose_attention(exported_program))
        # Imported in the non-QNN path only: the qnn backend never uses this transform, and
        # some ExecuTorch installs don't ship it -- a top-level import would raise ImportError
        # they never needed.
        try:
            from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
        except ImportError as exc:
            raise ImportError(
                "AddmmToLinearTransform is unavailable in this ExecuTorch install; upgrade "
                "executorch to a version shipping executorch.backends.transforms.addmm_mm_to_linear."
            ) from exc
        # AddmmToLinearTransform recombines the addmm/mm ops that torch.export decomposes
        # nn.Linear into, back into aten.linear. The handful the partitioner leaves un-delegated (the
        # two-stage encoder-output projections over the full token sequence) otherwise run on the
        # portable addmm.out kernel, which is ~2 orders of magnitude slower than linear.out:
        # 111 ms -> 44 ms end-to-end on RFDETRNano/XNNPACK, outputs identical to ~1e-5.
        edge_program = to_edge_transform_and_lower(
            exported_program,
            transform_passes=[AddmmToLinearTransform()],
            partitioner=_build_partitioner(backend),
        )
        return edge_program.to_executorch()

    def _lower(self, graph: ExportGraph, backend: str, output_file: Path) -> Any:
        """Switch *graph* into eval mode and lower it for *backend*, normalizing converter failures.

        Args:
            graph: The prepared model and its graph metadata.
            backend: Resolved delegation backend name.
            output_file: Destination the lowered program will be written to; named in the progress log.

        Returns:
            An ExecuTorch program manager (``.buffer`` holds the ``.pte`` bytes).

        Raises:
            ImportError: If ``executorch``'s backend extension is not installed.
            ValueError: If a SoC-locked backend was given a SoC ExecuTorch does not know.
            RuntimeError: If ``torch.export`` or ExecuTorch lowering fails for any other reason.
        """
        model = graph.model.eval()
        logger.info(f"Exporting model to ExecuTorch ({backend}) format: {output_file}")
        try:
            with torch.no_grad():
                if backend == "qnn":
                    # QNN has its own lowering entry point + workarounds; see _lower_qnn.
                    logger.warning(
                        "ExecuTorch QNN export is EXPERIMENTAL. It requires a source build of ExecuTorch "
                        "against the QAIRT SDK (not the pip wheel) and so cannot be CI-tested. Validated "
                        "on-device on the Snapdragon HTP: top detections match the "
                        "PyTorch model to sub-pixel accuracy, with the two-stage selection ops (topk/max.dim) "
                        "kept on CPU -- the HTP fp16 path computes wrong indices for them (see "
                        "_QNN_CPU_FALLBACK_OPS). Remaining differences are fp16-level; validate detections on "
                        "your target Snapdragon before relying on this export."
                    )
                    return _lower_qnn(model, graph.input_tensors, soc_model=self._resolve_soc())
                if backend == "tensorrt":
                    return _lower_tensorrt(model, graph.input_tensors)
                return self._lower_generic(model, graph.input_tensors, backend)
        except (ImportError, ValueError):
            raise
        except Exception as exc:
            logger.exception("ExecuTorch export failed")
            raise RuntimeError(f"ExecuTorch export failed: {exc}") from exc

    def _write(self, program: Any, output_file: Path) -> Path:
        """Write the lowered program's serialized bytes to *output_file*.

        Args:
            program: ExecuTorch program manager returned by the lowering step.
            output_file: Destination ``.pte`` path.

        Returns:
            *output_file*, now holding the serialized program.
        """
        output_file.write_bytes(program.buffer)
        logger.info(f"Successfully exported ExecuTorch model to: {output_file}")
        return output_file

    def _convert(self, graph: ExportGraph) -> Path:
        """Write the ``.pte`` file and return its path.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the exported ``.pte`` file.

        Raises:
            ImportError: If the backend's extension is not installed (``executorch`` itself is checked first, by
                :meth:`check_dependencies`).
            ValueError: If the SoC a SoC-locked backend compiles for is not supported.
            RuntimeError: If ``torch.export`` or ExecuTorch lowering fails.
        """
        backend = self._resolve_backend()
        output_dir = self._prepare_output_dir()
        output_file = self._resolve_output_file(output_dir, backend, backbone_only=graph.backbone_only)
        program = self._lower(graph, backend, output_file)
        return self._write(program, output_file)

    def _metadata_for_artifact(self, metadata: ExportMetadata, path: Path) -> ExportMetadata:
        """Use ExecuTorch's positional method inputs and outputs."""
        return positional_metadata(metadata)
