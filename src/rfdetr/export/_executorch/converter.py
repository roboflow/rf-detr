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

XNNPACK covers portable CPU; QNN and CoreML cover the mobile accelerators.

RF-DETR's detection graph exports cleanly once the deformable-attention modules are switched into their export-friendly
path (handled by :meth:`model.export`, which the caller invokes before reaching this function).  This module only
performs the conversion; numerical parity against eager PyTorch is checked by the test suite (mirroring the ONNX and
TFLite exporters, which likewise convert without an in-library validation step).

Note:
    The produced ``.pte`` expects the same input normalization as the ONNX export: ImageNet mean/std
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``).  The caller is responsible for applying this
    normalization at inference time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# Backends accepted by :func:`export_executorch`.  XNNPACK (portable CPU, fp32) and CoreML (Apple devices, fp16) are
# validated end-to-end.  Qualcomm QNN (Snapdragon HTP/NPU, fp16) lowers cleanly and delegates
# the bulk of the network to the HTP -- but its delegate ships neither in the ``executorch`` wheel nor runs off a
# Snapdragon device, so it requires a source build of ExecuTorch against the Qualcomm AI Engine Direct SDK.  Vulkan is
# omitted (blocked by two upstream ExecuTorch bugs + missing ops; see triage notes).
_VALID_BACKENDS: frozenset[str] = frozenset({"xnnpack", "coreml", "qnn"})

# Backends whose ahead-of-time compilation bakes a target SoC into the ``.pte`` (see :func:`_lower_qnn`), so a SoC
# must be supplied at export time.  XNNPACK (portable CPU) and CoreML (Apple's runtime selects the device at load)
# need none.  Public :meth:`rfdetr.detr.RFDETR.export` consults this set (via
# :func:`rfdetr.export._backend._resolve_export_backend`) to decide whether its ``soc`` argument is required for the
# requested ``backend``.
_SOC_BACKENDS: frozenset[str] = frozenset({"qnn"})

# Public aliases so detr.py can import these without crossing a private-name boundary.
VALID_BACKENDS: frozenset[str] = _VALID_BACKENDS
SOC_BACKENDS: frozenset[str] = _SOC_BACKENDS

_INSTALL_HINT = "ExecuTorch export requires the `executorch` package. Install it with: pip install rfdetr[executorch]"
_COREML_HINT = "CoreML export requires `coremltools`. Install it with: pip install coremltools"
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


def _check_executorch_available() -> None:
    """Verify that the ``executorch`` package is importable and meets the minimum version requirement.

    Raises:
        ImportError: If ``executorch`` is not installed or is older than 1.3.
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
        return [CoreMLPartitioner()]
    # QNN uses _lower_qnn instead of this function; this raise is reached only if a new
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
    # XNNPACK/CoreML path avoids by exporting non-strict; see export_executorch). Force the wrapper's internal
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


def export_executorch(
    model: nn.Module,
    input_tensors: torch.Tensor,
    output_dir: str | os.PathLike[str],
    *,
    backend: Literal["xnnpack", "coreml", "qnn"] = "xnnpack",
    variant_name: str | None = None,
    soc: str = "SM8650",
    dynamic_batch: bool = False,
) -> Path:
    """Export an RF-DETR model to an ExecuTorch ``.pte`` file.

    The model must already be switched into export mode (``model.export()``) and moved to CPU by the caller -- the
    public :meth:`rfdetr.detr.RFDETR.export` entry point handles both.

    The ``backend``/``soc`` defaults below are a convenience for direct/internal calls only; the public
    :meth:`rfdetr.detr.RFDETR.export` contract requires ``backend`` to be passed explicitly (and ``soc`` for
    SoC-locked backends) -- it never relies on these defaults.

    Args:
        model: The RF-DETR PyTorch module to export, in export mode and on CPU.
        input_tensors: Example input tensor ``(batch, channels, height, width)`` used to trace the graph.  Its shape is
            baked into the exported program.
        output_dir: Directory where the ``.pte`` file is written.
        backend: ExecuTorch delegation backend.  One of ``"xnnpack"`` (portable CPU, fp32), ``"coreml"`` (Apple
            devices, fp16; requires ``coremltools``), or ``"qnn"`` (Qualcomm Snapdragon HTP, fp16; requires an
            ExecuTorch source build against the QNN SDK).
        variant_name: Model variant identifier (e.g. ``"rfdetr-nano"``).  When provided, the file is named
            ``{variant_name}.pte`` instead of the generic ``inference_model.pte``.
        soc: Target SoC for backends that compile for a specific chip (currently ``"qnn"``), a ``QcomChipset`` name
            (default ``"SM8650"`` = Snapdragon 8 Gen 3).  Ignored for backends that do not target a specific SoC.
        dynamic_batch: Variable batch size at runtime.  Not supported on executorch 1.3.1 (raises
            ``NotImplementedError``): ``torch.export`` keeps the batch axis symbolic, but the runtime cannot resize
            RF-DETR's windowed-attention reshapes, so a dynamic ``.pte`` runs only at the traced batch.  Export one
            ``.pte`` per batch size for now.

    Returns:
        Path to the exported ``.pte`` file.

    Raises:
        ImportError: If ``executorch`` (or the backend extension) is not installed.
        ValueError: If *backend* is not supported.
        NotImplementedError: If *dynamic_batch* is requested (unsupported on executorch 1.3.1).
        RuntimeError: If ``torch.export`` or ExecuTorch lowering fails.

    Examples:
        Export for portable CPU inference (XNNPACK, fp32)::

            pte_path = export_executorch(model, input_tensor, "output/", backend="xnnpack")

        Export for Apple Neural Engine (CoreML, fp16; detections correct, raw diffs from fp16 expected)::

            pte_path = export_executorch(model, input_tensor, "output/", backend="coreml")

        Export for Qualcomm HTP (QNN, fp16; requires ExecuTorch source build against QAIRT SDK)::

            pte_path = export_executorch(model, input_tensor, "output/", backend="qnn", soc="SM8650")
    """
    backend_name: str = backend.lower()
    if backend_name not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported ExecuTorch backend {backend_name!r}. Choose from: {sorted(_VALID_BACKENDS)}.")
    if dynamic_batch:
        # torch.export keeps the batch dim symbolic (verified: range stays [1, N] through export), but ExecuTorch
        # 1.3.1 cannot carry it through lowering, for two independent reasons:
        #   1. to_edge's IR-validity check (EdgeOpArgValidator) calls len() on a Tensor[]-returning op's first
        #      result (RF-DETR's deformable-attention value.split(...)), which guards the batch to the example
        #      value and bakes a fixed-batch .pte.
        #   2. even with that check disabled, the windowed-attention reshapes (B <-> B*num_windows**2) lower to
        #      view_copy/et_view ops the runtime cannot resize for a dynamic batch (check_view_copy_args fails).
        # The resulting .pte runs only at the traced batch and silently mis-computes others, so it is refused
        # rather than shipped. (QNN additionally compiles a fixed-shape, SoC-locked binary.) Revisit on an
        # ExecuTorch release that fixes the verifier specialization and dynamic view resize.
        raise NotImplementedError(
            "ExecuTorch export does not support dynamic_batch on executorch 1.3.1 (the edge verifier specializes "
            "the batch dim and the runtime cannot resize the windowed-attention reshapes). Export one .pte per "
            "batch size instead."
        )

    _check_executorch_available()
    from executorch.exir import to_edge_transform_and_lower

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if variant_name:
        # Sanitize against path traversal (mirrors export_onnx): "foo/bar" -> "bar".
        variant_name = os.path.splitext(os.path.basename(variant_name))[0]
        export_name = variant_name
    else:
        export_name = "inference_model"
    output_file = output_dir / f"{export_name}.pte"

    model = model.eval()
    logger.info(f"Exporting model to ExecuTorch ({backend_name}) format: {output_file}")
    try:
        with torch.no_grad():
            if backend_name == "qnn":
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
                executorch_program = _lower_qnn(model, input_tensors, soc_model=soc)
            else:
                # strict=False (non-strict capture). strict=True is the forward-looking default and captures this
                # model fine, but the resulting program lifts a small spatial-shape constant (spatial_shapes /
                # level_start_index) into the `transformer` submodule, and ExecuTorch 1.3.1's un-lift step
                # (torch/export/_unlift.py) cannot resolve a submodule-qualified lift_fresh_copy constant ->
                # lowering fails. Non-strict keeps those inline and lowers cleanly with verified parity. Revisit
                # strict=True when that upstream torch.export <-> ExecuTorch interaction is fixed.
                exported_program = torch.export.export(model, (input_tensors,), strict=False)
                edge_program = to_edge_transform_and_lower(
                    exported_program, partitioner=_build_partitioner(backend_name)
                )
                executorch_program = edge_program.to_executorch()
    except (ImportError, ValueError):
        raise
    except Exception as exc:
        logger.exception("ExecuTorch export failed")
        raise RuntimeError(f"ExecuTorch export failed: {exc}") from exc

    output_file.write_bytes(executorch_program.buffer)
    logger.info(f"Successfully exported ExecuTorch model to: {output_file}")
    return output_file
