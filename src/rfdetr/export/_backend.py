# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backend/format resolution and export-format dispatch helpers for :meth:`rfdetr.detr.RFDETR.export`.

These are utility functions shared by the export CLI (:mod:`rfdetr.export.main`) and the public
:meth:`rfdetr.detr.RFDETR.export` API — kept in their own module so ``main.py`` stays focused on CLI orchestration.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, cast

from torch import Tensor

from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# Every format accepted by :meth:`rfdetr.detr.RFDETR.export`.
_EXPORT_FORMATS: frozenset[str] = frozenset({"onnx", "tflite", "tensorrt", "executorch", "coreml"})
# The subset of :data:`_EXPORT_FORMATS` that specialize for a hardware backend, and so require a ``backend`` argument
# (the rest are backend-agnostic).  The accepted backends per format, and the backends that further require a ``soc``,
# are owned by the converter (``_VALID_BACKENDS`` / ``_SOC_BACKENDS``).
# Note: ``format="coreml"`` (native ``.mlpackage``) is backend-agnostic; ExecuTorch's ``backend="coreml"`` is separate
# and still goes through ``format="executorch"``.
_BACKEND_FORMATS: frozenset[str] = frozenset({"executorch"})


def _resolve_export_backend(format: str, backend: str | None, soc: str | None) -> tuple[str | None, str | None]:
    """Validate a ``format`` / ``backend`` / ``soc`` combination and return the effective ``(backend, soc)``.

    Driven by the :data:`_EXPORT_FORMATS` / :data:`_BACKEND_FORMATS` registries (and the converter's backend/SoC sets)
    rather than per-format branches, so adding a format or backend is a data change:

    * A format not in :data:`_BACKEND_FORMATS` is backend-agnostic — it takes neither ``backend`` nor ``soc``;
      supplying one warns and it is ignored (returned ``None``).
    * A format in :data:`_BACKEND_FORMATS` requires ``backend`` to be one of the backends its converter accepts
      (looked up by format).
    * A backend that compiles for a specific chip (looked up by format+backend against the converter's SoC set)
      requires ``soc``; any other backend warns if a ``soc`` is supplied and ignores it.

    Args:
        format: Export format; one of :data:`_EXPORT_FORMATS`.
        backend: Requested hardware backend, or ``None``.
        soc: Requested target SoC, or ``None``.

    Returns:
        ``(backend, soc)`` with each value set to ``None`` when the format/backend does not use it.

    Raises:
        ValueError: On an unknown format, a missing or unknown required backend, or a missing required SoC.

    Examples:
        >>> _resolve_export_backend("onnx", None, None)
        (None, None)
    """
    if format not in _EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format {format!r}. Choose from: {sorted(_EXPORT_FORMATS)}.")

    if format not in _BACKEND_FORMATS:
        # Backend-agnostic format: warn on any supplied (and therefore unused) backend/soc.
        for name, value in (("backend", backend), ("soc", soc)):
            if value is not None:
                warnings.warn(
                    f"`{name}={value!r}` is ignored for format={format!r}; this format does not require a hardware "
                    f"backend specialization.",
                    UserWarning,
                    stacklevel=3,
                )
        return None, None

    # Backend-bearing format: the converter owns the authoritative capability sets.  These are keyed by format
    # (accepted backends) and by format+backend (which backends require a ``soc``).  Adding a second backend-bearing
    # format is primarily a data change (add entries below + update _EXPORT_FORMATS / _BACKEND_FORMATS), but also
    # requires a lazy import and an elif branch in export(). Imported lazily so that backend-agnostic exports never
    # pull in the (optional, heavy) executorch dependency.
    from rfdetr.export._executorch.converter import SOC_BACKENDS, VALID_BACKENDS

    accepted_backends: dict[str, frozenset[str]] = {"executorch": VALID_BACKENDS}
    soc_backends: dict[str, frozenset[str]] = {"executorch": SOC_BACKENDS}
    valid = accepted_backends.get(format, frozenset())
    soc_required = soc_backends.get(format, frozenset())

    if backend is None:
        raise ValueError(f"format {format!r} requires a valid backend (one of {sorted(valid)}), but none was provided.")
    # Normalise case so RFDETR.export(backend="XNNPACK") and backend="xnnpack" behave identically.
    backend = backend.lower()
    if backend not in valid:
        raise ValueError(f"Unsupported backend {backend!r} for format {format!r}. Choose from: {sorted(valid)}.")

    if backend not in soc_required:
        if soc is not None:
            warnings.warn(
                f"`soc={soc!r}` is ignored for backend={backend!r}; this backend does not target a specific SoC.",
                UserWarning,
                stacklevel=3,
            )
        return backend, None

    if soc is None:
        raise ValueError(f"backend {backend!r} requires a valid soc, but none was provided.")
    return backend, soc


def _export_executorch_format(
    model: LWDETR,
    input_tensors: Tensor,
    output_dir_path: Path,
    *,
    backend: str | None,
    soc: str | None,
    variant_name: str | None,
    dynamic_batch: bool,
    notes: object,
) -> Path:
    """Dispatch :meth:`rfdetr.detr.RFDETR.export` to the ExecuTorch converter.

    Args:
        model: The prepared (CPU, export-mode) PyTorch module to export.
        input_tensors: Example input tensor used to trace the graph.
        output_dir_path: Directory where the ``.pte`` file is written.
        backend: ExecuTorch delegation backend; must not be ``None`` (invariant enforced by
            :func:`_resolve_export_backend`, which always sets a backend for ``format="executorch"``).
        soc: Target SoC for backends that compile for a specific chip, or ``None``.
        variant_name: Model variant identifier used to name the output file.
        dynamic_batch: Whether a dynamic batch dimension was requested (always rejected below).
        notes: User-supplied export metadata; ExecuTorch has no metadata slot, so a non-``None`` value warns.

    Returns:
        Path to the exported ``.pte`` file.

    Raises:
        RuntimeError: If ``backend`` is ``None`` (invariant violation — see Args).
        ImportError: If the optional ``executorch`` dependency is not installed.

    Examples:
        >>> _export_executorch_format(  # doctest: +SKIP
        ...     model, input_tensors, output_dir_path,
        ...     backend="xnnpack", soc=None, variant_name="small",
        ...     dynamic_batch=False, notes=None,
        ... )
        PosixPath('out/model.pte')
    """
    if notes is not None:
        warnings.warn(
            "`notes` is not forwarded to format='executorch' (ExecuTorch .pte has no metadata slot). "
            "This argument is ignored.",
            UserWarning,
            stacklevel=3,
        )
    # Invariant: _resolve_export_backend always sets backend for executorch formats.
    if backend is None:  # pragma: no cover
        raise RuntimeError("backend must not be None for format='executorch' — invariant: _resolve_export_backend.")
    warnings.warn(
        "ExecuTorch export is experimental and work-in-progress.",
        UserWarning,
        stacklevel=3,
    )
    try:
        from rfdetr.export._executorch.converter import export_executorch
    except ImportError:
        logger.error(
            "It seems some dependencies for ExecuTorch export are missing."
            " Please run `pip install rfdetr[executorch]` and try again.",
        )
        raise
    # ExecuTorch consumes a torch.export graph directly, so switch the model into its
    # export-friendly forward (the ONNX path does this inside export_onnx).
    if hasattr(model, "export"):
        model.export()
    # soc only applies to the qnn backend; _resolve_export_backend leaves it None otherwise.
    soc_kwargs = {"soc": soc} if soc is not None else {}
    # _resolve_export_backend already validated backend against _VALID_BACKENDS at runtime;
    # narrow the static type to match export_executorch's Literal signature.

    backend_literal = cast(Literal["xnnpack", "coreml", "qnn"], backend)
    pte_path = export_executorch(
        model=model,
        input_tensors=input_tensors,
        output_dir=str(output_dir_path),
        backend=backend_literal,
        variant_name=variant_name,
        dynamic_batch=dynamic_batch,
        **soc_kwargs,
    )
    logger.info(f"Successfully exported ExecuTorch model to: {pte_path}")
    return pte_path


def _export_coreml_format(
    model: LWDETR,
    input_tensors: Tensor,
    output_dir_path: Path,
    *,
    variant_name: str | None,
    verbose: bool,
    notes: object,
    compute_precision: str | None = None,
) -> Path:
    """Dispatch :meth:`rfdetr.detr.RFDETR.export` to the native CoreML converter.

    This is ``format="coreml"`` → ``.mlpackage`` via ``torch.export`` + ``coremltools``. It is distinct from
    ExecuTorch's ``format="executorch", backend="coreml"`` path, which produces a ``.pte``.

    Args:
        model: The prepared (CPU) PyTorch module to export.
        input_tensors: Example input tensor used to trace the graph.
        output_dir_path: Directory where the ``.mlpackage`` is written.
        variant_name: Model variant identifier used to name the output bundle.
        verbose: Forwarded to :func:`~rfdetr.export._coreml.converter.export_coreml`.
        notes: User-supplied export metadata; CoreML has no ONNX-style metadata slot, so a non-``None`` value warns.
        compute_precision: ``"float32"``/``"float16"``/``None`` — forwarded to
            :func:`~rfdetr.export._coreml.converter.export_coreml`'s ``compute_precision``.

    Note:
        Unlike the ONNX path, output names are not forwarded to ``ct.convert`` — coremltools infers
        its own output names for the ``.mlpackage`` spec. Consumers must rely on **output position**,
        not name, to match the ``(dets, labels)`` / ``(dets, labels, masks)`` / ``(dets, labels,
        keypoints)`` contract documented on :meth:`rfdetr.detr.RFDETR.export`.

    Returns:
        Path to the exported ``.mlpackage`` bundle.

    Raises:
        ImportError: If the optional ``coreml`` dependency is not installed.
        NotImplementedError: If the exported graph has CoreML registry gaps.

    Examples:
        >>> _export_coreml_format(  # doctest: +SKIP
        ...     model, input_tensors, output_dir_path,
        ...     variant_name="small", verbose=True, notes=None,
        ... )
        PosixPath('out/model.mlpackage')
    """
    if notes is not None:
        warnings.warn(
            "`notes` is not forwarded to format='coreml' (CoreML .mlpackage has no ONNX-style metadata slot). "
            "This argument is ignored.",
            UserWarning,
            stacklevel=3,
        )
    warnings.warn(
        "CoreML export is experimental and work-in-progress. Dynamic batch is not supported.",
        UserWarning,
        stacklevel=3,
    )
    try:
        from rfdetr.export._coreml.converter import export_coreml
    except ImportError:
        logger.error(
            "It seems some dependencies for CoreML export are missing."
            " Please run `pip install rfdetr[coreml]` and try again.",
        )
        raise
    # CoreML consumes a torch.export graph directly, so switch the model into its
    # export-friendly forward (the ONNX path does this inside export_onnx).
    if hasattr(model, "export"):
        model.export()
    mlpackage_path = export_coreml(
        model=model,
        input_tensors=input_tensors,
        output_dir=str(output_dir_path),
        variant_name=variant_name,
        verbose=verbose,
        compute_precision=compute_precision,
    )
    logger.info(f"Successfully exported CoreML model to: {mlpackage_path}")
    return mlpackage_path
