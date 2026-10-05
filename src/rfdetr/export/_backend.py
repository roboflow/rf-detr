# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backend/format resolution and export-format dispatch helpers for :meth:`rfdetr.detr.RFDETR.export`.

These are utility functions shared by :meth:`rfdetr.detr.RFDETR.export` and the exporter classes — kept in their own
module so neither the public entry point nor any one format package owns them.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import Protocol, cast

from torch import Tensor, nn

from rfdetr.export.registry import REGISTRY
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class _ExportableModule(Protocol):
    """Structural type for a module exposing a zero-arg export-mode switch.

    Narrows the static type before calling ``.export()`` directly on an ``nn.Module`` -- ``nn.Module.__getattr__``'s
    stub resolves that attribute access to ``Tensor | Module``, which mypy refuses to call (mirrors the ``cast(Backbone,
    ...)`` precedent used for backbone-only export in :meth:`rfdetr.detr.RFDETR.export`).
    """

    def export(self) -> None: ...


def _switch_to_export_mode(model: nn.Module) -> None:
    """Switch *model* into its export-friendly forward, if it exposes one.

    Shared by the ONNX exporter (``export_onnx``) and the ExecuTorch, CoreML, and OpenVINO dispatch
    functions below, so every export path switches through one guarded choke point. A model without a
    callable ``export`` attribute (e.g. a plain ``nn.Module`` in a unit test) is left untouched rather
    than raising.

    Switching a module that is already in export mode is a no-op here, because it is *not* a no-op in
    the models: :meth:`rfdetr.models.lwdetr.LWDETR.export`,
    :meth:`rfdetr.models.backbone.backbone.Backbone.export` and
    :meth:`rfdetr.models.position_encoding.PositionEmbeddingSine.export` each stash
    ``self._forward_origin = self.forward`` before swapping in ``forward_export``, so a second call
    overwrites the saved original with the export forward and loses the real one for good.
    (``DinoV2.export`` already guards itself; these three do not.) The guard lives here rather than in
    the models so every export path shares one choke point.

    Args:
        model: The module to switch into export mode, if supported.
    """
    if getattr(model, "_export", False):
        return
    export_method = getattr(model, "export", None)
    if callable(export_method):
        cast(_ExportableModule, model).export()


class _BackboneExport(nn.Module):
    """Expose all feature projectors from a backbone already prepared for export."""

    def __init__(self, backbone: Backbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: Tensor) -> list[Tensor]:
        """Return primary feature levels followed by cross-attention levels, when present."""
        features, _, cross_attn_features = self.backbone.forward_export(images)
        return features if cross_attn_features is None else features + cross_attn_features


# Every format accepted by :meth:`rfdetr.detr.RFDETR.export`, derived from the exporter registry so adding a
# format stays a one-line data change there.
_EXPORT_FORMATS: frozenset[str] = frozenset(REGISTRY)
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
    if not isinstance(format, str) or format not in _EXPORT_FORMATS:
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
    from rfdetr.export._executorch.exporter import SOC_BACKENDS, VALID_BACKENDS

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


def check_onnx_available(
    install_hint: str = 'Install with: pip install "rfdetr[onnx]"', *, stage: str = "ONNX export"
) -> None:
    """Raise the install hint when ``onnx`` is missing.

    ``torch.onnx.export`` needs ``onnx`` too, but reports it only once the whole trace has run, without the hint. The
    TFLite and TensorRT exporters, which export through the ONNX stage, check it with this too — from here rather than
    from :mod:`rfdetr.export._onnx.exporter` so that neither format has to reach into another format's module (and its
    eager ``onnx`` import) just to run this probe.

    ``onnx`` is imported here rather than read from a module-level binding, which is fixed once first imported: a user
    who installs it after a refused export and retries in the same process (a notebook) would otherwise be refused by
    this check until they restart it. Only an ``onnx`` that is not installed gets the hint; an installed one that
    fails to import (a broken native extension, say) raises its own error unchanged.

    Args:
        install_hint: The sentence that tells the user what to install. A format that exports through the ONNX stage
            names its own extra, which installs ``onnx`` along with everything else the format needs.
        stage: The export stage to name in the message. A caller exporting through the ONNX stage on the way to its
            own format (TFLite, TensorRT) passes its own label so the message doesn't misname ONNX as the culprit.

    Raises:
        ImportError: If ``onnx`` is not installed, or, unchanged, if an installed ``onnx`` fails to import.
    """
    try:
        importlib.import_module("onnx")
    except ModuleNotFoundError as error:
        if error.name != "onnx":
            raise
        raise ImportError(f"{stage} dependencies are missing (onnx). {install_hint}") from error


#: Whether the onnx-before-TensorFlow import-order warning has already been logged in this process. The order it
#: reports cannot be repaired once both libraries are loaded, so repeating it on every preload — up to four times in
#: one TFLite export — only buries the first one.
_ONNX_ORDER_WARNED = False


def _onnx_imported_before_tensorflow() -> bool:
    """Report whether ``onnx`` entered ``sys.modules`` ahead of ``tensorflow``.

    ``sys.modules`` is an ordinary dict, so iterating it yields keys in insertion order — which for a top-level package
    is the order the two libraries were first imported in. That is a heuristic, not a loader guarantee: deleting and
    re-importing a module moves it to the end. It only ever decides whether to emit a warning, never what gets
    imported, so a wrong answer costs a log line.

    ``onnx2tf`` is deliberately not treated as ``onnx``: it is a pure-Python package whose import does not load ONNX's
    compiled extension.

    Returns:
        ``True`` when an ``onnx`` module precedes every ``tensorflow`` module, or when ``onnx`` is imported and
        ``tensorflow`` is not. ``False`` otherwise, including when neither is imported.
    """
    for name in tuple(sys.modules):
        if name == "onnx" or name.startswith("onnx."):
            return True
        if name == "tensorflow" or name.startswith("tensorflow."):
            return False
    return False


def preload_tensorflow_before_onnx() -> None:
    """Import TensorFlow before ONNX's C extension so TFLite conversion cannot deadlock.

    ``onnx``'s compiled extension and TensorFlow both statically link Abseil and export its symbols as *weak external*
    definitions.  The dynamic loader coalesces weak definitions onto the first image that provides them, so whichever
    library is imported first supplies Abseil's synchronization primitives — including the per-thread semaphore that
    ``absl::Mutex`` and ``absl::Notification`` block on — to *both* libraries.

    When ONNX wins that race, TensorFlow's executor blocks in ``absl::Notification::WaitForNotification()`` while
    restoring the SavedModel bundle and is never woken, hanging the export at 0% CPU with no traceback and no
    ``.tflite``.  ``format="tflite"`` reaches ``onnx2tf``'s converter module (``onnx2tf.onnx2tf``) only after a full
    ONNX export, so ONNX always wins unless TensorFlow is preloaded here.  See
    https://github.com/roboflow/rf-detr/issues/1322 for the measured comparison.

    Importing ``onnx`` *after* TensorFlow is safe, so the warning below is keyed on the relative order of the two
    imports (:func:`_onnx_imported_before_tensorflow`) rather than on ``onnx`` merely being imported.  A TFLite export
    preloads several times, and the order it reports is the same every time, so the warning is logged once per process.

    Note:
        Does not re-import TensorFlow when it is already loaded, and stays silent when TensorFlow is not installed —
        on the export path, the actionable missing-dependency error is raised by
        :meth:`~rfdetr.export._tflite.exporter.TFLiteExporter.check_dependencies`; a direct
        :meth:`~rfdetr.export._tflite.exporter.TFLiteExporter.convert_onnx` call checks only ``onnx2tf``.

    Examples:
        >>> preload_tensorflow_before_onnx()  # returns when the top-level tensorflow package is unavailable
    """
    onnx_won_the_race = _onnx_imported_before_tensorflow()

    if "tensorflow" not in sys.modules:
        try:
            importlib.import_module("tensorflow")
        except ModuleNotFoundError as error:
            if error.name != "tensorflow":
                raise
            return

    global _ONNX_ORDER_WARNED
    if onnx_won_the_race and not _ONNX_ORDER_WARNED:
        _ONNX_ORDER_WARNED = True
        logger.warning(
            "onnx was imported before TensorFlow. Both statically link Abseil and export its symbols weakly, so "
            "TensorFlow can block forever while restoring the SavedModel bundle during TFLite conversion. That order "
            "cannot be repaired once both are loaded. If the export hangs with no output, import tensorflow before "
            "onnx or run the export in a fresh process."
        )
