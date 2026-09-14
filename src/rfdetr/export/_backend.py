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
    ``.tflite``.  ``format="tflite"`` reaches ``onnx2tf`` only after a full ONNX export, so ONNX always wins unless
    TensorFlow is preloaded here.  See https://github.com/roboflow/rf-detr/issues/1322 for the measured comparison.

    Importing ``onnx`` *after* TensorFlow is safe, so the warning below is keyed on the relative order of the two
    imports (:func:`_onnx_imported_before_tensorflow`) rather than on ``onnx`` merely being imported.

    Note:
        Does not re-import TensorFlow when it is already loaded, and stays silent when TensorFlow is not installed —
        the actionable missing-dependency error is raised later, by
        :func:`~rfdetr.export._tflite.exporter._check_onnx2tf_available`.

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

    if onnx_won_the_race:
        logger.warning(
            "onnx was imported before TensorFlow. Both statically link Abseil and export its symbols weakly, so "
            "TensorFlow can block forever while restoring the SavedModel bundle during TFLite conversion. That order "
            "cannot be repaired once both are loaded. If the export hangs with no output, import tensorflow before "
            "onnx or run the export in a fresh process."
        )
