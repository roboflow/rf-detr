# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Which exporter class implements which export format, and how to reach it without importing the others.

Every format except ONNX sits behind an optional dependency — ``executorch``, ``coremltools``, ``openvino``,
``onnx2tf``/``tensorflow``, ``tensorrt`` — and none of them may be imported by ``import rfdetr``. The registry is
therefore *data*: a format maps to the dotted path of its exporter class, and only :func:`resolve_exporter` imports it.
Adding a format is one entry here plus the module it names, which owns both the exporter class and the configuration
dataclass it is built from.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.export.base import Exporter

logger = get_logger()


@dataclass(frozen=True, slots=True)
class ExporterEntry:
    """Where one format's exporter class lives, what installs it, and the little that must be known before importing it.

    ``label``, ``supports_dynamic_batch`` and ``dynamic_batch_reason`` mirror the exporter class's own
    ``display_name``, ``supports_dynamic_batch`` and ``dynamic_batch_reason``. The duplication is deliberate and
    test-enforced (``tests/export/test_registry.py``): all three are needed *before* the format's module — and with it
    its heavy optional dependency — is imported, so a doomed request can be refused, in the format's own words, without
    paying for ``coremltools`` or TensorFlow.

    Attributes:
        module: Dotted path of the module defining the exporter class.
        attribute: Name of the exporter class within *module*.
        pip_extra: The ``rfdetr[...]`` extra that installs the format's dependencies, or ``None`` when it needs none.
        label: How the format is spelled in messages addressed to users.
        supports_dynamic_batch: Whether the format can bake a dynamic batch dimension into its artifact.
        dynamic_batch_reason: Why a fixed-batch format cannot honour ``dynamic_batch``, and what to do instead.
            Empty for formats that support it. Mirrors the exporter class's own ``dynamic_batch_reason`` so the
            pre-import refusal and the one the constructed exporter raises read identically.
        preimport: ``"module:function"`` to call before importing *module*, or ``None``. TFLite needs one: TensorFlow
            has to be loaded before anything pulls in ONNX's C extension, and importing the exporter's own package
            already reaches third-party code that could load ONNX first.
    """

    module: str
    attribute: str
    pip_extra: str | None
    label: str
    supports_dynamic_batch: bool = False
    dynamic_batch_reason: str = ""
    preimport: str | None = None


#: Every format :meth:`rfdetr.detr.RFDETR.export` accepts, mapped to the exporter that writes it.
REGISTRY: Mapping[str, ExporterEntry] = {
    "onnx": ExporterEntry("rfdetr.export._onnx.exporter", "OnnxExporter", "onnx", "ONNX", supports_dynamic_batch=True),
    "tflite": ExporterEntry(
        "rfdetr.export._tflite.exporter",
        "TFLiteExporter",
        "tflite",
        "TFLite",
        supports_dynamic_batch=True,
        preimport="rfdetr.export._backend:preload_tensorflow_before_onnx",
    ),
    "tensorrt": ExporterEntry(
        "rfdetr.export._tensorrt.exporter",
        "TensorRTExporter",
        "tensorrt",
        "TensorRT",
        dynamic_batch_reason=(
            "(the engine is compiled without a TensorRT optimization profile, so it accepts only the exported batch"
            " size). Export one engine per batch size instead."
        ),
    ),
    "executorch": ExporterEntry(
        "rfdetr.export._executorch.exporter",
        "ExecuTorchExporter",
        "executorch",
        "ExecuTorch",
        dynamic_batch_reason="(see the ExecuTorch exporter for details). Export one .pte per batch size instead.",
    ),
    "coreml": ExporterEntry(
        "rfdetr.export._coreml.exporter",
        "CoreMLExporter",
        "coreml",
        "CoreML",
        dynamic_batch_reason=(
            "(fixed shapes are required for reliable ANE / GPU scheduling)."
            " Export one .mlpackage per batch size instead."
        ),
    ),
    "openvino": ExporterEntry(
        "rfdetr.export._openvino.exporter",
        "OpenVINOExporter",
        "openvino",
        "OpenVINO",
        dynamic_batch_reason="(the IR graph bakes a fixed input shape). Export one model per batch size instead.",
    ),
}

#: Short spellings accepted for a format, mapped to the canonical name.
ALIASES: Mapping[str, str] = {"trt": "tensorrt", "pte": "executorch"}


def normalize_format(format: str) -> str:
    """Resolve an alias to the canonical format name.

    Args:
        format: Format name or alias as the caller spelled it.

    Returns:
        The canonical format name, unchanged when *format* is already canonical.

    Examples:
        >>> normalize_format("trt")
        'tensorrt'
        >>> normalize_format("onnx")
        'onnx'
    """
    return ALIASES.get(format, format)


def resolve_exporter(format: str) -> type[Exporter[Any]]:
    """Import and return the exporter class implementing *format*.

    Args:
        format: Canonical format name (pass it through :func:`normalize_format` first).

    Returns:
        The exporter class registered for *format*.

    Raises:
        ValueError: If *format* is not a known export format.
        ImportError: If the format's optional dependencies are not installed.

    Examples:
        >>> resolve_exporter("onnx").format
        'onnx'
        >>> sorted(REGISTRY)
        ['coreml', 'executorch', 'onnx', 'openvino', 'tensorrt', 'tflite']
        >>> resolve_exporter("nonesuch")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Unsupported export format 'nonesuch'.
    """
    entry = require_entry(format)
    if entry.preimport is not None:
        preimport_module, _, preimport_attribute = entry.preimport.partition(":")
        getattr(importlib.import_module(preimport_module), preimport_attribute)()
    try:
        module = importlib.import_module(entry.module)
    except ImportError:
        if entry.pip_extra is not None:
            logger.error(
                f"It seems some dependencies for {entry.label} export are missing."
                f" Please run `pip install rfdetr[{entry.pip_extra}]` and try again.",
            )
        raise
    exporter_class: type[Exporter[Any]] = getattr(module, entry.attribute)
    return exporter_class


def require_entry(format: str) -> ExporterEntry:
    """Look up *format*'s registry entry without importing anything.

    Args:
        format: Canonical format name.

    Returns:
        The registry entry for *format*.

    Raises:
        ValueError: If *format* is not a known export format.

    Examples:
        >>> require_entry("tensorrt").label
        'TensorRT'
    """
    entry = REGISTRY.get(format)
    if entry is None:
        raise ValueError(f"Unsupported export format {format!r}. Choose from: {sorted(REGISTRY)}.")
    return entry
