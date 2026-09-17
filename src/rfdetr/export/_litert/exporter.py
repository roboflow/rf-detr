# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Direct PyTorch -> LiteRT (``.tflite``) export via ``torch.export`` + ``litert-torch``.

LiteRT (formerly TensorFlow Lite) is Google's on-device runtime.  This route hands the export-mode RF-DETR module to
``litert_torch.convert``, which captures it with :func:`torch.export.export` and lowers the captured graph to a
``.tflite`` flatbuffer itself -- no ONNX and no TensorFlow step, unlike ``format="tflite"``
(:mod:`rfdetr.export._tflite`), which goes PyTorch -> ONNX -> ``onnx2tf`` -> TFLite.  Both routes produce a
``.tflite`` that the ``ai_edge_litert`` interpreter runs.

The exported graph is the full detector in one file, two-stage query selection (``topk``/``gather``) included, so it
runs on the CPU (XNNPACK) delegate.  On CPU the pretrained Nano / Seg-Nano exports track eager PyTorch to ~1e-7
(boxes) and ~3e-5 (class logits, mask probabilities) over the confident queries, and to ~2e-3 at worst over all 300 raw
mask logits; the ``e2e_litert`` test suite gates on looser regression bounds (see its docstrings for the measurements).
Keypoint models are not supported on litert-torch 0.9.4: its converter rejects the rank-4 ``batch_matmul`` that the
keypoint head's ``nn.Linear`` lowers to.

Note:
    The produced ``.tflite`` expects the same input normalization as the ONNX export: ImageNet mean/std
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``), NCHW float32.  Output tensor names are
    litert-torch's own (``serving_default_output_<i>_output``); consumers match outputs by **position**, mirroring the
    CoreML and OpenVINO exports.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from rfdetr.export._naming import append_backbone_marker, resolve_export_stem
from rfdetr.export.base import ExportConfig, Exporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_INSTALL_HINT = 'LiteRT export requires `litert-torch`. Install it with: pip install "rfdetr[litert]"'


def _check_litert_available() -> None:
    """Verify that ``litert_torch`` is importable.

    Shared by :class:`LiteRTExporter` and the package-level ``_IS_LITERT_AVAILABLE`` flag so both surface the same
    actionable message, and so tests can monkeypatch a single choke point instead of relying on ``litert_torch``
    actually being absent from the environment.

    Raises:
        ImportError: If ``litert_torch`` cannot be imported.
    """
    try:
        import litert_torch  # noqa: F401
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error


@dataclass(frozen=True, slots=True)
class LiteRTConfig(ExportConfig):
    """Settings for ``format="litert"``.

    This route has no format-specific knob yet: it always writes one float32 ``.tflite`` with the shape baked in.
    ``RFDETR.export``'s ``quantization`` keyword is validated rather than stored -- see
    :meth:`LiteRTExporter._format_settings` -- so a value this route cannot honour is refused instead of being
    silently ignored.
    """


class ModelWrapper(nn.Module):
    """Normalize an export-mode RF-DETR forward into a plain tensor tuple for ``litert_torch.convert``.

    The wrapped *model* must already be switched into export mode by the caller (:meth:`Exporter.__call__` does this
    before :meth:`LiteRTExporter._convert` runs) -- ``forward_export`` always returns a tuple of tensors for the full
    detector (``(dets, labels)`` / ``(dets, labels, masks)``), and the backbone-only export graph
    (:class:`rfdetr.export._backend._BackboneExport`) returns a plain list of tensors.  A dict output only reaches this
    wrapper when the caller forgot the mode-switch -- that is a caller bug, not a shape this wrapper can flatten, so it
    raises instead of silently dropping keys.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            return tuple(output)
        if isinstance(output, dict):
            raise NotImplementedError(
                f"LiteRT export received a dict-valued model output (keys={sorted(output)}); this means the model "
                "was not switched into export mode before wrapping (forward_export always returns a tuple/list). "
                "Call model.export() before wrapping."
            )
        raise TypeError(f"Unsupported model output type for LiteRT export: {type(output)!r}")


class LiteRTExporter(Exporter[LiteRTConfig]):
    """Convert a prepared graph straight to a LiteRT ``.tflite``, with no ONNX or TensorFlow step in between.

    The conversion runs as a short sequence of steps, each its own method: import ``litert_torch``, create the
    output directory, resolve the artifact name, wrap the graph for capture, then convert and write the ``.tflite``.
    Mirrors :class:`~rfdetr.export._openvino.exporter.OpenVINOExporter`, the other direct-from-PyTorch route.

    Examples:
        Requires the optional ``litert-torch`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        LiteRTExporter(LiteRTConfig(variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small.tflite')
        ```
    """

    config_class = LiteRTConfig
    format = "litert"
    display_name = "LiteRT"
    dynamic_batch_reason = "(the .tflite bakes a fixed input shape). Export one model per batch size instead."
    experimental = True
    experimental_note = "Upstream dependency instabilities (litert-torch) may affect results."
    pip_extra = "litert"
    notes_reason = "a .tflite flatbuffer has no ONNX-style metadata slot"

    @classmethod
    def _format_settings(cls, settings: Mapping[str, Any]) -> dict[str, Any]:
        """Refuse a ``quantization`` this route cannot apply; the configuration itself stores nothing.

        A ``.tflite`` caller may reasonably expect ``quantization`` to apply, so any value other than ``None`` /
        ``"fp32"`` is rejected here -- at :meth:`~rfdetr.export.base.Exporter.build_config` time, before the caller
        pays for a forward pass -- rather than silently writing a float32 file.

        Args:
            settings: The keyword arguments ``RFDETR.export`` was called with.

        Returns:
            An empty mapping: :class:`LiteRTConfig` declares no format-specific field.

        Raises:
            NotImplementedError: If *quantization* is anything but ``None`` / ``"fp32"``.
        """
        quantization = settings.get("quantization")
        if quantization not in (None, "fp32"):
            raise NotImplementedError(
                f"LiteRT export writes a float32 .tflite; quantization={quantization!r} is not supported on this "
                "route yet. Use format='tflite' for its fp16/int8 modes, or quantize the exported file with "
                "ai-edge-quantizer."
            )
        return {}

    def _import_converter(self) -> ModuleType:
        """Verify ``litert_torch`` is installed and return the module the conversion calls into.

        Imported here rather than at module scope so ``format="litert"`` is the only export path that pays for the
        (optional, heavy) dependency.

        Returns:
            The imported ``litert_torch`` package.

        Raises:
            ImportError: If ``litert_torch`` is not installed.
        """
        _check_litert_available()
        return importlib.import_module("litert_torch")

    def _prepare_output_dir(self) -> Path:
        """Create the configured output directory if it does not exist yet.

        Returns:
            The directory the ``.tflite`` is written into.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_export_name(self, *, backbone_only: bool) -> str:
        """Resolve the ``.tflite`` filename stem.

        Args:
            backbone_only: Whether the graph is a backbone-only export.

        Returns:
            The artifact name, without extension.
        """
        stem, _ = resolve_export_stem(
            self.config.variant_name,
            self.config.output_name,
            default="backbone_model" if backbone_only else "inference_model",
        )
        return append_backbone_marker(
            stem,
            backbone_only=backbone_only,
            named=bool(self.config.variant_name or self.config.output_name),
        )

    def _prepare_module_for_tracing(self, graph: ExportGraph) -> tuple[ModelWrapper, torch.Tensor]:
        """Announce the conversion, then move the graph onto CPU and wrap it for ``litert_torch.convert``.

        ``litert_torch.convert`` captures on CPU, and :class:`ModelWrapper` normalizes the export-mode forward's
        tuple/list output into the plain tensor tuple the converter accepts.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            The wrapped model and the example input tensor, both on CPU and in eval mode.
        """
        if self.config.verbose:
            logger.info("Converting PyTorch model to LiteRT (.tflite) with litert-torch...")
            logger.info(f"Input shape: {tuple(graph.input_tensors.shape)}")

        model = graph.model.eval().cpu()
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        return wrapped_model, graph.input_tensors.cpu()

    def _convert_and_save(
        self,
        litert_torch: ModuleType,
        wrapped_model: ModelWrapper,
        input_tensors: torch.Tensor,
        output_file: Path,
    ) -> None:
        """Capture *wrapped_model* with ``litert_torch.convert`` and write the ``.tflite`` to *output_file*.

        Args:
            litert_torch: The imported ``litert_torch`` package from :meth:`_import_converter`.
            wrapped_model: The CPU, eval-mode module to capture.
            input_tensors: Example input the capture traces with; its shape is baked into the file.
            output_file: Destination path for the ``.tflite``.

        Raises:
            ImportError: If litert-torch's lazily imported converter/quantizer companions fail on a partial install.
            NotImplementedError: If the model was not switched into export mode first (see :class:`ModelWrapper`).
            TypeError: If the model's forward returns an output type :class:`ModelWrapper` cannot wrap.
            RuntimeError: If ``torch.export`` capture, lowering, or writing the file otherwise fails.
        """
        try:
            with torch.no_grad():
                edge_model = litert_torch.convert(wrapped_model, (input_tensors,))
                edge_model.export(str(output_file))
        except (ImportError, NotImplementedError, TypeError):
            # ImportError: litert-torch lazily imports its converter/quantizer companions, which can still fail on a
            # partial install after the top-level import succeeded.  NotImplementedError/TypeError: raised by
            # ModelWrapper.forward's own documented contract (dict output / unsupported output type) -- must reach
            # the caller as-is, not be relabeled RuntimeError by the broad except below.  Mirrors the OpenVINO
            # exporter's passthrough tier.
            raise
        except Exception as e:
            logger.exception("LiteRT export failed")
            raise RuntimeError(f"Failed to export model to LiteRT: {e}") from e

    def _convert(self, graph: ExportGraph) -> Path:
        """Write the ``.tflite`` and return its path."""
        litert_torch = self._import_converter()
        output_dir = self._prepare_output_dir()
        output_file = output_dir / f"{self._resolve_export_name(backbone_only=graph.backbone_only)}.tflite"

        wrapped_model, input_tensors = self._prepare_module_for_tracing(graph)
        self._convert_and_save(litert_torch, wrapped_model, input_tensors, output_file)

        if self.config.verbose:
            logger.info(f"✓ LiteRT model saved to {output_file}")
        return output_file
