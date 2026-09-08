# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Direct PyTorch → OpenVINO IR export."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rfdetr.export._naming import append_backbone_marker, resolve_export_stem
from rfdetr.export.base import Exporter, OpenVINOConfig
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _check_openvino_available() -> None:
    """Verify that ``openvino`` is importable.

    Shared by :class:`OpenVINOExporter` and :class:`~rfdetr.export._openvino.inference.OpenVINOInference`
    so both surface the same actionable message and tests can monkeypatch a single choke point instead
    of relying on ``openvino`` actually being absent from the environment.

    Raises:
        ImportError: If ``openvino`` cannot be imported.
    """
    try:
        import openvino  # noqa: F401
    except ImportError as error:
        raise ImportError('OpenVINO requires `openvino`. Install it with: pip install "rfdetr[openvino]"') from error


class ModelWrapper(nn.Module):
    """Normalize an export-mode RF-DETR forward into a plain tensor tuple for ``openvino.convert_model``.

    The wrapped *model* must already be switched into export mode by the caller (see
    :class:`OpenVINOExporter`) -- ``forward_export`` always returns a
    tuple of tensors for the full detector (``(dets, labels)`` / ``(dets, labels, masks)`` /
    ``(dets, labels, keypoints)``), and the backbone-only export graph
    (:class:`rfdetr.export._backend._BackboneExport`) returns a plain list of tensors. A dict output
    only reaches this wrapper when the caller forgot the mode-switch -- that is a caller bug, not a
    shape this wrapper can flatten, so it raises instead of silently dropping keys.
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
                f"OpenVINO export received a dict-valued model output (keys={sorted(output)}); this "
                "means the model was not switched into export mode before wrapping (forward_export "
                "always returns a tuple/list). Call model.export() before wrapping."
            )
        raise TypeError(f"Unsupported model output type for OpenVINO export: {type(output)!r}")


class OpenVINOExporter(Exporter[OpenVINOConfig]):
    """Convert a prepared graph straight to OpenVINO IR, with no ONNX step in between.

    The conversion runs as a short sequence of steps, each its own method: import OpenVINO's converters, create
    the output directory, resolve the artifact name, translate *precision* into OpenVINO's compression flag,
    wrap the graph for tracing, then convert and save the ``.xml``/``.bin`` pair.

    Note:
        Output tensor names in the saved IR are OpenVINO-inferred, not renamed to ``dets``/``labels``/etc. --
        consumers must match outputs by **position**, mirroring the CoreML export's naming limitation.

    Examples:
        Requires the optional ``openvino`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        OpenVINOExporter(OpenVINOConfig(variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small.xml')
        ```
    """

    format = "openvino"
    display_name = "OpenVINO"
    pip_extra = "openvino"
    notes_reason = "OpenVINO IR has no ONNX-style metadata slot"

    def _import_converters(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """Verify ``openvino`` is installed and return the two entry points the conversion needs.

        Imported here rather than at module scope so ``format="openvino"`` is the only export path that pays
        for the (optional, heavy) dependency.

        Returns:
            The ``(convert_model, save_model)`` pair from the ``openvino`` package.

        Raises:
            ImportError: If ``openvino`` is not installed.
        """
        _check_openvino_available()
        from openvino import convert_model, save_model

        return convert_model, save_model

    def _prepare_output_dir(self) -> Path:
        """Create the configured output directory if it does not exist yet.

        Returns:
            The directory the ``.xml``/``.bin`` pair is written into.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_export_name(self, *, backbone_only: bool) -> str:
        """Resolve the filename stem the ``.xml`` and its companion ``.bin`` share.

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

    def _resolve_compress_to_fp16(self) -> bool:
        """Translate the configured *precision* into ``save_model``'s ``compress_to_fp16`` flag.

        ``compress_to_fp16`` controls IR *storage* precision only -- execution precision still depends on the
        compiled device and is not guaranteed to match eager PyTorch on non-CPU devices. ``None`` keeps
        OpenVINO's own compressing default, which ``"float16"`` states explicitly.

        Returns:
            Whether weights are stored compressed to FP16.

        Raises:
            ValueError: If *precision* is not ``"float32"``, ``"float16"``, or ``None``.
        """
        precision = self.config.precision
        if precision is None or precision == "float16":
            return True
        if precision == "float32":
            return False
        raise ValueError(f"precision must be 'float32', 'float16', or None, got {precision!r}")

    def _prepare_module_for_tracing(self, graph: ExportGraph) -> tuple[ModelWrapper, torch.Tensor]:
        """Announce the conversion, then move the graph onto CPU and wrap it for ``convert_model``.

        ``convert_model`` traces on CPU, and :class:`ModelWrapper` normalizes the export-mode forward's
        tuple/list output into the plain tensor tuple the converter accepts.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            The wrapped model and the example input tensor, both on CPU and in eval mode.
        """
        if self.config.verbose:
            logger.info("Converting PyTorch model to OpenVINO IR...")
            logger.info(f"Input shape: {graph.input_tensors.shape}")

        model = graph.model.eval().cpu()
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        return wrapped_model, graph.input_tensors.cpu()

    def _convert_and_save(
        self,
        converters: tuple[Callable[..., Any], Callable[..., Any]],
        wrapped_model: ModelWrapper,
        input_tensors: torch.Tensor,
        output_xml: Path,
        *,
        compress_to_fp16: bool,
    ) -> None:
        """Trace *wrapped_model* into an IR graph and write it to *output_xml* (plus its ``.bin``).

        Args:
            converters: The ``(convert_model, save_model)`` pair from :meth:`_import_converters`.
            wrapped_model: The CPU, eval-mode module to trace.
            input_tensors: Example input the conversion traces with.
            output_xml: Destination path for the IR graph; the weights land beside it as ``.bin``.
            compress_to_fp16: Whether to store the weights compressed to FP16.

        Raises:
            ImportError: If ``convert_model``'s lazy submodule imports fail on a partial/ABI-mismatched install.
            NotImplementedError: If the model was not switched into export mode first (see :class:`ModelWrapper`).
            TypeError: If the model's forward returns an output type :class:`ModelWrapper` cannot wrap.
            ValueError: If the conversion rejects an argument.
            RuntimeError: If conversion or saving otherwise fails.
        """
        convert_model, save_model = converters
        try:
            with torch.no_grad():
                ov_model = convert_model(wrapped_model, example_input=input_tensors)
            save_model(ov_model, str(output_xml), compress_to_fp16=compress_to_fp16)
        except (ImportError, NotImplementedError, TypeError, ValueError):
            # ImportError: convert_model lazily imports private submodules that can still fail on a
            # partial/ABI-mismatched install even after the top-level `openvino` import succeeded.
            # NotImplementedError/TypeError: raised by ModelWrapper.forward's own documented contract
            # (dict output / unsupported output type) -- must reach the caller as-is, not be relabeled
            # RuntimeError by the broad except below. Mirrors the CoreML exporter's passthrough tier.
            raise
        except Exception as e:
            logger.exception("OpenVINO export failed")
            raise RuntimeError(f"Failed to export model to OpenVINO IR: {e}") from e

    def _convert(self, graph: ExportGraph) -> str:
        """Write the ``.xml``/``.bin`` IR pair and return the ``.xml`` path."""
        converters = self._import_converters()
        output_dir = self._prepare_output_dir()
        export_name = self._resolve_export_name(backbone_only=graph.backbone_only)
        output_xml = output_dir / f"{export_name}.xml"
        output_bin = output_dir / f"{export_name}.bin"
        compress_to_fp16 = self._resolve_compress_to_fp16()

        wrapped_model, input_tensors = self._prepare_module_for_tracing(graph)
        self._convert_and_save(converters, wrapped_model, input_tensors, output_xml, compress_to_fp16=compress_to_fp16)

        if self.config.verbose:
            logger.info(f"✓ OpenVINO IR model saved to {output_xml}")
            logger.info(f"✓ Model binary saved to {output_bin}")
        return str(output_xml)
