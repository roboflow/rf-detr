# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""ONNX export: trace the prepared graph with ``torch.onnx.export`` and embed the optional notes."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

import torch

from rfdetr.export._backend import check_onnx_available as _check_onnx_available
from rfdetr.export._naming import append_backbone_marker, resolve_export_stem
from rfdetr.export.base import ExportConfig, Exporter, serialize_notes, shared_settings
from rfdetr.export.prepare import ExportGraph


@dataclass(frozen=True, slots=True)
class OnnxConfig(ExportConfig):
    """Settings for ``format="onnx"``.

    Attributes:
        opset_version: ONNX opset the graph targets.
    """

    opset_version: int = 17

    @classmethod
    def derive(cls, config: ExportConfig, *, opset_version: int) -> OnnxConfig:
        """Build the intermediate configuration a two-stage format exports through.

        TFLite and TensorRT both write an ONNX graph first and convert it. The intermediate graph inherits every
        setting the ONNX stage understands, so the artifact a two-stage export passes on is named and annotated
        exactly as a direct ``format="onnx"`` export would be.

        Args:
            config: The two-stage format's configuration.
            opset_version: ONNX opset the intermediate graph targets.

        Returns:
            The configuration for the ONNX stage.

        Examples:
            >>> OnnxConfig.derive(ExportConfig(variant_name="rfdetr-small"), opset_version=17).variant_name
            'rfdetr-small'
        """
        return cls(**shared_settings(config), opset_version=opset_version)


class OnnxExporter(Exporter[OnnxConfig]):
    """Export a prepared graph to ONNX.

    The only format with no optional runtime of its own, and the one both two-stage formats export through, so it is
    also the only one supporting a dynamic batch dimension together with embedded *notes* metadata.

    The conversion runs in three steps, each its own method: resolve the output filename, trace the graph with
    ``torch.onnx.export``, then — only when the caller supplied *notes* — reopen the written file to inject them.

    Examples:
        Requires a prepared graph, so this is documentation only (not a doctest):

        ```python
        OnnxExporter(OnnxConfig(output_dir=Path("output"), variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small.onnx')
        ```
    """

    config_class = OnnxConfig
    setting_names = {"opset_version": "opset_version"}
    format = "onnx"
    display_name = "ONNX"
    supports_dynamic_batch = True
    supports_notes = True
    pip_extra = "onnx"

    @classmethod
    def check_dependencies(cls) -> None:
        """Raise the ``rfdetr[onnx]`` install hint when ``onnx`` is missing.

        Raises:
            ImportError: If ``onnx`` is not installed.
        """
        _check_onnx_available()

    def _resolve_output_file(self, *, backbone_only: bool) -> str:
        """Return the path the ``.onnx`` file is written to.

        Args:
            backbone_only: Whether the graph is a backbone-only export.

        Returns:
            Absolute or relative path of the ``.onnx`` file, inside the configured output directory.
        """
        stem, _ = resolve_export_stem(
            self.config.variant_name,
            self.config.output_name,
            default="backbone_model" if backbone_only else "inference_model",
        )
        export_name = append_backbone_marker(
            stem,
            backbone_only=backbone_only,
            named=bool(self.config.variant_name or self.config.output_name),
        )
        return os.path.join(str(self.config.output_dir), f"{export_name}.onnx")

    def _trace(self, graph: ExportGraph, output_file: str) -> None:
        """Trace *graph* with ``torch.onnx.export`` and write the result to *output_file*.

        Args:
            graph: The prepared model and its graph metadata.
            output_file: Destination path for the traced model.
        """
        export_kwargs: dict[str, Any] = {}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            # Torch 2.10+ may default to the dynamo exporter which requires extra deps
            # (e.g. onnxscript). Use the legacy path for compatibility.
            export_kwargs["dynamo"] = False

        input_tensors = graph.input_tensors
        torch.onnx.export(
            graph.model,
            (input_tensors,) if isinstance(input_tensors, torch.Tensor) else tuple(input_tensors),
            output_file,
            input_names=list(graph.input_names),
            output_names=list(graph.output_names),
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            verbose=self.config.verbose,
            opset_version=self.config.opset_version,
            dynamic_axes=graph.dynamic_axes,
            **export_kwargs,
        )

    def _embed_notes(self, output_file: str) -> None:
        """Write the configured *notes* into the already-exported file's ``rfdetr_notes`` metadata property.

        ``torch.onnx.export`` writes to disk only and hands back no in-memory handle, so the model is reloaded and
        resaved (~1-2 s on large models). Does nothing when no notes were supplied.

        Args:
            output_file: Path of the exported model to annotate.
        """
        if self.config.notes is None:
            return
        # Imported here, not at module load: loading this exporter imports no ONNX package, so a TFLite export can load
        # TensorFlow first (issue #1322), and an `onnx` installed after a refused export is found without a restart.
        import onnx

        onnx_model = onnx.load(output_file)
        notes_value = serialize_notes(self.config.notes)
        existing = next((prop for prop in onnx_model.metadata_props if prop.key == "rfdetr_notes"), None)
        if existing is not None:
            existing.value = notes_value
        else:
            meta = onnx_model.metadata_props.add()
            meta.key = "rfdetr_notes"
            meta.value = notes_value
        onnx.save(onnx_model, output_file)

    def _convert(self, graph: ExportGraph) -> str:
        """Write the ``.onnx`` file and return its path."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_file = self._resolve_output_file(backbone_only=graph.backbone_only)
        # Composed TFLite/TensorRT exporters route their intermediate ONNX through here before creating their own
        # output directory, so this stage cannot rely on a caller having made it.
        os.makedirs(str(self.config.output_dir), exist_ok=True)
        self._trace(graph, output_file)
        self._embed_notes(output_file)
        return output_file
