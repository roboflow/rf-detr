# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PyTorch -> Apple Core AI (``.aimodel``) conversion via ``torch.export`` and ``coreai-torch``.

Core AI is Apple's on-device inference framework for iOS, iPadOS and macOS 27 and later. Like the native CoreML path,
it consumes a :func:`torch.export.export` graph directly — no ONNX step — and ``coreai-torch`` lowers it to an
``.aimodel`` asset that the Core AI runtime specializes for the CPU, GPU or Neural Engine when it is loaded.

The graph needs two entries on top of ``coreai_torch.get_decomp_table()``, see
:mod:`rfdetr.export._coreai.decompositions`: the deformable attention's ``aten.grid_sampler`` — and the
``aten.grid_sampler_2d`` a base table rewrites it into — has no Core AI lowering, and a float16 ``aten.topk`` runs in
float32, because the Neural Engine's float16 ``topk`` returns corrupt indices and the two-stage query selection would
gather the wrong encoder tokens.

Note:
    The ``.aimodel`` keeps the contract of the other formats: a fixed ``[batch, 3, H, W]`` float input named
    ``input`` that the caller has resized and ImageNet-normalized (``mean=[0.485, 0.456, 0.406]``,
    ``std=[0.229, 0.224, 0.225]``). Unlike CoreML, the outputs keep their names (``dets``, ``labels``, and
    ``masks`` or ``keypoints``), so consumers can look them up by name. A ``precision="float16"`` export also takes
    and returns float16 tensors.

Note:
    Converting only needs ``coreai-torch`` (macOS 26+ on Apple silicon, or Linux x86-64, Python 3.11-3.13). Loading
    and running the ``.aimodel`` needs the Core AI runtime, which ships with iOS, iPadOS and macOS 27.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rfdetr.export._coreai import _IS_COREAI_TORCH_AVAILABLE
from rfdetr.export._coreai.decompositions import coreai_decomposition_table
from rfdetr.export._naming import append_backbone_marker, resolve_export_stem
from rfdetr.export.base import ExportConfig, Exporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import get_version

if TYPE_CHECKING:
    from torch.export import ExportedProgram

logger = get_logger()

#: Precision name -> (torch dtype the graph is traced in, filename token).
_PRECISIONS: dict[str, tuple[torch.dtype, str]] = {
    "float32": (torch.float32, "fp32"),
    "float16": (torch.float16, "fp16"),
}


def _check_coreai_torch_available(*, raise_error: bool = True) -> bool:
    """Return whether ``coreai-torch`` is importable.

    Args:
        raise_error: Raise an :class:`ImportError` with the install command instead of returning ``False``.

    Returns:
        Whether ``coreai-torch`` is installed.

    Raises:
        ImportError: If ``coreai-torch`` is missing and *raise_error* is set.
    """
    if not _IS_COREAI_TORCH_AVAILABLE:
        if raise_error:
            raise ImportError(
                "Core AI export requires `coreai-torch` (Python 3.11-3.13)."
                ' Install it with: pip install "rfdetr[coreai]"'
            )
        return False
    return True


@dataclass(frozen=True, slots=True)
class CoreAIConfig(ExportConfig):
    """Settings for ``format="coreai"``.

    Attributes:
        precision: ``"float32"`` (default when ``None``) or ``"float16"``, the dtype the graph is traced and stored in.
    """

    precision: str | None = None


class CoreAIExporter(Exporter[CoreAIConfig]):
    """Convert a prepared graph to an Apple Core AI ``.aimodel`` via ``torch.export`` + ``coreai-torch``.

    Examples:
        Requires the optional ``coreai-torch`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        CoreAIExporter(CoreAIConfig(variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small_fp32.aimodel')
        ```
    """

    config_class = CoreAIConfig
    setting_names = {"precision": "coreai_precision"}
    format = "coreai"
    display_name = "Core AI"
    dynamic_batch_reason = "(the asset bakes a fixed input shape). Export one .aimodel per batch size instead."
    supports_notes = True
    experimental = True
    experimental_note = "The .aimodel runs on iOS, iPadOS and macOS 27 or later."
    pip_extra = "coreai"

    def _convert(self, graph: ExportGraph) -> Path:
        """Write the ``.aimodel`` asset and return its path.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the ``.aimodel`` asset directory.

        Raises:
            ImportError: If ``coreai-torch``, or the ``coreai`` runtime package the asset metadata comes from,
                is not installed.
            ValueError: If the configured precision is not ``"float32"`` or ``"float16"``.
            RuntimeError: If ``torch.export`` or the Core AI conversion fails. Writing the asset is not covered:
                a failing ``save_asset`` raises whatever the Core AI runtime raises.
        """
        _check_coreai_torch_available()
        dtype, precision_token = self._resolve_precision()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self._export_name(precision_token, backbone_only=graph.backbone_only)}.aimodel"
        if self.config.verbose:
            logger.info(f"Exporting model to Core AI format: {output_path}")
        # Built first: it imports the `coreai` runtime distribution, which `_check_coreai_torch_available` does not
        # cover. A missing one must not surface only after a full trace, conversion and optimize().
        metadata = self._asset_metadata()
        program = self._build_program(graph, dtype)
        program.save_asset(output_path, metadata=metadata)
        return output_path

    def _resolve_precision(self) -> tuple[torch.dtype, str]:
        """Resolve the configured precision to the traced dtype and the filename token.

        Returns:
            ``(dtype, token)``, float32 when no precision was configured.

        Raises:
            ValueError: If the precision is neither ``"float32"`` nor ``"float16"``.
        """
        precision = self.config.precision or "float32"
        try:
            return _PRECISIONS[precision]
        except KeyError:
            raise ValueError(f"precision must be 'float32', 'float16', or None, got {precision!r}") from None

    def _export_name(self, precision_token: str, *, backbone_only: bool) -> str:
        """Resolve the artifact's filename stem.

        Args:
            precision_token: ``"fp32"`` or ``"fp16"``, appended unless the caller chose the exact name.
            backbone_only: Whether the graph is a backbone-only export.

        Returns:
            The artifact name, without extension.
        """
        stem, is_custom = resolve_export_stem(
            self.config.variant_name,
            self.config.output_name,
            default="backbone_model" if backbone_only else "inference_model",
        )
        if not is_custom:
            stem = f"{stem}_{precision_token}"
        return append_backbone_marker(
            stem, backbone_only=backbone_only, named=bool(self.config.variant_name or self.config.output_name)
        )

    def _export_program(self, graph: ExportGraph, dtype: torch.dtype) -> ExportedProgram:
        """Trace the model in *dtype* and decompose it into ops ``coreai-torch`` can lower.

        Args:
            graph: The prepared model and its graph metadata.
            dtype: Dtype the model and its example input are traced in.

        Returns:
            The decomposed exported program.

        Raises:
            NotImplementedError: If a grid-sampling op outlives the decomposition table.
        """
        from coreai_torch import get_decomp_table

        model = graph.model.eval().to(dtype)
        example = graph.input_tensors.to(dtype)
        # strict=False: same rationale as CoreML and ExecuTorch — submodule-lifted spatial_shapes constants
        # break lowering under strict=True.
        exported_program = torch.export.export(model, (example,), strict=False)
        decomposed: ExportedProgram = exported_program.run_decompositions(
            coreai_decomposition_table(get_decomp_table())
        )
        # A surviving grid sampler would otherwise fail deep inside the converter, long after this is fixable here.
        survivors = sorted(
            {
                str(node.target)
                for node in decomposed.graph.nodes
                if node.op == "call_function" and str(node.target).startswith("aten.grid_sampler")
            }
        )
        if survivors:
            raise NotImplementedError(
                f"The decomposition table left {', '.join(survivors)} in the graph and Core AI cannot lower grid"
                " sampling. This needs a decomposition entry for that op in rfdetr.export._coreai.decompositions;"
                " please report it with your coreai-torch version."
            )
        return decomposed

    def _build_program(self, graph: ExportGraph, dtype: torch.dtype) -> Any:
        """Trace, decompose and lower the graph to an optimized Core AI program.

        Args:
            graph: The prepared model and its graph metadata.
            dtype: Dtype the graph is traced in.

        Returns:
            The optimized ``coreai`` program, ready for ``save_asset``.

        Raises:
            ImportError: If a lazily imported part of the Core AI stack fails to load.
            NotImplementedError: If the graph needs a grid-sampling mode the decomposition does not cover, or a
                grid-sampling op outlives the decomposition table.
            ValueError: If ``torch.export`` or ``coreai-torch`` rejects the graph with it (for example an
                unsupported ATen op).
            RuntimeError: If the export or conversion fails for any other reason.
        """
        from coreai_torch import TorchConverter

        try:
            with torch.no_grad():
                exported_program = self._export_program(graph, dtype)
                program = (
                    TorchConverter()
                    .add_exported_program(
                        exported_program,
                        input_names=list(graph.input_names),
                        output_names=list(graph.output_names),
                    )
                    .to_coreai()
                )
            program.optimize()
        except (ImportError, NotImplementedError, ValueError):
            raise
        except Exception as exc:
            logger.exception("Core AI export failed")
            raise RuntimeError(f"Core AI export failed: {exc}") from exc
        return program

    def _asset_metadata(self) -> Any:
        """Build the asset metadata, embedding *notes* under the key the ONNX export uses.

        Returns:
            A ``coreai.runtime.AIModelAssetMetadata``.
        """
        from coreai.runtime import AIModelAssetMetadata

        metadata = AIModelAssetMetadata()
        metadata.model_description = "RF-DETR real-time detection transformer (https://github.com/roboflow/rf-detr)"
        rfdetr_version = get_version()
        if rfdetr_version is not None:
            metadata.set_custom("rfdetr_version", rfdetr_version)
        notes = self.config.notes
        if notes is not None:
            metadata.set_custom("rfdetr_notes", notes if isinstance(notes, str) else json.dumps(notes, allow_nan=False))
        return metadata
