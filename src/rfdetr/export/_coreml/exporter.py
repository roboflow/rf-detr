# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PyTorch -> CoreML (``.mlpackage``) conversion via ``torch.export`` and coremltools.

Unlike TFLite (ONNX → onnx2tf), native CoreML consumes a :func:`torch.export.export` graph
directly and lowers it with ``coremltools`` to an ``mlprogram`` ``.mlpackage`` suitable for
Xcode / Core ML.  This is distinct from ExecuTorch's ``format="executorch", backend="coreml"``
path, which produces a ``.pte`` for the ExecuTorch runtime.

The model must already be in export mode (``model.export()``) with the rank-≤5 deformable-attention
path — :meth:`rfdetr.detr.RFDETR.export` handles that before :class:`CoreMLExporter` runs.

Note:
    The produced ``.mlpackage`` expects ImageNet mean/std normalization
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``), same as ONNX.
    :class:`CoreMLExporter` defaults to ``compute_precision=FLOAT32`` for tight CPU parity with
    eager PyTorch. Pass ``coremltools.precision.FLOAT16`` (or the string ``"float16"``) when you
    want a smaller ANE-oriented bundle (expect larger numeric drift) — either directly on
    :class:`~rfdetr.export.base.CoreMLConfig`, or via :meth:`rfdetr.detr.RFDETR.export`'s
    ``coreml_precision`` argument (string form only, so callers don't need to import ``coremltools``).

Note:
    ``torch>=2.12`` sharply raises the rate of a CoreML/eager numeric-parity divergence on
    real-image input with coremltools 9.0 (bisected: torch<2.12 failed ~1/6 repeat runs,
    torch>=2.12.0 failed ~6/7) — the ``coreml`` extra pins ``torch<2.12`` in ``pyproject.toml``
    until this is understood/fixed upstream. That pin reduces the failure rate, it does not
    guarantee determinism. See ``tests/export/test_coreml_export.py::TestCoreMLEndToEnd::
    test_outputs_match_pytorch_supervision_image``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._coreml.op_coverage import unsupported_coreml_ops
from rfdetr.export._naming import append_backbone_marker, resolve_export_stem
from rfdetr.export.base import CoreMLConfig, Exporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from coremltools import precision as ct_precision
    from torch.export import ExportedProgram

logger = get_logger()


def _check_coremltools_available(*, raise_error: bool = True) -> bool:
    """Return whether ``coremltools`` is importable."""
    if not _IS_COREMLTOOLS_AVAILABLE:
        if raise_error:
            raise ImportError("CoreML export requires `coremltools`. Install it with: pip install rfdetr[coreml]")
        return False
    return True


@dataclass(frozen=True, slots=True)
class _CoreMLApi:
    """The ``coremltools`` entry points one conversion needs, bound by a single lazy import.

    ``coremltools`` is an optional dependency imported inside the conversion rather than at module scope, and
    ``precision`` is an Enum class whose members are reached by attribute lookup rather than independently
    ``from``-importable. Binding all four here keeps every phase reading the same names instead of repeating
    the import — and keeps the precision members identity-stable across the phases that compare against them.

    Attributes:
        convert: ``coremltools.convert``.
        target: ``coremltools.target``, the deployment-target enum.
        float32: ``coremltools.precision.FLOAT32``.
        float16: ``coremltools.precision.FLOAT16``.
    """

    convert: Callable[..., Any]
    target: Any
    float32: ct_precision
    float16: ct_precision


class CoreMLExporter(Exporter[CoreMLConfig]):
    """Convert a prepared graph to a native CoreML ``.mlpackage`` via ``torch.export`` + ``coremltools``.

    Distinct from ExecuTorch's ``backend="coreml"``, which produces a ``.pte`` instead.

    Examples:
        Requires the optional ``coremltools`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        CoreMLExporter(CoreMLConfig(variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small_fp32.mlpackage')
        ```
    """

    format = "coreml"
    display_name = "CoreML"
    experimental = True
    experimental_note = "Dynamic batch is not supported."
    pip_extra = "coreml"
    notes_reason = "CoreML .mlpackage has no ONNX-style metadata slot"

    def _convert(self, graph: ExportGraph) -> Path:
        """Write the ``.mlpackage`` bundle and return its path.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the exported ``.mlpackage`` bundle. Output tensor names in the saved spec are
            coremltools-inferred, not renamed to ``dets``/``labels``/etc. — consumers must match outputs
            by **position**, in the same order as the model's ONNX ``output_names`` contract.

        Raises:
            ImportError: If ``coremltools`` is not installed, or if ``coremltools.convert`` triggers a
                lazy import of a private submodule that fails even though the top-level package is
                installed (e.g. a partial/ABI-mismatched install).
            NotImplementedError: If the exported graph contains op kinds missing from coremltools'
                Torch registry (fast-fail checklist).
            ValueError: If the configured precision is unrecognized, or if ``torch.export`` /
                ``coremltools.convert`` raises it directly (e.g. invalid shape arguments) — passed
                through unwrapped rather than re-wrapped as ``RuntimeError``.
            RuntimeError: If ``torch.export`` or ``coremltools.convert`` fails for any other reason.
        """
        api = self._import_coreml_api()
        output_dir = self._prepare_output_dir()
        compute_precision = self._resolve_compute_precision(api)
        output_file = self._resolve_output_file(output_dir, compute_precision, api, backbone_only=graph.backbone_only)
        if self.config.verbose:
            logger.info(f"Exporting model to CoreML format: {output_file}")
        mlmodel = self._build_mlmodel(graph, compute_precision, api)
        return self._save_mlmodel(mlmodel, output_file)

    def _import_coreml_api(self) -> _CoreMLApi:
        """Verify ``coremltools`` is installed and bind the entry points the conversion uses.

        Returns:
            The bound ``coremltools`` entry points.

        Raises:
            ImportError: If ``coremltools`` is not installed.
        """
        _check_coremltools_available()
        from coremltools import convert as ct_convert
        from coremltools import precision as ct_precision
        from coremltools import target as ct_target

        return _CoreMLApi(
            convert=ct_convert,
            target=ct_target,
            float32=ct_precision.FLOAT32,
            float16=ct_precision.FLOAT16,
        )

    def _prepare_output_dir(self) -> Path:
        """Create the configured output directory.

        Returns:
            The directory the ``.mlpackage`` bundle is written into.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_compute_precision(self, api: _CoreMLApi) -> ct_precision:
        """Resolve the configured precision to a ``coremltools.precision`` member.

        Args:
            api: The bound ``coremltools`` entry points.

        Returns:
            The configured ``coremltools.precision`` value, the member its string names, or
            ``FLOAT32`` when unset (tight CPU parity with eager PyTorch).

        Raises:
            ValueError: If the configured precision is a string naming neither ``"float32"`` nor
                ``"float16"``.
        """
        compute_precision: Any = self.config.compute_precision
        if compute_precision is None:
            return api.float32
        if isinstance(compute_precision, str):
            try:
                return {"float32": api.float32, "float16": api.float16}[compute_precision]
            except KeyError:
                raise ValueError(
                    f"compute_precision must be 'float32', 'float16', a coremltools.precision value, or "
                    f"None, got {compute_precision!r}"
                ) from None
        return compute_precision

    def _resolve_output_file(
        self,
        output_dir: Path,
        compute_precision: ct_precision,
        api: _CoreMLApi,
        *,
        backbone_only: bool,
    ) -> Path:
        """Name the ``.mlpackage`` bundle from the configured names, the precision, and the graph kind.

        Args:
            output_dir: Directory the bundle is written into.
            compute_precision: The already-resolved coremltools precision.
            api: The bound ``coremltools`` entry points.
            backbone_only: Whether *graph* is a backbone-only export.

        Returns:
            Path to the ``.mlpackage`` bundle this export writes.
        """
        variant_name, output_name = self.config.variant_name, self.config.output_name
        stem, is_custom = resolve_export_stem(
            variant_name, output_name, default="backbone_model" if backbone_only else "inference_model"
        )
        # Precision materially changes the artifact (fp16 has larger numeric drift, per the module
        # docstring) — always encode it, unless the caller asked for an exact custom filename.
        if compute_precision == api.float16:
            precision_token = "fp16"
        elif compute_precision == api.float32:
            precision_token = "fp32"
        else:
            precision_token = "fp32"
            logger.warning(
                f"Unrecognized CoreML compute precision {compute_precision!r}; using the fp32 filename label."
            )
        export_name = stem if is_custom else f"{stem}_{precision_token}"
        # The marker goes on after the precision token: it names a distinct model graph, not a detail of
        # how that graph was lowered.
        export_name = append_backbone_marker(
            export_name, backbone_only=backbone_only, named=bool(variant_name or output_name)
        )
        return output_dir / f"{export_name}.mlpackage"

    def _export_program(self, graph: ExportGraph) -> ExportedProgram:
        """Trace the model into a decomposed ``torch.export`` program.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            The exported program, after ``run_decompositions``.
        """
        model = graph.model.eval()
        # strict=False: same rationale as ExecuTorch — submodule-lifted spatial_shapes constants
        # break lowering under strict=True on current torch.export + converter stacks.
        exported_program = torch.export.export(model, (graph.input_tensors,), strict=False)
        decomposed: ExportedProgram = exported_program.run_decompositions({})
        return decomposed

    @staticmethod
    def _check_op_coverage(exported_program: ExportedProgram) -> None:
        """Fail fast when the graph uses op kinds coremltools has no Torch translation for.

        Args:
            exported_program: The decomposed program about to be converted.

        Raises:
            NotImplementedError: If any op kind in the graph is missing from coremltools' registry.
        """
        # unsupported_coreml_ops() applies ensure_coreml_torch_op_patches() itself before
        # scanning (registry gaps, e.g. aten.alias → coremltools noop, must be patched before
        # the checklist runs) — no separate call needed here.
        coverage = unsupported_coreml_ops(exported_program)
        if coverage:
            summary = f"CoreML op registry gaps: {dict(coverage)}"
            logger.error(summary)
            raise NotImplementedError(
                f"{summary}. Fix these ops before convert (see tests/export/test_coreml_op_coverage.py)."
            )

    def _build_mlmodel(self, graph: ExportGraph, compute_precision: ct_precision, api: _CoreMLApi) -> Any:
        """Trace, screen for op-registry gaps, and lower the graph to a CoreML ``mlprogram``.

        The three steps share one guard: any of them can fail inside the converter stack, and every such
        failure is reported the same way — except the three types callers are expected to handle, which
        keep their own identity instead of being flattened into ``RuntimeError``.

        Args:
            graph: The prepared model and its graph metadata.
            compute_precision: The already-resolved coremltools precision.
            api: The bound ``coremltools`` entry points.

        Returns:
            The converted ``coremltools.models.MLModel``.

        Raises:
            ImportError: If ``coremltools.convert`` triggers a lazy import of a private submodule that
                fails even though the top-level package is installed.
            NotImplementedError: If the exported graph contains op kinds missing from coremltools'
                Torch registry.
            ValueError: If ``torch.export`` or ``coremltools.convert`` raises it directly.
            RuntimeError: If ``torch.export`` or ``coremltools.convert`` fails for any other reason.
        """
        try:
            with torch.no_grad():
                exported_program = self._export_program(graph)
                self._check_op_coverage(exported_program)
                return api.convert(
                    exported_program,
                    convert_to="mlprogram",
                    minimum_deployment_target=api.target.iOS16,
                    compute_precision=compute_precision,
                )
        except (ImportError, NotImplementedError, ValueError):
            # ImportError here is not dead code even though _check_coremltools_available() already
            # verified the top-level `coremltools` package above: ct.convert() lazily imports private
            # submodules (MIL passes, backend components) that can still fail on a partial/ABI-mismatched
            # install even after the top-level import succeeds.
            raise
        except Exception as exc:
            logger.exception("CoreML export failed")
            raise RuntimeError(f"CoreML export failed: {exc}") from exc

    def _save_mlmodel(self, mlmodel: Any, output_file: Path) -> Path:
        """Write the converted model to *output_file*.

        Args:
            mlmodel: The converted ``coremltools.models.MLModel``.
            output_file: Path the ``.mlpackage`` bundle is written to.

        Returns:
            *output_file*, now on disk.
        """
        mlmodel.save(str(output_file))
        if self.config.verbose:
            logger.info(f"Successfully exported CoreML model to: {output_file}")
        return output_file
