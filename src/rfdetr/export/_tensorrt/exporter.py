# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""TensorRT export helper: build a serialized engine from ONNX in-process.

The engine is built with the TensorRT Python API (via `polygraphy`), so no
``trtexec`` binary on ``PATH`` is required — only ``pip install rfdetr[tensorrt]``.

For TensorRT *inference*, use the ``inference-models`` library which provides
multi-backend RF-DETR support (PyTorch, ONNX, TensorRT) with automatic backend
selection::

    from inference_models import AutoModel

    model = AutoModel.from_pretrained("rfdetr-small")

See https://github.com/roboflow/inference/tree/main/inference_models for details.
"""

from __future__ import annotations

import os

from rfdetr.export._naming import resolve_export_stem
from rfdetr.export.base import Exporter, TensorRTConfig
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# polygraphy ships in the ``rfdetr[tensorrt]`` extra alongside ``tensorrt``. Import it
# lazily at module scope (guarded) so importing this module never fails on hosts
# without TensorRT, and so tests can monkeypatch these names without polygraphy
# installed.
try:
    from polygraphy.backend.trt import (
        CreateConfig,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )

    _IS_TENSORRT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via TensorRTExporter._require_tensorrt
    CreateConfig = None
    engine_from_network = None
    network_from_onnx_path = None
    save_engine = None

    _IS_TENSORRT_AVAILABLE = False


class TensorRTExporter(Exporter[TensorRTConfig]):
    """Export to TensorRT by running an ONNX export first and compiling its output into an engine.

    Unlike the portable formats, the engine is compiled for the machine that builds it: it is tied to that GPU and
    TensorRT version and does not move to another host.

    Examples:
        Requires the optional ``tensorrt`` dependency and a prepared graph, so this is documentation only
        (not a doctest):

        ```python
        TensorRTExporter(TensorRTConfig(variant_name="rfdetr-small"))(graph)
        # -> PosixPath('output/rfdetr-small_fp16.trt')
        ```
    """

    format = "tensorrt"
    display_name = "TensorRT"
    supports_dynamic_batch = True
    supports_notes = True
    pip_extra = "tensorrt"

    def _convert(self, graph: ExportGraph) -> str:
        """Export to ONNX, build the engine from it, and return the engine's path."""
        from rfdetr.export._onnx.exporter import OnnxExporter

        onnx_path = OnnxExporter(self.config.onnx_stage())(graph)
        # A backbone-only export already carries the "-backbone" marker in the ONNX stem; reuse that stem so a
        # custom output_name does not silently produce an engine indistinguishable from a full-detector one.
        output_name = onnx_path.stem if graph.backbone_only and self.config.output_name else self.config.output_name
        logger.info("Converting ONNX model to TensorRT engine")
        return self.build_engine(str(onnx_path), output_name=output_name)

    def build_engine(self, onnx_path: str, *, dry_run: bool = False, output_name: str | None = None) -> str:
        """Build a serialized TensorRT engine from an already-exported ONNX model, in-process.

        Uses the TensorRT Python API through ``polygraphy`` — no ``trtexec`` subprocess. Workspace size is left to
        the TensorRT default (it auto-sizes to the available device memory), which meets or exceeds the historical
        4 GiB cap. Precision and progress logging come from the exporter's configuration (``fp16``, ``verbose``);
        FP16 is automatically downgraded to FP32 (with a warning) on TensorRT builds that do not expose the FP16
        builder flag, and the engine filename reflects the precision actually built — except under *dry_run*, where
        nothing is probed or built so the requested precision is used as-is.

        Args:
            onnx_path: Path to the source ``.onnx`` file. Its stem (typically the model variant name, e.g.
                ``"rfdetr-medium"``) is reused for the engine filename unless an output name is given.
            dry_run: Log the intended build and return the engine path without building anything (no TensorRT /
                GPU required).
            output_name: Full filename override (without extension), or ``None`` to fall back to the
                configuration's ``output_name``. Takes precedence over the ONNX stem and suppresses the
                ``_fp16``/``_fp32`` suffix — the engine is named ``{output_name}.trt`` verbatim, written alongside
                *onnx_path*. :meth:`_convert` passes the backbone-marked ONNX stem through here.

        Returns:
            Path to the generated ``.trt`` engine file.

        Raises:
            ImportError: If ``polygraphy``/``tensorrt`` are not installed.

        Examples:
            The build logs its progress, so this is documentation rather than a doctest:

            ```python
            TensorRTExporter(TensorRTConfig()).build_engine("output/rfdetr-medium.onnx", dry_run=True)
            # -> 'output/rfdetr-medium_fp16.trt'
            ```
        """
        name = output_name if output_name is not None else self.config.output_name
        fp16 = self.config.fp16
        engine_path = self._engine_path(onnx_path, fp16_used=fp16, output_name=name)

        if dry_run:
            logger.info(f"[dry-run] Would build TensorRT engine (fp16={fp16}): {onnx_path} -> {engine_path}")
            return engine_path

        self._require_tensorrt()

        if fp16 and not self._supports_fp16_flag():
            fp16 = False
            engine_path = self._engine_path(onnx_path, fp16_used=fp16, output_name=name)

        self._compile(onnx_path, engine_path, fp16=fp16)
        return engine_path

    def _engine_path(self, onnx_path: str, *, fp16_used: bool, output_name: str | None) -> str:
        """Derive the ``.trt`` path the engine is written to, beside *onnx_path*.

        Args:
            onnx_path: Path to the source ``.onnx`` file, whose directory prefix and stem the engine inherits.
            fp16_used: The precision actually being built, which the filename encodes.
            output_name: Full filename override (without extension), or ``None`` to derive the name from the ONNX
                stem plus a precision suffix.

        Returns:
            Path to the ``.trt`` file the engine is written to.
        """
        if output_name:
            # Delegate output_name sanitize to the shared resolver so the custom-name stem is derived
            # identically to the ONNX/CoreML/ExecuTorch backends (single source of truth for basename +
            # extension stripping); TensorRT still owns its own path prefix and precision suffix below.
            stem = resolve_export_stem(None, output_name)[0]
            # Preserve onnx_path's directory prefix verbatim rather than rebuilding it via
            # os.path.dirname + os.path.join, which inject os.sep (a backslash on Windows) regardless
            # of onnx_path's own separator style and mis-parse a foreign-separator path. The sibling
            # suffix branch below deliberately avoids pathlib/os.path for the same reason.
            sep_idx = max(onnx_path.rfind("/"), onnx_path.rfind("\\"))
            prefix = onnx_path[: sep_idx + 1] if sep_idx != -1 else ""
            return f"{prefix}{stem}.trt"
        # Precision materially changes the engine (fp16 vs fp32 accuracy/speed), so it is always
        # encoded — unless a custom name was requested. Swapping only the final suffix (rather than
        # rebuilding the whole path) keeps any earlier ".onnx"-like segment intact and never aliases
        # the input path; a string-level split (not pathlib) preserves separators verbatim (pathlib
        # rewrites "/" to "\\" on Windows).
        onnx_stem = os.path.splitext(onnx_path)[0]
        return f"{onnx_stem}_{'fp16' if fp16_used else 'fp32'}.trt"

    def _require_tensorrt(self) -> None:
        """Fail early when the ``rfdetr[tensorrt]`` extra is missing.

        Raises:
            ImportError: If ``polygraphy``/``tensorrt`` are not installed.
        """
        if engine_from_network is None:
            raise ImportError(
                "TensorRT export requires the 'tensorrt' extra. Install with: pip install rfdetr[tensorrt]"
            )

    def _supports_fp16_flag(self) -> bool:
        """Report whether the installed TensorRT exposes the FP16 builder flag, warning when it does not.

        Some TensorRT builds (e.g. lean/partial wheels) do not expose the flag. polygraphy aborts when asked to set
        an unavailable one, so probe for it up front and let the caller fall back to an FP32 engine instead of
        crashing the whole export.

        Returns:
            ``False`` only when TensorRT imports but lacks the flag; a missing/broken ``tensorrt`` import is left to
            the polygraphy build chain to surface, so it reports ``True``.
        """
        try:
            import tensorrt as trt
        except ImportError:
            return True  # a missing/broken tensorrt import is surfaced by the polygraphy build chain

        if hasattr(trt.BuilderFlag, "FP16"):
            return True
        logger.warning(
            "TensorRT %s does not expose the FP16 builder flag; building an FP32 engine instead. "
            "Pass fp16=False to silence this warning.",
            getattr(trt, "__version__", "unknown"),
        )
        return False

    def _compile(self, onnx_path: str, engine_path: str, *, fp16: bool) -> None:
        """Build the engine through polygraphy and serialize it to *engine_path*.

        Args:
            onnx_path: Path to the source ``.onnx`` file.
            engine_path: Path the serialized engine is written to.
            fp16: The precision the engine is built with, after the FP16 availability probe.
        """
        if self.config.verbose:
            logger.info(f"Building TensorRT engine (fp16={fp16}) from {onnx_path}")

        engine = engine_from_network(
            network_from_onnx_path(onnx_path),
            config=CreateConfig(fp16=fp16),
        )
        save_engine(engine, path=engine_path)

        logger.info(f"Successfully built TensorRT engine: {engine_path}")
