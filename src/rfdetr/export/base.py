# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The exporter interface every export format implements, and the configuration each one takes.

An exporter is constructed from its format's configuration and then called with a prepared
:class:`~rfdetr.export.prepare.ExportGraph`, so the two halves of an export — *what the user asked for* and *what the
model looks like* — stay separate and independently testable.

The base class owns everything that is the same for all six formats: rejecting a capability the format does not have,
switching the model into its export-friendly forward exactly once, normalizing the returned path, and logging the
result. A format subclass implements :meth:`Exporter._convert` and declares its capabilities as class attributes; it
never repeats a guard.

Configuration is per format rather than one flat object, so a knob that does not apply cannot be passed: there is no
``opset_version`` on :class:`CoreMLConfig` to silently ignore.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypedDict, TypeVar, cast

from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export.prepare import ExportGraph
from rfdetr.export.registry import require_entry
from rfdetr.utilities.logger import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Settings every export format understands.

    Attributes:
        output_dir: Directory the artifact is written to.
        output_name: Full filename override (without extension), or ``None`` to derive one from *variant_name*.
        variant_name: Model variant identifier used to name the artifact when *output_name* is unset.
        backbone_only: Whether the graph is a backbone-only export, which the filename marks.
        dynamic_batch: Whether the graph carries a dynamic batch dimension.
        verbose: Whether the format's converter should log its progress.
        notes: User-supplied metadata. Only formats declaring ``supports_notes`` embed it; the rest warn and drop it.
    """

    output_dir: Path = Path("output")
    output_name: str | None = None
    variant_name: str | None = None
    backbone_only: bool = False
    dynamic_batch: bool = False
    verbose: bool = True
    notes: object = None


@dataclass(frozen=True, slots=True)
class OnnxConfig(ExportConfig):
    """Settings for ``format="onnx"``.

    Attributes:
        opset_version: ONNX opset the graph targets.
    """

    opset_version: int = 17


@dataclass(frozen=True, slots=True)
class OpenVINOConfig(ExportConfig):
    """Settings for ``format="openvino"``.

    Attributes:
        precision: ``"float32"``, ``"float16"``, or ``None`` to keep OpenVINO's own FP16 compression default.
    """

    precision: str | None = None


@dataclass(frozen=True, slots=True)
class CoreMLConfig(ExportConfig):
    """Settings for ``format="coreml"``.

    Attributes:
        compute_precision: ``"float32"``, ``"float16"``, or ``None`` for coremltools' default.
    """

    compute_precision: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutorchConfig(ExportConfig):
    """Settings for ``format="executorch"``.

    Attributes:
        backend: Delegation backend the ``.pte`` is lowered for.
        soc: Target SoC, required by backends that compile ahead-of-time for one chip.
    """

    backend: Literal["xnnpack", "coreml", "qnn"] = "xnnpack"
    soc: str | None = None


@dataclass(frozen=True, slots=True)
class TFLiteConfig(ExportConfig):
    """Settings for ``format="tflite"``, which converts an ONNX export rather than the model directly.

    Attributes:
        opset_version: ONNX opset the intermediate graph targets.
        quantization: Quantization mode; ``"int8"`` additionally writes a dynamic-range model.
        calibration_data: Data written to a scratch file beside the artifacts but not consumed by the conversion.
        max_images: Maximum images read from a *calibration_data* directory.
    """

    opset_version: int = 17
    quantization: str | None = None
    calibration_data: Any = None
    max_images: int = 100

    def onnx_stage(self) -> OnnxConfig:
        """Return the configuration for the ONNX export this format converts from.

        Returns:
            An :class:`OnnxConfig` carrying the settings the intermediate graph needs.

        Examples:
            >>> TFLiteConfig(quantization="int8").onnx_stage().opset_version
            17
        """
        return _derive_onnx_stage(self, opset_version=self.opset_version)


@dataclass(frozen=True, slots=True)
class TensorRTConfig(ExportConfig):
    """Settings for ``format="tensorrt"``, which builds an engine from an ONNX export.

    Attributes:
        opset_version: ONNX opset the intermediate graph targets.
        fp16: Whether to build the engine with FP16 precision.
    """

    opset_version: int = 17
    fp16: bool = True

    def onnx_stage(self) -> OnnxConfig:
        """Return the configuration for the ONNX export this format builds from.

        Returns:
            An :class:`OnnxConfig` carrying the settings the intermediate graph needs.

        Examples:
            >>> TensorRTConfig(fp16=False).onnx_stage().opset_version
            17
        """
        return _derive_onnx_stage(self, opset_version=self.opset_version)


def _derive_onnx_stage(config: ExportConfig, *, opset_version: int) -> OnnxConfig:
    """Build the intermediate :class:`OnnxConfig` a two-stage format exports through.

    The intermediate graph inherits every setting the ONNX stage understands, *notes* included — the two-stage
    formats have always embedded them in the ``.onnx`` they pass on, and dropping them here would silently change
    what a ``format="tflite"`` export writes.

    Args:
        config: The two-stage format's configuration.
        opset_version: ONNX opset the intermediate graph targets.

    Returns:
        The configuration for the ONNX stage.

    Examples:
        >>> _derive_onnx_stage(ExportConfig(variant_name="rfdetr-small"), opset_version=17).variant_name
        'rfdetr-small'
    """
    return OnnxConfig(
        output_dir=config.output_dir,
        output_name=config.output_name,
        variant_name=config.variant_name,
        backbone_only=config.backbone_only,
        dynamic_batch=config.dynamic_batch,
        verbose=config.verbose,
        notes=config.notes,
        opset_version=opset_version,
    )


#: Why each format that bakes a fixed batch size cannot honour ``dynamic_batch``, and what to do instead. Kept here
#: rather than on the exporter classes so the refusal can be raised before the format's module — and its heavy
#: optional dependency — is imported; formats absent from this mapping accept a dynamic batch dimension.
_DYNAMIC_BATCH_REASONS: Mapping[str, str] = {
    "executorch": "(see the ExecuTorch exporter for details). Export one .pte per batch size instead.",
    "coreml": (
        "(fixed shapes are required for reliable ANE / GPU scheduling). Export one .mlpackage per batch size instead."
    ),
    "openvino": "(the IR graph bakes a fixed input shape). Export one model per batch size instead.",
}


def _dynamic_batch_message(label: str, reason: str) -> str:
    """Compose the message shown when a format cannot bake a dynamic batch dimension.

    Args:
        label: How the format is spelled in messages addressed to users.
        reason: Why the format cannot do it, and what to do instead.

    Returns:
        The rejection message.

    Examples:
        >>> _dynamic_batch_message("CoreML", "(fixed shapes are required). Export one per batch size instead.")
        'CoreML export does not support dynamic_batch (fixed shapes are required). Export one per batch size instead.'
    """
    return f"{label} export does not support dynamic_batch {reason}".rstrip()


def reject_unsupported_dynamic_batch(format: str, *, dynamic_batch: bool) -> None:
    """Raise when *format* cannot honour ``dynamic_batch``, without importing the format's dependencies.

    The exporter class rejects this too, but only once it exists — and constructing it means importing
    ``coremltools``/``executorch``/``openvino`` first, tens of seconds and hundreds of megabytes for a request that
    was already doomed. The registry mirrors the capability precisely so the refusal can happen before that.

    Args:
        format: Canonical format name.
        dynamic_batch: Whether the caller asked for a dynamic batch dimension.

    Raises:
        NotImplementedError: If *format* bakes a fixed batch size and *dynamic_batch* is set.
        ValueError: If *format* is not a known export format.

    Examples:
        >>> reject_unsupported_dynamic_batch("onnx", dynamic_batch=True)
        >>> reject_unsupported_dynamic_batch("coreml", dynamic_batch=False)
    """
    if not dynamic_batch:
        return
    entry = require_entry(format)
    if entry.supports_dynamic_batch:
        return
    raise NotImplementedError(_dynamic_batch_message(entry.label, _DYNAMIC_BATCH_REASONS[format]))


class _SharedSettings(TypedDict):
    """The :class:`ExportConfig` fields every format's configuration is built from."""

    output_dir: Path
    output_name: str | None
    variant_name: str | None
    backbone_only: bool
    dynamic_batch: bool
    verbose: bool
    notes: object


def build_export_config(
    format: str,
    *,
    output_dir: Path,
    output_name: str | None = None,
    variant_name: str | None = None,
    backbone_only: bool = False,
    dynamic_batch: bool = False,
    verbose: bool = True,
    notes: object = None,
    opset_version: int = 17,
    backend: str | None = None,
    soc: str | None = None,
    fp16: bool = True,
    coreml_precision: str | None = None,
    openvino_precision: str | None = None,
    quantization: str | None = None,
    calibration_data: Any = None,
    max_images: int = 100,
) -> ExportConfig:
    """Translate :meth:`rfdetr.detr.RFDETR.export`'s flat keyword arguments into *format*'s configuration.

    This is the one place the public method's union-of-every-format signature is narrowed to the settings the
    chosen format actually reads. Arguments belonging to other formats are dropped here rather than travelling
    down into a converter that would have to know to ignore them.

    Args:
        format: Canonical format name.
        output_dir: Directory the artifact is written to.
        output_name: Full filename override, without extension.
        variant_name: Model variant identifier used to name the artifact.
        backbone_only: Whether the graph is a backbone-only export.
        dynamic_batch: Whether a dynamic batch dimension was requested.
        verbose: Whether the converter logs its progress.
        notes: User-supplied metadata.
        opset_version: ONNX opset, for the formats that write or pass through an ONNX graph.
        backend: ExecuTorch delegation backend.
        soc: ExecuTorch target SoC.
        fp16: TensorRT engine precision.
        coreml_precision: CoreML compute precision.
        openvino_precision: OpenVINO IR storage precision.
        quantization: TFLite quantization mode.
        calibration_data: TFLite calibration data.
        max_images: Maximum images read from a TFLite calibration directory.

    Returns:
        The configuration class registered for *format*.

    Raises:
        ValueError: If *format* is not a known export format.

    Examples:
        >>> build_export_config("coreml", output_dir=Path("out"), coreml_precision="float16").compute_precision
        'float16'
        >>> build_export_config("onnx", output_dir=Path("out"), opset_version=18).opset_version
        18
    """
    shared: _SharedSettings = {
        "output_dir": output_dir,
        "output_name": output_name,
        "variant_name": variant_name,
        "backbone_only": backbone_only,
        "dynamic_batch": dynamic_batch,
        "verbose": verbose,
        "notes": notes,
    }
    if format == "onnx":
        return OnnxConfig(**shared, opset_version=opset_version)
    if format == "openvino":
        return OpenVINOConfig(**shared, precision=openvino_precision)
    if format == "coreml":
        return CoreMLConfig(**shared, compute_precision=coreml_precision)
    if format == "executorch":
        if backend is None:
            # _resolve_export_backend always sets a backend for this format; defaulting silently here would turn a
            # bypassed resolver into a quietly-wrong xnnpack artifact instead of an error.
            raise ValueError("format='executorch' requires a backend; none was resolved.")
        # The resolver already validated it against the converter's accepted set; narrow the static type to the
        # Literal the converter's signature declares.
        executorch_backend = cast("Literal['xnnpack', 'coreml', 'qnn']", backend)
        return ExecutorchConfig(**shared, backend=executorch_backend, soc=soc)
    if format == "tflite":
        return TFLiteConfig(
            **shared,
            opset_version=opset_version,
            quantization=quantization,
            calibration_data=calibration_data,
            max_images=max_images,
        )
    if format == "tensorrt":
        return TensorRTConfig(**shared, opset_version=opset_version, fp16=fp16)
    raise ValueError(f"Unsupported export format {format!r}.")


_ConfigT = TypeVar("_ConfigT", bound=ExportConfig)


class Exporter(ABC, Generic[_ConfigT]):
    """Write one export format's artifact from a prepared graph.

    Subclasses declare what their format can do as class attributes and implement :meth:`_convert`. Constructing
    an exporter validates the configuration against those capabilities, so an unsupported combination is rejected
    before the caller pays for a full forward pass through the model.

    Attributes:
        format: The format name this exporter is registered under.
        display_name: How the format is spelled in messages addressed to users.
        supports_dynamic_batch: Whether the format can bake a dynamic batch dimension into its artifact.
        supports_notes: Whether the artifact has a metadata slot for the user's *notes*.
        experimental: Whether constructing this exporter warns that the format is work-in-progress.
        pip_extra: The ``rfdetr[...]`` extra that installs this format's dependencies, or ``None`` when it needs none.
        notes_reason: Which metadata slot the artifact lacks, named in the dropped-``notes`` warning.
        experimental_note: Extra sentence appended to the experimental warning.
    """

    format: ClassVar[str]
    display_name: ClassVar[str] = ""
    supports_dynamic_batch: ClassVar[bool] = False
    supports_notes: ClassVar[bool] = False
    experimental: ClassVar[bool] = False
    pip_extra: ClassVar[str | None] = None
    notes_reason: ClassVar[str] = "this artifact has no ONNX-style metadata slot"
    experimental_note: ClassVar[str] = ""

    def __init__(self, config: _ConfigT) -> None:
        """Validate *config* against this format's capabilities and keep it for the conversion.

        Args:
            config: The format's configuration.

        Raises:
            NotImplementedError: If the configuration asks for a capability the format does not have.
        """
        self.config = config
        self._check_capabilities()

    def _check_capabilities(self) -> None:
        """Reject or warn about settings this format cannot honour.

        Raises:
            NotImplementedError: If ``dynamic_batch`` was requested and the format bakes a fixed shape.
        """
        if self.config.dynamic_batch and not self.supports_dynamic_batch:
            raise NotImplementedError(
                _dynamic_batch_message(
                    self.display_name or self.format, _DYNAMIC_BATCH_REASONS.get(self.format, "")
                )
            )
        # stacklevel=4, not 3: the warning is raised two frames below the public entry point
        # (_check_capabilities -> __init__ -> RFDETR.export -> the user's call), and pointing at RFDETR.export
        # would break `warnings.filterwarnings(..., module=...)` filters and collapse all six formats onto one
        # reported location.
        if self.config.notes is not None and not self.supports_notes:
            warnings.warn(
                f"`notes` is not forwarded to format={self.format!r} ({self.notes_reason}). This argument is ignored.",
                UserWarning,
                stacklevel=4,
            )
        if self.experimental:
            name = self.display_name or self.format
            warnings.warn(
                f"{name} export is experimental and work-in-progress. {self.experimental_note}".strip(),
                UserWarning,
                stacklevel=4,
            )

    def __call__(self, graph: ExportGraph) -> Path:
        """Export *graph* and return the path to the artifact.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the exported artifact.
        """
        # Once for every format — the model arrives from prepare_export_graph in its training forward, and the
        # switch is idempotent so a two-stage format composing another exporter stays safe.
        _switch_to_export_mode(graph.model)
        path = Path(self._convert(graph))
        logger.info(f"Successfully exported {self.display_name or self.format} model to: {path}")
        return path

    @abstractmethod
    def _convert(self, graph: ExportGraph) -> Path | str:
        """Write this format's artifact and return where it landed.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the written artifact, as a :class:`~pathlib.Path` or a string.
        """
