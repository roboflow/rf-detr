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
``opset_version`` on ``CoreMLConfig`` to silently ignore. Each format's configuration class lives beside the exporter
that reads it, not here — this module names no format, so adding one touches its own package and the registry rather
than the abstraction they share.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Generic, TypeVar, cast

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
    was already doomed. The registry mirrors the capability — and the explanation — precisely so the refusal can
    happen before that, with the same wording either route reaches it by.

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
    raise NotImplementedError(_dynamic_batch_message(entry.label, entry.dynamic_batch_reason))


#: The :class:`ExportConfig` fields every format's configuration is built from. Both the shared half of
#: :meth:`Exporter.build_config` and the intermediate configuration a two-stage format derives are assembled from
#: exactly these names, so a new shared setting is added in one place.
SHARED_FIELDS: Final[tuple[str, ...]] = (
    "output_dir",
    "output_name",
    "variant_name",
    "backbone_only",
    "dynamic_batch",
    "verbose",
    "notes",
)


def shared_settings(config: ExportConfig) -> dict[str, Any]:
    """Extract the format-independent half of *config*, ready to splat into another configuration class.

    Used by the two-stage formats to build the intermediate ONNX configuration they export through: the
    intermediate graph inherits every setting the ONNX stage understands, *notes* included — the two-stage formats
    have always embedded them in the ``.onnx`` they pass on, and dropping them would silently change what a
    ``format="tflite"`` export writes.

    Args:
        config: Any format's configuration.

    Returns:
        The :data:`SHARED_FIELDS` values, keyed by field name.

    Examples:
        >>> shared_settings(ExportConfig(variant_name="rfdetr-small"))["variant_name"]
        'rfdetr-small'
    """
    return {name: getattr(config, name) for name in SHARED_FIELDS}


_ConfigT = TypeVar("_ConfigT", bound=ExportConfig)


class Exporter(ABC, Generic[_ConfigT]):
    """Write one export format's artifact from a prepared graph.

    Subclasses declare what their format can do as class attributes and implement :meth:`_convert`. Constructing
    an exporter validates the configuration against those capabilities, so an unsupported combination is rejected
    before the caller pays for a full forward pass through the model.

    A subclass also owns its configuration: :attr:`config_class` names the dataclass it is constructed from, and
    :attr:`setting_names` maps that dataclass's format-specific fields onto the keyword arguments
    :meth:`rfdetr.detr.RFDETR.export` accepts. :meth:`build_config` then narrows the public method's
    union-of-every-format signature down to one format's settings without the base class knowing which formats
    exist — the registry is the only place that enumerates them.

    Attributes:
        config_class: The configuration dataclass :meth:`build_config` instantiates.
        setting_names: This format's configuration fields, mapped to the ``RFDETR.export`` keyword each is read
            from. Shared fields (:data:`SHARED_FIELDS`) are handled by the base class and must not be listed.
        format: The format name this exporter is registered under.
        display_name: How the format is spelled in messages addressed to users.
        supports_dynamic_batch: Whether the format can bake a dynamic batch dimension into its artifact.
        supports_notes: Whether the artifact has a metadata slot for the user's *notes*.
        experimental: Whether constructing this exporter warns that the format is work-in-progress.
        pip_extra: The ``rfdetr[...]`` extra that installs this format's dependencies, or ``None`` when it needs none.
        notes_reason: Which metadata slot the artifact lacks, named in the dropped-``notes`` warning.
        experimental_note: Extra sentence appended to the experimental warning.
        dynamic_batch_reason: Why a fixed-batch format cannot honour ``dynamic_batch``, and what to do instead. Left
            empty by formats that support it. Mirrored by the format's registry entry so the same sentence is reached
            whether the refusal happens before or after the format's module is imported.
    """

    config_class: ClassVar[type[ExportConfig]] = ExportConfig
    setting_names: ClassVar[Mapping[str, str]] = {}
    format: ClassVar[str]
    display_name: ClassVar[str] = ""
    supports_dynamic_batch: ClassVar[bool] = False
    supports_notes: ClassVar[bool] = False
    experimental: ClassVar[bool] = False
    pip_extra: ClassVar[str | None] = None
    notes_reason: ClassVar[str] = "this artifact has no ONNX-style metadata slot"
    experimental_note: ClassVar[str] = ""
    dynamic_batch_reason: ClassVar[str] = ""

    @classmethod
    def build_config(cls, **settings: Any) -> _ConfigT:
        """Build this format's configuration from :meth:`rfdetr.detr.RFDETR.export`'s flat keyword arguments.

        Settings belonging to other formats are dropped here rather than travelling down into a converter that
        would have to know to ignore them, and a keyword absent from *settings* falls back to the configuration
        dataclass's own default instead of being restated.

        Args:
            **settings: The keyword arguments ``RFDETR.export`` was called with, shared and format-specific alike.

        Returns:
            An instance of :attr:`config_class`.
        """
        shared = {name: settings[name] for name in SHARED_FIELDS if name in settings}
        specific = cls._format_settings(settings)
        return cast("_ConfigT", cls.config_class(**shared, **specific))

    @classmethod
    def _format_settings(cls, settings: Mapping[str, Any]) -> dict[str, Any]:
        """Pick this format's own settings out of ``RFDETR.export``'s flat keyword arguments.

        The default reads :attr:`setting_names`. Override it only when a format needs to validate or derive a
        setting rather than copy it across.

        Args:
            settings: The keyword arguments ``RFDETR.export`` was called with.

        Returns:
            The format-specific keyword arguments for :attr:`config_class`.
        """
        return {field: settings[keyword] for field, keyword in cls.setting_names.items() if keyword in settings}

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
                _dynamic_batch_message(self.display_name or self.format, self.dynamic_batch_reason)
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
