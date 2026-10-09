# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The exporter interface every export format implements, and the configuration each one takes.

An exporter is constructed from its format's configuration and then called with a prepared
:class:`~rfdetr.export.prepare.ExportGraph`, so the two halves of an export — *what the user asked for* and *what the
model looks like* — stay separate and independently testable.

The base class owns everything that is the same for every format: rejecting a capability the format does not have and
``notes`` that JSON cannot encode, warning about settings it ignores, running the format's dependency check before the
conversion, switching the model into its export-friendly forward exactly once, normalizing the returned path, and
logging the result. A format subclass implements :meth:`Exporter._convert` and declares its capabilities as class
attributes; it adds its own checks through the hooks below rather than repeating the base class's.

Configuration is per format rather than one flat object, so a knob that does not apply cannot be passed: there is no
``opset_version`` on ``CoreMLConfig`` to silently ignore. Each format's configuration class lives beside the exporter
that reads it, not here — this module names no format, so adding one touches its own package and the registry rather
than the abstraction they share.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Generic, TypeVar, cast

from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export._runtime.metadata import ExportMetadata, write_metadata
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


def serialize_notes(notes: object) -> str:
    """Render *notes* as the string an artifact's metadata slot stores.

    A string is stored as-is, so readers can use it without decoding. Anything else is JSON-encoded, and strictly:
    ``NaN`` and ``Infinity`` are not valid JSON, and a strict parser on the reading side would reject them.

    Args:
        notes: The user's notes.

    Returns:
        The value to store under the artifact's ``rfdetr_notes`` key.

    Raises:
        ValueError: If *notes* holds a non-finite float or a circular reference.
        TypeError: If *notes* holds a value JSON cannot encode.

    Examples:
        >>> serialize_notes("trained on pallets")
        'trained on pallets'
        >>> serialize_notes({"run": 3, "classes": ["box"]})
        '{"run": 3, "classes": ["box"]}'
        >>> serialize_notes(float("nan"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: notes must be a string or a JSON-serializable value: ...
    """
    if isinstance(notes, str):
        return notes
    try:
        return json.dumps(notes, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"notes must be a string or a JSON-serializable value: {error}") from error
    except TypeError as error:
        raise TypeError(f"notes must be a string or a JSON-serializable value: {error}") from error


_ConfigT = TypeVar("_ConfigT", bound=ExportConfig)


class Exporter(ABC, Generic[_ConfigT]):
    """Write one export format's artifact from a prepared graph.

    Subclasses declare what their format can do as class attributes and implement :meth:`_convert`. A check that needs
    only the request or the installed packages belongs before the caller pays for a full forward pass through the
    model, in one of these steps, which :meth:`rfdetr.detr.RFDETR.export` runs before it prepares the graph:

    1. :meth:`build_config` reads the keyword arguments. A subclass overrides :meth:`_format_settings` only for a
       keyword that must be derived, one the configuration does not store, or one whose absence the configuration's
       default would hide.
    2. Constructing the exporter validates the configuration: :meth:`_check_capabilities` rejects a capability the
       class attributes deny and ``notes`` that JSON cannot encode, for a format that embeds them. A subclass
       overrides it (calling ``super()`` first) to validate its own settings — an unknown precision name, or
       cross-field consistency within the format's configuration — raising ``ValueError`` (the base class raises
       ``NotImplementedError`` for a capability the class attributes deny). It reads the configuration only, never
       the environment. Only a configuration that passes is warned about (dropped ``notes``, an experimental format).
    3. Where :meth:`_check_capabilities` judges the request, :meth:`check_dependencies` judges the host: it refuses a
       package the format cannot run without. A subclass overrides it and never calls it: ``RFDETR.export`` calls it
       before the forward pass, and :meth:`__call__` calls it again before :meth:`_convert`, so an exporter handed a
       graph directly checks the same packages first.
    4. :meth:`check_environment` then asks whether the installed packages can build what the configuration requests
       (a separate runtime library, a feature of a newer release). It runs right after :meth:`check_dependencies`, at
       the same two places, and is an instance method because the answer depends on the configuration.

    A refusal that depends on the prepared graph (an output the converter cannot lower, say) comes first in
    :meth:`_convert`, before the conversion runs.

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

        The default reads :attr:`setting_names`. Override it only for a keyword that must be derived rather than
        copied, one the configuration does not store, or one whose absence the configuration's default would hide. A
        value the configuration holds is validated in :meth:`_check_capabilities` instead.

        Args:
            settings: The keyword arguments ``RFDETR.export`` was called with.

        Returns:
            The format-specific keyword arguments for :attr:`config_class`.
        """
        return {field: settings[keyword] for field, keyword in cls.setting_names.items() if keyword in settings}

    def __init__(self, config: _ConfigT) -> None:
        """Validate *config* against this format's capabilities, warn about what it ignores, and keep it.

        Args:
            config: The format's configuration.

        Raises:
            NotImplementedError: If the configuration asks for a capability the format does not have.
            ValueError: If *notes* the format embeds is not JSON-serializable (e.g. ``NaN``), or a subclass's
                :meth:`_check_capabilities` override rejects the configuration for a format-specific reason the class
                attributes alone cannot express (e.g. an unknown precision, or TensorRT's dynamic-batch
                optimization-profile bounds).
            TypeError: If *notes* the format embeds holds a value JSON cannot encode.
        """
        self.config = config
        self._check_capabilities()
        # Warned only once the checks above pass, so a configuration this constructor refuses does not warn first.
        # stacklevel=3 points past __init__ and RFDETR.export at the user's call: pointing at RFDETR.export would break
        # `warnings.filterwarnings(..., module=...)` filters and collapse all the formats onto one reported location.
        if self.config.notes is not None and not self.supports_notes:
            warnings.warn(
                f"`notes` is not forwarded to format={self.format!r} ({self.notes_reason}). This argument is ignored.",
                UserWarning,
                stacklevel=3,
            )
        if self.experimental:
            name = self.display_name or self.format
            warnings.warn(
                f"{name} export is experimental and work-in-progress. {self.experimental_note}".strip(),
                UserWarning,
                stacklevel=3,
            )

    def _check_capabilities(self) -> None:
        """Reject settings this format cannot honour.

        A subclass override should call ``super()._check_capabilities()`` first, then add its own format-specific
        checks — see :class:`~rfdetr.export._tensorrt.exporter.TensorRTExporter` for an example.

        Raises:
            NotImplementedError: If ``dynamic_batch`` was requested and the format bakes a fixed shape.
            ValueError: If the format embeds *notes* and they hold a non-finite float or a circular reference. A
                subclass override may also raise it for its own format-specific validation failures.
            TypeError: If the format embeds *notes* and they hold a value JSON cannot encode.
        """
        if self.config.dynamic_batch and not self.supports_dynamic_batch:
            raise NotImplementedError(
                _dynamic_batch_message(self.display_name or self.format, self.dynamic_batch_reason)
            )
        # Serialized here and again when the artifact is written: a value no metadata slot can hold must fail before
        # the forward pass, not after the trace has already written a file without it.
        if self.config.notes is not None and self.supports_notes:
            serialize_notes(self.config.notes)

    @classmethod
    def check_dependencies(cls) -> None:
        """Refuse a host missing this format's optional dependencies, before any work on the model starts.

        The default is a no-op, for a format that needs no optional package. A format that needs one overrides this
        with the availability check its conversion relies on, kept cheap — a metadata probe such as
        :func:`~rfdetr.utilities.package.is_installed`, or an import the conversion performs anyway — so the refusal
        lands before :func:`~rfdetr.export.prepare.prepare_export_graph` has paid for a full forward pass through the
        model.

        :meth:`rfdetr.detr.RFDETR.export` calls it on the resolved exporter class right after constructing the
        exporter, and :meth:`__call__` calls it again before :meth:`_convert`, so an exporter called on a graph
        directly is checked too. A subclass overrides it and never calls it itself; it runs twice on the export path, so
        an override stays idempotent. An override is an addition, not a move, for a public entry point that bypasses
        :meth:`__call__`: TFLite's ``convert_onnx`` and TensorRT's ``build_engine`` keep their own checks. Construction
        does not call it: a caller that only builds the exporter (a TensorRT ``dry_run``, a test that stubs the
        conversion) needs none of the format's packages.

        Raises:
            ImportError: If an override finds a package its format needs is not installed, naming the extra that
                installs it.

        Note:
            Not thread-safe across concurrent exports in the same process: an override may probe or mutate
            process-global state (``sys.modules``, import order, warning filters) that a concurrent call to this
            method, on any format, could race with. Callers are assumed to invoke exports one at a time.

        Examples:
            >>> Exporter.check_dependencies() is None
            True
        """
        return

    def check_environment(self) -> None:
        """Refuse a configuration the installed packages cannot build, before any work on the model starts.

        The default is a no-op. A format overrides it when a setting needs more than :meth:`check_dependencies` can
        see, such as a runtime library that ships separately from the format's package.
        :meth:`rfdetr.detr.RFDETR.export` and :meth:`__call__` call it right after :meth:`check_dependencies`, so an
        override may import the format's packages, and it stays idempotent. A public entry point that bypasses
        :meth:`__call__` calls it itself.

        Raises:
            ImportError: If an override finds that a library the configuration needs cannot be loaded.
            ValueError: If an override finds that the installed package cannot honour a setting.
        """
        return

    def __call__(self, graph: ExportGraph) -> Path:
        """Export *graph* and return the path to the artifact.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the exported artifact.

        Raises:
            ImportError: If :meth:`check_dependencies` finds a package the format needs missing, or
                :meth:`check_environment` a library the configuration needs; both are checked before the conversion
                starts, so a caller that hands the exporter a graph directly gets the same message.
            ValueError: If :meth:`check_environment` finds that the installed package cannot honour a setting.
        """
        self.check_dependencies()
        self.check_environment()
        # Once for every format — the model arrives from prepare_export_graph in its training forward, and the
        # switch is idempotent so a two-stage format composing another exporter stays safe.
        _switch_to_export_mode(graph.model)
        path = Path(self._convert(graph))
        if graph.metadata is not None and graph.metadata.format == self.format:
            try:
                artifacts = self._metadata_artifacts(path)
            except Exception as error:
                logger.warning(
                    "Could not select inference metadata artifacts for %s: %s. "
                    "Load the artifact with metadata= set to its inference metadata.",
                    path,
                    error,
                )
            else:
                for artifact in artifacts:
                    try:
                        write_metadata(artifact, self._metadata_for_artifact(graph.metadata, artifact))
                    except Exception as error:
                        logger.warning(
                            "Could not write inference metadata for %s: %s. "
                            "Load this artifact with metadata= set to its inference metadata.",
                            artifact,
                            error,
                        )
        logger.info(f"Successfully exported {self.display_name or self.format} model to: {path}")
        return path

    def _metadata_artifacts(self, path: Path) -> tuple[Path, ...]:
        """List final artifacts that need their own inference metadata."""
        return (path,)

    def _metadata_for_artifact(self, metadata: ExportMetadata, path: Path) -> ExportMetadata:
        """Return metadata unchanged unless the format overrides the graph interface."""
        return metadata

    @abstractmethod
    def _convert(self, graph: ExportGraph) -> Path | str:
        """Write this format's artifact and return where it landed.

        Args:
            graph: The prepared model and its graph metadata.

        Returns:
            Path to the written artifact, as a :class:`~pathlib.Path` or a string.
        """
