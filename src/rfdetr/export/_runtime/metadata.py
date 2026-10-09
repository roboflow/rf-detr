# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Versioned prediction metadata for exported RF-DETR artifacts."""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
import os
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rfdetr.utilities.class_names import prediction_labels
from rfdetr.utilities.files import _mkstemp_default_mode, _replace_keeping_mode

METADATA_KEY = "rfdetr_inference"
#: The one metadata schema this reader understands. Any change to the ``ExportMetadata`` fields bumps it.
SCHEMA_VERSION: Final = 1
ExportFormat = Literal["onnx", "tensorrt", "openvino", "tflite", "litert", "coreml", "coreai", "executorch"]
ExportTask = Literal["detect", "segment", "keypoints", "backbone"]


class ExportMetadata(BaseModel):
    """Describe an exported graph and the native prediction rules it uses."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    producer_version: str | None = None
    format: ExportFormat
    task: ExportTask
    variant: str | None = None
    backend: str | None = None
    input_shape: tuple[int, int, int, int]
    input_layout: Literal["NCHW", "NHWC"] = "NCHW"
    input_dtype: str = "float32"
    input_name: str | int = "input"
    max_batch_size: int | None = None
    outputs: dict[str, str | int]
    means: list[float]
    stds: list[float]
    pixel_scale: float = 1.0 / 255.0
    channel_order: Literal["RGB"] = "RGB"
    resize_interpolation: Literal["bilinear"] = "bilinear"
    resize_antialias: Literal[False] = False
    class_names: list[str]
    class_id_to_name: dict[int, str] = Field(default_factory=dict)
    num_classes: int
    num_select: int
    selection_policy: Literal["all_logits"] = "all_logits"
    box_format: Literal["cxcywh_normalized"] = "cxcywh_normalized"
    num_keypoints_per_class: list[int] = Field(default_factory=list)
    trace_alpha: float
    upsample_masks_to_image_size: bool = True
    patch_size: int
    num_windows: int

    @model_validator(mode="after")
    def validate_contract(self) -> ExportMetadata:
        """Reject metadata that cannot describe one safe prediction contract."""
        batch, channels, height, width = self.input_shape
        if batch == 0 or batch < -1 or min(channels, height, width) <= 0:
            raise ValueError(
                "input_shape must have a positive channel and spatial shape, and batch must be -1 or positive"
            )
        if self.max_batch_size is not None and (batch != -1 or self.max_batch_size <= 0):
            raise ValueError("max_batch_size requires a dynamic batch and must be positive")
        if (
            len(self.means) != channels
            or len(self.stds) != channels
            or any(not math.isfinite(value) for value in self.means + self.stds)
            or any(value <= 0 for value in self.stds)
        ):
            raise ValueError("means and positive stds must match the input channel count")
        if self.task != "backbone":
            needed = {"pred_boxes", "pred_logits"}
            if self.task == "segment":
                needed.add("pred_masks")
            if self.task == "keypoints":
                needed.add("pred_keypoints")
            if not needed.issubset(self.outputs):
                raise ValueError(f"outputs missing semantic values: {sorted(needed - self.outputs.keys())}")
        identifiers = list(self.outputs.values())
        if len(set(identifiers)) != len(identifiers) or any(
            isinstance(identifier, int) and identifier < 0 for identifier in identifiers
        ):
            raise ValueError(
                "outputs must map each semantic value to its own runtime output, and output positions must be"
                " non-negative"
            )
        if self.task != "backbone" and not self.class_names:
            raise ValueError("class_names are required for prediction")
        if self.num_classes < 0 or self.num_select < 0 or self.patch_size <= 0 or self.num_windows <= 0:
            raise ValueError("num_classes, num_select, patch_size, and num_windows must describe a valid model")
        if not math.isfinite(self.trace_alpha) or self.trace_alpha < 0:
            raise ValueError("trace_alpha must be finite and non-negative")
        if self.pixel_scale != 1.0 / 255.0:
            raise ValueError("pixel_scale must match native 1/255 preprocessing")
        if self.task == "keypoints" and not self.num_keypoints_per_class:
            raise ValueError("num_keypoints_per_class is required for keypoint prediction")
        if self.task == "keypoints" and not any(self.num_keypoints_per_class):
            raise ValueError("num_keypoints_per_class has no active keypoint class")
        if not self.class_id_to_name and self.num_classes == len(self.class_names):
            self.class_id_to_name = dict(enumerate(self.class_names))
        if self.task != "backbone" and not self.class_id_to_name:
            raise ValueError("class_id_to_name is required when labels are not contiguous")
        return self

    @property
    def num_channels(self) -> int:
        """Return the channel count in canonical NCHW order."""
        return self.input_shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the fixed input height and width."""
        return self.input_shape[2:]


def positional_metadata(metadata: ExportMetadata) -> ExportMetadata:
    """Use positional input and output identifiers for tuple-returning runtimes."""
    return metadata.model_copy(
        update={"input_name": 0, "outputs": {name: index for index, name in enumerate(metadata.outputs)}}
    )


def _artifact_digest(path: Path) -> str:
    """Hash a file or bundle, including OpenVINO's companion weights file."""
    digest = hashlib.sha256()
    files = sorted(file for file in path.rglob("*") if file.is_file()) if path.is_dir() else [path]
    if path.suffix == ".xml":
        weights = path.with_suffix(".bin")
        if weights.exists():
            files.append(weights)
    for file in files:
        if path.is_dir():
            digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    """Name the companion file without colliding with another artifact extension."""
    return path.with_name(f"{path.name}.rfdetr.json")


def _temporary_sibling(path: Path) -> Path:
    """Create a temporary sibling so replacement stays on the same filesystem.

    The file gets the umask-derived mode a plain ``open()`` would give a new file, not ``mkstemp``'s owner-only
    ``0o600``; move it into place with ``_replace_keeping_mode`` so an existing target keeps its own mode.
    """
    descriptor, name = _mkstemp_default_mode(path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    os.close(descriptor)
    return Path(name)


def write_metadata(path: str | Path, metadata: ExportMetadata) -> Path | None:
    """Store metadata in ONNX or in a digest-bound adjacent JSON file."""
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    if metadata.format == "onnx" and artifact.suffix == ".onnx":
        onnx = importlib.import_module("onnx")
        model = onnx.load(str(artifact), load_external_data=False)
        serialized = metadata.model_dump_json()
        existing = next((item for item in model.metadata_props if item.key == METADATA_KEY), None)
        item = existing if existing is not None else model.metadata_props.add()
        item.key = METADATA_KEY
        item.value = serialized
        temporary = _temporary_sibling(artifact)
        try:
            onnx.save(model, str(temporary))
            _replace_keeping_mode(temporary, artifact)
        finally:
            temporary.unlink(missing_ok=True)
        return None
    sidecar = _sidecar_path(artifact)
    payload = {"artifact_sha256": _artifact_digest(artifact), "metadata": metadata.model_dump(mode="json")}
    temporary = _temporary_sibling(sidecar)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _replace_keeping_mode(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def _json_object(text: str, source: str) -> dict[str, Any]:
    """Parse JSON text that must hold one object.

    Args:
        text: Serialized JSON.
        source: What the text came from, named in the error.

    Returns:
        The parsed object.

    Raises:
        ValueError: If the text is not valid JSON, nests too deeply to parse, or holds something other than an object.

    Examples:
        >>> _json_object('{"task": "detect"}', "metadata=")
        {'task': 'detect'}
    """
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{source} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{source} must hold a JSON object, got {type(value).__name__}")
    return value


def _envelope_metadata(
    envelope: dict[str, Any], source: str, artifact: Path, artifact_digest: Callable[[], str]
) -> dict[str, Any]:
    """Unwrap the metadata a digest envelope carries once its digest matches the artifact.

    Args:
        envelope: Parsed ``{"artifact_sha256": ..., "metadata": ...}`` object.
        source: What the envelope came from, named in the error.
        artifact: Artifact the envelope must describe.
        artifact_digest: Returns the artifact's digest; called at most once.

    Returns:
        The metadata object inside the envelope.

    Raises:
        ValueError: If a key is missing, the digest belongs to other bytes, or the metadata is not an object.
    """
    missing = sorted({"artifact_sha256", "metadata"} - envelope.keys())
    if missing:
        raise ValueError(f"{source} is missing {missing}")
    if envelope["artifact_sha256"] != artifact_digest():
        raise ValueError(f"Metadata digest does not match artifact: {artifact}")
    metadata = envelope["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{source} must hold a JSON object under 'metadata', got {type(metadata).__name__}")
    return metadata


def _override_metadata(
    override: Mapping[str, Any] | str | Path, artifact: Path, artifact_digest: Callable[[], str]
) -> dict[str, Any]:
    """Load caller-supplied metadata as plain JSON values, unwrapping a digest envelope bound to the artifact.

    Args:
        override: A mapping, or the path of a JSON file, holding metadata or a digest envelope.
        artifact: Artifact the metadata must describe.
        artifact_digest: Returns the artifact's digest; called at most once.

    Returns:
        The metadata object, with tuples as lists and keys as strings, as stored metadata has them.

    Raises:
        ValueError: If the file or mapping does not hold JSON-serializable metadata, or an envelope is stale.
    """
    if isinstance(override, (str, Path)):
        source = f"Metadata file {override}"
        text = Path(override).read_text(encoding="utf-8")
    else:
        source = "metadata= mapping"
        mapping = dict(override)
        try:
            text = json.dumps(mapping)
        except (TypeError, RecursionError) as error:
            raise ValueError(f"{source} must hold JSON-serializable values: {error}") from error
    explicit = _json_object(text, source)
    if "metadata" in explicit and "artifact_sha256" in explicit:
        return _envelope_metadata(explicit, source, artifact, artifact_digest)
    return explicit


def read_metadata(path: str | Path, override: Mapping[str, Any] | str | Path | None = None) -> ExportMetadata:
    """Load validated metadata and reject stale companions or conflicting overrides.

    Args:
        path: Exported artifact. ONNX embeds its metadata; other formats keep it in ``<artifact>.rfdetr.json``.
        override: Missing semantics for an older artifact, as a mapping or a JSON file path, holding metadata or a
            digest envelope bound to this artifact.

    Returns:
        The validated prediction contract.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ImportError: If the artifact is ``.onnx`` and ``onnx`` is not installed.
        ValueError: If metadata is missing, malformed, stale, conflicting, or from an unsupported schema version.
    """
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    # The companion and an envelope override check the same digest, so a large artifact is hashed once, on first use.
    artifact_digest = functools.cache(functools.partial(_artifact_digest, artifact))
    stored: dict[str, Any] | None = None
    if artifact.suffix == ".onnx":
        try:
            onnx = importlib.import_module("onnx")
        except ImportError as error:
            raise ImportError("Reading ONNX inference metadata requires onnx. Install rfdetr[onnx].") from error
        model = onnx.load(str(artifact), load_external_data=False)
        raw = next((item.value for item in model.metadata_props if item.key == METADATA_KEY), None)
        if raw is not None:
            stored = _json_object(raw, f"Embedded metadata in {artifact}")
    sidecar = _sidecar_path(artifact)
    if sidecar.exists():
        source = f"Metadata companion {sidecar}"
        envelope = _json_object(sidecar.read_text(encoding="utf-8"), source)
        companion = _envelope_metadata(envelope, source, artifact, artifact_digest)
        if stored is not None and stored != companion:
            raise ValueError("Embedded metadata conflicts with companion metadata")
        stored = companion
    explicit = None if override is None else _override_metadata(override, artifact, artifact_digest)
    if stored is None and explicit is None:
        raise ValueError(f"No RF-DETR inference metadata for {artifact}. Pass metadata= with the missing semantics.")
    merged = dict(stored or {})
    if explicit is not None:
        for key, value in explicit.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Metadata override conflicts with embedded value for {key!r}")
            merged[key] = value
    # Checked before validation: a newer schema's unknown fields would otherwise surface as a raw pydantic error.
    schema_version = merged.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        producer = merged.get("producer_version")
        writer = f"rfdetr {producer}" if producer else "an unknown rfdetr version"
        raise ValueError(
            f"Metadata schema_version {schema_version!r} was written by {writer}; this rfdetr reads schema_version "
            f"{SCHEMA_VERSION} only. Upgrade rfdetr to load this artifact."
        )
    return ExportMetadata.model_validate_json(json.dumps(merged))


def metadata_from_model(
    model: Any,
    *,
    format: str,
    shape: tuple[int, int],
    batch_size: int,
    dynamic_batch: bool,
    backbone_only: bool = False,
    backend: str | None = None,
    max_batch_size: int | None = None,
) -> ExportMetadata:
    """Capture prediction semantics from a live RFDETR wrapper before conversion."""
    config = model.model_config
    task: ExportTask = (
        "backbone"
        if backbone_only
        else "segment"
        if config.segmentation_head
        else "keypoints"
        if config.use_grouppose_keypoints
        else "detect"
    )
    # The same derivation as native prediction, so the artifact maps class IDs to the names predict() returns.
    labels = prediction_labels(model)
    outputs: dict[str, str | int] = {"pred_boxes": "dets", "pred_logits": "labels"}
    if task == "segment":
        outputs["pred_masks"] = "masks"
    elif task == "keypoints":
        outputs["pred_keypoints"] = "keypoints"
    elif task == "backbone":
        outputs = {}
    try:
        producer_version = version("rfdetr")
    except PackageNotFoundError:
        producer_version = None
    return ExportMetadata(
        producer_version=producer_version,
        format=cast(ExportFormat, format),
        task=task,
        variant=getattr(model, "size", None),
        backend=backend,
        input_shape=(-1 if dynamic_batch else batch_size, config.num_channels, *shape),
        max_batch_size=max_batch_size,
        outputs=outputs,
        means=list(model.means),
        stds=list(model.stds),
        class_names=labels.class_names,
        class_id_to_name=labels.class_id_to_name,
        num_classes=labels.num_classes,
        num_select=model.model.postprocess.num_select,
        num_keypoints_per_class=labels.num_keypoints_per_class,
        trace_alpha=model.model.postprocess.trace_alpha,
        upsample_masks_to_image_size=model.model.postprocess.upsample_masks_to_image_size,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
