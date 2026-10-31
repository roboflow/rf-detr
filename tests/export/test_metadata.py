# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Inference metadata is part of the exported artifact contract."""

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from torchvision.transforms import functional

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from rfdetr.detr import RFDETR
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._runtime.metadata import (
    METADATA_KEY,
    ExportMetadata,
    metadata_from_model,
    positional_metadata,
    read_metadata,
    write_metadata,
)
from rfdetr.export._tflite.exporter import TFLiteConfig, TFLiteExporter
from rfdetr.export.imports import _IS_OPENVINO_INSTALLED
from rfdetr.export.prepare import ExportGraph, resolve_output_names
from rfdetr.inference import RFDETRInference
from rfdetr.utilities.class_names import class_id_to_name
from tests._markers import onnx_and_onnxruntime_only, onnx_only

#: Digest of the ``b"model"`` artifact bytes, as a companion records it.
_MODEL_DIGEST = hashlib.sha256(b"model").hexdigest()


#: Fields of a valid TFLite detection contract; the tests below override only what their scenario changes.
_METADATA_FIELDS: dict[str, Any] = {
    "format": "tflite",
    "task": "detect",
    "input_shape": (1, 3, 448, 448),
    "outputs": {"pred_boxes": 0, "pred_logits": 1},
    "class_names": ["object"],
    "num_classes": 1,
    "means": [0.485, 0.456, 0.406],
    "stds": [0.229, 0.224, 0.225],
    "num_select": 100,
    "trace_alpha": 0.2,
    "patch_size": 16,
    "num_windows": 1,
}


def _make_metadata(**overrides: Any) -> ExportMetadata:
    """Build a valid detection ``ExportMetadata`` with ``overrides`` applied on top of ``_METADATA_FIELDS``.

    Examples:
        >>> _make_metadata().format, _make_metadata(format="onnx", num_select=2).num_select
        ('tflite', 2)
    """
    return ExportMetadata(**{**_METADATA_FIELDS, **overrides})


def _make_graph(
    model: torch.nn.Module,
    metadata: ExportMetadata,
    *,
    size: int = 32,
    output_names: tuple[str, str] = ("dets", "labels"),
) -> ExportGraph:
    """Build a static ``size`` x ``size`` detection ``ExportGraph`` around ``model`` that carries ``metadata``.

    Examples:
        >>> _make_graph(torch.nn.Identity(), _make_metadata()).shape
        (32, 32)
    """
    return ExportGraph(
        model=model,
        input_tensors=torch.zeros(1, 3, size, size),
        input_names=("input",),
        output_names=output_names,
        dynamic_axes=None,
        shape=(size, size),
        backbone_only=False,
        metadata=metadata,
    )


def _make_model(
    *,
    class_names: list[str],
    num_classes: int,
    num_select: int = 100,
    num_keypoints_per_class: Sequence[int] = (),
    keypoints: bool = False,
    size: str = "rfdetr-nano",
) -> SimpleNamespace:
    """Build a live-model double with the attributes ``metadata_from_model`` reads, for a detection or keypoint head.

    Examples:
        >>> model = _make_model(class_names=["a"], num_classes=1, num_keypoints_per_class=[0, 3], keypoints=True)
        >>> model.model.args.num_keypoints_per_class, model.size
        ([0, 3], 'rfdetr-nano')
    """
    return SimpleNamespace(
        model_config=SimpleNamespace(
            segmentation_head=False,
            use_grouppose_keypoints=keypoints,
            num_channels=3,
            num_classes=num_classes,
            num_keypoints_per_class=list(num_keypoints_per_class),
            patch_size=16,
            num_windows=1,
        ),
        model=SimpleNamespace(
            args=SimpleNamespace(num_classes=num_classes, num_keypoints_per_class=list(num_keypoints_per_class)),
            postprocess=SimpleNamespace(num_select=num_select, trace_alpha=0.2, upsample_masks_to_image_size=True),
        ),
        class_names=class_names,
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        size=size,
    )


@pytest.fixture
def metadata_source() -> RFDETR:
    """Provide a live wrapper's metadata state without constructing a backbone.

    Examples:
        A pytest fixture cannot be called directly.

        >>> model = metadata_source()  # doctest: +SKIP
        >>> model.class_names  # doctest: +SKIP
        ['object']
    """
    model = object.__new__(RFDETR)
    model.model_config = SimpleNamespace(
        segmentation_head=False,
        use_grouppose_keypoints=False,
        num_channels=3,
        num_keypoints_per_class=[],
        patch_size=16,
        num_windows=1,
    )
    model.model = SimpleNamespace(
        resolution=32,
        class_names=["object"],
        args=SimpleNamespace(num_classes=1, num_keypoints_per_class=[]),
        postprocess=SimpleNamespace(num_select=2, trace_alpha=0.2, upsample_masks_to_image_size=True),
    )
    model.means = [0.485, 0.456, 0.406]
    model.stds = [0.229, 0.224, 0.225]
    return model


class TestGraphOutputOrder:
    """Keep named and positional metadata aligned with the graph's output order."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.parametrize("reverse_order", [False, True])
    def test_metadata_follows_resolved_output_order(
        self, metadata_source: RFDETR, mocker: MockerFixture, task: str, reverse_order: bool
    ) -> None:
        """A resolver order change must reach both named and positional metadata."""
        config = metadata_source.model_config
        config.segmentation_head = task == "segment"
        config.use_grouppose_keypoints = task == "keypoints"
        if task == "keypoints":
            config.num_keypoints_per_class = [3]
            metadata_source.model.args.num_keypoints_per_class = [3]
        graph_names = resolve_output_names(config, backbone_only=False, backbone=None)
        if reverse_order:
            graph_names.reverse()
            # Simulate a future graph order change without duplicating the resolver's task rules.
            mocker.patch("rfdetr.export.prepare.resolve_output_names", return_value=graph_names)

        metadata = metadata_from_model(
            metadata_source, format="onnx", shape=(32, 32), batch_size=1, dynamic_batch=False
        )

        assert metadata.task == task
        assert list(metadata.outputs.values()) == graph_names
        semantic_names = {
            "dets": "pred_boxes",
            "labels": "pred_logits",
            "masks": "pred_masks",
            "keypoints": "pred_keypoints",
        }
        assert metadata.outputs == {semantic_names[name]: name for name in graph_names}
        assert positional_metadata(metadata).outputs == {
            semantic_names[name]: graph_names.index(name) for name in graph_names
        }

    def test_backbone_metadata_has_no_prediction_outputs(self, metadata_source: RFDETR) -> None:
        """Backbone metadata does not need a backbone instance to resolve prediction outputs."""
        metadata = metadata_from_model(
            metadata_source, format="onnx", shape=(32, 32), batch_size=1, dynamic_batch=False, backbone_only=True
        )

        assert metadata.task == "backbone"
        assert metadata.outputs == {}


class TestWriteExportMetadata:
    """Verify public metadata publication for artifacts exported earlier."""

    @pytest.mark.skipif(not _IS_OPENVINO_INSTALLED, reason="openvino not installed")
    def test_openvino_positional_sidecar(self, tmp_path: Path, metadata_source: RFDETR) -> None:
        """Apply the real exporter mapping and bind metadata to XML and weights."""
        artifact = tmp_path / "model.xml"
        artifact.write_bytes(b"<model/>")
        weights = artifact.with_suffix(".bin")
        weights.write_bytes(b"weights")

        sidecar = metadata_source.write_export_metadata(artifact, format="openvino", batch_size=2)

        assert sidecar == tmp_path / "model.xml.rfdetr.json"
        metadata = read_metadata(artifact)
        assert metadata.format == "openvino"
        assert metadata.outputs == {"pred_boxes": 0, "pred_logits": 1}
        assert metadata.input_shape == (2, 3, 32, 32)
        assert metadata.means == metadata_source.means
        assert metadata.stds == metadata_source.stds
        assert metadata.num_select == 2
        assert metadata.trace_alpha == 0.2
        weights.write_bytes(b"changed weights")
        with pytest.raises(ValueError, match="digest"):
            read_metadata(artifact)

    def test_missing_artifact(self, tmp_path: Path, metadata_source: RFDETR) -> None:
        """Refuse a missing artifact before attempting metadata publication."""
        artifact = tmp_path / "missing.xml"

        with pytest.raises(FileNotFoundError) as error:
            metadata_source.write_export_metadata(artifact, format="openvino")

        assert error.value.args == (artifact,)
        assert not artifact.with_name("missing.xml.rfdetr.json").exists()


class _TinyDetector(torch.nn.Module):
    """Provide deterministic linear layers for real ONNX export and calibration.

    Examples:
        >>> model = _TinyDetector()
        >>> [tuple(output.shape) for output in model(torch.zeros(1, 3, 32, 32))]
        [(1, 2, 4), (1, 2, 2)]
    """

    def __init__(self, num_classes: int = 1) -> None:
        """Initialize fixed weights so export does not depend on the random seed.

        Args:
            num_classes: Foreground class slots, excluding the trailing no-object slot.

        Examples:
            >>> model = _TinyDetector()
            >>> bool(torch.all(model.hidden.weight == 0.1))
            True
        """
        super().__init__()
        self.hidden = torch.nn.Linear(3, 8)
        self.boxes = torch.nn.Linear(8, 8)
        self.logits = torch.nn.Linear(8, 2 * (num_classes + 1))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized boxes and logits from the spatially averaged image.

        Examples:
            >>> boxes, logits = _TinyDetector()(torch.zeros(1, 3, 32, 32))
            >>> tuple(boxes.shape), tuple(logits.shape)
            ((1, 2, 4), (1, 2, 2))
        """
        hidden = self.hidden(images.mean(dim=(2, 3))).relu()
        return (
            self.boxes(hidden).sigmoid().reshape(-1, 2, 4),
            self.logits(hidden).reshape(-1, 2, self.logits.out_features // 2),
        )


class TestSidecarMetadata:
    """Verify that sidecars stay bound to their artifacts."""

    def test_round_trip_rejects_changed_artifact(self, tmp_path: Path) -> None:
        """A companion file identifies the exact artifact it describes."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model-v1")
        metadata = _make_metadata(format="litert")

        write_metadata(artifact, metadata)
        assert read_metadata(artifact) == metadata

        artifact.write_bytes(b"model-v2")
        with pytest.raises(ValueError, match="digest"):
            read_metadata(artifact)

    def test_survives_renaming_artifact_and_companion(self, tmp_path: Path) -> None:
        """A single-file artifact's bytes define its digest, so renaming both files is safe."""
        artifact = tmp_path / "exported_model.tflite"
        artifact.write_bytes(b"model")
        metadata = _make_metadata()
        companion = write_metadata(artifact, metadata)
        assert companion is not None

        renamed_artifact = tmp_path / "model.tflite"
        renamed_companion = tmp_path / "model.tflite.rfdetr.json"
        artifact.rename(renamed_artifact)
        companion.rename(renamed_companion)

        assert read_metadata(renamed_artifact) == metadata

    def test_openvino_survives_renaming_xml_and_weights(self, tmp_path: Path) -> None:
        """OpenVINO metadata binds XML and weights bytes without binding their names."""
        artifact = tmp_path / "exported_model.xml"
        weights = tmp_path / "exported_model.bin"
        artifact.write_text("<model/>", encoding="utf-8")
        weights.write_bytes(b"weights")
        metadata = _make_metadata(format="openvino")
        companion = write_metadata(artifact, metadata)
        assert companion is not None
        renamed_artifact = tmp_path / "model.xml"
        renamed_weights = tmp_path / "model.bin"
        renamed_companion = tmp_path / "model.xml.rfdetr.json"
        artifact.rename(renamed_artifact)
        weights.rename(renamed_weights)
        companion.rename(renamed_companion)

        assert read_metadata(renamed_artifact) == metadata

    def test_sparse_coco_mapping_survives_round_trip(self, tmp_path: Path) -> None:
        """JSON sidecars preserve sparse COCO category IDs as integer keys."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model")
        metadata = _make_metadata(
            class_names=list(COCO_CLASS_NAMES),
            class_id_to_name=class_id_to_name(list(COCO_CLASS_NAMES), 90, []),
            num_classes=90,
        )

        write_metadata(artifact, metadata)

        restored = read_metadata(artifact)
        assert restored.class_id_to_name == metadata.class_id_to_name
        assert restored.class_id_to_name[18] == "dog"
        assert 12 not in restored.class_id_to_name

    def test_bundle_digest_keeps_internal_relative_paths(self, tmp_path: Path) -> None:
        """Directory bundles include internal file paths in their digest."""
        artifact = tmp_path / "model.mlpackage"
        data = artifact / "Data" / "weights.bin"
        data.parent.mkdir(parents=True)
        data.write_bytes(b"weights")
        metadata = _make_metadata(format="coreml")
        write_metadata(artifact, metadata)

        data.rename(data.with_name("renamed.bin"))

        with pytest.raises(ValueError, match="digest"):
            read_metadata(artifact)

    def test_openvino_covers_weights_and_rejects_conflicting_override(self, tmp_path: Path) -> None:
        """An IR companion covers both model files and fixed semantics cannot change."""
        artifact = tmp_path / "model.xml"
        weights = tmp_path / "model.bin"
        artifact.write_text("<model/>", encoding="utf-8")
        weights.write_bytes(b"weights-v1")
        metadata = _make_metadata(format="openvino")

        write_metadata(artifact, metadata)
        with pytest.raises(ValueError, match="conflicts"):
            read_metadata(artifact, {"task": "segment"})
        weights.write_bytes(b"weights-v2")
        with pytest.raises(ValueError, match="digest"):
            read_metadata(artifact)

    def test_explicit_companion_cannot_describe_another_artifact(self, tmp_path: Path) -> None:
        """An explicit JSON companion remains bound to its source artifact."""
        source = tmp_path / "source.tflite"
        destination = tmp_path / "destination.tflite"
        source.write_bytes(b"source graph")
        destination.write_bytes(b"different graph")
        metadata = _make_metadata(format="litert")
        companion = write_metadata(source, metadata)
        assert companion is not None
        envelope = json.loads(companion.read_text(encoding="utf-8"))

        with pytest.raises(ValueError, match="digest"):
            read_metadata(destination, envelope)

    def test_missing_legacy_metadata_requires_explicit_semantics(self, tmp_path: Path) -> None:
        """A raw artifact cannot choose a task or label space by shape alone."""
        artifact = tmp_path / "legacy.tflite"
        artifact.write_bytes(b"legacy graph")
        with pytest.raises(ValueError, match="Pass metadata="):
            read_metadata(artifact)

    def test_newer_schema_names_its_producer_and_asks_to_upgrade(self, tmp_path: Path) -> None:
        """A companion from a newer schema asks for an upgrade instead of failing on its unknown field."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model")
        metadata = _make_metadata()
        companion = write_metadata(artifact, metadata)
        assert companion is not None
        envelope = json.loads(companion.read_text(encoding="utf-8"))
        envelope["metadata"].update(schema_version=2, producer_version="9.9.9", future_field=True)
        companion.write_text(json.dumps(envelope), encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            read_metadata(artifact)

        assert "written by rfdetr 9.9.9" in str(exc_info.value)
        assert "Upgrade rfdetr" in str(exc_info.value)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_new_file_mode_follows_umask(self, tmp_path: Path, process_umask: int) -> None:
        """A first sidecar gets the mode ``open()`` would give it, so other users can read it, not ``0o600``."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model")
        metadata = _make_metadata()

        companion = write_metadata(artifact, metadata)

        assert companion is not None
        assert stat.S_IMODE(companion.stat().st_mode) == 0o666 & ~process_umask

    @pytest.mark.parametrize(
        ("companion", "message"),
        [
            ("{", "not valid JSON"),
            pytest.param("[" * 100_000, "not valid JSON", id="nested-too-deep"),
            ("[]", "must hold a JSON object"),
            ('{"metadata": {}}', "missing ['artifact_sha256']"),
            pytest.param(json.dumps({"artifact_sha256": _MODEL_DIGEST}), "missing ['metadata']", id="missing-metadata"),
            pytest.param(
                json.dumps({"artifact_sha256": _MODEL_DIGEST, "metadata": []}),
                "JSON object under 'metadata'",
                id="metadata-not-an-object",
            ),
        ],
    )
    def test_malformed_companion_raises_value_error(self, tmp_path: Path, companion: str, message: str) -> None:
        """A truncated or wrong-shaped companion raises ``ValueError``, not ``KeyError``, ``TypeError`` or recursion."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model")
        (tmp_path / "model.tflite.rfdetr.json").write_text(companion, encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            read_metadata(artifact)

        assert message in str(exc_info.value)

    @onnx_only
    def test_companion_conflicting_with_embedded_onnx_metadata_is_rejected(self, tmp_path: Path) -> None:
        """A companion cannot relabel an ONNX model whose embedded metadata says something else."""
        import onnx  # optional dependency, gated by onnx_only

        artifact = tmp_path / "model.onnx"
        onnx.save(onnx.helper.make_model(onnx.helper.make_graph([], "empty", [], [])), artifact)
        metadata = _make_metadata(format="onnx", outputs={"pred_boxes": "dets", "pred_logits": "labels"})
        write_metadata(artifact, metadata)
        # A non-ONNX format takes the companion path, bound to the bytes that now embed the ONNX metadata.
        write_metadata(artifact, metadata.model_copy(update={"format": "tensorrt"}))

        with pytest.raises(ValueError, match="Embedded metadata conflicts with companion metadata"):
            read_metadata(artifact)


class TestResizeAntialias:
    """Verify metadata defaults and per-call overrides for exported prediction."""

    def test_antialiased_resize_validates(self) -> None:
        """An antialiased resize is a valid artifact preprocessing contract."""
        metadata = _make_metadata(input_shape=(1, 3, 32, 32), num_select=2, resize_antialias=True)

        assert metadata.resize_antialias is True

    @pytest.mark.parametrize("metadata_antialias", [False, True])
    @pytest.mark.parametrize(
        "prediction_kwargs",
        [
            pytest.param({}, id="omitted"),
            pytest.param({"antialias": None}, id="none"),
            pytest.param({"antialias": False}, id="explicit-false"),
            pytest.param({"antialias": True}, id="explicit-true"),
        ],
    )
    @pytest.mark.parametrize("stream", [False, True])
    def test_prediction_resolves_antialias(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        metadata_antialias: bool,
        prediction_kwargs: dict[str, bool | None],
        stream: bool,
    ) -> None:
        """Legacy metadata supplies the default, while either explicit boolean wins."""
        artifact = tmp_path / "legacy.tflite"
        artifact.write_bytes(b"legacy graph")
        metadata = {
            "format": "tflite",
            "task": "detect",
            "input_shape": (1, 3, 32, 32),
            "outputs": {"pred_boxes": 0, "pred_logits": 1},
            "class_names": ["object"],
            "num_classes": 1,
            "means": [0.485, 0.456, 0.406],
            "stds": [0.229, 0.224, 0.225],
            "num_select": 2,
            "trace_alpha": 0.2,
            "patch_size": 16,
            "num_windows": 1,
            "resize_antialias": metadata_antialias,
        }
        runtime = SimpleNamespace(
            device=torch.device("cpu"),
            run=mocker.Mock(
                return_value={
                    "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
                    "pred_logits": torch.tensor([[[4.0, -4.0]]]),
                }
            ),
            info={"backend": "tflite"},
        )
        mocker.patch("rfdetr.export._runtime.context.load_runtime", return_value=runtime)
        resize = mocker.patch("rfdetr._prediction.core.F.resize", wraps=functional.resize)
        predictor = RFDETRInference(artifact, metadata=metadata)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        results = list(predictor.predict([image], stream=stream, **prediction_kwargs))

        expected_antialias = prediction_kwargs.get("antialias")
        if expected_antialias is None:
            expected_antialias = metadata_antialias
        assert resize.call_args.kwargs["antialias"] is expected_antialias
        assert len(results) == 1
        assert predictor._export_context.antialias is metadata_antialias


class TestMetadataCapture:
    """Verify model metadata capture and label mapping."""

    def test_positional_metadata_maps_names_without_mutating_source(self) -> None:
        """Copy named metadata into positional input and output identifiers.

        Tuple-returning runtimes use output mapping order, while the source retains its named interface and semantics.
        """
        metadata = _make_metadata(
            format="onnx",
            input_shape=(1, 3, 8, 8),
            input_name="images",
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
            num_select=2,
        )
        original = metadata.model_dump()

        result = positional_metadata(metadata)

        assert result is not metadata
        assert result.input_name == 0
        assert result.outputs == {"pred_boxes": 0, "pred_logits": 1}
        assert result.model_dump() == original | {"input_name": 0, "outputs": {"pred_boxes": 0, "pred_logits": 1}}
        assert metadata.model_dump() == original

    def test_metadata_from_model_normalizes_numpy_scalars(self) -> None:
        """Checkpoint NumPy scalars become Python numerics before strict validation."""
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=True,
                num_channels=np.int64(3),
                patch_size=np.int64(16),
                num_windows=np.int64(1),
            ),
            model=SimpleNamespace(
                args=SimpleNamespace(num_classes=np.int64(2), num_keypoints_per_class=[np.int64(0), np.int64(3)]),
                postprocess=SimpleNamespace(
                    num_select=np.int64(100), trace_alpha=np.float32(0.2), upsample_masks_to_image_size=True
                ),
            ),
            class_names=["person"],
            means=[np.float32(0.485), np.float32(0.456), np.float32(0.406)],
            stds=[np.float32(0.229), np.float32(0.224), np.float32(0.225)],
        )

        metadata = metadata_from_model(model, format="onnx", shape=(32, 32), batch_size=1, dynamic_batch=False)

        assert (metadata.num_classes, metadata.num_select, metadata.patch_size, metadata.num_windows) == (2, 100, 16, 1)
        assert metadata.num_keypoints_per_class == [0, 3]
        assert metadata.class_id_to_name == {1: "person"}
        assert all(
            type(value) is int
            for value in (
                metadata.num_classes,
                metadata.num_select,
                metadata.patch_size,
                metadata.num_windows,
                *metadata.input_shape,
                *metadata.num_keypoints_per_class,
                *metadata.class_id_to_name,
            )
        )
        assert type(metadata.trace_alpha) is float
        assert all(type(value) is float for value in metadata.means + metadata.stds)

    def test_backbone_metadata_survives_sidecar_round_trip(self, tmp_path: Path) -> None:
        """Backbone-only exports retain their task and empty prediction labels."""
        model = _make_model(class_names=[], num_classes=0, num_select=0)
        metadata = metadata_from_model(
            model,
            format="tflite",
            shape=(448, 448),
            batch_size=1,
            dynamic_batch=False,
            backbone_only=True,
        )
        artifact = tmp_path / "backbone.tflite"
        artifact.write_bytes(b"backbone")

        write_metadata(artifact, metadata)

        restored = read_metadata(artifact)
        assert restored.task == "backbone"
        assert restored.class_names == []
        assert restored.outputs == {}

    def test_metadata_from_model_preserves_sparse_coco_ids(self) -> None:
        """COCO names stay mapped to sparse category IDs, including gaps."""
        model = _make_model(class_names=list(COCO_CLASS_NAMES), num_classes=90)

        metadata = metadata_from_model(model, format="onnx", shape=(448, 448), batch_size=1, dynamic_batch=False)

        assert metadata.class_id_to_name[1] == "person"
        assert metadata.class_id_to_name[90] == "toothbrush"
        assert 12 not in metadata.class_id_to_name

    def test_class_id_to_name_preserves_sparse_coco_ids(self) -> None:
        """COCO class names use category IDs when the model includes all logit slots."""
        names = list(COCO_CLASS_NAMES)

        mapping = class_id_to_name(names, 90, [])

        assert mapping[1] == "person"
        assert mapping[18] == "dog"
        assert mapping[90] == "toothbrush"
        assert 12 not in mapping

    def test_class_id_to_name_skips_background_keypoint_slot(self) -> None:
        """Background-first keypoint names map to active slots only."""
        assert class_id_to_name(["person", "car"], 3, [0, 17, 4]) == {1: "person", 2: "car"}

    def test_metadata_from_model_preserves_background_first_keypoints(self) -> None:
        """Legacy keypoint slot zero has no class name."""
        model = _make_model(
            class_names=["person"],
            num_classes=2,
            num_keypoints_per_class=[0, 3],
            keypoints=True,
            size="rfdetr-keypoint-preview",
        )

        metadata = metadata_from_model(model, format="onnx", shape=(448, 448), batch_size=1, dynamic_batch=False)

        assert metadata.class_id_to_name == {1: "person"}
        assert metadata.num_keypoints_per_class == [0, 3]


class TestFormatMetadata:
    """Verify metadata behavior for format-specific interfaces."""

    @onnx_and_onnxruntime_only
    def test_missing_args_export_preserves_head_size_and_zero_indexed_names(
        self, tmp_path: Path, metadata_source: RFDETR
    ) -> None:
        """A 90-class head with 80 COCO names exports and predicts without training arguments."""
        del metadata_source.model.args
        metadata_source.model_config.num_classes = 90
        metadata_source.model.class_names = list(COCO_CLASS_NAMES)
        detector = _TinyDetector(num_classes=90).eval()
        with torch.no_grad():
            detector.logits.weight.zero_()
            detector.logits.bias.fill_(-10)
            detector.logits.bias[1] = 10
            detector.logits.bias[92] = 10
        metadata = metadata_from_model(
            metadata_source, format="onnx", shape=(32, 32), batch_size=1, dynamic_batch=False
        )
        graph = _make_graph(detector, metadata)

        artifact = OnnxExporter(OnnxConfig(output_dir=tmp_path, variant_name="rfdetr-nano", verbose=False))(graph)
        detections = RFDETRInference(artifact, device="cpu").predict(
            np.full((32, 32, 3), 128, dtype=np.uint8), threshold=0.5
        )

        assert read_metadata(artifact).num_classes == 90
        assert read_metadata(artifact).class_id_to_name == dict(enumerate(COCO_CLASS_NAMES))
        np.testing.assert_array_equal(detections.class_id, [1, 1])
        np.testing.assert_array_equal(detections.data["class_name"], ["bicycle", "bicycle"])

    @onnx_and_onnxruntime_only
    @pytest.mark.parametrize("quantization", [None, "fp32", "int8"])
    @pytest.mark.parametrize("composed", [False, True])
    def test_onnx_final_artifacts_load_and_predict(
        self, tmp_path: Path, quantization: str | None, composed: bool
    ) -> None:
        """Every final ONNX file, including the INT8 export's FP32 baseline, carries a usable contract."""
        import onnx  # optional dependency, gated by onnx_and_onnxruntime_only

        metadata = _make_metadata(
            format="tensorrt" if composed else "onnx",
            input_shape=(1, 3, 32, 32),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
            num_select=2,
        )
        graph = _make_graph(_TinyDetector().eval(), metadata)
        exporter = OnnxExporter(
            OnnxConfig(
                output_dir=tmp_path,
                variant_name="rfdetr-nano",
                quantization=quantization,
                calibration_data=np.full((2, 3, 32, 32), 0.25, dtype=np.float32),
                verbose=False,
            )
        )

        returned = exporter(graph)

        baseline = tmp_path / "rfdetr-nano.onnx"
        artifacts = [baseline]
        if quantization == "int8":
            assert returned == tmp_path / "rfdetr-nano_int8.onnx"
            artifacts.append(returned)
            assert any(node.op_type == "QuantizeLinear" for node in onnx.load(returned).graph.node)
        else:
            assert returned == baseline
        assert set(tmp_path.glob("*.onnx")) == set(artifacts)
        for artifact in artifacts:
            if composed:
                assert METADATA_KEY not in {prop.key for prop in onnx.load(artifact).metadata_props}
                continue
            assert read_metadata(artifact) == metadata
            detections = RFDETRInference(artifact, device="cpu").predict(
                np.full((32, 32, 3), 128, dtype=np.uint8), threshold=0.0
            )
            assert len(detections) == 2
            assert detections.xyxy.shape == (2, 4)
            assert np.isfinite(detections.xyxy).all()
            assert np.isfinite(detections.confidence).all()

    @pytest.mark.parametrize(("boxes_name", "logits_name"), [("dets", "labels"), ("box_output", "class_output")])
    def test_tflite_export_uses_signature_to_map_reordered_outputs(
        self, tmp_path: Path, mocker: MockerFixture, boxes_name: str, logits_name: str
    ) -> None:
        """Signature tensor indices identify boxes when output detail names are generic."""
        artifact = tmp_path / "model_fp32.tflite"
        metadata = _make_metadata(outputs={"pred_boxes": boxes_name, "pred_logits": logits_name})
        graph = _make_graph(torch.nn.Identity(), metadata, size=448, output_names=(boxes_name, logits_name))
        interpreter = mocker.Mock()
        interpreter.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
        interpreter.get_output_details.return_value = [
            {"name": "Identity", "index": 12},
            {"name": "Identity_1", "index": 11},
        ]
        interpreter.get_signature_list.return_value = {"serving_default": {"outputs": [boxes_name, logits_name]}}
        interpreter.get_signature_runner.return_value.get_output_details.return_value = {
            boxes_name: {"index": 11},
            logits_name: {"index": 12},
        }
        mocker.patch("rfdetr.export._tflite.inference._create_interpreter", return_value=interpreter)
        exporter = TFLiteExporter(TFLiteConfig(output_dir=tmp_path))
        mocker.patch.object(TFLiteExporter, "check_dependencies")
        artifact.write_bytes(b"converted graph")
        mocker.patch.object(exporter, "_convert", return_value=artifact)

        exporter(graph)

        loaded = read_metadata(artifact)
        assert loaded.outputs == {"pred_boxes": 1, "pred_logits": 0}
        assert loaded.input_layout == "NHWC"

    @onnx_only
    def test_onnx_metadata_keeps_user_notes(self, tmp_path: Path) -> None:
        """The reserved inference key does not replace a caller's notes."""
        import onnx  # optional dependency, gated by onnx_only

        artifact = tmp_path / "model.onnx"
        graph = onnx.helper.make_graph([], "empty", [], [])
        model = onnx.helper.make_model(graph)
        notes = model.metadata_props.add()
        notes.key = "rfdetr_notes"
        notes.value = '{"run": 7}'
        onnx.save(model, artifact)
        metadata = _make_metadata(format="onnx", outputs={"pred_boxes": "dets", "pred_logits": "labels"})

        write_metadata(artifact, metadata)

        assert read_metadata(artifact) == metadata
        saved = onnx.load(artifact)
        assert next(item.value for item in saved.metadata_props if item.key == "rfdetr_notes") == '{"run": 7}'
        assert not artifact.with_name("model.onnx.rfdetr.json").exists()

    @onnx_only
    def test_onnx_metadata_write_failure_preserves_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed metadata save leaves the successfully converted ONNX file intact."""
        import onnx  # optional dependency, gated by onnx_only

        artifact = tmp_path / "model.onnx"
        graph = onnx.helper.make_graph([], "empty", [], [])
        onnx.save(onnx.helper.make_model(graph), artifact)
        original = artifact.read_bytes()
        metadata = _make_metadata(format="onnx", outputs={"pred_boxes": "dets", "pred_logits": "labels"})

        def fail_after_writing_temporary_model(model: object, destination: str) -> None:
            """Simulate an interrupted save to the writer's temporary file.

            Examples:
                Requires the temporary path managed by the enclosing test.
                >>> fail_after_writing_temporary_model(model, destination)  # doctest: +SKIP
            """
            Path(destination).write_bytes(b"partial metadata output")
            raise OSError("simulated disk failure")

        monkeypatch.setattr(onnx, "save", fail_after_writing_temporary_model)

        with pytest.raises(OSError, match="simulated disk failure"):
            write_metadata(artifact, metadata)

        assert artifact.read_bytes() == original
        assert not list(tmp_path.glob(".model.*.onnx"))

    @pytest.mark.parametrize(
        ("update", "message"),
        [
            pytest.param({"input_shape": (0, 3, 448, 448)}, "batch must be -1 or positive", id="zero-batch"),
            pytest.param({"input_shape": (-2, 3, 448, 448)}, "batch must be -1 or positive", id="batch-below-dynamic"),
            pytest.param({"input_shape": (1, 3, 0, 448)}, "positive channel and spatial shape", id="zero-height"),
            pytest.param({"max_batch_size": 4}, "max_batch_size requires a dynamic batch", id="max-batch-static-batch"),
            pytest.param({"means": [0.485, 0.456]}, "means and positive stds must match", id="means-length"),
            pytest.param({"means": [float("nan"), 0.456, 0.406]}, "means and positive stds must match", id="nan-mean"),
            pytest.param({"stds": [0.229, 0.0, 0.225]}, "means and positive stds must match", id="zero-std"),
            pytest.param({"task": "segment"}, "outputs missing semantic values: ['pred_masks']", id="segment-no-masks"),
            pytest.param(
                {"task": "keypoints", "num_keypoints_per_class": [17]},
                "outputs missing semantic values: ['pred_keypoints']",
                id="keypoints-no-output",
            ),
            pytest.param(
                {"outputs": {"pred_boxes": "dets", "pred_logits": "dets"}},
                "its own runtime output",
                id="shared-output-name",
            ),
            pytest.param(
                {"outputs": {"pred_boxes": -1, "pred_logits": 0}},
                "output positions must be non-negative",
                id="negative-output-position",
            ),
            pytest.param({"class_names": []}, "class_names are required", id="empty-class-names"),
            pytest.param({"num_classes": -1}, "num_classes, num_select, patch_size", id="negative-num-classes"),
            pytest.param({"num_select": -1}, "num_classes, num_select, patch_size", id="negative-num-select"),
            pytest.param({"patch_size": 0}, "num_classes, num_select, patch_size", id="zero-patch-size"),
            pytest.param(
                {"trace_alpha": -0.1}, "trace_alpha must be finite and non-negative", id="negative-trace-alpha"
            ),
            pytest.param({"trace_alpha": float("inf")}, "trace_alpha must be finite", id="infinite-trace-alpha"),
            pytest.param({"pixel_scale": 1.0}, "pixel_scale must match native", id="pixel-scale"),
            pytest.param(
                {
                    "task": "keypoints",
                    "outputs": {"pred_boxes": "dets", "pred_logits": "labels", "pred_keypoints": "kp"},
                },
                "num_keypoints_per_class is required",
                id="keypoints-no-schema",
            ),
            pytest.param(
                {
                    "task": "keypoints",
                    "outputs": {"pred_boxes": "dets", "pred_logits": "labels", "pred_keypoints": "kp"},
                    "num_keypoints_per_class": [0],
                },
                "no active keypoint class",
                id="keypoints-all-inactive",
            ),
            pytest.param(
                {"num_keypoints_per_class": [-1]},
                "num_keypoints_per_class counts must be non-negative",
                id="detect-negative-keypoint-count",
            ),
            pytest.param(
                {
                    "task": "keypoints",
                    "outputs": {"pred_boxes": "dets", "pred_logits": "labels", "pred_keypoints": "kp"},
                    "num_keypoints_per_class": [-1, 17],
                },
                "num_keypoints_per_class counts must be non-negative",
                id="keypoints-mixed-negative-positive-counts",
            ),
            pytest.param({"num_classes": 91}, "class_id_to_name is required", id="sparse-labels-without-map"),
            pytest.param({"schema_version": 2}, "schema_version", id="schema-version"),
            pytest.param({"unknown_field": 1}, "Extra inputs are not permitted", id="extra-field"),
            pytest.param({"num_classes": "1"}, "Input should be a valid integer", id="strict-integer"),
        ],
    )
    def test_rejects_contract_violations(self, update: dict[str, object], message: str) -> None:
        """Metadata that cannot describe one safe prediction contract fails at construction.

        Each case changes one field of an otherwise valid detection payload, so the message pins the rejecting rule: an
        explicit legacy config cannot change native scaling, the schema, the output mapping, or the label layout.
        """
        payload = {**_METADATA_FIELDS, "format": "onnx", "outputs": {"pred_boxes": "dets", "pred_logits": "labels"}}

        with pytest.raises(ValueError) as exc_info:
            ExportMetadata.model_validate({**payload, **update})

        assert message in str(exc_info.value)
