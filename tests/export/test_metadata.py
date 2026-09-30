# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Inference metadata is part of the exported artifact contract."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from rfdetr.export._runtime.metadata import ExportMetadata, metadata_from_model, read_metadata, write_metadata
from rfdetr.export._tflite.exporter import TFLiteConfig, TFLiteExporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.class_names import class_id_to_name


class TestSidecarMetadata:
    """Verify that sidecars stay bound to their artifacts."""

    def test_sidecar_round_trip_rejects_changed_artifact(self, tmp_path: Path) -> None:
        """A companion file identifies the exact artifact it describes."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model-v1")
        metadata = ExportMetadata(
            format="litert",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )

        write_metadata(artifact, metadata)
        assert read_metadata(artifact) == metadata

        artifact.write_bytes(b"model-v2")
        with pytest.raises(ValueError, match="digest"):
            read_metadata(artifact)

    def test_sidecar_survives_renaming_artifact_and_companion(self, tmp_path: Path) -> None:
        """A single-file artifact's bytes define its digest, so renaming both files is safe."""
        artifact = tmp_path / "exported_model.tflite"
        artifact.write_bytes(b"model")
        metadata = ExportMetadata(
            format="tflite",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )
        companion = write_metadata(artifact, metadata)
        assert companion is not None

        renamed_artifact = tmp_path / "model.tflite"
        renamed_companion = tmp_path / "model.tflite.rfdetr.json"
        artifact.rename(renamed_artifact)
        companion.rename(renamed_companion)

        assert read_metadata(renamed_artifact) == metadata

    def test_openvino_sidecar_survives_renaming_xml_and_weights(self, tmp_path: Path) -> None:
        """OpenVINO metadata binds XML and weights bytes without binding their names."""
        artifact = tmp_path / "exported_model.xml"
        weights = tmp_path / "exported_model.bin"
        artifact.write_text("<model/>", encoding="utf-8")
        weights.write_bytes(b"weights")
        metadata = ExportMetadata(
            format="openvino",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )
        companion = write_metadata(artifact, metadata)
        assert companion is not None
        renamed_artifact = tmp_path / "model.xml"
        renamed_weights = tmp_path / "model.bin"
        renamed_companion = tmp_path / "model.xml.rfdetr.json"
        artifact.rename(renamed_artifact)
        weights.rename(renamed_weights)
        companion.rename(renamed_companion)

        assert read_metadata(renamed_artifact) == metadata

    def test_sparse_coco_mapping_survives_metadata_sidecar_round_trip(self, tmp_path: Path) -> None:
        """JSON sidecars preserve sparse COCO category IDs as integer keys."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model")
        metadata = ExportMetadata(
            format="tflite",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=list(COCO_CLASS_NAMES),
            class_id_to_name=class_id_to_name(list(COCO_CLASS_NAMES), 90, []),
            num_classes=90,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
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
        metadata = ExportMetadata(
            format="coreml",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )
        write_metadata(artifact, metadata)

        data.rename(data.with_name("renamed.bin"))

        with pytest.raises(ValueError, match="digest"):
            read_metadata(artifact)

    def test_openvino_sidecar_covers_weights_and_rejects_conflicting_override(self, tmp_path: Path) -> None:
        """An IR companion covers both model files and fixed semantics cannot change."""
        artifact = tmp_path / "model.xml"
        weights = tmp_path / "model.bin"
        artifact.write_text("<model/>", encoding="utf-8")
        weights.write_bytes(b"weights-v1")
        metadata = ExportMetadata(
            format="openvino",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )

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
        metadata = ExportMetadata(
            format="litert",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": 0, "pred_logits": 1},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )
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


class TestMetadataCapture:
    """Verify model metadata capture and label mapping."""

    def test_backbone_metadata_survives_sidecar_round_trip(self, tmp_path: Path) -> None:
        """Backbone-only exports retain their task and empty prediction labels."""
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                num_channels=3,
                num_classes=0,
                num_keypoints_per_class=[],
                patch_size=16,
                num_windows=1,
            ),
            model=SimpleNamespace(
                args=SimpleNamespace(num_classes=0, num_keypoints_per_class=[]),
                postprocess=SimpleNamespace(num_select=0, trace_alpha=0.2, upsample_masks_to_image_size=True),
            ),
            class_names=[],
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            size="rfdetr-nano",
        )
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

    def test_metadata_from_model_preserves_sparse_coco_ids(
        self,
    ) -> None:
        """COCO names stay mapped to sparse category IDs, including gaps."""
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                num_channels=3,
                num_classes=90,
                num_keypoints_per_class=[],
                patch_size=16,
                num_windows=1,
            ),
            model=SimpleNamespace(
                args=SimpleNamespace(num_classes=90, num_keypoints_per_class=[]),
                postprocess=SimpleNamespace(num_select=100, trace_alpha=0.2, upsample_masks_to_image_size=True),
            ),
            class_names=list(COCO_CLASS_NAMES),
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            size="rfdetr-nano",
        )

        metadata = metadata_from_model(model, format="onnx", shape=(448, 448), batch_size=1, dynamic_batch=False)

        assert metadata.class_id_to_name[1] == "person"
        assert metadata.class_id_to_name[90] == "toothbrush"
        assert 12 not in metadata.class_id_to_name

    def test_class_id_to_name_preserves_sparse_coco_ids(
        self,
    ) -> None:
        """COCO class names use category IDs when the model includes all logit slots."""
        names = list(COCO_CLASS_NAMES)

        mapping = class_id_to_name(names, 90, [])

        assert mapping[1] == "person"
        assert mapping[18] == "dog"
        assert mapping[90] == "toothbrush"
        assert 12 not in mapping

    def test_class_id_to_name_skips_background_keypoint_slot(
        self,
    ) -> None:
        """Background-first keypoint names map to active slots only."""
        assert class_id_to_name(["person", "car"], 3, [0, 17, 4]) == {1: "person", 2: "car"}

    def test_metadata_from_model_preserves_background_first_keypoints(
        self,
    ) -> None:
        """Legacy keypoint slot zero has no class name."""
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=True,
                num_channels=3,
                num_classes=2,
                num_keypoints_per_class=[0, 3],
                patch_size=16,
                num_windows=1,
            ),
            model=SimpleNamespace(
                args=SimpleNamespace(num_classes=2, num_keypoints_per_class=[0, 3]),
                postprocess=SimpleNamespace(num_select=100, trace_alpha=0.2, upsample_masks_to_image_size=True),
            ),
            class_names=["person"],
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            size="rfdetr-keypoint-preview",
        )

        metadata = metadata_from_model(model, format="onnx", shape=(448, 448), batch_size=1, dynamic_batch=False)

        assert metadata.class_id_to_name == {1: "person"}
        assert metadata.num_keypoints_per_class == [0, 3]


class TestFormatMetadata:
    """Verify metadata behavior for format-specific interfaces."""

    def test_tflite_export_uses_signature_to_map_reordered_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Signature tensor indices identify boxes when output detail names are generic."""
        artifact = tmp_path / "model_fp32.tflite"
        metadata = ExportMetadata(
            format="tflite",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )
        graph = ExportGraph(
            model=torch.nn.Identity(),
            input_tensors=torch.zeros(1, 3, 448, 448),
            input_names=("input",),
            output_names=("dets", "labels"),
            dynamic_axes=None,
            shape=(448, 448),
            backbone_only=False,
            metadata=metadata,
        )
        interpreter = Mock()
        interpreter.get_input_details.return_value = [{"index": 0, "dtype": np.float32}]
        interpreter.get_output_details.return_value = [
            {"name": "Identity", "index": 12},
            {"name": "Identity_1", "index": 11},
        ]
        interpreter.get_signature_list.return_value = {"serving_default": {"outputs": ["dets", "labels"]}}
        interpreter.get_signature_runner.return_value.get_output_details.return_value = {
            "dets": {"index": 11},
            "labels": {"index": 12},
        }
        monkeypatch.setattr("rfdetr.export._tflite.inference._create_interpreter", Mock(return_value=interpreter))
        exporter = TFLiteExporter(TFLiteConfig(output_dir=tmp_path))
        monkeypatch.setattr(TFLiteExporter, "check_dependencies", Mock())
        artifact.write_bytes(b"converted graph")
        monkeypatch.setattr(exporter, "_convert", Mock(return_value=artifact))

        exporter(graph)

        loaded = read_metadata(artifact)
        assert loaded.outputs == {"pred_boxes": 1, "pred_logits": 0}
        assert loaded.input_layout == "NHWC"

    def test_onnx_metadata_keeps_user_notes(self, tmp_path: Path) -> None:
        """The reserved inference key does not replace a caller's notes."""
        onnx = pytest.importorskip("onnx")
        artifact = tmp_path / "model.onnx"
        graph = onnx.helper.make_graph([], "empty", [], [])
        model = onnx.helper.make_model(graph)
        notes = model.metadata_props.add()
        notes.key = "rfdetr_notes"
        notes.value = '{"run": 7}'
        onnx.save(model, artifact)
        metadata = ExportMetadata(
            format="onnx",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )

        write_metadata(artifact, metadata)

        assert read_metadata(artifact) == metadata
        saved = onnx.load(artifact)
        assert next(item.value for item in saved.metadata_props if item.key == "rfdetr_notes") == '{"run": 7}'
        assert not artifact.with_name("model.onnx.rfdetr.json").exists()

    def test_onnx_metadata_write_failure_preserves_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed metadata save leaves the successfully converted ONNX file intact."""
        onnx = pytest.importorskip("onnx")
        artifact = tmp_path / "model.onnx"
        graph = onnx.helper.make_graph([], "empty", [], [])
        onnx.save(onnx.helper.make_model(graph), artifact)
        original = artifact.read_bytes()
        metadata = ExportMetadata(
            format="onnx",
            task="detect",
            input_shape=(1, 3, 448, 448),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
            class_names=["object"],
            num_classes=1,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            num_select=100,
            trace_alpha=0.2,
            patch_size=16,
            num_windows=1,
        )

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

    def test_metadata_rejects_unsupported_preprocessing_and_schema(
        self,
    ) -> None:
        """An explicit legacy config cannot change native scaling or schema rules."""
        payload = {
            "format": "onnx",
            "task": "detect",
            "input_shape": (1, 3, 448, 448),
            "outputs": {"pred_boxes": "dets", "pred_logits": "labels"},
            "class_names": ["object"],
            "num_classes": 1,
            "means": [0.485, 0.456, 0.406],
            "stds": [0.229, 0.224, 0.225],
            "num_select": 100,
            "trace_alpha": 0.2,
            "patch_size": 16,
            "num_windows": 1,
        }
        with pytest.raises(ValueError, match="pixel_scale"):
            ExportMetadata(**payload, pixel_scale=1.0)
        with pytest.raises(ValueError, match="schema_version"):
            ExportMetadata(**payload, schema_version=2)
