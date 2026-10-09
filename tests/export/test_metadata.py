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
from tests._markers import onnx_only

#: Digest of the ``b"model"`` artifact bytes, as a companion records it.
_MODEL_DIGEST = hashlib.sha256(b"model").hexdigest()


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

    def test_newer_schema_names_its_producer_and_asks_to_upgrade(self, tmp_path: Path) -> None:
        """A companion from a newer schema asks for an upgrade instead of failing on its unknown field."""
        artifact = tmp_path / "model.tflite"
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
        envelope = json.loads(companion.read_text(encoding="utf-8"))
        envelope["metadata"].update(schema_version=2, producer_version="9.9.9", future_field=True)
        companion.write_text(json.dumps(envelope), encoding="utf-8")

        with pytest.raises(ValueError, match=r"written by rfdetr 9\.9\.9.*Upgrade rfdetr"):
            read_metadata(artifact)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_new_sidecar_mode_follows_umask(self, tmp_path: Path, process_umask: int) -> None:
        """A first sidecar gets the mode ``open()`` would give it, so other users can read it, not ``0o600``."""
        artifact = tmp_path / "model.tflite"
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
        assert stat.S_IMODE(companion.stat().st_mode) == 0o666 & ~process_umask

    @pytest.mark.parametrize(
        ("companion", "match"),
        [
            pytest.param("{", "not valid JSON", id="truncated"),
            pytest.param("[" * 100_000, "not valid JSON", id="nested-too-deep"),
            pytest.param("[]", "must hold a JSON object", id="not-an-object"),
            pytest.param('{"metadata": {}}', r"missing \['artifact_sha256'\]", id="missing-digest"),
            pytest.param(
                json.dumps({"artifact_sha256": _MODEL_DIGEST}), r"missing \['metadata'\]", id="missing-metadata"
            ),
            pytest.param(
                json.dumps({"artifact_sha256": _MODEL_DIGEST, "metadata": []}),
                "JSON object under 'metadata'",
                id="metadata-not-an-object",
            ),
        ],
    )
    def test_malformed_companion_raises_value_error(self, tmp_path: Path, companion: str, match: str) -> None:
        """A truncated or wrong-shaped companion raises ``ValueError``, not ``KeyError``, ``TypeError`` or recursion."""
        artifact = tmp_path / "model.tflite"
        artifact.write_bytes(b"model")
        (tmp_path / "model.tflite.rfdetr.json").write_text(companion, encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            read_metadata(artifact)

    @onnx_only
    def test_companion_conflicting_with_embedded_onnx_metadata_is_rejected(self, tmp_path: Path) -> None:
        """A companion cannot relabel an ONNX model whose embedded metadata says something else."""
        import onnx  # optional dependency, gated by onnx_only

        artifact = tmp_path / "model.onnx"
        onnx.save(onnx.helper.make_model(onnx.helper.make_graph([], "empty", [], [])), artifact)
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
        # A non-ONNX format takes the companion path, bound to the bytes that now embed the ONNX metadata.
        write_metadata(artifact, metadata.model_copy(update={"format": "tensorrt"}))

        with pytest.raises(ValueError, match="Embedded metadata conflicts with companion metadata"):
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

    @pytest.mark.parametrize(
        ("update", "match"),
        [
            pytest.param({"input_shape": (0, 3, 448, 448)}, "batch must be -1 or positive", id="zero-batch"),
            pytest.param({"input_shape": (-2, 3, 448, 448)}, "batch must be -1 or positive", id="batch-below-dynamic"),
            pytest.param({"input_shape": (1, 3, 0, 448)}, "positive channel and spatial shape", id="zero-height"),
            pytest.param({"max_batch_size": 4}, "max_batch_size requires a dynamic batch", id="max-batch-static-batch"),
            pytest.param({"means": [0.485, 0.456]}, "means and positive stds must match", id="means-length"),
            pytest.param({"means": [float("nan"), 0.456, 0.406]}, "means and positive stds must match", id="nan-mean"),
            pytest.param({"stds": [0.229, 0.0, 0.225]}, "means and positive stds must match", id="zero-std"),
            pytest.param({"task": "segment"}, "outputs missing semantic values.*pred_masks", id="segment-no-masks"),
            pytest.param(
                {"task": "keypoints", "num_keypoints_per_class": [17]},
                "outputs missing semantic values.*pred_keypoints",
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
            pytest.param({"num_classes": 91}, "class_id_to_name is required", id="sparse-labels-without-map"),
            pytest.param({"schema_version": 2}, "schema_version", id="schema-version"),
            pytest.param({"unknown_field": 1}, "Extra inputs are not permitted", id="extra-field"),
            pytest.param({"num_classes": "1"}, "Input should be a valid integer", id="strict-integer"),
        ],
    )
    def test_rejects_contract_violations(self, update: dict[str, object], match: str) -> None:
        """Metadata that cannot describe one safe prediction contract fails at construction.

        Each case changes one field of an otherwise valid detection payload, so the message pins the rejecting rule: an
        explicit legacy config cannot change native scaling, the schema, the output mapping, or the label layout.
        """
        payload: dict[str, object] = {
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

        with pytest.raises(ValueError, match=match):
            ExportMetadata.model_validate({**payload, **update})
