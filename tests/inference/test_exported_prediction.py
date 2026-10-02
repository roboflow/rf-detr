# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public exported-model loading and prediction contracts."""

from pathlib import Path

import pytest

from rfdetr import RFDETRInference
from rfdetr.detr import RFDETR


class TestInferenceLoading:
    """Export loading refuses invalid artifacts before runtime execution."""

    def test_missing_artifact(self, tmp_path: Path) -> None:
        """A missing artifact reports its path rather than constructing a native network."""
        with pytest.raises(FileNotFoundError, match="missing.onnx"):
            RFDETRInference(tmp_path / "missing.onnx")


@pytest.fixture
def exported_detection(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Return a tiny real ONNX graph and its legacy inference metadata.

    Examples:
        >>> exported_detection()  # doctest: +SKIP
        # Pytest supplies the temporary directory.
    """
    import numpy as np

    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path / "detector.onnx"
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.2, 0.2]]], dtype=np.float32)
    logits = np.array([[[4.0, -4.0], [-4.0, -4.0]]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["boxes"], value=numpy_helper.from_array(boxes)),
            helper.make_node("Constant", [], ["logits"], value=numpy_helper.from_array(logits)),
        ],
        "detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 48])],
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 2]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.save(model, path)
    return path, {
        "format": "onnx",
        "task": "detect",
        "variant": "rfdetr-nano",
        "input_shape": [1, 3, 32, 48],
        "input_name": "images",
        "outputs": {"pred_boxes": "boxes", "pred_logits": "logits"},
        "class_names": ["object"],
        "class_id_to_name": {"0": "object"},
        "num_classes": 1,
        "num_select": 2,
        "patch_size": 16,
        "num_windows": 1,
        "trace_alpha": 0.2,
        "means": [0.485, 0.456, 0.406],
        "stds": [0.229, 0.224, 0.225],
    }


class TestInferenceCapabilities:
    """The public constructor returns an inference-only type."""

    def test_distinct_type(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """The public constructor loads an export into an inference-only instance."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        assert type(model) is RFDETRInference
        assert not isinstance(model, RFDETR)


class TestExportedPrediction:
    """A real runtime returns the existing Supervision prediction contract."""

    def test_detection(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """The exported model scales boxes and retains class names and source metadata."""
        import numpy as np
        import supervision as sv

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        result = model.predict(image)
        assert isinstance(result, sv.Detections)
        np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]])
        assert list(result.data["class_name"]) == ["object"]
        np.testing.assert_array_equal(result.metadata["source_image"], image)

    @pytest.mark.parametrize("input_kind", ["pil", "path", "tensor"])
    def test_input_forms_preserve_prediction_and_source(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path, input_kind: str
    ) -> None:
        """PIL, file paths, and CHW tensors match the array prediction and source image."""
        import numpy as np
        import supervision as sv
        import torch
        from PIL import Image

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.full((64, 96, 3), (48, 96, 144), dtype=np.uint8)
        baseline = model.predict(image)
        assert isinstance(baseline, sv.Detections)
        pil_image = Image.fromarray(image)
        if input_kind == "pil":
            supplied = pil_image
        elif input_kind == "path":
            source_path = tmp_path / "source.png"
            pil_image.save(source_path)
            supplied = str(source_path)
        else:
            supplied = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)

        result = model.predict(supplied)

        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.xyxy, baseline.xyxy)
        np.testing.assert_array_equal(result.class_id, baseline.class_id)
        np.testing.assert_array_equal(result.confidence, baseline.confidence)
        np.testing.assert_array_equal(result.metadata["source_image"], image)
        np.testing.assert_array_equal(result.data["source_shape"], baseline.data["source_shape"])

    @pytest.mark.parametrize(
        ("device", "error", "message"),
        [("mps", ValueError, "unsupported"), ("cuda:0", RuntimeError, "CUDAExecutionProvider is unavailable")],
    )
    def test_explicit_device_never_falls_back(
        self,
        exported_detection: tuple[Path, dict[str, object]],
        monkeypatch: pytest.MonkeyPatch,
        device: str,
        error: type[Exception],
        message: str,
    ) -> None:
        """An unsupported or unavailable provider fails before session construction."""
        import onnxruntime as ort

        path, metadata = exported_detection
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        with pytest.raises(error, match=message):
            RFDETRInference(path, metadata=metadata, device=device)

    @pytest.mark.parametrize("as_list", [False, True])
    def test_result_container(self, exported_detection: tuple[Path, dict[str, object]], as_list: bool) -> None:
        """A one-item list stays a list, while a single image stays a single result."""
        import numpy as np
        import supervision as sv

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        result = model.predict([image] if as_list else image, include_source_image=False)
        if as_list:
            assert isinstance(result, list)
            prediction = result[0]
        else:
            prediction = result
        assert isinstance(prediction, sv.Detections)
        assert "source_image" not in prediction.metadata

    def test_batch_mismatch(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """Fixed-batch artifacts reject lists that cannot execute as one batch."""
        import numpy as np

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model.predict([image, image])

    def test_shape_mismatch(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """A prediction shape cannot override the artifact's fixed shape."""
        import numpy as np

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        with pytest.raises(ValueError, match="requires shape"):
            model.predict(np.zeros((32, 48, 3), dtype=np.uint8), shape=(32, 32))

    def test_invalid_shape(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """Invalid shape types retain the native ValueError contract."""
        import numpy as np

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        with pytest.raises(ValueError, match="shape must be a sequence"):
            model.predict(np.zeros((32, 48, 3), dtype=np.uint8), shape=42)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "operation",
        [
            "train",
            "evaluate",
            "inference",
            "remove_optimized_model",
            "export",
            "deploy_to_roboflow",
            "export_for_roboflow",
        ],
    )
    def test_native_operations_absent(self, exported_detection: tuple[Path, dict[str, object]], operation: str) -> None:
        """The inference type does not advertise native-model capabilities."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        assert not hasattr(model, operation)

    def test_embedded_metadata(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """A self-describing ONNX artifact needs no caller configuration."""
        import json

        import numpy as np
        import onnx
        import supervision as sv

        from rfdetr import RFDETRInference

        path, metadata = exported_detection
        graph = onnx.load(path)
        entry = graph.metadata_props.add()
        entry.key = "rfdetr_inference"
        entry.value = json.dumps(metadata)
        onnx.save(graph, path)
        model = RFDETRInference(path, device="cpu")
        result = model.predict(np.zeros((64, 96, 3), dtype=np.uint8))
        assert isinstance(result, sv.Detections)
        assert result.class_id is not None
        assert result.class_id.tolist() == [0]
        assert model.class_names == ["object"]
        assert model.runtime_info["backend"] == "onnx"

    @pytest.mark.parametrize("task", ["segment", "keypoints"])
    def test_task_outputs(self, exported_detection: tuple[Path, dict[str, object]], task: str) -> None:
        """Masks and keypoints use native postprocessing and Supervision output types."""
        import numpy as np
        import onnx
        import supervision as sv
        from onnx import TensorProto, helper, numpy_helper

        path, metadata = exported_detection
        metadata = dict(metadata, task=task)
        graph = onnx.load(path)
        if task == "segment":
            values = np.ones((1, 2, 4, 6), dtype=np.float32)
            semantic = "pred_masks"
        else:
            values = np.array([[[[0.5, 0.25, 2.0, 0.0, 0.0, 0.0, 0.0]]] * 2], dtype=np.float32)
            semantic = "pred_keypoints"
            metadata["num_keypoints_per_class"] = [1]
            metadata["trace_alpha"] = 0.0
        output_map = metadata["outputs"]
        assert isinstance(output_map, dict)
        metadata["outputs"] = {**output_map, semantic: "task_output"}
        graph.graph.node.append(
            helper.make_node("Constant", [], ["task_output"], value=numpy_helper.from_array(values))
        )
        graph.graph.output.append(helper.make_tensor_value_info("task_output", TensorProto.FLOAT, list(values.shape)))
        onnx.save(graph, path)
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        result = model.predict(np.zeros((64, 96, 3), dtype=np.uint8))
        if task == "segment":
            assert isinstance(result, sv.Detections)
            assert result.mask is not None
            assert np.asarray(result.mask).shape == (1, 64, 96)
            assert np.asarray(result.mask).all()
        else:
            assert isinstance(result, sv.KeyPoints)
            np.testing.assert_allclose(result.xy, [[[48, 16]]])
            assert result.keypoint_confidence is not None
            np.testing.assert_allclose(result.keypoint_confidence, [[0.880797]], rtol=1e-5)
            assert np.asarray(result.data["covariance"]).shape == (1, 1, 2, 2)
