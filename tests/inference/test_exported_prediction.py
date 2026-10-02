# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public exported-model loading and prediction contracts."""

from pathlib import Path

import pytest
from supervision import Detections

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


class TestExportedStreaming:
    """Exported runtimes use the same source expansion as native models."""

    @pytest.mark.parametrize("input_kind", ["directory", "glob", "bchw"])
    @pytest.mark.parametrize("stream", [False, True])
    def test_expanded_sources(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path, input_kind: str, stream: bool
    ) -> None:
        """Expanded inputs preserve image order and boxes in eager and lazy modes."""
        import numpy as np
        import torch
        from PIL import Image

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        images = np.stack([np.full((64, 96, 3), value, dtype=np.uint8) for value in [40, 80, 120]])
        source_dir = tmp_path / "images"
        source_dir.mkdir()
        for index in [2, 0, 1]:
            Image.fromarray(images[index]).save(source_dir / f"{index}.png")
        if input_kind == "directory":
            source = source_dir
        elif input_kind == "glob":
            source = str(source_dir / "*.png")
        else:
            source = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255)

        results = list(model.predict(source, stream=stream))

        assert len(results) == 3
        for result, image in zip(results, images):
            assert isinstance(result, Detections)
            np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]])
            np.testing.assert_array_equal(result.metadata["source_image"], image)

    @pytest.mark.parametrize("stream", [False, True])
    def test_video_stride(
        self, exported_detection: tuple[Path, dict[str, object]], prediction_video: Path, stream: bool
    ) -> None:
        """Video stride selects every second frame and stops at finite EOF."""
        import numpy as np

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")

        results = list(model.predict(prediction_video, stream=stream, vid_stride=2))

        assert len(results) == 3
        for result, value in zip(results, [40, 80, 120]):
            assert isinstance(result, Detections)
            np.testing.assert_allclose(result.metadata["source_image"], value, atol=2)
            np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]])

    @pytest.mark.parametrize("ending", ["exhaustion", "close", "error"])
    def test_video_capture_cleanup(
        self,
        exported_detection: tuple[Path, dict[str, object]],
        prediction_video: Path,
        monkeypatch: pytest.MonkeyPatch,
        ending: str,
    ) -> None:
        """Real video captures close on completion, early close, and a rejected batch."""
        from unittest.mock import Mock

        import cv2

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        capture = cv2.VideoCapture(str(prediction_video))
        assert capture.isOpened()
        capture_factory = Mock(return_value=capture)
        monkeypatch.setattr(cv2, "VideoCapture", capture_factory)
        results = model.predict(prediction_video, stream=True, batch=2 if ending == "error" else 1)
        capture_factory.assert_not_called()
        if ending == "error":
            with pytest.raises(ValueError, match="Batch size mismatch"):
                next(results)
        elif ending == "close":
            next(results)
            results.close()
        else:
            assert len(list(results)) == 6

        assert not capture.isOpened()

    def test_partial_batch_rejected(self, exported_batch_two: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
        """A fixed batch-two export yields its full batch then rejects the final image."""
        import numpy as np
        from PIL import Image

        path, metadata = exported_batch_two
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        source = tmp_path / "images"
        source.mkdir()
        for index in range(3):
            Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8)).save(source / f"{index}.png")
        results = model.predict(source, stream=True, batch=2)

        for _ in range(2):
            result = next(results)
            assert isinstance(result, Detections)
            np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]])
        with pytest.raises(ValueError, match="Batch size mismatch"):
            next(results)

    def test_dynamic_batch_flushes_final_image(
        self, exported_dynamic_batch: tuple[Path, dict[str, object]], tmp_path: Path
    ) -> None:
        """A dynamic export executes batches of two, two, then one without losing images."""
        from unittest.mock import patch

        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        path, metadata = exported_dynamic_batch
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        source = tmp_path / "images"
        source.mkdir()
        for index, value in enumerate([20, 40, 60, 80, 100]):
            Image.fromarray(np.full((64, 96, 3), value, dtype=np.uint8)).save(source / f"{index}.png")

        with patch.object(ort.InferenceSession, "run", autospec=True, side_effect=ort.InferenceSession.run) as execute:
            results = list(model.predict(source, stream=True, batch=2))

        assert [call.args[2]["images"].shape[0] for call in execute.call_args_list] == [2, 2, 1]
        assert len(results) == 5
        for result, value in zip(results, [20, 40, 60, 80, 100]):
            assert isinstance(result, Detections)
            np.testing.assert_array_equal(result.metadata["source_image"], value)
            np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]])


@pytest.fixture
def prediction_video(tmp_path: Path) -> Path:
    """Create six grayscale frames with known pixel values.

    Examples:
        >>> prediction_video()  # doctest: +SKIP
        # Pytest supplies the temporary directory and OpenCV codec support.
    """
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer unavailable")
    try:
        for value in [20, 40, 60, 80, 100, 120]:
            writer.write(np.full((64, 96, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.fixture
def exported_batch_two(exported_detection: tuple[Path, dict[str, object]]) -> tuple[Path, dict[str, object]]:
    """Resize the real constant ONNX detector to a fixed batch of two.

    Examples:
        >>> exported_batch_two()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact fixture.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    path, metadata = exported_detection
    model = onnx.load(path)
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = 2
    for node in model.graph.node:
        tensor = node.attribute[0].t
        tensor.CopyFrom(numpy_helper.from_array(np.repeat(numpy_helper.to_array(tensor), 2, axis=0)))
    onnx.save(model, path)
    return path, dict(metadata, input_shape=[2, 3, 32, 48])


@pytest.fixture
def exported_dynamic_batch(exported_detection: tuple[Path, dict[str, object]]) -> tuple[Path, dict[str, object]]:
    """Expand the constant detector outputs to the input's actual batch size.

    Examples:
        >>> exported_dynamic_batch()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact fixture.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    path, metadata = exported_detection
    model = onnx.load(path)
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    nodes = [helper.make_node("Shape", ["images"], ["batch_shape"], start=0, end=1)]
    for node, tail in zip(model.graph.node, [[2, 4], [2, 2]]):
        output = node.output[0]
        node.output[0] = f"{output}_single"
        model.graph.initializer.append(numpy_helper.from_array(np.array(tail, dtype=np.int64), f"{output}_tail"))
        nodes.extend(
            [
                helper.make_node("Concat", ["batch_shape", f"{output}_tail"], [f"{output}_shape"], axis=0),
                helper.make_node("Expand", [f"{output}_single", f"{output}_shape"], [output]),
            ]
        )
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path, dict(metadata, input_shape=[-1, 3, 32, 48])
