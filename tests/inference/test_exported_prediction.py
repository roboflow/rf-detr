# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public exported-model loading and prediction contracts."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr import RFDETRInference
from rfdetr.detr import RFDETR
from tests._markers import onnx_and_onnxruntime_only


class TestInferenceLoading:
    """Export loading refuses invalid artifacts before runtime execution."""

    def test_missing_artifact(self, tmp_path: Path) -> None:
        """A missing artifact reports its path rather than constructing a native network."""
        with pytest.raises(FileNotFoundError, match="missing.onnx"):
            RFDETRInference(tmp_path / "missing.onnx")


# The graph helper doctest writes a real ONNX file, so it needs the package the class-level skipif cannot gate.
__doctest_requires__ = {("_channel_readout_graph",): ["onnx"]}

#: Native ImageNet normalization the fixture metadata declares and the tests apply by hand.
_MEANS = (0.485, 0.456, 0.406)
_STDS = (0.229, 0.224, 0.225)
#: How far query 0's box moves per unit of a normalized channel mean.
_READOUT_SCALE = 0.1
#: Query 0's class-0 logit before the green channel mean is added, high enough to stay above the 0.5 threshold.
_LOGIT_OFFSET = 4.0


def _channel_readout_graph(path: Path, channels: int = 3) -> None:
    """Save a two-query ONNX detector whose first query reads the normalized input channel means.

    Query 0's box centre x, centre y and width move with the red, green and blue channel means, and its class-0 logit
    moves with the green mean, so a wrong mean, std or channel order changes the prediction. With one channel only
    query 0's centre x reads the grayscale input. Query 1 is a constant
    low-score box. The batch axis is symbolic, so the same graph serves fixed- and dynamic-batch metadata.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "detector.onnx"
        >>> _channel_readout_graph(path)
        >>> path.stat().st_size > 0
        True
        >>> _channel_readout_graph(path, channels=1)
        >>> path.stat().st_size > 0
        True
    """
    import onnx
    from onnx import TensorProto, helper

    # Rows are the red, green and blue means; columns are query 0 then query 1, each as (cx, cy, w, h).
    box_weights = [
        [_READOUT_SCALE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, _READOUT_SCALE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, _READOUT_SCALE, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    box_bias = [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.2, 0.2]
    # Columns are (query 0, class 0), (query 0, class 1), (query 1, class 0), (query 1, class 1).
    logit_weights = [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    logit_bias = [_LOGIT_OFFSET, -4.0, -4.0, -4.0]
    box_weights = box_weights[:channels]
    logit_weights = logit_weights[:channels]
    initializers = [
        helper.make_tensor(
            "box_weights", TensorProto.FLOAT, [channels, 8], [value for row in box_weights for value in row]
        ),
        helper.make_tensor("box_bias", TensorProto.FLOAT, [8], box_bias),
        helper.make_tensor("box_shape", TensorProto.INT64, [3], [-1, 2, 4]),
        helper.make_tensor(
            "logit_weights", TensorProto.FLOAT, [channels, 4], [value for row in logit_weights for value in row]
        ),
        helper.make_tensor("logit_bias", TensorProto.FLOAT, [4], logit_bias),
        helper.make_tensor("logit_shape", TensorProto.INT64, [3], [-1, 2, 2]),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["channel_means"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["channel_means", "box_weights"], ["box_flat"]),
        helper.make_node("Add", ["box_flat", "box_bias"], ["box_values"]),
        helper.make_node("Reshape", ["box_values", "box_shape"], ["boxes"]),
        helper.make_node("MatMul", ["channel_means", "logit_weights"], ["logit_flat"]),
        helper.make_node("Add", ["logit_flat", "logit_bias"], ["logit_values"]),
        helper.make_node("Reshape", ["logit_values", "logit_shape"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", channels, 32, 48])],
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["batch", 2, 4]),
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2, 2]),
        ],
        initializer=initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)


def _expected_detection(rgb: tuple[int, int, int], height: int, width: int) -> tuple[list[float], float]:
    """Compute the box and score the channel-readout graph gives a uniform image under native preprocessing.

    Each channel is normalized by hand as ``(pixel / 255 - mean) / std`` in RGB order; a uniform image keeps that value
    through any bilinear resize, so the expected prediction depends only on normalization and channel order.

    Examples:
        >>> box, score = _expected_detection((0, 0, 0), height=64, width=96)
        >>> [round(value, 1) for value in box]
        [12.3, 3.0, 43.0, 35.0]
        >>> round(score, 3)
        0.877
    """
    red, green, blue = ((value / 255 - mean) / std for value, mean, std in zip(rgb, _MEANS, _STDS))
    center_x, center_y = 0.5 + _READOUT_SCALE * red, 0.5 + _READOUT_SCALE * green
    box_width, box_height = 0.5 + _READOUT_SCALE * blue, 0.5
    box = [
        (center_x - box_width / 2) * width,
        (center_y - box_height / 2) * height,
        (center_x + box_width / 2) * width,
        (center_y + box_height / 2) * height,
    ]
    return box, 1 / (1 + math.exp(-(_LOGIT_OFFSET + green)))


@pytest.fixture
def exported_detection(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Return a tiny real ONNX graph that reads its input, and its legacy inference metadata.

    Examples:
        >>> exported_detection()  # doctest: +SKIP
        # Pytest supplies the temporary directory.
    """
    path = tmp_path / "detector.onnx"
    _channel_readout_graph(path)
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
        "means": list(_MEANS),
        "stds": list(_STDS),
    }


@onnx_and_onnxruntime_only
class TestInferenceCapabilities:
    """The public constructor returns an inference-only type."""

    def test_distinct_type(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """The public constructor loads an export into an inference-only instance."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        assert type(model) is RFDETRInference
        assert not isinstance(model, RFDETR)


@onnx_and_onnxruntime_only
class TestExportedPrediction:
    """A real runtime returns the existing Supervision prediction contract."""

    def test_detection(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """Native preprocessing reaches the graph, and boxes scale back to a non-square source.

        The graph reads each normalized channel mean, so a wrong mean, std or channel order on this channel-distinct
        image moves the box or the score; the 64x96 source checks scaling to its own height and width.
        """
        import numpy as np
        import supervision as sv

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.full((64, 96, 3), (48, 96, 144), dtype=np.uint8)
        expected_box, expected_score = _expected_detection((48, 96, 144), height=64, width=96)

        result = model.predict(image)

        assert isinstance(result, sv.Detections)
        np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)
        np.testing.assert_allclose(result.confidence, [expected_score], rtol=0, atol=1e-5)
        assert list(result.data["class_name"]) == ["object"]
        np.testing.assert_array_equal(result.metadata["source_image"], image)

    def test_dynamic_batch_keeps_images_apart(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """A dynamic-batch artifact predicts two different images in one call, each from its own pixels.

        Two uniform images of different colours and sizes run as one batch of two; a batch-axis mix-up or a shared
        source size would give both results the same box or score.
        """
        import numpy as np

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=dict(metadata, input_shape=[-1, 3, 32, 48]), device="cpu")
        images = [np.full((64, 96, 3), (48, 96, 144), np.uint8), np.full((40, 80, 3), (200, 30, 90), np.uint8)]
        first_box, first_score = _expected_detection((48, 96, 144), height=64, width=96)
        second_box, second_score = _expected_detection((200, 30, 90), height=40, width=80)

        first, second = model.predict(images, include_source_image=False)

        np.testing.assert_allclose([first.xyxy[0], second.xyxy[0]], [first_box, second_box], rtol=0, atol=1e-3)
        np.testing.assert_allclose(
            [first.confidence[0], second.confidence[0]], [first_score, second_score], rtol=0, atol=1e-5
        )

    @pytest.mark.parametrize("keyword", ["threhsold", "confidence"])
    def test_unknown_keyword_cannot_change_threshold_intent(
        self, exported_detection: tuple[Path, dict[str, object]], keyword: str
    ) -> None:
        """Unknown options fail through the same real runtime that honors the declared threshold."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        assert len(model.predict(image, threshold=1.0)) == 0
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            model.predict(image, **{keyword: 1.0})

    @pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
    @pytest.mark.parametrize("input_kind", ["pil", "str", "path", "array", "tensor"])
    def test_single_channel_input_forms_match(
        self,
        exported_detection: tuple[Path, dict[str, object]],
        tmp_path: Path,
        image_format: str,
        input_kind: str,
    ) -> None:
        """One real graph consumes the same decoded grayscale pixels across image forms."""
        path, metadata = exported_detection
        _channel_readout_graph(path, channels=1)
        metadata = dict(metadata, input_shape=[1, 1, 32, 48], means=[_MEANS[0]], stds=[_STDS[0]])
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        pixels = (np.arange(40 * 64).reshape(40, 64) % 256).astype(np.uint8)
        image_path = tmp_path / f"gray.{image_format.lower()}"
        Image.fromarray(pixels).save(image_path, format=image_format)
        with Image.open(image_path) as decoded:
            pil = decoded.convert("L")
        expected_source = np.array(pil)
        array = expected_source[:, :, None].copy()
        expected = model.predict(array)
        supplied = {
            "pil": pil,
            "str": str(image_path),
            "path": image_path,
            "array": array,
            "tensor": torch.from_numpy(array).permute(2, 0, 1).float() / 255,
        }[input_kind]
        actual = model.predict(supplied)

        np.testing.assert_allclose(actual.xyxy, expected.xyxy, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(actual.confidence, expected.confidence, rtol=1e-4, atol=1e-6)
        source = actual.metadata["source_image"]
        assert source.dtype == np.uint8
        assert source.shape == ((40, 64) if input_kind in {"pil", "str", "path"} else (40, 64, 1))
        np.testing.assert_array_equal(source.squeeze(), expected_source)
        assert source.flags.owndata
        array.fill(0)
        np.testing.assert_array_equal(source.squeeze(), expected_source)

    def test_rgb_pil_converts_to_the_single_channel_contract(
        self, exported_detection: tuple[Path, dict[str, object]]
    ) -> None:
        """A one-channel artifact selects Pillow luminance conversion for RGB images."""
        path, metadata = exported_detection
        _channel_readout_graph(path, channels=1)
        metadata = dict(metadata, input_shape=[1, 1, 32, 48], means=[_MEANS[0]], stds=[_STDS[0]])
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = Image.new("RGB", (64, 40), color=(17, 127, 231))
        gray = np.array(image.convert("L"))

        actual = model.predict(image)
        expected = model.predict(gray)

        np.testing.assert_array_equal(actual.xyxy, expected.xyxy)
        np.testing.assert_array_equal(actual.metadata["source_image"], gray)

    def test_pathlike_tuple_keeps_the_exported_batch_contract(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path
    ) -> None:
        """PathLike tuples preserve ordering and the one-call dynamic batch contract."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=dict(metadata, input_shape=[-1, 3, 32, 48]), device="cpu")
        first = tmp_path / "first.png"
        second = tmp_path / "second.png"
        Image.new("RGB", (64, 40), (17, 127, 231)).save(first)
        Image.new("RGB", (48, 32), (231, 17, 127)).save(second)
        actual = model.predict((first, second))

        assert isinstance(actual, list)
        assert len(actual) == 2
        assert not np.array_equal(actual[0].xyxy, actual[1].xyxy)
        np.testing.assert_array_equal(actual[0].metadata["source_image"][0, 0], [17, 127, 231])
        np.testing.assert_array_equal(actual[1].metadata["source_image"][0, 0], [231, 17, 127])

    @pytest.mark.parametrize("batch", [1, -1])
    def test_runtime_information_exposes_exported_constraints(
        self, exported_detection: tuple[Path, dict[str, object]], batch: int
    ) -> None:
        """Inspection exposes fixed/dynamic batch and spatial constraints without leaking mutable state."""
        path, metadata = exported_detection
        metadata = dict(metadata, input_shape=[batch, 3, 32, 48])
        if batch == -1:
            metadata["max_batch_size"] = 4
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        required_batch = None if batch == -1 else batch
        assert model.runtime_info == {
            "backend": "onnx",
            "device": "CPUExecutionProvider",
            "task": "detect",
            "input_shape": (required_batch, 3, 32, 48),
            "default_shape": (32, 48),
            "batch_size": required_batch,
            "max_batch_size": 4 if batch == -1 else None,
        }
        info = model.runtime_info
        info["task"] = "segment"
        assert model.runtime_info["task"] == "detect"

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

    def test_dynamic_metadata_rejects_fixed_graph(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """Metadata cannot widen a fixed graph to accept batches its runtime rejects."""
        import onnx

        path, metadata = exported_detection
        graph = onnx.load(path)
        batch = graph.graph.input[0].type.tensor_type.shape.dim[0]
        batch.ClearField("dim_param")
        batch.dim_value = 1
        onnx.save(graph, path)
        metadata = dict(metadata, input_shape=[-1, 3, 32, 48])

        with pytest.raises(ValueError, match="ONNX input shape.*disagrees"):
            RFDETRInference(path, metadata=metadata, device="cpu")

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
        assert model.runtime_info["task"] == task
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
