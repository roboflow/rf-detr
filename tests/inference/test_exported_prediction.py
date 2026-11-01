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
from supervision import Detections, KeyPoints

from rfdetr import ClassThresholds, RFDETRInference
from rfdetr.detr import RFDETR
from rfdetr.export.imports import _IS_ONNXRUNTIME_INSTALLED
from tests._markers import _requires_onnx_and_onnxruntime

from .helpers import _requires_opencv


def _onnxruntime_has_cuda() -> bool:
    """Report whether ONNX Runtime offers its CUDA execution provider on a host where torch sees a GPU.

    Examples:
        >>> isinstance(_onnxruntime_has_cuda(), bool)
        True
    """
    if not (_IS_ONNXRUNTIME_INSTALLED and torch.cuda.is_available()):
        return False
    import onnxruntime

    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


# GPU CI installs the CPU onnxruntime, so this runs only where onnxruntime-gpu is installed.
_requires_onnxruntime_cuda = pytest.mark.skipif(
    not _onnxruntime_has_cuda(), reason="requires onnxruntime-gpu with the CUDA execution provider and a CUDA device"
)


class TestInferenceLoading:
    """Export loading refuses invalid artifacts before runtime execution."""

    def test_missing_artifact(self, tmp_path: Path) -> None:
        """A missing artifact reports its path rather than constructing a native network."""
        with pytest.raises(FileNotFoundError) as exc_info:
            RFDETRInference(tmp_path / "missing.onnx")

        assert "missing.onnx" in str(exc_info.value)


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
    query 0's centre x reads the grayscale input. Query 1 is a constant low-score box. The batch axis is symbolic, so
    the same graph serves fixed- and dynamic-batch metadata.

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


def _expected_detection(rgb: tuple[float, float, float], height: int, width: int) -> tuple[list[float], float]:
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


@pytest.fixture
def exported_sparse_classes(exported_detection: tuple[Path, dict[str, object]]) -> tuple[Path, dict[str, object]]:
    """Return the detector regraphed to class ids 0 and 2 (named ``a`` and ``b``) plus the no-object slot.

    Class id 1 is unused, as in a pretrained COCO checkpoint with gaps in its category ids. Query 0 scores class id 2
    at 0.982 and query 1 scores class id 0 at 0.269; every other slot stays at 0.018.

    Examples:
        >>> exported_sparse_classes()  # doctest: +SKIP
        # Pytest supplies the detection fixture it edits.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path, metadata = exported_detection
    graph = onnx.load(path)
    initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    # Columns are (query 0, ids 0..3) then (query 1, ids 0..3); id 3 is the no-object slot.
    logit_bias = np.array([-4, -4, 4, -4, -1, -4, -4, -4], dtype=np.float32)
    initializers["logit_weights"].CopyFrom(numpy_helper.from_array(np.zeros((3, 8), np.float32), "logit_weights"))
    initializers["logit_bias"].CopyFrom(numpy_helper.from_array(logit_bias, "logit_bias"))
    initializers["logit_shape"].CopyFrom(numpy_helper.from_array(np.array([-1, 2, 4], np.int64), "logit_shape"))
    logits = next(output for output in graph.graph.output if output.name == "logits")
    logits.CopyFrom(helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2, 4]))
    onnx.save(graph, path)
    return path, dict(metadata, class_names=["a", "b"], class_id_to_name={"0": "a", "2": "b"}, num_classes=3)


@pytest.fixture
def exported_segmentation(exported_detection: tuple[Path, dict[str, object]]) -> tuple[Path, dict[str, object]]:
    """Return the detector with an all-ones ``pred_masks`` output added, and its segmentation metadata.

    Examples:
        >>> exported_segmentation()  # doctest: +SKIP
        # Pytest supplies the detection fixture it edits.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path, metadata = exported_detection
    outputs = metadata["outputs"]
    assert isinstance(outputs, dict)
    masks = np.ones((1, 2, 4, 6), dtype=np.float32)
    graph = onnx.load(path)
    graph.graph.node.append(helper.make_node("Constant", [], ["task_output"], value=numpy_helper.from_array(masks)))
    graph.graph.output.append(helper.make_tensor_value_info("task_output", TensorProto.FLOAT, list(masks.shape)))
    onnx.save(graph, path)
    return path, dict(metadata, task="segment", outputs={**outputs, "pred_masks": "task_output"})


@_requires_onnx_and_onnxruntime
class TestInferenceCapabilities:
    """The public constructor returns an inference-only type."""

    def test_load_returns_inference_only_instance(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """The public constructor loads an export into an inference-only instance."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        assert type(model) is RFDETRInference
        assert not isinstance(model, RFDETR)


@_requires_onnx_and_onnxruntime
class TestExportedPrediction:
    """A real runtime returns the existing Supervision prediction contract."""

    def test_detection_scales_boxes_to_source_size(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """Native preprocessing reaches the graph, and boxes scale back to a non-square source.

        The graph reads each normalized channel mean, so a wrong mean, std or channel order on this channel-distinct
        image moves the box or the score; the 64x96 source checks scaling to its own height and width.
        """
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.full((64, 96, 3), (48, 96, 144), dtype=np.uint8)
        expected_box, expected_score = _expected_detection((48, 96, 144), height=64, width=96)

        result = model.predict(image)

        assert isinstance(result, Detections)
        np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)
        np.testing.assert_allclose(result.confidence, [expected_score], rtol=0, atol=1e-5)
        assert list(result.data["class_name"]) == ["object"]
        np.testing.assert_array_equal(result.metadata["source_image"], image)

    def test_dynamic_batch_keeps_images_apart(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """A dynamic-batch artifact predicts two different images in one call, each from its own pixels.

        Two uniform images of different colours and sizes run as one batch of two; a batch-axis mix-up or a shared
        source size would give both results the same box or score.
        """
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

    @pytest.mark.gpu
    @_requires_onnxruntime_cuda
    def test_cuda_provider_predicts_a_directory_as_the_cpu_provider_does(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path
    ) -> None:
        """The CUDA provider prepares and binds each batch on the GPU, and predicts what the CPU provider does.

        A directory runs in batches that the next batch's preparation overlaps, so a batch handed to ONNX Runtime before
        its GPU preprocessing finished would give a wrong box or score.
        """
        path, metadata = exported_detection
        metadata = dict(metadata, input_shape=[-1, 3, 32, 48])
        source = tmp_path / "images"
        source.mkdir()
        for index in range(10):
            Image.fromarray(np.full((64, 96, 3), (20 * index, 90, 200 - 15 * index), np.uint8)).save(
                source / f"{index}.png"
            )
        expected = RFDETRInference(path, metadata=metadata, device="cpu").predict(source)
        model = RFDETRInference(path, metadata=metadata, device="cuda")

        actual = model.predict(source)

        assert model._prediction_context().device.type == "cuda"
        np.testing.assert_allclose([r.xyxy for r in actual], [r.xyxy for r in expected], rtol=0, atol=1e-3)
        np.testing.assert_allclose([r.confidence for r in actual], [r.confidence for r in expected], rtol=0, atol=1e-5)

    @pytest.mark.parametrize(("threshold", "kept"), [([0.5], 1), ([1.0], 0)])
    def test_per_class_threshold_filters_the_exported_detections(
        self, exported_detection: tuple[Path, dict[str, object]], threshold: list[float], kept: int
    ) -> None:
        """A one-class threshold sequence reaches the same keep-filter as a native model's, through a real runtime."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")

        result = model.predict(np.zeros((32, 48, 3), dtype=np.uint8), threshold=ClassThresholds(threshold))

        assert len(result) == kept

    def test_per_class_threshold_of_the_wrong_length_raises(
        self, exported_detection: tuple[Path, dict[str, object]]
    ) -> None:
        """An exported model with one class refuses two thresholds before it runs."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")

        with pytest.raises(ValueError, match="2 entries but the model has 1 classes"):
            model.predict(np.zeros((32, 48, 3), dtype=np.uint8), threshold=ClassThresholds([0.5, 0.5]))

    @pytest.mark.parametrize(
        ("threshold", "expected_class_id", "expected_class_name"),
        [
            # Name "b" is class id 2 (score 0.982); the list reads its entry by position, so 0.1 keeps it.
            pytest.param([0.99, 0.1], 2, "b", id="second-name-owns-id-2"),
            # Name "a" is class id 0 (score 0.269); its entry is first, so 0.1 keeps it and 0.99 drops id 2.
            pytest.param([0.1, 0.99], 0, "a", id="first-name-owns-id-0"),
        ],
    )
    def test_per_class_threshold_follows_sparse_class_ids(
        self,
        exported_sparse_classes: tuple[Path, dict[str, object]],
        threshold: list[float],
        expected_class_id: int,
        expected_class_name: str,
    ) -> None:
        """With class ids 0 and 2 the list is still one entry per class name, in id order, not indexed by raw id.

        Reading the list by raw id would pair id 2 with a missing third entry, and pairing the entries with ids 0 and 1
        would hand id 2 no threshold, which drops its detection.
        """
        path, metadata = exported_sparse_classes
        model = RFDETRInference(path, metadata=metadata, device="cpu")

        result = model.predict(np.zeros((32, 48, 3), dtype=np.uint8), threshold=ClassThresholds(threshold))

        assert result.class_id.tolist() == [expected_class_id]
        assert list(result.data["class_name"]) == [expected_class_name]

    @pytest.mark.parametrize(("threshold", "kept"), [([0.5], 1), ([1.0], 0)])
    def test_per_class_threshold_filters_the_exported_masks(
        self, exported_segmentation: tuple[Path, dict[str, object]], threshold: list[float], kept: int
    ) -> None:
        """A segmentation export keeps one mask per surviving detection when thresholds are given per class."""
        path, metadata = exported_segmentation
        model = RFDETRInference(path, metadata=metadata, device="cpu")

        result = model.predict(np.zeros((64, 96, 3), dtype=np.uint8), threshold=ClassThresholds(threshold))

        assert (len(result), np.asarray(result.mask).shape) == (kept, (kept, 64, 96))

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
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.full((64, 96, 3), (48, 96, 144), dtype=np.uint8)
        baseline = model.predict(image)
        assert isinstance(baseline, Detections)
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

        assert isinstance(result, Detections)
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
    def test_list_input_returns_list_and_single_image_returns_single(
        self, exported_detection: tuple[Path, dict[str, object]], as_list: bool
    ) -> None:
        """A one-item list stays a list, while a single image stays a single result."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        result = model.predict([image] if as_list else image, include_source_image=False)
        if as_list:
            assert isinstance(result, list)
            prediction = result[0]
        else:
            prediction = result
        assert isinstance(prediction, Detections)
        assert "source_image" not in prediction.metadata

    def test_list_larger_than_fixed_batch_raises(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """Fixed-batch artifacts reject lists that cannot execute as one batch."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model.predict([image, image])

    def test_explicit_batch_splits_image_list(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """A fixed batch-one artifact predicts an image list when the caller passes the export's batch size.

        Without the explicit bound the list runs as one batch of two and the artifact rejects it.
        """
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        image = np.zeros((32, 48, 3), dtype=np.uint8)

        results = model.predict([image, image], batch=1)

        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.parametrize(("shape", "message"), [((32, 32), "requires shape"), (42, "shape must be a sequence")])
    def test_unusable_prediction_shape_raises(
        self, exported_detection: tuple[Path, dict[str, object]], shape: object, message: str
    ) -> None:
        """A shape that differs from the fixed artifact shape, or is not a sequence, raises the native ValueError."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        with pytest.raises(ValueError, match=message):
            model.predict(np.zeros((32, 48, 3), dtype=np.uint8), shape=shape)  # type: ignore[arg-type]

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
    def test_native_operations_are_absent(
        self, exported_detection: tuple[Path, dict[str, object]], operation: str
    ) -> None:
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

        with pytest.raises(ValueError) as exc_info:
            RFDETRInference(path, metadata=metadata, device="cpu")

        assert "ONNX input shape" in str(exc_info.value)
        assert "disagrees" in str(exc_info.value)

    def test_embedded_metadata_needs_no_caller_config(self, exported_detection: tuple[Path, dict[str, object]]) -> None:
        """A self-describing ONNX artifact needs no caller configuration."""
        import json

        import onnx

        path, metadata = exported_detection
        graph = onnx.load(path)
        entry = graph.metadata_props.add()
        entry.key = "rfdetr_inference"
        entry.value = json.dumps(metadata)
        onnx.save(graph, path)
        model = RFDETRInference(path, device="cpu")
        result = model.predict(np.zeros((64, 96, 3), dtype=np.uint8))
        assert isinstance(result, Detections)
        assert result.class_id is not None
        assert result.class_id.tolist() == [0]
        assert model.class_names == ["object"]
        assert model.runtime_info["backend"] == "onnx"

    @pytest.mark.parametrize("task", ["segment", "keypoints"])
    def test_segmentation_and_keypoint_outputs_use_native_postprocessing(
        self, exported_detection: tuple[Path, dict[str, object]], task: str
    ) -> None:
        """Masks and keypoints use native postprocessing and Supervision output types."""
        import onnx
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
            assert isinstance(result, Detections)
            assert result.mask is not None
            assert np.asarray(result.mask).shape == (1, 64, 96)
            assert np.asarray(result.mask).all()
        else:
            assert isinstance(result, KeyPoints)
            np.testing.assert_allclose(result.xy, [[[48, 16]]])
            assert result.keypoint_confidence is not None
            np.testing.assert_allclose(result.keypoint_confidence, [[0.880797]], rtol=1e-5)
            assert np.asarray(result.data["covariance"]).shape == (1, 1, 2, 2)


@_requires_onnx_and_onnxruntime
class TestExportedStreaming:
    """Exported runtimes use the same source expansion as native models."""

    @pytest.mark.parametrize("input_kind", ["directory", "glob", "bchw"])
    @pytest.mark.parametrize("stream", [False, True])
    def test_expanded_sources(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path, input_kind: str, stream: bool
    ) -> None:
        """Expanded inputs preserve image order and boxes in eager and lazy modes."""
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

        # The batch-one export needs an explicit batch to split the BCHW tensor; file sources already use one.
        results = list(model.predict(source, stream=stream, batch=1))

        assert len(results) == 3
        for result, image, value in zip(results, images, [40, 80, 120]):
            assert isinstance(result, Detections)
            expected_box, _ = _expected_detection((value, value, value), height=64, width=96)
            np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)
            np.testing.assert_array_equal(result.metadata["source_image"], image)

    @_requires_opencv
    @pytest.mark.parametrize("stream", [False, True])
    def test_bchw_default_batch_matches_fixed_export(
        self, exported_batch_two: tuple[Path, dict[str, object]], stream: bool
    ) -> None:
        """A BCHW tensor runs as one batch by default, so a fixed batch-two export accepts its own B=2 tensor.

        Splitting the tensor into single images would make the export reject every batch with a size mismatch, leaving
        callers to discover the ``batch`` argument from an error.
        """
        path, metadata = exported_batch_two
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        images = np.stack([np.full((64, 96, 3), value, dtype=np.uint8) for value in [40, 120]])
        source = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255)

        results = list(model.predict(source, stream=stream))

        assert len(results) == 2
        for result, image, value in zip(results, images, [40, 120]):
            assert isinstance(result, Detections)
            expected_box, _ = _expected_detection((value, value, value), height=64, width=96)
            np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)
            np.testing.assert_array_equal(result.metadata["source_image"], image)

    @_requires_opencv
    @pytest.mark.parametrize("stream", [False, True])
    def test_video_stride(
        self, exported_detection: tuple[Path, dict[str, object]], prediction_video: Path, stream: bool
    ) -> None:
        """Video stride keeps every second frame from the first and stops at finite EOF."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")

        results = list(model.predict(prediction_video, stream=stream, vid_stride=2))

        assert len(results) == 3
        for result, value in zip(results, [20, 60, 100]):
            assert isinstance(result, Detections)
            source_image = result.metadata["source_image"]
            np.testing.assert_allclose(source_image, value, atol=2)
            # MJPEG may shift decoded colors by a few levels across codec backends.
            red, green, blue = source_image.mean(axis=(0, 1))
            expected_box, _ = _expected_detection((red, green, blue), height=64, width=96)
            np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)

    @_requires_opencv
    @pytest.mark.parametrize("ending", ["exhaustion", "close", "error"])
    def test_video_capture_cleanup(
        self,
        exported_detection: tuple[Path, dict[str, object]],
        prediction_video: Path,
        mocker,
        ending: str,
    ) -> None:
        """Real video captures close on completion, early close, and a rejected batch."""
        import cv2  # optional dependency, guarded by the test's skipif

        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        capture = cv2.VideoCapture(str(prediction_video))
        assert capture.isOpened()
        capture_factory = mocker.patch.object(cv2, "VideoCapture", return_value=capture)
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
        path, metadata = exported_batch_two
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        source = tmp_path / "images"
        source.mkdir()
        for index in range(3):
            Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8)).save(source / f"{index}.png")
        results = model.predict(source, stream=True, batch=2)

        expected_box, _ = _expected_detection((0, 0, 0), height=64, width=96)
        for _ in range(2):
            result = next(results)
            assert isinstance(result, Detections)
            np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            next(results)

    def test_dynamic_batch_flushes_final_image(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path, mocker
    ) -> None:
        """A dynamic export executes batches of two, two, then one without losing images."""
        import onnxruntime as ort

        path, metadata = exported_detection
        metadata = dict(metadata, input_shape=[-1, 3, 32, 48])
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        source = tmp_path / "images"
        source.mkdir()
        for index, value in enumerate([20, 40, 60, 80, 100]):
            Image.fromarray(np.full((64, 96, 3), value, dtype=np.uint8)).save(source / f"{index}.png")

        execute = mocker.patch.object(ort.InferenceSession, "run", autospec=True, side_effect=ort.InferenceSession.run)
        results = list(model.predict(source, stream=True, batch=2))

        assert [call.args[2]["images"].shape[0] for call in execute.call_args_list] == [2, 2, 1]
        assert len(results) == 5
        for result, value in zip(results, [20, 40, 60, 80, 100]):
            assert isinstance(result, Detections)
            np.testing.assert_array_equal(result.metadata["source_image"], value)
            expected_box, _ = _expected_detection((value, value, value), height=64, width=96)
            np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)

    def test_dynamic_batch_default_for_a_directory(
        self, exported_detection: tuple[Path, dict[str, object]], tmp_path: Path, mocker
    ) -> None:
        """Without ``batch``, a dynamic-batch export runs an eager directory in batches of its ``max_batch_size``.

        The bound comes from the artifact's metadata, so a ten-image directory reaches the ONNX session as batches of
        four, four and two, in source order.
        """
        import onnxruntime as ort

        path, metadata = exported_detection
        metadata = dict(metadata, input_shape=[-1, 3, 32, 48], max_batch_size=4)
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        source = tmp_path / "images"
        source.mkdir()
        for index in range(10):
            Image.fromarray(np.full((64, 96, 3), 20 * index, dtype=np.uint8)).save(source / f"{index}.png")

        execute = mocker.patch.object(ort.InferenceSession, "run", autospec=True, side_effect=ort.InferenceSession.run)
        results = model.predict(source)

        assert [call.args[2]["images"].shape[0] for call in execute.call_args_list] == [4, 4, 2]
        assert [int(result.metadata["source_image"][0, 0, 0]) for result in results] == [20 * i for i in range(10)]

    @_requires_opencv
    def test_two_camera_streams_fill_a_batch_two_export(
        self, exported_batch_two: tuple[Path, dict[str, object]], tmp_path: Path, mocker
    ) -> None:
        """A `.streams` manifest of two cameras feeds one batch-two export frame by frame, in source order."""
        path, metadata = exported_batch_two
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        manifest = tmp_path / "cameras.streams"
        manifest.write_text("0\n1\n")
        cameras = [mocker.Mock(), mocker.Mock()]
        for camera, value in zip(cameras, [40, 80]):
            camera.read.return_value = (True, np.full((64, 96, 3), value, dtype=np.uint8))
        mocker.patch("cv2.VideoCapture", side_effect=cameras)

        results = model.predict(manifest, stream=True, stream_buffer=True)
        try:
            first, second = next(results), next(results)
        finally:
            results.close()

        for result, value in zip([first, second], [40, 80]):
            assert isinstance(result, Detections)
            expected_box, _ = _expected_detection((value, value, value), height=64, width=96)
            np.testing.assert_allclose(result.xyxy, [expected_box], rtol=0, atol=1e-3)
        for camera in cameras:
            camera.release.assert_called_once()

    @_requires_opencv
    def test_eager_live_source_warns_about_memory(
        self, exported_detection: tuple[Path, dict[str, object]], mocker
    ) -> None:
        """Eager prediction on a live camera warns that results accumulate in memory before opening the device."""
        path, metadata = exported_detection
        model = RFDETRInference(path, metadata=metadata, device="cpu")
        camera = mocker.Mock()
        camera.isOpened.return_value = False
        open_capture = mocker.patch("cv2.VideoCapture", return_value=camera)

        with pytest.warns(UserWarning, match="accumulate in memory"), pytest.raises(ValueError, match="Could not open"):
            model.predict(0)

        open_capture.assert_called_once()


@pytest.fixture
def prediction_video(tmp_path: Path) -> Path:
    """Create six grayscale frames with known pixel values.

    Examples:
        >>> prediction_video()  # doctest: +SKIP
        # Pytest supplies the temporary directory and OpenCV codec support.
    """
    import cv2  # optional dependency, only requested by tests guarded with a skipif

    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (96, 64))
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
    """Fix the channel-readout detector to a batch of two.

    Examples:
        >>> exported_batch_two()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact fixture.
    """
    import onnx

    path, metadata = exported_detection
    model = onnx.load(path)
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = 2
    onnx.save(model, path)
    return path, dict(metadata, input_shape=[2, 3, 32, 48])
