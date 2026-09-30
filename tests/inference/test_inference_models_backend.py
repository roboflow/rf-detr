# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public inference_models backend contracts."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from rfdetr import RFDETRInference
from rfdetr.detr import RFDETR

_NO_OPTIONAL_RUNTIME_SCRIPT = """
import importlib.abc
import json
import sys
from rfdetr.detr import RFDETR

class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'inference_models'}:
            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)
        return None

for name in list(sys.modules):
    if name.split('.')[0] in {'onnxruntime', 'inference_models'}:
        del sys.modules[name]
sys.meta_path.insert(0, BlockOptionalRuntime())
try:
    RFDETR.from_export(sys.argv[1], metadata=json.loads(sys.argv[2]), backend='inference_models', device='cpu')
except Exception as error:
    print(f'{type(error).__name__}: {error}')
else:
    print('NO_ERROR')
"""


@pytest.fixture(scope="module")
def exported_detection(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Create a tiny ONNX detector whose cat score depends on the red input channel.

    Examples:
        >>> exported_detection()  # doctest: +SKIP
        # Pytest supplies the temporary directory and optional ONNX packages.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path_factory.mktemp("inference-models") / "detector.onnx"
    values = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="red_channel"),
        numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), name="logit_shape"),
        numpy_helper.from_array(np.array([12.0], dtype=np.float32), name="score_scale"),
        numpy_helper.from_array(np.array([-6.0], dtype=np.float32), name="score_offset"),
        numpy_helper.from_array(np.full((1, 1, 17), -10.0, dtype=np.float32), name="before_cat"),
        numpy_helper.from_array(np.full((1, 1, 72), -10.0, dtype=np.float32), name="after_cat"),
    ]
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["images", "red_channel"], ["red"], axis=1),
            helper.make_node("ReduceMean", ["red"], ["red_mean"], axes=[2, 3], keepdims=1),
            helper.make_node("Reshape", ["red_mean", "logit_shape"], ["red_scalar"]),
            helper.make_node("Mul", ["red_scalar", "score_scale"], ["scaled_red"]),
            helper.make_node("Add", ["scaled_red", "score_offset"], ["cat_score"]),
            helper.make_node("Concat", ["before_cat", "cat_score", "after_cat"], ["labels"], axis=2),
            helper.make_node("Constant", [], ["dets"], value=numpy_helper.from_array(boxes)),
        ],
        "input-sensitive-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 16, 16])],
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 1, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 1, 90]),
        ],
        initializer=values,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path, {
        "format": "onnx",
        "task": "detect",
        "variant": "rfdetr-nano",
        "input_shape": [1, 3, 16, 16],
        "input_name": "images",
        "outputs": {"pred_boxes": "dets", "pred_logits": "labels"},
        "class_names": ["cat"],
        "class_id_to_name": {"17": "cat"},
        "num_classes": 90,
        "num_select": 1,
        "patch_size": 16,
        "num_windows": 1,
        "trace_alpha": 0.2,
        "means": [0.0, 0.0, 0.0],
        "stds": [1.0, 1.0, 1.0],
    }


@pytest.fixture
def model(exported_detection: tuple[Path, dict[str, Any]]) -> RFDETRInference:
    """Load the tiny detector through the public inference_models backend.

    Examples:
        >>> model()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact and installed inference_models runtime.
    """
    pytest.importorskip("inference_models")
    path, metadata = exported_detection
    return RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")


@pytest.fixture
def red_image() -> np.ndarray[Any, Any]:
    """Return a red source image larger than the fixed model input.

    Examples:
        >>> red_image()  # doctest: +SKIP
        # Pytest supplies this fixture to prediction tests.
    """
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    return image


class TestBackendSelection:
    """Backend selection gives clear errors before runtime loading."""

    def test_unknown_backend(self, tmp_path: Path) -> None:
        """An unknown backend name fails before artifact loading."""
        artifact = tmp_path / "model.onnx"
        artifact.touch()
        with pytest.raises(ValueError, match="backend"):
            RFDETR.from_export(artifact, backend="not-a-backend", device="cpu")

    def test_missing_dependency(self, exported_detection: tuple[Path, dict[str, Any]]) -> None:
        """A valid graph reports the missing SDK even without ONNX Runtime."""
        path, metadata = exported_detection
        result = subprocess.run(
            [sys.executable, "-c", _NO_OPTIONAL_RUNTIME_SCRIPT, str(path), json.dumps(metadata)],
            check=True,
            text=True,
            capture_output=True,
        )
        assert result.stdout.strip().startswith("ImportError:")
        assert "requires inference-models" in result.stdout

    def test_unsupported_backbone(self, exported_detection: tuple[Path, dict[str, Any]]) -> None:
        """A backbone artifact has no full SDK prediction pipeline."""
        path, metadata = exported_detection
        with pytest.raises(ValueError, match="backbone|task|support"):
            RFDETR.from_export(
                path, metadata=dict(metadata, task="backbone", outputs={}), backend="inference_models", device="cpu"
            )

    def test_unsupported_format(self, tmp_path: Path, exported_detection: tuple[Path, dict[str, Any]]) -> None:
        """The SDK bridge rejects artifact formats outside its supported set."""
        artifact = tmp_path / "detector.tflite"
        artifact.touch()
        _, metadata = exported_detection
        with pytest.raises(ValueError, match="format|support|ONNX|TensorRT"):
            RFDETR.from_export(artifact, metadata=dict(metadata, format="tflite"), backend="inference_models")


class TestPublicPrediction:
    """The optional backend keeps the exported prediction contract."""

    def test_rgb_coordinates_and_sparse_class(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """The red input produces one cat at source coordinates, with source metadata."""
        result = model.predict(red_image)
        assert isinstance(result, sv.Detections)
        np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]], atol=1)
        np.testing.assert_array_equal(result.class_id, [17])
        assert result.xyxy.dtype == np.float32
        assert result.class_id is not None
        assert result.class_id.dtype == np.int64
        assert list(result.data["class_name"]) == ["cat"]
        np.testing.assert_array_equal(result.data["source_shape"], [[64, 96]])
        np.testing.assert_array_equal(result.metadata["source_image"], red_image)
        assert model.class_names == ["cat"]
        assert model.runtime_info["backend"] == "inference_models"

    @pytest.mark.parametrize("input_kind", ["pil", "path"])
    def test_source_image_from_pil_or_path_is_writeable(
        self, model: RFDETRInference, red_image: np.ndarray[Any, Any], tmp_path: Path, input_kind: str
    ) -> None:
        """Detection source images from Pillow can be drawn on in place."""
        image = Image.fromarray(red_image)
        supplied: Image.Image | str = image
        if input_kind == "path":
            path = tmp_path / "red.png"
            image.save(path)
            supplied = str(path)
        result = model.predict(supplied)
        assert isinstance(result, sv.Detections)
        source = result.metadata["source_image"]
        assert isinstance(source, np.ndarray)
        assert source.flags.writeable
        source[0, 0, 0] = 0

    @pytest.mark.parametrize("input_kind", ["array", "pil", "path", "tensor"])
    def test_rgb_input_forms(
        self, model: RFDETRInference, red_image: np.ndarray[Any, Any], tmp_path: Path, input_kind: str
    ) -> None:
        """All public RGB input forms preserve the red-sensitive detection."""
        if input_kind == "array":
            supplied = red_image
        elif input_kind == "pil":
            supplied = Image.fromarray(red_image)
        elif input_kind == "path":
            path = tmp_path / "red.png"
            Image.fromarray(red_image).save(path)
            supplied = str(path)
        else:
            supplied = torch.from_numpy(red_image.copy()).permute(2, 0, 1).float().div(255)
        result = model.predict(supplied, include_source_image=False)
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.class_id, [17])
        assert "source_image" not in result.metadata

    def test_blue_input_has_no_detection(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """The graph reads RGB pixels rather than returning constant predictions."""
        blue = red_image.copy()
        blue[:, :, 0] = 0
        blue[:, :, 2] = 255
        result = model.predict(blue, threshold=0.5)
        assert isinstance(result, sv.Detections)
        assert len(result) == 0

    def test_threshold_filters_detection(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """A threshold above the cat score filters the detection."""
        result = model.predict(red_image, threshold=0.999)
        assert isinstance(result, sv.Detections)
        assert len(result) == 0

    def test_one_item_list_returns_list(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """A list input stays a list after one SDK batch call."""
        result = model.predict([red_image])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0].class_id, [17])

    def test_fixed_batch_rejected(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """A fixed-batch artifact rejects two images before SDK execution."""
        with pytest.raises(ValueError, match="Batch size|batch"):
            model.predict([red_image, red_image])

    def test_shape_mismatch(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """Prediction cannot override the artifact's fixed input shape."""
        with pytest.raises(ValueError, match="shape"):
            model.predict(red_image, shape=(32, 32))

    def test_patch_size_mismatch(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """Prediction cannot override the artifact's patch size."""
        with pytest.raises(ValueError, match="patch"):
            model.predict(red_image, patch_size=8)


@pytest.fixture
def batched_detection(exported_detection: tuple[Path, dict[str, Any]], tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Make a fixed-batch two-image version of the red-sensitive ONNX detector.

    Examples:
        >>> batched_detection()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact and temporary output directory.
    """
    import onnx
    from onnx import numpy_helper

    path, metadata = exported_detection
    graph = onnx.load(path)
    graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    for output in graph.graph.output:
        output.type.tensor_type.shape.dim[0].dim_value = 2
    replacements = {
        "logit_shape": np.array([2, 1, 1], dtype=np.int64),
        "before_cat": np.full((2, 1, 17), -10.0, dtype=np.float32),
        "after_cat": np.full((2, 1, 72), -10.0, dtype=np.float32),
    }
    for initializer in graph.graph.initializer:
        if initializer.name in replacements:
            initializer.CopyFrom(numpy_helper.from_array(replacements[initializer.name], name=initializer.name))
    for node in graph.graph.node:
        if "dets" in node.output:
            boxes = np.tile(np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32), (2, 1, 1))
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(boxes))
    onnx.checker.check_model(graph)
    batched_path = tmp_path / "detector-batch2.onnx"
    onnx.save(graph, batched_path)
    return batched_path, dict(metadata, input_shape=[2, 3, 16, 16])


class TestArtifactInterface:
    """The SDK backend checks artifact metadata before inference."""

    def test_malformed_graph_without_optional_runtimes(self, exported_detection: tuple[Path, dict[str, Any]]) -> None:
        """Graph errors win even when both the SDK and ONNX Runtime are absent."""
        path, metadata = exported_detection
        result = subprocess.run(
            [sys.executable, "-c", _NO_OPTIONAL_RUNTIME_SCRIPT, str(path), json.dumps(dict(metadata, num_select=2))],
            check=True,
            text=True,
            capture_output=True,
        )
        assert result.stdout.strip().startswith("ValueError:")
        assert "query" in result.stdout

    def test_num_select_mismatch(self, exported_detection: tuple[Path, dict[str, Any]]) -> None:
        """Metadata cannot request more queries than the graph provides."""
        path, metadata = exported_detection
        with pytest.raises(ValueError, match="num_select|quer"):
            RFDETR.from_export(path, metadata=dict(metadata, num_select=2), backend="inference_models", device="cpu")

    @pytest.mark.parametrize("identifier", ["input", "output"])
    def test_onnx_io_mismatch(self, exported_detection: tuple[Path, dict[str, Any]], identifier: str) -> None:
        """Metadata names must identify graph inputs and outputs."""
        path, metadata = exported_detection
        metadata = dict(metadata)
        if identifier == "input":
            metadata["input_name"] = "missing_images"
        else:
            metadata["outputs"] = {"pred_boxes": "dets", "pred_logits": "missing_labels"}
        with pytest.raises(ValueError, match="input|output|metadata|interface"):
            RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")

    def test_unavailable_explicit_device(self, exported_detection: tuple[Path, dict[str, Any]]) -> None:
        """An unavailable explicit device never falls back to CPU."""
        path, metadata = exported_detection
        with pytest.raises(ValueError, match="cpu, cuda:N, or auto"):
            RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="mps")

    def test_mixed_fixed_batch(
        self, batched_detection: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """One fixed-batch SDK call keeps per-image detections in source order."""
        pytest.importorskip("inference_models")
        path, metadata = batched_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        blue = red_image.copy()
        blue[:, :, 0] = 0
        blue[:, :, 2] = 255
        results = model.predict([red_image, blue], threshold=0.5)
        assert isinstance(results, list)
        assert len(results) == 2
        np.testing.assert_array_equal(results[0].class_id, [17])
        assert len(results[1]) == 0


class TestDevicePolicy:
    """Device selection checks the actual optional runtime provider."""

    def test_auto_uses_cpu_without_cuda_provider(
        self, exported_detection: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A visible CUDA device does not imply an ONNX CUDA provider."""
        pytest.importorskip("inference_models")
        import onnxruntime as ort

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        path, metadata = exported_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="auto")
        assert model.runtime_info["device"] == "cpu"

    def test_explicit_cuda_rejects_missing_provider(
        self, exported_detection: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit CUDA request fails when ONNX Runtime has no CUDA provider."""
        pytest.importorskip("inference_models")
        import onnxruntime as ort

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        path, metadata = exported_detection
        with pytest.raises((RuntimeError, ValueError), match="CUDAExecutionProvider|CUDA"):
            RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cuda:0")


class TestInputOwnership:
    """Source capture and float pixels retain the native input contract."""

    def test_source_image_omitted_when_disabled(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """Disabling source capture leaves no image in result metadata."""
        result = model.predict(red_image, include_source_image=False)
        assert isinstance(result, sv.Detections)
        assert "source_image" not in result.metadata

    def test_source_image_is_snapshot(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """Changing caller storage after prediction does not change the result image."""
        result = model.predict(red_image)
        assert isinstance(result, sv.Detections)
        red_image[:, :, 0] = 0
        assert np.all(result.metadata["source_image"][:, :, 0] == 255)

    def test_float_numpy_pixels_keep_sub_byte_precision(self, model: RFDETRInference) -> None:
        """Float RGB input around the score cutoff is not rounded to uint8 first."""
        low = np.zeros((16, 16, 3), dtype=np.float32)
        low[:, :, 0] = 0.4999
        high = low.copy()
        high[:, :, 0] = 0.5001
        low_result = model.predict(low, threshold=0.5, include_source_image=False)
        high_result = model.predict(high, threshold=0.5, include_source_image=False)
        assert isinstance(low_result, sv.Detections)
        assert isinstance(high_result, sv.Detections)
        assert len(low_result) == 0
        np.testing.assert_array_equal(high_result.class_id, [17])


class TestShapeTypes:
    """Prediction rejects values that look numeric but are not valid dimensions."""

    @pytest.mark.parametrize("invalid_height", [True, 16.0])
    def test_invalid_shape_dimension(
        self, model: RFDETRInference, red_image: np.ndarray[Any, Any], invalid_height: Any
    ) -> None:
        """Boolean and floating dimensions do not pass integer shape validation."""
        with pytest.raises(ValueError, match="shape|integer"):
            model.predict(red_image, shape=(invalid_height, 16))

    def test_boolean_patch_size(self, model: RFDETRInference, red_image: np.ndarray[Any, Any]) -> None:
        """True is not an accepted patch size even though it equals one."""
        with pytest.raises(ValueError, match="patch_size"):
            model.predict(red_image, patch_size=True)  # type: ignore[arg-type]


class TestSdkBoundary:
    """The public backend detects SDK device violations."""

    def test_explicit_cuda_rejects_sdk_cpu_predictions(
        self,
        exported_detection: tuple[Path, dict[str, Any]],
        red_image: np.ndarray[Any, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CUDA request fails if the SDK silently returns CPU tensors."""
        from unittest.mock import Mock

        pytest.importorskip("inference_models")
        import onnxruntime as ort
        from inference_models import AutoModel

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        fake_model = Mock()
        fake_model.pre_process.return_value = (Mock(device=torch.device("cuda:0")), None)
        fake_model.infer.return_value = [Mock(xyxy=Mock(device=torch.device("cpu")))]
        monkeypatch.setattr(AutoModel, "from_pretrained", Mock(return_value=fake_model))
        path, metadata = exported_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cuda:0")
        with pytest.raises(RuntimeError, match="instead of requested"):
            model.predict(red_image)

    def test_sdk_infer_is_outside_forced_inference_mode(
        self,
        exported_detection: tuple[Path, dict[str, Any]],
        red_image: np.ndarray[Any, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The SDK can reuse cached buffers after one prediction."""
        from unittest.mock import Mock

        pytest.importorskip("inference_models")
        from inference_models import AutoModel

        prediction = Mock(xyxy=Mock(device=torch.device("cpu")))
        prediction.to_supervision.return_value = sv.Detections.empty()
        fake_model = Mock()
        fake_model.pre_process.return_value = (Mock(device=torch.device("cpu")), None)
        fake_model.infer.side_effect = lambda *_args, **_kwargs: (
            [prediction]
            if not torch.is_inference_mode_enabled()
            else pytest.fail("SDK infer ran inside forced torch.inference_mode")
        )
        monkeypatch.setattr(AutoModel, "from_pretrained", Mock(return_value=fake_model))
        path, metadata = exported_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.Detections)
        assert fake_model.infer.call_count == 1


@pytest.fixture
def dynamic_detection(exported_detection: tuple[Path, dict[str, Any]], tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create a red-sensitive ONNX detector with a two-image dynamic batch limit.

    Examples:
        >>> dynamic_detection()  # doctest: +SKIP
        # Pytest supplies ONNX and a temporary output directory.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    _, metadata = exported_detection
    values = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="red_channel"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="batch_axis"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), name="unsqueeze_axis"),
        numpy_helper.from_array(np.array([1, 17], dtype=np.int64), name="before_tail"),
        numpy_helper.from_array(np.array([1, 72], dtype=np.int64), name="after_tail"),
        numpy_helper.from_array(np.array([1, 4], dtype=np.int64), name="boxes_tail"),
        numpy_helper.from_array(np.array([12.0], dtype=np.float32), name="score_scale"),
        numpy_helper.from_array(np.array([-6.0], dtype=np.float32), name="score_offset"),
        numpy_helper.from_array(np.full((1, 1, 17), -10.0, dtype=np.float32), name="before_base"),
        numpy_helper.from_array(np.full((1, 1, 72), -10.0, dtype=np.float32), name="after_base"),
        numpy_helper.from_array(np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32), name="boxes_base"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["images"], ["image_shape"]),
            helper.make_node("Gather", ["image_shape", "batch_axis"], ["batch_count"], axis=0),
            helper.make_node("Concat", ["batch_count", "before_tail"], ["before_shape"], axis=0),
            helper.make_node("Concat", ["batch_count", "after_tail"], ["after_shape"], axis=0),
            helper.make_node("Concat", ["batch_count", "boxes_tail"], ["boxes_shape"], axis=0),
            helper.make_node("Expand", ["before_base", "before_shape"], ["before_cat"]),
            helper.make_node("Expand", ["after_base", "after_shape"], ["after_cat"]),
            helper.make_node("Expand", ["boxes_base", "boxes_shape"], ["dets"]),
            helper.make_node("Gather", ["images", "red_channel"], ["red"], axis=1),
            helper.make_node("ReduceMean", ["red"], ["red_mean"], axes=[2, 3], keepdims=0),
            helper.make_node("Unsqueeze", ["red_mean", "unsqueeze_axis"], ["red_scalar"]),
            helper.make_node("Mul", ["red_scalar", "score_scale"], ["scaled_red"]),
            helper.make_node("Add", ["scaled_red", "score_offset"], ["cat_score"]),
            helper.make_node("Concat", ["before_cat", "cat_score", "after_cat"], ["labels"], axis=2),
        ],
        "dynamic-input-sensitive-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 16, 16])],
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, ["batch", 1, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, ["batch", 1, 90]),
        ],
        initializer=values,
    )
    graph_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(graph_model)
    path = tmp_path / "detector-dynamic.onnx"
    onnx.save(graph_model, path)
    return path, dict(metadata, input_shape=[-1, 3, 16, 16], max_batch_size=2)


class TestDynamicBatch:
    """A dynamic ONNX artifact executes each requested batch as one SDK call."""

    def test_one_image(self, dynamic_detection: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]) -> None:
        """A dynamic export accepts one image as a single result."""
        pytest.importorskip("inference_models")
        path, metadata = dynamic_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.Detections)
        np.testing.assert_array_equal(result.class_id, [17])

    def test_two_mixed_images(
        self, dynamic_detection: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """A dynamic export keeps two different image results in source order."""
        pytest.importorskip("inference_models")
        path, metadata = dynamic_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        blue = red_image.copy()
        blue[:, :, 0] = 0
        blue[:, :, 2] = 255
        results = model.predict([red_image, blue])
        assert isinstance(results, list)
        assert len(results) == 2
        np.testing.assert_array_equal(results[0].class_id, [17])
        assert len(results[1]) == 0

    def test_exceeds_maximum(
        self, dynamic_detection: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """The declared dynamic batch maximum rejects three images."""
        pytest.importorskip("inference_models")
        path, metadata = dynamic_detection
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        with pytest.raises(ValueError, match="batch|Batch|maximum"):
            model.predict([red_image, red_image, red_image])


@pytest.fixture
def exported_segmentation(
    exported_detection: tuple[Path, dict[str, Any]], tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    """Add one left-half mask to the red-sensitive ONNX detector.

    Examples:
        >>> exported_segmentation()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact and a temporary output directory.
    """
    pytest.importorskip("inference_models")
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path, metadata = exported_detection
    graph_model = onnx.load(path)
    masks = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
    masks[:, :, :, :2] = 10.0
    graph_model.graph.node.append(helper.make_node("Constant", [], ["masks"], value=numpy_helper.from_array(masks)))
    graph_model.graph.output.append(helper.make_tensor_value_info("masks", TensorProto.FLOAT, [1, 1, 4, 4]))
    onnx.checker.check_model(graph_model)
    segmentation_path = tmp_path / "segmenter.onnx"
    onnx.save(graph_model, segmentation_path)
    return segmentation_path, dict(metadata, task="segment", outputs={**metadata["outputs"], "pred_masks": "masks"})


class TestInstanceSegmentation:
    """The SDK backend returns full-resolution instance masks through public predict."""

    def test_red_image_mask_and_source_metadata(
        self, exported_segmentation: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """The image-sensitive graph returns a source-aligned mask and class."""
        pytest.importorskip("inference_models")
        path, metadata = exported_segmentation
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.Detections)
        np.testing.assert_allclose(result.xyxy, [[24, 16, 72, 48]], atol=1)
        np.testing.assert_array_equal(result.class_id, [17])
        assert result.xyxy.dtype == np.float32
        assert result.class_id is not None
        assert result.class_id.dtype == np.int64
        assert list(result.data["class_name"]) == ["cat"]
        np.testing.assert_array_equal(result.data["source_shape"], [[64, 96]])
        np.testing.assert_array_equal(result.metadata["source_image"], red_image)
        mask = result.mask
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (1, 64, 96)
        assert mask.dtype == np.bool_
        assert mask[0, 32, 12]
        assert not mask[0, 32, 84]

    @pytest.mark.parametrize("input_kind", ["pil", "path"])
    def test_source_image_from_pil_or_path_is_writeable(
        self,
        exported_segmentation: tuple[Path, dict[str, Any]],
        red_image: np.ndarray[Any, Any],
        tmp_path: Path,
        input_kind: str,
    ) -> None:
        """Segmentation source images from Pillow can be drawn on in place."""
        path, metadata = exported_segmentation
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        image = Image.fromarray(red_image)
        supplied: Image.Image | str = image
        if input_kind == "path":
            image_path = tmp_path / "red.png"
            image.save(image_path)
            supplied = str(image_path)
        result = model.predict(supplied)
        assert isinstance(result, sv.Detections)
        source = result.metadata["source_image"]
        assert isinstance(source, np.ndarray)
        assert source.flags.writeable
        source[0, 0, 0] = 0

    def test_blue_image_has_no_instance(
        self, exported_segmentation: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """A blue image leaves the input-sensitive segmentation output empty."""
        path, metadata = exported_segmentation
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        blue = red_image.copy()
        blue[:, :, 0] = 0
        blue[:, :, 2] = 255
        result = model.predict(blue, include_source_image=False)
        assert isinstance(result, sv.Detections)
        assert len(result) == 0
        mask = result.mask
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (0, 64, 96)
        assert "source_image" not in result.metadata

    def test_source_image_can_be_omitted(
        self, exported_segmentation: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """Mask predictions need no copied source image when capture is disabled."""
        path, metadata = exported_segmentation
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image, include_source_image=False)
        assert isinstance(result, sv.Detections)
        assert len(result) == 1
        assert "source_image" not in result.metadata

    def test_refuses_non_upsampled_mask_contract(self, exported_segmentation: tuple[Path, dict[str, Any]]) -> None:
        """The SDK cannot match native low-resolution mask output semantics."""
        path, metadata = exported_segmentation
        with pytest.raises(ValueError, match="mask|upsampl"):
            RFDETR.from_export(
                path,
                metadata=dict(metadata, upsample_masks_to_image_size=False),
                backend="inference_models",
                device="cpu",
            )


@pytest.fixture
def exported_keypoints(exported_detection: tuple[Path, dict[str, Any]], tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Add a one-point legacy-background-first pose to the red-sensitive ONNX graph.

    Examples:
        >>> exported_keypoints()  # doctest: +SKIP
        # Pytest supplies the ONNX artifact and a temporary output directory.
    """
    pytest.importorskip("inference_models")
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path, metadata = exported_detection
    graph_model = onnx.load(path)
    next(node for node in graph_model.graph.node if "labels" in node.output).output[0] = "all_labels"
    graph_model.graph.initializer.append(numpy_helper.from_array(np.array([0, 17], dtype=np.int64), name="pose_slots"))
    graph_model.graph.node.append(helper.make_node("Gather", ["all_labels", "pose_slots"], ["labels"], axis=2))
    point = np.zeros((1, 1, 2, 8), dtype=np.float32)
    point[0, 0, 1] = [0.25, 0.75, 10.0, 10.0, 2.0, 0.0, 2.0, 10.0]
    graph_model.graph.node.append(helper.make_node("Constant", [], ["keypoints"], value=numpy_helper.from_array(point)))
    graph_model.graph.output[1].type.tensor_type.shape.dim[2].dim_value = 2
    graph_model.graph.output.append(helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, [1, 1, 2, 8]))
    onnx.checker.check_model(graph_model)
    keypoint_path = tmp_path / "keypoints.onnx"
    onnx.save(graph_model, keypoint_path)
    return keypoint_path, dict(
        metadata,
        task="keypoints",
        outputs={**metadata["outputs"], "pred_keypoints": "keypoints"},
        class_names=["person"],
        class_id_to_name={"1": "person"},
        num_classes=2,
        num_keypoints_per_class=[0, 1],
    )


class TestKeypointPrediction:
    """The SDK backend returns native Supervision keypoints with source metadata."""

    def test_legacy_background_first_preserves_raw_class_id(
        self, exported_keypoints: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """One red-sensitive person pose keeps RF-DETR's original class slot."""
        pytest.importorskip("inference_models")
        path, metadata = exported_keypoints
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.KeyPoints)
        assert result.xy.shape == (1, 1, 2)
        assert result.xy.dtype == np.float32
        np.testing.assert_allclose(result.xy[0, 0], [24, 48], atol=1)
        np.testing.assert_array_equal(result.class_id, [1])
        assert result.class_id is not None
        assert result.class_id.dtype == np.int64
        assert list(result.data["class_name"]) == ["person"]
        np.testing.assert_allclose(np.asarray(result.data["xyxy"], dtype=np.float32), [[24, 16, 72, 48]], atol=1)
        np.testing.assert_array_equal(result.data["source_shape"], [[64, 96]])
        np.testing.assert_array_equal(result.data["source_image"][0], red_image)
        assert np.asarray(result.data["covariance"]).shape == (1, 1, 2, 2)
        assert result.keypoint_confidence is not None
        assert result.detection_confidence is not None
        assert result.keypoint_confidence[0, 0] > 0.9
        assert result.detection_confidence[0] > 0.5

    @pytest.mark.parametrize("input_kind", ["pil", "path"])
    def test_source_image_from_pil_or_path_is_writeable(
        self,
        exported_keypoints: tuple[Path, dict[str, Any]],
        red_image: np.ndarray[Any, Any],
        tmp_path: Path,
        input_kind: str,
    ) -> None:
        """Keypoint source images from Pillow can be drawn on in place."""
        path, metadata = exported_keypoints
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        image = Image.fromarray(red_image)
        supplied: Image.Image | str = image
        if input_kind == "path":
            image_path = tmp_path / "red.png"
            image.save(image_path)
            supplied = str(image_path)
        result = model.predict(supplied)
        assert isinstance(result, sv.KeyPoints)
        source = result.data["source_image"][0]
        assert isinstance(source, np.ndarray)
        assert source.flags.writeable
        source[0, 0, 0] = 0

    def test_foreground_name_resembling_background_is_preserved(
        self, exported_keypoints: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """An actual class named like an internal placeholder survives remapping."""
        path, metadata = exported_keypoints
        unusual_name = "__background_0__"
        metadata = dict(metadata, class_names=[unusual_name], class_id_to_name={"1": unusual_name})
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.KeyPoints)
        np.testing.assert_array_equal(result.class_id, [1])
        assert list(result.data["class_name"]) == [unusual_name]

    def test_active_first_preserves_zero_class_id(
        self, exported_keypoints: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any], tmp_path: Path
    ) -> None:
        """An active-first keypoint artifact keeps class slot zero."""
        import onnx
        from onnx import numpy_helper

        path, metadata = exported_keypoints
        graph_model = onnx.load(path)
        for initializer in graph_model.graph.initializer:
            if initializer.name == "pose_slots":
                initializer.CopyFrom(numpy_helper.from_array(np.array([17], dtype=np.int64), name="pose_slots"))
        for node in graph_model.graph.node:
            if "keypoints" in node.output:
                point = np.zeros((1, 1, 1, 8), dtype=np.float32)
                point[0, 0, 0] = [0.25, 0.75, 10.0, 10.0, 2.0, 0.0, 2.0, 10.0]
                node.attribute[0].t.CopyFrom(numpy_helper.from_array(point))
        graph_model.graph.output[1].type.tensor_type.shape.dim[2].dim_value = 1
        graph_model.graph.output[2].type.tensor_type.shape.dim[2].dim_value = 1
        onnx.checker.check_model(graph_model)
        active_path = tmp_path / "active-keypoints.onnx"
        onnx.save(graph_model, active_path)
        active_metadata = dict(
            metadata,
            class_id_to_name={"0": "person"},
            num_classes=1,
            num_keypoints_per_class=[1],
        )
        model = RFDETR.from_export(active_path, metadata=active_metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.KeyPoints)
        np.testing.assert_array_equal(result.class_id, [0])
        assert list(result.data["class_name"]) == ["person"]
        np.testing.assert_allclose(result.xy[0, 0], [24, 48], atol=1)

    def test_blue_image_has_no_pose(
        self, exported_keypoints: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """A blue image leaves the input-sensitive pose output empty."""
        path, metadata = exported_keypoints
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        blue = red_image.copy()
        blue[:, :, 0] = 0
        blue[:, :, 2] = 255
        result = model.predict(blue, include_source_image=False)
        assert isinstance(result, sv.KeyPoints)
        assert len(result) == 0
        assert result.xy.shape == (0, 1, 2)
        assert "source_image" not in result.data

    def test_source_image_is_snapshot(
        self, exported_keypoints: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """Pose source capture owns the image after the caller mutates it."""
        path, metadata = exported_keypoints
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        result = model.predict(red_image)
        assert isinstance(result, sv.KeyPoints)
        red_image[:, :, 0] = 0
        assert np.all(result.data["source_image"][0][:, :, 0] == 255)

    def test_fixed_batch_rejected(
        self, exported_keypoints: tuple[Path, dict[str, Any]], red_image: np.ndarray[Any, Any]
    ) -> None:
        """A fixed one-image keypoint artifact refuses two source images."""
        path, metadata = exported_keypoints
        model = RFDETR.from_export(path, metadata=metadata, backend="inference_models", device="cpu")
        with pytest.raises(ValueError, match="Batch size|batch"):
            model.predict([red_image, red_image])

    def test_refuses_other_trace_alpha(self, exported_keypoints: tuple[Path, dict[str, Any]]) -> None:
        """A different native keypoint score rule cannot be represented by the SDK."""
        path, metadata = exported_keypoints
        with pytest.raises(ValueError, match="trace_alpha|alpha"):
            RFDETR.from_export(path, metadata=dict(metadata, trace_alpha=0.1), backend="inference_models", device="cpu")

    def test_refuses_rank_three_keypoints(
        self, exported_keypoints: tuple[Path, dict[str, Any]], tmp_path: Path
    ) -> None:
        """Malformed packed keypoint output fails before SDK inference."""
        import onnx
        from onnx import numpy_helper

        path, metadata = exported_keypoints
        graph_model = onnx.load(path)
        for node in graph_model.graph.node:
            if "keypoints" in node.output:
                packed = np.zeros((1, 1, 8), dtype=np.float32)
                node.attribute[0].t.CopyFrom(numpy_helper.from_array(packed))
        graph_model.graph.output[2].type.tensor_type.shape.dim.pop()
        onnx.checker.check_model(graph_model)
        malformed_path = tmp_path / "rank-three-keypoints.onnx"
        onnx.save(graph_model, malformed_path)
        with pytest.raises(ValueError, match="keypoint|rank|shape"):
            RFDETR.from_export(malformed_path, metadata=metadata, backend="inference_models", device="cpu")
