# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Distinct prediction rows and real MPS contracts for ``ClassThresholds``, without weights."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr import ClassThresholds
from tests._markers import _requires_mps
from tests.inference.helpers import _DummyModel, _DummyRFDETR, _record_model_inputs


def _row_result(
    labels: list[int], image_index: int, device: str, score_dtype: torch.dtype, keypoints: bool
) -> dict[str, torch.Tensor]:
    """Build distinct postprocessing rows for one image on the requested device."""
    rows = torch.arange(4, device=device, dtype=torch.float32) + image_index * 10
    result = {
        "scores": torch.tensor([0.65, 0.75, 0.85, 0.9], device=device, dtype=score_dtype),
        "labels": torch.tensor(labels, device=device),
        "boxes": rows[:, None] + torch.arange(4, device=device),
    }
    if keypoints:
        result["keypoints"] = rows[:, None, None].repeat(1, 2, 3)
        result["keypoints"][:, :, 2] = torch.tensor([[0.0, 0.25], [0.5, 0.0], [0.75, 1.0], [0.0, 1.0]], device=device)
        result["keypoint_precision_cholesky"] = (rows + 1)[:, None, None].repeat(1, 2, 3)
    else:
        pixels = torch.arange(16, device=device).reshape(1, 1, 4, 4)
        result["masks"] = pixels == (rows % 16)[:, None, None, None]
    assert all(tensor.device.type == device for tensor in result.values())
    return result


def _assert_selected_rows(
    prediction: sv.Detections | sv.KeyPoints,
    source: dict[str, torch.Tensor],
    selected: list[int],
    expected_ids: list[int],
    expected_names: list[str],
    keypoints: bool,
) -> None:
    """Check that all prediction fields retain the selected rows of one image."""
    assert prediction.class_id.tolist() == expected_ids
    assert prediction.data["class_name"].tolist() == expected_names
    np.testing.assert_array_equal(
        prediction.data["xyxy"] if keypoints else prediction.xyxy, source["boxes"].cpu().numpy()[selected]
    )
    np.testing.assert_array_equal(
        prediction.detection_confidence if keypoints else prediction.confidence,
        source["scores"].float().cpu().numpy()[selected],
    )
    if keypoints:
        assert isinstance(prediction, sv.KeyPoints)
        expected = source["keypoints"].cpu().numpy()[selected]
        np.testing.assert_array_equal(prediction.xy, expected[:, :, :2])
        np.testing.assert_array_equal(prediction.keypoint_confidence, expected[:, :, 2])
        np.testing.assert_array_equal(prediction.visible, expected[:, :, 2] > 0)
        np.testing.assert_array_equal(
            prediction.data["keypoint_precision_cholesky"],
            source["keypoint_precision_cholesky"].cpu().numpy()[selected],
        )
    else:
        assert isinstance(prediction, sv.Detections)
        np.testing.assert_array_equal(prediction.mask, source["masks"].cpu().numpy()[selected, 0])


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=_requires_mps)])
@pytest.mark.parametrize(
    "score_dtype", [pytest.param(torch.float16, id="float16"), pytest.param(torch.float32, id="float32")]
)
class TestClassThresholdRowAlignment:
    """Non-prefix selections must carry every output field from the same rows of the same image."""

    @pytest.mark.parametrize("schema", ["contiguous", "coco", "active-keypoint", "legacy-keypoint"])
    def test_distinct_rows_follow_each_images_class_thresholds(
        self, device: str, score_dtype: torch.dtype, schema: str, mocker: MockerFixture
    ) -> None:
        """Keep rows 0/2, then 1/3, so prefix slicing and cross-image row reuse change observable data."""
        keypoints = "keypoint" in schema
        labels = [1, 18, 3, 18] if schema == "coco" else ([1, 2, 3, 0] if schema == "legacy-keypoint" else [0, 1, 2, 1])
        model = _DummyRFDETR()
        model.model = _DummyModel(class_names=None if schema == "coco" else ["a", "b", "c"], device=device)
        model.model.args = SimpleNamespace(
            num_classes=90 if schema == "coco" else 3,
            num_keypoints_per_class=[0, 2, 2, 2] if schema == "legacy-keypoint" else ([2, 2, 2] if keypoints else []),
        )
        threshold = ClassThresholds({"dog" if schema == "coco" else "b": 0.95}, default=0.5)
        per_image = [
            _row_result(labels, 0, device, score_dtype, keypoints),
            _row_result(list(reversed(labels)), 1, device, score_dtype, keypoints),
        ]
        image = Image.new("RGB", (28, 28))
        batches: list[torch.Tensor] = []
        mocker.patch.object(model.model, "postprocess", return_value=per_image)
        with model.model.model.register_forward_pre_hook(_record_model_inputs(batches)):
            output = model.predict([image, image], threshold=threshold)

        assert [batch.device.type for batch in batches] == [device]
        assert len(output) == 2
        expected_ids = [1, 3] if schema in ("coco", "legacy-keypoint") else [0, 2]
        expected_names = ["person", "car"] if schema == "coco" else ["a", "c"]
        _assert_selected_rows(output[0], per_image[0], [0, 2], expected_ids, expected_names, keypoints)
        _assert_selected_rows(output[1], per_image[1], [1, 3], expected_ids[::-1], expected_names[::-1], keypoints)

    def test_empty_keypoint_result_keeps_per_object_field_shapes(self, device: str, score_dtype: torch.dtype) -> None:
        """Dropping every class empties all keypoint fields, including metadata, along the detection axis."""
        model = _DummyRFDETR()
        model.model = _DummyModel(
            class_names=["a"],
            labels=[0, 0],
            scores=[0.5, 0.75],
            include_keypoints=True,
            num_keypoints=2,
            device=device,
            score_dtype=score_dtype,
        )
        model.model.args = SimpleNamespace(num_classes=1, num_keypoints_per_class=[2])

        result = model.predict(Image.new("RGB", (28, 28)), threshold=ClassThresholds([float("inf")]))

        assert isinstance(result, sv.KeyPoints)
        assert result.xy.shape == (0, 2, 2)
        assert result.keypoint_confidence.shape == result.visible.shape == (0, 2)
        assert result.detection_confidence.shape == result.class_id.shape == (0,)
        assert result.data["xyxy"].shape == (0, 4)
        assert result.data["keypoint_precision_cholesky"].shape == (0, 2, 3)
        assert result.data["class_name"].shape == (0,)


@_requires_mps
@pytest.mark.parametrize(
    "score_dtype", [pytest.param(torch.float16, id="float16"), pytest.param(torch.float32, id="float32")]
)
class TestMpsClassThresholds:
    """Run the existing prediction code on real MPS tensors; the network and postprocess outputs are synthetic."""

    def test_boundary_rounding_matches_scalar_threshold_on_mps(self, score_dtype: torch.dtype) -> None:
        """FP16 rounds 0.4001 up to score equality; FP32 keeps a score of 0.400146484375."""
        model = _DummyRFDETR()
        model.model = _DummyModel(
            class_names=["a"], labels=[0], scores=[0.400146484375], device="mps", score_dtype=score_dtype
        )
        image = Image.new("RGB", (28, 28))
        batches: list[torch.Tensor] = []

        with model.model.model.register_forward_pre_hook(_record_model_inputs(batches)):
            per_class = model.predict(image, threshold=ClassThresholds([0.4001]))
            scalar = model.predict(image, threshold=0.4001)

        expected_count = 0 if score_dtype == torch.float16 else 1
        assert len(per_class) == len(scalar) == expected_count
        np.testing.assert_array_equal(per_class.class_id, scalar.class_id)
        assert [batch.device.type for batch in batches] == ["mps", "mps"]

    def test_grad_bearing_mps_threshold_tensor_is_detached(self, score_dtype: torch.dtype) -> None:
        """A device-resident threshold container can be read without NumPy conversion or autograd warnings."""
        model = _DummyRFDETR()
        model.model = _DummyModel(
            class_names=["a", "b"], labels=[0, 1], scores=[0.5, 0.75], device="mps", score_dtype=score_dtype
        )
        values = torch.tensor([0.5, 0.5], device="mps", dtype=score_dtype, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            threshold = ClassThresholds(values)

        result = model.predict(Image.new("RGB", (28, 28)), threshold=threshold)

        assert values.device.type == "mps" and values.requires_grad
        assert result.class_id.tolist() == [1]

    @pytest.mark.parametrize("form", ["pil", "list-one", "numpy", "cpu-tensor", "mps-tensor", "bchw"])
    def test_image_forms_preserve_selection_and_return_shape_on_mps(self, score_dtype: torch.dtype, form: str) -> None:
        """Every input route reaches MPS and applies the named class override and default to its results."""
        model = _DummyRFDETR()
        model.model = _DummyModel(
            class_names=["a", "b"], labels=[0, 1], scores=[0.7, 0.75], device="mps", score_dtype=score_dtype
        )
        image = Image.new("RGB", (28, 28))
        tensor = torch.zeros((3, 28, 28))
        images = {
            "pil": image,
            "list-one": [image],
            "numpy": np.zeros((28, 28, 3), dtype=np.uint8),
            "cpu-tensor": tensor,
            "mps-tensor": tensor.to("mps"),
            "bchw": torch.stack((tensor, tensor)).to("mps"),
        }[form]
        batches: list[torch.Tensor] = []

        with model.model.model.register_forward_pre_hook(_record_model_inputs(batches)):
            result = model.predict(images, threshold=ClassThresholds({"b": 0.8}, default=0.6))

        is_batch = form in ("list-one", "bchw")
        assert isinstance(result, list) == is_batch
        outputs = result if is_batch else [result]
        expected_count = 2 if form == "bchw" else 1
        assert len(outputs) == expected_count
        assert [prediction.class_id.tolist() for prediction in outputs] == [[0]] * expected_count
        assert [(batch.device.type, len(batch)) for batch in batches] == [("mps", expected_count)]
