# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import socket
from types import SimpleNamespace
from typing import Any

import numpy as np
import PIL.Image
import pytest
import supervision as sv
import torch

from rfdetr import RFDETRNano, RFDETRSegNano
from rfdetr.detr import RFDETR

_HTTP_IMAGE_URL = "http://images.cocodataset.org/val2017/000000397133.jpg"
_HTTP_HOST = "images.cocodataset.org"
_HTTP_PORT = 80


def _is_online(host: str, port: int, timeout_s: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class _DummyModel:
    def __init__(self, class_names: list[str] | None = None, labels: list[int] | None = None) -> None:
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = torch.nn.Identity()
        self.class_names = class_names
        self._labels = labels if labels is not None else [1]

    def postprocess(self, predictions: Any, target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            results.append(
                {
                    "scores": torch.tensor([0.9] * len(self._labels)),
                    "labels": torch.tensor(self._labels),
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(self._labels)),
                }
            )
        return results


class _DummyRFDETR(RFDETR):
    def maybe_download_pretrain_weights(self) -> None:
        return None

    def get_model_config(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace()

    def get_model(self, config: SimpleNamespace) -> _DummyModel:
        return _DummyModel()


class TestPredictReturnTypes:
    """``RFDETR.predict()`` API contract tests using synthetic images.

    Quality is not assessed here — see ``tests/benchmarks/test_inference_coco.py``.
    """

    def test_detection_returns_sv_detections(self) -> None:
        """Detection model returns a list of ``sv.Detections``."""
        img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))
        model = RFDETRNano()
        detections = model.predict([img, img], threshold=0.3)
        assert isinstance(detections, list), "predict() must return a list for multiple inputs"
        assert all(isinstance(d, sv.Detections) for d in detections), "Each result must be sv.Detections"

    def test_segmentation_returns_sv_detections_with_masks(self) -> None:
        """Segmentation model returns ``sv.Detections`` with the mask field always set."""
        img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))
        model = RFDETRSegNano()
        detections = model.predict([img, img], threshold=0.3)
        assert isinstance(detections, list), "predict() must return a list for multiple inputs"
        assert all(isinstance(d, sv.Detections) for d in detections), "Each result must be sv.Detections"
        assert all(d.mask is not None for d in detections), (
            "Segmentation predict() must always set the mask field, even when no objects are detected"
        )


def test_predict_accepts_image_url() -> None:
    if not _is_online(_HTTP_HOST, _HTTP_PORT):
        pytest.skip("Offline environment, skipping HTTP predict URL test.")
    model = _DummyRFDETR()
    detections = model.predict(_HTTP_IMAGE_URL)
    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (1, 4)


class TestPredictSourceData:
    """Verify ``predict()`` source metadata behavior."""

    def test_source_image_included_by_default(self) -> None:
        """source_image remains included by default for API compatibility."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img)
        assert "source_image" in detections.data
        assert isinstance(detections.data["source_image"], np.ndarray)
        assert detections.data["source_image"].shape == (48, 64, 3)
        assert detections.data["source_shape"] == (48, 64)

    def test_source_image_included_by_default_tensor(self) -> None:
        """Tensor input keeps source_image by default for API compatibility."""
        tensor = torch.rand(3, 48, 64)
        model = _DummyRFDETR()
        detections = model.predict(tensor)
        assert "source_image" in detections.data
        assert isinstance(detections.data["source_image"], np.ndarray)
        assert detections.data["source_image"].dtype == np.uint8
        assert detections.data["source_image"].shape == (48, 64, 3)
        assert detections.data["source_shape"] == (48, 64)

    def test_source_image_can_be_disabled(self) -> None:
        """include_source_image=False omits source_image for memory-sensitive paths."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img, include_source_image=False)
        assert "source_image" not in detections.data
        assert detections.data["source_shape"] == (48, 64)

    def test_source_image_from_pil(self) -> None:
        """PIL input stores the original image as a numpy array."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img, include_source_image=True)
        assert "source_image" in detections.data
        assert isinstance(detections.data["source_image"], np.ndarray)
        assert detections.data["source_image"].shape == (48, 64, 3)

    def test_source_shape_from_pil(self) -> None:
        """PIL input stores the original (height, width) tuple."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img)
        assert "source_shape" in detections.data
        assert detections.data["source_shape"] == (48, 64)

    def test_source_image_from_tensor(self) -> None:
        """Tensor input stores the original image as a uint8 numpy array."""
        tensor = torch.rand(3, 48, 64)
        model = _DummyRFDETR()
        detections = model.predict(tensor, include_source_image=True)
        assert "source_image" in detections.data
        assert isinstance(detections.data["source_image"], np.ndarray)
        assert detections.data["source_image"].dtype == np.uint8
        assert detections.data["source_image"].shape == (48, 64, 3)

    def test_tensor_with_negative_values_raises(self) -> None:
        """Tensor with negative pixel values raises ValueError."""
        tensor = torch.full((3, 48, 64), -0.1)
        model = _DummyRFDETR()
        with pytest.raises(ValueError, match="below 0"):
            model.predict(tensor)

    def test_source_image_batch(self) -> None:
        """Batch predict stores a source_image per detection."""
        img1 = PIL.Image.new("RGB", (64, 48), color=(100, 100, 100))
        img2 = PIL.Image.new("RGB", (32, 24), color=(200, 200, 200))
        model = _DummyRFDETR()
        detections_list = model.predict([img1, img2], include_source_image=True)
        assert isinstance(detections_list, list)
        assert detections_list[0].data["source_image"].shape == (48, 64, 3)
        assert detections_list[1].data["source_image"].shape == (24, 32, 3)
        assert detections_list[0].data["source_shape"] == (48, 64)
        assert detections_list[1].data["source_shape"] == (24, 32)


class TestPredictShape:
    """Verify that ``predict(shape=...)`` controls the resize target.

    Regression tests for https://github.com/roboflow/rf-detr/issues/682.
    """

    def test_predict_uses_resolution_when_no_shape_provided(self) -> None:
        """Without ``shape=``, resize uses ``(resolution, resolution)``."""
        from unittest.mock import patch

        import torchvision.transforms.functional as F

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr.detr.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img)

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [28, 28], f"Expected resize to model resolution (28, 28), got {resize_size}"

    def test_predict_uses_provided_rectangular_shape(self) -> None:
        # Regression test for #682
        from unittest.mock import patch

        import torchvision.transforms.functional as F

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr.detr.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, shape=(378, 672))

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [378, 672], (
            f"Expected resize to user-provided shape (378, 672), got {resize_size}. "
            "predict() must honour the shape parameter instead of falling back "
            "to (resolution, resolution)."
        )

    def test_predict_shape_square_override(self) -> None:
        # Regression test for #682 — square shape different from model resolution.
        from unittest.mock import patch

        import torchvision.transforms.functional as F

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr.detr.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, shape=(56, 56))

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [56, 56], (
            f"Expected resize to user-provided shape (56, 56), got {resize_size}. "
            "predict() must honour the shape parameter even for square sizes "
            "that differ from the model's default resolution."
        )

    @pytest.mark.parametrize(
        "int_shape",
        [
            pytest.param((np.int64(378), np.int64(672)), id="numpy_int64"),
            pytest.param((np.int32(378), np.int32(672)), id="numpy_int32"),
            pytest.param((torch.tensor(378), torch.tensor(672)), id="torch_scalar"),
        ],
    )
    def test_predict_shape_accepts_integer_like_types(self, int_shape: tuple) -> None:
        """predict() accepts integer-like types (numpy, torch) via the __index__ protocol."""
        from unittest.mock import patch

        import torchvision.transforms.functional as F

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr.detr.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, shape=int_shape)  # type: ignore[arg-type]

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [378, 672], f"predict() must accept integer-like shape types, got resize {resize_size}"

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((378, 671), id="width_not_div_14"),  # 671 % 14 != 0
            pytest.param((371, 672), id="height_not_div_14"),  # 371 % 14 != 0
        ],
    )
    def test_predict_shape_not_divisible_by_14_raises(self, bad_shape: tuple[int, int]) -> None:
        """predict() must reject shapes with dimensions not divisible by 14."""
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="divisible by 14"):
            model.predict(img, shape=bad_shape)

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((378.0, 672.0), id="float_dims"),
            pytest.param((378,), id="wrong_arity_one_element"),
            pytest.param((378, 672, 3), id="wrong_arity_three_elements"),
            pytest.param((0, 56), id="zero_height"),
            pytest.param((-14, 56), id="negative_height"),
            pytest.param((56, 0), id="zero_width"),
            pytest.param((56, -14), id="negative_width"),
            pytest.param((True, 56), id="bool_height"),
            pytest.param((56, False), id="bool_width"),
        ],
    )
    def test_predict_shape_invalid_raises(self, bad_shape: tuple[int | float | bool, ...]) -> None:
        """predict() must raise ValueError for invalid shape values."""
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="shape"):
            model.predict(img, shape=bad_shape)  # type: ignore[arg-type]


class TestPredictPatchSize:
    """predict() patch_size resolution and validation tests."""

    def _make_model_with_config(self, patch_size: int, num_windows: int) -> _DummyRFDETR:
        """Return a _DummyRFDETR whose model_config carries patch_size and num_windows."""
        from types import SimpleNamespace

        model = _DummyRFDETR()
        model.model_config = SimpleNamespace(patch_size=patch_size, num_windows=num_windows)
        return model

    def test_predict_defaults_patch_size_from_model_config(self) -> None:
        """predict() reads patch_size from model_config when not provided by the caller."""
        # patch_size=16, num_windows=2 → block_size=32; shape=(64,64) is valid
        model = self._make_model_with_config(patch_size=16, num_windows=2)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        # Should not raise — 64 % 32 == 0
        model.predict(img, shape=(64, 64))

    def test_predict_shape_must_be_divisible_by_block_size(self) -> None:
        """predict() rejects shapes not divisible by patch_size * num_windows."""
        # patch_size=16, num_windows=2 → block_size=32; shape (48, 64) fails (48%32==16)
        model = self._make_model_with_config(patch_size=16, num_windows=2)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="divisible by 32"):
            model.predict(img, shape=(48, 64))

    @pytest.mark.parametrize("bad_patch_size", [0, -1, True, False])
    def test_predict_invalid_patch_size_raises(self, bad_patch_size: int) -> None:
        """predict() must raise ValueError when patch_size is not a positive integer."""
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            model.predict(img, patch_size=bad_patch_size)  # type: ignore[arg-type]

    def test_predict_patch_size_mismatch_raises(self) -> None:
        """predict() must raise ValueError when caller's patch_size != model_config.patch_size."""
        # model has patch_size=16; passing patch_size=14 should raise immediately
        model = self._make_model_with_config(patch_size=16, num_windows=1)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="does not match"):
            model.predict(img, shape=(16, 16), patch_size=14)

    def test_predict_explicit_patch_size_matching_config_succeeds(self) -> None:
        """predict(patch_size=X) must succeed when X matches model_config.patch_size."""
        # patch_size=16, num_windows=2 → block_size=32; shape=(64,64) is valid
        model = self._make_model_with_config(patch_size=16, num_windows=2)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        # Should not raise — patch_size matches config, 64 % 32 == 0
        model.predict(img, shape=(64, 64), patch_size=16)

    @pytest.mark.parametrize("bad_num_windows", [0, -1, True])
    def test_predict_invalid_num_windows_raises(self, bad_num_windows: int) -> None:
        """predict() must raise ValueError when model_config.num_windows is not a positive integer."""
        model = self._make_model_with_config(patch_size=14, num_windows=1)
        model.model_config.num_windows = bad_num_windows
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="num_windows must be a positive integer"):
            model.predict(img, shape=(14, 14))

    def test_predict_default_resolution_not_divisible_by_block_size_raises(self) -> None:
        """predict() with shape=None must raise ValueError when model.resolution % block_size != 0."""
        # patch_size=14, num_windows=1 → block_size=14; set resolution=25 (not divisible)
        model = self._make_model_with_config(patch_size=14, num_windows=1)
        model.model.resolution = 25
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="default resolution"):
            model.predict(img)


class TestPredictClassNameData:
    """Verify that ``predict()`` populates ``data["class_name"]`` in the returned Detections.

    class IDs are always 0-indexed (COCO category IDs are remapped during training);
    including the class name string in ``data`` lets callers read the class directly
    without a separate lookup into ``model.class_names``.
    """

    def _make_model_with_class_names(self, class_names: list[str], labels: list[int]) -> _DummyRFDETR:
        """Return a _DummyRFDETR whose inner model carries custom class_names and returns given labels."""
        model = _DummyRFDETR()
        model.model = _DummyModel(class_names=class_names, labels=labels)
        return model

    def test_class_name_key_present_in_detections_data(self) -> None:
        """predict() must include 'class_name' in detections.data when class_names is set."""
        model = self._make_model_with_class_names(["cat", "dog"], labels=[0])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert "class_name" in detections.data, "data['class_name'] must be present"

    def test_class_name_values_match_class_id(self) -> None:
        """class_name at each position must equal class_names[class_id]."""
        model = self._make_model_with_class_names(["cat", "dog", "bird"], labels=[0, 1, 2])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        np.testing.assert_array_equal(
            detections.data["class_name"],
            np.array(["cat", "dog", "bird"]),
            err_msg="class_name must match class_names[class_id] for each detection",
        )

    def test_class_name_with_remapped_coco_dataset(self) -> None:
        """Simulates a single-class COCO dataset where category_id=1 is remapped to label=0.

        After training with remap_category_ids=True, the model outputs class_id=0 for the
        first class.  class_name must correctly map 0 → the first class name.
        """
        # Single-class model: category_id=1 was remapped to label=0 during training.
        model = self._make_model_with_class_names(["myclass"], labels=[0])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.class_id[0] == 0, "class_id must be 0 (0-indexed)"
        assert detections.data["class_name"][0] == "myclass", (
            "class_name must be 'myclass' even though the original COCO category_id was 1"
        )

    def test_class_name_falls_back_to_coco_when_no_custom_names(self) -> None:
        """Without custom class_names, class_name maps class_id via COCO_CLASS_NAMES."""
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        # _DummyModel with no custom class_names; labels=[1] → COCO_CLASS_NAMES[1]
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert "class_name" in detections.data
        assert detections.data["class_name"][0] == COCO_CLASS_NAMES[1], (
            "class_name must fall back to COCO_CLASS_NAMES[class_id]"
        )

    def test_class_name_empty_array_when_no_detections(self) -> None:
        """When threshold filters all detections, data['class_name'] must be an empty array."""
        model = self._make_model_with_class_names(["cat"], labels=[0])
        img = PIL.Image.new("RGB", (28, 28))
        # threshold=1.1 filters out all detections (confidence=0.9 < 1.1)
        detections = model.predict(img, threshold=1.1)
        assert "class_name" in detections.data
        assert len(detections.data["class_name"]) == 0, "class_name must be empty when no detections pass threshold"
        assert detections.data["class_name"].dtype == object, (
            "class_name dtype must be object even when the array is empty (not float64)"
        )

    def test_class_name_out_of_bounds_class_id_returns_empty_string(self) -> None:
        """A class_id >= len(class_names) must map to an empty string (no IndexError)."""
        # class_names has 2 entries but labels includes out-of-bounds id=5
        model = self._make_model_with_class_names(["cat", "dog"], labels=[5])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.data["class_name"][0] == "", "Out-of-bounds class_id must produce empty string"

    def test_class_name_negative_class_id_returns_empty_string(self) -> None:
        """A negative class_id must map to an empty string (bounds check: 0 <= cid)."""
        model = self._make_model_with_class_names(["cat", "dog"], labels=[-1])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.data["class_name"][0] == "", "Negative class_id must produce empty string"

    def test_class_name_populated_for_each_image_in_batch(self) -> None:
        """class_name must be correctly populated for every Detections in a batch prediction."""
        model = self._make_model_with_class_names(["cat", "dog"], labels=[0, 1])
        img1 = PIL.Image.new("RGB", (28, 28))
        img2 = PIL.Image.new("RGB", (28, 28))
        results = model.predict([img1, img2])
        assert isinstance(results, list), "batch predict must return a list"
        assert len(results) == 2, "one Detections per input image"
        for idx, det in enumerate(results):
            assert "class_name" in det.data, f"image {idx}: class_name must be present"
            assert list(det.data["class_name"]) == ["cat", "dog"], (
                f"image {idx}: class_name must match class_names[class_id]"
            )
