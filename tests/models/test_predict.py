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
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = torch.nn.Identity()

    def postprocess(self, predictions: Any, target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            results.append(
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
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
