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
        self.resolution = 32
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
        assert resize_size == [32, 32], f"Expected resize to model resolution (32, 32), got {resize_size}"

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
