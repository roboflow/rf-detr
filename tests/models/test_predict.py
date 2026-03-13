# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import socket
from types import SimpleNamespace
from typing import Any

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

    Quality is not assessed here — see ``tests/benchmarks/test_coco_inference.py``.
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
