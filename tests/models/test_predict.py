# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import socket
from types import SimpleNamespace
from typing import Any

import pytest
import supervision as sv
import torch

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


def test_predict_accepts_image_url() -> None:
    if not _is_online(_HTTP_HOST, _HTTP_PORT):
        pytest.skip("Offline environment, skipping HTTP predict URL test.")
    model = _DummyRFDETR()
    detections = model.predict(_HTTP_IMAGE_URL)
    assert isinstance(detections, sv.Detections)
    assert detections.xyxy.shape == (1, 4)
