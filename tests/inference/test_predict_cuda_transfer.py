# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression coverage for the coalesced GPU->CPU output transfer in ``predict()``.

The existing ``predict()`` tests (``test_predict.py``) use ``_DummyModel``, whose
``postprocess()`` builds result tensors on CPU (``helpers.py``) — that path never
exercises the CUDA branch added to batch the output transfers, so it cannot catch an
incomplete device->host read or a synchronization call targeting the wrong device.
These tests require CUDA and are skipped otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from rfdetr.detr import RFDETR

from .helpers import _BaseFakeRFDETR

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _CudaDummyModel:
    """Like ``helpers._DummyModel``, but ``postprocess()`` returns CUDA tensors."""

    def __init__(self, labels: list[int] | None = None) -> None:
        self.device = torch.device("cuda:0")
        self.resolution = 28
        self.model = torch.nn.Identity().to(self.device)
        self.class_names = None
        self._labels = labels if labels is not None else [1, 2, 3]

    def postprocess(
        self,
        predictions: Any,
        target_sizes: torch.Tensor,
        score_threshold: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            n = len(self._labels)
            results.append(
                {
                    "scores": torch.full((n,), 0.9, device=self.device),
                    "labels": torch.tensor(self._labels, device=self.device),
                    "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]] * n, device=self.device),
                }
            )
        return results


class _CudaDummyRFDETR(RFDETR):
    """Weight-free RFDETR whose model lives on CUDA, for exercising the async path."""

    def maybe_download_pretrain_weights(self) -> None:
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(num_channels=3)

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _CudaDummyModel:
        return _CudaDummyModel()


class TestPredictCudaTransfer:
    """Verifies the batched CUDA->CPU transfer is both correct and synchronized right."""

    def test_output_values_are_complete_after_async_transfer(self) -> None:
        """Every field must be fully populated — not read mid-copy — after the batched transfer."""
        img = torch.rand(3, 28, 28, device="cuda:0")
        model = _CudaDummyRFDETR(pretrain_weights=None)
        detections = model.predict(img, threshold=0.5)
        assert len(detections) == 3
        assert detections.confidence == pytest.approx([0.9, 0.9, 0.9])
        assert detections.class_id.tolist() == [1, 2, 3]
        assert np.allclose(detections.xyxy, [[0.1, 0.2, 0.3, 0.4]] * 3, atol=1e-6)

    def test_synchronize_targets_the_result_tensors_own_device(self) -> None:
        """`torch.cuda.synchronize` must be called with the device the tensors actually live
        on, not left to default to `torch.cuda.current_device()` — the two can differ on a
        multi-GPU host."""
        img = torch.rand(3, 28, 28, device="cuda:0")
        model = _CudaDummyRFDETR(pretrain_weights=None)
        with patch("torch.cuda.synchronize", wraps=torch.cuda.synchronize) as spy:
            model.predict(img, threshold=0.5)
        assert spy.call_count >= 1
        called_devices = [torch.device(c.args[0]) if c.args else None for c in spy.call_args_list]
        assert torch.device("cuda:0") in called_devices, (
            f"expected a synchronize(cuda:0) call, got: {called_devices}"
        )
