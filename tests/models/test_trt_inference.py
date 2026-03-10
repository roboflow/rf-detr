# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from unittest.mock import Mock

import torch
from PIL import Image

from rfdetr.deploy.benchmark import TRTInference, infer_transforms


class TestTRTInference:
    def test_synchronize_sync_mode_does_not_require_stream(self, monkeypatch) -> None:
        """`synchronize()` should not access stream in sync mode."""
        inference = TRTInference.__new__(TRTInference)
        inference.sync_mode = True

        mock_is_available = Mock(return_value=True)
        mock_cuda_sync = Mock()
        monkeypatch.setattr("torch.cuda.is_available", mock_is_available)
        monkeypatch.setattr("torch.cuda.synchronize", mock_cuda_sync)

        inference.synchronize()

        mock_is_available.assert_called_once()
        mock_cuda_sync.assert_called_once()

    def test_synchronize_async_mode_uses_stream_sync(self, monkeypatch) -> None:
        """`synchronize()` should use stream synchronization in async mode."""
        inference = TRTInference.__new__(TRTInference)
        inference.sync_mode = False
        inference.stream = Mock()

        mock_cuda_sync = Mock()
        monkeypatch.setattr("torch.cuda.synchronize", mock_cuda_sync)

        inference.synchronize()

        inference.stream.synchronize.assert_called_once()
        mock_cuda_sync.assert_not_called()

    def test_infer_transforms_accepts_none_target(self) -> None:
        """Benchmark inference preprocessing should support image-only input."""
        image = Image.new("RGB", (320, 240))

        image_tensor, target = infer_transforms()(image, None)

        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 640, 640)
        assert image_tensor.dtype == torch.float32
        assert target is None
