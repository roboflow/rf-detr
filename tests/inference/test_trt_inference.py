# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from rfdetr.export.benchmark import TRTInference, infer_transforms


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


class TestBenchmarkShapeParameterization:
    """Benchmark preprocessing/postprocessing read input size and query count instead of hardcoding 640/300."""

    def test_infer_transforms_uses_requested_size(self) -> None:
        """infer_transforms resizes to the caller-supplied (height, width)."""
        image = Image.new("RGB", (320, 240))

        image_tensor, _ = infer_transforms((512, 384))(image, None)

        assert image_tensor.shape == (3, 512, 384)

    def test_infer_transforms_defaults_to_640(self) -> None:
        """The default input size stays 640x640 for callers that do not pass a size."""
        image = Image.new("RGB", (320, 240))

        image_tensor, _ = infer_transforms()(image, None)

        assert image_tensor.shape == (3, 640, 640)

    def test_static_dim_returns_concrete_int(self) -> None:
        """A concrete positive dimension is returned unchanged."""
        from rfdetr.export.benchmark import _static_dim

        assert _static_dim(384, 640) == 384

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("height", id="dynamic-string"),
            pytest.param(None, id="none"),
            pytest.param(-1, id="negative"),
        ],
    )
    def test_static_dim_falls_back_for_dynamic_axis(self, value) -> None:
        """Dynamic/unknown axes fall back to the provided default."""
        from rfdetr.export.benchmark import _static_dim

        assert _static_dim(value, 640) == 640

    def test_post_process_respects_num_queries(self) -> None:
        """post_process selects exactly num_queries detections per image."""
        from rfdetr.export.benchmark import post_process

        num_queries = 5
        outputs = {
            "labels": torch.rand(1, 20, 3),
            "dets": torch.rand(1, 20, 4),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = post_process(outputs, target_sizes, num_queries=num_queries)

        assert results[0]["scores"].shape == (num_queries,)
