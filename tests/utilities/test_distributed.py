# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for distributed utility helpers."""

from unittest.mock import patch

import torch

from rfdetr.utilities.distributed import all_gather


def _fake_all_gather(output_tensors, input_tensor) -> None:
    """Stand-in for dist.all_gather: broadcast the local tensor to every output slot."""
    for out in output_tensors:
        out.copy_(input_tensor)


def test_all_gather_supports_cpu_without_tensor_truthiness_error() -> None:
    """all_gather derives a cpu device and works when the backend is not nccl (e.g. gloo)."""
    with (
        patch("rfdetr.utilities.distributed.get_world_size", return_value=2),
        patch("rfdetr.utilities.distributed.dist.all_gather", side_effect=_fake_all_gather),
        patch("rfdetr.utilities.distributed.is_dist_avail_and_initialized", return_value=True),
        patch("rfdetr.utilities.distributed.dist.get_backend", return_value="gloo"),
    ):
        result = all_gather({"value": 7})

    assert result == [{"value": 7}, {"value": 7}]


def test_all_gather_explicit_device_bypasses_backend_probe() -> None:
    """A caller-supplied device (e.g. an XLA device) is used as-is; no backend derivation runs."""
    with (
        patch("rfdetr.utilities.distributed.get_world_size", return_value=2),
        patch("rfdetr.utilities.distributed.dist.all_gather", side_effect=_fake_all_gather),
        patch("rfdetr.utilities.distributed.dist.get_backend") as mock_get_backend,
    ):
        result = all_gather({"value": 7}, device=torch.device("cpu"))

    mock_get_backend.assert_not_called()
    assert result == [{"value": 7}, {"value": 7}]
