# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for distributed utility helpers."""

from unittest.mock import patch

from rfdetr.utilities.distributed import all_gather


def test_all_gather_supports_cpu_without_tensor_truthiness_error() -> None:
    """all_gather should work on CPU-only setups and return gathered objects."""

    def _fake_all_gather(output_tensors, input_tensor) -> None:
        for out in output_tensors:
            out.copy_(input_tensor)

    with (
        patch("rfdetr.utilities.distributed.get_world_size", return_value=2),
        patch("rfdetr.utilities.distributed.dist.all_gather", side_effect=_fake_all_gather),
        patch("rfdetr.utilities.distributed.torch.cuda.is_available", return_value=False),
    ):
        result = all_gather({"value": 7})

    assert result == [{"value": 7}, {"value": 7}]
