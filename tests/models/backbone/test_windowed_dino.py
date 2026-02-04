# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch

from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersConfig,
    WindowedDinov2WithRegistersEmbeddings,
)


def test_window_partition_forward_rectangular_preserves_shapes():
    """
    Regression test for WindowedDinov2WithRegistersEmbeddings.forward with rectangular input.
    Ensures window partitioning logic correctly handles H != W.
    """
    # Params: H_patches=6, W_patches=4, num_windows=2 -> 3x2 patches per window
    batch_size, hidden_size, patch_size, num_windows = 1, 64, 16, 2
    hp, wp, nr = 6, 4, 4
    h, w = hp * patch_size, wp * patch_size

    config = WindowedDinov2WithRegistersConfig(
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_windows=num_windows,
        image_size=h,  # square image_size for positional embeddings
        num_register_tokens=nr,
    )
    model = WindowedDinov2WithRegistersEmbeddings(config)

    # Input is rectangular
    pixel_values = torch.randn(batch_size, 3, h, w)
    result = model(pixel_values)

    expected_batch = batch_size * (num_windows**2)
    expected_seq_len = 1 + nr + (hp // num_windows) * (wp // num_windows)

    assert result.shape == (expected_batch, expected_seq_len, hidden_size)
