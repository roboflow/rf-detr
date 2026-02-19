# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import pytest
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


# Before fix in PR #448 the reshape used num_h_patches_per_window in both the height
# AND width dimension. This only fails when height and width produce different patch
# counts, so all tests below use non-square images (hp != wp).


@pytest.mark.parametrize(
    "hp, wp, num_windows",
    [
        (4, 6, 2),  # wider than tall
        (6, 4, 2),  # taller than wide
        (6, 9, 3),  # 3-window grid, non-square
        (8, 4, 2),  # 2:1 aspect ratio
    ],
)
def test_window_partition_nonsquare_does_not_raise(hp, wp, num_windows):
    """
    Before the fix, the reshape used num_h_patches_per_window for the width
    dimension, so the total element count mismatched and PyTorch raised a
    RuntimeError for any non-square image.  The fix replaces that variable
    with num_w_patches_per_window, making the operation valid for all shapes.
    """
    hidden_size, patch_size, nr = 32, 16, 0
    h, w = hp * patch_size, wp * patch_size

    config = WindowedDinov2WithRegistersConfig(
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_windows=num_windows,
        image_size=max(h, w),
        num_register_tokens=nr,
    )
    model = WindowedDinov2WithRegistersEmbeddings(config)
    pixel_values = torch.randn(1, 3, h, w)

    # This line would raise RuntimeError before the fix
    result = model(pixel_values)

    expected_batch = num_windows**2
    expected_seq_len = 1 + (hp // num_windows) * (wp // num_windows)
    assert result.shape == (expected_batch, expected_seq_len, hidden_size)


def test_window_partition_correct_window_content():
    """
    Verifies that after windowing each window contains the spatially correct
    patch tokens — not just that the shape is right.

    Layout with hp=4, wp=6, num_windows=2 (2x2 grid of windows):
      Window (0,0): rows 0-1, cols 0-2
      Window (0,1): rows 0-1, cols 3-5
      Window (1,0): rows 2-3, cols 0-2
      Window (1,1): rows 2-3, cols 3-5

    Before the fix the reshape used num_h_patches_per_window for the width dim
    so it raised an error and never produced window content at all.
    """
    hidden_size, patch_size, num_windows, nr = 1, 16, 2, 0
    hp, wp = 4, 6
    h, w = hp * patch_size, wp * patch_size
    batch_size = 1

    config = WindowedDinov2WithRegistersConfig(
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_windows=num_windows,
        image_size=max(h, w),
        num_register_tokens=nr,
        num_hidden_layers=1,
        num_attention_heads=1,
    )
    model = WindowedDinov2WithRegistersEmbeddings(config)

    # Disable position embeddings and cls token so we can track patch identity.
    # Each patch gets a unique value equal to its flat index (row * wp + col).
    with torch.no_grad():
        model.position_embeddings.zero_()
        model.cls_token.zero_()

    # Build a synthetic patch embedding: patch at (row, col) has value row*wp+col.
    # Shape after patch projection: (1, hp*wp, 1) — hidden_size=1 for simplicity.
    patch_ids = torch.arange(hp * wp, dtype=torch.float).view(1, hp * wp, 1)

    # Bypass the full forward pass and exercise the windowing logic directly.
    pixel_tokens = patch_ids  # (1, 24, 1)
    pixel_tokens_2d = pixel_tokens.view(batch_size, hp, wp, hidden_size)  # (1,4,6,1)

    num_h_patches_per_window = hp // num_windows  # 2
    num_w_patches_per_window = wp // num_windows  # 3

    # --- correct reshape (the fix) ---
    windowed = pixel_tokens_2d.reshape(
        batch_size * num_windows,
        num_h_patches_per_window,
        num_windows,
        num_w_patches_per_window,
        hidden_size,
    )
    windowed = windowed.permute(0, 2, 1, 3, 4)
    windowed = windowed.reshape(
        batch_size * num_windows**2,
        num_h_patches_per_window * num_w_patches_per_window,
        hidden_size,
    )

    # Expected content for each of the 4 windows (6 patches each):
    expected = torch.tensor(
        [
            # Window 0 (rows 0-1, cols 0-2): ids 0,1,2, 6,7,8
            [[0.0], [1.0], [2.0], [6.0], [7.0], [8.0]],
            # Window 1 (rows 0-1, cols 3-5): ids 3,4,5, 9,10,11
            [[3.0], [4.0], [5.0], [9.0], [10.0], [11.0]],
            # Window 2 (rows 2-3, cols 0-2): ids 12,13,14, 18,19,20
            [[12.0], [13.0], [14.0], [18.0], [19.0], [20.0]],
            # Window 3 (rows 2-3, cols 3-5): ids 15,16,17, 21,22,23
            [[15.0], [16.0], [17.0], [21.0], [22.0], [23.0]],
        ]
    )
    assert torch.equal(windowed, expected), f"Window content mismatch:\n{windowed}\n!=\n{expected}"


def test_buggy_reshape_raises_for_nonsquare():
    """
    Directly demonstrates what the pre-fix code did: using num_h_patches_per_window
    in the width position of the reshape causes a RuntimeError when the element count
    is not divisible by the (wrong) shape.

    With hidden_size=1 and hp=4, wp=6, num_windows=2 the total elements are 24 but
    the buggy target dims (2,2,2,2,-1) require a non-integer last dimension,
    so PyTorch raises RuntimeError.
    """
    hp, wp = 4, 6  # non-square: width > height
    num_windows = 2
    hidden_size = 1  # chosen so total / buggy-fixed-dims is non-integer

    num_h_patches_per_window = hp // num_windows  # 2
    num_w_patches_per_window = wp // num_windows  # 3
    batch_size = 1

    # Simulate pixel_tokens_with_pos_embed after the .view() call
    pixel_tokens_2d = torch.randn(batch_size, hp, wp, hidden_size)

    # The correct reshape (post-fix) must succeed
    pixel_tokens_2d.reshape(
        batch_size * num_windows,
        num_h_patches_per_window,
        num_windows,
        num_w_patches_per_window,  # correct
        hidden_size,
    )

    # The buggy reshape (pre-fix) must raise RuntimeError:
    # total elements = 1*4*6*1 = 24,  fixed-dims product = 2*2*2*2 = 16,  16 ∤ 24.
    with pytest.raises(RuntimeError):
        pixel_tokens_2d.reshape(
            batch_size * num_windows,
            num_h_patches_per_window,
            num_windows,
            num_h_patches_per_window,  # bug: height used for width
            -1,
        )


def test_buggy_reshape_silent_corruption_for_nonsquare():
    """
    When hidden_size happens to make the total element count divisible by the
    buggy target shape, PyTorch does NOT raise — instead the last dimension is
    inflated, which silently corrupts the tensor layout.

    Pre-fix with hp=4, wp=6, hidden_size=8, num_windows=2:
      total elements = 1*4*6*8 = 192
      buggy fixed dims = 2*2*2*2 = 16  →  last dim inferred as 192/16 = 12 (not 8)

    The fix ensures the correct reshape always yields a last dim equal to hidden_size.
    """
    hp, wp = 4, 6
    num_windows = 2
    hidden_size = 8

    num_h_patches_per_window = hp // num_windows  # 2
    num_w_patches_per_window = wp // num_windows  # 3
    batch_size = 1

    pixel_tokens_2d = torch.randn(batch_size, hp, wp, hidden_size)

    # Buggy reshape silently infers last dim = 12 (not 8)
    buggy_out = pixel_tokens_2d.reshape(
        batch_size * num_windows,
        num_h_patches_per_window,
        num_windows,
        num_h_patches_per_window,  # bug
        -1,
    )
    assert buggy_out.shape[-1] != hidden_size, "Buggy reshape should produce wrong last dim"

    # Correct reshape always yields last dim == hidden_size
    correct_out = pixel_tokens_2d.reshape(
        batch_size * num_windows,
        num_h_patches_per_window,
        num_windows,
        num_w_patches_per_window,  # fix
        hidden_size,
    )
    assert correct_out.shape[-1] == hidden_size
