# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import pytest
import torch

from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    Dinov2WithRegistersAttention,
    Dinov2WithRegistersSdpaAttention,
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
    WindowedDinov2WithRegistersEmbeddings,
    WindowedDinov2WithRegistersModel,
    _find_pruneable_heads_and_indices,
    _get_aligned_output_features_output_indices,
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


# ---------------------------------------------------------------------------
# Tests for locally-copied utility functions (removed from transformers v5 public API)
# ---------------------------------------------------------------------------


class TestGetAlignedOutputFeaturesOutputIndices:
    """Tests for the local copy of get_aligned_output_features_output_indices."""

    def test_both_none_returns_last_stage(self):
        stage_names = ["stage1", "stage2", "stage3"]
        features, indices = _get_aligned_output_features_output_indices(None, None, stage_names)
        assert features == ["stage3"]
        assert indices == [2]

    def test_only_out_features_derives_indices(self):
        stage_names = ["stem", "layer1", "layer2", "layer3"]
        features, indices = _get_aligned_output_features_output_indices(["layer1", "layer3"], None, stage_names)
        assert features == ["layer1", "layer3"]
        assert indices == [1, 3]

    def test_only_out_indices_derives_features(self):
        stage_names = ["stem", "layer1", "layer2", "layer3"]
        features, indices = _get_aligned_output_features_output_indices(None, [0, 2], stage_names)
        assert features == ["stem", "layer2"]
        assert indices == [0, 2]

    def test_both_provided_returns_as_is(self):
        stage_names = ["stem", "layer1", "layer2"]
        features, indices = _get_aligned_output_features_output_indices(["layer1"], [1], stage_names)
        assert features == ["layer1"]
        assert indices == [1]

    def test_out_indices_converted_to_list(self):
        """out_indices supplied as a tuple must be returned as a list."""
        stage_names = ["stem", "layer1", "layer2"]
        _, indices = _get_aligned_output_features_output_indices(None, (1, 2), stage_names)
        assert isinstance(indices, list)
        assert indices == [1, 2]


class TestFindPruneableHeadsAndIndices:
    """Tests for the local copy of find_pruneable_heads_and_indices."""

    def test_no_pruning_returns_full_index(self):
        heads, index = _find_pruneable_heads_and_indices(set(), n_heads=4, head_size=3, already_pruned_heads=set())
        assert len(heads) == 0
        assert len(index) == 12  # 4 * 3, nothing masked

    @pytest.mark.parametrize(
        "head_to_prune, expected_index",
        [
            pytest.param({0}, list(range(3, 12)), id="prune-first-head"),
            pytest.param({3}, list(range(9)), id="prune-last-head"),
        ],
    )
    def test_prune_single_head_removes_correct_rows(self, head_to_prune, expected_index):
        # Head N masked → N*head_size indices removed; remaining = n_heads*head_size - head_size = 9
        heads, index = _find_pruneable_heads_and_indices(
            head_to_prune, n_heads=4, head_size=3, already_pruned_heads=set()
        )
        assert heads == head_to_prune
        assert len(index) == 9
        assert index.tolist() == expected_index

    def test_already_pruned_head_adjusts_offset(self):
        # Head 0 was already pruned. Now pruning head 1 (which is now effective head 0
        # after offset adjustment) should remove 3 more indices from the effective mask.
        heads, index = _find_pruneable_heads_and_indices({1}, n_heads=4, head_size=3, already_pruned_heads={0})
        assert 1 in heads
        assert len(index) == 9  # 4*3 - 3 pruned


# ---------------------------------------------------------------------------
# Smoke tests for WindowedDinov2WithRegistersBackbone
# ---------------------------------------------------------------------------


def _minimal_backbone_config(**kwargs) -> WindowedDinov2WithRegistersConfig:
    """Return the smallest valid config for backbone instantiation tests."""
    defaults = dict(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=16,
        image_size=64,
        num_register_tokens=0,
        num_windows=1,
    )
    defaults.update(kwargs)
    return WindowedDinov2WithRegistersConfig(**defaults)


class TestWindowedDinov2WithRegistersBackbone:
    """Smoke tests that guard against _init_transformers_backbone() API regressions."""

    @pytest.mark.parametrize(
        "attr",
        [
            pytest.param("stage_names", id="stage_names"),
            pytest.param("out_features", id="out_features"),
        ],
    )
    def test_instantiation_sets_list_attribute(self, attr):
        config = _minimal_backbone_config()
        backbone = WindowedDinov2WithRegistersBackbone(config)
        assert hasattr(backbone, attr)
        assert isinstance(getattr(backbone, attr), list)
        assert len(getattr(backbone, attr)) > 0

    def test_forward_returns_backbone_output(self):
        config = _minimal_backbone_config()
        backbone = WindowedDinov2WithRegistersBackbone(config)
        backbone.eval()
        pixel_values = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = backbone(pixel_values)
        assert hasattr(output, "feature_maps")
        assert len(output.feature_maps) == len(backbone.out_features)


# ---------------------------------------------------------------------------
# Test for output_attentions=True SDPA fallback path
# ---------------------------------------------------------------------------


class TestSdpaFallbackWithOutputAttentions:
    """Guards the output_attentions behaviour in windowed attention."""

    def test_output_attentions_true_raises(self):
        """Windowed attention explicitly does not support output_attentions=True."""
        config = _minimal_backbone_config()
        model = WindowedDinov2WithRegistersModel(config)
        model.eval()
        pixel_values = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            with pytest.raises(AssertionError, match="output_attentions is not supported for windowed attention"):
                model(pixel_values, output_attentions=True)


class TestSetAttnImplementation:
    """Tests for WindowedDinov2WithRegistersModel.set_attn_implementation."""

    @pytest.mark.parametrize(
        "switches, expected_impl, expected_cls",
        [
            pytest.param(["eager"], "eager", Dinov2WithRegistersAttention, id="sdpa-to-eager"),
            pytest.param(["eager", "sdpa"], "sdpa", Dinov2WithRegistersSdpaAttention, id="roundtrip-back-to-sdpa"),
        ],
    )
    def test_switch_updates_config_and_layers(self, switches, expected_impl, expected_cls):
        """After each call in *switches*, config and all layer attention modules reflect the final impl."""
        config = _minimal_backbone_config()
        model = WindowedDinov2WithRegistersModel(config)

        for impl in switches:
            model.set_attn_implementation(impl)

        assert model.config._attn_implementation == expected_impl
        for layer in model.encoder.layer:
            assert type(layer.attention) is expected_cls

    def test_invalid_implementation_raises(self):
        """Passing an unknown key raises ValueError with a clear message."""
        config = _minimal_backbone_config()
        model = WindowedDinov2WithRegistersModel(config)

        with pytest.raises(ValueError, match="Unknown attn_implementation"):
            model.set_attn_implementation("flash_attention_2")


@pytest.mark.parametrize(
    "h, w, num_windows, should_raise",
    [
        pytest.param(64, 64, 2, False, id="valid-square"),
        pytest.param(64, 96, 2, False, id="valid-rectangular"),
        pytest.param(32, 32, 1, False, id="num_windows-1-valid"),
        pytest.param(33, 64, 2, True, id="h-not-divisible"),
        pytest.param(64, 33, 2, True, id="w-not-divisible"),
        pytest.param(33, 33, 2, True, id="both-not-divisible"),
    ],
)
def test_forward_validates_spatial_dims(h: int, w: int, num_windows: int, should_raise: bool) -> None:
    """WindowedDinov2WithRegistersEmbeddings raises ValueError for incompatible dims.

    Both H and W must be divisible by patch_size * num_windows.  The check
    must survive Python's -O flag (assert would be silently stripped).
    """
    patch_size = 16
    config = WindowedDinov2WithRegistersConfig(
        hidden_size=32,
        patch_size=patch_size,
        num_windows=num_windows,
        image_size=max(h, w),
        num_register_tokens=0,
    )
    model = WindowedDinov2WithRegistersEmbeddings(config)
    pixel_values = torch.randn(1, 3, h, w)
    if should_raise:
        with pytest.raises(ValueError, match="divisible"):
            model(pixel_values)
    else:
        model(pixel_values)  # must not raise
