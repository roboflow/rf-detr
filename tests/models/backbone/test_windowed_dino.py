import types
from types import SimpleNamespace

import pytest
import torch


def test_window_partition_forward_rectangular_preserves_shapes(monkeypatch):
    """
    Regression test that exercises WindowedDinov2WithRegistersEmbeddings.forward
    with a rectangular input (H != W) to hit the window partitioning logic.

    The test stubs Dinov2WithRegistersPatchEmbeddings but:
    - returns patch embeddings whose last dimension equals config.hidden_size
      (avoids spurious hidden-dim mismatch errors)
    - provides num_patches used to initialize position_embeddings

    The test asserts the final embeddings tensor has the expected shape:
    (batch_size * num_windows**2, 1 + num_h_patches_per_window * num_w_patches_per_window, hidden_size)

    On implementations with the height/width mix-up bug this will raise a RuntimeError
    (or produce an unexpected shape) and the test will fail.
    """

    # Parameters
    batch_size = 1
    in_channels = 3
    hidden_size = 64
    patch_size = 16
    num_windows = 2  # exercise the windowing path, must be > 1

    # Rectangular pixel dims that produce different patch-grid sizes (H_patches != W_patches)
    H_pixels = 96   # 96 // 16 == 6 patches
    W_pixels = 64   # 64 // 16 == 4 patches

    num_h_patches = H_pixels // patch_size
    num_w_patches = W_pixels // patch_size
    assert num_h_patches != num_w_patches, "Test requires non-square patch grid"

    # Ensure both patch dims are divisible by num_windows for clean per-window sizes
    assert num_h_patches % num_windows == 0 and num_w_patches % num_windows == 0, (
        "Choose H_pixels and W_pixels such that H_patches and W_patches are divisible by num_windows"
    )

    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_register_tokens=0,
        patch_size=patch_size,
        hidden_dropout_prob=0.0,
        num_windows=num_windows,
    )

    # Import module under test from the repository codebase
    import rfdetr.models.backbone.dinov2_with_windowed_attn as emb_mod

    # Stub: pretend the pretrained positional grid had a square number of patches (e.g., 4x4 = 16)
    fake_num_patches_pretrained = 16

    # Build a stub that:
    # - exposes num_patches for __init__
    # - exposes projection.weight.dtype used by the real class
    # - when called returns embeddings with last dim == config.hidden_size
    def _stub_patch_embeddings_factory(cfg):
        class StubPatchEmb:
            def __init__(self):
                self.num_patches = fake_num_patches_pretrained
                # used as target dtype in the real code
                self.projection = types.SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32))

            def __call__(self, pixel_values):
                B, C, H, W = pixel_values.shape
                nh = H // cfg.patch_size
                nw = W // cfg.patch_size
                # Return tensor shaped (B, nh * nw, hidden_size) matching config.hidden_size
                return torch.randn(B, nh * nw, cfg.hidden_size, dtype=torch.float32)

        return StubPatchEmb()

    # Monkeypatch heavy dependency to our stub
    monkeypatch.setattr(
        emb_mod, "Dinov2WithRegistersPatchEmbeddings", _stub_patch_embeddings_factory
    )

    # Instantiate the real embeddings module (uses stub during __init__)
    emb_module = emb_mod.WindowedDinov2WithRegistersEmbeddings(config)

    # Build rectangular pixel input
    pixel_values = torch.randn(batch_size, in_channels, H_pixels, W_pixels, dtype=torch.float32)

    # Expected values after windowing:
    num_h_patches_per_window = num_h_patches // num_windows
    num_w_patches_per_window = num_w_patches // num_windows
    expected_batch = batch_size * (num_windows ** 2)
    expected_seq_len = 1 + (num_h_patches_per_window * num_w_patches_per_window)

    # Call forward. If the develop branch still has the height/width mix-up bug,
    # this call is expected to raise (shape mismatch during reshape/permute/reshape or concatenation),
    # causing the test to fail. If the branch has the fix, this should succeed and assertions below pass.
    result = emb_module.forward(pixel_values)

    assert isinstance(result, torch.Tensor), "forward should return a tensor of embeddings"

    assert result.shape == (
        expected_batch,
        expected_seq_len,
        hidden_size,
    ), (
        f"Unexpected embedding shape {result.shape}, expected "
        f"({expected_batch}, {expected_seq_len}, {hidden_size})"
    )
