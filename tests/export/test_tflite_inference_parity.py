# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Preprocessing parity tests: ``_run_inference`` must produce essentially the same input tensor as ``RFDETR.predict``
for the same source image, otherwise the TFLite-exported model is fed inputs the PyTorch graph never saw and detections
drift.

History: an earlier version of ``_run_inference`` called ``PIL.Image.resize`` without a filter argument, picking up
PIL's default (BICUBIC since Pillow 9.1.0). PyTorch's predict() path uses torchvision ``F.resize`` (BILINEAR). The
mismatch caused IoU drift up to 0.36 on detail-rich images and a 2-class-mismatch FP16 disaster on the ``dog`` test
image. This test exists to keep ``_preprocess_image`` locked to BILINEAR -- any regression that re-introduces BICUBIC or
otherwise shifts the resize filter will surface here.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image as PILImage

from rfdetr.export._tflite.inference import _bilinear_resize_half_pixel, _preprocess_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Bound for max abs diff in normalised space between the PyTorch and TFLite preprocessing pipelines.
# With torchvision available (which is always the case in this repo's CI -- it's a hard rfdetr
# dependency) _preprocess_image runs the same torchvision call PyTorch's predict() uses, so the
# tensors are bit-exact. The 0.05 bound is generous so the torch-free fallback path (which uses
# PIL.BILINEAR and shows ~0.016 max diff) also passes; the BICUBIC regression would push max diff
# to ~0.5, well above the bound.
MAX_ABS_DIFF_BOUND = 0.05
# When torchvision is available the inference path matches torchvision-resize byte-for-byte, so
# the diff is effectively zero modulo floating-point noise.
BIT_EXACT_BOUND = 1e-5


def _pytorch_preprocess(pil_img: PILImage.Image, hw: tuple[int, int]) -> np.ndarray:
    """Mirror of the PyTorch predict() preprocessing: to_tensor -> resize -> normalize."""
    img = F.to_tensor(pil_img)
    img = F.resize(img, list(hw))
    img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img.unsqueeze(0).numpy()


def _tflite_preprocess_to_nchw(pil_img: PILImage.Image, hw: tuple[int, int]) -> np.ndarray:
    """Call ``_preprocess_image`` and convert NHWC -> NCHW for apples-to-apples comparison."""
    nhwc = _preprocess_image(pil_img, hw, channels=3)
    return nhwc.transpose(0, 3, 1, 2)


def _make_synthetic_rgb(seed: int, size: tuple[int, int]) -> PILImage.Image:
    """Deterministic synthetic RGB image with structure (not pure noise) so resize filtering matters."""
    rng = np.random.default_rng(seed)
    height, width = size
    base = rng.integers(0, 256, size=(height // 8, width // 8, 3), dtype=np.uint8)
    pil_small = PILImage.fromarray(base, mode="RGB")
    return pil_small.resize((width, height), getattr(PILImage, "Resampling", PILImage).NEAREST)


class TestPreprocessingParity:
    """``_preprocess_image`` must match PyTorch's predict() preprocessing within MAX_ABS_DIFF_BOUND.

    Three shapes cover the common downscale ratios produced by RFDETR exports:
      - 1280x720 -> 384x384 (nano default): heavy downscale, the case that surfaced the BICUBIC bug
      - 800x600  -> 384x384: moderate downscale, mixed aspect ratio
      - 384x384  -> 384x384: identity resize -- only normalisation differs (rounding noise only)
    """

    @pytest.mark.parametrize(
        ("src_size", "tgt_size", "seed"),
        [
            pytest.param((1280, 720), (384, 384), 0, id="1280x720_to_384x384"),
            pytest.param((800, 600), (384, 384), 1, id="800x600_to_384x384"),
            pytest.param((384, 384), (384, 384), 2, id="identity_384x384"),
        ],
    )
    def test_matches_pytorch_predict_preprocessing(
        self, src_size: tuple[int, int], tgt_size: tuple[int, int], seed: int
    ) -> None:
        pil = _make_synthetic_rgb(seed, (src_size[1], src_size[0]))  # _make takes (H, W)
        pt = _pytorch_preprocess(pil, tgt_size)
        tf = _tflite_preprocess_to_nchw(pil, tgt_size)

        assert pt.shape == tf.shape, f"shape mismatch: PT {pt.shape} vs TF {tf.shape}"
        max_diff = float(np.abs(pt - tf).max())
        # torchvision is a hard rfdetr dependency, so in this test environment _preprocess_image
        # uses the torchvision path and matches PyTorch byte-for-byte. The torch-free fallback is
        # exercised separately by test_torch_free_fallback_still_close.
        assert max_diff < BIT_EXACT_BOUND, (
            f"PyTorch vs TFLite preprocessing diverged: max|diff|={max_diff:.6f} exceeds "
            f"{BIT_EXACT_BOUND}. With torchvision available, _preprocess_image should be using "
            f"torchvision.transforms.functional.resize and the diff should be effectively zero. "
            f"If this fires, check that the torch/torchvision import path inside _preprocess_image "
            f"hasn't been broken."
        )

    def test_grayscale_channel_handling(self) -> None:
        """Grayscale (channels=1) path must produce shape (1, H, W, 1)."""
        rng = np.random.default_rng(3)
        height, width = 256, 256
        pil = PILImage.fromarray(rng.integers(0, 256, size=(height, width), dtype=np.uint8), mode="L")
        tf = _preprocess_image(pil, (128, 128), channels=1)
        assert tf.shape == (1, 128, 128, 1), f"unexpected shape: {tf.shape}"
        assert tf.dtype == np.float32

    def test_returns_nhwc_float32(self) -> None:
        """``_preprocess_image`` returns NHWC float32 with a leading batch dim."""
        pil = _make_synthetic_rgb(seed=7, size=(64, 64))
        tf = _preprocess_image(pil, (32, 32), channels=3)
        assert tf.shape == (1, 32, 32, 3)
        assert tf.dtype == np.float32

    def test_normalisation_uses_imagenet_stats(self) -> None:
        """A mid-gray (128) image should land near zero on all channels after normalisation."""
        gray = np.full((64, 64, 3), 128, dtype=np.uint8)
        pil = PILImage.fromarray(gray, mode="RGB")
        tf = _preprocess_image(pil, (32, 32), channels=3)
        # 128/255 ~= 0.502; expected normalised values per channel:
        #   (0.502 - 0.485) / 0.229 ~=  0.074
        #   (0.502 - 0.456) / 0.224 ~=  0.205
        #   (0.502 - 0.406) / 0.225 ~=  0.426
        expected = np.array([(128 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c] for c in range(3)], dtype=np.float32)
        per_channel_mean = tf[0].mean(axis=(0, 1))
        np.testing.assert_allclose(per_channel_mean, expected, atol=1e-3)

    def test_torch_free_fallback_still_close(self) -> None:
        """Simulate the torch-free environment by masking torch imports; assert the PIL fallback still stays within the
        looser MAX_ABS_DIFF_BOUND.

        This documents the gap users on edge deployments without torch installed will see (versus the bit-exact
        torchvision path).
        """
        from unittest import mock

        pil = _make_synthetic_rgb(seed=11, size=(720, 1280))
        tgt = (384, 384)
        pt = _pytorch_preprocess(pil, tgt)

        # Hide torch from _preprocess_image's lazy import, forcing the PIL fallback.
        with mock.patch.dict(sys.modules, {"torch": None}):
            tf = _tflite_preprocess_to_nchw(pil, tgt)

        max_diff = float(np.abs(pt - tf).max())
        assert max_diff < MAX_ABS_DIFF_BOUND, (
            f"Torch-free PIL fallback diverged: max|diff|={max_diff:.4f} > {MAX_ABS_DIFF_BOUND}. "
            "The fallback uses PIL.BILINEAR which should keep diff ~0.016; a regression to "
            "BICUBIC would push it ~30x larger."
        )


class TestPreprocessingFilterRegression:
    """Direct comparison of BILINEAR vs BICUBIC.

    Asserts the current code stays on BILINEAR by showing BICUBIC would produce a much larger divergence.
    """

    def test_bicubic_would_be_much_worse(self) -> None:
        """If a future change reverts to BICUBIC default, this confirms how much worse it gets."""
        pil = _make_synthetic_rgb(seed=42, size=(720, 1280))
        tgt = (384, 384)

        pt = _pytorch_preprocess(pil, tgt)
        tf_current = _tflite_preprocess_to_nchw(pil, tgt)

        # Simulate the regression: PIL default (BICUBIC since 9.1.0).
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        height, width = tgt
        arr_bicubic = np.array(pil.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
        tf_bicubic = ((arr_bicubic - mean) / std)[np.newaxis].transpose(0, 3, 1, 2)

        max_diff_current = float(np.abs(pt - tf_current).max())
        max_diff_bicubic = float(np.abs(pt - tf_bicubic).max())

        # The BILINEAR fix must be at least 5x closer to PyTorch than the BICUBIC regression.
        # In practice it's ~30x closer; the 5x floor is forgiving of pillow / numpy drift.
        assert max_diff_current * 5 < max_diff_bicubic, (
            f"_preprocess_image is too close to BICUBIC behaviour: "
            f"current max|diff|={max_diff_current:.4f}, BICUBIC max|diff|={max_diff_bicubic:.4f}. "
            f"Check that _PIL_BILINEAR is being passed to .resize()."
        )


class TestBilinearResizeHalfPixelParity:
    """``_bilinear_resize_half_pixel`` is the torch-free fallback used by ``_decode_masks``.

    It must match ``torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)``
    -- the same call ``PostProcess.forward`` uses -- byte-for-byte modulo float noise. Sharp-edge
    inputs are the worst case: even a sub-pixel shift in the half-pixel convention flips boundary
    pixels and tanks mask IoU.
    """

    @staticmethod
    def _torch_interpolate(src: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        """Reference implementation: ``F.interpolate`` with ``align_corners=False``."""
        import torch
        import torch.nn.functional as TF  # noqa: N812

        with torch.no_grad():
            t = torch.from_numpy(src.astype(np.float32)).unsqueeze(0)
            out = TF.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return out.squeeze(0).cpu().numpy()

    @pytest.mark.parametrize(
        ("src_hw", "out_hw"),
        [
            pytest.param((28, 28), (384, 384), id="upsample_28_to_384"),
            pytest.param((56, 56), (256, 256), id="upsample_56_to_256"),
            pytest.param((100, 100), (100, 100), id="identity_100"),
            pytest.param((100, 100), (50, 50), id="downsample_100_to_50"),
            pytest.param((40, 60), (200, 400), id="non_square_upsample"),
        ],
    )
    def test_matches_torch_interpolate_on_random_logits(self, src_hw: tuple[int, int], out_hw: tuple[int, int]) -> None:
        """Random logits over a small batch must resize identically to ``F.interpolate``."""
        rng = np.random.default_rng(0)
        src = rng.standard_normal((3, *src_hw)).astype(np.float32) * 4.0

        ours = _bilinear_resize_half_pixel(src, out_hw[0], out_hw[1])
        ref = self._torch_interpolate(src, out_hw)

        max_diff = float(np.abs(ours - ref).max())
        # 1e-4 absorbs the float32 op-order noise that accumulates on large upsample ratios
        # (mine: split bilinear sums in pure numpy; torch: fused kernel). Half-pixel-convention
        # drift would push this several orders of magnitude higher.
        assert max_diff < 1e-4, (
            f"_bilinear_resize_half_pixel diverged from F.interpolate(align_corners=False): "
            f"max|diff|={max_diff:.2e} on shape {src_hw} -> {out_hw}. "
            "Half-pixel convention drift would surface here."
        )

    def test_sharp_edge_mask_matches_torch(self) -> None:
        """A mask with a sharp left/right boundary is the regression-prone case.

        This is the shape ``_decode_masks`` actually consumes (logits with a zero-crossing). A half-pixel shift would
        flip the boundary column and is exactly what the original PIL.BILINEAR path got wrong.
        """
        src = np.full((1, 28, 28), -10.0, dtype=np.float32)
        src[0, :, 14:] = 10.0  # sharp vertical edge at column 14

        out_hw = (224, 224)
        ours = _bilinear_resize_half_pixel(src, out_hw[0], out_hw[1])
        ref = self._torch_interpolate(src, out_hw)

        max_diff = float(np.abs(ours - ref).max())
        assert max_diff < 1e-4, (
            f"Sharp-edge resize diverged from F.interpolate: max|diff|={max_diff:.2e}. "
            "This is the case that previously dropped mask IoU below 0.6 with PIL.BILINEAR."
        )

        # Also assert the thresholded output matches: this is what _decode_masks actually returns.
        assert np.array_equal(ours > 0, ref > 0), (
            "Boolean mask after thresholding diverged from F.interpolate. Even a single column of "
            "flipped pixels would show up here -- the exact failure mode the original PR fixes."
        )
