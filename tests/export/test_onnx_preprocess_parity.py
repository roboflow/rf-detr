# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Preprocessing parity tests for the ONNX Runtime inference helper.

``_preprocess_pil_to_nchw`` must produce the same input tensor as ``RFDETR.predict()`` for the same source image — the
ONNX sibling of ``tests/export/test_tflite_inference_parity.py``. History: this path resized with PIL BILINEAR, which
applies an adaptive antialias filter when downscaling; after predict() switched to ``antialias=False`` (#1206) the ONNX
inputs diverged on 91%+ of float bits.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image as PILImage

from rfdetr.export._onnx.inference import _preprocess_pil_to_nchw

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same bound as the TFLite parity file: the torch-free NumPy fallback matches the torchvision
# path up to float32 op-order noise (~5e-5 in normalised space).
FALLBACK_MAX_ABS_DIFF_BOUND = 1e-3


def _pytorch_preprocess(pil_img: PILImage.Image, hw: tuple[int, int]) -> np.ndarray:
    """Mirror of the PyTorch predict() preprocessing: to_tensor -> resize(antialias=False) -> normalize.

    Examples:
        >>> import numpy as np
        >>> from PIL import Image as PILImage
        >>> img = PILImage.new("RGB", (64, 64), color=(128, 64, 32))
        >>> arr = _pytorch_preprocess(img, (32, 32))
        >>> arr.shape
        (1, 3, 32, 32)
        >>> arr.dtype
        dtype('float32')
    """
    img = F.to_tensor(pil_img)
    img = F.resize(img, list(hw), antialias=False)
    img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img.unsqueeze(0).numpy()


def _make_synthetic_rgb(seed: int, size: tuple[int, int]) -> PILImage.Image:
    """Deterministic synthetic RGB image with structure (not pure noise) so resize filtering matters.

    Examples:
        >>> img = _make_synthetic_rgb(0, (64, 64))
        >>> img.size
        (64, 64)
        >>> img.mode
        'RGB'
    """
    rng = np.random.default_rng(seed)
    height, width = size
    base = rng.integers(0, 256, size=(height // 8, width // 8, 3), dtype=np.uint8)
    pil_small = PILImage.fromarray(base, mode="RGB")
    return pil_small.resize((width, height), getattr(PILImage, "Resampling", PILImage).NEAREST)


class TestOnnxPreprocessingParity:
    """``_preprocess_pil_to_nchw`` must match predict()'s preprocessing bit-for-bit with torchvision."""

    @pytest.mark.parametrize(
        ("src_hw", "tgt", "seed"),
        [
            pytest.param((720, 1280), (384, 384), 5, id="1280x720_to_384x384"),
            pytest.param((600, 800), (384, 384), 8, id="800x600_to_384x384"),
            pytest.param((500, 500), (256, 256), 11, id="500x500_to_256x256"),
            pytest.param((384, 384), (384, 384), 9, id="identity_384x384"),
        ],
    )
    def test_matches_pytorch_predict_preprocessing_bitwise(
        self, src_hw: tuple[int, int], tgt: tuple[int, int], seed: int
    ) -> None:
        pil = _make_synthetic_rgb(seed=seed, size=src_hw)

        pt = _pytorch_preprocess(pil, tgt)
        onnx = _preprocess_pil_to_nchw(pil, tgt[0], tgt[1], channels=3)

        assert pt.shape == onnx.shape, f"shape mismatch: PT {pt.shape} vs ONNX {onnx.shape}"
        pt32 = np.ascontiguousarray(pt, dtype=np.float32).view(np.uint32)
        on32 = np.ascontiguousarray(onnx, dtype=np.float32).view(np.uint32)
        n_diff = int((pt32 != on32).sum())
        assert n_diff == 0, (
            f"ONNX preprocessing diverged from predict(): {n_diff}/{pt32.size} float bits differ "
            f"(max|diff|={float(np.abs(pt - onnx).max()):.6f}). With torchvision importable it must "
            f"run predict()'s exact call chain (to_tensor -> resize(antialias=False) -> normalize)."
        )

    def test_antialias_true_would_be_much_worse(self) -> None:
        """Discriminator: torchvision's ``antialias=True`` default must diverge far from predict().

        The regression this file guards against — the ONNX path silently resized with antialiasing after predict()
        switched to ``antialias=False`` (#1206), flipping 91%+ of float bits. The current code must stay bit-identical
        to predict() while the antialiased variant stays far away, proving the bitwise parity assertion discriminates on
        the antialias flag rather than passing vacuously.
        """
        pil = _make_synthetic_rgb(seed=13, size=(720, 1280))
        tgt = (384, 384)

        pt = _pytorch_preprocess(pil, tgt)
        onnx = _preprocess_pil_to_nchw(pil, tgt[0], tgt[1], channels=3)

        img = F.to_tensor(pil.convert("RGB"))
        img = F.resize(img, list(tgt), antialias=True)
        img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        antialias = img.unsqueeze(0).numpy()

        pt32 = np.ascontiguousarray(pt, dtype=np.float32).view(np.uint32)
        on32 = np.ascontiguousarray(onnx, dtype=np.float32).view(np.uint32)
        assert int((pt32 != on32).sum()) == 0, "current ONNX code must stay bit-identical to predict()"

        max_diff_antialias = float(np.abs(pt - antialias).max())
        assert max_diff_antialias > 0.1, (
            f"antialias=True variant unexpectedly close to predict() (max|diff|={max_diff_antialias:.4f}); "
            "this test can no longer discriminate the antialias regression."
        )

    def test_torch_free_fallback_still_close(self) -> None:
        """With torch masked out, the NumPy fallback stays within float32 op-order noise of predict()."""
        from unittest import mock

        pil = _make_synthetic_rgb(seed=6, size=(720, 1280))
        tgt = (384, 384)
        pt = _pytorch_preprocess(pil, tgt)

        with mock.patch.dict(sys.modules, {"torch": None}):
            onnx = _preprocess_pil_to_nchw(pil, tgt[0], tgt[1], channels=3)

        max_diff = float(np.abs(pt - onnx).max())
        assert max_diff < FALLBACK_MAX_ABS_DIFF_BOUND, (
            f"Torch-free NumPy fallback diverged: max|diff|={max_diff:.4f} > {FALLBACK_MAX_ABS_DIFF_BOUND}. "
            "The fallback must use _bilinear_resize_half_pixel (antialias-free, half-pixel centres)."
        )

    def test_grayscale_channel_handling(self) -> None:
        """Grayscale (channels=1) path must produce shape (1, 1, H, W) float32."""
        rng = np.random.default_rng(7)
        pil = PILImage.fromarray(rng.integers(0, 256, size=(256, 256), dtype=np.uint8), mode="L")
        onnx = _preprocess_pil_to_nchw(pil, 128, 128, channels=1)
        assert onnx.shape == (1, 1, 128, 128), f"unexpected shape: {onnx.shape}"
        assert onnx.dtype == np.float32
