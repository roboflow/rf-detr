# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Preprocessing parity tests for the benchmark and export example-input pipelines.

``infer_transforms`` (ONNX/TensorRT benchmark) and ``make_infer_image`` (traced example input for export) must feed the
model the same tensors ``RFDETR.predict()`` produces. History: both pipelines resized the PIL image before tensorising,
going through PIL's adaptive antialias filter; after predict() switched to ``antialias=False`` (#1206) they benchmarked
and traced on inputs predict() never produces. They now tensorise first and resize with ``antialias=False``; the only
residual versus predict()'s ``to_tensor`` chain is 1 ULP from v2's uint8->float scaling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image as PILImage

from rfdetr.export.benchmark import infer_transforms
from rfdetr.export.main import make_infer_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# v2's ToDtype(scale=True) and to_tensor may round the uint8->float conversion 1 ULP apart, which
# propagates through resize+normalize to ~1e-6 in normalised space. The old resize-then-tensorise
# pipelines (PIL antialiased filter) diverged by ~1.6 and would blow far past this bound.
MAX_ABS_DIFF_BOUND = 1e-4


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


def test_infer_transforms_matches_predict_preprocessing() -> None:
    """The benchmark pipeline must resize with predict()'s convention (bilinear, half-pixel, antialias-free)."""
    pil = _make_synthetic_rgb(seed=3, size=(720, 1280))
    tgt = (384, 384)

    pt = _pytorch_preprocess(pil, tgt)
    tensor, _ = infer_transforms(tgt)(pil, None)
    bench = tensor.unsqueeze(0).numpy()

    assert bench.shape == pt.shape, f"shape mismatch: predict {pt.shape} vs benchmark {bench.shape}"
    max_diff = float(np.abs(pt - bench).max())
    assert max_diff < MAX_ABS_DIFF_BOUND, (
        f"Benchmark preprocessing diverged from predict(): max|diff|={max_diff:.4f}. infer_transforms "
        f"must tensorise before resizing and pass antialias=False, matching detr.py's predict()."
    )

    # The antialiased variant (what a Resize without the flag, or a PIL-first pipeline, produces)
    # must stay far away, proving the assertion discriminates on the resize convention.
    img = F.to_tensor(pil)
    img = F.resize(img, list(tgt), antialias=True)
    img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    max_diff_antialias = float(np.abs(pt - img.unsqueeze(0).numpy()).max())
    assert max_diff_antialias > 0.1, (
        f"antialias=True variant unexpectedly close to predict() (max|diff|={max_diff_antialias:.4f}); "
        "this test can no longer discriminate the antialias regression."
    )


def test_make_infer_image_matches_predict_preprocessing(tmp_path: Path) -> None:
    """The traced example input must go through predict()'s resize convention, batched on the CPU."""
    pil = _make_synthetic_rgb(seed=4, size=(720, 1280))
    img_path = tmp_path / "img.png"
    pil.save(img_path)

    out = make_infer_image(str(img_path), shape=(384, 384), batch_size=2, device="cpu")

    assert out.shape == (2, 3, 384, 384), f"unexpected shape: {tuple(out.shape)}"
    pt = _pytorch_preprocess(pil, (384, 384))
    max_diff = float(np.abs(pt[0] - out[0].numpy()).max())
    assert max_diff < MAX_ABS_DIFF_BOUND, (
        f"Export example input diverged from predict(): max|diff|={max_diff:.4f}. make_infer_image "
        f"must tensorise before resizing and pass antialias=False, matching detr.py's predict()."
    )
    np.testing.assert_array_equal(
        out[0].numpy(), out[1].numpy(), err_msg="batch entries must be identical copies of the same image"
    )
