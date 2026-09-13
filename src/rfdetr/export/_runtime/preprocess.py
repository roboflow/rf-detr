# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Image preprocessing shared by the export inference helpers.

Every exported format consumes the same tensor: ImageNet-normalised float32 at the model's spatial resolution, resized
with :meth:`rfdetr.detr.RFDETR.predict`'s exact convention. Only the memory layout differs -- ONNX, OpenVINO and
ExecuTorch take NCHW, while TFLite takes NHWC because ``onnx2tf`` transposes at export time. That transpose is the
caller's job; this module produces NCHW.
"""

from __future__ import annotations

import contextlib

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from rfdetr.export._resize import _bilinear_resize_half_pixel

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


def preprocess_to_nchw(
    image: PILImage.Image,
    height: int,
    width: int,
    channels: int = 3,
) -> NDArray[np.float32]:
    """Resize and normalise a PIL image to a ``(1, C, H, W)`` float32 NCHW tensor.

    Resizes with ``RFDETR.predict()``'s exact convention -- bilinear, half-pixel centers,
    ``antialias=False`` -- via ``torchvision`` when importable (bit-exact parity) or the pure-NumPy
    :func:`~rfdetr.export._resize._bilinear_resize_half_pixel` otherwise (float32 op-order noise only). PIL resize is
    not used: both its BILINEAR and BICUBIC filters apply adaptive antialiasing when downscaling and diverge from
    predict(), shifting confidence scores. Normalises with ImageNet statistics: ``mean=[0.485, 0.456, 0.406]``,
    ``std=[0.229, 0.224, 0.225]``.

    Args:
        image: Input PIL image; any mode -- converted to ``"RGB"`` (3-channel) or ``"L"`` (1-channel) internally.
        height: Target spatial height expected by the model.
        width: Target spatial width expected by the model.
        channels: Number of channels the model expects (``1`` for grayscale, ``3`` for RGB).

    Returns:
        Float32 ndarray of shape ``(1, channels, height, width)``.

    Examples:
        >>> from PIL import Image
        >>> preprocess_to_nchw(Image.new("RGB", (8, 8)), height=4, width=4).shape
        (1, 3, 4, 4)
    """
    pil_mode = "L" if channels == 1 else "RGB"
    pil_img = image.convert(pil_mode)
    mean_list = [IMAGENET_MEAN[i % 3] for i in range(channels)]
    std_list = [IMAGENET_STD[i % 3] for i in range(channels)]

    with contextlib.suppress(ImportError):
        # Match predict() exactly: torchvision to_tensor -> resize(antialias=False) -> normalize.
        # antialias=False mirrors detr.py's predict(); torchvision's float-tensor default is True.
        import torch
        import torchvision.transforms.functional as _F  # noqa: N812

        with torch.no_grad():
            t = _F.to_tensor(pil_img)
            t = _F.resize(t, [height, width], antialias=False)
            t = _F.normalize(t, mean_list, std_list)
        return np.asarray(t.unsqueeze(0).cpu().numpy(), dtype=np.float32)

    # Torch-free fallback: same antialias-free half-pixel bilinear as predict(), in NumPy.
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # "L" -> (H, W); needs (H, W, 1)
        arr = arr[:, :, np.newaxis]
    chw = _bilinear_resize_half_pixel(arr.transpose(2, 0, 1), height, width)
    mean = np.array(mean_list, dtype=np.float32)[:, np.newaxis, np.newaxis]
    std = np.array(std_list, dtype=np.float32)[:, np.newaxis, np.newaxis]
    chw = (chw - mean) / std
    return np.expand_dims(chw, axis=0).astype(np.float32)  # (1, C, H, W)
