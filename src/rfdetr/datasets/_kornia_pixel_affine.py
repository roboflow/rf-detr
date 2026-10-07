# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Kornia affine transform with integer-pixel translation sampling."""

from typing import Any, cast

import torch
from kornia.augmentation import RandomAffine
from kornia.constants import SamplePadding
from torch import Tensor


class PixelTranslatedAffine(RandomAffine):  # type: ignore[misc]
    """Apply pure x/y translations sampled in whole pixels per image.

    Kornia's built-in ``translate`` samples a symmetric fraction of image size, which cannot express Albumentations'
    fixed or asymmetric ``translate_px``.
    """

    def __init__(
        self,
        translation_bounds: tuple[tuple[int, int], tuple[int, int]],
        *,
        p: float,
        padding_mode: str,
    ) -> None:
        """Construct an affine transform with inclusive x/y pixel bounds.

        Args:
            translation_bounds: Inclusive horizontal and vertical pixel ranges.
            p: Probability of applying the transform.
            padding_mode: Image and mask border mode.
        """
        super().__init__(
            degrees=0.0,
            padding_mode=padding_mode,
            align_corners=True,
            p=p,
        )
        self.translation_bounds = translation_bounds

    def generate_parameters(self, batch_shape: tuple[int, ...]) -> dict[str, Tensor]:
        """Replace Kornia's fractional translation draw with integer pixels.

        Args:
            batch_shape: Shape of the images selected for this transform.

        Returns:
            Kornia affine parameters with independently sampled x/y offsets.
        """
        params = cast(dict[str, Tensor], super().generate_parameters(batch_shape))
        translations = params["translations"]
        for axis, (lower, upper) in enumerate(self.translation_bounds):
            translations[:, axis] = torch.randint(
                lower, upper + 1, (translations.shape[0],), device=translations.device
            ).to(translations.dtype)
        return params

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Index pixels directly so an integer shift cannot interpolate masks.

        Args:
            input: Images or packed mask channels.
            params: Sampled affine parameters.
            flags: Kornia border-mode flags.
            transform: Unused matrix passed by the Kornia pipeline.

        Returns:
            Translated values with the same shape and dtype as the input.
        """
        batch_size, _, height, width = input.shape
        offsets = params["translations"].to(device=input.device, dtype=torch.long)
        source_y = torch.arange(height, device=input.device)[None, :] - offsets[:, 1:2]
        source_x = torch.arange(width, device=input.device)[None, :] - offsets[:, 0:1]
        padding_mode = flags["padding_mode"]
        valid_y: Tensor | None
        valid_x: Tensor | None
        if padding_mode == SamplePadding.ZEROS:
            valid_y = (source_y >= 0) & (source_y < height)
            valid_x = (source_x >= 0) & (source_x < width)
        elif padding_mode == SamplePadding.BORDER:
            valid_y = valid_x = None
        else:
            raise ValueError(f"Unsupported pixel translation padding mode: {padding_mode!r}")
        source_y = source_y.clamp(0, height - 1)
        source_x = source_x.clamp(0, width - 1)
        batch_indices = torch.arange(batch_size, device=input.device)[:, None, None]
        translated = input.permute(0, 2, 3, 1)[batch_indices, source_y[:, :, None], source_x[:, None, :]]
        if valid_y is not None and valid_x is not None:
            translated *= (valid_y[:, :, None] & valid_x[:, None, :]).unsqueeze(-1)
        return translated.permute(0, 3, 1, 2).contiguous()
