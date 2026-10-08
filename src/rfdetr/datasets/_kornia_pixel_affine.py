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
            Kornia affine parameters with x/y offsets sampled per image, or drawn once and shared by the whole batch
            when ``same_on_batch`` is set.
        """
        params = cast(dict[str, Tensor], super().generate_parameters(batch_shape))
        translations = params["translations"]
        batch_size = translations.shape[0]
        num_draws = 1 if self.same_on_batch else batch_size
        for axis, (lower, upper) in enumerate(self.translation_bounds):
            offsets = torch.randint(lower, upper + 1, (num_draws,), device=translations.device)
            translations[:, axis] = offsets.to(translations.dtype).expand(batch_size)
        return params

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Shift by whole pixels through direct indexing instead of grid sampling.

        Kornia's ``apply_transform_mask`` already switches the ``resample`` flag to nearest before it calls this
        method for masks. Indexing never reads that flag: it copies source pixels exactly, so the integer shift is
        bit-exact for images as well as masks, and it avoids building a sampling grid.

        Args:
            input: Images or packed mask channels.
            params: Sampled affine parameters.
            flags: Kornia border-mode flags.
            transform: Unused matrix passed by the Kornia pipeline.

        Returns:
            A new contiguous tensor shaped like ``input`` in which each image is moved by its sampled offset. Pixels
            shifted in from outside the frame are zero with ``"zeros"`` padding and repeat the nearest edge pixel with
            ``"border"`` padding.
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

    def inverse_transform(
        self,
        input: Tensor,
        flags: dict[str, Any],
        transform: Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Shift images or masks back by the negated forward offsets.

        Kornia's inverse path passes the inverted matrix to ``apply_transform``, which reads only the sampled
        ``translations``; negating those keeps the inverse exact for any input dtype, whereas the matrix is cast to the
        input dtype and would round large shifts in float16 or bfloat16.

        Args:
            input: Translated images or packed mask channels.
            flags: Kornia border-mode flags.
            transform: Unused inverse matrix passed by the Kornia pipeline.
            size: Unused output size passed by the Kornia pipeline.

        Returns:
            Values shifted back to their original positions, with the same shape and dtype as the input.
        """
        params = {**self._params, "translations": -self._params["translations"]}
        return self.apply_transform(input, params, flags, transform)
