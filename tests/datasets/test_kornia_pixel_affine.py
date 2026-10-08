# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the Kornia integer-pixel translation transform ``PixelTranslatedAffine``.

All tests in this module are CPU-compatible — the transform indexes pixels directly, which behaves identically on CPU
and GPU tensors, so no ``@pytest.mark.gpu`` is needed.
"""

import pytest
import torch

from rfdetr.utilities.imports import _IS_KORNIA_INSTALLED

#: Skip tests that construct ``PixelTranslatedAffine``, which subclasses Kornia (optional extra).
kornia_only = pytest.mark.skipif(not _IS_KORNIA_INSTALLED, reason="kornia not installed; skip Kornia transform tests")


@kornia_only
class TestPixelTranslatedAffineInverse:
    """``inverse()`` must undo the integer shift on images, masks, and boxes alike."""

    def test_round_trip_restores_image_masks_and_boxes(self) -> None:
        """A forward shift followed by ``inverse()`` returns every target to its original position.

        Kornia's inverse path hands ``apply_transform`` the inverted matrix, which this transform ignores in favour of
        its sampled offsets, while boxes are inverted through that matrix. Images and masks must therefore be shifted
        back too, or they drift away from the boxes. The shift moves right by 3 and up by 2 so both axes and both signs
        are exercised; the bands the forward pass pushed out of frame were zero-filled and stay zero after the inverse.
        """
        from kornia.augmentation import AugmentationSequential

        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((3, 3), (-2, -2)), p=1.0, padding_mode="zeros")
        pipeline = AugmentationSequential(transform, data_keys=["input", "bbox_xyxy", "mask"])
        image = torch.arange(1, 8 * 10 + 1, dtype=torch.float32).reshape(1, 1, 8, 10)
        boxes = torch.tensor([[[4.0, 3.0, 6.0, 5.0]]])
        masks = torch.zeros(1, 1, 8, 10)
        masks[0, 0, 3:5, 4:6] = 1.0

        restored_image, restored_boxes, restored_masks = pipeline.inverse(*pipeline(image, boxes, masks))

        expected_image = torch.zeros_like(image)
        expected_image[:, :, 2:, :7] = image[:, :, 2:, :7]
        torch.testing.assert_close(restored_image, expected_image, rtol=0, atol=0)
        torch.testing.assert_close(restored_masks, masks, rtol=0, atol=0)
        torch.testing.assert_close(restored_boxes, boxes, rtol=0, atol=1e-5)

    def test_round_trip_with_partial_selection_restores_each_image(self) -> None:
        """With ``p=0.5`` only the selected images are shifted back, each by its own forward offset.

        Kornia samples offsets only for the images it selects and inverts just that subset, so the stored offsets must
        line up with the selected rows. Unselected images were never shifted and must come back untouched; selected ones
        lose only the column band the forward shift pushed out of frame.
        """
        from kornia.augmentation import AugmentationSequential

        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((2, 2), (0, 0)), p=0.5, padding_mode="zeros")
        pipeline = AugmentationSequential(transform, data_keys=["input", "bbox_xyxy"])
        image = torch.arange(1, 6 * 8 + 1, dtype=torch.float32).reshape(1, 1, 6, 8).repeat(16, 1, 1, 1)
        boxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]]).repeat(16, 1, 1)

        shifted_image, shifted_boxes = pipeline(image, boxes)
        restored_image, restored_boxes = pipeline.inverse(shifted_image, shifted_boxes)

        selected = shifted_boxes[:, 0, 0] != boxes[:, 0, 0]
        cropped = image.clone()
        cropped[:, :, :, 6:] = 0.0
        expected_image = torch.where(selected[:, None, None, None], cropped, image)
        assert selected.any() and (~selected).any()
        torch.testing.assert_close(restored_image, expected_image, rtol=0, atol=0)
        torch.testing.assert_close(restored_boxes, boxes, rtol=0, atol=1e-5)


@kornia_only
class TestPixelTranslatedAffineSameOnBatch:
    """``same_on_batch=True`` must give every image in a batch one shared shift."""

    def test_container_flag_shifts_every_image_identically(self) -> None:
        """A container-level ``same_on_batch=True`` shifts a batch of identical images to identical outputs.

        ``AugmentationSequential(same_on_batch=True)`` sets the flag on each child, and Kornia's own generators then
        draw one value and repeat it across the batch. The integer offsets replace Kornia's translation draw, so they
        must honour the flag too; with 49 possible offsets per image, independent draws make 16 outputs differ.
        """
        from kornia.augmentation import AugmentationSequential

        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((-3, 3), (-3, 3)), p=1.0, padding_mode="zeros")
        pipeline = AugmentationSequential(transform, data_keys=["input"], same_on_batch=True)
        image = torch.arange(1, 8 * 8 + 1, dtype=torch.float32).reshape(1, 1, 8, 8).repeat(16, 1, 1, 1)

        shifted = pipeline(image)

        torch.testing.assert_close(shifted, shifted[:1].expand_as(shifted), rtol=0, atol=0)

    def test_empty_selection_returns_empty_offsets(self) -> None:
        """A shared draw still yields no offset rows when no image in the batch was selected.

        With ``p < 1`` Kornia may select zero images and then asks for parameters of an empty batch; expanding the one
        shared draw must produce an empty ``(0, 2)`` offset tensor rather than a stray row.
        """
        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((-3, 3), (-3, 3)), p=0.5, padding_mode="zeros")
        transform.same_on_batch = True

        translations = transform.generate_parameters((0, 1, 8, 8))["translations"]

        assert translations.shape == (0, 2)
