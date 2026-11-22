# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the EXIF orientation helpers in ``rfdetr.utilities.exif``."""

import io

import numpy as np
import pytest
from PIL import Image, ImageOps
from pytest_mock import MockerFixture

from rfdetr.utilities.exif import exif_orientation, swaps_axes, upright_image


def _jpeg_with_orientation(orientation: object, width: int = 6, height: int = 4) -> Image.Image:
    """Open an in-memory JPEG whose EXIF orientation tag holds ``orientation``.

    Examples:
        >>> _jpeg_with_orientation(6).getexif()[0x0112]
        6
    """
    exif = Image.Exif()
    exif[0x0112] = orientation
    encoded = io.BytesIO()
    Image.new("RGB", (width, height)).save(encoded, format="JPEG", exif=exif.tobytes())
    return Image.open(io.BytesIO(encoded.getvalue()))


def _noise_image(width: int = 7, height: int = 5) -> Image.Image:
    """Build an RGB image whose every pixel differs, so any flip or turn changes it.

    Examples:
        >>> _noise_image().size
        (7, 5)
    """
    pixels = ((np.arange(height * width * 3).reshape(height, width, 3) * 37 + 13) % 256).astype(np.uint8)
    return Image.fromarray(pixels)


class TestExifOrientation:
    @pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_returns_the_tag(self, orientation: int) -> None:
        """Each valid orientation is read back from the header."""
        assert exif_orientation(_jpeg_with_orientation(orientation)) == orientation

    def test_missing_tag_is_upright(self) -> None:
        """A photo without an orientation tag is stored upright."""
        encoded = io.BytesIO()
        Image.new("RGB", (6, 4)).save(encoded, format="JPEG")

        assert exif_orientation(Image.open(encoded)) == 1

    @pytest.mark.parametrize("orientation", [0, 9, 65535])
    def test_out_of_range_tag_is_upright(self, orientation: int) -> None:
        """Values outside 1 to 8 name no transpose, so the photo is read as stored."""
        assert exif_orientation(_jpeg_with_orientation(orientation)) == 1

    def test_unparsable_exif_is_upright(self, mocker: MockerFixture) -> None:
        """An EXIF block Pillow fails to parse must not stop the decode."""
        image = _jpeg_with_orientation(6)
        mocker.patch.object(Image.Image, "getexif", side_effect=SyntaxError("not a TIFF file"))

        assert exif_orientation(image) == 1

    def test_tiff_has_nothing_left_to_turn(self) -> None:
        """Pillow opens and loads a tagged TIFF upright itself, so the tag must not be applied a second time."""
        exif = Image.Exif()
        exif[0x0112] = 6
        encoded = io.BytesIO()
        Image.new("RGB", (6, 4)).save(encoded, format="TIFF", exif=exif.tobytes())
        image = Image.open(io.BytesIO(encoded.getvalue()))

        assert image.size == (4, 6)
        assert exif_orientation(image) == 1

    def test_png_tag_is_read_without_decoding_the_pixels(self, mocker: MockerFixture) -> None:
        """A header-only read of a PNG stays header-only, as dataset indexing and ``predict()`` rely on."""
        exif = Image.Exif()
        exif[0x0112] = 8
        encoded = io.BytesIO()
        Image.new("RGB", (6, 4)).save(encoded, format="PNG", exif=exif.tobytes())
        image = Image.open(io.BytesIO(encoded.getvalue()))
        mocker.patch("PIL.ImageFile.ImageFile.load", side_effect=AssertionError("Pillow decoded the pixels"))

        assert exif_orientation(image) == 8


class TestSwapsAxes:
    @pytest.mark.parametrize(("orientation", "expected"), [(1, False), (2, False), (3, False), (4, False)])
    def test_flips_and_half_turn_keep_the_axes(self, orientation: int, expected: bool) -> None:
        """Orientations 1 to 4 keep width and height."""
        assert swaps_axes(orientation) is expected

    @pytest.mark.parametrize("orientation", [5, 6, 7, 8])
    def test_quarter_turns_swap_the_axes(self, orientation: int) -> None:
        """Orientations 5 to 8 swap width and height."""
        assert swaps_axes(orientation) is True


class TestUprightImage:
    @pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
    def test_matches_pillow_exif_transpose(self, orientation: int) -> None:
        """The turn equals ``ImageOps.exif_transpose`` for every orientation that needs one."""
        image = _noise_image()
        tagged = image.copy()
        tagged.getexif()[0x0112] = orientation
        expected = np.asarray(ImageOps.exif_transpose(tagged))

        result = upright_image(image, orientation)

        np.testing.assert_array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize("orientation", [1, 0, 9])
    def test_upright_image_is_returned_unchanged(self, orientation: int) -> None:
        """No copy is made when the pixels are already upright."""
        image = _noise_image()

        assert upright_image(image, orientation) is image
