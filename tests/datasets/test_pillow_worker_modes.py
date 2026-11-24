# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for which Pillow image modes ``convert_image`` accepts for 1 and 4 channel targets."""

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfdetr.datasets._pillow_worker import convert_image, to_8bit_grayscale
from rfdetr.datasets.io_utils import decode_image


def _16_bit_ramp() -> np.ndarray:
    """Return 16-bit pixels spanning the full range, as a one-row image."""
    return np.array([[0, 257, 32768, 65535]], dtype=np.uint16)


class TestTo8BitGrayscale:
    """16-bit grayscale scales to 8 bits by 1/257 instead of clipping to white (#1728)."""

    def test_scales_16_bit_array_image(self) -> None:
        """An ``I;16`` image scales to ``L`` with its full dynamic range."""
        scaled = to_8bit_grayscale(Image.fromarray(_16_bit_ramp()))
        assert scaled.mode == "L"
        np.testing.assert_array_equal(np.asarray(scaled), [[0, 1, 127, 255]])

    @pytest.mark.parametrize("mode", ["I;16B", "I;16L"])
    def test_scales_explicit_endianness_modes(self, mode: str) -> None:
        """The explicit-endianness ``I;16`` spellings scale the same way."""
        order = ">" if mode.endswith("B") else "<"
        raw = np.array([[0, 257, 65535]], dtype=f"{order}u2").tobytes()
        scaled = to_8bit_grayscale(Image.frombytes(mode, (3, 1), raw))
        assert scaled.mode == "L"
        np.testing.assert_array_equal(np.asarray(scaled), [[0, 1, 255]])

    def test_scales_older_pillow_32_bit_int_mode(self) -> None:
        """Mode ``I`` pixels above the 8-bit range scale too (older Pillow opens 16-bit PNGs as ``I``)."""
        image = Image.fromarray(_16_bit_ramp().astype(np.int32))
        assert image.mode == "I"
        np.testing.assert_array_equal(np.asarray(to_8bit_grayscale(image)), [[0, 1, 127, 255]])

    def test_leaves_in_range_32_bit_int_pixels(self) -> None:
        """Mode ``I`` pixels already in [0, 255] are genuine 32-bit data and keep their levels."""
        image = Image.fromarray(np.array([[0, 128, 255]], dtype=np.int32))
        assert image.mode == "I"
        assert to_8bit_grayscale(image) is image

    def test_leaves_out_of_range_32_bit_int_pixels(self) -> None:
        """Mode ``I`` pixels outside the 16-bit range are not 16-bit data and stay untouched."""
        image = Image.fromarray(np.array([[0, 70000]], dtype=np.int32))
        assert to_8bit_grayscale(image) is image

    def test_leaves_8_bit_and_float_images(self) -> None:
        """8-bit and float images pass through unchanged."""
        rgb = Image.new("RGB", (2, 2))
        assert to_8bit_grayscale(rgb) is rgb
        assert to_8bit_grayscale(Image.new("F", (2, 2))).mode == "F"


class TestDeepGrayscale:
    """A target of 1 channel refuses pixels deeper than 16 bits instead of saturating them at 255."""

    def test_refuses_32_bit_int_pixels_naming_the_file(self) -> None:
        """``I`` pixels outside the 16-bit range raise a ``ValueError`` that names the image."""
        image = Image.fromarray(np.array([[0, 70000]], dtype=np.int32))
        with pytest.raises(ValueError, match=r"thermal\.png.*deeper than 8 bits"):
            convert_image(image, "L", "thermal.png")

    def test_keeps_in_range_32_bit_int_pixels_on_rgb_target(self) -> None:
        """In-range mode ``I`` pixels convert to RGB with their levels intact, not black."""
        image = Image.fromarray(np.array([[0, 128, 255]], dtype=np.int32))
        converted = convert_image(image, "RGB", "int32.png")
        assert converted.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(converted), [[[0, 0, 0], [128, 128, 128], [255, 255, 255]]])

    def test_refuses_float_pixels_naming_the_file(self) -> None:
        """``F`` images raise a ``ValueError`` that names the image."""
        with pytest.raises(ValueError, match=r"thermal\.png.*deeper than 8 bits"):
            convert_image(Image.new("F", (2, 2)), "L", "thermal.png")

    def test_scales_16_bit_png_file(self, tmp_path: Path) -> None:
        """A 16-bit PNG decoded with ``num_channels=1`` scales to 8-bit grayscale."""
        path = tmp_path / "depth.png"
        Image.fromarray(_16_bit_ramp()).save(path)
        pixels, _ = decode_image(path, num_channels=1)
        assert pixels.shape == (1, 4)
        np.testing.assert_array_equal(pixels, [[0, 1, 127, 255]])

    def test_scales_16_bit_png_on_default_path(self, tmp_path: Path) -> None:
        """The 3-channel default path scales a 16-bit PNG instead of clipping it to white."""
        path = tmp_path / "depth.png"
        Image.fromarray(_16_bit_ramp()).save(path)
        pixels, _ = decode_image(path)
        assert pixels.shape == (1, 4, 3)
        np.testing.assert_array_equal(pixels, [[[0, 0, 0], [1, 1, 1], [127, 127, 127], [255, 255, 255]]])

    @pytest.mark.parametrize("mode", ["L", "LA", "RGB"])
    def test_accepts_8_bit_modes(self, mode: str) -> None:
        """8-bit modes still convert to ``L``."""
        assert convert_image(Image.new(mode, (2, 2)), "L", "image").mode == "L"


def _png_with_transparency(mode: str, transparency: int | tuple[int, int, int]) -> Image.Image:
    """Return *mode* image re-opened from a PNG that carries a tRNS chunk.

    Examples:
        >>> _png_with_transparency("RGB", (1, 2, 3)).info["transparency"]
        (1, 2, 3)
    """
    buffer = io.BytesIO()
    Image.new(mode, (2, 2)).save(buffer, "PNG", transparency=transparency)
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    return image


class TestTransparencyKey:
    """The ``transparency`` info key is a fourth channel for palette images only."""

    @pytest.mark.parametrize(
        ("mode", "key"),
        [pytest.param("RGB", (1, 2, 3), id="rgb-colour-key"), pytest.param("L", 5, id="gray-colour-key")],
    )
    def test_colour_key_is_not_a_fourth_channel(self, mode: str, key: int | tuple[int, int, int]) -> None:
        """A tRNS colour key on an RGB or L PNG is refused for 4 channels."""
        image = _png_with_transparency(mode, key)
        assert "transparency" in image.info
        with pytest.raises(ValueError, match="has no fourth channel"):
            convert_image(image, "RGBA", "key.png")

    def test_palette_alpha_is_a_fourth_channel(self) -> None:
        """A palette image with tRNS keeps its alpha when converted to RGBA."""
        image = _png_with_transparency("P", 0)
        assert convert_image(image, "RGBA", "palette.png").mode == "RGBA"

    @pytest.mark.parametrize("mode", ["LA", "PA", "RGBA"])
    def test_alpha_modes_stay_admitted(self, mode: str) -> None:
        """Modes that end in ``A`` are real alpha and stay admitted."""
        assert convert_image(Image.new(mode, (2, 2)), "RGBA", "image").mode == "RGBA"


def _four_sample_tiff(extra_samples: int) -> Image.Image:
    """Return a 4-sample TIFF of stored ``(100, 50, 25, 128)`` pixels whose ExtraSamples tag is *extra_samples*.

    Pillow rewrites ExtraSamples on save, so the tag value is patched in the written bytes.

    Examples:
        >>> _four_sample_tiff(2).mode
        'RGBA'
        >>> _four_sample_tiff(0).mode
        'RGB'
    """
    pixels = np.zeros((4, 4, 4), np.uint8)
    pixels[...] = (100, 50, 25, 128)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buffer, "TIFF")
    data = bytearray(buffer.getvalue())
    order = "<" if data[:2] == b"II" else ">"
    (ifd_offset,) = struct.unpack_from(order + "I", data, 4)
    (entries,) = struct.unpack_from(order + "H", data, ifd_offset)
    for index in range(entries):
        entry = ifd_offset + 2 + 12 * index
        (tag,) = struct.unpack_from(order + "H", data, entry)
        if tag == 338:  # ExtraSamples
            struct.pack_into(order + "H", data, entry + 8, extra_samples)
    return Image.open(io.BytesIO(bytes(data)))


class TestExtraSamplesTiff:
    """Diagnosis and decode of 4-sample TIFF files by their ExtraSamples tag."""

    def test_unspecified_extra_sample_is_diagnosed(self) -> None:
        """An unspecified fourth sample is refused with a message that says so and names the fix."""
        with pytest.raises(ValueError, match=r"4 samples per pixel.*unspecified.*ExtraSamples=2"):
            convert_image(_four_sample_tiff(0), "RGBA", "extra.tif")

    def test_plain_rgb_keeps_generic_message(self) -> None:
        """An RGB image that is not a 4-sample TIFF gets the generic refusal."""
        with pytest.raises(ValueError, match=r"has no fourth channel \(mode 'RGB'\)"):
            convert_image(Image.new("RGB", (2, 2)), "RGBA", "plain.png")

    def test_associated_alpha_is_unpremultiplied(self) -> None:
        """ExtraSamples=1 reads as premultiplied alpha: Pillow un-premultiplies the colour bands."""
        decoded = np.asarray(convert_image(_four_sample_tiff(1), "RGBA", "assoc.tif"))
        assert tuple(decoded[0, 0]) == (199, 99, 49, 128)

    def test_unassociated_alpha_is_unchanged(self) -> None:
        """ExtraSamples=2 keeps the stored colour bands, the way to carry a data plane."""
        decoded = np.asarray(convert_image(_four_sample_tiff(2), "RGBA", "data.tif"))
        assert tuple(decoded[0, 0]) == (100, 50, 25, 128)
