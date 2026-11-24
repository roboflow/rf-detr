# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the EXIF orientation helpers in ``rfdetr.utilities.exif``."""

import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps, features
from pytest_mock import MockerFixture

from rfdetr.utilities.exif import exif_orientation, swaps_axes, upright_image
from tests._exif import move_exif_after_pixels

_requires_webp = pytest.mark.skipif(not features.check("webp"), reason="Pillow was built without WebP support")

#: The largest PNG eXIf read the walker may issue; ``_MAX_PNG_EXIF_BYTES`` is documented as 1 MiB.
_PNG_EXIF_READ_CAP = 1 << 20


def _tagged_png(image: Image.Image, orientation: int) -> bytes:
    """Encode ``image`` as PNG bytes whose eXIf chunk, written before the pixel data, holds ``orientation``.

    Examples:
        >>> Image.open(io.BytesIO(_tagged_png(Image.new("RGB", (6, 4)), 8))).getexif()[0x0112]
        8
    """
    exif = Image.Exif()
    exif[0x0112] = orientation
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", exif=exif.tobytes())
    return encoded.getvalue()


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


def _jpeg_with_garbage_exif() -> bytes:
    """Encode a JPEG whose APP1 Exif block is not a TIFF structure, so Pillow cannot parse its EXIF.

    Examples:
        >>> image = Image.open(io.BytesIO(_jpeg_with_garbage_exif()))
        >>> image.size, b"garbage" in image.info["exif"]
        ((6, 4), True)
    """
    encoded = io.BytesIO()
    Image.new("RGB", (6, 4)).save(encoded, format="JPEG")
    data = encoded.getvalue()
    payload = b"Exif\x00\x00" + b"garbage, neither II nor MM"
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return data[:2] + app1 + data[2:]


def _noise_image(width: int = 7, height: int = 5) -> Image.Image:
    """Build an RGB image whose every pixel differs, so any flip or turn changes it.

    Examples:
        >>> _noise_image().size
        (7, 5)
    """
    pixels = ((np.arange(height * width * 3).reshape(height, width, 3) * 37 + 13) % 256).astype(np.uint8)
    return Image.fromarray(pixels)


def _png_chunk(chunk_type: bytes, data: bytes = b"", declared_length: int | None = None) -> bytes:
    """Encode one PNG chunk, optionally declaring a length that differs from the data it carries.

    Examples:
        >>> _png_chunk(b"IEND").hex()
        '0000000049454e44ae426082'
        >>> struct.unpack(">I", _png_chunk(b"eXIf", b"ab", declared_length=0xFFFFFFFF)[:4])
        (4294967295,)
    """
    length = len(data) if declared_length is None else declared_length
    return struct.pack(">I", length) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data))


def _exif_block(orientation: int) -> bytes:
    """Encode an EXIF block holding ``orientation`` as the data of a PNG eXIf chunk.

    Examples:
        >>> Image.Exif().tobytes() != _exif_block(6)
        True
    """
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


def _png_without_iend(trailer: bytes, animated: bool = False) -> bytes:
    """Encode a PNG whose pixel data is followed by ``trailer`` instead of the usual IEND chunk.

    ``animated`` writes two different frames, so Pillow opens the file as an animated PNG.

    Examples:
        >>> Image.open(io.BytesIO(_png_without_iend(b""))).size
        (6, 4)
        >>> Image.open(io.BytesIO(_png_without_iend(b"", animated=True))).is_animated
        True
    """
    encoded = io.BytesIO()
    first = Image.new("RGB", (6, 4))
    if animated:
        first.save(encoded, format="PNG", save_all=True, append_images=[Image.new("RGB", (6, 4), (9, 9, 9))])
    else:
        first.save(encoded, format="PNG")
    data = encoded.getvalue()
    return data[: data.rindex(b"IEND") - 4] + trailer


def _tagged_file(image_format: str, orientation: int) -> bytes:
    """Encode noise as a ``image_format`` file whose EXIF orientation tag holds ``orientation``.

    ``MPO`` saves two frames, the way a phone writes a multi-picture JPEG; ``WEBP`` is lossless.

    Examples:
        >>> Image.open(io.BytesIO(_tagged_file("MPO", 6))).getexif()[0x0112]
        6
    """
    options: dict[str, object] = {"exif": _exif_block(orientation)}
    if image_format == "MPO":
        options.update(save_all=True, append_images=[_noise_image().rotate(90)])
    elif image_format == "WEBP":
        options.update(lossless=True)
    encoded = io.BytesIO()
    _noise_image().save(encoded, format=image_format, **options)
    return encoded.getvalue()


class _ReadRecorder(io.BytesIO):
    """In-memory stream that remembers the largest read it was asked for, ``-1`` counting as unbounded.

    Examples:
        >>> stream = _ReadRecorder(b"abcdef")
        >>> stream.read(2), stream.read()
        (b'ab', b'cdef')
        >>> stream.largest_read
        inf
    """

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.largest_read: float = 0

    def read(self, size: int | None = -1) -> bytes:
        self.largest_read = max(self.largest_read, float("inf") if size is None or size < 0 else size)
        return super().read(size)


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

    def test_unparsable_exif_is_upright(self) -> None:
        """An APP1 Exif block Pillow fails to parse must not stop the decode."""
        image = Image.open(io.BytesIO(_jpeg_with_garbage_exif()))

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

    def test_png_tag_after_the_pixels_is_read_without_decoding_them(self, mocker: MockerFixture) -> None:
        """An eXIf chunk stored after the pixel data is found from the chunk headers, still without a decode."""
        image = Image.open(io.BytesIO(move_exif_after_pixels(_tagged_png(_noise_image(), 6))))
        mocker.patch("PIL.ImageFile.ImageFile.load", side_effect=AssertionError("Pillow decoded the pixels"))

        assert exif_orientation(image) == 6

    def test_png_tag_after_the_pixels_leaves_the_image_decodable(self) -> None:
        """Walking the chunks puts the stream back, so Pillow then decodes the stored pixels unchanged."""
        noise = _noise_image()
        image = Image.open(io.BytesIO(move_exif_after_pixels(_tagged_png(noise, 6))))

        exif_orientation(image)

        np.testing.assert_array_equal(np.asarray(image.convert("RGB")), np.asarray(noise))

    def test_png_cut_short_after_the_pixels_is_upright(self) -> None:
        """A PNG that ends after its pixel data, with no IEND, has no chunk to find, and the walk stops at the end."""
        encoded = io.BytesIO()
        Image.new("RGB", (6, 4)).save(encoded, format="PNG")
        data = encoded.getvalue()

        assert exif_orientation(Image.open(io.BytesIO(data[: data.rindex(b"IEND") - 4]))) == 1

    @pytest.mark.parametrize(
        ("trailer", "expected"),
        [
            pytest.param(_png_chunk(b"eXIf", _exif_block(6)), 6, id="exif-after-pixels-and-no-iend"),
            pytest.param(
                _png_chunk(b"tEXt", b"k\x00v") + _png_chunk(b"eXIf", _exif_block(8)) + _png_chunk(b"IEND"),
                8,
                id="exif-after-another-chunk",
            ),
            pytest.param(
                _png_chunk(b"tEXt", b"k\x00v", declared_length=0xFFFFFFFF) + _png_chunk(b"eXIf", _exif_block(6)),
                1,
                id="oversized-length-on-another-chunk-skips-past-the-end",
            ),
            pytest.param(
                _png_chunk(b"eXIf", _exif_block(6), declared_length=0xFFFFFFFF),
                6,
                id="oversized-length-on-exif-reads-only-the-bytes-present",
            ),
            pytest.param(
                _png_chunk(b"eXIf", _exif_block(6) + bytes(2 * _PNG_EXIF_READ_CAP)),
                6,
                id="exif-over-the-cap-keeps-its-leading-ifd",
            ),
            pytest.param(_png_chunk(b"eXIf", b"not a tiff block"), 1, id="garbage-exif-body"),
            pytest.param(_png_chunk(b"eXIf"), 1, id="empty-exif-body"),
            pytest.param(b"\x00\x00\x00", 1, id="chunk-header-cut-short"),
            pytest.param(b"\x00\x00\x00\x08eXIf" + b"II*\x00", 1, id="exif-body-cut-short"),
            pytest.param(_png_chunk(b"IEND") + _png_chunk(b"eXIf", _exif_block(6)), 1, id="walk-stops-at-iend"),
        ],
    )
    def test_png_chunk_walk_is_bounded_and_reads_the_trailing_tag(self, trailer: bytes, expected: int) -> None:
        """Crafted chunk streams after the pixels end the walk without a huge read, and yield only a readable tag."""
        stream = _ReadRecorder(_png_without_iend(trailer))
        image = Image.open(stream)
        stream.largest_read = 0

        orientation = exif_orientation(image)

        assert (orientation, stream.largest_read <= _PNG_EXIF_READ_CAP) == (expected, True)

    def test_animated_png_is_not_walked_for_trailing_chunks(self) -> None:
        """An animated PNG's later chunks belong to frames Pillow has not loaded, so its tag is never taken from
        them."""
        image = Image.open(io.BytesIO(_png_without_iend(_png_chunk(b"eXIf", _exif_block(6)), animated=True)))

        assert (image.is_animated, exif_orientation(image)) == (True, 1)

    @pytest.mark.parametrize(("tagged", "expected"), [(True, 6), (False, 1)])
    def test_png_closed_after_decoding_is_read_without_the_stream(
        self, tmp_path: Path, tagged: bool, expected: int
    ) -> None:
        """Pillow closes a path-opened PNG once it has decoded it, and the orientation is then read from what it
        parsed."""
        path = tmp_path / "photo.png"
        if tagged:
            path.write_bytes(move_exif_after_pixels(_tagged_png(_noise_image(), 6)))
        else:
            _noise_image().save(path)
        with Image.open(path) as image:
            image.load()
            stream_was_released = image.fp is None

            orientation = exif_orientation(image)

        assert (stream_was_released, orientation) == (True, expected)


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
    @pytest.mark.parametrize("image_format", ["JPEG", "MPO", pytest.param("WEBP", marks=_requires_webp, id="WEBP")])
    @pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_matches_pillow_exif_transpose(self, image_format: str, orientation: int) -> None:
        """The orientation read from a file and the turn it names equal ``ImageOps.exif_transpose`` of that file."""
        encoded = _tagged_file(image_format, orientation)
        with Image.open(io.BytesIO(encoded)) as oracle:
            expected = np.asarray(ImageOps.exif_transpose(oracle).convert("RGB"))

        with Image.open(io.BytesIO(encoded)) as image:
            result = upright_image(image.convert("RGB"), exif_orientation(image))

        np.testing.assert_array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize("orientation", [1, 0, 9])
    def test_upright_image_is_returned_unchanged(self, orientation: int) -> None:
        """No copy is made when the pixels are already upright."""
        image = _noise_image()

        assert upright_image(image, orientation) is image
