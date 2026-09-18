# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the shared image decoder in ``rfdetr.datasets.io_utils``."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rfdetr.datasets import io_utils
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.io_utils import decode_image, decode_image_bytes


def write_test_jpeg(path: Path, width: int, height: int, mode: str = "RGB") -> None:
    """Write a deterministic noisy JPEG so decoder comparisons exercise real DCT content.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     jpeg_path = Path(tmp) / "img.jpg"
        ...     write_test_jpeg(jpeg_path, 8, 6)
        ...     Image.open(jpeg_path).size
        (8, 6)
    """
    pixels = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).convert(mode).save(path, format="JPEG", quality=90)


def pillow_decode(path: Path, draft_size: int | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Reference decode through Pillow alone, mirroring what ``decode_image`` did before ``simplejpeg`` support.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     jpeg_path = Path(tmp) / "img.jpg"
        ...     write_test_jpeg(jpeg_path, 64, 32)
        ...     pixels, scales = pillow_decode(jpeg_path, draft_size=16)
        ...     pixels.shape, scales
        ((16, 32, 3), (0.5, 0.5))
    """
    with Image.open(path) as image:
        full_width, full_height = image.size
        if draft_size is not None:
            image.draft("RGB", (draft_size, draft_size))
        pixels = np.asarray(image.convert("RGB"))
    return pixels, (pixels.shape[1] / full_width, pixels.shape[0] / full_height)


def write_cmyk_jpeg(path: Path, width: int, height: int) -> None:
    """Write an Adobe-marked CMYK JPEG with gradient content, for tolerance checks against RGB decoders.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     jpeg_path = Path(tmp) / "cmyk.jpg"
        ...     write_cmyk_jpeg(jpeg_path, 8, 6)
        ...     Image.open(jpeg_path).mode
        'CMYK'
    """
    cyan = np.linspace(0, 255, width, dtype=np.uint8)
    magenta = np.linspace(0, 255, height, dtype=np.uint8)
    image = Image.new("CMYK", (width, height))
    image.putdata([(int(cyan[x]), int(magenta[y]), 128, 64) for y in range(height) for x in range(width)])
    image.save(path, format="JPEG", quality=90)


requires_simplejpeg = pytest.mark.skipif(io_utils.simplejpeg is None, reason="simplejpeg is not installed")


class TestDecodeImage:
    """``decode_image`` yields Pillow's pixel arrays whichever decoder handles the file.

    Comparisons between ``simplejpeg`` and Pillow assert exact equality for RGB and grayscale JPEG sources, on
    purpose.  The two packages may bundle different libjpeg-turbo builds, so equality is not guaranteed in general,
    but it holds for the PyPI wheels on every CI leg, and exactness is what catches a decoder-setting drift:
    ``fastdct=True`` moves these noisy test images by a mean of ~1.1 levels and smooth content by ~0.5, so a
    mean-difference tolerance near 1 would barely or not at all separate it from rounding.  If a future wheel pair
    diverges by rounding alone, loosen these to a bounded difference rather than chasing the pixels.

    CMYK sources are the one exception: ``simplejpeg`` and Pillow both apply the Adobe CMYK-to-RGB conversion but
    round it slightly differently, so that comparison uses a small tolerance (measured max absolute difference of 2
    on the fixture below) instead of exact equality.
    """

    @requires_simplejpeg
    def test_jpeg_pixels_match_pillow(self, tmp_path: Path) -> None:
        """Full-resolution simplejpeg output equals Pillow's."""
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 457, 301)
        expected, _ = pillow_decode(jpeg_path)

        pixels, scales = decode_image(jpeg_path)

        assert pixels.dtype == np.uint8
        assert scales == (1.0, 1.0)
        np.testing.assert_array_equal(pixels, expected)

    @requires_simplejpeg
    @pytest.mark.parametrize(
        ("width", "height", "draft_size"),
        [
            (1000, 1000, 350),
            (961, 541, 256),
            (640, 480, 560),
            (513, 1024, 512),
            (1000, 1000, 100),
        ],
    )
    def test_draft_reduction_matches_pillow(self, tmp_path: Path, width: int, height: int, draft_size: int) -> None:
        """Reduced decodes pick Pillow's power-of-two factor, not libjpeg-turbo's finer N/8 steps."""
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, width, height)
        expected, expected_scales = pillow_decode(jpeg_path, draft_size)

        pixels, scales = decode_image(jpeg_path, draft_size)

        assert pixels.shape == expected.shape
        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    @requires_simplejpeg
    def test_grayscale_jpeg_decodes_to_rgb(self, tmp_path: Path) -> None:
        """Single-channel JPEG sources come back as three-channel RGB like Pillow's ``convert``."""
        jpeg_path = tmp_path / "gray.jpg"
        write_test_jpeg(jpeg_path, 40, 30, mode="L")
        expected, _ = pillow_decode(jpeg_path)

        pixels, _ = decode_image(jpeg_path)

        assert pixels.shape == (30, 40, 3)
        np.testing.assert_array_equal(pixels, expected)

    @requires_simplejpeg
    def test_cmyk_jpeg_matches_pillow_within_rounding_tolerance(self, tmp_path: Path) -> None:
        """CMYK JPEG sources decode to RGB pixels within a small rounding tolerance of Pillow's conversion.

        ``simplejpeg`` and Pillow both apply the Adobe CMYK-to-RGB conversion but round it slightly differently
        (measured max absolute difference of 2 on this fixture), so unlike every other comparison in this class this one
        does not assert exact equality — see the class docstring.
        """
        jpeg_path = tmp_path / "cmyk.jpg"
        write_cmyk_jpeg(jpeg_path, 40, 32)
        expected, _ = pillow_decode(jpeg_path)

        pixels, scales = decode_image(jpeg_path)

        assert scales == (1.0, 1.0)
        assert pixels.shape == expected.shape
        np.testing.assert_allclose(pixels.astype(int), expected.astype(int), atol=2)

    @requires_simplejpeg
    def test_truncated_jpeg_raises_pillow_error(self, tmp_path: Path) -> None:
        """A JPEG simplejpeg rejects falls through to Pillow, so callers see the same ``OSError`` as before."""
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 457, 301)
        jpeg_path.write_bytes(jpeg_path.read_bytes()[:3000])

        with pytest.raises(OSError, match="truncated"):
            decode_image(jpeg_path)

    @requires_simplejpeg
    def test_jpeg_over_pixel_limit_raises_decompression_bomb_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The simplejpeg path enforces Pillow's pixel limit on the header size, as ``Image.open`` does."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 64, 32)  # 2048 pixels, above the 2 * MAX_IMAGE_PIXELS raise tier

        with pytest.raises(Image.DecompressionBombError):
            decode_image(jpeg_path)

    @requires_simplejpeg
    def test_jpeg_between_pixel_limit_tiers_warns_and_decodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Above ``MAX_IMAGE_PIXELS`` but within twice it, the simplejpeg path warns and still decodes, like Pillow."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1500)
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 64, 32)  # 2048 pixels

        with pytest.warns(Image.DecompressionBombWarning):
            pixels, _ = decode_image(jpeg_path)

        assert pixels.shape == (32, 64, 3)

    def test_png_uses_pillow_and_ignores_draft(self, tmp_path: Path) -> None:
        """Non-JPEG files decode through Pillow at full resolution regardless of ``draft_size``."""
        png_path = tmp_path / "img.png"
        expected = np.random.default_rng(0).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
        Image.fromarray(expected).save(png_path)

        pixels, scales = decode_image(png_path, draft_size=8)

        assert scales == (1.0, 1.0)
        np.testing.assert_array_equal(pixels, expected)

    def test_bytes_png_uses_pillow_and_ignores_draft(self, tmp_path: Path) -> None:
        """Non-JPEG bytes decode through Pillow at full resolution regardless of ``draft_size``.

        Every other ``decode_image_bytes`` case in this file feeds JPEG bytes; this is the only one confirming its final
        ``Image.open`` fallback also handles a non-JPEG payload, which is what an in-memory reader (an archive member
        whose extension does not guarantee JPEG content) can hand it.
        """
        png_path = tmp_path / "img.png"
        expected = np.random.default_rng(0).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
        Image.fromarray(expected).save(png_path)

        pixels, scales = decode_image_bytes(png_path.read_bytes(), draft_size=8)

        assert scales == (1.0, 1.0)
        np.testing.assert_array_equal(pixels, expected)

    @pytest.mark.parametrize("draft_size", [None, 256])
    def test_bytes_entry_point_matches_path_entry_point(self, tmp_path: Path, draft_size: int | None) -> None:
        """``decode_image_bytes`` is the policy ``decode_image`` applies, so in-memory readers get the same result."""
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 961, 541)
        expected, expected_scales = decode_image(jpeg_path, draft_size)

        pixels, scales = decode_image_bytes(jpeg_path.read_bytes(), draft_size)

        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    def test_without_simplejpeg_falls_back_to_pillow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``simplejpeg`` unavailable, JPEG decoding still drafts through Pillow."""
        monkeypatch.setattr(io_utils, "simplejpeg", None)
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 961, 541)
        expected, expected_scales = pillow_decode(jpeg_path, 256)

        pixels, scales = decode_image(jpeg_path, 256)

        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    @pytest.mark.parametrize("draft_size", [None, 256])
    def test_bytes_without_simplejpeg_falls_back_to_pillow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft_size: int | None
    ) -> None:
        """With ``simplejpeg`` unavailable, ``decode_image_bytes`` still drafts through Pillow.

        ``WebDatasetDetection._decode`` calls ``decode_image_bytes`` directly rather than going through
        ``decode_image``, so a reader without the ``[train]`` extra takes exactly this fallback path; the
        ``test_without_simplejpeg_falls_back_to_pillow`` case above only exercises it via ``decode_image``'s file-based
        wrapper.
        """
        monkeypatch.setattr(io_utils, "simplejpeg", None)
        jpeg_path = tmp_path / "img.jpg"
        write_test_jpeg(jpeg_path, 961, 541)
        expected, expected_scales = pillow_decode(jpeg_path, draft_size)

        pixels, scales = decode_image_bytes(jpeg_path.read_bytes(), draft_size)

        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    @requires_simplejpeg
    def test_bytes_garbage_after_soi_raises_pillow_error(self) -> None:
        """Bytes with a valid JPEG SOI marker but a garbage payload raise the same error Pillow would.

        Exercises the ``except ValueError: pass`` fallback in ``decode_image_bytes``: ``simplejpeg.decode_jpeg_header``
        rejects the corrupt payload with a ``ValueError``, so decoding must fall through to Pillow and surface Pillow's
        own error instead of swallowing it.
        """
        data = b"\xff\xd8" + bytes(range(256))

        with pytest.raises(Image.UnidentifiedImageError):
            decode_image_bytes(data)


class TestCocoDetectionRealPathParity:
    """The dataset's real read path preserves the pixel-parity contract ``decode_image`` provides directly."""

    def test_getitem_image_matches_pillow_reference(self, tmp_path: Path) -> None:
        """``CocoDetection.__getitem__`` returns pixels identical to a direct Pillow decode of the same file.

        Every other test in this file calls ``decode_image``/``decode_image_bytes`` directly.  This drives the real
        consumer path instead -- ``CocoDetection._decode_image`` -> ``Image.fromarray`` -> ``ConvertCoco`` -> the
        returned image -- so a mismatch introduced anywhere along that chain, not only inside ``io_utils``, would
        surface here.
        """
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        jpeg_path = img_dir / "img1.jpg"
        write_test_jpeg(jpeg_path, 64, 48)
        expected, _ = pillow_decode(jpeg_path)
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "img1.jpg", "width": 64, "height": 48}],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
                }
            )
        )

        dataset = CocoDetection(img_dir, ann_file, transforms=None)
        image, _ = dataset[0]

        np.testing.assert_array_equal(np.array(image), expected)
