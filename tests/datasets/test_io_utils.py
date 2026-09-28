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
from rfdetr.datasets.io_utils import decode_image, decode_image_bytes, decode_pil_image, decode_pil_image_bytes
from tests.datasets._memory import peak_traced_bytes


def _write_test_image(path: Path, width: int, height: int, mode: str = "RGB") -> None:
    """Write deterministic noise in the format ``path``'s suffix names, so JPEG comparisons exercise real DCT content.

    JPEG files are written at quality 90; lossless formats keep the noise exactly.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     jpeg_path = Path(tmp) / "img.jpg"
        ...     _write_test_image(jpeg_path, 8, 6)
        ...     with Image.open(jpeg_path) as image:
        ...         print(image.format, image.size)
        JPEG (8, 6)
    """
    pixels = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(pixels).convert(mode)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(path, quality=90)
    else:
        image.save(path)


def _pillow_decode(path: Path, draft_size: int | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Reference decode through Pillow alone, mirroring what ``decode_image`` did before ``simplejpeg`` support.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     jpeg_path = Path(tmp) / "img.jpg"
        ...     _write_test_image(jpeg_path, 64, 32)
        ...     pixels, scales = _pillow_decode(jpeg_path, draft_size=16)
        ...     pixels.shape, scales
        ((16, 32, 3), (0.5, 0.5))
    """
    with Image.open(path) as image:
        full_width, full_height = image.size
        if draft_size is not None:
            image.draft("RGB", (draft_size, draft_size))
        pixels = np.asarray(image.convert("RGB"))
    return pixels, (pixels.shape[1] / full_width, pixels.shape[0] / full_height)


def _write_cmyk_jpeg(path: Path, width: int, height: int) -> None:
    """Write an Adobe-marked CMYK JPEG with gradient content, for tolerance checks against RGB decoders.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     jpeg_path = Path(tmp) / "cmyk.jpg"
        ...     _write_cmyk_jpeg(jpeg_path, 8, 6)
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
        _write_test_image(jpeg_path, 457, 301)
        expected, _ = _pillow_decode(jpeg_path)

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
        _write_test_image(jpeg_path, width, height)
        expected, expected_scales = _pillow_decode(jpeg_path, draft_size)

        pixels, scales = decode_image(jpeg_path, draft_size)

        assert pixels.shape == expected.shape
        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    @requires_simplejpeg
    def test_grayscale_jpeg_decodes_to_rgb(self, tmp_path: Path) -> None:
        """Single-channel JPEG sources come back as three-channel RGB like Pillow's ``convert``."""
        jpeg_path = tmp_path / "gray.jpg"
        _write_test_image(jpeg_path, 40, 30, mode="L")
        expected, _ = _pillow_decode(jpeg_path)

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
        _write_cmyk_jpeg(jpeg_path, 40, 32)
        expected, _ = _pillow_decode(jpeg_path)

        pixels, scales = decode_image(jpeg_path)

        assert scales == (1.0, 1.0)
        assert pixels.shape == expected.shape
        np.testing.assert_allclose(pixels.astype(int), expected.astype(int), atol=2)

    @requires_simplejpeg
    def test_truncated_jpeg_raises_pillow_error(self, tmp_path: Path) -> None:
        """A JPEG simplejpeg rejects falls through to Pillow, so callers see the same ``OSError`` as before."""
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 457, 301)
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
        _write_test_image(jpeg_path, 64, 32)  # 2048 pixels, above the 2 * MAX_IMAGE_PIXELS raise tier

        with pytest.raises(Image.DecompressionBombError):
            decode_image(jpeg_path)

    @requires_simplejpeg
    def test_jpeg_between_pixel_limit_tiers_warns_and_decodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Above ``MAX_IMAGE_PIXELS`` but within twice it, the simplejpeg path warns and still decodes, like Pillow."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1500)
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 64, 32)  # 2048 pixels

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

        Confirms the final ``Image.open`` fallback also handles a non-JPEG payload, which is what an archive member
        whose extension does not guarantee JPEG content can be.
        """
        png_path = tmp_path / "img.png"
        expected = np.random.default_rng(0).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
        Image.fromarray(expected).save(png_path)

        pixels, scales = decode_image_bytes(png_path.read_bytes(), draft_size=8)

        assert scales == (1.0, 1.0)
        np.testing.assert_array_equal(pixels, expected)

    @pytest.mark.parametrize("draft_size", [None, 256])
    def test_bytes_entry_point_matches_path_entry_point(self, tmp_path: Path, draft_size: int | None) -> None:
        """``decode_image_bytes`` is the policy ``decode_image`` applies, so bytes and files decode alike."""
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 961, 541)
        expected, expected_scales = decode_image(jpeg_path, draft_size)

        pixels, scales = decode_image_bytes(jpeg_path.read_bytes(), draft_size)

        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    def test_without_simplejpeg_falls_back_to_pillow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``simplejpeg`` unavailable, JPEG decoding still drafts through Pillow."""
        monkeypatch.setattr(io_utils, "simplejpeg", None)
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 961, 541)
        expected, expected_scales = _pillow_decode(jpeg_path, 256)

        pixels, scales = decode_image(jpeg_path, 256)

        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    @pytest.mark.parametrize("draft_size", [None, 256])
    def test_bytes_without_simplejpeg_falls_back_to_pillow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft_size: int | None
    ) -> None:
        """With ``simplejpeg`` unavailable, ``decode_image_bytes`` still drafts through Pillow.

        The ``test_without_simplejpeg_falls_back_to_pillow`` case above only exercises this fallback through
        ``decode_image``'s file-based wrapper.
        """
        monkeypatch.setattr(io_utils, "simplejpeg", None)
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 961, 541)
        expected, expected_scales = _pillow_decode(jpeg_path, draft_size)

        pixels, scales = decode_image_bytes(jpeg_path.read_bytes(), draft_size)

        assert scales == expected_scales
        np.testing.assert_array_equal(pixels, expected)

    @requires_simplejpeg
    def test_bytes_garbage_after_soi_raises_pillow_error(self) -> None:
        """Bytes with a valid JPEG SOI marker but a garbage payload raise the same error Pillow would.

        Exercises the ``except ValueError`` fallback in ``_decode_with_simplejpeg``: ``simplejpeg.decode_jpeg_header``
        rejects the corrupt payload with a ``ValueError``, so decoding must fall through to Pillow and surface Pillow's
        own error instead of swallowing it.
        """
        data = b"\xff\xd8" + bytes(range(256))

        with pytest.raises(Image.UnidentifiedImageError):
            decode_image_bytes(data)


#: Each PIL-out entry point and the array-out entry point whose decoder policy it shares, by the source both take.
_ENTRY_POINTS = {
    "path": (decode_pil_image, decode_image),
    "bytes": (decode_pil_image_bytes, decode_image_bytes),
}


class TestDecodePilImage:
    """``decode_pil_image``/``decode_pil_image_bytes`` hand PIL consumers what the array entry points decode.

    Same decoder policy, same pixels and decode scales; the difference is that a Pillow-decoded image is returned as is
    instead of being copied into an array that the caller would copy straight back into a PIL image (#1544).
    """

    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    @pytest.mark.parametrize("extension", ["jpg", "png", "bmp"])
    @pytest.mark.parametrize("draft_size", [None, 256])
    def test_pixels_and_scales_match_array_entry_point(
        self, tmp_path: Path, source_kind: str, extension: str, draft_size: int | None
    ) -> None:
        """Every format, drafted or not, yields the array entry point's RGB pixels and decode scales."""
        decode_pil, decode_array = _ENTRY_POINTS[source_kind]
        image_path = tmp_path / f"img.{extension}"
        _write_test_image(image_path, 961, 541)
        source = image_path.read_bytes() if source_kind == "bytes" else image_path
        expected, expected_scales = decode_array(source, draft_size)

        image, scales = decode_pil(source, draft_size)

        assert (image.mode, scales) == ("RGB", expected_scales)
        np.testing.assert_array_equal(np.asarray(image), expected)

    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    @pytest.mark.parametrize("mode", ["L", "P", "RGBA"])
    def test_non_rgb_png_decodes_to_rgb(self, tmp_path: Path, source_kind: str, mode: str) -> None:
        """Grayscale, palette and alpha sources come back as the three-channel RGB of Pillow's ``convert("RGB")``."""
        decode_pil, _ = _ENTRY_POINTS[source_kind]
        image_path = tmp_path / "img.png"
        _write_test_image(image_path, 40, 30, mode=mode)
        source = image_path.read_bytes() if source_kind == "bytes" else image_path
        expected, _ = _pillow_decode(image_path)

        image, _ = decode_pil(source)

        np.testing.assert_array_equal(np.asarray(image), expected)

    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    def test_without_simplejpeg_drafts_through_pillow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
    ) -> None:
        """With ``simplejpeg`` unavailable, a JPEG still gets Pillow's reduced-scale decode."""
        monkeypatch.setattr(io_utils, "simplejpeg", None)
        decode_pil, _ = _ENTRY_POINTS[source_kind]
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 961, 541)
        source = jpeg_path.read_bytes() if source_kind == "bytes" else jpeg_path
        expected, expected_scales = _pillow_decode(jpeg_path, 256)

        image, scales = decode_pil(source, 256)

        assert scales == expected_scales
        np.testing.assert_array_equal(np.asarray(image), expected)

    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    @pytest.mark.parametrize("extension", ["png", "bmp"])
    def test_pillow_decode_is_not_copied_through_numpy(self, tmp_path: Path, source_kind: str, extension: str) -> None:
        """A Pillow-decoded image allocates no frame-sized NumPy buffer on its way to a PIL consumer.

        Regression test for #1544: the round trip through ``np.array`` and back through ``Image.fromarray`` slowed data
        loading on large PNG and BMP files. A Pillow-only decode peaks at its read buffers (about 0.14 MB here); the
        round trip peaks at about twice the 1.44 MB frame.
        """
        decode_pil, _ = _ENTRY_POINTS[source_kind]
        image_path = tmp_path / f"img.{extension}"
        _write_test_image(image_path, 800, 600)
        source = image_path.read_bytes() if source_kind == "bytes" else image_path

        # The bound is loose on purpose: the real floor is Pillow's read buffers (~138-142 KB, frame-independent),
        # not a fraction of the 800x600 frame. The bytes variant stays near that floor too, because CPython's
        # `io.BytesIO(data)` shares `data`'s buffer instead of copying it.
        assert peak_traced_bytes(decode_pil, source) < 800 * 600 * 3 // 2

    @requires_simplejpeg
    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    def test_jpeg_over_pixel_limit_raises_decompression_bomb_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
    ) -> None:
        """The simplejpeg path reached through the PIL entry points still enforces Pillow's pixel limit."""
        decode_pil, _ = _ENTRY_POINTS[source_kind]
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 64, 32)  # 2048 pixels, above the 2 * MAX_IMAGE_PIXELS raise tier
        source = jpeg_path.read_bytes() if source_kind == "bytes" else jpeg_path

        with pytest.raises(Image.DecompressionBombError):
            decode_pil(source)

    @requires_simplejpeg
    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    def test_truncated_jpeg_raises_pillow_error(self, tmp_path: Path, source_kind: str) -> None:
        """A truncated JPEG falls through to Pillow, raising the same ``OSError`` for the PIL entry points."""
        decode_pil, _ = _ENTRY_POINTS[source_kind]
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 457, 301)
        jpeg_path.write_bytes(jpeg_path.read_bytes()[:3000])
        source = jpeg_path.read_bytes() if source_kind == "bytes" else jpeg_path

        with pytest.raises(OSError, match="truncated"):
            decode_pil(source)

    @requires_simplejpeg
    @pytest.mark.parametrize("source_kind", ["path", "bytes"])
    def test_garbage_after_soi_raises_pillow_error(self, tmp_path: Path, source_kind: str) -> None:
        """A JPEG SOI marker followed by a garbage payload raises Pillow's error through the PIL entry points.

        Exercises the ``except ValueError`` fallback in ``_decode_with_simplejpeg`` the same way
        ``TestDecodeImage.test_bytes_garbage_after_soi_raises_pillow_error`` does for the array entry point.
        """
        decode_pil, _ = _ENTRY_POINTS[source_kind]
        jpeg_path = tmp_path / "img.jpg"
        data = b"\xff\xd8" + bytes(range(256))
        jpeg_path.write_bytes(data)
        source = data if source_kind == "bytes" else jpeg_path

        with pytest.raises(Image.UnidentifiedImageError):
            decode_pil(source)


class TestReadSimplejpegCandidate:
    """``_read_simplejpeg_candidate`` decides, by marker, whether a file is ``simplejpeg``'s to decode."""

    @requires_simplejpeg
    def test_jpeg_file_returns_its_bytes(self, tmp_path: Path) -> None:
        """A JPEG file's raw bytes come back unchanged, the positive branch the module doctest does not cover."""
        jpeg_path = tmp_path / "img.jpg"
        _write_test_image(jpeg_path, 8, 6)

        candidate = io_utils._read_simplejpeg_candidate(jpeg_path)

        assert candidate == jpeg_path.read_bytes()


def _single_image_coco_dataset(root: Path, file_name: str, width: int, height: int) -> CocoDetection:
    """Write one noise image named ``file_name`` under ``root/images`` with an empty COCO annotation file and load it.

    Examples:
        >>> import contextlib, io, tempfile
        >>> with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        ...     size = _single_image_coco_dataset(Path(tmp), "img1.png", 8, 6)[0][0].size
        >>> size
        (8, 6)
    """
    img_dir = root / "images"
    img_dir.mkdir()
    _write_test_image(img_dir / file_name, width, height)
    ann_file = root / "annotations.json"
    ann_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": file_name, "width": width, "height": height}],
                "annotations": [],
                "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
            }
        )
    )
    return CocoDetection(img_dir, ann_file, transforms=None)


class TestCocoDetectionRealPathParity:
    """The dataset's real read path preserves the pixel-parity contract the ``io_utils`` entry points provide."""

    @pytest.mark.parametrize("extension", ["jpg", "png"])
    def test_getitem_image_matches_pillow_reference(self, tmp_path: Path, extension: str) -> None:
        """``CocoDetection.__getitem__`` returns pixels identical to a direct Pillow decode of the same file.

        Every other test in this file calls the ``io_utils`` entry points directly.  This drives the real consumer path
        instead -- ``CocoDetection._decode_image`` -> ``ConvertCoco`` -> the returned image -- so a mismatch introduced
        anywhere along that chain, not only inside ``io_utils``, would surface here.
        """
        dataset = _single_image_coco_dataset(tmp_path, f"img1.{extension}", 64, 48)
        expected, _ = _pillow_decode(tmp_path / "images" / f"img1.{extension}")

        image, _ = dataset[0]

        np.testing.assert_array_equal(np.array(image), expected)

    @pytest.mark.parametrize("extension", ["png", "bmp"])
    def test_getitem_does_not_copy_pillow_decode_through_numpy(self, tmp_path: Path, extension: str) -> None:
        """A PNG or BMP training sample reaches the transforms without a frame-sized NumPy copy.

        #1544's scenario: 1.11.0 routed every non-JPEG sample through ``np.array`` and ``Image.fromarray``, which made
        the train DataLoader slower than 1.10.1 on large PNG and BMP files.
        """
        dataset = _single_image_coco_dataset(tmp_path, f"img1.{extension}", 800, 600)

        assert peak_traced_bytes(dataset.__getitem__, 0) < 800 * 600 * 3 // 2
