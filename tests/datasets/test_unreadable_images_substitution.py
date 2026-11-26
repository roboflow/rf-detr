# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for which read failures the opt-in substitution of unreadable images covers, and what it costs and reports.

The substitution itself is pinned in ``test_unreadable_images.py``; these tests cover the decoder errors it must catch,
the reads it must not retry, the pause before it gives up, and the detail its errors carry.
"""

from __future__ import annotations

import errno
import json
import struct
import subprocess
import zlib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.datasets import io_utils
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.io_utils import (
    _KnownUnreadableImageError,
    decode_image,
    decode_pil_image,
    decode_pil_image_bytes,
    read_image_or_substitute,
    read_image_with_retries,
)
from rfdetr.datasets.yolo import YoloDetection
from tests.datasets.helpers import _patch_retry_sleep, _path_of, _Reader, _write_coco_dataset, _write_yolo_dataset

sleep = pytest.fixture(autouse=True)(_patch_retry_sleep)

# The datasets of this module hold three PNG images, which its tests overwrite by file name.
_write_coco_dataset = partial(_write_coco_dataset, extension="png", num_images=3)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    """Return one PNG chunk: data length, chunk type, data and CRC.

    Examples:
        >>> _png_chunk(b"IEND", b"").hex()
        '0000000049454e44ae426082'
    """
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _two_idat_png() -> tuple[bytes, int]:
    """Return an 8 x 8 RGB PNG whose pixel data is split over two IDAT chunks, and the offset of the second one.

    Pillow reads the header of a chunk that follows an IDAT chunk only while it loads the pixels, so damage there
    passes ``Image.open`` and surfaces from ``load``.

    Examples:
        >>> import io
        >>> data, offset = _two_idat_png()
        >>> Image.open(io.BytesIO(data)).convert("RGB").size, data[offset + 4 : offset + 8]
        ((8, 8), b'IDAT')
    """
    width = height = 8
    rows = (b"\x00" + bytes((17 * row + column) % 256 for column in range(width * 3)) for row in range(height))
    compressed = zlib.compress(b"".join(rows))
    half = len(compressed) // 2
    header = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    first = _png_chunk(b"IDAT", compressed[:half])
    second = _png_chunk(b"IDAT", compressed[half:])
    return header + first + second + _png_chunk(b"IEND", b""), len(header) + len(first)


def _damage_second_chunk_type(data: bytes, offset: int) -> bytes:
    """Zero one byte of the type of the chunk at ``offset``, as bit rot would.

    Examples:
        >>> _damage_second_chunk_type(b"0123456789", 0)
        b'012345\\x00789'
    """
    damaged = bytearray(data)
    damaged[offset + 6] = 0
    return bytes(damaged)


def _cut_inside_second_chunk_header(data: bytes, offset: int) -> bytes:
    """Cut the file two bytes into the type of the chunk at ``offset``, as an interrupted copy would.

    Examples:
        >>> _cut_inside_second_chunk_header(b"0123456789", 0)
        b'012345'
    """
    return data[: offset + 6]


def _broken_png(damage: Callable[[bytes, int], bytes]) -> bytes:
    """Return the two-IDAT PNG of :func:`_two_idat_png` with ``damage`` applied to the header of its second IDAT chunk.

    Examples:
        >>> len(_broken_png(_cut_inside_second_chunk_header)) < len(_two_idat_png()[0])
        True
    """
    return damage(*_two_idat_png())


_DAMAGES = [
    pytest.param(_damage_second_chunk_type, id="bit-rotted-chunk-type"),
    pytest.param(_cut_inside_second_chunk_header, id="cut-inside-chunk-header"),
]
_DECODERS = [
    pytest.param(decode_pil_image, id="pil"),
    pytest.param(decode_image, id="array"),
]


class TestBrokenPngChunk:
    """Pillow raises ``SyntaxError`` for a broken chunk header after an IDAT chunk; the decoders raise ``OSError``."""

    @pytest.mark.parametrize("damage", _DAMAGES)
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_decoder_raises_os_error(
        self, tmp_path: Path, decoder: Callable[[Path], Any], damage: Callable[[bytes, int], bytes]
    ) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(_broken_png(damage))

        with pytest.raises(OSError, match="broken PNG file"):
            decoder(path)

    def test_os_error_is_chained_to_the_pillow_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(_broken_png(_cut_inside_second_chunk_header))

        with pytest.raises(OSError) as error:
            decode_pil_image(path)

        assert isinstance(error.value.__cause__, SyntaxError)

    def test_decoder_process_raises_os_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A per-decode limit above the process limit decodes in a separate process, which must raise the same type."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        path = tmp_path / "broken.png"
        path.write_bytes(_broken_png(_cut_inside_second_chunk_header))

        with pytest.raises(OSError, match="broken PNG file"):
            decode_pil_image(path, max_image_pixels=10_000)

    @pytest.mark.parametrize(
        ("error_name", "expected"),
        [
            ("SyntaxError", OSError),
            ("OSError", OSError),
            ("FileNotFoundError", FileNotFoundError),
            ("PermissionError", PermissionError),
            ("TimeoutError", TimeoutError),
            ("IsADirectoryError", IsADirectoryError),
            ("NotADirectoryError", NotADirectoryError),
            ("ConnectionResetError", ConnectionResetError),
        ],
    )
    def test_decoder_process_error_is_raised_as_an_os_error(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, error_name: str, expected: type[OSError]
    ) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        response = json.dumps({"error": error_name, "message": "worker failure"}).encode("utf-8") + b"\n"
        mocker.patch.object(
            io_utils.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=response, stderr=b"")
        )

        with pytest.raises(OSError, match="worker failure") as error:
            decode_pil_image_bytes(_two_idat_png()[0], max_image_pixels=10_000)

        assert type(error.value) is expected

    @pytest.mark.parametrize("error_name", ["KeyError", "MemoryError", "exit", "NoSuchError"])
    def test_decoder_process_error_of_another_type_is_a_runtime_error(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, error_name: str
    ) -> None:
        """Only ``OSError`` types are restored by name; any other name, built-in or not, stays a ``RuntimeError``."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        response = json.dumps({"error": error_name, "message": "worker failure"}).encode("utf-8") + b"\n"
        mocker.patch.object(
            io_utils.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=response, stderr=b"")
        )

        with pytest.raises(RuntimeError, match="worker failure"):
            decode_pil_image_bytes(_two_idat_png()[0], max_image_pixels=10_000)

    def test_decoder_process_substitutes_a_directory_in_place_of_an_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decoder process opens a non-RGB read itself, so its ``IsADirectoryError`` must reach the substitution."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        paths = [tmp_path / "directory.png", tmp_path / "image.png"]
        paths[0].mkdir()
        Image.new("L", (4, 4)).save(paths[1])

        index, _ = read_image_or_substitute(
            lambda i: decode_pil_image(paths[i], max_image_pixels=10_000, num_channels=1), 0, 2, lambda i: paths[i]
        )

        assert index == 1

    def test_coco_sample_with_a_broken_chunk_is_substituted(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        (tmp_path / "image1.png").write_bytes(_broken_png(_cut_inside_second_chunk_header))
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        _, target = dataset[1]

        assert int(target["image_id"]) == 3


class TestKnownUnreadableImage:
    """A read known to fail on every attempt is substituted at once, with no retry, pause or warning."""

    def test_read_raises_on_the_first_attempt(self) -> None:
        reader = _Reader({3: _KnownUnreadableImageError("unreadable when the dataset was built")})

        with pytest.raises(_KnownUnreadableImageError):
            read_image_with_retries(lambda: reader(3), _path_of(3))

        assert reader.calls == [3]

    def test_read_does_not_pause_or_warn(self, mocker: MockerFixture, sleep: Any) -> None:
        reader = _Reader({3: _KnownUnreadableImageError("unreadable when the dataset was built")})
        log = mocker.patch.object(io_utils, "logger")

        with pytest.raises(_KnownUnreadableImageError):
            read_image_with_retries(lambda: reader(3), _path_of(3))

        assert (sleep.called, log.method_calls) == (False, [])

    def test_sample_is_substituted_by_the_next_one(self) -> None:
        reader = _Reader({3: _KnownUnreadableImageError("unreadable when the dataset was built")})

        assert read_image_or_substitute(reader, 3, 10, _path_of) == (4, "image-4")


def _stalled_reader(num_samples: int, stalled_reads: int) -> _Reader:
    """Return a reader whose sample 0 never reads and whose other samples fail their first ``stalled_reads`` reads.

    Examples:
        >>> reader = _stalled_reader(2, stalled_reads=1)
        >>> try:
        ...     reader(1)
        ... except OSError as exc:
        ...     print(exc)
        Input/output error
        >>> reader(1)
        'image-1'
    """
    errors = {index: OSError("Input/output error") for index in range(num_samples)}
    return _Reader(errors, failures={index: stalled_reads for index in range(1, num_samples)})


class TestStorageStall:
    """A storage stall that outlasts the retries gets one more pause before the dataset is judged broken."""

    def test_stall_that_ends_during_the_pause_yields_a_substitute(self) -> None:
        assert read_image_or_substitute(_stalled_reader(20, stalled_reads=1), 0, 20, _path_of) == (1, "image-1")

    def test_pauses_once_more_for_the_longest_retry_pause(self, sleep: Any) -> None:
        read_image_or_substitute(_stalled_reader(20, stalled_reads=1), 0, 20, _path_of)

        backoff = io_utils._IMAGE_READ_BACKOFF_SECONDS
        assert [call.args[0] for call in sleep.call_args_list] == [backoff, 2 * backoff, 2 * backoff]

    def test_two_sample_dataset_survives_one_failed_read_of_the_other_sample(self) -> None:
        assert read_image_or_substitute(_stalled_reader(2, stalled_reads=1), 0, 2, _path_of) == (1, "image-1")

    def test_broken_dataset_still_raises(self) -> None:
        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            read_image_or_substitute(_stalled_reader(20, stalled_reads=2), 0, 20, _path_of)

    def test_reads_each_substitute_twice_before_raising(self) -> None:
        """Retries and a sweep, then after the pause one more read of the sample and a second sweep."""
        reader = _stalled_reader(20, stalled_reads=2)

        with pytest.raises(RuntimeError):
            read_image_or_substitute(reader, 0, 20, _path_of)

        substitutes = list(range(1, io_utils._MAX_SUBSTITUTIONS + 1))
        assert reader.calls == [0] * io_utils._IMAGE_READ_ATTEMPTS + substitutes + [0] + substitutes

    def test_error_names_a_sustained_storage_outage(self) -> None:
        with pytest.raises(RuntimeError, match="stayed unavailable through every retry, as in a sustained outage"):
            read_image_or_substitute(_stalled_reader(20, stalled_reads=2), 0, 20, _path_of)

    def test_known_unreadable_substitutes_are_not_paused_for(self, sleep: Any) -> None:
        reader = _Reader({index: _KnownUnreadableImageError("unreadable at build") for index in range(20)})

        with pytest.raises(RuntimeError):
            read_image_or_substitute(reader, 0, 20, _path_of)

        assert (sleep.called, len(reader.calls)) == (False, io_utils._MAX_SUBSTITUTIONS + 1)

    def test_requested_sample_that_recovers_during_the_pause_is_returned(self) -> None:
        """The stall fails the requested sample's three reads and every substitute's first; the sample itself wins."""
        errors = {index: OSError("Input/output error") for index in range(20)}
        reader = _Reader(errors, failures={index: 3 if index == 0 else 1 for index in range(20)})

        assert read_image_or_substitute(reader, 0, 20, _path_of) == (0, "image-0")

    def test_requested_sample_that_recovers_during_the_pause_is_not_logged_unreadable(
        self, mocker: MockerFixture
    ) -> None:
        errors = {index: OSError("Input/output error") for index in range(20)}
        reader = _Reader(errors, failures={index: 3 if index == 0 else 1 for index in range(20)})
        log = mocker.patch.object(io_utils, "logger")

        read_image_or_substitute(reader, 0, 20, _path_of)

        assert log.error.called is False

    def test_pauses_for_the_requested_sample_when_every_substitute_is_known_unreadable(self) -> None:
        errors: dict[int, OSError] = {
            index: _KnownUnreadableImageError("unreadable at build") for index in range(1, 20)
        }
        reader = _Reader({0: OSError("Input/output error"), **errors}, failures={0: 3})

        assert read_image_or_substitute(reader, 0, 20, _path_of) == (0, "image-0")

    def test_single_sample_that_recovers_during_the_pause_is_returned(self) -> None:
        reader = _Reader({0: OSError("Input/output error")}, failures={0: 3})

        assert read_image_or_substitute(reader, 0, 1, _path_of) == (0, "image-0")


class TestSubstituteErrors:
    """The error of each substitute that fails is logged, and kept in the error raised when every substitute fails."""

    def test_each_failed_substitute_is_logged_at_debug_level(self, mocker: MockerFixture) -> None:
        log = mocker.patch.object(io_utils, "logger")
        reader = _Reader({index: OSError("image file is truncated") for index in (3, 4, 5)})

        read_image_or_substitute(reader, 3, 10, _path_of)

        assert [call.args[1:3] for call in log.debug.call_args_list] == [(_path_of(4), 4), (_path_of(5), 5)]

    def test_error_names_each_distinct_substitute_error(self) -> None:
        reader = _Reader({0: OSError("truncated"), 1: FileNotFoundError("gone"), 2: PermissionError("no")})

        with pytest.raises(RuntimeError) as error:
            read_image_or_substitute(reader, 0, 3, _path_of)

        assert "failed with: FileNotFoundError: gone; PermissionError: no." in str(error.value)

    def test_error_names_a_repeated_substitute_error_once(self) -> None:
        with pytest.raises(RuntimeError) as error:
            read_image_or_substitute(_stalled_reader(20, stalled_reads=2), 0, 20, _path_of)

        assert str(error.value).count("OSError: Input/output error") == 1

    def test_single_sample_error_lists_no_substitute(self) -> None:
        with pytest.raises(RuntimeError) as error:
            read_image_or_substitute(_Reader({0: OSError("image file is truncated")}), 0, 1, _path_of)

        assert "tried in its place" not in str(error.value)


class TestYoloSampleUnreadableAtBuild:
    """A YOLO sample whose header was unreadable when the dataset was built is known to stay unreadable."""

    def test_access_does_not_pause_or_warn(self, tmp_path: Path, mocker: MockerFixture, sleep: Any) -> None:
        image_dir, label_dir, data_file = _write_yolo_dataset(tmp_path)
        (image_dir / "image1.png").write_bytes(b"not an image")
        dataset = YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)
        sleep.reset_mock()  # building the dataset retried the header read; only the access below is measured
        log = mocker.patch.object(io_utils, "logger")

        dataset[1]

        assert (sleep.call_count, log.warning.call_count) == (0, 0)


def _short_phys_png(after_idat: bool) -> bytes:
    """Return a 4 x 4 RGB PNG whose ``pHYs`` chunk holds 4 bytes instead of 9, placed before or after its IDAT chunk.

    Pillow raises ``ValueError`` for the short chunk: while opening the file when the chunk comes before the pixel data,
    as PNG requires, and only while loading the pixels when it comes after.

    Examples:
        >>> import io
        >>> Image.open(io.BytesIO(_short_phys_png(after_idat=True))).size
        (4, 4)
    """
    width = height = 4
    rows = b"".join(b"\x00" + bytes(width * 3) for _ in range(height))
    header = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    phys = _png_chunk(b"pHYs", b"\x00\x00\x00\x01")
    idat = _png_chunk(b"IDAT", zlib.compress(rows))
    chunks = idat + phys if after_idat else phys + idat
    return header + chunks + _png_chunk(b"IEND", b"")


_PHYS_PLACEMENTS = [
    pytest.param(False, id="raised-on-open"),
    pytest.param(True, id="raised-on-load"),
]


class TestShortPhysChunk:
    """Pillow raises ``ValueError`` for a ``pHYs`` chunk shorter than 9 bytes; the decoders raise ``OSError``."""

    @pytest.mark.parametrize("after_idat", _PHYS_PLACEMENTS)
    @pytest.mark.parametrize("decoder", _DECODERS)
    def test_decoder_raises_os_error(self, tmp_path: Path, decoder: Callable[[Path], Any], after_idat: bool) -> None:
        path = tmp_path / "short-phys.png"
        path.write_bytes(_short_phys_png(after_idat))

        with pytest.raises(OSError, match="Truncated pHYs chunk"):
            decoder(path)

    @pytest.mark.parametrize("after_idat", _PHYS_PLACEMENTS)
    def test_decoder_with_a_per_decode_limit_raises_os_error(self, tmp_path: Path, after_idat: bool) -> None:
        """A per-decode limit below Pillow's own calls the format plugins directly instead of ``Image.open``."""
        path = tmp_path / "short-phys.png"
        path.write_bytes(_short_phys_png(after_idat))

        with pytest.raises(OSError, match="Truncated pHYs chunk"):
            decode_pil_image(path, max_image_pixels=10_000)

    @pytest.mark.parametrize("after_idat", _PHYS_PLACEMENTS)
    def test_decoder_process_raises_os_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_idat: bool
    ) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        path = tmp_path / "short-phys.png"
        path.write_bytes(_short_phys_png(after_idat))

        with pytest.raises(OSError, match="Truncated pHYs chunk"):
            decode_pil_image(path, max_image_pixels=10_000)

    @pytest.mark.parametrize("after_idat", _PHYS_PLACEMENTS)
    def test_coco_sample_is_substituted(self, tmp_path: Path, after_idat: bool) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        (tmp_path / "image1.png").write_bytes(_short_phys_png(after_idat))
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        _, target = dataset[1]

        assert int(target["image_id"]) == 3

    @pytest.mark.parametrize("after_idat", _PHYS_PLACEMENTS)
    def test_decoder_process_sample_is_substituted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_idat: bool
    ) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        paths = [tmp_path / "short-phys.png", tmp_path / "image.png"]
        paths[0].write_bytes(_short_phys_png(after_idat))
        Image.new("RGB", (4, 4)).save(paths[1])

        index, _ = read_image_or_substitute(
            lambda i: decode_pil_image(paths[i], max_image_pixels=10_000), 0, 2, lambda i: paths[i]
        )

        assert index == 1


_PROCESS_RESOURCE_ERRNOS = [
    pytest.param(errno.EMFILE, id="EMFILE"),
    pytest.param(errno.ENFILE, id="ENFILE"),
    pytest.param(errno.ENOMEM, id="ENOMEM"),
    pytest.param(errno.EAGAIN, id="EAGAIN"),
]


class TestProcessResourceError:
    """A read that fails because the process is out of a resource is retried, but the healthy file is never replaced."""

    @pytest.mark.parametrize("error_number", _PROCESS_RESOURCE_ERRNOS)
    def test_requested_sample_error_is_raised_after_the_retries(self, error_number: int) -> None:
        error = OSError(error_number, "resource exhausted")

        with pytest.raises(OSError) as raised:
            read_image_or_substitute(_Reader({3: error}), 3, 10, _path_of)

        assert raised.value is error

    def test_requested_sample_is_retried(self) -> None:
        reader = _Reader({3: OSError(errno.EMFILE, "Too many open files")})

        with pytest.raises(OSError):
            read_image_or_substitute(reader, 3, 10, _path_of)

        assert reader.calls == [3, 3, 3]

    def test_requested_sample_is_not_logged_unreadable(self, mocker: MockerFixture) -> None:
        log = mocker.patch.object(io_utils, "logger")

        with pytest.raises(OSError):
            read_image_or_substitute(_Reader({3: OSError(errno.EMFILE, "Too many open files")}), 3, 10, _path_of)

        assert log.error.called is False

    def test_substitute_error_is_raised_at_once(self) -> None:
        """The requested file is unreadable, but the read of its substitute runs out of descriptors."""
        error = OSError(errno.EMFILE, "Too many open files")
        reader = _Reader({3: OSError("image file is truncated"), 4: error})

        with pytest.raises(OSError) as raised:
            read_image_or_substitute(reader, 3, 10, _path_of)

        assert (raised.value, reader.calls) == (error, [3, 3, 3, 4])

    def test_requested_sample_error_after_the_pause_is_raised(self) -> None:
        """The stall ends, but the read of the requested sample after the pause runs out of descriptors."""
        error = OSError(errno.EMFILE, "Too many open files")
        errors = {index: OSError("Input/output error") for index in range(20)}
        reader = _Reader(errors, failures={index: 3 if index == 0 else 1 for index in range(20)})

        def read(index: int) -> str:
            if index == 0 and reader.calls.count(0) == 3:
                raise error
            return reader(index)

        with pytest.raises(OSError) as raised:
            read_image_or_substitute(read, 0, 20, _path_of)

        assert raised.value is error

    def test_decoder_process_that_cannot_start_raises(
        self, tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        error = OSError(errno.EMFILE, "Too many open files")
        mocker.patch.object(io_utils.subprocess, "run", side_effect=error)
        paths = [tmp_path / "image0.png", tmp_path / "image1.png"]
        for path in paths:
            Image.new("RGB", (4, 4)).save(path)

        with pytest.raises(OSError) as raised:
            read_image_or_substitute(
                lambda i: decode_pil_image(paths[i], max_image_pixels=10_000), 0, 2, lambda i: paths[i]
            )

        assert raised.value is error

    def test_decoder_process_error_keeps_its_errno(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
        header = {"error": "OSError", "message": "Too many open files in system", "errno": errno.ENFILE}
        response = json.dumps(header).encode("utf-8") + b"\n"
        mocker.patch.object(
            io_utils.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=response, stderr=b"")
        )

        with pytest.raises(OSError) as raised:
            decode_pil_image_bytes(_two_idat_png()[0], max_image_pixels=10_000)

        assert raised.value.errno == errno.ENFILE

    def test_decoder_process_reports_the_errno(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The worker sends the errno of its own error, here from opening a missing file, back with the message."""
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)

        with pytest.raises(OSError) as raised:
            decode_pil_image(tmp_path / "missing.png", max_image_pixels=10_000, num_channels=1)

        assert raised.value.errno == errno.ENOENT
