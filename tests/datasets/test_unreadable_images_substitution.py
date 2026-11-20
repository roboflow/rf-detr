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

import json
import struct
import subprocess
import zlib
from collections.abc import Callable, Mapping
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


@pytest.fixture(autouse=True)
def sleep(mocker: MockerFixture) -> Any:
    """Replace the retry pause so the tests do not wait, and expose the mock to tests that check the pauses."""
    return mocker.patch.object(io_utils.time, "sleep")


class _Reader:
    """Stands in for a dataset's image read: raises a chosen error per index and records every index it is asked for.

    Args:
        errors: Error raised by the read of each failing index.
        failures: Number of leading reads of an index that raise its error; an index left out always raises.

    Examples:
        >>> reader = _Reader({1: OSError("short read")}, failures={1: 1})
        >>> try:
        ...     reader(1)
        ... except OSError as exc:
        ...     print(exc)
        short read
        >>> reader(1), reader(2), reader.calls
        ('image-1', 'image-2', [1, 1, 2])
    """

    def __init__(self, errors: Mapping[int, OSError] | None = None, failures: Mapping[int, int] | None = None) -> None:
        self._errors = dict(errors or {})
        self._failures = dict(failures or {})
        self.calls: list[int] = []

    def __call__(self, index: int) -> str:
        """Return a stand-in image for ``index``, or raise its error while it still has failures left."""
        self.calls.append(index)
        error = self._errors.get(index)
        if error is not None and self._failures.get(index, 1) > 0:
            if index in self._failures:
                self._failures[index] -= 1
            raise error
        return f"image-{index}"


def _path_of(index: int) -> str:
    """Return the stand-in image path of a dataset index.

    Examples:
        >>> _path_of(3)
        '/data/frame000003.png'
    """
    return f"/data/frame{index:06d}.png"


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


def _write_coco_dataset(root: Path, num_images: int = 3) -> Path:
    """Write ``num_images`` 8 x 8 PNG images, image ``i`` with COCO id ``i + 1`` and one box; return the annotations.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     annotation_file = _write_coco_dataset(Path(tmp))
        ...     sorted(path.name for path in Path(tmp).iterdir())
        ['_annotations.coco.json', 'image0.png', 'image1.png', 'image2.png']
    """
    root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for index in range(num_images):
        file_name = f"image{index}.png"
        Image.new("RGB", (8, 8), color=(40 * (index + 1),) * 3).save(root / file_name)
        images.append({"id": index + 1, "file_name": file_name, "width": 8, "height": 8})
        annotations.append(
            {"id": index + 1, "image_id": index + 1, "category_id": 1, "bbox": [2, 2, 4, 4], "area": 16, "iscrowd": 0}
        )
    annotation_file = root / "_annotations.coco.json"
    annotation_file.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "object"}]}),
        encoding="utf-8",
    )
    return annotation_file


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
        [("SyntaxError", OSError), ("FileNotFoundError", FileNotFoundError), ("PermissionError", PermissionError)],
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
        reader = _stalled_reader(20, stalled_reads=2)

        with pytest.raises(RuntimeError):
            read_image_or_substitute(reader, 0, 20, _path_of)

        assert max(reader.calls.count(index) for index in range(1, 20)) == 2

    def test_error_names_a_sustained_storage_outage(self) -> None:
        with pytest.raises(RuntimeError, match="stayed unavailable through every retry, as in a sustained outage"):
            read_image_or_substitute(_stalled_reader(20, stalled_reads=2), 0, 20, _path_of)

    def test_known_unreadable_substitutes_are_not_paused_for(self, sleep: Any) -> None:
        reader = _Reader({index: _KnownUnreadableImageError("unreadable at build") for index in range(20)})

        with pytest.raises(RuntimeError):
            read_image_or_substitute(reader, 0, 20, _path_of)

        assert (sleep.called, len(reader.calls)) == (False, io_utils._MAX_SUBSTITUTIONS + 1)


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


def _write_yolo_dataset(root: Path, num_images: int = 3) -> tuple[Path, Path, Path]:
    """Write ``num_images`` 8 x 8 PNG images, image ``i`` with ``i + 1`` boxes; return the image, label and data paths.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     image_dir, label_dir, data_file = _write_yolo_dataset(Path(tmp))
        ...     sorted(path.name for path in image_dir.iterdir()), (label_dir / "image2.txt").read_text().count("\\n")
        (['image0.png', 'image1.png', 'image2.png'], 3)
    """
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    for index in range(num_images):
        Image.new("RGB", (8, 8), color=(40 * (index + 1),) * 3).save(image_dir / f"image{index}.png")
        (label_dir / f"image{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n" * (index + 1), encoding="utf-8")
    data_file = root / "data.yaml"
    data_file.write_text("names:\n  - object\n", encoding="utf-8")
    return image_dir, label_dir, data_file


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
