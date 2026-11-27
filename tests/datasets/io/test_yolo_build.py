# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the build-time handling of unreadable image headers in the YOLO dataset builders.

``skip_unreadable`` makes the builder retry every header read, about 1.5 s of pauses per unreadable file, and keep an
image that stays unreadable as a sample to substitute.  These tests pin the bound that stops a systemically broken
dataset from stalling startup for every image, and the summary the builder logs about the samples it kept.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.datasets import io_utils, yolo
from rfdetr.datasets.io_utils import _MAX_SUBSTITUTIONS
from rfdetr.datasets.yolo import YoloDetection
from tests.datasets.helpers import _patch_retry_sleep

_CORRUPT_BYTES = b"not an image"


sleep = pytest.fixture(autouse=True)(_patch_retry_sleep)


def _write_yolo_images(root: Path, num_images: int, corrupt: set[int] | None = None) -> tuple[Path, Path, Path]:
    """Write ``num_images`` small PNG images with one label each, ``corrupt`` of them with an unreadable header.

    The builder lists images in sorted path order, so the names are zero-padded to keep the sample index of an image
    equal to the index it is written with.

    Args:
        root: Directory that receives ``images/``, ``labels/`` and ``data.yaml``.
        num_images: Number of images ``image000.png`` ... to write.
        corrupt: Indices of the images overwritten with bytes no decoder identifies.

    Returns:
        The image directory, the label directory and the data file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     image_dir, label_dir, data_file = _write_yolo_images(Path(tmp), 3, corrupt={1})
        ...     [(path.name, path.read_bytes() == _CORRUPT_BYTES) for path in sorted(image_dir.iterdir())]
        [('image000.png', False), ('image001.png', True), ('image002.png', False)]
    """
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    for index in range(num_images):
        image_path = image_dir / f"image{index:03d}.png"
        if index in (corrupt or set()):
            image_path.write_bytes(_CORRUPT_BYTES)
        else:
            Image.new("RGB", (16, 12), color=(40,) * 3).save(image_path)
        (label_dir / f"image{index:03d}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data_file = root / "data.yaml"
    data_file.write_text("names:\n  - object\n", encoding="utf-8")
    return image_dir, label_dir, data_file


def _build(root: Path, num_images: int, corrupt: set[int] | None = None, skip_unreadable: bool = True) -> Any:
    """Write a dataset with :func:`_write_yolo_images` and build its lazy detection dataset.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     len(_build(Path(tmp), 3))
        3
    """
    image_dir, label_dir, data_file = _write_yolo_images(root, num_images, corrupt)
    return yolo._build_lazy_yolo_detection_dataset(
        str(image_dir), str(label_dir), str(data_file), skip_unreadable=skip_unreadable
    )


def _readable_flags(dataset: Any) -> list[bool]:
    """Return, in dataset order, whether each sample kept a readable header.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _readable_flags(_build(Path(tmp), 3))
        [True, True, True]
    """
    return [dataset.get_image_info(index).readable for index in range(len(dataset))]


class TestConsecutiveUnreadableHeaders:
    """More than ``_MAX_SUBSTITUTIONS`` unreadable headers in a row mean a broken dataset, which ends the build."""

    def test_raises_once_run_exceeds_substitution_limit(self, tmp_path: Path) -> None:
        """The build fails once more consecutive headers are unreadable than the substitution limit allows."""
        run = set(range(_MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match=f"{_MAX_SUBSTITUTIONS + 1} consecutive samples were unreadable"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

    def test_keeps_run_of_exactly_substitution_limit(self, tmp_path: Path) -> None:
        """A run of unreadable headers exactly at the limit is accepted and marked unreadable."""
        run = set(range(_MAX_SUBSTITUTIONS))

        dataset = _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

        assert _readable_flags(dataset) == [False] * _MAX_SUBSTITUTIONS + [True] * 5

    def test_error_names_first_file_of_run(self, tmp_path: Path) -> None:
        """The error names the sample and file where the unreadable run started."""
        run = set(range(2, 2 + _MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match=r"starting at sample 2 \(.*image002\.png\)"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

    def test_error_wording_matches_runtime_error(self, tmp_path: Path) -> None:
        """The build-time error ends with the sentence ``read_image_or_substitute`` raises for the same run."""
        run = set(range(_MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match="The dataset or the filesystem holding it is broken, not a single file"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

    def test_error_is_chained_from_first_failure(self, tmp_path: Path) -> None:
        """The build error keeps the first header failure as its cause."""
        run = set(range(_MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError) as excinfo:
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

        assert isinstance(excinfo.value.__cause__, OSError)

    def test_stops_reading_headers_at_bound(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """The build stops reading headers as soon as the run limit is passed."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3 * _MAX_SUBSTITUTIONS)
        read_size = mocker.patch.object(yolo, "_read_image_size", side_effect=OSError("header unreadable"))

        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            yolo._build_lazy_yolo_detection_dataset(
                str(image_dir), str(label_dir), str(data_file), skip_unreadable=True
            )

        assert read_size.call_count == (_MAX_SUBSTITUTIONS + 1) * io_utils._IMAGE_READ_ATTEMPTS

    def test_pauses_only_for_headers_up_to_bound(self, tmp_path: Path, mocker: MockerFixture, sleep: Any) -> None:
        """Retry pauses are taken only for the headers read before the limit is passed."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3 * _MAX_SUBSTITUTIONS)
        mocker.patch.object(yolo, "_read_image_size", side_effect=OSError("header unreadable"))

        with pytest.raises(RuntimeError):
            yolo._build_lazy_yolo_detection_dataset(
                str(image_dir), str(label_dir), str(data_file), skip_unreadable=True
            )

        assert sleep.call_count == (_MAX_SUBSTITUTIONS + 1) * (io_utils._IMAGE_READ_ATTEMPTS - 1)

    def test_constructor_raises_for_same_run(self, tmp_path: Path) -> None:
        """YoloDetection raises for the same too-long run of unreadable headers."""
        image_dir, label_dir, data_file = _write_yolo_images(
            tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=set(range(_MAX_SUBSTITUTIONS + 1))
        )

        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)


class TestReadableHeaderResetsTheRun:
    """Only consecutive failures count: a readable header between two runs starts the count again."""

    def test_runs_separated_by_readable_header_do_not_add_up(self, tmp_path: Path) -> None:
        """Two runs below the limit, split by one readable header, are not added together."""
        first_run = set(range(_MAX_SUBSTITUTIONS))
        second_run = set(range(_MAX_SUBSTITUTIONS + 1, 2 * _MAX_SUBSTITUTIONS + 1))

        dataset = _build(tmp_path, 2 * _MAX_SUBSTITUTIONS + 2, corrupt=first_run | second_run)

        assert _readable_flags(dataset).count(False) == 2 * _MAX_SUBSTITUTIONS

    def test_isolated_failures_across_large_dataset_never_trip_bound(self, tmp_path: Path) -> None:
        """Single unreadable headers spread across a large dataset never reach the run limit."""
        num_images = 4 * _MAX_SUBSTITUTIONS
        scattered = set(range(0, num_images, 2))

        dataset = _build(tmp_path, num_images, corrupt=scattered)

        assert len(dataset) == num_images


class TestWholeDatasetUnreadable:
    """A dataset too small for the run bound to trip is still refused when no header can be read."""

    @pytest.mark.parametrize("num_images", [1, 3, _MAX_SUBSTITUTIONS])
    def test_raises_when_every_header_is_unreadable(self, tmp_path: Path, num_images: int) -> None:
        """A dataset with no readable header raises, even when it is smaller than the run limit."""
        with pytest.raises(RuntimeError, match=f"{num_images} consecutive samples were unreadable"):
            _build(tmp_path, num_images, corrupt=set(range(num_images)))

    def test_one_readable_header_is_enough_to_build(self, tmp_path: Path) -> None:
        """One readable header among unreadable ones is enough for the build to succeed."""
        dataset = _build(tmp_path, 3, corrupt={0, 2})

        assert _readable_flags(dataset) == [False, True, False]

    def test_empty_directory_still_builds(self, tmp_path: Path) -> None:
        """An image directory with no images builds an empty dataset."""
        assert len(_build(tmp_path, 0)) == 0


class TestDefaultPathUnchanged:
    """With ``skip_unreadable`` off the first unreadable header still raises its own error, with no retry."""

    def test_raises_decoder_error_for_first_unreadable_header(self, tmp_path: Path, sleep: Any) -> None:
        """With substitution off, the first unreadable header raises the decoder error without any retry."""
        with pytest.raises(OSError, match="cannot identify image file"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=set(range(_MAX_SUBSTITUTIONS + 1)), skip_unreadable=False)

        assert sleep.called is False

    def test_does_not_use_retry_helper(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """With substitution off, the retry helper is never called."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3)
        spy = mocker.spy(yolo, "read_image_with_retries")

        yolo._build_lazy_yolo_detection_dataset(str(image_dir), str(label_dir), str(data_file))

        assert spy.called is False


def _message(call: Any) -> str:
    """Return the text a mocked ``logger`` call would have logged.

    Examples:
        >>> from unittest.mock import call
        >>> _message(call("%d of %d samples", 2, 4))
        '2 of 4 samples'
    """
    return call.args[0] % call.args[1:]


class TestUnreadableSummary:
    """Each unreadable header logs its own error; the builder then logs one line that totals them."""

    def test_warning_counts_unreadable_samples_not_last_run(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """The summary warning counts all unreadable samples, not just the last run."""
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 4, corrupt={0, 2})

        (summary,) = log.warning.call_args_list
        assert "2 of 4 samples" in _message(summary)

    def test_warning_says_samples_stay_substituted_for_run(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """The summary warning says the unreadable samples stay substituted for the rest of the run."""
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 3, corrupt={1})

        (summary,) = log.warning.call_args_list
        assert "stay substituted for the rest of the run" in _message(summary)

    def test_keeps_one_error_per_unreadable_header_and_logs_summary_last(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Each unreadable header logs an error, and the summary warning comes last."""
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 4, corrupt={0, 2})

        assert [name for name, _, _ in log.method_calls] == ["error", "error", "warning"]

    def test_logs_nothing_when_every_header_is_readable(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A dataset with only readable headers logs nothing."""
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 3)

        assert log.method_calls == []

    def test_logs_no_summary_when_build_raises(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A build that fails logs no summary warning."""
        log = mocker.patch.object(yolo, "logger")

        with pytest.raises(RuntimeError):
            _build(tmp_path, 3, corrupt={0, 1, 2})

        assert log.warning.called is False


class TestRunWrapsAroundTheEndOfTheDataset:
    """A read-time substitution wraps around the end of the dataset, so a run at the end continues at the start."""

    def test_raises_when_leading_and_trailing_run_together_exceed_limit(self, tmp_path: Path) -> None:
        """Unreadable images at the start and at the end count as one run that can exceed the limit."""
        num_images = 30
        leading = set(range(5))
        trailing = set(range(num_images - (_MAX_SUBSTITUTIONS + 1 - 5), num_images))

        with pytest.raises(RuntimeError, match=f"{_MAX_SUBSTITUTIONS + 1} consecutive samples were unreadable"):
            _build(tmp_path, num_images, corrupt=leading | trailing)

    def test_error_names_first_sample_of_trailing_run(self, tmp_path: Path) -> None:
        """The error names the first sample of the run at the end of the dataset."""
        num_images = 30
        leading = set(range(5))
        first = num_images - 6
        trailing = set(range(first, num_images))

        with pytest.raises(RuntimeError, match=rf"starting at sample {first} \(.*image{first:03d}\.png\)"):
            _build(tmp_path, num_images, corrupt=leading | trailing)

    def test_keeps_split_run_of_exactly_substitution_limit(self, tmp_path: Path) -> None:
        """A run split between the start and the end is accepted when it is exactly at the limit."""
        num_images = 30
        leading = set(range(4))
        trailing = set(range(num_images - (_MAX_SUBSTITUTIONS - 4), num_images))

        dataset = _build(tmp_path, num_images, corrupt=leading | trailing)

        assert _readable_flags(dataset).count(False) == _MAX_SUBSTITUTIONS

    def test_leading_run_alone_is_not_joined_to_readable_last_sample(self, tmp_path: Path) -> None:
        """A run at the start is not joined to a readable last image."""
        run = set(range(_MAX_SUBSTITUTIONS))

        dataset = _build(tmp_path, 30, corrupt=run)

        assert _readable_flags(dataset) == [False] * _MAX_SUBSTITUTIONS + [True] * (30 - _MAX_SUBSTITUTIONS)

    def test_constructor_raises_for_same_split_run(self, tmp_path: Path) -> None:
        """YoloDetection raises for a run split between the start and the end of the dataset."""
        num_images = 30
        image_dir, label_dir, data_file = _write_yolo_images(
            tmp_path, num_images, corrupt=set(range(5)) | set(range(num_images - 6, num_images))
        )

        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)


def _png_with_truncated_phys(chunk_length: int = 4) -> bytes:
    """Return a small valid PNG whose ``pHYs`` chunk is declared ``chunk_length`` bytes long, fewer than the 9 it needs.

    Pillow reads the chunk while opening the file and raises ``ValueError("Truncated pHYs chunk")`` for it, instead of
    the ``OSError`` it raises for an unidentifiable or truncated file.

    Examples:
        >>> data = _png_with_truncated_phys()
        >>> data.startswith(b"\\x89PNG") and b"pHYs" in data
        True
    """
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color=(40,) * 3).save(buffer, format="PNG")
    data = buffer.getvalue()
    payload = b"\x00" * chunk_length
    chunk = struct.pack(">I", chunk_length) + b"pHYs" + payload + struct.pack(">I", zlib.crc32(b"pHYs" + payload))
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature, then the length, type, data and CRC of the IHDR chunk
    return data[:ihdr_end] + chunk + data[ihdr_end:]


class TestCorruptHeaderRaisingValueError:
    """A header Pillow rejects with ``ValueError`` is as unreadable as one it rejects with ``OSError``."""

    def test_premise_pillow_raises_value_error_for_png(self, tmp_path: Path) -> None:
        """Pillow raises ValueError, not OSError, for the PNG with a too-short pHYs chunk."""
        path = tmp_path / "bad.png"
        path.write_bytes(_png_with_truncated_phys())

        with pytest.raises(ValueError, match="Truncated pHYs chunk"):
            Image.open(path)

    def test_builds_and_marks_sample_unreadable(self, tmp_path: Path) -> None:
        """A header that raises ValueError is marked unreadable and the build still succeeds."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 4)
        (image_dir / "image001.png").write_bytes(_png_with_truncated_phys())

        dataset = yolo._build_lazy_yolo_detection_dataset(
            str(image_dir), str(label_dir), str(data_file), skip_unreadable=True
        )

        assert _readable_flags(dataset) == [True, False, True, True]

    def test_retries_corrupt_header_like_any_unreadable_file(self, tmp_path: Path, sleep: Any) -> None:
        """A header that raises ValueError is read again like any other unreadable header."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3)
        (image_dir / "image000.png").write_bytes(_png_with_truncated_phys())

        yolo._build_lazy_yolo_detection_dataset(str(image_dir), str(label_dir), str(data_file), skip_unreadable=True)

        assert sleep.call_count == io_utils._IMAGE_READ_ATTEMPTS - 1

    def test_counts_toward_run_bound(self, tmp_path: Path) -> None:
        """Headers that raise ValueError count toward the consecutive-run limit."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, _MAX_SUBSTITUTIONS + 5)
        for index in range(_MAX_SUBSTITUTIONS + 1):
            (image_dir / f"image{index:03d}.png").write_bytes(_png_with_truncated_phys())

        with pytest.raises(RuntimeError, match=f"{_MAX_SUBSTITUTIONS + 1} consecutive samples were unreadable"):
            yolo._build_lazy_yolo_detection_dataset(
                str(image_dir), str(label_dir), str(data_file), skip_unreadable=True
            )

    def test_default_path_keeps_decoder_error(self, tmp_path: Path) -> None:
        """With substitution off, the ValueError from the decoder is raised as it is."""
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3)
        (image_dir / "image001.png").write_bytes(_png_with_truncated_phys())

        with pytest.raises(ValueError, match="Truncated pHYs chunk"):
            yolo._build_lazy_yolo_detection_dataset(str(image_dir), str(label_dir), str(data_file))


class TestRetryPauseBudget:
    """Unreadable headers scattered through a large dataset share one pause budget, so they cannot stall the build."""

    num_images = 80
    scattered = set(range(0, 60, 2))

    @staticmethod
    def _retried_headers() -> int:
        """Return how many unreadable headers the budget pays a full retry for."""
        return int(yolo._BUILD_RETRY_BUDGET_SECONDS // yolo._retry_pause_seconds(io_utils._IMAGE_READ_ATTEMPTS))

    def test_total_pause_stays_within_budget(self, tmp_path: Path, sleep: Any) -> None:
        """The total waiting time for scattered unreadable headers stays within the budget."""
        dataset = _build(tmp_path, self.num_images, corrupt=self.scattered)

        assert len(dataset) == self.num_images
        assert 0 < sum(call.args[0] for call in sleep.call_args_list) <= yolo._BUILD_RETRY_BUDGET_SECONDS

    def test_pays_full_retry_for_each_header_budget_covers(self, tmp_path: Path, sleep: Any) -> None:
        """Each unreadable header the budget can pay for gets the full set of retry pauses."""
        assert self._retried_headers() < len(self.scattered)

        _build(tmp_path, self.num_images, corrupt=self.scattered)

        assert sleep.call_count == self._retried_headers() * (io_utils._IMAGE_READ_ATTEMPTS - 1)

    def test_reads_each_header_once_after_budget_is_spent(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Once the budget is spent, each remaining header is read only once."""
        read_size = mocker.spy(yolo, "_read_image_size")

        _build(tmp_path, self.num_images, corrupt=self.scattered)

        retried = self._retried_headers()
        unretried = self.num_images - retried
        assert read_size.call_count == retried * io_utils._IMAGE_READ_ATTEMPTS + unretried

    def test_unretried_headers_are_still_marked_unreadable(self, tmp_path: Path) -> None:
        """Headers read without retries are still marked unreadable."""
        dataset = _build(tmp_path, self.num_images, corrupt=self.scattered)

        assert [not flag for flag in _readable_flags(dataset)] == [
            index in self.scattered for index in range(self.num_images)
        ]

    def test_logs_one_warning_when_budget_is_spent(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """One warning is logged when the retry budget runs out."""
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, self.num_images, corrupt=self.scattered)

        spent = [call for call in log.warning.call_args_list if "budget" in _message(call)]
        assert len(spent) == 1

    def test_logs_no_budget_warning_for_few_unreadable_headers(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """No budget warning is logged when only a few headers are unreadable."""
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 10, corrupt={1, 5})

        assert not any("budget" in _message(call) for call in log.warning.call_args_list)

    def test_long_run_still_ends_build_after_budget_is_spent(self, tmp_path: Path) -> None:
        """A too-long run of unreadable headers still ends the build after the budget is spent."""
        run = set(range(60, 60 + _MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match=f"{_MAX_SUBSTITUTIONS + 1} consecutive samples were unreadable"):
            _build(tmp_path, self.num_images, corrupt=self.scattered | run)
