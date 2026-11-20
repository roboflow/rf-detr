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

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.datasets import io_utils, yolo
from rfdetr.datasets.io_utils import _MAX_SUBSTITUTIONS
from rfdetr.datasets.yolo import YoloDetection

_CORRUPT_BYTES = b"not an image"


@pytest.fixture(autouse=True)
def sleep(mocker: MockerFixture) -> Any:
    """Replace the retry pause so the tests do not wait, and expose the mock to tests that count the pauses."""
    return mocker.patch.object(io_utils.time, "sleep")


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

    def test_raises_once_the_run_exceeds_the_substitution_limit(self, tmp_path: Path) -> None:
        run = set(range(_MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match=f"{_MAX_SUBSTITUTIONS + 1} consecutive samples were unreadable"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

    def test_keeps_a_run_of_exactly_the_substitution_limit(self, tmp_path: Path) -> None:
        run = set(range(_MAX_SUBSTITUTIONS))

        dataset = _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

        assert _readable_flags(dataset) == [False] * _MAX_SUBSTITUTIONS + [True] * 5

    def test_error_names_the_first_file_of_the_run(self, tmp_path: Path) -> None:
        run = set(range(2, 2 + _MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match=r"starting at sample 2 \(.*image002\.png\)"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

    def test_error_wording_matches_the_runtime_error(self, tmp_path: Path) -> None:
        """The build-time error ends with the sentence ``read_image_or_substitute`` raises for the same run."""
        run = set(range(_MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError, match="The dataset or the filesystem holding it is broken, not a single file"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

    def test_error_is_chained_from_the_first_failure(self, tmp_path: Path) -> None:
        run = set(range(_MAX_SUBSTITUTIONS + 1))

        with pytest.raises(RuntimeError) as excinfo:
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=run)

        assert isinstance(excinfo.value.__cause__, OSError)

    def test_stops_reading_headers_at_the_bound(self, tmp_path: Path, mocker: MockerFixture) -> None:
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3 * _MAX_SUBSTITUTIONS)
        read_size = mocker.patch.object(yolo, "_read_image_size", side_effect=OSError("header unreadable"))

        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            yolo._build_lazy_yolo_detection_dataset(
                str(image_dir), str(label_dir), str(data_file), skip_unreadable=True
            )

        assert read_size.call_count == (_MAX_SUBSTITUTIONS + 1) * io_utils._IMAGE_READ_ATTEMPTS

    def test_pauses_only_for_the_headers_up_to_the_bound(
        self, tmp_path: Path, mocker: MockerFixture, sleep: Any
    ) -> None:
        image_dir, label_dir, data_file = _write_yolo_images(tmp_path, 3 * _MAX_SUBSTITUTIONS)
        mocker.patch.object(yolo, "_read_image_size", side_effect=OSError("header unreadable"))

        with pytest.raises(RuntimeError):
            yolo._build_lazy_yolo_detection_dataset(
                str(image_dir), str(label_dir), str(data_file), skip_unreadable=True
            )

        assert sleep.call_count == (_MAX_SUBSTITUTIONS + 1) * (io_utils._IMAGE_READ_ATTEMPTS - 1)

    def test_constructor_raises_for_the_same_run(self, tmp_path: Path) -> None:
        image_dir, label_dir, data_file = _write_yolo_images(
            tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=set(range(_MAX_SUBSTITUTIONS + 1))
        )

        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)


class TestReadableHeaderResetsTheRun:
    """Only consecutive failures count: a readable header between two runs starts the count again."""

    def test_runs_separated_by_a_readable_header_do_not_add_up(self, tmp_path: Path) -> None:
        first_run = set(range(_MAX_SUBSTITUTIONS))
        second_run = set(range(_MAX_SUBSTITUTIONS + 1, 2 * _MAX_SUBSTITUTIONS + 1))

        dataset = _build(tmp_path, 2 * _MAX_SUBSTITUTIONS + 2, corrupt=first_run | second_run)

        assert _readable_flags(dataset).count(False) == 2 * _MAX_SUBSTITUTIONS

    def test_isolated_failures_across_a_large_dataset_never_trip_the_bound(self, tmp_path: Path) -> None:
        num_images = 4 * _MAX_SUBSTITUTIONS
        scattered = set(range(0, num_images, 2))

        dataset = _build(tmp_path, num_images, corrupt=scattered)

        assert len(dataset) == num_images


class TestWholeDatasetUnreadable:
    """A dataset too small for the run bound to trip is still refused when no header can be read."""

    @pytest.mark.parametrize("num_images", [1, 3, _MAX_SUBSTITUTIONS])
    def test_raises_when_every_header_is_unreadable(self, tmp_path: Path, num_images: int) -> None:
        with pytest.raises(RuntimeError, match=f"{num_images} consecutive samples were unreadable"):
            _build(tmp_path, num_images, corrupt=set(range(num_images)))

    def test_one_readable_header_is_enough_to_build(self, tmp_path: Path) -> None:
        dataset = _build(tmp_path, 3, corrupt={0, 2})

        assert _readable_flags(dataset) == [False, True, False]

    def test_an_empty_directory_still_builds(self, tmp_path: Path) -> None:
        assert len(_build(tmp_path, 0)) == 0


class TestDefaultPathUnchanged:
    """With ``skip_unreadable`` off the first unreadable header still raises its own error, with no retry."""

    def test_raises_the_decoder_error_for_the_first_unreadable_header(self, tmp_path: Path, sleep: Any) -> None:
        with pytest.raises(OSError, match="cannot identify image file"):
            _build(tmp_path, _MAX_SUBSTITUTIONS + 5, corrupt=set(range(_MAX_SUBSTITUTIONS + 1)), skip_unreadable=False)

        assert sleep.called is False

    def test_does_not_use_the_retry_helper(self, tmp_path: Path, mocker: MockerFixture) -> None:
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

    def test_warning_counts_the_unreadable_samples_not_the_last_run(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 4, corrupt={0, 2})

        (summary,) = log.warning.call_args_list
        assert "2 of 4 samples" in _message(summary)

    def test_warning_says_the_samples_stay_substituted_for_the_run(self, tmp_path: Path, mocker: MockerFixture) -> None:
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 3, corrupt={1})

        (summary,) = log.warning.call_args_list
        assert "stay substituted for the rest of the run" in _message(summary)

    def test_keeps_one_error_per_unreadable_header_and_logs_the_summary_last(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 4, corrupt={0, 2})

        assert [name for name, _, _ in log.method_calls] == ["error", "error", "warning"]

    def test_logs_nothing_when_every_header_is_readable(self, tmp_path: Path, mocker: MockerFixture) -> None:
        log = mocker.patch.object(yolo, "logger")

        _build(tmp_path, 3)

        assert log.method_calls == []

    def test_logs_no_summary_when_the_build_raises(self, tmp_path: Path, mocker: MockerFixture) -> None:
        log = mocker.patch.object(yolo, "logger")

        with pytest.raises(RuntimeError):
            _build(tmp_path, 3, corrupt={0, 1, 2})

        assert log.warning.called is False
