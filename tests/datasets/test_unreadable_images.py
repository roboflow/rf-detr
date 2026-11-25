# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the opt-in retry and substitution of unreadable images in the loose-file dataset readers.

An ``OSError`` that escapes ``__getitem__`` in a DataLoader worker ends that process's training run, and Pillow's
message does not name the file. Under DDP, Lightning's launcher terminates the other ranks when a child process exits
non-zero; only a failure on rank 0 can leave them blocked in their next collective operation until the process-group
timeout. A YOLO image whose header cannot be read fails earlier still, while the dataset is built and before any
DataLoader worker starts. These tests pin the behaviour that keeps one bad read from ending a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.datasets import coco, io_utils, o365, yolo
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.io_utils import read_image_or_substitute
from rfdetr.datasets.yolo import YoloDetection
from tests._markers import _requires_simplejpeg
from tests.datasets.helpers import (
    _COCO_IMAGE_SIZES,
    _Reader,
    _coco_box,
    _path_of,
    _write_coco_dataset,
    _write_yolo_dataset,
    patch_retry_sleep,
)


sleep = pytest.fixture(autouse=True)(patch_retry_sleep)


def _read(reader: _Reader, index: int, num_samples: int = 10) -> tuple[int, str]:
    """Read sample ``index`` of a ``num_samples`` dataset through :func:`read_image_or_substitute`.

    Examples:
        >>> _read(_Reader(), 3)
        (3, 'image-3')
    """
    return read_image_or_substitute(reader, index, num_samples, _path_of)


class TestReadableImage:
    """A healthy sample is returned directly."""

    def test_returns_the_requested_sample(self) -> None:
        assert _read(_Reader(), 3) == (3, "image-3")

    def test_does_not_retry(self) -> None:
        reader = _Reader()
        _read(reader, 3)
        assert reader.calls == [3]

    def test_logs_nothing(self, mocker: Any) -> None:
        log = mocker.patch.object(io_utils, "logger")
        _read(_Reader(), 3)
        assert log.method_calls == []


class TestTransientFailure:
    """A read that succeeds on retry yields the originally requested sample."""

    def test_returns_the_real_sample_after_one_failure(self) -> None:
        assert _read(_Reader(flaky={3: 1}), 3) == (3, "image-3")

    def test_retries_the_same_index_not_a_neighbour(self) -> None:
        reader = _Reader(flaky={3: 1})
        _read(reader, 3)
        assert reader.calls == [3, 3]

    def test_succeeds_on_the_third_attempt(self) -> None:
        """Three reads of the requested sample are made before it is given up on: two failures, then a success."""
        reader = _Reader(flaky={3: 2})
        assert (_read(reader, 3), reader.calls) == ((3, "image-3"), [3, 3, 3])

    def test_logs_a_warning_naming_the_file_and_no_error(self, mocker: Any) -> None:
        log = mocker.patch.object(io_utils, "logger")
        failure = OSError("image file is truncated")
        _read(_Reader(flaky={3: 1}, exc=failure), 3)
        (warning,) = log.warning.call_args_list
        assert (warning.args, log.error.called) == (
            ("Could not read image %s (attempt %d of %d): %s", _path_of(3), 1, io_utils._IMAGE_READ_ATTEMPTS, failure),
            False,
        )

    def test_pauses_half_a_second_then_a_second_before_the_retries(self, sleep: Any) -> None:
        _read(_Reader(flaky={3: 2}), 3)
        assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


class TestPersistentFailure:
    """A permanently unreadable image is substituted."""

    def test_substitutes_the_next_index(self) -> None:
        assert _read(_Reader(failing={3}), 3) == (4, "image-4")

    def test_reads_the_requested_sample_three_times_before_substituting(self) -> None:
        reader = _Reader(failing={3})
        _read(reader, 3)
        assert reader.calls.count(3) == 3

    def test_pauses_half_a_second_then_a_second_before_substituting(self, sleep: Any) -> None:
        """Only the leading pauses are pinned; a failed sweep of substitutes may pause again."""
        _read(_Reader(failing={3}), 3)
        assert [call.args[0] for call in sleep.call_args_list][:2] == [0.5, 1.0]

    def test_skips_consecutive_bad_images(self) -> None:
        assert _read(_Reader(failing={3, 4, 5}), 3) == (6, "image-6")

    def test_wraps_around_the_end_of_the_dataset(self) -> None:
        assert _read(_Reader(failing={3}), 3, num_samples=4) == (0, "image-0")

    def test_logs_one_error_naming_the_file(self, mocker: Any) -> None:
        log = mocker.patch.object(io_utils, "logger")
        failure = OSError("image file is truncated")
        _read(_Reader(failing={3}, exc=failure), 3)
        (error,) = log.error.call_args_list
        assert error.args == (
            "Image %s (sample %d) is unreadable: %s. Using sample %d (%s) in its place.",
            _path_of(3),
            3,
            failure,
            4,
            _path_of(4),
        )


class TestWholesaleFailure:
    """A broken dataset still surfaces as an error instead of silent duplication."""

    def test_raises_when_every_candidate_fails(self) -> None:
        with pytest.raises(RuntimeError, match="broken|unreadable"):
            _read(_Reader(failing=set(range(100))), 0, num_samples=100)

    def test_error_names_the_first_bad_file(self) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            _read(_Reader(failing=set(range(100))), 0, num_samples=100)
        assert _path_of(0) in str(exc_info.value)

    def test_substitutes_the_tenth_following_sample(self) -> None:
        """The requested sample and the nine after it are unreadable; the tenth following one is the last one tried."""
        assert _read(_Reader(failing=set(range(10))), 0, num_samples=100) == (10, "image-10")

    def test_raises_when_the_ten_following_samples_are_unreadable(self) -> None:
        """The requested sample and the ten after it are unreadable, so every substitute tried fails."""
        with pytest.raises(RuntimeError, match="broken|unreadable"):
            _read(_Reader(failing=set(range(11))), 0, num_samples=100)

    def test_reads_at_most_ten_substitutes(self) -> None:
        reader = _Reader(failing=set(range(100)))
        with pytest.raises(RuntimeError, match="broken|unreadable"):
            _read(reader, 0, num_samples=100)
        assert set(reader.calls) == set(range(11))

    def test_single_sample_dataset_is_not_substituted_with_itself(self) -> None:
        """Three reads with retries, then one more after the pause for a storage stall, and never a substitute."""
        reader = _Reader(failing={0})
        with pytest.raises(RuntimeError, match="broken|unreadable"):
            _read(reader, 0, num_samples=1)
        assert reader.calls == [0, 0, 0, 0]


class TestNonImageErrors:
    """Errors other than ``OSError`` must not be masked by a retry or a substitution."""

    def test_value_error_propagates(self) -> None:
        with pytest.raises(ValueError, match="bad annotation"):
            _read(_Reader(failing={3}, exc=ValueError("bad annotation")), 3)

    def test_value_error_is_not_retried(self) -> None:
        reader = _Reader(failing={3}, exc=ValueError("bad annotation"))
        with pytest.raises(ValueError, match="bad annotation"):
            _read(reader, 3)
        assert reader.calls == [3]

    def test_decompression_bomb_propagates(self) -> None:
        reader = _Reader(failing={3}, exc=Image.DecompressionBombError("too many pixels"))
        with pytest.raises(Image.DecompressionBombError):
            _read(reader, 3)


def _expected_coco_sample(index: int, reduction: int = 1) -> tuple[tuple[int, int], list[float]]:
    """Return the decoded size and the ``[x_min, y_min, x_max, y_max]`` target box of image ``index``.

    Args:
        index: Image index in :func:`_write_coco_dataset`.
        reduction: Factor by which the decoder shrank the image, ``1`` for a full-size decode.

    Examples:
        >>> _expected_coco_sample(2)
        ((128, 112), [12.0, 12.0, 28.0, 32.0])
        >>> _expected_coco_sample(2, reduction=4)
        ((32, 28), [3.0, 3.0, 7.0, 8.0])
    """
    width, height = _COCO_IMAGE_SIZES[index]
    x, y, box_width, box_height = _coco_box(index)
    box = [x, y, x + box_width, y + box_height]
    return (width // reduction, height // reduction), [value / reduction for value in box]


def _truncate(path: Path) -> None:
    """Cut a file to half its size, as an interrupted copy leaves it.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "data.bin"
        ...     _ = path.write_bytes(b"0123456789")
        ...     _truncate(path)
        ...     path.read_bytes()
        b'01234'
    """
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _gray_level(image: Image.Image) -> int:
    """Return ``i + 1`` for an image filled with the gray level ``40 * (i + 1)``, which names its source file.

    Examples:
        >>> _gray_level(Image.new("RGB", (4, 4), color=(120, 120, 120)))
        3
    """
    return int(round(float(np.asarray(image).mean()) / 40))


class TestCocoDetectionUnreadableFile:
    """``CocoDetection`` on real files, through both decoders and with and without a draft decode."""

    # ``reduction`` is the factor the third image, 128 x 112, is shrunk by: a JPEG decoded with ``draft_size=16`` keeps
    # at least 16 pixels on both sides, so the largest power of two that does is 4.  The second image, the one that is
    # truncated, would have been reduced by 2.
    @pytest.mark.parametrize(
        ("extension", "use_simplejpeg", "draft_size", "reduction"),
        [
            pytest.param("jpg", True, None, 1, marks=_requires_simplejpeg, id="jpg-simplejpeg"),
            pytest.param("jpg", True, 16, 4, marks=_requires_simplejpeg, id="jpg-simplejpeg-draft"),
            pytest.param("jpg", False, None, 1, id="jpg-pillow"),
            pytest.param("jpg", False, 16, 4, id="jpg-pillow-draft"),
            pytest.param("png", False, None, 1, id="png-pillow"),
        ],
    )
    def test_truncated_file_is_replaced_by_the_next_sample(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        extension: str,
        use_simplejpeg: bool,
        draft_size: int | None,
        reduction: int,
    ) -> None:
        if not use_simplejpeg:
            monkeypatch.setattr(io_utils, "simplejpeg", None)
        annotation_file = _write_coco_dataset(tmp_path, extension)
        _truncate(tmp_path / f"image1.{extension}")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, draft_size=draft_size, skip_unreadable=True)

        image, target = dataset[1]

        assert (_gray_level(image), int(target["image_id"])) == (3, 3)
        # The box is the third image's, scaled by that image's own decode scale and not by the truncated one's.
        expected_size, expected_box = _expected_coco_sample(2, reduction)
        assert (image.size, target["boxes"].tolist()) == (expected_size, [expected_box])

    def test_truncated_file_logs_one_error_naming_its_path(self, tmp_path: Path, mocker: Any) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)
        error = mocker.patch.object(io_utils.logger, "error")

        dataset[1]

        (call,) = error.call_args_list
        assert str(tmp_path / "image1.jpg") in call.args[0] % call.args[1:]

    def test_missing_file_is_replaced_by_the_next_sample(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        (tmp_path / "image3.jpg").unlink()
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        image, target = dataset[3]

        expected_size, expected_box = _expected_coco_sample(0)
        assert (int(target["image_id"]), image.size, target["boxes"].tolist()) == (1, expected_size, [expected_box])

    def test_length_is_kept_and_the_bad_sample_serves_its_neighbour(self, tmp_path: Path) -> None:
        """Every DDP rank must keep the same batch count, so a bad file does not shorten the dataset."""
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        image_ids = [int(dataset[index][1]["image_id"]) for index in range(len(dataset))]

        assert (len(dataset), image_ids) == (4, [1, 3, 3, 4])

    def test_default_raises_the_decoder_error(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None)

        with pytest.raises(OSError):
            dataset[1]

    def test_default_does_not_retry_or_log(self, tmp_path: Path, mocker: Any, sleep: Any) -> None:
        """Off is the behaviour before the option existed: one read, no pause, nothing logged."""
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None)
        decode = mocker.spy(dataset, "_decode_image")
        log = mocker.patch.object(io_utils, "logger")

        with pytest.raises(OSError):
            dataset[1]

        assert (decode.call_count, sleep.called, log.method_calls) == (1, False, [])

    def test_default_returns_the_same_sample_as_enabled_for_a_readable_file(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        image, target = CocoDetection(tmp_path, annotation_file, transforms=None)[2]
        enabled = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)
        enabled_image, enabled_target = enabled[2]

        assert (image.tobytes(), int(target["image_id"])) == (enabled_image.tobytes(), int(enabled_target["image_id"]))

    def test_broken_annotation_still_raises(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        content = json.loads(annotation_file.read_text(encoding="utf-8"))
        del content["annotations"][1]["bbox"]
        annotation_file.write_text(json.dumps(content), encoding="utf-8")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        with pytest.raises(KeyError, match="bbox"):
            dataset[1]

    def test_transform_os_error_still_raises(self, tmp_path: Path) -> None:
        """Only the image read is tolerated: an ``OSError`` raised after it is not retried or substituted."""
        annotation_file = _write_coco_dataset(tmp_path)

        def failing_transform(image: Any, target: Any) -> Any:
            raise OSError("transform failed")

        dataset = CocoDetection(tmp_path, annotation_file, transforms=failing_transform, skip_unreadable=True)

        with pytest.raises(OSError, match="transform failed"):
            dataset[1]

    def test_all_unreadable_dataset_raises_runtime_error(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        for index in range(4):
            _truncate(tmp_path / f"image{index}.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        with pytest.raises(RuntimeError, match="broken|unreadable"):
            dataset[0]


def _open_yolo(paths: tuple[Path, Path, Path], **kwargs: Any) -> YoloDetection:
    """Open a ``YoloDetection`` on ``(image_dir, label_dir, data_file)``; ``transforms`` defaults to ``None``.

    Examples:
        >>> import contextlib, io, tempfile
        >>> with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        ...     count = len(_open_yolo(_write_yolo_dataset(Path(tmp)), skip_unreadable=True))
        >>> count
        3
    """
    image_dir, label_dir, data_file = paths
    kwargs.setdefault("transforms", None)
    return YoloDetection(str(image_dir), str(label_dir), str(data_file), **kwargs)


def _yolo_dataset(root: Path, **kwargs: Any) -> YoloDetection:
    """Write the dataset of :func:`_write_yolo_dataset` under ``root`` and open it with ``kwargs``.

    Examples:
        >>> import contextlib, io, tempfile
        >>> with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        ...     count = len(_yolo_dataset(Path(tmp)))
        >>> count
        3
    """
    return _open_yolo(_write_yolo_dataset(root), **kwargs)


def _write_yolo_segmentation_dataset(root: Path, num_images: int = 3) -> tuple[Path, Path, Path]:
    """Write the images of :func:`_write_yolo_dataset`, each labelled with one square polygon that grows with its index.

    Image ``i`` carries the polygon spanning ``0.1`` to ``0.1 + 0.2 * (i + 1)`` on both axes, so the box and the mask of
    a sample identify the image they belong to.  At most four images fit inside the unit square.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _, label_dir, _ = _write_yolo_segmentation_dataset(Path(tmp))
        ...     (label_dir / "image1.txt").read_text(encoding="utf-8").split()
        ['0', '0.1', '0.1', '0.5', '0.1', '0.5', '0.5', '0.1', '0.5']
    """
    image_dir, label_dir, data_file = _write_yolo_dataset(root, num_images)
    for index in range(num_images):
        edge = f"{0.1 + 0.2 * (index + 1):.1f}"
        polygon = f"0.1 0.1 {edge} 0.1 {edge} {edge} 0.1 {edge}"
        (label_dir / f"image{index}.txt").write_text(f"0 {polygon}\n", encoding="utf-8")
    return image_dir, label_dir, data_file


def _write_yolo_pose_dataset(root: Path, num_images: int = 3) -> tuple[Path, Path, Path]:
    """Write the images of :func:`_write_yolo_dataset`, each labelled with one pose of two keypoints.

    Image ``i`` carries a visible first keypoint at ``0.2 * (i + 1)`` on both axes and a second one, of visibility 1, at
    ``0.9``, so the keypoints of a sample identify the image they belong to.  At most four images fit inside the unit
    square.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _, label_dir, data_file = _write_yolo_pose_dataset(Path(tmp))
        ...     print((label_dir / "image2.txt").read_text(encoding="utf-8").strip())
        ...     print("kpt_shape: [2, 3]" in data_file.read_text(encoding="utf-8"))
        0 0.5 0.5 0.5 0.5 0.6 0.6 2 0.9 0.9 1
        True
    """
    image_dir, label_dir, data_file = _write_yolo_dataset(root, num_images)
    for index in range(num_images):
        first = f"{0.2 * (index + 1):.1f}"
        pose = f"0.5 0.5 0.5 0.5 {first} {first} 2 0.9 0.9 1"
        (label_dir / f"image{index}.txt").write_text(f"0 {pose}\n", encoding="utf-8")
    data_file.write_text(
        "names:\n  0: person\nkpt_shape: [2, 3]\nkpt_names:\n  0: [left_eye, right_eye]\nflip_idx: [1, 0]\n",
        encoding="utf-8",
    )
    return image_dir, label_dir, data_file


class TestYoloDetectionUnreadableFile:
    """``YoloDetection`` on real files."""

    def test_truncated_file_is_replaced_by_the_next_sample(self, tmp_path: Path) -> None:
        dataset = _yolo_dataset(tmp_path, skip_unreadable=True)
        _truncate(tmp_path / "images" / "image1.png")

        image, target = dataset[1]

        assert (_gray_level(image), int(target["image_id"]), len(target["boxes"])) == (3, 2, 3)

    def test_truncated_file_logs_one_error_naming_its_path(self, tmp_path: Path, mocker: Any) -> None:
        dataset = _yolo_dataset(tmp_path, skip_unreadable=True)
        _truncate(tmp_path / "images" / "image1.png")
        error = mocker.patch.object(io_utils.logger, "error")

        dataset[1]

        (call,) = error.call_args_list
        assert str(tmp_path / "images" / "image1.png") in call.args[0] % call.args[1:]

    def test_default_raises_value_error_naming_the_file(self, tmp_path: Path) -> None:
        dataset = _yolo_dataset(tmp_path)
        _truncate(tmp_path / "images" / "image1.png")

        with pytest.raises(ValueError, match=r"Could not read image from path: .*image1\.png"):
            dataset[1]

    def test_default_does_not_retry_or_log(self, tmp_path: Path, mocker: Any, sleep: Any) -> None:
        """Off is the behaviour before the option existed: one read, no pause, nothing logged."""
        dataset = _yolo_dataset(tmp_path)
        _truncate(tmp_path / "images" / "image1.png")
        decode = mocker.spy(yolo, "decode_image")
        log = mocker.patch.object(io_utils, "logger")

        with pytest.raises(ValueError, match="Could not read image"):
            dataset[1]

        assert (decode.call_count, sleep.called, log.method_calls) == (1, False, [])

    def test_all_unreadable_dataset_raises_runtime_error(self, tmp_path: Path) -> None:
        dataset = _yolo_dataset(tmp_path, skip_unreadable=True)
        for index in range(3):
            _truncate(tmp_path / "images" / f"image{index}.png")

        with pytest.raises(RuntimeError, match="broken|unreadable"):
            dataset[0]

    def test_transform_os_error_still_raises(self, tmp_path: Path, mocker: Any, sleep: Any) -> None:
        """Only the image read is tolerated: an ``OSError`` raised after it is not retried or substituted."""
        transform = mocker.Mock(side_effect=OSError("transform failed"))
        dataset = _open_yolo(_write_yolo_dataset(tmp_path), transforms=transform, skip_unreadable=True)

        with pytest.raises(OSError, match="transform failed"):
            dataset[1]

        assert (transform.call_count, sleep.called) == (1, False)


class TestYoloSegmentationUnreadableFile:
    """``YoloDetection`` with polygon labels: a substituted sample brings its own boxes and masks."""

    def _dataset(self, root: Path) -> YoloDetection:
        """Open a polygon dataset whose second image is truncated after the dataset was built.

        Examples:
            >>> import contextlib, io, tempfile
            >>> with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ...     count = len(TestYoloSegmentationUnreadableFile()._dataset(Path(tmp)))
            >>> count
            3
        """
        image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(root)
        dataset = _open_yolo((image_dir, label_dir, data_file), include_masks=True, skip_unreadable=True)
        _truncate(image_dir / "image1.png")
        return dataset

    def test_truncated_file_is_replaced_by_the_box_of_the_next_sample(self, tmp_path: Path) -> None:
        image, target = self._dataset(tmp_path)[1]

        assert (_gray_level(image), int(target["image_id"])) == (3, 2)
        torch.testing.assert_close(target["boxes"], torch.tensor([[6.4, 4.8, 44.8, 33.6]]))

    def test_truncated_file_is_replaced_by_the_mask_of_the_next_sample(self, tmp_path: Path) -> None:
        dataset = self._dataset(tmp_path)

        _, target = dataset[1]
        _, next_target = dataset[2]
        _, first_target = dataset[0]

        assert torch.equal(target["masks"], next_target["masks"])
        # The masks differ between samples, so the equality above cannot hold by accident.
        assert not torch.equal(target["masks"], first_target["masks"])


class TestYoloPoseUnreadableFile:
    """``YoloDetection`` with pose labels: a substituted sample brings its own keypoints."""

    def _dataset(self, root: Path) -> YoloDetection:
        """Open a pose dataset whose second image is truncated after the dataset was built.

        Examples:
            >>> import contextlib, io, tempfile
            >>> with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ...     count = len(TestYoloPoseUnreadableFile()._dataset(Path(tmp)))
            >>> count
            3
        """
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(root)
        dataset = _open_yolo(
            (image_dir, label_dir, data_file), include_keypoints=True, num_keypoints_per_class=[2], skip_unreadable=True
        )
        _truncate(image_dir / "image1.png")
        return dataset

    def test_truncated_file_is_replaced_by_the_keypoints_of_the_next_sample(self, tmp_path: Path) -> None:
        image, target = self._dataset(tmp_path)[1]

        assert (_gray_level(image), int(target["image_id"])) == (3, 2)
        torch.testing.assert_close(target["keypoints"], torch.tensor([[[38.4, 28.8, 2.0], [57.6, 43.2, 1.0]]]))


def _corrupt_header(path: Path) -> None:
    """Overwrite a file with bytes no decoder identifies, so that even its header cannot be read.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "image.png"
        ...     _corrupt_header(path)
        ...     path.read_bytes()
        b'not an image'
    """
    path.write_bytes(b"not an image")


class TestYoloDetectionUnreadableHeader:
    """``YoloDetection`` reads every image header when it is built, so the corrupt file exists before construction."""

    def _dataset(self, root: Path, **kwargs: Any) -> YoloDetection:
        """Write the YOLO dataset, corrupt the header of its second image, and open it with ``kwargs``.

        Examples:
            >>> import contextlib, io, tempfile
            >>> from unittest import mock
            >>> with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(io_utils.time, "sleep"):
            ...     with tempfile.TemporaryDirectory() as tmp:
            ...         count = len(TestYoloDetectionUnreadableHeader()._dataset(Path(tmp), skip_unreadable=True))
            >>> count
            3
        """
        image_dir, label_dir, data_file = _write_yolo_dataset(root)
        _corrupt_header(image_dir / "image1.png")
        return YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, **kwargs)

    def test_default_raises_from_the_constructor(self, tmp_path: Path, sleep: Any) -> None:
        with pytest.raises(OSError, match="cannot identify image file"):
            self._dataset(tmp_path)
        assert sleep.called is False

    def test_construction_keeps_the_dataset_length(self, tmp_path: Path) -> None:
        assert len(self._dataset(tmp_path, skip_unreadable=True)) == 3

    def test_construction_logs_one_error_naming_the_file(self, tmp_path: Path, mocker: Any) -> None:
        error = mocker.patch.object(yolo.logger, "error")

        self._dataset(tmp_path, skip_unreadable=True)

        (call,) = error.call_args_list
        assert str(tmp_path / "images" / "image1.png") in call.args[0] % call.args[1:]

    def test_unreadable_sample_is_replaced_by_the_next_sample(self, tmp_path: Path) -> None:
        dataset = self._dataset(tmp_path, skip_unreadable=True)

        image, target = dataset[1]

        assert (_gray_level(image), int(target["image_id"]), len(target["boxes"])) == (3, 2, 3)

    @pytest.mark.parametrize("index", [0, 2])
    def test_other_samples_keep_their_index_and_labels(self, tmp_path: Path, index: int) -> None:
        dataset = self._dataset(tmp_path, skip_unreadable=True)

        image, target = dataset[index]

        assert (_gray_level(image), int(target["image_id"]), len(target["boxes"])) == (index + 1, index, index + 1)

    def test_file_repaired_after_construction_is_still_substituted(self, tmp_path: Path) -> None:
        """Its labels were not kept, so returning its pixels would pair a labelled image with no boxes."""
        dataset = self._dataset(tmp_path, skip_unreadable=True)
        Image.new("RGB", (64, 48), color=(80, 80, 80)).save(tmp_path / "images" / "image1.png")

        _, target = dataset[1]

        assert int(target["image_id"]) == 2

    def test_malformed_label_of_the_unreadable_image_still_raises(self, tmp_path: Path) -> None:
        image_dir, label_dir, data_file = _write_yolo_dataset(tmp_path)
        _corrupt_header(image_dir / "image1.png")
        (label_dir / "image1.txt").write_text("7 0.5 0.5 0.25 0.25\n", encoding="utf-8")

        with pytest.raises(ValueError, match="image1.txt"):
            YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)

    def test_header_that_reads_on_retry_keeps_its_labels(self, tmp_path: Path, mocker: Any) -> None:
        image_dir, label_dir, data_file = _write_yolo_dataset(tmp_path)
        read_size = yolo._read_image_size
        failures = [OSError("short read")]

        def flaky_read_size(path: str) -> tuple[int, int]:
            if path.endswith("image1.png") and failures:
                raise failures.pop()
            return read_size(path)

        mocker.patch.object(yolo, "_read_image_size", side_effect=flaky_read_size)
        dataset = YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)

        _, target = dataset[1]

        assert (int(target["image_id"]), len(target["boxes"])) == (1, 2)


def _builder_args(dataset_dir: Path, **train_kwargs: Any) -> Any:
    """Build the argument namespace the dataset builders read; ``train_kwargs`` override the ``TrainConfig`` defaults.

    Examples:
        >>> args = _builder_args(Path("/data"), skip_unreadable_images=True)
        >>> args.skip_unreadable_images, args.square_resize_div_64
        (True, True)
    """
    return _namespace_from_configs(RFDETRNanoConfig(), TrainConfig(dataset_dir=str(dataset_dir), **train_kwargs))


def _roboflow_yolo_root(root: Path) -> Path:
    """Write a one-image ``train`` and ``valid`` split in the Roboflow YOLO layout under ``root``, and return ``root``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     print(sorted(path.name for path in _roboflow_yolo_root(Path(tmp)).iterdir()))
        ['data.yaml', 'train', 'valid']
    """
    for split in ("train", "valid"):
        _write_yolo_dataset(root / split, num_images=1)
    (root / "data.yaml").write_text("names:\n  - object\n", encoding="utf-8")
    return root


def _roboflow_coco_root(root: Path) -> Path:
    """Write a one-image ``train`` and ``valid`` split in the Roboflow COCO layout under ``root``, and return ``root``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     print(sorted(path.name for path in _roboflow_coco_root(Path(tmp)).iterdir()))
        ['train', 'valid']
    """
    for split in ("train", "valid"):
        _write_coco_dataset(root / split, num_images=1)
    return root


class TestBuilderWiring:
    """Builders turn substitution on only when ``TrainConfig`` asks for it, and only for the training split."""

    def test_train_config_default_is_off(self) -> None:
        assert TrainConfig(dataset_dir="/data").skip_unreadable_images is False

    # Every builder constructs its dataset in an ``if square_resize_div_64: ... else: ...`` pair with its own
    # ``skip_unreadable`` keyword, so both resize modes are parametrized to reach each call site.
    @pytest.mark.parametrize("square_resize_div_64", [True, False])
    @pytest.mark.parametrize(
        ("image_set", "enabled", "expected"),
        [("train", True, True), ("train", False, False), ("val", True, False)],
    )
    def test_roboflow_coco_builder(
        self, tmp_path: Path, image_set: str, enabled: bool, expected: bool, square_resize_div_64: bool
    ) -> None:
        args = _builder_args(
            _roboflow_coco_root(tmp_path), skip_unreadable_images=enabled, square_resize_div_64=square_resize_div_64
        )
        dataset = coco.build_roboflow_from_coco(image_set, args, resolution=64)
        assert dataset._skip_unreadable is expected

    @pytest.mark.parametrize("square_resize_div_64", [True, False])
    @pytest.mark.parametrize(
        ("image_set", "enabled", "expected"),
        [("train", True, True), ("train", False, False), ("val", True, False)],
    )
    def test_roboflow_yolo_builder(
        self, tmp_path: Path, image_set: str, enabled: bool, expected: bool, square_resize_div_64: bool
    ) -> None:
        args = _builder_args(
            _roboflow_yolo_root(tmp_path), skip_unreadable_images=enabled, square_resize_div_64=square_resize_div_64
        )
        dataset = yolo.build_roboflow_from_yolo(image_set, args, resolution=64)
        assert dataset._skip_unreadable is expected

    @pytest.mark.parametrize("square_resize_div_64", [True, False])
    @pytest.mark.parametrize(
        ("image_set", "enabled", "expected"),
        [("train", True, True), ("train", False, False), ("val", True, False)],
    )
    @pytest.mark.parametrize(
        "module_and_builder",
        [pytest.param((coco, "build_coco"), id="coco"), pytest.param((o365, "build_o365_raw"), id="o365")],
    )
    def test_fixed_layout_builders(
        self,
        tmp_path: Path,
        mocker: Any,
        module_and_builder: tuple[Any, str],
        image_set: str,
        enabled: bool,
        expected: bool,
        square_resize_div_64: bool,
    ) -> None:
        module, builder = module_and_builder
        dataset_class = mocker.patch.object(module, "CocoDetection")
        args = _builder_args(tmp_path, skip_unreadable_images=enabled, square_resize_div_64=square_resize_div_64)

        getattr(module, builder)(image_set, args, resolution=64)

        assert dataset_class.call_args.kwargs["skip_unreadable"] is expected

    def test_namespace_without_the_field_keeps_it_off(self, tmp_path: Path) -> None:
        """A hand-built namespace that predates the field keeps the ``TrainConfig`` default."""
        args = _builder_args(_roboflow_coco_root(tmp_path))
        del args.skip_unreadable_images
        dataset = coco.build_roboflow_from_coco("train", args, resolution=64)
        assert dataset._skip_unreadable is False
