# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the opt-in retry and substitution of unreadable images in the loose-file dataset readers.

An ``OSError`` that escapes ``__getitem__`` in a DataLoader worker ends that process's training run; under DDP the other
ranks then wait in their next collective operation until the backend's watchdog aborts the job, without naming the file.
These tests pin the behaviour that keeps one bad read from doing that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.datasets import coco, io_utils, o365, yolo
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.io_utils import read_image_or_substitute
from rfdetr.datasets.yolo import YoloDetection

_requires_simplejpeg = pytest.mark.skipif(io_utils.simplejpeg is None, reason="simplejpeg is not installed")


@pytest.fixture(autouse=True)
def sleep(mocker: Any) -> Any:
    """Replace the retry pause so the tests do not wait, and expose the mock to tests that check the pauses."""
    return mocker.patch.object(io_utils.time, "sleep")


class _Reader:
    """Stands in for a dataset's image read: fails for chosen indices and records every index it is asked for.

    Args:
        failing: Indices whose read always raises.
        exc: Exception raised for a failing index.
        flaky: Number of leading reads that fail, per index, before the read starts to succeed.
    """

    def __init__(
        self, failing: set[int] | None = None, exc: Exception | None = None, flaky: dict[int, int] | None = None
    ) -> None:
        self._failing = failing or set()
        self._exc = exc or OSError("image file is truncated")
        self._flaky = dict(flaky or {})
        self.calls: list[int] = []

    def __call__(self, index: int) -> str:
        self.calls.append(index)
        if index in self._failing:
            raise self._exc
        if self._flaky.get(index, 0) > 0:
            self._flaky[index] -= 1
            raise self._exc
        return f"image-{index}"


def _path_of(index: int) -> str:
    return f"/data/frame{index:06d}.png"


def _read(reader: _Reader, index: int, num_samples: int = 10) -> tuple[int, str]:
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

    def test_survives_failures_up_to_the_attempt_limit(self) -> None:
        reader = _Reader(flaky={3: io_utils._IMAGE_READ_ATTEMPTS - 1})
        assert _read(reader, 3) == (3, "image-3")

    def test_logs_a_warning_naming_the_file_and_no_error(self, mocker: Any) -> None:
        log = mocker.patch.object(io_utils, "logger")
        _read(_Reader(flaky={3: 1}), 3)
        (warning,) = log.warning.call_args_list
        assert (_path_of(3) in warning.args, log.error.called) == (True, False)

    def test_pauses_longer_before_each_retry(self, sleep: Any) -> None:
        _read(_Reader(flaky={3: 2}), 3)
        backoff = io_utils._IMAGE_READ_BACKOFF_SECONDS
        assert [call.args[0] for call in sleep.call_args_list] == [backoff, 2 * backoff]


class TestPersistentFailure:
    """A permanently unreadable image is substituted."""

    def test_substitutes_the_next_index(self) -> None:
        assert _read(_Reader(failing={3}), 3) == (4, "image-4")

    def test_retries_before_substituting(self) -> None:
        reader = _Reader(failing={3})
        _read(reader, 3)
        assert reader.calls.count(3) == io_utils._IMAGE_READ_ATTEMPTS

    def test_skips_consecutive_bad_images(self) -> None:
        assert _read(_Reader(failing={3, 4, 5}), 3) == (6, "image-6")

    def test_wraps_around_the_end_of_the_dataset(self) -> None:
        assert _read(_Reader(failing={3}), 3, num_samples=4) == (0, "image-0")

    def test_logs_one_error_naming_the_file(self, mocker: Any) -> None:
        log = mocker.patch.object(io_utils, "logger")
        _read(_Reader(failing={3}), 3)
        (error,) = log.error.call_args_list
        assert _path_of(3) in error.args


class TestWholesaleFailure:
    """A broken dataset still surfaces as an error instead of silent duplication."""

    def test_raises_when_every_candidate_fails(self) -> None:
        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            _read(_Reader(failing=set(range(100))), 0, num_samples=100)

    def test_error_names_the_first_bad_file(self) -> None:
        with pytest.raises(RuntimeError, match=r"/data/frame000000\.png"):
            _read(_Reader(failing=set(range(100))), 0, num_samples=100)

    def test_tries_a_bounded_number_of_substitutes(self) -> None:
        reader = _Reader(failing=set(range(100)))
        with pytest.raises(RuntimeError, match="unreadable"):
            _read(reader, 0, num_samples=100)
        assert len(set(reader.calls)) == io_utils._MAX_SUBSTITUTIONS + 1

    def test_single_sample_dataset_is_not_substituted_with_itself(self) -> None:
        reader = _Reader(failing={0})
        with pytest.raises(RuntimeError, match="1 consecutive samples were unreadable"):
            _read(reader, 0, num_samples=1)
        assert len(reader.calls) == io_utils._IMAGE_READ_ATTEMPTS


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


def _write_coco_dataset(root: Path, extension: str = "jpg", num_images: int = 4) -> Path:
    """Write ``num_images`` 64 x 48 images, one box each, and return the annotation file.

    Image ``i`` is filled with gray level ``40 * (i + 1)`` and has COCO image id ``i + 1``, so a decoded sample
    identifies its source file.
    """
    root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for index in range(num_images):
        file_name = f"image{index}.{extension}"
        Image.new("RGB", (64, 48), color=(40 * (index + 1),) * 3).save(root / file_name)
        images.append({"id": index + 1, "file_name": file_name, "width": 64, "height": 48})
        annotation = {"id": index + 1, "image_id": index + 1, "category_id": 1, "bbox": [8, 8, 16, 16], "area": 256}
        annotations.append({**annotation, "iscrowd": 0})
    annotation_file = root / "_annotations.coco.json"
    annotation_file.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "object"}]}),
        encoding="utf-8",
    )
    return annotation_file


def _truncate(path: Path) -> None:
    """Cut a file to half its size, as an interrupted copy leaves it."""
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def _gray_level(image: Image.Image) -> int:
    return int(round(float(np.asarray(image).mean()) / 40))


class TestCocoDetectionUnreadableFile:
    """``CocoDetection`` on real files, through both decoders and with and without a draft decode."""

    @pytest.mark.parametrize(
        ("extension", "use_simplejpeg", "draft_size"),
        [
            pytest.param("jpg", True, None, marks=_requires_simplejpeg, id="jpg-simplejpeg"),
            pytest.param("jpg", True, 16, marks=_requires_simplejpeg, id="jpg-simplejpeg-draft"),
            pytest.param("jpg", False, None, id="jpg-pillow"),
            pytest.param("jpg", False, 16, id="jpg-pillow-draft"),
            pytest.param("png", False, None, id="png-pillow"),
        ],
    )
    def test_truncated_file_is_replaced_by_the_next_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str, use_simplejpeg: bool, draft_size: int
    ) -> None:
        if not use_simplejpeg:
            monkeypatch.setattr(io_utils, "simplejpeg", None)
        annotation_file = _write_coco_dataset(tmp_path, extension)
        _truncate(tmp_path / f"image1.{extension}")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, draft_size=draft_size, skip_unreadable=True)

        image, target = dataset[1]

        assert (_gray_level(image), int(target["image_id"])) == (3, 3)

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

        _, target = dataset[3]

        assert int(target["image_id"]) == 1

    def test_length_is_unchanged(self, tmp_path: Path) -> None:
        """Every DDP rank must keep the same batch count, so a bad file does not shorten the dataset."""
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)

        assert len([dataset[index] for index in range(len(dataset))]) == 4

    def test_default_raises_the_decoder_error(self, tmp_path: Path) -> None:
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None)

        with pytest.raises(OSError, match="(?i)truncated"):
            dataset[1]

    def test_default_does_not_retry_or_log(self, tmp_path: Path, mocker: Any, sleep: Any) -> None:
        """Off is the behaviour before the option existed: one read, no pause, nothing logged."""
        annotation_file = _write_coco_dataset(tmp_path)
        _truncate(tmp_path / "image1.jpg")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None)
        decode = mocker.spy(dataset, "_decode_image")
        log = mocker.patch.object(io_utils, "logger")

        with pytest.raises(OSError, match="(?i)truncated"):
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

        with pytest.raises(RuntimeError, match="dataset or the filesystem holding it is broken"):
            dataset[0]


def _write_yolo_dataset(root: Path, num_images: int = 3) -> tuple[Path, Path, Path]:
    """Write ``num_images`` 64 x 48 PNG images, image ``i`` holding ``i + 1`` boxes, and return the three paths."""
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    for index in range(num_images):
        Image.new("RGB", (64, 48), color=(40 * (index + 1),) * 3).save(image_dir / f"image{index}.png")
        (label_dir / f"image{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n" * (index + 1), encoding="utf-8")
    data_file = root / "data.yaml"
    data_file.write_text("names:\n  - object\n", encoding="utf-8")
    return image_dir, label_dir, data_file


def _yolo_dataset(root: Path, **kwargs: Any) -> YoloDetection:
    image_dir, label_dir, data_file = _write_yolo_dataset(root)
    return YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, **kwargs)


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

        with pytest.raises(RuntimeError, match="dataset or the filesystem holding it is broken"):
            dataset[0]


def _builder_args(dataset_dir: Path, **train_kwargs: Any) -> Any:
    return _namespace_from_configs(RFDETRSmallConfig(), TrainConfig(dataset_dir=str(dataset_dir), **train_kwargs))


def _roboflow_yolo_root(root: Path) -> Path:
    for split in ("train", "valid"):
        _write_yolo_dataset(root / split, num_images=1)
    (root / "data.yaml").write_text("names:\n  - object\n", encoding="utf-8")
    return root


def _roboflow_coco_root(root: Path) -> Path:
    for split in ("train", "valid"):
        _write_coco_dataset(root / split, num_images=1)
    return root


class TestBuilderWiring:
    """Builders turn substitution on only when ``TrainConfig`` asks for it, and only for the training split."""

    def test_train_config_default_is_off(self) -> None:
        assert TrainConfig(dataset_dir="/data").skip_unreadable_images is False

    @pytest.mark.parametrize(
        ("image_set", "enabled", "expected"),
        [("train", True, True), ("train", False, False), ("val", True, False)],
    )
    def test_roboflow_coco_builder(self, tmp_path: Path, image_set: str, enabled: bool, expected: bool) -> None:
        args = _builder_args(_roboflow_coco_root(tmp_path), skip_unreadable_images=enabled)
        dataset = coco.build_roboflow_from_coco(image_set, args, resolution=64)
        assert dataset._skip_unreadable is expected

    @pytest.mark.parametrize(
        ("image_set", "enabled", "expected"),
        [("train", True, True), ("train", False, False), ("val", True, False)],
    )
    def test_roboflow_yolo_builder(self, tmp_path: Path, image_set: str, enabled: bool, expected: bool) -> None:
        args = _builder_args(_roboflow_yolo_root(tmp_path), skip_unreadable_images=enabled)
        dataset = yolo.build_roboflow_from_yolo(image_set, args, resolution=64)
        assert dataset._skip_unreadable is expected

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
    ) -> None:
        module, builder = module_and_builder
        dataset_class = mocker.patch.object(module, "CocoDetection")
        args = _builder_args(tmp_path, skip_unreadable_images=enabled)

        getattr(module, builder)(image_set, args, resolution=64)

        assert dataset_class.call_args.kwargs["skip_unreadable"] is expected

    def test_namespace_without_the_field_keeps_it_off(self, tmp_path: Path) -> None:
        """A hand-built namespace that predates the field keeps the ``TrainConfig`` default."""
        args = _builder_args(_roboflow_coco_root(tmp_path))
        del args.skip_unreadable_images
        dataset = coco.build_roboflow_from_coco("train", args, resolution=64)
        assert dataset._skip_unreadable is False
