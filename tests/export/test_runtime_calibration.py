# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the calibration data shared by the static-quantization export paths.

``calibration_batches`` is the single reader behind the TensorRT INT8 export (and the OpenVINO one), so its contract is
exercised here directly rather than only through an export. Covered: which files a directory yields and in what order,
how an image is preprocessed, what a ``.npy`` file or an array must look like, how a ``.npy`` file is read, and the
refusals that keep a mismatched array from failing later inside a runtime.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from rfdetr.export._runtime import calibration
from rfdetr.export._runtime.calibration import IMAGE_SUFFIXES, _arrays_from_samples, _image_paths, calibration_batches
from rfdetr.export._runtime.preprocess import preprocess_to_nchw

Image = pytest.importorskip("PIL.Image")


def _save_image(path: Path, *, mode: str = "RGB", size: tuple[int, int] = (12, 9), seed: int = 0) -> Path:
    """Write a seeded random image of the given PIL *mode* and ``(width, height)`` *size* to *path*.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     with Image.open(_save_image(Path(tmp) / "a.png", mode="L", size=(5, 3))) as image:
        ...         (image.mode, image.size)
        ('L', (5, 3))
    """
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").convert(mode).save(path)
    return path


def _save_npy(path: Path, shape: tuple[int, ...], dtype: type = np.float32) -> Path:
    """Write a seeded random array of *shape* and *dtype* to *path*.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     np.load(_save_npy(Path(tmp) / "a.npy", (2, 3))).shape
        (2, 3)
    """
    np.save(path, np.random.default_rng(0).standard_normal(shape).astype(dtype))
    return path


class TestImagePaths:
    """A calibration directory yields its images sorted by name, capped at ``max_images``."""

    @pytest.fixture
    def directory(self, tmp_path: Path) -> Path:
        """A directory holding ``c.png``, ``a.jpg``, ``b.bmp`` and a text file that is not an image.

        Examples:
            A pytest fixture cannot be called directly.

            >>> directory(tmp_path)  # doctest: +SKIP
        """
        for name in ("c.png", "a.jpg", "b.bmp"):
            _save_image(tmp_path / name)
        (tmp_path / "notes.txt").write_text("not an image")
        return tmp_path

    @pytest.mark.parametrize(
        ("max_images", "expected"),
        [
            pytest.param(1, ["a.jpg"], id="one"),
            pytest.param(2, ["a.jpg", "b.bmp"], id="two"),
            pytest.param(3, ["a.jpg", "b.bmp", "c.png"], id="all"),
            pytest.param(100, ["a.jpg", "b.bmp", "c.png"], id="more-than-there-are"),
        ],
    )
    def test_images_are_sorted_by_name_and_capped(self, directory: Path, max_images: int, expected: list[str]) -> None:
        assert [p.name for p in _image_paths(directory, max_images)] == expected

    @pytest.mark.parametrize("suffix", sorted(IMAGE_SUFFIXES))
    def test_every_supported_suffix_is_read(self, tmp_path: Path, suffix: str) -> None:
        (tmp_path / f"image{suffix}").write_bytes(b"")
        assert [p.name for p in _image_paths(tmp_path, 5)] == [f"image{suffix}"]

    @pytest.mark.parametrize("name", ["IMAGE.JPG", "image.Png", "image.JPEG"])
    def test_suffix_is_matched_case_insensitively(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name).write_bytes(b"")
        assert [p.name for p in _image_paths(tmp_path, 5)] == [name]

    @pytest.mark.parametrize("name", ["notes.txt", "image.gif", "image.tiff", "image", "image.jpg.bak"])
    def test_other_files_are_not_images(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name).write_bytes(b"")
        with pytest.raises(ValueError, match="No calibration images found"):
            _image_paths(tmp_path, 5)

    def test_empty_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Supported suffixes"):
            _image_paths(tmp_path, 5)


class TestArraysFromSamples:
    """A preprocessed array is checked against the graph, then split into one single-sample batch per row."""

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((3, 8, 8), id="rank-3"),
            pytest.param((8, 8), id="rank-2"),
            pytest.param((1, 1, 3, 8, 8), id="rank-5"),
        ],
    )
    def test_rank_other_than_four_is_refused(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="must be rank 4"):
            list(_arrays_from_samples(np.zeros(shape, np.float32), height=8, width=8, channels=3))

    @pytest.mark.parametrize(("height", "width"), [(8, 6), (6, 8), (16, 16)])
    def test_spatial_mismatch_is_refused(self, height: int, width: int) -> None:
        with pytest.raises(ValueError, match="but the graph expects"):
            list(_arrays_from_samples(np.zeros((1, 3, height, width), np.float32), height=8, width=8, channels=3))

    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_channel_mismatch_is_refused(self, channels: int) -> None:
        with pytest.raises(ValueError, match=f"has {channels} channels"):
            list(_arrays_from_samples(np.zeros((1, channels, 8, 8), np.float32), height=8, width=8, channels=3))

    def test_empty_array_yields_nothing(self) -> None:
        assert list(_arrays_from_samples(np.zeros((0, 3, 8, 8), np.float32), height=8, width=8, channels=3)) == []

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_each_sample_is_a_float32_batch_of_one(self, dtype: type) -> None:
        samples = np.random.default_rng(0).standard_normal((3, 3, 4, 4)).astype(dtype)
        batches = list(_arrays_from_samples(samples, height=4, width=4, channels=3))
        assert [(b.shape, b.dtype) for b in batches] == [((1, 3, 4, 4), np.float32)] * 3
        np.testing.assert_array_equal(np.concatenate(batches), samples.astype(np.float32))

    def test_batch_is_an_owned_writable_copy(self) -> None:
        samples = np.zeros((2, 3, 4, 4), np.float32)
        batch = next(_arrays_from_samples(samples, height=4, width=4, channels=3))
        batch[...] = 1.0
        assert (samples == 0).all() and batch.flags.writeable


class TestCalibrationBatches:
    """``calibration_batches`` accepts a directory, a ``.npy`` path or an array, and refuses what it cannot read."""

    def test_missing_path_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            next(calibration_batches(tmp_path / "missing", height=8, width=8))

    @pytest.mark.parametrize("name", ["images.npz", "images.txt", "images", "images.jpg"])
    def test_file_that_is_not_a_npy_array_is_refused(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name).write_bytes(b"")
        with pytest.raises(ValueError, match="must be a .npy array"):
            next(calibration_batches(tmp_path / name, height=8, width=8))

    @pytest.mark.parametrize("source", ["array", "npy"])
    @pytest.mark.parametrize(
        ("shape", "message"),
        [
            pytest.param((2, 3, 8), "must be rank 4", id="rank"),
            pytest.param((2, 1, 8, 8), "has 1 channels", id="channels"),
            pytest.param((2, 3, 8, 6), "but the graph expects", id="spatial"),
        ],
    )
    def test_array_that_does_not_match_the_graph_is_refused(
        self, tmp_path: Path, source: str, shape: tuple[int, ...], message: str
    ) -> None:
        data: Any = _save_npy(tmp_path / "images.npy", shape) if source == "npy" else np.zeros(shape, np.float32)
        with pytest.raises(ValueError, match=message):
            list(calibration_batches(data, height=8, width=8, channels=3))

    @pytest.mark.parametrize("source", ["array", "npy"])
    def test_array_is_passed_through_unchanged(self, tmp_path: Path, source: str) -> None:
        samples = np.random.default_rng(1).standard_normal((3, 3, 8, 8)).astype(np.float32)
        np.save(tmp_path / "images.npy", samples)
        data: Any = tmp_path / "images.npy" if source == "npy" else samples
        batches = list(calibration_batches(data, height=8, width=8))
        np.testing.assert_array_equal(np.concatenate(batches), samples)

    def test_npy_file_reaches_the_samples_memory_mapped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        path = _save_npy(tmp_path / "images.npy", (2, 3, 8, 8))
        spy = mock.Mock(wraps=calibration._arrays_from_samples)
        monkeypatch.setattr(calibration, "_arrays_from_samples", spy)
        assert len(list(calibration_batches(path, height=8, width=8))) == 2
        assert isinstance(spy.call_args.args[0], np.memmap)

    def test_batches_do_not_alias_the_calibration_file(self, tmp_path: Path) -> None:
        path = _save_npy(tmp_path / "images.npy", (2, 3, 8, 8))
        batch = next(calibration_batches(path, height=8, width=8))
        assert batch.flags.writeable and batch.base is None

    @pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
    def test_directory_images_are_preprocessed_as_predict_does(self, tmp_path: Path, mode: str) -> None:
        path = _save_image(tmp_path / "image.png", mode=mode, size=(20, 14))
        (batch,) = calibration_batches(tmp_path, height=8, width=10)
        with Image.open(path) as image:
            np.testing.assert_array_equal(batch, preprocess_to_nchw(image, height=8, width=10))

    def test_grayscale_graph_gets_one_channel(self, tmp_path: Path) -> None:
        _save_image(tmp_path / "image.png", mode="RGB")
        (batch,) = calibration_batches(tmp_path, height=8, width=8, channels=1)
        assert batch.shape == (1, 1, 8, 8)

    @pytest.mark.parametrize("max_images", [1, 2, 3])
    def test_directory_is_capped_at_max_images(self, tmp_path: Path, max_images: int) -> None:
        for index in range(3):
            _save_image(tmp_path / f"{index}.png", seed=index)
        assert len(list(calibration_batches(tmp_path, height=8, width=8, max_images=max_images))) == max_images

    def test_images_are_read_one_at_a_time(self, tmp_path: Path) -> None:
        _save_image(tmp_path / "0.png")
        (tmp_path / "1.png").write_bytes(b"not an image")
        batches = calibration_batches(tmp_path, height=8, width=8)
        assert next(batches).shape == (1, 3, 8, 8)  # the unreadable second file is only opened by the next step
        assert inspect.getgeneratorstate(batches) == inspect.GEN_SUSPENDED

    def test_empty_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No calibration images found"):
            next(calibration_batches(tmp_path, height=8, width=8))
