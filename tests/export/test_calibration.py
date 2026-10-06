# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for calibration-data loading shared by the INT8 export paths (:mod:`rfdetr.export._runtime.calibration`)."""

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rfdetr.export._openvino.quantize import quantize_int8 as openvino_quantize_int8
from rfdetr.export._runtime import calibration
from rfdetr.export._runtime.calibration import (
    MIN_CALIBRATION_SAMPLES,
    _image_paths,
    calibration_batches,
    warn_if_too_few_samples,
)


@pytest.fixture
def fake_nncf(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stand-in ``nncf`` whose ``quantize`` returns the model unchanged.

    Lets the OpenVINO quantizer's own logic run on hosts without NNCF or OpenVINO.

    Examples:
        A pytest fixture -- it needs ``monkeypatch`` to install the module, so the example is not run:

        >>> fake_nncf(monkeypatch).quantize("model", [])  # doctest: +SKIP
        'model'
    """
    nncf = types.ModuleType("nncf")
    nncf.Dataset = list
    nncf.ModelType = types.SimpleNamespace(TRANSFORMER="TRANSFORMER")
    nncf.quantize = lambda model, dataset, **kwargs: model
    monkeypatch.setitem(sys.modules, "nncf", nncf)
    return nncf


class TestSmallCalibrationSetWarning:
    """The shared floor below which both INT8 paths warn about too little calibration data."""

    @pytest.mark.parametrize(
        ("count", "warned"),
        [
            pytest.param(1, True, id="single-sample"),
            pytest.param(MIN_CALIBRATION_SAMPLES - 1, True, id="just-below-floor"),
            pytest.param(MIN_CALIBRATION_SAMPLES, False, id="at-floor"),
        ],
    )
    def test_warns_only_below_the_floor(self, count: int, warned: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """A warning is logged for fewer than ``MIN_CALIBRATION_SAMPLES`` samples and not at or above it.

        Too few samples gives a model that loads, runs and is quietly less accurate, so the user must hear about it; at
        the floor the log must stay clean so the warning keeps meaning something.
        """
        warnings: list[str] = []
        monkeypatch.setattr(calibration.logger, "warning", warnings.append)
        warn_if_too_few_samples(count)
        assert bool(warnings) is warned

    @pytest.mark.usefixtures("fake_nncf")
    def test_openvino_path_warns_below_the_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two calibration samples trigger the warning on the OpenVINO path too.

        The floor applies to both formats; this pins that the NNCF quantizer consults it.
        """
        warnings: list[str] = []
        monkeypatch.setattr(calibration.logger, "warning", warnings.append)
        openvino_quantize_int8("model", np.zeros((2, 3, 8, 8), dtype=np.float32), height=8, width=8)
        assert len(warnings) == 1


class TestDirectoryEdgeForms:
    """Directory entries that look like calibration images but are not."""

    @pytest.mark.parametrize(
        ("name", "create"),
        [
            pytest.param("x.jpg", Path.mkdir, id="directory-named-like-an-image"),
            pytest.param("._a.jpg", lambda path: path.write_bytes(b"\x00\x05\x16\x07"), id="appledouble-sidecar"),
        ],
    )
    def test_skips_non_image_entries(self, name: str, create: Any, tmp_path: Path) -> None:
        """A subdirectory or AppleDouble sidecar with an image suffix is skipped, not read.

        Either used to abort the whole export after the trace (``IsADirectoryError`` / ``UnidentifiedImageError``); both
        are routine in real folders, especially ones copied from macOS.
        """
        Image.new("RGB", (16, 16)).save(tmp_path / "a.jpg")
        create(tmp_path / name)
        assert len(list(calibration_batches(tmp_path, height=8, width=8))) == 1

    def test_unreadable_image_raises_value_error_naming_the_file(self, tmp_path: Path) -> None:
        """A file with an image suffix that Pillow cannot identify raises ``ValueError`` naming that file.

        ``ValueError`` is the documented contract for bad calibration data; the file name tells the user what to remove.
        """
        (tmp_path / "broken.jpg").write_bytes(b"not an image")
        with pytest.raises(ValueError, match="broken.jpg"):
            list(calibration_batches(tmp_path, height=8, width=8))

    def test_reads_first_images_in_name_order(self, tmp_path: Path) -> None:
        """``max_images`` keeps the alphabetically first files, whatever order the filesystem lists them in.

        A stable choice makes two exports from the same folder calibrate on the same images.
        """
        for name in ("c.png", "a.png", "b.png"):
            (tmp_path / name).write_bytes(b"")
        assert [path.name for path in _image_paths(tmp_path, max_images=2)] == ["a.png", "b.png"]

    def test_normalizes_pixels_like_inference(self, tmp_path: Path) -> None:
        """A solid-colour image becomes the ImageNet-normalized value of that colour in every channel.

        Calibration ranges are only valid if the samples carry the same normalization the deployed model is fed.
        """
        Image.new("RGB", (16, 16), color=(255, 0, 128)).save(tmp_path / "solid.png")
        expected = (np.array([255, 0, 128]) / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch = next(calibration_batches(tmp_path, height=8, width=8))
        np.testing.assert_allclose(batch[0, :, 0, 0], expected, atol=1e-5)


class TestArraySources:
    """How ``.npy`` files and in-memory arrays are read."""

    def test_rejects_array_with_wrong_channel_count(self) -> None:
        """A single-channel array for a three-channel graph is refused, naming the expected shape.

        Only the spatial size used to be checked, so the mismatch surfaced later as an opaque runtime shape error.
        """
        with pytest.raises(ValueError, match=r"\(N, 3, 8, 8\)"):
            list(calibration_batches(np.zeros((2, 1, 8, 8), dtype=np.float32), height=8, width=8, channels=3))

    def test_rejects_integer_array(self) -> None:
        """Raw ``uint8`` pixels are refused: arrays must already be normalized like inference input.

        An un-normalized array calibrates ranges ~100x too wide; the model loads and runs and is quietly wrong.
        """
        with pytest.raises(ValueError, match="floating point"):
            list(calibration_batches(np.zeros((2, 3, 8, 8), dtype=np.uint8), height=8, width=8))

    def test_accepts_half_precision_array_as_float32(self) -> None:
        """A float16 array passes the dtype check and is handed to the quantizer as float32.

        Any floating dtype is a normalized array; only the batch dtype the runtimes consume is fixed.
        """
        batch = next(calibration_batches(np.zeros((1, 3, 8, 8), dtype=np.float16), height=8, width=8))
        assert batch.dtype == np.float32

    @pytest.mark.usefixtures("fake_nncf")
    def test_openvino_path_checks_channels(self) -> None:
        """The OpenVINO quantizer forwards its graph's channel count to the array check.

        The exporter reads the channel count from the traced input; this pins that it reaches the validation.
        """
        with pytest.raises(ValueError, match=r"\(N, 1, 8, 8\)"):
            openvino_quantize_int8("model", np.zeros((2, 3, 8, 8), dtype=np.float32), height=8, width=8, channels=1)

    def test_rejects_pickled_object_npy(self, tmp_path: Path) -> None:
        """A ``.npy`` file holding pickled objects is refused with ``ValueError`` instead of being unpickled.

        Unpickling runs arbitrary code; a calibration file must be plain numeric data.
        """
        array_path = tmp_path / "objects.npy"
        np.save(array_path, np.array([{"payload": 1}], dtype=object), allow_pickle=True)
        with pytest.raises(ValueError, match="objects.npy"):
            list(calibration_batches(array_path, height=8, width=8))

    def test_npy_file_is_memory_mapped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``.npy`` calibration file is opened read-only memory-mapped instead of being read up front.

        Large calibration sets are materialized batch by batch; mapping the file keeps the source array itself off the
        heap, so only one copy of the data is resident.
        """
        array_path = tmp_path / "calib.npy"
        np.save(array_path, np.zeros((2, 3, 8, 8), dtype=np.float32))
        real_load = np.load
        calls: list[dict[str, Any]] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return real_load(*args, **kwargs)

        monkeypatch.setattr(calibration.np, "load", spy)
        list(calibration_batches(array_path, height=8, width=8))
        assert calls[0]["mmap_mode"] == "r"

    def test_array_is_not_capped_by_max_images(self) -> None:
        """An in-memory array is used whole: ``max_images`` only limits directory reads.

        The cap exists to bound image decoding; silently truncating an array the caller prepared would calibrate on a
        different set than the one they passed.
        """
        batches = list(calibration_batches(np.zeros((5, 3, 8, 8), dtype=np.float32), height=8, width=8, max_images=2))
        assert len(batches) == 5
