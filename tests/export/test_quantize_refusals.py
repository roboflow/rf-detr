# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Refusals of a bad INT8 *calibration_data*, and *when* they land.

``quantization="int8"`` reads *calibration_data* only after the forward pass and the conversion, so a mistyped path used
to cost the whole export before it failed. :func:`~rfdetr.export._runtime.calibration_checks.check_calibration_data`
now judges what the configuration alone can tell -- a missing path, a directory without images, a file that is not
``.npy``, an array of the wrong rank -- from both exporters' ``_check_capabilities``. These tests pin the message *and*
the timing: the refusal comes from constructing the exporter, and ``RFDETR.export`` raises it before
``prepare_export_graph`` runs the model.

Nothing here needs ``onnx``, ``onnxruntime``, ``openvino`` or ``nncf``: the checks read metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from numpy.typing import NDArray

from rfdetr.detr import RFDETR
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._openvino.exporter import OpenVINOConfig, OpenVINOExporter
from rfdetr.export._runtime.calibration_checks import check_calibration_data

_EXPORTERS = [
    pytest.param(OnnxExporter, OnnxConfig, "onnx", id="onnx"),
    pytest.param(OpenVINOExporter, OpenVINOConfig, "openvino", id="openvino"),
]

#: Calibration inputs that must be refused, as (attribute of :class:`CalibrationLayout`, message fragment).
_REFUSED = [
    pytest.param(attrgetter("missing"), "does not exist", id="missing-path"),
    pytest.param(attrgetter("empty_dir"), "No calibration images found", id="empty-directory"),
    pytest.param(attrgetter("text_only_dir"), "No calibration images found", id="directory-without-images"),
    pytest.param(attrgetter("sidecar_only_dir"), "No calibration images found", id="directory-of-image-lookalikes"),
    pytest.param(attrgetter("text_file"), r"must be a \.npy array", id="non-npy-file"),
    pytest.param(attrgetter("rank3"), "must be rank 4", id="rank-3-array"),
    pytest.param(attrgetter("raw_pixels"), "must be floating point", id="integer-array"),
    pytest.param(attrgetter("no_samples"), "at least one image", id="empty-array"),
    pytest.param(attrgetter("empty_path"), "calibration_data to be a directory", id="empty-path"),
    pytest.param(attrgetter("image_list"), "calibration_data to be a directory", id="list-of-paths"),
]

#: Calibration inputs that must be accepted: each carries what the check looks for and nothing more.
_ACCEPTED = [
    pytest.param(attrgetter("images_dir"), id="directory-with-an-image"),
    pytest.param(attrgetter("npy_file"), id="npy-file"),
    pytest.param(attrgetter("rank4"), id="rank-4-array"),
]


@dataclass(frozen=True)
class CalibrationLayout:
    """Calibration inputs on disk and in memory, one per case the check tells apart.

    Attributes:
        missing: A path that does not exist.
        empty_dir: A directory with nothing in it.
        text_only_dir: A directory holding only a ``.txt`` file.
        sidecar_only_dir: A directory holding only entries named like images that the reader skips: a macOS
            AppleDouble sidecar (``._a.jpg``) and a subdirectory (``sub.jpg``).
        text_file: A file that is not a ``.npy``.
        images_dir: A directory holding one file with an image suffix (never decoded).
        npy_file: A ``.npy`` file.
        rank3: A rank-3 array.
        rank4: A rank-4 ``(N, C, H, W)`` array.
        raw_pixels: A rank-4 ``uint8`` array: raw pixels, not normalized.
        no_samples: A rank-4 array with no sample in it.
        empty_path: An empty path, which would otherwise name the working directory.
        image_list: A list of image paths, which is not one of the accepted forms.
    """

    missing: Path
    empty_dir: Path
    text_only_dir: Path
    sidecar_only_dir: Path
    text_file: Path
    images_dir: Path
    npy_file: Path
    rank3: NDArray[np.float32]
    rank4: NDArray[np.float32]
    raw_pixels: NDArray[np.uint8]
    no_samples: NDArray[np.float32]
    empty_path: str
    image_list: list[str]


class _ForwardPassReachedError(Exception):
    """Raised by the patched ``prepare_export_graph`` to prove the export got as far as the forward pass."""


@pytest.fixture
def layout(tmp_path: Path) -> CalibrationLayout:
    """Lay out every calibration input the tests need under ``tmp_path``."""
    (tmp_path / "empty").mkdir()
    (tmp_path / "text_only").mkdir()
    (tmp_path / "text_only" / "notes.txt").write_text("not an image")
    (tmp_path / "sidecars").mkdir()
    (tmp_path / "sidecars" / "._a.jpg").write_bytes(b"")
    (tmp_path / "sidecars" / "sub.jpg").mkdir()
    (tmp_path / "images").mkdir()
    # Only the suffix is read at configuration time; the bytes are never decoded, so an empty file is enough.
    (tmp_path / "images" / "frame.jpg").write_bytes(b"")
    (tmp_path / "calibration.txt").write_text("not an array")
    np.save(tmp_path / "calibration.npy", np.zeros((1, 3, 8, 8), dtype=np.float32))
    return CalibrationLayout(
        missing=tmp_path / "nowhere",
        empty_dir=tmp_path / "empty",
        text_only_dir=tmp_path / "text_only",
        sidecar_only_dir=tmp_path / "sidecars",
        text_file=tmp_path / "calibration.txt",
        images_dir=tmp_path / "images",
        npy_file=tmp_path / "calibration.npy",
        rank3=np.zeros((3, 8, 8), dtype=np.float32),
        rank4=np.zeros((1, 3, 8, 8), dtype=np.float32),
        raw_pixels=np.zeros((1, 3, 8, 8), dtype=np.uint8),
        no_samples=np.zeros((0, 3, 8, 8), dtype=np.float32),
        empty_path="",
        image_list=[str(tmp_path / "images" / "frame.jpg")],
    )


@pytest.fixture
def rfdetr_stub() -> Any:
    """Return an ``RFDETR`` with mocked internals, enough to reach ``export``'s exporter construction."""
    obj = RFDETR.__new__(RFDETR)
    obj.model = mock.MagicMock()
    obj.model.resolution = 560
    obj.model.device = "cpu"
    obj.model.model.to.return_value = obj.model.model
    obj.model_config = mock.MagicMock()
    obj.model_config.segmentation_head = False
    obj.model_config.patch_size = 14
    obj.model_config.num_windows = 1
    return obj


@pytest.fixture
def forward_pass() -> Any:
    """Patch ``prepare_export_graph`` to raise :class:`_ForwardPassReachedError`, and return the patch.

    The host check is stubbed out for both formats: the point is the order of the refusals, not which packages the test
    machine has.
    """
    with (
        mock.patch("rfdetr.export.prepare.prepare_export_graph", side_effect=_ForwardPassReachedError) as prepare,
        mock.patch.object(OnnxExporter, "check_dependencies"),
        mock.patch.object(OpenVINOExporter, "check_dependencies"),
    ):
        yield prepare


class TestCheckCalibrationData:
    """The format-free check both exporters share."""

    @pytest.mark.parametrize(("select", "message"), _REFUSED)
    def test_refuses_input_that_cannot_calibrate(self, layout: CalibrationLayout, select: Any, message: str) -> None:
        """Each unusable input is named, so the caller knows which part of it to fix."""
        with pytest.raises(ValueError, match=message):
            check_calibration_data(select(layout))

    @pytest.mark.parametrize("select", _ACCEPTED)
    def test_accepts_input_that_can_calibrate(self, layout: CalibrationLayout, select: Any) -> None:
        """An input of the right kind passes without being read."""
        assert check_calibration_data(select(layout)) is None


class TestExporterConstructionRefuses:
    """The refusal is raised by constructing the exporter, so it precedes anything that traces."""

    @pytest.mark.parametrize(("exporter_class", "config_class", "format_name"), _EXPORTERS)
    @pytest.mark.parametrize(("select", "message"), _REFUSED)
    def test_int8_with_unusable_calibration_data(
        self,
        tmp_path: Path,
        layout: CalibrationLayout,
        exporter_class: Any,
        config_class: Any,
        format_name: str,
        select: Any,
        message: str,
    ) -> None:
        """INT8 with calibration data that cannot calibrate fails at construction, with the reason.

        There is no graph at this point -- construction takes the configuration only -- so nothing can have been traced.
        """
        config = config_class(output_dir=tmp_path, quantization="int8", calibration_data=select(layout))
        with pytest.raises(ValueError, match=message):
            exporter_class(config)

    @pytest.mark.parametrize(("exporter_class", "config_class", "format_name"), _EXPORTERS)
    @pytest.mark.parametrize("select", _ACCEPTED)
    def test_int8_with_usable_calibration_data(
        self,
        tmp_path: Path,
        layout: CalibrationLayout,
        exporter_class: Any,
        config_class: Any,
        format_name: str,
        select: Any,
    ) -> None:
        """Calibration data of the right kind constructs; nothing else about it is judged yet."""
        config = config_class(output_dir=tmp_path, quantization="int8", calibration_data=select(layout))
        assert exporter_class(config).config.calibration_data is select(layout)

    @pytest.mark.parametrize(("exporter_class", "config_class", "format_name"), _EXPORTERS)
    @pytest.mark.parametrize("max_images", [0, -1, True, 2.5])
    def test_int8_with_max_images_that_is_not_a_positive_integer(
        self,
        tmp_path: Path,
        layout: CalibrationLayout,
        exporter_class: Any,
        config_class: Any,
        format_name: str,
        max_images: object,
    ) -> None:
        """A cap that is not a positive integer would silently drop images from a directory, so it is refused."""
        config = config_class(
            output_dir=tmp_path, quantization="int8", calibration_data=layout.images_dir, max_images=max_images
        )
        with pytest.raises(ValueError, match="max_images must be a positive integer"):
            exporter_class(config)

    @pytest.mark.parametrize(("exporter_class", "config_class", "format_name"), _EXPORTERS)
    @pytest.mark.parametrize("quantization", [None, "fp32"])
    def test_float_modes_do_not_read_calibration_data(
        self,
        tmp_path: Path,
        layout: CalibrationLayout,
        exporter_class: Any,
        config_class: Any,
        format_name: str,
        quantization: str | None,
    ) -> None:
        """Only INT8 consumes calibration data, so a bad value beside a float mode is not a refusal."""
        config = config_class(output_dir=tmp_path, quantization=quantization, calibration_data=layout.missing)
        assert exporter_class(config).config.quantization == quantization


class TestExportRefusesBeforeForwardPass:
    """``RFDETR.export`` raises the refusal before ``prepare_export_graph`` runs the model."""

    @pytest.mark.parametrize(("exporter_class", "config_class", "format_name"), _EXPORTERS)
    @pytest.mark.parametrize(("select", "message"), _REFUSED)
    def test_refused_before_prepare_export_graph(
        self,
        tmp_path: Path,
        layout: CalibrationLayout,
        rfdetr_stub: Any,
        forward_pass: mock.MagicMock,
        exporter_class: Any,
        config_class: Any,
        format_name: str,
        select: Any,
        message: str,
    ) -> None:
        """A refused request never reaches the model: the forward pass is where the export starts paying.

        ``prepare_export_graph`` is the full forward pass through the detector; it is patched to raise if reached, so a
        ``ValueError`` carrying *message* can only have come from an earlier stage.
        """
        with pytest.raises(ValueError, match=message):
            rfdetr_stub.export(
                format=format_name,
                output_dir=str(tmp_path / "out"),
                quantization="int8",
                calibration_data=select(layout),
            )
        forward_pass.assert_not_called()

    @pytest.mark.parametrize(("exporter_class", "config_class", "format_name"), _EXPORTERS)
    def test_usable_calibration_data_reaches_the_forward_pass(
        self,
        tmp_path: Path,
        layout: CalibrationLayout,
        rfdetr_stub: Any,
        forward_pass: mock.MagicMock,
        exporter_class: Any,
        config_class: Any,
        format_name: str,
    ) -> None:
        """Control for the test above: a valid request does reach ``prepare_export_graph``.

        Without this, the "not called" assertion could hold only because the patch targets a function ``export`` no
        longer calls.
        """
        with pytest.raises(_ForwardPassReachedError):
            rfdetr_stub.export(
                format=format_name,
                output_dir=str(tmp_path / "out"),
                quantization="int8",
                calibration_data=layout.rank4,
            )
        forward_pass.assert_called_once()
