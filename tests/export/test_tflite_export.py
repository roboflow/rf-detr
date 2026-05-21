# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the ONNX → TFLite export pipeline.

Tests cover:
* ``export_tflite()`` — the main conversion function (mocked ``onnx2tf``)
* ``_check_onnx2tf_available()`` — import-based availability check
* ``_numpy_allow_pickle()`` — NumPy monkey-patch context manager
* ``_patch_validation_download()`` — validation download redirect
* ``_get_onnx_input_info()`` — ONNX model input metadata reader
* ``_prepare_calibration_data()`` — calibration data preparation
* ``format="tflite"`` parameter wiring through ``RFDETR.export()``
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Generator
from unittest import mock

import numpy as np
import pytest

from rfdetr.export._tflite import _IS_ONNX2TF_AVAILABLE
from rfdetr.export._tflite.converter import (
    _DEFAULT_CALIB_SAMPLES,
    _DEFAULT_DIR_CALIB_SAMPLES,
    _IMAGE_EXTENSIONS,
    _VALID_QUANTIZATIONS,
    _check_onnx2tf_available,
    _get_onnx_input_info,
    _load_calibration_images,
    _numpy_allow_pickle,
    _patch_validation_download,
    _prepare_calibration_data,
    export_tflite,
)

onnx2tf_available = pytest.mark.skipif(not _IS_ONNX2TF_AVAILABLE, reason="onnx2tf not installed")

# ---------------------------------------------------------------------------
# Helpers — fake onnx2tf module injected into sys.modules
# ---------------------------------------------------------------------------


class _FakeOnnx2tfModule:
    """Namespace that mimics ``onnx2tf`` for testing."""

    def __init__(self) -> None:
        self.convert = mock.MagicMock()


_ONNX2TF_KEYS = ("onnx2tf", "onnx2tf.onnx2tf", "onnx2tf.utils", "onnx2tf.utils.common_functions")


def _install_fake_onnx2tf() -> tuple[_FakeOnnx2tfModule, mock.MagicMock, dict[str, object]]:
    """Insert a fake ``onnx2tf`` package into ``sys.modules``.

    Saves any pre-existing real modules under the same keys so they can be
    restored by ``_remove_fake_onnx2tf`` (Copilot: do not silently clobber
    real modules that a prior test may have imported).

    Returns:
        Tuple of (fake_module, convert_mock, saved_originals).
    """
    # Snapshot originals before overwriting (None means the key was absent).
    saved: dict[str, object] = {k: sys.modules.get(k) for k in _ONNX2TF_KEYS}

    fake = _FakeOnnx2tfModule()
    pkg = types.ModuleType("onnx2tf")
    pkg.convert = fake.convert  # type: ignore[attr-defined]
    pkg.__version__ = "2.4.0"  # type: ignore[attr-defined]

    # onnx2tf.onnx2tf — force-imported by export_tflite() before patching
    inner_mod = types.ModuleType("onnx2tf.onnx2tf")
    inner_mod.download_test_image_data = mock.MagicMock(  # type: ignore[attr-defined]
        return_value=np.zeros((20, 128, 128, 3), dtype=np.float32),
    )

    # onnx2tf.utils and onnx2tf.utils.common_functions
    utils_mod = types.ModuleType("onnx2tf.utils")
    cf_mod = types.ModuleType("onnx2tf.utils.common_functions")
    cf_mod.download_test_image_data = mock.MagicMock(  # type: ignore[attr-defined]
        return_value=np.zeros((20, 128, 128, 3), dtype=np.float32),
    )

    # Wire up module hierarchy
    pkg.onnx2tf = inner_mod  # type: ignore[attr-defined]
    pkg.utils = utils_mod  # type: ignore[attr-defined]
    utils_mod.common_functions = cf_mod  # type: ignore[attr-defined]

    sys.modules["onnx2tf"] = pkg
    sys.modules["onnx2tf.onnx2tf"] = inner_mod
    sys.modules["onnx2tf.utils"] = utils_mod
    sys.modules["onnx2tf.utils.common_functions"] = cf_mod
    return fake, fake.convert, saved


def _remove_fake_onnx2tf(saved: dict[str, object] | None = None) -> None:
    """Remove fake ``onnx2tf`` entries from ``sys.modules`` and restore originals.

    Args:
        saved: Snapshot returned by ``_install_fake_onnx2tf``.  If a key was
            present before installation its original value is restored; if it
            was absent it is deleted.  When *saved* is ``None`` all
            ``onnx2tf*`` keys are simply deleted (legacy behaviour).
    """
    if saved is not None:
        for key in _ONNX2TF_KEYS:
            original = saved.get(key)
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original  # type: ignore[assignment]
    else:
        for key in list(sys.modules):
            if key == "onnx2tf" or key.startswith("onnx2tf."):
                del sys.modules[key]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_onnx2tf():
    """Provide a fake ``onnx2tf`` that records *convert()* calls.

    Also patches ``_replace_gridsample_for_tflite`` to return the input path
    unchanged so tests that supply stub ONNX bytes do not depend on
    ``onnx.load`` tolerating those bytes.
    """
    fake, convert_mock, saved = _install_fake_onnx2tf()
    with mock.patch(
        "rfdetr.export._tflite.converter._replace_gridsample_for_tflite",
        side_effect=lambda path, _dir: path,
    ):
        yield fake, convert_mock
    _remove_fake_onnx2tf(saved)


@pytest.fixture()
def onnx_model(tmp_path: Path) -> Path:
    """Create a dummy ``.onnx`` file."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"\x08\x06")  # minimal bytes
    return p


@pytest.fixture()
def mock_prepare_calib(tmp_path: Path) -> Generator:
    """Mock ``_prepare_calibration_data`` so dummy ONNX files work.

    ``export_tflite`` calls ``_prepare_calibration_data`` which calls
    ``_get_onnx_input_info`` (requiring a real ONNX file).  Since the
    ``onnx_model`` fixture writes only stub bytes, this mock prevents
    the ONNX parse from being attempted.
    """
    dummy_npy = tmp_path / "_rfdetr_calib_data.npy"
    np.save(str(dummy_npy), np.zeros((1, 64, 64, 3), dtype=np.float32))
    with mock.patch(
        "rfdetr.export._tflite.converter._prepare_calibration_data",
        return_value=dummy_npy,
    ) as m:
        yield m


@pytest.fixture()
def tflite_output(tmp_path: Path, onnx_model: Path) -> Path:
    """Create expected TFLite output file so export_tflite finds it."""
    out = tmp_path / "output"
    out.mkdir()
    (out / f"{onnx_model.stem}_float32.tflite").write_bytes(b"tflite")
    return out


# ---------------------------------------------------------------------------
# TestExportTfliteConverter
# ---------------------------------------------------------------------------


@onnx2tf_available
class TestExportTfliteConverter:
    """Tests for ``export_tflite()``."""

    def test_missing_onnx_raises_file_not_found(self, tmp_path: Path, fake_onnx2tf: Any) -> None:
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            export_tflite(tmp_path / "nope.onnx", tmp_path / "out")

    def test_invalid_quantization_raises_value_error(self, onnx_model: Path, tmp_path: Path, fake_onnx2tf: Any) -> None:
        with pytest.raises(ValueError, match="Unsupported quantization"):
            export_tflite(onnx_model, tmp_path / "out", quantization="q4")

    @pytest.mark.parametrize(
        "static_mode",
        [
            pytest.param("int8_static", id="int8_static"),
            pytest.param("full_int8", id="full_int8"),
            pytest.param("integer_quant", id="integer_quant"),
        ],
    )
    def test_static_int8_raises(self, onnx_model: Path, tmp_path: Path, fake_onnx2tf: Any, static_mode: str) -> None:
        """A static / full-integer INT8 request must raise a ValueError.

        Static INT8 is intentionally unsupported; only dynamic-range 'int8' is offered.
        """
        with pytest.raises(ValueError, match="[Ss]tatic / full-integer INT8 is not supported"):
            export_tflite(onnx_model, tmp_path / "out", quantization=static_mode)

    def test_default_quantization_calls_convert(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        _, convert_mock = fake_onnx2tf
        result = export_tflite(onnx_model, tflite_output)

        convert_mock.assert_called_once()
        kwargs = convert_mock.call_args.kwargs
        assert kwargs["input_onnx_file_path"] == str(onnx_model)
        assert kwargs["output_folder_path"] == str(tflite_output)
        assert kwargs["output_signaturedefs"] is True
        assert kwargs["non_verbose"] is True
        assert "output_integer_quantized_tflite" not in kwargs
        assert result == tflite_output / "model_float32.tflite"

    def test_custom_input_not_passed_to_convert(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """custom_input_op_name_np_data_path must NOT be passed to convert().

        The onnx2tf custom_input code path triggers a tf.tile rank mismatch
        with DINOv2-style backbones when N > 1.  We rely on patching
        download_test_image_data() instead.
        """
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output)

        kwargs = convert_mock.call_args.kwargs
        assert "custom_input_op_name_np_data_path" not in kwargs

    def test_output_signaturedefs_always_enabled(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """output_signaturedefs must always be True.

        Segmentation models produce ONNX node names with leading "/"
        characters that violate the TF saved_model naming pattern.
        Enabling signature defs bypasses this restriction.
        """
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output)

        kwargs = convert_mock.call_args.kwargs
        assert kwargs["output_signaturedefs"] is True

    def test_tflite_backend_forced_to_tf_converter(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """tflite_backend must always be 'tf_converter' to avoid the TFLite TopK_V2 kernel check.

        onnx2tf 2.x defaults to flatbuffer_direct, which trips a
        "k > internal dimension" error at AllocateTensors() time on
        RF-DETR's encoder TopK node.  tf_converter is forced unconditionally.
        """
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output)

        assert convert_mock.call_args.kwargs["tflite_backend"] == "tf_converter"

    def test_replace_to_pseudo_operators_contains_erf_and_gelu(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """replace_to_pseudo_operators must include Erf and GeLU.

        Without this, AllocateTensors() fails with "FlexErf failed to prepare"
        because the TFLite runtime lacks native Erf / GeLU kernels.
        """
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output)

        pseudo_ops = convert_mock.call_args.kwargs.get("replace_to_pseudo_operators", [])
        assert "Erf" in pseudo_ops
        assert "GeLU" in pseudo_ops

    def test_fp32_quantization_no_int8_flag(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output, quantization="fp32")
        assert "output_integer_quantized_tflite" not in convert_mock.call_args.kwargs

    def test_fp16_quantization_no_int8_flag(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output, quantization="fp16")
        assert "output_integer_quantized_tflite" not in convert_mock.call_args.kwargs

    def test_int8_quantization_produces_dynamic_range(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """int8 export derives a dynamic-range model and avoids onnx2tf's -oiqt path.

        onnx2tf's ``output_integer_quantized_tflite`` (-oiqt) only yields static
        quantization, which RF-DETR's transformer activations do not survive.
        The converter instead builds dynamic-range INT8 from the SavedModel via
        ``_quantize_dynamic_range``, so the onnx2tf call must NOT carry the
        ``output_integer_quantized_tflite`` flag.
        """
        _, convert_mock = fake_onnx2tf
        dyn_path = tflite_output / "model_dynamic_range_quant.tflite"
        with mock.patch(
            "rfdetr.export._tflite.converter._quantize_dynamic_range",
            return_value=dyn_path,
        ) as quant_mock:
            result = export_tflite(onnx_model, tflite_output, quantization="int8")
        assert "output_integer_quantized_tflite" not in convert_mock.call_args.kwargs
        quant_mock.assert_called_once()
        assert result == dyn_path

    def test_verbosity_forwarded(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        _, convert_mock = fake_onnx2tf
        export_tflite(onnx_model, tflite_output, verbosity="debug")
        assert convert_mock.call_args.kwargs["verbosity"] == "debug"

    def test_convert_failure_raises_runtime_error(
        self,
        onnx_model: Path,
        tmp_path: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        _, convert_mock = fake_onnx2tf
        convert_mock.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="onnx2tf conversion failed"):
            export_tflite(onnx_model, tmp_path / "out")

    def test_fallback_when_primary_tflite_missing(
        self,
        onnx_model: Path,
        tmp_path: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """Fallback returns a stem-scoped file when the primary *_float32.tflite is absent."""
        out = tmp_path / "out"
        out.mkdir()
        # Scoped fallback: must match {stem}_*.tflite (stem == "model" here).
        (out / "model_float16.tflite").write_bytes(b"fb")
        result = export_tflite(onnx_model, out)
        assert result.name == "model_float16.tflite"

    def test_fallback_does_not_return_unrelated_tflite(
        self,
        onnx_model: Path,
        tmp_path: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """Stale artifacts from a different export are never returned as fallback."""
        out = tmp_path / "out"
        out.mkdir()
        # Unrelated file — does NOT match model_*.tflite.
        (out / "other_model.tflite").write_bytes(b"stale")
        with pytest.raises(RuntimeError, match="no .tflite file matching"):
            export_tflite(onnx_model, out)

    def test_no_tflite_output_raises_runtime_error(
        self,
        onnx_model: Path,
        tmp_path: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """Empty output directory raises RuntimeError after conversion."""
        out = tmp_path / "empty_out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="no .tflite file matching"):
            export_tflite(onnx_model, out)

    def test_returns_path_object(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        result = export_tflite(onnx_model, tflite_output)
        assert isinstance(result, Path)

    def test_calibration_data_forwarded_to_prepare(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """Verify that calibration_data is passed to _prepare_calibration_data."""
        calib_path = "/some/calib.npy"
        export_tflite(onnx_model, tflite_output, calibration_data=calib_path)
        call_args = mock_prepare_calib.call_args
        assert call_args[0][1] == calib_path  # second positional arg

    def test_max_images_forwarded_to_prepare(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """Verify that max_images is passed to _prepare_calibration_data."""
        export_tflite(onnx_model, tflite_output, max_images=42)
        call_kwargs = mock_prepare_calib.call_args
        assert call_kwargs.kwargs.get("max_images") == 42

    def test_max_images_default_is_100(
        self,
        onnx_model: Path,
        tflite_output: Path,
        fake_onnx2tf: Any,
        mock_prepare_calib: Any,
    ) -> None:
        """Verify that max_images defaults to 100 when not specified."""
        export_tflite(onnx_model, tflite_output)
        call_kwargs = mock_prepare_calib.call_args
        assert call_kwargs.kwargs.get("max_images") == 100

    def test_valid_quantizations_set(self) -> None:
        assert _VALID_QUANTIZATIONS == {None, "fp32", "fp16", "int8"}


# ---------------------------------------------------------------------------
# TestExportFormatParameter
# ---------------------------------------------------------------------------


class TestExportFormatParameter:
    """Tests for ``format="tflite"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> None:
        """Mock the heavy export dependencies so ``RFDETR.export()`` is fast."""
        self._tmp_path = tmp_path
        onnx_out = tmp_path / "inference_model.onnx"
        onnx_out.write_bytes(b"onnx")

        import contextlib

        self._mock_stack = contextlib.ExitStack()

        # Mock export_onnx to return a fake ONNX file path
        self._mock_export_onnx = self._mock_stack.enter_context(
            mock.patch("rfdetr.export.main.export_onnx", return_value=str(onnx_out))
        )
        # Mock make_infer_image to return a small tensor
        import torch

        self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export.main.make_infer_image",
                return_value=torch.zeros(1, 3, 560, 560),
            )
        )
        # Mock export_tflite
        self._mock_export_tflite = self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export._tflite.converter.export_tflite",
                return_value=tmp_path / "inference_model_float32.tflite",
            )
        )
        yield
        self._mock_stack.close()

    @staticmethod
    def _make_rfdetr() -> Any:
        """Create a minimal RFDETR instance with mocked internals."""
        from rfdetr.detr import RFDETR

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

    def test_tflite_format_calls_export_tflite(self) -> None:
        obj = self._make_rfdetr()
        obj.export(format="tflite", output_dir=str(self._tmp_path / "out"))
        self._mock_export_tflite.assert_called_once()

    def test_onnx_format_does_not_call_export_tflite(self) -> None:
        obj = self._make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_export_tflite.assert_not_called()

    def test_quantization_forwarded(self) -> None:
        obj = self._make_rfdetr()
        obj.export(
            format="tflite",
            output_dir=str(self._tmp_path / "out"),
            quantization="int8",
        )
        call_kwargs = self._mock_export_tflite.call_args
        assert call_kwargs[1].get("quantization") == "int8" or call_kwargs.kwargs.get("quantization") == "int8"

    @pytest.mark.parametrize(
        "quant",
        [
            pytest.param(None, id="none"),
            pytest.param("fp32", id="fp32"),
            pytest.param("fp16", id="fp16"),
            pytest.param("int8", id="int8"),
        ],
    )
    def test_all_quantization_modes_accepted(self, quant: str | None) -> None:
        obj = self._make_rfdetr()
        obj.export(
            format="tflite",
            output_dir=str(self._tmp_path / "out"),
            quantization=quant,
        )
        self._mock_export_tflite.assert_called_once()

    def test_unsupported_format_raises(self) -> None:
        obj = self._make_rfdetr()
        with pytest.raises(ValueError, match="[Uu]nsupported.*format"):
            obj.export(format="banana", output_dir=str(self._tmp_path / "out"))

    def test_calibration_data_forwarded(self) -> None:
        """Verify calibration_data kwarg reaches export_tflite."""
        obj = self._make_rfdetr()
        calib = "/my/calib.npy"
        obj.export(
            format="tflite",
            output_dir=str(self._tmp_path / "out"),
            calibration_data=calib,
        )
        call_kwargs = self._mock_export_tflite.call_args
        assert call_kwargs[1].get("calibration_data") == calib or call_kwargs.kwargs.get("calibration_data") == calib

    def test_max_images_forwarded(self) -> None:
        """Verify max_images kwarg reaches export_tflite."""
        obj = self._make_rfdetr()
        obj.export(
            format="tflite",
            output_dir=str(self._tmp_path / "out"),
            max_images=50,
        )
        call_kwargs = self._mock_export_tflite.call_args
        assert call_kwargs[1].get("max_images") == 50 or call_kwargs.kwargs.get("max_images") == 50

    def test_max_images_default_is_100(self) -> None:
        """Verify max_images defaults to 100 when not specified."""
        obj = self._make_rfdetr()
        obj.export(
            format="tflite",
            output_dir=str(self._tmp_path / "out"),
        )
        call_kwargs = self._mock_export_tflite.call_args
        assert call_kwargs[1].get("max_images") == 100 or call_kwargs.kwargs.get("max_images") == 100


# ---------------------------------------------------------------------------
# TestNumpyAllowPickle
# ---------------------------------------------------------------------------


class TestNumpyAllowPickle:
    """Tests for ``_numpy_allow_pickle()`` context manager."""

    def test_patches_np_load(self) -> None:
        original = np.load
        with _numpy_allow_pickle():
            assert np.load is not original
        assert np.load is original

    def test_sets_allow_pickle_default(self) -> None:
        calls: list[dict[str, Any]] = []
        original = np.load

        def _spy(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs.copy())
            raise ValueError("stop")

        np.load = _spy  # type: ignore[assignment]
        try:
            with _numpy_allow_pickle():
                with pytest.raises(ValueError, match="stop"):
                    np.load("dummy.npy")
            assert calls[0].get("allow_pickle") is True
        finally:
            np.load = original  # type: ignore[assignment]

    def test_restores_on_exception(self) -> None:
        original = np.load
        try:
            with _numpy_allow_pickle():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert np.load is original


# ---------------------------------------------------------------------------
# TestPatchValidationDownload
# ---------------------------------------------------------------------------


class TestPatchValidationDownload:
    """Tests for ``_patch_validation_download()`` context manager."""

    def test_patches_download_in_common_functions(self, tmp_path: Path, fake_onnx2tf: Any) -> None:
        """The patch replaces download_test_image_data in common_functions."""
        data = np.random.rand(5, 128, 128, 3).astype(np.float32)
        npy_path = tmp_path / "calib.npy"
        np.save(str(npy_path), data)

        cf_mod = sys.modules["onnx2tf.utils.common_functions"]
        original_fn = cf_mod.download_test_image_data

        with _patch_validation_download(str(npy_path)):
            assert cf_mod.download_test_image_data is not original_fn
            result = cf_mod.download_test_image_data()
            np.testing.assert_array_equal(result, data)

        # Restored after exit
        assert cf_mod.download_test_image_data is original_fn

    def test_patches_download_in_onnx2tf_module(self, tmp_path: Path, fake_onnx2tf: Any) -> None:
        """The patch also covers onnx2tf.onnx2tf if it has the function."""
        data = np.random.rand(3, 64, 64, 3).astype(np.float32)
        npy_path = tmp_path / "calib.npy"
        np.save(str(npy_path), data)

        # Create onnx2tf.onnx2tf submodule with download_test_image_data
        onnx2tf_inner = types.ModuleType("onnx2tf.onnx2tf")
        original_fn = mock.MagicMock()
        onnx2tf_inner.download_test_image_data = original_fn  # type: ignore[attr-defined]
        sys.modules["onnx2tf.onnx2tf"] = onnx2tf_inner

        try:
            with _patch_validation_download(str(npy_path)):
                assert onnx2tf_inner.download_test_image_data is not original_fn
                result = onnx2tf_inner.download_test_image_data()
                np.testing.assert_array_equal(result, data)

            assert onnx2tf_inner.download_test_image_data is original_fn
        finally:
            sys.modules.pop("onnx2tf.onnx2tf", None)

    def test_restores_on_exception(self, tmp_path: Path, fake_onnx2tf: Any) -> None:
        """Functions are restored even when an exception occurs."""
        npy_path = tmp_path / "calib.npy"
        np.save(str(npy_path), np.zeros((1, 8, 8, 3), dtype=np.float32))

        cf_mod = sys.modules["onnx2tf.utils.common_functions"]
        original_fn = cf_mod.download_test_image_data

        try:
            with _patch_validation_download(str(npy_path)):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert cf_mod.download_test_image_data is original_fn

    def test_skips_missing_modules(self, tmp_path: Path) -> None:
        """Does not raise when onnx2tf modules are not in sys.modules."""
        npy_path = tmp_path / "calib.npy"
        np.save(str(npy_path), np.zeros((1, 8, 8, 3), dtype=np.float32))

        # Ensure onnx2tf is NOT in sys.modules
        keys = [k for k in sys.modules if k == "onnx2tf" or k.startswith("onnx2tf.")]
        saved = {k: sys.modules.pop(k) for k in keys}

        try:
            with _patch_validation_download(str(npy_path)):
                pass  # should not raise
        finally:
            sys.modules.update(saved)


class TestGetOnnxInputInfo:
    """Tests for ``_get_onnx_input_info()``."""

    def test_reads_input_name_and_shape(self, tmp_path: Path) -> None:
        """Build a minimal ONNX model and verify we read back its metadata."""
        onnx = pytest.importorskip("onnx", reason="onnx not installed")
        TensorProto, helper = onnx.TensorProto, onnx.helper  # noqa: N806

        inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 560, 560])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 100, 4])
        node = helper.make_node("Identity", inputs=["images"], outputs=["output"])
        graph = helper.make_graph([node], "test", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(onnx_path))

        name, dims = _get_onnx_input_info(onnx_path)
        assert name == "images"
        assert dims == [1, 3, 560, 560]

    def test_different_input_shape(self, tmp_path: Path) -> None:
        """Verify non-square resolution reads correctly."""
        onnx = pytest.importorskip("onnx", reason="onnx not installed")
        TensorProto, helper = onnx.TensorProto, onnx.helper  # noqa: N806

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 448, 640])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 4])
        node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
        graph = helper.make_graph([node], "test", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(onnx_path))

        name, dims = _get_onnx_input_info(onnx_path)
        assert name == "input"
        assert dims == [1, 3, 448, 640]


# ---------------------------------------------------------------------------
# TestPrepareCalibrationData
# ---------------------------------------------------------------------------


class TestPrepareCalibrationData:
    """Tests for ``_prepare_calibration_data()``."""

    @pytest.fixture()
    def _mock_onnx_info(self) -> Generator:
        """Mock ``_get_onnx_input_info`` to return a known shape."""
        with mock.patch(
            "rfdetr.export._tflite.converter._get_onnx_input_info",
            return_value=("input", [1, 3, 256, 256]),
        ):
            yield

    def test_none_generates_random_data(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")

        npy_path = _prepare_calibration_data(onnx_path, None, tmp_path, "fp32")

        assert isinstance(npy_path, Path)
        assert npy_path.is_file()

        data = np.load(str(npy_path))
        assert data.shape == (_DEFAULT_CALIB_SAMPLES, 256, 256, 3)
        assert data.dtype == np.float32

    def test_none_int8_emits_warning(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")

        with mock.patch("rfdetr.export._tflite.converter.logger") as mock_logger:
            _prepare_calibration_data(onnx_path, None, tmp_path, "int8")
            mock_logger.warning.assert_called_once()
            assert "INT8" in mock_logger.warning.call_args[0][0]

    def test_ndarray_saves_to_npy(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")
        calib = np.random.rand(10, 256, 256, 3).astype(np.float32)

        npy_path = _prepare_calibration_data(onnx_path, calib, tmp_path, "fp32")

        loaded = np.load(str(npy_path))
        np.testing.assert_array_equal(loaded, calib)

    def test_path_string_used_directly(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")
        npy_file = tmp_path / "my_calib.npy"
        np.save(str(npy_file), np.zeros((5, 256, 256, 3), dtype=np.float32))

        npy_path = _prepare_calibration_data(onnx_path, str(npy_file), tmp_path, "fp32")

        assert npy_path == npy_file

    def test_directory_loads_images(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        """A directory path triggers image loading and .npy creation."""
        from PIL import Image

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(5):
            img = Image.new("RGB", (100, 80), color=(i * 50, 0, 0))
            img.save(img_dir / f"img_{i:03d}.jpg")

        npy_path = _prepare_calibration_data(onnx_path, str(img_dir), tmp_path, "int8")

        assert npy_path.is_file()
        data = np.load(str(npy_path))
        # _mock_onnx_info returns [1, 3, 256, 256] → H=256, W=256
        assert data.shape == (5, 256, 256, 3)
        assert data.dtype == np.float32

    def test_directory_respects_max_images(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        """Verify that max_images limits the number of images loaded from a directory."""
        from PIL import Image

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(10):
            img = Image.new("RGB", (100, 80), color=(i * 25, 0, 0))
            img.save(img_dir / f"img_{i:03d}.jpg")

        npy_path = _prepare_calibration_data(onnx_path, str(img_dir), tmp_path, "int8", max_images=3)

        assert npy_path.is_file()
        data = np.load(str(npy_path))
        assert data.shape[0] == 3  # only 3 of 10 images loaded

    def test_missing_path_raises_file_not_found(self, tmp_path: Path, _mock_onnx_info: None) -> None:
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"\x00")

        with pytest.raises(FileNotFoundError, match="Calibration data path not found"):
            _prepare_calibration_data(onnx_path, "/nonexistent/calib.npy", tmp_path, "fp32")


# ---------------------------------------------------------------------------
# TestLoadCalibrationImages
# ---------------------------------------------------------------------------


class TestLoadCalibrationImages:
    """Tests for ``_load_calibration_images()``."""

    @staticmethod
    def _make_images(directory: Path, count: int = 5, size: tuple[int, int] = (100, 80)) -> list[Path]:
        """Create *count* small JPEG images in *directory*."""
        from PIL import Image

        paths: list[Path] = []
        for i in range(count):
            p = directory / f"img_{i:04d}.jpg"
            Image.new("RGB", size, color=(i * 40 % 256, 0, 0)).save(p)
            paths.append(p)
        return paths

    def test_loads_images_with_correct_shape(self, tmp_path: Path) -> None:
        self._make_images(tmp_path, count=3)
        result = _load_calibration_images(tmp_path, height=128, width=256)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 128, 256, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_respects_max_images(self, tmp_path: Path) -> None:
        self._make_images(tmp_path, count=10)
        result = _load_calibration_images(tmp_path, height=64, width=64, max_images=4)
        assert result.shape[0] == 4

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No supported image files"):
            _load_calibration_images(empty, height=64, width=64)

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Calibration image directory not found"):
            _load_calibration_images(tmp_path / "does_not_exist", height=64, width=64)

    def test_unsupported_extensions_ignored(self, tmp_path: Path) -> None:
        """Only image extensions are loaded; .txt files are skipped."""
        self._make_images(tmp_path, count=2)
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")

        result = _load_calibration_images(tmp_path, height=32, width=32)
        assert result.shape[0] == 2

    def test_skips_unreadable_files(self, tmp_path: Path) -> None:
        """Corrupt images are skipped without raising."""
        self._make_images(tmp_path, count=3)
        # Write a corrupt file with a supported extension
        (tmp_path / "corrupt.jpg").write_bytes(b"not-a-jpeg")

        result = _load_calibration_images(tmp_path, height=32, width=32)
        assert result.shape[0] == 3  # only the 3 valid images

    def test_all_unreadable_raises(self, tmp_path: Path) -> None:
        """If all files are unreadable, raises FileNotFoundError."""
        (tmp_path / "bad1.jpg").write_bytes(b"garbage")
        (tmp_path / "bad2.png").write_bytes(b"garbage")

        with pytest.raises(FileNotFoundError, match="No readable images"):
            _load_calibration_images(tmp_path, height=32, width=32)

    def test_png_and_jpeg_both_loaded(self, tmp_path: Path) -> None:
        """Both .jpg and .png formats are loaded."""
        from PIL import Image

        Image.new("RGB", (50, 50), "red").save(tmp_path / "a.jpg")
        Image.new("RGB", (50, 50), "blue").save(tmp_path / "b.png")

        result = _load_calibration_images(tmp_path, height=32, width=32)
        assert result.shape[0] == 2

    def test_constants_are_reasonable(self) -> None:
        """Sanity-check the module-level constants."""
        assert _DEFAULT_DIR_CALIB_SAMPLES > 0
        assert ".jpg" in _IMAGE_EXTENSIONS
        assert ".png" in _IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# TestCheckOnnx2tfAvailable
# ---------------------------------------------------------------------------


@onnx2tf_available
class TestCheckOnnx2tfAvailable:
    """Tests for ``_check_onnx2tf_available()``."""

    def test_available_when_importable(self, fake_onnx2tf: Any) -> None:
        _check_onnx2tf_available()  # should not raise

    def test_raises_when_not_importable(self) -> None:
        """ImportError is raised with install hint when onnx2tf is absent."""
        _remove_fake_onnx2tf()
        with mock.patch.dict(sys.modules, {"onnx2tf": None}):
            with pytest.raises(ImportError, match="onnx2tf is not installed"):
                _check_onnx2tf_available()


# ---------------------------------------------------------------------------
# TestGridSampleOnnxRewrite
# ---------------------------------------------------------------------------

onnx_gs_available = pytest.mark.skipif(
    not all(
        __import__("importlib").util.find_spec(p) is not None for p in ("onnx", "onnx_graphsurgeon", "onnxruntime")
    ),
    reason="onnx, onnx_graphsurgeon, and onnxruntime required",
)


def _build_gridsample_onnx(
    path: Path,
    *,
    n: int = 1,
    c: int = 4,
    h: int = 8,
    w: int = 8,
    h_out: int = 4,
    w_out: int = 4,
) -> None:
    """Write a minimal ONNX model with one GridSample node to *path*."""
    import onnx
    from onnx import TensorProto, helper

    im = helper.make_tensor_value_info("im", TensorProto.FLOAT, [n, c, h, w])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [n, h_out, w_out, 2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [n, c, h_out, w_out])
    node = helper.make_node(
        "GridSample",
        inputs=["im", "grid"],
        outputs=["out"],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=0,
    )
    graph = helper.make_graph([node], "gs_test", [im, grid], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    onnx.save(model, str(path))


@pytest.fixture()
def gridsample_onnx(tmp_path: Path) -> Path:
    """Build a tiny ONNX file with a single GridSample node."""
    p = tmp_path / "gs_model.onnx"
    _build_gridsample_onnx(p)
    return p


class TestGridSampleOnnxRewrite:
    """Tests for the ONNX-level GridSample → Gather(axis=0) rewrite."""

    def test_module_import_does_not_raise(self) -> None:
        """Importing the converter module must succeed regardless of onnx2tf version."""
        import rfdetr.export._tflite.converter  # noqa: F401

    @onnx_gs_available
    def test_no_gridsample_nodes_after_rewrite(self, gridsample_onnx: Path, tmp_path: Path) -> None:
        """_replace_gridsample_for_tflite removes all GridSample nodes from the graph."""
        import onnx
        import onnx_graphsurgeon as gs

        from rfdetr.export._tflite.converter import _replace_gridsample_for_tflite

        patched_path = _replace_gridsample_for_tflite(gridsample_onnx, tmp_path)

        model = onnx.load(str(patched_path))
        graph = gs.import_onnx(model)
        remaining = [n for n in graph.nodes if n.op == "GridSample"]
        assert remaining == [], f"Expected no GridSample nodes; found {len(remaining)}"

    @onnx_gs_available
    def test_gather_nodes_present_after_rewrite(self, gridsample_onnx: Path, tmp_path: Path) -> None:
        """Rewritten graph contains Gather nodes (the TFLite-safe replacement ops)."""
        import onnx
        import onnx_graphsurgeon as gs

        from rfdetr.export._tflite.converter import _replace_gridsample_for_tflite

        patched_path = _replace_gridsample_for_tflite(gridsample_onnx, tmp_path)

        model = onnx.load(str(patched_path))
        graph = gs.import_onnx(model)
        gather_nodes = [n for n in graph.nodes if n.op == "Gather"]
        assert len(gather_nodes) >= 4, f"Expected ≥4 Gather nodes (one per bilinear corner); found {len(gather_nodes)}"

    @onnx_gs_available
    def test_numerical_equivalence_vs_pytorch(self, gridsample_onnx: Path, tmp_path: Path) -> None:
        """Rewritten ONNX output matches torch.nn.functional.grid_sample within 1e-5."""
        import onnxruntime as ort
        import torch
        import torch.nn.functional as F  # noqa: N812

        from rfdetr.export._tflite.converter import _replace_gridsample_for_tflite

        rng = np.random.default_rng(0)
        im_np = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
        grid_np = rng.uniform(-1, 1, (1, 4, 4, 2)).astype(np.float32)

        ref = F.grid_sample(
            torch.from_numpy(im_np),
            torch.from_numpy(grid_np),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).numpy()

        patched_path = _replace_gridsample_for_tflite(gridsample_onnx, tmp_path)
        sess = ort.InferenceSession(str(patched_path), providers=["CPUExecutionProvider"])
        (result,) = sess.run(None, {"im": im_np, "grid": grid_np})

        np.testing.assert_allclose(
            result, ref, atol=1e-5, rtol=0, err_msg="Gather(axis=0) rewrite output diverges from F.grid_sample"
        )
