# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for static INT8 quantization of OpenVINO exports (:mod:`rfdetr.export._openvino.quantize`)."""

from pathlib import Path

import pytest

from rfdetr.export._openvino.exporter import OpenVINOConfig, OpenVINOExporter
from rfdetr.export._openvino.quantize import VALID_QUANTIZATIONS, quantize_int8


class TestConfigValidation:
    """Refusals that happen before the model is traced."""

    def test_int8_is_a_valid_mode(self) -> None:
        """``"int8"`` is accepted by this format's mode list.

        Guards against the mode being wired into the exporter while the list it is validated against still rejects it,
        which would make every INT8 request fail at construction.
        """
        assert "int8" in VALID_QUANTIZATIONS

    def test_rejects_unknown_quantization_mode(self, tmp_path: Path) -> None:
        """An unrecognized mode is refused rather than silently ignored.

        OpenVINO has no FP16 *quantization* mode -- that is `precision` -- so a caller reaching for ``"fp16"`` here has
        confused the two knobs and must be told, not handed an unquantized model.
        """
        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            OpenVINOExporter(OpenVINOConfig(output_dir=tmp_path, quantization="fp16"))

    def test_rejects_int8_without_calibration_data(self, tmp_path: Path) -> None:
        """INT8 without calibration data is refused at construction.

        Static quantization reads activation ranges from data; defaulting to none would emit a model that loads and runs
        while being quietly wrong, which is the failure this refusal exists to prevent.
        """
        with pytest.raises(ValueError, match="requires calibration_data"):
            OpenVINOExporter(OpenVINOConfig(output_dir=tmp_path, quantization="int8"))

    def test_accepts_int8_with_calibration_data(self, tmp_path: Path) -> None:
        """INT8 with calibration data constructs successfully.

        The positive case for the refusal above: a caller who supplies data reaches the conversion.
        """
        exporter = OpenVINOExporter(OpenVINOConfig(output_dir=tmp_path, quantization="int8", calibration_data=tmp_path))
        assert exporter.config.quantization == "int8"

    @pytest.mark.parametrize("quantization", [None, "fp32"])
    def test_accepts_float_modes_without_calibration_data(self, quantization: str | None, tmp_path: Path) -> None:
        """Float modes need no calibration data.

        Only the INT8 path consumes calibration data, so requiring it for an FP32 export would be a pointless obstacle
        for the common case.
        """
        exporter = OpenVINOExporter(OpenVINOConfig(output_dir=tmp_path, quantization=quantization))
        assert exporter.config.quantization == quantization

    def test_quantization_is_independent_of_precision(self, tmp_path: Path) -> None:
        """``precision`` and ``quantization`` are separate settings.

        ``precision`` controls IR storage width and ``quantization`` controls arithmetic width; a caller may set one
        without the other, and conflating them would silently change what an export produces.
        """
        exporter = OpenVINOExporter(OpenVINOConfig(output_dir=tmp_path, precision="float32"))
        assert exporter.config.quantization is None


class TestSettingPlumbing:
    """The keywords ``RFDETR.export`` forwards to this format."""

    def test_quantization_settings_reach_the_config(self) -> None:
        """``quantization``, ``calibration_data`` and ``max_images`` are carried into the config.

        These arrive through ``RFDETR.export``'s flat keyword signature, so a missing `setting_names` entry would drop
        them silently and export an unquantized model without complaint.
        """
        config = OpenVINOExporter.build_config(quantization="int8", calibration_data="images/", max_images=16)
        assert (config.quantization, config.calibration_data, config.max_images) == ("int8", "images/", 16)

    def test_openvino_precision_still_maps_to_precision(self) -> None:
        """The pre-existing ``openvino_precision`` keyword keeps working.

        Added settings share the `setting_names` mapping with it, so this guards the older keyword against being
        displaced by the new entries.
        """
        assert OpenVINOExporter.build_config(openvino_precision="float32").precision == "float32"


class TestNncfRequirement:
    """What happens on a host without NNCF."""

    def test_missing_nncf_raises_with_install_hint(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A host without NNCF is told exactly what to install.

        NNCF is deliberately outside the ``rfdetr[openvino]`` extra, so an INT8 request on an otherwise complete
        OpenVINO install is an expected path and must fail with an actionable message rather than an ImportError from
        deep inside the conversion.
        """
        import builtins

        real_import = builtins.__import__

        def _refuse_nncf(name: str, *args: object, **kwargs: object) -> object:
            if name == "nncf":
                raise ImportError("No module named 'nncf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _refuse_nncf)
        with pytest.raises(ImportError, match="pip install nncf"):
            quantize_int8(object(), tmp_path, height=8, width=8)
