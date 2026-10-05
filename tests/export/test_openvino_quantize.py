# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for static INT8 quantization of OpenVINO exports (:mod:`rfdetr.export._openvino.quantize`)."""

import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from rfdetr.export._openvino.exporter import OpenVINOConfig, OpenVINOExporter
from rfdetr.export._openvino.quantize import VALID_QUANTIZATIONS, quantize_int8
from rfdetr.export.prepare import ExportGraph


class _TinyAttention(torch.nn.Module):
    """A two-linear-layer model with a softmax in between, small enough to quantize in a couple of seconds."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Project, attend and map to four outputs per row."""
        q = self.proj(x)
        weights = torch.softmax(q @ q.transpose(-1, -2), dim=-1)
        return (self.head(weights @ q),)


def _export_graph(model: torch.nn.Module | None = None) -> ExportGraph:
    """Build a throwaway ``ExportGraph`` for driving ``OpenVINOExporter`` without a real RF-DETR model.

    Args:
        model: Module to export; an ``Identity`` when omitted.

    Returns:
        A graph over a ``[1, 3, 8, 8]`` input.

    Examples:
        >>> tuple(_export_graph().input_tensors.shape)
        (1, 3, 8, 8)
    """
    return ExportGraph(
        model=torch.nn.Identity() if model is None else model,
        input_tensors=torch.zeros(1, 3, 8, 8),
        input_names=("input",),
        output_names=("dets",),
        dynamic_axes=None,
        shape=(8, 8),
        backbone_only=False,
    )


def _stub_openvino_module() -> types.ModuleType:
    """Build a fake ``openvino`` module whose ``convert_model``/``save_model`` are mocks.

    Returns:
        A fresh fake module.

    Examples:
        >>> fake = _stub_openvino_module()
        >>> callable(fake.convert_model) and callable(fake.save_model)
        True
    """
    fake = types.ModuleType("openvino")
    fake.convert_model = mock.MagicMock(return_value=mock.MagicMock(name="ov_model"))
    fake.save_model = mock.MagicMock()
    return fake


@pytest.fixture
def fake_nncf(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Stand-in ``nncf`` module recording how ``quantize_int8`` calls it, installed in ``sys.modules``."""
    module = types.ModuleType("nncf")
    module.ModelType = types.SimpleNamespace(TRANSFORMER=object())
    module.Dataset = mock.MagicMock(name="Dataset")
    module.quantize = mock.MagicMock(name="quantize")
    monkeypatch.setitem(sys.modules, "nncf", module)
    return module


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
        deep inside the conversion. A ``None`` entry in ``sys.modules`` makes the import statement itself fail, whether
        or not NNCF is installed.
        """
        monkeypatch.setitem(sys.modules, "nncf", None)
        with pytest.raises(ImportError, match="pip install nncf"):
            quantize_int8(object(), tmp_path, height=8, width=8)


class TestQuantizeInt8NncfCall:
    """How ``quantize_int8`` drives ``nncf.quantize``, observed through a stub NNCF."""

    def test_requests_transformer_recipe(self, fake_nncf: types.ModuleType) -> None:
        """NNCF is asked for its transformer recipe, which keeps normalization and elementwise math in float."""
        quantize_int8(object(), np.zeros((3, 3, 8, 8), dtype=np.float32), height=8, width=8)
        assert fake_nncf.quantize.call_args.kwargs["model_type"] is fake_nncf.ModelType.TRANSFORMER

    @pytest.mark.parametrize("samples", [1, 5])
    def test_subset_size_matches_number_of_calibration_samples(self, fake_nncf: types.ModuleType, samples: int) -> None:
        """NNCF is told to use exactly the samples supplied, not its default of 300."""
        quantize_int8(object(), np.zeros((samples, 3, 8, 8), dtype=np.float32), height=8, width=8)
        assert fake_nncf.quantize.call_args.kwargs["subset_size"] == samples

    def test_quantizes_given_model_on_calibration_dataset(self, fake_nncf: types.ModuleType) -> None:
        """The model passed in is quantized against a dataset wrapping the calibration batches."""
        model = object()
        quantize_int8(model, np.zeros((2, 3, 8, 8), dtype=np.float32), height=8, width=8)
        (quantized_model, dataset), _ = fake_nncf.quantize.call_args
        assert quantized_model is model and dataset is fake_nncf.Dataset.return_value
        assert [batch.shape for batch in fake_nncf.Dataset.call_args.args[0]] == [(1, 3, 8, 8)] * 2

    def test_returns_what_nncf_returns(self, fake_nncf: types.ModuleType) -> None:
        """The quantized model NNCF produces is handed back unchanged for saving."""
        result = quantize_int8(object(), np.zeros((1, 3, 8, 8), dtype=np.float32), height=8, width=8)
        assert result is fake_nncf.quantize.return_value


class TestExportOpenvinoInt8:
    """``format="openvino"`` INT8 export through the exporter, with stub ``openvino`` and ``nncf`` modules."""

    @pytest.mark.parametrize(
        ("precision", "expected_compress"),
        [
            pytest.param("float32", False, id="float32-disables-compression"),
            pytest.param("float16", True, id="float16-enables-compression"),
        ],
    )
    def test_saves_quantized_model_with_precision_compression_flag(
        self, fake_nncf: types.ModuleType, tmp_path: Path, precision: str, expected_compress: bool
    ) -> None:
        """``save_model`` gets the NNCF result and the compression flag ``precision`` selects.

        ``precision`` (IR storage width) and ``quantization`` (arithmetic width) are independent, so an INT8 request
        must not change which flag a given ``precision`` maps to.
        """
        fake_ov = _stub_openvino_module()
        config = OpenVINOConfig(
            output_dir=tmp_path,
            quantization="int8",
            calibration_data=np.zeros((2, 3, 8, 8), dtype=np.float32),
            precision=precision,
            verbose=False,
        )
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            OpenVINOExporter(config)(_export_graph())
        saved_model = fake_ov.save_model.call_args.args[0]
        assert saved_model is fake_nncf.quantize.return_value
        assert fake_ov.save_model.call_args.kwargs["compress_to_fp16"] is expected_compress


@pytest.mark.integration
@pytest.mark.e2e_openvino
class TestOpenvinoInt8EndToEnd:
    """Real NNCF quantization of a tiny model (``-m e2e_openvino``; needs ``openvino`` and ``nncf`` installed)."""

    def test_exports_runnable_quantized_ir(self, tmp_path: Path) -> None:
        """INT8 export of a tiny attention model writes an IR that compiles and runs on CPU."""
        openvino: Any = pytest.importorskip("openvino", reason="openvino not installed")
        pytest.importorskip("nncf", reason="nncf not installed; `pip install nncf` to run the INT8 export")
        data = np.stack([np.sin(np.arange(192).reshape(3, 8, 8) * (index + 1)) for index in range(3)]).astype(
            np.float32
        )
        config = OpenVINOConfig(output_dir=tmp_path, quantization="int8", calibration_data=data, verbose=False)

        xml_path = OpenVINOExporter(config)(_export_graph(_TinyAttention().eval()))

        compiled = openvino.Core().compile_model(str(xml_path), "CPU")
        (output,) = compiled(data[:1]).values()
        assert output.shape == (1, 3, 8, 4) and np.isfinite(output).all()
