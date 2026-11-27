# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU tests of the ExecuTorch ``backend="tensorrt"`` lowering and its environment checks, with Torch-TensorRT mocked.

``test_tensorrt.py`` exports real models through Torch-TensorRT and needs the ``executorch-tensorrt`` extra and a CUDA
device. These tests pin what the exporter asks of Torch-TensorRT and what it tells the caller when that fails, so a
regression shows on a host with neither.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import re
from pathlib import Path
from unittest import mock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.export._executorch.exporter import ExecutorchConfig, ExecuTorchExporter, _lower_tensorrt
from tests.export.conftest import _make_rfdetr
from tests.export.executorch.test_export import _export_graph

_EXPORTER_MODULE = "rfdetr.export._executorch.exporter"
_EXPORTER = f"{_EXPORTER_MODULE}.ExecuTorchExporter"
_DELEGATE_PACKAGE = "torch_tensorrt_executorch_runtime"
_PARTIAL_CONVERSION_ERROR = "require_full_compilation=True was specified, but model is not fully supported"


@pytest.fixture
def torch_tensorrt_api(mocker: MockerFixture) -> mock.MagicMock:
    """Install a mocked ``torch_tensorrt`` package and a no-op ``torch.export.export``, and return the package mock.

    The package's ``executorch`` submodule is the ``executorch`` attribute of the returned mock.
    """
    package = mock.MagicMock()
    mocker.patch.dict("sys.modules", {"torch_tensorrt": package, "torch_tensorrt.executorch": package.executorch})
    mocker.patch("torch.export.export")
    return package


class TestLowerTensorRTFullConversion:
    """``_lower_tensorrt`` asks for a full conversion and explains a refusal."""

    def test_compile_requires_full_conversion(self, torch_tensorrt_api: mock.MagicMock) -> None:
        """Torch-TensorRT is told to fail on a partial conversion rather than return a hybrid graph."""
        _lower_tensorrt(mock.MagicMock(), mock.MagicMock())

        assert torch_tensorrt_api.dynamo.compile.call_args.kwargs["require_full_compilation"] is True

    def test_partial_conversion_names_models_and_alternatives(self, torch_tensorrt_api: mock.MagicMock) -> None:
        """A partial conversion raises with the models known to convert fully and what to do for any other."""
        cause = AssertionError(_PARTIAL_CONVERSION_ERROR)
        torch_tensorrt_api.dynamo.compile.side_effect = cause

        with pytest.raises(RuntimeError) as excinfo:
            _lower_tensorrt(mock.MagicMock(), mock.MagicMock())

        message = str(excinfo.value)
        assert "model is not fully supported" in message
        assert "Detection Nano to Large, segmentation Nano and the keypoint preview model" in message
        assert "backend='xnnpack'" in message
        assert "format='tensorrt'" in message
        assert excinfo.value.__cause__ is cause

    def test_unrelated_assertion_error_is_not_relabelled(self, torch_tensorrt_api: mock.MagicMock) -> None:
        """An assertion that is not about full compilation keeps its own message."""
        torch_tensorrt_api.dynamo.compile.side_effect = AssertionError("Cannot specify both inputs and arg_inputs")

        with pytest.raises(AssertionError, match="Cannot specify both inputs and arg_inputs"):
            _lower_tensorrt(mock.MagicMock(), mock.MagicMock())

    def test_remedy_survives_the_exporter_wrapping(self, torch_tensorrt_api: mock.MagicMock, tmp_path: Path) -> None:
        """``ExecuTorchExporter`` re-wraps lowering failures, and the remedy is still in the message the caller sees."""
        torch_tensorrt_api.dynamo.compile.side_effect = AssertionError(_PARTIAL_CONVERSION_ERROR)
        exporter = ExecuTorchExporter(ExecutorchConfig(output_dir=tmp_path, backend="tensorrt"))
        # A CPU tensor would need a CUDA device for ``.cuda()``; the mocked Torch-TensorRT never reads the example.
        graph = dataclasses.replace(_export_graph(), input_tensors=mock.MagicMock(spec=torch.Tensor))

        with pytest.raises(RuntimeError, match=r"ExecuTorch export failed: .*backend='xnnpack'"):
            exporter._lower(graph, "tensorrt", tmp_path / "model.pte")


class TestExportDeviceScope:
    """``RFDETR.export`` makes the model's CUDA device current around the exporter, so a GPU build lands on it."""

    @pytest.fixture
    def cuda_scope(self, mocker: MockerFixture, tmp_path: Path) -> tuple[mock.MagicMock, list[bool]]:
        """Mock everything ``RFDETR.export`` does around the conversion and return how the CUDA scope was used.

        Returns:
            The mocked ``torch.cuda.device`` class, and one entry per conversion saying whether the scope it returned
            was entered and not yet left while the conversion ran.
        """
        cuda_device = mocker.patch("torch.cuda.device")
        mocker.patch("torch.cuda.is_available", return_value=True)
        scope_open_during_conversion: list[bool] = []

        def convert(_exporter: ExecuTorchExporter, _graph: object) -> Path:
            scope = cuda_device.return_value
            scope_open_during_conversion.append(scope.__enter__.called and not scope.__exit__.called)
            return tmp_path / "model.pte"

        mocker.patch(f"{_EXPORTER}._convert", autospec=True, side_effect=convert)
        mocker.patch(f"{_EXPORTER}.check_dependencies")
        mocker.patch(f"{_EXPORTER}.check_environment")
        mocker.patch("rfdetr.export.prepare.prepare_export_graph", return_value=_export_graph())
        mocker.patch("rfdetr.export.base.write_metadata")
        return cuda_device, scope_open_during_conversion

    @pytest.mark.parametrize(
        "device",
        [pytest.param(torch.device("cuda", 1), id="indexed"), pytest.param(torch.device("cuda"), id="index-less")],
    )
    def test_converts_on_the_models_cuda_device(
        self, cuda_scope: tuple[mock.MagicMock, list[bool]], tmp_path: Path, device: torch.device
    ) -> None:
        """A CUDA-configured model is exported with its own device current, an index-less one with the current."""
        cuda_device, scope_open_during_conversion = cuda_scope
        obj = _make_rfdetr()
        obj.model.device = device

        obj.export(format="executorch", backend="tensorrt", output_dir=str(tmp_path / "out"))

        cuda_device.assert_called_once_with(device)
        assert scope_open_during_conversion == [True]

    def test_cpu_model_keeps_the_current_device(
        self, cuda_scope: tuple[mock.MagicMock, list[bool]], tmp_path: Path
    ) -> None:
        """A CPU-configured model is exported without touching the current CUDA device, whatever the host has."""
        cuda_device, scope_open_during_conversion = cuda_scope
        obj = _make_rfdetr()
        obj.model.device = "cpu"

        obj.export(format="executorch", backend="xnnpack", output_dir=str(tmp_path / "out"))

        cuda_device.assert_not_called()
        assert scope_open_during_conversion == [False]

    def test_cuda_model_on_a_host_without_cuda_does_not_enter_the_scope(
        self, cuda_scope: tuple[mock.MagicMock, list[bool]], mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Entering ``torch.cuda.device`` needs a CUDA runtime, so a CUDA model on a host without one is left alone."""
        cuda_device, _ = cuda_scope
        mocker.patch("torch.cuda.is_available", return_value=False)
        obj = _make_rfdetr()
        obj.model.device = torch.device("cuda", 0)

        obj.export(format="executorch", backend="xnnpack", output_dir=str(tmp_path / "out"))

        cuda_device.assert_not_called()


class TestTensorRTIntegrationCheck:
    """``check_environment`` refuses a Torch-TensorRT installed without its ExecuTorch integration."""

    @pytest.fixture
    def exporter(self, mocker: MockerFixture, tmp_path: Path) -> ExecuTorchExporter:
        """A ``backend="tensorrt"`` exporter on a host that has Torch-TensorRT, its delegate package and a CUDA
        device."""
        mocker.patch(f"{_EXPORTER_MODULE}.is_installed", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=True)
        return ExecuTorchExporter(ExecutorchConfig(output_dir=tmp_path, backend="tensorrt"))

    @pytest.mark.parametrize("version", ["1.6.0", "1.6", "1.7.2"])
    def test_accepts_the_validated_executorch_release_and_newer(
        self, exporter: ExecuTorchExporter, mocker: MockerFixture, version: str
    ) -> None:
        """The release the ``executorch-tensorrt`` extra was validated with, or a newer one, passes the check."""
        mocker.patch("importlib.metadata.version", return_value=version)

        exporter.check_environment()

    def test_accepts_a_source_build_without_package_metadata(
        self, exporter: ExecuTorchExporter, mocker: MockerFixture
    ) -> None:
        """An ExecuTorch built from source has no installed-package version to compare, so it is not refused."""
        mocker.patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("executorch"))

        exporter.check_environment()

    @pytest.mark.parametrize("version", ["1.3.1", "1.5.0"])
    def test_refuses_executorch_older_than_the_validated_release(
        self, exporter: ExecuTorchExporter, mocker: MockerFixture, version: str
    ) -> None:
        """An older ExecuTorch (the plain ``executorch`` extra's) is refused, naming both extras and the version."""
        mocker.patch("importlib.metadata.version", return_value=version)

        with pytest.raises(ImportError, match=rf"rfdetr\[executorch-tensorrt\] Found executorch {re.escape(version)},"):
            exporter.check_environment()

    def test_refuses_a_missing_delegate_package(self, exporter: ExecuTorchExporter, mocker: MockerFixture) -> None:
        """Torch-TensorRT without its ExecuTorch delegate package is refused, naming the extra to install."""
        mocker.patch(f"{_EXPORTER_MODULE}.is_installed", side_effect=lambda name: name != _DELEGATE_PACKAGE)
        mocker.patch("importlib.metadata.version", return_value="1.6.0")

        with pytest.raises(ImportError, match=r"Torch-TensorRT .*rfdetr\[executorch-tensorrt\]"):
            exporter.check_environment()
