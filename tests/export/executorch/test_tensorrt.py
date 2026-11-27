# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CPU tests of the ExecuTorch ``backend="tensorrt"`` lowering and its environment checks, with Torch-TensorRT mocked.

``test_tensorrt.py`` exports real models through Torch-TensorRT and needs the ``executorch-tensorrt`` extra and a CUDA
device. These tests pin what the exporter asks of Torch-TensorRT and what it tells the caller when that fails, so a
regression shows on a host with neither.

ExecuTorch ``backend="tensorrt"`` lowering and runtime on an NVIDIA GPU (``@pytest.mark.e2e_executorch_tensorrt``).

The unit tests in ``test_export.py`` mock Torch-TensorRT away. These export real models through it and run the ``.pte``
with ExecuTorch's runtime and the Torch-TensorRT delegate, so they need the ``executorch-tensorrt`` extra and a CUDA
device.

Check which devices the ExecuTorch TensorRT loader accepts, with CUDA, the delegate and the program mocked.

The delegate runs on the current CUDA device and takes no device argument, so ``cuda:N`` is only an assertion that ``N``
is that device. Every test makes device 1 current, so an accepted ``cuda:1`` cannot be a hardcoded ``cuda:0``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import re
import sys
import types
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr import RFDETRNano
from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export._executorch.exporter import ExecutorchConfig, ExecuTorchExporter, _lower_tensorrt
from rfdetr.export._executorch.inference import load_export_runtime
from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.export.imports import _IS_TORCH_TENSORRT_EXECUTORCH_RUNTIME_INSTALLED, _IS_TORCH_TENSORRT_INSTALLED
from rfdetr.utilities.reproducibility import seed_all
from tests._markers import _requires_cuda
from tests.export.conftest import _make_rfdetr, _prediction_model_for_task, assert_prediction_roundtrip
from tests.export.executorch.test_export import _export_graph, _portable_kernel_call_names

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


# Same reason as test_coreml.py's `_EXPORT_SEED`: the module-scoped fixture runs before the function-scoped
# autouse seed reset.
_EXPORT_SEED = 0


_requires_executorch_tensorrt = pytest.mark.skipif(
    not _IS_EXECUTORCH_AVAILABLE
    or not _IS_TORCH_TENSORRT_INSTALLED
    or not _IS_TORCH_TENSORRT_EXECUTORCH_RUNTIME_INSTALLED,
    reason="needs the executorch-tensorrt extra",
)


#: Skip on a GPU older than Ampere: Torch-TensorRT 2.15 pulls in TensorRT 11.4, which builds attention with
#: ``IAttention``, and that layer needs compute capability 8.0 or higher. On a Turing T4 (7.5) the engine build fails.
#: A host without CUDA is left to ``_requires_cuda``. The reason's opening words are what the CI job's skip allow-list
#: matches, so change both together.
_requires_ampere = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0),
    reason="TensorRT 11 attention needs compute capability 8.0 or newer",
)


#: Device-to-host copies the ``.pte`` of each task holds: one per model output (detection ``dets`` and ``labels``;
#: segmentation adds ``masks``; keypoints add ``keypoints``). Every ``.pte`` also holds one host-to-device input copy.
_EXPECTED_D2H_COPIES = {"detect": 2, "segment": 3, "keypoints": 3}


@pytest.fixture(scope="module", params=list(_EXPECTED_D2H_COPIES))
def tensorrt_pte(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Lower an untrained model of one task through the TensorRT delegate and return the task and its ``.pte``.

    Detection is the default ``RFDETRNano``; segmentation and keypoints are the small random models the prediction round
    trips use.
    """
    task = request.param
    seed_all(_EXPORT_SEED)
    detector = RFDETRNano(pretrain_weights=None) if task == "detect" else _prediction_model_for_task(task)
    out_dir = tmp_path_factory.mktemp(f"executorch_tensorrt_{task}")
    return task, Path(detector.export(output_dir=str(out_dir), format="executorch", backend="tensorrt", verbose=False))


def _delegate_ids(pte_path: Path) -> list[str]:
    """Return the backend id of every delegate in the serialized ``.pte``, in program order.

    Args:
        pte_path: Path to a serialized ExecuTorch program.

    Returns:
        One backend id per delegate in the first execution plan, such as ``"TensorRTBackend"``.

    Examples:
        Requires a real ``.pte`` artifact and the ``executorch`` package, so it is not runnable standalone.

        >>> callable(_delegate_ids)
        True
    """
    from executorch.exir._serialize import _deserialize_pte_binary

    plan = _deserialize_pte_binary(pte_path.read_bytes()).program.execution_plan[0]
    return [delegate.id for delegate in plan.delegates]


@_requires_executorch_tensorrt
@_requires_cuda
@_requires_ampere
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_executorch_tensorrt
class TestExecuTorchTensorRTDelegate:
    """The TensorRT delegate must take the whole graph and produce a ``.pte`` that predicts like the PyTorch model."""

    def test_shipped_pte_is_one_tensorrt_engine(self, tensorrt_pte: tuple[str, Path]) -> None:
        """The exported ``.pte`` of every task holds a single TensorRT delegate, so no op runs outside the engine."""
        _, pte = tensorrt_pte
        assert _delegate_ids(pte) == ["TensorRTBackend"]

    def test_shipped_pte_only_copies_outside_the_engine(self, tensorrt_pte: tuple[str, Path]) -> None:
        """The only kernel calls outside the engine move the input to the GPU and each output back to the CPU.

        Any other kernel call would be a model op TensorRT did not convert, running on ExecuTorch's CPU kernels. A task
        with more outputs holds more device-to-host copies.
        """
        task, pte = tensorrt_pte
        expected = ["et_copy::_d2h_copy.out"] * _EXPECTED_D2H_COPIES[task] + ["et_copy::_h2d_copy.out"]
        assert sorted(_portable_kernel_call_names(pte)) == sorted(expected)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    def test_prediction_matches_native(self, task: str, tmp_path: Path) -> None:
        """``RFDETRInference`` on the TensorRT ``.pte`` returns the native boxes, scores, labels and task data."""
        assert_prediction_roundtrip("executorch", task, tmp_path, backend="tensorrt")


_CURRENT_DEVICE = 1


def _tensorrt_metadata() -> ExportMetadata:
    """Make the inference metadata of a detection ``.pte`` delegated to TensorRT.

    Examples:
        >>> _tensorrt_metadata().backend
        'tensorrt'
    """
    return ExportMetadata(
        format="executorch",
        task="detect",
        backend="tensorrt",
        input_shape=(1, 3, 8, 8),
        outputs={"pred_boxes": 0, "pred_logits": 1},
        class_names=["one", "two"],
        class_id_to_name={0: "one", 1: "two"},
        num_classes=2,
        num_select=2,
        patch_size=16,
        num_windows=1,
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        trace_alpha=0.2,
    )


@pytest.fixture
def pte_path(tmp_path: Path) -> Path:
    """Write a placeholder ``.pte``; the mocked program loader never reads it."""
    path = tmp_path / "model.pte"
    path.write_bytes(b"pte")
    return path


@pytest.fixture
def program_loader(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Make CUDA device 1 current, register a stub delegate and return the mocked program loader."""
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.current_device", return_value=_CURRENT_DEVICE)
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", types.ModuleType("stub_delegate"))
    return mocker.patch("rfdetr.export._executorch.inference.load_executorch_method")


class TestTensorRTDeviceAccepted:
    """Devices naming the current CUDA device load the program."""

    @pytest.mark.parametrize(
        "device",
        [
            "auto",
            "cuda",
            "cuda:1",
            pytest.param(str(torch.device("cuda", _CURRENT_DEVICE)), id="torch-device-current-index"),
        ],
    )
    def test_device_naming_the_current_device_loads(
        self, device: str, pte_path: Path, program_loader: MagicMock
    ) -> None:
        """``auto``, ``cuda`` and ``cuda:<current index>`` load the program on the current CUDA device."""
        runtime = load_export_runtime(pte_path, _tensorrt_metadata(), device, {})

        program_loader.assert_called_once_with(pte_path)
        assert runtime.info == {"backend": "executorch", "device": "cuda"}

    def test_device_selection_leaves_cuda_state_alone(
        self, mocker: MockerFixture, pte_path: Path, program_loader: MagicMock
    ) -> None:
        """Accepting ``cuda:N`` never switches the process-wide current CUDA device."""
        set_device = mocker.patch("torch.cuda.set_device")

        load_export_runtime(pte_path, _tensorrt_metadata(), "cuda:1", {})

        set_device.assert_not_called()


class TestTensorRTDeviceRefused:
    """Devices the delegate cannot honour are refused before the program loads."""

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:2"])
    def test_other_cuda_index_is_refused(self, device: str, pte_path: Path, program_loader: MagicMock) -> None:
        """An index other than the current CUDA device is refused with the forms that are accepted."""
        with pytest.raises(ValueError, match=r"Request auto, cuda or cuda:1"):
            load_export_runtime(pte_path, _tensorrt_metadata(), device, {})

        program_loader.assert_not_called()

    @pytest.mark.parametrize("device", ["cuda:abc", "cuda:", "cuda:-1"])
    def test_malformed_cuda_device_is_refused(self, device: str, pte_path: Path, program_loader: MagicMock) -> None:
        """A ``cuda:`` device without a non-negative integer index is refused by its spelling."""
        with pytest.raises(ValueError, match="ExecuTorch TensorRT requires cuda or auto"):
            load_export_runtime(pte_path, _tensorrt_metadata(), device, {})

        program_loader.assert_not_called()

    def test_cuda_index_without_cuda_raises_runtime_error(self, mocker: MockerFixture, pte_path: Path) -> None:
        """Without CUDA, ``cuda:N`` reports the missing device instead of querying the current one."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        current_device = mocker.patch("torch.cuda.current_device")

        with pytest.raises(RuntimeError, match="requires an available CUDA device"):
            load_export_runtime(pte_path, _tensorrt_metadata(), "cuda:1", {})

        current_device.assert_not_called()
