# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Check which devices the ExecuTorch TensorRT loader accepts, with CUDA, the delegate and the program mocked.

The delegate runs on the current CUDA device and takes no device argument, so ``cuda:N`` is only an assertion that
``N`` is that device. Every test makes device 1 current, so an accepted ``cuda:1`` cannot be a hardcoded ``cuda:0``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.export._executorch.inference import load_export_runtime
from rfdetr.export._runtime.metadata import ExportMetadata

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
    """Write a placeholder ``.pte``; the mocked program loader never reads it.

    Examples:
        Skipped: a pytest fixture, so it cannot run standalone.

        >>> pte_path.read_bytes()  # doctest: +SKIP
        b'pte'
    """
    path = tmp_path / "model.pte"
    path.write_bytes(b"pte")
    return path


@pytest.fixture
def program_loader(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Make CUDA device 1 current, register a stub delegate and return the mocked program loader.

    Examples:
        Skipped: a pytest fixture, so it cannot run standalone.

        >>> program_loader.assert_not_called()  # doctest: +SKIP
    """
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
