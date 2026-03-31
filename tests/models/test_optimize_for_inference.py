# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for RFDETR.optimize_for_inference()."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rfdetr.detr import RFDETR


class _FakeModel(torch.nn.Module):
    """Minimal nn.Module that satisfies the optimize_for_inference contract."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pred_boxes": self.linear(x[:, :1, :1, :1].squeeze(-1).squeeze(-1))}

    def export(self) -> None:
        pass


class _FakeModelContext:
    def __init__(self, device: torch.device | str = torch.device("cpu"), resolution: int = 28) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.resolution = resolution
        self.model = _FakeModel()
        self.inference_model = None


class _FakeRFDETR(RFDETR):
    def maybe_download_pretrain_weights(self) -> None:
        return None

    def get_model_config(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace()

    def get_model(self, config: SimpleNamespace) -> _FakeModelContext:
        return _FakeModelContext()


class TestOptimizeForInferenceDtype:
    """dtype coercion and validation tests."""

    def test_string_dtype_float32_is_accepted(self) -> None:
        """Passing dtype='float32' (str) should be coerced to torch.float32."""
        rfdetr = _FakeRFDETR()

        with patch.object(rfdetr.model.model, "export"):
            rfdetr.optimize_for_inference(compile=False, dtype="float32")

        assert rfdetr._optimized_dtype == torch.float32

    def test_string_dtype_float16_is_accepted(self) -> None:
        """Passing dtype='float16' (str) should be coerced to torch.float16."""
        rfdetr = _FakeRFDETR()

        with patch.object(rfdetr.model.model, "export"):
            rfdetr.optimize_for_inference(compile=False, dtype="float16")

        assert rfdetr._optimized_dtype == torch.float16

    def test_torch_dtype_is_passed_through(self) -> None:
        """Passing dtype=torch.float32 directly should work as before."""
        rfdetr = _FakeRFDETR()

        with patch.object(rfdetr.model.model, "export"):
            rfdetr.optimize_for_inference(compile=False, dtype=torch.float32)

        assert rfdetr._optimized_dtype == torch.float32

    def test_invalid_dtype_type_raises_type_error(self) -> None:
        """Passing an invalid dtype type (e.g. int) should raise TypeError."""
        rfdetr = _FakeRFDETR()

        with pytest.raises(TypeError, match="dtype must be a torch.dtype or a string name of a dtype"):
            rfdetr.optimize_for_inference(compile=False, dtype=42)  # type: ignore[arg-type]

    def test_invalid_dtype_string_raises_attribute_error(self) -> None:
        """Passing a non-existent dtype string should raise AttributeError."""
        rfdetr = _FakeRFDETR()

        with pytest.raises(AttributeError):
            rfdetr.optimize_for_inference(compile=False, dtype="not_a_dtype")


class TestOptimizeForInferenceCudaDeviceContext:
    """Verify that optimize_for_inference wraps operations in the correct device context."""

    def test_cuda_device_context_manager_is_used_for_cuda_device(self) -> None:
        """torch.cuda.device() context should be entered when model is on CUDA."""
        rfdetr = _FakeRFDETR()
        # Simulate a CUDA device without actually requiring CUDA hardware
        rfdetr.model.device = torch.device("cuda", 0)

        entered_devices: list = []

        class _CapturingDeviceCtx:
            def __init__(self, captured_device):
                entered_devices.append(captured_device)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with (
            patch("torch.cuda.device", side_effect=_CapturingDeviceCtx),
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
        ):
            rfdetr.optimize_for_inference(compile=False, dtype=torch.float32)

        assert len(entered_devices) == 1
        assert entered_devices[0] == torch.device("cuda", 0)

    def test_nullcontext_used_for_cpu_device(self) -> None:
        """contextlib.nullcontext() should be used when model is on CPU (no CUDA init)."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cpu")

        # torch.cuda.device should NOT be called for CPU devices
        with (
            patch("torch.cuda.device") as mock_cuda_device,
            patch.object(rfdetr.model.model, "export"),
        ):
            rfdetr.optimize_for_inference(compile=False, dtype=torch.float32)

        mock_cuda_device.assert_not_called()

    def test_cuda_device_context_uses_model_device(self) -> None:
        """The device passed to torch.cuda.device() should match self.model.device."""
        rfdetr = _FakeRFDETR()
        expected_device = torch.device("cuda", 2)
        rfdetr.model.device = expected_device

        captured: dict = {}

        class _CapturingCtx:
            def __init__(self, captured_device):
                captured["device"] = captured_device

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with (
            patch("torch.cuda.device", side_effect=_CapturingCtx),
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
        ):
            rfdetr.optimize_for_inference(compile=False)

        assert captured.get("device") == expected_device


class TestOptimizeForInferenceState:
    """Verify that optimize_for_inference correctly sets internal state flags."""

    def test_is_optimized_flag_set(self) -> None:
        """_is_optimized_for_inference should be True after optimization."""
        rfdetr = _FakeRFDETR()

        with patch.object(rfdetr.model.model, "export"):
            rfdetr.optimize_for_inference(compile=False)

        assert rfdetr._is_optimized_for_inference is True

    def test_inference_model_set(self) -> None:
        """model.inference_model should be set after optimization."""
        rfdetr = _FakeRFDETR()

        with patch.object(rfdetr.model.model, "export"):
            rfdetr.optimize_for_inference(compile=False)

        assert rfdetr.model.inference_model is not None

    def test_remove_optimized_model_clears_state(self) -> None:
        """remove_optimized_model() should clear all optimization flags."""
        rfdetr = _FakeRFDETR()

        with patch.object(rfdetr.model.model, "export"):
            rfdetr.optimize_for_inference(compile=False)

        rfdetr.remove_optimized_model()

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_dtype is None
        assert rfdetr._optimized_resolution is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None
