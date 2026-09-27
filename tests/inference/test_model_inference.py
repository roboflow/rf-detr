# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETR.inference()."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from rfdetr import detr as detr_module
from rfdetr.detr import RFDETR
from rfdetr.inference import ModelContext


class _FakeModel(torch.nn.Module):
    """Minimal nn.Module that satisfies the inference contract."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pred_boxes": self.linear(x[:, :1, :1, :1].squeeze(-1).squeeze(-1))}

    def export(self) -> None:
        pass


class _TupleModule(torch.nn.Module):
    """Parameter-free module whose forward returns two tensors, like the exported detector."""

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``value + 1`` and ``value * 2``.

        Examples:
            >>> first, second = _TupleModule()(torch.tensor([2.0]))
            >>> first.tolist(), second.tolist()
            ([3.0], [4.0])
        """
        return value + 1, value * 2

    def export(self) -> None:
        """Do nothing; ``RFDETR.inference()`` calls ``export()`` on the module it optimizes.

        Examples:
            >>> _TupleModule().export() is None
            True
        """


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
        return SimpleNamespace(num_channels=3)

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _FakeModelContext:
        return _FakeModelContext()


class TestOptimizeForInferenceRemoved:
    """The ``optimize_for_inference`` alias was removed in v1.11.0; only :meth:`RFDETR.inference` remains."""

    def test_alias_is_gone(self) -> None:
        """The deprecated alias must no longer exist on the class."""
        assert not hasattr(RFDETR, "optimize_for_inference")


class TestModelInferenceDtype:
    """Dtype coercion and validation tests."""

    def test_string_dtype_float32_is_accepted(self) -> None:
        """Passing dtype='float32' (str) should be coerced to torch.float32."""
        rfdetr = _FakeRFDETR()

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False, dtype="float32")

        assert rfdetr._optimized_dtype == torch.float32

    def test_string_dtype_float16_is_accepted(self) -> None:
        """Passing dtype='float16' (str) should be coerced to torch.float16."""
        rfdetr = _FakeRFDETR()

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False, dtype="float16")

        assert rfdetr._optimized_dtype == torch.float16

    def test_torch_dtype_is_passed_through(self) -> None:
        """Passing dtype=torch.float32 directly should work as before."""
        rfdetr = _FakeRFDETR()

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False, dtype=torch.float32)

        assert rfdetr._optimized_dtype == torch.float32

    def test_invalid_dtype_type_raises_type_error(self) -> None:
        """Passing an invalid dtype type (e.g. int) should raise TypeError."""
        rfdetr = _FakeRFDETR()

        with pytest.raises(TypeError, match="dtype must be a torch.dtype or a string name of a dtype"):
            rfdetr.inference(compile=False, dtype=42)  # type: ignore[arg-type]

    def test_invalid_dtype_string_raises_type_error(self) -> None:
        """Passing a non-existent dtype string should raise TypeError with a descriptive message."""
        rfdetr = _FakeRFDETR()

        with pytest.raises(TypeError, match="dtype must be a torch.dtype or a string name of a dtype"):
            rfdetr.inference(compile=False, dtype="not_a_dtype")

    def test_valid_torch_attr_that_is_not_dtype_raises_type_error(self) -> None:
        """'Tensor' is a valid torch attribute but not a torch.dtype — should raise TypeError."""
        rfdetr = _FakeRFDETR()

        with pytest.raises(TypeError, match="dtype must be a torch.dtype or a string name of a dtype"):
            rfdetr.inference(compile=False, dtype="Tensor")  # type: ignore[arg-type]

    @pytest.mark.parametrize("dtype_str", ["float32", "float16", "bfloat16"])
    def test_string_dtype_variants_are_accepted(self, dtype_str: str) -> None:
        """Common dtype string names should be accepted and coerced to the matching torch.dtype."""
        rfdetr = _FakeRFDETR()
        expected = getattr(torch, dtype_str)

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False, dtype=dtype_str)

        assert rfdetr._optimized_dtype == expected


class TestModelInferenceCudaDeviceContext:
    """Verify that inference() wraps operations in the correct device context."""

    @pytest.mark.gpu
    @patch("rfdetr.detr._move_model_context_to_device")
    @patch("rfdetr.detr.deepcopy")
    @patch("torch.cuda.device")
    def test_cuda_device_context_manager_is_used_for_cuda_device(
        self,
        mock_cuda_device,
        mock_deepcopy,
        _mock_move_model_context_to_device,
    ) -> None:
        """torch.cuda.device() context should be entered when model is on CUDA."""
        rfdetr = _FakeRFDETR()
        # Simulate a CUDA device without actually requiring CUDA hardware
        rfdetr.model.device = torch.device("cuda", 0)
        mock_deepcopy.return_value = rfdetr.model.model

        entered_devices: list[torch.device] = []

        class _CapturingDeviceCtx:
            def __init__(self, captured_device):
                entered_devices.append(captured_device)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_cuda_device.side_effect = _CapturingDeviceCtx
        rfdetr.inference(compile=False, dtype=torch.float32)

        assert len(entered_devices) == 1
        assert entered_devices[0] == torch.device("cuda", 0)

    def test_nullcontext_used_for_cpu_device(self) -> None:
        """contextlib.nullcontext() should be used when model is on CPU (no CUDA init)."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cpu")

        # torch.cuda.device should NOT be called for CPU devices
        with (
            patch("torch.cuda.device") as mock_cuda_device,
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
        ):
            rfdetr.inference(compile=False, dtype=torch.float32)

        mock_cuda_device.assert_not_called()

    @pytest.mark.gpu
    @patch("rfdetr.detr._move_model_context_to_device")
    @patch("rfdetr.detr.deepcopy")
    @patch("torch.cuda.device")
    def test_cuda_device_context_uses_model_device(
        self,
        mock_cuda_device,
        mock_deepcopy,
        _mock_move_model_context_to_device,
    ) -> None:
        """The device passed to torch.cuda.device() should match self.model.device."""
        rfdetr = _FakeRFDETR()
        expected_device = torch.device("cuda", 2)
        rfdetr.model.device = expected_device
        mock_deepcopy.return_value = rfdetr.model.model

        captured: dict[str, torch.device] = {}

        class _CapturingCtx:
            def __init__(self, captured_device):
                captured["device"] = captured_device

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_cuda_device.side_effect = _CapturingCtx
        rfdetr.inference(compile=False)

        assert captured.get("device") == expected_device


class TestModelInferenceCompile:
    """Tests for the compile=True paths."""

    def test_cudagraph_backend_traces_freezes_and_captures(self) -> None:
        """The CUDA Graph backend should capture a frozen TorchScript module and its fixed input."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cuda")
        dummy_input = torch.randn(2, 3, 28, 28)
        traced = Mock()
        frozen = Mock()
        captured = Mock()

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("rfdetr.detr.torch.jit.trace", return_value=traced) as mock_trace,
            patch("rfdetr.detr.torch.jit.freeze", return_value=frozen) as mock_freeze,
            patch("rfdetr.detr._CUDAGraphInferenceModel", return_value=captured) as mock_capture,
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", return_value=rfdetr.model.model),
        ):
            rfdetr.inference(compile=True, batch_size=2, compile_backend="cudagraph")

        mock_trace.assert_called_once_with(rfdetr.model.model, dummy_input)
        mock_freeze.assert_called_once_with(traced)
        mock_capture.assert_called_once_with(frozen, dummy_input, rfdetr.model.device)
        assert rfdetr.model.inference_model is captured

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(torch.float32, id="float32"), pytest.param(torch.float16, id="float16")],
    )
    def test_cudagraph_backend_replays_real_module_through_inference(self, dtype: torch.dtype) -> None:
        """The public cudagraph route should capture a real CUDA module and return outputs that survive a replay."""
        device = torch.device("cuda")
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = device
        rfdetr.model.model = _TupleModule().to(device)

        rfdetr.inference(compile=True, batch_size=2, dtype=dtype, compile_backend="cudagraph")

        graph_model = rfdetr.model.inference_model
        assert isinstance(graph_model, detr_module._CUDAGraphInferenceModel)
        first_input = torch.randn(2, 3, 28, 28, device=device, dtype=dtype)
        second_input = torch.randn(2, 3, 28, 28, device=device, dtype=dtype)
        first = graph_model(first_input)
        second = graph_model(second_input)

        torch.cuda.synchronize(device)
        assert first[0].dtype == dtype
        assert torch.equal(first[0], first_input + 1)
        assert torch.equal(first[1], first_input * 2)
        assert torch.equal(second[0], second_input + 1)
        assert torch.equal(second[1], second_input * 2)

    def test_cudagraph_backend_rejects_cpu_before_copying_model(self) -> None:
        """CUDA Graph capture should fail clearly when the selected device is not CUDA."""
        rfdetr = _FakeRFDETR()

        with (
            patch("rfdetr.detr.deepcopy") as mock_deepcopy,
            pytest.raises(ValueError, match="cudagraph.*CUDA"),
        ):
            rfdetr.inference(compile=True, compile_backend="cudagraph")

        mock_deepcopy.assert_not_called()

    def test_compile_true_calls_jit_trace(self) -> None:
        """torch.jit.trace should be called with the model and a correctly-shaped dummy input."""
        rfdetr = _FakeRFDETR()
        mock_traced = rfdetr.model.model

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace", return_value=mock_traced) as mock_trace,
            patch("torch.jit.freeze") as mock_freeze,
        ):
            rfdetr.inference(compile=True, batch_size=2)

        assert mock_trace.called
        mock_freeze.assert_not_called()
        dummy_input: torch.Tensor = mock_trace.call_args.args[1]
        resolution = rfdetr.model.resolution
        assert dummy_input.shape == (2, 3, resolution, resolution)

    def test_compile_true_sets_compiled_flags(self) -> None:
        """_optimized_has_been_compiled=True and _optimized_batch_size should be set after compile=True."""
        rfdetr = _FakeRFDETR()

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace", return_value=rfdetr.model.model),
        ):
            rfdetr.inference(compile=True, batch_size=4)

        assert rfdetr._optimized_has_been_compiled is True
        assert rfdetr._optimized_batch_size == 4

    def test_inductor_backend_compiles_and_warms_up_with_dummy_input(self) -> None:
        """The CUDA Inductor backend should complete allocator warmup and graph recording before success."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cuda")
        original_forward = rfdetr.model.model.forward
        warmup: dict[str, object] = {}
        dummy_input = torch.randn(2, 3, 28, 28)
        compiled_model = Mock(
            side_effect=lambda x: (
                warmup.update({"shape": x.shape, "inference_mode": torch.is_inference_mode_enabled()}),
                original_forward(x),
            )[1]
        )

        def assert_unoptimized_before_sync(device: torch.device) -> None:
            """Assert synchronization completes before the optimized wrapper becomes visible.

            Examples:
                >>> callable(assert_unoptimized_before_sync)
                True
            """
            assert device == rfdetr.model.device
            assert rfdetr._is_optimized_for_inference is False
            assert rfdetr._optimized_has_been_compiled is False
            assert rfdetr._optimized_batch_size is None
            assert rfdetr.model.inference_model is None

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.compile", return_value=compiled_model) as mock_compile,
            patch("torch.jit.trace") as mock_trace,
            patch("torch.jit.freeze") as mock_freeze,
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch("rfdetr.detr.torch.cuda.synchronize", side_effect=assert_unoptimized_before_sync) as mock_synchronize,
            patch.object(_FakeModel, "to", return_value=rfdetr.model.model),
        ):
            rfdetr.inference(compile=True, batch_size=2, compile_backend="inductor")

        mock_trace.assert_not_called()
        mock_freeze.assert_not_called()
        mock_compile.assert_called_once_with(rfdetr.model.model, mode="reduce-overhead")
        assert compiled_model.call_count == 2
        mock_synchronize.assert_called_once_with(rfdetr.model.device)
        assert warmup == {"shape": torch.Size((2, 3, 28, 28)), "inference_mode": True}
        assert rfdetr._optimized_has_been_compiled is True
        assert rfdetr._optimized_batch_size == 2

    def test_invalid_compile_backend_raises(self) -> None:
        """An unknown backend should fail before copying or exporting the model."""
        rfdetr = _FakeRFDETR()

        with (
            patch("rfdetr.detr.deepcopy") as mock_deepcopy,
            pytest.raises(ValueError, match="compile_backend must be 'torchscript', 'cudagraph', or 'inductor'"),
        ):
            rfdetr.inference(compile=True, compile_backend="unknown")  # type: ignore[arg-type]

        mock_deepcopy.assert_not_called()

    def test_compile_false_skips_jit_trace(self) -> None:
        """torch.jit.trace should NOT be called when compile=False."""
        rfdetr = _FakeRFDETR()

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace") as mock_trace,
            patch("torch.jit.freeze") as mock_freeze,
        ):
            rfdetr.inference(compile=False)

        mock_trace.assert_not_called()
        mock_freeze.assert_not_called()
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None


class TestModelInferenceState:
    """Verify that inference() correctly sets internal state flags."""

    def test_is_optimized_inplace_false_before_optimization(self) -> None:
        """is_optimized_inplace is False before any optimization is applied."""
        rfdetr = _FakeRFDETR()
        assert rfdetr.is_optimized_inplace is False

    def test_is_optimized_flag_set(self) -> None:
        """_is_optimized_for_inference should be True after optimization."""
        rfdetr = _FakeRFDETR()

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False)

        assert rfdetr._is_optimized_for_inference is True

    def test_inference_model_set(self) -> None:
        """model.inference_model should be set after optimization."""
        rfdetr = _FakeRFDETR()

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False)

        assert rfdetr.model.inference_model is not None

    def test_remove_optimized_model_clears_state(self) -> None:
        """remove_optimized_model() should clear all optimization flags."""
        rfdetr = _FakeRFDETR()

        with patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model):
            rfdetr.inference(compile=False)

        rfdetr.remove_optimized_model()

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_dtype is None
        assert rfdetr._optimized_resolution is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None
        assert rfdetr.is_optimized_inplace is False


class TestModelInferenceInplace:
    """Tests for the low-memory in-place optimization path."""

    def test_inplace_false_keeps_deepcopy_behavior(self) -> None:
        """The default path should still deep-copy the loaded module."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model
        copied_model = _FakeModel()

        with patch("rfdetr.detr.deepcopy", return_value=copied_model) as mock_deepcopy:
            rfdetr.inference(compile=False)

        mock_deepcopy.assert_called_once_with(original_model)
        assert rfdetr.model.model is original_model
        assert rfdetr.model.inference_model is copied_model
        assert rfdetr._is_optimized_for_inference is True
        assert rfdetr.is_optimized_inplace is False

    @patch("rfdetr.detr.deepcopy")
    def test_inplace_true_compile_false_does_not_deepcopy(self, mock_deepcopy) -> None:
        """Inplace=True with compile=False should use the loaded module directly."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model

        rfdetr.inference(compile=False, inplace=True)

        mock_deepcopy.assert_not_called()
        assert rfdetr.model.model is None
        assert rfdetr.model.inference_model is original_model
        assert rfdetr._is_optimized_for_inference is True
        assert rfdetr.is_optimized_inplace is True

    def test_remove_optimized_model_after_inplace_warns_and_preserves_state(self) -> None:
        """remove_optimized_model() after inplace optimization issues UserWarning and no-ops."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model

        rfdetr.inference(compile=False, inplace=True)

        with pytest.warns(UserWarning, match="no effect after inplace optimization"):
            rfdetr.remove_optimized_model()

        assert rfdetr.model.model is None
        assert rfdetr.model.inference_model is original_model
        assert rfdetr._is_optimized_for_inference is True
        assert rfdetr.is_optimized_inplace is True

    def test_second_optimize_after_inplace_raises_runtime_error(self) -> None:
        """Calling inference() again after inplace=True raises RuntimeError."""
        rfdetr = _FakeRFDETR()

        rfdetr.inference(compile=False, inplace=True)

        with pytest.raises(RuntimeError, match="base model has been cleared"):
            rfdetr.inference(compile=False)

    def test_inplace_true_default_dtype_float32_does_not_cast(self) -> None:
        """Inplace=True with default dtype (float32) leaves weights unchanged — no casting occurs."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model
        original_dtype = original_model.linear.weight.dtype

        rfdetr.inference(compile=False, inplace=True)

        assert rfdetr.model.inference_model is original_model
        assert original_model.linear.weight.dtype == original_dtype
        assert rfdetr._optimized_dtype == torch.float32
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float16, id="float16"),
            pytest.param(torch.bfloat16, id="bfloat16"),
        ],
    )
    def test_inplace_true_allows_destructive_dtype_casting(self, dtype: torch.dtype) -> None:
        """In-place optimization may cast the original module to the target dtype."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model

        rfdetr.inference(compile=False, dtype=dtype, inplace=True)

        assert rfdetr.model.model is None
        assert rfdetr.model.inference_model is original_model
        assert original_model.linear.weight.dtype == dtype
        assert rfdetr._optimized_dtype == dtype

    def test_inplace_export_failure_keeps_base_model(self) -> None:
        """Export failure in the in-place path should not clear model.model."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model

        with (
            patch("rfdetr.detr.deepcopy") as mock_deepcopy,
            patch.object(original_model, "export", side_effect=RuntimeError("export failed")),
            pytest.raises(RuntimeError, match="export failed"),
        ):
            rfdetr.inference(compile=False, inplace=True)

        mock_deepcopy.assert_not_called()
        assert rfdetr.model.model is original_model
        assert rfdetr.model.inference_model is None
        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.is_optimized_inplace is False

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.int8, id="torch-int8"),
            pytest.param("int8", id="string-int8"),
        ],
    )
    def test_inplace_non_floating_dtype_raises_before_export(self, dtype: torch.dtype | str) -> None:
        """In-place optimization rejects non-floating dtypes before mutating the base model."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model

        with (
            patch("rfdetr.detr.deepcopy") as mock_deepcopy,
            patch.object(original_model, "export") as mock_export,
            pytest.raises(ValueError, match="floating-point torch.dtype"),
        ):
            rfdetr.inference(compile=False, dtype=dtype, inplace=True)

        mock_deepcopy.assert_not_called()
        mock_export.assert_not_called()
        assert rfdetr.model.model is original_model
        assert rfdetr.model.inference_model is None
        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.is_optimized_inplace is False

    def test_inplace_compile_true_raises_before_export_or_trace(self) -> None:
        """In-place optimization rejects compile=True before mutating the base model."""
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model

        with (
            patch("rfdetr.detr.deepcopy") as mock_deepcopy,
            patch.object(original_model, "export") as mock_export,
            patch("torch.jit.trace") as mock_trace,
            pytest.raises(ValueError, match="inplace=True.*compile=False"),
        ):
            rfdetr.inference(compile=True, inplace=True)

        mock_deepcopy.assert_not_called()
        mock_export.assert_not_called()
        mock_trace.assert_not_called()
        assert rfdetr.model.model is original_model
        assert rfdetr.model.inference_model is None
        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None
        assert rfdetr.is_optimized_inplace is False


class TestModelInferenceExceptionRecovery:
    """Verify state consistency when optimization fails mid-execution."""

    def test_inductor_second_setup_call_failure_leaves_model_fully_unoptimized(self) -> None:
        """A graph-recording failure after allocator warmup should clear all optimized state."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cuda")
        dummy_input = torch.randn(1, 3, 28, 28)
        compiled_model = Mock(side_effect=[rfdetr.model.model(dummy_input), RuntimeError("recording failed")])

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.compile", return_value=compiled_model),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", return_value=rfdetr.model.model),
            pytest.raises(RuntimeError, match="recording failed"),
        ):
            rfdetr.inference(compile=True, compile_backend="inductor")

        assert compiled_model.call_count == 2
        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    def test_deepcopy_failure_leaves_clean_state(self) -> None:
        """If deepcopy raises, inference_model should be None and _is_optimized_for_inference False."""
        rfdetr = _FakeRFDETR()
        # Simulate a previously-optimized state to confirm remove_optimized_model ran
        rfdetr._is_optimized_for_inference = True
        rfdetr.model.inference_model = rfdetr.model.model

        with (
            patch("rfdetr.detr.deepcopy", side_effect=RuntimeError("deepcopy failed")),
            pytest.raises(RuntimeError, match="deepcopy failed"),
        ):
            rfdetr.inference(compile=False)

        assert rfdetr.model.inference_model is None
        assert rfdetr._is_optimized_for_inference is False

    def test_export_failure_leaves_is_optimized_false(self) -> None:
        """If export() raises after deepcopy succeeds, _is_optimized_for_inference stays False."""
        rfdetr = _FakeRFDETR()
        fake_copy = _FakeModel()

        with (
            patch("rfdetr.detr.deepcopy", return_value=fake_copy),
            patch.object(fake_copy, "export", side_effect=RuntimeError("export failed")),
            pytest.raises(RuntimeError, match="export failed"),
        ):
            rfdetr.inference(compile=False)

        assert rfdetr._is_optimized_for_inference is False

    def test_jit_trace_failure_leaves_compiled_flags_false(self) -> None:
        """If jit.trace raises, _optimized_has_been_compiled and _optimized_batch_size stay unset."""
        rfdetr = _FakeRFDETR()

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace", side_effect=RuntimeError("trace failed")),
            pytest.raises(RuntimeError, match="trace failed"),
        ):
            rfdetr.inference(compile=True, batch_size=2)

        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    def test_jit_trace_failure_leaves_model_fully_unoptimized(self) -> None:
        """jit.trace failure leaves both _is_optimized_for_inference=False and inference_model=None."""
        rfdetr = _FakeRFDETR()

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace", side_effect=RuntimeError("trace failed")),
            pytest.raises(RuntimeError, match="trace failed"),
        ):
            rfdetr.inference(compile=True)

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None

    def test_jit_freeze_failure_leaves_model_fully_unoptimized(self) -> None:
        """A freeze failure after tracing must not publish a partially optimized model."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cuda")
        dummy_input = torch.randn(1, 3, 28, 28)

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace", return_value=rfdetr.model.model),
            patch("torch.jit.freeze", side_effect=RuntimeError("freeze failed")),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", return_value=rfdetr.model.model),
            pytest.raises(RuntimeError, match="freeze failed"),
        ):
            rfdetr.inference(compile=True, compile_backend="cudagraph")

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    def test_cudagraph_capture_failure_leaves_model_fully_unoptimized(self) -> None:
        """A capture failure after tracing and freezing must not publish partial state."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cuda")
        dummy_input = torch.randn(1, 3, 28, 28)

        with (
            patch("rfdetr.detr.deepcopy", return_value=rfdetr.model.model),
            patch("torch.jit.trace", return_value=rfdetr.model.model),
            patch("torch.jit.freeze", return_value=rfdetr.model.model),
            patch("rfdetr.detr._CUDAGraphInferenceModel", side_effect=RuntimeError("capture failed")),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", return_value=rfdetr.model.model),
            pytest.raises(RuntimeError, match="capture failed"),
        ):
            rfdetr.inference(compile=True, compile_backend="cudagraph")

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    def test_inplace_export_failure_module_mutations_are_not_undone(self) -> None:
        """RFDETR resets flags on export failure but cannot undo module-level mutations.

        Production export() may mutate the module (e.g. forward->forward_export) before raising; those changes are not
        reversed by the exception-recovery path.
        """
        rfdetr = _FakeRFDETR()
        original_model = rfdetr.model.model
        mutated: dict[str, bool] = {"happened": False}

        def _mutating_export() -> None:
            mutated["happened"] = True
            raise RuntimeError("export failed mid-mutation")

        with (
            patch("rfdetr.detr.deepcopy"),
            patch.object(original_model, "export", side_effect=_mutating_export),
            pytest.raises(RuntimeError, match="export failed mid-mutation"),
        ):
            rfdetr.inference(compile=False, inplace=True)

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.is_optimized_inplace is False
        assert rfdetr.model.model is original_model
        # The mutation happened and cannot be undone by RFDETR's recovery path
        assert mutated["happened"] is True


class TestCudaGraphInferenceModel:
    """Coverage for the static buffers and stream ordering owned by the direct graph wrapper."""

    def test_waits_for_previous_stream_before_replay(self) -> None:
        """A call waits on the prior completion event, then replays, clones, and only then records its own stream."""
        order = Mock()
        order.clone.side_effect = detr_module._CUDAGraphInferenceModel._clone_output
        wrapper = detr_module._CUDAGraphInferenceModel.__new__(detr_module._CUDAGraphInferenceModel)
        wrapper._lock = nullcontext()
        wrapper._static_input = order.static_input
        wrapper._static_output = torch.tensor([3.0])
        wrapper._graph = order.graph
        wrapper._completed = order.completed
        value = Mock()

        with (
            patch("rfdetr.detr.torch.cuda.current_stream", return_value=order.stream) as mock_current_stream,
            patch.object(detr_module._CUDAGraphInferenceModel, "_clone_output", order.clone),
        ):
            result = wrapper(value)

        mock_current_stream.assert_called_once_with(value.device)
        assert order.mock_calls == [
            call.stream.wait_event(order.completed),
            call.static_input.copy_(value),
            call.graph.replay(),
            call.clone(wrapper._static_output),
            call.completed.record(order.stream),
        ]
        assert torch.equal(result, wrapper._static_output)
        assert result.data_ptr() != wrapper._static_output.data_ptr()

    @pytest.mark.gpu
    def test_replays_from_two_streams_keep_inputs_and_returned_outputs_separate(self) -> None:
        """A replay held back on the device sees its own input, and its returned outputs survive the next replay."""
        device = torch.device("cuda")
        module = torch.jit.trace(_TupleModule().eval().to(device), torch.zeros(1, device=device))
        wrapper = detr_module._CUDAGraphInferenceModel(module, torch.zeros(1, device=device), device)
        graph = wrapper._graph

        def delayed_replay() -> None:
            """Queue a device-side delay ahead of the replay so a call from another stream could overtake it.

            Examples:
                Requires the live CUDA graph captured by this test.
                >>> delayed_replay()  # doctest: +SKIP
            """
            torch.cuda._sleep(20_000_000)
            graph.replay()

        wrapper._graph = Mock(replay=delayed_replay)
        first_stream = torch.cuda.Stream(device=device)
        second_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(first_stream):
            first = wrapper(torch.tensor([2.0], device=device))
        with torch.cuda.stream(second_stream):
            second = wrapper(torch.tensor([5.0], device=device))

        torch.cuda.synchronize(device)
        assert torch.equal(first[0], torch.tensor([3.0], device=device))
        assert torch.equal(first[1], torch.tensor([4.0], device=device))
        assert torch.equal(second[0], torch.tensor([6.0], device=device))
        assert torch.equal(second[1], torch.tensor([10.0], device=device))

    @pytest.mark.gpu
    def test_capture_failure_reports_restart_and_keeps_the_cause(self) -> None:
        """A rejected capture should tell the caller to restart the process and chain the original error."""
        device = torch.device("cuda")
        module = torch.jit.trace(_TupleModule().eval().to(device), torch.zeros(1, device=device))

        with (
            patch("rfdetr.detr.torch.cuda.graph", side_effect=RuntimeError("unsupported operator")),
            pytest.raises(RuntimeError, match="capture failed.*Restart the process") as exc_info,
        ):
            detr_module._CUDAGraphInferenceModel(module, torch.zeros(1, device=device), device)

        assert str(exc_info.value.__cause__) == "unsupported operator"

    @pytest.mark.gpu
    def test_error_at_the_completion_fence_reports_restart_and_keeps_the_cause(self) -> None:
        """An error CUDA reports only at the post-capture fence should get the same restart message and cause."""
        device = torch.device("cuda")
        module = torch.jit.trace(_TupleModule().eval().to(device), torch.zeros(1, device=device))
        real_synchronize = torch.cuda.synchronize
        fences = 0

        def fail_completion_fence(target: torch.device | None = None) -> None:
            """Fail the wrapper's second fenced synchronize; torch's own argument-less calls pass through.

            Examples:
                Requires the live CUDA device and closure created by this test.
                >>> fail_completion_fence()  # doctest: +SKIP
            """
            nonlocal fences
            if target is not None:
                fences += 1
                if fences == 2:
                    raise RuntimeError("deferred capture error")
            real_synchronize(target)

        with (
            patch("rfdetr.detr.torch.cuda.synchronize", side_effect=fail_completion_fence),
            pytest.raises(RuntimeError, match="capture failed.*Restart the process") as exc_info,
        ):
            detr_module._CUDAGraphInferenceModel(module, torch.zeros(1, device=device), device)

        assert fences == 2
        assert str(exc_info.value.__cause__) == "deferred capture error"


class TestModelContextClearedWeights:
    """``ModelContext`` rejects work needing the underlying module once ``inplace=True`` cleared it."""

    @staticmethod
    def _context() -> ModelContext:
        """Build a ModelContext whose weights have already been cleared."""
        context = ModelContext(
            model=_FakeModel(),
            postprocess=SimpleNamespace(),
            device=torch.device("cpu"),
            resolution=28,
            args=SimpleNamespace(num_classes=2),
        )
        context.model = None
        return context

    def test_reinitialize_detection_head_raises_runtime_error(self) -> None:
        """Reinitializing the head after the weights were cleared raises RuntimeError, not AttributeError."""
        with pytest.raises(RuntimeError, match="Cannot reinitialize the detection head"):
            self._context().reinitialize_detection_head(5)

    def test_reinitialize_detection_head_leaves_num_classes_untouched(self) -> None:
        """The guard fires before ``args.num_classes`` is mutated, so the context is not left half-updated."""
        context = self._context()

        with pytest.raises(RuntimeError):
            context.reinitialize_detection_head(5)

        assert context.args.num_classes == 2
