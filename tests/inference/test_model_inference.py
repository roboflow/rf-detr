# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RFDETR.inference()."""

import threading
import weakref
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
import torch

from rfdetr import RFDETRNano, RFDETRSegNano
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


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Collect the tensors nested in tuples, lists and dict values, in traversal order.

    Examples:
        >>> leaves = _tensor_leaves(({"a": torch.tensor([1.0])}, [torch.tensor([2.0]), None]))
        >>> [leaf.tolist() for leaf in leaves]
        [[1.0], [2.0]]
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _tensor_leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _tensor_leaves(item)]
    return []


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

        predictor = rfdetr.inference(compile=False, dtype="float32")

        assert predictor._native_model._optimized_dtype == torch.float32

    def test_string_dtype_float16_is_accepted(self) -> None:
        """Passing dtype='float16' (str) should be coerced to torch.float16."""
        rfdetr = _FakeRFDETR()

        predictor = rfdetr.inference(compile=False, dtype="float16")

        assert predictor._native_model._optimized_dtype == torch.float16

    def test_torch_dtype_is_passed_through(self) -> None:
        """Passing dtype=torch.float32 directly should work as before."""
        rfdetr = _FakeRFDETR()

        predictor = rfdetr.inference(compile=False, dtype=torch.float32)

        assert predictor._native_model._optimized_dtype == torch.float32

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

        predictor = rfdetr.inference(compile=False, dtype=dtype_str)

        assert predictor._native_model._optimized_dtype == expected


class TestModelInferenceCudaDeviceContext:
    """Device placement belongs to the copied runtime and preserves a cold source."""

    @pytest.mark.parametrize("device_index", [0, 2])
    def test_cuda_device_context_uses_snapshot_device(self, device_index: int) -> None:
        """CUDA work enters the source's configured device while moving only the copied module."""
        source = _FakeRFDETR()
        target = torch.device("cuda", device_index)
        source.model.device = target
        original = source.model.model
        with (
            patch("torch.cuda.device", return_value=nullcontext()) as device_context,
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module) as move,
        ):
            predictor = source.inference(compile=False)

        device_context.assert_called_once_with(target)
        assert move.call_args.args[0] is not original
        assert predictor._native_model.model.device == target
        assert original.linear.weight.device.type == "cpu"
        assert source.model.device == target

    def test_nullcontext_used_for_cpu_device(self) -> None:
        """A CPU factory does not initialize a CUDA device context."""
        source = _FakeRFDETR()
        with patch("torch.cuda.device") as device_context:
            source.inference(compile=False)
        device_context.assert_not_called()

    @pytest.mark.gpu
    def test_cold_cuda_source_stays_on_cpu_and_snapshot_fixes_device_index(self) -> None:
        """A deferred CUDA source stays on CPU while its predictor owns a concrete CUDA device."""
        source = _FakeRFDETR()
        source.model.device = torch.device("cuda")
        original = source.model.model
        weights = original.linear.weight.detach().clone()
        expected_device = torch.device("cuda", torch.cuda.current_device())

        predictor = source.inference(compile=False)
        snapshot = predictor._native_model

        assert snapshot.model.device == expected_device
        assert snapshot.model.inference_model.linear.weight.device == expected_device
        assert source.model.model is original
        assert original.linear.weight.device.type == "cpu"
        assert torch.equal(original.linear.weight, weights)
        assert original.training is True
        assert source.model.device == torch.device("cuda")
        assert source.model.inference_model is None
        # The device stays explicit even if a later call observes another current CUDA device.
        with patch("torch.cuda.current_device", return_value=expected_device.index + 1):
            assert snapshot.model.device == expected_device


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
            patch("rfdetr.detr.torch.jit.trace", return_value=traced) as mock_trace,
            patch("rfdetr.detr.torch.jit.freeze", return_value=frozen) as mock_freeze,
            patch("rfdetr.detr._CUDAGraphInferenceModel", return_value=captured) as mock_capture,
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module),
        ):
            predictor = rfdetr.inference(compile=True, batch_size=2, compile_backend="cudagraph")

        assert mock_trace.call_args.args[0] is not rfdetr.model.model
        assert mock_trace.call_args.args[1] is dummy_input
        mock_freeze.assert_called_once_with(traced)
        mock_capture.assert_called_once_with(frozen, dummy_input, torch.device("cuda", 0))
        assert predictor._native_model.model.inference_model is captured

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

        predictor = rfdetr.inference(compile=True, batch_size=2, dtype=dtype, compile_backend="cudagraph")

        graph_model = predictor._native_model.model.inference_model
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

    @pytest.fixture
    def nondeterministic_algorithms(self) -> Generator[None, None, None]:
        """Turn deterministic algorithms off for one test, restoring the previous mode afterwards.

        The autouse ``reset_random_seeds`` fixture enables them, which routes CUDA ``F.interpolate`` through a
        decomposition that fails under ``torch.jit.trace`` (its sizes become CPU tensors).
        """
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    @pytest.mark.gpu
    @pytest.mark.usefixtures("nondeterministic_algorithms")
    @pytest.mark.parametrize(
        "model_class",
        [pytest.param(RFDETRNano, id="detection"), pytest.param(RFDETRSegNano, id="segmentation")],
    )
    @pytest.mark.parametrize(
        ("dtype", "tolerance"),
        [pytest.param(torch.float32, 1e-3, id="float32"), pytest.param(torch.float16, 1e-2, id="float16")],
    )
    def test_cudagraph_predict_matches_torchscript_backend(
        self, model_class: type[RFDETR], dtype: torch.dtype, tolerance: float
    ) -> None:
        """The cudagraph backend gives predict() the same raw predictions as the default TorchScript backend.

        Elsewhere only the parameter-free ``_TupleModule`` is captured; this runs a real detector through ``predict()``,
        including its tuple-to-dict mapping of the cloned graph outputs. Raw predictions are compared because a top-k
        order flip between near-equal random-weight scores would make final detections unstable. The tolerance allows
        for ``torch.jit.freeze``, whose default passes do not strictly preserve numerics; float32 matmuls run in TF32
        because ``rfdetr.detr`` sets ``torch.set_float32_matmul_precision("high")`` on import.
        """
        model = model_class(pretrain_weights=None, device="cuda")
        image = torch.rand(3, 480, 640)
        torchscript = model.inference(compile=True, batch_size=1, dtype=dtype)
        with patch.object(
            torchscript._native_model.model, "postprocess", wraps=torchscript._native_model.model.postprocess
        ) as torchscript_postprocess:
            torchscript.predict(image)
        cudagraph = model.inference(compile=True, batch_size=1, dtype=dtype, compile_backend="cudagraph")

        with patch.object(
            cudagraph._native_model.model, "postprocess", wraps=cudagraph._native_model.model.postprocess
        ) as cudagraph_postprocess:
            cudagraph.predict(image)

        inference_model = cudagraph._native_model.model.inference_model
        assert isinstance(inference_model, detr_module._CUDAGraphInferenceModel)
        if dtype == torch.float16:
            # Half precision on random weights flips near-tied top-k proposals between the two separate traces. Each
            # decoder slot starts from its own learned query embedding, so a flip changes the logits themselves and
            # not only their order; sorting cannot absorb it. Compare the graph with eager replay of the very module
            # it captured, which is what the backend adds. The float32 case covers the predict() mapping.
            static_input = inference_model._static_input.clone()
            with torch.inference_mode():
                expected = inference_model._model(static_input)
                actual = inference_model(static_input)
        else:
            actual = cudagraph_postprocess.call_args.args[0]
            expected = torchscript_postprocess.call_args.args[0]
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    @pytest.mark.gpu
    def test_cudagraph_reoptimize_replays_at_the_new_batch_size(self) -> None:
        """Calling inference() again with another batch size recaptures, so the new batch size replays correctly.

        Users are told to re-run ``inference()`` per batch size; the new graph must take the new static shape instead of
        reusing the earlier capture.
        """
        device = torch.device("cuda")
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = device
        rfdetr.model.model = _TupleModule().to(device)
        first_predictor = rfdetr.inference(compile=True, batch_size=1, compile_backend="cudagraph")
        value = torch.randn(2, 3, 28, 28, device=device)

        predictor = rfdetr.inference(compile=True, batch_size=2, compile_backend="cudagraph")
        first, second = predictor._native_model.model.inference_model(value)

        assert first_predictor._native_model._optimized_batch_size == 1
        torch.cuda.synchronize(device)
        assert torch.equal(first, value + 1)
        assert torch.equal(second, value * 2)

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

        with (
            patch("torch.jit.trace", side_effect=lambda module, _input: module) as mock_trace,
            patch("torch.jit.freeze") as mock_freeze,
        ):
            rfdetr.inference(compile=True, batch_size=2)

        assert mock_trace.call_count == 1
        assert mock_trace.call_args.args[0] is not rfdetr.model.model
        assert rfdetr.model.model.training is True
        mock_freeze.assert_not_called()
        dummy_input: torch.Tensor = mock_trace.call_args.args[1]
        resolution = rfdetr.model.resolution
        assert dummy_input.shape == (2, 3, resolution, resolution)

    def test_torchscript_backend_runs_real_trace_without_mutating_source(self) -> None:
        """The default backend traces a copied module and preserves its numerical outputs."""
        source = _FakeRFDETR()
        original = _TupleModule()
        source.model.model = original
        value = torch.randn(2, 3, 28, 28)

        predictor = source.inference(batch_size=2)
        runtime = predictor._native_model.model.inference_model

        assert isinstance(runtime, torch.jit.ScriptModule)
        torch.testing.assert_close(runtime(value), original(value))
        assert source.model.model is original
        assert original.training is True
        assert source.model.inference_model is None

    def test_compile_true_sets_compiled_flags(self) -> None:
        """_optimized_has_been_compiled=True and _optimized_batch_size should be set after compile=True."""
        rfdetr = _FakeRFDETR()

        with patch("torch.jit.trace", side_effect=lambda module, _input: module):
            predictor = rfdetr.inference(compile=True, batch_size=4)

        assert predictor._native_model._optimized_has_been_compiled is True
        assert predictor._native_model._optimized_batch_size == 4

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
            assert device == torch.device("cuda", 0)
            assert rfdetr._is_optimized_for_inference is False
            assert rfdetr._optimized_has_been_compiled is False
            assert rfdetr._optimized_batch_size is None
            assert rfdetr.model.inference_model is None

        with (
            patch("torch.compile", return_value=compiled_model) as mock_compile,
            patch("torch.jit.trace") as mock_trace,
            patch("torch.jit.freeze") as mock_freeze,
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch("rfdetr.detr.torch.cuda.synchronize", side_effect=assert_unoptimized_before_sync) as mock_synchronize,
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module),
        ):
            predictor = rfdetr.inference(compile=True, batch_size=2, compile_backend="inductor")

        mock_trace.assert_not_called()
        mock_freeze.assert_not_called()
        assert mock_compile.call_args.args[0] is not rfdetr.model.model
        assert mock_compile.call_args.kwargs == {"mode": "reduce-overhead"}
        assert compiled_model.call_count == 2
        mock_synchronize.assert_called_once_with(torch.device("cuda", 0))
        assert warmup == {"shape": torch.Size((2, 3, 28, 28)), "inference_mode": True}
        assert predictor._native_model._optimized_has_been_compiled is True
        assert predictor._native_model._optimized_batch_size == 2

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
            patch("torch.jit.trace") as mock_trace,
            patch("torch.jit.freeze") as mock_freeze,
        ):
            predictor = rfdetr.inference(compile=False)

        mock_trace.assert_not_called()
        mock_freeze.assert_not_called()
        assert predictor._native_model._optimized_has_been_compiled is False
        assert predictor._native_model._optimized_batch_size is None

    @pytest.mark.parametrize("compile_backend", ["cudagraph", "inductor"])
    def test_compile_false_ignores_compile_backend(self, compile_backend: str) -> None:
        """With compile=False the backend is ignored: no CUDA check runs and the eager exported model is used.

        ``compile_backend`` is documented as applying only when ``compile=True``. On a CPU model ``"cudagraph"`` with
        ``compile=True`` raises, so this pins that ``compile=False`` neither checks the device nor wraps the model.
        """
        rfdetr = _FakeRFDETR()

        predictor = rfdetr.inference(compile=False, compile_backend=compile_backend)

        assert isinstance(predictor._native_model.model.inference_model, _FakeModel)
        assert predictor._native_model._optimized_has_been_compiled is False


class TestModelInferenceState:
    """The returned predictor owns optimization state and its runtime lifetime."""

    def test_snapshot_owns_flags_and_runtime(self) -> None:
        """The source remains eager while the returned snapshot owns only its exported copy."""
        source = _FakeRFDETR()
        original = source.model.model
        predictor = source.inference(compile=False)
        snapshot = predictor._native_model

        assert snapshot is not source
        assert snapshot._is_optimized_for_inference is True
        assert snapshot.is_optimized_inplace is False
        assert snapshot.model.model is None
        assert isinstance(snapshot.model.inference_model, _FakeModel)
        assert snapshot.model.inference_model is not original
        assert source.model.model is original
        assert source.model.inference_model is None
        assert source._is_optimized_for_inference is False
        assert source.is_optimized_inplace is False

    def test_legacy_cleanup_does_not_change_predictor(self) -> None:
        """Clearing old source optimization state does not discard a returned predictor's runtime."""
        source = _FakeRFDETR()
        predictor = source.inference(compile=False)
        runtime = predictor._native_model.model.inference_model
        source._is_optimized_for_inference = True
        source._optimized_dtype = torch.float32
        source._optimized_resolution = 28
        source._optimized_has_been_compiled = True
        source._optimized_batch_size = 2
        source.model.inference_model = _FakeModel()

        source.remove_optimized_model()

        assert source.model.inference_model is None
        assert source._is_optimized_for_inference is False
        assert source._optimized_dtype is None
        assert source._optimized_resolution is None
        assert source._optimized_has_been_compiled is False
        assert source._optimized_batch_size is None
        assert source.is_optimized_inplace is False
        assert predictor._native_model.model.inference_model is runtime

    def test_legacy_inplace_cleanup_warns_and_preserves_runtime(self) -> None:
        """Cleanup cannot recreate base weights removed by an older destructive optimization."""
        source = _FakeRFDETR()
        runtime = source.model.model
        source.model.model = None
        source.model.inference_model = runtime
        source._optimized_inplace = True
        source._is_optimized_for_inference = True

        with pytest.warns(UserWarning, match="no effect after inplace optimization"):
            source.remove_optimized_model()

        assert source.model.model is None
        assert source.model.inference_model is runtime
        assert source._is_optimized_for_inference is True
        assert source.is_optimized_inplace is True

    def test_deleting_predictor_releases_runtime(self) -> None:
        """The source retains no reference to a predictor's runtime after the predictor is deleted."""
        source = _FakeRFDETR()
        predictor = source.inference(compile=False)
        released = weakref.ref(predictor._native_model.model.inference_model)

        del predictor

        assert released() is None
        assert source.model.inference_model is None

    def test_deleting_predictor_releases_cudagraph_wrapper(self) -> None:
        """The predictor exclusively retains its CUDA graph wrapper after capture completes."""
        source = _FakeRFDETR()
        source.model.device = torch.device("cuda")
        wrapper_class = detr_module._CUDAGraphInferenceModel
        dummy_input = torch.randn(1, 3, 28, 28)
        with (
            patch("torch.jit.trace", side_effect=lambda module, _input: module),
            patch("torch.jit.freeze", side_effect=lambda module: module),
            patch("rfdetr.detr._CUDAGraphInferenceModel", side_effect=lambda *_: wrapper_class.__new__(wrapper_class)),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module),
        ):
            predictor = source.inference(compile_backend="cudagraph")
        released = weakref.ref(predictor._native_model.model.inference_model)

        del predictor

        assert released() is None
        assert source.model.inference_model is None


class TestModelInferenceIndependence:
    """Factories preserve source storage, dtype and mode across repeated calls and rejected requests."""

    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("compile", [False, True])
    def test_inplace_keyword_is_rejected_without_mutation(self, inplace: bool, compile: bool) -> None:
        """The removed keyword cannot activate destructive optimization, even with its former default."""
        source = _FakeRFDETR()
        original = source.model.model
        weights = original.linear.weight.detach().clone()

        with pytest.raises(TypeError, match="inplace"):
            source.inference(compile=compile, inplace=inplace)

        assert source.model.model is original
        assert torch.equal(original.linear.weight, weights)
        assert original.training is True
        assert source.model.inference_model is None
        assert source._is_optimized_for_inference is False

    @pytest.mark.parametrize(
        "dtype", [pytest.param(torch.float16, id="float16"), pytest.param(torch.bfloat16, id="bfloat16")]
    )
    def test_dtype_cast_only_changes_independent_runtime(self, dtype: torch.dtype) -> None:
        """Casting a predictor preserves source values, storage, dtype and training mode."""
        source = _FakeRFDETR()
        original = source.model.model
        weights = original.linear.weight.detach().clone()
        predictor = source.inference(compile=False, dtype=dtype)
        runtime = predictor._native_model.model.inference_model

        assert runtime.linear.weight.dtype == dtype
        assert runtime.linear.weight.data_ptr() != original.linear.weight.data_ptr()
        assert runtime.training is False
        assert original.linear.weight.dtype == torch.float32
        assert torch.equal(original.linear.weight, weights)
        assert original.training is True

    def test_repeated_calls_copy_current_base_weights(self) -> None:
        """Each factory copies current base weights even when stale legacy optimization remains on the source."""
        source = _FakeRFDETR()
        legacy = _FakeModel()
        source.model.inference_model = legacy
        source._is_optimized_for_inference = True
        first = source.inference(compile=False)
        first_runtime = first._native_model.model.inference_model
        original_weights = first_runtime.linear.weight.detach().clone()
        with torch.no_grad():
            source.model.model.linear.weight.add_(2)
        second = source.inference(compile=False)
        second_runtime = second._native_model.model.inference_model

        assert torch.equal(first_runtime.linear.weight, original_weights)
        assert torch.equal(second_runtime.linear.weight, source.model.model.linear.weight)
        assert first_runtime.linear.weight.data_ptr() != second_runtime.linear.weight.data_ptr()
        assert source.model.inference_model is legacy
        assert source._is_optimized_for_inference is True

    @pytest.mark.parametrize("dtype", [pytest.param(torch.int8, id="torch-int8"), "int8"])
    def test_non_floating_dtype_rejected_before_copy(self, dtype: torch.dtype | str) -> None:
        """Integer dtypes fail before exporting or copying the source module."""
        source = _FakeRFDETR()
        with (
            patch("rfdetr.detr.deepcopy") as copy,
            patch.object(_FakeModel, "export") as export,
            pytest.raises(ValueError, match="floating-point torch.dtype"),
        ):
            source.inference(compile=False, dtype=dtype)
        copy.assert_not_called()
        export.assert_not_called()
        assert source.model.inference_model is None


class TestModelInferenceExceptionRecovery:
    """Verify state consistency when optimization fails mid-execution."""

    def test_inductor_second_setup_call_failure_leaves_model_fully_unoptimized(self) -> None:
        """A graph-recording failure after allocator warmup should clear all optimized state."""
        rfdetr = _FakeRFDETR()
        rfdetr.model.device = torch.device("cuda")
        dummy_input = torch.randn(1, 3, 28, 28)
        compiled_model = Mock(side_effect=[rfdetr.model.model(dummy_input), RuntimeError("recording failed")])

        with (
            patch("torch.compile", return_value=compiled_model),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module),
            pytest.raises(RuntimeError, match="recording failed"),
        ):
            rfdetr.inference(compile=True, compile_backend="inductor")

        assert compiled_model.call_count == 2
        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    def test_deepcopy_failure_preserves_legacy_state(self) -> None:
        """A copy failure leaves any legacy source runtime and flags untouched."""
        rfdetr = _FakeRFDETR()
        # A failed independent factory must preserve even legacy source optimization state.
        rfdetr._is_optimized_for_inference = True
        rfdetr.model.inference_model = rfdetr.model.model

        with (
            patch("rfdetr.detr.deepcopy", side_effect=RuntimeError("deepcopy failed")),
            pytest.raises(RuntimeError, match="deepcopy failed"),
        ):
            rfdetr.inference(compile=False)

        assert rfdetr.model.inference_model is rfdetr.model.model
        assert rfdetr._is_optimized_for_inference is True

    def test_export_failure_leaves_is_optimized_false(self) -> None:
        """If export() raises after deepcopy succeeds, _is_optimized_for_inference stays False."""
        rfdetr = _FakeRFDETR()

        with (
            patch.object(_FakeModel, "export", side_effect=RuntimeError("export failed")),
            pytest.raises(RuntimeError, match="export failed"),
        ):
            rfdetr.inference(compile=False)

        assert rfdetr._is_optimized_for_inference is False

    def test_jit_trace_failure_leaves_compiled_flags_false(self) -> None:
        """If jit.trace raises, _optimized_has_been_compiled and _optimized_batch_size stay unset."""
        rfdetr = _FakeRFDETR()

        with (
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
            patch("torch.jit.trace", side_effect=lambda module, _input: module),
            patch("torch.jit.freeze", side_effect=RuntimeError("freeze failed")),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module),
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
            patch("torch.jit.trace", side_effect=lambda module, _input: module),
            patch("torch.jit.freeze", side_effect=lambda module: module),
            patch("rfdetr.detr._CUDAGraphInferenceModel", side_effect=RuntimeError("capture failed")),
            patch("rfdetr.detr.torch.randn", return_value=dummy_input),
            patch("rfdetr.detr.torch.cuda.current_device", return_value=0),
            patch("rfdetr.detr.torch.cuda.device", return_value=nullcontext()),
            patch.object(_FakeModel, "to", autospec=True, side_effect=lambda module, *args, **kwargs: module),
            pytest.raises(RuntimeError, match="capture failed"),
        ):
            rfdetr.inference(compile=True, compile_backend="cudagraph")

        assert rfdetr._is_optimized_for_inference is False
        assert rfdetr.model.inference_model is None
        assert rfdetr._optimized_has_been_compiled is False
        assert rfdetr._optimized_batch_size is None

    def test_rejected_cudagraph_request_keeps_previous_predictor(self) -> None:
        """A rejected CUDA request leaves an earlier independent predictor usable and unchanged."""
        source = _FakeRFDETR()
        predictor = source.inference(compile=False, dtype="float16")
        runtime = predictor._native_model.model.inference_model

        with pytest.raises(ValueError, match="cudagraph.*CUDA"):
            source.inference(compile=True, compile_backend="cudagraph")

        assert predictor._native_model.model.inference_model is runtime
        assert predictor._native_model._optimized_dtype == torch.float16
        assert source.model.inference_model is None
        assert source._is_optimized_for_inference is False

    def test_export_failure_isolates_module_mutations(self) -> None:
        """An export that mutates its receiver before failing cannot mutate the source module."""
        source = _FakeRFDETR()
        original = source.model.model

        def mutate_and_fail(module: _FakeModel) -> None:
            """Mutate the copied module and fail before publication.

            Examples:
                >>> module = _FakeModel()
                >>> try:
                ...     mutate_and_fail(module)
                ... except RuntimeError:
                ...     print(module.training)
                False
            """
            module.eval()
            with torch.no_grad():
                module.linear.weight.zero_()
            raise RuntimeError("export failed mid-mutation")

        weights = original.linear.weight.detach().clone()
        with (
            patch.object(_FakeModel, "export", autospec=True, side_effect=mutate_and_fail),
            pytest.raises(RuntimeError, match="export failed mid-mutation"),
        ):
            source.inference(compile=False)

        assert source.model.model is original
        assert original.training is True
        assert torch.equal(original.linear.weight, weights)
        assert source.model.inference_model is None
        assert source._is_optimized_for_inference is False


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
        value = Mock(shape=order.static_input.shape, dtype=order.static_input.dtype, device=order.static_input.device)

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

    @pytest.mark.parametrize(
        ("batch", "size", "dtype", "device"),
        [
            (1, 4, "float16", "cpu"),
            (2, 8, "float16", "cpu"),
            (2, 4, "float32", "cpu"),
            (2, 4, "float16", "meta"),
        ],
    )
    def test_rejects_input_unlike_the_captured_one(self, batch: int, size: int, dtype: str, device: str) -> None:
        """An input differing from the captured batch, size, dtype or device fails instead of being broadcast or cast.

        ``copy_`` into the static buffer would replicate a smaller batch or cast float32 into a float16 graph, so a
        direct ``model.model.inference_model`` caller would get plausible but wrong results without this check.
        """
        wrapper = detr_module._CUDAGraphInferenceModel.__new__(detr_module._CUDAGraphInferenceModel)
        wrapper._static_input = torch.zeros(2, 3, 4, 4, dtype=torch.float16)
        value = torch.zeros(batch, 3, size, size, dtype=getattr(torch, dtype), device=device)

        with pytest.raises(ValueError, match=r"captured for input shape \(2, 3, 4, 4\), dtype torch\.float16"):
            wrapper(value)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([torch.ones(2), torch.zeros(3)], id="list"),
            pytest.param({"pred_logits": torch.ones(2, 3), "pred_boxes": torch.zeros(2, 4)}, id="dict"),
            pytest.param(
                ({"spatial_features": torch.ones(2, 2), "bias": torch.zeros(1)}, [torch.ones(1)]), id="nested"
            ),
            pytest.param((torch.ones(2), torch.zeros(2), torch.full((2,), 3.0)), id="three-tuple"),
            pytest.param((torch.ones(2), None, 7), id="non-tensor-leaf"),
            pytest.param((torch.arange(12.0).reshape(3, 4).t(),), id="non-contiguous"),
        ],
    )
    def test_clone_output_copies_every_tensor_leaf(self, value: Any) -> None:
        """Every tensor leaf is copied into new storage with its layout; containers and other leaves are kept.

        Detection, segmentation and keypoint graphs return different tuple and dict shapes. A tensor that escaped
        uncloned would be overwritten by the next replay, so every branch of ``_clone_output`` must copy its leaves.
        """
        result = detr_module._CUDAGraphInferenceModel._clone_output(value)

        result_leaves = _tensor_leaves(result)
        source_leaves = _tensor_leaves(value)
        assert type(result) is type(value)
        torch.testing.assert_close(result, value, rtol=0, atol=0)
        assert set(map(torch.Tensor.data_ptr, result_leaves)).isdisjoint(map(torch.Tensor.data_ptr, source_leaves))
        assert list(map(torch.Tensor.stride, result_leaves)) == list(map(torch.Tensor.stride, source_leaves))

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
    def test_next_replay_waits_for_a_delayed_clone(self) -> None:
        """A replay from another stream waits until the previous call's clone has read the graph outputs.

        The clone is held back on the device. Had the completion event been recorded before the clone, the second
        stream's replay could overwrite the static outputs first and the first caller would get the second's result.
        """
        device = torch.device("cuda")
        module = torch.jit.trace(_TupleModule().eval().to(device), torch.zeros(1, device=device))
        wrapper = detr_module._CUDAGraphInferenceModel(module, torch.zeros(1, device=device), device)

        def delayed_clone(value: Any) -> Any:
            """Queue a device-side delay ahead of the real clone so a replay from another stream could overtake it.

            Examples:
                Requires a live CUDA device.
                >>> delayed_clone(torch.zeros(1, device="cuda"))  # doctest: +SKIP
            """
            torch.cuda._sleep(20_000_000)
            return detr_module._CUDAGraphInferenceModel._clone_output(value)

        first_stream = torch.cuda.Stream(device=device)
        second_stream = torch.cuda.Stream(device=device)
        with patch.object(wrapper, "_clone_output", delayed_clone):
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
    def test_concurrent_threads_each_get_their_own_result(self) -> None:
        """Threads calling the wrapper at once, each from its own stream, each get the result for their own input.

        A barrier releases every thread together and each replay is held back on the device, so without the host lock
        and the completion event one thread's input copy could land before another thread's replay had read its input.
        """
        device = torch.device("cuda")
        module = torch.jit.trace(_TupleModule().eval().to(device), torch.zeros(1, device=device))
        wrapper = detr_module._CUDAGraphInferenceModel(module, torch.zeros(1, device=device), device)
        graph = wrapper._graph
        inputs = torch.tensor([2.0, 3.0, 5.0, 7.0])
        barrier = threading.Barrier(len(inputs), timeout=30)

        def delayed_replay() -> None:
            """Queue a device-side delay ahead of the replay so another thread's input copy could overtake it.

            Examples:
                Requires the live CUDA graph captured by this test.
                >>> delayed_replay()  # doctest: +SKIP
            """
            torch.cuda._sleep(20_000_000)
            graph.replay()

        def call_from_own_stream(value: torch.Tensor) -> torch.Tensor:
            """Wait for every caller, then call the wrapper from a new stream and stack its two outputs.

            Examples:
                Requires the live CUDA graph and barrier created by this test.
                >>> call_from_own_stream(torch.tensor(2.0))  # doctest: +SKIP
            """
            barrier.wait()
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                return torch.stack(wrapper(value.reshape(1).to(device)))

        with (
            patch.object(wrapper, "_graph", Mock(replay=delayed_replay)),
            ThreadPoolExecutor(max_workers=len(inputs)) as executor,
        ):
            outputs = list(executor.map(call_from_own_stream, inputs))

        torch.cuda.synchronize(device)
        assert torch.equal(torch.cat(outputs, dim=1), torch.stack([inputs + 1, inputs * 2]).to(device))

    @pytest.mark.gpu
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
    def test_capture_on_second_device_replays_the_callers_input(self) -> None:
        """A graph captured on cuda:1 after one on cuda:0 must compute on the caller's input, not the dummy.

        The first capture on ``cuda:0`` is what would create torch's shared default capture stream on that device; the
        later ``cuda:1`` capture must still record its own work and match eager outputs for a new input.
        """
        first_device = torch.device("cuda:0")
        second_device = torch.device("cuda:1")
        with torch.cuda.device(first_device):
            first_module = torch.jit.trace(_TupleModule().eval().to(first_device), torch.zeros(1, device=first_device))
            detr_module._CUDAGraphInferenceModel(first_module, torch.zeros(1, device=first_device), first_device)
        with torch.cuda.device(second_device):
            second_module = torch.jit.trace(
                _TupleModule().eval().to(second_device), torch.zeros(1, device=second_device)
            )
            wrapper = detr_module._CUDAGraphInferenceModel(
                second_module, torch.zeros(1, device=second_device), second_device
            )
        value = torch.tensor([5.0], device=second_device)
        expected = _TupleModule()(value)

        output = wrapper(value)

        torch.cuda.synchronize(second_device)
        assert torch.equal(output[0], expected[0])
        assert torch.equal(output[1], expected[1])

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
            """Fail the wrapper's second synchronize on the test device; any other synchronize call passes through.

            Examples:
                Requires the live CUDA device and closure created by this test.
                >>> fail_completion_fence()  # doctest: +SKIP
            """
            nonlocal fences
            if target == device:
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
    """Inference-only contexts reject work that requires base weights."""

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
