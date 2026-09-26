# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import contextlib
import json
import re
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
from PIL import Image

import rfdetr.export.benchmark as benchmark
from rfdetr.export._tensorrt import inference as trt_inference
from rfdetr.export._tensorrt.inference import TimeProfiler, TRTInference
from rfdetr.export.benchmark import infer_transforms

#: Minimal indexed COCO dataset used to verify evaluator construction.
_MINIMAL_COCO = {
    "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0}],
    "categories": [{"id": 1, "name": "widget"}],
}


class TestTRTInference:
    def test_synchronize_sync_mode_does_not_require_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`synchronize()` should not access stream in sync mode."""
        inference = TRTInference.__new__(TRTInference)
        inference.sync_mode = True
        inference._engine_device = torch.device("cuda", 0)

        mock_is_available = Mock(return_value=True)
        mock_cuda_sync = Mock()
        monkeypatch.setattr("torch.cuda.is_available", mock_is_available)
        monkeypatch.setattr("torch.cuda.synchronize", mock_cuda_sync)

        inference.synchronize()

        mock_is_available.assert_called_once()
        mock_cuda_sync.assert_called_once()

    def test_synchronize_async_mode_uses_stream_sync(self, monkeypatch) -> None:
        """`synchronize()` should use stream synchronization in async mode."""
        inference = TRTInference.__new__(TRTInference)
        inference.sync_mode = False
        inference.stream = Mock()

        mock_cuda_sync = Mock()
        monkeypatch.setattr("torch.cuda.synchronize", mock_cuda_sync)

        inference.synchronize()

        inference.stream.synchronize.assert_called_once()
        mock_cuda_sync.assert_not_called()

    @pytest.mark.parametrize("sync_mode", [True, False])
    def test_synchronize_waits_on_the_engine_device(self, monkeypatch: pytest.MonkeyPatch, sync_mode: bool) -> None:
        """Without a stream to drain, ``synchronize()`` waits on the engine's device, not whichever one is current.

        A bare ``torch.cuda.synchronize()`` waits on the current device, which is ``cuda:0`` for a caller that never
        switched -- so an engine on ``cuda:1`` would be reported finished while its work is still running.
        """
        inference = TRTInference.__new__(TRTInference)
        inference.sync_mode = sync_mode
        inference.stream = None
        inference._engine_device = torch.device("cuda", 1)
        monkeypatch.setattr("torch.cuda.is_available", Mock(return_value=True))
        mock_cuda_sync = Mock()
        monkeypatch.setattr("torch.cuda.synchronize", mock_cuda_sync)

        inference.synchronize()

        mock_cuda_sync.assert_called_once_with(torch.device("cuda", 1))

    def test_time_profiler_synchronizes_its_own_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The profiler waits on the device it times, so work queued there is not charged to the next measurement."""
        monkeypatch.setattr("torch.cuda.is_available", Mock(return_value=True))
        mock_cuda_sync = Mock()
        monkeypatch.setattr("torch.cuda.synchronize", mock_cuda_sync)

        TimeProfiler(device="cuda:1").time()

        mock_cuda_sync.assert_called_once_with("cuda:1")

    def test_infer_transforms_accepts_none_target(self) -> None:
        """Benchmark inference preprocessing should support image-only input."""
        image = Image.new("RGB", (320, 240))

        image_tensor, target = infer_transforms()(image, None)

        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 640, 640)
        assert image_tensor.dtype == torch.float32
        assert target is None


class _FakeRuntime:
    """``tensorrt.Runtime`` stand-in whose ``deserialize_cuda_engine`` hands back :attr:`engine`.

    ``None`` models a file TensorRT cannot deserialize (another TensorRT version or GPU, or a truncated file), which
    TensorRT reports by returning ``None`` rather than raising.

    Examples:
        >>> runtime = _FakeRuntime()
        >>> runtime.deserialize_cuda_engine(b"engine bytes") is None
        True
        >>> runtime.engine = "engine"
        >>> runtime.deserialize_cuda_engine(b"engine bytes")
        'engine'
    """

    def __init__(self) -> None:
        self.engine: _FakeEngine | None = None
        self.deserialize_cuda_engine = Mock(side_effect=lambda payload: self.engine)

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeTensorRTModule(ModuleType):
    """Stand-in ``tensorrt`` module covering what ``TRTInference.__init__`` and ``get_bindings`` touch.

    Every ``trt.Runtime(...)`` returns the same :attr:`runtime`, so a test sets ``runtime.engine`` to the engine the
    file should deserialize to before constructing a ``TRTInference``.
    """

    def __init__(self) -> None:
        super().__init__("tensorrt")
        self.__version__ = "11.3.0.99"
        self.TensorIOMode = SimpleNamespace(INPUT="input", OUTPUT="output")
        self.nptype = lambda dtype: dtype
        self.Logger = Mock()
        self.init_libnvinfer_plugins = Mock()
        self.runtime = _FakeRuntime()
        self.Runtime = Mock(return_value=self.runtime)


class _FakeEngine:
    """Deserialized-engine stand-in: iterates tensor names and answers the shape/dtype/mode/profile queries.

    Shapes use ``-1`` for a dynamic batch axis, as TensorRT reports them; ``profile_max`` is the batch upper bound the
    single optimization profile declares on every dynamic input, whose minimum is batch 1. ``input_dtype`` is the numpy
    type the engine reports for its inputs (outputs are always float32).
    """

    def __init__(
        self,
        tensors: dict[str, tuple[str, tuple[int, ...]]],
        profile_max: int = 4,
        input_dtype: type = np.float32,
    ) -> None:
        self._tensors = tensors
        self.profile_max = profile_max
        self.input_dtype = input_dtype

    def __iter__(self):
        return iter(self._tensors)

    def get_tensor_mode(self, name: str) -> str:
        return self._tensors[name][0]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._tensors[name][1]

    def get_tensor_dtype(self, name: str) -> type:
        return self.input_dtype if self.get_tensor_mode(name) == "input" else np.float32

    def get_tensor_profile_shape(self, name: str, profile_index: int):
        shape = self._tensors[name][1]
        return ((1, *shape[1:]), (2, *shape[1:]), (self.profile_max, *shape[1:]))

    def create_execution_context(self) -> "_FakeContext":
        return _FakeContext(self)


class _FakeContext:
    """Execution-context stand-in that resolves every dynamic shape from the input shapes it has been given.

    TensorRT sizes a dynamic engine's tensors on the execution context, not on the engine, so ``set_input_shape``
    records the batch a call declares and ``get_tensor_shape`` then reports every dynamic tensor at it. A shape outside
    the profile -- a batch below 1 or above the engine's profile maximum, or any other axis differing from the engine's
    -- is refused by returning ``False``, which is how TensorRT reports it instead of raising. ``output_batch`` pins the
    outputs to a batch of their own, modelling an engine whose output batch is not its input batch. The execution calls
    are plain ``Mock`` s so tests can assert on them.
    """

    def __init__(self, engine: _FakeEngine, output_batch: int | None = None) -> None:
        self._engine = engine
        self._output_batch = output_batch
        self._batch: int | None = None
        self.set_input_shape = Mock(side_effect=self._set_input_shape)
        self.get_tensor_shape = Mock(side_effect=self._get_tensor_shape)
        self.set_tensor_address = Mock()
        self.execute_v2 = Mock()
        self.execute_async_v3 = Mock()

    def _set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        if not 1 <= shape[0] <= self._engine.profile_max or shape[1:] != self._engine.get_tensor_shape(name)[1:]:
            return False
        self._batch = int(shape[0])
        return True

    def _get_tensor_shape(self, name: str) -> tuple[int, ...]:
        shape = self._engine.get_tensor_shape(name)
        if shape[0] != -1:
            return shape
        if self._output_batch is not None and self._engine.get_tensor_mode(name) == "output":
            return (self._output_batch, *shape[1:])
        # No input was declared, so TensorRT has nothing to resolve the batch from and still reports it as -1.
        return (-1 if self._batch is None else self._batch, *shape[1:])


class _DeviceRecorder:
    """Stand-in for ``torch.cuda.device``: tracks which device the code under test made current.

    A CPU-only torch cannot enter ``torch.cuda.device`` at all, and a one-GPU machine cannot switch to a second device,
    so the tests observe the device ``TRTInference`` makes current around each TensorRT call instead of TensorRT
    itself. :attr:`current` is ``None`` outside every scope.

    Examples:
        >>> recorder = _DeviceRecorder()
        >>> with recorder("cuda:1"):
        ...     recorder.current
        device(type='cuda', index=1)
        >>> recorder.current is None
        True
    """

    def __init__(self) -> None:
        self.current: torch.device | None = None

    @contextlib.contextmanager
    def __call__(self, device: str | torch.device) -> Iterator[None]:
        previous, self.current = self.current, torch.device(device)
        try:
            yield
        finally:
            self.current = previous


@pytest.fixture
def fake_tensorrt(monkeypatch: pytest.MonkeyPatch) -> _FakeTensorRTModule:
    """Point the module-level ``trt`` handle at a :class:`_FakeTensorRTModule` so no real TensorRT is needed.

    Examples:
        A pytest fixture, so it only runs when a test requests it:

        >>> fake_tensorrt.runtime.engine = _FakeEngine(_STATIC_ENGINE_TENSORS)  # doctest: +SKIP
    """
    module = _FakeTensorRTModule()
    monkeypatch.setattr(trt_inference, "trt", module)
    return module


@pytest.fixture
def cuda_device_recorder(monkeypatch: pytest.MonkeyPatch) -> _DeviceRecorder:
    """Replace ``torch.cuda.device`` with a :class:`_DeviceRecorder`, which CPU-only CI can enter.

    Examples:
        A pytest fixture, so it only runs when a test requests it:

        >>> cuda_device_recorder.current is None  # doctest: +SKIP
        True
    """
    recorder = _DeviceRecorder()
    monkeypatch.setattr(torch.cuda, "device", recorder)
    return recorder


def _runtime_around(
    engine: _FakeEngine, context: _FakeContext | None = None, *, sync_mode: bool = True
) -> TRTInference:
    """Assemble a ``TRTInference`` around a fake engine and context without touching ``__init__`` (needs a GPU).

    A context matching *engine* is built here unless the test needs a non-default one (see :class:`_FakeContext`).
    The engine "runs" on the CPU, so the fakes can hand it ordinary CPU tensors. Calling the runtime needs the
    ``fake_tensorrt`` and ``cuda_device_recorder`` fixtures; the doctest only builds it, and patches ``trt`` itself.

    Examples:
        >>> from unittest.mock import patch
        >>> engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        >>> with patch.object(trt_inference, "trt", _FakeTensorRTModule()):
        ...     runtime = _runtime_around(engine)
        >>> runtime.input_names, runtime.output_names, runtime.bindings["input"].shape
        (['input'], ['dets'], (4, 3, 8, 8))
    """
    runtime = TRTInference.__new__(TRTInference)
    runtime.engine = engine
    runtime.context = _FakeContext(engine) if context is None else context
    runtime.sync_mode = sync_mode
    runtime.device = "cpu"
    runtime._engine_device = torch.device("cpu")
    runtime.stream = None if sync_mode else Mock(handle=7)
    runtime.bindings = runtime.get_bindings(engine, runtime.context, device="cpu")
    runtime.bindings_addr = OrderedDict((n, v.ptr) for n, v in runtime.bindings.items())
    runtime.input_names = runtime.get_input_names()
    runtime.output_names = runtime.get_output_names()
    runtime._prime_context()
    return runtime


@pytest.mark.usefixtures("fake_tensorrt", "cuda_device_recorder")
class TestTRTInferenceDynamicBatch:
    """``TRTInference`` serves engines built with ``dynamic_batch=True`` (a ``-1`` batch axis on every tensor)."""

    def test_dynamic_tensors_are_allocated_at_the_profile_max(self) -> None:
        """A ``-1`` batch axis becomes the profile's max batch so any batch within the profile fits."""
        engine = _FakeEngine(
            {"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4)), "labels": ("output", (-1, 5, 3))},
            profile_max=4,
        )

        runtime = _runtime_around(engine)

        assert runtime.bindings["input"].shape == (4, 3, 8, 8)
        assert runtime.bindings["dets"].shape == (4, 5, 4)
        assert runtime.bindings["dets"].dynamic is True
        assert tuple(runtime.bindings["labels"].data.shape) == (4, 5, 3)

    def test_every_dynamic_input_is_declared_at_its_profile_maximum(self) -> None:
        """Allocation declares each dynamic input at its profile max before it reads any shape back.

        That declaration is what lets the execution context resolve the rest of the graph, so it has to happen for every
        dynamic input and at the maximum the profile allows — anything smaller would under-allocate.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)

        runtime = _runtime_around(engine)

        assert runtime.context.set_input_shape.call_args_list == [call("input", (4, 3, 8, 8))]

    def test_output_buffers_are_sized_by_the_execution_context(self) -> None:
        """Outputs are allocated at the batch TensorRT resolves, not at the batch the inputs were declared with.

        An engine is free to emit an output whose batch axis does not track its input's. Carrying the input's profile
        maximum straight into the output buffer assumes it does, and silently mis-sizes the buffer when it does not, so
        the size has to be read back off the context.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)

        runtime = _runtime_around(engine, _FakeContext(engine, output_batch=7))

        assert runtime.bindings["dets"].shape == (7, 5, 4)

    def test_an_unresolved_non_batch_axis_names_the_tensor_and_the_axis(self) -> None:
        """A dimension left at ``-1`` past the batch axis is reported with its tensor and axis.

        Only axis 0 is exported dynamic, so an engine with another dynamic axis is shaped differently than this runtime
        assumes. Handing that shape to ``np.empty`` reports only "negative dimensions are not allowed", which names
        neither the tensor nor the axis that caused it.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, -1, 4))}, profile_max=4)

        with pytest.raises(ValueError, match=r"'dets'.*axis 1"):
            _runtime_around(engine)

    def test_an_unresolvable_batch_axis_is_reported_rather_than_allocated(self) -> None:
        """An engine with a dynamic output but no dynamic input leaves the batch axis unresolved.

        Nothing declares a shape to the context in that case, so TensorRT keeps reporting ``-1`` and there is no profile
        to size the output buffer from.
        """
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (-1, 5, 4))})

        with pytest.raises(ValueError, match=r"'dets'.*axis 0"):
            _runtime_around(engine)

    def test_input_bindings_carry_no_device_buffer(self) -> None:
        """An input binding allocates nothing, keeping only the shape and a placeholder address.

        Every execution binds the caller's own tensor over the input address, so a buffer allocated here would be paid
        for on the device and never read. The address entry itself still has to exist because ``run_sync`` hands
        ``execute_v2`` the whole address list positionally.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)

        runtime = _runtime_around(engine)

        assert runtime.bindings["input"].data is None
        assert runtime.bindings_addr["input"] == 0
        assert runtime.bindings["input"].shape == (4, 3, 8, 8)

    def test_an_unchanged_input_shape_is_declared_only_once(self) -> None:
        """A repeated batch does not re-issue ``set_input_shape``; the context already holds that shape.

        The construction-time declaration at the profile maximum does not seed the memo, so the first call still
        declares its own shape -- only an exact repeat of the previous call is skipped.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine)
        blob = {"input": torch.zeros(3, 3, 8, 8)}

        runtime(blob)
        runtime(blob)

        assert runtime.context.set_input_shape.call_args_list == [
            call("input", (4, 3, 8, 8)),
            call("input", (3, 3, 8, 8)),
        ]

    def test_a_changed_input_shape_is_declared_again(self) -> None:
        """A different batch always reaches the context, so the memo can never suppress a needed declaration."""
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine)

        runtime({"input": torch.zeros(3, 3, 8, 8)})
        runtime({"input": torch.zeros(2, 3, 8, 8)})

        assert runtime.context.set_input_shape.call_args_list[-2:] == [
            call("input", (3, 3, 8, 8)),
            call("input", (2, 3, 8, 8)),
        ]

    def test_output_addresses_are_registered_once_rather_than_per_call(self) -> None:
        """Output buffers never move, so the async path registers them at construction and only rebinds inputs."""
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine, sync_mode=False)
        blob = {"input": torch.zeros(3, 3, 8, 8)}

        runtime(blob)
        runtime(blob)

        assert [name for (name, _), _ in runtime.context.set_tensor_address.call_args_list] == [
            "dets",
            "input",
            "input",
        ]

    def test_static_tensors_keep_their_shape(self) -> None:
        """A fixed-batch engine is allocated exactly as declared and marked static."""
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (2, 5, 4))})

        runtime = _runtime_around(engine)

        assert runtime.bindings["input"].shape == (2, 3, 8, 8)
        assert runtime.bindings["input"].dynamic is False

    def test_run_sync_declares_the_input_shape_and_trims_outputs(self) -> None:
        """Each call sets the real input shape on the context and returns only the rows the engine produced."""
        engine = _FakeEngine(
            {"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))},
            profile_max=4,
        )
        runtime = _runtime_around(engine)
        blob = {"input": torch.zeros(3, 3, 8, 8)}

        outputs = runtime(blob)

        assert runtime.context.set_input_shape.call_args == call("input", (3, 3, 8, 8))
        runtime.context.execute_v2.assert_called_once()
        assert tuple(outputs["dets"].shape) == (3, 5, 4)
        assert runtime.bindings_addr["input"] == blob["input"].data_ptr()

    def test_run_sync_on_a_static_engine_returns_the_whole_buffer(self) -> None:
        """A fixed-batch engine neither declares shapes nor trims, so the old behaviour is unchanged."""
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (2, 5, 4))})
        runtime = _runtime_around(engine)

        outputs = runtime({"input": torch.zeros(2, 3, 8, 8)})

        runtime.context.set_input_shape.assert_not_called()
        runtime.context.get_tensor_shape.assert_not_called()
        assert tuple(outputs["dets"].shape) == (2, 5, 4)

    def test_batch_beyond_the_profile_is_refused_before_execution(self) -> None:
        """TensorRT reports an out-of-profile shape by returning ``False`` from ``set_input_shape``, not by raising.

        Ignoring that result would execute anyway and hand back the previous call's output buffer contents, so the
        helper must stop before touching the context's execution path.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine)

        with pytest.raises(ValueError, match="outside the engine's optimization profile"):
            runtime({"input": torch.zeros(5, 3, 8, 8)})

        runtime.context.execute_v2.assert_not_called()
        runtime.context.execute_async_v3.assert_not_called()

    def test_a_static_engine_refuses_a_blob_of_the_wrong_shape(self) -> None:
        """A fixed-batch engine validates the blob it is handed instead of binding it by raw pointer.

        Nothing declares a static engine's shape to TensorRT, so there is no ``set_input_shape`` to reject a
        mismatch: the engine would read as many elements as it was built for straight off the blob's device
        pointer, past the end of a smaller tensor, and report nothing.
        """
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (2, 5, 4))})
        runtime = _runtime_around(engine)

        with pytest.raises(ValueError, match="does not match the fixed shape"):
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        runtime.context.execute_v2.assert_not_called()

    def test_run_async_synchronizes_the_stream_before_reading_output_shapes(self) -> None:
        """The stream is drained before the produced batch is read back off the context.

        ``execute_async_v3`` only enqueues the work. Until the stream completes, the context still describes the
        previous call, so reading shapes first would trim this call's outputs to a stale batch.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine, sync_mode=False)
        order = Mock()
        order.attach_mock(runtime.stream.synchronize, "synchronize")
        order.attach_mock(runtime.context.get_tensor_shape, "read_shape")

        runtime({"input": torch.zeros(3, 3, 8, 8)})

        assert [name for name, _, _ in order.mock_calls] == ["synchronize", "read_shape"]

    def test_run_async_registers_every_tensor_address_and_trims_outputs(self) -> None:
        """The async path binds each tensor by name, launches ``execute_async_v3`` on the stream, then syncs it."""
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine, sync_mode=False)
        blob = {"input": torch.zeros(3, 3, 8, 8)}

        outputs = runtime(blob)

        assert runtime.context.set_input_shape.call_args == call("input", (3, 3, 8, 8))
        addresses = {name: address for (name, address), _ in runtime.context.set_tensor_address.call_args_list}
        assert addresses == {"input": blob["input"].data_ptr(), "dets": runtime.bindings["dets"].ptr}
        runtime.context.execute_async_v3.assert_called_once_with(stream_handle=7)
        runtime.context.execute_v2.assert_not_called()
        runtime.stream.synchronize.assert_called_once()
        assert tuple(outputs["dets"].shape) == (3, 5, 4)

    def test_serves_a_sequence_of_differing_batches_on_one_long_lived_runtime(self) -> None:
        """One ``TRTInference`` instance must serve batch 4 -> 1 -> 3 in sequence without cross-call contamination.

        Reproduces the DeepStream/Triton usage from issue #376: one engine, one process, one runtime object reused call
        after call with a different batch each time. Every other test in this class constructs a fresh runtime or
        exercises exactly one batch per instance; this is the only test that keeps one ``TRTInference`` alive across a
        batch sequence and checks both shape and values survive it.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        context = Mock()
        # get_bindings() resolves every dynamic tensor's shape off the context during construction, at the profile
        # maximum -- the return value has to exist before _runtime_around() runs, not just before the first call.
        context.get_tensor_shape.return_value = (4, 5, 4)
        runtime = _runtime_around(engine, context)
        # Construction itself declares "input" once at the profile maximum (_declare_profile_max_inputs); the
        # assertion below is only about the three per-call declarations that follow, so drop that call now.
        context.set_input_shape.reset_mock()

        # Step 1: batch 4 -- fills the whole profile-max buffer. ``execute_v2`` reports launch success as a bool
        # return; the side effect fills the buffer as its effect but must still hand back ``True`` for it.
        def _fill_dets(batch: int, value: float) -> bool:
            runtime.bindings["dets"].data[:batch].fill_(value)
            return True

        context.execute_v2.side_effect = lambda *_a: _fill_dets(4, 4.0)
        outputs = runtime({"input": torch.full((4, 3, 8, 8), 4.0)})
        assert tuple(outputs["dets"].shape) == (4, 5, 4)
        assert torch.equal(outputs["dets"], torch.full((4, 5, 4), 4.0))

        # Step 2: batch 1 -- the smallest legal batch, right after the largest.
        context.get_tensor_shape.return_value = (1, 5, 4)
        context.execute_v2.side_effect = lambda *_a: _fill_dets(1, 1.0)
        outputs = runtime({"input": torch.full((1, 3, 8, 8), 1.0)})
        assert tuple(outputs["dets"].shape) == (1, 5, 4)
        assert torch.equal(outputs["dets"], torch.full((1, 5, 4), 1.0))

        # Step 3: batch 3 -- a third, different size, still on the same runtime object.
        context.get_tensor_shape.return_value = (3, 5, 4)
        context.execute_v2.side_effect = lambda *_a: _fill_dets(3, 3.0)
        outputs = runtime({"input": torch.full((3, 3, 8, 8), 3.0)})
        assert tuple(outputs["dets"].shape) == (3, 5, 4)
        assert torch.equal(outputs["dets"], torch.full((3, 5, 4), 3.0))

        assert context.set_input_shape.call_args_list == [
            call("input", (4, 3, 8, 8)),
            call("input", (1, 3, 8, 8)),
            call("input", (3, 3, 8, 8)),
        ]

    def test_get_dummy_input_uses_the_configured_device(self) -> None:
        """The dummy input must land on the engine's own device at the requested batch, not a hardcoded ``"cuda:0"``.

        Regression guard: ``get_dummy_input`` used to hardcode ``.to("cuda:0")``, so a runtime built for the CPU (as
        every fake-engine test in this module is) would still hand back a CUDA tensor.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine)
        # ``meta`` stands in for a device other than the CPU, which is also torch's default device.
        runtime._engine_device = torch.device("meta")

        blob = runtime.get_dummy_input(batch_size=2)

        assert blob["input"].shape == (2, 3, 8, 8)
        assert blob["input"].device == torch.device("meta")


#: A fixed-batch engine with one ``(1, 3, 8, 8)`` float32 input and one output, as the unit tests below build it.
_STATIC_ENGINE_TENSORS = {"input": ("input", (1, 3, 8, 8)), "dets": ("output", (1, 5, 4))}


@pytest.mark.usefixtures("fake_tensorrt", "cuda_device_recorder")
class TestTRTInferenceInputValidation:
    """TensorRT reads each input straight off its pointer, as a dense buffer of the engine's own dtype and device.

    Anything else -- another memory layout, dtype or device -- is read as if it were that buffer, so the runtime has to
    refuse it before binding rather than return detections computed from misread memory.
    """

    @pytest.mark.parametrize(
        "make_input",
        [
            pytest.param(lambda: torch.rand(1, 3, 8, 8).to(memory_format=torch.channels_last), id="channels_last"),
            pytest.param(lambda: torch.rand(1, 3, 8, 16)[..., ::2], id="strided"),
        ],
    )
    def test_rejects_a_non_contiguous_input(self, make_input: Callable[[], torch.Tensor]) -> None:
        """A strided tensor is read as dense memory; on a real engine ``channels_last`` took 13 detections to 0."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS))

        with pytest.raises(ValueError, match="not contiguous"):
            runtime({"input": make_input()})

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(torch.float16, id="float16"), pytest.param(torch.float64, id="float64")],
    )
    def test_rejects_an_input_of_another_dtype(self, dtype: torch.dtype) -> None:
        """The engine reads ``4 * numel`` bytes of float32 whatever the tensor holds, past the end of a float16 one."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS))

        with pytest.raises(ValueError, match="dtype"):
            runtime({"input": torch.rand(1, 3, 8, 8, dtype=dtype)})

    def test_the_accepted_dtype_is_the_engines_own(self) -> None:
        """An engine that reports float16 inputs takes a float16 tensor: the dtype comes from the engine, not a literal.

        Every engine RF-DETR exports today keeps float32 inputs, so this passes with a hard-coded float32 check too --
        except for the engine this test builds.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS, input_dtype=np.float16))
        blob = {"input": torch.rand(1, 3, 8, 8, dtype=torch.float16)}

        runtime(blob)

        assert runtime.bindings_addr["input"] == blob["input"].data_ptr()

    @pytest.mark.parametrize(
        ("engine_device", "input_device"),
        [("cpu", "meta"), ("cuda:0", "cpu"), ("cuda:0", "cuda:1")],
    )
    def test_rejects_an_input_on_another_device(self, engine_device: str, input_device: str) -> None:
        """A tensor on another device, including another GPU, hands TensorRT a pointer it cannot read.

        On a GPU engine a CPU tensor ended in ``cudaError 700: an illegal memory access``, which leaves the process's
        CUDA context unusable. CPU CI has no CUDA tensor to hand over, so the input is a stand-in carrying the
        attributes the checks read; the device is checked first.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS))
        runtime._engine_device = torch.device(engine_device)
        stand_in = SimpleNamespace(
            device=torch.device(input_device),
            dtype=torch.float32,
            shape=torch.Size((1, 3, 8, 8)),
            is_contiguous=lambda: True,
            data_ptr=lambda: 0,
        )

        with pytest.raises(ValueError, match="device"):
            runtime({"input": stand_in})

    @pytest.mark.parametrize(
        ("engine_shape", "shape", "misleading_advice"),
        [
            pytest.param((-1, 3, 8, 8), (0, 3, 8, 8), "max_batch_size", id="dynamic-empty-batch"),
            pytest.param((-1, 3, 8, 8), (2, 3, 9, 9), "max_batch_size", id="dynamic-other-image-size"),
            pytest.param((-1, 3, 8, 8), (5, 3, 8), "max_batch_size", id="dynamic-missing-axis"),
            pytest.param((-1, 3, 8, 8), (5, 3, 9, 9), "max_batch_size", id="dynamic-above-max-other-image-size"),
            pytest.param((2, 3, 8, 8), (2, 3, 9, 9), "dynamic_batch", id="static-other-image-size"),
            pytest.param((2, 3, 8, 8), (0, 3, 8, 8), "dynamic_batch", id="static-empty-batch"),
            pytest.param((4,), (), "dynamic_batch", id="static-scalar-input"),
        ],
    )
    def test_a_shape_refusal_advises_only_a_fix_that_applies(
        self, engine_shape: tuple[int, ...], shape: tuple[int, ...], misleading_advice: str
    ) -> None:
        """A larger ``max_batch_size`` or ``dynamic_batch=True`` only fixes a batch the engine cannot take.

        An empty batch or a different image size is refused too, and pointing at the batch bounds would have the user
        export again an engine that refuses the same input.
        """
        engine = _FakeEngine({"input": ("input", engine_shape), "dets": ("output", (engine_shape[0], 5, 4))})
        runtime = _runtime_around(engine)

        with pytest.raises(ValueError, match=re.escape(str(shape))) as refusal:
            runtime({"input": torch.zeros(shape)})

        assert misleading_advice not in str(refusal.value)

    @pytest.mark.parametrize(
        ("engine_shape", "shape", "advice"),
        [
            pytest.param((-1, 3, 8, 8), (5, 3, 8, 8), "Export with a larger max_batch_size", id="dynamic-above-max"),
            pytest.param((2, 3, 8, 8), (1, 3, 8, 8), "Export with dynamic_batch=True", id="static-other-batch"),
        ],
    )
    def test_a_batch_refusal_keeps_its_advice(
        self, engine_shape: tuple[int, ...], shape: tuple[int, ...], advice: str
    ) -> None:
        """Where only the batch is wrong, the refusal still names the export setting that fixes it (unchanged)."""
        engine = _FakeEngine({"input": ("input", engine_shape), "dets": ("output", (engine_shape[0], 5, 4))})
        runtime = _runtime_around(engine)

        with pytest.raises(ValueError, match=re.escape(advice)):
            runtime({"input": torch.zeros(shape)})

    @pytest.mark.parametrize(
        ("shape", "advised"),
        [
            pytest.param((5, 3, 4, 4), True, id="smallest-image"),
            pytest.param((5, 3, 6, 6), True, id="mid-range-image"),
            pytest.param((5, 3, 2, 2), False, id="image-below-range"),
            pytest.param((5, 3, 9, 9), False, id="image-above-range"),
        ],
    )
    def test_the_batch_advice_on_an_engine_with_a_dynamic_image_size(
        self, shape: tuple[int, ...], advised: bool
    ) -> None:
        """Each image axis is compared with its own profile range (4x4 to 8x8 here), not with the maximum alone.

        Five images of an in-range size are refused for the batch alone, so a larger ``max_batch_size`` fixes it; five
        images outside the range would still be refused after that re-export.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        engine.get_tensor_profile_shape = lambda name, profile_index: ((1, 3, 4, 4), (2, 3, 8, 8), (4, 3, 8, 8))
        runtime = _runtime_around(engine)

        with pytest.raises(ValueError, match=re.escape(str(shape))) as refusal:
            runtime({"input": torch.zeros(shape)})

        assert ("Export with a larger max_batch_size" in str(refusal.value)) is advised


@pytest.mark.usefixtures("cuda_device_recorder")
class TestTRTInferenceEngineLoading:
    """TensorRT reports an engine or context it cannot create by returning ``None``, never by raising."""

    def test_an_undeserializable_engine_raises_a_rebuild_hint(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """An engine from another TensorRT version or GPU, or a truncated copy, names the file and the fix.

        Using the ``None`` unchecked surfaced as ``AttributeError: 'NoneType' object has no attribute
        'create_execution_context'``, which says nothing about the engine file.
        """
        engine_file = tmp_path / "foreign.trt"
        engine_file.write_bytes(b"not an engine for this machine")
        fake_tensorrt.runtime.engine = None

        with pytest.raises(RuntimeError, match=re.escape(str(engine_file)) + ".*Rebuild"):
            TRTInference(str(engine_file), device="cuda:0", sync_mode=True)

    def test_a_missing_execution_context_is_reported(self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path) -> None:
        """A context TensorRT could not create is reported at construction, not at the first call."""
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))})
        engine.create_execution_context = lambda: None
        fake_tensorrt.runtime.engine = engine

        with pytest.raises(RuntimeError, match="execution context"):
            TRTInference(str(engine_file), device="cuda:0", sync_mode=True)


@pytest.mark.usefixtures("fake_tensorrt")
class TestTRTInferenceDevice:
    """TensorRT binds an engine to the device current when it is loaded, and every launch must find it current again.

    ``benchmark.main(device=1)`` asks for ``cuda:1``; before this was honoured, the engine was loaded and run on
    whichever device was current (``cuda:0``) while its buffers and inputs sat on ``cuda:1``.
    """

    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            ("cuda:1", "cuda:1"),
            ("cuda", "cuda:3"),
            pytest.param(torch.device("cuda", 2), "cuda:2", id="torch.device"),
        ],
    )
    def test_the_engine_is_loaded_on_the_requested_device(
        self,
        fake_tensorrt: _FakeTensorRTModule,
        cuda_device_recorder: _DeviceRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        device: str | torch.device,
        expected: str,
    ) -> None:
        """Deserialization and context creation both run with the requested device current.

        A bare ``"cuda"`` is pinned to the device current at construction (3 here), since that is where TensorRT puts
        the engine. The engine has no outputs, so no buffer is allocated on a CUDA device a CPU-only CI does not have.
        """
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))})
        active: dict[str, torch.device | None] = {}

        def deserialize(payload: bytes) -> _FakeEngine:
            active["deserialize"] = cuda_device_recorder.current
            return engine

        def create_context() -> _FakeContext:
            active["context"] = cuda_device_recorder.current
            return _FakeContext(engine)

        fake_tensorrt.runtime.deserialize_cuda_engine.side_effect = deserialize
        engine.create_execution_context = create_context
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")

        TRTInference(str(engine_file), device=device, sync_mode=True)

        assert active == {"deserialize": torch.device(expected), "context": torch.device(expected)}

    @pytest.mark.usefixtures("cuda_device_recorder")
    def test_the_runtime_timer_waits_on_the_engine_device(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """The ``TimeProfiler`` that ``speed()`` times with waits on the engine's device, not on the current one."""
        fake_tensorrt.runtime.engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))})
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")

        runtime = TRTInference(str(engine_file), device="cuda:1", sync_mode=True)

        assert runtime.time_profile.device == torch.device("cuda", 1)

    @pytest.mark.parametrize("sync_mode", [True, False])
    def test_execution_runs_on_the_engine_device(self, cuda_device_recorder: _DeviceRecorder, sync_mode: bool) -> None:
        """Each launch makes the engine's device current, not whatever ``device`` resolves to at call time.

        The caller asked for a bare ``"cuda"``, which construction pinned (to the fakes' CPU here); re-resolving
        ``"cuda"`` per call would follow the caller's current device away from the engine.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=sync_mode)
        runtime.device = "cuda"
        active: list[torch.device | None] = []

        def launch(*args: object, **kwargs: object) -> bool:
            active.append(cuda_device_recorder.current)
            return True

        runtime.context.execute_v2.side_effect = launch
        runtime.context.execute_async_v3.side_effect = launch

        runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert active == [torch.device("cpu")]

    def test_a_non_cuda_device_is_refused(self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path) -> None:
        """TensorRT cannot run on the CPU; ``device="cpu"`` used to load the engine with its buffers in host memory."""
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        fake_tensorrt.runtime.engine = _FakeEngine(_STATIC_ENGINE_TENSORS)

        with pytest.raises(ValueError, match="CUDA"):
            TRTInference(str(engine_file), device="cpu", sync_mode=True)

    def test_output_buffers_default_to_the_engine_device(self) -> None:
        """``get_bindings`` without a ``device`` allocates where the engine runs, as its docstring promises.

        ``.to(None)`` is a no-op, so the default used to leave every output buffer in host memory. ``meta`` stands in
        for a device other than the CPU the buffers start on.
        """
        engine = _FakeEngine(_STATIC_ENGINE_TENSORS)
        runtime = _runtime_around(engine)
        runtime._engine_device = torch.device("meta")

        bindings = runtime.get_bindings(engine, _FakeContext(engine))

        assert bindings["dets"].data.device == torch.device("meta")

    @pytest.mark.parametrize(
        "input_dtype", [pytest.param(np.float32, id="float32"), pytest.param(np.float16, id="float16")]
    )
    @pytest.mark.usefixtures("cuda_device_recorder")
    def test_the_runtime_accepts_the_dummy_input_it_builds(self, input_dtype: type) -> None:
        """``get_dummy_input`` builds what the input checks accept: the engine's own dtype, on the engine's device.

        The caller asked for a bare ``"cuda"``, which construction pinned (to the fakes' CPU here); building on
        ``"cuda"`` again would follow the caller's current device, and a float32 dummy is refused by a float16 engine.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS, input_dtype=input_dtype))
        runtime.device = "cuda"

        runtime(runtime.get_dummy_input(batch_size=1))

        runtime.context.execute_v2.assert_called_once()


class TestBenchmarkMain:
    @pytest.mark.parametrize(
        ("device", "expected_torch_device"),
        [
            pytest.param(0, "cuda:0", id="default-device"),
            pytest.param(7, "cuda:7", id="non-default-device"),
        ],
    )
    def test_onnx_benchmark_uses_requested_cuda_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
        device: int,
        expected_torch_device: str,
    ) -> None:
        """ONNX Runtime and PyTorch should use the requested CUDA device."""
        session = Mock()
        inference_session = Mock(return_value=session)
        onnxruntime = ModuleType("onnxruntime")
        onnxruntime.InferenceSession = inference_session  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)

        monkeypatch.setattr(benchmark, "get_image_list", Mock(return_value=[]))
        infer_onnx = Mock()
        monkeypatch.setattr(benchmark, "infer_onnx", infer_onnx)

        benchmark.main("model.onnx", device=device, disable_eval=True)

        inference_session.assert_called_once_with(
            "model.onnx",
            providers=[("CUDAExecutionProvider", {"device_id": device})],
        )
        infer_onnx.assert_called_once()
        assert infer_onnx.call_args.args[0] is session
        assert infer_onnx.call_args.kwargs["device"] == expected_torch_device
        assert infer_onnx.call_args.kwargs["repeats"] == 1

    @pytest.mark.parametrize("device", [0, 7])
    def test_trt_benchmark_uses_requested_cuda_device(self, monkeypatch: pytest.MonkeyPatch, device: int) -> None:
        """The TensorRT branch hands the requested device to the runtime, the input pipeline and the latency timer."""
        monkeypatch.setattr(benchmark, "get_image_list", Mock(return_value=[]))
        runtime_class = Mock()
        monkeypatch.setattr(benchmark, "TRTInference", runtime_class)
        infer_engine = Mock()
        monkeypatch.setattr(benchmark, "infer_engine", infer_engine)

        benchmark.main("model.trt", device=device, disable_eval=True)

        runtime_class.assert_called_once_with("model.trt", sync_mode=True, device=f"cuda:{device}")
        assert infer_engine.call_args.kwargs["device"] == f"cuda:{device}"
        assert infer_engine.call_args.args[2].device == f"cuda:{device}"

    def test_eval_enabled_passes_a_loaded_coco_object_to_the_evaluator(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``main`` loads the annotation file so ``CocoEvaluator`` receives a COCO object, not a path string."""
        pytest.importorskip("faster_coco_eval")
        annotations = tmp_path / "annotations"
        annotations.mkdir()
        (annotations / "instances_val2017.json").write_text(json.dumps(_MINIMAL_COCO))
        onnxruntime = ModuleType("onnxruntime")
        onnxruntime.InferenceSession = Mock(return_value=Mock())  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
        infer_onnx = Mock()
        monkeypatch.setattr(benchmark, "infer_onnx", infer_onnx)

        benchmark.main("model.onnx", coco_path=str(tmp_path), disable_eval=False)

        infer_onnx.assert_called_once()
        evaluator = infer_onnx.call_args.args[1]
        assert evaluator.cat_ids == {1}


class TestBenchmarkShapeParameterization:
    """Benchmark preprocessing/postprocessing read input size and query count instead of hardcoding 640/300."""

    def test_infer_transforms_uses_requested_size(self) -> None:
        """infer_transforms resizes to the caller-supplied (height, width)."""
        image = Image.new("RGB", (320, 240))

        image_tensor, _ = infer_transforms((512, 384))(image, None)

        assert image_tensor.shape == (3, 512, 384)

    def test_infer_transforms_defaults_to_640(self) -> None:
        """The default input size stays 640x640 for callers that do not pass a size."""
        image = Image.new("RGB", (320, 240))

        image_tensor, _ = infer_transforms()(image, None)

        assert image_tensor.shape == (3, 640, 640)

    def test_static_dim_returns_concrete_int(self) -> None:
        """A concrete positive dimension is returned unchanged."""
        from rfdetr.export.benchmark import _static_dim

        assert _static_dim(384, 640) == 384

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("height", id="dynamic-string"),
            pytest.param(None, id="none"),
            pytest.param(-1, id="negative"),
        ],
    )
    def test_static_dim_falls_back_for_dynamic_axis(self, value) -> None:
        """Dynamic/unknown axes fall back to the provided default."""
        from rfdetr.export.benchmark import _static_dim

        assert _static_dim(value, 640) == 640

    def test_post_process_respects_num_queries(self) -> None:
        """post_process selects exactly num_queries detections per image."""
        from rfdetr.export.benchmark import post_process

        num_queries = 5
        outputs = {
            "labels": torch.rand(1, 20, 3),
            "dets": torch.rand(1, 20, 4),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = post_process(outputs, target_sizes, num_queries=num_queries)

        assert results[0]["scores"].shape == (num_queries,)

    def test_post_process_repeats_boxes_for_duplicated_topk_queries(self) -> None:
        """Top-k over the flattened [Q, C] scores can pick the same query under two classes.

        Each pick must reproduce that query's exact box, so duplicated and out-of-order query indices have to copy the
        source row verbatim for every occurrence.
        """
        from rfdetr.export.benchmark import box_cxcywh_to_xyxy, post_process

        logits = torch.full((1, 4, 3), -10.0)
        logits[0, 2, 0] = 3.0  # query 2, class 0 -> rank 1
        logits[0, 2, 1] = 2.0  # query 2, class 1 -> rank 2 (same query twice)
        logits[0, 1, 2] = 1.0  # query 1, class 2 -> rank 3
        dets = torch.rand(1, 4, 4)
        target_sizes = torch.tensor([[480, 640]])

        results = post_process({"labels": logits, "dets": dets}, target_sizes, num_queries=3)

        scale = torch.tensor([640.0, 480.0, 640.0, 480.0])
        expected = box_cxcywh_to_xyxy(dets[0]) * scale
        assert torch.equal(results[0]["labels"], torch.tensor([0, 1, 2]))
        assert torch.equal(results[0]["boxes"][0], expected[2])
        assert torch.equal(results[0]["boxes"][1], expected[2])
        assert torch.equal(results[0]["boxes"][2], expected[1])
