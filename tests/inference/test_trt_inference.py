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
from rfdetr.export._tensorrt.exporter import TensorRTExporter
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

        mock_cuda_sync.assert_called_once_with(torch.device("cuda", 1))

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
        #: TensorRT's own default: an engine that carries host code is refused unless the caller opts in.
        self.engine_host_code_allowed = False
        self.deserialize_cuda_engine = Mock(side_effect=lambda payload: self.engine)

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _NoSwitchRuntime(_FakeRuntime):
    """``tensorrt.Runtime`` from a release that predates ``engine_host_code_allowed``: setting it to ``True`` fails.

    Real ``pybind11`` classes raise ``AttributeError`` for an attribute they do not define; the initial ``False`` is
    what :class:`_FakeRuntime` assigns to model TensorRT's default, so only ``True`` raises.

    Examples:
        >>> runtime = _NoSwitchRuntime()
        >>> runtime.engine_host_code_allowed = True
        Traceback (most recent call last):
        ...
        AttributeError: 'Runtime' object has no attribute 'engine_host_code_allowed'
    """

    def __setattr__(self, name: str, value: object) -> None:
        if name == "engine_host_code_allowed" and value is True:
            raise AttributeError("'Runtime' object has no attribute 'engine_host_code_allowed'")
        super().__setattr__(name, value)


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
    type the engine reports for its inputs (outputs are always float32). An image axis reported as ``-1`` ranges over
    ``image_profile`` (minimum, maximum) in profile 0; ``second_image_profile`` adds a second profile, identical but for
    ranging the image axes over that instead. A profile's optimal shape is batch 2, halfway through its image range.
    """

    def __init__(
        self,
        tensors: dict[str, tuple[str, tuple[int, ...]]],
        profile_max: int = 4,
        input_dtype: type = np.float32,
        image_profile: tuple[int, int] = (4, 16),
        second_image_profile: tuple[int, int] | None = None,
    ) -> None:
        self._tensors = tensors
        self.profile_max = profile_max
        self.input_dtype = input_dtype
        self.image_profile = image_profile
        self.second_image_profile = second_image_profile
        self.num_optimization_profiles = 1 if second_image_profile is None else 2

    def __iter__(self):
        return iter(self._tensors)

    def get_tensor_mode(self, name: str) -> str:
        return self._tensors[name][0]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._tensors[name][1]

    def get_tensor_dtype(self, name: str) -> type:
        return self.input_dtype if self.get_tensor_mode(name) == "input" else np.float32

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Report the (minimum, optimal, maximum) shape of input *name* in profile *profile_index*: batch from 1 to
        ``profile_max``, images over that profile's range."""
        low, high = self.image_profile if profile_index == 0 else self.second_image_profile
        rest = self._tensors[name][1][1:]
        floor = tuple(low if dim == -1 else dim for dim in rest)
        middle = tuple((low + high) // 2 if dim == -1 else dim for dim in rest)
        ceiling = tuple(high if dim == -1 else dim for dim in rest)
        return ((1, *floor), (2, *middle), (self.profile_max, *ceiling))

    def create_execution_context(self) -> "_FakeContext":
        return _FakeContext(self)


class _FakeContext:
    """Execution-context stand-in that resolves every dynamic shape from the input shapes it has been given.

    TensorRT sizes a dynamic engine's tensors on the execution context, not on the engine, so ``set_input_shape``
    records the shape a call declares and ``get_tensor_shape`` then reports a declared input at that shape and every
    other dynamic tensor at its batch. A shape outside the profile -- a batch below 1 or above the engine's profile
    maximum, or any other axis outside the range the profile gives it -- is refused by returning ``False``, which is how
    TensorRT reports it instead of raising. ``output_batch`` pins the outputs to a batch of their own, modelling an
    engine whose output batch is not its input batch. The execution calls are plain ``Mock`` objects so tests can assert
    on them.
    """

    def __init__(self, engine: _FakeEngine, output_batch: int | None = None) -> None:
        self._engine = engine
        self._output_batch = output_batch
        self._batch: int | None = None
        self._declared: dict[str, tuple[int, ...]] = {}
        self.set_input_shape = Mock(side_effect=self._set_input_shape)
        self.get_tensor_shape = Mock(side_effect=self._get_tensor_shape)
        self.set_tensor_address = Mock()
        self.execute_v2 = Mock()
        self.execute_async_v3 = Mock()

    def _set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        """Accept *shape* for input *name* if every axis is inside the profile, and remember it; else report
        ``False``."""
        low, _, high = self._engine.get_tensor_profile_shape(name, 0)
        if len(shape) != len(low) or not all(lo <= dim <= hi for dim, lo, hi in zip(shape, low, high)):
            return False
        self._batch = int(shape[0])
        self._declared[name] = tuple(shape)
        return True

    def _get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Report tensor *name*'s shape the way TensorRT would, resolving a dynamic batch from what was declared."""
        shape = self._engine.get_tensor_shape(name)
        if shape[0] != -1:
            return shape
        if self._output_batch is not None and self._engine.get_tensor_mode(name) == "output":
            return (self._output_batch, *shape[1:])
        if name in self._declared:
            return self._declared[name]
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


class _FakeStream:
    """``torch.cuda.Stream`` stand-in: a handle TensorRT could launch on, plus counters for the calls a test asserts on.

    Examples:
        >>> events = []
        >>> stream = _FakeStream(handle=11, events=events)
        >>> stream.synchronize()
        >>> stream.wait_stream("other")
        >>> stream.cuda_stream, stream.synchronized, stream.waited_on, events
        (11, 1, ['other'], ['sync', 'wait'])
    """

    def __init__(self, handle: int = 11, events: list[str] | None = None) -> None:
        self.cuda_stream = handle
        self.synchronized = 0
        self.waited_on: list[object] = []
        self._events = events

    def synchronize(self) -> None:
        """Count one wait for the stream, and log it if the test keeps an event log."""
        self.synchronized += 1
        if self._events is not None:
            self._events.append("sync")

    def wait_stream(self, other: object) -> None:
        """Record the stream this one was made to wait for, and log it if the test keeps an event log."""
        self.waited_on.append(other)
        if self._events is not None:
            self._events.append("wait")


class _FakeCudaGraph:
    """``torch.cuda.CUDAGraph`` stand-in: notes whether a capture is open and counts replays.

    Examples:
        >>> recorder = _FakeCudaGraphs()
        >>> graph = _FakeCudaGraph(recorder)
        >>> graph.replay(), graph.replays
        (None, 1)
    """

    def __init__(self, recorder: "_FakeCudaGraphs") -> None:
        self._recorder = recorder
        self.capturing = False
        self.replays = 0

    def replay(self) -> None:
        """Count one replay and run the recorder's ``on_replay`` hook, if a test set one."""
        self.replays += 1
        self._recorder.events.append("replay")
        if self._recorder.on_replay is not None:
            self._recorder.on_replay(self)


class _FakeCudaGraphs:
    """Stand-in for ``torch.cuda.CUDAGraph`` / ``graph`` / ``stream`` / ``Stream`` / ``current_stream``.

    A CPU-only torch cannot capture, so the code under test runs against this instead. Capturing ``graph`` runs its body
    (so the TensorRT launch is recorded) and marks it as captured; ``on_replay`` lets a test make a replay do what the
    captured launch would, and ``fail_next_capture`` makes the next capture raise the way an uncapturable engine does --
    and, like ``torch.cuda.graph`` when ending a capture fails, leave the capture stream current. ``events`` logs the
    order of stream scopes and replays; the other lists record what the code under test asked torch for.

    Examples:
        >>> recorder = _FakeCudaGraphs()
        >>> graph = recorder.new_graph()
        >>> with recorder.capture(graph):
        ...     graph.capturing
        True
        >>> graph.capturing, len(recorder.graphs)
        (False, 1)
        >>> with recorder.stream("any stream"):
        ...     "still runs"
        'still runs'
        >>> before, leaked = recorder.current, _FakeStream(handle=5)
        >>> recorder.fail_next_capture = True
        >>> try:
        ...     with recorder.capture(recorder.new_graph(), stream=leaked):
        ...         pass
        ... except RuntimeError as error:
        ...     print(error)
        capture refused
        >>> recorder.current is leaked
        True
        >>> recorder.set_stream(before)
        >>> recorder.current is before, recorder.capture_streams
        (True, [None, <...>])
        >>> recorder.new_stream(device="cuda:1").cuda_stream, recorder.stream_kwargs
        (11, [{'device': 'cuda:1'}])
        >>> recorder.get_current_stream("cuda:1") is before, recorder.current_stream_args
        (True, [('cuda:1',)])
    """

    def __init__(self) -> None:
        self.graphs: list[_FakeCudaGraph] = []
        self.on_replay: Callable[[_FakeCudaGraph], None] | None = None
        self.fail_next_capture = False
        self.current = _FakeStream(handle=1)
        self.events: list[str] = []
        self.capture_streams: list[object | None] = []
        self.stream_kwargs: list[dict[str, object]] = []
        self.current_stream_args: list[tuple[object, ...]] = []

    def new_graph(self) -> _FakeCudaGraph:
        """Hand out a graph and remember it."""
        graph = _FakeCudaGraph(self)
        self.graphs.append(graph)
        return graph

    @contextlib.contextmanager
    def capture(self, graph: _FakeCudaGraph, stream: object | None = None) -> Iterator[None]:
        """Stand-in for ``torch.cuda.graph``: refuse once if asked, else mark *graph* as capturing for the body."""
        self.capture_streams.append(stream)
        if self.fail_next_capture:
            self.fail_next_capture = False
            self.current = stream
            raise RuntimeError("capture refused")
        graph.capturing = True
        try:
            yield
        finally:
            graph.capturing = False

    @contextlib.contextmanager
    def stream(self, stream: object) -> Iterator[None]:
        """Stand-in for ``torch.cuda.stream``: only logs when its scope opens and closes."""
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")

    def record_copies(self, patch: pytest.MonkeyPatch) -> None:
        """Log ``"copy"`` to ``events`` whenever a tensor is copied into another, so a test sees the scope it ran in.

        Examples:
            >>> recorder = _FakeCudaGraphs()
            >>> with pytest.MonkeyPatch.context() as patch:
            ...     recorder.record_copies(patch)
            ...     _ = torch.zeros(2).copy_(torch.ones(2))
            >>> recorder.events
            ['copy']
        """
        original, events = torch.Tensor.copy_, self.events

        def logged_copy(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            events.append("copy")
            return original(tensor, *args, **kwargs)

        patch.setattr(torch.Tensor, "copy_", logged_copy)

    def set_stream(self, stream: object) -> None:
        """Stand-in for ``torch.cuda.set_stream``."""
        self.current = stream

    def new_stream(self, **kwargs: object) -> _FakeStream:
        """Stand-in for ``torch.cuda.Stream``: remember what it was asked for."""
        self.stream_kwargs.append(kwargs)
        return _FakeStream(handle=11)

    def get_current_stream(self, *args: object) -> _FakeStream:
        """Stand-in for ``torch.cuda.current_stream``: remember which device it was asked about."""
        self.current_stream_args.append(args)
        return self.current


@pytest.fixture
def fake_cuda_graphs(monkeypatch: pytest.MonkeyPatch) -> _FakeCudaGraphs:
    """Replace the torch CUDA-graph and stream API with a :class:`_FakeCudaGraphs`, which CPU-only CI can enter.

    Examples:
        A pytest fixture, so it only runs when a test requests it:

        >>> fake_cuda_graphs.graphs  # doctest: +SKIP
        []
    """
    recorder = _FakeCudaGraphs()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", recorder.new_graph)
    monkeypatch.setattr(torch.cuda, "graph", recorder.capture)
    monkeypatch.setattr(torch.cuda, "stream", recorder.stream)
    monkeypatch.setattr(torch.cuda, "Stream", recorder.new_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", recorder.get_current_stream)
    monkeypatch.setattr(torch.cuda, "set_stream", recorder.set_stream)
    return recorder


def _runtime_around(
    engine: _FakeEngine, context: _FakeContext | None = None, *, sync_mode: bool = True, cuda_graph: bool = False
) -> TRTInference:
    """Assemble a ``TRTInference`` around a fake engine and context without touching ``__init__`` (needs a GPU).

    A context matching *engine* is built here unless the test needs a non-default one (see :class:`_FakeContext`).
    Skipping ``__init__`` also skips its refusals, such as that of an engine whose profile varies more than the batch
    under ``cuda_graph=True``, so a test must not build a runtime here that construction would refuse.
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
    runtime.stream = None if sync_mode or cuda_graph else Mock(handle=7)
    runtime._graph_stream = _FakeStream(handle=11) if cuda_graph else None
    runtime._graphs = {}
    runtime._static_inputs = {}
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

    def test_a_bad_input_is_named_among_several_good_ones(self) -> None:
        """``_check_input_memory`` attributes the refusal to the one bad input, not the whole call.

        Every other input-validation test above uses a single-input engine, so nothing confirms the per-name loop in
        ``_bind_inputs`` reports the right name once an engine has more than one input.
        """
        engine = _FakeEngine(
            {"a": ("input", (1, 3, 8, 8)), "b": ("input", (1, 3, 8, 8)), "dets": ("output", (1, 5, 4))}
        )
        runtime = _runtime_around(engine)
        good = torch.rand(1, 3, 8, 8)
        bad = torch.rand(1, 3, 8, 8).to(memory_format=torch.channels_last)  # non-contiguous, same shape as "good"

        with pytest.raises(ValueError, match="'b'.*not contiguous"):
            runtime({"a": good, "b": bad})

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
        ("engine_shape", "shape", "export_batch_size"),
        [
            pytest.param((-1, 3, 8, 8), (5, 3, 8, 8), 2, id="dynamic-above-max"),
            pytest.param((2, 3, 8, 8), (3, 3, 8, 8), 2, id="static-larger-batch"),
            pytest.param((2, 3, 8, 8), (1, 3, 8, 8), 2, id="static-smaller-batch"),
        ],
    )
    def test_a_batch_refusal_advises_an_export_that_serves_the_refused_batch(
        self, engine_shape: tuple[int, ...], shape: tuple[int, ...], export_batch_size: int
    ) -> None:
        """The advised settings, added to the original export, pass the TensorRT exporter's checks and fit the batch.

        ``dynamic_batch=True`` alone is refused by that exporter, which needs ``max_batch_size`` and ``batch_size <=
        max_batch_size``. *export_batch_size* is the ``batch_size`` the engine was exported with: the static engine's
        own batch, or the fakes' profile ``opt`` for the dynamic one.
        """
        engine = _FakeEngine({"input": ("input", engine_shape), "dets": ("output", (engine_shape[0], 5, 4))})
        runtime = _runtime_around(engine)
        with pytest.raises(ValueError) as refusal:
            runtime({"input": torch.zeros(shape)})
        advised = dict(re.findall(r"(dynamic_batch|max_batch_size)=(\w+)", str(refusal.value)))

        exporter = TensorRTExporter(
            TensorRTExporter.build_config(
                batch_size=export_batch_size,
                dynamic_batch=advised["dynamic_batch"] == "True",
                max_batch_size=int(advised["max_batch_size"]),
            )
        )

        assert exporter.config.max_batch_size >= shape[0]

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

        assert ("max_batch_size=5" in str(refusal.value)) is advised


#: Tensors of an engine with a dynamic batch axis on its input and output; each test sets the profile maximum.
_DYNAMIC_ENGINE_TENSORS = {"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}

#: What identifies the one graph a static engine captures: the shape of its only input.
_STATIC_SHAPES = ((1, 3, 8, 8),)


@pytest.mark.usefixtures("fake_tensorrt", "cuda_device_recorder", "fake_cuda_graphs")
class TestTRTInferenceCudaGraph:
    """``cuda_graph=True`` captures the engine's launch once per set of input shapes and replays it on every later call.

    Each call still goes through the plain path's validation and dynamic-shape declaration, and returns the same output
    buffers, so only the launch differs: the input is copied into a static buffer the graph was captured on.
    """

    def test_replay_reads_the_input_of_each_call(self, fake_cuda_graphs: _FakeCudaGraphs) -> None:
        """Two calls with different tensors each get their own values: the caller's tensor is copied, not captured."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        fake_cuda_graphs.on_replay = lambda graph: runtime.bindings["dets"].data.fill_(
            runtime._graphs[_STATIC_SHAPES].inputs["input"].sum().item()
        )
        first, second = torch.full((1, 3, 8, 8), 1.0), torch.full((1, 3, 8, 8), 2.0)

        seen_first = runtime({"input": first})["dets"][0, 0, 0].item()
        seen_second = runtime({"input": second})["dets"][0, 0, 0].item()

        assert (seen_first, seen_second) == (first.sum().item(), second.sum().item())

    def test_capture_points_the_input_at_the_static_buffer_once(self) -> None:
        """The captured launch reads the static buffer: the caller's pointers are never handed to the context."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        first, second = torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8)

        runtime({"input": first})
        runtime({"input": second})

        addresses = [
            call.args[1] for call in runtime.context.set_tensor_address.call_args_list if call.args[0] == "input"
        ]
        assert addresses == [runtime._graphs[_STATIC_SHAPES].inputs["input"].data_ptr()]
        assert first.data_ptr() not in addresses
        assert second.data_ptr() not in addresses

    def test_one_graph_per_batch_size_on_a_dynamic_engine(self, fake_cuda_graphs: _FakeCudaGraphs) -> None:
        """Batches 1 and 3 in mixed order capture two graphs, each launched once outside and once inside its capture."""
        runtime = _runtime_around(_FakeEngine(_DYNAMIC_ENGINE_TENSORS, profile_max=4), sync_mode=False, cuda_graph=True)
        inside_capture: list[bool] = []
        runtime.context.execute_async_v3.side_effect = lambda *args, **kwargs: (
            inside_capture.append(any(graph.capturing for graph in fake_cuda_graphs.graphs)) or True
        )

        for batch in (1, 3, 1, 3):
            runtime({"input": torch.zeros(batch, 3, 8, 8)})

        assert sorted(runtime._graphs) == [((1, 3, 8, 8),), ((3, 3, 8, 8),)]
        assert len(fake_cuda_graphs.graphs) == 2
        assert (inside_capture.count(False), inside_capture.count(True)) == (2, 2)
        assert [graph.replays for graph in fake_cuda_graphs.graphs] == [2, 2]

    @pytest.mark.parametrize("batch", [1, 3])
    def test_dynamic_outputs_are_trimmed_to_the_batch_of_the_call(self, batch: int) -> None:
        """A dynamic engine's output buffers sit at the profile max; the call returns only the batch it ran."""
        runtime = _runtime_around(_FakeEngine(_DYNAMIC_ENGINE_TENSORS, profile_max=4), sync_mode=False, cuda_graph=True)

        outputs = runtime({"input": torch.zeros(batch, 3, 8, 8)})

        assert outputs["dets"].shape == (batch, 5, 4)

    @pytest.mark.parametrize(
        "make_input",
        [
            pytest.param(lambda: torch.rand(1, 3, 4, 4), id="wrong_shape"),
            pytest.param(lambda: torch.rand(1, 3, 8, 8, dtype=torch.float64), id="wrong_dtype"),
            pytest.param(lambda: torch.rand(1, 3, 8, 16)[..., ::2], id="strided"),
        ],
    )
    def test_a_refused_input_captures_nothing(
        self, fake_cuda_graphs: _FakeCudaGraphs, make_input: Callable[[], torch.Tensor]
    ) -> None:
        """The plain path's input checks still run first, so a bad tensor never reaches the copy or the capture."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)

        with pytest.raises(ValueError):
            runtime({"input": make_input()})

        assert (fake_cuda_graphs.graphs, runtime._graphs) == ([], {})

    def test_a_failed_capture_raises_and_the_next_call_captures_again(self, fake_cuda_graphs: _FakeCudaGraphs) -> None:
        """A user who asked for a graph is told when capture fails instead of silently getting the plain launch.

        Nothing is cached for the failed shape, so the next call at it captures again.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        example = {"input": torch.zeros(1, 3, 8, 8)}
        fake_cuda_graphs.fail_next_capture = True

        with pytest.raises(RuntimeError, match="CUDA graph.*sync_mode=True"):
            runtime(example)
        assert runtime._graphs == {}

        runtime(example)
        assert sorted(runtime._graphs) == [_STATIC_SHAPES]

    def test_a_refused_launch_raises(self) -> None:
        """``execute_async_v3`` reports failure by returning ``False``; that must not be mistaken for a capture."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        runtime.context.execute_async_v3.return_value = False

        with pytest.raises(RuntimeError, match="launch failure"):
            runtime({"input": torch.zeros(1, 3, 8, 8)})

    def test_run_graph_needs_a_runtime_built_with_cuda_graph(self) -> None:
        """Calling ``run_graph`` directly on a plain runtime says what is missing instead of failing on a ``None``
        stream."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS))

        with pytest.raises(RuntimeError, match="cuda_graph=True"):
            runtime.run_graph({"input": torch.zeros(1, 3, 8, 8)})

    def test_a_refused_input_address_raises_before_any_launch(self) -> None:
        """``set_tensor_address`` reports a refusal by returning ``False``; the capture must not go on without it."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        runtime.context.set_tensor_address.return_value = False

        with pytest.raises(RuntimeError, match="refused the tensor address"):
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        runtime.context.execute_async_v3.assert_not_called()

    def test_a_warm_up_under_inference_mode_leaves_buffers_later_calls_can_write(self) -> None:
        """Warming up under ``torch.inference_mode()`` must not make the static buffers inference tensors."""
        runtime = _runtime_around(_FakeEngine(_DYNAMIC_ENGINE_TENSORS, profile_max=4), sync_mode=False, cuda_graph=True)
        with torch.inference_mode():
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        runtime({"input": torch.ones(3, 3, 8, 8)})
        runtime({"input": torch.ones(1, 3, 8, 8)})

        assert runtime._graphs[((1, 3, 8, 8),)].inputs["input"].sum().item() == 3 * 8 * 8

    @pytest.mark.parametrize(("input_dtype", "torch_dtype"), [(np.float32, torch.float32), (np.float16, torch.float16)])
    def test_the_static_buffer_has_the_dtype_the_engine_reads(
        self, input_dtype: type, torch_dtype: torch.dtype
    ) -> None:
        """TensorRT reads the buffer as its own input dtype, so a buffer of another one would be read as garbage."""
        runtime = _runtime_around(
            _FakeEngine(_STATIC_ENGINE_TENSORS, input_dtype=input_dtype), sync_mode=False, cuda_graph=True
        )

        runtime({"input": torch.zeros(1, 3, 8, 8, dtype=torch_dtype)})

        assert runtime._static_inputs["input"].dtype == torch_dtype

    def test_the_warm_up_launch_reads_the_callers_input(self) -> None:
        """The launch that precedes a capture runs on the call's own input, not on whatever the buffer was allocated
        with."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        seen: list[float] = []
        runtime.context.execute_async_v3.side_effect = lambda *args, **kwargs: (
            seen.append(runtime._static_inputs["input"].sum().item()) or True
        )

        runtime({"input": torch.ones(1, 3, 8, 8)})

        assert seen == [3 * 8 * 8, 3 * 8 * 8]

    def test_a_failed_capture_keeps_the_cause(self, fake_cuda_graphs: _FakeCudaGraphs) -> None:
        """The error a user sees is ours, and the reason torch gave is chained to it."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        fake_cuda_graphs.fail_next_capture = True

        with pytest.raises(RuntimeError, match="CUDA graph") as refusal:
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert str(refusal.value.__cause__) == "capture refused"

    def test_an_input_that_requires_grad_leaves_no_autograd_history_on_the_buffer(self) -> None:
        """The buffers outlive every call, so they must not collect the graph of an input produced by a differentiable
        op."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)

        for _ in range(2):
            runtime({"input": torch.zeros(1, 3, 8, 8, requires_grad=True) * 1})

        assert runtime._static_inputs["input"].requires_grad is False

    def test_an_input_that_requires_grad_is_accepted_after_a_no_grad_warm_up(self) -> None:
        """A view made under ``torch.no_grad()`` by the warm-up call can still be written by a call that needs grad."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        with torch.no_grad():
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        runtime({"input": torch.zeros(1, 3, 8, 8, requires_grad=True) * 1})

        assert runtime._static_inputs["input"].requires_grad is False

    def test_every_graph_reads_one_shared_static_buffer(self) -> None:
        """Graphs at different batch sizes are views of one buffer as large as the profile maximum, not a copy each."""
        runtime = _runtime_around(_FakeEngine(_DYNAMIC_ENGINE_TENSORS, profile_max=4), sync_mode=False, cuda_graph=True)

        for batch in (1, 3, 2):
            runtime({"input": torch.zeros(batch, 3, 8, 8)})

        buffer = runtime._static_inputs["input"]
        views = {captured.inputs["input"].data_ptr() for captured in runtime._graphs.values()}
        assert (views, buffer.numel()) == ({buffer.data_ptr()}, 4 * 3 * 8 * 8)

    def test_every_input_of_a_multi_input_engine_is_copied_to_its_own_buffer(
        self, fake_cuda_graphs: _FakeCudaGraphs
    ) -> None:
        """Both inputs reach the graph with the values of the call, each through a buffer of its own."""
        tensors = {"input": ("input", (1, 3, 8, 8)), "aux": ("input", (1, 2)), "dets": ("output", (1, 5, 4))}
        runtime = _runtime_around(_FakeEngine(tensors), sync_mode=False, cuda_graph=True)
        shapes = ((1, 3, 8, 8), (1, 2))
        fake_cuda_graphs.on_replay = lambda graph: runtime.bindings["dets"].data.fill_(
            runtime._graphs[shapes].inputs["input"].sum().item()
            + 100 * runtime._graphs[shapes].inputs["aux"].sum().item()
        )

        first = runtime({"input": torch.full((1, 3, 8, 8), 1.0), "aux": torch.full((1, 2), 1.0)})["dets"][
            0, 0, 0
        ].item()
        second = runtime({"input": torch.full((1, 3, 8, 8), 2.0), "aux": torch.full((1, 2), 3.0)})["dets"][
            0, 0, 0
        ].item()

        assert (first, second) == (192 + 200, 384 + 600)

    def test_a_replay_waits_for_the_caller_then_copies_and_replays_on_the_graph_stream_then_synchronizes(
        self, fake_cuda_graphs: _FakeCudaGraphs, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The order of a replayed call: wait for the caller's stream, then copy and replay inside the graph stream."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        runtime._graph_stream = _FakeStream(handle=11, events=fake_cuda_graphs.events)
        runtime({"input": torch.zeros(1, 3, 8, 8)})
        fake_cuda_graphs.events.clear()
        fake_cuda_graphs.record_copies(monkeypatch)

        runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert fake_cuda_graphs.events == ["wait", "enter", "copy", "replay", "exit", "sync"]

    def test_the_warm_up_launch_and_the_capture_use_the_graph_stream(self, fake_cuda_graphs: _FakeCudaGraphs) -> None:
        """TensorRT is launched on the graph's stream both times, and the capture is told which stream that is."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)

        runtime({"input": torch.zeros(1, 3, 8, 8)})

        handles = [call.kwargs["stream_handle"] for call in runtime.context.execute_async_v3.call_args_list]
        assert (handles, fake_cuda_graphs.capture_streams) == ([11, 11], [runtime._graph_stream])

    def test_a_failed_capture_gives_the_caller_its_stream_back(self, fake_cuda_graphs: _FakeCudaGraphs) -> None:
        """``torch.cuda.graph`` leaves the capture stream current when a capture fails; the caller must not keep it."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        caller_stream = fake_cuda_graphs.current
        fake_cuda_graphs.fail_next_capture = True

        with pytest.raises(RuntimeError, match="CUDA graph"):
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert fake_cuda_graphs.current is caller_stream

    def test_every_launch_and_replay_runs_on_the_engine_device(
        self, cuda_device_recorder: _DeviceRecorder, fake_cuda_graphs: _FakeCudaGraphs
    ) -> None:
        """The warm-up launch, the capture and the replay all run with the engine's device current, and ask it for a
        stream.

        The caller asked for a bare ``"cuda"``, which construction pinned (to the fakes' CPU here), as in the plain
        paths' device test above.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)
        runtime.device = "cuda"
        engine_device = runtime._engine_device
        active: list[torch.device | None] = []
        runtime.context.execute_async_v3.side_effect = lambda *args, **kwargs: (
            active.append(cuda_device_recorder.current) or True
        )
        fake_cuda_graphs.on_replay = lambda graph: active.append(cuda_device_recorder.current)

        runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert (active, set(fake_cuda_graphs.current_stream_args)) == ([engine_device] * 3, {(engine_device,)})

    def test_the_graph_stream_waits_for_the_callers_stream_before_each_copy(
        self, fake_cuda_graphs: _FakeCudaGraphs
    ) -> None:
        """The input may still be in flight on the caller's stream, so the graph stream waits for it on every call."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)

        runtime({"input": torch.zeros(1, 3, 8, 8)})
        runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert runtime._graph_stream.waited_on == [fake_cuda_graphs.current, fake_cuda_graphs.current]

    def test_synchronize_waits_on_the_graph_stream(self) -> None:
        """``synchronize()`` drains the stream the graph runs on."""
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=False, cuda_graph=True)

        runtime.synchronize()

        assert runtime._graph_stream.synchronized == 1


@pytest.mark.usefixtures("cuda_device_recorder", "fake_cuda_graphs")
class TestTRTInferenceCudaGraphConstruction:
    """The ``cuda_graph`` option is opt-in, needs no pycuda, and is refused where it cannot take effect or where its
    graphs would have no bound."""

    @staticmethod
    def _engine_file(fake_tensorrt: _FakeTensorRTModule, tmp_path: Path, engine: _FakeEngine | None = None) -> str:
        """Point the fake runtime at *engine* and return a path TensorRT can be asked to load.

        Without *engine* it is one with a single fixed ``(1, 3, 8, 8)`` input. An engine given here should have no
        output when construction is meant to succeed: it allocates the output buffers on the CUDA device, which
        CPU-only CI does not have.

        Examples:
            >>> import tempfile
            >>> from unittest.mock import patch
            >>> fake = _FakeTensorRTModule()
            >>> with tempfile.TemporaryDirectory() as directory, patch.object(trt_inference, "trt", fake):
            ...     path = TestTRTInferenceCudaGraphConstruction._engine_file(fake, Path(directory))
            ...     Path(path).name, fake.runtime.engine.get_tensor_shape("input")
            ('model.trt', (1, 3, 8, 8))
            >>> with tempfile.TemporaryDirectory() as directory, patch.object(trt_inference, "trt", fake):
            ...     engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8))})
            ...     _ = TestTRTInferenceCudaGraphConstruction._engine_file(fake, Path(directory), engine)
            ...     fake.runtime.engine is engine
            True
        """
        fake_tensorrt.runtime.engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))}) if engine is None else engine
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        return str(engine_file)

    def test_cuda_graph_is_off_by_default(self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path) -> None:
        """A caller that does not ask for a graph gets no graph stream."""
        runtime = TRTInference(self._engine_file(fake_tensorrt, tmp_path), sync_mode=True)

        assert runtime._graph_stream is None

    def test_cuda_graph_with_sync_mode_is_refused_before_loading(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """``sync_mode`` runs ``execute_v2``, which cannot be captured, so asking for both is an error."""
        engine_file = self._engine_file(fake_tensorrt, tmp_path)

        with pytest.raises(ValueError, match="sync_mode"):
            TRTInference(engine_file, sync_mode=True, cuda_graph=True)

        fake_tensorrt.runtime.deserialize_cuda_engine.assert_not_called()

    @pytest.mark.parametrize(
        "tensors",
        [
            pytest.param({"input": ("input", (-1, 3, -1, -1))}, id="dynamic-image-size"),
            pytest.param({"input": ("input", (-1, 3, -1, 8))}, id="dynamic-height-only"),
            pytest.param({"input": ("input", (-1, 3, 8, 8)), "aux": ("input", (-1, -1))}, id="dynamic-second-input"),
        ],
    )
    def test_an_engine_whose_profile_varies_more_than_the_batch_is_refused(
        self,
        fake_tensorrt: _FakeTensorRTModule,
        tmp_path: Path,
        tensors: dict[str, tuple[str, tuple[int, ...]]],
    ) -> None:
        """A graph is kept per input shape, so an engine whose profile takes any image size would keep one for every
        size it is ever called with; only a profile that varies the batch alone bounds the graphs."""
        engine_file = self._engine_file(fake_tensorrt, tmp_path, _FakeEngine(tensors))

        with pytest.raises(ValueError, match="varies only the batch"):
            TRTInference(engine_file, cuda_graph=True)

    def test_the_refusal_comes_before_an_execution_context_is_created(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """The decision needs only the engine, so nothing the refused runtime would hold is allocated first."""
        engine = _FakeEngine({"input": ("input", (-1, 3, -1, -1))})
        engine.create_execution_context = Mock(side_effect=engine.create_execution_context)
        engine_file = self._engine_file(fake_tensorrt, tmp_path, engine)

        with pytest.raises(ValueError, match="varies only the batch"):
            TRTInference(engine_file, cuda_graph=True)

        engine.create_execution_context.assert_not_called()

    def test_the_refusal_advises_sync_mode(self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path) -> None:
        """``sync_mode=True`` is the mode that needs neither a graph nor pycuda, like the capture error's advice."""
        engine_file = self._engine_file(fake_tensorrt, tmp_path, _FakeEngine({"input": ("input", (-1, 3, -1, -1))}))

        with pytest.raises(ValueError, match="sync_mode=True instead of cuda_graph=True"):
            TRTInference(engine_file, cuda_graph=True)

    def test_a_fixed_batch_engine_with_a_dynamic_image_size_gets_the_error_of_every_mode(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """No mode can size such an input, so the refusal must not advise ``sync_mode=True`` for it."""
        engine_file = self._engine_file(fake_tensorrt, tmp_path, _FakeEngine({"input": ("input", (1, 3, -1, -1))}))

        with pytest.raises(ValueError, match="unresolved dimension"):
            TRTInference(engine_file, cuda_graph=True)

    def test_the_refused_engine_loads_with_the_advised_sync_mode(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """The refusal's advice works when followed verbatim: the same engine loads with ``sync_mode=True``."""
        engine_file = self._engine_file(fake_tensorrt, tmp_path, _FakeEngine({"input": ("input", (-1, 3, -1, -1))}))

        runtime = TRTInference(engine_file, sync_mode=True)

        assert runtime.bindings["input"].shape == (4, 3, 16, 16)

    @pytest.mark.parametrize(
        "engine",
        [
            pytest.param(_FakeEngine({"input": ("input", (-1, 3, 8, 8))}), id="dynamic-batch"),
            pytest.param(
                _FakeEngine({"input": ("input", (-1, 3, -1, -1))}, image_profile=(8, 8), second_image_profile=(4, 16)),
                id="image-size-pinned-in-profile-0",
            ),
        ],
    )
    def test_an_engine_whose_profile_varies_only_the_batch_is_accepted(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path, engine: _FakeEngine
    ) -> None:
        """What ``RFDETR.export(format="tensorrt", dynamic_batch=True, max_batch_size=...)`` builds keeps graph replay,
        and so does an engine that reports a dynamic image size because another profile varies it, while profile 0, the
        one the runtime runs, pins it."""
        engine_file = self._engine_file(fake_tensorrt, tmp_path, engine)

        runtime = TRTInference(engine_file, cuda_graph=True)

        assert runtime._graph_stream is not None

    def test_cuda_graph_does_not_need_pycuda(
        self, fake_tensorrt: _FakeTensorRTModule, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The graph runs on a torch stream, so the ``tensorrt-bench`` extra is not required for it."""
        monkeypatch.setattr(trt_inference, "cuda", None)

        runtime = TRTInference(self._engine_file(fake_tensorrt, tmp_path), cuda_graph=True)

        assert (runtime._graph_stream is not None, runtime.stream) == (True, None)

    def test_the_graph_stream_is_created_on_the_engine_device(
        self, fake_tensorrt: _FakeTensorRTModule, fake_cuda_graphs: _FakeCudaGraphs, tmp_path: Path
    ) -> None:
        """A stream made without a device would belong to whichever device is current, not to the engine's."""
        TRTInference(self._engine_file(fake_tensorrt, tmp_path), device="cuda:1", cuda_graph=True)

        assert fake_cuda_graphs.stream_kwargs == [{"device": torch.device("cuda", 1)}]

    def test_async_mode_without_a_graph_still_needs_pycuda(
        self, fake_tensorrt: _FakeTensorRTModule, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only the graph option lifts the pycuda requirement; the default async path keeps it."""
        monkeypatch.setattr(trt_inference, "cuda", None)

        with pytest.raises(ImportError, match="pycuda"):
            TRTInference(self._engine_file(fake_tensorrt, tmp_path))


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

    @pytest.mark.parametrize(
        ("keywords", "allowed"),
        [
            pytest.param({}, False, id="not-given"),
            pytest.param({"engine_host_code_allowed": False}, False, id="false"),
            pytest.param({"engine_host_code_allowed": True}, True, id="true"),
        ],
    )
    def test_host_code_is_allowed_only_when_asked_for(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path, keywords: dict[str, bool], allowed: bool
    ) -> None:
        """An engine that carries host code (a version-compatible one) loads only on request, and the runtime is told
        before it deserializes, not after.

        TensorRT refuses such an engine by default because running host code from an engine file is only safe for a file
        the caller trusts, so ``TRTInference`` never turns the switch on by itself.
        """
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))})
        fake_tensorrt.runtime.engine = engine
        seen: dict[str, bool] = {}

        def deserialize(payload: bytes) -> _FakeEngine:
            seen["allowed_while_deserializing"] = fake_tensorrt.runtime.engine_host_code_allowed
            return engine

        fake_tensorrt.runtime.deserialize_cuda_engine.side_effect = deserialize

        TRTInference(str(engine_file), device="cuda:0", sync_mode=True, **keywords)

        assert seen == {"allowed_while_deserializing": allowed}

    @pytest.mark.parametrize("value", ["false", 1, None])
    def test_only_a_bool_can_allow_host_code(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path, value: object
    ) -> None:
        """A truthy string such as ``"false"`` from a config file must not switch a security setting on."""
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")

        with pytest.raises(ValueError, match="engine_host_code_allowed"):
            TRTInference(str(engine_file), device="cuda:0", sync_mode=True, engine_host_code_allowed=value)

    def test_a_failure_with_host_code_allowed_does_not_suggest_allowing_it(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """The caller already opted in, so the message points at the other reasons an engine does not load."""
        engine_file = tmp_path / "foreign.trt"
        engine_file.write_bytes(b"not an engine for this machine")
        fake_tensorrt.runtime.engine = None

        with pytest.raises(RuntimeError) as refusal:
            TRTInference(str(engine_file), device="cuda:0", sync_mode=True, engine_host_code_allowed=True)

        message = str(refusal.value)
        assert "could not deserialize" in message
        assert "engine_host_code_allowed" not in message

    def test_a_tensorrt_without_the_host_code_switch_still_loads_the_engine(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """A release that predates the switch has no gate to open, so the opt-in must not become an AttributeError."""
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        runtime = _NoSwitchRuntime()
        runtime.engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))})
        fake_tensorrt.Runtime.return_value = runtime

        loaded = TRTInference(str(engine_file), device="cuda:0", sync_mode=True, engine_host_code_allowed=True)

        assert loaded.engine is runtime.engine

    def test_a_tensorrt_without_the_host_code_switch_says_the_opt_in_had_no_effect(
        self, monkeypatch: pytest.MonkeyPatch, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path
    ) -> None:
        """If the property is ever renamed, the caller who passed ``True`` must be told it did nothing."""
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        runtime = _NoSwitchRuntime()
        runtime.engine = _FakeEngine({"input": ("input", (1, 3, 8, 8))})
        fake_tensorrt.Runtime.return_value = runtime
        logged: list[str] = []
        monkeypatch.setattr(trt_inference.logger, "warning", lambda message, *args: logged.append(message % args))

        TRTInference(str(engine_file), device="cuda:0", sync_mode=True, engine_host_code_allowed=True)

        assert len(logged) == 1
        assert "engine_host_code_allowed" in logged[0]

    @pytest.mark.parametrize(
        "phrase", ["trt_hardware_compatibility", "trt_version_compatible", "engine_host_code_allowed=True"]
    )
    def test_the_rebuild_hint_names_the_portable_builds(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path, phrase: str
    ) -> None:
        """An engine that will not load may be a portable one: the message names both keywords and the opt-in."""
        engine_file = tmp_path / "foreign.trt"
        engine_file.write_bytes(b"not an engine for this machine")
        fake_tensorrt.runtime.engine = None

        with pytest.raises(RuntimeError, match=re.escape(phrase)):
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

    @pytest.mark.parametrize("sync_mode", [True, False])
    def test_a_launch_failure_propagates_and_exits_the_device_scope(
        self, cuda_device_recorder: _DeviceRecorder, sync_mode: bool
    ) -> None:
        """An exception from the real TensorRT launch call is neither swallowed nor left holding the device scope.

        Neither ``run_sync`` nor ``run_async`` wraps ``execute_v2``/``execute_async_v3`` in a ``try``/``except``, so an
        exception there should propagate through Python's own ``with`` statement guarantee -- this proves that stays
        true, and that ``torch.cuda.device``'s ``__exit__`` still runs on the way out.
        """
        runtime = _runtime_around(_FakeEngine(_STATIC_ENGINE_TENSORS), sync_mode=sync_mode)
        runtime.context.execute_v2.side_effect = RuntimeError("launch failed")
        runtime.context.execute_async_v3.side_effect = RuntimeError("launch failed")

        with pytest.raises(RuntimeError, match="launch failed"):
            runtime({"input": torch.zeros(1, 3, 8, 8)})

        assert cuda_device_recorder.current is None

    @pytest.mark.parametrize("device", ["cpu", "mps", "xpu", pytest.param("cuda:x", id="malformed-index")])
    def test_a_non_cuda_device_is_refused(
        self, fake_tensorrt: _FakeTensorRTModule, tmp_path: Path, device: str
    ) -> None:
        """TensorRT cannot run on anything but CUDA; ``device="cpu"`` used to load the engine with buffers in host
        memory.

        A malformed index (``"cuda:x"``) fails inside ``torch.device()`` itself, before rfdetr's own type check --
        parametrized alongside the other non-CUDA types to prove both paths raise the same named ``ValueError`` rather
        than a bare ``RuntimeError`` leaking out of ``torch.device()``.
        """
        engine_file = tmp_path / "model.trt"
        engine_file.write_bytes(b"engine")
        fake_tensorrt.runtime.engine = _FakeEngine(_STATIC_ENGINE_TENSORS)

        with pytest.raises(ValueError, match="CUDA|valid device"):
            TRTInference(str(engine_file), device=device, sync_mode=True)

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

        runtime_class.assert_called_once_with(
            "model.trt", sync_mode=True, device=f"cuda:{device}", engine_host_code_allowed=False
        )
        assert infer_engine.call_args.kwargs["device"] == f"cuda:{device}"
        assert infer_engine.call_args.args[2].device == torch.device(f"cuda:{device}")

    @pytest.mark.parametrize("engine_path", ["model.trt", "model.engine"])
    def test_trt_benchmark_forwards_the_host_code_opt_in(
        self, monkeypatch: pytest.MonkeyPatch, engine_path: str
    ) -> None:
        """A version-compatible engine can be benchmarked once the caller opts in, as with ``TRTInference`` itself.

        Both engine suffixes the benchmark routes to TensorRT carry the opt-in.
        """
        monkeypatch.setattr(benchmark, "get_image_list", Mock(return_value=[]))
        runtime_class = Mock()
        monkeypatch.setattr(benchmark, "TRTInference", runtime_class)
        monkeypatch.setattr(benchmark, "infer_engine", Mock())

        benchmark.main(engine_path, disable_eval=True, engine_host_code_allowed=True)

        assert runtime_class.call_args.kwargs["engine_host_code_allowed"] is True

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
