# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
from PIL import Image

import rfdetr.export.benchmark as benchmark
from rfdetr.export._tensorrt.inference import TRTInference
from rfdetr.export.benchmark import infer_transforms

#: Minimal indexed COCO dataset used to verify evaluator construction.
_MINIMAL_COCO = {
    "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0}],
    "categories": [{"id": 1, "name": "widget"}],
}


class TestTRTInference:
    def test_synchronize_sync_mode_does_not_require_stream(self, monkeypatch) -> None:
        """`synchronize()` should not access stream in sync mode."""
        inference = TRTInference.__new__(TRTInference)
        inference.sync_mode = True

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

    def test_infer_transforms_accepts_none_target(self) -> None:
        """Benchmark inference preprocessing should support image-only input."""
        image = Image.new("RGB", (320, 240))

        image_tensor, target = infer_transforms()(image, None)

        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 640, 640)
        assert image_tensor.dtype == torch.float32
        assert target is None


class _FakeTensorRTModule(ModuleType):
    """Stand-in ``tensorrt`` module with the two symbols ``get_bindings`` reads: ``TensorIOMode`` and ``nptype``."""

    def __init__(self) -> None:
        super().__init__("tensorrt")
        self.TensorIOMode = SimpleNamespace(INPUT="input", OUTPUT="output")
        self.nptype = lambda dtype: dtype


class _FakeEngine:
    """Deserialized-engine stand-in: iterates tensor names and answers the shape/dtype/mode/profile queries.

    Shapes use ``-1`` for a dynamic batch axis, as TensorRT reports them; ``profile_max`` is the batch upper bound the
    single optimization profile declares on every dynamic input.
    """

    def __init__(self, tensors: dict[str, tuple[str, tuple[int, ...]]], profile_max: int = 4) -> None:
        self._tensors = tensors
        self.profile_max = profile_max

    def __iter__(self):
        return iter(self._tensors)

    def get_tensor_mode(self, name: str) -> str:
        return self._tensors[name][0]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._tensors[name][1]

    def get_tensor_dtype(self, name: str):
        return np.float32

    def get_tensor_profile_shape(self, name: str, profile_index: int):
        shape = self._tensors[name][1]
        return ((1, *shape[1:]), (2, *shape[1:]), (self.profile_max, *shape[1:]))


class _FakeContext:
    """Execution-context stand-in that resolves every dynamic shape from the input shapes it has been given.

    TensorRT sizes a dynamic engine's tensors on the execution context, not on the engine, so ``set_input_shape``
    records the batch a call declares and ``get_tensor_shape`` then reports every dynamic tensor at it. A batch above
    the engine's profile maximum is refused by returning ``False``, which is how TensorRT reports it instead of raising.
    ``output_batch`` pins the outputs to a batch of their own, modelling an engine whose output batch is not its input
    batch. The execution calls are plain ``Mock`` s so tests can assert on them.
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
        if shape[0] > self._engine.profile_max:
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


def _runtime_around(
    engine: _FakeEngine, context: _FakeContext | None = None, *, sync_mode: bool = True
) -> TRTInference:
    """Assemble a ``TRTInference`` around a fake engine and context without touching ``__init__`` (needs a GPU).

    A context matching *engine* is built here unless the test needs a non-default one (see :class:`_FakeContext`).
    The module-level ``trt`` handle must already point at ``_FakeTensorRTModule`` (see the autouse fixture in
    ``TestTRTInferenceDynamicBatch``); the doctest patches it itself.

    Examples:
        >>> from unittest.mock import patch
        >>> from rfdetr.export._tensorrt import inference as trt_inference
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
    runtime.stream = None if sync_mode else Mock(handle=7)
    runtime.bindings = runtime.get_bindings(engine, runtime.context, device="cpu")
    runtime.bindings_addr = OrderedDict((n, v.ptr) for n, v in runtime.bindings.items())
    runtime.input_names = runtime.get_input_names()
    runtime.output_names = runtime.get_output_names()
    runtime._prime_context()
    return runtime


class TestTRTInferenceDynamicBatch:
    """``TRTInference`` serves engines built with ``dynamic_batch=True`` (a ``-1`` batch axis on every tensor)."""

    @pytest.fixture(autouse=True)
    def _fake_tensorrt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point the module-level ``trt`` handle at the stand-in so no real TensorRT is needed."""
        from rfdetr.export._tensorrt import inference as trt_inference

        monkeypatch.setattr(trt_inference, "trt", _FakeTensorRTModule())

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

    def test_get_dummy_input_uses_the_runtime_device(self) -> None:
        """Dummy input tensors land on ``self.device``, not a hardcoded ``cuda:0``.

        Regression guard: ``get_dummy_input`` used to build every tensor with ``.to("cuda:0")``
        regardless of the device the runtime was constructed with.
        """
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (2, 5, 4))})
        runtime = _runtime_around(engine, context=Mock())
        runtime.device = "cpu"

        blob = runtime.get_dummy_input(batch_size=3)

        assert set(blob) == {"input"}
        assert blob["input"].shape == (3, 3, 8, 8)
        assert blob["input"].device.type == "cpu"

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
        """The dummy input must land on the runtime's own configured device, not a hardcoded ``"cuda:0"``.

        Regression guard: ``get_dummy_input`` used to hardcode ``.to("cuda:0")``, so a runtime built for the CPU (as
        every fake-engine test in this module is) would still hand back a CUDA tensor.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        runtime = _runtime_around(engine)
        runtime.device = "cpu"

        blob = runtime.get_dummy_input(batch_size=2)

        assert blob["input"].shape == (2, 3, 8, 8)
        assert torch.device(blob["input"].device) == torch.device("cpu")


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
