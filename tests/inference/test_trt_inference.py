# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import sys
from collections import OrderedDict
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

import rfdetr.export.benchmark as benchmark
from rfdetr.export._tensorrt.inference import TRTInference
from rfdetr.export.benchmark import infer_transforms


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
        self._profile_max = profile_max

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
        return ((1, *shape[1:]), (2, *shape[1:]), (self._profile_max, *shape[1:]))


def _runtime_around(engine: _FakeEngine, context: Mock, *, sync_mode: bool = True) -> TRTInference:
    """Assemble a ``TRTInference`` around a fake engine and context without touching ``__init__`` (needs a GPU).

    The module-level ``trt`` handle must already point at ``_FakeTensorRTModule`` (see the autouse fixture in
    ``TestTRTInferenceDynamicBatch``); the doctest patches it itself.

    Examples:
        >>> from unittest.mock import patch
        >>> from rfdetr.export._tensorrt import inference as trt_inference
        >>> engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        >>> with patch.object(trt_inference, "trt", _FakeTensorRTModule()):
        ...     runtime = _runtime_around(engine, Mock())
        >>> runtime.input_names, runtime.output_names, runtime.bindings["input"].shape
        (['input'], ['dets'], (4, 3, 8, 8))
    """
    runtime = TRTInference.__new__(TRTInference)
    runtime.engine = engine
    runtime.context = context
    runtime.sync_mode = sync_mode
    runtime.stream = None if sync_mode else Mock(handle=7)
    runtime.bindings = runtime.get_bindings(engine, context, device="cpu")
    runtime.bindings_addr = OrderedDict((n, v.ptr) for n, v in runtime.bindings.items())
    runtime.input_names = runtime.get_input_names()
    runtime.output_names = runtime.get_output_names()
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

        runtime = _runtime_around(engine, context=Mock())

        assert runtime.bindings["input"].shape == (4, 3, 8, 8)
        assert runtime.bindings["dets"].shape == (4, 5, 4)
        assert runtime.bindings["dets"].dynamic is True
        assert tuple(runtime.bindings["labels"].data.shape) == (4, 5, 3)

    def test_static_tensors_keep_their_shape(self) -> None:
        """A fixed-batch engine is allocated exactly as declared and marked static."""
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (2, 5, 4))})

        runtime = _runtime_around(engine, context=Mock())

        assert runtime.bindings["input"].shape == (2, 3, 8, 8)
        assert runtime.bindings["input"].dynamic is False

    def test_run_sync_declares_the_input_shape_and_trims_outputs(self) -> None:
        """Each call sets the real input shape on the context and returns only the rows the engine produced."""
        engine = _FakeEngine(
            {"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))},
            profile_max=4,
        )
        context = Mock()
        context.get_tensor_shape.return_value = (3, 5, 4)
        runtime = _runtime_around(engine, context)
        blob = {"input": torch.zeros(3, 3, 8, 8)}

        outputs = runtime(blob)

        context.set_input_shape.assert_called_once_with("input", (3, 3, 8, 8))
        context.execute_v2.assert_called_once()
        assert tuple(outputs["dets"].shape) == (3, 5, 4)
        assert runtime.bindings_addr["input"] == blob["input"].data_ptr()

    def test_run_sync_on_a_static_engine_returns_the_whole_buffer(self) -> None:
        """A fixed-batch engine neither declares shapes nor trims, so the old behaviour is unchanged."""
        engine = _FakeEngine({"input": ("input", (2, 3, 8, 8)), "dets": ("output", (2, 5, 4))})
        context = Mock()
        runtime = _runtime_around(engine, context)

        outputs = runtime({"input": torch.zeros(2, 3, 8, 8)})

        context.set_input_shape.assert_not_called()
        context.get_tensor_shape.assert_not_called()
        assert tuple(outputs["dets"].shape) == (2, 5, 4)

    def test_batch_beyond_the_profile_is_refused_before_execution(self) -> None:
        """TensorRT reports an out-of-profile shape by returning ``False`` from ``set_input_shape``, not by raising.

        Ignoring that result would execute anyway and hand back the previous call's output buffer contents, so the
        helper must stop before touching the context's execution path.
        """
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        context = Mock()
        context.set_input_shape.return_value = False
        runtime = _runtime_around(engine, context)

        with pytest.raises(ValueError, match="outside the engine's optimization profile"):
            runtime({"input": torch.zeros(5, 3, 8, 8)})

        context.execute_v2.assert_not_called()
        context.execute_async_v3.assert_not_called()

    def test_run_async_registers_every_tensor_address_and_trims_outputs(self) -> None:
        """The async path binds each tensor by name, launches ``execute_async_v3`` on the stream, then syncs it."""
        engine = _FakeEngine({"input": ("input", (-1, 3, 8, 8)), "dets": ("output", (-1, 5, 4))}, profile_max=4)
        context = Mock()
        context.get_tensor_shape.return_value = (3, 5, 4)
        runtime = _runtime_around(engine, context, sync_mode=False)
        blob = {"input": torch.zeros(3, 3, 8, 8)}

        outputs = runtime(blob)

        context.set_input_shape.assert_called_once_with("input", (3, 3, 8, 8))
        addresses = {name: address for (name, address), _ in context.set_tensor_address.call_args_list}
        assert addresses == {"input": blob["input"].data_ptr(), "dets": runtime.bindings["dets"].ptr}
        context.execute_async_v3.assert_called_once_with(stream_handle=7)
        context.execute_v2.assert_not_called()
        runtime.stream.synchronize.assert_called_once()
        assert tuple(outputs["dets"].shape) == (3, 5, 4)


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
