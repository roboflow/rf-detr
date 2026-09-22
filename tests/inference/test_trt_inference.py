# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

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
