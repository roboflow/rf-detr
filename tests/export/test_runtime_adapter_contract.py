# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Boundary tests for exported-model runtime adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rfdetr.export._runtime.adapters import load_runtime
from rfdetr.export._runtime.metadata import ExportMetadata


def _metadata(**fields: Any) -> ExportMetadata:
    """Make a complete prediction contract for runtime tests.

    Examples:
        >>> contract = _metadata(format="onnx", task="detect", input_shape=(1, 3, 8, 8),
        ...     outputs={"pred_boxes": "dets", "pred_logits": "labels"})
        >>> contract.num_classes
        3
    """
    return ExportMetadata(
        class_names=["one", "two", "three"],
        class_id_to_name={0: "one", 1: "two", 2: "three"},
        num_classes=3,
        num_select=2,
        patch_size=16,
        num_windows=1,
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        trace_alpha=0.2,
        **fields,
    )


class TestONNXRuntimeAdapter:
    """Exercise provider choice and semantic output mapping through the loader."""

    def test_cpu_run_returns_owned_semantic_tensors(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An ONNX call maps graph names and owns returned arrays."""
        path = tmp_path / "model.onnx"
        path.write_bytes(b"model")
        output = np.ones((1, 2, 4), dtype=np.float32)
        logits = np.ones((1, 2, 3), dtype=np.float32)
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 8, 8])]
        session.get_outputs.return_value = [SimpleNamespace(name="dets"), SimpleNamespace(name="labels")]
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.run.return_value = [output, logits]
        monkeypatch.setitem(
            sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        )
        monkeypatch.setattr("rfdetr.export._onnx.inference._create_onnx_session", lambda *args, **kwargs: session)
        metadata = _metadata(
            format="onnx",
            task="detect",
            input_shape=(1, 3, 8, 8),
            input_name="input",
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        )

        runtime = load_runtime(path, metadata, device="cpu")
        result = runtime.run(torch.zeros(1, 3, 8, 8))
        output.fill(9)

        assert runtime.info == {"backend": "onnx", "device": "CPUExecutionProvider"}
        assert runtime.device == torch.device("cpu")
        assert result["pred_boxes"][0, 0, 0] == 1
        session.run.assert_called_once()

    def test_explicit_cuda_refuses_missing_provider(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An explicit CUDA request cannot fall back to CPU."""
        path = tmp_path / "model.onnx"
        path.write_bytes(b"model")
        monkeypatch.setitem(
            sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        )
        metadata = _metadata(
            format="onnx",
            task="detect",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        )

        with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
            load_runtime(path, metadata, device="cuda:0")


class TestTFLiteRuntimeAdapter:
    """Exercise layout conversion, dynamic batches, and positional outputs."""

    def test_nhwc_run_maps_positional_outputs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A LiteRT artifact receives NHWC data and returns semantic outputs."""
        path = tmp_path / "model.tflite"
        path.write_bytes(b"model")
        interpreter = Mock()
        interpreter.get_input_details.return_value = [
            {"name": "input", "index": 7, "shape": [1, 8, 8, 3], "dtype": np.float32}
        ]
        interpreter.get_output_details.return_value = [
            {"name": "opaque_0", "index": 9},
            {"name": "opaque_1", "index": 10},
        ]
        interpreter.get_tensor.side_effect = [np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 3), np.float32)]
        monkeypatch.setattr("rfdetr.export._tflite.inference._create_interpreter", lambda path: interpreter)
        metadata = _metadata(
            format="litert",
            task="detect",
            input_shape=(1, 3, 8, 8),
            input_layout="NHWC",
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        runtime = load_runtime(path, metadata)
        result = runtime.run(torch.zeros(1, 3, 8, 8))

        assert interpreter.set_tensor.call_args.args[0] == 7
        assert interpreter.set_tensor.call_args.args[1].shape == (1, 8, 8, 3)
        assert result["pred_boxes"].shape == (1, 2, 4)
        interpreter.invoke.assert_called_once()

    def test_rejects_wrong_fixed_batch_before_invoke(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A fixed batch artifact rejects a two-image call before execution."""
        path = tmp_path / "model.tflite"
        path.write_bytes(b"model")
        interpreter = Mock()
        interpreter.get_input_details.return_value = [
            {"name": "input", "index": 0, "shape": [1, 8, 8, 3], "dtype": np.float32}
        ]
        interpreter.get_output_details.return_value = [
            {"name": "opaque_0", "index": 1},
            {"name": "opaque_1", "index": 2},
        ]
        monkeypatch.setattr("rfdetr.export._tflite.inference._create_interpreter", lambda path: interpreter)
        metadata = _metadata(
            format="tflite",
            task="detect",
            input_shape=(1, 3, 8, 8),
            input_layout="NHWC",
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        runtime = load_runtime(path, metadata)
        with pytest.raises(ValueError, match="Input axis 0 must be 1"):
            runtime.run(torch.zeros(2, 3, 8, 8))
        interpreter.invoke.assert_not_called()

    def test_dynamic_batch_resizes_one_interpreter(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A dynamic model resizes once and still executes one two-image batch."""
        path = tmp_path / "model.tflite"
        path.write_bytes(b"model")
        interpreter = Mock()
        interpreter.get_input_details.return_value = [
            {
                "name": "input",
                "index": 0,
                "shape": [1, 8, 8, 3],
                "shape_signature": [-1, 8, 8, 3],
                "dtype": np.float32,
            }
        ]
        interpreter.get_output_details.return_value = [{"name": "dets", "index": 1}, {"name": "labels", "index": 2}]
        interpreter.get_tensor.side_effect = [np.zeros((2, 2, 4), np.float32), np.zeros((2, 2, 3), np.float32)]
        monkeypatch.setattr("rfdetr.export._tflite.inference._create_interpreter", lambda path: interpreter)
        metadata = _metadata(
            format="tflite",
            task="detect",
            input_shape=(-1, 3, 8, 8),
            input_layout="NHWC",
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        runtime = load_runtime(path, metadata)
        result = runtime.run(torch.zeros(2, 3, 8, 8))

        interpreter.resize_tensor_input.assert_called_once_with(0, (2, 8, 8, 3), strict=True)
        interpreter.invoke.assert_called_once()
        assert result["pred_boxes"].shape[0] == 2


class TestTensorRTRuntimeAdapter:
    """Check the existing TensorRT wrapper's buffer ownership."""

    def test_engine_waits_for_prepared_cuda_input(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A stream-free engine call cannot read input before its torch stream finishes."""
        path = tmp_path / "model.engine"
        path.write_bytes(b"engine")
        events: list[str] = []
        session = Mock()
        session.input_names = ["input"]
        session.output_names = ["dets", "labels"]
        session.bindings = {"input": SimpleNamespace(dtype=np.float32, shape=(1, 3, 8, 8))}
        session.engine_device = torch.device("cuda:0")
        session.side_effect = lambda _: events.append("execute") or {}
        stream = SimpleNamespace(synchronize=lambda: events.append("input ready"))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
        monkeypatch.setattr(torch.Tensor, "to", lambda self, **kwargs: self)
        monkeypatch.setattr("rfdetr.export._tensorrt.inference.TRTInference", lambda *args, **kwargs: session)
        metadata = _metadata(
            format="tensorrt",
            task="detect",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        )

        runtime = load_runtime(path, metadata)
        runtime._execute(torch.zeros(1, 3, 8, 8))

        assert events == ["input ready", "execute"]

    def test_engine_run_copies_reused_outputs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A later engine call cannot change a prior prediction."""
        path = tmp_path / "model.engine"
        path.write_bytes(b"engine")
        boxes = torch.ones(1, 2, 4)
        logits = torch.ones(1, 2, 3)
        session = Mock()
        session.input_names = ["input"]
        session.output_names = ["dets", "labels"]
        session.bindings = {"input": SimpleNamespace(dtype=np.float32, shape=(1, 3, 8, 8))}
        session.engine_device = torch.device("cpu")
        session.side_effect = lambda _: {"dets": boxes, "labels": logits}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("rfdetr.export._tensorrt.inference.TRTInference", lambda *args, **kwargs: session)
        metadata = _metadata(
            format="tensorrt",
            task="detect",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        )

        runtime = load_runtime(path, metadata)
        batch = torch.zeros(1, 3, 8, 8)
        first = runtime.run(batch)
        boxes.fill_(9)

        assert first["pred_boxes"][0, 0, 0] == 1
        assert runtime.device == session.engine_device
        assert session.call_args.args[0]["input"] is batch
        session.assert_called_once()


class TestCoreAIRuntimeAdapter:
    """Check the documented Core AI load and execution APIs."""

    def test_float16_keypoints_auto_uses_cpu(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Automatic execution avoids the unsafe Neural Engine path."""
        path = tmp_path / "model.aimodel"
        path.write_bytes(b"asset")
        function = Mock(
            return_value={
                "dets": SimpleNamespace(numpy=lambda: np.zeros((1, 2, 4), np.float32)),
                "labels": SimpleNamespace(numpy=lambda: np.zeros((1, 2, 3), np.float32)),
                "keypoints": SimpleNamespace(numpy=lambda: np.zeros((1, 2, 17, 3), np.float32)),
            }
        )
        model = Mock()
        model.load_function.return_value = function
        options = SimpleNamespace(cpu_only=Mock(return_value="cpu-option"), default=Mock(return_value="default-option"))
        ai_model = SimpleNamespace(load=Mock(return_value=model))
        runtime_module = SimpleNamespace(
            AIModel=ai_model, ComputeUnitKind=Mock(), SpecializationOptions=options, NDArray=lambda array: array
        )
        monkeypatch.setitem(sys.modules, "coreai", SimpleNamespace(runtime=runtime_module))
        monkeypatch.setitem(sys.modules, "coreai.runtime", runtime_module)
        monkeypatch.setattr("rfdetr.export._runtime.adapters.platform.system", lambda: "Darwin")
        metadata = _metadata(
            format="coreai",
            task="keypoints",
            input_shape=(1, 3, 8, 8),
            input_dtype="float16",
            num_keypoints_per_class=[17, 17, 17],
            outputs={"pred_boxes": "dets", "pred_logits": "labels", "pred_keypoints": "keypoints"},
        )

        runtime = load_runtime(path, metadata)
        result = runtime.run(torch.zeros(1, 3, 8, 8))

        ai_model.load.assert_called_once_with(path, "cpu-option")
        assert runtime.info["device"] == "cpu"
        assert result["pred_keypoints"].shape == (1, 2, 17, 3)
        del runtime


class TestOpenVINORuntimeAdapter:
    """Check the existing OpenVINO wrapper's positional interface."""

    def test_cpu_run_maps_positional_outputs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The OpenVINO wrapper returns a tuple in metadata order."""
        path = tmp_path / "model.xml"
        path.write_text("<net/>")
        session = Mock()
        raw_boxes = np.zeros((1, 2, 4), np.float32)
        session.infer.return_value = (raw_boxes, np.zeros((1, 2, 3), np.float32))
        session.input_layer.partial_shape = [
            SimpleNamespace(is_static=True, get_length=Mock(return_value=size)) for size in (1, 3, 8, 8)
        ]
        session.output_layers = ["dets", "labels"]
        core = Mock(available_devices=["CPU"])
        monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
        monkeypatch.setattr("rfdetr.export._openvino.inference.OpenVINOInference", lambda *args, **kwargs: session)
        metadata = _metadata(
            format="openvino",
            task="detect",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        runtime = load_runtime(path, metadata, device="cpu")
        result = runtime.run(torch.zeros(1, 3, 8, 8))

        assert runtime.info["device"] == "CPU"
        assert runtime.device == torch.device("cpu")
        assert result["pred_logits"].shape == (1, 2, 3)
        assert result["pred_boxes"].data_ptr() == raw_boxes.ctypes.data
        session.infer.assert_called_once()


class TestExecuTorchRuntimeAdapter:
    """Check the documented ExecuTorch Python runtime method."""

    def test_xnnpack_run_uses_forward_method(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The loader keeps the program alive and executes one batch."""
        path = tmp_path / "model.pte"
        path.write_bytes(b"pte")
        method = Mock()
        method.execute.return_value = [torch.zeros(1, 2, 4), torch.zeros(1, 2, 3)]
        program = Mock()
        program.load_method.return_value = method
        engine = Mock()
        engine.load_program.return_value = program
        runtime_class = SimpleNamespace(get=lambda: engine)
        monkeypatch.setitem(sys.modules, "executorch", SimpleNamespace(runtime=SimpleNamespace(Runtime=runtime_class)))
        monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(Runtime=runtime_class))
        metadata = _metadata(
            format="executorch",
            task="detect",
            backend="xnnpack",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        runtime = load_runtime(path, metadata)
        result = runtime.run(torch.zeros(1, 3, 8, 8))

        program.load_method.assert_called_once_with("forward")
        method.execute.assert_called_once()
        assert result["pred_boxes"].shape == (1, 2, 4)

    def test_runtime_compatibility_error_prevents_program_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A mismatched ExecuTorch runtime stops before the program loader runs."""
        path = tmp_path / "model.pte"
        path.write_bytes(b"pte")
        import rfdetr.export._executorch.exporter as exporter

        monkeypatch.setattr(
            exporter, "_check_executorch_available", Mock(side_effect=ImportError("ABI-compatibility gap"))
        )
        metadata = _metadata(
            format="executorch",
            task="detect",
            backend="xnnpack",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        with pytest.raises(ImportError, match="ABI-compatibility gap"):
            load_runtime(path, metadata)


class TestCoreMLRuntimeAdapter:
    """Check native CoreML output ordering and device policy."""

    def test_cpu_run_uses_spec_output_order(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Position mappings follow the model spec despite dictionary order."""
        path = tmp_path / "model.mlpackage"
        path.mkdir()
        session = Mock()
        session.get_spec.return_value = SimpleNamespace(
            description=SimpleNamespace(
                input=[
                    SimpleNamespace(
                        name="input",
                        type=SimpleNamespace(
                            WhichOneof=lambda field: "multiArrayType",
                            multiArrayType=SimpleNamespace(shape=(1, 3, 8, 8), dataType=65568),
                        ),
                    )
                ],
                output=[SimpleNamespace(name="dets"), SimpleNamespace(name="labels")],
            )
        )
        session.predict.return_value = {
            "labels": np.zeros((1, 2, 3), np.float32),
            "dets": np.zeros((1, 2, 4), np.float32),
        }
        coreml = SimpleNamespace(
            ComputeUnit=SimpleNamespace(ALL="all", CPU_ONLY="cpu"),
            models=SimpleNamespace(MLModel=Mock(return_value=session)),
        )
        monkeypatch.setitem(sys.modules, "coremltools", coreml)
        monkeypatch.setattr("rfdetr.export._runtime.adapters.platform.system", lambda: "Darwin")
        metadata = _metadata(
            format="coreml",
            task="detect",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        runtime = load_runtime(path, metadata, device="cpu")
        result = runtime.run(torch.zeros(1, 3, 8, 8))

        assert result["pred_boxes"].shape == (1, 2, 4)
        assert result["pred_logits"].shape == (1, 2, 3)
        coreml.models.MLModel.assert_called_once_with(str(path), compute_units="cpu")

    def test_explicit_ane_is_refused(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """CoreML cannot guarantee an ANE-only execution request."""
        path = tmp_path / "model.mlpackage"
        path.mkdir()
        coreml = SimpleNamespace(ComputeUnit=SimpleNamespace(ALL="all", CPU_ONLY="cpu"))
        monkeypatch.setitem(sys.modules, "coremltools", coreml)
        monkeypatch.setattr("rfdetr.export._runtime.adapters.platform.system", lambda: "Darwin")
        metadata = _metadata(
            format="coreml",
            task="detect",
            input_shape=(1, 3, 8, 8),
            outputs={"pred_boxes": 0, "pred_logits": 1},
        )

        with pytest.raises(ValueError, match="also permit CPU"):
            load_runtime(path, metadata, device="ane")
