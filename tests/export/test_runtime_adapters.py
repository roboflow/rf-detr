# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Boundary tests for exported inference runtime policies."""

import asyncio
import sys
import threading
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rfdetr.export._runtime.adapters import load_runtime
from rfdetr.export._runtime.metadata import ExportMetadata


def test_coreai_sync_calls_share_one_loop_inside_running_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The sync factory and prediction call work inside an active caller loop."""
    artifact = tmp_path / "model.aimodel"
    artifact.write_bytes(b"coreai")
    contract = ExportMetadata(
        format="coreai",
        task="detect",
        input_shape=(1, 3, 2, 2),
        outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    worker_loops: list[asyncio.AbstractEventLoop] = []
    worker_threads: list[threading.Thread] = []

    class NDArray:
        """Stand in for Core AI's input array."""

        def __init__(self, array: np.ndarray) -> None:
            self.array = array

    class OutputValue:
        """Expose the Core AI output conversion method."""

        def __init__(self, array: np.ndarray) -> None:
            self.array = array

        def numpy(self) -> np.ndarray:
            """Return one raw output array."""
            return self.array

    class Outputs:
        """Expose name indexing without inheriting from dict."""

        def __init__(self) -> None:
            self.values = {
                "dets": OutputValue(np.array([[[0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)),
                "labels": OutputValue(np.array([[[4.0, -4.0]]], dtype=np.float32)),
            }

        def __getitem__(self, name: str) -> OutputValue:
            """Return one named output."""
            return self.values[name]

    async def execute(_inputs: dict[str, NDArray]) -> Outputs | dict[str, np.ndarray]:
        """Record the runtime loop and return both supported output map styles."""
        worker_loops.append(asyncio.get_running_loop())
        worker_threads.append(threading.current_thread())
        if len(worker_loops) == 4:
            return {
                "dets": np.array([[[0.5, 0.5, 1.0, 1.0]]], dtype=np.float32),
                "labels": np.array([[[4.0, -4.0]]], dtype=np.float32),
            }
        return Outputs()

    class LoadedModel:
        """Expose the Core AI function loader."""

        async def load_function(self, _name: str) -> object:
            """Return the model function on the runtime loop."""
            worker_loops.append(asyncio.get_running_loop())
            worker_threads.append(threading.current_thread())
            return execute

    class AIModel:
        """Expose the Core AI model loader."""

        @staticmethod
        async def load(_path: Path, _options: object) -> LoadedModel:
            """Return a model on the runtime loop."""
            worker_loops.append(asyncio.get_running_loop())
            worker_threads.append(threading.current_thread())
            return LoadedModel()

    runtime_module = types.ModuleType("coreai.runtime")
    runtime_module.AIModel = AIModel
    runtime_module.NDArray = NDArray
    runtime_module.ComputeUnitKind = object
    runtime_module.SpecializationOptions = types.SimpleNamespace(default=lambda: object())
    monkeypatch.setitem(sys.modules, "coreai", types.ModuleType("coreai"))
    monkeypatch.setitem(sys.modules, "coreai.runtime", runtime_module)
    monkeypatch.setattr("rfdetr.export._runtime.adapters._require_apple", lambda _name: None)

    async def caller() -> None:
        """Exercise the synchronous runtime API from an already running loop."""
        runtime = load_runtime(artifact, contract, device="auto")
        first = runtime.run(torch.zeros(1, 3, 2, 2))
        second = runtime.run(torch.zeros(1, 3, 2, 2))
        assert first["pred_boxes"].shape == (1, 1, 4)
        assert second["pred_logits"].shape == (1, 1, 2)
        runtime._coreai_bridge._finalizer()

    asyncio.run(caller())
    assert len({id(loop) for loop in worker_loops}) == 1
    assert len({id(thread) for thread in worker_threads}) == 1
    assert worker_threads[0] is not threading.current_thread()


@pytest.mark.parametrize("device", ["gpu", "ane"])
def test_coreai_rejects_accelerator_preference_as_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    """A preferred compute unit does not guarantee an accelerator-only run."""
    artifact = tmp_path / "model.aimodel"
    artifact.write_bytes(b"coreai")
    contract = ExportMetadata(
        format="coreai",
        task="detect",
        input_shape=(1, 3, 2, 2),
        outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    runtime_module = types.ModuleType("coreai.runtime")
    runtime_module.AIModel = object
    runtime_module.SpecializationOptions = object
    monkeypatch.setitem(sys.modules, "coreai", types.ModuleType("coreai"))
    monkeypatch.setitem(sys.modules, "coreai.runtime", runtime_module)
    monkeypatch.setattr("rfdetr.export._runtime.adapters._require_apple", lambda _name: None)

    with pytest.raises(ValueError, match="preferred compute unit"):
        load_runtime(artifact, contract, device=device)


def test_openvino_accepts_family_alias_but_checks_numbered_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A GPU family request can use GPU.0, but GPU.1 must exist to be selected."""
    artifact = tmp_path / "model.xml"
    artifact.write_text("<model/>", encoding="utf-8")
    contract = ExportMetadata(
        format="openvino",
        task="detect",
        input_shape=(1, 3, 2, 2),
        input_name=0,
        outputs={"pred_boxes": 0, "pred_logits": 1},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )

    class Dimension:
        """Expose one fixed OpenVINO input dimension."""

        is_static = True

        def __init__(self, size: int) -> None:
            self.size = size

        def get_length(self) -> int:
            """Return the fixed dimension size."""
            return self.size

    class FakeSession:
        """Expose the input and output interface of an OpenVINO IR."""

        def __init__(self, _path: Path, device: str, inference_precision: str | None = None) -> None:
            self.device = device
            self.inference_precision = inference_precision
            self.input_layer = types.SimpleNamespace(partial_shape=[Dimension(size) for size in (1, 3, 2, 2)])
            self.output_layers = [object(), object()]

    ov_module = types.ModuleType("openvino")
    ov_module.Core = lambda: types.SimpleNamespace(available_devices=["CPU", "GPU.0"])
    inference_module = types.ModuleType("rfdetr.export._openvino.inference")
    inference_module.OpenVINOInference = FakeSession
    monkeypatch.setitem(sys.modules, "openvino", ov_module)
    monkeypatch.setitem(sys.modules, "rfdetr.export._openvino.inference", inference_module)

    gpu_runtime = load_runtime(artifact, contract, device="gpu")
    assert gpu_runtime.session.device == "GPU"
    assert gpu_runtime.session.inference_precision == "f32"
    assert load_runtime(artifact, contract, device="gpu.0").session.device == "GPU.0"
    with pytest.raises(RuntimeError, match="GPU.1 is unavailable"):
        load_runtime(artifact, contract, device="gpu.1")


def test_onnx_rejects_input_rank_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An ONNX graph must declare all four input axes."""
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"onnx")
    contract = ExportMetadata(
        format="onnx",
        task="detect",
        input_shape=(1, 3, 8, 8),
        outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    session = Mock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.get_inputs.return_value = [types.SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 8])]
    monkeypatch.setitem(
        sys.modules, "onnxruntime", types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    )
    monkeypatch.setattr("rfdetr.export._onnx.inference._create_onnx_session", lambda *args, **kwargs: session)

    with pytest.raises(ValueError, match="input rank"):
        load_runtime(artifact, contract, device="cpu")


def test_tensorrt_rejects_input_rank_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A TensorRT engine must expose a four-axis image binding."""
    artifact = tmp_path / "model.engine"
    artifact.write_bytes(b"engine")
    contract = ExportMetadata(
        format="tensorrt",
        task="detect",
        input_shape=(1, 3, 8, 8),
        outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    session = Mock()
    session.input_names = ["input"]
    session.bindings = {"input": types.SimpleNamespace(dtype=np.float32, shape=(1, 3, 8))}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("rfdetr.export._tensorrt.inference.TRTInference", lambda *args, **kwargs: session)

    with pytest.raises(ValueError, match="input rank"):
        load_runtime(artifact, contract)


def test_tflite_rejects_input_rank_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A TFLite graph must expose a four-axis image input."""
    artifact = tmp_path / "model.tflite"
    artifact.write_bytes(b"tflite")
    contract = ExportMetadata(
        format="tflite",
        task="detect",
        input_shape=(1, 3, 8, 8),
        input_layout="NHWC",
        input_name=0,
        outputs={"pred_boxes": 0, "pred_logits": 1},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    session = Mock()
    session.get_input_details.return_value = [{"name": "input", "index": 0, "shape": [1, 8, 8], "dtype": np.float32}]
    monkeypatch.setattr("rfdetr.export._tflite.inference._create_interpreter", lambda *args: session)

    with pytest.raises(ValueError, match="input rank"):
        load_runtime(artifact, contract)


@pytest.mark.parametrize("graph_shape,graph_dtype", [((1, 3, 8), 65568), ((1, 3, 8, 9), 65568), ((1, 3, 8, 8), 65552)])
def test_coreml_rejects_input_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, graph_shape: tuple[int, ...], graph_dtype: int
) -> None:
    """CoreML input rank, shape, and dtype must agree with artifact metadata."""
    artifact = tmp_path / "model.mlpackage"
    artifact.mkdir()
    contract = ExportMetadata(
        format="coreml",
        task="detect",
        input_shape=(1, 3, 8, 8),
        outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    array_type = types.SimpleNamespace(shape=graph_shape, dataType=graph_dtype)
    feature_type = types.SimpleNamespace(WhichOneof=lambda _name: "multiArrayType", multiArrayType=array_type)
    session = Mock()
    session.get_spec.return_value = types.SimpleNamespace(
        description=types.SimpleNamespace(
            input=[types.SimpleNamespace(name="input", type=feature_type)],
            output=[types.SimpleNamespace(name="dets"), types.SimpleNamespace(name="labels")],
        )
    )
    coreml = types.SimpleNamespace(
        ComputeUnit=types.SimpleNamespace(ALL="all", CPU_ONLY="cpu"),
        models=types.SimpleNamespace(MLModel=Mock(return_value=session)),
    )
    monkeypatch.setitem(sys.modules, "coremltools", coreml)
    monkeypatch.setattr("rfdetr.export._runtime.adapters.platform.system", lambda: "Darwin")

    with pytest.raises(ValueError, match="CoreML input (rank|shape|dtype)"):
        load_runtime(artifact, contract, device="cpu")
