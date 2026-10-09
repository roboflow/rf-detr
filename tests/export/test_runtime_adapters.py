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
from rfdetr.export.registry import REGISTRY
from tests.export._openvino_shapes import FakePartialShape


class TestRuntimePolicies:
    """Validate device and tensor contracts at the runtime boundary."""

    def test_coreai_sync_calls_share_one_loop_inside_running_loop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
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
            del runtime

        asyncio.run(caller())
        assert len({id(loop) for loop in worker_loops}) == 1
        assert len({id(thread) for thread in worker_threads}) == 1
        assert worker_threads[0] is not threading.current_thread()
        assert not worker_threads[0].is_alive()

    @pytest.mark.parametrize("device", ["gpu", "ane"])
    def test_coreai_rejects_accelerator_preference_as_device(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

        class FakeSession:
            """Expose the input and output interface of an OpenVINO IR."""

            def __init__(self, _path: Path, device: str, inference_precision: str | None = None) -> None:
                self.device = device
                self.inference_precision = inference_precision
                self.input_layer = types.SimpleNamespace(partial_shape=FakePartialShape((1, 3, 2, 2)))
                self.output_layers = [object(), object()]

        ov_module = types.ModuleType("openvino")
        ov_module.Core = lambda: types.SimpleNamespace(available_devices=["CPU", "GPU.0"])
        monkeypatch.setitem(sys.modules, "openvino", ov_module)
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", FakeSession)

        gpu_runtime = load_runtime(artifact, contract, device="gpu")
        assert gpu_runtime.session.device == "GPU"
        assert gpu_runtime.session.inference_precision == "f32"
        assert load_runtime(artifact, contract, device="gpu.0").session.device == "GPU.0"
        with pytest.raises(RuntimeError, match="GPU.1 is unavailable"):
            load_runtime(artifact, contract, device="gpu.1")

    def test_onnx_rejects_input_rank_mismatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    @pytest.mark.parametrize(
        "partial_shape",
        [
            pytest.param(FakePartialShape((1, 3, 8)), id="rank-3"),
            pytest.param(FakePartialShape((1, 3, 8, 8, 1)), id="rank-5"),
            pytest.param(FakePartialShape((), static_rank=False), id="dynamic-rank"),
        ],
    )
    def test_openvino_rejects_input_rank_mismatch(
        self, partial_shape: FakePartialShape, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An OpenVINO IR must declare exactly the four input axes its metadata records.

        A three-axis input used to index past its last axis (``IndexError``), and a five-axis input passed with its
        extra axis unchecked; a dynamic rank has no axes to compare. All three are refused with the same message.
        """
        artifact = tmp_path / "model.xml"
        artifact.write_text("<model/>", encoding="utf-8")
        contract = ExportMetadata(
            format="openvino",
            task="detect",
            input_shape=(1, 3, 8, 8),
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
        session = types.SimpleNamespace(
            input_layer=types.SimpleNamespace(partial_shape=partial_shape), output_layers=[object(), object()]
        )
        ov_module = types.ModuleType("openvino")
        ov_module.Core = lambda: types.SimpleNamespace(available_devices=["CPU"])
        monkeypatch.setitem(sys.modules, "openvino", ov_module)
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", lambda *_a, **_kw: session)

        with pytest.raises(ValueError, match="OpenVINO input rank must be 4"):
            load_runtime(artifact, contract, device="cpu")

    def test_tensorrt_rejects_input_rank_mismatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        monkeypatch.setattr("rfdetr.export._tensorrt.inference._load_tensorrt_session", lambda *args, **kwargs: session)

        with pytest.raises(ValueError, match="input rank"):
            load_runtime(artifact, contract)

    def test_tflite_rejects_input_rank_mismatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        session.get_input_details.return_value = [
            {"name": "input", "index": 0, "shape": [1, 8, 8], "dtype": np.float32}
        ]
        monkeypatch.setattr("rfdetr.export._tflite.inference._create_interpreter", lambda *args: session)

        with pytest.raises(ValueError, match="input rank"):
            load_runtime(artifact, contract)

    @pytest.mark.parametrize(
        "graph_shape,graph_dtype", [((1, 3, 8), 65568), ((1, 3, 8, 9), 65568), ((1, 3, 8, 8), 65552)]
    )
    def test_coreml_rejects_input_contract_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, graph_shape: tuple[int, ...], graph_dtype: int
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


def _detect_metadata(format: str) -> ExportMetadata:
    """Build minimal detection metadata for one export format.

    Examples:
        >>> _detect_metadata("onnx").format
        'onnx'
    """
    return ExportMetadata(
        format=format,
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


class TestRuntimeOptions:
    """Format loaders read only their own runtime options and forward them to the session loader."""

    @pytest.mark.parametrize("format", sorted(REGISTRY))
    def test_unknown_key_is_refused_before_runtime_import(self, tmp_path: Path, format: str) -> None:
        """A misspelled or wrong-format key fails on every host, before the format's runtime package is needed.

        No runtime package is faked here, so a loader that imported its runtime, or checked the host, before the options
        would fail with ImportError or RuntimeError instead of naming the key.
        """
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"artifact")

        with pytest.raises(ValueError, match="does not accept 'cache_dirr'"):
            load_runtime(artifact, _detect_metadata(format), options={"cache_dirr": "cache"})

    def test_non_mapping_options_are_refused(self, tmp_path: Path) -> None:
        """A list of pairs is not silently read as options."""
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"artifact")

        with pytest.raises(TypeError, match="must be a mapping"):
            load_runtime(artifact, _detect_metadata("tensorrt"), options=[("cuda_graph", True)])

    @pytest.mark.parametrize(
        "options,expected",
        [
            pytest.param(
                {},
                {"sync_mode": True, "verbose": False, "cuda_graph": False, "engine_host_code_allowed": False},
                id="default-execute-v2",
            ),
            pytest.param(
                {"cuda_graph": True, "engine_host_code_allowed": True, "verbose": True},
                {"sync_mode": False, "verbose": True, "cuda_graph": True, "engine_host_code_allowed": True},
                id="cuda-graph-trusted-verbose",
            ),
        ],
    )
    def test_tensorrt_forwards_options_to_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, options: dict[str, bool], expected: dict[str, bool]
    ) -> None:
        """The default stays execute_v2; a CUDA graph drops sync_mode, which the session loader cannot combine with it.

        The session loader is where a dynamic-profile engine is refused for cuda_graph=True, so forwarding the flag is
        what makes that refusal reach RFDETRInference.
        """
        artifact = tmp_path / "model.engine"
        artifact.write_bytes(b"engine")
        session_loader = Mock(side_effect=RuntimeError("stop after session load"))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("rfdetr.export._tensorrt.inference._load_tensorrt_session", session_loader)

        with pytest.raises(RuntimeError, match="stop after session load"):
            load_runtime(artifact, _detect_metadata("tensorrt"), options=options)

        session_loader.assert_called_once_with(str(artifact), device="cuda:0", **expected)

    def test_tensorrt_refuses_non_bool_option(self, tmp_path: Path) -> None:
        """A truthy string such as "false" from a config file must not switch graph capture on."""
        artifact = tmp_path / "model.engine"
        artifact.write_bytes(b"engine")

        with pytest.raises(ValueError, match="cuda_graph must be a bool"):
            load_runtime(artifact, _detect_metadata("tensorrt"), options={"cuda_graph": "false"})

    def test_openvino_forwards_options_to_session(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The compile cache, precision hint and compile properties reach the shared OpenVINO session loader.

        The export-side precision alias resolves to OpenVINO's own spelling, and a path-like cache directory is passed
        on as the string OpenVINO's CACHE_DIR property takes.
        """
        artifact = tmp_path / "model.xml"
        artifact.write_text("<model/>", encoding="utf-8")
        session_loader = Mock(side_effect=RuntimeError("stop after session load"))
        monkeypatch.setitem(sys.modules, "openvino", types.ModuleType("openvino"))
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", session_loader)
        options = {
            "cache_dir": tmp_path / "cache",
            "inference_precision": "float16",
            "config": {"INFERENCE_NUM_THREADS": 2},
        }

        with pytest.raises(RuntimeError, match="stop after session load"):
            load_runtime(artifact, _detect_metadata("openvino"), options=options)

        session_loader.assert_called_once_with(
            artifact,
            device="AUTO",
            inference_precision="f16",
            cache_dir=str(tmp_path / "cache"),
            config={"INFERENCE_NUM_THREADS": 2},
        )
