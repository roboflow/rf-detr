# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Boundary tests for exported inference runtime policies."""

import asyncio
import re
import sys
import threading
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from rfdetr.export._runtime import adapters
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
        monkeypatch.setattr("rfdetr.export._runtime.adapters.platform.system", lambda: "Darwin")

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
        monkeypatch.setattr("rfdetr.export._runtime.adapters.platform.system", lambda: "Darwin")

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
                self.compiled_model = types.SimpleNamespace(inputs=[object()])
                self.input_layer = types.SimpleNamespace(partial_shape=FakePartialShape((1, 3, 2, 2)))
                self.output_layers = [object(), object()]

        ov_module = types.ModuleType("openvino")
        ov_module.Core = lambda: types.SimpleNamespace(available_devices=["CPU", "GPU.0"])
        monkeypatch.setitem(sys.modules, "openvino", ov_module)
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", FakeSession)

        gpu_runtime = load_runtime(artifact, contract, device="gpu")
        assert gpu_runtime.session.device == "GPU"
        assert gpu_runtime.session.inference_precision == "auto"
        assert load_runtime(artifact, contract, device="gpu.0").session.device == "GPU.0"
        with pytest.raises(RuntimeError, match="GPU.1 is unavailable"):
            load_runtime(artifact, contract, device="gpu.1")

    @pytest.mark.parametrize("device", ["auto", "AUTO", "Auto"])
    def test_openvino_accepts_auto_in_any_case(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
    ) -> None:
        """Every casing of auto selects the OpenVINO AUTO plugin."""
        artifact = tmp_path / "model.xml"
        artifact.write_text("<model/>", encoding="utf-8")
        session_loader = Mock(side_effect=RuntimeError("stop after session load"))
        monkeypatch.setitem(sys.modules, "openvino", types.ModuleType("openvino"))
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", session_loader)

        with pytest.raises(RuntimeError, match="stop after session load"):
            load_runtime(artifact, _detect_metadata("openvino"), device=device)

        session_loader.assert_called_once()
        assert session_loader.call_args.kwargs["device"] == "AUTO"

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
            compiled_model=types.SimpleNamespace(inputs=[object()]),
            input_layer=types.SimpleNamespace(partial_shape=partial_shape),
            output_layers=[object(), object()],
        )
        ov_module = types.ModuleType("openvino")
        ov_module.Core = lambda: types.SimpleNamespace(available_devices=["CPU"])
        monkeypatch.setitem(sys.modules, "openvino", ov_module)
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", lambda *_a, **_kw: session)

        with pytest.raises(ValueError, match="OpenVINO input rank must be 4"):
            load_runtime(artifact, contract, device="cpu")

    def test_tensorrt_rejects_non_cuda_device(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An unsupported device spelling fails before querying CUDA availability."""
        artifact = tmp_path / "model.engine"
        artifact.write_bytes(b"engine")
        availability = Mock(return_value=False)
        monkeypatch.setattr(torch.cuda, "is_available", availability)

        with pytest.raises(ValueError, match="auto, cuda, or cuda:N"):
            load_runtime(artifact, _detect_metadata("tensorrt"), device="cpu")

        availability.assert_not_called()

    @pytest.mark.parametrize("device", ["auto", "cuda", "cuda:1"])
    def test_tensorrt_requires_available_cuda(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
    ) -> None:
        """Accepted CUDA spellings fail with a runtime error when CUDA is unavailable."""
        artifact = tmp_path / "model.engine"
        artifact.write_bytes(b"engine")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="TensorRT requires an available CUDA device"):
            load_runtime(artifact, _detect_metadata("tensorrt"), device=device)

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


class TestRuntimeSuffixes:
    """Artifact suffixes must agree with the metadata before a runtime is imported."""

    @pytest.mark.parametrize(
        "format,allowed",
        [
            pytest.param("onnx", [".onnx"], id="onnx"),
            pytest.param("tflite", [".tflite"], id="tflite"),
            pytest.param("litert", [".tflite"], id="litert"),
            pytest.param("executorch", [".pte"], id="executorch"),
            pytest.param("coreml", [".mlmodel", ".mlpackage"], id="coreml"),
            pytest.param("coreai", [".aimodel"], id="coreai"),
            pytest.param("openvino", [".xml"], id="openvino"),
        ],
    )
    def test_wrong_suffix_is_refused_before_loader_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str, allowed: list[str]
    ) -> None:
        """A mismatch names the format, sorted suffixes, filename and corrective action before dispatch."""
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"artifact")
        resolver = Mock(side_effect=AssertionError("runtime loader must not be resolved"))
        monkeypatch.setattr("rfdetr.export._runtime.adapters.resolve_runtime_loader", resolver)
        message = (
            f"{REGISTRY[format].label} artifacts require one of {allowed}, got {artifact.name!r}. "
            "Pass the file the exporter wrote or fix the format in the metadata."
        )

        with pytest.raises(ValueError, match=re.escape(message)):
            load_runtime(artifact, _detect_metadata(format))

        resolver.assert_not_called()

    def test_tensorrt_arbitrary_suffix_reaches_options_check(self, tmp_path: Path) -> None:
        """TensorRT engines have no required suffix and still reach their loader's validation."""
        artifact = tmp_path / "model.arbitrary"
        artifact.write_bytes(b"engine")

        with pytest.raises(ValueError, match="does not accept 'cache_dirr'"):
            load_runtime(artifact, _detect_metadata("tensorrt"), options={"cache_dirr": "cache"})

    @pytest.mark.parametrize(
        "format,filename",
        [
            pytest.param("onnx", "model.ONNX", id="onnx"),
            pytest.param("tflite", "model.TFLITE", id="tflite"),
            pytest.param("litert", "model.TFLITE", id="litert"),
            pytest.param("executorch", "model.PTE", id="executorch"),
            pytest.param("coreml", "model.MLPACKAGE", id="coreml-package"),
            pytest.param("coreml", "model.MLMODEL", id="coreml-model"),
            pytest.param("coreai", "model.AIMODEL", id="coreai"),
            pytest.param("openvino", "model.XML", id="openvino"),
        ],
    )
    def test_uppercase_suffix_reaches_options_check(self, tmp_path: Path, format: str, filename: str) -> None:
        """Valid suffixes are case insensitive and dispatch to the real format loader."""
        artifact = tmp_path / filename
        artifact.write_bytes(b"artifact")

        with pytest.raises(ValueError, match="does not accept 'cache_dirr'"):
            load_runtime(artifact, _detect_metadata(format), options={"cache_dirr": "cache"})


class TestGraphInterfaceChecks:
    """The shared graph-interface checks every format loader runs before it builds a runtime."""

    def test_onnx_refuses_multi_input_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A second graph input is named in the error instead of failing a tuple unpack.

        The image input is the only one the shared runtime feeds, so a graph with an extra input cannot run; the loader
        used to unpack the input list and raise a bare "too many values to unpack".
        """
        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"onnx")
        session = Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        session.get_inputs.return_value = [
            types.SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 8, 8]),
            types.SimpleNamespace(name="extra", type="tensor(float)", shape=[1, 3, 8, 8]),
        ]
        monkeypatch.setitem(
            sys.modules, "onnxruntime", types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        )
        monkeypatch.setattr("rfdetr.export._onnx.inference._create_onnx_session", lambda *_a, **_kw: session)

        with pytest.raises(ValueError, match=r"ONNX expected one input, got 2\."):
            load_runtime(artifact, _detect_metadata("onnx"), device="cpu")

    def test_openvino_refuses_multi_input_graph(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An OpenVINO IR with a second input is refused; the loader used to read only its first input.

        OpenVINO exposes the image input through ``compiled_model.input(0)``, which silently picks the first of several
        inputs, so the count has to be checked against the compiled model's full input list.
        """
        artifact = tmp_path / "model.xml"
        artifact.write_text("<model/>", encoding="utf-8")
        session = types.SimpleNamespace(
            compiled_model=types.SimpleNamespace(inputs=[object(), object()]),
            input_layer=types.SimpleNamespace(partial_shape=FakePartialShape((1, 3, 8, 8))),
            output_layers=[object(), object()],
        )
        monkeypatch.setitem(sys.modules, "openvino", types.ModuleType("openvino"))
        monkeypatch.setattr("rfdetr.export._openvino.inference._load_openvino_session", lambda *_a, **_kw: session)
        contract = _detect_metadata("openvino").model_copy(update={"outputs": {"pred_boxes": 0, "pred_logits": 1}})

        with pytest.raises(ValueError, match=r"OpenVINO expected one input, got 2\."):
            load_runtime(artifact, contract)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((None, 3, 8, 8), id="dynamic-batch-axis"),
            pytest.param((1, 3, None, None), id="dynamic-spatial-axes"),
        ],
    )
    def test_dynamic_graph_axis_accepts_fixed_metadata(self, shape: tuple[int | None, ...]) -> None:
        """A graph axis left dynamic accepts the fixed size the metadata records.

        ONNX reports a symbolic dimension as a string, which the loader turns into ``None`` before the check.
        """
        assert adapters._check_input_interface("ONNX", _detect_metadata("onnx"), "input", shape, "float32") is None

    def test_metadata_batch_wildcard_accepts_fixed_graph_batch(self) -> None:
        """A ``-1`` metadata batch accepts any graph batch for a format that opts into ``dynamic_batch``.

        TFLite and TensorRT report a concrete batch for a graph exported with a dynamic batch axis.
        """
        contract = _detect_metadata("tflite").model_copy(update={"input_shape": (-1, 3, 8, 8)})

        assert (
            adapters._check_input_interface("TFLite", contract, None, (1, 3, 8, 8), "float32", dynamic_batch=True)
            is None
        )

    def test_metadata_batch_wildcard_refused_for_fixed_shape_format(self) -> None:
        """A ``-1`` metadata batch is refused where the graph batch must match exactly, as for CoreML.

        A CoreML model has a fixed input shape, so metadata claiming a dynamic batch disagrees with the artifact.
        """
        contract = _detect_metadata("coreml").model_copy(update={"input_shape": (-1, 3, 8, 8)})

        with pytest.raises(ValueError, match=r"CoreML input shape \(1, 3, 8, 8\) disagrees"):
            adapters._check_input_interface("CoreML", contract, None, (1, 3, 8, 8), "float32")

    @pytest.mark.parametrize(
        "outputs,match",
        [
            pytest.param(["dets"], "output name 'labels' is absent", id="name-absent-from-graph"),
            pytest.param(2, "must use positions", id="name-against-output-count"),
        ],
    )
    def test_output_name_mapping_refused(self, outputs: list[str] | int, match: str) -> None:
        """A named output mapping is refused when the graph lacks that name or exposes only an output count.

        OpenVINO passes only its output count, so a named mapping there can never be resolved.
        """
        with pytest.raises(ValueError, match=match):
            adapters._check_output_interface("Runtime", _detect_metadata("onnx"), outputs)

    def test_output_position_past_last_output_refused(self) -> None:
        """A positional output mapping past the graph's last output is refused, also for a format exposing names.

        ONNX and TensorRT used to check named mappings only, so a bad position surfaced at the first prediction.
        """
        contract = _detect_metadata("onnx").model_copy(update={"outputs": {"pred_boxes": 0, "pred_logits": 2}})

        with pytest.raises(ValueError, match="output position 2 is absent"):
            adapters._check_output_interface("ONNX", contract, ["dets", "labels"])


class TestRuntimeOptions:
    """Format loaders read only their own runtime options and forward them to the session loader."""

    @pytest.mark.parametrize(
        "format,filename",
        [
            pytest.param("onnx", "model.onnx", id="onnx"),
            pytest.param("tflite", "model.tflite", id="tflite"),
            pytest.param("litert", "model.tflite", id="litert"),
            pytest.param("tensorrt", "model.engine", id="tensorrt"),
            pytest.param("executorch", "model.pte", id="executorch"),
            pytest.param("coreml", "model.mlpackage", id="coreml"),
            pytest.param("coreai", "model.aimodel", id="coreai"),
            pytest.param("openvino", "model.xml", id="openvino"),
        ],
    )
    def test_unknown_key_is_refused_before_runtime_import(self, tmp_path: Path, format: str, filename: str) -> None:
        """A misspelled or wrong-format key fails on every host, before the format's runtime package is needed.

        No runtime package is faked here, so a loader that imported its runtime, or checked the host, before the options
        would fail with ImportError or RuntimeError instead of naming the key.
        """
        artifact = tmp_path / filename
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

        The export-side precision alias is passed through for the session loader to resolve, and a path-like cache
        directory is passed on as the string OpenVINO's CACHE_DIR property takes.
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
            inference_precision="float16",
            cache_dir=str(tmp_path / "cache"),
            config={"INFERENCE_NUM_THREADS": 2},
        )
