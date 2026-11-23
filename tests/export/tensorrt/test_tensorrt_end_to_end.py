# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end TensorRT tests: real FP16 and INT8 engines built on a GPU and compared with the eager model.

The classes are marked ``gpu``, ``integration`` and ``e2e_tensorrt``, need ``rfdetr[tensorrt]`` and are opt-in, so the
TensorRT end-to-end CI job selects them by marker. The builder's unit tests are in ``test_tensorrt_build_engine.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import inference as tensorrt_inference
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou
from tests._markers import _requires_cuda, _requires_tensorrt, _requires_two_cuda_devices
from tests.export.conftest import (
    _structured_parity_input,
    assert_prediction_roundtrip,
    eager_reference_tensors,
    max_abs_output_diffs,
)

# FP32 TensorRT fuses kernels, so it diverges more than the XNNPACK CPU path (~1e-5); a structural regression
# collapses outputs by >=1e-1. Recalibrate once real GPU numbers are observed in the tensorrt-parity CI job.
_TENSORRT_MAX_ABS_DIFF = 1e-2

# FP16 has 2^-11 (~4.9e-4) relative resolution and the error compounds across a whole-graph cast with no FP32
# fallback; the review measured 0.11-0.16 max-abs on ``labels`` under onnxruntime-CPU. The bound sits above that to
# catch NaN/Inf, collapsed outputs or a silent FP32 engine labelled FP16. Recalibrate in the tensorrt-parity CI job.
_TENSORRT_FP16_MAX_ABS_DIFF = 3e-1


def _distinct_batch(batch: int, resolution: int) -> torch.Tensor:
    """Stack *batch* structured inputs with different per-image scaling, so batch positions are not interchangeable.

    ``_structured_parity_input`` repeats one sample across the batch; a dynamic-batch engine that mixed up or
    duplicated batch positions would still pass on that. Scaling each image differently makes every position
    distinguishable.

    Args:
        batch: Number of images to stack.
        resolution: Square spatial size of each image.

    Returns:
        Contiguous float tensor shaped ``(batch, 3, resolution, resolution)``.

    Examples:
        >>> t = _distinct_batch(3, 8)
        >>> t.shape
        torch.Size([3, 3, 8, 8])
        >>> bool(torch.equal(t[0], t[1]))
        False
    """
    base = _structured_parity_input(1, 3, resolution, resolution)
    return torch.cat([base * (1.0 + 0.15 * index) for index in range(batch)], dim=0).contiguous()


@_requires_tensorrt
@_requires_cuda
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestTensorRTEndToEnd:
    """Real ONNX -> TensorRT engine build + runtime parity on GPU (requires ``rfdetr[tensorrt]`` and CUDA)."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    def test_public_export_prediction_matches_native(self, task: str, tmp_path: Path) -> None:
        """The public facade preserves predictions after a real TensorRT export."""
        assert_prediction_roundtrip("tensorrt", task, tmp_path)

    @pytest.fixture(scope="class")
    def trt_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
        """Export RFDETRNano to ONNX, build a FP32 ``.trt`` engine, and reuse it across the parity checks."""
        from rfdetr import RFDETRNano

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt")
        detector = RFDETRNano(pretrain_weights=None)
        onnx_path = detector.export(output_dir=str(out_dir), format="onnx", verbose=False)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False))
        engine_path = exporter.build_engine(str(onnx_path))

        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        return model, example, Path(engine_path)

    def test_engine_file_written(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """build_engine must produce a non-empty ``.trt`` engine from the exported ONNX."""
        _, _, engine_path = trt_engine
        assert engine_path.is_file()
        assert engine_path.suffix == ".trt"
        assert engine_path.stat().st_size > 0

    def test_runtime_output_matches_pytorch(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """The TensorRT engine's outputs (dets, labels) must match eager PyTorch within FP32 tolerance."""
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, example, engine_path = trt_engine
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.asarray(outputs[name], dtype=np.float32)) for name in output_names]

        diffs = max_abs_output_diffs(eager_tensors, trt_tensors, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_MAX_ABS_DIFF, (
            f"TensorRT outputs diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_MAX_ABS_DIFF})"
        )

    @pytest.fixture(
        scope="class",
        params=[
            pytest.param((False, False), id="static-fp32"),
            pytest.param((True, False), id="dynamic-fp32"),
            pytest.param((False, True), id="static-fp16"),
            pytest.param((True, True), id="dynamic-fp16"),
        ],
    )
    def trt_sidecar_engine(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
        """Export RFDETRNano to a TensorRT engine with ``trt_metadata=True`` and deserialize the engine it wrote.

        Returns:
            A namespace with the parsed ``sidecar``, the ``engine_path``, the deserialized ``engine`` (and the
            ``runtime`` that must outlive it), the model's ``resolution``, and the ``dynamic_batch`` and ``fp16``
            settings of this parameter.
        """
        import tensorrt as trt

        from rfdetr import RFDETRNano

        dynamic_batch, fp16 = request.param
        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt_sidecar")
        detector = RFDETRNano(pretrain_weights=None)
        engine_path = detector.export(
            output_dir=str(out_dir),
            format="tensorrt",
            fp16=fp16,
            verbose=False,
            trt_metadata=True,
            dynamic_batch=dynamic_batch,
            batch_size=2,
            max_batch_size=4 if dynamic_batch else None,
        )
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        return types.SimpleNamespace(
            sidecar=json.loads(engine_path.with_suffix(".json").read_text()),
            engine_path=engine_path,
            engine=engine,
            runtime=runtime,
            resolution=int(detector.model.resolution),
            dynamic_batch=dynamic_batch,
            fp16=fp16,
        )

    @staticmethod
    def _tensor_names(engine: Any) -> tuple[list[str], list[str]]:
        """Return the deserialized engine's input tensor names and output tensor names, in binding order.

        Examples:
            Needs a real engine on a GPU, so this is documentation only (not a doctest):

            >>> TestTensorRTEndToEnd._tensor_names(engine)  # doctest: +SKIP
            (['input'], ['dets', 'labels'])
        """
        import tensorrt as trt

        modes = {name: engine.get_tensor_mode(name) for name in engine}
        return (
            [name for name, mode in modes.items() if mode == trt.TensorIOMode.INPUT],
            [name for name, mode in modes.items() if mode == trt.TensorIOMode.OUTPUT],
        )

    def test_the_sidecar_input_is_the_engines_input(self, trt_sidecar_engine: Any) -> None:
        """The input's name and the spatial size match what the engine expects (a static engine's shape, a dynamic one's
        profile)."""
        sidecar, engine = trt_sidecar_engine.sidecar, trt_sidecar_engine.engine
        (engine_input,), _ = self._tensor_names(engine)

        assert sidecar["input"]["name"] == engine_input
        resolution = trt_sidecar_engine.resolution
        assert (sidecar["input"]["height"], sidecar["input"]["width"]) == (resolution, resolution)
        assert tuple(engine.get_tensor_shape(engine_input)[1:]) == (3, resolution, resolution)

    def test_the_sidecar_outputs_are_the_engines_outputs_in_order_and_float32(self, trt_sidecar_engine: Any) -> None:
        """Names and order match the engine's outputs, and every tensor is FP32, also for an FP16 build."""
        import tensorrt as trt

        engine = trt_sidecar_engine.engine
        (engine_input,), engine_outputs = self._tensor_names(engine)

        assert [output["name"] for output in trt_sidecar_engine.sidecar["outputs"]] == engine_outputs
        assert {engine.get_tensor_dtype(name) for name in (engine_input, *engine_outputs)} == {trt.float32}

    def test_the_sidecar_batch_is_the_engines_batch(self, trt_sidecar_engine: Any) -> None:
        """A dynamic engine's min/opt/max equal its optimization profile; a static one's size is its batch axis."""
        sidecar, engine = trt_sidecar_engine.sidecar, trt_sidecar_engine.engine
        (engine_input,), _ = self._tensor_names(engine)

        if trt_sidecar_engine.dynamic_batch:
            profile = engine.get_tensor_profile_shape(engine_input, 0)
            assert sidecar["batch"] == {"dynamic": True, "min": 1, "opt": 2, "max": 4}
            assert [shape[0] for shape in profile] == [1, 2, 4]
        else:
            assert sidecar["batch"] == {"dynamic": False, "size": 2}
            assert engine.get_tensor_shape(engine_input)[0] == 2

    def test_the_sidecar_identifies_the_engine_file(self, trt_sidecar_engine: Any) -> None:
        """The recorded size and SHA-256 are those of the ``.trt`` the export returned, so a consumer's check passes."""
        engine_bytes = trt_sidecar_engine.engine_path.read_bytes()

        assert trt_sidecar_engine.sidecar["engine"] == {
            "size": len(engine_bytes),
            "sha256": hashlib.sha256(engine_bytes).hexdigest(),
        }

    def test_the_sidecar_records_the_build_that_ran(self, trt_sidecar_engine: Any) -> None:
        """The precision requested, the TensorRT version and the GPU the test itself sees.

        The last two are read from the same sources the exporter reads, so they check the plumbing, not TensorRT. The
        precision equals the request because a full TensorRT wheel never takes the lean-wheel fallback.
        """
        import tensorrt as trt

        build = trt_sidecar_engine.sidecar["build"]
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())

        assert build["precision"] == ("fp16" if trt_sidecar_engine.fp16 else "fp32")
        assert build["tensorrt_version"] == trt.__version__
        assert build["gpu"] == {"name": properties.name, "compute_capability": f"{properties.major}.{properties.minor}"}

    @pytest.fixture(scope="class")
    def trt_fp16_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
        """Export RFDETRNano to ONNX, build an FP16 ``.trt`` engine, and reuse it across the parity checks.

        Mirrors ``trt_engine`` with ``fp16=True``: on a strongly typed TensorRT that routes through the whole-graph FP16
        cast, which is the path with no numerical evidence behind it otherwise.
        """
        from rfdetr import RFDETRNano

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt_fp16")
        detector = RFDETRNano(pretrain_weights=None)
        onnx_path = detector.export(output_dir=str(out_dir), format="onnx", verbose=False)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, verbose=False))
        engine_path = exporter.build_engine(str(onnx_path))

        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        return model, example, Path(engine_path)

    def test_fp16_engine_file_written(self, trt_fp16_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """An FP16 request must produce an engine named ``_fp16``, not a silently downgraded ``_fp32``."""
        _, _, engine_path = trt_fp16_engine

        assert engine_path.stem.endswith("_fp16")

    def test_fp16_runtime_output_matches_pytorch(
        self, trt_fp16_engine: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """The FP16 engine's outputs must stay within the FP16 bound of eager PyTorch.

        Scenario: the FP16 path casts the whole graph, so every non-block-listed op runs in FP16 with no
        per-layer FP32 fallback. This is the only check in the repo that evaluates a number from an FP16
        engine — engine file size proves weight storage, not detections. Values are compared sorted
        rather than positionally: the graph ends in ``TopK`` query selection, and on this randomly
        initialised fixture the logits sit in a narrow band where FP16 rounding can reorder near-tied
        queries. A positional diff would then explode to O(1) on a numerically healthy engine, so the
        comparison is deliberately order-insensitive. NaN or Inf still fails, because neither compares
        less than the bound.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, example, engine_path = trt_fp16_engine
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.asarray(outputs[name], dtype=np.float32)) for name in output_names]
        sorted_eager = [torch.sort(tensor.flatten()).values for tensor in eager_tensors]
        sorted_trt = [torch.sort(tensor.flatten()).values for tensor in trt_tensors]

        diffs = max_abs_output_diffs(sorted_eager, sorted_trt, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_FP16_MAX_ABS_DIFF, (
            f"FP16 TensorRT outputs diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_FP16_MAX_ABS_DIFF})"
        )

    @pytest.fixture(scope="class")
    def trt_dynamic_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, int, Path]:
        """Export RFDETRNano with a dynamic batch axis and build one FP32 engine spanning batch 1 through 4.

        Built through ``RFDETR.export(format="tensorrt", ...)`` rather than ``build_engine`` directly, so the
        ``batch_size`` / ``max_batch_size`` keywords are exercised end to end.
        """
        from rfdetr import RFDETRNano

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt_dynamic")
        detector = RFDETRNano(pretrain_weights=None)
        engine_path = detector.export(
            output_dir=str(out_dir),
            format="tensorrt",
            fp16=False,
            dynamic_batch=True,
            batch_size=2,
            max_batch_size=4,
            verbose=False,
        )

        model = detector.model.model.to("cpu").eval()
        model.export()
        return model, int(detector.model.resolution), Path(engine_path)

    @pytest.mark.parametrize("batch", [1, 3, 4])
    def test_dynamic_engine_matches_pytorch_at_each_batch(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], batch: int
    ) -> None:
        """One engine must serve every batch inside its profile, each image matching eager PyTorch.

        The batch repeats the same structured image the static parity tests use: on this randomly initialised
        fixture the two-stage top-k sits on near-ties for other inputs, where a rank swap turns a healthy engine
        into an O(1) positional diff (see ``test_fp16_runtime_output_matches_pytorch``). Whether distinct images
        stay independent inside a batch is ``test_dynamic_engine_batch_positions_are_independent``'s job.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, resolution, engine_path = trt_dynamic_engine
        example = _structured_parity_input(batch, 3, resolution, resolution)
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.array(outputs[name], dtype=np.float32)) for name in output_names]

        diffs = max_abs_output_diffs(eager_tensors, trt_tensors, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_MAX_ABS_DIFF, (
            f"dynamic TensorRT outputs at batch {batch} diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_MAX_ABS_DIFF})"
        )

    def test_dynamic_engine_batch_positions_are_independent(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """Four distinct images run as one batch must each equal the same image run alone through the same engine.

        ``TrtRunner.infer`` reuses its host output buffers between calls, so every result is copied out before the next
        call.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        _, resolution, engine_path = trt_dynamic_engine
        example = np.ascontiguousarray(_distinct_batch(4, resolution).numpy())
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            batched = {name: np.array(value) for name, value in runner.infer(feed_dict={"input": example}).items()}
            alone = [
                {name: np.array(value) for name, value in runner.infer(feed_dict={"input": example[i : i + 1]}).items()}
                for i in range(4)
            ]

        for name in ("dets", "labels"):
            for index in range(4):
                diff = float(np.abs(batched[name][index] - alone[index][name][0]).max())
                assert diff < 1e-4, f"{name} for image {index} differs between batch 4 and batch 1: {diff}"

    def test_dynamic_engine_rejects_a_batch_beyond_the_profile(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A batch above ``max_batch_size`` is outside the profile and must not silently run.

        Polygraphy's ``TrtRunner.infer`` reports an out-of-profile shape by having ``G_LOGGER.critical`` raise a
        ``PolygraphyException`` naming the failed ``set_input_shape`` call -- narrower than a bare ``Exception``, which
        would also swallow an unrelated crash (OOM, a driver error) as a false pass.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        from polygraphy.exception import PolygraphyException

        _, resolution, engine_path = trt_dynamic_engine
        feed = {"input": np.ascontiguousarray(_distinct_batch(5, resolution).numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner, pytest.raises(PolygraphyException, match="failed to set shape"):
            runner.infer(feed_dict=feed)

    @staticmethod
    def _polygraphy_outputs(engine_path: str | Path, example: torch.Tensor) -> dict:
        """Run one engine through Polygraphy on *example* and return its outputs as NumPy arrays.

        Polygraphy's loader always allows host code, so this also opens a version-compatible engine.

        Args:
            engine_path: The serialized engine.
            example: The input batch, on any device.

        Returns:
            Each output's name mapped to its values.

        Examples:
            Needs a real engine on a GPU, so this is documentation only (not a doctest):

            >>> TestTensorRTEndToEnd._polygraphy_outputs("model.trt", example)  # doctest: +SKIP
            {'dets': array(...), 'labels': array(...)}
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        with TrtRunner(EngineFromBytes(BytesFromPath(str(engine_path)))) as runner:
            return {name: np.array(value) for name, value in runner.infer(feed_dict=feed).items()}

    @pytest.mark.parametrize("level", ["ampere_plus", "same_compute_capability"])
    def test_hardware_compatible_engine_records_its_level_and_matches_the_default_engine(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], level: str
    ) -> None:
        """A hardware-compatible engine is tagged with the requested level and computes what the default one does.

        ``AMPERE_PLUS`` needs compute capability 8.0 or newer, so it always skips on a T4 (7.5). That skip is the only
        one the TensorRT CI job whitelists. Loading an engine on a different GPU or TensorRT release than built it is
        not CI-verified: this test builds and loads on one device with one release.
        """
        import numpy as np
        import tensorrt as trt

        if not hasattr(trt.HardwareCompatibilityLevel, level.upper()):
            pytest.skip(f"this TensorRT has no HardwareCompatibilityLevel.{level.upper()}")
        if level == "ampere_plus" and torch.cuda.get_device_capability() < (8, 0):
            pytest.skip("AMPERE_PLUS engines need compute capability 8.0 or newer")
        _, example, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        config = TensorRTConfig(fp16=False, verbose=False, hardware_compatibility=level)

        portable_path = TensorRTExporter(config).build_engine(str(onnx_path), output_name=f"hardware-{level}")

        with open(portable_path, "rb") as engine_file:
            engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(engine_file.read())
        assert engine.hardware_compatibility_level == getattr(trt.HardwareCompatibilityLevel, level.upper())
        default, portable = (self._polygraphy_outputs(path, example) for path in (engine_path, portable_path))
        diffs = {name: float(np.abs(default[name] - portable[name]).max()) for name in ("dets", "labels")}
        assert max(diffs.values()) < _TENSORRT_MAX_ABS_DIFF, (
            f"{level} engine differs from the default engine's: {diffs}"
        )

    @pytest.fixture(scope="class")
    def trt_version_compatible_engine(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> Path:
        """Build a version-compatible engine from the FP32 engine's ONNX, or skip when TensorRT has no lean runtime.

        The lean runtime is a separate package (``tensorrt-lean-cu*-libs``) that the ``rfdetr[tensorrt]`` extra does not
        install, so this is skipped wherever it is missing. CI's TensorRT job installs it and fails if it does not load.
        """
        _, _, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False, version_compatible=True))
        try:
            exporter.check_environment()
        except ImportError as error:
            pytest.skip(str(error))
        return Path(exporter.build_engine(str(onnx_path), output_name="version-compatible"))

    def test_a_version_compatible_engine_is_refused_by_trt_inference_without_host_code_on_tensorrt_11(
        self, trt_version_compatible_engine: Path
    ) -> None:
        """TensorRT 11 will not deserialize the host code such an engine carries unless the caller says it is trusted.

        TensorRT 10.16 loads its own version-compatible engine without the opt-in, so the refusal is asserted only where
        it exists.
        """
        import tensorrt as trt

        if int(trt.__version__.split(".")[0]) < 11:
            pytest.skip("TensorRT 10 loads a version-compatible engine without the host-code opt-in")

        with pytest.raises(RuntimeError, match="engine_host_code_allowed=True"):
            tensorrt_inference.TRTInference(str(trt_version_compatible_engine), device="cuda:0", sync_mode=True)

    def test_a_version_compatible_engine_computes_what_the_default_engine_does_once_host_code_is_allowed(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], trt_version_compatible_engine: Path
    ) -> None:
        """With the opt-in ``TRTInference`` runs it, and its outputs agree with the default engine's."""
        import numpy as np

        _, example, engine_path = trt_engine
        runtime = tensorrt_inference.TRTInference(
            str(trt_version_compatible_engine), device="cuda:0", sync_mode=True, engine_host_code_allowed=True
        )
        outputs = runtime({"input": example.to("cuda:0")})

        default = self._polygraphy_outputs(engine_path, example)
        diffs = {
            name: float(np.abs(outputs[name].detach().float().cpu().numpy() - default[name]).max())
            for name in ("dets", "labels")
        }
        assert max(diffs.values()) < _TENSORRT_MAX_ABS_DIFF, (
            f"version-compatible engine differs from the default's: {diffs}"
        )

    @pytest.mark.parametrize("device", ["cuda:0", "cuda"])
    def test_trt_inference_helper_serves_the_dynamic_engine(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], device: str
    ) -> None:
        """``TRTInference`` allocates at the profile's max batch, trims outputs to the batch run, and agrees with
        polygraphy.

        A bare ``"cuda"`` resolves to the current device, so its ``cuda:0`` input must be accepted as the engine's own.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        _, resolution, engine_path = trt_dynamic_engine
        example = _distinct_batch(3, resolution)
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            reference = {
                name: np.array(value)
                for name, value in runner.infer(feed_dict={"input": np.ascontiguousarray(example.numpy())}).items()
            }

        runtime = tensorrt_inference.TRTInference(str(engine_path), device=device, sync_mode=True)
        assert runtime.bindings["input"].shape[0] == 4
        outputs = runtime({"input": example.to("cuda:0")})

        for name in ("dets", "labels"):
            got = outputs[name].detach().float().cpu().numpy()
            assert got.shape == reference[name].shape
            diff = float(np.abs(got - reference[name]).max())
            assert diff < 1e-4, f"TRTInference {name} differs from polygraphy on the same engine: {diff}"

    def test_public_prediction_waits_for_non_default_input_stream(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stream-free TensorRT launch reads normalized input only after its torch stream finishes."""
        from supervision import Detections

        from rfdetr import RFDETRInference

        _, resolution, engine_path = trt_dynamic_engine
        image = np.random.default_rng(42).integers(0, 256, (resolution, resolution, 3), dtype=np.uint8)
        model = RFDETRInference(engine_path, device="cuda:0")
        baseline = model.predict(image, threshold=0.0, include_source_image=False)
        assert isinstance(baseline, Detections)
        assert baseline.confidence is not None and baseline.class_id is not None
        baseline_boxes = baseline.xyxy.copy()
        baseline_scores = baseline.confidence.copy()
        baseline_classes = baseline.class_id.copy()

        producer = torch.cuda.Stream(device="cuda:0")
        warm_image = image.copy()
        warm_image[0, 0, 0] ^= 1
        with torch.cuda.stream(producer):
            model.predict(warm_image, threshold=0.0, include_source_image=False)
        release = torch.cuda.Stream(device="cuda:0")
        gate = torch.cuda.Event()
        run_sync = tensorrt_inference._run_tensorrt_sync
        engine_input_ready: list[bool] = []

        def monitored_run(
            self: tensorrt_inference._TensorRTSession, blob: Mapping[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            """Record whether input preparation finished before TensorRT executes.

            Examples:
                Requires a live CUDA TensorRT engine.
                >>> monitored_run({})  # doctest: +SKIP
            """
            engine_input_ready.append(producer.query())
            return run_sync(self, blob)

        monkeypatch.setattr(tensorrt_inference, "_run_tensorrt_sync", monitored_run)
        with torch.cuda.stream(release):
            torch.cuda._sleep(1_000_000_000)
            gate.record()
        with torch.cuda.stream(producer):
            producer.wait_event(gate)
            assert not producer.query(), "The producer stream must still be waiting before prediction."
            actual = model.predict(image, threshold=0.0, include_source_image=False)

        assert engine_input_ready == [True]

        assert isinstance(actual, Detections)
        assert actual.confidence is not None and actual.class_id is not None
        np.testing.assert_array_equal(actual.class_id, baseline_classes)
        np.testing.assert_allclose(actual.xyxy, baseline_boxes, atol=1e-4, rtol=0)
        np.testing.assert_allclose(actual.confidence, baseline_scores, atol=1e-6, rtol=0)

        later = model.predict(image, threshold=0.0, include_source_image=False)
        assert isinstance(later, Detections)
        np.testing.assert_allclose(later.xyxy, baseline_boxes, atol=1e-4, rtol=0)
        np.testing.assert_array_equal(baseline.xyxy, baseline_boxes)

    def test_trt_inference_helper_refuses_a_batch_beyond_the_profile(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A real context returns ``False`` from ``set_input_shape`` for batch 5; the helper must raise, not run."""
        _, resolution, engine_path = trt_dynamic_engine
        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)

        with pytest.raises(ValueError, match="outside the engine's optimization profile"):
            runtime({"input": _distinct_batch(5, resolution).to("cuda:0")})

    @_requires_two_cuda_devices
    def test_trt_inference_runs_on_a_non_default_device(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A runtime constructed for ``cuda:1`` gives the same output as one on ``cuda:0``, not a device no-op.

        Every other device test in this suite runs on a single-GPU host and only asserts which device the
        constructor *asked* torch to make current -- a hardware no-op would pass those too. This is the one
        real-placement check: it needs a second GPU, so it is skipped everywhere except a multi-GPU runner.
        """
        _, resolution, engine_path = trt_dynamic_engine
        example = _distinct_batch(1, resolution)

        reference = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        reference_out = reference({"input": example.to("cuda:0")})

        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:1", sync_mode=True)
        assert runtime.engine_device == torch.device("cuda", 1)
        outputs = runtime({"input": example.to("cuda:1")})

        for name in ("dets", "labels"):
            got = outputs[name].detach().float().cpu().numpy()
            expected = reference_out[name].detach().float().cpu().numpy()
            diff = float(np.abs(got - expected).max())
            assert diff < 1e-4, f"TRTInference on cuda:1 {name} differs from cuda:0: {diff}"

    def test_trt_inference_refuses_a_channels_last_input(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A ``channels_last`` input is refused before launch rather than read as dense NCHW memory.

        Bound by pointer, it took pretrained RF-DETR Nano from 13 detections above 0.5 to none on a real image.
        """
        _, resolution, engine_path = trt_dynamic_engine
        runtime = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        example = _distinct_batch(2, resolution).to("cuda:0", memory_format=torch.channels_last)

        with pytest.raises(ValueError, match="not contiguous"):
            runtime({"input": example})

    def test_trt_inference_reports_a_truncated_engine(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], tmp_path: Path
    ) -> None:
        """Real TensorRT returns ``None`` for a truncated engine rather than raising; the runtime reports it as such."""
        _, _, engine_path = trt_dynamic_engine
        truncated = tmp_path / "truncated.trt"
        with open(engine_path, "rb") as engine_file:
            truncated.write_bytes(engine_file.read(4096))

        with pytest.raises(RuntimeError, match="Rebuild"):
            tensorrt_inference.TRTInference(str(truncated), device="cuda:0", sync_mode=True)

    def test_benchmark_build_engine_keeps_the_previous_engine_on_a_failed_build(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path], tmp_path: Path
    ) -> None:
        """Real TensorRT refuses a dynamic-batch ONNX in ``TRTInference.build_engine``, which declares no profile.

        It reports that by returning ``None``; the engine already at the target path must survive the failure.
        """
        import tensorrt as trt

        _, _, engine_path = trt_dynamic_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        target = tmp_path / "benchmark.trt"
        target.write_bytes(b"engine from an earlier build")
        runtime_stand_in = types.SimpleNamespace(logger=trt.Logger(trt.Logger.ERROR))

        with pytest.raises(RuntimeError, match="could not build an engine"):
            tensorrt_inference.TRTInference.build_engine(runtime_stand_in, str(onnx_path), str(target))

        assert target.read_bytes() == b"engine from an earlier build"

    @pytest.mark.parametrize("engine_fixture", ["trt_engine", "trt_fp16_engine"])
    def test_cuda_graph_matches_the_plain_call_on_new_inputs(
        self, request: pytest.FixtureRequest, engine_fixture: str
    ) -> None:
        """A replayed graph returns what the plain call returns, on tensors it was not captured on, in FP32 and FP16.

        The first call captures; the later ones replay on other pointers and values and end on the first input again, so
        a graph that had baked in the captured tensor instead of copying each input would fail. The FP16 engine is
        compared with its own ``sync_mode=True`` call, on inputs in the dtype that engine reads.
        """
        _, example, engine_path = request.getfixturevalue(engine_fixture)
        plain = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        graphed = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", cuda_graph=True)
        images = _distinct_batch(3, example.shape[-1]).to("cuda:0", plain._input_dtypes["input"])

        matches = []
        for index in (0, 1, 2, 0):
            image = images[index : index + 1].contiguous()
            expected = {name: tensor.clone() for name, tensor in plain({"input": image}).items()}
            got = graphed({"input": image})
            matches.append(
                got.keys() == expected.keys()
                and all(torch.allclose(got[name], expected[name], rtol=1e-3, atol=1e-3) for name in expected)
            )

        assert matches == [True, True, True, True]

    def test_cuda_graph_waits_for_an_input_still_in_flight_on_the_callers_stream(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """An input a side stream is still producing reaches the graph complete, in each of 20 rounds.

        The runtime is called inside ``torch.cuda.stream(side)`` right after a long sleep kernel and the copy that
        produces the input were enqueued there, so the graph stream must wait on the caller's current stream, not the
        default one: copying the input early would read rows left over from an earlier round. The capturing call, which
        synchronizes the device, runs before the rounds so every round replays.
        """
        _, example, engine_path = trt_engine
        images = _distinct_batch(3, example.shape[-1]).cuda()
        plain = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", sync_mode=True)
        graphed = tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", cuda_graph=True)
        graphed({"input": images[:1].contiguous()})
        side = torch.cuda.Stream(device="cuda:0")
        in_flight_cycles = 50_000_000  # tens of milliseconds of GPU clock: far longer than the host takes to call

        matches = []
        for round_index in range(20):
            source = images[round_index % 3 : round_index % 3 + 1]
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                produced = torch.empty_like(source)
                torch.cuda._sleep(in_flight_cycles)
                produced.copy_(source)
                got = {name: tensor.clone() for name, tensor in graphed({"input": produced}).items()}
            torch.cuda.synchronize()
            expected = plain({"input": produced})
            matches.append(all(torch.allclose(got[name], expected[name], rtol=1e-3, atol=1e-3) for name in expected))

        assert matches == [True] * 20

    def test_cuda_graph_is_refused_for_the_dynamic_batch_engine(
        self, trt_dynamic_engine: tuple[torch.nn.Module, int, Path]
    ) -> None:
        """A real ``dynamic_batch=True`` engine is refused for graph replay at construction, with the sync-mode advice.

        Graphs captured at different batches would share one execution context whose input shapes each call changes
        between replays, which TensorRT leaves undefined, so the real engine's profile must be read as varying.
        """
        _, _, engine_path = trt_dynamic_engine

        with pytest.raises(ValueError, match="sync_mode=True instead of cuda_graph=True"):
            tensorrt_inference.TRTInference(str(engine_path), device="cuda:0", cuda_graph=True)

    def test_warm_timing_cache_builds_an_equivalent_engine(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first build writes the timing cache, and a build that reuses it produces the cold build's outputs.

        Reusing measured timings changes which kernels a build may skip timing, never what the network computes, so on
        an FP32 engine the two engines must agree to numerical noise.
        """
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        _, example, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        cache = tmp_path / "engine.cache"
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False, timing_cache=cache))
        # The real ``CreateConfig`` receives the keywords, so Polygraphy itself vouches for their names.
        real_create_config = tensorrt_export.CreateConfig
        create_config_calls: list[dict[str, object]] = []

        def recording_create_config(**kwargs: object) -> object:
            """Record the keywords and build the real configuration."""
            create_config_calls.append(kwargs)
            return real_create_config(**kwargs)

        monkeypatch.setattr(tensorrt_export, "CreateConfig", recording_create_config)

        cold_path = exporter.build_engine(str(onnx_path), output_name="cold-with-cache")
        assert cache.is_file() and cache.stat().st_size > 0, "the first build must write the timing cache"
        warm_path = exporter.build_engine(str(onnx_path), output_name="warm-with-cache")
        assert ["load_timing_cache" in call for call in create_config_calls] == [False, True]
        assert create_config_calls[1]["load_timing_cache"] == str(cache)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        outputs = []
        for path in (cold_path, warm_path):
            with TrtRunner(EngineFromBytes(BytesFromPath(path))) as runner:
                outputs.append({name: np.array(value) for name, value in runner.infer(feed_dict=feed).items()})
        diffs = {name: float(np.abs(outputs[0][name] - outputs[1][name]).max()) for name in ("dets", "labels")}
        assert max(diffs.values()) < 1e-4, f"the warm-cache engine differs from the cold build's: {diffs}"

    @pytest.fixture(scope="class")
    def populated_timing_cache(
        self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path], tmp_path_factory: pytest.TempPathFactory
    ) -> bytes:
        """The timing cache a real build writes, built once for the tests that damage it."""
        _, _, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        cache = tmp_path_factory.mktemp("timing-cache") / "engine.cache"
        config = TensorRTConfig(fp16=False, verbose=False, timing_cache=cache)

        TensorRTExporter(config).build_engine(str(onnx_path), output_name="populate-timing-cache")

        return cache.read_bytes()

    @pytest.mark.parametrize("damage", ["empty", "garbage", "truncated"])
    def test_a_damaged_timing_cache_is_replaced_and_does_not_fail_the_build(
        self,
        trt_engine: tuple[torch.nn.Module, torch.Tensor, Path],
        populated_timing_cache: bytes,
        tmp_path: Path,
        damage: str,
    ) -> None:
        """A cache TensorRT cannot read does not stop the build, and the build writes its own timings over it.

        The documented behaviour of ``trt_timing_cache`` for an empty, a garbage and a truncated file (a real cache cut
        in half). TensorRT logs a serialization error for the last two; the export still succeeds.
        """
        _, _, engine_path = trt_engine
        onnx_path = engine_path.with_name(engine_path.stem.removesuffix("_fp32") + ".onnx")
        damaged = {
            "empty": b"",
            "garbage": b"this is not a timing cache" * 40,
            "truncated": populated_timing_cache[: len(populated_timing_cache) // 2],
        }[damage]
        cache = tmp_path / "engine.cache"
        cache.write_bytes(damaged)

        built = TensorRTExporter(TensorRTConfig(fp16=False, verbose=False, timing_cache=cache)).build_engine(
            str(onnx_path), output_name=f"after-a-{damage}-cache"
        )

        assert Path(built).stat().st_size > 0
        assert len(cache.read_bytes()) > len(damaged), "the build must write a fuller timing cache over the damaged one"


def _people_photo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Download supervision's PEOPLE_WALKING photo; return a directory of its crops and flips, and the photo itself.

    Examples:
        Needs a network download, so it is documentation rather than a doctest:

        >>> calibration_dir, photo_path = _people_photo(tmp_path_factory)  # doctest: +SKIP
    """
    from PIL import Image, ImageOps
    from supervision.assets import ImageAssets, download_assets

    asset_dir = tmp_path_factory.mktemp("int8_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        photo = Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)
    calibration = tmp_path_factory.mktemp("int8_calibration")
    with Image.open(photo) as image:
        image = image.convert("RGB")
        width, height = image.size
        for index, box in enumerate(
            [(0, 0, width, height), (0, 0, width // 2, height), (width // 2, 0, width, height)]
        ):
            crop = image.crop(box)
            crop.save(calibration / f"{index}.jpg")
            ImageOps.mirror(crop).save(calibration / f"{index}_flipped.jpg")
    return calibration, photo


def _photo_input(photo: Path, resolution: int) -> np.ndarray:
    """Preprocess *photo* to a ``(1, 3, resolution, resolution)`` model input the way ``predict()`` does.

    Examples:
        >>> import tempfile
        >>> from PIL import Image
        >>> with tempfile.TemporaryDirectory() as root:
        ...     Image.new("RGB", (20, 10)).save(Path(root) / "p.jpg")
        ...     _photo_input(Path(root) / "p.jpg", 8).shape
        (1, 3, 8, 8)
    """
    from PIL import Image

    from rfdetr.export._runtime.preprocess import preprocess_to_nchw

    with Image.open(photo) as image:
        return preprocess_to_nchw(image.convert("RGB"), height=resolution, width=resolution)


#: Score above which an FP16 detection of the PEOPLE_WALKING photo by RFDETRNano counts as confident. One person there
#: scores 0.498, 0.501 and 0.503 in FP16 engines of batch 1, 2 and 4, and INT8 boxes it with an IoU of only 0.82 to 0.84
#: at every batch size (NVIDIA L4, TensorRT 11.4), so a cutoff of 0.5 checked that person at batch 2 and 4 alone. Above
#: 0.55 the lowest FP16 score is 0.57, and eleven detections remain.
_DETECTION_CONFIDENCE_THRESHOLD = 0.55


def _assert_confident_detections_match(
    fp16_engine: Path,
    int8_engine: Path,
    example: np.ndarray,
    batch: int,
    *,
    threshold: float = _DETECTION_CONFIDENCE_THRESHOLD,
) -> None:
    """Run both engines on *batch* copies of *example*; every confident FP16 detection needs an INT8 match.

    A match is an INT8 box with IoU above 0.85 and a score within 0.1 of the FP16 one. An FP16 detection is confident
    above *threshold*, and at least three must be for the check to mean anything. The INT8 engine must also give
    every row of the batch the same boxes for its confident queries.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(np.repeat(example, batch, axis=0)).cuda()}
    results = []
    for path in (fp16_engine, int8_engine):
        outputs = TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs)
        results.append((outputs["dets"].float().cpu(), outputs["labels"].float().sigmoid().max(dim=-1).values.cpu()))
    (fp16_boxes, fp16_scores), (int8_boxes, int8_scores) = results
    assert torch.isfinite(int8_boxes).all() and torch.isfinite(int8_scores).all()
    # Rows are compared on the queries the INT8 engine is confident about: the boxes of near-zero-score queries are
    # unconstrained, and a different INT8 tactic per batch row moves them by more than any useful tolerance.
    confident_int8 = int8_scores[0] > 0.5
    assert torch.allclose(
        int8_boxes[:, confident_int8], int8_boxes[:1, confident_int8].expand(batch, -1, -1), atol=2e-2
    ), "rows of one batch differ"
    for row in range(batch):
        confident = fp16_scores[row] > threshold
        assert int(confident.sum()) >= 3, "the photo must give the FP16 engine several confident people"
        # (confident FP16 boxes, all INT8 boxes); several INT8 queries can cover one person, so a match is the closest
        # score among the INT8 boxes that overlap, not the score of the single best-overlapping box.
        overlaps = box_iou(box_cxcywh_to_xyxy(fp16_boxes[row][confident]), box_cxcywh_to_xyxy(int8_boxes[row]))[0]
        gaps = (int8_scores[row][None, :] - fp16_scores[row][confident][:, None]).abs()
        gaps = gaps.masked_fill(overlaps <= 0.85, float("inf")).min(dim=1).values
        unmatched = [
            (round(float(o), 3), round(float(g), 3)) for o, g in zip(overlaps.max(dim=1).values, gaps) if g >= 0.1
        ]
        assert unmatched == [], (
            f"row {row}: confident FP16 detections without an INT8 match (best IoU, score gap): {unmatched}"
        )


#: Smallest mask IoU between an FP16 and an INT8 mask for a confident detection, and largest keypoint offset (as a
#: fraction of the image side) between them. Measured on RFDETRSegNano and RFDETRKeypointPreview (RTX 5070, TensorRT
#: 11.4): the lowest mask IoU was 0.90 and the largest offset 0.015; the bounds leave room for another GPU.
_MIN_MASK_IOU, _MAX_KEYPOINT_OFFSET = 0.8, 0.05

#: Score above which an FP16 detection of the PEOPLE_WALKING photo counts as confident, per extra output. The keypoint
#: model scores the photo's people low (0.62, 0.50, 0.48, 0.43 on RTX 5070), so 0.5 would let a rebuild flip one.
_CONFIDENCE_THRESHOLD = {"masks": 0.5, "keypoints": 0.4}


def _assert_heads_agree(fp16_engine: Path, int8_engine: Path, example: np.ndarray, head: str) -> None:
    """Run both engines on *example*; every confident FP16 detection needs an INT8 detection with the same *head*.

    A match is the INT8 query whose box has IoU above 0.85 with the FP16 box and whose score is closest within 0.1.
    For ``head="masks"`` the matched masks (logits above zero) need an IoU of at least :data:`_MIN_MASK_IOU`; for
    ``head="keypoints"`` the person keypoints' normalized x and y may differ by at most :data:`_MAX_KEYPOINT_OFFSET`.
    FP16 detections are confident above ``_CONFIDENCE_THRESHOLD[head]``, and at least three must be confident and
    matched for the check to mean anything.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_heads_agree(fp16_path, int8_path, example, head="masks")  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(example).cuda()}
    fp16, int8 = (
        {
            name: value.float().cpu()
            for name, value in TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs).items()
        }
        for path in (fp16_engine, int8_engine)
    )
    for outputs in (fp16, int8):
        assert torch.isfinite(outputs[head]).all(), f"{head} hold NaN or infinite values"
    fp16_scores = fp16["labels"][0].sigmoid().max(dim=-1).values
    int8_scores = int8["labels"][0].sigmoid().max(dim=-1).values
    confident = torch.nonzero(fp16_scores > _CONFIDENCE_THRESHOLD[head]).flatten()
    overlaps = box_iou(box_cxcywh_to_xyxy(fp16["dets"][0][confident]), box_cxcywh_to_xyxy(int8["dets"][0]))[0]
    gaps = (int8_scores[None, :] - fp16_scores[confident][:, None]).abs().masked_fill(overlaps <= 0.85, float("inf"))
    matched = [
        (int(q), int(m)) for q, m, gap in zip(confident, gaps.argmin(dim=1), gaps.min(dim=1).values) if gap < 0.1
    ]
    assert len(matched) >= 3, f"only {len(matched)} of {len(confident)} confident FP16 detections have an INT8 match"
    for query, other in matched:
        if head == "masks":
            fp16_mask, int8_mask = fp16["masks"][0, query] > 0, int8["masks"][0, other] > 0
            agreement = float((fp16_mask & int8_mask).sum() / (fp16_mask | int8_mask).sum().clamp(min=1))
            assert agreement >= _MIN_MASK_IOU, f"query {query}: mask IoU {agreement:.3f}"
        else:
            offset = float(
                (fp16["keypoints"][0, query, 17:34, :2] - int8["keypoints"][0, other, 17:34, :2]).abs().max()
            )
            assert offset <= _MAX_KEYPOINT_OFFSET, f"query {query}: keypoint offset {offset:.3f}"


@_requires_tensorrt
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained RFDETRNano agree on a photo's confident detections."""

    @pytest.fixture(scope="class")
    def photo(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
        """Crops and flips of supervision's PEOPLE_WALKING photo for calibration, and the photo as a model input.

        Examples:
            A pytest fixture, and it needs a network download.

            >>> calibration_dir, example = photo(tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, image = _people_photo(tmp_path_factory)
        return calibration, _photo_input(image, int(RFDETRNano().model.resolution))

    @pytest.fixture(scope="class")
    def engines(
        self, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray]:
        """Export FP16 and INT8 engines at batch 1, calibrating on the photo's crops and flips.

        Examples:
            A pytest fixture, and it needs a GPU and TensorRT.

            >>> fp16_engine, int8_engine, example = engines(photo, tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp("int8_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        return Path(fp16), Path(int8), example

    def test_engine_is_written_without_leftovers(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        """The INT8 export leaves only the ONNX, the engine and its description."""
        _, int8, _ = engines
        assert int8.name == "rfdetr-nano_int8.trt"
        assert sorted(p.name for p in int8.parent.iterdir()) == [
            "rfdetr-nano.onnx",
            "rfdetr-nano_int8.trt",
            "rfdetr-nano_int8.trt.rfdetr.json",
        ]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        """Every confident FP16 detection has a matching INT8 detection."""
        fp16_path, int8_path, example = engines
        _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)

    @pytest.mark.parametrize("source", ["npy-path", "array"])
    def test_preprocessed_calibration_gives_a_matching_engine(
        self,
        source: str,
        photo: tuple[Path, np.ndarray],
        engines: tuple[Path, Path, np.ndarray],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Calibrating on a preprocessed .npy path or array gives an engine matching FP16."""
        from rfdetr import RFDETRNano
        from rfdetr.export._runtime.calibration import calibration_batches

        calibration, example = photo
        fp16_path, _, _ = engines
        size = example.shape[-1]
        samples = np.concatenate(list(calibration_batches(calibration, height=size, width=size)))
        data: Any = samples
        if source == "npy-path":
            data = tmp_path_factory.mktemp("int8_samples") / "calibration.npy"
            np.save(data, samples)
        int8 = RFDETRNano().export(
            output_dir=str(tmp_path_factory.mktemp(f"int8_{source}")),
            format="tensorrt",
            quantization="int8",
            calibration_data=data,
            verbose=False,
        )
        assert Path(int8).name == "rfdetr-nano_int8.trt"
        _assert_confident_detections_match(fp16_path, Path(int8), example, batch=1)

    @pytest.mark.parametrize("batch", [2, 4])
    def test_static_batch_engine_matches_fp16_on_every_row(
        self, batch: int, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """A static-batch INT8 engine matches FP16 on every row of the batch."""
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp(f"int8_batch_{batch}")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", batch_size=batch, verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            batch_size=batch,
            verbose=False,
        )
        _assert_confident_detections_match(Path(fp16), Path(int8), example, batch=batch)


@_requires_tensorrt
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestSegmentationAndKeypointInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained segmentation and keypoint models agree on a photo."""

    @pytest.fixture(scope="class", params=["RFDETRSegNano", "RFDETRKeypointPreview"])
    def engines(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray, str]:
        """Export FP16 and INT8 engines of the model at batch 1, calibrating on the photo's crops and flips.

        Returns the two engine paths, the photo as that model's input, and the name of its extra output.

        Examples:
            A pytest fixture, and it needs a GPU, TensorRT and a network download.

            >>> fp16_engine, int8_engine, example, head = engines(request, tmp_path_factory)  # doctest: +SKIP
        """
        import rfdetr

        calibration, photo = _people_photo(tmp_path_factory)
        model = getattr(rfdetr, request.param)()
        out_dir = tmp_path_factory.mktemp("int8_head_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        head = "masks" if request.param == "RFDETRSegNano" else "keypoints"
        return Path(fp16), Path(int8), _photo_input(photo, int(model.model.resolution)), head

    def test_engine_keeps_the_extra_output(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """The INT8 engine keeps the mask or keypoint output after dets and labels."""
        from rfdetr.export._tensorrt.inference import TRTInference

        _, int8, _, head = engines
        assert TRTInference(str(int8), sync_mode=True, device="cuda:0").get_output_names() == ["dets", "labels", head]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """Every confident FP16 detection has a matching INT8 detection."""
        fp16, int8, example, head = engines
        _assert_confident_detections_match(fp16, int8, example, batch=1, threshold=_CONFIDENCE_THRESHOLD[head])

    def test_head_matches_fp16(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """The INT8 mask or keypoint head agrees with FP16 on confident detections."""
        fp16, int8, example, head = engines
        _assert_heads_agree(fp16, int8, example, head)
