# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the in-process TensorRT engine builder (`build_engine`).

The unit tests monkeypatch the polygraphy entry points so they run without TensorRT, a GPU, or `polygraphy` installed.
The end-to-end class (``@pytest.mark.e2e_tensorrt``, GPU + ``rfdetr[tensorrt]``, opt-in) builds a real engine from an
exported RF-DETR ONNX and checks runtime parity — mirroring the CoreML and ExecuTorch export suites.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from rfdetr.export import _tensorrt as tensorrt_export
from rfdetr.export._tensorrt import _IS_TENSORRT_AVAILABLE
from tests.export.conftest import (
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)

tensorrt_only = pytest.mark.skipif(not _IS_TENSORRT_AVAILABLE, reason="tensorrt not installed")

# A FP32 TensorRT engine still fuses/reorders kernels relative to eager PyTorch, so it diverges more
# than the XNNPACK CPU path (~1e-5). The bound tolerates kernel-level numerical differences while still
# failing on a structural regression (outputs collapse by >=1e-1). Recalibrate once real GPU numbers are
# observed in the tensorrt-parity CI job.
_TENSORRT_MAX_ABS_DIFF = 1e-2


def _patch_polygraphy_chain(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub the polygraphy build chain and return the dict that captures CreateConfig kwargs."""
    config_kwargs: dict = {}
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
    monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config")
    monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: "engine")
    monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
    return config_kwargs


class TestBuildEngineDryRun:
    """Dry-run derives the ``.trt`` path without touching the polygraphy build chain."""

    @pytest.mark.parametrize(
        ("onnx_path", "expected_engine"),
        [
            pytest.param("/output/rfdetr.onnx", "/output/rfdetr_fp16.trt", id="plain-path"),
            pytest.param("/path with spaces/model.onnx", "/path with spaces/model_fp16.trt", id="path-with-spaces"),
            pytest.param("/model;rm -rf /.onnx", "/model;rm -rf /.onnx_fp16.trt", id="shell-metachar"),
            pytest.param(
                "/data/my.onnx.backup/model.onnx",
                "/data/my.onnx.backup/model_fp16.trt",
                id="earlier-onnx-in-dir",
            ),
            pytest.param(
                "/output/model_v1.onnx.old.onnx",
                "/output/model_v1.onnx.old_fp16.trt",
                id="double-onnx-in-filename",
            ),
            pytest.param(
                "/output/model_without_extension",
                "/output/model_without_extension_fp16.trt",
                id="no-onnx-extension",
            ),
        ],
    )
    def test_derives_trt_path(self, onnx_path: str, expected_engine: str) -> None:
        """Only the final suffix is swapped to ``_fp16.trt``; earlier ``.onnx`` segments are never corrupted."""
        result = tensorrt_export.build_engine(onnx_path, dry_run=True)

        assert result == expected_engine

    def test_does_not_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry-run must return the engine path without invoking the polygraphy build chain."""
        called: list[str] = []
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda *a, **k: called.append("built"))

        result = tensorrt_export.build_engine("/tmp/model.onnx", dry_run=True)

        assert result == "/tmp/model_fp16.trt"
        assert not called, "dry_run must not invoke the polygraphy build chain"

    def test_output_name_overrides_and_suppresses_precision_suffix(self) -> None:
        """``output_name`` names the engine verbatim, in the ONNX's directory, with no ``_fp16``/``_fp32`` suffix."""
        result = tensorrt_export.build_engine("/output/rfdetr-medium.onnx", dry_run=True, output_name="my-engine")

        assert result == "/output/my-engine.trt"

    def test_output_name_preserves_windows_directory_separators(self) -> None:
        """A Windows-style ``onnx_path`` keeps its backslash directory prefix verbatim (no ``os.sep`` rewrite)."""
        result = tensorrt_export.build_engine(r"C:\out\m.onnx", dry_run=True, output_name="my-engine")

        assert result == r"C:\out\my-engine.trt"


class TestBuildEngineDependencyGuard:
    """A missing polygraphy/tensorrt install raises an actionable ImportError."""

    def test_missing_polygraphy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing polygraphy/tensorrt install must raise an actionable ImportError."""
        monkeypatch.setattr(tensorrt_export, "engine_from_network", None)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            tensorrt_export.build_engine("/tmp/model.onnx")


class TestBuildEngineWiring:
    """``build_engine`` wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""

    @pytest.mark.parametrize("fp16", [pytest.param(True, id="fp16"), pytest.param(False, id="fp32")])
    def test_invokes_polygraphy_and_saves_trt(self, monkeypatch: pytest.MonkeyPatch, fp16: bool) -> None:
        """build_engine wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""
        config_kwargs: dict = {}
        build_args: dict = {}
        saved: dict = {}

        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
        monkeypatch.setattr(
            tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config-sentinel"
        )

        def _engine_from_network(network, config):
            build_args["network"] = network
            build_args["config"] = config
            return "engine-sentinel"

        def _save_engine(engine, path):
            saved["engine"] = engine
            saved["path"] = path

        monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
        monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)

        result = tensorrt_export.build_engine("/tmp/model.onnx", fp16=fp16)
        expected_path = f"/tmp/model_{'fp16' if fp16 else 'fp32'}.trt"

        assert result == expected_path
        assert config_kwargs == {"fp16": fp16}
        assert build_args == {"network": ("network", "/tmp/model.onnx"), "config": "config-sentinel"}
        assert saved == {"engine": "engine-sentinel", "path": expected_path}


class TestBuildEngineFp16Fallback:
    """A TensorRT build lacking the FP16 builder flag falls back to FP32 instead of aborting."""

    def test_downgrades_to_fp32_when_fp16_flag_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A TensorRT build without the FP16 builder flag must fall back to an FP32 engine instead of aborting."""

        class _FakeTrt:
            __version__ = "11.1.0.106"

            class BuilderFlag:  # deliberately lacks an ``FP16`` attribute
                INT8 = 0

        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _FakeTrt)

        result = tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert result == "/tmp/model_fp32.trt"
        assert config_kwargs == {"fp16": False}


@tensorrt_only
@pytest.mark.gpu
@pytest.mark.e2e_tensorrt
class TestTensorRTEndToEnd:
    """Real ONNX -> TensorRT engine build + runtime parity on GPU (requires ``rfdetr[tensorrt]`` and CUDA)."""

    @pytest.fixture(scope="class")
    def trt_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
        """Export RFDETRNano to ONNX, build a FP32 ``.trt`` engine, and reuse it across the parity checks."""
        from rfdetr import RFDETRNano
        from rfdetr.export._tensorrt import build_engine

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt")
        detector = RFDETRNano(pretrain_weights=None)
        onnx_path = detector.export(output_dir=str(out_dir), format="onnx", verbose=False)
        engine_path = build_engine(str(onnx_path), fp16=False, verbose=False)

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
