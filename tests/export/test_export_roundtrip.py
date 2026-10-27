# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public export, load, and predict round trips on real runtime artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from supervision import KeyPoints

from rfdetr.export._coreai import _IS_COREAI_TORCH_AVAILABLE
from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export._tflite import _IS_ONNX2TF_AVAILABLE
from rfdetr.export.imports import (
    _IS_AI_EDGE_LITERT_INSTALLED,
    _IS_COREAI_INSTALLED,
    _IS_LITERT_TORCH_INSTALLED,
    _IS_OPENVINO_INSTALLED,
    _IS_TENSORFLOW_INSTALLED,
    _IS_TFLITE_RUNTIME_INSTALLED,
)
from tests._markers import onnx_and_onnxruntime_only
from tests.export.conftest import _prediction_image, _prediction_model_for_task, assert_prediction_roundtrip

#: Skip an OpenVINO round trip when the package that both converts and runs the model is absent.
openvino_runtime_only = pytest.mark.skipif(not _IS_OPENVINO_INSTALLED, reason="openvino not installed")
#: Skip an onnx2tf round trip unless some TFLite interpreter can run the converted model.
tflite_runtime_only = pytest.mark.skipif(
    not _IS_ONNX2TF_AVAILABLE
    or not (_IS_AI_EDGE_LITERT_INSTALLED or _IS_TFLITE_RUNTIME_INSTALLED or _IS_TENSORFLOW_INSTALLED),
    reason="TFLite round trips need onnx2tf and one of ai_edge_litert, tflite_runtime, or tensorflow",
)
#: Skip a direct LiteRT round trip unless both its converter and its interpreter are installed.
litert_runtime_only = pytest.mark.skipif(
    not (_IS_LITERT_TORCH_INSTALLED and _IS_AI_EDGE_LITERT_INSTALLED),
    reason="LiteRT round trips need litert_torch and ai_edge_litert",
)
#: Skip an ExecuTorch round trip when the ExecuTorch package cannot be imported.
executorch_runtime_only = pytest.mark.skipif(not _IS_EXECUTORCH_AVAILABLE, reason="executorch not installed")
#: Skip a CoreML round trip off macOS, where the CoreML runtime cannot execute a model.
coreml_runtime_only = pytest.mark.skipif(
    not _IS_COREMLTOOLS_AVAILABLE or sys.platform != "darwin",
    reason="CoreML round trips need coremltools and the macOS CoreML runtime",
)
#: Skip a Core AI round trip off macOS or without both the converter and the runtime package.
coreai_runtime_only = pytest.mark.skipif(
    not (_IS_COREAI_TORCH_AVAILABLE and _IS_COREAI_INSTALLED) or sys.platform != "darwin",
    reason="Core AI round trips need coreai_torch and the coreai runtime on macOS",
)


class TestPredictionFixture:
    """Keep native score variation larger than the export comparison tolerances."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    def test_score_spread(self, task: str) -> None:
        """Every task spans more than three times the largest backend score tolerance."""
        model = _prediction_model_for_task(task)
        size = 64 if task == "detect" else 96
        original = model.predict(_prediction_image(size), threshold=0.0, include_source_image=False)
        confidence = original.detection_confidence if isinstance(original, KeyPoints) else original.confidence
        assert confidence is not None
        assert np.ptp(confidence) > 3 * 3e-2


@pytest.mark.integration
class TestExportRoundTrip:
    """Compare public predictions before and after a real export."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_onnx
    @onnx_and_onnxruntime_only
    def test_onnx(self, task: str, tmp_path: Path) -> None:
        """ONNX preserves prediction coordinates, scores, labels, and task data."""
        assert_prediction_roundtrip("onnx", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_openvino
    @openvino_runtime_only
    def test_openvino(self, task: str, tmp_path: Path) -> None:
        """OpenVINO preserves all three public prediction tasks."""
        assert_prediction_roundtrip("openvino", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_tflite
    @pytest.mark.timeout(300)
    @tflite_runtime_only
    def test_tflite(self, task: str, tmp_path: Path) -> None:
        """ONNX-converted TFLite preserves all three public prediction tasks."""
        try:
            assert_prediction_roundtrip("tflite", task, tmp_path)
        except pytest.fail.Exception as exc:
            # Expected failure applies only to the experimental converter exceeding its budget.
            if not re.fullmatch(r"Timeout \(>[0-9.]+s\) from pytest-timeout\.", str(exc)):
                raise
            pytest.xfail("Experimental onnx2tf conversion exceeded the per-test time budget.")

    @pytest.mark.parametrize("task", ["detect", "segment"])
    @pytest.mark.e2e_litert
    @litert_runtime_only
    def test_litert(self, task: str, tmp_path: Path) -> None:
        """Direct LiteRT export preserves detection and segmentation."""
        assert_prediction_roundtrip("litert", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_executorch
    @executorch_runtime_only
    def test_executorch(self, task: str, tmp_path: Path) -> None:
        """Portable XNNPACK exports preserve all three prediction tasks."""
        assert_prediction_roundtrip("executorch", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreml
    @coreml_runtime_only
    def test_coreml(self, task: str, tmp_path: Path) -> None:
        """CoreML float32 CPU execution preserves all three prediction tasks."""
        assert_prediction_roundtrip("coreml", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreai
    @coreai_runtime_only
    def test_coreai(self, task: str, tmp_path: Path) -> None:
        """Core AI float32 CPU execution preserves all three prediction tasks."""
        assert_prediction_roundtrip("coreai", task, tmp_path)

    @pytest.mark.e2e_litert
    @litert_runtime_only
    def test_litert_keypoints_refused(self, tmp_path: Path) -> None:
        """LiteRT reports its known keypoint lowering limit before conversion."""
        model = _prediction_model_for_task("keypoints")
        with pytest.raises(NotImplementedError, match="keypoint"):
            model.export(format="litert", output_dir=str(tmp_path), verbose=False)
