# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public export, load, and predict round trips on real runtime artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.export.conftest import (
    _prediction_model_for_task,
    _require_prediction_runtime,
    assert_prediction_roundtrip,
)


@pytest.mark.integration
class TestExportRoundTrip:
    """Compare public predictions before and after a real export."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_onnx
    def test_onnx(self, task: str, tmp_path: Path) -> None:
        """ONNX preserves prediction coordinates, scores, labels, and task data."""
        assert_prediction_roundtrip("onnx", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_openvino
    def test_openvino(self, task: str, tmp_path: Path) -> None:
        """OpenVINO preserves all three public prediction tasks."""
        assert_prediction_roundtrip("openvino", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_tflite
    @pytest.mark.timeout(1200)
    def test_tflite(self, task: str, tmp_path: Path) -> None:
        """ONNX-converted TFLite preserves all three public prediction tasks."""
        assert_prediction_roundtrip("tflite", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment"])
    @pytest.mark.e2e_litert
    def test_litert(self, task: str, tmp_path: Path) -> None:
        """Direct LiteRT export preserves detection and segmentation."""
        assert_prediction_roundtrip("litert", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_executorch
    def test_executorch(self, task: str, tmp_path: Path) -> None:
        """Portable XNNPACK exports preserve all three prediction tasks."""
        assert_prediction_roundtrip("executorch", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreml
    def test_coreml(self, task: str, tmp_path: Path) -> None:
        """CoreML float32 CPU execution preserves all three prediction tasks."""
        assert_prediction_roundtrip("coreml", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreai
    def test_coreai(self, task: str, tmp_path: Path) -> None:
        """Core AI float32 CPU execution preserves all three prediction tasks."""
        assert_prediction_roundtrip("coreai", task, tmp_path)

    @pytest.mark.e2e_litert
    def test_litert_keypoints_refused(self, tmp_path: Path) -> None:
        """LiteRT reports its known keypoint lowering limit before conversion."""
        _require_prediction_runtime("litert")
        model = _prediction_model_for_task("keypoints")
        with pytest.raises(NotImplementedError, match="keypoint"):
            model.export(format="litert", output_dir=str(tmp_path), verbose=False)
