# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public export, load, and predict round trips on real runtime artifacts."""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from supervision import Detections, KeyPoints

from rfdetr import RFDETRInference, RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
from rfdetr.detr import RFDETR


def _model_for_task(task: str) -> RFDETR:
    """Build a small random model with explicit labels and no downloaded weights.

    Examples:
        Constructs a full RF-DETR model and takes several seconds, so the live doctest is skipped.

        >>> _model_for_task("detect").class_names  # doctest: +SKIP
        ['alpha', 'beta']
    """
    kwargs: dict[str, Any] = {
        "pretrain_weights": None,
        "device": "cpu",
        "resolution": 64 if task == "detect" else 96,
        "num_queries": 4,
        "num_select": 4,
        "num_classes": 2,
    }
    if task == "keypoints":
        kwargs["num_keypoints_per_class"] = [3]
        model = RFDETRKeypointPreview(**kwargs)
        model.model.class_names = ["alpha"]
    else:
        model = (RFDETRSegNano if task == "segment" else RFDETRNano)(**kwargs)
        model.model.class_names = ["alpha", "beta"]
    return model


def _image(size: int) -> np.ndarray:
    """Make an RGB image with stable spatial structure.

    Examples:
        >>> _image(4).shape
        (4, 4, 3)
    """
    y, x = np.indices((size, size))
    return np.stack(((x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256), axis=-1).astype(np.uint8)


def _require_runtime(format_name: str) -> None:
    """Skip a runtime that this host cannot import or execute.

    Examples:
        Needs the caller's optional runtime installation, so this check is only used in integration tests.

        >>> _require_runtime("onnx")  # doctest: +SKIP
    """
    requirements = {
        "onnx": ("onnx", "onnxruntime"),
        "openvino": ("openvino",),
        "tflite": ("onnx2tf",),
        "litert": ("litert_torch", "ai_edge_litert"),
        "executorch": ("executorch",),
        "coreml": ("coremltools",),
        "coreai": ("coreai_torch", "coreai"),
        "tensorrt": ("tensorrt",),
    }
    if format_name in {"coreml", "coreai"} and platform.system() != "Darwin":
        pytest.skip(f"{format_name} runtime requires macOS")
    if format_name == "tensorrt" and not torch.cuda.is_available():
        pytest.skip("TensorRT runtime requires CUDA")
    for package in requirements[format_name]:
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{format_name} runtime requires {package}")
    if format_name == "tflite" and not any(
        importlib.util.find_spec(package) is not None for package in ("ai_edge_litert", "tflite_runtime", "tensorflow")
    ):
        pytest.skip("TFLite inference requires ai_edge_litert, tflite_runtime, or tensorflow")


@pytest.mark.integration
class TestExportRoundTrip:
    """Compare public predictions before and after a real export."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_onnx
    def test_onnx(self, task: str, tmp_path: Path) -> None:
        """ONNX preserves prediction coordinates, scores, labels, and task data."""
        self._roundtrip("onnx", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_openvino
    def test_openvino(self, task: str, tmp_path: Path) -> None:
        """OpenVINO preserves all three public prediction tasks."""
        self._roundtrip("openvino", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_tflite
    @pytest.mark.timeout(1200)
    def test_tflite(self, task: str, tmp_path: Path) -> None:
        """ONNX-converted TFLite preserves all three public prediction tasks."""
        self._roundtrip("tflite", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment"])
    @pytest.mark.e2e_litert
    def test_litert(self, task: str, tmp_path: Path) -> None:
        """Direct LiteRT export preserves detection and segmentation."""
        self._roundtrip("litert", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_executorch
    def test_executorch(self, task: str, tmp_path: Path) -> None:
        """Portable XNNPACK exports preserve all three prediction tasks."""
        self._roundtrip("executorch", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreml
    def test_coreml(self, task: str, tmp_path: Path) -> None:
        """CoreML float32 CPU execution preserves all three prediction tasks."""
        self._roundtrip("coreml", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreai
    def test_coreai(self, task: str, tmp_path: Path) -> None:
        """Core AI float32 CPU execution preserves all three prediction tasks."""
        self._roundtrip("coreai", task, tmp_path)

    @pytest.mark.gpu
    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_tensorrt
    def test_tensorrt(self, task: str, tmp_path: Path) -> None:
        """TensorRT float32 execution preserves all three prediction tasks."""
        self._roundtrip("tensorrt", task, tmp_path)

    @pytest.mark.e2e_litert
    def test_litert_keypoints_refused(self, tmp_path: Path) -> None:
        """LiteRT reports its known keypoint lowering limit before conversion."""
        _require_runtime("litert")
        model = _model_for_task("keypoints")
        with pytest.raises(NotImplementedError, match="keypoint"):
            model.export(format="litert", output_dir=str(tmp_path), verbose=False)

    def _roundtrip(self, format_name: str, task: str, tmp_path: Path) -> None:
        """Run the public export and prediction APIs on one image.

        Examples:
            Requires a real export runtime and a temporary artifact directory.

            >>> TestExportRoundTrip()._roundtrip("onnx", "detect", Path("output"))  # doctest: +SKIP
        """
        _require_runtime(format_name)
        torch.manual_seed(17)
        native = _model_for_task(task)
        size = 64 if task == "detect" else 96
        image = _image(size)
        original = native.predict(image, threshold=0.0, include_source_image=False)
        settings: dict[str, Any] = {}
        if format_name == "tensorrt":
            settings["fp16"] = False
        elif format_name == "executorch":
            settings["backend"] = "xnnpack"
        elif format_name == "coreml":
            settings["coreml_precision"] = "float32"
        elif format_name == "coreai":
            settings["coreai_precision"] = "float32"
        elif format_name == "openvino":
            settings["openvino_precision"] = "float32"
        artifact = native.export(format=format_name, output_dir=str(tmp_path), verbose=False, **settings)
        exported = RFDETRInference(artifact, device="cuda:0" if format_name == "tensorrt" else "cpu")
        actual = exported.predict(image, threshold=0.0, include_source_image=False)

        assert exported.class_names == native.class_names
        assert type(actual) is type(original)
        normalized_box_atol = {
            "onnx": 1e-3,
            "openvino": 1e-2,
            "tflite": 2e-2,
            "litert": 1e-2,
            "executorch": 1e-2,
            "coreml": 2e-2,
            "coreai": 2e-2,
            "tensorrt": 2e-2,
        }[format_name]
        score_atol = {
            "onnx": 1e-3,
            "openvino": 2e-2,
            "tflite": 3e-2,
            "litert": 2e-2,
            "executorch": 2e-2,
            "coreml": 3e-2,
            "coreai": 3e-2,
            "tensorrt": 3e-2,
        }[format_name]
        if task == "keypoints":
            assert isinstance(actual, KeyPoints) and isinstance(original, KeyPoints)
            assert actual.detection_confidence is not None and original.detection_confidence is not None
            assert actual.keypoint_confidence is not None and original.keypoint_confidence is not None
            np.testing.assert_allclose(actual.xy, original.xy, atol=normalized_box_atol * size, rtol=0)
            np.testing.assert_allclose(
                actual.detection_confidence, original.detection_confidence, atol=score_atol, rtol=0
            )
            np.testing.assert_allclose(
                actual.keypoint_confidence, original.keypoint_confidence, atol=score_atol, rtol=0
            )
        else:
            assert isinstance(actual, Detections) and isinstance(original, Detections)
            assert actual.confidence is not None and original.confidence is not None
            np.testing.assert_array_equal(actual.class_id, original.class_id)
            np.testing.assert_allclose(actual.xyxy, original.xyxy, atol=normalized_box_atol * size, rtol=0)
            np.testing.assert_allclose(actual.confidence, original.confidence, atol=score_atol, rtol=0)
            if task == "segment":
                assert actual.mask is not None and original.mask is not None
                assert np.mean(actual.mask != original.mask) < 0.02
