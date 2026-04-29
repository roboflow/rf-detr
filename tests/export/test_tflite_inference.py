# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for TFLite inference helpers.

Covers:
* ``_create_interpreter()`` — interpreter loading with tflite_runtime / tensorflow fallback
* ``_run_inference()`` — image preprocessing, invocation, and detection decoding
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import supervision as sv
from PIL import Image as PILImage

from rfdetr.export._tflite.inference import _create_interpreter, _run_inference

# ---------------------------------------------------------------------------
# Shared helpers / factories
# ---------------------------------------------------------------------------

_INPUT_SHAPE = [1, 224, 224, 3]
_DET_OUTPUT = {"shape": [1, 10, 4], "name": "serving_default_dets:0", "index": 1}
_LABEL_OUTPUT = {"shape": [1, 10, 82], "name": "serving_default_labels:0", "index": 2}


def _make_boxes() -> np.ndarray:
    """Return (1, 10, 4) array of normalised cxcywh boxes all centred at 0.5."""
    return np.array([[[0.5, 0.5, 0.1, 0.1]] * 10], dtype=np.float32)


def _make_logits(high_conf_idx: int | None = 0) -> np.ndarray:
    """Return (1, 10, 82) logits with one high-confidence entry when requested."""
    logits = np.zeros((1, 10, 82), dtype=np.float32)
    if high_conf_idx is not None:
        logits[0, high_conf_idx, 0] = 10.0
    return logits


def _make_interp(
    input_shape: list[int] | None = None,
    out_dets: list[dict] | None = None,
    boxes: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> mock.MagicMock:
    """Build a mock TFLite interpreter with configurable I/O details."""
    if input_shape is None:
        input_shape = _INPUT_SHAPE
    out_dets = out_dets if out_dets is not None else [_DET_OUTPUT, _LABEL_OUTPUT]
    if boxes is None:
        boxes = _make_boxes()
    if logits is None:
        logits = _make_logits()

    def _get_tensor(index: int) -> np.ndarray:
        if index == _DET_OUTPUT["index"]:
            return boxes
        if index == _LABEL_OUTPUT["index"]:
            return logits
        raise ValueError(f"Unknown tensor index: {index}")

    interp = mock.MagicMock()
    interp.get_input_details.return_value = [{"shape": input_shape, "index": 0, "dtype": np.float32}]
    interp.get_output_details.return_value = out_dets
    interp.get_tensor.side_effect = _get_tensor
    return interp


def _save_rgb_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Write a small solid-colour RGB JPEG to *path*."""
    PILImage.new("RGB", size, color=(100, 150, 200)).save(path)


def _save_grayscale_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Write a small solid-colour grayscale PNG to *path*."""
    PILImage.new("L", size, color=128).save(path)


# ---------------------------------------------------------------------------
# TestCreateInterpreter
# ---------------------------------------------------------------------------


class TestCreateInterpreter:
    """Tests for ``_create_interpreter()``."""

    @pytest.fixture()
    def _mock_tflite_runtime(self):
        """Inject a fake tflite_runtime.interpreter into sys.modules.

        Python's import machinery resolves ``import tflite_runtime.interpreter``
        by looking up ``sys.modules["tflite_runtime.interpreter"]`` directly.
        We also set the ``interpreter`` attribute on the parent package mock so
        attribute-path resolution is consistent regardless of Python version.
        """
        interp_instance = mock.MagicMock()
        interp_instance.get_input_details.return_value = [{"shape": [1, 640, 640, 3], "dtype": np.float32}]
        interp_instance.get_output_details.return_value = [
            {"shape": [1, 300, 4], "name": "dets"},
            {"shape": [1, 300, 81], "name": "labels"},
        ]
        interp_cls = mock.MagicMock(return_value=interp_instance)

        # Build the submodule with a real Interpreter attribute
        import types

        mod = types.ModuleType("tflite_runtime.interpreter")
        mod.Interpreter = interp_cls  # type: ignore[attr-defined]

        # Build parent package that exposes mod as .interpreter
        parent_mod = types.ModuleType("tflite_runtime")
        parent_mod.interpreter = mod  # type: ignore[attr-defined]

        saved_sub = sys.modules.get("tflite_runtime.interpreter")
        saved_parent = sys.modules.get("tflite_runtime")
        sys.modules["tflite_runtime.interpreter"] = mod  # type: ignore[assignment]
        sys.modules["tflite_runtime"] = parent_mod  # type: ignore[assignment]

        yield interp_cls, interp_instance

        if saved_sub is None:
            sys.modules.pop("tflite_runtime.interpreter", None)
        else:
            sys.modules["tflite_runtime.interpreter"] = saved_sub

        if saved_parent is None:
            sys.modules.pop("tflite_runtime", None)
        else:
            sys.modules["tflite_runtime"] = saved_parent

    def test_uses_tflite_runtime_when_available(self, _mock_tflite_runtime) -> None:
        """Interpreter is constructed from tflite_runtime when it is importable."""
        interp_cls, interp_instance = _mock_tflite_runtime
        _create_interpreter("model.tflite")
        interp_cls.assert_called_once_with(model_path="model.tflite")

    def test_allocate_tensors_called(self, _mock_tflite_runtime) -> None:
        """allocate_tensors() is always called after construction."""
        _, interp_instance = _mock_tflite_runtime
        _create_interpreter("model.tflite")
        interp_instance.allocate_tensors.assert_called_once()

    def test_falls_back_to_tensorflow_when_tflite_runtime_missing(self) -> None:
        """tensorflow.lite.Interpreter is used when tflite_runtime is absent."""
        interp_instance = mock.MagicMock()
        interp_instance.get_input_details.return_value = [{"shape": [1, 640, 640, 3], "dtype": np.float32}]
        interp_instance.get_output_details.return_value = [{"shape": [1, 300, 4], "name": "dets"}]
        tf_interp_cls = mock.MagicMock(return_value=interp_instance)

        tf_lite_mod = mock.MagicMock()
        tf_lite_mod.Interpreter = tf_interp_cls
        tf_mod = mock.MagicMock()
        tf_mod.lite = tf_lite_mod

        with mock.patch.dict(sys.modules, {"tflite_runtime": None, "tflite_runtime.interpreter": None}):
            with mock.patch.dict(sys.modules, {"tensorflow": tf_mod}):
                _create_interpreter("model.tflite")

        tf_interp_cls.assert_called_once_with(model_path="model.tflite")

    def test_returns_interpreter(self, _mock_tflite_runtime) -> None:
        """Return value is the interpreter instance (not the class)."""
        _, interp_instance = _mock_tflite_runtime
        result = _create_interpreter("model.tflite")
        assert result is interp_instance

    def test_prints_input_and_output_shapes(self, _mock_tflite_runtime, capsys) -> None:
        """stdout contains 'Input' and 'Output' lines with shape info."""
        _create_interpreter("model.tflite")
        captured = capsys.readouterr().out
        assert "Input" in captured
        assert "Output" in captured

    def test_accepts_path_object(self, _mock_tflite_runtime) -> None:
        """Path objects are converted to strings before passing to Interpreter."""
        interp_cls, _ = _mock_tflite_runtime
        _create_interpreter(Path("model.tflite"))
        call_kwargs = interp_cls.call_args[1]
        assert call_kwargs["model_path"] == "model.tflite"
        assert isinstance(call_kwargs["model_path"], str)


# ---------------------------------------------------------------------------
# TestRunInference
# ---------------------------------------------------------------------------


class TestRunInference:
    """Tests for ``_run_inference()``."""

    @pytest.fixture()
    def rgb_image(self, tmp_path: Path) -> Path:
        """Write a small RGB JPEG to a temp file and return its path."""
        p = tmp_path / "image.jpg"
        _save_rgb_image(p)
        return p

    @pytest.fixture()
    def grayscale_image(self, tmp_path: Path) -> Path:
        """Write a small grayscale PNG to a temp file and return its path."""
        p = tmp_path / "gray.png"
        _save_grayscale_image(p)
        return p

    def test_returns_detections_and_image(self, rgb_image: Path) -> None:
        """Return type is tuple[sv.Detections, PIL.Image.Image]."""
        interp = _make_interp()
        result = _run_inference(interp, rgb_image)
        assert isinstance(result, tuple)
        dets, img = result
        assert isinstance(dets, sv.Detections)
        assert isinstance(img, PILImage.Image)

    def test_detections_above_threshold_kept(self, rgb_image: Path) -> None:
        """At least one detection is returned when one logit is high-confidence."""
        interp = _make_interp(logits=_make_logits(high_conf_idx=0))
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert len(dets) >= 1

    def test_detections_below_threshold_filtered(self, rgb_image: Path) -> None:
        """No detections survive when all logits are zero (uniform probs < 0.3)."""
        interp = _make_interp(logits=_make_logits(high_conf_idx=None))
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert len(dets) == 0

    def test_boxes_in_pixel_space(self, rgb_image: Path) -> None:
        """xyxy coordinates are scaled to image pixel dimensions, not 0–1 range."""
        img_size = (200, 100)  # (width, height) for PIL
        PILImage.new("RGB", img_size, color=(100, 150, 200)).save(rgb_image)

        # One centred box: cx=0.5, cy=0.5, w=0.2, h=0.2
        boxes = np.array([[[0.5, 0.5, 0.2, 0.2]] + [[0.0, 0.0, 0.0, 0.0]] * 9], dtype=np.float32)
        logits = _make_logits(high_conf_idx=0)
        interp = _make_interp(boxes=boxes, logits=logits)

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        # With cx=0.5*200=100, cy=0.5*100=50, bw=0.2*200=40, bh=0.2*100=20
        # xyxy expected: [80, 40, 120, 60]
        assert dets.xyxy[0, 0] > 1.0  # x1 in pixel coords, not 0–1

    def test_set_tensor_called_with_correct_shape(self, rgb_image: Path) -> None:
        """set_tensor receives a tensor matching (1, H, W, C)."""
        _, H, W, C = _INPUT_SHAPE  # noqa: N806
        interp = _make_interp()
        _run_inference(interp, rgb_image)
        call_args = interp.set_tensor.call_args
        tensor_arg = call_args[0][1]
        assert tensor_arg.shape == (1, H, W, C)

    def test_invoke_called_exactly_once(self, rgb_image: Path) -> None:
        """interp.invoke() is called exactly once per inference call."""
        interp = _make_interp()
        _run_inference(interp, rgb_image)
        interp.invoke.assert_called_once()

    def test_grayscale_image_accepted(self, grayscale_image: Path) -> None:
        """Grayscale (L-mode) input with C=1 is accepted without error."""
        input_shape = [1, 224, 224, 1]
        det_out = {"shape": [1, 10, 4], "name": "serving_default_dets:0", "index": 1}
        label_out = {"shape": [1, 10, 82], "name": "serving_default_labels:0", "index": 2}
        interp = _make_interp(input_shape=input_shape, out_dets=[det_out, label_out])
        dets, _ = _run_inference(interp, grayscale_image)
        assert isinstance(dets, sv.Detections)

    def test_output_lookup_by_name_robust_to_ordering(self, rgb_image: Path) -> None:
        """Swapping dets/labels order in get_output_details returns same detections."""
        logits = _make_logits(high_conf_idx=0)
        boxes = _make_boxes()

        # Canonical order: dets first, labels second
        interp_normal = _make_interp(boxes=boxes, logits=logits)
        dets_normal, _ = _run_inference(interp_normal, rgb_image, threshold=0.3)

        # Swapped order: labels first, dets second
        det_out_swapped = {"shape": [1, 10, 4], "name": "serving_default_dets:0", "index": 1}
        label_out_swapped = {"shape": [1, 10, 82], "name": "serving_default_labels:0", "index": 2}
        interp_swapped = _make_interp(
            out_dets=[label_out_swapped, det_out_swapped],
            boxes=boxes,
            logits=logits,
        )
        dets_swapped, _ = _run_inference(interp_swapped, rgb_image, threshold=0.3)

        assert len(dets_normal) == len(dets_swapped)
