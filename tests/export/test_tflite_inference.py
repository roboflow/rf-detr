# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for TFLite inference helpers.

Covers:
* ``_create_interpreter()`` — interpreter loading with tflite_runtime / tensorflow fallback
* ``_run_inference()`` — image preprocessing, invocation, and detection decoding
* ``_decode_masks()`` — segmentation mask upsampling and thresholding
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import supervision as sv
from PIL import Image as PILImage

from rfdetr.export._resize import _bilinear_resize_half_pixel
from rfdetr.export._tflite.inference import (
    _create_interpreter,
    _decode_masks,
    _preprocess_image,
    _run_inference,
)

# ---------------------------------------------------------------------------
# Shared helpers / factories
# ---------------------------------------------------------------------------

_INPUT_SHAPE = [1, 224, 224, 3]
_DET_OUTPUT = {"shape": [1, 10, 4], "name": "serving_default_dets:0", "index": 1}
_LABEL_OUTPUT = {"shape": [1, 10, 82], "name": "serving_default_labels:0", "index": 2}


def _make_boxes() -> np.ndarray:
    """Return (1, 10, 4) array of normalised cxcywh boxes all centred at 0.5.

    Examples:
        >>> boxes = _make_boxes()
        >>> boxes.shape
        (1, 10, 4)
        >>> float(boxes[0, 0, 0])
        0.5
    """
    return np.array([[[0.5, 0.5, 0.1, 0.1]] * 10], dtype=np.float32)


def _make_logits(high_conf_idx: int | None = 0) -> np.ndarray:
    """Return (1, 10, 82) logits with one high-confidence entry when requested.

    Low-confidence fill is -10.0 so sigmoid scores are near zero (~0.0001) for all entries except the explicitly boosted
    one (logit=+10.0, sigmoid≈0.9999). This ensures the helper works correctly under per-class sigmoid scoring.

    Examples:
        >>> logits = _make_logits()
        >>> logits.shape
        (1, 10, 82)
        >>> float(logits[0, 0, 0])
        10.0
        >>> float(logits[0, 1, 0])
        -10.0
        >>> float(_make_logits(high_conf_idx=None)[0, 0, 0])
        -10.0
    """
    logits = np.full((1, 10, 82), -10.0, dtype=np.float32)
    if high_conf_idx is not None:
        logits[0, high_conf_idx, 0] = 10.0
    return logits


def _make_interp(
    input_shape: list[int] | None = None,
    out_dets: list[dict] | None = None,
    boxes: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> mock.MagicMock:
    """Build a mock TFLite interpreter with configurable I/O details.

    Examples:
        >>> interp = _make_interp()
        >>> len(interp.get_input_details())
        1
        >>> len(interp.get_output_details())
        2
    """
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
    """Write a small solid-colour RGB JPEG to *path*.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "img.jpg"
        ...     _save_rgb_image(p)
        ...     p.exists()
        True
    """
    PILImage.new("RGB", size, color=(100, 150, 200)).save(path)


def _save_grayscale_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Write a small solid-colour grayscale PNG to *path*.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "img.png"
        ...     _save_grayscale_image(p)
        ...     p.exists()
        True
    """
    PILImage.new("L", size, color=128).save(path)


# ---------------------------------------------------------------------------
# TestCreateInterpreter
# ---------------------------------------------------------------------------

# Shared masking entries for mock.patch.dict(sys.modules, ...) that force
# ``_create_interpreter`` to skip the ai_edge_litert backend probe.
_AI_EDGE_LITERT_MASK: dict[str, None] = {
    "ai_edge_litert": None,
    "ai_edge_litert.interpreter": None,
}


class TestCreateInterpreter:
    """Tests for ``_create_interpreter()``."""

    @pytest.fixture()
    def _mock_tflite_runtime(self):
        """Inject a fake tflite_runtime.interpreter into sys.modules and mask ai_edge_litert.

        ``_create_interpreter`` probes backends in priority order: ``ai_edge_litert`` first, then ``tflite_runtime``,
        then ``tensorflow``. Masking ``ai_edge_litert`` and ``ai_edge_litert.interpreter`` to ``None`` forces the import
        loop to fall through to the ``tflite_runtime`` path so tests exercise that branch regardless of what is
        installed.

        Python's import machinery resolves ``import tflite_runtime.interpreter`` by looking up
        ``sys.modules["tflite_runtime.interpreter"]`` directly. We also set the ``interpreter`` attribute on the parent
        package mock so attribute-path resolution is consistent regardless of Python version.
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

        with mock.patch.dict(
            sys.modules,
            {
                **_AI_EDGE_LITERT_MASK,
                "tflite_runtime": parent_mod,
                "tflite_runtime.interpreter": mod,
            },
        ):
            yield interp_cls, interp_instance

    def test_uses_tflite_runtime_when_ai_edge_litert_absent(self, _mock_tflite_runtime) -> None:
        """tflite_runtime is used as backend when ai_edge_litert is masked from the environment."""
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

        with mock.patch.dict(
            sys.modules,
            {
                **_AI_EDGE_LITERT_MASK,
                "tflite_runtime": None,
                "tflite_runtime.interpreter": None,
                "tensorflow": tf_mod,
                "tensorflow.lite": tf_lite_mod,
            },
        ):
            _create_interpreter("model.tflite")

        tf_interp_cls.assert_called_once_with(model_path="model.tflite")
        interp_instance.allocate_tensors.assert_called_once()

    def test_returns_interpreter(self, _mock_tflite_runtime) -> None:
        """Return value is the interpreter instance (not the class)."""
        _, interp_instance = _mock_tflite_runtime
        result = _create_interpreter("model.tflite")
        assert result is interp_instance

    def test_logs_input_and_output_shapes(self, _mock_tflite_runtime) -> None:
        """Logger.debug is called with 'Input' and 'Output' shape lines."""
        with mock.patch("rfdetr.export._tflite.inference.logger") as mock_logger:
            _create_interpreter("model.tflite")
        debug_msgs = [call.args[0] for call in mock_logger.debug.call_args_list]
        assert any("Input" in m for m in debug_msgs)
        assert any("Output" in m for m in debug_msgs)

    def test_accepts_path_object(self, _mock_tflite_runtime) -> None:
        """Path objects are converted to strings before passing to Interpreter."""
        interp_cls, _ = _mock_tflite_runtime
        _create_interpreter(Path("model.tflite"))
        call_kwargs = interp_cls.call_args[1]
        assert call_kwargs["model_path"] == "model.tflite"
        assert isinstance(call_kwargs["model_path"], str)

    @pytest.fixture()
    def _mock_ai_edge_litert(self):
        """Inject a fake ai_edge_litert.interpreter into sys.modules and mask lower-priority backends.

        Mirrors ``_mock_tflite_runtime`` for the first-priority backend so the ``ai_edge_litert.interpreter`` branch of
        ``_create_interpreter`` can be exercised independently of whether the real package is installed.
        """
        interp_instance = mock.MagicMock()
        interp_instance.get_input_details.return_value = [{"shape": [1, 640, 640, 3], "dtype": np.float32}]
        interp_instance.get_output_details.return_value = [
            {"shape": [1, 300, 4], "name": "dets"},
            {"shape": [1, 300, 81], "name": "labels"},
        ]
        interp_cls = mock.MagicMock(return_value=interp_instance)

        import types

        mod = types.ModuleType("ai_edge_litert.interpreter")
        mod.Interpreter = interp_cls  # type: ignore[attr-defined]

        parent_mod = types.ModuleType("ai_edge_litert")
        parent_mod.interpreter = mod  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules,
            {
                "ai_edge_litert": parent_mod,
                "ai_edge_litert.interpreter": mod,
                "tflite_runtime": None,
                "tflite_runtime.interpreter": None,
            },
        ):
            yield interp_cls, interp_instance

    def test_uses_ai_edge_litert_when_available(self, _mock_ai_edge_litert) -> None:
        """ai_edge_litert is used as the first-priority backend when it is importable."""
        interp_cls, _ = _mock_ai_edge_litert
        _create_interpreter("model.tflite")
        interp_cls.assert_called_once_with(model_path="model.tflite")

    def test_ai_edge_litert_allocate_tensors_called(self, _mock_ai_edge_litert) -> None:
        """allocate_tensors() is called after construction via the ai_edge_litert backend."""
        _, interp_instance = _mock_ai_edge_litert
        _create_interpreter("model.tflite")
        interp_instance.allocate_tensors.assert_called_once()

    def test_ai_edge_litert_returns_interpreter(self, _mock_ai_edge_litert) -> None:
        """Return value is the ai_edge_litert interpreter instance."""
        _, interp_instance = _mock_ai_edge_litert
        result = _create_interpreter("model.tflite")
        assert result is interp_instance

    def test_raises_when_no_backend_available(self) -> None:
        """ImportError with a helpful install message is raised when all backends are absent."""
        with mock.patch.dict(
            sys.modules,
            {
                **_AI_EDGE_LITERT_MASK,
                "tflite_runtime": None,
                "tflite_runtime.interpreter": None,
                "tensorflow": None,
                "tensorflow.lite": None,
            },
        ):
            with pytest.raises(ImportError, match="TFLite inference requires"):
                _create_interpreter("model.tflite")


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

    def test_explicit_num_select_caps_export_decode(self, rgb_image: Path) -> None:
        """An explicit exported-model cap limits query/class pairs before thresholding."""
        logits = np.full((1, 10, 82), -100.0, dtype=np.float32)
        logits[0, :, 0] = 10.0
        interp = _make_interp(logits=logits)

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3, num_select=3)

        assert len(dets) == 3

    def test_detections_below_threshold_filtered(self, rgb_image: Path) -> None:
        """No detections survive when all logits are zero (uniform probs < 0.3)."""
        interp = _make_interp(logits=_make_logits(high_conf_idx=None))
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert len(dets) == 0

    def test_active_first_keypoint_layout_excludes_final_background(self, rgb_image: Path) -> None:
        """The default excludes a higher final background score while retaining active keypoint slot 0."""
        logits = np.full((1, 10, 2), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 9.0
        logits[0, 0, 1] = 10.0
        label_out = {"shape": [1, 10, 2], "name": "serving_default_labels:0", "index": 2}
        interp = _make_interp(logits=logits, out_dets=[_DET_OUTPUT, label_out])

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3, num_select=1)

        assert len(dets) == 1
        assert dets.class_id.tolist() == [0]

    def test_zero_foreground_classes_returns_no_detections(self, rgb_image: Path) -> None:
        """A raw output with only the no-object logit excludes it, leaving no class dimension to select from."""
        logits = np.zeros((1, 10, 1), dtype=np.float32)
        label_out = {"shape": [1, 10, 1], "name": "serving_default_labels:0", "index": 2}
        interp = _make_interp(logits=logits, out_dets=[_DET_OUTPUT, label_out])

        dets, _ = _run_inference(interp, rgb_image)

        assert len(dets) == 0

    def test_boxes_in_pixel_space(self, rgb_image: Path) -> None:
        """Xyxy coordinates are scaled to image pixel dimensions, not 0–1 range."""
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

    def test_raises_for_non_float32_input_dtype(self, rgb_image: Path) -> None:
        """ValueError raised when model input dtype is not float32."""
        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.uint8}]
        interp.get_output_details.return_value = [_DET_OUTPUT, _LABEL_OUTPUT]
        with pytest.raises(ValueError, match="float32"):
            _run_inference(interp, rgb_image)


# ---------------------------------------------------------------------------
# TestSigmoidScoring
# ---------------------------------------------------------------------------


class TestSigmoidScoring:
    """Tests for per-class sigmoid scoring introduced in _run_inference."""

    @pytest.fixture()
    def rgb_image(self, tmp_path: Path) -> Path:
        """Write a small RGB JPEG to a temp file and return its path."""
        p = tmp_path / "image.jpg"
        _save_rgb_image(p)
        return p

    def test_high_logit_yields_confidence_near_one(self, rgb_image: Path) -> None:
        """Logit of 10.0 produces sigmoid ≈ 0.9999; confidence[0] > 0.99."""
        logits = _make_logits(high_conf_idx=0)  # logits[0, 0, 0] = 10.0
        interp = _make_interp(logits=logits)
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert dets.confidence[0] > 0.99

    def test_low_logit_filtered_at_threshold(self, rgb_image: Path) -> None:
        """Logit of -10.0 produces sigmoid ≈ 0.0001; detection filtered at threshold=0.3."""
        logits = np.full((1, 10, 82), -10.0, dtype=np.float32)
        interp = _make_interp(logits=logits)
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert len(dets) == 0

    def test_multiclass_query_reports_every_class_above_threshold(self, rgb_image: Path) -> None:
        """A query scoring above threshold on more than one class must produce one detection per class, not just the
        argmax class.

        History: this test used to be ``test_multiclass_class_id_is_argmax_of_logits`` and asserted
        the *opposite* — that only the single highest-scoring class (argmax) survived — codifying a
        real bug as the expected contract. RF-DETR uses independent per-class sigmoids (not a
        mutually exclusive softmax): a query with logits [5, 2, 1, ...] has three classes
        (sigmoid ≈ 0.993, 0.881, 0.731) all above threshold=0.3, and `PostProcess._select_topk`
        (postprocess.py) — the real predict() selection this decode must mirror — ranks Q*C
        query/class pairs together, so all three are legitimate separate detections of the same
        box. See rfdetr/export/_topk.py for the shared selection helper this now uses.
        """
        # Shape (1, 10, 82): first query has logits [5, 2, 1, 0, ...], rest are -100
        logits = np.full((1, 10, 82), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0
        logits[0, 0, 1] = 2.0
        logits[0, 0, 2] = 1.0
        interp = _make_interp(logits=logits)
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)

        assert len(dets) == 3, "query 0 clears threshold=0.3 on 3 classes; all 3 must be reported"
        assert sorted(dets.class_id.tolist()) == [0, 1, 2]
        # All 3 detections share query 0's box (see _make_boxes: identical box per query).
        np.testing.assert_allclose(dets.xyxy, np.tile(dets.xyxy[0], (3, 1)))
        # Sorted by descending confidence, matching PostProcess._select_topk order.
        assert list(dets.confidence) == sorted(dets.confidence, reverse=True)
        assert dets.class_id[0] == 0  # highest logit (5.0) still ranks first

    def test_final_logit_column_is_decoded_without_background_class(self, rgb_image: Path) -> None:
        """Passing ``None`` keeps the final slot for sparse COCO checkpoints."""
        logits = np.full((1, 10, 91), -100.0, dtype=np.float32)
        logits[0, 0, -1] = 10.0
        label_out = {"shape": [1, 10, 91], "name": "serving_default_labels:0", "index": 2}
        interp = _make_interp(logits=logits, out_dets=[_DET_OUTPUT, label_out])

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3, background_class_id=None)

        assert len(dets) == 1
        assert dets.class_id.tolist() == [90]

    def test_background_first_layout_preserves_foreground_slot_id(self, rgb_image: Path) -> None:
        """Excluding slot 0 keeps the original foreground class ID instead of shifting it."""
        logits = np.full((1, 10, 2), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 10.0
        logits[0, 0, 1] = 9.0
        label_out = {"shape": [1, 10, 2], "name": "serving_default_labels:0", "index": 2}
        interp = _make_interp(logits=logits, out_dets=[_DET_OUTPUT, label_out])

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3, num_select=1, background_class_id=0)

        assert len(dets) == 1
        assert dets.class_id.tolist() == [1]


# ---------------------------------------------------------------------------
# TestShapeBasedOutputFallback
# ---------------------------------------------------------------------------

# Generic output detail dicts used across shape-based fallback tests.
# Indices mirror the canonical ones so _make_interp's _get_tensor dispatch works.
_GENERIC_DET_OUTPUT = {"shape": [1, 10, 4], "name": "Identity_0", "index": 1}
_GENERIC_LABEL_OUTPUT = {"shape": [1, 10, 82], "name": "Identity_1", "index": 2}


class TestShapeBasedOutputFallback:
    """Tests for the shape-based output matching fallback in _run_inference."""

    @pytest.fixture()
    def rgb_image(self, tmp_path: Path) -> Path:
        """Write a small RGB JPEG to a temp file and return its path."""
        p = tmp_path / "image.jpg"
        _save_rgb_image(p)
        return p

    def test_unambiguous_shapes_inferred_correctly(self, rgb_image: Path) -> None:
        """Generic names with shapes [1,10,4] and [1,10,82] resolve without error."""
        interp = _make_interp(
            out_dets=[_GENERIC_DET_OUTPUT, _GENERIC_LABEL_OUTPUT],
            logits=_make_logits(high_conf_idx=0),
        )
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert isinstance(dets, sv.Detections)
        assert len(dets) >= 1

    def test_three_outputs_with_rank4_keypoints_are_not_decoded_as_masks(self, rgb_image: Path) -> None:
        """A named rank-4 keypoints export (dets/labels/keypoints) is not fed to the mask decoder."""
        boxes = _make_boxes()
        logits = _make_logits(high_conf_idx=0)
        keypoints = np.zeros((1, 10, 17, 8), dtype=np.float32)

        def _get_tensor(index: int) -> np.ndarray:
            tensors = {1: boxes, 2: logits, 3: keypoints}
            return tensors[index]

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "serving_default_dets:0", "index": 1},
            {"shape": [1, 10, 82], "name": "serving_default_labels:0", "index": 2},
            {"shape": [1, 10, 17, 8], "name": "serving_default_keypoints:0", "index": 3},
        ]
        interp.get_tensor.side_effect = _get_tensor

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)

        assert isinstance(dets, sv.Detections)
        assert len(dets) >= 1
        assert dets.mask is None

    def test_ambiguous_shapes_two_outputs_positional_fallback(self, rgb_image: Path) -> None:
        """When both outputs have last-dim==4 (num_classes==3) and there are exactly 2, positional fallback is used."""
        # num_classes=3 → logits shape last-dim==4; boxes last-dim==4 → ambiguous
        # Positional order: index 0 = boxes (Identity_0, tensor index 1), index 1 = logits (Identity_1, tensor index 2)
        ambiguous_dets = {"shape": [1, 10, 4], "name": "Identity_0", "index": 1}
        ambiguous_labels = {"shape": [1, 10, 4], "name": "Identity_1", "index": 2}
        # Build logits of shape (1, 10, 4) so last col is dropped → (10, 3) per-class
        logits_ambiguous = np.full((1, 10, 4), -10.0, dtype=np.float32)
        logits_ambiguous[0, 0, 0] = 10.0  # first query, first class → high confidence
        interp = _make_interp(
            out_dets=[ambiguous_dets, ambiguous_labels],
            logits=logits_ambiguous,
        )
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert isinstance(dets, sv.Detections)
        assert len(dets) >= 1

    def test_three_outputs_all_dim4_raises_value_error(self, rgb_image: Path) -> None:
        """3 outputs all with last-dim==4 and no name match raises ValueError with expected message."""
        # Need a third tensor index; extend _get_tensor via a custom mock
        third_output = {"shape": [1, 10, 4], "name": "Identity_2", "index": 3}
        boxes = _make_boxes()
        logits = _make_logits()

        def _get_tensor(index: int) -> np.ndarray:
            if index == 1:
                return boxes
            if index in (2, 3):
                return logits
            raise ValueError(f"Unknown tensor index: {index}")

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "Identity_0", "index": 1},
            {"shape": [1, 10, 4], "name": "Identity_1", "index": 2},
            third_output,
        ]
        interp.get_tensor.side_effect = _get_tensor

        with pytest.raises(ValueError, match="Shape-based TFLite output matching failed"):
            _run_inference(interp, rgb_image, threshold=0.3)

    def test_three_outputs_with_rank4_masks_resolves_correctly(self, rgb_image: Path) -> None:
        """3-output segmentation export (boxes/logits/masks) with generic names resolves without error.

        Ensures the shape fallback ignores the rank-4 masks tensor and correctly identifies boxes [1,Q,4] and logits
        [1,Q,C] as rank-3 candidates.
        """
        boxes = _make_boxes()
        logits = _make_logits(high_conf_idx=0)
        masks = np.zeros((1, 10, 28, 28), dtype=np.float32)

        def _get_tensor(index: int) -> np.ndarray:
            if index == 1:
                return boxes
            if index == 2:
                return logits
            if index == 3:
                return masks
            raise ValueError(f"Unknown tensor index: {index}")

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "Identity_0", "index": 1},
            {"shape": [1, 10, 82], "name": "Identity_1", "index": 2},
            {"shape": [1, 10, 28, 28], "name": "Identity_2", "index": 3},
        ]
        interp.get_tensor.side_effect = _get_tensor

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert isinstance(dets, sv.Detections)
        assert len(dets) >= 1


# ---------------------------------------------------------------------------
# TestRank4OutputKind
# ---------------------------------------------------------------------------


def _make_rank4_interp(rank4_tensor: np.ndarray, rank4_name: str = "StatefulPartitionedCall:2") -> mock.MagicMock:
    """Build a mock interpreter for a 3-output export whose rank-4 output carries no kind in its name.

    ``onnx2tf`` does not carry the ONNX output names into the TFLite graph, so RF-DETR's own exports report
    ``StatefulPartitionedCall:N``. Segmentation masks and keypoints then look identical from the name alone.

    Args:
        rank4_tensor: The rank-4 output, shaped ``(1, Q, *, *)``.
        rank4_name: Output name the interpreter reports for *rank4_tensor*.

    Returns:
        A mock interpreter exposing boxes, logits, and *rank4_tensor* at tensor indices 1, 2, and 3.

    Examples:
        >>> interp = _make_rank4_interp(np.zeros((1, 10, 17, 8), dtype=np.float32))
        >>> [od["name"] for od in interp.get_output_details()]
        ['StatefulPartitionedCall:0', 'StatefulPartitionedCall:1', 'StatefulPartitionedCall:2']
        >>> interp.get_tensor(3).shape
        (1, 10, 17, 8)
    """
    boxes = _make_boxes()
    logits = _make_logits(high_conf_idx=0)
    tensors = {1: boxes, 2: logits, 3: rank4_tensor}

    interp = mock.MagicMock()
    interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
    interp.get_output_details.return_value = [
        {"shape": [1, 10, 4], "name": "StatefulPartitionedCall:0", "index": 1},
        {"shape": [1, 10, 82], "name": "StatefulPartitionedCall:1", "index": 2},
        {"shape": list(rank4_tensor.shape), "name": rank4_name, "index": 3},
    ]
    interp.get_tensor.side_effect = lambda index: tensors[index]
    return interp


class TestRank4OutputKind:
    """Tests for classifying a rank-4 output whose name does not say what it holds."""

    @pytest.fixture()
    def rgb_image(self, tmp_path: Path) -> Path:
        """Write a small RGB JPEG to a temp file and return its path."""
        p = tmp_path / "image.jpg"
        _save_rgb_image(p)
        return p

    def test_nameless_rank4_keypoints_are_not_decoded_as_masks_by_default(self, rgb_image: Path) -> None:
        """The safe default does not interpret an anonymous keypoint tensor as segmentation masks."""
        keypoints = np.zeros((1, 10, 17, 8), dtype=np.float32)

        dets, _ = _run_inference(_make_rank4_interp(keypoints), rgb_image, threshold=0.3)

        assert dets.mask is None

    def test_nameless_rank4_masks_are_decoded_when_declared(self, rgb_image: Path) -> None:
        """A segmentation caller can explicitly decode a name-stripped rank-4 mask tensor."""
        masks = np.full((1, 10, 28, 28), -10.0, dtype=np.float32)
        masks[0, 0] = 10.0

        dets, _ = _run_inference(_make_rank4_interp(masks), rgb_image, threshold=0.3, rank4_output="masks")

        assert dets.mask is not None
        assert dets.mask[0].all()

    @pytest.mark.parametrize(
        "rank4_output",
        [pytest.param("keypoints", id="declared-keypoints"), pytest.param(None, id="declared-unknown")],
    )
    def test_nameless_rank4_output_is_not_decoded_as_a_mask_when_declared_otherwise(
        self,
        rgb_image: Path,
        rank4_output: str | None,
    ) -> None:
        """A name-stripped keypoint export's ``pred_keypoints`` tensor never reaches the mask decoder."""
        keypoints = np.zeros((1, 10, 17, 8), dtype=np.float32)

        dets, _ = _run_inference(_make_rank4_interp(keypoints), rgb_image, threshold=0.3, rank4_output=rank4_output)

        assert len(dets) >= 1
        assert dets.mask is None

    @pytest.mark.parametrize(
        "rank4_output",
        [pytest.param(None, id="undeclared"), pytest.param("keypoints", id="declared-keypoints")],
    )
    def test_named_mask_output_is_decoded_whatever_the_declared_kind(
        self, rgb_image: Path, rank4_output: str | None
    ) -> None:
        """An output named ``masks`` is evidence from the artifact and outranks the caller's fallback declaration."""
        masks = np.full((1, 10, 28, 28), -10.0, dtype=np.float32)
        masks[0, 0] = 10.0
        interp = _make_rank4_interp(masks, rank4_name="serving_default_masks:0")

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3, rank4_output=rank4_output)

        assert dets.mask is not None
        assert dets.mask[0].all()

    def test_unrecognised_rank4_output_kind_is_rejected(self, rgb_image: Path) -> None:
        """A misspelled kind fails fast instead of silently falling through to mask decoding."""
        keypoints = np.zeros((1, 10, 17, 8), dtype=np.float32)

        with pytest.raises(ValueError, match="rank4_output"):
            _run_inference(_make_rank4_interp(keypoints), rgb_image, rank4_output="keypoint")


# ---------------------------------------------------------------------------
# TestMaskDecoding
# ---------------------------------------------------------------------------


class TestMaskDecoding:
    """Tests for ``_decode_masks()`` and mask decoding in ``_run_inference()``."""

    @pytest.fixture()
    def rgb_image(self, tmp_path: Path) -> Path:
        """Write a small RGB JPEG to a temp file and return its path."""
        p = tmp_path / "image.jpg"
        _save_rgb_image(p)
        return p

    def test_decode_masks_shape_and_dtype(self) -> None:
        """Output shape is (K, height, width) from out_size=(width, height); dtype is bool."""
        out = _decode_masks(np.zeros((3, 10, 10), dtype=np.float32), (40, 20))
        assert out.shape == (3, 20, 40)
        assert out.dtype == bool

    def test_decode_masks_thresholds_at_zero(self) -> None:
        """Positive logits decode to True, negative logits to False."""
        logits = np.stack(
            [
                np.full((8, 8), 5.0, dtype=np.float32),
                np.full((8, 8), -5.0, dtype=np.float32),
            ]
        )
        out = _decode_masks(logits, (16, 16))
        assert out[0].all()
        assert not out[1].any()

    def test_decode_masks_empty_input(self) -> None:
        """Zero masks in yields a (0, height, width) array, not an error."""
        out = _decode_masks(np.zeros((0, 10, 10), dtype=np.float32), (32, 32))
        assert out.shape == (0, 32, 32)

    def test_run_inference_decodes_declared_masks_for_seg_model(self, rgb_image: Path) -> None:
        """A declared name-stripped segmentation export populates Detections.mask at image size."""
        boxes = _make_boxes()
        logits = _make_logits(high_conf_idx=0)
        masks = np.full((1, 10, 28, 28), -10.0, dtype=np.float32)
        masks[0, 0] = 10.0  # query 0 (the kept detection) gets an all-positive mask

        def _get_tensor(index: int) -> np.ndarray:
            return {1: boxes, 2: logits, 3: masks}[index]

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "Identity_0", "index": 1},
            {"shape": [1, 10, 82], "name": "Identity_1", "index": 2},
            {"shape": [1, 10, 28, 28], "name": "Identity_2", "index": 3},
        ]
        interp.get_tensor.side_effect = _get_tensor

        dets, img = _run_inference(interp, rgb_image, threshold=0.3, rank4_output="masks")
        assert dets.mask is not None
        assert dets.mask.shape == (len(dets), img.height, img.width)
        assert dets.mask.dtype == bool
        assert dets.mask[0].all()  # query 0's all-positive logits decode to a full mask

    def test_final_logit_keeps_selected_query_mask(self, rgb_image: Path) -> None:
        """A final-column detection gathers the mask belonging to its selected query."""
        boxes = _make_boxes()
        logits = np.full((1, 10, 2), -100.0, dtype=np.float32)
        logits[0, 3, 1] = 9.0
        masks = np.full((1, 10, 28, 28), -10.0, dtype=np.float32)
        masks[0, 3] = 10.0

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "Identity_0", "index": 1},
            {"shape": [1, 10, 2], "name": "Identity_1", "index": 2},
            {"shape": [1, 10, 28, 28], "name": "Identity_2", "index": 3},
        ]
        tensors = {1: boxes, 2: logits, 3: masks}
        interp.get_tensor.side_effect = tensors.__getitem__

        dets, _ = _run_inference(
            interp,
            rgb_image,
            threshold=0.3,
            background_class_id=None,
            rank4_output="masks",
        )

        assert len(dets) == 1
        assert dets.class_id.tolist() == [1]
        assert dets.mask is not None
        assert dets.mask.shape[0] == 1
        assert dets.mask[0].all()

    def test_run_inference_multilabel_query_repeats_its_mask(self, rgb_image: Path) -> None:
        """A query contributing more than one detection (multi-label) must gather its mask once per detection, repeats
        included -- not a boolean mask over unique queries.

        Before the fix, mask gathering was ``raw_masks[keep]`` with ``keep`` a boolean vector over queries (at most one
        True per query, matching the old argmax-per-query decode). With the fix, more than one detection can share a
        query index, so gathering must be by (possibly repeating) integer index, not a boolean mask -- see
        _tflite/inference.py's comment on this.
        """
        boxes = _make_boxes()
        logits = np.full((1, 10, 82), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0  # sigmoid ~0.993, above threshold
        logits[0, 0, 1] = 2.0  # sigmoid ~0.881, also above threshold -> query 0 yields 2 detections
        masks = np.full((1, 10, 28, 28), -10.0, dtype=np.float32)
        masks[0, 0] = 10.0  # query 0's mask: all-positive

        def _get_tensor(index: int) -> np.ndarray:
            return {1: boxes, 2: logits, 3: masks}[index]

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "Identity_0", "index": 1},
            {"shape": [1, 10, 82], "name": "Identity_1", "index": 2},
            {"shape": [1, 10, 28, 28], "name": "Identity_2", "index": 3},
        ]
        interp.get_tensor.side_effect = _get_tensor

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3, rank4_output="masks")
        assert len(dets) == 2
        assert sorted(dets.class_id.tolist()) == [0, 1]
        # Both detections came from query 0, so both must carry query 0's (all-positive) mask.
        assert dets.mask.shape[0] == 2
        assert dets.mask[0].all()
        assert dets.mask[1].all()

    def test_run_inference_no_mask_for_detection_model(self, rgb_image: Path) -> None:
        """A 2-output detection export leaves Detections.mask as None."""
        interp = _make_interp(logits=_make_logits(high_conf_idx=0))
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert dets.mask is None

    def test_run_inference_name_based_mask_detection(self, rgb_image: Path) -> None:
        """Output named 'masks:0' exercises the name-based path and sets Detections.mask."""
        boxes = _make_boxes()
        logits = _make_logits(high_conf_idx=0)
        masks = np.full((1, 10, 28, 28), 10.0, dtype=np.float32)

        def _get_tensor(index: int) -> np.ndarray:
            return {1: boxes, 2: logits, 3: masks}[index]

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "serving_default_dets:0", "index": 1},
            {"shape": [1, 10, 82], "name": "serving_default_labels:0", "index": 2},
            {"shape": [1, 10, 28, 28], "name": "serving_default_masks:0", "index": 3},
        ]
        interp.get_tensor.side_effect = _get_tensor

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert dets.mask is not None

    def test_run_inference_seg_model_no_detections_returns_none_mask(self, rgb_image: Path) -> None:
        """Seg model with all scores below threshold returns mask=None (keep.any() is False)."""
        boxes = _make_boxes()
        logits = _make_logits(high_conf_idx=None)  # all scores near zero, below threshold
        masks = np.full((1, 10, 28, 28), 10.0, dtype=np.float32)

        def _get_tensor(index: int) -> np.ndarray:
            return {1: boxes, 2: logits, 3: masks}[index]

        interp = mock.MagicMock()
        interp.get_input_details.return_value = [{"shape": _INPUT_SHAPE, "index": 0, "dtype": np.float32}]
        interp.get_output_details.return_value = [
            {"shape": [1, 10, 4], "name": "Identity_0", "index": 1},
            {"shape": [1, 10, 82], "name": "Identity_1", "index": 2},
            {"shape": [1, 10, 28, 28], "name": "Identity_2", "index": 3},
        ]
        interp.get_tensor.side_effect = _get_tensor

        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        assert len(dets) == 0
        assert dets.mask is None

    def test_decode_masks_raises_on_wrong_rank(self) -> None:
        """_decode_masks raises ValueError when input is not rank-3."""
        with pytest.raises(ValueError, match="rank-3"):
            _decode_masks(np.zeros((10, 28, 28, 1), dtype=np.float32), (56, 56))

    def test_decode_masks_exact_zero_logit_decodes_to_false(self) -> None:
        """Logit exactly 0.0 is not > 0.0 and decodes to False (strict threshold)."""
        zero_logits = np.zeros((1, 8, 8), dtype=np.float32)
        out = _decode_masks(zero_logits, (16, 16))
        assert not out.any()

    def test_decode_masks_non_square_logit_input(self) -> None:
        """Non-square logit map (K, Hm, Wm) with Hm != Wm resizes to the correct output shape."""
        logits = np.full((3, 7, 14), 5.0, dtype=np.float32)
        out = _decode_masks(logits, (56, 28))  # out_size=(width=56, height=28)
        assert out.shape == (3, 28, 56)
        assert out.all()  # all-positive logits → all True

    def test_decode_masks_parity_positive_negative_regions(self) -> None:
        """Positive/negative logit regions map correctly after bilinear upsample + threshold.

        Uses high-magnitude logits (±10) so no ambiguity near the boundary; verifies the core _decode_masks contract
        matches the >0 PostProcess.forward equivalent.
        """
        logits = np.full((1, 14, 14), -10.0, dtype=np.float32)
        logits[0, :7, :] = 10.0  # top half strongly positive, bottom half strongly negative
        out = _decode_masks(logits, (28, 28))
        # Interior rows well away from the half-way boundary
        assert out[0, 1:6, :].all()  # top rows → all True
        assert not out[0, 15:27, :].any()  # bottom rows → all False


# ---------------------------------------------------------------------------
# TestBilinearResizeHalfPixel
# ---------------------------------------------------------------------------


class TestBilinearResizeHalfPixel:
    """Tests for ``_bilinear_resize_half_pixel()``."""

    def test_near_scale_ratio_uses_separable_horizontal_pass(self) -> None:
        """Near-scale resizes interpolate source rows once before gathering output rows."""
        src = np.arange(2 * 7 * 8, dtype=np.float32).reshape(2, 7, 8)

        with mock.patch("rfdetr.export._resize.np.take", wraps=np.take) as take:
            out = _bilinear_resize_half_pixel(src, 6, 5)

        assert out.shape == (2, 6, 5)
        assert take.call_count == 2

    def test_large_downscale_retains_bounded_output_grid(self) -> None:
        """Large height reductions avoid a separable intermediate proportional to the source height."""
        src = np.arange(25 * 9, dtype=np.float32).reshape(1, 25, 9)

        with mock.patch("rfdetr.export._resize.np.take", wraps=np.take) as take:
            out = _bilinear_resize_half_pixel(src, 6, 5)

        assert out.shape == (1, 6, 5)
        take.assert_not_called()

    def test_four_thirds_boundary_retains_output_grid(self) -> None:
        """The exact 4:3 height boundary avoids the larger three-source-grid intermediate."""
        src = np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9)

        with mock.patch("rfdetr.export._resize.np.take", wraps=np.take) as take:
            out = _bilinear_resize_half_pixel(src, 6, 5)

        assert out.shape == (2, 6, 5)
        take.assert_not_called()

    def test_output_shape(self) -> None:
        """Output shape is (K, out_h, out_w)."""
        src = np.ones((3, 8, 8), dtype=np.float32)
        out = _bilinear_resize_half_pixel(src, 16, 16)
        assert out.shape == (3, 16, 16)

    def test_output_dtype_is_float32(self) -> None:
        """Output dtype is float32 regardless of input magnitude."""
        src = np.ones((1, 4, 4), dtype=np.float32)
        out = _bilinear_resize_half_pixel(src, 8, 8)
        assert out.dtype == np.float32

    def test_identity_when_no_resize(self) -> None:
        """Output equals input when target dimensions match source dimensions."""
        rng = np.random.default_rng(0)
        src = rng.random((2, 8, 8)).astype(np.float32)
        out = _bilinear_resize_half_pixel(src, 8, 8)
        np.testing.assert_allclose(out, src, atol=1e-6)

    @pytest.mark.parametrize(
        ("src_shape", "out_h", "out_w"),
        [
            pytest.param((1, 4, 4), 8, 8, id="upsample_square"),
            pytest.param((3, 7, 5), 14, 10, id="upsample_nonsquare"),
            pytest.param((2, 8, 8), 4, 4, id="downsample"),
            pytest.param((1, 1, 1), 3, 3, id="degenerate_1x1"),
        ],
    )
    def test_parity_with_torch_interpolate(self, src_shape: tuple[int, int, int], out_h: int, out_w: int) -> None:
        """Output matches F.interpolate(mode='bilinear', align_corners=False) to within 1e-5."""
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F  # noqa: N812

        rng = np.random.default_rng(42)
        src = rng.random(src_shape).astype(np.float32)

        result = _bilinear_resize_half_pixel(src, out_h, out_w)

        t = torch.from_numpy(src).unsqueeze(0)
        with torch.no_grad():
            ref = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
        ref_np = ref.squeeze(0).numpy()

        np.testing.assert_allclose(result, ref_np, atol=1e-5)


# ---------------------------------------------------------------------------
# TestPreprocessImage
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    """Tests for ``_preprocess_image()``."""

    def test_output_shape_rgb(self) -> None:
        """RGB image returns float32 array of shape (1, H, W, 3)."""
        pil_img = PILImage.new("RGB", (100, 80))
        out = _preprocess_image(pil_img, (64, 64))
        assert out.shape == (1, 64, 64, 3)
        assert out.dtype == np.float32

    def test_output_shape_grayscale(self) -> None:
        """Grayscale image with channels=1 returns float32 array of shape (1, H, W, 1)."""
        pil_img = PILImage.new("L", (100, 80))
        out = _preprocess_image(pil_img, (64, 64), channels=1)
        assert out.shape == (1, 64, 64, 1)
        assert out.dtype == np.float32

    def test_output_values_are_normalized(self) -> None:
        """ImageNet normalization shifts black-pixel output below -1.0."""
        pil_img = PILImage.new("RGB", (32, 32), color=(0, 0, 0))
        out = _preprocess_image(pil_img, (32, 32))
        # pixel 0 → 0.0 → (0.0 - 0.485) / 0.229 ≈ -2.12
        assert out.min() < -1.0

    def test_numpy_fallback_when_torch_unavailable(self) -> None:
        """The NumPy resize path is used when torch is masked from sys.modules."""
        pil_img = PILImage.new("RGB", (100, 80))
        with mock.patch.dict(
            sys.modules,
            {
                "torch": None,
                "torchvision": None,
                "torchvision.transforms": None,
                "torchvision.transforms.functional": None,
            },
        ):
            out = _preprocess_image(pil_img, (64, 64))
        assert out.shape == (1, 64, 64, 3)
        assert out.dtype == np.float32
