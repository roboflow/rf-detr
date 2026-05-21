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

from rfdetr.export._tflite.inference import _create_interpreter, _decode_masks, _run_inference

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
    """Return (1, 10, 82) logits with one high-confidence entry when requested.

    Background fill is -10.0 so sigmoid scores are near zero (~0.0001) for all
    entries except the explicitly boosted one (logit=+10.0, sigmoid≈0.9999).
    This ensures the helper works correctly under per-class sigmoid scoring.
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

        with mock.patch.dict(sys.modules, {"tflite_runtime": parent_mod, "tflite_runtime.interpreter": mod}):
            yield interp_cls, interp_instance

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

        with mock.patch.dict(
            sys.modules,
            {
                "ai_edge_litert": None,
                "ai_edge_litert.interpreter": None,
                "tflite_runtime": None,
                "tflite_runtime.interpreter": None,
                "tensorflow": tf_mod,
                "tensorflow.lite": tf_lite_mod,
            },
        ):
            _create_interpreter("model.tflite")

        tf_interp_cls.assert_called_once_with(model_path="model.tflite")

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

    def test_multiclass_class_id_is_argmax_of_logits(self, rgb_image: Path) -> None:
        """argmax of sigmoid equals argmax of logits; query with [5,2,1,...] gets class_id==0."""
        # Shape (1, 10, 82): first query has logits [5, 2, 1, 0, ...], rest are -100
        logits = np.full((1, 10, 82), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0
        logits[0, 0, 1] = 2.0
        logits[0, 0, 2] = 1.0
        interp = _make_interp(logits=logits)
        dets, _ = _run_inference(interp, rgb_image, threshold=0.3)
        # argmax of sigmoid == argmax of logits because sigmoid is monotone increasing
        assert dets.class_id[0] == 0


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

        Ensures the shape fallback ignores the rank-4 masks tensor and correctly
        identifies boxes [1,Q,4] and logits [1,Q,C+1] as rank-3 candidates.
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

    def test_run_inference_decodes_masks_for_seg_model(self, rgb_image: Path) -> None:
        """A 3-output segmentation export populates Detections.mask at image size."""
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

        dets, img = _run_inference(interp, rgb_image, threshold=0.3)
        assert dets.mask is not None
        assert dets.mask.shape == (len(dets), img.height, img.width)
        assert dets.mask.dtype == bool
        assert dets.mask[0].all()  # query 0's all-positive logits decode to a full mask

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

        Uses high-magnitude logits (±10) so no ambiguity near the boundary; verifies
        the core _decode_masks contract matches the >0 PostProcess.forward equivalent.
        """
        logits = np.full((1, 14, 14), -10.0, dtype=np.float32)
        logits[0, :7, :] = 10.0  # top half strongly positive, bottom half strongly negative
        out = _decode_masks(logits, (28, 28))
        # Interior rows well away from the half-way boundary
        assert out[0, 1:6, :].all()  # top rows → all True
        assert not out[0, 15:27, :].any()  # bottom rows → all False
