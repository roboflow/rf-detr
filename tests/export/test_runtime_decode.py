# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the detection decoder shared by the export inference helpers.

``decode_detections`` is the single implementation behind the ONNX and TFLite ``_run_inference`` decoders, so its
contract is exercised here directly rather than only through a session or an interpreter. Covered: normalised ``cxcywh``
to pixel ``xyxy`` conversion, background-slot exclusion with original class IDs preserved, the multi-class selection
that lets one query produce more than one detection, thresholding, and the empty-input edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rfdetr.export._runtime.decode import decode_detections
from rfdetr.models.postprocess import PostProcess
from tests._markers import cuda_marks


class TestDecodeDetections:
    """``decode_detections()`` — sigmoid, background exclusion, top-k selection, box conversion."""

    def test_converts_normalised_boxes_to_pixel_xyxy(self) -> None:
        """A full-frame normalised box decodes to the image's pixel corners.

        This pins the coordinate contract every exported format depends on: boxes leave the model as
        normalised ``cxcywh`` and must arrive as pixel ``xyxy`` scaled by the *source* image size,
        not the model's input resolution.
        """
        boxes = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
        logits = np.array([[9.0, -9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (100, 50), background_class_id=None)

        np.testing.assert_allclose(decoded.xyxy, np.array([[0.0, 0.0, 100.0, 50.0]], dtype=np.float32))

    def test_clamps_out_of_bounds_boxes_to_source_image(self) -> None:
        """Boxes extending beyond the source image stop at its pixel bounds.

        Box regression is unbounded, so an exported model can produce coordinates below zero or beyond the image edge.
        The shared decoder must match ``PostProcess`` by retaining those values only through conversion, then clipping
        every pixel coordinate to the source width and height.
        """
        boxes = np.array([[0.5, 0.5, 1.5, 2.0]], dtype=np.float32)
        logits = np.array([[9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (100, 50), background_class_id=None)

        np.testing.assert_allclose(decoded.xyxy, np.array([[0.0, 0.0, 100.0, 50.0]], dtype=np.float32))

    def test_scores_are_per_class_sigmoid(self) -> None:
        """Confidence is an independent per-class sigmoid, not a softmax over classes.

        RF-DETR's head is multi-label; a softmax here would renormalise the scores and shift every detection's
        confidence away from what ``RFDETR.predict()`` reports.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[2.0, -5.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), background_class_id=None)

        np.testing.assert_allclose(decoded.confidence, [1.0 / (1.0 + np.exp(-2.0))], rtol=1e-6)

    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda", marks=cuda_marks)],
    )
    def test_fp16_threshold_matches_torch_sigmoid(self, device: str) -> None:
        """An FP16 score just above 0.75 must survive like PyTorch's CPU and CUDA sigmoid.

        Rounding each sigmoid operation in FP16 produces exactly 0.75 for this representable logit, so the exported
        runtime used to discard a detection retained by ``PostProcess``.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[1.1005859375]], dtype=np.float16)
        expected = torch.from_numpy(logits).to(device).sigmoid().cpu().numpy().ravel()

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.75)

        assert expected[0] > 0.75
        assert decoded.query_index.tolist() == [0]
        assert decoded.class_id.tolist() == [0]
        assert decoded.confidence.dtype == logits.dtype
        np.testing.assert_array_equal(decoded.confidence, expected)

    def test_fp16_topk_preserves_high_confidence_order(self) -> None:
        """Rounding intermediate values must not turn distinct high confidences into a tie.

        Both logits used to decode to 1.0, allowing the lower-confidence query/class pair to take the only selected
        slot. PyTorch's FP16 sigmoid still distinguishes the two scores.
        """
        boxes = np.array([[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]], dtype=np.float32)
        logits = np.array([[8.0, -9.0], [-9.0, 9.0]], dtype=np.float16)

        decoded = decode_detections(boxes, logits, (10, 10), num_select=1)

        assert decoded.query_index.tolist() == [1]
        assert decoded.class_id.tolist() == [1]
        np.testing.assert_array_equal(decoded.confidence, torch.from_numpy(logits[1, 1:]).sigmoid().numpy())

    @pytest.mark.parametrize("logit", [-65504.0, -12.0, 12.0, 65504.0])
    def test_fp16_extreme_logits_do_not_overflow(self, logit: float) -> None:
        """Finite FP16 inputs must decode without overflowing the exponential, even with ``over="raise"``."""
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[logit]], dtype=np.float16)
        expected = torch.from_numpy(logits).sigmoid().numpy().ravel()

        with np.errstate(over="raise"):
            decoded = decode_detections(boxes, logits, (10, 10), threshold=-1.0)

        assert np.isfinite(decoded.confidence).all()
        assert decoded.confidence.dtype == logits.dtype
        np.testing.assert_array_equal(decoded.confidence, expected)

    def test_fp16_threshold_applies_to_the_rounded_score(self) -> None:
        """A score that only exceeds the threshold before FP16 rounding is dropped, like PyTorch's FP16 sigmoid.

        ``sigmoid(0.0005)`` is just above 0.5 in float32 but rounds to exactly 0.5 in FP16, and the strict ``>``
        comparison must see the rounded value.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[0.0005]], dtype=np.float16)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.5)

        assert torch.from_numpy(logits).sigmoid().item() == 0.5
        assert decoded.confidence.size == 0

    def test_fp16_excluded_background_slot_keeps_original_class_ids(self) -> None:
        """Dropping the background slot from FP16 scores keeps the surviving class IDs and the FP16 dtype."""
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[-9.0, 3.0, 9.0]], dtype=np.float16)

        decoded = decode_detections(boxes, logits, (10, 10), background_class_id=-1)

        assert decoded.class_id.tolist() == [1]
        assert decoded.confidence.dtype == logits.dtype
        np.testing.assert_array_equal(decoded.confidence, torch.from_numpy(logits[:, 1:2]).sigmoid().numpy().ravel())

    def test_fp16_empty_logits_return_empty_fp16_confidence(self) -> None:
        """Zero queries decode to empty arrays that still carry the FP16 score dtype."""
        boxes = np.zeros((0, 4), dtype=np.float32)
        logits = np.zeros((0, 3), dtype=np.float16)

        decoded = decode_detections(boxes, logits, (10, 10))

        assert decoded.confidence.shape == (0,)
        assert decoded.confidence.dtype == logits.dtype
        assert decoded.xyxy.shape == (0, 4)

    def test_fp16_non_finite_logits_match_torch_sigmoid(self) -> None:
        """NaN is never selected, while infinite logits score 1.0 and 0.0 exactly as PyTorch's FP16 sigmoid does."""
        boxes = np.full((3, 4), 0.5, dtype=np.float32)
        logits = np.array([[np.nan], [np.inf], [-np.inf]], dtype=np.float16)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=-1.0)

        assert decoded.query_index.tolist() == [1, 2]
        np.testing.assert_array_equal(decoded.confidence, torch.from_numpy(logits[1:]).sigmoid().numpy().ravel())

    @pytest.mark.parametrize("dtype", ["int8", "int32", "bool"])
    def test_non_floating_logits_keep_floating_confidence(self, dtype: str) -> None:
        """Integer and boolean logits decode to floating probabilities instead of truncating them to 0 or 1.

        Such inputs are outside the annotated contract, but the decoder returned floating scores for them before the
        single rounding to the logits dtype, and casting a probability to an integer dtype silently drops detections.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[0], [1]]).astype(dtype)

        decoded = decode_detections(boxes, logits, (10, 10))

        assert decoded.query_index.tolist() == [1, 0]
        assert np.issubdtype(decoded.confidence.dtype, np.floating)
        np.testing.assert_allclose(decoded.confidence, [1.0 / (1.0 + np.exp(-1.0)), 0.5], rtol=1e-6)

    @pytest.mark.parametrize("width", ["f2", "f4", "f8"])
    def test_non_native_byte_order_logits_give_native_confidence(self, width: str) -> None:
        """Big-endian logits produce native-byte-order confidence.

        A non-native score array would skip the float32 fast top-k path and be rejected by ``torch.from_numpy``.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[1.1005859375]], dtype=np.dtype(f">{width}"))

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.75)

        assert decoded.confidence.dtype.isnative
        assert decoded.confidence.dtype == np.dtype(width)
        assert decoded.confidence.size == 1

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
    def test_threshold_remains_strict(self, dtype: str) -> None:
        """Higher-precision sigmoid arithmetic must still drop a score exactly equal to the threshold."""
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[0.0]], dtype=dtype)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.5)

        assert decoded.confidence.size == 0
        assert decoded.confidence.dtype == logits.dtype

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_sigmoid_preserves_floating_point_precision(self, dtype: str) -> None:
        """FP32 and FP64 confidence keeps its input precision instead of being forced through a lower dtype."""
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[1.1005859375]], dtype=dtype)
        expected = torch.from_numpy(logits).sigmoid().numpy().ravel()

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.75)

        assert decoded.confidence.dtype == logits.dtype
        np.testing.assert_allclose(decoded.confidence, expected, rtol=4 * np.finfo(dtype).eps, atol=0.0)

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
    @pytest.mark.parametrize("read_only", [False, True])
    def test_decode_does_not_modify_logits(self, dtype: str, read_only: bool) -> None:
        """Decoding must leave caller-owned logits intact, including strided and read-only views.

        Reusing a sigmoid work buffer must not overwrite model outputs or aliases held by the caller, even when no dtype
        conversion is needed.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=np.float32)
        storage = np.array([[-100.0, 2.0, 1.1005859375, 3.0], [100.0, 4.0, -12.0, 5.0]], dtype=dtype)
        original = storage.copy()
        logits = storage[:, ::2]
        if read_only:
            logits.flags.writeable = False

        decode_detections(boxes, logits, (10, 10))

        np.testing.assert_array_equal(storage, original)

    def test_excluded_background_slot_keeps_original_class_ids(self) -> None:
        """Dropping the background slot must not renumber the surviving classes.

        The exported class IDs are the label space the user trained against. Excluding the final slot by *position*
        while returning compacted indices would silently shift every class after it.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        # Slot 2 scores highest but is the background slot; slot 1 must win with its ID intact.
        logits = np.array([[-9.0, 3.0, 9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), background_class_id=-1)

        assert decoded.class_id.tolist() == [1]

    def test_one_query_can_produce_two_detections(self) -> None:
        """A query scoring above threshold on two classes yields two detections sharing a query row.

        This is the regression that a per-query ``argmax`` decoder loses: with independent sigmoids a single query
        legitimately fires on, say, "car" and "truck". ``query_index`` therefore repeats, which is what lets a caller
        gather per-query outputs such as segmentation masks correctly.

        Two queries are needed to show it: the default cap is the query count, so query 0 can only claim both slots
        because query 1 scores nothing.
        """
        boxes = np.array([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]], dtype=np.float32)
        logits = np.array([[4.0, 4.0], [-9.0, -9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.3, background_class_id=None)

        assert decoded.query_index.tolist() == [0, 0]

    def test_drops_detections_at_or_below_threshold(self) -> None:
        """Scores that do not clear the threshold are discarded rather than returned with low confidence.

        Callers hand the result straight to ``supervision.Detections``, so filtering has to happen here; a leaked sub-
        threshold row becomes a visible false positive downstream.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[5.0, -9.0], [-5.0, -9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.5, background_class_id=None)

        assert decoded.query_index.tolist() == [0]

    def test_num_select_caps_pairs_before_thresholding(self) -> None:
        """``num_select`` limits the query/class pairs considered, mirroring ``PostProcess._select_topk``.

        The cap is applied before the threshold, not after, so a small ``num_select`` can legitimately return fewer
        detections than the threshold alone would allow.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[9.0, -9.0], [8.0, -9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), num_select=1, background_class_id=None)

        assert decoded.query_index.tolist() == [0]

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_default_keeps_final_coco_slot_like_predict(self, dtype: str) -> None:
        """With the default, slot 90 of a 91-slot COCO head survives and matches ``PostProcess`` (``predict()``).

        Official COCO checkpoints keep every slot, and the final one is the real class 90. Excluding it by default would
        silently drop those detections, so the decoder must agree with ``PostProcess`` on labels, scores and boxes. FP16
        logits must agree exactly on scores, since both sides round once from higher-precision arithmetic.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=np.float32)
        logits = np.full((2, 91), -9.0, dtype=dtype)
        logits[0, 90] = 9.0  # slot 90 is the top score
        logits[1, 5] = 3.0

        decoded = decode_detections(boxes, logits, (100, 50))

        expected = PostProcess(num_select=2)(
            {"pred_logits": torch.from_numpy(logits)[None], "pred_boxes": torch.from_numpy(boxes)[None]},
            torch.tensor([[50, 100]]),
        )[0]
        keep = expected["scores"] > 0.3
        assert decoded.class_id.tolist() == expected["labels"][keep].tolist() == [90, 5]
        np.testing.assert_allclose(
            decoded.confidence, expected["scores"][keep].numpy(), rtol=1e-5 if dtype == "float32" else 0.0
        )
        np.testing.assert_allclose(decoded.xyxy, expected["boxes"][keep].numpy(), rtol=1e-5, atol=1e-4)

    def test_returns_empty_arrays_when_nothing_clears_threshold(self) -> None:
        """No surviving detection yields empty arrays, not ``None`` or a ragged shape.

        The ONNX and TFLite callers build a ``Detections`` object unconditionally, so the empty case has to stay array-
        shaped or construction raises on an image with no objects.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[-9.0, -9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.5, background_class_id=None)

        assert decoded.xyxy.shape == (0, 4)
