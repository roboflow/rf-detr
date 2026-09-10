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

from rfdetr.export._runtime.decode import decode_detections


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

    def test_returns_empty_arrays_when_nothing_clears_threshold(self) -> None:
        """No surviving detection yields empty arrays, not ``None`` or a ragged shape.

        The ONNX and TFLite callers build a ``Detections`` object unconditionally, so the empty case has to stay array-
        shaped or construction raises on an image with no objects.
        """
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        logits = np.array([[-9.0, -9.0]], dtype=np.float32)

        decoded = decode_detections(boxes, logits, (10, 10), threshold=0.5, background_class_id=None)

        assert decoded.xyxy.shape == (0, 4)
