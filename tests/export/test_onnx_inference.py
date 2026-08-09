# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ONNX Runtime inference decoding (``_run_inference`` in ``rfdetr/export/_onnx/inference.py``).

Before this file, ``_run_inference``'s detection decode had no dedicated unit test at all — only its preprocessing half
was covered (``test_onnx_preprocess_parity.py``). This file covers the multi-label query/class selection, using the same
``types.SimpleNamespace`` fake-session pattern the rest of the export test suite uses for mocking backend objects (see
``test_export.py``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage

from rfdetr.export._onnx.inference import _run_inference

_INPUT_SHAPE = [1, 3, 224, 224]


def _make_boxes() -> np.ndarray:
    """Return (1, 10, 4) array of normalised cxcywh boxes, all centred at 0.5."""
    return np.array([[[0.5, 0.5, 0.1, 0.1]] * 10], dtype=np.float32)


def _make_logits(high_conf_idx: int | None = 0) -> np.ndarray:
    """Return (1, 10, 82) logits with one high-confidence entry when requested (mirrors the TFLite test helper of the
    same name in test_tflite_inference.py)."""
    logits = np.full((1, 10, 82), -10.0, dtype=np.float32)
    if high_conf_idx is not None:
        logits[0, high_conf_idx, 0] = 10.0
    return logits


class _FakeSession:
    """Minimal ``onnxruntime.InferenceSession`` stand-in for ``_run_inference``."""

    def __init__(self, boxes: np.ndarray, logits: np.ndarray, input_shape: list[int] | None = None) -> None:
        self._boxes = boxes
        self._logits = logits
        self._input_shape = input_shape if input_shape is not None else _INPUT_SHAPE

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input", shape=self._input_shape)]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="dets"), SimpleNamespace(name="labels")]

    def run(self, output_names: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names, feeds
        return [self._boxes, self._logits]


@pytest.fixture()
def rgb_image(tmp_path: Path) -> Path:
    """Write a small RGB JPEG to a temp file and return its path."""
    p = tmp_path / "image.jpg"
    PILImage.new("RGB", (64, 64), color=(100, 150, 200)).save(p)
    return p


class TestRunInferenceBasics:
    def test_returns_detections_and_image(self, rgb_image: Path) -> None:
        session = _FakeSession(_make_boxes(), _make_logits())
        dets, img = _run_inference(session, rgb_image)
        import supervision as sv

        assert isinstance(dets, sv.Detections)
        assert isinstance(img, PILImage.Image)

    def test_detections_below_threshold_filtered(self, rgb_image: Path) -> None:
        session = _FakeSession(_make_boxes(), _make_logits(high_conf_idx=None))
        dets, _ = _run_inference(session, rgb_image, threshold=0.3)
        assert len(dets) == 0


class TestMulticlassSelection:
    """``_run_inference`` must select query/class pairs the same way ``PostProcess._select_topk`` does — flatten (Q, C)
    and rank all pairs together, not a per-query argmax.

    See the analogous test in ``test_tflite_inference.py`` and the shared selection helper in
    ``rfdetr/export/_topk.py``.
    """

    def test_multiclass_query_reports_every_class_above_threshold(self, rgb_image: Path) -> None:
        logits = np.full((1, 10, 82), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0  # sigmoid ~0.9933
        logits[0, 0, 1] = 2.0  # sigmoid ~0.8808
        logits[0, 0, 2] = 1.0  # sigmoid ~0.7311
        session = _FakeSession(_make_boxes(), logits)

        dets, _ = _run_inference(session, rgb_image, threshold=0.3)

        assert len(dets) == 3, "query 0 clears threshold=0.3 on 3 classes; all 3 must be reported"
        assert sorted(dets.class_id.tolist()) == [0, 1, 2]
        assert list(dets.confidence) == sorted(dets.confidence, reverse=True)
        assert dets.class_id[0] == 0  # highest logit (5.0) still ranks first
