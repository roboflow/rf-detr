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

from rfdetr.export._onnx.inference import _create_onnx_session, _run_inference

_INPUT_SHAPE = [1, 3, 224, 224]


def _make_boxes() -> np.ndarray:
    """Return normalised cxcywh boxes, all centred at 0.5.

    Examples:
        >>> _make_boxes().shape
        (1, 10, 4)
    """
    return np.array([[[0.5, 0.5, 0.1, 0.1]] * 10], dtype=np.float32)


def _make_logits(high_conf_idx: int | None = 0) -> np.ndarray:
    """Return logits with one high-confidence entry when requested.

    Examples:
        >>> _make_logits().shape
        (1, 10, 82)
        >>> float(_make_logits()[0, 0, 0])
        10.0
    """
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

    def test_active_first_keypoint_layout_excludes_final_background(self, rgb_image: Path) -> None:
        """The default excludes a higher final background score while retaining active keypoint slot 0."""
        logits = np.full((1, 10, 2), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 9.0
        logits[0, 0, 1] = 10.0
        session = _FakeSession(_make_boxes(), logits)

        dets, _ = _run_inference(session, rgb_image, threshold=0.3, num_select=1)

        assert len(dets) == 1
        assert dets.class_id.tolist() == [0]

    def test_zero_foreground_classes_returns_no_detections(self, rgb_image: Path) -> None:
        """A raw output with only the no-object logit excludes it, leaving no class dimension to select from."""
        session = _FakeSession(_make_boxes(), np.zeros((1, 10, 1), dtype=np.float32))

        dets, _ = _run_inference(session, rgb_image)

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

    def test_final_logit_column_is_decoded_without_background_class(self, rgb_image: Path) -> None:
        """Passing ``None`` keeps the final slot for sparse COCO checkpoints."""
        logits = np.full((1, 10, 91), -100.0, dtype=np.float32)
        logits[0, 0, -1] = 10.0
        session = _FakeSession(_make_boxes(), logits)

        dets, _ = _run_inference(session, rgb_image, threshold=0.3, background_class_id=None)

        assert len(dets) == 1
        assert dets.class_id.tolist() == [90]

    def test_background_first_layout_preserves_foreground_slot_id(self, rgb_image: Path) -> None:
        """Excluding slot 0 keeps the original foreground class ID instead of shifting it."""
        logits = np.full((1, 10, 2), -100.0, dtype=np.float32)
        logits[0, 0, 0] = 10.0
        logits[0, 0, 1] = 9.0
        session = _FakeSession(_make_boxes(), logits)

        dets, _ = _run_inference(session, rgb_image, threshold=0.3, num_select=1, background_class_id=0)

        assert len(dets) == 1
        assert dets.class_id.tolist() == [1]

    def test_explicit_num_select_caps_export_decode(self, rgb_image: Path) -> None:
        """An explicit exported-model cap limits query/class pairs before thresholding."""
        logits = np.full((1, 10, 82), -100.0, dtype=np.float32)
        logits[0, :, 0] = 10.0
        session = _FakeSession(_make_boxes(), logits)

        dets, _ = _run_inference(session, rgb_image, threshold=0.3, num_select=3)

        assert len(dets) == 3


@pytest.fixture()
def tiny_onnx_model(tmp_path: Path) -> Path:
    """Write a minimal single-node ONNX model and return its path.

    ``_create_onnx_session`` only needs a loadable graph; an ``Identity`` node avoids depending on the real RF-DETR
    export path for a session-construction test.

    Examples:
        Pytest fixture functions cannot be called directly outside fixture injection.
        >>> tiny_onnx_model(Path("."))  # doctest: +SKIP
    """
    onnx = pytest.importorskip("onnx", reason="onnx not installed")
    tensor_proto, helper = onnx.TensorProto, onnx.helper

    inp = helper.make_tensor_value_info("input", tensor_proto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("output", tensor_proto.FLOAT, [1, 3, 8, 8])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = tmp_path / "identity.onnx"
    onnx.save(model, str(onnx_path))
    return onnx_path


class TestCreateOnnxSession:
    """Tests for ``_create_onnx_session``'s CPU thread-pool configuration.

    ``_run_inference``'s own preprocessing (``preprocess_to_nchw``, via ``torchvision`` when installed) runs in the same
    process as the session it feeds. ONNX Runtime's default CPU thread pool busy-spins between calls rather than
    blocking, so those idle-but-spinning threads contend for CPU with that preprocessing step -- and with the session's
    own next call -- for as long as the process lives. Disabling spinning only changes how idle threads wait; it does
    not change computed values.
    """

    def test_disables_intra_op_spinning_on_cpu(self, tiny_onnx_model: Path) -> None:
        """The constructed session must not let its CPU threads busy-spin between calls."""
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

        session = _create_onnx_session(tiny_onnx_model, providers=["CPUExecutionProvider"])

        options = session.get_session_options()
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"

    def test_still_produces_correct_output(self, tiny_onnx_model: Path) -> None:
        """Disabling spinning must not change what the session computes."""
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

        session = _create_onnx_session(tiny_onnx_model, providers=["CPUExecutionProvider"])

        feed = np.arange(3 * 8 * 8, dtype=np.float32).reshape(1, 3, 8, 8)
        (actual,) = session.run(None, {session.get_inputs()[0].name: feed})

        np.testing.assert_array_equal(actual, feed)

    def test_disables_spinning_regardless_of_requested_provider_list(self, tiny_onnx_model: Path) -> None:
        """The spin-config entries attach before provider selection, so they apply the same way for any requested
        provider list -- including one naming a GPU provider unavailable on the machine running the test.

        ONNX Runtime falls back to an available provider with a warning rather than raising in that case.
        """
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

        session = _create_onnx_session(tiny_onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        options = session.get_session_options()
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"
