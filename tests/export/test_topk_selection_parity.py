# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Decode-vs-``PostProcess`` selection parity: ``_select_topk_multiclass`` (``rfdetr/export/_topk.py``, shared by the
ONNX and TFLite reference decoders) must select the exact same set of ``(query, class, score)`` triples that
``PostProcess._select_topk`` (``rfdetr/models/postprocess.py``) selects for the same raw logits — at any query
multiplicity, i.e. a query scoring above threshold on more than one class must produce more than one detection in both.

History: before this fix, ``_run_inference`` in both ``_onnx/inference.py`` and ``_tflite/inference.py`` selected a
single class per query via ``scores_all.argmax(axis=-1)``. ``PostProcess._select_topk`` instead flattens ``(Q, C)`` to
``Q * C`` query/class pairs and ranks all of them together, so a query can contribute more than one detection — exactly
what happens when RF-DETR's independent per-class sigmoids put more than one class above threshold on the same query. No
test exercised this divergence: ``test_onnx_preprocess_parity.py`` / ``test_tflite_inference_parity.py`` only cover
*preprocessing* (resize/normalise) parity, and the one existing decode test
(``test_multiclass_class_id_is_argmax_of_logits`` in ``test_tflite_inference.py``) asserted the old, wrong, single-
detection-per-query behaviour as correct. This file locks the numeric selection itself, independent of either runtime
backend.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from rfdetr.export._topk import _select_topk_multiclass
from rfdetr.models.postprocess import PostProcess


def _reference_select_topk(
    logits_with_bg: torch.Tensor, num_select: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call ``PostProcess._select_topk`` directly on one image's logits (batch size 1).

    Examples:
        >>> logits = torch.zeros((2, 3))
        >>> tuple(value.shape for value in _reference_select_topk(logits, num_select=2))
        (torch.Size([2]), torch.Size([2]), torch.Size([2]))
    """
    pp = PostProcess(num_select=num_select)
    scores, labels, boxes = pp._select_topk(logits_with_bg.unsqueeze(0))
    return scores[0], labels[0], boxes[0]


class TestTopkSelectionParity:
    def test_float32_partial_selection_uses_partition_without_full_sort(self) -> None:
        """The common float32 partial-selection path partitions instead of sorting the complete score grid."""
        rng = np.random.default_rng(12)
        scores_all = rng.random((300, 91), dtype=np.float32)
        num_select = 300
        flat_scores = scores_all.reshape(-1)
        flat_idx = np.arange(flat_scores.size, dtype=np.int64)
        expected_idx = np.lexsort((flat_idx, -flat_scores))[:num_select]

        with (
            mock.patch("rfdetr.export._topk.np.argpartition", wraps=np.argpartition) as partition,
            mock.patch("rfdetr.export._topk.np.lexsort", wraps=np.lexsort) as lexsort,
        ):
            scores, labels, queries = _select_topk_multiclass(scores_all, threshold=-1.0, num_select=num_select)

        partition.assert_called_once()
        lexsort.assert_not_called()
        np.testing.assert_array_equal(scores.view(np.uint32), flat_scores[expected_idx].view(np.uint32))
        np.testing.assert_array_equal(queries * scores_all.shape[1] + labels, expected_idx)

    def test_dense_float32_selection_retains_full_sort(self) -> None:
        """Selecting more than one quarter of the grid avoids partition overhead."""
        rng = np.random.default_rng(13)
        scores_all = rng.random((300, 2), dtype=np.float32)
        num_select = 300
        flat_scores = scores_all.reshape(-1)
        flat_idx = np.arange(flat_scores.size, dtype=np.int64)
        expected_idx = np.lexsort((flat_idx, -flat_scores))[:num_select]

        with (
            mock.patch("rfdetr.export._topk.np.argpartition", wraps=np.argpartition) as partition,
            mock.patch("rfdetr.export._topk.np.lexsort", wraps=np.lexsort) as lexsort,
        ):
            scores, labels, queries = _select_topk_multiclass(scores_all, threshold=-1.0, num_select=num_select)

        partition.assert_not_called()
        lexsort.assert_called_once()
        np.testing.assert_array_equal(scores.view(np.uint32), flat_scores[expected_idx].view(np.uint32))
        np.testing.assert_array_equal(queries * scores_all.shape[1] + labels, expected_idx)

    def test_float32_partition_preserves_special_value_and_tie_order(self) -> None:
        """NaN, infinity, signed zero, and tied cutoff scores retain the established lexicographic order."""
        scores_all = np.full((8, 4), -np.inf, dtype=np.float32)
        scores_all.reshape(-1)[:8] = np.array(
            [np.nan, np.inf, 1.0, 1.0, 0.0, -0.0, -1.0, -np.inf],
            dtype=np.float32,
        )
        flat_scores = scores_all.reshape(-1)
        flat_idx = np.arange(flat_scores.size, dtype=np.int64)
        sort_scores = np.where(np.isnan(flat_scores), np.inf, flat_scores)
        expected_idx = np.lexsort((flat_idx, -sort_scores))[:8]
        expected_scores = flat_scores[expected_idx]
        expected_keep = expected_scores > -np.inf

        with mock.patch("rfdetr.export._topk.np.argpartition", wraps=np.argpartition) as partition:
            scores, labels, queries = _select_topk_multiclass(scores_all, threshold=-np.inf, num_select=8)

        partition.assert_called_once()
        np.testing.assert_array_equal(scores.view(np.uint32), expected_scores[expected_keep].view(np.uint32))
        np.testing.assert_array_equal(
            queries * scores_all.shape[1] + labels,
            expected_idx[expected_keep],
        )

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(np.float16, id="float16"), pytest.param(np.float64, id="float64")],
    )
    def test_non_float32_fallback_preserves_exact_order(self, dtype: type[np.floating]) -> None:
        """Other floating dtypes retain the complete lexicographic fallback."""
        scores_all = np.array([[0.5, 0.5, 0.25], [0.75, np.nan, -0.0]], dtype=dtype)
        flat_scores = scores_all.reshape(-1)
        flat_idx = np.arange(flat_scores.size, dtype=np.int64)
        sort_scores = np.where(np.isnan(flat_scores), np.inf, flat_scores)
        expected_idx = np.lexsort((flat_idx, -sort_scores))[:4]
        expected_scores = flat_scores[expected_idx]
        expected_keep = expected_scores > -1.0

        scores, labels, queries = _select_topk_multiclass(scores_all, threshold=-1.0, num_select=4)

        np.testing.assert_array_equal(scores.view(np.uint8), expected_scores[expected_keep].view(np.uint8))
        np.testing.assert_array_equal(queries * scores_all.shape[1] + labels, expected_idx[expected_keep])

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_matches_postprocess_select_topk(self, seed: int) -> None:
        """Random logits: the (query, class, score) set our NumPy helper keeps above threshold must equal what
        PostProcess._select_topk + the same threshold keeps, for arbitrary multiplicity."""
        rng = np.random.default_rng(seed)
        num_queries, num_fg_classes = 40, 12
        threshold = 0.3

        fg_logits = rng.normal(loc=-1.0, scale=3.0, size=(num_queries, num_fg_classes)).astype(np.float32)
        # No-object slot: deeply suppressed, matching real training (class_embed never receives a
        # positive target for it — see rfdetr/export/_topk.py) so it never wins topk in the
        # reference either. This keeps the comparison apples-to-apples with what _run_inference
        # actually computes: it drops this column before calling our helper (inference.py:238-240).
        bg_logits = np.full((num_queries, 1), -100.0, dtype=np.float32)
        logits_with_bg = torch.from_numpy(np.concatenate([fg_logits, bg_logits], axis=1))

        ref_scores, ref_labels, ref_boxes = _reference_select_topk(logits_with_bg, num_select=300)
        ref_keep = ref_scores > threshold
        ref_by_pair = {
            (int(q), int(c)): float(s)
            for q, c, s in zip(
                ref_boxes[ref_keep].tolist(), ref_labels[ref_keep].tolist(), ref_scores[ref_keep].tolist()
            )
        }

        # float32 throughout, matching what _run_inference actually computes (inference.py:250:
        # `one = np.asarray(1, dtype=logits.dtype)`).
        one = np.asarray(1, dtype=np.float32)
        scores_all = one / (one + np.exp(-fg_logits))
        got_scores, got_labels, got_query = _select_topk_multiclass(scores_all, threshold, num_select=300)
        got_by_pair = {(int(q), int(c)): float(s) for q, c, s in zip(got_query, got_labels, got_scores)}

        # Compare which (query, class) pairs were selected exactly; compare their scores with a
        # tight tolerance rather than exact equality — torch's fused sigmoid kernel and this
        # module's `1/(1+exp(-x))` NumPy formula can differ in the last float32 ULP on the same
        # input, which is expected numerical noise, not a selection-logic discrepancy.
        assert set(got_by_pair) == set(ref_by_pair)
        for pair, ref_s in ref_by_pair.items():
            assert got_by_pair[pair] == pytest.approx(ref_s, abs=1e-4)

    def test_multilabel_query_yields_multiple_detections(self) -> None:
        """A single query with two classes above threshold must produce two detections — the exact case the old per-
        query argmax silently collapsed to one."""
        num_queries, num_fg_classes = 5, 4
        threshold = 0.3
        fg_logits = np.full((num_queries, num_fg_classes), -100.0, dtype=np.float32)
        fg_logits[0, 0] = 5.0  # sigmoid ~0.9933
        fg_logits[0, 1] = 2.0  # sigmoid ~0.8808
        fg_logits[0, 2] = 1.0  # sigmoid ~0.7311
        fg_logits[0, 3] = -1.0  # sigmoid ~0.2689, below threshold

        one = np.asarray(1, dtype=np.float32)
        scores_all = one / (one + np.exp(-fg_logits.clip(-88, 88)))
        got_scores, got_labels, got_query = _select_topk_multiclass(scores_all, threshold, num_select=300)

        assert got_query.tolist() == [0, 0, 0]
        assert got_labels.tolist() == [0, 1, 2]
        assert list(got_scores) == sorted(got_scores, reverse=True)

    def test_num_select_caps_pairs_like_torch_topk(self) -> None:
        """When more than num_select pairs exceed threshold, only the top num_select by score survive — matching
        PostProcess._select_topk (topk happens before, not after, thresholding)."""
        num_queries, num_fg_classes = 10, 10
        threshold = 0.0
        rng = np.random.default_rng(3)
        # loc=3.0 keeps sigmoid well above threshold=0.0 for every pair, so all 100 pairs clear the
        # threshold and only the num_select cap decides which survive.
        fg_logits = rng.uniform(1.0, 5.0, size=(num_queries, num_fg_classes)).astype(np.float32)
        scores_all = 1.0 / (1.0 + np.exp(-fg_logits))

        num_select = 17
        got_scores, _, _ = _select_topk_multiclass(scores_all, threshold, num_select=num_select)
        assert got_scores.shape[0] == num_select
        assert list(got_scores) == sorted(got_scores, reverse=True)

    @pytest.mark.parametrize("num_select", [100, 200])
    def test_model_selection_caps_are_preserved(self, num_select: int) -> None:
        """Export decoding retains the configured cap used by shipped segmentation variants."""
        scores_all = np.full((250, 1), 0.9, dtype=np.float32)

        got_scores, got_labels, got_query = _select_topk_multiclass(scores_all, threshold=0.0, num_select=num_select)

        assert got_scores.shape == (num_select,)
        assert got_labels.shape == (num_select,)
        assert got_query.tolist() == list(range(num_select))

    def test_zero_selection_returns_empty_arrays(self) -> None:
        """Selecting zero pairs returns no detections, even when scores clear the threshold."""
        scores, labels, queries = _select_topk_multiclass(np.array([[0.9, 0.8]], dtype=np.float32), 0.0, num_select=0)

        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert queries.shape == (0,)

    def test_tied_cutoff_uses_flattened_index_order(self) -> None:
        """Equal scores use the same ascending flattened-index tie rule as ``PostProcess``."""
        logits = torch.tensor([[0.0, 0.0], [0.0, -1.0]])
        reference_scores, reference_labels, reference_queries = _reference_select_topk(logits, num_select=2)
        scores_all = torch.sigmoid(logits).numpy()
        scores, labels, queries = _select_topk_multiclass(scores_all, 0.0, num_select=2)

        np.testing.assert_allclose(scores, reference_scores.numpy())
        np.testing.assert_array_equal(labels, reference_labels.numpy())
        np.testing.assert_array_equal(queries, reference_queries.numpy())

    def test_negative_selection_is_rejected(self) -> None:
        """Negative selection caps are invalid for both export and PyTorch postprocessing."""
        with pytest.raises(ValueError, match="non-negative"):
            _select_topk_multiclass(np.ones((1, 1), dtype=np.float32), 0.0, num_select=-1)
        with pytest.raises(ValueError, match="non-negative"):
            PostProcess(num_select=-1)

    def test_nan_cutoff_matches_postprocess_filtering(self) -> None:
        """NaN scores consume the same ranked slot and are then removed by the threshold filter."""
        logits = torch.tensor([[float("nan"), 3.0], [2.0, -1.0]])
        reference_scores, _, _ = _reference_select_topk(logits, num_select=1)
        reference_kept = reference_scores[reference_scores > 0.3]
        scores, labels, queries = _select_topk_multiclass(torch.sigmoid(logits).numpy(), 0.3, num_select=1)

        assert reference_kept.shape == (0,)
        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert queries.shape == (0,)

    def test_empty_query_grid_returns_empty_arrays(self) -> None:
        """An empty query grid is valid and produces typed empty outputs."""
        scores, labels, queries = _select_topk_multiclass(np.empty((0, 3), dtype=np.float32), 0.0)

        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert queries.shape == (0,)
