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
test exercised this divergence: ``tests/export/onnx/test_preprocess_parity.py`` / ``test_tflite_inference_parity.py``
only cover *preprocessing* (resize/normalise) parity, and the one existing decode test
(``test_multiclass_class_id_is_argmax_of_logits`` in ``test_tflite_inference.py``) asserted the old, wrong, single-
detection-per-query behaviour as correct. This file locks the numeric selection itself, independent of either runtime
backend.

Error-path coverage for the shared export-parity helpers in ``tests/export/conftest.py``.

These guards only fire when a backend diverges structurally (output count/shape mismatch, an empty forward, or a missing
fixture image), so the green-path end-to-end suites never exercise them. The checks here are backend-agnostic and CPU-
only.

Preprocessing parity tests for the benchmark and export example-input pipelines.

``infer_transforms`` (ONNX/TensorRT benchmark) and ``make_infer_image`` (traced example input for export) must feed the
model the same tensors ``RFDETR.predict()`` produces. History: both pipelines resized the PIL image before tensorising,
going through PIL's adaptive antialias filter; after predict() switched to ``antialias=False`` (#1206) they benchmarked
and traced on inputs predict() never produces. They now tensorise first and resize with ``antialias=False``; the only
residual versus predict()'s ``to_tensor`` chain is 1 ULP from v2's uint8->float scaling.
"""

from __future__ import annotations
from collections.abc import Callable
from unittest import mock
import numpy as np
import pytest
import torch
from rfdetr.export._topk import _select_topk_multiclass
from rfdetr.models.postprocess import PostProcess
from pathlib import Path
from tests.export.conftest import (
    _parity_input_from_image,
    eager_reference_tensors,
    max_abs_output_diffs,
)
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image as PILImage
from rfdetr.export.benchmark import infer_transforms
from rfdetr.export.prepare import make_infer_image

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
    """The export decoder's top-k selection matches ``PostProcess._select_topk`` for the same scores."""

    @pytest.mark.parametrize(
        ("seed", "num_classes", "partition_calls", "lexsort_calls"),
        [
            pytest.param(12, 91, 1, 0, id="sparse-partitions"),
            pytest.param(13, 2, 0, 1, id="dense-full-sort"),
        ],
    )
    def test_float32_selection_partitions_only_when_sparse(
        self, seed: int, num_classes: int, partition_calls: int, lexsort_calls: int
    ) -> None:
        """Partial selection partitions the score grid; selecting over a quarter of it keeps the full sort.

        The common float32 path avoids sorting the complete grid, while selecting more than one quarter of it avoids the
        partition overhead.
        """
        rng = np.random.default_rng(seed)
        scores_all = rng.random((300, num_classes), dtype=np.float32)
        num_select = 300
        flat_scores = scores_all.reshape(-1)
        flat_idx = np.arange(flat_scores.size, dtype=np.int64)
        expected_idx = np.lexsort((flat_idx, -flat_scores))[:num_select]

        with (
            mock.patch("rfdetr.export._topk.np.argpartition", wraps=np.argpartition) as partition,
            mock.patch("rfdetr.export._topk.np.lexsort", wraps=np.lexsort) as lexsort,
        ):
            scores, labels, queries = _select_topk_multiclass(scores_all, threshold=-1.0, num_select=num_select)

        assert (partition.call_count, lexsort.call_count) == (partition_calls, lexsort_calls)
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
        # No-object slot is deeply suppressed as in real training (see rfdetr/export/_topk.py) so it never wins topk;
        # _run_inference drops this column before calling our helper (inference.py:238-240).
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

        # Pairs must match exactly; scores get a tight tolerance because torch's fused sigmoid and the NumPy
        # `1/(1+exp(-x))` can differ in the last float32 ULP (numerical noise, not a selection discrepancy).
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

    @pytest.mark.parametrize(
        ("scores_all", "kwargs"),
        [
            pytest.param(np.array([[0.9, 0.8]], dtype=np.float32), {"num_select": 0}, id="zero-selection"),
            pytest.param(np.empty((0, 3), dtype=np.float32), {}, id="empty-query-grid"),
        ],
    )
    def test_nothing_to_select_returns_empty_arrays(self, scores_all: np.ndarray, kwargs: dict[str, int]) -> None:
        """Selecting zero pairs, or an empty query grid, returns typed empty outputs even when scores clear the
        threshold."""
        scores, labels, queries = _select_topk_multiclass(scores_all, 0.0, **kwargs)

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

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(
                lambda: _select_topk_multiclass(np.ones((1, 1), dtype=np.float32), 0.0, num_select=-1), id="export"
            ),
            pytest.param(lambda: PostProcess(num_select=-1), id="postprocess"),
        ],
    )
    def test_negative_selection_is_rejected(self, build: Callable[[], object]) -> None:
        """Negative selection caps are invalid for both export and PyTorch postprocessing."""
        with pytest.raises(ValueError) as refusal:
            build()

        assert "non-negative" in str(refusal.value)

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


class _NoTensorModule(torch.nn.Module):
    """Stub export-mode module whose forward yields no tensor outputs."""

    def forward(self, x: torch.Tensor) -> tuple[()]:
        """Return an empty output tuple regardless of input."""
        return ()


class TestMaxAbsOutputDiffs:
    """Guards on positional pairing of eager and backend output tensors."""

    def test_count_mismatch_raises(self) -> None:
        """A differing number of eager vs backend tensors must raise AssertionError."""
        eager = [torch.zeros(2, 2)]
        other = [torch.zeros(2, 2), torch.zeros(2, 2)]
        with pytest.raises(AssertionError, match="output count"):
            max_abs_output_diffs(eager, other)

    def test_shape_mismatch_raises(self) -> None:
        """Paired outputs with differing shapes must raise AssertionError when check_shape is set."""
        eager = [torch.zeros(2, 2)]
        other = [torch.zeros(2, 3)]
        with pytest.raises(AssertionError, match="shape mismatch"):
            max_abs_output_diffs(eager, other, check_shape=True)


class TestEagerReferenceTensors:
    """Guard on the eager reference forward producing comparable tensors."""

    def test_no_tensor_outputs_raises(self) -> None:
        """A model that yields no tensor outputs must raise AssertionError."""
        with pytest.raises(AssertionError, match="no tensor outputs"):
            eager_reference_tensors(_NoTensorModule(), torch.zeros(1, 3, 8, 8))


class TestParityInputFromImage:
    """Guard on the image-input builder for missing fixture files."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A path that does not exist must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _parity_input_from_image(tmp_path / "missing.png", 64)


IMAGENET_MEAN = [0.485, 0.456, 0.406]


IMAGENET_STD = [0.229, 0.224, 0.225]


# v2's ToDtype(scale=True) and to_tensor may round the uint8->float conversion 1 ULP apart, which
# propagates through resize+normalize to ~1e-6 in normalised space. The old resize-then-tensorise
# pipelines (PIL antialiased filter) diverged by ~1.6 and would blow far past this bound.
MAX_ABS_DIFF_BOUND = 1e-4


def _pytorch_preprocess(pil_img: PILImage.Image, hw: tuple[int, int]) -> np.ndarray:
    """Mirror of the PyTorch predict() preprocessing: to_tensor -> resize(antialias=False) -> normalize.

    Examples:
        >>> import numpy as np
        >>> from PIL import Image as PILImage
        >>> img = PILImage.new("RGB", (64, 64), color=(128, 64, 32))
        >>> arr = _pytorch_preprocess(img, (32, 32))
        >>> arr.shape
        (1, 3, 32, 32)
        >>> arr.dtype
        dtype('float32')
    """
    img = F.to_tensor(pil_img)
    img = F.resize(img, list(hw), antialias=False)
    img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    return img.unsqueeze(0).numpy()


def _make_synthetic_rgb(seed: int, size: tuple[int, int]) -> PILImage.Image:
    """Deterministic synthetic RGB image with structure (not pure noise) so resize filtering matters.

    Examples:
        >>> img = _make_synthetic_rgb(0, (64, 64))
        >>> img.size
        (64, 64)
        >>> img.mode
        'RGB'
    """
    rng = np.random.default_rng(seed)
    height, width = size
    base = rng.integers(0, 256, size=(height // 8, width // 8, 3), dtype=np.uint8)
    pil_small = PILImage.fromarray(base, mode="RGB")
    return pil_small.resize((width, height), getattr(PILImage, "Resampling", PILImage).NEAREST)


def test_infer_transforms_matches_predict_preprocessing() -> None:
    """The benchmark pipeline must resize with predict()'s convention (bilinear, half-pixel, antialias-free)."""
    pil = _make_synthetic_rgb(seed=3, size=(720, 1280))
    tgt = (384, 384)

    pt = _pytorch_preprocess(pil, tgt)
    tensor, _ = infer_transforms(tgt)(pil, None)
    bench = tensor.unsqueeze(0).numpy()

    assert bench.shape == pt.shape, f"shape mismatch: predict {pt.shape} vs benchmark {bench.shape}"
    max_diff = float(np.abs(pt - bench).max())
    assert max_diff < MAX_ABS_DIFF_BOUND, (
        f"Benchmark preprocessing diverged from predict(): max|diff|={max_diff:.4f}. infer_transforms "
        f"must tensorise before resizing and pass antialias=False, matching detr.py's predict()."
    )

    # The antialiased variant (what a Resize without the flag, or a PIL-first pipeline, produces)
    # must stay far away, proving the assertion discriminates on the resize convention.
    img = F.to_tensor(pil)
    img = F.resize(img, list(tgt), antialias=True)
    img = F.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    max_diff_antialias = float(np.abs(pt - img.unsqueeze(0).numpy()).max())
    assert max_diff_antialias > 0.1, (
        f"antialias=True variant unexpectedly close to predict() (max|diff|={max_diff_antialias:.4f}); "
        "this test can no longer discriminate the antialias regression."
    )


def test_make_infer_image_matches_predict_preprocessing(tmp_path: Path) -> None:
    """The traced example input must go through predict()'s resize convention, batched on the CPU."""
    pil = _make_synthetic_rgb(seed=4, size=(720, 1280))
    img_path = tmp_path / "img.png"
    pil.save(img_path)

    out = make_infer_image(str(img_path), shape=(384, 384), batch_size=2, device="cpu")

    assert out.shape == (2, 3, 384, 384), f"unexpected shape: {tuple(out.shape)}"
    pt = _pytorch_preprocess(pil, (384, 384))
    max_diff = float(np.abs(pt[0] - out[0].numpy()).max())
    assert max_diff < MAX_ABS_DIFF_BOUND, (
        f"Export example input diverged from predict(): max|diff|={max_diff:.4f}. make_infer_image "
        f"must tensorise before resizing and pass antialias=False, matching detr.py's predict()."
    )
    np.testing.assert_array_equal(
        out[0].numpy(), out[1].numpy(), err_msg="batch entries must be identical copies of the same image"
    )
