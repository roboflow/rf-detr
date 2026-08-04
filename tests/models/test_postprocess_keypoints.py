# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for keypoint decoding in PostProcess."""

from unittest.mock import patch

import pytest
import torch

from rfdetr.models.postprocess import PostProcess


@pytest.mark.parametrize(
    ("schema", "expected_scalar_reads"),
    [
        pytest.param([17], 1, id="single-active-class"),
        pytest.param([0, 17], 3, id="legacy-background-first"),
        pytest.param([0, 3, 0, 5], 5, id="mixed-detection-and-keypoint-classes"),
    ],
)
def test_static_keypoint_schema_avoids_redundant_device_presence_checks(
    schema: list[int], expected_scalar_reads: int
) -> None:
    """Zero-keypoint slots and a lone valid class must not trigger scalar tensor reads.

    The remaining reads are one global valid-class guard plus one presence guard in
    each of the decode and trace-fusion loops for every active class in a multi-class
    schema. Calling ``Tensor.__bool__`` on the scalar returned by ``class_mask.any()``
    synchronizes CUDA, so the count must depend on active rather than padded classes.
    """
    num_classes = len(schema)
    max_keypoints = max(schema)
    postprocess = PostProcess(num_select=num_classes, num_keypoints_per_class=schema)
    outputs = {
        "pred_logits": torch.arange(num_classes, dtype=torch.float32).view(1, 1, num_classes),
        "pred_boxes": torch.full((1, 1, 4), 0.5),
        "pred_keypoints": torch.randn(1, 1, num_classes * max_keypoints, 8),
    }
    target_sizes = torch.tensor([[100, 200]])
    original_bool = torch.Tensor.__bool__
    scalar_reads = 0

    def counting_bool(tensor: torch.Tensor) -> bool:
        nonlocal scalar_reads
        scalar_reads += 1
        return original_bool(tensor)

    with patch.object(torch.Tensor, "__bool__", counting_bool):
        results = postprocess(outputs, target_sizes)

    assert scalar_reads == expected_scalar_reads
    assert results[0]["keypoints"].shape == (num_classes, max_keypoints, 3)


def test_static_keypoint_schema_skips_absent_active_class_without_stale_output() -> None:
    """An active keypoint class fully absent from an image's detections must still decode to zeros.

    The parametrized test above always builds ``pred_logits`` with ``arange``, which picks the single
    active class every time, so it never exercises the ``class_mask.any()`` branch actually returning
    ``False`` for a present-in-schema class. This drives the real checkpoint schema ``[0, 17]`` with
    every detection classified as the zero-keypoint background class, so the active class (1) has zero
    matches and the presence check must still run once per loop and correctly skip.
    """
    postprocess = PostProcess(num_select=2, num_keypoints_per_class=[0, 17])
    outputs = {
        "pred_logits": torch.tensor([[[10.0, -10.0], [10.0, -10.0]]], dtype=torch.float32),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
        "pred_keypoints": torch.randn(1, 2, 34, 8),
    }
    target_sizes = torch.tensor([[100, 200]])
    original_bool = torch.Tensor.__bool__
    scalar_reads = 0

    def counting_bool(tensor: torch.Tensor) -> bool:
        nonlocal scalar_reads
        scalar_reads += 1
        return original_bool(tensor)

    with patch.object(torch.Tensor, "__bool__", counting_bool):
        results = postprocess(outputs, target_sizes)

    assert scalar_reads == 3
    keypoints = results[0]["keypoints"]
    assert keypoints.shape == (2, 17, 3)
    assert torch.equal(keypoints, torch.zeros_like(keypoints))


def test_postprocess_keypoints_shape_and_scores() -> None:
    """PostProcess should emit keypoints and raw precision parameters for top detections."""
    postprocess = PostProcess(num_select=2, num_keypoints_per_class=[17])
    outputs = {
        "pred_logits": torch.tensor([[[10.0, -10.0], [9.0, -10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.4, 0.6, 0.2, 0.3]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 2, 17, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, :, :, 0] = 0.5
    outputs["pred_keypoints"][0, :, :, 1] = 0.25
    outputs["pred_keypoints"][0, :, :, 2] = 3.0
    outputs["pred_keypoints"][0, :, :, 4] = 0.25
    outputs["pred_keypoints"][0, :, :, 5] = 0.5
    outputs["pred_keypoints"][0, :, :, 6] = -0.25

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)
    keypoints = results[0]["keypoints"]
    keypoint_precision = results[0]["keypoint_precision_cholesky"]

    assert keypoints.shape == (2, 17, 3)
    assert torch.allclose(keypoints[:, :, 0], torch.full((2, 17), 100.0))
    assert torch.allclose(keypoints[:, :, 1], torch.full((2, 17), 25.0))
    assert torch.all((keypoints[:, :, 2] > 0) & (keypoints[:, :, 2] < 1))
    assert keypoint_precision.shape == (2, 17, 3)
    torch.testing.assert_close(keypoint_precision[:, :, 0], torch.full((2, 17), 0.25))
    torch.testing.assert_close(keypoint_precision[:, :, 1], torch.full((2, 17), 0.5))
    torch.testing.assert_close(keypoint_precision[:, :, 2], torch.full((2, 17), -0.25))


def test_gather_keypoints_for_queries_repeats_duplicated_indices() -> None:
    """Duplicated and out-of-order query indices must each reproduce that query's exact keypoint rows.

    Top-k selection can pick the same query under two classes, so the gather has to copy the full per-query keypoint
    block verbatim for every occurrence.
    """
    out_keypoints_i = torch.randn(4, 17, 8)
    query_indices = torch.tensor([3, 1, 3])  # query 3 selected twice, out of order

    gathered = PostProcess._gather_keypoints_for_queries(out_keypoints_i, query_indices)

    assert gathered.shape == (3, 17, 8)
    assert torch.equal(gathered[0], out_keypoints_i[3])
    assert torch.equal(gathered[1], out_keypoints_i[1])
    assert torch.equal(gathered[2], out_keypoints_i[3])


def test_postprocess_keypoints_decodes_duplicated_and_out_of_order_queries() -> None:
    """Duplicated and out-of-order query selection must decode each occurrence from the correct source query.

    Drives dupe/out-of-order ``topk_boxes`` through the full class-filtering path (``_decode_keypoints_for_image``)
    rather than the raw gather helper: two detections that select the same query under the same class must produce
    identical decoded keypoints, while an interleaved detection must decode a different query's block.
    """
    postprocess = PostProcess(num_keypoints_per_class=[2, 2])
    out_keypoints = torch.randn(1, 4, 4, 8)  # (B, Q, num_classes * max_kpts, D); D >= 7 for precision cols
    topk_boxes = torch.tensor([[2, 1, 2]])  # query 2 selected twice, query 1 interleaved (out of order)
    labels = torch.zeros(1, 3, dtype=torch.long)  # all class 0 so the two query-2 picks decode identically
    scores = torch.rand(1, 3)
    boxes = torch.zeros(1, 3, 4)
    target_sizes = torch.tensor([[100, 200]])  # (h, w)

    results = postprocess._postprocess_keypoints(out_keypoints, scores, labels, boxes, topk_boxes, target_sizes)
    keypoints = results[0]["keypoints"]  # (3, max_num_keypoints=2, 3)

    assert keypoints.shape == (3, 2, 3)
    assert torch.equal(keypoints[0], keypoints[2])  # same query + class → identical decoded keypoints
    # Independent decode of query 1's class-0 block (flat slots 0,1) confirms out-of-order gather picks the right query.
    img_h, img_w = 100, 200
    q1_class0 = out_keypoints[0, 1, 0:2]
    expected1 = torch.stack([q1_class0[:, 0] * img_w, q1_class0[:, 1] * img_h, q1_class0[:, 2].sigmoid()], dim=-1)
    torch.testing.assert_close(keypoints[1], expected1)


def test_postprocess_keypoints_class_filtering() -> None:
    """Class-specific keypoint slots should be selected from padded per-class keypoint tensors."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[2, 1])
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 4, 8), dtype=torch.float32),
    }
    # class 0 slots: [0, 1], class 1 slots: [2, 3]
    outputs["pred_keypoints"][0, 0, 2, 0] = 0.25
    outputs["pred_keypoints"][0, 0, 2, 1] = 0.4
    outputs["pred_keypoints"][0, 0, 2, 2] = 2.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)
    keypoints = results[0]["keypoints"]
    keypoint_precision = results[0]["keypoint_precision_cholesky"]

    assert keypoints.shape == (1, 2, 3)
    assert torch.allclose(keypoints[0, 0, 0], torch.tensor(50.0))
    assert torch.allclose(keypoints[0, 0, 1], torch.tensor(40.0))
    assert 0.0 < keypoints[0, 0, 2].item() < 1.0
    torch.testing.assert_close(keypoints[0, 1], torch.zeros(3))
    torch.testing.assert_close(keypoint_precision[0, 1], torch.full((3,), float("nan")), equal_nan=True)


def test_postprocess_keypoints_trace_alpha_rescores_active_keypoints_only() -> None:
    """Trace fusion should use active keypoints for the predicted class and ignore padded slots."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[2, 1], trace_alpha=1.0)
    outputs = {
        "pred_logits": torch.tensor([[[-10.0, 0.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 4, 8), dtype=torch.float32),
    }
    # class 1 has one active slot at flat index 2 and one padded inactive slot at flat index 3.
    outputs["pred_keypoints"][0, 0, 2, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 2, 4] = 0.0
    outputs["pred_keypoints"][0, 0, 2, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 2, 6] = 0.0
    outputs["pred_keypoints"][0, 0, 3, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 3, 4] = -2.0
    outputs["pred_keypoints"][0, 0, 3, 6] = -2.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    expected_score = torch.tensor([0.2], dtype=torch.float32)
    torch.testing.assert_close(results[0]["scores"], expected_score, rtol=1e-4, atol=1e-6)


def test_postprocess_keypoints_trace_alpha_normalizes_large_fused_scores() -> None:
    """Trace-fused keypoint scores should be bounded after empirical normalization."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[1], trace_alpha=1.0)
    outputs = {
        "pred_logits": torch.tensor([[[10.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 1, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, 0, 0, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 0, 4] = 2.0
    outputs["pred_keypoints"][0, 0, 0, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 0, 6] = 2.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    original_score = torch.sigmoid(torch.tensor([10.0], dtype=torch.float32))
    mean_trace = torch.tensor([2.0], dtype=torch.float32) * torch.exp(torch.tensor([-4.0], dtype=torch.float32))
    fused_score = original_score * mean_trace.pow(-1.0)
    expected_score = fused_score / (1.0 + fused_score)
    assert fused_score.item() > 1.0
    assert 0.0 < results[0]["scores"].item() < 1.0
    torch.testing.assert_close(results[0]["scores"], expected_score, rtol=1e-4, atol=1e-6)


def test_postprocess_keypoints_trace_alpha_clamps_overflowing_fused_scores() -> None:
    """Trace fusion should stay finite and strictly below 1.0 when the raw fused score overflows."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[1], trace_alpha=1.0)
    outputs = {
        "pred_logits": torch.tensor([[[0.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 1, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, 0, 0, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 0, 4] = 50.0
    outputs["pred_keypoints"][0, 0, 0, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 0, 6] = 50.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    score = results[0]["scores"]
    expected_score = torch.nextafter(torch.ones_like(score), torch.zeros_like(score))
    assert torch.isfinite(score).all()
    assert 0.0 < score.item() < 1.0
    torch.testing.assert_close(score, expected_score, rtol=0.0, atol=0.0)


def test_postprocess_keypoints_trace_alpha_uses_log_space_for_extreme_trace() -> None:
    """Trace fusion should stay finite for extreme covariance terms."""
    postprocess = PostProcess(num_select=1, num_keypoints_per_class=[1])
    outputs = {
        "pred_logits": torch.tensor([[[0.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        "pred_keypoints": torch.zeros((1, 1, 1, 8), dtype=torch.float32),
    }
    outputs["pred_keypoints"][0, 0, 0, 2] = 10.0
    outputs["pred_keypoints"][0, 0, 0, 4] = -50.0
    outputs["pred_keypoints"][0, 0, 0, 5] = 0.0
    outputs["pred_keypoints"][0, 0, 0, 6] = 0.0

    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
    results = postprocess(outputs, target_sizes)

    expected_score = torch.tensor([0.5], dtype=torch.float32) * torch.exp(torch.tensor([-20.0], dtype=torch.float32))
    torch.testing.assert_close(results[0]["scores"], expected_score, rtol=1e-4, atol=1e-12)


def test_postprocess_validate_outputs_raises_when_masks_and_keypoints_both_present() -> None:
    """PostProcess should raise ValueError when both pred_masks and pred_keypoints are present."""
    postprocess = PostProcess(num_select=10)
    outputs = {
        "pred_logits": torch.zeros((1, 2, 2)),
        "pred_boxes": torch.zeros((1, 2, 4)),
        "pred_masks": torch.zeros((1, 2, 4, 4)),
        "pred_keypoints": torch.zeros((1, 2, 17, 8)),
    }
    target_sizes = torch.tensor([[100, 200]], dtype=torch.int64)

    with pytest.raises(ValueError, match="cannot be used together"):
        postprocess(outputs, target_sizes)
