# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Scores of keypoint-schema slots with zero keypoints in PostProcess.

A schema such as ``[2, 0]`` gives class 1 a slot but no keypoints, so it has no localization uncertainty to fuse. Its
rows must keep their raw score, as rows whose label lies past the schema already do, instead of the ``s / (1 + s)`` a
zero trace would produce: that value never exceeds 0.5, so ``predict()``'s default threshold would drop every row.
"""

import math

import pytest
import torch

from rfdetr.models.postprocess import PostProcess

#: Per-class keypoint counts: class 0 has two keypoints, class 1 a slot with none; class 2 lies past the schema.
_SCHEMA = [2, 0]
#: Object logit of the one confident class of each query; query ``q`` predicts class ``q``.
_LOGITS = [2.0, 1.5, 1.0]
_TARGET_SIZES = torch.tensor([[4, 4]])


def _mixed_schema_outputs(include_masks: bool) -> dict[str, torch.Tensor]:
    """Return outputs for three queries, query ``q`` confident only in class ``q``, under the ``[2, 0]`` schema.

    Every keypoint has an identity precision Cholesky factor and a findable logit of 0, so the class-0 row fuses with a
    mean covariance trace of 2. Query ``q``'s mask is positive only in column ``q``.

    Args:
        include_masks: Whether to add ``pred_masks``, as a model with a segmentation head does.

    Examples:
        >>> outputs = _mixed_schema_outputs(include_masks=True)
        >>> outputs["pred_logits"].shape, outputs["pred_keypoints"].shape, outputs["pred_masks"].shape
        (torch.Size([1, 3, 3]), torch.Size([1, 3, 4, 8]), torch.Size([1, 3, 4, 4]))
    """
    num_queries = len(_LOGITS)
    logits = torch.full((1, num_queries, num_queries), -10.0)
    for query, logit in enumerate(_LOGITS):
        logits[0, query, query] = logit
    outputs = {
        "pred_logits": logits,
        "pred_boxes": torch.full((1, num_queries, 4), 0.5),
        "pred_keypoints": torch.zeros((1, num_queries, len(_SCHEMA) * max(_SCHEMA), 8)),
    }
    if include_masks:
        masks = torch.full((1, num_queries, 4, 4), -1.0)
        for query in range(num_queries):
            masks[0, query, :, query] = 1.0
        outputs["pred_masks"] = masks
    return outputs


def _raw_score(label: int) -> float:
    """Return the sigmoid score of the class ``label`` query, before any fusion.

    Examples:
        >>> round(_raw_score(1), 4)
        0.8176
    """
    return 1.0 / (1.0 + math.exp(-_LOGITS[label]))


class TestZeroCountKeypointSlotScores:
    """Rows whose class has no keypoints keep their raw score; rows of an active class stay fused."""

    @pytest.mark.parametrize("include_masks", [False, True])
    def test_zero_count_class_keeps_raw_score(self, include_masks: bool) -> None:
        """The class-1 row reports its sigmoid score, in the keypoint-only and in the two-head path."""
        postprocess = PostProcess(num_select=3, num_keypoints_per_class=_SCHEMA)

        (result,) = postprocess(_mixed_schema_outputs(include_masks), _TARGET_SIZES)

        assert result["labels"].tolist() == [0, 1, 2]
        assert result["scores"][1].item() == pytest.approx(_raw_score(1), rel=1e-6)

    def test_zero_count_and_out_of_schema_rows_share_the_raw_scale(self) -> None:
        """A zero-count slot is scored like a label past the schema, so one result holds one score scale."""
        postprocess = PostProcess(num_select=3, num_keypoints_per_class=_SCHEMA)

        (result,) = postprocess(_mixed_schema_outputs(include_masks=False), _TARGET_SIZES)

        assert result["scores"][1:].tolist() == pytest.approx([_raw_score(1), _raw_score(2)], rel=1e-6)

    def test_active_class_is_still_fused(self) -> None:
        """The class-0 row is still divided by its covariance trace to the power ``trace_alpha`` and normalized."""
        postprocess = PostProcess(num_select=3, num_keypoints_per_class=_SCHEMA, trace_alpha=0.2)

        (result,) = postprocess(_mixed_schema_outputs(include_masks=False), _TARGET_SIZES)

        fused = _raw_score(0) * 2.0**-0.2
        assert result["scores"][0].item() == pytest.approx(fused / (1.0 + fused), rel=1e-5)

    def test_mask_only_class_keeps_its_mask_at_the_default_threshold(self) -> None:
        """With both heads, the class-1 row survives ``score_threshold=0.5`` with its own mask and raw score."""
        postprocess = PostProcess(num_select=3, num_keypoints_per_class=_SCHEMA)

        (result,) = postprocess(_mixed_schema_outputs(include_masks=True), _TARGET_SIZES, score_threshold=0.5)

        labels = result["labels"].tolist()
        assert 1 in labels
        row = labels.index(1)
        assert result["scores"][row].item() == pytest.approx(_raw_score(1), rel=1e-6)
        # Query 1's mask is positive only in column 1.
        assert result["masks"][row, 0, 0].tolist() == [False, True, False, False]
        assert result["keypoints"].shape[0] == len(labels)
