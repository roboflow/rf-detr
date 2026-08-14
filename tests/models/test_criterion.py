# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for SetCriterion edge paths: _output_device and num_boxes_for_targets."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.utilities.tensors import NestedTensor


class _MatcherStub:
    """Minimal matcher that returns identity indices for every target in the batch."""

    def __call__(self, outputs, targets, group_detr=1, target_side_safety=None):
        return [(torch.arange(len(t["labels"])), torch.arange(len(t["labels"]))) for t in targets]


class _LegacyMatcherStub:
    """Matcher predating the target-side-safety cache: two-argument signature, no precompute method.

    Deliberately has neither a ``target_side_safety`` parameter nor a ``_precompute_target_side_safety`` attribute, so
    it raises TypeError if ``SetCriterion.forward`` passes the kwarg unconditionally.
    """

    def __call__(self, outputs, targets, group_detr=1):
        return [(torch.arange(len(t["labels"])), torch.arange(len(t["labels"]))) for t in targets]


def _bare_criterion() -> SetCriterion:
    """Return a SetCriterion with no losses so forward() is a no-op."""
    criterion = SetCriterion.__new__(SetCriterion)
    criterion.training = True
    criterion.group_detr = 1
    criterion.sum_group_losses = False
    criterion.losses = []
    criterion.weight_dict = {}
    criterion.matcher = _MatcherStub()
    criterion.num_keypoints_per_class = []
    return criterion


class TestOutputDevice:
    """Tests for SetCriterion._output_device — probes top-level tensor values only."""

    def test_returns_device_of_first_tensor(self):
        """Device inferred from the first tensor value in outputs."""
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")

    def test_raises_when_no_tensor_present(self):
        """ValueError raised when no top-level value is a tensor."""
        outputs = {"meta": "string_value", "count": 42}

        with pytest.raises(ValueError, match="at least one tensor"):
            SetCriterion._output_device(outputs)

    def test_skips_non_tensor_values(self):
        """Non-tensor entries at the top level are skipped; first tensor wins."""
        outputs = {"meta": "ignored", "pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")


class TestNumBoxesForTargets:
    """Tests for SetCriterion.num_boxes_for_targets — clamp and empty-target edge cases."""

    def test_returns_tensor_gte_one(self):
        """Result must be clamped to >= 1.0 to prevent division by zero."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.tensor([0, 1])}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() >= 1.0

    def test_clamps_zero_box_count_to_one(self):
        """Empty targets (no labels) must clamp to 1.0 to avoid zero denominator."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.zeros(0, dtype=torch.int64)}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_clamps_empty_target_list(self):
        """Empty target list (batch_size=0 edge case) must also clamp to 1.0."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = []

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_counts_labels_correctly(self):
        """Box count equals total number of labels across all targets in the batch."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [
            {"labels": torch.tensor([0, 1])},
            {"labels": torch.tensor([0])},
        ]

        result = criterion.num_boxes_for_targets(outputs, targets)

        # 2 + 1 = 3 boxes; single-process so no all-reduce
        assert result.item() == pytest.approx(3.0)


class TestLossMasksEmptyMatch:
    """Tests for the dict-path zero-GT branch of SetCriterion.loss_masks."""

    def test_dict_path_zero_gt_stays_connected_to_graph(self):
        """Zero-match dict path returns a loss that back-propagates to every segmentation-head output."""
        criterion = _bare_criterion()
        spatial_features = torch.randn(1, 4, 8, 8, requires_grad=True)
        query_features = torch.randn(1, 5, 4, requires_grad=True)
        bias = torch.randn(1, requires_grad=True)
        outputs = {
            "pred_masks": {
                "spatial_features": spatial_features,
                "query_features": query_features,
                "bias": bias,
            }
        }
        empty = torch.empty(0, dtype=torch.long)
        indices = [(empty, empty)]

        losses = criterion.loss_masks(outputs, targets=[{}], indices=indices, num_boxes=1)

        assert losses["loss_mask_ce"].requires_grad
        (losses["loss_mask_ce"] + losses["loss_mask_dice"]).backward()
        assert spatial_features.grad is not None
        assert query_features.grad is not None
        assert bias.grad is not None


class TestLossMasksNonEmptyMatchUsesRealSegmentationHeadOutput:
    """The non-empty-match branch of the dict path (``criterion.py``'s einsum over
    ``outputs["pred_masks"]["spatial_features"]``) reads that key straight off the dict with no projection of its own —
    it trusts whatever produced it.

    ``SegmentationHead``-only tests can't catch a regression here, since they never call ``loss_masks``. This builds the
    dict with a real ``SegmentationHead`` (``sparse_forward(skip_blocks=True)``, non-identity ``spatial_features_proj``)
    instead of hand-built tensors, and checks the loss is sensitive to whether the projection was applied and that
    gradient reaches ``spatial_features_proj.weight``.
    """

    def test_loss_backprops_to_projection_weight_and_is_sensitive_to_it(self) -> None:
        torch.manual_seed(0)
        hidden, mask_size = 4, 8
        head = SegmentationHead(in_dim=hidden, num_blocks=1, bottleneck_ratio=1, downsample_ratio=1)
        spatial_features = torch.randn(1, hidden, mask_size, mask_size)
        query_features = torch.randn(1, 2, hidden)
        targets = [{"masks": torch.rand(2, mask_size, mask_size)}]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        criterion = _bare_criterion()
        criterion.mask_point_sample_ratio = 16

        # Real head output: spatial_features is already projected by sparse_forward.
        real_dict = head.sparse_forward(spatial_features, [query_features], (mask_size, mask_size), skip_blocks=True)[0]
        torch.manual_seed(1)  # get_uncertain_point_coords_with_randomness draws torch.rand internally.
        real_losses = criterion.loss_masks({"pred_masks": real_dict}, targets, indices, num_boxes=torch.tensor(2.0))
        (real_losses["loss_mask_ce"] + real_losses["loss_mask_dice"]).backward()
        assert head.spatial_features_proj.weight.grad is not None
        assert torch.isfinite(head.spatial_features_proj.weight.grad).all()

        # Corrupted dict: same query_features/bias, but spatial_features is the RAW, unprojected
        # tensor — exactly what the pre-fix skip_blocks=True branch used to hand loss_masks.
        with torch.no_grad():
            raw_resized = torch.nn.functional.interpolate(
                spatial_features, size=(mask_size, mask_size), mode="bilinear", align_corners=False
            )
        corrupted_dict = {
            "spatial_features": raw_resized,
            "query_features": real_dict["query_features"].detach(),
            "bias": real_dict["bias"].detach(),
        }
        torch.manual_seed(1)  # same point-sampling draw as the real-dict call above, for a fair comparison.
        corrupted_losses = criterion.loss_masks(
            {"pred_masks": corrupted_dict}, targets, indices, num_boxes=torch.tensor(2.0)
        )

        assert not torch.allclose(real_losses["loss_mask_ce"], corrupted_losses["loss_mask_ce"])
        assert not torch.allclose(real_losses["loss_mask_dice"], corrupted_losses["loss_mask_dice"])


def _build_encoder_only_features(batch_size: int, hidden_dim: int, size: int) -> list[NestedTensor]:
    """Build the single-level backbone feature map an ``LWDETR`` forward pass consumes.

    Examples:
        >>> len(_build_encoder_only_features(batch_size=1, hidden_dim=4, size=4))
        1
    """
    return [
        NestedTensor(
            torch.randn(batch_size, hidden_dim, size, size),
            torch.zeros(batch_size, size, size, dtype=torch.bool),
        )
    ]


class TestEncOutputsMaskLossThroughRealForwardPath:
    """No prior test drove a real ``SegmentationHead`` through the actual production path: ``LWDETR.forward()`` building
    ``enc_outputs["pred_masks"]`` with ``skip_blocks=True``, then ``SetCriterion.forward()``'s ``"enc_outputs"`` branch
    (``lwdetr.py``'s two-stage block, ``criterion.py``'s ``_enc``-suffixed losses).

    The other tests in this module and in ``test_matcher.py`` call ``SegmentationHead.sparse_forward()``,
    ``loss_masks()``, and ``HungarianMatcher.__call__()`` directly instead — real coverage of each unit, but none of
    them exercise the forward/criterion wiring that actually reaches ``enc_outputs`` in training, where ``two_stage``
    and ``aux_loss`` both default to ``True`` for every released ``RFDETRSeg*`` model.
    """

    def test_loss_mask_ce_enc_backprops_to_projection_weight(self) -> None:
        torch.manual_seed(0)
        batch_size, hidden_dim, num_queries, num_queries_enc, num_classes, mask_size = 1, 4, 2, 3, 3, 8

        backbone = MagicMock()
        backbone.return_value = (_build_encoder_only_features(batch_size, hidden_dim, size=4), [torch.zeros(1)], None)

        transformer = MagicMock()
        transformer.d_model = hidden_dim
        transformer.return_value = (
            torch.randn(1, batch_size, num_queries, hidden_dim),  # hs (1 decoder layer)
            torch.rand(1, batch_size, num_queries, 4) * 0.4 + 0.3,  # ref_unsigmoid
            torch.randn(batch_size, num_queries_enc, hidden_dim),  # hs_enc
            torch.rand(batch_size, num_queries_enc, 4) * 0.4 + 0.3,  # ref_enc
        )

        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=SegmentationHead(in_dim=hidden_dim, num_blocks=1, bottleneck_ratio=1, downsample_ratio=1),
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=False,
            group_detr=1,
            two_stage=True,
            bbox_reparam=False,
        )

        outputs = model(torch.randn(batch_size, 3, mask_size, mask_size))
        assert isinstance(outputs["enc_outputs"]["pred_masks"], dict)

        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=HungarianMatcher(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["masks"],
            group_detr=1,
            mask_point_sample_ratio=16,
        )
        targets: list[dict[str, Tensor]] = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.rand(1, 4) * 0.4 + 0.3,
                "masks": torch.rand(1, mask_size, mask_size),
            }
        ]

        losses = criterion(outputs, targets)

        assert "loss_mask_ce_enc" in losses
        (losses["loss_mask_ce_enc"] + losses["loss_mask_dice_enc"]).backward()
        proj_grad = model.segmentation_head.spatial_features_proj.weight.grad
        assert proj_grad is not None
        assert torch.isfinite(proj_grad).all()


class TestMatcherContract:
    """``target_side_safety`` is an optimization SetCriterion offers, not part of the matcher contract it requires."""

    def test_forward_supports_matcher_without_target_side_safety_kwarg(self) -> None:
        """A duck-typed matcher with a two-argument signature drives a full multi-call step without TypeError.

        The step deliberately carries a non-empty ``aux_outputs`` plus ``enc_outputs`` -- the shape that makes
        ``SetCriterion.forward`` want a precomputed safety value in the first place. A step without them short-circuits
        before any precompute and would pass even against an unguarded call site, proving nothing.
        """
        batch_size, num_queries, num_classes = 2, 4, 3
        layers = [
            {
                "pred_logits": torch.zeros(batch_size, num_queries, num_classes),
                "pred_boxes": torch.full((batch_size, num_queries, 4), 0.5),
            }
            for _ in range(3)
        ]
        outputs = {**layers[0], "aux_outputs": [layers[1]], "enc_outputs": layers[2]}
        targets = [
            {"labels": torch.zeros(2, dtype=torch.int64), "boxes": torch.full((2, 4), 0.5)} for _ in range(batch_size)
        ]
        criterion = _bare_criterion()
        criterion.matcher = _LegacyMatcherStub()

        assert criterion.forward(outputs, targets, num_boxes=1.0) == {}
