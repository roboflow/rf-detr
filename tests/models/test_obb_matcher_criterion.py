# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
from types import SimpleNamespace

import pytest
import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.models.matcher import HungarianMatcher

_BUILDER_ARGS = dict(
    device="cpu",
    two_stage=True,
    aux_loss=True,
    dec_layers=3,
    cls_loss_coef=1.0,
    bbox_loss_coef=5.0,
    giou_loss_coef=2.0,
    segmentation_head=None,
    use_grouppose_keypoints=False,
    sum_group_losses=False,
    focal_alpha=0.25,
    set_cost_class=2.0,
    set_cost_bbox=5.0,
    set_cost_giou=2.0,
    num_classes=15,
    ia_bce_loss=True,
    group_detr=13,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    num_select=300,
    use_focal_loss=True,
)


def _make_oriented_matcher() -> HungarianMatcher:
    return HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, oriented=True)


def _make_oriented_criterion(*, ia_bce_loss: bool = False) -> SetCriterion:
    matcher = _make_oriented_matcher()
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    return SetCriterion(
        num_classes=16,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        ia_bce_loss=ia_bce_loss,
    )


class TestOrientedWeightDict:
    """SetCriterion drops any loss key absent from weight_dict, without warning."""

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("loss_giou_enc", id="encoder-giou"),
            pytest.param("loss_kld", id="decoder-probiou"),
            pytest.param("loss_bbox", id="l1"),
        ],
    )
    def test_oriented_two_stage_weights_every_reported_loss(self, key: str) -> None:
        criterion, _ = build_criterion_and_postprocessors(SimpleNamespace(oriented=True, **_BUILDER_ARGS))
        assert criterion.weight_dict.get(key) is not None

    def test_non_oriented_weight_dict_has_no_probiou_term(self) -> None:
        """The oriented-only loss_kld key must not leak into axis-aligned training."""
        criterion, _ = build_criterion_and_postprocessors(SimpleNamespace(oriented=False, **_BUILDER_ARGS))
        assert "loss_kld" not in criterion.weight_dict
        assert criterion.weight_dict["loss_giou_enc"] == 2.0


class TestOrientedMatcher:
    def test_returns_valid_indices(self) -> None:
        matcher = _make_oriented_matcher()
        outputs = {
            "pred_logits": torch.randn(1, 10, 16),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        targets = [
            {
                "labels": torch.tensor([0, 3]),
                "boxes_obb": torch.tensor(
                    [
                        [0.5, 0.5, 0.2, 0.1, 0.3],
                        [0.3, 0.7, 0.15, 0.1, 1.0],
                    ]
                ),
            }
        ]
        indices = matcher(outputs, targets)
        assert len(indices) == 1
        src_idx, tgt_idx = indices[0]
        assert len(src_idx) == 2
        assert len(tgt_idx) == 2

    def test_oriented_flag_stored(self) -> None:
        matcher = _make_oriented_matcher()
        assert matcher.oriented is True

    def test_non_oriented_default(self) -> None:
        matcher = HungarianMatcher()
        assert matcher.oriented is False


class TestOrientedCriterion:
    def test_loss_boxes_returns_kld(self) -> None:
        criterion = _make_oriented_criterion()
        outputs = {
            "pred_logits": torch.randn(1, 10, 16),
            "pred_boxes": torch.rand(1, 10, 5) * 0.5 + 0.1,
        }
        outputs["pred_boxes"][..., 4] = outputs["pred_boxes"][..., 4] * math.pi
        targets = [
            {
                "labels": torch.tensor([0, 3]),
                "boxes_obb": torch.tensor(
                    [
                        [0.5, 0.5, 0.2, 0.1, 0.3],
                        [0.3, 0.7, 0.15, 0.1, 1.0],
                    ]
                ),
            }
        ]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=2)
        # 5D decoder boxes take the oriented branch: ProbIoU under the "loss_kld"
        # key, with no "loss_giou" term (which is registered in weight_dict only
        # for the non-oriented path).
        assert set(losses) == {"loss_bbox", "loss_kld"}
        assert losses["loss_bbox"].item() >= 0
        assert losses["loss_kld"].item() >= 0

    def test_loss_kld_is_zero_for_perfect_predictions(self) -> None:
        """A ProbIoU stub returning a constant would not distinguish these two cases."""
        criterion = _make_oriented_criterion()
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3], [0.3, 0.7, 0.15, 0.1, 1.0]])
        outputs = {"pred_logits": torch.randn(1, 2, 16), "pred_boxes": boxes[None]}
        targets = [{"labels": torch.tensor([0, 3]), "boxes_obb": boxes}]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=2)

        assert losses["loss_kld"].item() == pytest.approx(0.0, abs=1e-4)
        assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_kld_is_positive_for_mismatched_angle(self) -> None:
        """An angle-only error must still register — L1 does not see the angle."""
        criterion = _make_oriented_criterion()
        target_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.05, 0.0]])
        pred_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.05, math.pi / 2]])
        outputs = {"pred_logits": torch.randn(1, 1, 16), "pred_boxes": pred_boxes[None]}
        targets = [{"labels": torch.tensor([0]), "boxes_obb": target_boxes}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=1)

        assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)
        assert losses["loss_kld"].item() > 0.5

    def test_ia_bce_oriented_branch_runs(self) -> None:
        """The IA-BCE classification loss uses ProbIoU when oriented.

        criterion.py takes a separate oriented path here; the default fixture has ia_bce_loss=False, so this branch was
        previously never executed.
        """
        criterion = _make_oriented_criterion(ia_bce_loss=True)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3], [0.3, 0.7, 0.15, 0.1, 1.0]])
        outputs = {"pred_logits": torch.randn(1, 10, 16), "pred_boxes": torch.rand(1, 10, 5) * 0.5 + 0.1}
        outputs["pred_boxes"][0, :2] = boxes
        targets = [{"labels": torch.tensor([0, 3]), "boxes_obb": boxes}]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        losses = criterion.loss_labels(outputs, targets, indices, num_boxes=2)

        assert "loss_ce" in losses
        assert torch.isfinite(losses["loss_ce"]).all()

    def test_oriented_flag_propagated(self) -> None:
        criterion = _make_oriented_criterion()
        assert criterion.oriented is True

    def test_encoder_path_compares_against_the_axis_aligned_envelope(self) -> None:
        """4D encoder proposals must be scored against the target's envelope.

        A 0.4x0.1 box at 45 degrees has a 0.354x0.354 envelope. A prediction that matches that envelope exactly should
        incur ~0 L1; slicing [..., :4] instead would score it against the raw 0.4x0.1 sides and report a large error.
        """
        criterion = _make_oriented_criterion()
        angle = math.pi / 4
        target = torch.tensor([[0.5, 0.5, 0.4, 0.1, angle]])
        side = 0.4 * math.cos(angle) + 0.1 * math.sin(angle)
        pred = torch.tensor([[0.5, 0.5, side, side]])
        outputs = {"pred_logits": torch.randn(1, 1, 16), "pred_boxes": pred[None]}
        targets = [{"labels": torch.tensor([0]), "boxes_obb": target}]

        losses = criterion.loss_boxes(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))], num_boxes=1)

        assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-5)
        assert losses["loss_giou"].item() == pytest.approx(0.0, abs=1e-5)

    def test_encoder_matcher_prefers_the_envelope_match(self) -> None:
        """The matcher must rank an envelope-shaped proposal above a raw-sides one."""
        matcher = _make_oriented_matcher()
        angle = math.pi / 4
        side = 0.4 * math.cos(angle) + 0.1 * math.sin(angle)
        # Query 0 matches the raw rotated sides, query 1 matches the true envelope.
        pred = torch.tensor([[[0.5, 0.5, 0.4, 0.1], [0.5, 0.5, side, side]]])
        outputs = {"pred_logits": torch.zeros(1, 2, 16), "pred_boxes": pred}
        targets = [{"labels": torch.tensor([0]), "boxes_obb": torch.tensor([[0.5, 0.5, 0.4, 0.1, angle]])}]

        src_idx, _ = matcher(outputs, targets)[0]

        assert src_idx.tolist() == [1]

    def test_non_oriented_criterion(self) -> None:
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=91,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            focal_alpha=0.25,
            losses=["labels", "boxes"],
        )
        assert criterion.oriented is False
