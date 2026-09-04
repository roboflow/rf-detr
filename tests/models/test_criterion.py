# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for SetCriterion edge paths: _output_device and num_boxes_for_targets."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

import rfdetr.models.criterion as criterion_module
from rfdetr.models.criterion import (
    SetCriterion,
    dice_loss,
    dice_loss_jit,
    sigmoid_ce_loss,
    sigmoid_ce_loss_jit,
)
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
    """Return a SetCriterion with no losses so forward() is a no-op.

    Examples:
        >>> isinstance(_bare_criterion(), SetCriterion)
        True
    """
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


class TestTargetMaskPointSampling:
    """Tests for the guarded direct sampling of matched ground-truth masks."""

    @pytest.mark.parametrize(
        ("height", "width", "groups", "expected_fallback_calls"),
        [
            pytest.param(24, 24, 13, 1, id="small-repeated-targets-fall-back"),
            pytest.param(256, 256, 2, 0, id="large-targets-use-direct-indexing"),
        ],
    )
    def test_matches_grid_sample_bitwise_and_routes_by_cost_guard(
        self,
        monkeypatch: pytest.MonkeyPatch,
        height: int,
        width: int,
        groups: int,
        expected_fallback_calls: int,
    ) -> None:
        """Direct and fallback paths match nearest ``grid_sample`` bit-for-bit."""
        num_targets = 8
        masks = torch.arange(num_targets * height * width, dtype=torch.int32).reshape(num_targets, height, width)
        masks = masks.remainder(251).to(torch.uint8)
        matched = torch.arange(num_targets).repeat(groups)
        indices = [(matched, matched)]
        generator = torch.Generator().manual_seed(0)
        point_coords = torch.rand((matched.numel(), 17, 2), generator=generator)
        point_coords[:, :4] = torch.tensor(
            [
                # 0.0/1.0 exactly unnormalize to an exact pixel-center tie for any mask size
                # (-0.5 / size-0.5) and are covered separately by
                # ``test_pixel_center_tie_corrects_only_the_tied_points``; these stay
                # close to the edges without landing on that tie, so this case still exercises
                # boundary clamping on the direct path.
                [0.001, 0.001],
                [0.999, 0.999],
                [-0.2, 1.2],
                [0.3, 0.7],
            ]
        )
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        assert fallback.call_count == expected_fallback_calls

    def test_many_small_image_groups_fall_back_despite_clearing_the_aggregate_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An aggregate element count above the cost floor is not enough if no single image clears it.

        The direct path pays a fixed per-image loop iteration cost (slicing, index computation, a device transfer, a
        gather). 16 images with 4 matches each of 64x64 masks total 1,048,576 elements -- far above
        ``_MIN_DIRECT_MASK_ELEMENTS`` (1048576) -- but each image alone contributes only 16,384 elements, well under the
        floor. Measured on this machine: taking the direct path here is ~4.4x SLOWER than ``point_sample``, a stable
        regression across 65536/262144/1048576-element totals, not machine noise -- the per-group floor below exists to
        keep the guard from taking it.
        """
        height = width = 64
        n_images = 16
        matches_per_image = 4
        masks = [torch.rand(2, height, width) for _ in range(n_images)]
        targets = [{"masks": image_masks} for image_masks in masks]
        matched = torch.arange(2).repeat(matches_per_image // 2)
        indices = [(matched, matched) for _ in range(n_images)]
        point_coords = torch.rand(n_images * matches_per_image, 64, 2)
        expected_masks = torch.cat([image_masks[matched] for image_masks in masks])
        expected = criterion_module.point_sample(
            expected_masks.unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points(targets, indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()

    def test_single_match_per_image_groups_fall_back_despite_large_masks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A per-group element floor is not enough if a single large mask clears it on its own.

        8 images with exactly 1 match each of 300x300 masks: each group's element count (90,000)
        clears ``_MIN_DIRECT_MASK_ELEMENTS`` (1048576) by itself, so the existing per-group element
        floor does not reject it -- but the direct path's fixed per-image loop overhead (slicing,
        index computation, a device transfer, a gather) is not amortized when a group has only one
        match to gather. Measured on this machine: the direct path here is ~1.25-1.5x SLOWER than
        ``point_sample``, a stable regression across 1-8 images, all with a single match per group --
        the per-group MATCH COUNT floor below (independent of mask resolution) exists to keep the
        guard from taking it.
        """
        height = width = 300
        n_images = 8
        masks = [torch.rand(1, height, width) for _ in range(n_images)]
        targets = [{"masks": image_masks} for image_masks in masks]
        matched = torch.zeros(1, dtype=torch.int64)
        indices = [(matched, matched) for _ in range(n_images)]
        point_coords = torch.rand(n_images, 380, 2)
        expected_masks = torch.cat([image_masks[matched] for image_masks in masks])
        expected = criterion_module.point_sample(
            expected_masks.unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points(targets, indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()

    def test_pixel_center_tie_corrects_only_the_tied_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A coordinate landing on an exact pixel-center tie is corrected via ``point_sample``, not a full fallback.

        ``x = 1 / 673`` unnormalizes to exactly ``0.5`` for ``width=673``, a tie where PyTorch's compiled
        ``grid_sample`` nearest-mode kernel does not agree with ``torch.round`` (verified: it does for
        ``width=96``, but not for ``width=673`` -- a float32 evaluation-order artifact of the kernel, not a
        simple rounding-convention mismatch). Without a per-point correction this mask/coordinate pair
        samples label ``0`` on the direct path where ``point_sample`` samples ``1``. A real point set of a
        few hundred points routinely contains a tie like this one (verified: 22/30 runs of a
        104-match x 380-point set did), so falling back for the WHOLE call whenever ANY point ties would
        give away most of the optimization for no reason -- only the tied points should pay the
        ``point_sample`` cost, not the other 152 points sampled in this call.
        """
        height, width = 560, 673
        masks = torch.zeros(8, height, width, dtype=torch.uint8)
        masks[:, :, 1] = 1  # column 1 is the only way to distinguish "rounded to 0" from "rounded to 1".
        matched = torch.arange(8)
        indices = [(matched, matched)]
        generator = torch.Generator().manual_seed(0)
        point_coords = torch.rand(8, 20, 2, generator=generator)
        point_coords[:, 0, 0] = 1.0 / 673.0  # ties every row's first point; the other 19 columns per row do not.
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()
        # The correction call must cover only the 8 tied points (one per row's column 0), not the full
        # 8x20 = 160-point batch -- that's the whole point of correcting in place instead of falling back.
        (correction_masks, correction_coords), _ = fallback.call_args
        assert correction_masks.shape[0] == 8
        assert correction_coords.shape == (8, 1, 2)

    def test_pixel_center_y_tie_corrects_only_the_tied_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A y-axis pixel-center tie is corrected without falling back for all sampled points."""
        height, width = 673, 560
        masks = torch.zeros(8, height, width, dtype=torch.uint8)
        masks[:, 1, :] = 1
        matched = torch.arange(8)
        indices = [(matched, matched)]
        generator = torch.Generator().manual_seed(0)
        point_coords = torch.rand(8, 20, 2, generator=generator)
        point_coords[:, 0, 1] = 1.0 / 673.0
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()
        (correction_masks, correction_coords), _ = fallback.call_args
        assert correction_masks.shape[0] == 8
        assert correction_coords.shape == (8, 1, 2)

    def test_float64_point_coords_preserve_fallback_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Float64 coordinates remain on the established fallback path."""
        masks = torch.rand(8, 96, 96)
        matched = torch.arange(8)
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 11, 2, dtype=torch.float64)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        with pytest.raises(RuntimeError, match="scalar type Float but found Double"):
            criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        fallback.assert_called_once()

    def test_noncontiguous_masks_preserve_fallback_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-contiguous masks remain on the established fallback path."""
        masks = torch.rand(8, 96, 96).transpose(1, 2)
        assert not masks.is_contiguous()
        matched = torch.arange(8)
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 11, 2)
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()

    def test_non_int64_target_indices_preserve_fallback_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-int64 matcher indices remain on the established fallback path."""
        masks = torch.rand(8, 96, 96)
        matched = torch.arange(8, dtype=torch.int32)
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 11, 2)
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()

    def test_positive_out_of_range_indices_preserve_fallback_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Positive out-of-range matcher indices remain on the established fallback path."""
        masks = torch.rand(8, 96, 96)
        matched = torch.tensor([0, 1, 2, 3, 4, 5, 6, 8])
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 11, 2)
        fallback_targets = [{"masks": masks}]
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        with pytest.raises(IndexError):
            criterion_module._sample_target_masks_at_points(fallback_targets, indices, point_coords)

        fallback.assert_not_called()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_point_coords_preserve_fallback_for_cpu_masks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA point coordinates with CPU masks remain on the established fallback path."""
        masks = torch.rand(8, 96, 96)
        matched = torch.arange(8)
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 11, 2, device="cuda")
        fallback = MagicMock(return_value=torch.zeros(8, 1, 11, device="cuda"))
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        fallback.assert_called_once()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_matcher_indices_preserve_fallback_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA matcher indices remain on the established fallback path."""
        masks = torch.rand(8, 96, 96, device="cuda")
        matched = torch.arange(8, device="cuda")
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 11, 2, device="cuda")
        fallback = MagicMock(return_value=torch.zeros(8, 1, 11, device="cuda"))
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        fallback.assert_called_once()

    def test_negative_indices_preserve_existing_fallback_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid negative advanced indices stay on the existing ``point_sample`` path."""
        height = width = 96
        masks = torch.rand(8, height, width)
        matched = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1])
        indices = [(matched, matched)]
        point_coords = torch.rand(matched.numel(), 11, 2)
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()

    def test_direct_path_preserves_multi_image_match_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-image direct samples concatenate in matcher order, including an empty middle image.

        Each non-empty group uses 16 matched targets at 256x256 (1,048,576 elements) so it clears the per-group cost
        floor on its own -- a group below that floor is expected to fall back regardless of the other
        groups' size (see ``test_matches_grid_sample_bitwise_and_routes_by_cost_guard``'s
        ``small-repeated-targets-fall-back`` case and the per-group guard in
        ``_sample_target_masks_at_points``).
        """
        torch.manual_seed(0)  # deterministic and, at this point count, verified to land no exact-tie coordinate.
        masks = [torch.rand(8, 256, 256), torch.empty(0, 256, 256), torch.rand(8, 256, 256)]
        first = torch.arange(8).repeat(2)
        empty = torch.empty(0, dtype=torch.int64)
        last = torch.tensor([7, 5, 1, 0, 6, 2, 4, 3]).repeat(2)
        indices = [(first, first), (empty, empty), (last, last)]
        point_coords = torch.rand(32, 19, 2)
        expected_masks = torch.cat((masks[0][first], masks[1][empty], masks[2][last]))
        expected = criterion_module.point_sample(
            expected_masks.unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points(
            [{"masks": image_masks} for image_masks in masks], indices, point_coords
        )

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_not_called()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_masks_use_fallback_until_cuda_path_is_benchmarked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA masks retain the fallback path until direct-route synchronization is benchmarked."""
        torch.manual_seed(0)  # deterministic and, at this point count, verified to land no exact-tie coordinate.
        masks = (torch.rand(8, 96, 96, device="cuda") > 0.5).contiguous()
        matched = torch.arange(8)
        indices = [(matched, matched)]
        point_coords = torch.rand(8, 17, 2, device="cuda")
        expected = criterion_module.point_sample(
            masks[matched].unsqueeze(1).float(),
            point_coords,
            align_corners=False,
            mode="nearest",
        ).squeeze(1)
        fallback = MagicMock(wraps=criterion_module.point_sample)
        monkeypatch.setattr(criterion_module, "point_sample", fallback)

        actual = criterion_module._sample_target_masks_at_points([{"masks": masks}], indices, point_coords)

        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        fallback.assert_called_once()

    def test_loss_masks_uses_guarded_target_sampler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tensor ``loss_masks`` path delegates target labels to the guarded sampler."""
        criterion = _bare_criterion()
        criterion.mask_point_sample_ratio = 16
        pred_masks = torch.randn(1, 8, 24, 24, requires_grad=True)
        targets = [{"masks": torch.rand(8, 96, 96) > 0.5}]
        matched = torch.arange(8)
        indices = [(matched, matched)]
        sampler = MagicMock(wraps=criterion_module._sample_target_masks_at_points)
        monkeypatch.setattr(criterion_module, "_sample_target_masks_at_points", sampler)

        losses = criterion.loss_masks({"pred_masks": pred_masks}, targets, indices, num_boxes=torch.tensor(8.0))
        (losses["loss_mask_ce"] + losses["loss_mask_dice"]).backward()

        sampler.assert_called_once()
        assert pred_masks.grad is not None


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


class TestMatchedTargetCache:
    """Matched labels and boxes are shared by all detection losses for one output layer."""

    def test_matched_targets_collects_each_target_field_once(self) -> None:
        """The cache preserves target ordering while providing one shared loss context.

        This prevents the criterion from reconstructing the same labels and boxes for classification and box losses
        independently. It fails before the cache exists.
        """
        criterion = _bare_criterion()
        targets = [
            {
                "labels": torch.tensor([2, 1], dtype=torch.int64),
                "boxes": torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.2, 0.3]]),
            },
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.2]]),
            },
        ]
        indices = [(torch.tensor([1]), torch.tensor([0])), (torch.tensor([0]), torch.tensor([0]))]

        matched = criterion._get_matched_targets(targets, indices)

        assert torch.equal(matched.source_indices[0], torch.tensor([0, 1]))
        assert torch.equal(matched.source_indices[1], torch.tensor([1, 0]))
        assert torch.equal(matched.labels, torch.tensor([2, 0]))
        assert torch.equal(matched.boxes, torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.5, 0.5, 0.1, 0.2]]))

    def test_forward_reuses_one_cached_call_per_layer_for_labels_and_boxes(self) -> None:
        """``forward()`` must call ``_get_matched_targets`` exactly once per output layer, not once per loss.

        The other test in this class only checks ``_get_matched_targets``'s own return values in isolation -- it would
        still pass if ``forward()`` stopped supplying the cache and ``loss_labels``/``loss_boxes`` rebuilt their matched
        tensors independently. This drives a real ``forward()`` call with ``losses=["labels", "boxes"]`` across three
        output layers (final, one aux, one enc) and spies on ``_get_matched_targets`` to prove one cached call serves
        both losses for each layer.
        """
        batch_size, num_queries, num_classes = 2, 4, 3
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=_MatcherStub(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["labels", "boxes"],
            group_detr=1,
        )
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4) * 0.4 + 0.3},
            {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4) * 0.4 + 0.3},
        ]

        def _layer() -> dict[str, torch.Tensor]:
            return {
                "pred_logits": torch.randn(batch_size, num_queries, num_classes),
                "pred_boxes": torch.rand(batch_size, num_queries, 4) * 0.4 + 0.3,
            }

        outputs = {**_layer(), "aux_outputs": [_layer()], "enc_outputs": _layer()}

        with patch.object(criterion, "_get_matched_targets", wraps=criterion._get_matched_targets) as spy:
            losses = criterion(outputs, targets, num_boxes=1.0)

        assert spy.call_count == 3, "one cached call per output layer (final + aux + enc), not per loss"
        for suffix in ("", "_0", "_enc"):
            assert f"loss_ce{suffix}" in losses
            assert f"loss_bbox{suffix}" in losses
            assert f"loss_giou{suffix}" in losses

    @pytest.mark.parametrize(
        "loss_flag",
        [
            pytest.param("ia_bce_loss", id="ia_bce_loss"),
            pytest.param("use_position_supervised_loss", id="use_position_supervised_loss"),
        ],
    )
    def test_loss_labels_matches_cached_and_recomputed_matched_targets(self, loss_flag: str) -> None:
        """``loss_labels`` must return the same ``loss_ce`` whether it recomputes ``idx``/``target_classes_o`` (and, for
        these two flags, ``target_boxes``) from ``targets``+``indices`` itself (``matched_targets=None``) or consumes a
        ``_MatchedTargets`` cache built from those same ``indices``.

        ``ia_bce_loss`` and ``use_position_supervised_loss`` are the only two ``loss_labels`` branches that read
        ``matched_targets.boxes``, and are mutually exclusive (``if``/``elif``). Elsewhere they were only ever checked
        for flag *forwarding* onto the criterion, never for loss correctness against a populated cache -- this pins that
        the cache is a pure optimization, not a behavior change.
        """
        batch_size, num_queries, num_classes = 2, 4, 3
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=_MatcherStub(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["labels"],
            group_detr=1,
            **{loss_flag: True},
        )
        torch.manual_seed(7)
        outputs = {
            "pred_logits": torch.randn(batch_size, num_queries, num_classes),
            "pred_boxes": torch.rand(batch_size, num_queries, 4) * 0.4 + 0.3,
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4) * 0.4 + 0.3},
            {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4) * 0.4 + 0.3},
        ]
        indices = [(torch.tensor([1, 3]), torch.tensor([0, 1])), (torch.tensor([2]), torch.tensor([0]))]
        num_boxes = torch.tensor(3.0)
        matched_targets = criterion._get_matched_targets(targets, indices)

        recomputed = criterion.loss_labels(outputs, targets, indices, num_boxes, log=False)
        cached = criterion.loss_labels(outputs, targets, indices, num_boxes, log=False, matched_targets=matched_targets)

        assert torch.allclose(recomputed["loss_ce"], cached["loss_ce"]), (
            f"{loss_flag}=True must compute the same loss_ce from a cached matched_targets as it does recomputing "
            "target_classes_o/idx/target_boxes from targets+indices directly"
        )


class TestBatchedFastPathRespectsMatcherOverrides:
    """The batched matcher fast path must decline whenever the matcher customizes matching."""

    def test_forward_uses_subclass_override_instead_of_batched_fast_path(self) -> None:
        """A ``HungarianMatcher`` subclass overriding ``forward`` must be called for every layer.

        The batched fast path reads ``_match_many`` off the matcher's class and calls it unbound, bypassing
        ``nn.Module.__call__``. Without the forward-identity veto, a subclass overriding ``forward`` (the sanctioned
        extension point) would have the base ``_match_many`` silently answer in its place instead, with no error.
        Driving a real multi-layer ``forward()`` call (final + aux + enc) through a logging subclass proves the override
        actually runs once per layer rather than being silently skipped.
        """
        call_log: list[int] = []

        class _LoggingMatcher(HungarianMatcher):
            def forward(self, outputs, targets, group_detr=1, target_side_safety=None):
                call_log.append(len(call_log))
                return super().forward(outputs, targets, group_detr=group_detr, target_side_safety=target_side_safety)

        batch_size, num_queries, num_classes = 2, 4, 3
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=_LoggingMatcher(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["labels", "boxes"],
            group_detr=1,
        )
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4) * 0.4 + 0.3},
            {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4) * 0.4 + 0.3},
        ]

        def _layer() -> dict[str, torch.Tensor]:
            return {
                "pred_logits": torch.randn(batch_size, num_queries, num_classes),
                "pred_boxes": torch.rand(batch_size, num_queries, 4) * 0.4 + 0.3,
            }

        outputs = {**_layer(), "aux_outputs": [_layer()], "enc_outputs": _layer()}

        criterion(outputs, targets, num_boxes=1.0)

        assert len(call_log) == 3, (
            "the overridden forward() must run once per output layer (final + aux + enc); a call count below 3 "
            "means the batched fast path silently bypassed the subclass override"
        )


class TestMaskLossDenominatorStaysOnDevice:
    """The JIT mask losses take their denominator as a Tensor, so the mask path never reads it back.

    ``loss_masks`` normalizes by ``num_boxes``, which is either ``num_boxes_for_targets``'s all-reduced (across
    distributed ranks) Tensor, or an explicit grad-accum-aware override a manual-optimization caller supplies instead --
    segmentation models train on Lightning's automatic-optimization path (``module_model.py``'s ``training_step`` calls
    ``self.criterion(outputs, targets)`` with no override), so in practice they get the former, not the latter. While
    ``dice_loss``/``sigmoid_ce_loss`` were TorchScripted with ``num_masks: float`` the caller had to unwrap that Tensor
    to a Python scalar, and on XLA every unwrap is a device-to-host sync that cuts the lazy graph.
    ``SetCriterion.forward`` calls ``loss_masks`` once per matched output layer (the final layer, every aux layer, and
    the enc layer), so a segmentation model's training step pays this sync several times, not once -- 5 times for
    SegNano/SegSmall (``dec_layers=4``), 6 for SegMedium/SegLarge (``dec_layers=5``), 7 for SegXLarge/Seg2XLarge
    (``dec_layers=6``).
    """

    def test_loss_masks_hands_the_jit_losses_the_num_boxes_tensor_unconverted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prove the *production* call site, not just the two JIT leaves in isolation.

        Before this fix, ``loss_masks`` called ``float(num_boxes)`` ONCE and reused that single Python float for both
        JIT calls -- one host read per ``loss_masks`` call, not two (and ``loss_masks`` itself runs once per matched
        output layer per training step, see the class docstring above).  Spying on the module-level JIT functions from
        ``loss_masks``'s real call site shows there is now no ``float()``/``.item()`` conversion anywhere on the path:
        the exact same ``num_boxes`` Tensor object reaches both calls untouched.
        """
        criterion = _bare_criterion()
        criterion.mask_point_sample_ratio = 16
        pred_masks = torch.randn(1, 8, 24, 24, requires_grad=True)
        targets = [{"masks": torch.rand(8, 96, 96) > 0.5}]
        matched = torch.arange(8)
        indices = [(matched, matched)]
        num_boxes = torch.tensor(8.0)
        dice_spy = MagicMock(wraps=criterion_module.dice_loss_jit)
        ce_spy = MagicMock(wraps=criterion_module.sigmoid_ce_loss_jit)
        monkeypatch.setattr(criterion_module, "dice_loss_jit", dice_spy)
        monkeypatch.setattr(criterion_module, "sigmoid_ce_loss_jit", ce_spy)

        criterion.loss_masks({"pred_masks": pred_masks}, targets, indices, num_boxes=num_boxes)

        dice_spy.assert_called_once()
        ce_spy.assert_called_once()
        assert dice_spy.call_args.args[2] is num_boxes
        assert ce_spy.call_args.args[2] is num_boxes

    def test_jit_signatures_declare_a_triple_typed_denominator(self) -> None:
        """Pin the scripted signature itself.

        TorchScript does not reject a Tensor passed for a ``float`` parameter -- it converts it inside the scripted
        function, which is exactly the host read this change removes.  Asserting on the value alone would therefore pass
        either way; the compiled schema is what actually distinguishes the two.  The signature accepts ``Union[Tensor,
        float, int]`` rather than ``Tensor`` alone, because ``dice_loss``/``sigmoid_ce_loss`` are re-exported from
        ``lwdetr.py`` as backward-compat symbols (``lwdetr.py``'s "Backward-compat re-exports" import block) -- an
        external caller of the old ``float``-only signature must keep working, and ``int`` is included because
        TorchScript's ``Union`` argument binding does not implicitly widen a Python ``int`` to ``float`` the way a
        plain single-typed ``float`` parameter does (see ``test_int_denominator_matches_the_pre_fix_signature`` below).
        """
        assert "Union(Tensor, float, int) num_masks" in str(dice_loss_jit.schema)
        assert "Union(Tensor, float, int) num_masks" in str(sigmoid_ce_loss_jit.schema)

    def test_int_denominator_matches_the_pre_fix_signature(self) -> None:
        """A bare Python ``int`` denominator must keep working, bit-for-bit against the pre-fix ``float``-only call.

        Before this PR, ``dice_loss_jit``/``sigmoid_ce_loss_jit`` declared ``num_masks: float``; TorchScript's binding
        for a single declared type widens a Python ``int`` to ``float`` implicitly, so ``dice_loss_jit(a, b, 5)``
        worked. A naive ``Union[Tensor, float]`` widening does NOT inherit that implicit int->float widening --
        TorchScript's ``Union`` argument binding requires an exact type match per member and rejects ``int`` outright
        with a ``RuntimeError`` (verified against this schema before ``int`` was added to the ``Union``). ``int`` must
        be its own explicit member of the ``Union`` for a bare-int caller to keep working.
        """
        torch.manual_seed(0)
        inputs = torch.randn(2, 16)
        targets = torch.randint(0, 2, (2, 16)).float()

        assert torch.equal(dice_loss_jit(inputs, targets, 5), dice_loss_jit(inputs, targets, 5.0))
        assert torch.equal(sigmoid_ce_loss_jit(inputs, targets, 5), sigmoid_ce_loss_jit(inputs, targets, 5.0))

    def test_numpy_scalar_denominator_is_a_documented_incompatibility(self) -> None:
        """A NumPy scalar (e.g. ``numpy.float32``) is not accepted, unlike the pre-fix ``float``-only signature.

        Under the old single-typed ``float`` signature, TorchScript's binding called a generic Python-to-double coercion
        that happened to also accept a NumPy scalar (or even a 0-d Tensor, silently reading it to the host). A ``Union``
        argument requires TorchScript to pick exactly one member without ambiguity, so it uses a strict type check per
        member instead of that generic coercion -- a NumPy scalar matches neither ``Tensor``, ``float``, nor ``int`` and
        is rejected. This is an inherent TorchScript ``Union``-binding limitation, not a choice made by this fix, and no
        caller inside this repository passes a NumPy scalar for this argument (production always converts through
        ``torch.as_tensor`` in ``SetCriterion.forward``, keeping this off the real training path). External callers of
        the re-exported ``dice_loss``/``sigmoid_ce_loss``/``_jit`` symbols must convert with ``float(...)`` first.
        """
        np = pytest.importorskip("numpy")
        torch.manual_seed(0)
        inputs = torch.randn(2, 16)
        targets = torch.randint(0, 2, (2, 16)).float()

        with pytest.raises(RuntimeError):
            dice_loss_jit(inputs, targets, np.float32(5.0))
        with pytest.raises(RuntimeError):
            sigmoid_ce_loss_jit(inputs, targets, np.float32(5.0))

    def test_jit_losses_accept_a_tensor_denominator(self) -> None:
        """The scripted and eager forms agree when handed an on-device denominator."""
        torch.manual_seed(0)
        inputs = torch.randn(2, 16)
        targets = torch.randint(0, 2, (2, 16)).float()
        denominator = torch.tensor(2.0)

        assert torch.allclose(dice_loss_jit(inputs, targets, denominator), dice_loss(inputs, targets, denominator))
        assert torch.allclose(
            sigmoid_ce_loss_jit(inputs, targets, denominator), sigmoid_ce_loss(inputs, targets, denominator)
        )

    def test_tensor_denominator_divides_exactly_as_before(self) -> None:
        """Normalizing by ``n`` must equal normalizing by 1 and dividing by ``n`` afterwards."""
        torch.manual_seed(0)
        inputs = torch.randn(3, 32)
        targets = torch.randint(0, 2, (3, 32)).float()
        one = torch.tensor(1.0)
        n = 5.0

        assert torch.allclose(dice_loss_jit(inputs, targets, torch.tensor(n)), dice_loss_jit(inputs, targets, one) / n)
        assert torch.allclose(
            sigmoid_ce_loss_jit(inputs, targets, torch.tensor(n)), sigmoid_ce_loss_jit(inputs, targets, one) / n
        )

    def test_float_denominator_still_matches_the_pre_fix_signature_bit_for_bit(self) -> None:
        """Backward compat for ``lwdetr.py``'s re-exports: a caller stuck on the old ``float`` signature must see the
        exact same numbers it always did, not merely "close" ones.

        Before this PR, ``dice_loss``/``sigmoid_ce_loss`` declared ``num_masks: float`` and divided by it directly.
        Wrapping that float in a Tensor before dividing (an earlier version of this fix did exactly that) changes the
        result under reduced precision, because ``tensor / python_float`` and ``tensor / tensor_wrapping_that_float``
        are not the same operation once the tensor's dtype has fewer mantissa bits than the float needs -- dividing by
        the wrapped Tensor quantizes the denominator to the tensor's dtype first, while dividing by the bare Python
        float does not.  So the fix must branch on ``num_masks``'s type and let the ``float`` branch divide by the
        unwrapped Python float exactly as the old code did, never materializing a Tensor for it.
        """
        torch.manual_seed(0)
        inputs = torch.randn(3, 47).to(torch.bfloat16)  # a non-power-of-two shape and a reduced dtype are
        targets = torch.randint(0, 2, (3, 47)).float()  # both required to expose a quantization regression
        denom = 17.0  # a non-power-of-two value: exact division by 2**k would hide a quantization bug

        dice_out = dice_loss_jit(inputs, targets, denom)
        ce_out = sigmoid_ce_loss_jit(inputs, targets, denom)

        # Reference: the literal pre-fix computation, with num_masks used as a bare Python float throughout.
        sig = inputs.sigmoid().flatten(1)
        numerator = 2 * (sig * targets).sum(-1)
        denominator = sig.sum(-1) + targets.sum(-1)
        expected_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / denom
        expected_ce = (
            torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none").mean(1).sum()
            / denom
        )

        assert torch.equal(dice_out, expected_dice)
        assert torch.equal(ce_out, expected_ce)

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
    def test_reduced_precision_predictions_with_production_shaped_targets_match_the_pre_fix_value(
        self, input_dtype: torch.dtype
    ) -> None:
        """The fix must not change loss values on the dtype combination training actually produces.

        ``_sample_target_masks_at_points`` always ends with ``.float()`` (criterion.py's point-sampling helper), so
        ``point_labels`` is ``float32`` in every real training step regardless of the model's autocast dtype --  only
        ``point_logits`` (the predictions) can be at a reduced dtype.  That asymmetry already promotes ``dice_loss``'s
        ``inputs * targets`` and ``sigmoid_ce_loss``'s ``binary_cross_entropy_with_logits`` to ``float32`` before the
        denominator is ever involved, so switching the denominator from a Python float to a Tensor changes neither the
        dtype nor the value on this, the only combination ``loss_masks`` actually feeds these functions.
        """
        torch.manual_seed(0)
        inputs = torch.randn(2, 16).to(input_dtype)
        targets = torch.randint(0, 2, (2, 16)).float()  # always float32, matching real point-label sampling
        denom = 2.0

        dice_new = dice_loss_jit(inputs, targets, torch.tensor(denom))
        ce_new = sigmoid_ce_loss_jit(inputs, targets, torch.tensor(denom))
        dice_old = dice_loss_jit(inputs, targets, denom)  # the pre-fix call shape: a bare Python float
        ce_old = sigmoid_ce_loss_jit(inputs, targets, denom)

        assert dice_new.dtype == torch.float32
        assert ce_new.dtype == torch.float32
        assert torch.equal(dice_new, dice_old)
        assert torch.equal(ce_new, ce_old)

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
    def test_reduced_precision_targets_are_a_documented_boundary_training_never_reaches(
        self, input_dtype: torch.dtype
    ) -> None:
        """Reducing BOTH inputs and targets -- a combination real training cannot produce, see the test above -- is the
        one case where the Tensor-denominator path's ``float32`` promotion measurably changes the value versus the old
        Python-float division, which stayed at the narrower dtype throughout.

        This pins that documented boundary without implying it is reachable from ``loss_masks``.
        """
        torch.manual_seed(0)
        inputs = torch.randn(2, 16).to(input_dtype)
        targets = torch.randint(0, 2, (2, 16)).float().to(input_dtype)
        denom = 2.0

        dice_new = dice_loss_jit(inputs, targets, torch.tensor(denom))
        ce_new = sigmoid_ce_loss_jit(inputs, targets, torch.tensor(denom))

        assert dice_new.dtype == torch.float32
        assert ce_new.dtype == torch.float32
        assert torch.allclose(dice_new, dice_loss(inputs, targets, torch.tensor(denom)))
        assert torch.allclose(ce_new, sigmoid_ce_loss(inputs, targets, torch.tensor(denom)))

    def test_tensor_denominator_now_carries_gradient_a_new_capability_not_a_regression(self) -> None:
        """Passing a ``requires_grad=True`` Tensor now backpropagates into the denominator; it never could before.

        Under the pre-fix ``float``-only signature, TorchScript's single-type binding accepted a Tensor too (by silently
        coercing it through the same generic Python-to-double path a NumPy scalar used, itself an undocumented host
        read), which detached it from the autograd graph -- the gradient was always ``None`` no matter what was passed,
        because a Tensor could never reach the function still carrying its graph connection. A ``Union[Tensor, float,
        int]`` denominator instead passes a Tensor through unchanged, so if that Tensor requires grad, the gradient now
        flows. No prior caller could have depended on the old dropped-gradient behavior for a Tensor input, because
        passing a Tensor through the scripted call boundary was never a documented, type-checked contract before this
        PR.
        """
        torch.manual_seed(0)
        inputs = torch.randn(2, 16)
        targets = torch.randint(0, 2, (2, 16)).float()
        denominator = torch.tensor(2.0, requires_grad=True)

        dice_loss_jit(inputs, targets, denominator).backward()

        assert denominator.grad is not None

    @pytest.mark.xla
    def test_denominator_is_not_read_back_to_the_host_on_xla(self) -> None:
        """No ``_local_scalar_dense`` and no ``aten::`` fallback: the whole call stays on device.

        Runs on any PJRT backend -- ``device.type`` is ``"xla"`` under ``PJRT_DEVICE=CPU`` too, which is all the
        host-sync counter depends on, so this needs no TPU silicon.
        """
        pytest.importorskip("torch_xla")
        import torch_xla
        import torch_xla.debug.metrics as met

        device = torch_xla.device()
        torch.manual_seed(0)
        inputs = torch.randn(2, 16, device=device)
        targets = torch.randint(0, 2, (2, 16), device=device).float()
        denominator = torch.tensor(2.0, device=device)

        # Warm up so one-off compilation transfers do not land in the measured counters.
        dice_loss_jit(inputs, targets, denominator)
        sigmoid_ce_loss_jit(inputs, targets, denominator)
        torch_xla.sync()

        met.clear_all()
        dice_loss_jit(inputs, targets, denominator)
        sigmoid_ce_loss_jit(inputs, targets, denominator)
        torch_xla.sync()

        assert met.counter_value("aten::_local_scalar_dense") is None
        assert [name for name in met.counter_names() if name.startswith("aten::")] == []
