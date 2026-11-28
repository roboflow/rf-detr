# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``HungarianMatcher`` with masks present (the compact hybrid path and its device routing) and for the
compiled L1 matching callable."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.models import matcher as matcher_module
from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.matcher import HungarianMatcher
from tests._markers import _cuda_marks, _requires_cpu_inductor, _requires_cuda
from tests.models.matcher.helpers import (
    _assert_assignment_lengths,
    _assert_same_indices,
    _random_detection_batch,
    _spy_on_compact_path,
)


def _spy_on_mask_cost_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``_compute_mask_costs`` to record how many times it draws a fresh ``torch.rand`` point sample.

    The masks-hybrid path and the full-cartesian path both call ``_compute_mask_costs`` for a masks-present batch, but
    the hybrid path's own fallback (a non-finite combined cost) must reuse its already-drawn mask cost instead of
    calling this a second time, or the two computed indices would come from two different random point samples
    instead of the single draw the full path has always made for this batch.

    Examples:
        >>> outputs, targets = _random_segmentation_batch(seed=1, sizes=[1, 2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     calls = _spy_on_mask_cost_calls(patched)
        ...     _ = HungarianMatcher()(outputs, targets)
        >>> calls
        [1]
    """
    calls: list[int] = []
    original = HungarianMatcher._compute_mask_costs

    def spy(
        self: HungarianMatcher, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(1)
        return original(self, outputs, targets)

    monkeypatch.setattr(HungarianMatcher, "_compute_mask_costs", spy)
    return calls


def _random_segmentation_batch(
    seed: int, sizes: list[int], num_queries: int = 12, num_classes: int = 5, mask_size: int = 8
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Random segmentation (masks-present) outputs/targets with the given per-image target counts.

    Mirrors ``_random_detection_batch`` but adds a ``pred_masks`` tensor and per-target ``masks``,
    the shape ``TestMasksPresentCompactHybrid`` needs to exercise the masks-hybrid path.

    Examples:
        >>> outputs, targets = _random_segmentation_batch(seed=1, sizes=[2, 0])
        >>> outputs["pred_masks"].shape
        torch.Size([2, 12, 8, 8])
        >>> [len(target["masks"]) for target in targets]
        [2, 0]
    """
    torch.manual_seed(seed)
    bs = len(sizes)
    outputs = {
        "pred_logits": torch.randn(bs, num_queries, num_classes),
        "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
        "pred_masks": torch.randn(bs, num_queries, mask_size, mask_size),
    }
    targets = [
        {
            "labels": torch.randint(0, num_classes, (size,), dtype=torch.int64),
            "boxes": torch.rand(size, 4) * 0.4 + 0.3,
            "masks": torch.rand(size, mask_size, mask_size),
        }
        for size in sizes
    ]
    return outputs, targets


def _total_masks_present_assignment_cost(
    matcher: HungarianMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Total combined class/bbox/GIoU + mask cost an assignment achieves, scored on a single CPU cost matrix built
    from ``outputs``/``targets``.

    Mirrors ``_total_assignment_cost``'s device-tolerant comparison (score two assignments against one shared matrix
    instead of demanding identical index tensors), extended with the mask cost diagonal the masks-hybrid path itself
    adds — ``_total_assignment_cost`` alone omits it and would compare an incomplete objective for a masks-present
    batch. Builds exactly one cost matrix (one random mask point-sample draw) and scores every candidate assignment
    against it, so both sides see the same objective.

    Calls ``_compute_compact_detection_cost_matrix`` and ``_compute_mask_costs``, so it bumps ``_spy_on_compact_path``;
    call it only after any routing assertion.

    Examples:
        >>> matcher = HungarianMatcher()
        >>> outputs, targets = _random_segmentation_batch(seed=1, sizes=[2, 1])
        >>> round(_total_masks_present_assignment_cost(matcher, outputs, targets, matcher(outputs, targets)), 6)
        -0.593316
    """
    cpu_outputs = {key: value.cpu() for key, value in outputs.items()}
    cpu_targets = [{key: value.cpu() for key, value in target.items()} for target in targets]
    bs, num_queries = cpu_outputs["pred_logits"].shape[:2]
    sizes = [len(target["boxes"]) for target in cpu_targets]
    compact_class_bbox_giou = matcher._compute_compact_detection_cost_matrix(cpu_outputs, cpu_targets).float()
    cost_mask_ce, cost_mask_dice = matcher._compute_mask_costs(cpu_outputs, cpu_targets)
    mask_cost = (matcher.cost_mask_ce * cost_mask_ce + matcher.cost_mask_dice * cost_mask_dice).view(
        bs, num_queries, -1
    )
    target_offsets = [0]
    for size in sizes:
        target_offsets.append(target_offsets[-1] + size)
    mask_cost_diagonal = torch.cat(
        [mask_cost[index, :, target_offsets[index] : target_offsets[index + 1]] for index in range(bs)],
        dim=-1,
    )
    cost_matrix = compact_class_bbox_giou + mask_cost_diagonal
    target_offset = 0
    total = 0.0
    for (query_indices, target_indices), target in zip(indices, cpu_targets):
        total += float(cost_matrix[query_indices, target_indices + target_offset].sum())
        target_offset += len(target["boxes"])
    return total


def _hybrid_and_full_indices(
    monkeypatch: pytest.MonkeyPatch,
    matcher: HungarianMatcher,
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
    rng_seed: int,
    group_detr: int = 1,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Match one masks-present batch on the masks-hybrid route and again on the full-cartesian route, same RNG draw.

    The RNG is reseeded before each call because ``_compute_mask_costs`` draws ``point_coords`` with ``torch.rand``, and
    the two routes must sample the same points to be comparable. Asserts each route really ran via the compact spy.

    Examples:
        >>> outputs, targets = _random_segmentation_batch(seed=1, sizes=[2, 3])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     hybrid, full = _hybrid_and_full_indices(patched, HungarianMatcher(), outputs, targets, rng_seed=7)
        >>> len(hybrid), len(full)
        (2, 2)
    """
    monkeypatch.setattr(HungarianMatcher, "_mask_compact_device_supported", staticmethod(lambda o: True))
    monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 1)
    torch.manual_seed(rng_seed)
    hybrid_calls = _spy_on_compact_path(monkeypatch)
    hybrid = matcher(outputs, targets, group_detr=group_detr)
    assert hybrid_calls == [1], "test is only meaningful if the hybrid route actually ran"

    monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 10**9)
    torch.manual_seed(rng_seed)
    full_calls = _spy_on_compact_path(monkeypatch)
    full = matcher(outputs, targets, group_detr=group_detr)
    assert full_calls == [], "test is only meaningful if the comparison route is the full cartesian path"
    return hybrid, full


class TestMasksPresentCompactHybrid:
    """The masks-hybrid path (``_compact_mask_path_applicable`` + ``_mask_compact_worth_it``) computes class/bbox/GIoU
    through the compact per-image route while leaving the mask cost on the full cross-image matrix, extracting its
    diagonal blocks to match.

    It must agree with the pre-existing full-cartesian path exactly, and must only engage once the batch is large enough
    to be worth it.
    """

    @pytest.fixture()
    def forced_mask_hybrid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the masks-hybrid route eligible on any device with a one-element worth-it threshold."""
        monkeypatch.setattr(HungarianMatcher, "_mask_compact_device_supported", staticmethod(lambda o: True))
        monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 1)

    def test_below_threshold_uses_fallback_path_regardless_of_batch_size(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A masks-present batch that has not reached ``_MASK_COMPACT_SAVED_ELEMENT_LIMIT`` must still skip the compact
        route, however large ``batch_size`` alone is — the gate measures avoided cross-image entries, not ``batch_size``
        by itself."""
        monkeypatch.setattr(HungarianMatcher, "_mask_compact_device_supported", staticmethod(lambda o: True))
        monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 10**9)
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_segmentation_batch(seed=301, sizes=[2, 3, 1, 4])

        matcher(outputs, targets)

        assert calls == []

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_above_threshold_uses_compact_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """Once the avoided class/bbox/GIoU entries reach the threshold, the masks-present batch must reach the compact
        route for those terms."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_segmentation_batch(seed=302, sizes=[2, 3])

        matcher(outputs, targets)

        assert calls == [1]

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_combined_cost_preserves_full_path_addition_order(
        self, mocker: MockerFixture, matcher: HungarianMatcher
    ) -> None:
        """The hybrid matrix must add detection, mask-CE, and mask-Dice terms in the full path's exact order.

        These finite float32 values make ``detection + (mask_ce + mask_dice)`` differ by one ULP from ``(detection +
        mask_ce) + mask_dice``. Capturing the matrix at the assignment boundary pins the latter order independently of
        which assignment a larger randomized fixture happens to select.
        """
        outputs, targets = _random_segmentation_batch(seed=300, sizes=[1, 1], num_queries=1)
        detection_cost = torch.full((1, 2), 14.94161605834961, dtype=torch.float32)
        mask_ce = torch.tensor([[5.640276908874512, 0.0], [0.0, 5.640276908874512]], dtype=torch.float32)
        mask_dice = torch.tensor([[4.21762752532959, 0.0], [0.0, 4.21762752532959]], dtype=torch.float32)
        assignment = mocker.patch.object(
            matcher,
            "_assign_compact_cost_matrix",
            return_value=[(torch.tensor([0]), torch.tensor([0])), (torch.tensor([0]), torch.tensor([0]))],
        )
        mocker.patch.object(matcher, "_compute_compact_detection_cost_matrix", return_value=detection_cost)
        mocker.patch.object(matcher, "_compute_mask_costs", return_value=(mask_ce, mask_dice))

        matcher(outputs, targets)

        combined_cost = assignment.call_args.args[0]
        expected = (detection_cost + mask_ce.diagonal().unsqueeze(0)) + mask_dice.diagonal().unsqueeze(0)
        regrouped = detection_cost + (mask_ce.diagonal().unsqueeze(0) + mask_dice.diagonal().unsqueeze(0))
        assert not torch.equal(expected, regrouped), "the fixture must expose the float32 addition-order difference"
        assert torch.equal(combined_cost, expected)

    @pytest.mark.parametrize("seed", [401, 402, 403, 404, 405])
    @pytest.mark.parametrize(
        "sizes",
        [
            pytest.param([2, 3], id="uniform"),
            pytest.param([0, 3, 1], id="zero_target_image"),
            pytest.param([5, 1, 4, 2], id="heterogeneous_four_images"),
        ],
    )
    def test_matches_full_cartesian_path_exactly(
        self, monkeypatch: pytest.MonkeyPatch, seed: int, sizes: list[int]
    ) -> None:
        """The hybrid path's assignment must be IDENTICAL to the pre-existing full-cartesian path's, for the exact same
        random mask point sample.

        Rather than a hand-written reference (itself a place to introduce a second, independent bug), this compares the
        real ``forward()`` under the hybrid gate against the same real ``forward()`` with only
        ``_mask_compact_worth_it`` forced false — the only other thing that changes is which route computes
        class/bbox/GIoU. ``torch.manual_seed`` is reset immediately before each call because ``_compute_mask_costs``
        draws ``point_coords`` with ``torch.rand``, and the two routes must sample the same points to be comparable (see
        ``rfdetr-cudagraphs-l4-confirmed-12pct``-style lessons on this exact RNG trap in this matcher).
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_segmentation_batch(seed=seed, sizes=sizes)

        hybrid, full = _hybrid_and_full_indices(monkeypatch, matcher, outputs, targets, rng_seed=seed + 10_000)

        _assert_same_indices(hybrid, full)

    @pytest.mark.parametrize("group_detr", [2, 3])
    def test_matches_full_cartesian_path_exactly_with_group_detr(
        self, monkeypatch: pytest.MonkeyPatch, group_detr: int
    ) -> None:
        """The hybrid path's assignment must stay IDENTICAL to the full-cartesian path's under ``group_detr > 1`` -- the
        actual shape real segmentation TRAINING calls the matcher with.

        ``SetCriterion.forward`` passes ``group_detr=self.group_detr`` (the real configured group count, not 1) whenever
        ``self.training`` is True, and every other parity test in this class calls ``matcher(outputs, targets)`` without
        a ``group_detr`` keyword, which defaults to 1 -- the eval-mode shape, not the training-mode one this diff is
        meant to speed up.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_segmentation_batch(seed=311, sizes=[2, 3], num_queries=12)

        hybrid, full = _hybrid_and_full_indices(
            monkeypatch, matcher, outputs, targets, rng_seed=930, group_detr=group_detr
        )

        _assert_same_indices(hybrid, full)

    def test_matches_full_cartesian_path_exactly_with_dict_shaped_pred_masks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hybrid path's assignment must stay IDENTICAL to the full-cartesian path's when ``pred_masks`` is the
        dict-shaped projected mask head output (``spatial_features``/``query_features``/``bias``), not only the plain
        ``Tensor`` form every other parity test in this class uses.

        ``_compute_mask_costs``'s dict branch (``TestMatcherDictMaskCostUsesProjectedFeatures``) was previously
        exercised only through the full-cartesian path -- that existing coverage never ran with
        ``_MASK_COMPACT_SAVED_ELEMENT_LIMIT`` low enough to reach the hybrid gate at all.
        """
        torch.manual_seed(15)
        hidden, mask_size, num_queries = 4, 8, 12
        head = SegmentationHead(in_dim=hidden, num_blocks=1, bottleneck_ratio=1, downsample_ratio=1)
        sizes = [2, 3]
        outputs, targets = _random_detection_batch(seed=16, sizes=sizes, num_queries=num_queries)
        bs = len(targets)
        spatial_features = torch.randn(bs, hidden, mask_size, mask_size)
        query_features = torch.randn(bs, num_queries, hidden)
        for target, size in zip(targets, sizes):
            target["masks"] = torch.rand(size, mask_size, mask_size)
        outputs["pred_masks"] = head.sparse_forward(
            spatial_features, [query_features], (mask_size, mask_size), skip_blocks=True
        )[0]
        matcher = HungarianMatcher()

        hybrid, full = _hybrid_and_full_indices(monkeypatch, matcher, outputs, targets, rng_seed=920)

        _assert_same_indices(hybrid, full)

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_fixed_row_padding_uses_fallback_path_even_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A masks-present batch carrying ``"valid"`` (XLA's fixed-row target padding) must never reach the compact
        route, however large, since the padding sentinel's interaction with a per-column mask cost is unverified."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_segmentation_batch(seed=303, sizes=[2, 3])
        for target in targets:
            target["valid"] = torch.ones(len(target["boxes"]), dtype=torch.bool)

        matcher(outputs, targets)

        assert calls == []

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_keypoints_alongside_masks_uses_fallback_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A batch with both mask and keypoint targets (not a real model output today, but defensively excluded) must
        never reach the masks-hybrid compact route."""
        calls = _spy_on_compact_path(monkeypatch)
        num_keypoints, pred_dim = 3, 7
        keypoint_matcher = HungarianMatcher(num_keypoints_per_class=[num_keypoints])
        outputs, targets = _random_segmentation_batch(seed=304, sizes=[2, 3])
        outputs["pred_keypoints"] = torch.randn(2, 12, num_keypoints, pred_dim)
        for target in targets:
            target["keypoints"] = torch.rand(len(target["boxes"]), num_keypoints, 3)

        keypoint_matcher(outputs, targets)

        assert calls == []

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_non_finite_mask_cost_falls_through_to_full_path(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """A NaN in ``pred_masks`` (not swept by ``_detection_inputs_are_safe``, which only checks boxes/labels) must
        make the hybrid route's combined matrix non-finite and fall through to the sanitizing full-cartesian path,
        exactly like the detection-only compact path already does for an overflowing weighted cost."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_segmentation_batch(seed=305, sizes=[2, 3])
        outputs["pred_masks"][0, 0, 0, 0] = float("nan")

        results = matcher(outputs, targets)

        # The compact class/bbox/GIoU matrix is still attempted (it is finite on its own); only the
        # combined matrix, once the NaN mask cost is added in, is non-finite and triggers the fall-through.
        assert calls == [1]
        assert len(results) == 2

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_non_finite_mask_cost_reuses_the_hybrid_attempts_draw(
        self, monkeypatch: pytest.MonkeyPatch, matcher: HungarianMatcher
    ) -> None:
        """The fall-through triggered by a non-finite combined cost must reuse the hybrid branch's own mask-cost draw
        instead of calling ``_compute_mask_costs`` a second time.

        A second call would sample a different random ``point_coords`` than the single draw the full-cartesian path has
        always made for a masks-present batch -- silently changing the produced assignment relative to a batch that
        never attempted the hybrid route at all (``_MASK_COMPACT_SAVED_ELEMENT_LIMIT`` set high enough that the `elif`
        is never entered), even though both start from the same seed.

        The whole mask for image 0's query 0 (not just one pixel) is set to NaN, so the corrupted cost is non-finite for
        *every* possible ``point_coords`` draw -- the fall-through must trigger regardless of the seed, instead of
        depending on a random point sample happening to land on one bad pixel.
        """
        outputs, targets = _random_segmentation_batch(seed=305, sizes=[2, 3])
        outputs["pred_masks"][0, 0] = float("nan")

        compact_calls = _spy_on_compact_path(monkeypatch)
        mask_cost_calls = _spy_on_mask_cost_calls(monkeypatch)
        torch.manual_seed(900)
        hybrid_then_fallback = matcher(outputs, targets)
        assert compact_calls == [1], "test is only meaningful if the hybrid route was attempted first"
        assert mask_cost_calls == [1], "the fallback must reuse the hybrid attempt's draw, not sample a second time"

        monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 10**9)
        torch.manual_seed(900)
        never_attempted_hybrid = matcher(outputs, targets)

        _assert_same_indices(hybrid_then_fallback, never_attempted_hybrid)

    def test_worth_it_gate_uses_avoided_work_not_compact_matrix_size(self) -> None:
        """Batches with the same old gate value must route differently when their avoided work differs.

        Both cases have ``batch_size * num_queries * max(sizes) == 800,000``. The imbalanced batch removes only 40,000
        entries, while the balanced one removes 800,000.
        """
        outputs = {"pred_logits": torch.zeros(2, 20_000, 1)}
        imbalanced_targets = [{"boxes": torch.zeros(20, 4)}, {"boxes": torch.zeros(1, 4)}]
        balanced_targets = [{"boxes": torch.zeros(20, 4)}, {"boxes": torch.zeros(20, 4)}]

        assert HungarianMatcher._mask_compact_worth_it(outputs, imbalanced_targets) is False
        assert HungarianMatcher._mask_compact_worth_it(outputs, balanced_targets) is True

    @pytest.mark.parametrize(
        ("num_queries", "expected"),
        [(90_999, False), (91_000, True), (91_001, True)],
    )
    def test_worth_it_gate_boundary_at_the_real_default_threshold(self, num_queries: int, expected: bool) -> None:
        """``_mask_compact_worth_it`` must be inclusive (``>=``) at exactly the real deployed
        ``_MASK_COMPACT_SAVED_ELEMENT_LIMIT`` default.

        With two images carrying four targets each, each extra query avoids eight entries, so these are the nearest
        representable points below, at, and above 728,000 avoided entries.
        """
        outputs = {"pred_logits": torch.zeros(2, num_queries, 1)}
        targets = [{"boxes": torch.zeros(4, 4)}, {"boxes": torch.zeros(4, 4)}]

        assert HungarianMatcher._mask_compact_worth_it(outputs, targets) is expected

    @pytest.mark.parametrize(
        ("sizes", "expected"),
        [
            pytest.param([10] * 4, False, id="measured_batch4_neutral"),
            pytest.param([10] * 8, True, id="measured_batch8_winner"),
            pytest.param(
                [3, 2, 23, 12, 1, 13, 3, 8, 5, 11, 12, 2, 1, 3, 5, 4, 8, 1, 3, 16],
                True,
                id="measured_batch20_real_density_winner",
            ),
        ],
    )
    def test_worth_it_gate_matches_measured_workloads(self, sizes: list[int], expected: bool) -> None:
        """The neutral measured point must stay off while both measured winners stay on."""
        outputs = {"pred_logits": torch.zeros(len(sizes), 1_300, 1)}
        targets = [{"boxes": torch.zeros(size, 4)} for size in sizes]

        assert HungarianMatcher._mask_compact_worth_it(outputs, targets) is expected

    @pytest.mark.usefixtures("forced_mask_hybrid")
    def test_unsafe_inputs_fall_through_to_full_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A masks-present batch above the worth-it threshold must still fall through to the full cartesian path when
        ``_detection_inputs_are_safe`` reports the batch unsafe — the third predicate the hybrid gate ANDs together,
        alongside ``_compact_mask_path_applicable`` and ``_mask_compact_worth_it`` already covered above.

        Mirrors ``TestCompactPathRouting.test_overflowing_cost_weight_falls_through_to_fallback_path`` for the
        detection-only compact path: that test forces the same shared predicate false and compares against the
        fallback directly, but never exercised it with mask targets present, where the hybrid `elif` ANDs it with two
        more conditions instead of gating a plain `if`.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_segmentation_batch(seed=308, sizes=[2, 3])

        calls = _spy_on_compact_path(monkeypatch)
        torch.manual_seed(600)
        hybrid_result = matcher(outputs, targets)
        assert calls == [1], "test is only meaningful if the hybrid route actually ran first"

        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t, s=None: False))
        torch.manual_seed(600)
        fallback_result = matcher(outputs, targets)
        assert calls == [1], "the compact route must not be attempted again once inputs are reported unsafe"

        _assert_same_indices(hybrid_result, fallback_result)


@pytest.mark.gpu
@_requires_cuda
class TestMasksHybridPathOnCUDA:
    """The masks-hybrid path's CUDA branch (``_assignment.assign_many_bucketed``, gated on
    ``combined_cost_matrix.is_cuda``) had zero coverage under real CUDA kernels: every ``TestMasksPresentCompactHybrid``
    case above uses CPU tensors, which always takes the ``_assign_compact_cost_matrix`` branch instead.

    Mirrors ``TestCompactPathOnCUDA``, which covers the same gap for the detection-only compact path.
    """

    def test_matches_fallback_on_cuda_float32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under real CUDA kernels, the masks-hybrid route must reach an assignment as good as the full cartesian
        fallback's on the same device.

        The achieved total cost comparison preserves the existing CUDA tolerance for differently shaped kernels, and the
        assignment indices are also required to match for this concrete case.
        """
        monkeypatch.setattr(matcher_module, "_MASK_COMPACT_SAVED_ELEMENT_LIMIT", 1)
        matcher = HungarianMatcher()
        outputs, targets = _random_segmentation_batch(seed=309, sizes=[2, 4, 1])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]

        calls = _spy_on_compact_path(monkeypatch)
        torch.manual_seed(700)
        actual = matcher(outputs, targets)
        assert calls == [1], "test is only meaningful if the hybrid route actually ran"
        assert all(query.device.type == "cpu" for query, _ in actual), "assignment indices must return on CPU"
        _assert_assignment_lengths(actual, num_queries=12, sizes=[2, 4, 1])

        monkeypatch.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t, s=None: False))
        # Reseed before every `_compute_mask_costs` call: hybrid and fallback assignments need the same random point
        # draw, and both are then scored on one shared draw (see `TestCompactPathOnCUDA`) -- otherwise the comparison
        # mixes three independent random mask-cost matrices.
        torch.manual_seed(700)
        expected = matcher(outputs, targets)

        torch.manual_seed(701)
        actual_cost = _total_masks_present_assignment_cost(matcher, outputs, targets, actual)
        torch.manual_seed(701)
        expected_cost = _total_masks_present_assignment_cost(matcher, outputs, targets, expected)
        assert actual_cost == pytest.approx(expected_cost)
        _assert_same_indices(actual, expected)


class TestMasksHybridDeviceRouting:
    """The new performance route stays on the backend where its crossover was measured."""

    def test_cpu_batch_stays_on_full_cartesian_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A large CPU batch must retain the established full-cartesian path."""
        calls = _spy_on_compact_path(monkeypatch)
        outputs, targets = _random_segmentation_batch(seed=310, sizes=[20, 20], num_queries=20_000)

        HungarianMatcher()(outputs, targets)

        assert calls == []


class TestCompiledL1Matching:
    """The compiled L1 callable must preserve real compact and full-path assignments."""

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu", marks=_requires_cpu_inductor),
            pytest.param(
                "cuda",
                marks=_cuda_marks,
            ),
        ],
    )
    def test_dynamic_shapes_preserve_assignments(self, mocker: MockerFixture, device: str) -> None:
        """One compiled callable handles ragged, empty, and smaller last batches without changing matches."""
        eager = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        compiled = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        compiled.enable_compiled_l1_cost()
        real_compile = torch.compile
        spies: list[MagicMock] = []

        def spy_compile(fn, **kwargs):
            spies.append(MagicMock(wraps=real_compile(fn, **kwargs)))
            return spies[-1]

        mocker.patch.object(matcher_module.torch, "compile", side_effect=spy_compile)
        for sizes, queries in [([2, 0, 3], 8), ([5, 1], 12), ([0, 0], 8), ([1], 8), ([0], 8)]:
            outputs, targets = _random_detection_batch(seed=721, sizes=sizes, num_queries=queries)
            outputs = {key: value.to(device) for key, value in outputs.items()}
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            expected = eager(outputs, targets, group_detr=2)

            actual = compiled(outputs, targets, group_detr=2)

            _assert_same_indices(actual, expected)

        assert len(spies) == 1, "one lazily built callable must serve every batch"
        assert {call.args[0].ndim for call in spies[0].call_args_list} == {2, 3}

    def test_enable_defers_compilation_to_first_use_with_matcher_owned_recipe(self, mocker: MockerFixture) -> None:
        """Enabling records the request; ``torch.compile`` runs once, on the first call, with the matcher's recipe.

        The caller's Inductor options are forwarded, but matcher CUDA graphs stay off whatever the caller asked for. A
        request-only state is what keeps the matcher picklable for spawn-based launchers.
        """
        outputs, targets = _random_detection_batch(seed=725, sizes=[2, 1])
        matcher = HungarianMatcher()
        compiled_cost = MagicMock(wraps=matcher_module.pairwise_box_l1_cost)
        compiler = mocker.patch.object(matcher_module.torch, "compile", return_value=compiled_cost)

        matcher.enable_compiled_l1_cost({"triton.cudagraphs": True, "triton.coalesce_tiling_analysis": False})
        compiler.assert_not_called()
        matcher(outputs, targets)
        matcher(outputs, targets)

        # No fullgraph=True: hitting Dynamo's recompile limit must fall back to eager, never abort training.
        compiler.assert_called_once_with(
            matcher_module.pairwise_box_l1_cost,
            dynamic=True,
            options={"triton.coalesce_tiling_analysis": False, "triton.cudagraphs": False},
        )
        assert compiled_cost.call_count == 2

    def test_compiled_full_path_keeps_mixed_dtype_failure(self, mocker: MockerFixture) -> None:
        """A mismatched FP32/FP64 pair still fails eagerly instead of entering Dynamo -- or even building it."""
        outputs, targets = _random_detection_batch(seed=722, sizes=[2, 3])
        targets[1]["boxes"] = targets[1]["boxes"].double()
        matcher = HungarianMatcher()
        matcher.enable_compiled_l1_cost()

        compiler = mocker.patch.object(
            matcher_module.torch, "compile", side_effect=AssertionError("Invalid dtype pair reached compiler")
        )

        with pytest.raises(RuntimeError) as exc_info:
            matcher(outputs, targets)

        assert "expected scalar type Float but found Double" in str(exc_info.value)
        compiler.assert_not_called()
