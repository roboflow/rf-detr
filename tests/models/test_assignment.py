# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for :mod:`rfdetr.models._assignment` and the ``HungarianMatcher`` batched assignment built on it.

``TestGpuAssignmentBucketing`` and ``TestGpuAssignmentPreservesEstablishedAssignments`` exercise
``assign_many_bucketed`` end-to-end against the matcher's SciPy path, and the ``TestBatchedDetectionMatching*``
classes check ``_match_many`` against matching each layer individually. The unit-level tests cover what the module
itself needs: the padding helper's block placement, the solver wrapper's index recovery, the routing decision between
the two solve paths, and the all-empty-batch branch neither of those matcher-level suites reaches.
"""

from __future__ import annotations

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from rfdetr.models import _assignment
from rfdetr.models.matcher import HungarianMatcher
from tests._markers import _requires_cuda
from tests.models._matcher_support import _assert_same_indices, _full_path_indices, _random_detection_batch


def _reference_indices(cost_matrix: torch.Tensor, sizes: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve one ungrouped compact cost matrix per-image with SciPy directly.

    Ground truth for ``group_detr=1`` cases: no query-group offsetting is involved, so each
    image's slice of the compact matrix is solved on its own.

    Args:
        cost_matrix: ``[num_queries, sum(sizes)]`` cost matrix for a single query group.
        sizes: Each image's target count, in batch order.

    Returns:
        Per-image ``(row_indices, col_indices)`` int64 tensor pairs.

    Examples:
        >>> costs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> [(r.tolist(), c.tolist()) for r, c in _reference_indices(costs, [1, 1])]
        [([0], [0]), ([1], [0])]
    """
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    results = []
    for image_index, size in enumerate(sizes):
        block = cost_matrix[:, offsets[image_index] : offsets[image_index] + size]
        rows, cols = linear_sum_assignment(block.numpy())
        results.append((torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)))
    return results


def _index_pairs(rows: torch.Tensor, cols: torch.Tensor) -> set[tuple[int, int]]:
    """Return the matched ``(row, col)`` pairs of one assignment as an order-independent set.

    Args:
        rows: Matched row indices.
        cols: Matched column indices, aligned with ``rows``.

    Returns:
        The set of ``(row, col)`` integer pairs.

    Examples:
        >>> sorted(_index_pairs(torch.tensor([1, 0]), torch.tensor([0, 1])))
        [(0, 1), (1, 0)]
    """
    return set(zip(rows.tolist(), cols.tolist()))


class TestSolveToIndices:
    """``_solve_to_indices`` must recover the same assignment SciPy would, as static-shaped indices."""

    def test_matches_scipy_linear_sum_assignment(self) -> None:
        """A single non-square problem's recovered indices equal SciPy's optimal assignment.

        ``group_width=3`` exceeds ``targets=2``, so only 2 of the 3 rows are matched; the row excluded from the optimal
        assignment must not appear in either returned index tensor.
        """
        cost = torch.tensor([[4.0, 1.0], [2.0, 0.5], [3.0, 3.0]])
        stacked = cost.unsqueeze(0)

        rows, cols = _assignment._solve_to_indices(stacked, num_matches=2)

        expected_rows, expected_cols = linear_sum_assignment(cost.numpy())
        assert _index_pairs(rows[0], cols[0]) == _index_pairs(
            torch.as_tensor(expected_rows), torch.as_tensor(expected_cols)
        )

    def test_batches_independent_problems_without_cross_contamination(self) -> None:
        """Two stacked problems with opposite optimal assignments are solved independently.

        Regression target: an indexing bug in the batched gather could silently apply one
        problem's assignment to another problem's rows.
        """
        cost_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        cost_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        stacked = torch.stack([cost_a, cost_b])

        rows, cols = _assignment._solve_to_indices(stacked, num_matches=2)

        assert _index_pairs(rows[0], cols[0]) == {(0, 0), (1, 1)}
        assert _index_pairs(rows[1], cols[1]) == {(0, 1), (1, 0)}


class TestStackPadded:
    """``_stack_padded`` must fold every layer/image/group problem into one zero-padded batch."""

    def test_pads_each_images_target_block_with_zero_cost(self) -> None:
        """Real cost values land at their offset; columns beyond an image's target count are zero."""
        cost_matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sizes = [1, 2]

        stacked = _assignment._stack_padded([cost_matrix], sizes, group_width=2, group_detr=1, max_size=2)

        assert stacked.shape == (2, 2, 2)
        assert torch.equal(stacked[0], torch.tensor([[1.0, 0.0], [4.0, 0.0]]))
        assert torch.equal(stacked[1], torch.tensor([[2.0, 3.0], [5.0, 6.0]]))

    def test_leaves_zero_size_image_as_an_all_zero_block(self) -> None:
        """An image with zero targets occupies its batch slot as an all-zero-cost problem.

        The ``if size:`` guard skips writing into that slot entirely, relying on the buffer's own zero-initialization --
        this pins that the resulting problem stays shaped and present in the batch rather than being silently dropped.
        """
        cost_matrix = torch.tensor([[9.0, 9.0]])
        sizes = [0, 2]

        stacked = _assignment._stack_padded([cost_matrix], sizes, group_width=1, group_detr=1, max_size=2)

        assert stacked.shape == (2, 1, 2)
        assert torch.equal(stacked[0], torch.zeros(1, 2))


class TestAssignPadded:
    """``_assign_padded`` must batch-solve every layer/image/group problem and drop the padding."""

    def test_matches_scipy_for_a_single_layer_single_group(self) -> None:
        """One layer, one query group: recovered indices equal SciPy's per-image assignment."""
        torch.manual_seed(11)
        cost_matrix = torch.rand(4, 5)
        sizes = [2, 3]

        [result] = _assignment._assign_padded([cost_matrix], sizes, group_width=4, group_detr=1)

        expected = _reference_indices(cost_matrix, sizes)
        for (rows, cols), (exp_rows, exp_cols) in zip(result, expected):
            assert _index_pairs(rows, cols) == _index_pairs(exp_rows, exp_cols)

    def test_offsets_row_indices_per_group_and_concatenates_them(self) -> None:
        """Multiple query groups: each group's rows are offset by ``group_index * group_width``.

        Compared against ``HungarianMatcher._assign_compact_cost_matrix``, the matcher's own reference for grouped
        compact-cost-matrix assignment.
        """
        torch.manual_seed(12)
        cost_matrix = torch.rand(6, 3)
        sizes = [1, 2]

        [result] = _assignment._assign_padded([cost_matrix], sizes, group_width=3, group_detr=2)

        expected = HungarianMatcher._assign_compact_cost_matrix(cost_matrix, sizes, group_detr=2)
        for (rows, cols), (exp_rows, exp_cols) in zip(result, expected):
            assert _index_pairs(rows, cols) == _index_pairs(exp_rows, exp_cols)


class TestAssignBucketedBySize:
    """``_assign_bucketed_by_size`` must batch by distinct problem shape and reassemble per layer."""

    def test_matches_scipy_when_targets_exceed_group_width(self) -> None:
        """Targets exceeding a layer's group width -- the case padding cannot serve -- still solves correctly."""
        torch.manual_seed(13)
        cost_matrix = torch.rand(4, 10)
        sizes = [3, 7]

        [result] = _assignment._assign_bucketed_by_size([cost_matrix], sizes, group_widths=[4], group_detr=1)

        expected = _reference_indices(cost_matrix, sizes)
        for (rows, cols), (exp_rows, exp_cols) in zip(result, expected):
            assert _index_pairs(rows, cols) == _index_pairs(exp_rows, exp_cols)

    def test_layers_with_different_group_widths_each_use_their_own(self) -> None:
        """Per-layer group widths route each layer's problems into their own shape bucket."""
        torch.manual_seed(14)
        cost_matrices = [torch.rand(4, 5), torch.rand(8, 5)]
        sizes = [2, 3]

        actual = _assignment._assign_bucketed_by_size(cost_matrices, sizes, group_widths=[4, 8], group_detr=1)

        expected = [_reference_indices(cost_matrix, sizes) for cost_matrix in cost_matrices]
        for actual_layer, expected_layer in zip(actual, expected):
            for (rows, cols), (exp_rows, exp_cols) in zip(actual_layer, expected_layer):
                assert _index_pairs(rows, cols) == _index_pairs(exp_rows, exp_cols)


class TestAssignManyBucketedRouting:
    """``assign_many_bucketed`` must route to the padding fast path only when padding is safe."""

    @pytest.mark.parametrize(
        ("shape", "sizes", "uses_padded"),
        [
            pytest.param((6, 5), [2, 3], True, id="uniform-width-covers-targets"),
            pytest.param((4, 10), [3, 7], False, id="targets-exceed-group-width"),
        ],
    )
    def test_routes_to_padded_only_when_group_width_covers_targets(self, mocker, shape, sizes, uses_padded) -> None:
        """A single shared group width covering ``max(sizes)`` takes the padding path; otherwise per-shape bucketing.

        Padding cannot serve ``max(sizes) > group_width``: ``min(group_width, targets)`` would stop equaling the real
        target count, so padded columns could no longer be told apart from real ones.
        """
        padded_spy = mocker.patch.object(_assignment, "_assign_padded", wraps=_assignment._assign_padded)
        bucketed_spy = mocker.patch.object(
            _assignment, "_assign_bucketed_by_size", wraps=_assignment._assign_bucketed_by_size
        )

        _assignment.assign_many_bucketed([torch.rand(*shape)], sizes, group_detr=1)

        assert padded_spy.call_count == int(uses_padded)
        assert bucketed_spy.call_count == int(not uses_padded)


class TestAssignManyBucketedEmptyBatch:
    """``assign_many_bucketed`` must short-circuit when every image has zero targets."""

    def test_returns_empty_index_pairs_for_every_layer_and_image(self) -> None:
        """All-zero ``sizes`` returns empty int64 index pairs without reaching either solve path.

        No detection cost matrix has any columns to assign here -- this pins the ``max(sizes) == 0`` branch, which
        neither the padding nor the bucketing path reaches.
        """
        cost_matrices = [torch.rand(4, 0), torch.rand(4, 0)]

        result = _assignment.assign_many_bucketed(cost_matrices, sizes=[0, 0, 0], group_detr=1)

        assert len(result) == 2
        for layer_result in result:
            assert len(layer_result) == 3
            for rows, cols in layer_result:
                assert rows.tolist() == []
                assert cols.tolist() == []
                assert rows.dtype == torch.int64
                assert cols.dtype == torch.int64


@pytest.mark.gpu
@_requires_cuda
class TestBatchedDetectionMatchingOnCUDA:
    """``_match_many``'s entire reason for existing -- solving every layer of a step together, under one host
    synchronization -- had zero test coverage under real CUDA kernels; every ``TestBatchedDetectionMatching`` case above
    uses CPU tensors, which never exercises the device path at all."""

    def test_match_many_matches_individual_detection_assignments_on_cuda(self) -> None:
        """Batched CUDA matching must reach the same per-image assignments as matching each layer individually.

        Mirrors ``TestBatchedDetectionMatching.test_match_many_matches_individual_detection_assignments`` but with
        CUDA tensors, so the device-resident cost build and batched solve are exercised under real CUDA kernels rather
        than assumed to behave like the CPU path.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=401, sizes=[2, 3])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
        layers = [outputs]
        for seed in (402, 403):
            layer, _ = _random_detection_batch(seed=seed, sizes=[2, 3])
            layers.append({key: value.cuda() for key, value in layer.items()})
        safety = matcher._precompute_target_side_safety(outputs, targets)

        actual = matcher._match_many(layers, targets, target_side_safety=safety)
        expected = [matcher(layer, targets, target_side_safety=safety) for layer in layers]

        assert actual is not None
        for actual_indices, expected_indices in zip(actual, expected):
            _assert_same_indices(actual_indices, expected_indices)

    def test_match_many_declines_non_finite_pred_boxes_on_cuda(self) -> None:
        """A CUDA layer with non-finite ``pred_boxes`` must still decline before anything reaches the solver.

        Proves the stacked safety reduction (every layer's ``target_safe & pred_safe`` plus cost-finiteness, reduced in
        one ``bool()``) actually catches an unsafe layer under real CUDA kernels -- not just on CPU, where no device
        synchronization is involved at all.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=408, sizes=[2, 3])
        outputs = {key: value.cuda() for key, value in outputs.items()}
        targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
        outputs["pred_boxes"][0, 0, 0] = float("nan")
        layer2, _ = _random_detection_batch(seed=409, sizes=[2, 3])
        layer2 = {key: value.cuda() for key, value in layer2.items()}

        assert matcher._match_many([outputs, layer2], targets) is None


class TestBatchedDetectionMatching:
    """Batching final, auxiliary, and encoder detection matchers must preserve every assignment."""

    def test_match_many_matches_individual_detection_assignments(self) -> None:
        """Three decoder/encoder-like outputs must produce their usual per-image assignments.

        This prevents batched solving from mixing a layer's cost matrix with a neighboring layer. It fails before the
        batched matching API exists.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=401, sizes=[2, 3])
        layers = [outputs]
        for seed in (402, 403):
            layer, _ = _random_detection_batch(seed=seed, sizes=[2, 3])
            layers.append(layer)
        safety = matcher._precompute_target_side_safety(outputs, targets)

        actual = matcher._match_many(layers, targets, target_side_safety=safety)
        expected = [matcher(layer, targets, target_side_safety=safety) for layer in layers]

        assert actual is not None
        for actual_indices, expected_indices in zip(actual, expected):
            _assert_same_indices(actual_indices, expected_indices)

    def test_match_many_declines_out_of_range_labels_for_full_path_fallback(self) -> None:
        """An invalid label must bypass batching so the legacy full path keeps its IndexError contract."""
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=404, sizes=[2, 3])
        targets[1]["labels"][0] = outputs["pred_logits"].shape[-1]

        assert matcher._match_many([outputs, outputs], targets) is None
        with pytest.raises(IndexError):
            matcher(outputs, targets)

    def test_match_many_declines_single_layer(self) -> None:
        """A single layer must use the per-layer matching path."""
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=405, sizes=[2, 3])

        assert matcher._match_many([outputs], targets) is None

    def test_match_many_declines_zero_targets(self) -> None:
        """A batch without targets must use the per-layer matching path."""
        matcher = HungarianMatcher()
        layer1, targets = _random_detection_batch(seed=406, sizes=[0, 0])
        layer2, _ = _random_detection_batch(seed=407, sizes=[0, 0])

        assert matcher._match_many([layer1, layer2], targets) is None

    def test_match_many_declines_cross_layer_num_classes_mismatch(self) -> None:
        """Two otherwise-compatible layers with different ``pred_logits`` class counts must decline batching.

        Every other ``TestBatchedDetectionMatching`` case uses homogeneous layer shapes, never exercising the cross-
        layer ``pred_logits.shape[-1] != reference_classes`` compatibility gate that guards the clamp used to build each
        layer's compact cost matrix.
        """
        matcher = HungarianMatcher()
        layer1, targets = _random_detection_batch(seed=410, sizes=[2, 3], num_classes=5)
        layer2, _ = _random_detection_batch(seed=411, sizes=[2, 3], num_classes=6)

        assert matcher._match_many([layer1, layer2], targets) is None

    def test_match_many_declines_cross_layer_pred_boxes_dtype_mismatch(self) -> None:
        """Two otherwise-compatible layers with different ``pred_boxes`` dtypes must decline batching.

        The compatibility gate checks ``pred_boxes.dtype`` equality across layers alongside ``pred_boxes.device``,
        ``pred_logits.device``, and ``pred_logits.shape[-1]`` -- only the ``num_classes`` branch had a dedicated test,
        leaving this branch unverified.
        """
        matcher = HungarianMatcher()
        layer1, targets = _random_detection_batch(seed=415, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=416, sizes=[2, 3])
        layer2["pred_boxes"] = layer2["pred_boxes"].double()

        assert matcher._match_many([layer1, layer2], targets) is None

    def test_match_many_declines_cross_layer_pred_logits_dtype_mismatch(self) -> None:
        """Two layers with different logit dtypes must not share one stacked cost construction.

        ``torch.cat`` promotes the logits when stacking, which subtly changes the focal classification cost for the
        lower-precision layer. Declining preserves the established per-layer calculation instead of accepting an
        assignment from mixed-precision arithmetic.
        """
        matcher = HungarianMatcher()
        layer1, targets = _random_detection_batch(seed=419, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=420, sizes=[2, 3])
        layer2["pred_logits"] = layer2["pred_logits"].double()

        assert matcher._match_many([layer1, layer2], targets) is None

    def test_match_many_declines_cross_layer_pred_logits_device_mismatch(self) -> None:
        """Two otherwise-compatible layers with different ``pred_logits`` devices must decline batching.

        The compatibility gate's ``pred_logits.device`` check is what item A5 (from the /oss:resolve PR #1361 review)
        added alongside the pre-existing ``pred_boxes`` dtype/device check -- it had no dedicated test. A ``meta``
        device is used only to make the device comparison itself differ; no tensor math runs on it, since the mismatch
        short-circuits ``_match_many`` before any compute.
        """
        matcher = HungarianMatcher()
        layer1, targets = _random_detection_batch(seed=417, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=418, sizes=[2, 3])
        layer2["pred_logits"] = layer2["pred_logits"].to("meta")

        assert matcher._match_many([layer1, layer2], targets) is None

    def test_match_many_declines_whole_batch_on_one_unsafe_layer(self) -> None:
        """One layer's non-finite ``pred_boxes`` must decline the entire batch, not just that layer.

        The only other unsafety test makes every layer unsafe simultaneously via a shared corrupted target, so it cannot
        distinguish a whole-batch decline from a partial one. Here the outer two layers stay safe and only the middle
        layer carries a non-finite box, pinning that the per-layer safety check inside the transferred-payload loop
        returns ``None`` for the whole call on its first failure rather than assigning the safe layers.
        """
        matcher = HungarianMatcher()
        layer1, targets = _random_detection_batch(seed=412, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=413, sizes=[2, 3])
        layer2["pred_boxes"][0, 0, 0] = float("nan")
        layer3, _ = _random_detection_batch(seed=414, sizes=[2, 3])

        assert matcher._match_many([layer1, layer2, layer3], targets) is None


class TestGpuAssignmentBucketing:
    """The bucketed batched solver must reproduce the SciPy per-layer loop exactly.

    ``assign_many_bucketed`` regroups every ``(layer, group, image)`` problem by target count, solves each bucket as one
    stacked call, then reassembles per-layer/per-image pairs with the group-query offsets applied. That bookkeeping —
    not the solver — is where a wrong answer would come from, and it is device-independent: the optional dependency's
    non-CUDA branch is SciPy, the same solver ``_assign_compact_cost_matrix`` calls. So these run on CPU, where CI and
    developer machines can actually catch a regression, rather than only on GPU hardware.
    """

    @pytest.mark.parametrize(
        ("sizes", "group_detr", "num_layers"),
        [
            pytest.param([2, 3], 1, 3, id="uniform-single-group"),
            pytest.param([1, 4, 2], 1, 2, id="mixed-sizes"),
            pytest.param([0, 2, 3], 1, 3, id="leading-empty-image"),
            pytest.param([2, 0], 2, 2, id="trailing-empty-image-two-groups"),
            pytest.param([3, 3], 3, 2, id="three-groups"),
            pytest.param([2, 2], 2, 1, id="single-layer"),
        ],
    )
    def test_matches_per_layer_scipy_assignment(self, sizes: list[int], group_detr: int, num_layers: int) -> None:
        """Bucketed batched assignment returns byte-identical indices to solving each layer with SciPy.

        Covers the combinations where the reassembly can go wrong: repeated vs mixed target counts (which decide how
        problems bucket), images with zero targets (whose empty column slice must still occupy its batch slot), more
        than one query group (whose row indices need a per-group offset), and a single layer (the degenerate bucket).
        """
        torch.manual_seed(900)
        num_queries = 6 * group_detr
        cost_matrices = [torch.rand(num_queries, sum(sizes)) for _ in range(num_layers)]

        actual = _assignment.assign_many_bucketed(cost_matrices, sizes, group_detr)

        expected = [
            HungarianMatcher._assign_compact_cost_matrix(cost_matrix, sizes, group_detr)
            for cost_matrix in cost_matrices
        ]
        assert len(actual) == len(expected)
        for layer_index, (actual_layer, expected_layer) in enumerate(zip(actual, expected)):
            assert len(actual_layer) == len(expected_layer), f"layer {layer_index} returned the wrong image count"
            _assert_same_indices(actual_layer, expected_layer)

    def test_layers_with_different_query_counts_use_their_own_group_width(self) -> None:
        """Layers whose query counts differ each keep their own group width instead of the first layer's.

        ``_match_many`` has always accepted layers with unequal query counts. Deriving one group width from
        ``cost_matrices[0]`` silently sliced every other layer at the wrong row offsets — producing a plausible but
        wrong assignment rather than an error — so this pins the per-layer derivation.
        """
        torch.manual_seed(905)
        sizes = [2, 3]
        cost_matrices = [torch.rand(4, sum(sizes)), torch.rand(8, sum(sizes))]

        actual = _assignment.assign_many_bucketed(cost_matrices, sizes, group_detr=2)

        expected = [
            HungarianMatcher._assign_compact_cost_matrix(cost_matrix, sizes, 2) for cost_matrix in cost_matrices
        ]
        for actual_layer, expected_layer in zip(actual, expected):
            _assert_same_indices(actual_layer, expected_layer)

    def test_rejects_group_detr_that_does_not_divide_queries(self) -> None:
        """An indivisible ``group_detr`` raises instead of silently mis-slicing the query dimension.

        Mirrors ``_assign_compact_cost_matrix``'s existing contract: slicing queries into unequal groups would produce a
        quietly wrong assignment rather than an error, so the failure has to be loud and it has to happen before any
        solve work is done.
        """
        cost_matrix = torch.rand(7, 4)

        with pytest.raises(ValueError) as exc_info:
            _assignment.assign_many_bucketed([cost_matrix], [2, 2], group_detr=2)

        assert "must be divisible by group_detr" in str(exc_info.value)


class TestGpuAssignmentPreservesEstablishedAssignments:
    """Routing the solve through the dependency must not change any assignment the matcher already produced.

    ``_match_many`` picks its solver by device: the batched dependency on CUDA, the SciPy loop everywhere else. These
    run on CPU, so they exercise the SciPy branch and pin that dropping the pinned host buffer left the assignments
    themselves untouched — the reorganisation must be behaviour-preserving, not merely working.
    """

    def test_forward_matches_the_full_cartesian_path(self, matcher: HungarianMatcher) -> None:
        """``forward``'s compact path still agrees with the untouched full-cartesian fallback.

        The full path is the reference implementation this optimization must never diverge from; it never routes through
        the dependency at all, so agreement here is an end-to-end check that the compact path's result is unchanged.
        """
        outputs, targets = _random_detection_batch(seed=901, sizes=[2, 3])

        actual = matcher(outputs, targets)

        _assert_same_indices(actual, _full_path_indices(matcher, outputs, targets))

    def test_match_many_matches_per_layer_forward(self, matcher: HungarianMatcher) -> None:
        """Batched multi-layer matching still equals matching each layer on its own.

        This is ``_match_many``'s core invariant and it must survive the removal of the pinned host buffer: batching
        layers together is an optimization, never a change in what each layer matches.
        """
        outputs, targets = _random_detection_batch(seed=902, sizes=[2, 3])
        layer2, _ = _random_detection_batch(seed=903, sizes=[2, 3])
        layers = [outputs, layer2]

        actual = matcher._match_many(layers, targets)

        assert actual is not None
        for actual_indices, layer in zip(actual, layers):
            _assert_same_indices(actual_indices, matcher(layer, targets))


@pytest.mark.gpu
@_requires_cuda
class TestGpuAssignmentOnCUDA:
    """The device solve must agree with the host SciPy solve under real CUDA execution.

    Solver choice is by device, so CUDA is the only place the batched dependency runs at all: every CPU test exercises
    the SciPy branch instead. That leaves the Triton kernel, the device-side safety reduction, and the index-only
    device-to-host transfer with no coverage outside this class, and makes these the genuine cross-backend parity checks
    — CUDA answers from the kernel, CPU answers from SciPy. ``ci-tests-gpu.yml`` selects ``-m gpu``.
    """

    def test_match_many_matches_the_cpu_solve_on_cuda(self) -> None:
        """A CUDA batch and its CPU copy produce identical assignments.

        The CUDA batch is solved by the Triton kernel and the CPU copy by SciPy, so this is the exact-parity check that
        the kernel's answer, the bucketing, and the group offsets all agree with the reference solver — a divergence
        would otherwise show up only as silently different training targets.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=910, sizes=[2, 3])
        layers = [outputs]
        for seed in (911, 912):
            layer, _ = _random_detection_batch(seed=seed, sizes=[2, 3])
            layers.append(layer)
        cuda_layers = [{key: value.cuda() for key, value in layer.items()} for layer in layers]
        cuda_targets = [{key: value.cuda() for key, value in target.items()} for target in targets]

        actual = matcher._match_many(cuda_layers, cuda_targets)

        expected = matcher._match_many(layers, targets)
        assert actual is not None
        assert expected is not None
        for actual_indices, expected_indices in zip(actual, expected):
            _assert_same_indices(actual_indices, expected_indices)

    def test_match_many_matches_the_cpu_solve_with_an_empty_image_on_cuda(self) -> None:
        """A batch containing a zero-target image matches the CPU solve exactly on CUDA.

        An empty image contributes a zero-width problem, which the solver short-circuits before reaching its kernel.
        That branch is otherwise only exercised on CPU, and a crowded batch is exactly where a mishandled empty slot
        would shift every following image's bucket position.
        """
        matcher = HungarianMatcher()
        outputs, targets = _random_detection_batch(seed=915, sizes=[0, 3, 2])
        layer2, _ = _random_detection_batch(seed=916, sizes=[0, 3, 2])
        layers = [outputs, layer2]
        cuda_layers = [{key: value.cuda() for key, value in layer.items()} for layer in layers]
        cuda_targets = [{key: value.cuda() for key, value in target.items()} for target in targets]

        actual = matcher._match_many(cuda_layers, cuda_targets)

        expected = matcher._match_many(layers, targets)
        assert actual is not None
        assert expected is not None
        for actual_indices, expected_indices in zip(actual, expected):
            _assert_same_indices(actual_indices, expected_indices)
