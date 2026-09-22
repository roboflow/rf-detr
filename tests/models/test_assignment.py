# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Direct unit tests for :mod:`rfdetr.models._assignment`.

``tests/models/test_matcher.py::TestGpuAssignmentBucketing`` and
``TestGpuAssignmentPreservesEstablishedAssignments`` already exercise
``assign_many_bucketed`` end-to-end against the matcher's SciPy path. This file adds the
unit-level coverage the module itself still lacked: the padding helper's block placement,
the solver wrapper's index recovery, the routing decision between the two solve paths, and
the all-empty-batch branch neither of those matcher-level suites reaches.
"""

from __future__ import annotations

from unittest.mock import patch

import torch
from scipy.optimize import linear_sum_assignment

from rfdetr.models import _assignment
from rfdetr.models.matcher import HungarianMatcher


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
        assert set(zip(rows[0].tolist(), cols[0].tolist())) == set(zip(expected_rows.tolist(), expected_cols.tolist()))

    def test_batches_independent_problems_without_cross_contamination(self) -> None:
        """Two stacked problems with opposite optimal assignments are solved independently.

        Regression target: an indexing bug in the batched gather could silently apply one
        problem's assignment to another problem's rows.
        """
        cost_a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        cost_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        stacked = torch.stack([cost_a, cost_b])

        rows, cols = _assignment._solve_to_indices(stacked, num_matches=2)

        assert set(zip(rows[0].tolist(), cols[0].tolist())) == {(0, 0), (1, 1)}
        assert set(zip(rows[1].tolist(), cols[1].tolist())) == {(0, 1), (1, 0)}


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
            assert set(zip(rows.tolist(), cols.tolist())) == set(zip(exp_rows.tolist(), exp_cols.tolist()))

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
            assert set(zip(rows.tolist(), cols.tolist())) == set(zip(exp_rows.tolist(), exp_cols.tolist()))


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
            assert set(zip(rows.tolist(), cols.tolist())) == set(zip(exp_rows.tolist(), exp_cols.tolist()))

    def test_layers_with_different_group_widths_each_use_their_own(self) -> None:
        """Per-layer group widths route each layer's problems into their own shape bucket."""
        torch.manual_seed(14)
        cost_matrices = [torch.rand(4, 5), torch.rand(8, 5)]
        sizes = [2, 3]

        actual = _assignment._assign_bucketed_by_size(cost_matrices, sizes, group_widths=[4, 8], group_detr=1)

        expected = [_reference_indices(cost_matrix, sizes) for cost_matrix in cost_matrices]
        for actual_layer, expected_layer in zip(actual, expected):
            for (rows, cols), (exp_rows, exp_cols) in zip(actual_layer, expected_layer):
                assert set(zip(rows.tolist(), cols.tolist())) == set(zip(exp_rows.tolist(), exp_cols.tolist()))


class TestAssignManyBucketedRouting:
    """``assign_many_bucketed`` must route to the padding fast path only when padding is safe."""

    def test_uses_padded_path_when_group_width_is_uniform_and_covers_targets(self) -> None:
        """A single shared group width with ``max(sizes) <= group_width`` takes the padding fast path."""
        cost_matrices = [torch.rand(6, 5)]

        with (
            patch.object(_assignment, "_assign_padded", wraps=_assignment._assign_padded) as padded_spy,
            patch.object(
                _assignment, "_assign_bucketed_by_size", wraps=_assignment._assign_bucketed_by_size
            ) as bucketed_spy,
        ):
            _assignment.assign_many_bucketed(cost_matrices, [2, 3], group_detr=1)

        padded_spy.assert_called_once()
        bucketed_spy.assert_not_called()

    def test_uses_bucketed_path_when_targets_exceed_the_shared_group_width(self) -> None:
        """A single group width but ``max(sizes) > group_width`` falls back to per-shape bucketing.

        Padding cannot serve this case: ``min(group_width, targets)`` would stop equaling the real
        target count, so padded columns could no longer be told apart from real ones.
        """
        cost_matrices = [torch.rand(4, 10)]

        with (
            patch.object(_assignment, "_assign_padded", wraps=_assignment._assign_padded) as padded_spy,
            patch.object(
                _assignment, "_assign_bucketed_by_size", wraps=_assignment._assign_bucketed_by_size
            ) as bucketed_spy,
        ):
            _assignment.assign_many_bucketed(cost_matrices, [3, 7], group_detr=1)

        bucketed_spy.assert_called_once()
        padded_spy.assert_not_called()


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
