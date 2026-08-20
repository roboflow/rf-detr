# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Batched linear assignment solver for :class:`~rfdetr.models.matcher.HungarianMatcher`.

Wraps ``torch_linear_assignment`` behind a narrow, matcher-shaped API: bucket every problem that shares a target count
across every decoder/aux/enc layer of one training step into as few batched calls as possible, then hand back only the
small integer index pairs. See ``.plans/active/plan-native-linear-assignment.md`` Phase M3 Item 2 for the measurement
evidence and design rationale — bucketing across layers, not within one layer, is what clears the measured SciPy/GPU
crossover.

Backend selection is deliberately *not* re-implemented here: the dependency owns that policy (its Triton CUDA kernel on
Linux NVIDIA compute capability >= 8.0 with Torch >= 2.4, SciPy on everything else) and falls back internally, so this
module is device-agnostic and its results are identical everywhere.

``torch_linear_assignment`` ships in the ``train`` extra, and it is imported inside :func:`assign_many_bucketed` rather
than at module scope: ``rfdetr.models`` imports ``criterion`` -> ``matcher`` -> this module, so a module-scope import
would make a plain ``import rfdetr`` fail on an inference-only install that has no reason to solve an assignment problem
at all.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def assign_many_bucketed(
    cost_matrices: list[Tensor],
    sizes: list[int],
    group_detr: int,
) -> list[list[tuple[Tensor, Tensor]]]:
    """Solve every layer's compact cost matrix with as few batched device calls as possible.

    Buckets every ``(layer, group, image)`` problem by its real target count (``sizes[image]``)
    across *all* layers at once, since every layer sharing one call has the same ``targets`` and
    therefore the same ``sizes``. Bucketing across layers instead of solving one layer at a time is
    what pushes the per-bucket problem count (up to ``len(cost_matrices) * len(sizes) *
    group_detr``) past the measured SciPy/GPU crossover (~50 problems on an NVIDIA L4); see the plan
    doc's M3 Item 2 design notes.

    Examples:
        >>> import torch
        >>> costs = [torch.arange(8, dtype=torch.float32).reshape(4, 2)]
        >>> [(rows.tolist(), cols.tolist()) for rows, cols in assign_many_bucketed(costs, [1, 1], 1)[0]]
        [([0], [0]), ([0], [0])]

    Args:
        cost_matrices: One ``[num_queries, sum(sizes)]`` compact cost matrix per layer, all on the
            same device with the same dtype and shape, as built by
            ``HungarianMatcher._compute_compact_detection_cost_matrix`` /
            ``_compute_stacked_compact_cost_matrices``. Any device is accepted; a non-CUDA
            accelerator tensor is moved to the host first (see below).
        sizes: Each image's real (unpadded) target count, in batch order, shared by every layer.
        group_detr: Number of query groups; ``num_queries`` must be evenly divisible by it.

    Returns:
        One list per layer (``cost_matrices`` order) of per-image ``(row_indices, col_indices)``
        CPU int64 tensor pairs, group-concatenated exactly like
        ``HungarianMatcher._assign_compact_cost_matrix``.

    Raises:
        ValueError: If ``num_queries`` is not evenly divisible by ``group_detr``.
        ModuleNotFoundError: If ``torch_linear_assignment`` is missing, which means an install
            without the ``train`` extra reached a training-only code path.
    """
    # Optional-dependency boundary: see the module docstring -- importing this at module scope would
    # break `import rfdetr` for inference-only installs, which never reach this function.
    from torch_linear_assignment import (  # type: ignore[import-untyped,unused-ignore]
        assignment_to_indices,
        batch_linear_assignment,
    )

    # Query counts may differ between layers (`_match_many` accepts that), so each layer carries its
    # own group width rather than inheriting the first layer's.
    group_widths: list[int] = []
    for cost_matrix in cost_matrices:
        num_queries = cost_matrix.shape[0]
        if num_queries % group_detr != 0:
            raise ValueError(f"num_queries ({num_queries}) must be divisible by group_detr ({group_detr})")
        group_widths.append(num_queries // group_detr)

    # The dependency dispatches on `is_cuda`, and its non-CUDA branch hands the tensor straight to
    # `.numpy()` without moving it to the host, which raises on MPS. CUDA tensors are passed through
    # untouched so its own backend selection (and, on an unsupported GPU, its own host fallback)
    # applies; every other accelerator is moved here instead. CPU tensors are already a no-op.
    if not cost_matrices[0].is_cuda:
        cost_matrices = [cost_matrix.cpu() for cost_matrix in cost_matrices]

    target_offsets = [0]
    for size in sizes:
        target_offsets.append(target_offsets[-1] + size)

    # (layer_index, group_index, image_index) triples, bucketed by the full problem shape they share.
    # Both dimensions must match to stack: the target count comes from the image, the row count from
    # the layer's own group width. Equal shapes are also what keeps `assignment_to_indices` valid,
    # since it rejects a batch whose problems match a differing number of rows.
    buckets: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for image_index, size in enumerate(sizes):
        for layer_index, group_width in enumerate(group_widths):
            for group_index in range(group_detr):
                buckets[(group_width, size)].append((layer_index, group_index, image_index))

    # Per-(layer, image) slot for each group's solved pair, so the reassembly below can walk
    # group_index in ascending order regardless of the order buckets happen to be solved in.
    per_group_pairs: dict[tuple[int, int], list[tuple[Tensor, Tensor]]] = {
        (layer_index, image_index): [(torch.empty(0), torch.empty(0))] * group_detr
        for layer_index in range(len(cost_matrices))
        for image_index in range(len(sizes))
    }

    for (group_width, size), problems in buckets.items():
        stacked = torch.stack(
            [
                cost_matrices[layer_index][
                    group_index * group_width : (group_index + 1) * group_width,
                    target_offsets[image_index] : target_offsets[image_index] + size,
                ]
                for layer_index, group_index, image_index in problems
            ]
        )
        row_indices, col_indices = assignment_to_indices(batch_linear_assignment(stacked))
        row_indices = row_indices.cpu()
        col_indices = col_indices.cpu()
        for problem_index, (layer_index, group_index, image_index) in enumerate(problems):
            per_group_pairs[(layer_index, image_index)][group_index] = (
                row_indices[problem_index],
                col_indices[problem_index],
            )

    results: list[list[tuple[Tensor, Tensor]]] = []
    for layer_index, group_width in enumerate(group_widths):
        layer_result: list[tuple[Tensor, Tensor]] = []
        for image_index in range(len(sizes)):
            rows, cols = per_group_pairs[(layer_index, image_index)][0]
            for group_index in range(1, group_detr):
                next_rows, next_cols = per_group_pairs[(layer_index, image_index)][group_index]
                rows = torch.cat([rows, next_rows + group_width * group_index])
                cols = torch.cat([cols, next_cols])
            layer_result.append((rows, cols))
        results.append(layer_result)
    return results
