# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Batched linear assignment solver for :class:`~rfdetr.models.matcher.HungarianMatcher`.

Wraps ``torch_linear_assignment`` behind a narrow, matcher-shaped API: fold every problem across
every decoder/aux/enc layer of one training step into a single batched call, then hand back only
the small integer index pairs. See ``.plans/active/plan-native-linear-assignment.md`` Phase M3
Item 2 for the measurement evidence and design rationale.

Two properties of the dependency drive the design, both measured on an L4 at RF-DETR shapes
(``bench_gpu_solver.py``, ``group_width=300``, targets 1-30):

* **The win scales with the batched problem count, not the problem size.** 52 problems measured
  1.15-1.37x SciPy (and 0.89-0.93x at 15-30 targets); 208 measured 2.0-2.5x; 624 measured ~5.9x.
  Splitting a step's problems across several calls therefore costs real throughput, so every
  problem in a step is padded to a common target count and solved in one call.
* **The convenience wrapper synchronizes.** ``assignment_to_indices`` reads a CUDA bool in an
  ``if``, calls ``.item()``, and runs ``nonzero``/``masked_select`` -- each a host-device stall,
  measured at ~8 per call. Since the match count is known analytically, this module converts
  assignments to index pairs with shape-static ops instead and pays a single fused transfer.

Backend selection is deliberately *not* re-implemented here: the dependency owns that policy (its
Triton CUDA kernel on Linux NVIDIA compute capability >= 8.0 with Torch >= 2.4, SciPy on everything
else) and falls back internally, so this module is device-agnostic and its results are identical
everywhere.

``torch_linear_assignment`` ships in the ``train`` extra, and it is imported inside the solve
helper rather than at module scope: ``rfdetr.models`` imports ``criterion`` -> ``matcher`` -> this
module, so a module-scope import would make a plain ``import rfdetr`` fail on an inference-only
install that has no reason to solve an assignment problem at all.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


def _solve_to_indices(stacked: Tensor, num_matches: int) -> tuple[Tensor, Tensor]:
    """Solve a uniform batch of problems and return host-side index pairs.

    Converts the solver's per-row assignment into ``(rows, cols)`` without any data-dependent
    device op: ``num_matches`` is known by construction, and a stable descending sort of the
    matched mask lists the matched rows in ascending order, which is what ``nonzero`` would have
    produced at the cost of a synchronization. The two index tensors cross the device boundary
    together as one transfer.

    Args:
        stacked: ``[num_problems, group_width, targets]`` costs, all problems the same shape.
        num_matches: Matches per problem, ``min(group_width, targets)``.

    Returns:
        ``(rows, cols)``, each a host ``[num_problems, num_matches]`` int64 tensor.

    Examples:
        >>> import torch
        >>> rows, cols = _solve_to_indices(torch.arange(4, dtype=torch.float32).reshape(1, 2, 2), 2)
        >>> rows.tolist(), cols.tolist()
        ([[0, 1]], [[0, 1]])
    """
    # Optional-dependency boundary: see the module docstring -- importing this at module scope
    # would break `import rfdetr` for inference-only installs, which never reach this function.
    from torch_linear_assignment import batch_linear_assignment  # type: ignore[import-untyped,unused-ignore]

    assignment = batch_linear_assignment(stacked)
    matched = assignment >= 0
    # Stable + descending puts every matched column first while preserving ascending row order.
    order = torch.argsort(matched.to(torch.int8), dim=1, descending=True, stable=True)
    rows = order[:, :num_matches]
    cols = assignment.gather(1, rows)
    # The single device-to-host transfer for the whole step.
    pair = torch.stack((rows, cols)).cpu()
    return pair[0], pair[1]


def _stack_padded(
    cost_matrices: list[Tensor],
    sizes: list[int],
    group_width: int,
    group_detr: int,
    max_size: int,
) -> Tensor:
    """Fold every ``(layer, image, group)`` problem into one uniformly shaped batch.

    Each image's target block is padded from its real count to ``max_size`` with a constant zero
    cost. Because ``group_width`` far exceeds ``max_size``, every feasible solution assigns all
    ``max_size`` columns exactly once, so a constant pad contributes the same amount to every
    arrangement and cannot change which assignment is optimal. Padded columns are discarded by the
    caller once the indices are back on the host.

    Args:
        cost_matrices: One ``[num_queries, sum(sizes)]`` compact cost matrix per layer.
        sizes: Each image's real target count, in batch order.
        group_width: Queries per group, shared by every layer here.
        group_detr: Number of query groups.
        max_size: Padded target count, ``max(sizes)``.

    Returns:
        ``[len(cost_matrices) * len(sizes) * group_detr, group_width, max_size]``, ordered
        layer-major, then image, then group.

    Examples:
        >>> import torch
        >>> _stack_padded([torch.ones(2, 3)], [1, 2], 2, 1, 2).shape
        torch.Size([2, 2, 2])
    """
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)

    per_layer: list[Tensor] = []
    for cost_matrix in cost_matrices:
        padded = cost_matrix.new_zeros(cost_matrix.shape[0], len(sizes), max_size)
        for image_index, size in enumerate(sizes):
            if size:
                padded[:, image_index, :size] = cost_matrix[:, offsets[image_index] : offsets[image_index] + size]
        # [group_detr, group_width, images, max_size] -> [images, group_detr, group_width, max_size]
        grouped = padded.view(group_detr, group_width, len(sizes), max_size).permute(2, 0, 1, 3)
        per_layer.append(grouped.reshape(len(sizes) * group_detr, group_width, max_size))
    return torch.cat(per_layer)


def _assign_padded(
    cost_matrices: list[Tensor],
    sizes: list[int],
    group_width: int,
    group_detr: int,
) -> list[list[tuple[Tensor, Tensor]]]:
    """Solve every layer in one batched call, then drop the padded columns on the host.

    Args:
        cost_matrices: One compact cost matrix per layer, all sharing ``group_width``.
        sizes: Each image's real target count, in batch order.
        group_width: Queries per group.
        group_detr: Number of query groups.

    Returns:
        One list per layer of per-image ``(row_indices, col_indices)`` CPU int64 tensor pairs.

    Examples:
        >>> import torch
        >>> costs = [torch.arange(8, dtype=torch.float32).reshape(4, 2)]
        >>> [(r.tolist(), c.tolist()) for r, c in _assign_padded(costs, [1, 1], 4, 1)[0]]
        [([0], [0]), ([0], [0])]
    """
    max_size = max(sizes)
    stacked = _stack_padded(cost_matrices, sizes, group_width, group_detr, max_size)
    rows, cols = _solve_to_indices(stacked, min(group_width, max_size))

    results: list[list[tuple[Tensor, Tensor]]] = []
    per_layer_problems = len(sizes) * group_detr
    for layer_index in range(len(cost_matrices)):
        layer_result: list[tuple[Tensor, Tensor]] = []
        for image_index, size in enumerate(sizes):
            row_parts: list[Tensor] = []
            col_parts: list[Tensor] = []
            for group_index in range(group_detr):
                problem = layer_index * per_layer_problems + image_index * group_detr + group_index
                problem_cols = cols[problem]
                # Padded columns sit at or above this image's real target count. Filtering on the
                # host keeps the device free of a data-dependent op.
                keep = problem_cols < size
                row_parts.append(rows[problem][keep] + group_width * group_index)
                col_parts.append(problem_cols[keep])
            layer_result.append((torch.cat(row_parts), torch.cat(col_parts)))
        results.append(layer_result)
    return results


def _assign_bucketed_by_size(
    cost_matrices: list[Tensor],
    sizes: list[int],
    group_widths: list[int],
    group_detr: int,
) -> list[list[tuple[Tensor, Tensor]]]:
    """Solve one batched call per distinct problem shape.

    The fallback for the two cases padding cannot serve: layers whose query counts differ, and
    target counts at or above ``group_width``, where ``min(group_width, targets)`` stops being the
    real target count and padded columns would no longer be separable from real ones.

    Args:
        cost_matrices: One compact cost matrix per layer.
        sizes: Each image's real target count, in batch order.
        group_widths: Each layer's queries per group.
        group_detr: Number of query groups.

    Returns:
        One list per layer of per-image ``(row_indices, col_indices)`` CPU int64 tensor pairs.

    Examples:
        >>> import torch
        >>> costs = [torch.arange(8, dtype=torch.float32).reshape(4, 2)]
        >>> [(r.tolist(), c.tolist()) for r, c in _assign_bucketed_by_size(costs, [1, 1], [4], 1)[0]]
        [([0], [0]), ([0], [0])]
    """
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)

    buckets: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for image_index, size in enumerate(sizes):
        for layer_index, group_width in enumerate(group_widths):
            for group_index in range(group_detr):
                buckets[(group_width, size)].append((layer_index, group_index, image_index))

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
                    offsets[image_index] : offsets[image_index] + size,
                ]
                for layer_index, group_index, image_index in problems
            ]
        )
        rows, cols = _solve_to_indices(stacked, min(group_width, size))
        for problem_index, (layer_index, group_index, image_index) in enumerate(problems):
            per_group_pairs[(layer_index, image_index)][group_index] = (rows[problem_index], cols[problem_index])

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


def assign_many_bucketed(
    cost_matrices: list[Tensor],
    sizes: list[int],
    group_detr: int,
) -> list[list[tuple[Tensor, Tensor]]]:
    """Solve every layer's compact cost matrix with as few batched device calls as possible.

    Folds every ``(layer, group, image)`` problem of one training step into a single call by
    padding to a common target count, since the solver's advantage grows steeply with the batched
    problem count. Callers are expected to route only CUDA work here; off CUDA the dependency
    resolves to the same SciPy solve behind extra bookkeeping. See the plan doc's M3 Item 2 notes.

    Examples:
        >>> import torch
        >>> costs = [torch.arange(8, dtype=torch.float32).reshape(4, 2)]
        >>> [(rows.tolist(), cols.tolist()) for rows, cols in assign_many_bucketed(costs, [1, 1], 1)[0]]
        [([0], [0]), ([0], [0])]

    Args:
        cost_matrices: One ``[num_queries, sum(sizes)]`` compact cost matrix per layer, all on the
            same device with the same dtype, as built by
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
    # Query counts may differ between layers (`_match_many` accepts that), so each layer carries
    # its own group width rather than inheriting the first layer's.
    group_widths: list[int] = []
    for cost_matrix in cost_matrices:
        num_queries = cost_matrix.shape[0]
        if num_queries % group_detr != 0:
            raise ValueError(f"num_queries ({num_queries}) must be divisible by group_detr ({group_detr})")
        group_widths.append(num_queries // group_detr)

    # The dependency dispatches on `is_cuda`, and its non-CUDA branch hands the tensor straight to
    # `.numpy()` without moving it to the host, which raises on MPS. CUDA tensors are passed
    # through untouched so its own backend selection (and, on an unsupported GPU, its own host
    # fallback) applies; every other accelerator is moved here instead. CPU tensors are a no-op.
    if not cost_matrices[0].is_cuda:
        cost_matrices = [cost_matrix.cpu() for cost_matrix in cost_matrices]

    max_size = max(sizes)
    if max_size == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return [[(empty, empty) for _ in sizes] for _ in cost_matrices]

    # Padding needs one shared group width, and needs `min(group_width, targets)` to still equal
    # the real target count so padded columns stay separable from real ones.
    if len(set(group_widths)) == 1 and max_size <= group_widths[0]:
        return _assign_padded(cost_matrices, sizes, group_widths[0], group_detr)
    return _assign_bucketed_by_size(cost_matrices, sizes, group_widths, group_detr)
