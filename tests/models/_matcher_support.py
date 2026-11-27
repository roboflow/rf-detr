# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Batch builders, path spies and index assertions shared by the ``HungarianMatcher`` test modules.

Import them by name, e.g. ``from tests.models._matcher_support import _assert_same_indices, _random_detection_batch``;
the shared ``matcher`` fixture lives in ``tests/models/conftest.py``.
"""

import pytest
import torch

from rfdetr.models.matcher import HungarianMatcher


def _spy_on_compact_path(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``_compute_compact_detection_cost_matrix`` to record how many times it runs, without changing its behavior —
    lets a test assert which branch ``forward`` actually took.

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[1, 2])
        >>> with pytest.MonkeyPatch.context() as patched:
        ...     calls = _spy_on_compact_path(patched)
        ...     _ = HungarianMatcher()(outputs, targets)
        >>> calls
        [1]
    """
    calls: list[int] = []
    original = HungarianMatcher._compute_compact_detection_cost_matrix

    def spy(
        self: HungarianMatcher,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append(1)
        return original(self, outputs, targets, *args, **kwargs)

    monkeypatch.setattr(HungarianMatcher, "_compute_compact_detection_cost_matrix", spy)
    return calls


def _random_detection_batch(
    seed: int, sizes: list[int], num_queries: int = 6, num_classes: int = 5
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Random detection-only outputs/targets with the given per-image target counts.

    Examples:
        >>> outputs, targets = _random_detection_batch(seed=1, sizes=[2, 0])
        >>> outputs["pred_logits"].shape, outputs["pred_boxes"].shape
        (torch.Size([2, 6, 5]), torch.Size([2, 6, 4]))
        >>> [len(target["labels"]) for target in targets]
        [2, 0]
    """
    torch.manual_seed(seed)
    bs = len(sizes)
    outputs = {
        "pred_logits": torch.randn(bs, num_queries, num_classes),
        "pred_boxes": torch.rand(bs, num_queries, 4) * 0.4 + 0.3,
    }
    targets = [
        {
            "labels": torch.randint(0, num_classes, (size,), dtype=torch.int64),
            "boxes": torch.rand(size, 4) * 0.4 + 0.3,
        }
        for size in sizes
    ]
    return outputs, targets


def _detection_batch_with_labels(
    seed: int, labels_per_image: list[list[int]], num_queries: int = 4, num_classes: int = 5
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Detection-only outputs/targets with exactly the given per-image class ids, so a test can place a non-finite logit
    in a class column that is consumed by its own image, consumed only by another image, or consumed by nobody.

    Examples:
        >>> outputs, targets = _detection_batch_with_labels(seed=1, labels_per_image=[[0, 1], [2, 3]])
        >>> outputs["pred_logits"].shape
        torch.Size([2, 4, 5])
        >>> [target["labels"].tolist() for target in targets]
        [[0, 1], [2, 3]]
    """
    torch.manual_seed(seed)
    batch_size = len(labels_per_image)
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4) * 0.4 + 0.3,
    }
    targets = [
        {
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.rand(len(labels), 4) * 0.4 + 0.3,
        }
        for labels in labels_per_image
    ]
    return outputs, targets


def _full_path_indices(
    matcher: HungarianMatcher,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Assignment the full path produces for these exact inputs, obtained by forcing the eligibility gate to ``False``
    inside a self-undoing monkeypatch context — the reference for every compact-vs-full equivalence assertion.

    Examples:
        >>> matcher = HungarianMatcher()
        >>> outputs, targets = _detection_batch_with_labels(seed=2, labels_per_image=[[0], [1]])
        >>> [(q.tolist(), t.tolist()) for q, t in _full_path_indices(matcher, outputs, targets)]
        [([1], [0]), ([3], [0])]
    """
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(HungarianMatcher, "_detection_inputs_are_safe", staticmethod(lambda o, t, s=None: False))
        return matcher(outputs, targets)


def _assert_same_indices(
    actual: list[tuple[torch.Tensor, torch.Tensor]], expected: list[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """Assert two per-image ``(query_indices, target_indices)`` assignments are element-for-element equal.

    Examples:
        >>> pair = [(torch.tensor([0]), torch.tensor([0]))]
        >>> _assert_same_indices(pair, [(torch.tensor([0]), torch.tensor([0]))])
    """
    assert len(actual) == len(expected)
    for image_idx, ((act_q, act_t), (exp_q, exp_t)) in enumerate(zip(actual, expected)):
        assert torch.equal(act_q, exp_q), f"query indices diverged for image {image_idx}"
        assert torch.equal(act_t, exp_t), f"target indices diverged for image {image_idx}"


def _assert_assignment_lengths(
    indices: list[tuple[torch.Tensor, torch.Tensor]], num_queries: int, sizes: list[int]
) -> None:
    """Assert every image matched exactly ``min(num_queries, size)`` query/target pairs.

    Guards a cost-only comparison against a degenerate empty assignment, which scores a total of ``0.0`` and would
    otherwise pass whatever it is compared against.

    Examples:
        >>> _assert_assignment_lengths([(torch.tensor([0, 3]), torch.tensor([1, 0]))], num_queries=6, sizes=[2])
    """
    assert len(indices) == len(sizes)
    for image_idx, ((query_indices, target_indices), size) in enumerate(zip(indices, sizes)):
        expected_length = min(num_queries, size)
        assert query_indices.shape == (expected_length,), f"query index count wrong for image {image_idx}"
        assert target_indices.shape == (expected_length,), f"target index count wrong for image {image_idx}"
