# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared backend lists, state builders and assertions for the one-pass COCO metric tests (``test_coco_map*.py``).

``_BACKEND_ONLY`` lists every backend the adapter accepts, with the skip mark for the package it needs; the other
backend lists derive from it. Import them with ``from tests.training._coco_map_support import ...``.
"""

from typing import Any

import pytest
import torch

from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision
from tests._markers import _requires_faster_coco_eval, _requires_hotcoco, _requires_ufcoco, _requires_vernier

# Every backend the adapter accepts, each carrying the skip mark for the package it needs.
_BACKEND_ONLY = {
    "faster_coco_eval": _requires_faster_coco_eval,
    "hotcoco": _requires_hotcoco,
    "hotcoco_streaming": _requires_hotcoco,
    "ufcoco": _requires_ufcoco,
    "vernier": _requires_vernier,
}
_ALL_BACKENDS = [pytest.param(backend, marks=mark) for backend, mark in _BACKEND_ONLY.items()]
# The backends that replace the evaluation of a `faster_coco_eval`-named TorchMetrics backend.
_ALTERNATIVE_BACKENDS = [
    pytest.param(backend, marks=_BACKEND_ONLY[backend])
    for backend in ("hotcoco", "hotcoco_streaming", "ufcoco", "vernier")
]
# Streaming is disabled under DDP, so the real two-rank spawn would only exercise the batch path it falls back to.
_NON_STREAMING_BACKENDS = [
    pytest.param(backend, marks=mark) for backend, mark in _BACKEND_ONLY.items() if backend != "hotcoco_streaming"
]
# The backends that load box-only detections from one array instead of one dict per detection.
_BOX_ARRAY_BACKENDS = [pytest.param(backend, marks=_BACKEND_ONLY[backend]) for backend in ("hotcoco", "ufcoco")]
# `hotcoco_streaming` streams only when the model's class count is known; without it the metric silently evaluates in
# one batch, and a shared test would pass without ever touching the streaming path.
_BACKEND_KWARGS: dict[str, dict[str, Any]] = {"hotcoco_streaming": {"num_classes": 90}}


def _assert_streams_when_backend_streams(metric: OnePassCocoMeanAveragePrecision) -> None:
    """Assert a metric on a streaming backend really opened its stream, between ``update()`` and ``compute()``.

    A streaming backend that fell back to the batch path returns the same numbers, so a parity test cannot tell the two
    apart; this is what keeps such a test from passing vacuously. Call it before ``compute()``, which closes the stream.

    Examples:
        >>> metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")
        >>> _assert_streams_when_backend_streams(metric)
    """
    assert (metric._streams is not None) == metric._coco_backend.streams, (
        "a streaming backend must have opened its stream (is num_classes passed?), and a batch backend must not"
    )


def _assert_results_close(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], *, rtol: float = 0.0, atol: float = 0.0
) -> None:
    """Assert every ``actual`` metric matches ``expected`` flattened, class IDs exactly and the rest within tolerance.

    Examples:
        >>> result = {"classes": torch.tensor([3]), "map": torch.tensor([0.5])}
        >>> _assert_results_close(result, dict(result))
    """
    for key in actual:
        if key == "classes":
            assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1))
        else:
            torch.testing.assert_close(actual[key].reshape(-1), expected[key].reshape(-1), rtol=rtol, atol=atol)


def multiclass_detection_state() -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Return predictions and targets spanning three images and three non-contiguous classes.

    The fixture mixes a perfect match, a partial-overlap match, a missed detection and a false positive so that
    per-class AP and AR are genuinely fractional. Backend parity checks need that: a fixture whose every value
    collapses to 1.0/0.0/-1.0 would agree between two backends even if one of them reduced the wrong axis.

    Returns:
        The prediction and target lists in TorchMetrics detection format.

    Examples:
        >>> predictions, targets = multiclass_detection_state()
        >>> [len(predictions), len(targets), int(predictions[0]["labels"][0])]
        [3, 3, 3]
    """
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 28.0, 28.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([3, 17]),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([42]),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
            "scores": torch.tensor([0.6, 0.4]),
            "labels": torch.tensor([17, 17]),
        },
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([3, 17]),
        },
        {"boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([17])},
    ]
    return predictions, targets


def _metric_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return tied predictions, a crowd target, and images missing predictions or targets.

    Examples:
        >>> predictions, targets = _metric_inputs()
        >>> len(predictions), len(targets)
        (3, 3)
    """
    boxes = torch.tensor([[1.0, 2.0, 11.0, 12.0], [1.0, 2.0, 11.0, 12.0], [20.0, 20.0, 25.0, 25.0]])
    masks = torch.zeros((3, 32, 32), dtype=torch.bool)
    masks[:2, 2:12, 1:11] = True
    masks[2, 20:25, 20:25] = True
    predictions = [
        {
            "boxes": boxes,
            "masks": masks,
            "labels": torch.tensor([3, 3, 17]),
            "scores": torch.tensor([0.8, 0.8, 0.4]),
        },
        {
            "boxes": boxes[:0],
            "masks": masks[:0],
            "labels": torch.empty(0, dtype=torch.long),
            "scores": torch.empty(0),
        },
        {"boxes": boxes[:1], "masks": masks[:1], "labels": torch.tensor([29]), "scores": torch.tensor([0.5])},
    ]
    targets = [
        {
            "boxes": boxes[[0, 2]],
            "masks": masks[[0, 2]],
            "labels": torch.tensor([3, 17]),
            "iscrowd": torch.tensor([0, 1]),
            "area": torch.tensor([100.0, 25.0]),
        },
        {
            "boxes": boxes[:1],
            "masks": masks[:1],
            "labels": torch.tensor([3]),
            "iscrowd": torch.tensor([0]),
            "area": torch.tensor([100.0]),
        },
        {
            "boxes": boxes[:0],
            "masks": masks[:0],
            "labels": torch.empty(0, dtype=torch.long),
            "iscrowd": torch.empty(0, dtype=torch.long),
            "area": torch.empty(0),
        },
    ]
    return predictions, targets
