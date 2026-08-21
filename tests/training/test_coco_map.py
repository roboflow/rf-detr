# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for RF-DETR's one-pass TorchMetrics COCO adapter."""

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision


def test_one_pass_metric_matches_noncontiguous_per_class_results() -> None:
    """One global evaluation must preserve class IDs and AP/AR values, including prediction-only classes."""
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 50.0, 50.0], [60.0, 60.0, 70.0, 70.0]]),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([3, 17, 29]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([3, 17]),
        }
    ]
    metric = OnePassCocoMeanAveragePrecision(class_metrics=True)

    metric.update(predictions, targets)
    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3, 17, 29], dtype=torch.int32))
    torch.testing.assert_close(
        result["map_per_class"].reshape(-1), torch.tensor([1.0, 0.0, -1.0]), rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        result["mar_100_per_class"].reshape(-1), torch.tensor([1.0, 0.0, -1.0]), rtol=1e-4, atol=1e-6
    )


def test_class_metrics_use_one_coco_evaluator() -> None:
    """Per-class AP and AR must come from the aggregate evaluator instead of constructing a second evaluator."""
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([3, 17]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([3, 17]),
        }
    ]
    metric = OnePassCocoMeanAveragePrecision(class_metrics=True)
    metric.update(predictions, targets)
    evaluator_factory = MagicMock(side_effect=metric._coco_backend.cocoeval)

    with patch.object(CocoBackend, "cocoeval", new_callable=PropertyMock, return_value=evaluator_factory):
        result = metric.compute()

    assert evaluator_factory.call_count == 1
    torch.testing.assert_close(result["map_per_class"], torch.ones(2), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(result["mar_100_per_class"], torch.ones(2), rtol=1e-4, atol=1e-6)


def test_metric_update_owns_cpu_state_and_ignores_non_metric_fields() -> None:
    """The adapter must detach supported tensors on CPU without copying callback-only fields into metric state."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
    predictions = [
        {
            "boxes": boxes,
            "scores": torch.tensor([0.9], requires_grad=True),
            "labels": torch.tensor([3]),
            "keypoints": torch.ones(1, 1, 3),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([3]),
            "orig_size": torch.tensor([100, 100]),
        }
    ]
    metric = OnePassCocoMeanAveragePrecision()

    metric.update(predictions, targets)

    assert metric.has_updates is True
    assert all(value.device.type == "cpu" for value in metric.detection_box + metric.detection_scores)
    assert all(value.requires_grad is False for value in metric.detection_box + metric.detection_scores)
    assert "keypoints" not in metric.metric_state
    assert "orig_size" not in metric.metric_state


def test_class_metrics_disabled_keep_compact_sentinels() -> None:
    """Disabling class metrics must retain TorchMetrics sentinels without extended evaluator tensors."""
    metric = OnePassCocoMeanAveragePrecision(class_metrics=False)
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])}],
    )

    result = metric.compute()

    assert torch.equal(result["map_per_class"].reshape(-1), torch.tensor([-1.0]))
    assert torch.equal(result["mar_100_per_class"].reshape(-1), torch.tensor([-1.0]))
    assert {"ious", "precision", "recall", "scores"}.isdisjoint(result)


def test_bbox_and_segmentation_results_match_torchmetrics() -> None:
    """One-pass extraction must preserve aggregate and per-class values for both callback IoU types."""
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[:, 1:5, 1:5] = True
    predictions = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            "masks": mask.clone(),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([7]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
            "masks": mask.clone(),
            "labels": torch.tensor([7]),
        }
    ]
    kwargs = {
        "iou_type": ("bbox", "segm"),
        "class_metrics": True,
        "max_detection_thresholds": [1, 10, 25],
        "backend": "faster_coco_eval",
        "sync_on_compute": False,
    }
    expected_metric = MeanAveragePrecision(**kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(**kwargs)
    expected_metric.update(predictions, targets)
    actual_metric.update(predictions, targets)

    expected = expected_metric.compute()
    actual = actual_metric.compute()

    assert actual.keys() == expected.keys()
    assert actual["classes"].ndim == expected["classes"].ndim
    assert actual["bbox_map_per_class"].ndim == expected["bbox_map_per_class"].ndim
    assert actual["segm_map_per_class"].ndim == expected["segm_map_per_class"].ndim
    for key in actual:
        if key == "classes":
            assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1))
        else:
            torch.testing.assert_close(actual[key].reshape(-1), expected[key].reshape(-1), rtol=1e-4, atol=1e-6)


def test_adapter_matches_torchmetrics_on_nontrivial_multiclass_multiimage_data() -> None:
    """One-pass extraction must match stock TorchMetrics when every compared value is not a trivial 1.0/0.0/-1.0.

    ``test_bbox_and_segmentation_results_match_torchmetrics`` above uses one image, one class, one perfect
    match — every compared value collapses to a degenerate sentinel, so an axis-order slip or class-permutation
    bug in the adapter's one-pass reduction would still pass. This fixture spans three images and three
    non-contiguous classes with a partial-overlap match, a missed detection, and a false positive, so per-class
    AP/AR values are genuinely fractional and a reduction bug has real values to disagree on.
    """
    predictions = [
        {
            # Image 1: class 3 is a perfect match; class 17's IoU = 64 / (100 + 64 - 64) = 0.64 -- a
            # partial, not perfect, match.
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 28.0, 28.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([3, 17]),
        },
        {
            # Image 2: false positive -- class 42 has no ground truth anywhere in this fixture.
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([42]),
        },
        {
            # Image 3: class 17's true positive plus an extra false-positive detection for the same class.
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
        # Image 2: missed detection -- class 3 has ground truth here but no matching prediction.
        {"boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([17])},
    ]
    kwargs = {"class_metrics": True, "backend": "faster_coco_eval", "sync_on_compute": False}
    expected_metric = MeanAveragePrecision(**kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(**kwargs)
    expected_metric.update(predictions, targets)
    actual_metric.update(predictions, targets)

    expected = expected_metric.compute()
    actual = actual_metric.compute()

    assert actual.keys() == expected.keys()
    # Guard the fixture itself: if every per-class value below were 1.0/0.0/-1.0, this test would be as
    # degenerate as the one it complements, and an axis-order bug could still slip through undetected.
    assert set(expected["map_per_class"].tolist()) - {-1.0, 0.0, 1.0}, (
        "fixture produced only degenerate per-class values; strengthen it before trusting this parity check"
    )
    for key in actual:
        if key == "classes":
            assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1))
        else:
            torch.testing.assert_close(actual[key].reshape(-1), expected[key].reshape(-1), rtol=1e-4, atol=1e-6)


def test_empty_predictions_preserve_zero_recall() -> None:
    """A class with ground truth but no predictions must report zero AP/AR instead of a missing-value sentinel."""
    metric = OnePassCocoMeanAveragePrecision(class_metrics=True)
    metric.update(
        [{"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])}],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))
    torch.testing.assert_close(result["map_per_class"].reshape(-1), torch.tensor([0.0]), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(result["mar_100_per_class"].reshape(-1), torch.tensor([0.0]), rtol=1e-4, atol=1e-6)


def test_prediction_only_class_preserves_negative_sentinel() -> None:
    """A prediction-only class must remain present in class order with COCO's negative AP/AR sentinel."""
    metric = OnePassCocoMeanAveragePrecision(class_metrics=True)
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([29]),
            }
        ],
        [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.long)}],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([29], dtype=torch.int32))
    assert torch.equal(result["map_per_class"].reshape(-1), torch.tensor([-1.0]))
    assert torch.equal(result["mar_100_per_class"].reshape(-1), torch.tensor([-1.0]))


def test_empty_predictions_and_targets_return_compact_empty_class_result() -> None:
    """An updated image with no predictions or ground truth must finish with aggregate sentinels and no class IDs."""
    metric = OnePassCocoMeanAveragePrecision(class_metrics=True)
    metric.update(
        [{"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}],
        [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.long)}],
    )

    result = metric.compute()

    assert result["classes"].numel() == 0
    assert result["map_per_class"].numel() == 0
    assert result["mar_100_per_class"].numel() == 0
    assert float(result["map"]) == -1.0


def test_adapter_rejects_unsupported_result_and_backend_modes() -> None:
    """Unsupported modes must fail at construction instead of bypassing the adapter's memory and backend contract."""
    with pytest.raises(ValueError, match="extended_summary"):
        OnePassCocoMeanAveragePrecision(extended_summary=True)
    with pytest.raises(ValueError, match="faster_coco_eval"):
        OnePassCocoMeanAveragePrecision(backend="pycocotools")
    with pytest.raises(ValueError, match="average='macro'"):
        OnePassCocoMeanAveragePrecision(average="micro")
    with pytest.raises(ValueError, match="sync_on_compute=False"):
        OnePassCocoMeanAveragePrecision(sync_on_compute=True)


def test_adapter_rejects_stale_torchmetrics_state_contract() -> None:
    """Construction must fail loudly when the installed list-state schema no longer matches the adapter contract."""
    with (
        patch("rfdetr.training.coco_map._MAP_STATE_ATTRS", ("missing_state",)),
        pytest.raises(RuntimeError, match="incompatible with installed torchmetrics"),
    ):
        OnePassCocoMeanAveragePrecision()


def test_adapter_rejects_backend_method_missing_a_relied_on_parameter() -> None:
    """Construction must fail when a backend method drops a keyword compute() calls by name.

    Existence-only checks would miss an upstream rename of e.g. ``average``/``prefix`` on the installed backend helpers;
    the adapter's compute() calls those by keyword, so a silent rename would otherwise surface as a raw TypeError deep
    inside compute() instead of at construction.
    """
    with (
        patch(
            "rfdetr.training.coco_map._BACKEND_METHOD_PARAMS",
            {"_get_coco_datasets": ("not_a_real_parameter",), "_coco_stats_to_tensor_dict": ()},
        ),
        pytest.raises(RuntimeError, match="incompatible signature"),
    ):
        OnePassCocoMeanAveragePrecision()


class TestMismatchedBackendSignatures:
    """Unit coverage for the two contract-validation static helpers in isolation."""

    def test_flags_a_method_missing_a_relied_on_parameter(self) -> None:
        """A backend method lacking one of the required parameter names is reported as mismatched."""

        class _Backend:
            def _get_coco_datasets(self, groundtruth_labels, average) -> None:  # missing most params
                pass

        backend = _Backend()

        mismatched = OnePassCocoMeanAveragePrecision._mismatched_backend_signatures(backend, ["_get_coco_datasets"])

        assert mismatched == ["_get_coco_datasets"]

    def test_accepts_a_method_with_every_relied_on_parameter(self) -> None:
        """A backend method whose signature is a superset of the required names is not flagged."""

        class _Backend:
            def _coco_stats_to_tensor_dict(self, stats, prefix, max_detection_thresholds, extra=None) -> None:
                pass

        backend = _Backend()

        mismatched = OnePassCocoMeanAveragePrecision._mismatched_backend_signatures(
            backend, ["_coco_stats_to_tensor_dict"]
        )

        assert mismatched == []


class TestEvaluatorMethodsNowRequiringArgs:
    """Unit coverage for the evaluator zero-arg-method regression check."""

    def test_flags_a_method_that_gained_a_required_argument(self) -> None:
        """A previously zero-arg evaluator method that now requires an argument is reported."""

        class _Evaluator:
            def evaluate(self, threshold) -> None:
                pass

        mismatched = OnePassCocoMeanAveragePrecision._evaluator_methods_now_requiring_args(_Evaluator, ["evaluate"])

        assert mismatched == ["evaluate"]

    def test_accepts_a_method_that_stayed_zero_arg(self) -> None:
        """An evaluator method still callable with no arguments is not flagged."""

        class _Evaluator:
            def evaluate(self) -> None:
                pass

        mismatched = OnePassCocoMeanAveragePrecision._evaluator_methods_now_requiring_args(_Evaluator, ["evaluate"])

        assert mismatched == []

    def test_accepts_a_method_whose_new_argument_has_a_default(self) -> None:
        """A new parameter with a default value does not break zero-argument calls."""

        class _Evaluator:
            def evaluate(self, threshold=0.5) -> None:
                pass

        mismatched = OnePassCocoMeanAveragePrecision._evaluator_methods_now_requiring_args(_Evaluator, ["evaluate"])

        assert mismatched == []


def test_crowd_annotations_do_not_reduce_class_metrics() -> None:
    """A detection matched to crowd ground truth must remain ignored while the non-crowd match stays perfect."""
    metric = OnePassCocoMeanAveragePrecision(class_metrics=True)
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([3, 3]),
            }
        ],
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                "labels": torch.tensor([3, 3]),
                "iscrowd": torch.tensor([0, 1]),
            }
        ],
    )

    result = metric.compute()

    torch.testing.assert_close(result["map_per_class"].reshape(-1), torch.ones(1), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(result["mar_100_per_class"].reshape(-1), torch.ones(1), rtol=1e-4, atol=1e-6)


@patch("rfdetr.training.coco_map.get_world_size", return_value=2)
@patch("rfdetr.training.coco_map.is_dist_avail_and_initialized", return_value=True)
def test_distributed_merge_concatenates_every_metric_state(_initialized: MagicMock, _world_size: MagicMock) -> None:
    """The explicit DDP boundary must gather every list state once and mark empty ranks as updated."""
    metric = OnePassCocoMeanAveragePrecision()
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])}],
    )
    metric._update_count = 0

    with patch("rfdetr.training.coco_map.all_gather", side_effect=lambda local: [local, local]) as gather:
        metric.merge_distributed_state()

    assert gather.call_count == 9
    assert len(metric.detection_box) == 2
    assert len(metric.groundtruth_area) == 2
    assert metric.has_updates is True


def test_distributed_merge_is_noop_without_process_group() -> None:
    """Single-process use must leave local state untouched and issue no object gathers."""
    metric = OnePassCocoMeanAveragePrecision()
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])}],
    )

    with patch("rfdetr.training.coco_map.all_gather") as gather:
        metric.merge_distributed_state()

    gather.assert_not_called()
    assert len(metric.detection_box) == 1


@patch("rfdetr.training.coco_map.get_world_size", return_value=1)
@patch("rfdetr.training.coco_map.is_dist_avail_and_initialized", return_value=True)
def test_distributed_merge_is_noop_for_world_size_one(_initialized: MagicMock, _world_size: MagicMock) -> None:
    """An initialized one-rank group must not perform redundant object gathers."""
    metric = OnePassCocoMeanAveragePrecision()

    with patch("rfdetr.training.coco_map.all_gather") as gather:
        metric.merge_distributed_state()

    gather.assert_not_called()


def _distributed_empty_rank_worker(rank: int, world_size: int, init_file: str) -> None:
    """Verify one populated and one empty rank converge on identical global COCO metrics.

    Args:
        rank: Process rank launched by ``torch.multiprocessing``.
        world_size: Total process count.
        init_file: File-store path used to initialize the local Gloo process group.

    Examples:
        This worker requires a multi-process Gloo rendezvous and is exercised by the test below.  # doctest: +SKIP
        >>> _distributed_empty_rank_worker(0, 2, "/tmp/rfdetr-coco-map-rendezvous")  # doctest: +SKIP
    """
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        metric = OnePassCocoMeanAveragePrecision(class_metrics=True)
        if rank == 0:
            metric.update(
                [
                    {
                        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([3]),
                    }
                ],
                [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])}],
            )
        metric.merge_distributed_state()
        result = metric.compute()
        assert metric.has_updates is True
        assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))
        torch.testing.assert_close(result["map_per_class"].reshape(-1), torch.ones(1), rtol=1e-4, atol=1e-6)
    finally:
        dist.destroy_process_group()


# Windows CI currently cannot run this spawn test because gloo DDP spawn fails with
# makeDeviceForHostname unsupported-device errors (see tests/training/test_trainer_smoke.py).
@pytest.mark.skipif(sys.platform == "win32", reason="gloo DDP spawn unsupported on Windows CI")
def test_distributed_merge_supports_uneven_shards_with_empty_rank(tmp_path) -> None:
    """Two real Gloo ranks must finish without deadlock when only rank zero receives a metric update."""
    init_file = tmp_path / "coco-map-gloo-init"

    mp.spawn(_distributed_empty_rank_worker, args=(2, str(init_file)), nprocs=2, join=True)
