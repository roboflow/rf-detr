# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for RF-DETR's one-pass TorchMetrics COCO adapter."""

import copy
import pickle
import sys
import warnings
from collections.abc import Callable
from typing import Any, get_args
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

from rfdetr.config import CocoEvalBackend
from rfdetr.training.coco_map import (
    _BACKENDS,
    OnePassCocoMeanAveragePrecision,
    _hotcoco,
    _ufcoco,
    _UfcocoBackend,
    _vernier,
    _vernier_thread_budget,
    _VernierBackend,
)

# Every backend the adapter accepts, with the package `pytest.importorskip` has to find for each.
_BACKEND_PACKAGES = {
    "faster_coco_eval": "faster_coco_eval",
    "hotcoco": "hotcoco",
    "ufcoco": "ultrafast_pycocotools",
    "vernier": "vernier",
}
_ALL_BACKENDS = list(_BACKEND_PACKAGES)
# The backends that replace the evaluation of a `faster_coco_eval`-named TorchMetrics backend.
_ALTERNATIVE_BACKENDS = ["hotcoco", "ufcoco", "vernier"]


def _require_backend(backend: str) -> None:
    """Skip the calling test when the package behind *backend* is not installed.

    Examples:
        >>> _require_backend("faster_coco_eval")
    """
    pytest.importorskip(_BACKEND_PACKAGES[backend])


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

    # Patched on the metric's own backend type rather than on CocoBackend: the hotcoco backend overrides the
    # property, so a patch applied to the base class would not be reached on the default path.
    with patch.object(
        type(metric._coco_backend), "cocoeval", new_callable=PropertyMock, return_value=evaluator_factory
    ):
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


class TestHoistedDetectionScores:
    """Prediction COCO datasets built with per-image score conversion instead of TorchMetrics' per-annotation read.

    Pinned to ``faster_coco_eval``: box-only evaluation on the default hotcoco backend loads detections from an array
    instead of building annotation dicts, so the hoist these tests describe only runs here and on the mask-only path.
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
    ]
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]), "labels": torch.tensor([3])},
    ]

    def _updated_metric(self) -> OnePassCocoMeanAveragePrecision:
        """Return a metric holding this class's two-image prediction and target state.

        Examples:
            >>> metric = TestHoistedDetectionScores()._updated_metric()
            >>> [scores.numel() for scores in metric.detection_scores]
            [2, 1]
        """
        metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", class_metrics=True)
        metric.update(self.predictions, self.targets)
        return metric

    def test_datasets_match_stock_construction(self) -> None:
        """Hoisted construction must produce the same COCO datasets TorchMetrics' own helper produces.

        The whole optimization rests on ``scores=None`` plus a second assignment pass being indistinguishable from the
        per-annotation read. Comparing the raw dataset dicts catches a divergence -- a dropped ``info`` key, a shifted
        ``annotation_id``, a misaligned score -- that aggregate mAP values could average away.
        """
        metric = self._updated_metric()
        expected_preds, expected_target = metric._coco_backend._get_coco_datasets(
            metric.groundtruth_labels,
            metric.groundtruth_box,
            metric.groundtruth_mask,
            metric.groundtruth_crowds,
            metric.groundtruth_area,
            metric.detection_labels,
            metric.detection_box,
            metric.detection_mask,
            metric.detection_scores,
            metric.iou_type,
            average=metric.average,
        )

        actual_preds, actual_target, _ = metric._coco_datasets(metric._observed_classes())

        assert actual_preds.dataset == expected_preds.dataset
        assert actual_target.dataset == expected_target.dataset

    def test_scores_are_converted_once_for_each_image(self) -> None:
        """Prediction scores must cross the CPU conversion boundary once per image.

        Dataset parity cannot detect a silent regression back to TorchMetrics' per-annotation score conversion. The two
        stored score tensors contain three detections, so recording two vector conversions proves this adapter converts
        scores once per image; a per-annotation implementation would instead record three scalar conversions.
        """
        metric = self._updated_metric()
        recorded = MagicMock(side_effect=metric._coco_backend._get_coco_format)
        original_cpu = torch.Tensor.cpu
        score_storage_addresses = {scores.untyped_storage().data_ptr() for scores in metric.detection_scores}
        score_conversion_shapes: list[tuple[int, ...]] = []

        def record_cpu(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if tensor.untyped_storage().data_ptr() in score_storage_addresses:
                score_conversion_shapes.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with (
            patch.object(CocoBackend, "_get_coco_format", recorded),
            patch.object(torch.Tensor, "cpu", new=record_cpu),
        ):
            metric._coco_datasets(metric._observed_classes())

        prediction_call = recorded.call_args_list[-1]
        assert prediction_call.kwargs["scores"] is None
        assert prediction_call.kwargs["labels"] is metric.detection_labels
        assert score_conversion_shapes == [(2,), (1,)]

    def test_annotation_count_mismatch_is_rejected(self) -> None:
        """A prediction annotation count that no longer matches stored scores must fail loudly.

        Assigning scores positionally is only sound while upstream emits one annotation per stored detection. If its
        loop ever starts dropping or adding annotations, silently zipping the shorter of the two would attach wrong
        scores to real detections and quietly corrupt every reported mAP.
        """
        metric = self._updated_metric()

        with (
            patch.object(
                CocoBackend, "_get_coco_format", return_value={"images": [], "annotations": [], "categories": []}
            ),
            pytest.raises(RuntimeError, match="prediction annotations"),
        ):
            metric._coco_datasets(metric._observed_classes())

    def test_mask_only_state_uses_stock_construction(self) -> None:
        """Segmentation-only predictions must keep using TorchMetrics' own helper.

        Without boxes, upstream drops an image that has no masks from the annotation list entirely
        (``helpers.py:508-511``), so annotation order stops tracking stored score order and positional assignment would
        attach one image's scores to another's detections. Nothing else in the suite reaches this branch: an empty ``(0,
        4)`` box tensor still leaves ``detection_box`` populated and takes the hoisted path.
        """
        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask[:, 1:5, 1:5] = True
        metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", iou_type="segm", class_metrics=True)
        metric.update(
            [{"masks": mask.clone(), "scores": torch.tensor([0.9]), "labels": torch.tensor([7])}],
            [{"masks": mask.clone(), "labels": torch.tensor([7])}],
        )
        recorded = MagicMock(side_effect=metric._coco_backend._get_coco_datasets)

        with patch.object(CocoBackend, "_get_coco_datasets", recorded):
            coco_preds, _, _ = metric._coco_datasets(metric._observed_classes())

        assert recorded.call_count == 1
        assert [annotation["score"] for annotation in coco_preds.dataset["annotations"]] == pytest.approx([0.9])

    def test_non_float_scores_are_rejected(self) -> None:
        """Integer score state must raise, matching the per-annotation type check the hoist replaces.

        TorchMetrics validates that scores are a tensor but not that they are floating point; the float check only
        happens during conversion. Converting a whole image at once skips that check, so it has to be restated.
        """
        metric = self._updated_metric()
        metric.detection_scores = [scores.long() for scores in metric.detection_scores]

        with pytest.raises(ValueError, match="expected floating point"):
            metric._coco_datasets(metric._observed_classes())

    def test_column_vector_scores_are_rejected(self) -> None:
        """Column-vector scores must fail instead of becoming nested COCO score lists.

        TorchMetrics' original per-annotation conversion rejects a list result. The hoisted conversion must preserve
        that scalar-score contract before assigning annotations, because nested scores can otherwise survive until a
        later backend operation and obscure the input error.
        """
        metric = self._updated_metric()
        metric.detection_scores = [scores.unsqueeze(1) for scores in metric.detection_scores]

        with pytest.raises(ValueError, match="one-dimensional"):
            metric._coco_datasets(metric._observed_classes())


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


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_empty_predictions_and_targets_return_compact_empty_class_result(backend: str) -> None:
    """An updated image with no predictions or ground truth must finish with aggregate sentinels and no class IDs.

    Every backend is covered because this is the one path that hands the COCO constructor a dataset with no annotations
    at all, and hotcoco builds its index there rather than in a later ``createIndex()`` call.
    """
    _require_backend(backend)
    metric = OnePassCocoMeanAveragePrecision(backend=backend, class_metrics=True)
    metric.update(
        [{"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}],
        [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.long)}],
    )

    result = metric.compute()

    assert result["classes"].numel() == 0
    assert result["map_per_class"].numel() == 0
    assert result["mar_100_per_class"].numel() == 0
    assert float(result["map"]) == -1.0


def test_backend_registry_matches_the_typed_eval_backend_names() -> None:
    """The runtime backend registry must accept exactly the names ``TrainConfig.eval_backend`` is typed with.

    The accepted names live in two places by necessity -- a ``Literal`` for type checkers and pydantic, a registry dict
    for construction -- so a backend added to one but not the other would validate in ``TrainConfig`` and then fail at
    metric construction, or construct fine yet be rejected by the config.
    """
    assert set(_BACKENDS) == set(get_args(CocoEvalBackend))


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


def test_adapter_rejects_non_callable_coco_backend_factory() -> None:
    """Construction must reject a non-callable COCO dataset factory before metric computation."""
    with (
        patch.object(CocoBackend, "coco", new=object()),
        pytest.raises(RuntimeError, match=r"missing backend methods: \['coco'\]"),
    ):
        OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")


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
        OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")


def test_hotcoco_contract_ignores_the_helper_it_never_calls() -> None:
    """A rename in ``_get_coco_datasets`` must not block the backend that never calls it.

    hotcoco builds its index in the COCO constructor, so this adapter assembles the dataset dictionaries itself and
    never reaches that helper. Guarding it for hotcoco would fail construction over an upstream change that cannot
    affect the metrics it produces — while the same rename must still fail loudly on faster-coco-eval.
    """
    stale_contract = {"_get_coco_datasets": ("not_a_real_parameter",), "_coco_stats_to_tensor_dict": ()}

    with patch("rfdetr.training.coco_map._BACKEND_METHOD_PARAMS", stale_contract):
        OnePassCocoMeanAveragePrecision(backend="hotcoco")
        with pytest.raises(RuntimeError, match="incompatible signature"):
            OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")


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

    def test_flags_keyword_call_to_a_positional_only_parameter(self) -> None:
        """A positional-only backend parameter must fail before its keyword call reaches compute().

        A parameter-name-only check accepts this signature even though ``_coco_datasets`` passes every
        ``_get_coco_format`` argument by keyword. Construction-time rejection turns a private API shift into the
        adapter's actionable compatibility error instead of a raw TypeError after state accumulation.
        """

        class _Backend:
            def _get_coco_format(
                self, labels, /, all_labels, boxes, masks, scores, crowds, area, iou_type, average
            ) -> None:
                pass

        mismatched = OnePassCocoMeanAveragePrecision._mismatched_backend_signatures(_Backend(), ["_get_coco_format"])

        assert mismatched == ["_get_coco_format"]


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


@patch("rfdetr.training.coco_map.all_gather")
def test_distributed_merge_is_noop_without_process_group(gather) -> None:
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


def _distributed_empty_rank_worker(rank: int, world_size: int, init_file: str, backend: str) -> None:
    """Verify one populated and one empty rank converge on identical global COCO metrics.

    Args:
        rank: Process rank launched by ``torch.multiprocessing``.
        world_size: Total process count.
        init_file: File-store path used to initialize the local Gloo process group.
        backend: COCO evaluation backend every rank constructs its metric with.

    Examples:
        This worker requires a multi-process Gloo rendezvous and is exercised by the test below.  # doctest: +SKIP
        >>> _distributed_empty_rank_worker(0, 2, "/tmp/rfdetr-coco-map-rendezvous", "hotcoco")  # doctest: +SKIP
    """
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        metric = OnePassCocoMeanAveragePrecision(class_metrics=True, backend=backend)
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
@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_distributed_merge_supports_uneven_shards_with_empty_rank(tmp_path, backend: str) -> None:
    """Two real Gloo ranks must finish without deadlock when only rank zero receives a metric update.

    Every backend is spawned because the merge hands each rank's list states through the repo's object gather and the
    empty rank then evaluates a dataset with no annotations on whichever evaluator was selected.
    """
    _require_backend(backend)
    init_file = tmp_path / "coco-map-gloo-init"

    mp.spawn(_distributed_empty_rank_worker, args=(2, str(init_file), backend), nprocs=2, join=True)


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


@pytest.mark.parametrize("backend", _ALTERNATIVE_BACKENDS)
def test_alternative_backend_matches_faster_coco_eval(backend: str) -> None:
    """Hotcoco, ufcoco and vernier must return the same metrics as the faster-coco-eval backend they replace.

    Exact equality is required of all three. ufcoco reproduces pycocotools' float64 summary to the byte, which differs
    from faster-coco-eval's in the last float64 digit on some entries (pycocotools adds ``np.spacing(1)`` to the
    precision denominator); the float32 tensors TorchMetrics reports absorb that here, so the same assertion holds.
    """
    _require_backend(backend)
    predictions, targets = multiclass_detection_state()
    kwargs: dict[str, Any] = {"class_metrics": True, "sync_on_compute": False}
    expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(backend=backend, **kwargs)
    expected_metric.update(predictions, targets)
    actual_metric.update(predictions, targets)

    expected = expected_metric.compute()
    actual = actual_metric.compute()

    assert actual.keys() == expected.keys()
    assert set(expected["map_per_class"].tolist()) - {-1.0, 0.0, 1.0}, (
        "fixture produced only degenerate per-class values; strengthen it before trusting this parity check"
    )
    for key in actual:
        if key == "classes":
            assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1))
        else:
            torch.testing.assert_close(actual[key].reshape(-1), expected[key].reshape(-1), rtol=0, atol=0)


@pytest.mark.parametrize("backend", _ALTERNATIVE_BACKENDS)
def test_alternative_backend_matches_faster_coco_eval_for_segmentation(backend: str) -> None:
    """Mask metrics must match across backends, including the per-IoU-type area swap.

    Segmentation is where the backends diverge structurally. hotcoco returns a copy from its ``dataset`` getter, so the
    ``area_bbox``/``area_segm`` swap the multi-IoU-type path performs cannot reach its evaluator unless the prediction
    dataset is rebuilt (hotcoco 1.0.1 fixed the earlier bytes-RLE constructor mismatch). ufcoco's RLE encoder rejects
    the boolean masks TorchMetrics hands over, so the adapter converts them; a mask AP that silently collapses to 0.0 is
    what either mistake would look like.
    """
    _require_backend(backend)
    mask = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask[0, 2:10, 2:10] = True
    mask[1, 11:15, 11:15] = True
    predicted_mask = mask.clone()
    predicted_mask[0, 2:4, 2:10] = False
    predictions = [
        {
            "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0], [11.0, 11.0, 15.0, 15.0]]),
            "masks": predicted_mask,
            "scores": torch.tensor([0.9, 0.6]),
            "labels": torch.tensor([3, 17]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0], [11.0, 11.0, 15.0, 15.0]]),
            "masks": mask,
            "labels": torch.tensor([3, 17]),
        }
    ]
    kwargs: dict[str, Any] = {"iou_type": ("bbox", "segm"), "class_metrics": True, "sync_on_compute": False}
    expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(backend=backend, **kwargs)
    expected_metric.update(predictions, targets)
    actual_metric.update(predictions, targets)

    expected = expected_metric.compute()
    actual = actual_metric.compute()

    assert actual.keys() == expected.keys()
    # A mask AP that silently collapses to 0.0 is the failure this fixture exists to catch, so the fixture itself
    # has to produce a non-zero one first.
    assert expected["segm_map"].item() > 0.0
    for key in actual:
        if key == "classes":
            assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1))
        else:
            torch.testing.assert_close(actual[key].reshape(-1), expected[key].reshape(-1), rtol=0, atol=0)


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_max_detection_thresholds_reach_the_evaluator(backend: str) -> None:
    """A configured maximum-detection threshold must change the metric it is supposed to change.

    hotcoco 0.5's ``params`` getter returned a copy, so writing a field through it changed nothing and raised nothing;
    1.0.0 makes field writes take effect but still documents pull-edit-assign as the idiom. Without this test the
    adapter could keep evaluating at COCO's default 100 detections while RF-DETR asked for ``eval_max_dets``, and every
    metric would still look plausible.
    """
    _require_backend(backend)
    boxes = torch.tensor([[float(index), 0.0, float(index) + 8.0, 8.0] for index in range(0, 60, 6)])
    predictions = [
        {
            "boxes": boxes,
            "scores": torch.linspace(0.9, 0.1, boxes.shape[0]),
            "labels": torch.zeros(boxes.shape[0], dtype=torch.long),
        }
    ]
    targets = [{"boxes": boxes, "labels": torch.zeros(boxes.shape[0], dtype=torch.long)}]

    def recall_at(max_detections: int) -> float:
        metric = OnePassCocoMeanAveragePrecision(
            backend=backend, max_detection_thresholds=[1, 10, max_detections], sync_on_compute=False
        )
        metric.update(predictions, targets)
        return float(metric.compute()[f"mar_{max_detections}"])

    assert recall_at(2) < recall_at(500)


def test_hotcoco_evaluation_prints_nothing(capfd: pytest.CaptureFixture[str]) -> None:
    """Selecting hotcoco must not add backend chatter to a training run's console output.

    hotcoco 1.0.1 routes its COCO summary table through ``sys.stdout`` and raises configuration diagnostics as
    ``UserWarning``s, so both are reachable with ordinary Python-level redirection. Without suppression, the table would
    land on the console on every validation epoch of every run. ``test_hotcoco_evaluation_raises_no_warnings`` covers
    the warning channel.
    """
    pytest.importorskip("hotcoco")
    predictions, targets = multiclass_detection_state()
    metric = OnePassCocoMeanAveragePrecision(
        backend="hotcoco", max_detection_thresholds=[1, 10, 500], sync_on_compute=False
    )
    metric.update(predictions, targets)
    capfd.readouterr()

    metric.compute()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_hotcoco_evaluation_raises_no_warnings() -> None:
    """Selecting hotcoco must not raise a warning per evaluation for configuration RF-DETR chose deliberately.

    hotcoco 1.0.1 reports every evaluator parameter differing from the COCO defaults as a Python warning (and prints a
    summary table on ``sys.stdout``). RF-DETR overrides ``maxDets``, and torchmetrics keeps its thresholds in float32,
    so the IoU and recall grids arrive off-reference by ~2.4e-8 and are reported too -- three warnings per ``compute()``
    on a real configuration. The stdout table is what ``test_hotcoco_evaluation_prints_nothing`` asserts on; the warning
    channel reaches a caller's ``catch_warnings``, a notebook cell, or a ``-W error`` run.
    """
    pytest.importorskip("hotcoco")
    predictions, targets = multiclass_detection_state()
    metric = OnePassCocoMeanAveragePrecision(
        backend="hotcoco", max_detection_thresholds=[1, 10, 500], sync_on_compute=False
    )
    metric.update(predictions, targets)

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        metric.compute()

    assert [str(warning.message) for warning in raised] == []


def test_ufcoco_reports_aggregate_ap_at_the_configured_detection_limit() -> None:
    """``map`` must be a real number, equal to faster-coco-eval's, when ``eval_max_dets`` is not 100.

    pycocotools summarizes ``stats[0]`` at ``maxDets=100`` regardless of the configured thresholds and returns ``-1``
    when 100 is not among them; ufcoco reproduces that, and faster-coco-eval and hotcoco read the largest configured
    threshold instead. RF-DETR evaluates at 500 by default, so without the adapter's evaluator override every
    ``val/mAP`` on this backend would read ``-1`` while the rest of the metrics looked plausible.
    """
    _require_backend("ufcoco")
    predictions, targets = multiclass_detection_state()

    def aggregate_ap(backend: str) -> float:
        metric = OnePassCocoMeanAveragePrecision(
            backend=backend, max_detection_thresholds=[1, 10, 500], sync_on_compute=False
        )
        metric.update(predictions, targets)
        return float(metric.compute()["map"])

    expected = aggregate_ap("faster_coco_eval")
    assert expected > 0.0, "fixture produced no aggregate AP; strengthen it before trusting this check"
    assert aggregate_ap("ufcoco") == expected


def test_ufcoco_evaluation_prints_nothing(capfd: pytest.CaptureFixture[str]) -> None:
    """Selecting ufcoco must not add backend chatter to a training run's console output.

    ufcoco prints pycocotools' evaluation progress lines and twelve-row summary table on ``sys.stdout``, once per IoU
    type per ``compute()``; the adapter's evaluation window has to swallow them the way it does for the other backends.
    """
    _require_backend("ufcoco")
    predictions, targets = multiclass_detection_state()
    metric = OnePassCocoMeanAveragePrecision(
        backend="ufcoco", max_detection_thresholds=[1, 10, 500], sync_on_compute=False
    )
    metric.update(predictions, targets)
    capfd.readouterr()

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        metric.compute()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert [str(warning.message) for warning in raised] == []


def test_missing_ufcoco_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting ufcoco without it installed must say how to install it.

    Same trap as for hotcoco: the private-contract check resolves the evaluator inside an ``except ImportError``, so an
    import deferred until then would be reported as a torchmetrics incompatibility instead of a missing extra.
    """

    def missing_dependency() -> Any:
        raise ImportError(
            "backend='ufcoco' requires the ultrafast-pycocotools package; install it with: pip install 'rfdetr[train]'"
        )

    monkeypatch.setattr("rfdetr.training.coco_map._ufcoco", missing_dependency)

    with pytest.raises(ImportError, match=r"rfdetr\[train\]"):
        OnePassCocoMeanAveragePrecision(backend="ufcoco")


@pytest.mark.parametrize(
    ("importer", "package"),
    [
        pytest.param(_hotcoco, "hotcoco", id="hotcoco"),
        pytest.param(_ufcoco, "ultrafast_pycocotools", id="ufcoco"),
        pytest.param(_vernier, "vernier", id="vernier"),
    ],
)
def test_missing_backend_package_names_the_extra(importer: Callable[[], Any], package: str) -> None:
    """A missing backend package must retain the actionable installation guidance."""
    missing_package = ModuleNotFoundError(f"No module named '{package}'", name=package)

    with (
        patch("builtins.__import__", side_effect=missing_package),
        pytest.raises(ImportError, match=r"rfdetr\[train\]"),
    ):
        importer()


def test_ufcoco_propagates_nested_import_error() -> None:
    """A failure inside ufcoco must retain the missing nested dependency name."""
    nested_error = ModuleNotFoundError("No module named 'nested_dependency'", name="nested_dependency")

    with patch("builtins.__import__", side_effect=nested_error), pytest.raises(ModuleNotFoundError) as raised:
        _ufcoco()

    assert raised.value is nested_error


def test_ufcoco_backend_picks_up_the_optional_package_and_survives_pickling() -> None:
    """The ufcoco backend must resolve its surfaces from ``ultrafast_pycocotools`` and pickle with the metric.

    Lightning's DDP spawn and the callback's checkpoint plumbing pickle metric objects; a backend that stored the
    imported module on itself would break there, which is why the surfaces are resolved on every access.
    """
    _require_backend("ufcoco")
    metric = OnePassCocoMeanAveragePrecision(backend="ufcoco", sync_on_compute=False)
    backend = metric._coco_backend

    assert isinstance(backend, _UfcocoBackend)
    assert backend.backend == "faster_coco_eval", "torchmetrics must still see the supported enum member"
    assert backend.coco.__module__.startswith("ultrafast_pycocotools")
    assert backend.cocoeval.__mro__[1].__module__.startswith("ultrafast_pycocotools")
    assert backend.mask_utils.area.__module__.startswith("ultrafast_pycocotools")

    restored = pickle.loads(pickle.dumps(metric))

    assert isinstance(restored._coco_backend, _UfcocoBackend)
    assert restored._coco_backend.cocoeval is backend.cocoeval


def test_vernier_backend_survives_pickling() -> None:
    """The vernier backend must pickle with the metric, which Lightning's DDP spawn and checkpoint plumbing require."""
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)

    restored = pickle.loads(pickle.dumps(metric))

    assert isinstance(restored._coco_backend, _VernierBackend)


@pytest.mark.parametrize(
    "local_world_size,expected",
    [
        pytest.param(None, 4, id="unset-uses-the-whole-pool"),
        pytest.param("2", 2, id="divides-evenly-among-local-ranks"),
        pytest.param("8", 1, id="oversized-world-size-clamps-to-one"),
    ],
)
def test_vernier_thread_budget_divides_by_local_world_size(
    monkeypatch: pytest.MonkeyPatch, local_world_size: str | None, expected: int
) -> None:
    """The per-rank thread budget divides the process-wide pool by ``LOCAL_WORLD_SIZE``, never below one.

    Every DDP rank sharing one node sees ``torch.get_num_threads()`` report the same process-wide count; handing vernier
    that whole budget on every rank oversubscribes the node's CPUs by a factor of ``LOCAL_WORLD_SIZE`` unless it is
    divided among the ranks running on that node, floored at one thread.
    """
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)

    assert _vernier_thread_budget() == expected


def test_vernier_rejects_mask_only_evaluation() -> None:
    """Mask-only evaluation must fail at construction, before an epoch of state is accumulated and discarded."""
    _require_backend("vernier")

    with pytest.raises(ValueError, match="requires 'bbox'"):
        OnePassCocoMeanAveragePrecision(backend="vernier", iou_type="segm", sync_on_compute=False)


def test_vernier_rejects_fractional_ground_truth_labels() -> None:
    """A fractional ground-truth class label must be rejected instead of silently truncated by ``.long()``.

    ``coco_inputs`` owns the check now, but a fractional value must still fail loudly rather than become a wrong class
    through an ``int64`` cast. Driven through ``compute()`` so it pins the behaviour wherever the check lives.
    """
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([3])}],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3.5])}],
    )

    with pytest.raises(ValueError, match="must be integral"):
        metric.compute()


def test_vernier_rejects_fractional_detection_labels_under_segmentation() -> None:
    """A fractional detection class label under ``segm`` must be rejected instead of truncated by ``.long()``.

    ``coco_inputs`` validates both sides and both routes, so this no longer depends on which route an IoU type takes.
    ``segm`` runs first so the failure is attributed to the detection labels.
    """
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", iou_type=("segm", "bbox"), sync_on_compute=False)
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[0, :2, :2] = True
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3.5]),
                "masks": mask,
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([3]), "masks": mask}],
    )

    with pytest.raises(ValueError, match="must be integral"):
        metric.compute()


def test_vernier_accepts_whole_valued_float_labels() -> None:
    """A whole-valued floating-point label must be accepted, matching the equivalent integer label.

    Only a fractional value is a truncation risk; rejecting every floating-point label outright would also reject an
    empty ``float32`` label tensor, which carries no values to truncate.
    """
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3.0]),
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3.0])}],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


def test_vernier_handles_one_dimensional_empty_boxes() -> None:
    """A 1-D empty box tensor on one image must not crash concatenation against another image's real boxes.

    TorchMetrics' ``_fix_empty_tensors`` reshapes a 1-D empty box tensor to ``(1, 0)`` rather than ``(0, 4)`` to avoid a
    DDP all-reduce hang (``torchmetrics.detection.helpers``). Concatenating that ``(1, 0)`` tensor against a ``(N, 4)``
    tensor from another image in the same batch fails on the mismatched second dimension unless each per-image tensor is
    reshaped to ``(-1, 4)`` first, for both the ground truth and the ``(N, 7)`` detection matrix ``bbox`` uses.
    """
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [
            {"boxes": torch.empty(0), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            },
        ],
        [
            {"boxes": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
        ],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


def test_vernier_handles_one_dimensional_empty_boxes_under_segmentation() -> None:
    """A 1-D empty box tensor on one image must not crash the columnar detection route ``segm`` uses.

    ``_vernier_detections`` takes the columnar route under ``segm`` instead of the ``(N, 7)`` matrix ``bbox`` uses, with
    its own ``torch.cat`` over ``detection_box`` that needs the same ``(-1, 4)`` reshape guard.
    """
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", iou_type=("bbox", "segm"), sync_on_compute=False)
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[0, :2, :2] = True
    empty_mask = torch.zeros((0, 4, 4), dtype=torch.bool)
    metric.update(
        [
            {
                "boxes": torch.empty(0),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
                "masks": empty_mask,
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
                "masks": mask,
            },
        ],
        [
            {"boxes": torch.empty(0), "labels": torch.empty(0, dtype=torch.long), "masks": empty_mask},
            {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([3]), "masks": mask},
        ],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


def test_vernier_ground_truth_handles_one_dimensional_empty_boxes() -> None:
    """A one-dimensional empty ground-truth box tensor must not crash concatenation against another image's boxes.

    TorchMetrics' ``_fix_empty_tensors`` (``torchmetrics.detection.helpers``) reshapes an empty ``torch.tensor([])`` box
    tensor to ``(1, 0)`` rather than ``(0, 4)``, to avoid a DDP all-reduce hang. ``_vernier_ground_truth`` concatenates
    every image's ground-truth boxes into one tensor, so a ``(1, 0)`` entry next to a real image's ``(N, 4)`` entry
    fails on the mismatched second dimension unless each per-image tensor is reshaped to ``(-1, 4)`` first.
    """
    _require_backend("vernier")
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [
            {"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            },
        ],
        [
            {"boxes": torch.tensor([]), "labels": torch.empty(0, dtype=torch.long)},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
        ],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


@pytest.mark.parametrize("iou_type", ["bbox", pytest.param(("bbox", "segm"), id="both")])
@pytest.mark.parametrize("max_dets", [100, 500])
def test_vernier_matches_faster_coco_eval_across_updates_and_reuse(iou_type: Any, max_dets: int) -> None:
    """Vernier's native path must match faster-coco-eval exactly on ties, crowds and images missing either side.

    It builds its own arrays instead of the COCO datasets every other backend shares, so it is held to exact equality on
    a fixture carrying everything those datasets encode, at ``eval_max_dets`` both at and above pycocotools' 100.
    """
    _require_backend("vernier")
    predictions, targets = _metric_inputs()
    kwargs: dict[str, Any] = {
        "iou_type": iou_type,
        "class_metrics": True,
        "max_detection_thresholds": [1, 10, max_dets],
    }
    reference = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual = OnePassCocoMeanAveragePrecision(backend="vernier", **kwargs)
    for _ in range(2):  # the second pass runs on a reset metric
        for metric in (reference, actual):
            for index in range(len(predictions)):
                metric.update(copy.deepcopy(predictions[index : index + 1]), copy.deepcopy(targets[index : index + 1]))
            metric.merge_distributed_state()
        expected, observed = reference.compute(), actual.compute()
        assert observed.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True)
        reference.reset()
        actual.reset()


#: Side of the canvas `_rasterize` draws on; the fixture keeps every coordinate inside it.
_CANVAS = 128


def _vernier_gt_state(
    seed: int,
    *,
    with_area: bool = True,
    with_crowd: bool = True,
    empty_images: bool = True,
    with_masks: bool = False,
    crowd_value: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ground truth shaped to exercise the columnar builder's edge cases, not just its happy path.

    Args:
        seed: Torch manual seed, so a case is reproducible.
        with_area: Whether targets carry an ``area``, half of it zero so the per-element fallback is exercised.
        with_crowd: Whether targets carry an ``iscrowd`` flag.
        empty_images: Whether every fifth image has no ground truth at all.
        with_masks: Whether both sides carry masks, so the run evaluates ``segm`` as well as ``bbox``.
        crowd_value: Value written for a crowd annotation, to exercise flags wider than ``uint8``.

    Returns:
        Predictions and targets in TorchMetrics detection format, one entry per image.

    Examples:
        >>> predictions, targets = _vernier_gt_state(1)
        >>> len(targets), sorted(targets[1])
        (40, ['area', 'boxes', 'iscrowd', 'labels'])
        >>> int(targets[1]["boxes"].shape[0]), int(targets[1]["iscrowd"].max())
        (6, 1)
    """
    torch.manual_seed(seed)
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for index in range(40):
        count = 0 if (empty_images and index % 5 == 0) else 6
        origin = torch.rand(count, 2) * 90
        extent = torch.rand(count, 2) * 24 + 8
        boxes = torch.cat([origin, origin + extent], 1) if count else torch.zeros((0, 4))
        labels = torch.randint(0, 15, (count,))
        target: dict[str, Any] = {"boxes": boxes, "labels": labels}
        if with_area and count:
            # Half the areas are 0, which upstream replaces with a computed one per element.
            area = (extent[:, 0] * extent[:, 1]).clone()
            area[::2] = 0.0
            target["area"] = area
        if with_crowd:
            target["iscrowd"] = (torch.arange(count) % 4 == 0).long() * crowd_value
        detections = boxes.repeat(4, 1) + torch.randn(count * 4, 4) * 5 if count else torch.zeros((0, 4))
        if count:
            detections[:, 2:] = torch.maximum(detections[:, 2:], detections[:, :2] + 1)
        prediction: dict[str, Any] = {
            "boxes": detections,
            "scores": torch.rand(count * 4),
            "labels": labels.repeat(4),
        }
        if with_masks:
            # A mask's area differs from its box's `w * h`, which tells the two area rules apart.
            target["masks"] = _rasterize(boxes)
            prediction["masks"] = _rasterize(detections)
        targets.append(target)
        predictions.append(prediction)
    return predictions, targets


def _rasterize(boxes: torch.Tensor) -> torch.Tensor:
    """Return one filled-rectangle mask per box, on the fixture's fixed canvas.

    Args:
        boxes: ``(N, 4)`` boxes in ``xyxy``.

    Returns:
        ``(N, _CANVAS, _CANVAS)`` boolean masks.

    Examples:
        >>> masks = _rasterize(torch.tensor([[1.0, 2.0, 5.0, 4.0]]))
        >>> masks.shape[0], int(masks.sum())
        (1, 8)
    """
    masks = torch.zeros((len(boxes), _CANVAS, _CANVAS), dtype=torch.bool)
    for index, (x0, y0, x1, y1) in enumerate(boxes.round().clamp(0, _CANVAS).int().tolist()):
        masks[index, y0:y1, x0:x1] = True
    return masks


@pytest.mark.parametrize(
    ("with_area", "with_crowd", "empty_images", "with_masks"),
    [
        pytest.param(False, False, False, False, id="boxes_only"),
        pytest.param(True, False, False, False, id="area"),
        pytest.param(False, True, False, False, id="crowd"),
        pytest.param(False, False, True, False, id="empty_images"),
        pytest.param(True, True, True, False, id="all"),
        pytest.param(True, True, True, True, id="masks"),
    ],
)
def test_vernier_columnar_ground_truth_is_the_same_document(
    with_area: bool, with_crowd: bool, empty_images: bool, with_masks: bool
) -> None:
    """The columnar ground truth must be the *document* TorchMetrics' COCO format is, not merely score the same.

    ``coco_inputs_from_columns`` builds the document, but mapping *this* metric's state onto it is still
    RF-DETR's, and that
    is what this pins -- by ``dataset_hash`` against the route it replaced, ``_get_coco_format`` through vernier's
    normalizers and JSON parser.

    Metrics would not catch it: most cases here differ only in ways AP cannot see, such as an image entry with no
    annotations on it or an ``area`` on an annotation that matches nothing. The cases are the rules easiest to get
    wrong -- the per-element ``area`` fallback, ``iscrowd``, an image with no ground truth, and under ``segm`` the
    size each image takes from its own first mask.
    """
    _require_backend("vernier")
    predictions, targets = _vernier_gt_state(
        1, with_area=with_area, with_crowd=with_crowd, empty_images=empty_images, with_masks=with_masks
    )
    iou_type = ("bbox", "segm") if with_masks else ("bbox",)
    metric = OnePassCocoMeanAveragePrecision(
        backend="vernier", box_format="xyxy", iou_type=iou_type, class_metrics=True
    )
    metric.update(predictions, targets)
    classes = sorted({int(label) for target in metric.groundtruth_labels for label in target.tolist()})

    adapters = _vernier().adapters
    reference = metric._coco_backend._get_coco_format(
        labels=metric.groundtruth_labels,
        boxes=metric.groundtruth_box,
        masks=metric.groundtruth_mask if with_masks else None,
        crowds=metric.groundtruth_crowds,
        area=metric.groundtruth_area,
        iou_type=metric.iou_type,
        all_labels=classes,
        average=metric.average,
    )
    detection_sizes = {
        image_id: tuple(masks[0][0]) for image_id, masks in enumerate(metric.detection_mask) if len(masks) > 0
    }
    sized = (
        adapters.with_mask_image_sizes(reference, detection_sizes)
        if with_masks
        else adapters.with_placeholder_image_sizes(reference)
    )
    from_json = _vernier().instance.CocoDataset.from_json(adapters.to_coco_json(sized))
    from_arrays, _ = adapters.coco_inputs_from_columns(
        *metric._vernier_columns(),
        box_format="xywh",
        categories=classes,
        area="auto",
    )

    assert from_arrays.dataset_hash == from_json.dataset_hash
    assert (from_arrays.num_images, from_arrays.num_annotations, from_arrays.num_categories) == (
        from_json.num_images,
        from_json.num_annotations,
        from_json.num_categories,
    )


def test_vernier_ground_truth_builder_keeps_crowd_flags_wider_than_uint8() -> None:
    """The ground-truth builder must keep ``iscrowd`` at its stored width, not through a column that wraps it.

    ``coco_inputs`` builds its columns directly from tensors rather than through JSON, so
    ``test_vernier_columnar_ground_truth_is_the_same_document`` cannot cover a value like 256: vernier's own JSON
    parser rejects anything but 0/1 for ``iscrowd``, so the reference side of that test would raise first. Comparing
    the built dataset's ``dataset_hash`` against one built with no crowd annotation at all catches the same defect
    without JSON in the loop -- a ``uint8`` column would wrap 256 back to 0 and make the two datasets identical.
    """
    _require_backend("vernier")

    def ground_truth(crowd_value: int) -> Any:
        predictions, targets = _vernier_gt_state(1, crowd_value=crowd_value)
        metric = OnePassCocoMeanAveragePrecision(
            backend="vernier", box_format="xyxy", iou_type="bbox", class_metrics=True
        )
        metric.update(predictions, targets)
        classes = sorted({int(label) for target in metric.groundtruth_labels for label in target.tolist()})
        ground_truth, _ = _vernier().adapters.coco_inputs_from_columns(
            *metric._vernier_columns(), box_format="xywh", categories=classes
        )
        return ground_truth

    wide_crowd, no_crowd = ground_truth(256), ground_truth(0)

    assert wide_crowd.dataset_hash != no_crowd.dataset_hash
    assert wide_crowd.num_annotations == no_crowd.num_annotations


def test_vernier_columnar_route_keeps_crowd_flags_wider_than_uint8() -> None:
    """A crowd flag must reach vernier at its stored width, not through a column that wraps it.

    ``iscrowd`` is 0/1 in COCO practice, so a ``uint8`` column looks harmless until 256 wraps to 0 and the annotation is
    matched as a normal one instead of ignored -- silently, and only on that value, since 255 and 257 both survive.
    """
    _require_backend("vernier")
    predictions, targets = _vernier_gt_state(1, crowd_value=256)
    kwargs: dict[str, Any] = {"box_format": "xyxy", "iou_type": "bbox", "class_metrics": True}
    reference = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual = OnePassCocoMeanAveragePrecision(backend="vernier", **kwargs)
    for metric in (reference, actual):
        metric.update(copy.deepcopy(predictions), copy.deepcopy(targets))
    expected, observed = reference.compute(), actual.compute()

    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("iou_type", ["bbox", pytest.param(("bbox", "segm"), id="both")])
@pytest.mark.parametrize("max_dets", [100, 500])
def test_vernier_parity_modes_differ_only_on_the_aggregate_at_a_non_default_max_dets(
    iou_type: Any, max_dets: int
) -> None:
    """Vernier's ``strict`` and ``corrected`` modes must differ on this path in exactly one metric, and only there.

    ``corrected`` is a bundle that has grown between releases, and ``_vernier_results`` claims exactly one member of
    it reaches RF-DETR: the aggregate ``map``, which pycocotools reads at a literal ``maxDets=100`` and so reports as
    ``-1`` whenever 100 is not among the configured limits. Everything else vernier corrects is unreachable here.

    A failure is therefore a signal about vernier, not a bug in RF-DETR: the numbers RF-DETR publishes would move on
    the version bump. Widen the allowed-to-differ set deliberately -- never by loosening the comparison.
    """
    _require_backend("vernier")
    predictions, targets = _metric_inputs()
    kwargs: dict[str, Any] = {
        "backend": "vernier",
        "iou_type": iou_type,
        "class_metrics": True,
        "max_detection_thresholds": [1, 10, max_dets],
        "sync_on_compute": False,
    }

    def results(parity_mode: str) -> dict[str, torch.Tensor]:
        with patch("rfdetr.training.coco_map._VERNIER_PARITY_MODE", parity_mode):
            metric = OnePassCocoMeanAveragePrecision(**kwargs)
            metric.update(copy.deepcopy(predictions), copy.deepcopy(targets))
            return metric.compute()

    strict, corrected = results("strict"), results("corrected")

    prefixes = ("",) if isinstance(iou_type, str) else tuple(f"{name}_" for name in iou_type)
    # It only bites on a ladder whose largest entry is not the literal 100 pycocotools' summary reads.
    differing_keys = set() if max_dets == 100 else {f"{prefix}map" for prefix in prefixes}
    assert strict.keys() == corrected.keys()
    assert set(corrected[f"{prefixes[0]}map_per_class"].tolist()) - {-1.0, 0.0, 1.0}, (
        "fixture produced only degenerate per-class values; strengthen it before trusting this parity check"
    )
    for key in corrected:
        if key in differing_keys:
            continue
        torch.testing.assert_close(strict[key], corrected[key], rtol=0, atol=0, equal_nan=True)
    for key in differing_keys:
        assert float(strict[key]) == -1.0
        assert float(corrected[key]) > 0.0


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


class TestUfcocoArraysMatchPycocotools:
    """The evaluator arrays the per-class reduction reads must be byte-identical to pycocotools' on the same datasets.

    The parity tests above compare the float32 tensors RF-DETR reports; this one goes one level down, through the
    adapter's own datasets, and checks the float64 precision, recall and score arrays the per-class AP/AR reduction is
    computed from. pycocotools is the reference ufcoco reproduces, and ``rfdetr[train]`` installs it.
    """

    @pytest.mark.parametrize("iou_type", ["bbox", "segm", pytest.param(("bbox", "segm"), id="both")])
    @pytest.mark.parametrize("max_dets", [100, 500])
    def test_evaluator_arrays_are_byte_identical(self, iou_type: Any, max_dets: int) -> None:
        """Precision, recall and score arrays must match pycocotools byte for byte across incremental updates and
        reuse."""
        _require_backend("ufcoco")
        pycocotools_coco = pytest.importorskip("pycocotools.coco").COCO
        pycocotools_cocoeval = pytest.importorskip("pycocotools.cocoeval").COCOeval
        predictions, targets = _metric_inputs()
        thresholds = [1, 10, max_dets]
        reference = OnePassCocoMeanAveragePrecision(
            backend="faster_coco_eval", iou_type=iou_type, class_metrics=True, max_detection_thresholds=thresholds
        )
        actual = OnePassCocoMeanAveragePrecision(
            backend="ufcoco", iou_type=iou_type, class_metrics=True, max_detection_thresholds=thresholds
        )
        for _ in range(2):  # the second pass runs on a reset metric
            for metric in (reference, actual):
                for index in range(len(predictions)):
                    metric.update(
                        copy.deepcopy(predictions[index : index + 1]), copy.deepcopy(targets[index : index + 1])
                    )
                metric.merge_distributed_state()
            expected, observed = reference.compute(), actual.compute()
            assert observed.keys() == expected.keys()
            for key in expected:
                torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True)

            coco_preds, coco_target, prediction_dataset = actual._coco_datasets(actual._observed_classes())
            assert prediction_dataset is not None
            oracle_gt, oracle_dt = pycocotools_coco(), pycocotools_coco()
            oracle_gt.dataset = copy.deepcopy(coco_target.dataset)
            oracle_dt.dataset = copy.deepcopy(prediction_dataset)
            oracle_gt.createIndex()
            oracle_dt.createIndex()
            for kind in actual.iou_type:
                if len(actual.iou_type) > 1:
                    for dataset in (prediction_dataset, oracle_dt.dataset):
                        for annotation in dataset["annotations"]:
                            annotation["area"] = annotation[f"area_{kind}"]
                evaluator = actual._coco_backend.cocoeval(coco_target, coco_preds, iouType=kind)
                oracle = pycocotools_cocoeval(oracle_gt, oracle_dt, iouType=kind)
                for instance in (evaluator, oracle):
                    instance.params.maxDets = thresholds
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        instance.evaluate()
                        instance.accumulate()
                for key in ("precision", "recall", "scores"):
                    assert np.asarray(evaluator.eval[key]).shape == np.asarray(oracle.eval[key]).shape
                    assert np.asarray(evaluator.eval[key]).tobytes() == np.asarray(oracle.eval[key]).tobytes()
            reference.reset()
            actual.reset()


def test_missing_hotcoco_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting hotcoco without it installed must say how to install it.

    The private-contract check that runs right after backend construction resolves the evaluator inside an ``except
    ImportError``, so an import failure deferred until then is reported as a torchmetrics incompatibility — the one
    message that tells a user nothing about the missing extra.
    """

    def missing_dependency() -> Any:
        raise ImportError("hotcoco requires the optional dependency; install it with: pip install 'rfdetr[hotcoco]'")

    monkeypatch.setattr("rfdetr.training.coco_map._hotcoco", missing_dependency)

    with pytest.raises(ImportError, match=r"rfdetr\[hotcoco\]"):
        OnePassCocoMeanAveragePrecision(backend="hotcoco")


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_multi_iou_type_areas_follow_their_own_iou_type(backend: str) -> None:
    """Each IoU type of a joint evaluation must bucket detections by that type's own area.

    TorchMetrics emits ``area_bbox`` and ``area_segm`` per annotation and the active ``area`` has to be switched between
    passes, because COCO's small/medium/large split reads that one field. The switch is invisible unless a detection's
    box and mask land in different buckets, so this fixture gives a false positive a 25x25 box (small) and a 60x60 mask
    (medium) and scores it above the true positive — an unmatched, top-ranked detection is the only kind whose area
    moves a reported number. Without the switch the segmentation area leaks into the box pass and ``bbox_map_small``
    doubles.
    """
    _require_backend(backend)
    masks = torch.zeros(2, 128, 128, dtype=torch.bool)
    masks[0, 0:20, 0:20] = True
    masks[1, 40:100, 40:100] = True
    predicted = masks.clone()
    predicted[0, 0:25, 0:25] = True
    metric = OnePassCocoMeanAveragePrecision(
        backend=backend, iou_type=("bbox", "segm"), class_metrics=True, sync_on_compute=False
    )
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0], [40.0, 40.0, 65.0, 65.0]]),
                "masks": predicted,
                "scores": torch.tensor([0.5, 0.95]),
                "labels": torch.tensor([1, 1]),
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0]]), "masks": masks[:1], "labels": torch.tensor([1])}],
    )

    result = metric.compute()

    # Box pass: the false positive's 25x25 box is "small" too and outranks the true positive, so precision at full
    # recall is 1/2. Let the segmentation area leak in and the 60x60 mask makes it "medium", the false positive
    # drops out of the bucket, and this reads 1.0 instead.
    assert float(result["bbox_map_small"]) == pytest.approx(0.5)
    # Mask pass: only the true positive is "small", and its 25x25 prediction over a 20x20 target is IoU 0.64 —
    # matched at 3 of the 10 COCO thresholds.
    assert float(result["segm_map_small"]) == pytest.approx(0.3)


def _straddling_disc_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records whose box and mask areas straddle the small/medium boundary.

    A disc of radius 17 has a 34x34 = 1156 box (*medium*) and a ~901 mask (*small*), so which area the
    ground truth is built from is visible in the AP buckets rather than only in the fourth decimal.

    Examples:
        >>> predictions, targets = _straddling_disc_records()
        >>> len(predictions), len(targets)
        (30, 30)
        >>> targets[0]["masks"].shape
        torch.Size([1, 128, 128])
    """
    size = 128
    rows, columns = np.ogrid[:size, :size]

    def disc(center_y: int, center_x: int, radius: int) -> Any:
        return (rows - center_y) ** 2 + (columns - center_x) ** 2 <= radius**2

    def boxes_of(masks: Any) -> torch.Tensor:
        corners = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            corners.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        return torch.tensor(np.asarray(corners, dtype=np.float64)).float()

    rng = np.random.default_rng(9)
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for _ in range(30):
        center_y, center_x = int(rng.integers(30, 98)), int(rng.integers(30, 98))
        gt_masks = np.stack([disc(center_y, center_x, 17)])
        dt_masks = np.stack([disc(center_y + 1, center_x, 17)])
        labels = torch.tensor([0])
        targets.append({"boxes": boxes_of(gt_masks), "labels": labels, "masks": torch.from_numpy(gt_masks)})
        predictions.append(
            {
                "boxes": boxes_of(dt_masks),
                "scores": torch.tensor([0.9]),
                "labels": labels,
                "masks": torch.from_numpy(dt_masks),
            }
        )
    return predictions, targets


def test_two_iou_type_run_buckets_bbox_ap_by_mask_area() -> None:
    """A two-IoU-type run must bucket ``bbox`` AP by *mask* area, as upstream does.

    ``_get_coco_format`` derives the ground-truth ``area`` from the mask whenever ``segm`` is among the IoU
    types -- for the ``bbox`` pass too. Deriving it per pass instead is invisible until an object's box and
    mask areas straddle 32**2 or 96**2, and then it moves AP between the small and medium buckets with no
    error. A disc of radius 17 does exactly that: box 34x34 = 1156 (medium), mask ~901 (small).
    """
    _require_backend("vernier")
    _require_backend("hotcoco")
    predictions, targets = _straddling_disc_records()

    def compute(backend: str) -> dict[str, torch.Tensor]:
        metric = OnePassCocoMeanAveragePrecision(box_format="xyxy", iou_type=("bbox", "segm"), backend=backend)
        metric.update(predictions, targets)
        return metric.compute()

    vernier_result, reference = compute("vernier"), compute("hotcoco")
    # Anti-vacuity: the fixture only bites if the mask area really is the small bucket while the box is not.
    assert float(vernier_result["bbox_map_small"]) > 0
    assert float(vernier_result["bbox_map_medium"]) == -1
    for key in vernier_result:
        if key == "classes":
            continue
        assert torch.equal(vernier_result[key], reference[key]), key


def test_bbox_only_run_buckets_by_box_area_and_carries_no_masks() -> None:
    """The mirror of the two-IoU-type case, and the reason ``rles`` is gated on the run.

    ``_get_coco_format`` derives the area from the mask only when ``segm`` is among the IoU types; a bbox-only run uses
    ``w * h``. vernier's ``area="auto"`` reads a mask whenever one is present, so carrying masks here would silently re-
    bucket the AP. The last assertion is the load-bearing one: what keeps masks out is the gate, not whether the state
    happens to be empty.
    """
    _require_backend("vernier")
    _require_backend("hotcoco")
    predictions, targets = _straddling_disc_records()

    def build(backend: str) -> OnePassCocoMeanAveragePrecision:
        metric = OnePassCocoMeanAveragePrecision(box_format="xyxy", iou_type=("bbox",), backend=backend)
        metric.update(predictions, targets)
        return metric

    metric = build("vernier")
    detection_columns, target_columns = metric._vernier_columns()
    assert "rles" not in detection_columns
    assert "rles" not in target_columns

    result = metric.compute()
    reference = build("hotcoco").compute()
    # Anti-vacuity: the box area really is the medium bucket while the mask area would be small.
    assert float(result["map_medium"]) > 0
    assert float(result["map_small"]) == -1
    for key in result:
        if key == "classes":
            continue
        assert torch.equal(result[key], reference[key]), key

    # The gate, not the emptiness of the state, is what keeps masks out.
    metric.groundtruth_mask = [(((128, 128), b"0"),)] * len(targets)
    metric.detection_mask = [(((128, 128), b"0"),)] * len(predictions)
    detection_columns, target_columns = metric._vernier_columns()
    assert "rles" not in detection_columns
    assert "rles" not in target_columns


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_dtypes_survive_the_round_trip_to_vernier(dtype: torch.dtype) -> None:
    """State stored under autocast must evaluate, whatever dtype the autocast ran in.

    ``bfloat16`` has no numpy dtype, so it used to raise ``TypeError`` out of ``np.asarray`` before vernier's f64 ingest
    boundary was reached; ``vernier>=0.5.3`` widens it there instead. Columns are handed over at their stored dtype on
    purpose — a cast here would hide that. Every value below is exactly representable in both dtypes, so the float32 run
    is a bit-exact oracle.
    """
    _require_backend("vernier")
    boxes = [[0.0, 0.0, 16.0, 16.0], [32.0, 32.0, 64.0, 64.0]]
    scores = [0.75, 0.5]

    def build(tensor_dtype: torch.dtype) -> OnePassCocoMeanAveragePrecision:
        metric = OnePassCocoMeanAveragePrecision(box_format="xyxy", iou_type=("bbox",), backend="vernier")
        metric.update(
            [
                {
                    "boxes": torch.tensor(boxes, dtype=tensor_dtype),
                    "scores": torch.tensor(scores, dtype=tensor_dtype),
                    "labels": torch.tensor([0, 1]),
                }
            ],
            [{"boxes": torch.tensor(boxes, dtype=tensor_dtype), "labels": torch.tensor([0, 1])}],
        )
        return metric

    metric = build(dtype)
    # The stored dtype reaches vernier unchanged; widening is vernier's job, not this metric's.
    assert metric._vernier_columns()[0]["scores"].dtype == dtype

    result = metric.compute()
    reference = build(torch.float32).compute()
    # Anti-vacuity: a perfect match scores 1.0, so a silently emptied run would not pass this.
    assert float(result["map"]) == 1.0
    for key in result:
        if key == "classes":
            continue
        assert torch.equal(result[key], reference[key]), key
