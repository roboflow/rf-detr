# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for RF-DETR's one-pass TorchMetrics COCO adapter."""

import sys
from typing import Any
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


@pytest.mark.parametrize("backend", ["faster_coco_eval", "hotcoco"])
def test_empty_predictions_and_targets_return_compact_empty_class_result(backend: str) -> None:
    """An updated image with no predictions or ground truth must finish with aggregate sentinels and no class IDs.

    Both backends are covered because this is the one path that hands the COCO constructor a dataset with no annotations
    at all, and hotcoco builds its index there rather than in a later ``createIndex()`` call.
    """
    pytest.importorskip(backend)
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


def test_hotcoco_backend_matches_faster_coco_eval() -> None:
    """The optional hotcoco backend must return the same metrics as the default faster-coco-eval backend."""
    pytest.importorskip("hotcoco")
    predictions, targets = multiclass_detection_state()
    kwargs: dict[str, Any] = {"class_metrics": True, "sync_on_compute": False}
    expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(backend="hotcoco", **kwargs)
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


def test_hotcoco_backend_matches_faster_coco_eval_for_segmentation() -> None:
    """Mask metrics must match across backends, including the per-IoU-type area swap.

    Segmentation is where the two backends diverge structurally: hotcoco returns a copy from its ``dataset``
    getter, so the ``area_bbox``/``area_segm`` swap the multi-IoU-type path performs cannot reach its evaluator
    unless the prediction dataset is rebuilt, and its RLE counts have to reach the constructor as text.
    """
    pytest.importorskip("hotcoco")
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
    actual_metric = OnePassCocoMeanAveragePrecision(backend="hotcoco", **kwargs)
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


@pytest.mark.parametrize("backend", ["faster_coco_eval", "hotcoco"])
def test_max_detection_thresholds_reach_the_evaluator(backend: str) -> None:
    """A configured maximum-detection threshold must change the metric it is supposed to change.

    hotcoco's ``params`` getter returns a copy, so writing a field through it changes nothing and raises nothing.
    Without this test the adapter could keep evaluating at COCO's default 100 detections while RF-DETR asked for
    ``eval_max_dets``, and every metric would still look plausible.
    """
    pytest.importorskip(backend)
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

    hotcoco prints from Rust straight to the output file descriptors, so Python-level redirection does not reach
    it: a COCO summary table plus one warning per overridden evaluator parameter would land on the console on
    every validation epoch of every run.
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


@pytest.mark.parametrize("backend", ["faster_coco_eval", "hotcoco"])
def test_multi_iou_type_areas_follow_their_own_iou_type(backend: str) -> None:
    """Each IoU type of a joint evaluation must bucket detections by that type's own area.

    TorchMetrics emits ``area_bbox`` and ``area_segm`` per annotation and the active ``area`` has to be switched between
    passes, because COCO's small/medium/large split reads that one field. The switch is invisible unless a detection's
    box and mask land in different buckets, so this fixture gives a false positive a 25x25 box (small) and a 60x60 mask
    (medium) and scores it above the true positive — an unmatched, top-ranked detection is the only kind whose area
    moves a reported number. Without the switch the segmentation area leaks into the box pass and ``bbox_map_small``
    doubles.
    """
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
