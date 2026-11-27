# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for RF-DETR's one-pass TorchMetrics COCO adapter: parity, backend contract, empty inputs and DDP."""

from collections.abc import Callable
from typing import Any, get_args
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytest_mock import MockerFixture
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

from rfdetr.config import CocoEvalBackend
from rfdetr.training.coco_map import _BACKENDS, OnePassCocoMeanAveragePrecision
from tests._markers import _requires_gloo_spawn
from tests.training.coco_map.helpers import (
    _ALL_BACKENDS,
    _BACKEND_KWARGS,
    _BACKEND_ONLY,
    _NON_STREAMING_BACKENDS,
    _assert_results_close,
    _assert_streams_when_backend_streams,
    multiclass_detection_state,
)


def _single_box_state() -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Return one image whose single class-3 detection matches its ground-truth box exactly.

    Examples:
        >>> predictions, targets = _single_box_state()
        >>> len(predictions), int(targets[0]["labels"][0])
        (1, 3)
    """
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    predictions = [{"boxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([3])}]
    return predictions, [{"boxes": box, "labels": torch.tensor([3])}]


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


def test_class_metrics_use_one_coco_evaluator(mocker: MockerFixture) -> None:
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
    mocker.patch.object(
        type(metric._coco_backend), "cocoeval", new_callable=PropertyMock, return_value=evaluator_factory
    )

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
    metric.update(*_single_box_state())

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
    _assert_results_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_adapter_matches_torchmetrics_on_nontrivial_multiclass_multiimage_data() -> None:
    """One-pass extraction must match stock TorchMetrics when every compared value is not a trivial 1.0/0.0/-1.0.

    ``test_bbox_and_segmentation_results_match_torchmetrics`` above uses one image, one class, one perfect
    match — every compared value collapses to a degenerate sentinel, so an axis-order slip or class-permutation
    bug in the adapter's one-pass reduction would still pass. This fixture spans three images and three
    non-contiguous classes with a partial-overlap match, a missed detection, and a false positive, so per-class
    AP/AR values are genuinely fractional and a reduction bug has real values to disagree on.
    """
    predictions, targets = multiclass_detection_state()
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
    _assert_results_close(actual, expected, rtol=1e-4, atol=1e-6)


class TestHoistedDetectionScores:
    """Prediction COCO datasets built with per-image score conversion instead of TorchMetrics' per-annotation read.

    Pinned to ``faster_coco_eval``: box-only evaluation on hotcoco and ufcoco loads detections from an array instead of
    building annotation dicts, so the hoist these tests describe only runs here and on the mask-only path.
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

    def test_scores_are_converted_once_for_each_image(self, mocker: MockerFixture) -> None:
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

        mocker.patch.object(CocoBackend, "_get_coco_format", recorded)
        mocker.patch.object(torch.Tensor, "cpu", new=record_cpu)

        metric._coco_datasets(metric._observed_classes())

        prediction_call = recorded.call_args_list[-1]
        assert prediction_call.kwargs["scores"] is None
        assert prediction_call.kwargs["labels"] is metric.detection_labels
        assert score_conversion_shapes == [(2,), (1,)]

    def test_annotation_count_mismatch_is_rejected(self, mocker: MockerFixture) -> None:
        """A prediction annotation count that no longer matches stored scores must fail loudly.

        Assigning scores positionally is only sound while upstream emits one annotation per stored detection. If its
        loop ever starts dropping or adding annotations, silently zipping the shorter of the two would attach wrong
        scores to real detections and quietly corrupt every reported mAP.
        """
        metric = self._updated_metric()

        mocker.patch.object(
            CocoBackend, "_get_coco_format", return_value={"images": [], "annotations": [], "categories": []}
        )

        with pytest.raises(RuntimeError) as exc_info:
            metric._coco_datasets(metric._observed_classes())
        assert "prediction annotations" in str(exc_info.value)

    def test_mask_only_state_uses_stock_construction(self, mocker: MockerFixture) -> None:
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

        mocker.patch.object(CocoBackend, "_get_coco_datasets", recorded)

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

        with pytest.raises(ValueError) as exc_info:
            metric._coco_datasets(metric._observed_classes())
        assert "expected floating point" in str(exc_info.value)

    def test_column_vector_scores_are_rejected(self) -> None:
        """Column-vector scores must fail instead of becoming nested COCO score lists.

        TorchMetrics' original per-annotation conversion rejects a list result. The hoisted conversion must preserve
        that scalar-score contract before assigning annotations, because nested scores can otherwise survive until a
        later backend operation and obscure the input error.
        """
        metric = self._updated_metric()
        metric.detection_scores = [scores.unsqueeze(1) for scores in metric.detection_scores]

        with pytest.raises(ValueError) as exc_info:
            metric._coco_datasets(metric._observed_classes())
        assert "one-dimensional" in str(exc_info.value)


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
    metric = OnePassCocoMeanAveragePrecision(backend=backend, class_metrics=True, **_BACKEND_KWARGS.get(backend, {}))
    metric.update(
        [{"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}],
        [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.long)}],
    )
    _assert_streams_when_backend_streams(metric)

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
    assert set(_BACKENDS) == set(_BACKEND_ONLY), "register a new backend in the shared test lists too"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"extended_summary": True}, "extended_summary", id="extended-summary"),
        pytest.param({"backend": "pycocotools"}, "faster_coco_eval", id="pycocotools-backend"),
        pytest.param({"average": "micro"}, "average='macro'", id="micro-average"),
        pytest.param({"sync_on_compute": True}, "sync_on_compute=False", id="sync-on-compute"),
    ],
)
def test_adapter_rejects_unsupported_result_and_backend_modes(kwargs: dict[str, Any], message: str) -> None:
    """Unsupported modes must fail at construction instead of bypassing the adapter's memory and backend contract."""
    with pytest.raises(ValueError) as exc_info:
        OnePassCocoMeanAveragePrecision(**kwargs)
    assert message in str(exc_info.value)


def test_adapter_rejects_stale_torchmetrics_state_contract(mocker: MockerFixture) -> None:
    """Construction must fail loudly when the installed list-state schema no longer matches the adapter contract."""
    mocker.patch("rfdetr.training.coco_map._MAP_STATE_ATTRS", ("missing_state",))

    with pytest.raises(RuntimeError) as exc_info:
        OnePassCocoMeanAveragePrecision()
    assert "incompatible with installed torchmetrics" in str(exc_info.value)


def test_adapter_rejects_non_callable_coco_backend_factory(mocker: MockerFixture) -> None:
    """Construction must reject a non-callable COCO dataset factory before metric computation."""
    mocker.patch.object(CocoBackend, "coco", new=object())

    with pytest.raises(RuntimeError) as exc_info:
        OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")

    assert "missing backend methods: ['coco']" in str(exc_info.value)


def test_adapter_rejects_backend_method_missing_a_relied_on_parameter(mocker: MockerFixture) -> None:
    """Construction must fail when a backend method drops a keyword compute() calls by name.

    Existence-only checks would miss an upstream rename of e.g. ``average``/``prefix`` on the installed backend helpers;
    the adapter's compute() calls those by keyword, so a silent rename would otherwise surface as a raw TypeError deep
    inside compute() instead of at construction.
    """
    mocker.patch(
        "rfdetr.training.coco_map._BACKEND_METHOD_PARAMS",
        {"_get_coco_datasets": ("not_a_real_parameter",), "_coco_stats_to_tensor_dict": ()},
    )

    with pytest.raises(RuntimeError) as exc_info:
        OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")
    assert "incompatible signature" in str(exc_info.value)


def test_hotcoco_contract_ignores_the_helper_it_never_calls(mocker: MockerFixture) -> None:
    """A rename in ``_get_coco_datasets`` must not block the backend that never calls it.

    hotcoco builds its index in the COCO constructor, so this adapter assembles the dataset dictionaries itself and
    never reaches that helper. Guarding it for hotcoco would fail construction over an upstream change that cannot
    affect the metrics it produces — while the same rename must still fail loudly on faster-coco-eval.
    """
    stale_contract = {"_get_coco_datasets": ("not_a_real_parameter",), "_coco_stats_to_tensor_dict": ()}

    mocker.patch("rfdetr.training.coco_map._BACKEND_METHOD_PARAMS", stale_contract)

    OnePassCocoMeanAveragePrecision(backend="hotcoco")
    with pytest.raises(RuntimeError) as exc_info:
        OnePassCocoMeanAveragePrecision(backend="faster_coco_eval")
    assert "incompatible signature" in str(exc_info.value)


class TestMismatchedBackendSignatures:
    """Unit coverage for the two contract-validation static helpers in isolation."""

    @pytest.mark.parametrize(
        ("method_name", "method", "expected"),
        [
            pytest.param(
                "_get_coco_datasets",
                lambda self, groundtruth_labels, average: None,
                ["_get_coco_datasets"],
                id="missing-parameters",
            ),
            pytest.param(
                "_coco_stats_to_tensor_dict",
                lambda self, stats, prefix, max_detection_thresholds, extra=None: None,
                [],
                id="superset-of-required-parameters",
            ),
            pytest.param(
                "_get_coco_format",
                lambda self, labels, /, all_labels, boxes, masks, scores, crowds, area, iou_type, average: None,
                ["_get_coco_format"],
                id="positional-only-parameter",
            ),
        ],
    )
    def test_flags_only_signatures_that_break_keyword_calls(
        self, method_name: str, method: Callable[..., None], expected: list[str]
    ) -> None:
        """A backend method missing a relied-on name or taking one positional-only is flagged; a superset is not.

        A parameter-name-only check accepts the positional-only signature even though ``_coco_datasets`` passes every
        ``_get_coco_format`` argument by keyword. Construction-time rejection turns a private API shift into the
        adapter's actionable compatibility error instead of a raw TypeError after state accumulation.
        """
        backend = type("_Backend", (), {method_name: method})()

        mismatched = OnePassCocoMeanAveragePrecision._mismatched_backend_signatures(backend, [method_name])

        assert mismatched == expected


class TestEvaluatorMethodsNowRequiringArgs:
    """Unit coverage for the evaluator zero-arg-method regression check."""

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            pytest.param(lambda self, threshold: None, ["evaluate"], id="gained-required-argument"),
            pytest.param(lambda self: None, [], id="stayed-zero-arg"),
            pytest.param(lambda self, threshold=0.5: None, [], id="new-argument-has-default"),
        ],
    )
    def test_flags_only_methods_that_gained_a_required_argument(
        self, method: Callable[..., None], expected: list[str]
    ) -> None:
        """A previously zero-arg evaluator method is reported only when it now requires an argument.

        A new parameter with a default value does not break zero-argument calls.
        """
        evaluator = type("_Evaluator", (), {"evaluate": method})

        mismatched = OnePassCocoMeanAveragePrecision._evaluator_methods_now_requiring_args(evaluator, ["evaluate"])

        assert mismatched == expected


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


def test_distributed_merge_concatenates_every_metric_state(mocker: MockerFixture) -> None:
    """The explicit DDP boundary must gather every list state once and mark empty ranks as updated."""
    mocker.patch("rfdetr.training.coco_map.get_world_size", return_value=2)
    mocker.patch("rfdetr.training.coco_map.is_dist_avail_and_initialized", return_value=True)
    metric = OnePassCocoMeanAveragePrecision()
    metric.update(*_single_box_state())
    metric._update_count = 0

    gather = mocker.patch("rfdetr.training.coco_map.all_gather", side_effect=lambda local: [local, local])

    metric.merge_distributed_state()

    assert gather.call_count == 9
    assert len(metric.detection_box) == 2
    assert len(metric.groundtruth_area) == 2
    assert metric.has_updates is True


def test_distributed_merge_is_noop_without_process_group(mocker: MockerFixture) -> None:
    """Single-process use must leave local state untouched and issue no object gathers."""
    gather = mocker.patch("rfdetr.training.coco_map.all_gather")
    metric = OnePassCocoMeanAveragePrecision()
    metric.update(*_single_box_state())

    metric.merge_distributed_state()

    gather.assert_not_called()
    assert len(metric.detection_box) == 1


def test_distributed_merge_is_noop_for_world_size_one(mocker: MockerFixture) -> None:
    """An initialized one-rank group must not perform redundant object gathers."""
    mocker.patch("rfdetr.training.coco_map.get_world_size", return_value=1)
    mocker.patch("rfdetr.training.coco_map.is_dist_avail_and_initialized", return_value=True)
    gather = mocker.patch("rfdetr.training.coco_map.all_gather")
    metric = OnePassCocoMeanAveragePrecision()

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
            metric.update(*_single_box_state())
        metric.merge_distributed_state()
        result = metric.compute()
        assert metric.has_updates is True
        assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))
        torch.testing.assert_close(result["map_per_class"].reshape(-1), torch.ones(1), rtol=1e-4, atol=1e-6)
    finally:
        dist.destroy_process_group()


# Windows CI currently cannot run this spawn test because gloo DDP spawn fails with
# makeDeviceForHostname unsupported-device errors (see tests/training/trainer/test_trainer_smoke.py).
@pytest.mark.ddp
@_requires_gloo_spawn
@pytest.mark.parametrize("backend", _NON_STREAMING_BACKENDS)
def test_distributed_merge_supports_uneven_shards_with_empty_rank(tmp_path, backend: str) -> None:
    """Two real Gloo ranks must finish without deadlock when only rank zero receives a metric update.

    Every backend is spawned because the merge hands each rank's list states through the repo's object gather and the
    empty rank then evaluates a dataset with no annotations on whichever evaluator was selected.
    """
    init_file = tmp_path / "coco-map-gloo-init"

    mp.spawn(_distributed_empty_rank_worker, args=(2, str(init_file), backend), nprocs=2, join=True)


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
def test_max_detection_thresholds_reach_the_evaluator(backend: str) -> None:
    """A configured maximum-detection threshold must change the metric it is supposed to change.

    hotcoco 0.5's ``params`` getter returned a copy, so writing a field through it changed nothing and raised nothing;
    1.0.0 makes field writes take effect but still documents pull-edit-assign as the idiom. Without this test the
    adapter could keep evaluating at COCO's default 100 detections while RF-DETR asked for ``eval_max_dets``, and every
    metric would still look plausible.
    """
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
            backend=backend,
            max_detection_thresholds=[1, 10, max_detections],
            sync_on_compute=False,
            **_BACKEND_KWARGS.get(backend, {}),
        )
        metric.update(predictions, targets)
        _assert_streams_when_backend_streams(metric)
        return float(metric.compute()[f"mar_{max_detections}"])

    assert recall_at(2) < recall_at(500)
