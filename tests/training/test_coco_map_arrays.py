# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""One-pass COCO adapter tests: evaluator and detection arrays, per-IoU-type area buckets and hotcoco streaming."""

import copy
import logging
import pickle
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision, _ufcoco_evaluator_type, _warn_streaming_fallback
from tests._markers import (
    _requires_faster_coco_eval,
    _requires_hotcoco,
    _requires_pycocotools,
    _requires_ufcoco,
    _requires_vernier,
)
from tests.training._coco_map_support import (
    _ALL_BACKENDS,
    _BACKEND_KWARGS,
    _BOX_ARRAY_BACKENDS,
    _assert_results_close,
    _assert_streams_when_backend_streams,
    _metric_inputs,
    multiclass_detection_state,
)

# pytest-doctestplus has no ``+REQUIRES`` directive, so a helper doctest that builds a hotcoco metric is gated on the
# optional package here instead, and skips rather than fails where the ``train`` extra is not installed.
__doctest_requires__ = {"_metric_fed_per_image": ["hotcoco"]}


def _warning_messages(records: list[logging.LogRecord]) -> list[str]:
    """List the message of each distinct WARNING emission among *records*, in order.

    pytest 9.1 installs its capture handler on the non-propagating ``rf-detr`` logger *and* on the root logger, so
    once a test forces ``propagate = True`` one emission lands in ``caplog.records`` twice. Both entries are the same
    ``LogRecord`` object; two real emissions are two objects, so counting objects still tells them apart.

    Examples:
        >>> def _record(message, level=logging.WARNING):
        ...     return logging.LogRecord("rf-detr", level, __file__, 0, message, (), None)
        >>> once, again, info = _record("once"), _record("once"), _record("fyi", logging.INFO)
        >>> _warning_messages([once, once, info])
        ['once']
        >>> _warning_messages([once, again])
        ['once', 'once']
    """
    distinct = {id(record): record for record in records if record.levelno == logging.WARNING}
    return [record.getMessage() for record in distinct.values()]


class TestUfcocoArraysMatchPycocotools:
    """The evaluator arrays the per-class reduction reads must be byte-identical to pycocotools' on the same datasets.

    The parity tests above compare the float32 tensors RF-DETR reports; this one goes one level down, through the
    adapter's own datasets, and checks the float64 precision, recall and score arrays the per-class AP/AR reduction is
    computed from. pycocotools is the reference ufcoco reproduces, and ``rfdetr[train]`` installs it.
    """

    @_requires_ufcoco
    def test_evaluator_keeps_the_inherited_collect(self) -> None:
        """The adapter's evaluator subclass must not define its own ``_collect``.

        ufcoco keeps array-loaded detections in columns only while the evaluator's ``_collect`` is its own; an override
        on the subclass would silently send every evaluation back through per-annotation dictionaries.
        """
        assert "_collect" not in vars(_ufcoco_evaluator_type())

    @_requires_pycocotools
    @_requires_ufcoco
    @pytest.mark.parametrize("iou_type", ["bbox", "segm", pytest.param(("bbox", "segm"), id="both")])
    @pytest.mark.parametrize("max_dets", [100, 500])
    def test_evaluator_arrays_are_byte_identical(self, iou_type: Any, max_dets: int) -> None:
        """Precision, recall and score arrays must match pycocotools byte for byte.

        Holds across incremental updates and reuse of the metric after ``reset()``.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

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
            oracle_gt = COCO()
            oracle_gt.dataset = copy.deepcopy(coco_target.dataset)
            oracle_gt.createIndex()
            if prediction_dataset is None:
                # Box-only state loads detections from one array; pycocotools loads the same array.
                assert tuple(actual.iou_type) == ("bbox",)
                oracle_dt = oracle_gt.loadRes(actual._detection_results_array())
            else:
                oracle_dt = COCO()
                oracle_dt.dataset = copy.deepcopy(prediction_dataset)
                oracle_dt.createIndex()
            for kind in actual.iou_type:
                if len(actual.iou_type) > 1:
                    for dataset in (prediction_dataset, oracle_dt.dataset):
                        for annotation in dataset["annotations"]:
                            annotation["area"] = annotation[f"area_{kind}"]
                evaluator = actual._coco_backend.cocoeval(coco_target, coco_preds, iouType=kind)
                oracle = COCOeval(oracle_gt, oracle_dt, iouType=kind)
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


@pytest.mark.parametrize(
    ("backend", "iou_type", "from_array"),
    [
        pytest.param("ufcoco", "bbox", True, marks=_requires_ufcoco),
        pytest.param("hotcoco", "bbox", True, marks=_requires_hotcoco),
        pytest.param("faster_coco_eval", "bbox", False, marks=_requires_faster_coco_eval),
        pytest.param("ufcoco", "segm", False, marks=_requires_ufcoco),
        pytest.param("ufcoco", ("bbox", "segm"), False, marks=_requires_ufcoco, id="ufcoco-both"),
    ],
)
def test_box_only_predictions_load_from_one_array(backend: str, iou_type: Any, from_array: bool) -> None:
    """Box-only hotcoco and ufcoco state must reach ``loadRes`` as one array, never as one dict per detection.

    faster-coco-eval's own ``loadRes`` is slower than the dict path, and an array carries no segmentation, so both keep
    building the prediction dataset.
    """
    predictions, targets = _metric_inputs()
    metric = OnePassCocoMeanAveragePrecision(backend=backend, iou_type=iou_type)
    metric.update(copy.deepcopy(predictions), copy.deepcopy(targets))
    metric.merge_distributed_state()
    _, _, prediction_dataset = metric._coco_datasets(metric._observed_classes())
    assert (prediction_dataset is None) is from_array


class TestDetectionArrayLabelValidation:
    """The detection-array path must reject floating-point labels the way upstream's annotation-dict path does.

    The array's category column is float64 whatever the stored dtype, so nothing downstream of it sees a fractional
    label: ufcoco evaluated ``3.5`` as a valid class, and hotcoco failed on it with a ``TypeError`` from its own
    ``COCO`` constructor or ``loadRes`` instead of upstream's ``ValueError``.
    """

    @pytest.mark.parametrize("backend", _BOX_ARRAY_BACKENDS)
    @pytest.mark.parametrize("label", [3.5, 3.0])
    def test_rejects_floating_point_labels(self, backend: str, label: float) -> None:
        """A floating-point detection label must raise upstream's ``ValueError``, whole-valued or not.

        Upstream rejects any label that is not a Python ``int`` after conversion, so ``3.0`` fails as ``3.5`` does;
        accepting the whole-valued one would make the two paths disagree on the same state.
        """
        metric = OnePassCocoMeanAveragePrecision(backend=backend, sync_on_compute=False)
        metric.update(
            [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([label]),
                }
            ],
            [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])}],
        )

        with pytest.raises(ValueError) as exc_info:
            metric.compute()
        assert "Invalid input class of sample 0" in str(exc_info.value)

    @_requires_ufcoco
    def test_accepts_empty_floating_point_labels(self) -> None:
        """An image with no detections must pass with a ``float32`` label tensor, as it does upstream.

        ``torch.tensor([])`` defaults to ``float32``; upstream checks labels element by element, so an empty tensor has
        nothing to reject, and a dtype check without that guard would refuse an ordinary empty prediction. hotcoco is
        left out: the float tensor promotes every observed class ID to float, which its ``loadRes`` rejects whatever
        the detection array's label dtype -- a separate limitation this check does not touch.
        """
        metric = OnePassCocoMeanAveragePrecision(backend="ufcoco", sync_on_compute=False)
        metric.update(
            [
                {"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0)},
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([3]),
                },
            ],
            [
                {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
                {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
            ],
        )

        result = metric.compute()

        assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


class TestDetectionArrayEdgeCasesMatchFasterCocoEval:
    """Edge cases of the detection array must evaluate as faster-coco-eval's annotation-dict path does.

    hotcoco and ufcoco load box-only detections from one array built from the stored state; these pin the stored shape
    the array construction reshapes and the row count the evaluator, not the array, has to truncate.
    """

    @pytest.mark.parametrize("backend", _BOX_ARRAY_BACKENDS)
    def test_one_dimensional_empty_boxes(self, backend: str) -> None:
        """An image whose prediction boxes are a 1-D empty tensor must count as zero detections, as upstream does.

        TorchMetrics stores that tensor as ``(1, 0)``; without the array's reshape back to ``(0, 4)`` it would either
        fail to concatenate or be counted as one detection, shifting later rows onto the wrong image ID.
        """
        predictions, targets = multiclass_detection_state()
        predictions[1] = {"boxes": torch.empty(0), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
        kwargs: dict[str, Any] = {"class_metrics": True, "sync_on_compute": False}
        expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
        actual_metric = OnePassCocoMeanAveragePrecision(backend=backend, **kwargs)
        expected_metric.update(predictions, targets)
        actual_metric.update(predictions, targets)

        expected = expected_metric.compute()
        actual = actual_metric.compute()

        assert actual.keys() == expected.keys()
        _assert_results_close(actual, expected)

    @pytest.mark.parametrize("backend", _BOX_ARRAY_BACKENDS)
    def test_detections_beyond_the_largest_detection_limit(self, backend: str) -> None:
        """Rows past ``maxDets[-1]`` must reach the evaluator and be truncated there, as upstream truncates them.

        The array carries every stored detection. With five same-class detections on one image and thresholds ``[1, 2,
        3]``, the two lowest-scoring true positives fall past the limit, so recall is 0.5 rather than the 1.0 an
        untruncated evaluation would report.
        """
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
                [60.0, 60.0, 70.0, 70.0],
                [80.0, 80.0, 90.0, 90.0],
            ]
        )
        predictions = [
            {"boxes": boxes, "scores": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]), "labels": torch.full((5,), 3)}
        ]
        targets = [{"boxes": boxes[[0, 2, 3, 4]], "labels": torch.full((4,), 3)}]
        kwargs: dict[str, Any] = {
            "class_metrics": True,
            "max_detection_thresholds": [1, 2, 3],
            "sync_on_compute": False,
        }
        expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
        actual_metric = OnePassCocoMeanAveragePrecision(backend=backend, **kwargs)
        expected_metric.update(predictions, targets)
        actual_metric.update(predictions, targets)

        expected = expected_metric.compute()
        actual = actual_metric.compute()

        assert float(expected["mar_3"]) == 0.5
        assert actual.keys() == expected.keys()
        _assert_results_close(actual, expected)


@pytest.mark.parametrize("backend", _ALL_BACKENDS)
@pytest.mark.parametrize("iou_type", [("bbox", "segm"), ("segm", "bbox")])
def test_multi_iou_type_areas_follow_their_own_iou_type(backend: str, iou_type: tuple[str, ...]) -> None:
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
        backend=backend,
        iou_type=iou_type,
        class_metrics=True,
        sync_on_compute=False,
        **_BACKEND_KWARGS.get(backend, {}),
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
    _assert_streams_when_backend_streams(metric)

    result = metric.compute()

    # Box pass: the false positive's 25x25 box is "small" too and outranks the true positive, so precision at full
    # recall is 1/2. Let the segmentation area leak in and the 60x60 mask makes it "medium", the false positive
    # drops out of the bucket, and this reads 1.0 instead.
    assert float(result["bbox_map_small"]) == pytest.approx(0.5)
    # Mask pass: only the true positive is "small", and its 25x25 prediction over a 20x20 target is IoU 0.64 —
    # matched at 3 of the 10 COCO thresholds.
    assert float(result["segm_map_small"]) == pytest.approx(0.3)


def _rectangle_records(*, with_predictions: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return two images of axis-aligned rectangle masks that fill their boxes exactly, so box and mask areas agree.

    Args:
        with_predictions: Whether each image carries detections; when false they are empty ``(0, 4)`` / ``(0, H, W)``.

    Returns:
        The prediction and target lists in TorchMetrics detection format.

    Examples:
        >>> predictions, targets = _rectangle_records(with_predictions=False)
        >>> predictions[0]["boxes"].shape, targets[0]["masks"].shape
        (torch.Size([0, 4]), torch.Size([1, 32, 32]))
    """
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for left in (2, 10):
        mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        mask[0, 4:20, left : left + 12] = True
        box = torch.tensor([[float(left), 4.0, float(left + 12), 20.0]])
        targets.append({"boxes": box, "masks": mask, "labels": torch.tensor([1])})
        predictions.append(
            {
                "boxes": box if with_predictions else torch.zeros(0, 4),
                "masks": mask if with_predictions else torch.zeros(0, 32, 32, dtype=torch.bool),
                "scores": torch.tensor([0.9]) if with_predictions else torch.zeros(0),
                "labels": torch.tensor([1]) if with_predictions else torch.zeros(0, dtype=torch.long),
            }
        )
    return predictions, targets


def _computed(
    backend: str, iou_type: tuple[str, ...], records: tuple[list[dict[str, Any]], list[dict[str, Any]]]
) -> dict[str, torch.Tensor]:
    """Return ``compute()`` of a ``backend`` metric fed ``records`` in one ``update()``.

    Examples:
        >>> predictions, targets = _rectangle_records(with_predictions=True)
        >>> float(_computed("faster_coco_eval", ("bbox",), (predictions, targets))["map"])
        1.0
    """
    metric = OnePassCocoMeanAveragePrecision(
        backend=backend, iou_type=iou_type, class_metrics=True, sync_on_compute=False
    )
    metric.update(*records)
    return metric.compute()


@_requires_hotcoco
class TestHotcocoMultiIouTypeMatchesFasterCocoEval:
    """A bbox+segm hotcoco run switches the active area with ``update_anns`` and must still equal the reference."""

    @pytest.mark.parametrize("iou_type", [("bbox", "segm"), ("segm", "bbox")])
    def test_identical_box_and_mask_area(self, iou_type: tuple[str, ...]) -> None:
        """Rectangle masks that fill their boxes make the area switch a no-op; the result must not move."""
        records = _rectangle_records(with_predictions=True)

        actual = _computed("hotcoco", iou_type, records)

        _assert_same_metrics(actual, _computed("faster_coco_eval", iou_type, records))
        assert float(actual["bbox_map"]) > 0, "fixture must produce a non-trivial score"

    @pytest.mark.parametrize("iou_type", [("bbox", "segm"), ("segm", "bbox")])
    def test_empty_predictions(self, iou_type: tuple[str, ...]) -> None:
        """Ground truth with no detections on any image must evaluate, not fail inside the area switch."""
        records = _rectangle_records(with_predictions=False)

        actual = _computed("hotcoco", iou_type, records)

        _assert_same_metrics(actual, _computed("faster_coco_eval", iou_type, records))


@_requires_hotcoco
def test_hotcoco_multi_iou_type_switches_areas_without_rebuilding_predictions(mocker: MockerFixture) -> None:
    """A bbox+segm run on hotcoco must build its datasets once, not once per IoU type.

    Before ``COCO.update_anns`` the only way to switch the active ``area`` on hotcoco was to build a new prediction
    dataset per IoU type. The area-bucket test above proves the switch still reaches the evaluator; this one pins
    that it no longer costs a rebuild: the array path builds one ground-truth and one prediction dataset for the whole
    computation, bypasses ``_build_coco``, and only switches the area once per IoU type.
    """
    predictions, targets = _straddling_disc_records()
    metric = OnePassCocoMeanAveragePrecision(iou_type=("bbox", "segm"), backend="hotcoco", sync_on_compute=False)
    metric.update(predictions, targets)

    build_coco = mocker.patch.object(metric, "_build_coco", wraps=metric._build_coco)
    from_arrays = mocker.patch.object(
        metric, "_hotcoco_datasets_from_arrays", wraps=metric._hotcoco_datasets_from_arrays
    )
    switch_area = mocker.patch.object(
        metric, "_prediction_dataset_for_iou_type", wraps=metric._prediction_dataset_for_iou_type
    )

    metric.compute()

    assert (build_coco.call_count, from_arrays.call_count, switch_area.call_count) == (0, 1, 2)


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


@_requires_vernier
@_requires_hotcoco
def test_two_iou_type_run_buckets_bbox_ap_by_mask_area() -> None:
    """A two-IoU-type run must bucket ``bbox`` AP by *mask* area, as upstream does.

    ``_get_coco_format`` derives the ground-truth ``area`` from the mask whenever ``segm`` is among the IoU
    types -- for the ``bbox`` pass too. Deriving it per pass instead is invisible until an object's box and
    mask areas straddle 32**2 or 96**2, and then it moves AP between the small and medium buckets with no
    error. A disc of radius 17 does exactly that: box 34x34 = 1156 (medium), mask ~901 (small).
    """
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


@_requires_vernier
@_requires_hotcoco
def test_bbox_only_run_buckets_by_box_area_and_carries_no_masks() -> None:
    """The mirror of the two-IoU-type case, and the reason ``rles`` is gated on the run.

    ``_get_coco_format`` derives the area from the mask only when ``segm`` is among the IoU types; a bbox-only run uses
    ``w * h``. vernier's ``area="auto"`` reads a mask whenever one is present, so carrying masks here would silently re-
    bucket the AP. The last assertion is the load-bearing one: what keeps masks out is the gate, not whether the state
    happens to be empty.
    """
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
@_requires_vernier
def test_autocast_dtypes_survive_the_round_trip_to_vernier(dtype: torch.dtype) -> None:
    """State stored under autocast must evaluate, whatever dtype the autocast ran in.

    ``bfloat16`` has no numpy dtype, so it used to raise ``TypeError`` out of ``np.asarray`` before vernier's f64 ingest
    boundary was reached; ``vernier>=0.5.3`` widens it there instead. Columns are handed over at their stored dtype on
    purpose — a cast here would hide that. Every value below is exactly representable in both dtypes, so the float32 run
    is a bit-exact oracle.
    """
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


def _metric_fed_per_image(
    backend: str,
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    **kwargs: Any,
) -> OnePassCocoMeanAveragePrecision:
    """Return a metric that received one ``update()`` per image, as a validation loop of batch size 1 would.

    Examples:
        >>> predictions, targets = multiclass_detection_state()
        >>> metric = _metric_fed_per_image("hotcoco", predictions, targets)
        >>> len(metric.groundtruth_labels)
        3
    """
    metric = OnePassCocoMeanAveragePrecision(backend=backend, sync_on_compute=False, **kwargs)
    for prediction, target in zip(predictions, targets):
        metric.update([prediction], [target])
    return metric


def _metric_fed_in_batches(
    backend: str,
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    batch_sizes: tuple[int, ...],
    **kwargs: Any,
) -> OnePassCocoMeanAveragePrecision:
    """Return a metric whose ``update()`` calls carried ``batch_sizes`` images each, in order.

    Examples:
        >>> predictions, targets = multiclass_detection_state()
        >>> metric = _metric_fed_in_batches("faster_coco_eval", predictions, targets, (2, 1))
        >>> len(metric.groundtruth_labels)
        3
    """
    metric = OnePassCocoMeanAveragePrecision(backend=backend, sync_on_compute=False, **kwargs)
    start = 0
    for size in batch_sizes:
        metric.update(predictions[start : start + size], targets[start : start + size])
        start += size
    return metric


def _no_detections() -> dict[str, torch.Tensor]:
    """Return the prediction of an image on which nothing was detected.

    Examples:
        >>> _no_detections()["boxes"].shape
        torch.Size([0, 4])
    """
    return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}


def _no_ground_truth() -> dict[str, torch.Tensor]:
    """Return the target of an image that holds no annotated object.

    Examples:
        >>> _no_ground_truth()["labels"].shape
        torch.Size([0])
    """
    return {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}


def _assert_same_metrics(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """Assert two ``compute()`` results carry the same keys and bit-identical values.

    Examples:
        >>> _assert_same_metrics({"map": torch.tensor(0.5)}, {"map": torch.tensor(0.5)})
    """
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1)), key


@_requires_hotcoco
class TestHotcocoStreaming:
    """``hotcoco_streaming`` matches batches in ``update()`` and must report exactly what ``hotcoco`` reports."""

    @pytest.mark.parametrize("iou_type", ["bbox", ("bbox", "segm")])
    def test_streamed_metrics_equal_batch_hotcoco(self, mocker: MockerFixture, iou_type: str | tuple[str, ...]) -> None:
        """Per-image streaming reproduces the batch evaluation bit for bit, per-class vectors included.

        The bbox+segm fixture straddles the small/medium boundary, so a detection area derived from the wrong geometry
        would move AP between buckets; the bbox fixture holds a prediction-only class.
        """
        records = multiclass_detection_state() if iou_type == "bbox" else _straddling_disc_records()
        kwargs: dict[str, Any] = {"iou_type": iou_type, "class_metrics": True, "num_classes": 90}
        streamed = _metric_fed_per_image("hotcoco_streaming", *records, **kwargs)
        assert streamed._streams is not None

        build_coco = mocker.patch.object(streamed, "_build_coco")
        actual = streamed.compute()

        build_coco.assert_not_called()
        _assert_same_metrics(actual, _metric_fed_per_image("hotcoco", *records, **kwargs).compute())

    @pytest.mark.parametrize("num_classes", [None, 10])
    def test_falls_back_to_batch_when_categories_do_not_cover_labels(self, num_classes: int | None) -> None:
        """Without a category list covering every label (fixture labels reach 42), the batch path runs instead.

        ``StreamingEval`` raises ``KeyError`` for an undeclared category, so the adapter stops the stream before handing
        it one, instead of failing the validation epoch at ``update()``.
        """
        records = multiclass_detection_state()
        streamed = _metric_fed_per_image("hotcoco_streaming", *records, class_metrics=True, num_classes=num_classes)

        assert streamed._streams is None
        _assert_same_metrics(
            streamed.compute(), _metric_fed_per_image("hotcoco", *records, class_metrics=True).compute()
        )

    @pytest.mark.parametrize(
        "round_trip",
        [
            pytest.param(lambda metric: pickle.loads(pickle.dumps(metric)), id="pickle"),
            pytest.param(copy.deepcopy, id="deepcopy"),
        ],
    )
    def test_copying_mid_epoch_falls_back_to_batch(self, round_trip: Callable[[Any], Any]) -> None:
        """A stream cannot be pickled, so a metric copied mid-epoch evaluates its whole state in one batch."""
        predictions, targets = multiclass_detection_state()
        metric = _metric_fed_per_image("hotcoco_streaming", predictions[:2], targets[:2], num_classes=90)

        restored = round_trip(metric)
        restored.update(predictions[2:], targets[2:])

        assert restored._streams is None
        _assert_same_metrics(restored.compute(), _metric_fed_per_image("hotcoco", predictions, targets).compute())

    def test_distributed_run_does_not_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under DDP each rank holds a shard that only ``merge_distributed_state()`` completes, so nothing streams."""
        monkeypatch.setattr("rfdetr.training.coco_map.is_dist_avail_and_initialized", lambda: True)
        monkeypatch.setattr("rfdetr.training.coco_map.get_world_size", lambda: 2)

        metric = _metric_fed_per_image("hotcoco_streaming", *multiclass_detection_state(), num_classes=90)

        assert metric._streams is None

    def test_reset_starts_a_fresh_stream(self) -> None:
        """``compute()`` consumes the stream; the next epoch after ``reset()`` must stream again from image 0."""
        predictions, targets = multiclass_detection_state()
        metric = _metric_fed_per_image("hotcoco_streaming", predictions, targets, num_classes=90)
        metric.compute()

        metric.reset()
        for prediction, target in zip(predictions[:2], targets[:2]):
            metric.update([prediction], [target])

        assert metric._streamed_images == 2
        _assert_same_metrics(metric.compute(), _metric_fed_per_image("hotcoco", predictions[:2], targets[:2]).compute())

    @pytest.mark.parametrize("batch_sizes", [pytest.param((2, 1), id="2-then-1"), pytest.param((3,), id="one-batch")])
    def test_multi_image_batches_of_boxes_equal_batch_hotcoco(self, batch_sizes: tuple[int, ...]) -> None:
        """Several images per ``update()`` get the same image ids as the one-shot batch path."""
        records = multiclass_detection_state()
        kwargs: dict[str, Any] = {"class_metrics": True, "num_classes": 90}
        streamed = _metric_fed_in_batches("hotcoco_streaming", *records, batch_sizes, **kwargs)
        _assert_streams_when_backend_streams(streamed)

        _assert_same_metrics(
            streamed.compute(), _metric_fed_in_batches("hotcoco", *records, (3,), class_metrics=True).compute()
        )

    def test_uneven_multi_image_batches_of_boxes_and_masks_equal_batch_hotcoco(self) -> None:
        """Unequally sized batches of box+mask images keep the id offset of every later batch right."""
        records = _straddling_disc_records()
        kwargs: dict[str, Any] = {"iou_type": ("bbox", "segm"), "class_metrics": True, "num_classes": 90}
        streamed = _metric_fed_in_batches("hotcoco_streaming", *records, (7, 11, 12), **kwargs)
        _assert_streams_when_backend_streams(streamed)

        reference = _metric_fed_in_batches("hotcoco", *records, (30,), iou_type=("bbox", "segm"), class_metrics=True)
        _assert_same_metrics(streamed.compute(), reference.compute())

    def test_image_without_detections_inside_a_batch(self) -> None:
        """An image the detector found nothing on, between two that it did, still streams the others unchanged."""
        predictions, targets = multiclass_detection_state()
        predictions[1] = _no_detections()
        streamed = _metric_fed_in_batches("hotcoco_streaming", predictions, targets, (3,), num_classes=90)
        _assert_streams_when_backend_streams(streamed)

        _assert_same_metrics(
            streamed.compute(), _metric_fed_in_batches("hotcoco", predictions, targets, (3,)).compute()
        )

    def test_image_without_ground_truth_inside_a_batch(self) -> None:
        """An image with detections but no annotated object counts its detections as false positives."""
        predictions, targets = multiclass_detection_state()
        targets[1] = _no_ground_truth()
        streamed = _metric_fed_in_batches("hotcoco_streaming", predictions, targets, (3,), num_classes=90)
        _assert_streams_when_backend_streams(streamed)

        _assert_same_metrics(
            streamed.compute(), _metric_fed_in_batches("hotcoco", predictions, targets, (3,)).compute()
        )

    def test_repeated_compute_returns_the_same_metrics(self) -> None:
        """``compute()`` twice on a streaming metric reports one result, as the batch backend does."""
        streamed = _metric_fed_per_image("hotcoco_streaming", *multiclass_detection_state(), num_classes=90)

        first = streamed.compute()

        _assert_same_metrics(streamed.compute(), first)

    def test_label_outside_the_categories_in_a_later_batch_stops_streaming(self) -> None:
        """A label beyond ``num_classes`` after a clean first batch stops the stream; the metrics stay exact."""
        predictions, targets = multiclass_detection_state()
        # Image 1 carries class 42, so ``num_classes=20`` accepts the first batch (labels 3 and 17) and not the second.
        streamed = _metric_fed_in_batches("hotcoco_streaming", predictions, targets, (1, 2), num_classes=20)

        _assert_same_metrics(
            streamed.compute(), _metric_fed_in_batches("hotcoco", predictions, targets, (3,)).compute()
        )

    def test_crowd_ground_truth_equals_batch_hotcoco(self) -> None:
        """A crowd annotation is ignored for matching identically whether the images stream or arrive at once."""
        predictions, targets = multiclass_detection_state()
        targets[0]["iscrowd"] = torch.tensor([0, 1])
        kwargs: dict[str, Any] = {"class_metrics": True}
        streamed = _metric_fed_per_image("hotcoco_streaming", predictions, targets, num_classes=90, **kwargs)
        _assert_streams_when_backend_streams(streamed)

        _assert_same_metrics(
            streamed.compute(), _metric_fed_per_image("hotcoco", predictions, targets, **kwargs).compute()
        )

    def test_unknown_class_count_logs_why_it_evaluates_in_one_batch(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting streaming without ``num_classes`` must say once that the batch path runs instead."""
        # The package logger does not propagate to caplog's root handler; the cached warning is process-wide.
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        _warn_streaming_fallback.cache_clear()

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            OnePassCocoMeanAveragePrecision(backend="hotcoco_streaming", sync_on_compute=False)

        assert _warning_messages(caplog.records) == [
            "eval_backend='hotcoco_streaming' evaluates in one batch at epoch end instead: "
            "the model's num_classes is unknown."
        ]

    def test_label_outside_the_categories_logs_the_declared_range(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stopping the stream mid-epoch on an out-of-range label must name the range the model declared."""
        # The package logger does not propagate to caplog's root handler; the cached warning is process-wide.
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        _warn_streaming_fallback.cache_clear()
        predictions, targets = multiclass_detection_state()

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _metric_fed_in_batches("hotcoco_streaming", predictions, targets, (1, 2), num_classes=20)

        assert "a label falls outside the declared categories [0, 20]" in caplog.text
