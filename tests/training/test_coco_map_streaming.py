# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lifecycle tests for the ``hotcoco_streaming`` backend of RF-DETR's one-pass COCO adapter."""

import copy
import logging
import pickle
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision, _HotCocoBackend, _warn_streaming_fallback
from rfdetr.utilities.imports import _IS_HOTCOCO_INSTALLED
from tests.training.callbacks.test_coco_eval_callback import _make_pl_module, _make_trainer

_requires_hotcoco = pytest.mark.skipif(not _IS_HOTCOCO_INSTALLED, reason="hotcoco is not installed")
# The two ways a metric gets copied mid-epoch: Lightning's spawn/checkpoint plumbing pickles it, EMA set-up deep-copies.
_ROUND_TRIPS = [
    pytest.param(lambda metric: pickle.loads(pickle.dumps(metric)), id="pickle"),
    pytest.param(copy.deepcopy, id="deepcopy"),
]
_DROPPED_STREAM_MESSAGE = "copied or unpickled mid-epoch"


def _disc_records(predicted: int, annotated: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return three images of one disc each, keeping the first ``predicted`` detections and ``annotated`` objects.

    Passing ``0`` for either side leaves every image with empty, correctly shaped tensors on that side, the state an
    epoch without detections or without ground truth stores.

    Examples:
        >>> predictions, targets = _disc_records(predicted=0, annotated=1)
        >>> predictions[0]["masks"].shape, targets[0]["masks"].shape
        (torch.Size([0, 64, 64]), torch.Size([1, 64, 64]))
    """
    size, radius = 64, 10
    rows, columns = np.ogrid[:size, :size]
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for center in (20, 28, 36):
        mask = torch.from_numpy((rows - center) ** 2 + (columns - center) ** 2 <= radius**2)[None]
        box = torch.tensor([[center - radius, center - radius, center + radius + 1, center + radius + 1]]).float()
        labels = torch.tensor([1])
        predictions.append(
            {
                "boxes": box[:predicted],
                "scores": torch.tensor([0.9])[:predicted],
                "labels": labels[:predicted],
                "masks": mask[:predicted],
            }
        )
        targets.append({"boxes": box[:annotated], "labels": labels[:annotated], "masks": mask[:annotated]})
    return predictions, targets


@_requires_hotcoco
class TestStreamConstruction:
    """The metric opens its streams through the backend's ``open_stream`` seam, never through a package directly."""

    def test_opens_one_stream_per_iou_type_through_the_backend(self) -> None:
        """Each IoU type's stream comes from ``open_stream``, so a backend owns which package streams.

        A streaming backend for another package then needs to override one method instead of the metric's lifecycle.
        """
        predictions, targets = _disc_records(predicted=1, annotated=1)
        metric = OnePassCocoMeanAveragePrecision(iou_type=("bbox", "segm"), backend="hotcoco_streaming", num_classes=2)
        backend = metric._coco_backend

        with patch.object(backend, "open_stream", wraps=backend.open_stream) as open_stream:
            metric.update(predictions, targets)

        assert [call.args[1] for call in open_stream.call_args_list] == ["bbox", "segm"]

    def test_non_streaming_backend_refuses_to_open_a_stream(self) -> None:
        """A backend that leaves ``streams`` false has no stream to open and says so.

        The hook's default is what a backend inheriting the flag's ``False`` gets, so a stray call fails loudly.
        """
        with pytest.raises(NotImplementedError, match="does not stream"):
            _HotCocoBackend().open_stream(
                [], "bbox", iou_thresholds=[0.5], rec_thresholds=[0.0], max_detection_thresholds=[1, 10, 100]
            )


@_requires_hotcoco
class TestCopyMidEpoch:
    """A copy taken mid-epoch drops its open stream, and the copy, the object that falls back, logs why."""

    @pytest.fixture(autouse=True)
    def _fresh_fallback_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forget earlier fallback warnings, logged once per process and reason, and let ``caplog`` see new ones.

        The ``rf-detr`` logger does not propagate to the root logger that ``caplog`` listens on.
        """
        _warn_streaming_fallback.cache_clear()
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

    @pytest.mark.parametrize("round_trip", _ROUND_TRIPS)
    def test_copy_logs_the_dropped_stream(self, round_trip: Any, caplog: pytest.LogCaptureFixture) -> None:
        """Restoring a metric whose stream was dropped logs the batch fallback instead of taking it silently.

        The docs promise a fallback that logs why; without the log the copy's slower epoch-end has no explanation.
        """
        predictions, targets = _disc_records(predicted=1, annotated=1)
        metric = OnePassCocoMeanAveragePrecision(backend="hotcoco_streaming", num_classes=2)
        metric.update(predictions, targets)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            round_trip(metric)

        assert [record.getMessage() for record in caplog.records if _DROPPED_STREAM_MESSAGE in record.getMessage()]

    @pytest.mark.parametrize("round_trip", _ROUND_TRIPS)
    def test_source_keeps_its_stream(self, round_trip: Any) -> None:
        """Copying never touches the source: it keeps streaming, so the warning belongs to the copy alone.

        ``__getstate__`` runs on the source, which is why the drop is reported from ``__setstate__`` instead.
        """
        predictions, targets = _disc_records(predicted=1, annotated=1)
        metric = OnePassCocoMeanAveragePrecision(backend="hotcoco_streaming", num_classes=2)
        metric.update(predictions, targets)

        round_trip(metric)

        assert metric._streams is not None

    @pytest.mark.parametrize("round_trip", _ROUND_TRIPS)
    def test_copy_without_an_open_stream_logs_nothing(self, round_trip: Any, caplog: pytest.LogCaptureFixture) -> None:
        """A metric copied before its first batch, as DDP spawn copies it, drops nothing and so reports nothing.

        Warning there would announce a fallback on every spawned run although each rank streams normally.
        """
        metric = OnePassCocoMeanAveragePrecision(backend="hotcoco_streaming", num_classes=2)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            round_trip(metric)

        assert not [record for record in caplog.records if _DROPPED_STREAM_MESSAGE in record.getMessage()]


def _assert_same_metrics(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """Assert two ``compute()`` results carry the same keys and bit-identical values.

    Examples:
        >>> _assert_same_metrics({"map": torch.tensor(0.5)}, {"map": torch.tensor(0.5)})
    """
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key].reshape(-1), expected[key].reshape(-1)), key


@_requires_hotcoco
class TestStreamedDetectionForms:
    """Each IoU type's stream gets the detection form that gives it the area the batch path uses."""

    @pytest.mark.parametrize("iou_type", ["bbox", "segm", ("bbox", "segm")])
    def test_bbox_streams_an_array_and_segm_streams_mask_dicts(self, iou_type: Any) -> None:
        """``bbox`` gets one ``(N, 7)`` array; ``segm`` gets dicts without a box, so its area stays the mask's.

        hotcoco 1.2 gives an array row its box area even with ``segmentation=``, which would move masks between the area
        buckets the batch path assigns them to.
        """
        predictions, targets = _disc_records(predicted=1, annotated=1)
        metric = OnePassCocoMeanAveragePrecision(iou_type=iou_type, backend="hotcoco_streaming", num_classes=2)
        metric.update(predictions, targets)

        detections = metric._stream_detections(num_images=3, offset=0)

        expected = {"bbox": (np.ndarray, (3, 7)), "segm": (list, 3)}
        assert {
            name: (type(value), value.shape if isinstance(value, np.ndarray) else len(value))
            for name, value in detections.items()
        } == {name: expected[name] for name in metric.iou_type}
        assert all("bbox" not in record for record in detections.get("segm", []))

    @pytest.mark.parametrize("iou_type", ["bbox", ("bbox", "segm")])
    def test_multi_image_batch_with_an_undetected_image_matches_batch_hotcoco(self, iou_type: Any) -> None:
        """A batch mixing detected and undetected images streams to exactly the metrics ``hotcoco`` reports.

        The image without detections contributes no array row, so later rows must still carry the right image ids.
        """
        detected, targets = _disc_records(predicted=1, annotated=1)
        undetected, _ = _disc_records(predicted=0, annotated=1)
        predictions = [detected[0], undetected[1], detected[2]]
        kwargs: dict[str, Any] = {"iou_type": iou_type, "class_metrics": True, "num_classes": 2}
        streamed = OnePassCocoMeanAveragePrecision(backend="hotcoco_streaming", **kwargs)
        streamed.update(predictions, targets)
        batch = OnePassCocoMeanAveragePrecision(backend="hotcoco", **kwargs)
        batch.update(predictions, targets)
        assert streamed._streams is not None

        _assert_same_metrics(streamed.compute(), batch.compute())


@_requires_hotcoco
@pytest.mark.parametrize("iou_type", ["bbox", "segm", ("bbox", "segm")])
@pytest.mark.parametrize(
    ("predicted", "annotated"),
    [pytest.param(0, 1, id="no-detections"), pytest.param(1, 0, id="no-ground-truth")],
)
def test_empty_side_state_matches_batch_hotcoco(iou_type: Any, predicted: int, annotated: int) -> None:
    """An epoch with no detection or no ground truth reports exactly what the batch path reports.

    The batch path owns the ``-1`` sentinels COCO reports for an empty side; streaming must not diverge. A segm-only
    epoch without predicted masks is where the paths split: TorchMetrics drops every prediction image, so the batch
    path reports sentinels while a stream would still finalize its ground-truth images.
    """
    predictions, targets = _disc_records(predicted=predicted, annotated=annotated)
    kwargs: dict[str, Any] = {"iou_type": iou_type, "class_metrics": True, "num_classes": 2}
    streamed = OnePassCocoMeanAveragePrecision(backend="hotcoco_streaming", **kwargs)
    streamed.update(predictions, targets)
    batch = OnePassCocoMeanAveragePrecision(backend="hotcoco", **kwargs)
    batch.update(predictions, targets)

    _assert_same_metrics(streamed.compute(), batch.compute())


@_requires_hotcoco
@pytest.mark.parametrize("class_metrics", [True, False])
def test_float_detection_labels_are_refused_while_streaming(class_metrics: bool) -> None:
    """A whole-valued float detection label raises the batch path's ``ValueError`` as the batch is streamed.

    hotcoco 1.2.1 rejects only a fractional id, so ``1.0`` would otherwise reach ``compute()``: an ``IndexError`` with
    per-class metrics, a silently accepted label without them.
    """
    predictions, targets = _disc_records(predicted=1, annotated=1)
    predictions[0]["labels"] = predictions[0]["labels"].float()
    metric = OnePassCocoMeanAveragePrecision(
        iou_type="bbox", backend="hotcoco_streaming", class_metrics=class_metrics, num_classes=2
    )

    with pytest.raises(ValueError, match="expected integer labels"):
        metric.update(predictions, targets)


@_requires_hotcoco
def test_float_detection_labels_are_refused_by_the_batch_backend() -> None:
    """The batch backend raises the same ``ValueError`` at ``compute()``, so both paths agree on the message."""
    predictions, targets = _disc_records(predicted=1, annotated=1)
    predictions[0]["labels"] = predictions[0]["labels"].float()
    metric = OnePassCocoMeanAveragePrecision(iou_type="bbox", backend="hotcoco", class_metrics=True, num_classes=2)
    metric.update(predictions, targets)

    with pytest.raises(ValueError, match="expected integer labels"):
        metric.compute()


@_requires_hotcoco
def test_float_detection_labels_in_a_later_batch_name_their_epoch_wide_sample() -> None:
    """A float label arriving after a valid batch is refused with its index in the whole epoch, as on the batch path.

    The second batch's image is sample ``1`` of the epoch; a batch-local count would call it sample ``0``.
    """
    predictions, targets = _disc_records(predicted=1, annotated=1)
    metric = OnePassCocoMeanAveragePrecision(
        iou_type="bbox", backend="hotcoco_streaming", class_metrics=False, num_classes=2
    )
    metric.update(predictions[:1], targets[:1])
    predictions[1]["labels"] = predictions[1]["labels"].float()

    with pytest.raises(ValueError, match=r"sample 1 \(expected integer labels"):
        metric.update(predictions[1:2], targets[1:2])


@_requires_hotcoco
@pytest.mark.parametrize("label", [1.0, 5.0])
def test_float_ground_truth_labels_are_refused_while_streaming(label: float) -> None:
    """A float target label in dictionary-backed ground truth raises, and streaming stays on.

    ``5.0`` lies outside the two declared categories: without the check it would take the range fallback and abandon
    streaming silently instead of raising, which is the behavior the check changes. ``1.0`` is in range, where
    TorchMetrics' own conversion error would otherwise name the label instead.
    """
    predictions, targets = _disc_records(predicted=1, annotated=1)
    targets[0]["labels"] = torch.tensor([label])
    metric = OnePassCocoMeanAveragePrecision(
        iou_type="bbox", backend="hotcoco_streaming", class_metrics=False, num_classes=2
    )

    with pytest.raises(ValueError, match="expected integer labels"):
        metric.update(predictions, targets)

    assert not metric._stream_stopped


@_requires_hotcoco
def test_callback_streams_validation_only_not_the_train_split() -> None:
    """``hotcoco_streaming`` streams validation only; the train-split metric evaluates in one batch on ``hotcoco``.

    Training-step updates must not run matching, so only the validation metric gets the streaming backend. With
    ``compute_train_metrics=True`` a streaming train metric would match inside ``on_train_batch_end``.
    """
    callback = COCOEvalCallback(eval_backend="hotcoco_streaming")

    callback.setup(_make_trainer(), _make_pl_module(), stage="fit")

    assert (callback.map_metric._coco_backend.streams, callback.map_metric_train._coco_backend.streams) == (
        True,
        False,
    )
