# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the hotcoco dataset-building path and backend capability check of the one-pass COCO adapter."""

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torchmetrics.detection.helpers import CocoBackend

from rfdetr.training import coco_map
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision, _HotCocoBackend, _HotCocoStreamingBackend

_CANVAS = 96


def _rect_mask(top: int, left: int, bottom: int, right: int) -> torch.Tensor:
    """Return a boolean ``_CANVAS`` x ``_CANVAS`` mask that is set on one rectangle.

    Args:
        top: First masked row.
        left: First masked column.
        bottom: Row past the last masked one.
        right: Column past the last masked one.

    Returns:
        The mask.

    Examples:
        >>> int(_rect_mask(10, 10, 30, 30).sum())
        400
    """
    mask = torch.zeros(_CANVAS, _CANVAS, dtype=torch.bool)
    mask[top:bottom, left:right] = True
    return mask


def _records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Predictions and targets whose boxes differ from their masks' boxes and whose areas straddle COCO buckets.

    Image 0: a target with no stored area whose 20x20 mask is *small* but whose 40x40 box is *medium*, so the
    fallback area rule decides its bucket; its prediction has the exact mask but a 30x30 box (box IoU 0.5625), so a
    box derived from the mask (IoU 0.25) would drop the match. A second target stores area 500 (*small*) over a
    *medium* mask, and a higher-scoring false positive outranks both. Image 1 holds a crowd target and a
    second-class miss, image 2 a false positive on an image with no targets, image 3 a target with no predictions.

    Returns:
        Predictions and targets in TorchMetrics' input format, boxes ``xyxy``.

    Examples:
        >>> predictions, targets = _records()
        >>> len(predictions), len(targets), predictions[0]["masks"].shape
        (4, 4, torch.Size([3, 96, 96]))
    """
    empty_masks = torch.zeros(0, _CANVAS, _CANVAS, dtype=torch.bool)
    predictions = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0], [50.0, 50.0, 88.0, 88.0], [10.0, 60.0, 22.0, 72.0]]),
            "masks": torch.stack([_rect_mask(10, 10, 30, 30), _rect_mask(52, 52, 88, 88), _rect_mask(60, 10, 70, 20)]),
            "scores": torch.tensor([0.9, 0.8, 0.95]),
            "labels": torch.tensor([1, 2, 1]),
        },
        {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
            "masks": _rect_mask(5, 5, 40, 40).unsqueeze(0),
            "scores": torch.tensor([0.7]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 12.0, 12.0]]),
            "masks": _rect_mask(0, 0, 10, 10).unsqueeze(0),
            "scores": torch.tensor([0.6]),
            "labels": torch.tensor([2]),
        },
        {
            "boxes": torch.empty(0, 4),
            "masks": empty_masks,
            "scores": torch.empty(0),
            "labels": torch.empty(0, dtype=torch.long),
        },
    ]
    targets = [
        {
            "boxes": torch.tensor([[5.0, 5.0, 45.0, 45.0], [50.0, 50.0, 90.0, 90.0]]),
            "masks": torch.stack([_rect_mask(10, 10, 30, 30), _rect_mask(50, 50, 90, 90)]),
            "labels": torch.tensor([1, 2]),
            "area": torch.tensor([0.0, 500.0]),
            "iscrowd": torch.tensor([0, 0]),
        },
        {
            "boxes": torch.tensor([[0.0, 0.0, 50.0, 50.0], [60.0, 60.0, 80.0, 80.0]]),
            "masks": torch.stack([_rect_mask(0, 0, 50, 50), _rect_mask(60, 60, 80, 80)]),
            "labels": torch.tensor([1, 2]),
            "area": torch.tensor([0.0, 0.0]),
            "iscrowd": torch.tensor([1, 0]),
        },
        {
            "boxes": torch.empty(0, 4),
            "masks": empty_masks,
            "labels": torch.empty(0, dtype=torch.long),
            "area": torch.empty(0),
            "iscrowd": torch.empty(0, dtype=torch.long),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
            "masks": _rect_mask(20, 20, 40, 40).unsqueeze(0),
            "labels": torch.tensor([1]),
            "area": torch.tensor([0.0]),
            "iscrowd": torch.tensor([0]),
        },
    ]
    return predictions, targets


def _updated_metric(backend: str, iou_type: Any) -> OnePassCocoMeanAveragePrecision:
    """Return a per-class metric on ``backend`` that has seen :func:`_records`.

    Args:
        backend: The ``eval_backend`` value.
        iou_type: One IoU type or a tuple of them.

    Returns:
        The updated metric.

    Examples:
        >>> len(_updated_metric("hotcoco", "bbox").groundtruth_labels)
        4
    """
    predictions, targets = _records()
    if "segm" not in iou_type:
        for record in (*predictions, *targets):
            del record["masks"]
    metric = OnePassCocoMeanAveragePrecision(
        backend=backend, iou_type=iou_type, class_metrics=True, sync_on_compute=False
    )
    metric.update(copy.deepcopy(predictions), copy.deepcopy(targets))
    return metric


_IOU_TYPES = [
    "bbox",
    "segm",
    pytest.param(("bbox", "segm"), id="bbox-segm"),
    pytest.param(("segm", "bbox"), id="segm-bbox"),
]


class TestHotcocoArrayIngest:
    """Hotcoco datasets built from columns must evaluate exactly as the annotation-dict path does."""

    @pytest.mark.parametrize("iou_type", _IOU_TYPES)
    def test_matches_faster_coco_eval(self, iou_type: Any) -> None:
        """Every metric must equal faster-coco-eval's to the bit, which reads TorchMetrics' own COCO dictionaries.

        The records put the fallback target area, the stored target area, crowd targets and box-versus-mask geometry in
        positions where a deviation from upstream's per-annotation rules changes an AP bucket.
        """
        expected = _updated_metric("faster_coco_eval", iou_type).compute()

        observed = _updated_metric("hotcoco", iou_type).compute()

        assert observed.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True, msg=key)

    @pytest.mark.parametrize("iou_type", _IOU_TYPES)
    def test_matches_the_annotation_dict_path(self, iou_type: Any) -> None:
        """Every metric must equal hotcoco's own result over the TorchMetrics dictionaries the array path replaces."""
        dict_path = _updated_metric("hotcoco", iou_type)
        with patch.object(dict_path, "_loads_detections_from_array", return_value=False):
            expected = dict_path.compute()

        observed = _updated_metric("hotcoco", iou_type).compute()

        assert observed.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True, msg=key)

    def test_predicted_boxes_are_not_derived_from_masks(self) -> None:
        """Loaded predictions must keep their stored boxes, which here differ from their masks' bounding boxes."""
        metric = _updated_metric("hotcoco", ("bbox", "segm"))
        expected = [box.tolist() for image in metric.detection_box for box in image.reshape(-1, 4)]

        coco_preds, _, _ = metric._coco_datasets(metric._observed_classes())

        assert [annotation["bbox"] for annotation in coco_preds.load_anns(sorted(coco_preds.get_ann_ids()))] == expected

    def test_builds_no_annotation_dicts(self) -> None:
        """A box-and-mask state must reach hotcoco without TorchMetrics' per-annotation COCO dictionaries."""
        metric = _updated_metric("hotcoco", ("bbox", "segm"))

        with patch.object(CocoBackend, "_get_coco_format") as get_coco_format:
            metric.compute()

        get_coco_format.assert_not_called()

    def test_mask_only_state_keeps_the_annotation_dict_path(self) -> None:
        """A segmentation-only state stores no boxes, which ``from_arrays`` and the detection array both need."""
        metric = _updated_metric("hotcoco", "segm")

        _, _, prediction_dataset = metric._coco_datasets(metric._observed_classes())

        assert isinstance(prediction_dataset, dict)

    def test_empty_state_matches_faster_coco_eval(self) -> None:
        """A metric that saw no update must report what faster-coco-eval reports instead of failing to build."""
        expected = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", iou_type=("bbox", "segm")).compute()

        observed = OnePassCocoMeanAveragePrecision(backend="hotcoco", iou_type=("bbox", "segm")).compute()

        assert observed.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True, msg=key)

    def test_floating_point_target_labels_are_rejected(self) -> None:
        """A float target label must raise upstream's ``ValueError``, not ``from_arrays``' ``TypeError``."""
        metric = _updated_metric("hotcoco", "bbox")
        metric.groundtruth_labels[0] = metric.groundtruth_labels[0].float()

        with pytest.raises(ValueError, match="Invalid input class of sample 0"):
            metric.compute()


class TestHotcocoCapabilityCheck:
    """An installed hotcoco older than the ``rfdetr[train]`` floor must be refused when the backend is built.

    The floor is enforced only when the environment is resolved; a stale install would otherwise fail with an
    ``AttributeError`` inside ``compute()`` after a whole validation epoch.
    """

    @staticmethod
    def _stale_package(**symbols: object) -> SimpleNamespace:
        """Return a stand-in hotcoco module whose ``COCO`` type carries only the given attributes.

        Args:
            **symbols: Attributes to set on the stand-in ``COCO`` type.

        Returns:
            A module-like namespace with ``COCO``, ``COCOeval`` and ``mask``.

        Examples:
            >>> package = TestHotcocoCapabilityCheck._stale_package(update_anns=None)
            >>> hasattr(package.COCO, "update_anns"), hasattr(package, "StreamingEval")
            (True, False)
        """
        return SimpleNamespace(COCO=type("COCO", (), symbols), COCOeval=object, mask=object)

    def test_stale_release_is_refused_as_too_old(self) -> None:
        """A release without a symbol the batch path calls must raise an upgrade hint, not a missing-package one."""
        with (
            patch.object(coco_map, "_hotcoco", return_value=self._stale_package()),
            patch("importlib.metadata.version", return_value="1.0.1"),
            pytest.raises(ImportError, match=r"hotcoco 1\.0\.1 is too old.*pip install -U 'rfdetr\[train\]'"),
        ):
            _HotCocoBackend()

    def test_streaming_backend_also_requires_streaming_eval(self) -> None:
        """The streaming backend must refuse a release that has every batch-path symbol but no ``StreamingEval``."""
        package = self._stale_package(**dict.fromkeys(name.split(".")[-1] for name in _HotCocoBackend.required_symbols))

        with (
            patch.object(coco_map, "_hotcoco", return_value=package),
            pytest.raises(ImportError, match=r"lacks hotcoco\.StreamingEval\."),
        ):
            _HotCocoStreamingBackend()

    def test_batch_backend_accepts_a_release_without_streaming_eval(self) -> None:
        """The batch backend must not demand ``StreamingEval``, which only the streaming backend calls."""
        package = self._stale_package(**dict.fromkeys(name.split(".")[-1] for name in _HotCocoBackend.required_symbols))

        with patch.object(coco_map, "_hotcoco", return_value=package):
            _HotCocoBackend()
