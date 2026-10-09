# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""State stored by the one-pass COCO adapter's ``update()``.

The adapter encodes masks to COCO RLE itself, on CUDA when the masks are there and on the CPU otherwise, instead of
letting TorchMetrics copy every mask to host memory and encode it there. Everything it stores must still be exactly what
TorchMetrics' own ``update()`` stores, because every backend evaluates that state.
"""

import warnings
from typing import Any

import pytest
import torch
from torchmetrics.detection import MeanAveragePrecision

from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

#: COCO RLE edge-case names exercised by the mask-state parity test.
_MASK_CASES = [
    "no_masks",
    "empty_1d",
    "no_pixels",
    "background",
    "foreground",
    "first_pixel",
    "last_pixel",
    "single_row",
    "single_column",
    "checkerboard",
    "random",
    "uint8",
    "large_runs",
    "count_at_group_boundary",
    "non_contiguous",
    "mask_innermost_row",
]
#: Devices every parity test runs on; the CUDA case needs the ``gpu`` marker and a visible device.
_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")],
    ),
]
#: Arguments for the adapter and its reference, stock TorchMetrics, which encodes masks with faster-coco-eval.
_KWARGS: dict[str, Any] = {"iou_type": ("bbox", "segm"), "backend": "faster_coco_eval"}


def _mask_stack(case: str) -> torch.Tensor:
    """Return a ``(K, H, W)`` mask stack for one COCO RLE edge case.

    ``background`` and ``foreground`` are single-run masks; ``first_pixel`` makes the leading background run empty and
    ``last_pixel`` makes the final run one pixel long. ``large_runs`` has counts that need several characters and
    differences against the count two back of both signs, and ``count_at_group_boundary`` has runs of all 900 pixels,
    a 10-bit count that needs a third 5-bit group only for its sign bit. ``checkerboard`` changes on every pixel.
    ``non_contiguous`` is a transposed view, and ``mask_innermost_row`` a one-row stack whose masks are its innermost
    dimension. ``no_pixels`` masks have no pixels at all, which COCO encodes as one empty run, ``empty_1d`` is an image
    without masks given as a 1-D tensor, and ``uint8`` holds 0/1 values instead of booleans.

    Args:
        case: One of ``_MASK_CASES``.

    Returns:
        The mask stack.

    Examples:
        >>> _mask_stack("first_pixel").nonzero().tolist()
        [[0, 0, 0]]
        >>> tuple(_mask_stack("single_column").shape)
        (3, 40, 1)
        >>> _mask_stack("mask_innermost_row").stride()
        (1, 120, 3)
    """
    generator = torch.Generator().manual_seed(0)
    masks = torch.zeros((2, 300, 400), dtype=torch.bool)
    if case == "no_masks":
        masks = masks[:0]
    elif case == "empty_1d":
        masks = torch.zeros(0, dtype=torch.bool)
    elif case == "no_pixels":
        masks = masks[:, :0]
    elif case == "foreground":
        masks[:] = True
    elif case == "first_pixel":
        masks = masks[:1]
        masks[0, 0, 0] = True
    elif case == "last_pixel":
        masks = masks[:1]
        masks[0, -1, -1] = True
    elif case == "single_row":
        masks = torch.rand((3, 1, 40), generator=generator) > 0.5
    elif case == "single_column":
        masks = torch.rand((3, 40, 1), generator=generator) > 0.5
    elif case == "checkerboard":
        masks = ((torch.arange(23)[:, None] + torch.arange(17)) % 2 == 0)[None]
    elif case == "random":
        masks = torch.rand((8, 37, 29), generator=generator) > 0.5
    elif case == "uint8":
        masks = (torch.rand((8, 37, 29), generator=generator) > 0.5).to(torch.uint8)
    elif case == "large_runs":
        masks[0, 40:260, 25:380] = True
        masks[1, 150:, :200] = True
        masks[1, 10:20, 300:302] = True
    elif case == "count_at_group_boundary":
        masks = torch.zeros((2, 30, 30), dtype=torch.bool)
        masks[0] = True
    elif case == "non_contiguous":
        masks = (torch.rand((4, 29, 37), generator=generator) > 0.5).transpose(1, 2)
    elif case == "mask_innermost_row":
        masks = (torch.rand((1, 40, 3), generator=generator) > 0.5).permute(2, 0, 1)
    return masks


def _segmentation_batch(
    masks: torch.Tensor, device: str = "cpu"
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Wrap a mask stack as one image of predictions, with the mirrored stack as that image's ground truth.

    Args:
        masks: ``(K, H, W)`` masks, one per detection.
        device: Device every returned tensor is placed on.

    Returns:
        TorchMetrics-format predictions and targets for one image.

    Examples:
        >>> preds, targets = _segmentation_batch(torch.ones((2, 4, 4), dtype=torch.bool))
        >>> tuple(preds[0]["masks"].shape), targets[0]["labels"].tolist()
        ((2, 4, 4), [0, 0])
    """
    count = masks.shape[0]
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]]).repeat(count, 1)
    labels = torch.zeros(count, dtype=torch.long)
    preds = [{"boxes": boxes, "scores": torch.linspace(0.9, 0.1, count), "labels": labels, "masks": masks}]
    targets = [{"boxes": boxes.clone(), "labels": labels.clone(), "masks": masks.flip(-1)}]
    to_device = [{name: value.to(device) for name, value in item.items()} for item in preds + targets]
    return to_device[:1], to_device[1:]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("case", _MASK_CASES)
def test_update_stores_the_masks_torchmetrics_stores(case: str, device: str) -> None:
    """Each stored mask must be the ``(size, counts)`` pair TorchMetrics' per-mask host encoding stores, byte for byte.

    The reference always encodes on the CPU; the adapter encodes where the masks are, so the CUDA case also proves the
    device path produces host-side bytes identical to the CPU encoder's.
    """
    reference = MeanAveragePrecision(**_KWARGS)
    reference.update(*_segmentation_batch(_mask_stack(case)))
    metric = OnePassCocoMeanAveragePrecision(**_KWARGS)

    metric.update(*_segmentation_batch(_mask_stack(case), device))

    assert (metric.detection_mask, metric.groundtruth_mask) == (reference.detection_mask, reference.groundtruth_mask)


@pytest.mark.parametrize("chunk_pixels", [1, 3500, 10**9])
def test_stored_masks_do_not_depend_on_the_encoding_chunk(monkeypatch: pytest.MonkeyPatch, chunk_pixels: int) -> None:
    """Encoding a stack in chunks of masks must not move a run across masks or change any count.

    The eight 37x29 masks are encoded one per chunk, three per chunk with a shorter last chunk, and all at once.
    """
    reference = MeanAveragePrecision(**_KWARGS)
    reference.update(*_segmentation_batch(_mask_stack("random")))
    monkeypatch.setattr("rfdetr.training.coco_map._RLE_CHUNK_PIXELS", chunk_pixels)
    metric = OnePassCocoMeanAveragePrecision(**_KWARGS)

    metric.update(*_segmentation_batch(_mask_stack("random")))

    assert (metric.detection_mask, metric.groundtruth_mask) == (reference.detection_mask, reference.groundtruth_mask)


def _state_case(case: str, device: str = "cpu") -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Return one image of predictions and ground truth exercising one rule of TorchMetrics' stored state.

    ``crowd_and_area`` gives the ground truth ``iscrowd`` and ``area``, ``no_crowd_or_area`` leaves both to their
    defaults, and ``empty_1d_boxes`` has 1-D empty boxes on both sides, which TorchMetrics stores as ``(1, 0)`` instead
    of converting them.

    Args:
        case: ``crowd_and_area``, ``no_crowd_or_area`` or ``empty_1d_boxes``.
        device: Device every returned tensor is placed on.

    Returns:
        TorchMetrics-format predictions and targets for one image.

    Examples:
        >>> preds, targets = _state_case("crowd_and_area")
        >>> targets[0]["iscrowd"].tolist(), tuple(preds[0]["masks"].shape)
        ([0, 1], (2, 8, 8))
    """
    masks = torch.zeros((2, 8, 8), dtype=torch.bool)
    masks[0, 1:4, 1:5] = True
    masks[1, 4:, 2:] = True
    boxes = torch.tensor([[1.0, 1.0, 5.0, 4.0], [2.0, 4.0, 8.0, 8.0]])
    labels = torch.tensor([1, 2])
    if case == "empty_1d_boxes":
        empty_labels = torch.empty(0, dtype=torch.long)
        preds = {"boxes": torch.empty(0), "scores": torch.empty(0), "labels": empty_labels, "masks": masks[:0]}
        target = {"boxes": torch.empty(0), "labels": empty_labels.clone(), "masks": masks[:0]}
    else:
        preds = {"boxes": boxes, "scores": torch.tensor([0.9, 0.4]), "labels": labels, "masks": masks}
        target = {"boxes": boxes.clone(), "labels": labels.clone(), "masks": masks.clone()}
    if case == "crowd_and_area":
        target.update(iscrowd=torch.tensor([0, 1]), area=torch.tensor([12.0, 0.0]))
    return [{name: value.to(device) for name, value in preds.items()}], [
        {name: value.to(device) for name, value in target.items()}
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("iou_type", ["bbox", "segm", pytest.param(("bbox", "segm"), id="both")])
@pytest.mark.parametrize("case", ["crowd_and_area", "no_crowd_or_area", "empty_1d_boxes"])
def test_update_stores_the_box_label_score_and_crowd_state_torchmetrics_stores(
    case: str, iou_type: Any, device: str
) -> None:
    """Boxes, scores, labels, crowd flags and areas must be stored as TorchMetrics stores them, as CPU tensors.

    The reference gets CPU inputs, so the CUDA case also proves every stored tensor reaches the CPU with its dtype.
    """
    reference = MeanAveragePrecision(iou_type=iou_type, backend="faster_coco_eval")
    reference.update(*_state_case(case))
    metric = OnePassCocoMeanAveragePrecision(iou_type=iou_type, backend="faster_coco_eval")

    metric.update(*_state_case(case, device))

    states = [name for name, default in reference._defaults.items() if isinstance(default, list)]
    mismatched = [
        name
        for name in states
        if len(getattr(metric, name)) != len(getattr(reference, name))
        or not all(
            stored.device.type == "cpu" and stored.dtype == expected.dtype and torch.equal(stored, expected)
            if isinstance(expected, torch.Tensor)
            else stored == expected
            for stored, expected in zip(getattr(metric, name), getattr(reference, name))
        )
    ]
    assert mismatched == []


def _warns_on_update(metric: MeanAveragePrecision, num_predictions: int, num_targets: int, enabled: bool) -> bool:
    """Update *metric* with one image of identical detections and report whether it warned about too many.

    Args:
        metric: A metric whose last maximum-detection threshold the detections are compared with.
        num_predictions: Detections on the image.
        num_targets: Ground-truth objects on the image.
        enabled: Value for the metric's ``warn_on_many_detections``.

    Returns:
        Whether ``update()`` raised TorchMetrics' too-many-detections warning.

    Examples:
        >>> _warns_on_update(MeanAveragePrecision(max_detection_thresholds=[1, 1, 2]), 3, 1, True)
        True
        >>> _warns_on_update(MeanAveragePrecision(max_detection_thresholds=[1, 1, 2]), 2, 1, True)
        False
    """
    metric.warn_on_many_detections = enabled
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]).repeat(num_predictions, 1),
            "scores": torch.full((num_predictions,), 0.5),
            "labels": torch.zeros(num_predictions, dtype=torch.long),
            "masks": torch.ones((num_predictions, 4, 4), dtype=torch.bool),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]).repeat(num_targets, 1),
            "labels": torch.zeros(num_targets, dtype=torch.long),
            "masks": torch.ones((num_targets, 4, 4), dtype=torch.bool),
        }
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric.update(preds, targets)
    return any("Encountered more than" in str(warning.message) for warning in caught)


@pytest.mark.parametrize(
    ("iou_type", "num_predictions", "num_targets", "enabled", "expected"),
    [
        pytest.param("bbox", 3, 1, True, True, id="bbox-above-limit"),
        pytest.param("segm", 3, 1, True, True, id="segm-above-limit"),
        pytest.param("bbox", 2, 1, True, False, id="at-limit"),
        pytest.param("bbox", 1, 3, True, False, id="targets-above-limit"),
        pytest.param("bbox", 3, 1, False, False, id="disabled"),
    ],
)
def test_too_many_detections_warns_as_torchmetrics_does(
    iou_type: str, num_predictions: int, num_targets: int, enabled: bool, expected: bool
) -> None:
    """The too-many-detections warning must fire exactly when TorchMetrics' own ``update()`` fires it.

    Only predictions count, against the last of ``max_detection_thresholds`` (2 here), and only while
    ``warn_on_many_detections`` is set.
    """
    kwargs: dict[str, Any] = {
        "iou_type": iou_type,
        "backend": "faster_coco_eval",
        "max_detection_thresholds": [1, 1, 2],
    }
    stock = MeanAveragePrecision(**kwargs)
    adapter = OnePassCocoMeanAveragePrecision(**kwargs)

    warned = [_warns_on_update(metric, num_predictions, num_targets, enabled) for metric in (stock, adapter)]

    assert warned == [expected, expected]
