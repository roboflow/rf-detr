# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.utilities.box_ops import (
    box_iou,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
    generalized_box_iou,
    masks_to_boxes,
)


def _random_xyxy_boxes(n: int, seed: int = 0) -> torch.Tensor:
    """Generate ``n`` non-degenerate random boxes in xyxy format."""
    gen = torch.Generator().manual_seed(seed)
    xy1 = torch.rand(n, 2, generator=gen)
    xy2 = xy1 + torch.rand(n, 2, generator=gen) * 0.5 + 0.01
    return torch.cat([xy1, xy2], dim=-1)


def test_elementwise_box_iou_matches_pairwise_diagonal() -> None:
    """Elementwise IoU/union equal the diagonal of the pairwise ``box_iou``, including gradients."""
    boxes1 = _random_xyxy_boxes(64, seed=2).requires_grad_(True)
    boxes2 = _random_xyxy_boxes(64, seed=3).requires_grad_(True)

    boxes1_ref = boxes1.detach().clone().requires_grad_(True)
    boxes2_ref = boxes2.detach().clone().requires_grad_(True)

    iou, union = elementwise_box_iou(boxes1, boxes2)
    iou_ref, union_ref = box_iou(boxes1_ref, boxes2_ref)

    torch.testing.assert_close(iou, torch.diag(iou_ref))
    torch.testing.assert_close(union, torch.diag(union_ref))

    iou.sum().backward()
    torch.diag(iou_ref).sum().backward()

    torch.testing.assert_close(boxes1.grad, boxes1_ref.grad)
    torch.testing.assert_close(boxes2.grad, boxes2_ref.grad)


def test_elementwise_generalized_box_iou_matches_pairwise_diagonal() -> None:
    """Elementwise GIoU equals the diagonal of the pairwise ``generalized_box_iou``, including gradients."""
    boxes1 = _random_xyxy_boxes(64, seed=0).requires_grad_(True)
    boxes2 = _random_xyxy_boxes(64, seed=1).requires_grad_(True)

    boxes1_ref = boxes1.detach().clone().requires_grad_(True)
    boxes2_ref = boxes2.detach().clone().requires_grad_(True)

    result = elementwise_generalized_box_iou(boxes1, boxes2)
    expected = torch.diag(generalized_box_iou(boxes1_ref, boxes2_ref))

    torch.testing.assert_close(result, expected)

    result.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(boxes1.grad, boxes1_ref.grad)
    torch.testing.assert_close(boxes2.grad, boxes2_ref.grad)


def test_elementwise_box_iou_zero_area_boxes_are_finite() -> None:
    """Zero-area boxes yield finite elementwise IoU/union instead of a 0/0 NaN."""
    zero_box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # w = h = 0

    iou, union = elementwise_box_iou(zero_box, zero_box)

    assert torch.isfinite(iou).all()
    assert torch.isfinite(union).all()


def test_elementwise_generalized_box_iou_zero_area_boxes_are_finite() -> None:
    """Degenerate zero-area boxes give finite elementwise GIoU instead of NaN/inf."""
    zero_box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # w = h = 0

    giou = elementwise_generalized_box_iou(zero_box, zero_box)

    assert torch.isfinite(giou).all()


def test_box_iou_zero_area_boxes_are_finite() -> None:
    """Zero-area boxes yield finite IoU/union instead of a 0/0 NaN."""
    zero_box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # w = h = 0

    iou, union = box_iou(zero_box, zero_box)

    assert torch.isfinite(iou).all()
    assert torch.isfinite(union).all()


def test_generalized_box_iou_zero_area_boxes_are_finite() -> None:
    """Degenerate zero-area boxes give finite GIoU instead of NaN/inf."""
    zero_box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # w = h = 0

    giou = generalized_box_iou(zero_box, zero_box)

    assert torch.isfinite(giou).all()


def test_masks_to_boxes_passes_ij_indexing_to_meshgrid(monkeypatch) -> None:
    """`masks_to_boxes` should call `torch.meshgrid` with explicit ij indexing."""
    original_meshgrid = torch.meshgrid
    call_count = 0

    def _meshgrid_with_indexing_assertion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("indexing") != "ij":
            raise AssertionError("torch.meshgrid must be called with indexing='ij'")
        return original_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", _meshgrid_with_indexing_assertion)

    masks = torch.zeros((1, 2, 3), dtype=torch.bool)
    masks[0, 0, 1] = True
    masks[0, 1, 2] = True

    boxes = masks_to_boxes(masks)

    assert call_count == 1
    assert boxes.shape == (1, 4)


def test_masks_to_boxes_builds_grid_on_masks_device(monkeypatch) -> None:
    """`masks_to_boxes` should construct arange tensors on the same device as masks."""
    original_arange = torch.arange
    observed_devices = []

    def _arange_with_device_capture(*args, **kwargs):
        observed_devices.append(kwargs.get("device"))
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", _arange_with_device_capture)

    masks = torch.zeros((1, 2, 3), dtype=torch.bool)
    masks[0, 1, 2] = True

    boxes = masks_to_boxes(masks)

    assert boxes.shape == (1, 4)
    assert observed_devices
    assert all(device == masks.device for device in observed_devices)
