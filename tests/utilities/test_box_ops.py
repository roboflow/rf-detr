# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.utilities import box_ops
from rfdetr.utilities.box_ops import (
    box_iou,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
    generalized_box_iou,
    masks_to_boxes,
    pairwise_box_l1_cost,
)


def _random_xyxy_boxes(n: int, seed: int = 0) -> torch.Tensor:
    """Generate ``n`` non-degenerate random boxes in xyxy format.

    Examples:
        >>> boxes = _random_xyxy_boxes(2, seed=0)
        >>> boxes.shape
        torch.Size([2, 4])
        >>> bool((boxes[:, 2:] > boxes[:, :2]).all())
        True
    """
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


@pytest.mark.parametrize(
    "boxes1,boxes2",
    [
        pytest.param(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[5.0, 5.0, 6.0, 6.0]]),
            id="disjoint",
        ),
        pytest.param(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 2.0, 1.0]]),
            id="edge-touch",
        ),
        pytest.param(
            torch.tensor([[0.0, 0.0, 1e8, 1e8]]),
            torch.tensor([[5e7, 5e7, 1.5e8, 1.5e8]]),
            id="large-coord",
        ),
    ],
)
def test_elementwise_matches_pairwise_diagonal_edge_regimes(boxes1: torch.Tensor, boxes2: torch.Tensor) -> None:
    """Elementwise IoU/GIoU match the pairwise diagonal across disjoint, edge-touch, and large-coord regimes."""
    iou, _ = elementwise_box_iou(boxes1, boxes2)
    giou = elementwise_generalized_box_iou(boxes1, boxes2)

    iou_ref, _ = box_iou(boxes1, boxes2)
    giou_ref = generalized_box_iou(boxes1, boxes2)

    torch.testing.assert_close(iou, torch.diag(iou_ref))
    torch.testing.assert_close(giou, torch.diag(giou_ref))


def test_elementwise_box_iou_identical_boxes_give_exact_unit_iou() -> None:
    """Identical boxes give IoU exactly 1.0 — the union clamp preserves the identity (not just assert_close)."""
    boxes = _random_xyxy_boxes(16, seed=7)

    iou, _ = elementwise_box_iou(boxes, boxes)

    assert torch.equal(iou, torch.ones_like(iou))


def test_elementwise_generalized_box_iou_identical_boxes_give_exact_unit_giou() -> None:
    """Identical boxes give GIoU exactly 1.0 (enclosing area equals union, so the correction is zero)."""
    boxes = _random_xyxy_boxes(16, seed=7)

    giou = elementwise_generalized_box_iou(boxes, boxes)

    assert torch.equal(giou, torch.ones_like(giou))


def test_elementwise_box_iou_mixed_degeneracy_batch_is_finite_and_matches_diagonal() -> None:
    """A normal/zero-area/disjoint batch stays finite and its non-degenerate rows match the pairwise diagonal."""
    boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 1.0, 1.0]])
    boxes2 = torch.tensor([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 11.0, 11.0]])

    iou, union = elementwise_box_iou(boxes1, boxes2)
    iou_ref, _ = box_iou(boxes1, boxes2)

    assert torch.isfinite(iou).all()
    assert torch.isfinite(union).all()
    torch.testing.assert_close(iou[[0, 2]], torch.diag(iou_ref)[[0, 2]])


def test_elementwise_box_iou_degenerate_row_does_not_pollute_neighbour_grads() -> None:
    """A degenerate zero-area row keeps the gradients of its neighbour rows finite under ``backward()``."""
    boxes1 = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 1.0, 1.0]],
        requires_grad=True,
    )
    boxes2 = torch.tensor(
        [[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 11.0, 11.0]],
        requires_grad=True,
    )

    iou, _ = elementwise_box_iou(boxes1, boxes2)
    iou.sum().backward()

    assert torch.isfinite(boxes1.grad[[0, 2]]).all()
    assert torch.isfinite(boxes2.grad[[0, 2]]).all()


def test_elementwise_box_iou_empty_input_returns_empty() -> None:
    """Empty (N=0) input returns empty IoU/union tensors without error."""
    empty = torch.empty(0, 4)

    iou, union = elementwise_box_iou(empty, empty)

    assert iou.shape == (0,)
    assert union.shape == (0,)


def test_elementwise_generalized_box_iou_empty_input_returns_empty() -> None:
    """Empty (N=0) input returns an empty GIoU tensor without error."""
    empty = torch.empty(0, 4)

    giou = elementwise_generalized_box_iou(empty, empty)

    assert giou.shape == (0,)


def test_elementwise_box_iou_rejects_unequal_length() -> None:
    """Mismatched operand lengths raise ValueError instead of silently broadcasting a length-1 side."""
    boxes1 = _random_xyxy_boxes(3, seed=4)
    boxes2 = _random_xyxy_boxes(1, seed=5)

    with pytest.raises(ValueError, match="same length"):
        elementwise_box_iou(boxes1, boxes2)


def test_elementwise_generalized_box_iou_rejects_unequal_length() -> None:
    """The GIoU variant also raises ValueError on mismatched operand lengths."""
    boxes1 = _random_xyxy_boxes(3, seed=4)
    boxes2 = _random_xyxy_boxes(1, seed=5)

    with pytest.raises(ValueError, match="same length"):
        elementwise_generalized_box_iou(boxes1, boxes2)


@pytest.mark.parametrize(
    "iou_fn",
    [
        pytest.param(box_iou, id="box_iou"),
        pytest.param(elementwise_box_iou, id="elementwise_box_iou"),
        pytest.param(generalized_box_iou, id="generalized_box_iou"),
        pytest.param(elementwise_generalized_box_iou, id="elementwise_generalized_box_iou"),
    ],
)
def test_zero_area_boxes_are_finite(iou_fn) -> None:
    """Degenerate zero-area boxes yield finite results (no 0/0 NaN) across every IoU/GIoU variant."""
    zero_box = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # w = h = 0

    result = iou_fn(zero_box, zero_box)
    tensors = result if isinstance(result, tuple) else (result,)

    assert all(torch.isfinite(t).all() for t in tensors)


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


class TestPairwiseBoxL1Cost:
    """`pairwise_box_l1_cost` replaces `torch.cdist(..., p=1)` in the matcher.

    The matcher feeds its output to a Hungarian solve, so equality with `cdist`
    has to hold exactly rather than approximately: a reassociated sum would
    change which assignment wins among near-ties.
    """

    @pytest.mark.parametrize(
        ("queries", "targets"),
        [(1, 1), (7, 1), (1, 7), (300, 12), (64, 97)],
    )
    def test_matches_cdist_for_two_dimensional_inputs(self, queries: int, targets: int) -> None:
        """The full-cartesian matcher path passes `[queries, 4]` against `[targets, 4]`."""
        boxes1 = _random_xyxy_boxes(queries, seed=1)
        boxes2 = _random_xyxy_boxes(targets, seed=2)

        assert torch.equal(pairwise_box_l1_cost(boxes1, boxes2), torch.cdist(boxes1, boxes2, p=1))

    @pytest.mark.parametrize(
        ("batch", "queries", "targets"),
        [(1, 8, 3), (4, 64, 12), (5, 130, 37)],
    )
    def test_matches_cdist_for_batched_inputs(self, batch: int, queries: int, targets: int) -> None:
        """The compact matcher path passes a leading batch (or batch x layer) dimension."""
        boxes1 = _random_xyxy_boxes(batch * queries, seed=3).reshape(batch, queries, 4)
        boxes2 = _random_xyxy_boxes(batch * targets, seed=4).reshape(batch, targets, 4)

        assert torch.equal(pairwise_box_l1_cost(boxes1, boxes2), torch.cdist(boxes1, boxes2, p=1))

    def test_matches_cdist_when_chunking_is_required(self, monkeypatch) -> None:
        """Chunking the target axis must not change any value.

        The budget is lowered so several chunks are needed for a small tensor, which is what makes this a test of the
        chunked branch rather than of the single-shot one.
        """
        monkeypatch.setattr(box_ops, "_L1_COST_ELEMENT_BUDGET", 64)
        boxes1 = _random_xyxy_boxes(2 * 17, seed=5).reshape(2, 17, 4)
        boxes2 = _random_xyxy_boxes(2 * 23, seed=6).reshape(2, 23, 4)

        assert torch.equal(pairwise_box_l1_cost(boxes1, boxes2), torch.cdist(boxes1, boxes2, p=1))

    def test_empty_target_set_returns_empty_cost(self) -> None:
        """An image with no ground-truth boxes contributes zero columns, not an error."""
        boxes1 = _random_xyxy_boxes(5, seed=7)
        boxes2 = torch.zeros((0, 4))

        cost = pairwise_box_l1_cost(boxes1, boxes2)

        assert cost.shape == (5, 0)

    def test_empty_query_set_returns_empty_cost(self) -> None:
        """Symmetric guard: no predictions means no rows."""
        cost = pairwise_box_l1_cost(torch.zeros((0, 4)), _random_xyxy_boxes(3, seed=8))

        assert cost.shape == (0, 3)

    def test_preserves_dtype_and_device(self) -> None:
        """The cost inherits the boxes' dtype, which the criterion relies on downstream."""
        boxes1 = _random_xyxy_boxes(4, seed=9).to(torch.float64)
        boxes2 = _random_xyxy_boxes(2, seed=10).to(torch.float64)

        cost = pairwise_box_l1_cost(boxes1, boxes2)

        assert cost.dtype is torch.float64
        assert cost.device == boxes1.device

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_matches_cdist_on_cuda_at_matcher_scale(self) -> None:
        """Equality has to hold on the device and at the shape that motivated the change.

        `[5 layers x 8 images, 300 queries x 13 groups, targets]` is what RF-DETR Medium's compact path builds during
        training.
        """
        boxes1 = torch.rand(40, 3900, 4, device="cuda")
        boxes2 = torch.rand(40, 30, 4, device="cuda")

        assert torch.equal(pairwise_box_l1_cost(boxes1, boxes2), torch.cdist(boxes1, boxes2, p=1))
