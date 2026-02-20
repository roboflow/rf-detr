# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.util.box_ops import masks_to_boxes


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
