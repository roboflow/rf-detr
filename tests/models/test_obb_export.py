# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
from unittest.mock import MagicMock

import torch

from rfdetr.models.lwdetr import LWDETR


def _make_exportable_oriented_lwdetr() -> LWDETR:
    """Build a minimal oriented LWDETR that can run forward_export."""
    hidden_dim = 8
    num_queries = 4
    num_classes = 3

    backbone = MagicMock()
    src = torch.randn(1, hidden_dim, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    pos = torch.randn(1, hidden_dim, 4, 4)
    # forward_export() calls the *exported* backbone, whose Joiner.forward_export
    # returns (feats, masks, poss, cross_attn_feats) — 4 values, unlike the 3-value
    # Joiner.forward used on the training path.
    backbone.return_value = ([src], [mask], [pos], None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    hs = torch.randn(1, 1, num_queries, hidden_dim)
    ref = torch.randn(1, 1, num_queries, 4)
    transformer.return_value = (hs, ref, None, None)
    transformer.decoder = MagicMock()
    transformer.decoder.bbox_embed = None

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        group_detr=1,
        oriented=True,
        bbox_reparam=False,
    )
    return model


class TestOrientedExportForward:
    def test_forward_export_output_has_angle(self) -> None:
        model = _make_exportable_oriented_lwdetr()
        model.eval()
        model._export = True
        tensors = torch.randn(1, 3, 32, 32)
        dets, labels = model.forward_export(tensors)
        assert dets.shape[-1] == 5
        assert labels.shape[-1] == 3

    def test_forward_export_angle_range(self) -> None:
        model = _make_exportable_oriented_lwdetr()
        model.eval()
        model._export = True
        tensors = torch.randn(1, 3, 32, 32)
        dets, _ = model.forward_export(tensors)
        angles = dets[..., 4]
        assert (angles >= 0).all()
        assert (angles <= math.pi).all()
