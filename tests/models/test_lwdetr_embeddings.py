# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for ``return_embeddings`` support in LWDETR's eager and exported forward passes."""

from unittest.mock import MagicMock

import torch
from torch import nn

from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def _build_feature_batch(batch_size: int, hidden_dim: int) -> list[NestedTensor]:
    return [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]


def _make_detection_model(
    *,
    batch_size: int = 2,
    num_queries: int = 3,
    hidden_dim: int = 8,
    num_classes: int = 5,
    num_decoder_layers: int = 2,
    two_stage: bool = False,
    segmentation_head: nn.Module | None = None,
) -> tuple[LWDETR, MagicMock]:
    """Build an LWDETR detection model whose backbone/transformer are mocked with fixed-shape outputs.

    The mock backbone's ``return_value`` is a plain 3-tuple ``(features, poss, cross_attn_features)``, matching the
    eager forward's unpacking (``forward``). For the exported/traced path (``forward_export``), which unpacks a 4-tuple
    ``(feats, masks, poss, cross_attn_feats)``, tests reconfigure ``backbone.return_value`` before calling ``export()``.
    """
    features = _build_feature_batch(batch_size=batch_size, hidden_dim=hidden_dim)
    poss = [torch.zeros(batch_size, hidden_dim, 4, 4)]

    backbone = MagicMock()
    backbone.return_value = (features, poss, None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = (
        torch.zeros(num_decoder_layers, batch_size, num_queries, hidden_dim),  # hs (all decoder layers)
        torch.zeros(num_decoder_layers, batch_size, num_queries, 4),  # ref_unsigmoid
        torch.zeros(batch_size, num_queries, hidden_dim),  # hs_enc
        torch.zeros(batch_size, num_queries, 4),  # ref_enc
    )

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=two_stage,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    )
    model.eval()
    return model, transformer


def _make_export_ready_model(
    *,
    batch_size: int = 2,
    num_queries: int = 3,
    hidden_dim: int = 8,
    num_classes: int = 5,
    segmentation_head: nn.Module | None = None,
) -> LWDETR:
    """Build an LWDETR model with mocked backbone/transformer shaped for ``forward_export`` (traced/optimized path).

    Unlike :func:`_make_detection_model`, the mock backbone here returns the export-time 4-tuple ``(feats, masks, poss,
    cross_attn_feats)`` expected by ``forward_export``, and the mock transformer returns the export-time last-decoder-
    layer-only shapes ``[B, Q, H]`` (not ``[L, B, Q, H]``).
    """
    features = _build_feature_batch(batch_size=batch_size, hidden_dim=hidden_dim)
    masks = [torch.zeros(batch_size, 4, 4, dtype=torch.bool)]
    poss = [torch.zeros(batch_size, hidden_dim, 4, 4)]

    backbone = MagicMock()
    backbone.return_value = (features, masks, poss, None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = (
        torch.zeros(batch_size, num_queries, hidden_dim),  # hs (last decoder layer only)
        torch.zeros(batch_size, num_queries, 4),  # ref_unsigmoid
        torch.zeros(batch_size, num_queries, hidden_dim),  # hs_enc
        torch.zeros(batch_size, num_queries, 4),  # ref_enc
    )

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    )
    model.eval()
    return model


class TestEagerForwardReturnEmbeddings:
    """``LWDETR.forward(return_embeddings=...)`` (unoptimized/eager path)."""

    def test_return_embeddings_false_omits_embeddings_key(self) -> None:
        """By default, no 'embeddings' key is added to the output dict."""
        model, _ = _make_detection_model()
        outputs = model(torch.ones(2, 3, 8, 8), return_embeddings=False)

        assert "embeddings" not in outputs

    def test_return_embeddings_true_adds_embeddings_with_expected_shape(self) -> None:
        """return_embeddings=True adds 'embeddings' with shape [B, Q, H] from the last decoder layer only."""
        batch_size, num_queries, hidden_dim, num_decoder_layers = 2, 3, 8, 2
        model, _ = _make_detection_model(
            batch_size=batch_size,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            num_decoder_layers=num_decoder_layers,
        )

        outputs = model(torch.ones(batch_size, 3, 8, 8), return_embeddings=True)

        assert "embeddings" in outputs
        assert outputs["embeddings"].shape == (batch_size, num_queries, hidden_dim)

    def test_return_embeddings_does_not_affect_other_outputs(self) -> None:
        """Turning on return_embeddings must not change pred_logits/pred_boxes shapes or values."""
        model, _ = _make_detection_model()
        x = torch.ones(2, 3, 8, 8)

        torch.manual_seed(0)
        out_without = model(x, return_embeddings=False)
        torch.manual_seed(0)
        out_with = model(x, return_embeddings=True)

        assert torch.equal(out_without["pred_logits"], out_with["pred_logits"])
        assert torch.equal(out_without["pred_boxes"], out_with["pred_boxes"])


class TestExportForwardReturnEmbeddings:
    """``LWDETR.export(return_embeddings=...)`` + traced ``forward_export`` (optimized path)."""

    def test_export_return_embeddings_false_appends_nothing(self) -> None:
        """export(return_embeddings=False) keeps the base 2-tuple (coord, class) for detection-only models."""
        model = _make_export_ready_model()
        model.export(return_embeddings=False)

        predictions = model(torch.ones(2, 3, 8, 8))

        assert isinstance(predictions, tuple)
        assert len(predictions) == 2

    def test_export_return_embeddings_true_appends_embeddings_last(self) -> None:
        """export(return_embeddings=True) appends embeddings as the last element of the tuple."""
        batch_size, num_queries, hidden_dim = 2, 3, 8
        model = _make_export_ready_model(batch_size=batch_size, num_queries=num_queries, hidden_dim=hidden_dim)
        model.export(return_embeddings=True)

        predictions = model(torch.ones(batch_size, 3, 8, 8))

        assert isinstance(predictions, tuple)
        assert len(predictions) == 3
        coord, logits, embeddings = predictions
        assert coord.shape == (batch_size, num_queries, 4)
        assert logits.shape[:2] == (batch_size, num_queries)
        # Exported decoder only returns the last layer's hidden state -> [B, Q, H], no L*H reshape.
        assert embeddings.shape == (batch_size, num_queries, hidden_dim)

    def test_export_return_embeddings_true_with_segmentation_head_still_appends_last(self) -> None:
        """When a mask head is present, embeddings remain the last tuple element (position 3, after masks)."""
        batch_size, num_queries, hidden_dim = 2, 3, 8
        mask_h, mask_w = 4, 4
        seg_head = MagicMock()
        seg_head.return_value = [torch.zeros(batch_size, num_queries, mask_h, mask_w)]

        model = _make_export_ready_model(
            batch_size=batch_size,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            segmentation_head=seg_head,
        )
        model.export(return_embeddings=True)

        predictions = model(torch.ones(batch_size, 3, 8, 8))

        assert isinstance(predictions, tuple)
        assert len(predictions) == 4
        coord, logits, masks, embeddings = predictions
        assert masks.shape == (batch_size, num_queries, mask_h, mask_w)
        assert embeddings.shape == (batch_size, num_queries, hidden_dim)

    def test_export_return_embeddings_defaults_to_false_when_omitted(self) -> None:
        """Export() with no arguments behaves exactly like export(return_embeddings=False)."""
        model = _make_export_ready_model()
        model.export()

        predictions = model(torch.ones(2, 3, 8, 8))

        assert isinstance(predictions, tuple)
        assert len(predictions) == 2
