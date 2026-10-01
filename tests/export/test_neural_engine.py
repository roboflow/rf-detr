# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Graph rewrites that keep an exported RF-DETR on the Apple Neural Engine (``coreml_neural_engine=True``)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rfdetr import RFDETRNano
from rfdetr.export._neural_engine import one_hot_top_rows, split_einsum_encoder
from rfdetr.models.transformer import Transformer
from rfdetr.utilities.reproducibility import seed_all


def _two_stage_transformer() -> Transformer:
    """Build a small eval-mode two-stage transformer, like the released detection models (``bbox_reparam=True``).

    Examples:
        >>> transformer = _two_stage_transformer()
        >>> transformer.two_stage, transformer.bbox_reparam, transformer.training
        (True, True, False)
    """
    hidden_dim, num_queries = 16, 5
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=2,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=2,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=True,
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 3)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])
    return transformer.eval()


class TestOneHotQuerySelection:
    def test_transformer_outputs_match_top_k_gather(self) -> None:
        torch.manual_seed(0)
        transformer = _two_stage_transformer()
        hidden_dim, feature_size = 16, 6
        srcs = [torch.randn(1, hidden_dim, feature_size, feature_size)]
        masks = [torch.zeros(1, feature_size, feature_size, dtype=torch.bool)]
        pos_embeds = [torch.randn(1, hidden_dim, feature_size, feature_size)]
        query_inputs = (torch.rand(5, 4), torch.randn(5, hidden_dim))

        with torch.no_grad():
            reference = transformer(srcs, masks, pos_embeds, *query_inputs)
            transformer.select_top_rows = one_hot_top_rows
            candidate = transformer(srcs, masks, pos_embeds, *query_inputs)

        reference_tensors, _ = torch.utils._pytree.tree_flatten(reference)
        candidate_tensors, _ = torch.utils._pytree.tree_flatten(candidate)
        assert len(candidate_tensors) == len(reference_tensors)
        for expected, actual in zip(reference_tensors, candidate_tensors):
            assert (actual is None and expected is None) or torch.equal(actual, expected)

    def test_equal_scores_select_the_lower_index_first(self) -> None:
        scores = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]])
        rows = torch.arange(5.0).reshape(1, 5, 1)

        selected = one_hot_top_rows(scores, 4)(rows)

        assert selected.flatten().tolist() == [1.0, 2.0, 4.0, 3.0]


class TestSplitEinsumEncoder:
    # RFDETRNano at 384 px: windowed layers attend over 4 windows of 145 tokens, global layers over 580 tokens. A
    # 256-token chunk splits only the global layers (3 x 194); a 100-token chunk splits both (73 + 72, 6 x ~97).
    @pytest.mark.parametrize("query_chunk", [256, 100])
    def test_backbone_features_match_the_original_encoder(self, query_chunk: int) -> None:
        seed_all(0)
        dinov2 = RFDETRNano(pretrain_weights=None).model.model.backbone[0].encoder.eval()
        image = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            reference = dinov2(image)
            dinov2.encoder.encoder = split_einsum_encoder(dinov2.encoder.encoder, query_chunk=query_chunk)
            candidate = dinov2(image)

        assert len(candidate) == len(reference)
        for expected, actual in zip(reference, candidate):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
