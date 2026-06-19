# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for GroupPose-oriented transformer streams."""

from types import SimpleNamespace

import torch
from torch import nn

from rfdetr.models.transformer import Transformer, TransformerDecoder, TransformerDecoderLayer, build_transformer


def _build_transformer_inputs(
    batch_size: int = 2,
    hidden_dim: int = 16,
    num_levels: int = 2,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build minimal synthetic multi-scale inputs for `Transformer.forward`.

    Args:
        batch_size: Mini-batch size.
        hidden_dim: Transformer and input channel size.
        num_levels: Number of feature pyramid levels.

    Returns:
        srcs, masks, pos_embeds, refpoint_embed, query_feat.
    """
    spatial_shapes = [(4, 4), (2, 2)]
    srcs = [
        torch.randn(batch_size, hidden_dim, spatial_shapes[idx][0], spatial_shapes[idx][1]) for idx in range(num_levels)
    ]
    masks = [torch.zeros(batch_size, h, w, dtype=torch.bool) for h, w in spatial_shapes[:num_levels]]
    pos_embeds = [
        torch.randn(batch_size, hidden_dim, spatial_shapes[idx][0], spatial_shapes[idx][1]) for idx in range(num_levels)
    ]
    refpoint_embed = torch.rand(6, 4)
    query_feat = torch.randn(6, hidden_dim)
    return srcs, masks, pos_embeds, refpoint_embed, query_feat


def test_transformer_keypoint_disabled_matches_default_contract() -> None:
    """Transformer without GroupPose should keep the 4-item return contract."""
    srcs, masks, pos_embeds, refpoint_embed, query_feat = _build_transformer_inputs()
    transformer = Transformer(
        d_model=16,
        num_queries=6,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        use_grouppose_keypoints=False,
    )

    outputs = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)
    assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"
    hs, references, memory_ts, boxes_ts = outputs
    assert hs is not None and references is not None
    assert memory_ts is None and boxes_ts is None


def test_transformer_keypoint_enabled_shapes() -> None:
    """GroupPose path should emit keypoint hidden states and encoder keypoint slots."""
    srcs, masks, pos_embeds, refpoint_embed, query_feat = _build_transformer_inputs()
    transformer = Transformer(
        d_model=16,
        num_queries=6,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(16, 2)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(16, 4)])

    outputs = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)
    assert len(outputs) == 7, f"Expected 7 outputs, got {len(outputs)}"
    hs, references, memory_ts, boxes_ts, keypoint_hs, enc_kp_predictions, keypoint_memory_ts = outputs

    assert isinstance(hs, torch.Tensor)
    assert hs.shape[-1] == 16
    assert references.shape[-1] == 4
    assert memory_ts is not None and boxes_ts is not None
    assert keypoint_hs is not None
    assert keypoint_hs.shape[3] == 17
    assert enc_kp_predictions is not None
    assert enc_kp_predictions.shape[-1] == 16
    assert keypoint_memory_ts is not None


def test_build_transformer_defaults_inter_instance_keypoint_attention_to_config_default() -> None:
    """Older args objects without `inter_instance_kp_attn` should keep the preview topology default."""
    args = SimpleNamespace(
        hidden_dim=16,
        sa_nheads=4,
        ca_nheads=4,
        num_queries=6,
        dropout=0.0,
        dim_feedforward=32,
        dec_layers=1,
        group_detr=1,
        num_feature_levels=2,
        dec_n_points=1,
        lite_refpoint_refine=True,
        decoder_norm="LN",
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )

    transformer = build_transformer(args)

    decoder_layer = transformer.decoder.layers[0]
    assert isinstance(decoder_layer, TransformerDecoderLayer)
    assert decoder_layer.enable_keypoint_processing
    assert not decoder_layer.inter_instance_kp_attn


def test_keypoint_class_mask_person_only() -> None:
    """Person-only schema `[17]` should build a keypoint class mask with only self-class tokens."""
    layer = TransformerDecoderLayer(
        d_model=16,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=32,
        num_feature_levels=2,
        enable_keypoint_processing=True,
        grouppose_keypoint_dim_downscale=1,
        keypoint_cross_attn=False,
        inter_instance_kp_attn=False,
    )
    decoder = TransformerDecoder(
        decoder_layer=layer,
        num_layers=1,
        return_intermediate=True,
        d_model=16,
        lite_refpoint_refine=True,
        enable_keypoint_processing=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )

    assert decoder.keypoint_pos_embed is not None
    assert decoder.keypoint_class_mask.shape == (18, 18)
    assert decoder.keypoint_class_mask.dtype == torch.bool
    assert not decoder.keypoint_class_mask.any()


def test_enc_keypoint_embed_eval_uses_only_head_zero() -> None:
    """Encoder keypoint path must use a single head (head 0) in eval mode.

    Regression: ``group_detr = len(self.enc_out_keypoint_embed)`` without a
    ``self.training`` guard caused eval mode to split ``num_queries`` across all
    group heads instead of routing every query through head 0. Fix:
    ``group_detr = len(...) if self.training else 1``.

    Strategy: zero all head weights/biases; set head-0 last-layer bias to
    ``sentinel``. In eval mode every query routes through head 0, so
    ``kp_pred[..., 2:]`` (the pure-delta dims unaffected by ref_xy/wh) must all
    equal ``sentinel``. In training mode only the first 1/group_detr queries go
    through head 0 (the rest equal 0.0).
    """
    group_detr = 3
    num_queries = 6  # divisible by group_detr
    hidden_dim = 16
    batch_size = 1
    sentinel = 50.0

    srcs, masks, pos_embeds, _, _ = _build_transformer_inputs(batch_size=batch_size, hidden_dim=hidden_dim)
    refpoint_embed = torch.rand(num_queries, 4)
    query_feat = torch.randn(num_queries, hidden_dim)

    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        group_detr=group_detr,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[2],
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(group_detr)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(group_detr)])

    # Zero all keypoint head weights and biases; give head 0 a distinctive output.
    with torch.no_grad():
        for _, head in enumerate(transformer.enc_out_keypoint_embed):
            for layer in head.layers:
                layer.weight.zero_()
                layer.bias.zero_()
        transformer.enc_out_keypoint_embed[0].layers[-1].bias.fill_(sentinel)

    transformer.eval()
    with torch.no_grad():
        outputs = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    _, _, _, _, _, enc_kp_predictions, _ = outputs
    assert enc_kp_predictions is not None, "enc_kp_predictions should not be None in keypoint mode"

    # kp_pred = [kp_xy(2 dims), kp_delta[2:]]; dims 2: are pure MLP output unaffected by ref_xy/wh.
    kp_beyond_xy = enc_kp_predictions[..., 2:]
    assert (kp_beyond_xy == sentinel).all(), (
        f"Eval mode must route all {num_queries} queries through head 0 (bias={sentinel}). "
        f"Got min={kp_beyond_xy.min().item():.2f}, max={kp_beyond_xy.max().item():.2f}. "
        "Bug: group_detr not guarded by self.training in enc_out_keypoint_embed loop."
    )


def test_cross_attn_srcs_none_backward_compat() -> None:
    """`cross_attn_srcs=None` must remain equivalent to passing the primary feature stream."""
    srcs, masks, pos_embeds, refpoint_embed, query_feat = _build_transformer_inputs()

    transformer = Transformer(
        d_model=16,
        num_queries=6,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        use_grouppose_keypoints=False,
    )
    outputs_default = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)
    outputs_explicit = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=srcs)

    assert len(outputs_default) == len(outputs_explicit) == 4
    for default_part, explicit_part in zip(outputs_default, outputs_explicit):
        if default_part is None:
            assert explicit_part is None
        else:
            torch.testing.assert_close(default_part, explicit_part)
