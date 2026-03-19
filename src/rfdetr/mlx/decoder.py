# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""RF-DETR decoder for MLX inference.

Includes the C2f projector, two-stage encoder, and transformer decoder
with deformable cross-attention via a fused Metal bilinear sampling kernel.

Supports both detection and segmentation model variants through
parameterization of hidden_dim, num_heads, num_points, num_layers,
num_queries, and num_classes. The decoder can optionally return
intermediate layer outputs for use by a segmentation head via the
``return_intermediate`` parameter on ``RFDETRDecoder.__call__``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

# ============================================================
# Metal bilinear sampling kernel
# ============================================================

_BILINEAR_METAL_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint C = out_shape[4];
    uint nP = out_shape[3];
    uint nH = out_shape[2];
    uint nQ = out_shape[1];
    uint N = out_shape[0];
    uint H = fm_shape[1];
    uint W = fm_shape[2];
    uint total = N * nQ * nH * nP * C;
    if (elem >= total) return;

    uint c = elem % C;
    uint rem = elem / C;
    uint p = rem % nP;
    rem = rem / nP;
    uint h = rem % nH;
    rem = rem / nH;
    uint q = rem % nQ;
    uint n = rem / nQ;

    uint loc_idx = ((n * nQ + q) * nH + h) * nP + p;
    float sx = locs[loc_idx * 2 + 0] * (float)W - 0.5f;
    float sy = locs[loc_idx * 2 + 1] * (float)H - 0.5f;

    int x0 = (int)metal::floor(sx);
    int y0 = (int)metal::floor(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    float wa = (1.0f - fx) * (1.0f - fy);
    float wb = (1.0f - fx) * fy;
    float wc = fx * (1.0f - fy);
    float wd = fx * fy;

    int iW = (int)W;
    int iH = (int)H;
    if (x0 < 0 || x0 >= iW || y0 < 0 || y0 >= iH) wa = 0.0f;
    if (x0 < 0 || x0 >= iW || y1 < 0 || y1 >= iH) wb = 0.0f;
    if (x1 < 0 || x1 >= iW || y0 < 0 || y0 >= iH) wc = 0.0f;
    if (x1 < 0 || x1 >= iW || y1 < 0 || y1 >= iH) wd = 0.0f;

    int x0c = metal::clamp(x0, 0, iW - 1);
    int x1c = metal::clamp(x1, 0, iW - 1);
    int y0c = metal::clamp(y0, 0, iH - 1);
    int y1c = metal::clamp(y1, 0, iH - 1);

    uint fm_stride_n = H * W * C;
    uint base = n * fm_stride_n;
    float va = (float)feat[base + (uint)y0c * W * C + (uint)x0c * C + c];
    float vb = (float)feat[base + (uint)y1c * W * C + (uint)x0c * C + c];
    float vc = (float)feat[base + (uint)y0c * W * C + (uint)x1c * C + c];
    float vd = (float)feat[base + (uint)y1c * W * C + (uint)x1c * C + c];

    out[elem] = (T)(wa * va + wb * vb + wc * vc + wd * vd);
"""

_bilinear_kernel = mx.fast.metal_kernel(
    name="bilinear_sample",
    input_names=["feat", "locs", "fm_shape", "out_shape"],
    output_names=["out"],
    source=_BILINEAR_METAL_SOURCE,
)


def bilinear_sample(feature_map: mx.array, sampling_locs: mx.array) -> mx.array:
    """Bilinear sampling via fused Metal kernel.

    Matches PyTorch grid_sample with align_corners=False and zero padding.

    Args:
        feature_map: (N, H, W, C) spatial feature map.
        sampling_locs: (N, nQ, nH, nP, 2) sampling locations in [0, 1].

    Returns:
        (N, nQ, nH, nP, C) sampled features.
    """
    N, H, W, C = feature_map.shape
    _, nQ, nH, nP, _ = sampling_locs.shape
    out_shape = (N, nQ, nH, nP, C)
    total_elems = N * nQ * nH * nP * C

    fm_shape_arr = mx.array([N, H, W, C], dtype=mx.uint32)
    out_shape_arr = mx.array([N, nQ, nH, nP, C], dtype=mx.uint32)
    locs_flat = sampling_locs.astype(mx.float32).reshape(-1)

    outputs = _bilinear_kernel(
        inputs=[feature_map, locs_flat, fm_shape_arr, out_shape_arr],
        template=[("T", feature_map.dtype)],
        grid=(total_elems, 1, 1),
        threadgroup=(min(256, total_elems), 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[feature_map.dtype],
    )
    return outputs[0]


# ============================================================
# Projector (C2f neck)
# ============================================================


class ConvBNSiLU(nn.Module):
    """Conv2d + LayerNorm + SiLU."""

    def __init__(self, c_in: int, c_out: int, k: int = 1, p: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, padding=p, bias=False)
        self.bn = nn.LayerNorm(c_out)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        return nn.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two 3x3 convolutions without residual connection."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.cv1 = ConvBNSiLU(c, c, k=3, p=1)
        self.cv2 = ConvBNSiLU(c, c, k=3, p=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        return self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, c_in: int, c_out: int, n: int = 3) -> None:
        super().__init__()
        self.c = c_out // 2
        self.cv1 = ConvBNSiLU(c_in, c_out, k=1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c_out, k=1)
        self.m = [Bottleneck(self.c) for _ in range(n)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        y = self.cv1(x)
        ch0 = y[..., : self.c]
        ch1 = y[..., self.c :]
        parts = [ch0, ch1]
        cur = ch1
        for m in self.m:
            cur = m(cur)
            parts.append(cur)
        return self.cv2(mx.concatenate(parts, axis=-1))


class Projector(nn.Module):
    """Single-scale projector: concat backbone features -> C2f -> LayerNorm."""

    def __init__(self, in_channels: int = 1536, out_channels: int = 256, n_blocks: int = 3) -> None:
        super().__init__()
        self.stages = [[C2f(in_channels, out_channels, n_blocks), nn.LayerNorm(out_channels)]]

    def __call__(self, features: List[mx.array]) -> mx.array:
        """Forward pass.

        Args:
            features: List of backbone feature maps, each (N, H, W, embed_dim).

        Returns:
            Projected features (N, H, W, out_channels).
        """
        x = mx.concatenate(features, axis=-1)
        c2f, ln = self.stages[0]
        return ln(c2f(x))


# ============================================================
# Sinusoidal position embeddings
# ============================================================


def gen_sineembed_for_position(pos_tensor: mx.array, dim: int = 128) -> mx.array:
    """Sinusoidal encoding for reference points.

    Matches PyTorch output order: [y, x, w, h] (y before x).

    Args:
        pos_tensor: (..., 2) or (..., 4) reference points.
        dim: Encoding dimension per coordinate.

    Returns:
        (..., n_coords * dim) sinusoidal encoding.
    """
    scale = 2 * math.pi
    dim_t = mx.arange(dim, dtype=mx.float32)
    dim_t = 10000.0 ** (2.0 * (dim_t // 2) / dim)

    def _encode_coord(coord: mx.array) -> mx.array:
        embed = coord[..., None] * scale
        pos = embed / dim_t
        pos_sin = mx.sin(pos[..., 0::2])
        pos_cos = mx.cos(pos[..., 1::2])
        interleaved = mx.stack([pos_sin, pos_cos], axis=-1)
        shape = list(interleaved.shape[:-2]) + [-1]
        return interleaved.reshape(*shape)

    n_coords = pos_tensor.shape[-1]
    if n_coords == 2:
        pos_x = _encode_coord(pos_tensor[..., 0])
        pos_y = _encode_coord(pos_tensor[..., 1])
        return mx.concatenate([pos_y, pos_x], axis=-1)
    elif n_coords == 4:
        pos_x = _encode_coord(pos_tensor[..., 0])
        pos_y = _encode_coord(pos_tensor[..., 1])
        pos_w = _encode_coord(pos_tensor[..., 2])
        pos_h = _encode_coord(pos_tensor[..., 3])
        return mx.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)
    else:
        parts = [_encode_coord(pos_tensor[..., i]) for i in range(n_coords)]
        return mx.concatenate(parts, axis=-1)


# ============================================================
# Deformable Attention
# ============================================================


class DeformableAttention(nn.Module):
    """Single-level deformable cross-attention with Metal bilinear kernel."""

    def __init__(self, d_model: int = 256, n_heads: int = 16, n_points: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * 1 * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * 1 * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def __call__(
        self,
        query: mx.array,
        reference_points: mx.array,
        value_map: mx.array,
        spatial_shape: Tuple[int, int],
    ) -> mx.array:
        """Forward pass.

        Args:
            query: (N, nQ, d_model) query features.
            reference_points: (N, nQ, 1, 4) reference points [cx, cy, w, h].
            value_map: (N, H*W, d_model) flattened feature map.
            spatial_shape: (H, W) spatial dimensions.

        Returns:
            (N, nQ, d_model) output features.
        """
        N, nQ, _ = query.shape
        H, W = spatial_shape
        nH = self.n_heads
        nP = self.n_points
        head_dim = self.head_dim

        v = self.value_proj(value_map).reshape(N, H, W, nH, head_dim)
        v = v.transpose(0, 3, 1, 2, 4).reshape(N * nH, H, W, head_dim)

        offsets = self.sampling_offsets(query).reshape(N, nQ, nH, nP, 2)

        if reference_points.ndim == 4:
            ref_xy = reference_points[:, :, 0, :2]
            ref_wh = reference_points[:, :, 0, 2:]
        else:
            ref_xy = reference_points[..., :2]
            ref_wh = reference_points[..., 2:]

        scaled_offsets = offsets / nP * ref_wh[:, :, None, None, :] * 0.5
        locs = ref_xy[:, :, None, None, :] + scaled_offsets

        locs = locs.transpose(0, 2, 1, 3, 4).reshape(N * nH, nQ, 1, nP, 2)

        sampled = bilinear_sample(v, locs)
        sampled = sampled.reshape(N * nH, nQ, nP, head_dim)

        attn = self.attention_weights(query).reshape(N, nQ, nH, nP)
        attn = mx.softmax(attn, axis=-1)
        attn = attn.transpose(0, 2, 1, 3).reshape(N * nH, nQ, nP)

        out = (sampled * attn[..., None]).sum(axis=2)
        out = out.reshape(N, nH, nQ, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(N, nQ, self.d_model)

        return self.output_proj(out)


# ============================================================
# Decoder Layer
# ============================================================


class DecoderLayer(nn.Module):
    """Single decoder layer: self-attention + deformable cross-attention + FFN."""

    def __init__(
        self,
        d_model: int = 256,
        sa_nhead: int = 8,
        ca_nhead: int = 16,
        ca_npoints: int = 2,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, sa_nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = DeformableAttention(d_model, ca_nhead, n_points=ca_npoints)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        query_pos: mx.array,
        reference_points: mx.array,
        spatial_shape: Tuple[int, int],
    ) -> mx.array:
        """Forward pass.

        Args:
            tgt: (N, nQ, d_model) target queries.
            memory: (N, H*W, d_model) encoder memory.
            query_pos: (N, nQ, d_model) query position embedding.
            reference_points: (N, nQ, 1, 4) reference points.
            spatial_shape: (H, W) spatial dimensions.

        Returns:
            (N, nQ, d_model) updated target queries.
        """
        q = k = tgt + query_pos
        v = tgt
        tgt2 = self.self_attn(q, k, v)
        tgt = self.norm1(tgt + tgt2)

        tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shape)
        tgt = self.norm2(tgt + tgt2)

        tgt2 = self.linear2(nn.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt


class _MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class _TransformerDecoder(nn.Module):
    """Transformer decoder with reference point head and output norm."""

    def __init__(
        self,
        d_model: int,
        sa_nhead: int,
        ca_nhead: int,
        ca_npoints: int,
        num_layers: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.layers = [
            DecoderLayer(d_model, sa_nhead, ca_nhead, ca_npoints, dim_feedforward) for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.ref_point_head = _MLP(2 * d_model, d_model, d_model, 2)

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        query_pos: mx.array,
        reference_points: mx.array,
        spatial_shape: Tuple[int, int],
        return_intermediate: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]:
        """Forward pass through all decoder layers.

        Args:
            tgt: (N, nQ, d_model) target queries.
            memory: (N, H*W, d_model) encoder memory.
            query_pos: (N, nQ, d_model) query position embedding.
            reference_points: (N, nQ, 1, 4) reference points.
            spatial_shape: (H, W) spatial dimensions.
            return_intermediate: If True, also return per-layer hidden states.

        Returns:
            If ``return_intermediate`` is False: (N, nQ, d_model) decoded features.
            If ``return_intermediate`` is True: tuple of (normed_output, intermediates)
            where ``intermediates`` is a list of (N, nQ, d_model) arrays, one per layer.
        """
        output = tgt
        intermediates: List[mx.array] = []
        last_normed: Optional[mx.array] = None
        for layer in self.layers:
            output = layer(output, memory, query_pos, reference_points, spatial_shape)
            if return_intermediate:
                last_normed = self.norm(output)
                intermediates.append(last_normed)
        if return_intermediate:
            if last_normed is None:
                last_normed = self.norm(output)
            return last_normed, intermediates
        normed = self.norm(output)
        return normed


# ============================================================
# Full Decoder
# ============================================================


class RFDETRDecoder(nn.Module):
    """Complete RF-DETR decoder: projector + two-stage encoder + transformer decoder.

    Args:
        d_model: Hidden dimension (256 or 384).
        sa_nhead: Self-attention heads (8 or 12).
        ca_nhead: Cross-attention heads (16 or 24).
        ca_npoints: Deformable attention points (2 or 4).
        num_layers: Number of decoder layers (2-6).
        num_queries: Number of object queries (300).
        num_classes: Number of output classes including background (91 for COCO).
        embed_dim: Backbone embedding dimension (384 or 768).
        num_features: Number of backbone feature levels (4).
        dim_feedforward: FFN hidden dimension (2048).
        group_detr: Number of groups for group DETR training (13).
    """

    def __init__(
        self,
        d_model: int = 256,
        sa_nhead: int = 8,
        ca_nhead: int = 16,
        ca_npoints: int = 2,
        num_layers: int = 2,
        num_queries: int = 300,
        num_classes: int = 91,
        embed_dim: int = 384,
        num_features: int = 4,
        dim_feedforward: int = 2048,
        group_detr: int = 13,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Projector
        self.projector = Projector(
            in_channels=embed_dim * num_features,
            out_channels=d_model,
        )

        # Two-stage encoder
        self.enc_output = [nn.Linear(d_model, d_model)]
        self.enc_output_norm = [nn.LayerNorm(d_model)]
        self.enc_out_class_embed = [nn.Linear(d_model, num_classes)]
        self.enc_out_bbox_embed = [[nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, 4)]]

        # Learned queries (full group_detr size, sliced at inference)
        self.query_feat = mx.zeros((num_queries * group_detr, d_model))
        self.refpoint_embed = mx.zeros((num_queries * group_detr, 4))

        # Decoder
        self.decoder = _TransformerDecoder(d_model, sa_nhead, ca_nhead, ca_npoints, num_layers, dim_feedforward)

        # Final prediction heads
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed_layers = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, 4),
        ]

    def _bbox_embed(self, x: mx.array) -> mx.array:
        """Apply bbox prediction MLP."""
        for i, layer in enumerate(self.bbox_embed_layers):
            x = layer(x)
            if i < len(self.bbox_embed_layers) - 1:
                x = nn.relu(x)
        return x

    def __call__(
        self,
        features: List[mx.array],
        return_intermediate: bool = False,
    ) -> Dict[str, mx.array]:
        """Forward pass.

        Args:
            features: List of backbone feature maps, each (N, H, W, embed_dim).
            return_intermediate: If True, add ``"spatial_features"`` and
                ``"hs_list"`` to the returned dict for segmentation head use.

        Returns:
            Dict with ``"pred_logits"`` (N, num_queries, num_classes) and
            ``"pred_boxes"`` (N, num_queries, 4) in cxcywh format.
            When ``return_intermediate`` is True, also includes
            ``"spatial_features"`` (N, H, W, d_model) and
            ``"hs_list"`` (list of per-layer (N, nQ, d_model) arrays).
        """
        projected = self.projector(features)
        spatial_features = projected  # (N, H, W, d_model) — before flatten
        N, H, W, _ = projected.shape

        memory = projected.reshape(N, H * W, self.d_model)

        refpoint_embed_ts, memory_ts = self._two_stage_encode(memory, H, W, N)

        query_feat = mx.broadcast_to(
            self.query_feat[: self.num_queries][None],
            (N, self.num_queries, self.d_model),
        )
        refpoint_learned = mx.broadcast_to(
            self.refpoint_embed[: self.num_queries][None],
            (N, self.num_queries, 4),
        )

        cxcy = refpoint_learned[..., :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2]
        wh = mx.exp(refpoint_learned[..., 2:]) * refpoint_embed_ts[..., 2:]
        refpoints = mx.concatenate([cxcy, wh], axis=-1)

        query_sine_embed = gen_sineembed_for_position(refpoints, dim=self.d_model // 2)
        query_pos = self.decoder.ref_point_head(query_sine_embed)

        refpoints_input = refpoints[:, :, None, :]

        if return_intermediate:
            hs, hs_list = self.decoder(query_feat, memory, query_pos, refpoints_input, (H, W), return_intermediate=True)
        else:
            hs = self.decoder(query_feat, memory, query_pos, refpoints_input, (H, W))

        delta = self._bbox_embed(hs)
        pred_cxcy = delta[..., :2] * refpoints[..., 2:] + refpoints[..., :2]
        pred_wh = mx.exp(delta[..., 2:]) * refpoints[..., 2:]
        pred_boxes = mx.concatenate([pred_cxcy, pred_wh], axis=-1)

        pred_logits = self.class_embed(hs)

        result: Dict[str, mx.array] = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        if return_intermediate:
            result["spatial_features"] = spatial_features
            result["hs_list"] = hs_list
        return result

    def _two_stage_encode(self, memory: mx.array, H: int, W: int, N: int) -> Tuple[mx.array, mx.array]:
        """Generate proposals, score them, select top-K.

        Args:
            memory: (N, H*W, d_model) encoder memory.
            H: Grid height.
            W: Grid width.
            N: Batch size.

        Returns:
            Tuple of (refpoint_embed_ts, memory_ts), each (N, num_queries, ...).
        """
        gy = mx.arange(H, dtype=mx.float32)
        gx = mx.arange(W, dtype=mx.float32)
        grid_y, grid_x = mx.meshgrid(gy, gx, indexing="ij")
        grid = mx.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
        grid = (grid + 0.5) / mx.array([W, H], dtype=mx.float32)
        wh = mx.ones_like(grid) * 0.05
        proposals = mx.concatenate([grid, wh], axis=-1)[None]

        valid = ((proposals > 0.01) & (proposals < 0.99)).min(axis=-1, keepdims=True)
        memory_masked = memory * valid

        output_memory = self.enc_output_norm[0](self.enc_output[0](memory_masked))

        class_scores = self.enc_out_class_embed[0](output_memory)

        bbox_layers = self.enc_out_bbox_embed[0]
        bbox_delta = bbox_layers[0](output_memory)
        bbox_delta = nn.relu(bbox_delta)
        bbox_delta = bbox_layers[1](bbox_delta)
        bbox_delta = nn.relu(bbox_delta)
        bbox_delta = bbox_layers[2](bbox_delta)

        cx = bbox_delta[..., :1] * proposals[..., 2:3] + proposals[..., :1]
        cy = bbox_delta[..., 1:2] * proposals[..., 3:4] + proposals[..., 1:2]
        w = mx.exp(bbox_delta[..., 2:3]) * proposals[..., 2:3]
        h = mx.exp(bbox_delta[..., 3:4]) * proposals[..., 3:4]
        enc_coords = mx.concatenate([cx, cy, w, h], axis=-1)

        max_scores = mx.max(class_scores, axis=-1)
        topk = min(self.num_queries, max_scores.shape[-1])
        topk_idx = mx.argpartition(-max_scores, kth=topk - 1, axis=-1)[:, :topk]
        topk_scores = mx.take_along_axis(max_scores, topk_idx, axis=1)
        sort_order = mx.argsort(-topk_scores, axis=-1)
        topk_idx = mx.take_along_axis(topk_idx, sort_order, axis=1)

        idx4 = mx.broadcast_to(topk_idx[:, :, None], (N, topk, 4))
        refpoint_embed_ts = mx.take_along_axis(enc_coords, idx4, axis=1)

        idx_d = mx.broadcast_to(topk_idx[:, :, None], (N, topk, self.d_model))
        memory_ts = mx.take_along_axis(output_memory, idx_d, axis=1)

        return refpoint_embed_ts, memory_ts
