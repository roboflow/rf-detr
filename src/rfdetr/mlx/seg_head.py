# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MLX segmentation head for RF-DETR.

Architecture mirrors ``rfdetr/models/segmentation_head.py``:

  DepthwiseConvBlock × num_blocks  (spatial feature refinement)
  spatial_features_proj  (1×1 Conv2d)
  query_features_block   (MLP with LayerNorm)
  query_features_proj    (Linear)
  masks = einsum(sp, q) + bias
"""

from __future__ import annotations

from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _bilinear_upsample(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear upsample matching PyTorch ``F.interpolate(align_corners=False)``.

    Args:
        x: (N, H, W, C) input features.
        target_h: Target height.
        target_w: Target width.

    Returns:
        (N, target_h, target_w, C) upsampled features.
    """
    N, H, W, C = x.shape

    # Source coordinates for each output pixel (align_corners=False)
    y_src = (mx.arange(target_h, dtype=mx.float32) + 0.5) * (H / target_h) - 0.5
    x_src = (mx.arange(target_w, dtype=mx.float32) + 0.5) * (W / target_w) - 0.5

    y0 = mx.clip(mx.floor(y_src).astype(mx.int32), 0, H - 1)
    y1 = mx.clip(y0 + 1, 0, H - 1)
    x0 = mx.clip(mx.floor(x_src).astype(mx.int32), 0, W - 1)
    x1 = mx.clip(x0 + 1, 0, W - 1)

    fy = (y_src - mx.floor(y_src))[:, None, None]  # (target_h, 1, 1)
    fx = (x_src - mx.floor(x_src))[None, :, None]  # (1, target_w, 1)

    # Gather 4 corner values via take along spatial axes
    v00 = mx.take(mx.take(x, y0, axis=1), x0, axis=2)
    v01 = mx.take(mx.take(x, y0, axis=1), x1, axis=2)
    v10 = mx.take(mx.take(x, y1, axis=1), x0, axis=2)
    v11 = mx.take(mx.take(x, y1, axis=1), x1, axis=2)

    return v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx + v10 * fy * (1 - fx) + v11 * fy * fx


class DepthwiseConvBlock(nn.Module):
    """Depthwise-separable conv block with residual.

    Computes ``x + gelu(pwconv(norm(dwconv(x))))``.

    Args:
        dim: Channel dimension (default 256).
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (N, H, W, C) spatial features.

        Returns:
            (N, H, W, C) refined features with residual connection.
        """
        return x + nn.gelu(self.pwconv(self.norm(self.dwconv(x))))


class MLPBlock(nn.Module):
    """MLP with LayerNorm and residual connection.

    Computes ``x + fc2(gelu(fc1(norm_in(x))))``.

    Args:
        dim: Channel dimension (default 256).
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (..., C) query features.

        Returns:
            (..., C) refined features with residual connection.
        """
        return x + self.fc2(nn.gelu(self.fc1(self.norm_in(x))))


class SegHead(nn.Module):
    """Segmentation head: produces per-query mask logits from spatial + query features.

    Takes projector output (spatial features) and intermediate decoder hidden
    states, upsamples and refines the spatial map, then generates masks via
    a dot-product between spatial features and query embeddings.

    Args:
        in_dim: Feature channel dimension (default 256).
        num_blocks: Number of ``DepthwiseConvBlock`` refinement stages (default 4).
    """

    def __init__(self, in_dim: int = 256, num_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = [DepthwiseConvBlock(in_dim) for _ in range(num_blocks)]
        self.spatial_features_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = nn.Linear(in_dim, in_dim)
        self.bias = mx.zeros((1,))

    def __call__(
        self,
        spatial_features: mx.array,
        hs_list: List[mx.array],
        img_size: int = 312,
        downsample_ratio: int = 4,
    ) -> mx.array:
        """Compute per-query mask logits.

        Args:
            spatial_features: (N, H_feat, W_feat, C) projector output.
            hs_list: List of (N, nQ, C) intermediate decoder hidden states.
                The last element is used for query features.
            img_size: Original input image size (square, e.g. 312).
            downsample_ratio: Mask resolution = img_size // downsample_ratio.

        Returns:
            (N, nQ, H_mask, W_mask) mask logits.
        """
        H_target = img_size // downsample_ratio  # e.g. 312 // 4 = 78
        W_target = H_target  # square output

        # Bilinear upsample: (N, H, W, C) → (N, H_target, W_target, C)
        sp = _bilinear_upsample(spatial_features, H_target, W_target)

        # Refine with depthwise blocks
        for block in self.blocks:
            sp = block(sp)

        # 1×1 conv projection
        sp = self.spatial_features_proj(sp)  # (N, H_target, W_target, C)

        # Process query features from the last decoder layer
        q = self.query_features_block(hs_list[-1])  # (N, nQ, C)
        q = self.query_features_proj(q)  # (N, nQ, C)

        # Dot-product between spatial and query → mask logits
        masks = mx.einsum("nhwc,nqc->nqhw", sp, q) + self.bias
        return masks  # (N, nQ, H_target, W_target)


def build_seg_head(seg_weights: Dict[str, np.ndarray], num_blocks: int = 4) -> SegHead:
    """Build a ``SegHead`` from a dict of converted MLX weights.

    Args:
        seg_weights: Dict mapping bare MLX key names (no ``segmentation_head.``
            prefix) to numpy arrays, as returned by ``convert_seg_weights()``.
        num_blocks: Number of ``DepthwiseConvBlock`` stages to build.

    Returns:
        Initialised ``SegHead`` with weights loaded.
    """
    seg_head = SegHead(num_blocks=num_blocks)
    seg_head.load_weights([(k, mx.array(v)) for k, v in seg_weights.items()], strict=False)
    return seg_head
