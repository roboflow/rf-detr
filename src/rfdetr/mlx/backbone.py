# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""DINOv2 backbone with windowed attention for MLX.

Supports all RF-DETR detection model variants:
- ViT-S (384d, 6 heads): Nano, Small, Medium, Base, Large
- ViT-B (768d, 12 heads): Large (deprecated)

Windowed attention pattern: every 3rd layer starting from layer 3 uses
full attention (layers 3, 6, 9 for 12-layer models). All other layers
use windowed attention with configurable window grid size.
"""

from __future__ import annotations

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings via strided convolution."""

    def __init__(
        self,
        img_size: int = 640,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, int, int]:
        """Forward pass.

        Args:
            x: Input tensor (N, H, W, C) in NHWC format.

        Returns:
            Tuple of (patches, grid_h, grid_w) where patches is (N, num_patches, embed_dim).
        """
        x = self.proj(x)
        N, H, W, C = x.shape
        return x.reshape(N, H * W, C), H, W


class LayerScale(nn.Module):
    """Learnable per-channel scale on residuals."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer scale."""
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections."""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (N, L, C).

        Returns:
            Output tensor (N, L, C).
        """
        N, L, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q(x).reshape(N, L, H, D).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(N, L, H, D).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(N, L, H, D).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(N, L, C)
        return self.out(out)


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer block with layer scale.

    Windowing is handled at the backbone level. Each block operates on
    whatever sequence it receives.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_windows: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)
        self.ls2 = LayerScale(dim)
        self.num_windows = num_windows

    def __call__(self, x: mx.array, run_full_attention: bool = False) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor. For windowed: (N*nW^2, tokens, C). For full: same shape.
            run_full_attention: If True, merge windows before attention.

        Returns:
            Output tensor, same shape as input.
        """
        nW2 = self.num_windows * self.num_windows

        if run_full_attention and nW2 > 1:
            B, HW, C = x.shape
            x_merged = x.reshape(B // nW2, nW2 * HW, C)
            attn_out = self.attn(self.norm1(x_merged))
            attn_out = attn_out.reshape(B, HW, C)
        else:
            attn_out = self.attn(self.norm1(x))

        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2Backbone(nn.Module):
    """DINOv2 backbone with windowed attention for RF-DETR.

    Supports ViT-S (384d) and ViT-B (768d) with configurable patch size,
    window count, and feature extraction indices.

    Args:
        img_size: Input image resolution (square).
        patch_size: Patch size for patch embedding (14 or 16).
        embed_dim: Embedding dimension (384 for ViT-S, 768 for ViT-B).
        depth: Number of transformer layers (12).
        num_heads: Number of attention heads (6 for ViT-S, 12 for ViT-B).
        num_windows: Window grid size (1, 2, or 4). Total windows = num_windows^2.
        feature_indices: 0-indexed layer indices at which to extract features.
        mlp_ratio: MLP hidden dimension ratio.
    """

    def __init__(
        self,
        img_size: int = 640,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_windows: int = 2,
        feature_indices: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_windows = num_windows
        self.depth = depth

        if feature_indices is None:
            feature_indices = [2, 5, 8, 11]
        self.feature_indices = feature_indices

        # Full attention at every 3rd layer starting from 3
        self.full_attn_layers = set(range(3, depth, 3))

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, 1 + num_patches, embed_dim))

        self.blocks = [Block(embed_dim, num_heads, num_windows, mlp_ratio) for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim)

    def _window_partition(self, patches: mx.array, H: int, W: int, N: int) -> mx.array:
        """Partition patches into windows.

        Args:
            patches: (N, H*W, C) patch tokens.
            H: Grid height.
            W: Grid width.
            N: Batch size.

        Returns:
            (N*nW^2, win_h*win_w, C) windowed patches.
        """
        nW = self.num_windows
        C = patches.shape[-1]
        win_h, win_w = H // nW, W // nW
        patches = patches.reshape(N, nW, win_h, nW, win_w, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(N * nW * nW, win_h * win_w, C)
        return patches

    def _unwindow(self, patches: mx.array, H: int, W: int, N: int) -> mx.array:
        """Reverse window partitioning.

        Args:
            patches: (N*nW^2, win_h*win_w, C) windowed patches.
            H: Grid height.
            W: Grid width.
            N: Batch size.

        Returns:
            (N, H, W, C) spatial feature map.
        """
        nW = self.num_windows
        nW2 = nW * nW
        C = patches.shape[-1]
        win_h, win_w = H // nW, W // nW
        patches = patches.reshape(N * nW2, win_h * win_w, C)
        patches = patches.reshape(N, nW2 * win_h * win_w, C)
        patches = patches.reshape(N * nW, nW, win_h, win_w, C)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(N, H, W, C)
        return patches

    def __call__(self, x: mx.array) -> List[mx.array]:
        """Forward pass extracting multi-scale features.

        Args:
            x: Input image tensor (N, H, W, C) in NHWC format, normalized.

        Returns:
            List of feature maps at configured layer indices,
            each (N, grid_H, grid_W, embed_dim).
        """
        N = x.shape[0]
        patches, H, W = self.patch_embed(x)

        cls = mx.broadcast_to(self.cls_token, (N, 1, self.embed_dim))
        x = mx.concatenate([cls, patches], axis=1) + self.pos_embed

        nW = self.num_windows
        nW2 = nW * nW

        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        win_patches = self._window_partition(patch_tokens, H, W, N)
        win_cls = mx.broadcast_to(cls_token, (N, 1, self.embed_dim))
        win_cls = mx.concatenate([win_cls] * nW2, axis=0)
        x = mx.concatenate([win_cls, win_patches], axis=1)

        features = []
        for i, block in enumerate(self.blocks):
            run_full = i in self.full_attn_layers
            x = block(x, run_full_attention=run_full)

            if i in self.feature_indices:
                normed = self.norm(x)
                patch_only = normed[:, 1:]
                feat = self._unwindow(patch_only, H, W, N)
                features.append(feat)

        return features


def interpolate_pos_embed(pos_embed: np.ndarray, target_patches: int) -> np.ndarray:
    """Interpolate positional embedding to a new resolution.

    Args:
        pos_embed: (1, 1+stored_patches, C) positional embedding with CLS token.
        target_patches: Target number of patch tokens.

    Returns:
        (1, 1+target_patches, C) interpolated positional embedding.
    """
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    stored_patches = patch_pos.shape[1]
    if stored_patches == target_patches:
        return pos_embed

    C = patch_pos.shape[2]
    src_side = int(round(stored_patches**0.5))
    tgt_side = int(round(target_patches**0.5))

    grid = patch_pos.reshape(src_side, src_side, C)

    from scipy.ndimage import zoom

    scale = tgt_side / src_side
    grid_resized = zoom(grid, (scale, scale, 1.0), order=3)

    patch_pos_new = grid_resized.reshape(1, tgt_side * tgt_side, C)
    return np.concatenate([cls_pos, patch_pos_new], axis=1)
