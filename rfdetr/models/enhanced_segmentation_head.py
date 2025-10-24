"""
Enhanced Segmentation Head with Color Attention and Boundary Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
from rfdetr.models.segmentation_head import DepthwiseConvBlock, MLPBlock
from rfdetr.models.enhancements import (
    ColorAttentionModule,
    BoundaryRefinementNetwork
)


class EnhancedSegmentationHead(nn.Module):
    """
    Enhanced segmentation head with:
    1. Color attention for better feature extraction
    2. Boundary refinement for precise mask prediction
    """
    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        bottleneck_ratio: int = 1,
        downsample_ratio: int = 4,
        use_color_attention: bool = True,
        use_boundary_refinement: bool = True
    ):
        super().__init__()

        self.downsample_ratio = downsample_ratio
        self.use_color_attention = use_color_attention
        self.use_boundary_refinement = use_boundary_refinement

        self.interaction_dim = in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim

        # Color attention module (applied to spatial features)
        if use_color_attention:
            self.color_attention = ColorAttentionModule(
                in_channels=in_dim,
                reduction=8,
                num_color_channels=64
            )

        # Standard blocks
        self.blocks = nn.ModuleList([
            DepthwiseConvBlock(in_dim) for _ in range(num_blocks)
        ])

        self.spatial_features_proj = (
            nn.Identity() if bottleneck_ratio is None
            else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )

        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = (
            nn.Identity() if bottleneck_ratio is None
            else nn.Linear(in_dim, self.interaction_dim)
        )

        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Boundary refinement network
        if use_boundary_refinement:
            self.boundary_refinement = BoundaryRefinementNetwork(
                in_channels=in_dim,
                hidden_channels=128,
                num_refinement_layers=3,
                use_edge_detection=True
            )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features: list[torch.Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False
    ) -> list[torch.Tensor]:
        """
        Args:
            spatial_features: (B, C, H, W)
            query_features: [(B, N, C)] for each decoder layer
            image_size: (H, W) original image size
            skip_blocks: Whether to skip the conv blocks
        Returns:
            mask_logits: List of mask predictions [(B, N, H*r, W*r)]
        """
        target_size = (
            image_size[0] // self.downsample_ratio,
            image_size[1] // self.downsample_ratio
        )
        spatial_features = F.interpolate(
            spatial_features,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Apply color attention to spatial features
        if self.use_color_attention:
            spatial_features = self.color_attention(spatial_features)

        # Store original spatial features for boundary refinement
        spatial_features_for_refinement = spatial_features

        mask_logits = []
        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf = self.query_features_proj(self.query_features_block(qf))
                coarse_masks = torch.einsum(
                    'bchw,bnc->bnhw',
                    spatial_features_proj,
                    qf
                ) + self.bias

                # Apply boundary refinement if enabled
                if self.use_boundary_refinement:
                    refined_masks = self.boundary_refinement(
                        spatial_features_for_refinement,
                        coarse_masks
                    )
                    mask_logits.append(refined_masks)
                else:
                    mask_logits.append(coarse_masks)
        else:
            assert len(query_features) == 1, "skip_blocks is only supported for length 1 query features"
            qf = self.query_features_proj(self.query_features_block(query_features[0]))
            coarse_masks = torch.einsum(
                'bchw,bnc->bnhw',
                spatial_features,
                qf
            ) + self.bias

            # Apply boundary refinement if enabled
            if self.use_boundary_refinement:
                refined_masks = self.boundary_refinement(
                    spatial_features_for_refinement,
                    coarse_masks
                )
                mask_logits.append(refined_masks)
            else:
                mask_logits.append(coarse_masks)

        return mask_logits

    def forward_export(
        self,
        spatial_features: torch.Tensor,
        query_features: list[torch.Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False
    ) -> list[torch.Tensor]:
        """Export-friendly forward pass."""
        assert len(query_features) == 1, "at export time, segmentation head expects exactly one query feature"

        target_size = (
            image_size[0] // self.downsample_ratio,
            image_size[1] // self.downsample_ratio
        )
        spatial_features = F.interpolate(
            spatial_features,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Apply color attention if enabled
        if self.use_color_attention:
            spatial_features = self.color_attention(spatial_features)

        spatial_features_for_refinement = spatial_features

        if not skip_blocks:
            for block in self.blocks:
                spatial_features = block(spatial_features)

        spatial_features_proj = self.spatial_features_proj(spatial_features)

        qf = self.query_features_proj(self.query_features_block(query_features[0]))
        coarse_masks = torch.einsum(
            'bchw,bnc->bnhw',
            spatial_features_proj,
            qf
        ) + self.bias

        # Apply boundary refinement if enabled
        if self.use_boundary_refinement:
            refined_masks = self.boundary_refinement(
                spatial_features_for_refinement,
                coarse_masks
            )
            return [refined_masks]
        else:
            return [coarse_masks]
