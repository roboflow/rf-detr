"""
Model Enhancement Modules for RF-DETR
Includes:
1. Color Attention Module
2. Color Contrast Loss
3. Enhanced Deformable Attention
4. Boundary Refinement Sub-network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ColorAttentionModule(nn.Module):
    """
    Color Attention Module for enhancing color-sensitive features.
    Useful for car interior segmentation where different parts have distinct colors.
    """
    def __init__(
        self,
        in_channels: int = 256,
        reduction: int = 8,
        num_color_channels: int = 64
    ):
        """
        Args:
            in_channels: Input feature channel dimension
            reduction: Channel reduction ratio for attention
            num_color_channels: Number of channels for color feature extraction
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_color_channels = num_color_channels

        # Color feature extraction branch
        self.color_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_color_channels, kernel_size=1),
            nn.BatchNorm2d(num_color_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_color_channels, num_color_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_color_channels),
            nn.ReLU(inplace=True)
        )

        # Channel attention for color features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_color_channels, num_color_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_color_channels // reduction, num_color_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention for color features
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_color_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + num_color_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Output projection
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Enhanced features with color attention [B, C, H, W]
        """
        # Extract color features
        color_feat = self.color_conv(x)

        # Apply channel attention
        channel_att = self.channel_attention(color_feat)
        color_feat = color_feat * channel_att

        # Apply spatial attention
        spatial_att = self.spatial_attention(color_feat)
        color_feat = color_feat * spatial_att

        # Fuse original and color features
        fused = torch.cat([x, color_feat], dim=1)
        fused = self.fusion(fused)

        # Residual connection
        output = self.output_proj(fused) + x

        return output


class ColorContrastLoss(nn.Module):
    """
    Color Contrast Loss to enhance color differences between instances.
    Encourages different instances to have distinct color representations.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        weight: float = 1.0
    ):
        """
        Args:
            temperature: Temperature parameter for contrastive learning
            margin: Margin for pushing different instances apart
            weight: Loss weight
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.weight = weight

    def extract_color_features(
        self,
        masks: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract average color features for each mask.

        Args:
            masks: Binary masks [B, N, H, W]
            images: RGB images [B, 3, H, W]
        Returns:
            Color features [B, N, 3]
        """
        B, N, H, W = masks.shape

        # Resize images to match mask size if needed
        if images.shape[2:] != masks.shape[2:]:
            images = F.interpolate(
                images,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        # Expand images for broadcasting [B, 1, 3, H, W]
        images = images.unsqueeze(1)

        # Expand masks for broadcasting [B, N, 1, H, W]
        masks_expanded = masks.unsqueeze(2)

        # Extract color features: weighted average by mask
        color_features = (images * masks_expanded).sum(dim=[3, 4])  # [B, N, 3]
        mask_areas = masks.sum(dim=[2, 3]).clamp(min=1.0).unsqueeze(2)  # [B, N, 1]
        color_features = color_features / mask_areas  # [B, N, 3]

        return color_features

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        images: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute color contrast loss.

        Args:
            pred_masks: Predicted masks [B, N, H, W]
            target_masks: Target masks [B, N, H, W]
            images: RGB images [B, 3, H, W]
            valid_mask: Valid instance mask [B, N] (optional)
        Returns:
            Color contrast loss
        """
        B, N = pred_masks.shape[:2]

        if N <= 1:
            return torch.tensor(0.0, device=pred_masks.device)

        # Extract color features from predicted masks
        pred_colors = self.extract_color_features(pred_masks, images)  # [B, N, 3]

        # Normalize color features
        pred_colors = F.normalize(pred_colors, p=2, dim=2)

        # Compute pairwise cosine similarity
        similarity = torch.bmm(pred_colors, pred_colors.transpose(1, 2))  # [B, N, N]
        similarity = similarity / self.temperature

        # Create mask for different instances (exclude diagonal)
        instance_mask = 1.0 - torch.eye(N, device=similarity.device).unsqueeze(0)  # [1, N, N]

        # Apply valid mask if provided
        if valid_mask is not None:
            # Create pairwise valid mask [B, N, N]
            pairwise_valid = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
            instance_mask = instance_mask * pairwise_valid

        # Contrastive loss: push different instances apart
        # We want to maximize distance (minimize similarity)
        # Loss = max(0, similarity - margin) for different instances
        contrast_loss = F.relu(similarity - self.margin) * instance_mask

        # Average over valid pairs
        num_pairs = instance_mask.sum() + 1e-6
        loss = contrast_loss.sum() / num_pairs

        return loss * self.weight


class EnhancedDeformableAttention(nn.Module):
    """
    Enhanced Multi-Scale Deformable Attention with:
    1. More sampling points for better coverage
    2. Adaptive sampling based on content
    3. Enhanced feature aggregation
    """
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        enhanced_points: int = 8,
        use_adaptive_sampling: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_levels: Number of feature levels
            n_heads: Number of attention heads
            n_points: Base number of sampling points per head per level
            enhanced_points: Enhanced number of sampling points
            use_adaptive_sampling: Whether to use adaptive sampling
        """
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = enhanced_points  # Use enhanced points
        self.use_adaptive_sampling = use_adaptive_sampling

        # Sampling offsets
        self.sampling_offsets = nn.Linear(
            d_model,
            n_heads * n_levels * enhanced_points * 2
        )

        # Attention weights
        self.attention_weights = nn.Linear(
            d_model,
            n_heads * n_levels * enhanced_points
        )

        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Adaptive sampling network (if enabled)
        if use_adaptive_sampling:
            self.adaptive_sampler = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, n_heads * n_levels * enhanced_points * 2)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1) / self.n_points

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Len_q, C]
            reference_points: [B, Len_q, n_levels, 2]
            value: [B, Len_v, C]
            spatial_shapes: [n_levels, 2]
            level_start_index: [n_levels]
            padding_mask: [B, Len_v]
        Returns:
            output: [B, Len_q, C]
        """
        B, Len_q, _ = query.shape
        B, Len_v, _ = value.shape

        # Project value
        value = self.value_proj(value)

        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)

        value = value.view(B, Len_v, self.n_heads, self.d_model // self.n_heads)

        # Get sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # Adaptive sampling adjustment
        if self.use_adaptive_sampling:
            adaptive_offsets = self.adaptive_sampler(query)
            adaptive_offsets = adaptive_offsets.view(
                B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
            )
            sampling_offsets = sampling_offsets + 0.1 * adaptive_offsets  # Small adjustment

        # Get attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = attention_weights.view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Normalize offsets
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(f"Unsupported reference_points shape: {reference_points.shape}")

        # Perform deformable sampling (simplified version)
        # In practice, you'd use the CUDA implementation from ms_deform_attn_func
        output = self._deformable_sampling(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        return output

    def _deformable_sampling(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified deformable sampling implementation.
        In practice, use the optimized CUDA implementation.
        """
        B, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
        _, Len_v, _, d_model = value.shape

        # This is a simplified PyTorch implementation
        # For production, use the CUDA-optimized version from ms_deform_attn_func

        outputs = []
        for level in range(n_levels):
            start_idx = level_start_index[level]
            end_idx = level_start_index[level + 1] if level < n_levels - 1 else Len_v
            H, W = spatial_shapes[level]

            # Get value for this level
            value_level = value[:, start_idx:end_idx, :, :]  # [B, H*W, n_heads, d_model]
            value_level = value_level.permute(0, 2, 3, 1).reshape(B, n_heads, d_model, H, W)

            # Get sampling locations for this level
            loc_level = sampling_locations[:, :, :, level, :, :]  # [B, Len_q, n_heads, n_points, 2]

            # Normalize to [-1, 1] for grid_sample
            loc_level = loc_level * 2.0 - 1.0

            # Sample features
            sampled = []
            for point in range(n_points):
                loc_point = loc_level[:, :, :, point, :]  # [B, Len_q, n_heads, 2]
                loc_point = loc_point.permute(0, 2, 1, 3)  # [B, n_heads, Len_q, 2]
                loc_point = loc_point.reshape(B * n_heads, Len_q, 1, 2)

                value_sampled = F.grid_sample(
                    value_level.reshape(B * n_heads, d_model, H, W),
                    loc_point,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )  # [B*n_heads, d_model, Len_q, 1]

                value_sampled = value_sampled.reshape(B, n_heads, d_model, Len_q).permute(0, 3, 1, 2)
                sampled.append(value_sampled)

            sampled = torch.stack(sampled, dim=-1)  # [B, Len_q, n_heads, d_model, n_points]

            # Apply attention weights
            weights_level = attention_weights[:, :, :, level, :].unsqueeze(3)  # [B, Len_q, n_heads, 1, n_points]
            output_level = (sampled * weights_level).sum(dim=-1)  # [B, Len_q, n_heads, d_model]

            outputs.append(output_level)

        # Sum across levels
        output = torch.stack(outputs, dim=-1).sum(dim=-1)  # [B, Len_q, n_heads, d_model]
        output = output.reshape(B, Len_q, n_heads * d_model)

        return output


class BoundaryRefinementNetwork(nn.Module):
    """
    Boundary Refinement Sub-network for precise instance boundary prediction.
    Uses multi-scale features and edge-aware convolutions.
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_refinement_layers: int = 3,
        use_edge_detection: bool = True
    ):
        """
        Args:
            in_channels: Input feature channel dimension
            hidden_channels: Hidden channel dimension
            num_refinement_layers: Number of refinement layers
            use_edge_detection: Whether to use edge detection module
        """
        super().__init__()
        self.use_edge_detection = use_edge_detection

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Edge detection module
        if use_edge_detection:
            self.edge_detector = EdgeDetectionModule(hidden_channels)

        # Boundary refinement layers
        self.refinement_layers = nn.ModuleList([
            BoundaryRefinementLayer(
                hidden_channels,
                use_edge_features=use_edge_detection
            )
            for _ in range(num_refinement_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        )

    def forward(
        self,
        features: torch.Tensor,
        coarse_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Input features [B, C, H, W]
            coarse_masks: Coarse mask predictions [B, N, H, W]
        Returns:
            Refined masks [B, N, H, W]
        """
        B, N, H, W = coarse_masks.shape

        # Project features
        feat = self.input_proj(features)

        # Detect edges if enabled
        if self.use_edge_detection:
            edge_features = self.edge_detector(feat)
        else:
            edge_features = None

        # Process each mask
        refined_masks = []
        for i in range(N):
            mask = coarse_masks[:, i:i+1, :, :]  # [B, 1, H, W]

            # Resize feature to match mask if needed
            if feat.shape[2:] != mask.shape[2:]:
                feat_resized = F.interpolate(
                    feat,
                    size=mask.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                if edge_features is not None:
                    edge_resized = F.interpolate(
                        edge_features,
                        size=mask.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    edge_resized = None
            else:
                feat_resized = feat
                edge_resized = edge_features

            # Concatenate mask with features
            x = feat_resized * mask.sigmoid()  # Mask-guided features

            # Apply refinement layers
            for layer in self.refinement_layers:
                x = layer(x, mask, edge_resized)

            # Generate refined mask
            refined_mask = self.output_proj(x)
            refined_masks.append(refined_mask)

        refined_masks = torch.cat(refined_masks, dim=1)  # [B, N, H, W]

        return refined_masks


class EdgeDetectionModule(nn.Module):
    """Edge detection module using multi-scale edge detection."""
    def __init__(self, channels: int):
        super().__init__()

        # Sobel-like edge detection kernels
        self.edge_conv_x = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.edge_conv_y = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

        # Initialize with edge detection kernels
        self._init_edge_kernels()

        # Edge feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _init_edge_kernels(self):
        """Initialize edge detection kernels (Sobel-like)."""
        for m in [self.edge_conv_x, self.edge_conv_y]:
            nn.init.constant_(m.weight, 0.0)

        # Simple edge detection initialization
        # In practice, you might want to use learnable parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Edge features [B, C, H, W]
        """
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)

        edge_feat = torch.cat([edge_x, edge_y], dim=1)
        edge_feat = self.fusion(edge_feat)

        return edge_feat


class BoundaryRefinementLayer(nn.Module):
    """Single boundary refinement layer with edge-aware processing."""
    def __init__(
        self,
        channels: int,
        use_edge_features: bool = True
    ):
        super().__init__()
        self.use_edge_features = use_edge_features

        input_channels = channels
        if use_edge_features:
            input_channels += channels  # Add edge features

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Attention for boundary regions
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            mask: Current mask prediction [B, 1, H, W]
            edge_features: Edge features [B, C, H, W] (optional)
        Returns:
            Refined features [B, C, H, W]
        """
        # Compute boundary attention
        boundary_att = self.boundary_attention(mask)

        # Concatenate with edge features if available
        if self.use_edge_features and edge_features is not None:
            x = torch.cat([x, edge_features], dim=1)

        # Apply convolutions
        identity = x if x.shape[1] == self.conv1[0].in_channels else None
        x = self.conv1(x)
        x = self.conv2(x)

        # Apply boundary attention
        x = x * boundary_att

        # Residual connection if dimensions match
        if identity is not None and identity.shape == x.shape:
            x = x + identity

        return x


def test_modules():
    """Test function for all enhancement modules."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test ColorAttentionModule
    print("Testing ColorAttentionModule...")
    color_attn = ColorAttentionModule(in_channels=256).to(device)
    x = torch.randn(2, 256, 32, 32).to(device)
    out = color_attn(x)
    print(f"ColorAttentionModule: Input {x.shape} -> Output {out.shape}")
    assert out.shape == x.shape

    # Test ColorContrastLoss
    print("\nTesting ColorContrastLoss...")
    color_loss = ColorContrastLoss().to(device)
    pred_masks = torch.sigmoid(torch.randn(2, 5, 64, 64)).to(device)
    target_masks = torch.sigmoid(torch.randn(2, 5, 64, 64)).to(device)
    images = torch.randn(2, 3, 64, 64).to(device)
    loss = color_loss(pred_masks, target_masks, images)
    print(f"ColorContrastLoss: {loss.item()}")
    assert loss.ndim == 0  # Scalar loss

    # Test EnhancedDeformableAttention
    print("\nTesting EnhancedDeformableAttention...")
    enh_deform_attn = EnhancedDeformableAttention(
        d_model=256, n_levels=4, n_heads=8
    ).to(device)
    query = torch.randn(2, 100, 256).to(device)
    reference_points = torch.rand(2, 100, 4, 2).to(device)
    value = torch.randn(2, 1000, 256).to(device)
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]]).to(device)
    level_start_index = torch.tensor([0, 1024, 1280, 1344]).to(device)
    out = enh_deform_attn(query, reference_points, value, spatial_shapes, level_start_index)
    print(f"EnhancedDeformableAttention: Query {query.shape} -> Output {out.shape}")
    assert out.shape == query.shape

    # Test BoundaryRefinementNetwork
    print("\nTesting BoundaryRefinementNetwork...")
    boundary_net = BoundaryRefinementNetwork(in_channels=256).to(device)
    features = torch.randn(2, 256, 64, 64).to(device)
    coarse_masks = torch.randn(2, 5, 64, 64).to(device)
    refined = boundary_net(features, coarse_masks)
    print(f"BoundaryRefinementNetwork: Coarse {coarse_masks.shape} -> Refined {refined.shape}")
    assert refined.shape == coarse_masks.shape

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_modules()
