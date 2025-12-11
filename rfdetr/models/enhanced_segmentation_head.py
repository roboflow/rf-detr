# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Enhanced Segmentation Head with Mask Quality Scoring
This module implements an improved segmentation head with mask quality prediction
and dynamic refinement for better instance segmentation performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MaskQualityPredictor(nn.Module):
    """
    Predict mask quality scores for better mask ranking and selection.
    """
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, mask_features: torch.Tensor, bbox_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict mask quality scores.
        
        Args:
            mask_features: Mask features (B, N, C)
            bbox_features: Optional bbox features (B, N, 4)
            
        Returns:
            quality_scores: Quality scores (B, N, 1)
        """
        if bbox_features is not None:
            combined_features = torch.cat([mask_features, bbox_features], dim=-1)
            # Adjust input dimension if bbox features are concatenated
            combined_dim = combined_features.shape[-1]
            if combined_dim != self.predictor[0].in_features:
                # Recreate first layer with correct dimensions
                self.predictor[0] = nn.Linear(combined_dim, self.predictor[0].out_features)
            return self.predictor(combined_features)
        else:
            return self.predictor(mask_features)


class DynamicMaskRefiner(nn.Module):
    """
    Dynamically refine masks using attention mechanisms.
    """
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
    def forward(self, query_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Refine mask features using attention.
        
        Args:
            query_features: Query features (B, N, C)
            context_features: Context features (B, H*W, C)
            
        Returns:
            refined_features: Refined features (B, N, C)
        """
        B, N, C = query_features.shape
        H_W = context_features.shape[1]
        
        # Multi-head attention
        q = self.query_proj(query_features)  # (B, N, C)
        k = self.key_proj(context_features)   # (B, H_W, C)
        v = self.value_proj(context_features) # (B, H_W, C)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.view(B, H_W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H_W, head_dim)
        v = v.view(B, H_W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H_W, head_dim)
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, H_W)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        
        # Output projection
        refined = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        refined = self.norm1(refined + query_features)
        
        # Feed-forward network
        refined = self.norm2(refined + self.ffn(refined))
        
        return refined


class EnhancedSegmentationHead(nn.Module):
    """
    Enhanced segmentation head with mask quality scoring and dynamic refinement.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 3,
        downsample_ratio: int = 4,
        use_quality_prediction: bool = True,
        use_dynamic_refinement: bool = True,
        num_refinement_heads: int = 8
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.downsample_ratio = downsample_ratio
        self.use_quality_prediction = use_quality_prediction
        self.use_dynamic_refinement = use_dynamic_refinement
        
        # Mask quality predictor
        if use_quality_prediction:
            self.quality_predictor = MaskQualityPredictor(feature_dim)
        
        # Dynamic mask refiner
        if use_dynamic_refinement:
            self.mask_refiner = DynamicMaskRefiner(feature_dim, num_refinement_heads)
        
        # Enhanced mask generation layers
        self.mask_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(8, feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(8, feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
        # Feature projection for mask generation
        self.spatial_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        
        # Final mask prediction
        self.mask_predictor = nn.Conv2d(feature_dim, 1, 1)
        
        self._export = False
    
    def export(self):
        """Export mode for deployment."""
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features: List[torch.Tensor],
        image_size: Tuple[int, int],
        bbox_features: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of enhanced segmentation head.
        
        Args:
            spatial_features: Spatial features (B, C, H, W)
            query_features: Query features from decoder layers [(B, N, C)]
            image_size: Original image size (H, W)
            bbox_features: Optional bbox features for quality prediction (B, N, 4)
            
        Returns:
            mask_logits: List of mask logits [(B, N, H', W')]
            quality_scores: Optional quality scores (B, N, 1)
        """
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode='bilinear', align_corners=False)
        
        # Apply enhanced mask layers
        refined_spatial = spatial_features
        for layer in self.mask_layers:
            refined_spatial = layer(refined_spatial)
        
        # Project spatial features
        spatial_proj = self.spatial_proj(refined_spatial)
        
        # Process query features
        mask_logits = []
        quality_scores = None
        
        for i, qf in enumerate(query_features):
            # Project query features
            qf_proj = self.query_proj(qf)
            
            # Apply dynamic refinement if enabled and not the first layer
            if self.use_dynamic_refinement and i > 0:
                # Flatten spatial features for attention
                B, C, H, W = spatial_proj.shape
                spatial_flat = spatial_proj.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
                
                # Refine query features
                qf_refined = self.mask_refiner(qf_proj, spatial_flat)
            else:
                qf_refined = qf_proj
            
            # Generate mask logits
            mask_logit = torch.einsum('bchw,bnc->bnhw', spatial_proj, qf_refined)
            mask_logits.append(mask_logit)
        
        # Predict mask quality if enabled
        if self.use_quality_prediction and bbox_features is not None:
            # Use the last layer's query features for quality prediction
            quality_scores = self.quality_predictor(query_features[-1], bbox_features)
        
        return mask_logits, quality_scores
    
    def forward_export(
        self,
        spatial_features: torch.Tensor,
        query_features: List[torch.Tensor],
        image_size: Tuple[int, int],
        bbox_features: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Export-friendly forward pass.
        """
        # Simplified version for export - only processes first query feature
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode='bilinear', align_corners=False)
        
        # Apply enhanced mask layers
        refined_spatial = spatial_features
        for layer in self.mask_layers:
            refined_spatial = layer(refined_spatial)
        
        # Project spatial features
        spatial_proj = self.spatial_proj(refined_spatial)
        
        # Process only the first query feature
        qf = self.query_proj(query_features[0])
        
        # Generate mask logits
        mask_logit = torch.einsum('bchw,bnc->bnhw', spatial_proj, qf)
        
        # Predict mask quality if enabled
        quality_scores = None
        if self.use_quality_prediction and bbox_features is not None:
            quality_scores = self.quality_predictor(qf, bbox_features)
        
        return [mask_logit], quality_scores


class AdaptiveMaskLoss(nn.Module):
    """
    Adaptive mask loss that considers mask quality.
    """
    
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, quality_weight: float = 0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.quality_weight = quality_weight
        
    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute dice loss.
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        
        intersection = (inputs * targets).sum(dim=1)
        cardinality = (inputs + targets).sum(dim=1)
        
        dice_loss = 1 - (2. * intersection + 1e-6) / (cardinality + 1e-6)
        return dice_loss.mean()
    
    def ce_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        """
        return F.binary_cross_entropy_with_logits(inputs, targets)
    
    def quality_loss(self, quality_scores: torch.Tensor, dice_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute quality prediction loss.
        """
        return F.mse_loss(quality_scores.squeeze(-1), dice_scores)
    
    def forward(
        self,
        mask_logits: List[torch.Tensor],
        targets: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive mask loss.
        
        Args:
            mask_logits: List of mask logits [(B, N, H, W)]
            targets: Target masks (B, N, H, W)
            quality_scores: Optional quality scores (B, N, 1)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        losses = {}
        
        # Use the last layer's predictions
        pred_masks = mask_logits[-1]
        
        # Dice loss
        dice_loss = self.dice_loss(pred_masks, targets)
        losses['dice_loss'] = dice_loss
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(pred_masks, targets)
        losses['ce_loss'] = ce_loss
        
        # Quality loss if quality scores are available
        quality_loss = torch.tensor(0.0, device=pred_masks.device)
        if quality_scores is not None:
            # Calculate dice scores for each mask
            with torch.no_grad():
                pred_masks_sigmoid = pred_masks.sigmoid()
                dice_scores = []
                for i in range(pred_masks.shape[1]):  # iterate over masks
                    pred_mask = pred_masks_sigmoid[:, i].flatten(1)
                    target_mask = targets[:, i].flatten(1)
                    intersection = (pred_mask * target_mask).sum(dim=1)
                    cardinality = (pred_mask + target_mask).sum(dim=1)
                    dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
                    dice_scores.append(dice_score)
                dice_scores = torch.stack(dice_scores, dim=1)  # (B, N)
            
            quality_loss = self.quality_loss(quality_scores, dice_scores)
            losses['quality_loss'] = quality_loss
        
        # Combine losses
        total_loss = (
            self.dice_weight * dice_loss + 
            self.ce_weight * ce_loss + 
            self.quality_weight * quality_loss
        )
        
        return total_loss, losses
