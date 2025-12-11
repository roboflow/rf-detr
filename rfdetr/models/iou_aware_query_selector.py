# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
IoU-aware Query Selection for RF-DETR
This module implements IoU-aware query selection inspired by RT-DETR v2
to improve object query initialization and focus on relevant objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class IoUAwareQuerySelector(nn.Module):
    """
    IoU-aware query selection mechanism for improved object query initialization.
    
    This module selects the most relevant queries based on IoU scores between
    predicted boxes and reference points, improving detection accuracy especially
    for small objects and reducing false positives.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 300,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_aware_score: bool = True,
        alpha: float = 0.5  # balance between classification and localization
    ):
        """
        Initialize IoU-aware query selector.
        
        Args:
            d_model: Feature dimension
            num_queries: Number of object queries
            num_layers: Number of MLP layers
            dropout: Dropout rate
            use_aware_score: Whether to use IoU-aware scoring
            alpha: Balance factor between classification and localization scores
        """
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.use_aware_score = use_aware_score
        self.alpha = alpha
        
        # Classification confidence prediction
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)  # binary classification (object vs background)
        )
        
        # Bounding box regression
        self.bbox_regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4)  # (cx, cy, w, h)
        )
        
        # IoU prediction network
        self.iou_predictor = nn.Sequential(
            nn.Linear(d_model + 4, d_model),  # features + bbox coords
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Query feature enhancement
        self.query_enhancer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for IoU-aware query selection.
        
        Args:
            memory: Feature memory from encoder (bs, N, d_model)
            spatial_shapes: Spatial shapes of feature levels (num_levels, 2)
            level_start_index: Start index for each level (num_levels,)
            reference_points: Reference points for queries (bs, num_queries, 4)
            
        Returns:
            selected_features: Selected query features (bs, num_queries, d_model)
            selection_scores: Selection scores (bs, N, 1)
        """
        bs, N, d_model = memory.shape
        
        # Predict classification confidence
        cls_scores = self.classifier(memory)  # (bs, N, 1)
        cls_scores = torch.sigmoid(cls_scores)
        
        # Predict bounding boxes
        bbox_deltas = self.bbox_regressor(memory)  # (bs, N, 4)
        bbox_deltas = torch.sigmoid(bbox_deltas)  # normalize to [0, 1]
        
        # Calculate IoU-aware scores
        if self.use_aware_score and reference_points is not None:
            # Use only the first num_queries reference points for memory positions
            if reference_points.shape[1] < N:
                # If we have fewer reference points than memory positions, repeat the reference points
                repeat_times = (N + reference_points.shape[1] - 1) // reference_points.shape[1]
                ref_points_expanded = reference_points.repeat(1, repeat_times, 1)[:, :N, :]
            else:
                # Use the first N reference points
                ref_points_expanded = reference_points[:, :N, :]
            
            # Calculate IoU between predicted boxes and reference points
            iou_scores = self._calculate_iou_aware_score(
                bbox_deltas, ref_points_expanded, memory
            )
            
            # Combine classification and IoU scores
            combined_scores = (
                self.alpha * cls_scores + 
                (1 - self.alpha) * iou_scores
            )
        else:
            combined_scores = cls_scores
        
        # Select top-K queries
        top_k_scores, top_k_indices = torch.topk(
            combined_scores.squeeze(-1), 
            k=min(self.num_queries, N), 
            dim=-1
        )  # (bs, num_queries)
        
        # Gather selected features
        selected_features = torch.gather(
            memory, 
            1, 
            top_k_indices.unsqueeze(-1).expand(-1, -1, d_model)
        )  # (bs, num_queries, d_model)
        
        # Enhance selected query features
        enhanced_features = self.query_enhancer(selected_features)
        enhanced_features = enhanced_features + selected_features  # residual connection
        
        return enhanced_features, top_k_scores.unsqueeze(-1)
    
    def _calculate_iou_aware_score(
        self, 
        pred_boxes: torch.Tensor, 
        ref_boxes: torch.Tensor, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate IoU-aware scores using predicted boxes and reference points.
        
        Args:
            pred_boxes: Predicted boxes (bs, N, 4) in (cx, cy, w, h) format
            ref_boxes: Reference boxes (bs, N, 4) in (cx, cy, w, h) format  
            features: Feature vectors (bs, N, d_model)
            
        Returns:
            iou_scores: IoU-aware scores (bs, N, 1)
        """
        # Convert to (x1, y1, x2, y2) format for IoU calculation
        pred_boxes_xyxy = self._cxcywh_to_xyxy(pred_boxes)
        ref_boxes_xyxy = self._cxcywh_to_xyxy(ref_boxes)
        
        # Calculate IoU
        iou = self._calculate_iou(pred_boxes_xyxy, ref_boxes_xyxy)  # (bs, N)
        
        # Predict IoU using features and box coordinates
        box_features = torch.cat([features, pred_boxes], dim=-1)  # (bs, N, d_model + 4)
        predicted_iou = self.iou_predictor(box_features).squeeze(-1)  # (bs, N)
        
        # Combine geometric IoU with predicted IoU
        combined_iou = 0.7 * iou + 0.3 * predicted_iou
        
        return combined_iou.unsqueeze(-1)
    
    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _calculate_iou(
        self, 
        boxes1: torch.Tensor, 
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate IoU between two sets of boxes.
        
        Args:
            boxes1: First set of boxes (bs, N, 4) in (x1, y1, x2, y2) format
            boxes2: Second set of boxes (bs, N, 4) in (x1, y1, x2, y2) format
            
        Returns:
            iou: IoU values (bs, N)
        """
        # Intersection
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-7)
        
        return iou


class AdaptiveQueryAllocator(nn.Module):
    """
    Adaptive query allocation based on image complexity.
    
    This module dynamically adjusts the number of queries based on
    image complexity to improve efficiency and accuracy.
    """
    
    def __init__(
        self,
        base_queries: int = 300,
        max_queries: int = 600,
        min_queries: int = 100,
        complexity_threshold: float = 0.5
    ):
        """
        Initialize adaptive query allocator.
        
        Args:
            base_queries: Base number of queries
            max_queries: Maximum number of queries
            min_queries: Minimum number of queries  
            complexity_threshold: Threshold for increasing queries
        """
        super().__init__()
        self.base_queries = base_queries
        self.max_queries = max_queries
        self.min_queries = min_queries
        self.complexity_threshold = complexity_threshold
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, memory: torch.Tensor) -> int:
        """
        Estimate image complexity and determine optimal number of queries.
        
        Args:
            memory: Feature memory (bs, N, d_model)
            
        Returns:
            num_queries: Optimal number of queries
        """
        # Average pool features to get global representation
        global_features = memory.mean(dim=1)  # (bs, d_model)
        
        # Estimate complexity
        complexity = self.complexity_estimator(global_features).mean()  # scalar
        
        # Adaptive query allocation
        if complexity > self.complexity_threshold:
            num_queries = min(
                int(self.base_queries * (1 + complexity)),
                self.max_queries
            )
        else:
            num_queries = max(
                int(self.base_queries * complexity),
                self.min_queries
            )
        
        return num_queries
