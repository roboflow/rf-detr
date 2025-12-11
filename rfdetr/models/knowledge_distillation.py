# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Knowledge Distillation Framework for RF-DETR
This module implements knowledge distillation to transfer knowledge from larger
teacher models to smaller student models for better efficiency-accuracy trade-offs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss using attention maps and intermediate features.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: Student features (B, C, H, W) or (B, N, C)
            teacher_features: Teacher features (B, C, H, W) or (B, N, C)
            
        Returns:
            loss: Feature distillation loss
        """
        # Ensure same spatial dimensions
        if student_features.dim() == 4:  # (B, C, H, W)
            # Spatial features
            student_flat = student_features.flatten(2)  # (B, C, H*W)
            teacher_flat = teacher_features.flatten(2)  # (B, C, H*W)
            
            # Channel-wise attention
            student_attention = torch.sum(student_flat ** 2, dim=2)  # (B, C)
            teacher_attention = torch.sum(teacher_flat ** 2, dim=2)  # (B, C)
            
            # Normalize attention
            student_attention = F.softmax(student_attention / self.temperature, dim=1)
            teacher_attention = F.softmax(teacher_attention / self.temperature, dim=1)
            
            # KL divergence loss
            attention_loss = F.kl_div(
                F.log_softmax(student_attention / self.temperature, dim=1),
                teacher_attention,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Feature similarity loss
            feature_loss = F.mse_loss(student_flat, teacher_flat)
            
            return self.alpha * attention_loss + (1 - self.alpha) * feature_loss
            
        else:  # (B, N, C) - sequence features
            # Sequence features
            student_norm = F.normalize(student_features, dim=-1)
            teacher_norm = F.normalize(teacher_features, dim=-1)
            
            # Cosine similarity loss
            cosine_loss = 1 - torch.sum(student_norm * teacher_norm, dim=-1).mean()
            
            # L2 loss
            l2_loss = F.mse_loss(student_features, teacher_features)
            
            return self.alpha * cosine_loss + (1 - self.alpha) * l2_loss


class DetectionDistillationLoss(nn.Module):
    """
    Detection-specific distillation loss for bounding boxes and class predictions.
    """
    
    def __init__(
        self, 
        cls_weight: float = 1.0, 
        bbox_weight: float = 1.0,
        temperature: float = 4.0
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.temperature = temperature
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute detection distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            targets: Ground truth targets (optional)
            
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Individual loss components
        """
        losses = {}
        
        # Classification distillation
        if 'pred_logits' in student_outputs and 'pred_logits' in teacher_outputs:
            student_logits = student_outputs['pred_logits']
            teacher_logits = teacher_outputs['pred_logits']
            
            # Soft targets with temperature scaling
            student_probs = F.softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            cls_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            losses['cls_distill'] = cls_loss
        
        # Bounding box distillation
        if 'pred_boxes' in student_outputs and 'pred_boxes' in teacher_outputs:
            student_boxes = student_outputs['pred_boxes']
            teacher_boxes = teacher_outputs['pred_boxes']
            
            # L1 loss for box coordinates
            bbox_loss = F.l1_loss(student_boxes, teacher_boxes)
            
            # GIoU loss
            try:
                from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
                student_boxes_xyxy = box_cxcywh_to_xyxy(student_boxes)
                teacher_boxes_xyxy = box_cxcywh_to_xyxy(teacher_boxes)
                
                giou_loss = 1 - torch.diag(generalized_box_iou(student_boxes_xyxy, teacher_boxes_xyxy))
                giou_loss = giou_loss.mean()
                
                bbox_loss = bbox_loss + giou_loss
            except:
                pass  # Fallback to L1 loss if import fails
            
            losses['bbox_distill'] = bbox_loss
        
        # Combine losses
        total_loss = 0
        if 'cls_distill' in losses:
            total_loss += self.cls_weight * losses['cls_distill']
        if 'bbox_distill' in losses:
            total_loss += self.bbox_weight * losses['bbox_distill']
        
        return total_loss, losses


class KnowledgeDistillationTrainer:
    """
    Knowledge distillation trainer for RF-DETR models.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        feature_loss_weight: float = 1.0,
        detection_loss_weight: float = 1.0,
        temperature: float = 4.0,
        alpha: float = 0.7
    ):
        """
        Initialize knowledge distillation trainer.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to be trained
            feature_loss_weight: Weight for feature distillation loss
            detection_loss_weight: Weight for detection distillation loss
            temperature: Temperature for soft targets
            alpha: Balance factor for feature loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.feature_loss_weight = feature_loss_weight
        self.detection_loss_weight = detection_loss_weight
        
        # Initialize loss functions
        self.feature_loss = FeatureDistillationLoss(temperature=temperature, alpha=alpha)
        self.detection_loss = DetectionDistillationLoss(temperature=temperature)
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def get_intermediate_features(self, model: nn.Module, samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features from model.
        
        Args:
            model: Model to extract features from
            samples: Input samples
            
        Returns:
            features: Dictionary of intermediate features
        """
        features = {}
        
        # Hook function to capture features
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, (list, tuple)):
                    features[name] = output[0] if len(output) > 0 else None
                else:
                    features[name] = output
            return hook
        
        # Register hooks for backbone features
        if hasattr(model, 'backbone'):
            for i, layer in enumerate(model.backbone):
                layer.register_forward_hook(hook_fn(f'backbone_{i}'))
        
        # Register hooks for transformer features
        if hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'decoder'):
                model.transformer.decoder.register_forward_hook(hook_fn('decoder'))
        
        # Forward pass
        with torch.no_grad():
            _ = model(samples)
        
        return features
    
    def compute_distillation_loss(
        self,
        samples: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute knowledge distillation loss.
        
        Args:
            samples: Input samples
            targets: Ground truth targets (optional)
            
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Individual loss components
        """
        losses = {}
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(samples)
            teacher_features = self.get_intermediate_features(self.teacher_model, samples)
        
        # Get student outputs
        student_outputs = self.student_model(samples)
        student_features = self.get_intermediate_features(self.student_model, samples)
        
        # Feature distillation loss
        feature_loss = 0
        feature_count = 0
        
        for key in teacher_features:
            if key in student_features and teacher_features[key] is not None and student_features[key] is not None:
                # Handle different tensor shapes
                teacher_feat = teacher_features[key]
                student_feat = student_features[key]
                
                # Resize if necessary
                if teacher_feat.shape != student_feat.shape:
                    if teacher_feat.dim() == 4:  # Spatial features
                        student_feat = F.interpolate(
                            student_feat, 
                            size=teacher_feat.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    elif teacher_feat.dim() == 3:  # Sequence features
                        min_seq_len = min(teacher_feat.shape[1], student_feat.shape[1])
                        teacher_feat = teacher_feat[:, :min_seq_len, :]
                        student_feat = student_feat[:, :min_seq_len, :]
                
                loss = self.feature_loss(student_feat, teacher_feat)
                feature_loss += loss
                feature_count += 1
                losses[f'feature_{key}'] = loss
        
        if feature_count > 0:
            feature_loss /= feature_count
            losses['feature_distill'] = feature_loss
        
        # Detection distillation loss
        detection_loss, detection_losses = self.detection_loss(student_outputs, teacher_outputs, targets)
        losses.update(detection_losses)
        
        # Combine losses
        total_loss = 0
        if 'feature_distill' in losses:
            total_loss += self.feature_loss_weight * losses['feature_distill']
        if 'cls_distill' in losses:
            total_loss += self.detection_loss_weight * losses['cls_distill']
        if 'bbox_distill' in losses:
            total_loss += self.detection_loss_weight * losses['bbox_distill']
        
        return total_loss, losses


class ProgressiveDistillationScheduler:
    """
    Progressive distillation scheduler that gradually shifts from distillation to ground truth training.
    """
    
    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 5,
        distillation_decay: str = 'linear'
    ):
        """
        Initialize progressive distillation scheduler.
        
        Args:
            total_epochs: Total training epochs
            warmup_epochs: Warmup epochs with pure distillation
            distillation_decay: Decay type ('linear', 'cosine', 'exponential')
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.distillation_decay = distillation_decay
    
    def get_distillation_weight(self, current_epoch: int) -> float:
        """
        Get distillation weight for current epoch.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            weight: Distillation weight (0.0 to 1.0)
        """
        if current_epoch <= self.warmup_epochs:
            return 1.0  # Pure distillation during warmup
        
        progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(progress, 1.0)
        
        if self.distillation_decay == 'linear':
            return 1.0 - progress
        elif self.distillation_decay == 'cosine':
            return 0.5 * (1 + math.cos(math.pi * progress))
        elif self.distillation_decay == 'exponential':
            return math.exp(-3 * progress)
        else:
            return 1.0 - progress


def create_teacher_student_pair(
    teacher_config: Dict,
    student_config: Dict,
    teacher_checkpoint: Optional[str] = None
) -> Tuple[nn.Module, nn.Module]:
    """
    Create teacher-student model pair for distillation.
    
    Args:
        teacher_config: Teacher model configuration
        student_config: Student model configuration  
        teacher_checkpoint: Path to teacher checkpoint
        
    Returns:
        teacher_model: Teacher model
        student_model: Student model
    """
    from rfdetr.models import build_model
    
    # Build teacher model
    teacher_args = type('Args', (), teacher_config)()
    teacher_model = build_model(teacher_args)
    
    # Load teacher checkpoint if provided
    if teacher_checkpoint:
        checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded teacher checkpoint from {teacher_checkpoint}")
    
    # Build student model
    student_args = type('Args', (), student_config)()
    student_model = build_model(student_args)
    
    return teacher_model, student_model
