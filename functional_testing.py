#!/usr/bin/env python3
"""
Test script for RF-DETR enhancements.
Validates IoU-aware query selection, enhanced segmentation, and knowledge distillation.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add rfdetr to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rfdetr.models.iou_aware_query_selector import IoUAwareQuerySelector, AdaptiveQueryAllocator
from rfdetr.models.enhanced_segmentation_head import EnhancedSegmentationHead, AdaptiveMaskLoss
from rfdetr.models.knowledge_distillation import KnowledgeDistillationTrainer, FeatureDistillationLoss


def test_iou_aware_query_selector():
    """Test IoU-aware query selector."""
    print("Testing IoU-aware query selector...")
    
    # Create dummy data
    batch_size = 2
    num_queries = 300
    feature_dim = 256
    num_memory = 1000
    
    # Initialize selector
    selector = IoUAwareQuerySelector(
        d_model=feature_dim,
        num_queries=num_queries
    )
    
    # Create dummy inputs
    memory = torch.randn(batch_size, num_memory, feature_dim)
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]])
    level_start_index = torch.tensor([0, 1024, 1536, 1792])
    reference_points = torch.rand(batch_size, num_queries, 4)
    
    # Forward pass
    try:
        selected_features, scores = selector(memory, spatial_shapes, level_start_index, reference_points)
        assert selected_features.shape == (batch_size, num_queries, feature_dim)
        assert scores.shape == (batch_size, num_queries, 1)
        print("✓ IoU-aware query selector test passed")
        return True
    except Exception as e:
        print(f"✗ IoU-aware query selector test failed: {e}")
        return False


def test_adaptive_query_allocator():
    """Test adaptive query allocator."""
    print("Testing adaptive query allocator...")
    
    # Create dummy data
    batch_size = 2
    num_queries = 300
    feature_dim = 256
    num_memory = 1000
    
    # Initialize allocator
    allocator = AdaptiveQueryAllocator(base_queries=num_queries)
    
    # Create dummy input
    memory = torch.randn(batch_size, num_memory, feature_dim)
    
    # Forward pass
    try:
        allocated_queries = allocator(memory)
        assert isinstance(allocated_queries, int)
        assert 100 <= allocated_queries <= 600  # Should be within min/max range
        print(f"✓ Adaptive query allocator test passed (allocated {allocated_queries} queries)")
        return True
    except Exception as e:
        print(f"✗ Adaptive query allocator test failed: {e}")
        return False


def test_enhanced_segmentation_head():
    """Test enhanced segmentation head."""
    print("Testing enhanced segmentation head...")
    
    # Create dummy data
    batch_size = 2
    num_queries = 100
    feature_dim = 256
    image_size = (512, 512)
    
    # Initialize enhanced segmentation head
    seg_head = EnhancedSegmentationHead(
        feature_dim=feature_dim,
        num_layers=3,
        use_quality_prediction=True,
        use_dynamic_refinement=True
    )
    
    # Create dummy inputs
    spatial_features = torch.randn(batch_size, feature_dim, 64, 64)
    query_features = [torch.randn(batch_size, num_queries, feature_dim) for _ in range(3)]
    bbox_features = torch.rand(batch_size, num_queries, 4)
    
    # Forward pass
    try:
        mask_logits, quality_scores = seg_head(
            spatial_features, query_features, image_size, bbox_features
        )
        
        assert len(mask_logits) == 3  # Should have 3 layers
        assert mask_logits[-1].shape == (batch_size, num_queries, 128, 128)
        assert quality_scores is not None
        assert quality_scores.shape == (batch_size, num_queries, 1)
        print("✓ Enhanced segmentation head test passed")
        return True
    except Exception as e:
        print(f"✗ Enhanced segmentation head test failed: {e}")
        return False


def test_knowledge_distillation():
    """Test knowledge distillation framework."""
    print("Testing knowledge distillation...")
    
    # Create dummy models
    feature_dim = 256
    num_classes = 91
    num_queries = 300
    
    # Simple teacher and student models
    teacher_model = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, feature_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(feature_dim, num_classes)
    )
    
    student_model = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, feature_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(feature_dim, num_classes)
    )
    
    # Initialize distillation trainer
    try:
        distiller = KnowledgeDistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model
        )
        print("✓ Knowledge distillation trainer initialization passed")
        return True
    except Exception as e:
        print(f"✗ Knowledge distillation test failed: {e}")
        return False


def test_feature_distillation_loss():
    """Test feature distillation loss."""
    print("Testing feature distillation loss...")
    
    # Create dummy data
    batch_size = 2
    feature_dim = 256
    spatial_size = 32
    
    # Initialize loss function
    loss_fn = FeatureDistillationLoss()
    
    # Create dummy features
    student_features = torch.randn(batch_size, feature_dim, spatial_size, spatial_size)
    teacher_features = torch.randn(batch_size, feature_dim, spatial_size, spatial_size)
    
    # Forward pass
    try:
        loss = loss_fn(student_features, teacher_features)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        print("✓ Feature distillation loss test passed")
        return True
    except Exception as e:
        print(f"✗ Feature distillation loss test failed: {e}")
        return False


def test_integration():
    """Test integration with existing RF-DETR components."""
    print("Testing integration...")
    
    try:
        # Test imports work correctly
        from rfdetr.models.transformer import Transformer
        from rfdetr.config import ModelConfig
        
        # Test configuration with new features
        config = ModelConfig(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[2, 5, 8, 11],
            dec_layers=3,
            projector_scale=["P3", "P4", "P5"],
            hidden_dim=256,
            patch_size=14,
            num_windows=4,
            sa_nheads=8,
            ca_nheads=8,
            dec_n_points=4,
            resolution=640,
            positional_encoding_size=10000,
            use_iou_aware_query=True,
            adaptive_query_allocation=True,
            enhanced_segmentation=True,
            mask_quality_prediction=True,
            dynamic_mask_refinement=True
        )
        
        assert config.use_iou_aware_query == True
        assert config.adaptive_query_allocation == True
        assert config.enhanced_segmentation == True
        
        print("✓ Integration test passed")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running RF-DETR Enhancement Tests\n")
    print("=" * 50)
    
    tests = [
        test_iou_aware_query_selector,
        test_adaptive_query_allocator,
        test_enhanced_segmentation_head,
        test_knowledge_distillation,
        test_feature_distillation_loss,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Ready for PR.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before creating PR.")
        return 1


if __name__ == "__main__":
    exit(main())
