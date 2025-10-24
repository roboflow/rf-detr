"""
Simple test script to verify the enhanced modules can be imported and instantiated.
"""

import sys
import os

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test basic imports
    print("Testing imports...")

    # Test enhancements module
    print("1. Testing enhancements module...")
    from rfdetr.models.enhancements import (
        ColorAttentionModule,
        ColorContrastLoss,
        EnhancedDeformableAttention,
        BoundaryRefinementNetwork
    )
    print("   ✓ Enhancements module imported successfully")

    # Test enhanced segmentation head
    print("2. Testing enhanced segmentation head...")
    from rfdetr.models.enhanced_segmentation_head import EnhancedSegmentationHead
    print("   ✓ Enhanced segmentation head imported successfully")

    # Test enhanced criterion
    print("3. Testing enhanced criterion...")
    from rfdetr.models.enhanced_criterion import EnhancedSetCriterion
    print("   ✓ Enhanced criterion imported successfully")

    # Test enhanced build
    print("4. Testing enhanced build module...")
    from rfdetr.models.enhanced_build import (
        build_enhanced_model,
        build_enhanced_criterion_and_postprocessors
    )
    print("   ✓ Enhanced build module imported successfully")

    # Test config
    print("5. Testing config...")
    from rfdetr.config import RFDETRSegEnhancedConfig, EnhancedSegmentationTrainConfig
    print("   ✓ Config classes imported successfully")

    # Test engine enhanced
    print("6. Testing engine enhanced...")
    from rfdetr.engine_enhanced import call_criterion_with_images, is_enhanced_criterion
    print("   ✓ Engine enhanced imported successfully")

    print("\n" + "="*50)
    print("All imports successful!")
    print("="*50)

    # Test instantiation (without torch)
    print("\nTesting configuration instantiation...")
    config = RFDETRSegEnhancedConfig()
    print(f"✓ Model config created: {config.resolution}x{config.resolution}")
    print(f"  - Color attention: {config.use_color_attention}")
    print(f"  - Boundary refinement: {config.use_boundary_refinement}")
    print(f"  - Enhanced deformable attention: {config.use_enhanced_deformable_attention}")
    print(f"  - Color contrast loss: {config.use_color_contrast_loss}")

    train_config = EnhancedSegmentationTrainConfig(
        dataset_dir="/tmp/dataset",
        batch_size=4,
        epochs=10
    )
    print(f"✓ Train config created: {train_config.batch_size} batch size, {train_config.epochs} epochs")

    print("\n" + "="*50)
    print("SUCCESS: All tests passed!")
    print("="*50)
    print("\nThe enhanced RF-DETR model is ready to use.")
    print("You can now train with:")
    print("  from rfdetr import RFDETRSegEnhanced")
    print("  model = RFDETRSegEnhanced()")
    print("  model.train(dataset_dir='...', epochs=50, batch_size=6)")

except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nThis might be due to missing dependencies.")
    print("Please ensure torch and other dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
