"""
Example script for training RF-DETR Enhanced model on car interior segmentation.

This script demonstrates how to use the enhanced RF-DETR model with:
1. Color Attention Module
2. Color Contrast Loss
3. Enhanced Deformable Attention
4. Boundary Refinement Network

These enhancements are particularly useful for car interior segmentation where:
- Different parts have distinct colors
- Precise boundaries are important
- Small details matter
"""

from rfdetr import RFDETRSegEnhanced

# Initialize the enhanced model
model = RFDETRSegEnhanced()

# Train with your car interior dataset
model.train(
    dataset_dir="path/to/your/car_interior_dataset",
    epochs=50,
    batch_size=6,
    grad_accum_steps=4,
    lr=1e-4,
    # Enhanced model configurations are automatically loaded
)

print("Training completed!")

# You can also customize the enhanced features:
# model_custom = RFDETRSegEnhanced()
# model_custom.train(
#     dataset_dir="path/to/dataset",
#     epochs=50,
#     batch_size=6,
#     grad_accum_steps=4,
#     lr=1e-4,
#     use_color_attention=True,           # Enable/disable color attention
#     use_boundary_refinement=True,       # Enable/disable boundary refinement
#     use_color_contrast_loss=True,       # Enable/disable color contrast loss
#     color_contrast_loss_weight=0.5,     # Adjust weight of color contrast loss
# )
