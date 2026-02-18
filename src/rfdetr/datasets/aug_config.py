# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Default Albumentations augmentation configuration for RF-DETR training.

This configuration defines the augmentation pipeline used during training.
The AlbumentationsWrapper automatically handles bounding box transformations
for geometric transforms (flips, rotations, crops) while preserving boxes
for pixel-level transforms (blur, color adjustments).

## Usage

Edit AUG_CONFIG to enable or customize augmentations:

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},  # 50% probability
    "Rotate": {"limit": 45, "p": 0.3},  # Rotate up to ±45°
    "GaussianBlur": {"p": 0.2},  # Pixel-level transform
}
```

## Transform Categories

**Geometric transforms** (automatically transform bounding boxes):
- Flips: HorizontalFlip, VerticalFlip
- Rotations: Rotate, Affine, ShiftScaleRotate
- Crops: RandomCrop, CenterCrop, RandomResizedCrop
- Perspective: Perspective, ElasticTransform, GridDistortion

**Pixel-level transforms** (preserve bounding boxes):
- Color: ColorJitter, HueSaturationValue, RandomBrightnessContrast
- Blur/Noise: GaussianBlur, GaussNoise, Blur
- Enhancement: CLAHE, Sharpen, Equalize

## Best Practices

1. **Start conservative**: Use moderate probabilities (p=0.3-0.5) and small parameter ranges
2. **Geometric caution**: Extreme rotations (>45°) or crops may remove too many boxes
3. **Performance**: Fewer transforms = faster training; prioritize transforms that match your domain
4. **Validation**: Monitor validation mAP - excessive augmentation can hurt performance
5. **Domain-specific**: Enable augmentations that reflect real-world variations in your data

## Adding Custom Transforms

For geometric transforms not in GEOMETRIC_TRANSFORMS set, add them in transforms.py:

```python
GEOMETRIC_TRANSFORMS = {
    ...
    "YourCustomTransform",  # Add here
}
```
"""

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    # "VerticalFlip": {"p": 0.5},
    # "Rotate": {"limit": 15, "p": 0.5},  # Better keep small angles
}
