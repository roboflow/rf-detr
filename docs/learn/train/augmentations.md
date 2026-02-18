# Custom Augmentations with Albumentations

RF-DETR supports custom data augmentations using the [Albumentations](https://albumentations.ai/) library, providing access to over 70 different image transformations optimized for object detection tasks.

## Why Albumentations?

- **Detection & Segmentation Support:** Geometric transforms automatically update bounding boxes and segmentation masks
- **Performance:** Highly optimized, faster than torchvision transforms
- **Flexibility:** Mix and match over 70 different augmentations
- **Battle-Tested:** Used in winning solutions of many Kaggle competitions

## Setup

Albumentations is installed automatically with RF-DETR:

```bash
pip install rfdetr
```

## Basic Usage

Augmentations are configured via the `AUG_CONFIG` dictionary in `src/rfdetr/augmentation_config.py`:

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": (90, 90), "p": 0.5},
}
```

Simply enable the augmentations you want by uncommenting them or adding new ones. The probability `p` controls how often each transform is applied.

## Available Augmentations

### Geometric Transforms

These transforms automatically update bounding boxes and segmentation masks:

- `HorizontalFlip` - Flip image horizontally
- `VerticalFlip` - Flip image vertically
- `Rotate` - Rotate image by random angle
- `Affine` - Apply affine transformations (scale, translate, rotate, shear)
- `RandomCrop` - Crop random region
- `ShiftScaleRotate` - Combination of shifting, scaling, and rotating
- `Perspective` - Apply perspective transformation
- `ElasticTransform` - Apply elastic deformation
- `GridDistortion` - Apply grid distortion
- `OpticalDistortion` - Apply optical distortion

### Pixel-Level Transforms

These transforms preserve bounding boxes and masks (no coordinate changes):

- `ColorJitter` - Randomly change brightness, contrast, saturation
- `GaussianBlur` - Apply Gaussian blur
- `GaussNoise` - Add Gaussian noise
- `CLAHE` - Contrast Limited Adaptive Histogram Equalization
- `RandomBrightnessContrast` - Adjust brightness and contrast
- `HueSaturationValue` - Randomly change hue, saturation, and value
- `ChannelShuffle` - Randomly shuffle image channels
- `CoarseDropout` - Apply coarse dropout (cutout)

## Configuration Examples

### Conservative Augmentations

Recommended for small datasets (under 500 images):

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
        "p": 0.3,
    },
}
```

### Aggressive Augmentations

For larger datasets (2000+ images):

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
}
```

### Aerial Imagery / Satellite Datasets

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},  # Important for overhead views
    "Rotate": {"limit": (90, 90), "p": 0.5},  # 90° rotations common
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
}
```

### Industrial / Manufacturing

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.3},  # Less common in manufacturing
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},  # Camera focus variations
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.3},  # Sensor noise
}
```

## How It Works

Augmentations are automatically applied during training:

1. The `AUG_CONFIG` is read when building the dataset
2. Transforms are composed into a pipeline
3. Each training sample is augmented on-the-fly
4. Bounding boxes and masks are automatically transformed for geometric augmentations

No code changes needed in your training script - just modify `augmentation_config.py`.

## Programmatic Configuration

You can also build augmentations programmatically for advanced use cases:

```python
from rfdetr.datasets.transforms import AlbumentationsWrapper, ComposeAugmentations

# Custom config
custom_config = {
    "HorizontalFlip": {"p": 0.7},
    "Rotate": {"limit": 15, "p": 0.5},
    "Blur": {"blur_limit": 3, "p": 0.2},
}

# Build and compose transforms using the static method
transforms = AlbumentationsWrapper.from_config(custom_config)
augmentation_pipeline = ComposeAugmentations(transforms)

# Apply to image and target (works with both detection and segmentation)
# For detection: target contains "boxes" and "labels"
# For segmentation: target contains "boxes", "labels", and "masks"
augmented_image, augmented_target = augmentation_pipeline(image, target)
```

## Best Practices

!!! tip "Start Conservative"

    Begin with simple augmentations (horizontal flip, small brightness changes) and gradually add more as needed.

!!! warning "Geometric Transforms"

    Be careful with aggressive rotations and crops on datasets where object orientation matters (e.g., text detection, oriented objects).

### Recommendations by Dataset Size

| Dataset Size     | Recommended Augmentations                                        |
| ---------------- | ---------------------------------------------------------------- |
| Under 500 images | Horizontal flip, small brightness/contrast adjustments           |
| 500-2000 images  | Add vertical flip (if applicable), color jitter, blur            |
| 2000+ images     | Add rotations, affine transforms, aggressive color augmentations |

### Performance Tips

- **CPU-bound:** Augmentations run on CPU during data loading
- **More augmentations = slower loading** (but better model generalization)
- **Use `num_workers`:** Parallelize augmentation with data loader workers
- **Monitor GPU utilization:** If GPU isn't saturated, you can add more augmentations

Example data loader configuration:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # Parallelize augmentations across 4 workers
    pin_memory=True,
)
```

## Monitoring Augmentations

### Visualize Augmented Images

To verify your augmentations are working as expected:

```python
import matplotlib.pyplot as plt
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.aug_config import AUG_CONFIG
from rfdetr.datasets.transforms import AlbumentationsWrapper, ComposeAugmentations

# Build dataset with augmentations
transforms = AlbumentationsWrapper.from_config(AUG_CONFIG)
augmentation_pipeline = ComposeAugmentations(transforms)

dataset = CocoDetection(
    img_folder="path/to/images",
    ann_file="path/to/annotations.json",
    transforms=augmentation_pipeline,
)

# Visualize
image, target = dataset[0]
plt.imshow(image)
plt.show()
```

### Expected Training Behavior

!!! note

    With augmentations enabled, it's normal to see:

    - **Training mAP lower than validation mAP** - Training uses augmented (harder) images
    - **Slower data loading** - CPU preprocessing time increases
    - **Better generalization** - Model learns from more diverse data

## Troubleshooting

### Problem: Training is very slow

**Solutions:**

- Reduce number of augmentations
- Reduce augmentation complexity (e.g., smaller rotation angles)
- Increase `num_workers` in data loader
- Profile to identify slow transforms

### Problem: Validation mAP is much higher than training mAP

**This is expected** with strong augmentations:

- Validation uses original images (no augmentation)
- Training mAP is artificially lower due to augmented data
- This gap is normal and indicates augmentations are working

### Problem: Some boxes or masks disappear after augmentation

**This is normal behavior:**

- Aggressive transforms (large rotations, crops) can move boxes outside boundaries
- Albumentations removes boxes and their corresponding masks that fall outside image

**Solutions:**

- Reduce augmentation intensity
- Use smaller rotation angles
- Avoid aggressive crops
- Advanced: Reduce `min_visibility` in `AlbumentationsWrapper` (requires code changes)

### Problem: Model not improving

**Check if augmentations are too aggressive:**

- Try reducing augmentation probabilities
- Remove geometric transforms temporarily
- Start with only color augmentations
- Gradually add back geometric transforms

## Advanced Topics

### Custom Transform Integration

To add a custom Albumentations transform not in the default config:

1. Import the transform in your config:

    ```python
    # aug_config.py
    AUG_CONFIG = {
        "HorizontalFlip": {"p": 0.5},
        "RandomShadow": {  # Custom shadow augmentation
            "shadow_roi": (0, 0, 1, 1),
            "num_shadows_limit": (1, 1),
            "shadow_dimension": 3,
            "p": 0.3,
        },
    }
    ```

2. The transform will be automatically loaded if it exists in Albumentations

### Conditional Augmentations

Use different augmentations for different scenarios:

```python
# For training
TRAIN_AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
}

# For fine-tuning (lighter augmentations)
FINETUNE_AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.3},
    "RandomBrightnessContrast": {"brightness_limit": 0.1, "contrast_limit": 0.1, "p": 0.2},
}
```

### Debug Mode

To see augmentation warnings and statistics:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Now you'll see messages like:
# INFO - Built 3 Albumentations transforms from config
# WARNING - Unknown Albumentations transform: CustomTransform. Skipping.
```

## Reference

- **Albumentations Documentation:** [https://albumentations.ai/docs/](https://albumentations.ai/docs/)
- **Available Transforms:** [https://albumentations.ai/docs/api_reference/augmentations/](https://albumentations.ai/docs/api_reference/augmentations/)
- **Examples Gallery:** [https://albumentations.ai/docs/examples/](https://albumentations.ai/docs/examples/)

## Next Steps

After configuring augmentations:

- [Monitor training with TensorBoard](advanced.md#logging-with-tensorboard)
- [Use early stopping](advanced.md#early-stopping) to prevent overfitting
- [Export your trained model](../export.md) for deployment
