# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Optional Albumentations augmentation presets for RF-DETR training.

RF-DETR's default training augmentation path is torchvision-native and does not require Albumentations. Importing and
passing the presets in this module as ``aug_config`` uses the optional Albumentations integration; install it with
``pip install 'rfdetr[augment]'``.

Import a preset and pass it as ``aug_config`` to your training call:

```python
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model.train(dataset_dir="...", aug_config=AUG_CONSERVATIVE) model.train(dataset_dir="...", aug_config=AUG_AGGRESSIVE)

# Disable all augmentations
model.train(dataset_dir="...", aug_config={})

# Fully custom
model.train(dataset_dir="...", aug_config={"HorizontalFlip": {"p": 0.5}})
```

## Available presets

| Preset         | Best for                                         |
| -------------- | ------------------------------------------------ |
| ``AUG_CONSERVATIVE``  | Small datasets (under 500 images)             |
| ``AUG_AGGRESSIVE``    | Large datasets (2000+ images)                 |
| ``AUG_AERIAL``        | Satellite / overhead imagery                  |
| ``AUG_INDUSTRIAL``    | Manufacturing / inspection data               |

## Transform Categories

**Geometric transforms** (automatically transform bounding boxes; this representative list is not exhaustive):
- Flips and square symmetries: HorizontalFlip, TimeReverse, VerticalFlip, D4, SquareSymmetry
- Rotations: Rotate, Affine, ShiftScaleRotate
- Crops: RandomCrop, CenterCrop, RandomResizedCrop
- Perspective: Perspective, ElasticTransform, GridDistortion

**Pixel-level transforms** (preserve bounding boxes):
- Color: ColorJitter, HueSaturationValue, RandomBrightnessContrast, ToGray
- Blur/Noise: GaussianBlur, GaussNoise, Blur
- Enhancement: CLAHE, Sharpen, Equalize

## Best Practices

1. **Start conservative**: Use moderate probabilities (p=0.3-0.5) and small parameter ranges
2. **Geometric caution**: Extreme rotations (>45°) or crops may remove too many boxes
3. **Keypoint safety**: Keypoint pipelines with an empty `keypoint_flip_pairs` list disable
   all horizontal-flip-capable aliases, including `TimeReverse`, `D4`, and `SquareSymmetry`;
   provide left/right pairs to retain them.
4. **Performance**: Fewer transforms = faster training; prioritize transforms that match your domain
5. **Validation**: Monitor validation mAP - excessive augmentation can hurt performance
6. **Domain-specific**: Enable augmentations that reflect real-world variations in your data

## Adding Custom Transforms

For geometric transforms not in GEOMETRIC_TRANSFORMS set, add them in transforms.py:

```python
GEOMETRIC_TRANSFORMS = {
    ...
    "YourCustomTransform",  # Add here
}
```

## Kornia GPU Backend

When ``augmentation_backend="kornia"`` is set in ``TrainConfig`` (or ``"auto"``/``"cpu"`` resolves to it because
Kornia is installed and CUDA is available), augmentations run on the GPU via Kornia instead of CPU Albumentations
or torchvision defaults. Install it with ``pip install 'rfdetr[augment]'``.

**Supported transforms** (all presets):

| Preset key | Kornia equivalent | Notes |
|---|---|---|
| ``HorizontalFlip`` | ``K.RandomHorizontalFlip`` | Direct |
| ``VerticalFlip`` | ``K.RandomVerticalFlip`` | Direct |
| ``Rotate`` | ``K.RandomRotation`` | ``limit`` may be scalar or tuple |
| ``Affine`` | ``K.RandomAffine`` | ``translate_percent`` treated as fraction |
| ``ColorJitter`` | ``K.ColorJiggle`` | Same multiplicative semantics |
| ``ToGray`` | ``K.RandomGrayscale`` | Grayscale, 3 channels; only ``p`` honored, method/num_output_channels ignored |
| ``RandomBrightnessContrast`` | ``K.ColorJiggle`` | ``brightness_limit`` / ``contrast_limit`` direct |
| ``GaussianBlur`` | ``K.RandomGaussianBlur`` | ``blur_limit`` rounded up to odd; ``sigma=(0.1, 2.0)`` |
| ``GaussNoise`` | ``K.RandomGaussianNoise`` | Upper bound of ``std_range`` used as fixed std |
| ``Blur`` | ``K.RandomBoxBlur`` | Box blur; ``blur_limit`` rounded up to odd, pair collapses to its upper bound |
| ``Sharpen`` | ``K.RandomSharpness`` | ``sharpness = 1.0 + alpha`` (1.0-pivoted); ``lightness``/``method`` ignored |
| ``Equalize`` | ``K.RandomEqualize`` | Only ``p`` honored; ``mode``/``by_channels``/``mask`` ignored |
| ``CLAHE`` | ``K.RandomClahe`` | ``clip_limit`` and ``tile_grid_size`` map directly |
| ``Perspective`` | ``K.RandomPerspective`` | Approximate; see the note below. ``keep_size=False`` raises |

``Perspective`` is the one entry in the table that is not a faithful mapping. Albumentations samples each
corner offset from ``abs(N(0, scale))``, Kornia samples uniformly from ``distortion_scale``, so the two
produce different distortion distributions for the same config. The upper bound of ``scale`` is used as
``distortion_scale`` and a scalar ``scale`` is read as ``(0, scale)``; the divergence is logged for every
config, not only for a range. ``keep_size=False`` raises rather than silently resizing, and
``fit_output``, ``interpolation``,
``mask_interpolation``, ``border_mode``, ``fill`` and ``fill_mask`` are ignored. Use the Albumentations
backend when the exact Albumentations semantics matter.

Not yet supported on Kornia: ``HueSaturationValue`` (Albumentations shifts hue/saturation/value additively,
Kornia's ``ColorJiggle`` scales them multiplicatively, so there is no faithful mapping), and the geometric
group ``ShiftScaleRotate``, ``RandomCrop``, ``CenterCrop``, ``RandomResizedCrop``,
``ElasticTransform`` and ``GridDistortion``, whose CPU behavior is not faithfully mapped by the current
Kornia pipeline. These still work on the Albumentations backend.

The GPU augmentation path transports the padded-batch mask with every geometric transform. Segmentation models also
carry instance-mask channels in the same synchronized mask tensor.
"""

# ---------------------------------------------------------------------------
# Default configuration (backward-compatible baseline)
# ---------------------------------------------------------------------------

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    # "VerticalFlip": {"p": 0.5},
    # "Rotate": {"limit": 15, "p": 0.5},  # Better keep small angles
}

# ---------------------------------------------------------------------------
# Named presets — import and pass directly as aug_config=<preset>
# ---------------------------------------------------------------------------

#: Minimal augmentations — safe for small datasets (under 500 images).
AUG_CONSERVATIVE = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
        "p": 0.3,
    },
}

#: Aggressive augmentations — for larger datasets (2000+ images).
AUG_AGGRESSIVE = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
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

#: Optimised for aerial / satellite imagery (overhead views, 90° rotations).
AUG_AERIAL = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": (90, 90), "p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
}

#: Optimised for industrial / manufacturing data (lighting & sensor noise).
AUG_INDUSTRIAL = {
    "HorizontalFlip": {"p": 0.3},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.3},
}
