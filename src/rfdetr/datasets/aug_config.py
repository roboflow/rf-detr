# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Kornia augmentation presets and default configuration for RF-DETR training.

Import a preset and pass it as ``aug_config`` to your training call:

```python
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model.train(dataset_dir="...", aug_config=AUG_CONSERVATIVE)
model.train(dataset_dir="...", aug_config=AUG_AGGRESSIVE)

# Disable optional augmentations
model.train(dataset_dir="...", aug_config={})

# Fully custom within RF-DETR's supported Kornia transform keys
model.train(dataset_dir="...", aug_config={"HorizontalFlip": {"p": 0.5}})
```

Supported optional augmentation keys are ``HorizontalFlip``, ``VerticalFlip``, ``Rotate``, ``Affine``, ``ColorJitter``,
``RandomBrightnessContrast``, ``GaussianBlur``, and ``GaussNoise``. RF-DETR also uses internal Kornia resize/container
keys for its dataset pipeline: ``Resize``, ``SmallestMaxSize``, ``LongestMaxSize``, ``RandomSizedCrop``, ``OneOf``, and
``Sequential``.

Kornia is used for both CPU dataset-time transforms and optional GPU-side augmentation when
``augmentation_backend="auto"`` or ``"gpu"`` is selected.
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
