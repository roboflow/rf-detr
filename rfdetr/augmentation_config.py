# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": (90, 90), "p": 0.5},  # Better keep small angles
    # "ColorJitter": {
    #     "brightness": 0.1,
    #     "contrast": 0.1,
    #     "saturation": 0.1,
    #     "hue": 0.0,
    #     "p": 0.5
    # },
    # "RandomCrop": {"height": 512, "width": 512, "p": 0.5},
    # "Affine": {
    #     "scale": (0.9, 1.1),
    #     "translate_percent": (0.25, 0.25),
    #     "rotate": (0, 0),  # Better keep small angles
    #     "shear": (0, 0),
    #     "p": 0.5
    # },
    # "CoarseDropout": {
    #     "num_holes_range": (1, 1),
    #     "hole_height_range": (0.05, 0.1),
    #     "hole_width_range": (0.05, 0.1),
    #     "p": 0.5
    # },
    # "Blur": {
    #     "blur_limit": 3,
    #     "p": 0.3
    # },
    # "GaussNoise": {
    #     "std_range": (0.01, 0.05),
    #     "p": 0.3
    # },
    # "HueSaturationValue": {
    #     "hue_shift_limit": 2,
    #     "sat_shift_limit": 2,
    #     "val_shift_limit": 2,
    #     "p": 0.5
    # },
    # "CLAHE": {
    #     "clip_limit": 2.0,
    #     "tile_grid_size": (8, 8),
    #     "p": 0.5
    # },
    # "ChannelShuffle": {"p": 0.5},
    # "RandomBrightnessContrast": {
    #     "brightness_limit": 0.1,
    #     "contrast_limit": 0.1,
    #     "p": 0.2
    # },
    # "RandomShadow": {
    #     "shadow_roi": (0, 0, 1, 1),
    #     "num_shadows_limit": (1, 1),
    #     "shadow_dimension": 3,
    #     "shadow_intensity_range": (0.2, 0.3),
    #     "p": 0.5
    # }
}
