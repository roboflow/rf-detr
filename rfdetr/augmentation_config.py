AUG_CONFIG = {
    "AlbumentationsHorizontalFlip": {"p": 0.5},
    "AlbumentationsRotate": {"limit": 30, "p": 0.5},
    # "AlbumentationsRandomBrightnessContrast": {"p": 0.2},
    # "AlbumentationsShiftScaleRotate": {"shift_limit": 0.0625, "scale_limit": 0.1, "rotate_limit": 15, "p": 0.5},
    "AlbumentationsGaussNoise": {"var_limit": (10.0, 50.0), "p": 0.3},
    # "AlbumentationsColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1, "p": 0.5},
    # "AlbumentationsBlur": {"blur_limit": 7, "p": 0.3},
    "AlbumentationsCoarseDropout": {"max_holes": 8, "max_height": 16, "max_width": 16, "p": 0.5},
    "AlbumentationsVerticalFlip": {"p": 0.5},
    # "AlbumentationsHueSaturationValue": {"hue_shift_limit": 20, "sat_shift_limit": 30, "val_shift_limit": 20, "p": 0.5},
    # "AlbumentationsCLAHE": {"clip_limit": 4.0, "tile_grid_size": (8, 8), "p": 0.5},
    # "AlbumentationsChannelShuffle": {"p": 0.5},
    # "AlbumentationsRandomCrop": {"height": 300, "width": 300, "p": 0.5},
    # "AlbumentationsAffine": {"scale": (0.9, 1.1), "translate_percent": (0.1, 0.1), "rotate": (-15, 15), "shear": (-10, 10), "p": 0.5},
    # "AlbumentationsRandomShadow": {
    #     "flare_roi": (0, 0.5, 1, 1),
    #     "angle_lower": 0.3, "angle_upper": 1.3,
    #     "num_flare_circles_lower": 1, "num_flare_circles_upper": 3,
    #     "p": 0.5
    # }
}
