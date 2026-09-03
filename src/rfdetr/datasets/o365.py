# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Dataset file for Object365."""

from pathlib import Path
from typing import Any

from PIL import Image

from rfdetr.datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.utilities.logger import get_logger

# O365 contains images larger than PIL's default 178M-pixel limit.
# Set a generous but finite cap (2 billion pixels ~ a 45k x 45k image) instead of
# disabling the guard entirely, so PIL still protects against decompression bombs.
Image.MAX_IMAGE_PIXELS = 2_000_000_000

logger = get_logger()


def build_o365_raw(image_set: str, args: Any, resolution: int) -> CocoDetection:
    """Build one Object365 detection split.

    Object365 currently uses detection annotations only and does not consume
    ``segmentation_head``. Direct callers must provide either ``dataset_dir``
    or ``coco_path``, plus ``square_resize_div_64``, ``multi_scale``,
    ``expanded_scales``, ``do_random_resize_via_padding``, ``patch_size``, and
    ``num_windows`` on ``args``. Optional ``scale_jitter`` and
    ``augmentation_backend`` values retain safe defaults.

    Args:
        image_set: Object365 split, either ``"train"`` or ``"val"``.
        args: Dataset and transform configuration namespace.
        resolution: Target square resolution in pixels.

    Returns:
        The configured Object365 detection dataset.

    Raises:
        AttributeError: If a required dataset or transform option is absent.
        KeyError: If ``image_set`` is not a supported Object365 split.
    """
    root = Path(getattr(args, "dataset_dir", None) or args.coco_path)
    PATHS = {  # noqa: N806
        "train": (root, root / "zhiyuan_objv2_train_val_wo_5k.json"),
        "val": (root, root / "zhiyuan_objv2_minival5k.json"),
    }
    img_folder, ann_file = PATHS[image_set]

    from rfdetr.datasets.kornia_transforms import is_gpu_postprocess, resolve_backend_for_build

    # These geometry values are model/config dependent, so direct calls must fail
    # instead of silently falling back to the transform factories' generic defaults.
    square_resize_div_64 = args.square_resize_div_64
    multi_scale = args.multi_scale
    expanded_scales = args.expanded_scales
    do_random_resize_via_padding = args.do_random_resize_via_padding
    patch_size = args.patch_size
    num_windows = args.num_windows
    scale_jitter = getattr(args, "scale_jitter", True)
    augmentation_backend = getattr(args, "augmentation_backend", "cpu")
    resolved_backend = resolve_backend_for_build(augmentation_backend)
    gpu_postprocess = is_gpu_postprocess(resolved_backend)

    if gpu_postprocess:
        logger.warning(
            "O365 dataset does not support custom aug_config with the Kornia GPU augmentation backend; "
            "Albumentations augmentation is skipped and normalization runs on GPU. "
            "Pass augmentation_backend='cpu' or 'albumentations' for full CPU augmentation pipeline with O365."
        )

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                scale_jitter=scale_jitter,
                gpu_postprocess=gpu_postprocess,
            ),
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                scale_jitter=scale_jitter,
                gpu_postprocess=gpu_postprocess,
            ),
        )
    return dataset


def build_o365(image_set: str, args: Any, resolution: int) -> CocoDetection:
    """Build a supported Object365 detection split.

    Args:
        image_set: Object365 split, either ``"train"`` or ``"val"``.
        args: Dataset and transform configuration namespace accepted by
            :func:`build_o365_raw`.
        resolution: Target square resolution in pixels.

    Returns:
        The configured Object365 detection dataset.

    Raises:
        AttributeError: If a required dataset or transform option is absent.
        ValueError: If ``image_set`` is not ``"train"`` or ``"val"``.
    """
    if image_set == "train":
        train_ds = build_o365_raw("train", args, resolution=resolution)
        return train_ds
    if image_set == "val":
        val_ds = build_o365_raw("val", args, resolution=resolution)
        return val_ds
    raise ValueError(f"Unknown image_set: {image_set}")
