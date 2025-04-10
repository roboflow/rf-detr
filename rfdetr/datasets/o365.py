# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""Dataset file for Object365."""
from pathlib import Path

from .coco import (
    CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def build_o365_raw(image_set, args, resolution):
    root = Path(args.coco_path)
    PATHS = {
        "train": (root, root / 'zhiyuan_objv2_train_val_wo_5k.json'),
        "val": (root, root / 'zhiyuan_objv2_minival5k.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(image_set, resolution, multi_scale=args.multi_scale, expanded_scales=args.expanded_scales))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, resolution, multi_scale=args.multi_scale, expanded_scales=args.expanded_scales))
    return dataset


def build_o365(image_set, args, resolution):
    if image_set == 'train':
        train_ds = build_o365_raw('train', args, resolution=resolution)
        return train_ds
    if image_set == 'val':
        val_ds = build_o365_raw('val', args, resolution=resolution)
        return val_ds
    raise ValueError('Unknown image_set: {}'.format(image_set))