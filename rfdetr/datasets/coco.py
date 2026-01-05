# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import pycocotools.mask as coco_mask

import rfdetr.datasets.transforms as T


def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, include_masks=False, include_attributes=False, attribute_names=None, num_attributes=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        self.include_attributes = include_attributes
        self.attribute_names = attribute_names
        self.num_attributes = num_attributes
        self.prepare = ConvertCoco(include_masks=include_masks, include_attributes=include_attributes, attribute_names=attribute_names, num_attributes=num_attributes)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __init__(self, include_masks=False, include_attributes=False, attribute_names=None, num_attributes=None):
        self.include_masks = include_masks
        self.include_attributes = include_attributes
        self.attribute_names = attribute_names
        self.num_attributes = num_attributes

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        # optional: parse attributes
        if self.include_attributes:
            attrs_list = []
            attr_len = None
            for obj in anno:
                attrs = obj.get("attributes", None)
                if attrs is None:
                    continue
                if isinstance(attrs, list):
                    vec = torch.as_tensor(attrs, dtype=torch.float32)
                elif isinstance(attrs, dict):
                    if self.attribute_names is not None:
                        vec = torch.as_tensor([float(attrs.get(k, 0.0)) for k in self.attribute_names], dtype=torch.float32)
                    else:
                        # stable order by key for deterministic behavior
                        keys = sorted(attrs.keys())
                        vec = torch.as_tensor([float(attrs[k]) for k in keys], dtype=torch.float32)
                else:
                    continue
                if self.num_attributes is not None:
                    # pad or truncate to desired length
                    if vec.numel() < self.num_attributes:
                        pad = torch.zeros(self.num_attributes - vec.numel(), dtype=torch.float32)
                        vec = torch.cat([vec, pad], dim=0)
                    elif vec.numel() > self.num_attributes:
                        vec = vec[:self.num_attributes]
                attrs_list.append(vec)

            if len(attrs_list) > 0:
                # make tensor and filter with keep mask, aligning with boxes/classes
                # fill missing ones with zeros to match anno length
                if len(attrs_list) != len(anno):
                    # create zero vectors for missing
                    if self.attribute_names is not None:
                        A = len(self.attribute_names)
                    elif self.num_attributes is not None:
                        A = int(self.num_attributes)
                    else:
                        A = int(attrs_list[0].numel())
                    full_attrs = []
                    ai = 0
                    for obj in anno:
                        a = obj.get("attributes", None)
                        if a is None:
                            full_attrs.append(torch.zeros(A, dtype=torch.float32))
                        else:
                            full_attrs.append(attrs_list[ai])
                            ai += 1
                    attrs_tensor = torch.stack(full_attrs, dim=0)
                else:
                    attrs_tensor = torch.stack(attrs_list, dim=0)
                if attrs_tensor.numel() > 0 and keep.numel() > 0:
                    target["attributes"] = attrs_tensor[keep]
                else:
                    target["attributes"] = torch.zeros((0, attrs_tensor.shape[-1] if attrs_tensor.ndim == 2 else 0), dtype=torch.float32)
            else:
                if self.num_attributes is not None:
                    target["attributes"] = torch.zeros((0, int(self.num_attributes)), dtype=torch.float32)

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([resolution], max_size=1333),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    scales = [resolution]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'test':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root /  "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    
    include_attributes = hasattr(args, 'num_attributes') and args.num_attributes is not None and int(args.num_attributes) > 0
    include_masks = getattr(args, 'segmentation_head', False)

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks, include_attributes=include_attributes, attribute_names=getattr(args, 'attribute_names', None), num_attributes=getattr(args, 'num_attributes', None))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks, include_attributes=include_attributes, attribute_names=getattr(args, 'attribute_names', None), num_attributes=getattr(args, 'num_attributes', None))
    return dataset

def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
    
    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False
    include_attributes = hasattr(args, 'num_attributes') and args.num_attributes is not None and int(args.num_attributes) > 0

    
    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks, include_attributes=include_attributes, attribute_names=getattr(args, 'attribute_names', None), num_attributes=getattr(args, 'num_attributes', None))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks, include_attributes=include_attributes, attribute_names=getattr(args, 'attribute_names', None), num_attributes=getattr(args, 'num_attributes', None))
    return dataset
