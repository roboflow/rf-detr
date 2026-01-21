# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
import supervision as sv

from rfdetr.datasets.coco import (
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)


class ConvertYolo:
    """
    Converts supervision Detections to the target dict format expected by RF-DETR.
    
    Args:
        include_masks: whether to include segmentation masks
    
    Examples:
        >>> import numpy as np
        >>> import supervision as sv
        >>> from PIL import Image
        >>> # Create a sample image and target
        >>> image = Image.new("RGB", (100, 100))
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[10, 20, 30, 40]]),
        ...     class_id=np.array([0])
        ... )
        >>> target = {"image_id": 0, "detections": detections}
        >>> # Create converter
        >>> converter = ConvertYolo(include_masks=False)
        >>> # Call converter
        >>> img, result = converter(image, target)
        >>> sorted(result.keys())
        ['area', 'boxes', 'image_id', 'iscrowd', 'labels', 'orig_size', 'size']
        >>> result["boxes"].shape
        torch.Size([1, 4])
        >>> result["labels"].tolist()
        [0]
        >>> result["image_id"].tolist()
        [0]
    """
    
    def __init__(self, include_masks: bool = False):
        self.include_masks = include_masks
    
    def __call__(self, image: Image.Image, target: dict) -> tuple:
        """
        Convert image and YOLO detections to RF-DETR format.
        
        Args:
            image: PIL Image
            target: dict with 'image_id' and 'detections' (sv.Detections)
            
        Returns:
            tuple of (image, target_dict)
        """
        w, h = image.size
        
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        
        detections = target["detections"]
        
        if len(detections) > 0:
            boxes = torch.from_numpy(detections.xyxy).to(torch.float32)
            classes = torch.from_numpy(detections.class_id).to(torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)

        # clamp and filter
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target_out = {}
        target_out["boxes"] = boxes
        target_out["labels"] = classes
        target_out["image_id"] = image_id

        # compute area after clamp
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target_out["area"] = area

        iscrowd = torch.zeros((classes.shape[0],), dtype=torch.int64)
        target_out["iscrowd"] = iscrowd

        if self.include_masks:
            if detections.mask is not None and len(detections.mask) > 0:
                masks = torch.from_numpy(detections.mask[keep.numpy()]).to(torch.uint8)
                target_out["masks"] = masks
            else:
                target_out["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            
            target_out["masks"] = target_out["masks"].bool()

        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_out["size"] = torch.as_tensor([int(h), int(w)])

        return image, target_out


class YoloDetection(VisionDataset):
    """
    YOLO format dataset using supervision.DetectionDataset.from_yolo().
    
    This class provides a VisionDataset interface compatible with RF-DETR training,
    matching the API of CocoDetection.
    
    Args:
        img_folder: Path to the directory containing images
        lb_folder: Path to the directory containing YOLO annotation .txt files
        data_file: Path to data.yaml file containing class names and dataset info
        transforms: Optional transforms to apply to images and targets
        include_masks: Whether to load segmentation masks (for YOLO segmentation format)
    """
    
    def __init__(
        self,
        img_folder: str,
        lb_folder: str,
        data_file: str,
        transforms=None,
        include_masks: bool = False,
    ):
        super(YoloDetection, self).__init__(img_folder)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertYolo(include_masks=include_masks)
        
        # Load dataset using supervision's from_yolo method
        self.sv_dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=img_folder,
            annotations_directory_path=lb_folder,
            data_yaml_path=data_file,
            force_masks=include_masks,
        )
        
        self.classes = self.sv_dataset.classes
        self.ids = list(range(len(self.sv_dataset)))

    def __len__(self) -> int:
        return len(self.sv_dataset)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image_path, cv2_image, detections = self.sv_dataset[idx]
        
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2_image[:, :, ::-1]
        img = Image.fromarray(rgb_image)
        
        target = {'image_id': image_id, 'detections': detections}
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_roboflow_from_yolo(image_set, args, resolution):
    """Build a Roboflow YOLO-format dataset.
    
    This uses Roboflow's standard YOLO directory structure
    (train/valid/test folders with images/ and labels/ subdirectories).
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    
    # YOLO format uses images/ and labels/ subdirectories
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "valid" / "images", root / "valid" / "labels"),
        "test": (root / "test" / "images", root / "test" / "labels"),
    }
    
    data_file = root / "data.yaml"
    img_folder, lb_folder = PATHS[image_set.split("_")[0]]
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    multi_scale = args.multi_scale
    expanded_scales = args.expanded_scales
    do_random_resize_via_padding = args.do_random_resize_via_padding
    patch_size = args.patch_size
    num_windows = args.num_windows

    if square_resize_div_64:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows
            ),
            include_masks=include_masks
        )
    else:
        dataset = YoloDetection(
            img_folder=str(img_folder),
            lb_folder=str(lb_folder),
            data_file=str(data_file),
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows
            ),
            include_masks=include_masks
        )
    return dataset
