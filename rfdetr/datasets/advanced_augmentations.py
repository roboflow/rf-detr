# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Advanced Data Augmentation Pipeline for RF-DETR
This module implements advanced augmentation techniques including Mosaic, MixUp,
and Copy-Paste augmentations specifically designed for DETR-based models.
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, List, Tuple, Optional, Union
import cv2

from rfdetr.util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class MosaicAugmentation:
    """
    Mosaic augmentation adapted for DETR models.
    Combines 4 images into one mosaic image.
    """
    
    def __init__(self, prob: float = 0.5):
        """
        Initialize Mosaic augmentation.
        
        Args:
            prob: Probability of applying mosaic augmentation
        """
        self.prob = prob
    
    def __call__(
        self, 
        images: List[Image.Image], 
        targets: List[Dict]
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply mosaic augmentation to a batch of images.
        
        Args:
            images: List of 4 PIL images
            targets: List of corresponding target dictionaries
            
        Returns:
            mosaic_image: Mosaic augmented image
            mosaic_target: Combined target dictionary
        """
        if random.random() > self.prob or len(images) < 4:
            return images[0], targets[0]
        
        # Get dimensions
        img0 = images[0]
        w, h = img0.size
        
        # Create mosaic canvas
        mosaic_img = Image.new('RGB', (w * 2, h * 2))
        
        # Calculate split points
        xc = random.randint(int(w * 0.5), int(w * 1.5))
        yc = random.randint(int(h * 0.5), int(h * 1.5))
        
        # Positions for 4 images
        positions = [
            (0, 0, xc, yc),      # top-left
            (xc, 0, w * 2, yc),  # top-right
            (0, yc, xc, h * 2),  # bottom-left
            (xc, yc, w * 2, h * 2)  # bottom-right
        ]
        
        mosaic_target = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
            'orig_size': torch.tensor([h * 2, w * 2]),
            'size': torch.tensor([h * 2, w * 2])
        }
        
        # Process each image
        for i, (img, target) in enumerate(zip(images[:4], targets[:4])):
            x1, y1, x2, y2 = positions[i]
            
            # Resize and paste image
            img_resized = img.resize((x2 - x1, y2 - y1), Image.BILINEAR)
            mosaic_img.paste(img_resized, (x1, y1))
            
            # Adjust bounding boxes
            if 'boxes' in target:
                boxes = target['boxes']
                # Convert from normalized to absolute coordinates
                orig_w, orig_h = img.size
                boxes_abs = boxes.clone()
                boxes_abs[:, [0, 2]] *= orig_w
                boxes_abs[:, [1, 3]] *= orig_h
                
                # Adjust for mosaic position
                boxes_abs[:, [0, 2]] += x1
                boxes_abs[:, [1, 3]] += y1
                
                # Normalize to mosaic canvas
                boxes_abs[:, [0, 2]] /= (w * 2)
                boxes_abs[:, [1, 3]] /= (h * 2)
                
                # Filter boxes that are within bounds
                mask = (boxes_abs[:, 0] < 1) & (boxes_abs[:, 2] > 0) & \
                       (boxes_abs[:, 1] < 1) & (boxes_abs[:, 3] > 0)
                
                if mask.any():
                    mosaic_target['boxes'].extend(boxes_abs[mask])
                    mosaic_target['labels'].extend(target['labels'][mask])
                    
                    # Calculate area
                    area = (boxes_abs[mask, 2] - boxes_abs[mask, 0]) * \
                           (boxes_abs[mask, 3] - boxes_abs[mask, 1])
                    mosaic_target['area'].extend(area)
                    
                    if 'iscrowd' in target:
                        mosaic_target['iscrowd'].extend(
                            [target['iscrowd'][j] for j in range(len(target['iscrowd'])) if mask[j]]
                        )
                    else:
                        mosaic_target['iscrowd'].extend([0] * mask.sum().item())
        
        # Convert to tensors
        if mosaic_target['boxes']:
            mosaic_target['boxes'] = torch.stack(mosaic_target['boxes'])
            mosaic_target['labels'] = torch.stack(mosaic_target['labels'])
            mosaic_target['area'] = torch.stack(mosaic_target['area'])
            mosaic_target['iscrowd'] = torch.tensor(mosaic_target['iscrowd'])
        else:
            # Empty target
            mosaic_target['boxes'] = torch.empty((0, 4))
            mosaic_target['labels'] = torch.empty((0,), dtype=torch.long)
            mosaic_target['area'] = torch.empty((0,))
            mosaic_target['iscrowd'] = torch.empty((0,), dtype=torch.long)
        
        return mosaic_img, mosaic_target


class MixUpAugmentation:
    """
    MixUp augmentation for object detection.
    Blends two images and their targets.
    """
    
    def __init__(self, prob: float = 0.5, alpha: float = 1.0):
        """
        Initialize MixUp augmentation.
        
        Args:
            prob: Probability of applying mixup
            alpha: Alpha parameter for beta distribution
        """
        self.prob = prob
        self.alpha = alpha
    
    def __call__(
        self, 
        img1: Image.Image, 
        target1: Dict,
        img2: Image.Image, 
        target2: Dict
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply mixup augmentation to two images.
        
        Args:
            img1: First image
            target1: First target
            img2: Second image  
            target2: Second target
            
        Returns:
            mixed_image: Mixup augmented image
            mixed_target: Combined target dictionary
        """
        if random.random() > self.prob:
            return img1, target1
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Convert to tensors
        img1_tensor = F.to_tensor(img1)
        img2_tensor = F.to_tensor(img2)
        
        # Ensure same size
        if img1_tensor.shape != img2_tensor.shape:
            img2_tensor = F.resize(img2_tensor, img1_tensor.shape[-2:])
        
        # Mix images
        mixed_tensor = lam * img1_tensor + (1 - lam) * img2_tensor
        mixed_image = F.to_pil_image(mixed_tensor)
        
        # Combine targets
        mixed_target = {
            'boxes': torch.cat([target1['boxes'], target2['boxes']], dim=0),
            'labels': torch.cat([target1['labels'], target2['labels']], dim=0),
            'area': torch.cat([target1['area'], target2['area']], dim=0),
            'iscrowd': torch.cat([target1.get('iscrowd', torch.zeros_like(target1['labels'])), 
                                 target2.get('iscrowd', torch.zeros_like(target2['labels']))], dim=0),
            'orig_size': target1['orig_size'],
            'size': target1['size']
        }
        
        return mixed_image, mixed_target


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for instance segmentation.
    Copies objects from one image to another.
    """
    
    def __init__(self, prob: float = 0.5, max_objects: int = 5):
        """
        Initialize Copy-Paste augmentation.
        
        Args:
            prob: Probability of applying copy-paste
            max_objects: Maximum number of objects to copy
        """
        self.prob = prob
        self.max_objects = max_objects
    
    def __call__(
        self, 
        img1: Image.Image, 
        target1: Dict,
        img2: Image.Image, 
        target2: Dict
    ) -> Tuple[Image.Image, Dict]:
        """
        Apply copy-paste augmentation.
        
        Args:
            img1: Target image (where objects will be pasted)
            target1: Target image target
            img2: Source image (where objects will be copied from)
            target2: Source image target
            
        Returns:
            result_image: Image with pasted objects
            result_target: Updated target dictionary
        """
        if random.random() > self.prob or 'boxes' not in target2:
            return img1, target1
        
        # Convert to numpy arrays
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        # Select random objects to copy
        num_objects = min(len(target2['boxes']), self.max_objects)
        if num_objects == 0:
            return img1, target1
        
        indices = random.sample(range(len(target2['boxes'])), num_objects)
        
        result_target = target1.copy()
        pasted_boxes = []
        pasted_labels = []
        pasted_areas = []
        
        for idx in indices:
            box = target2['boxes'][idx]
            label = target2['labels'][idx]
            
            # Convert to absolute coordinates
            h1, w1 = img1_np.shape[:2]
            h2, w2 = img2_np.shape[:2]
            
            box_abs = box.clone()
            box_abs[[0, 2]] *= w2
            box_abs[[1, 3]] *= h2
            
            x1, y1, x2, y2 = box_abs.int().tolist()
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w2, x2), min(h2, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract object
            obj_mask = np.zeros((h2, w2), dtype=np.uint8)
            obj_mask[y1:y2, x1:x2] = 255
            
            # Find random position in target image
            obj_w, obj_h = x2 - x1, y2 - y1
            if obj_w > w1 or obj_h > h1:
                continue
            
            paste_x = random.randint(0, w1 - obj_w)
            paste_y = random.randint(0, h1 - obj_h)
            
            # Simple paste (without sophisticated blending)
            try:
                obj_region = img2_np[y1:y2, x1:x2]
                img1_np[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj_region
                
                # Add to target
                new_box = torch.tensor([
                    paste_x / w1,
                    paste_y / h1,
                    (paste_x + obj_w) / w1,
                    (paste_y + obj_h) / h1
                ])
                
                pasted_boxes.append(new_box)
                pasted_labels.append(label)
                pasted_areas.append((new_box[2] - new_box[0]) * (new_box[3] - new_box[1]))
                
            except Exception:
                continue
        
        # Update target
        if pasted_boxes:
            result_target['boxes'] = torch.cat([
                target1['boxes'], 
                torch.stack(pasted_boxes)
            ], dim=0)
            result_target['labels'] = torch.cat([
                target1['labels'], 
                torch.stack(pasted_labels)
            ], dim=0)
            result_target['area'] = torch.cat([
                target1['area'], 
                torch.stack(pasted_areas)
            ], dim=0)
            
            iscrowd1 = target1.get('iscrowd', torch.zeros_like(target1['labels']))
            result_target['iscrowd'] = torch.cat([
                iscrowd1, 
                torch.zeros(len(pasted_labels), dtype=torch.long)
            ], dim=0)
        
        result_image = Image.fromarray(img1_np)
        return result_image, result_target


class AdvancedAugmentationPipeline:
    """
    Comprehensive augmentation pipeline combining multiple techniques.
    """
    
    def __init__(
        self,
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.3,
        copypaste_prob: float = 0.3,
        color_jitter: float = 0.2,
        gaussian_blur: float = 0.1,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize advanced augmentation pipeline.
        
        Args:
            mosaic_prob: Probability of mosaic augmentation
            mixup_prob: Probability of mixup augmentation
            copypaste_prob: Probability of copy-paste augmentation
            color_jitter: Color jitter strength
            gaussian_blur: Gaussian blur probability
            normalize_mean: Normalization mean values
            normalize_std: Normalization std values
        """
        self.mosaic = MosaicAugmentation(prob=mosaic_prob)
        self.mixup = MixUpAugmentation(prob=mixup_prob)
        self.copypaste = CopyPasteAugmentation(prob=copypaste_prob)
        
        self.color_jitter = T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter * 0.1
        )
        
        self.gaussian_blur_prob = gaussian_blur
        self.normalize = T.Normalize(mean=normalize_mean, std=normalize_std)
    
    def __call__(
        self, 
        image: Image.Image, 
        target: Dict,
        additional_images: Optional[List[Image.Image]] = None,
        additional_targets: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply advanced augmentation pipeline.
        
        Args:
            image: Input image
            target: Target dictionary
            additional_images: Additional images for complex augmentations
            additional_targets: Additional targets for complex augmentations
            
        Returns:
            augmented_tensor: Augmented image tensor
            augmented_target: Augmented target dictionary
        """
        # Apply color jitter
        if random.random() < 0.8:
            image = self.color_jitter(image)
        
        # Apply Gaussian blur
        if random.random() < self.gaussian_blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        
        # Apply complex augmentations if additional data is available
        if additional_images and additional_targets:
            # Try mosaic augmentation
            if len(additional_images) >= 3:
                mosaic_images = [image] + additional_images[:3]
                mosaic_targets = [target] + additional_targets[:3]
                image, target = self.mosaic(mosaic_images, mosaic_targets)
            
            # Try mixup augmentation
            elif len(additional_images) >= 1:
                image, target = self.mixup(image, target, additional_images[0], additional_targets[0])
            
            # Try copy-paste augmentation
            elif len(additional_images) >= 1:
                image, target = self.copypaste(image, target, additional_images[0], additional_targets[0])
        
        # Convert to tensor and normalize
        tensor = F.to_tensor(image)
        tensor = self.normalize(tensor)
        
        return tensor, target


def create_advanced_transforms(
    train: bool = True,
    image_size: Tuple[int, int] = (640, 640),
    **augmentation_kwargs
) -> T.Compose:
    """
    Create advanced transformation pipeline.
    
    Args:
        train: Whether to apply training augmentations
        image_size: Target image size
        **augmentation_kwargs: Additional augmentation parameters
        
    Returns:
        transforms: Composed transformation pipeline
    """
    transforms_list = []
    
    if train:
        # Advanced augmentations
        advanced_aug = AdvancedAugmentationPipeline(**augmentation_kwargs)
        
        def advanced_aug_wrapper(sample):
            image = sample['image']
            target = sample['target']
            
            # Handle additional images if available
            additional_images = sample.get('additional_images')
            additional_targets = sample.get('additional_targets')
            
            tensor, target = advanced_aug(image, target, additional_images, additional_targets)
            
            return {
                'image': tensor,
                'target': target
            }
        
        transforms_list.append(advanced_aug_wrapper)
    
    # Resize
    transforms_list.append(T.Resize(image_size))
    
    # Convert to tensor (if not already done)
    if not train:
        transforms_list.append(T.ToTensor())
        transforms_list.append(T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ))
    
    return T.Compose(transforms_list)
