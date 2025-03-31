"""
YOLO dataset loader.
Optimized for large datasets to avoid the memory overhead of converting beforehand.
"""
from pathlib import Path
import os
import yaml
import glob
from PIL import Image
import numpy as np
from collections import defaultdict

import torch
import torch.utils.data

from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.util.files import load_txt_lines

REQUIRED_YOLO_YAML_FILE = "data.yaml"
REQUIRED_SPLIT_DIRS = ["train", "valid"]
REQUIRED_DATA_SUBDIRS = ["images", "labels"]


def is_valid_yolo_dataset(dataset_dir: str) -> bool:
    """
    Checks if the specified dataset directory is in yolo format.

    We accept a dataset to be in yolo format if the following conditions are met:
    - The dataset_dir contains a data.yaml file
    - The dataset_dir contains "train" and "valid" subdirectories, each containing "images" and "labels" subdirectories
    - The "test" subdirectory is optional

    Returns a boolean indicating whether the dataset is in correct yolo format.
    """
    contains_required_data_yaml = os.path.exists(os.path.join(dataset_dir, REQUIRED_YOLO_YAML_FILE))
    contains_required_split_dirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir)) for split_dir in REQUIRED_SPLIT_DIRS
    )
    contains_required_data_subdirs = all(
        os.path.exists(os.path.join(dataset_dir, split_dir, data_subdir))
        for split_dir in REQUIRED_SPLIT_DIRS
        for data_subdir in REQUIRED_DATA_SUBDIRS
    )
    return contains_required_data_yaml and contains_required_split_dirs and contains_required_data_subdirs


def build_yolo(image_set, args, resolution):
    """Build YOLO dataset"""
    root = Path(args.dataset_dir)
    print(image_set)
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "valid" / "images", root / "valid" / "labels"),
        "test": (root / "test" / "images", root / "test" / "labels"),
    }
    
    img_folder, labels_folder = PATHS[image_set.split("_")[0]]
    data_yaml_path = root / "data.yaml"
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
    
    if square_resize_div_64:
        dataset = YoloDetection(
            img_folder, 
            labels_folder,
            data_yaml_path,
            transforms=make_coco_transforms_square_div_64(
                image_set, 
                resolution, 
                multi_scale=args.multi_scale, 
                expanded_scales=args.expanded_scales
            )
        )
    else:
        dataset = YoloDetection(
            img_folder, 
            labels_folder,
            data_yaml_path,
            transforms=make_coco_transforms(
                image_set, 
                resolution, 
                multi_scale=args.multi_scale, 
                expanded_scales=args.expanded_scales
            )
        )
    
    return dataset 


class YoloDetection(torch.utils.data.Dataset):
    """Dataset for YOLO format annotations"""
    def __init__(self, img_folder, labels_folder, data_yaml_path, transforms=None):
        super(YoloDetection, self).__init__()
        self.img_folder = img_folder
        self.labels_folder = labels_folder
        self.transforms = transforms
        self.img_files = sorted(glob.glob(os.path.join(img_folder, '*.jpg')) + 
                        glob.glob(os.path.join(img_folder, '*.jpeg')) + 
                        glob.glob(os.path.join(img_folder, '*.png')))
        
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.class_names = data.get('names', [])
            
        print(f"Loaded {len(self.class_names)} classes from YOLO dataset: {self.class_names}")
            
        self.class_to_coco_id = {i: i for i in range(len(self.class_names))}
        
        self.ids = [i+1 for i in list(range(len(self.img_files)))]
        
        self._coco = None
        
    def __len__(self):
        return len(self.img_files)

    def _parse_yolo_annotations(self, lines: list[str], width: int, height: int) -> tuple[list, list]:
        boxes = []
        labels = []
        for line in lines:
            data = line.strip().split()
            if len(data) == 5: 
                class_id = int(data[0])
                
                if class_id < 0 or class_id >= len(self.class_names):
                    print(f"Warning: Skipping invalid class ID {class_id}")
                    continue
                    
                x_center, y_center, width, height = map(float, data[1:5])
                
                if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                    print(f"Warning: Skipping invalid coordinates {x_center}, {y_center}, {width}, {height}. (Not normalized)")
                    continue
                
                x1 = (x_center - width / 2) * width
                y1 = (y_center - height / 2) * height
                x2 = (x_center + width / 2) * width
                y2 = (y_center + height / 2) * height
                
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_coco_id[class_id])
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = int(idx)
        img_path = self.img_files[idx]
        image_id = self.ids[idx]

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_folder, base_name + '.txt')
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        target = {}
        target["image_id"] = torch.tensor([image_id])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        boxes, labels = self._parse_yolo_annotations(load_txt_lines(label_path))
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # No objects in this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros(0)
        target["iscrowd"] = torch.zeros_like(labels, dtype=torch.int64)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    @property
    def coco(self):
        """
        Return a COCO-like API object for compatibility with pycocotools evaluation
        """
        if self._coco is None:
            self._coco = CocoLikeAPI(self)
        return self._coco


class CocoLikeAPI:
    """
    A COCO-like API for compatibility with pycocotools evaluation.
    This simulates the COCO API used for evaluation.
    """
    def __init__(self, dataset : YoloDetection):
        self.orig_dataset = dataset
        self.cats = self._create_category_mapping()
        self.imgs = self._create_image_mapping()
        self.anns = self._create_annotation_mapping()
        
        # Create lookup dictionaries
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)
        
        for ann in self.anns.values():
            self.imgToAnns[ann['image_id']].append(ann)
            self.catToImgs[ann['category_id']].append(ann['image_id'])
            
        # Create the dataset structure that COCO.loadRes expects
        self.dataset = {
            'images': self.imgs,
            'annotations': list(self.anns.values()),
            'categories': list(self.cats.values()),
        }
        
    def _create_category_mapping(self):
        """Create a category mapping similar to COCO format"""
        cats = {}
        for idx, name in enumerate(self.orig_dataset.class_names):
            cat_id = self.orig_dataset.class_to_coco_id[idx]
            cats[cat_id] = {
                'id': cat_id,
                'name': name,
                'supercategory': 'none'
            }
        return cats
        
    def _create_image_mapping(self):
        """Create an image mapping similar to COCO format"""
        imgs = []
        for idx, img_path in enumerate(self.orig_dataset.img_files):
            img = Image.open(img_path)
            width, height = img.size
            imgs.append({
                'id': self.orig_dataset.ids[idx],  # Use the same ID as the original dataset
                'file_name': os.path.basename(img_path),
                'width': width,
                'height': height
            })
        return imgs
        
    def _create_annotation_mapping(self):
        """Create an annotation mapping similar to COCO format"""
        anns = {}
        ann_id = 0
        
        for idx, img_path in enumerate(self.orig_dataset.img_files):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.orig_dataset.labels_folder, base_name + '.txt')
            
            # Skip if no label file exists
            if not os.path.exists(label_path):
                continue
                
            img = Image.open(img_path)
            width, height = img.size
            
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:  # class_id, x_center, y_center, width, height
                        class_id = int(data[0])
                        # YOLO format: [class_id, x_center, y_center, width, height] (normalized [0,1])
                        x_center, y_center, box_width, box_height = map(float, data[1:5])
                        
                        # Convert to COCO format (x, y, width, height) where x,y is top-left corner
                        x = (x_center - box_width / 2) * width
                        y = (y_center - box_height / 2) * height
                        w = box_width * width
                        h = box_height * height
                        
                        # COCO annotation
                        anns[ann_id] = {
                            'id': ann_id,
                            'image_id': self.orig_dataset.ids[idx],  # Use the same ID as the original dataset
                            'category_id': self.orig_dataset.class_to_coco_id[class_id],
                            'bbox': [x, y, w, h],
                            'area': w * h,
                            'iscrowd': 0
                        }
                        ann_id += 1
        
        return anns
    
    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """Get annotation IDs matching the given filter conditions"""
        anns = self.anns.values()
        
        if imgIds is not None:
            if not isinstance(imgIds, list):
                imgIds = [imgIds]
            anns = [ann for ann in anns if ann['image_id'] in imgIds]
            
        if catIds is not None:
            if not isinstance(catIds, list):
                catIds = [catIds]
            anns = [ann for ann in anns if ann['category_id'] in catIds]
            
        if areaRng is not None:
            anns = [ann for ann in anns if areaRng[0] <= ann['area'] <= areaRng[1]]
            
        if iscrowd is not None:
            anns = [ann for ann in anns if ann['iscrowd'] == iscrowd]
            
        return [ann['id'] for ann in anns]
    
    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Get category IDs matching the given filter conditions"""
        cats = self.cats.values()
        
        if catNms is not None:
            if not isinstance(catNms, list):
                catNms = [catNms]
            cats = [cat for cat in cats if cat['name'] in catNms]
            
        if supNms is not None:
            if not isinstance(supNms, list):
                supNms = [supNms]
            cats = [cat for cat in cats if cat['supercategory'] in supNms]
            
        if catIds is not None:
            if not isinstance(catIds, list):
                catIds = [catIds]
            cats = [cat for cat in cats if cat['id'] in catIds]
            
        return [cat['id'] for cat in cats]
    
    def getImgIds(self, imgIds=None, catIds=None):
        """Get image IDs matching the given filter conditions"""
        imgs = self.imgs
        
        if imgIds is not None:
            if not isinstance(imgIds, list):
                imgIds = [imgIds]
            imgs = [img for img in imgs if img['id'] in imgIds]
            
        if catIds is not None:
            if not isinstance(catIds, list):
                catIds = [catIds]
            # Use cached mapping for performance
            img_ids = set()
            for cat_id in catIds:
                img_ids.update(self.catToImgs[cat_id])
            imgs = [img for img in imgs if img['id'] in img_ids]
            
        return [img['id'] for img in imgs]
    
    def loadAnns(self, ids):
        """Load annotations with the specified IDs"""
        if isinstance(ids, int):
            ids = [ids]
        return [self.anns[id] for id in ids if id in self.anns]
    
    def loadCats(self, ids):
        """Load categories with the specified IDs"""
        if isinstance(ids, int):
            ids = [ids]
        return [self.cats[id] for id in ids if id in self.cats]
    
    def loadImgs(self, ids):
        """Load images with the specified IDs"""
        if isinstance(ids, int):
            ids = [ids]
        return [self.imgs[id] for id in ids if id in self.imgs]
    