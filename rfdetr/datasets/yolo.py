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
import supervision as sv
from supervision.utils.file import read_yaml_file, read_txt_file, list_files_with_extensions

import torch
import torch.utils.data

from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64

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
        dataset = YOLODataset(
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
        dataset = YOLODataset(
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


def _parse_yolo_annotations(lines: list[str], resolution_wh: tuple[int, int], class_names: list[str]) -> tuple[list, list]:
    boxes = []
    labels = []
    for line in lines:
        data = line.strip().split()
        if len(data) == 5: 
            class_id = int(data[0])
            
            if class_id < 0 or class_id >= len(class_names):
                print(f"Warning: Skipping invalid class ID {class_id}")
                continue
                
            x_center, y_center, width, height = map(float, data[1:5])
            
            if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                print(f"Warning: Skipping invalid coordinates {x_center}, {y_center}, {width}, {height}. (Not normalized)")
                continue
            
            x1 = (x_center - width / 2) * resolution_wh[0]
            y1 = (y_center - height / 2) * resolution_wh[1]
            x2 = (x_center + width / 2) * resolution_wh[0]
            y2 = (y_center + height / 2) * resolution_wh[1]
            
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)


def match_image_label_pairs(image_paths, label_paths):
    """
    Matches image paths with their corresponding label paths.
    
    Args:
        image_paths: List of paths to image files
        label_paths: List of paths to label files
        
    Returns:
        Tuple of (matched_image_paths, matched_label_paths) with paired files in sorted order
    """
    label_dict = {}
    label_basenames = set()
    for label_path in label_paths:
        base_name = os.path.splitext(os.path.basename(label_path))[0]
        label_dict[base_name] = label_path
        label_basenames.add(base_name)
    
    image_count = len(image_paths)
    label_count = len(label_paths)
    skipped_images = []
    unused_labels = set(label_basenames)
    
    matched_pairs = []
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if base_name in label_dict:
            matched_pairs.append((image_path, label_dict[base_name]))
            unused_labels.discard(base_name)
        else:
            skipped_images.append(os.path.basename(image_path))
    
    matched_pairs.sort(key=lambda x: x[0])
    
    matched_image_paths, matched_label_paths = zip(*matched_pairs) if matched_pairs else ([], [])
    
    if skipped_images:
        print(f"WARNING: Skipped {len(skipped_images)} images without matching labels")
        if len(skipped_images) <= 10:
            print(f"  Skipped images: {', '.join(skipped_images)}")
        else:
            print(f"  First 10 skipped images: {', '.join(skipped_images[:10])}...")
    
    if unused_labels:
        print(f"WARNING: Found {len(unused_labels)} label files without matching images")
        if len(unused_labels) <= 10:
            print(f"  Unused labels: {', '.join(unused_labels)}")
        else:
            print(f"  First 10 unused labels: {', '.join(list(unused_labels)[:10])}...")
    
    print(f"Matching complete: {len(matched_pairs)}/{image_count} images matched with labels ({len(matched_pairs)}/{label_count} labels used)")
    
    return list(matched_image_paths), list(matched_label_paths)


class YOLODataset(torch.utils.data.Dataset):
    """Dataset for YOLO format annotations"""
    def __init__(self, images_directory_path: str, annotations_directory_path: str, data_yaml_path: str, transforms=None):
        super(YOLODataset, self).__init__()
        self.images_directory_path = images_directory_path
        self.annotations_directory_path = annotations_directory_path
        self.transforms = transforms
        
        image_paths = list_files_with_extensions(
            directory=images_directory_path,
            extensions=["jpg", "jpeg", "png"],
        )
        
        label_paths = list_files_with_extensions(
            directory=annotations_directory_path,
            extensions=["txt"],
        )
        
        self.image_paths, self.label_paths = match_image_label_pairs(
            image_paths=image_paths, label_paths=label_paths)
        
        data = read_yaml_file(data_yaml_path)
        self.class_names = data.get('names', [])
            
        print(f"Loaded {len(self.class_names)} classes from YOLO dataset: {self.class_names}")
        print(f"Found {len(self.image_paths)} valid image-label pairs.")
            
        self.ids = [i+1 for i in list(range(len(self.image_paths)))]
        
        self._coco = None
        
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = int(idx)
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image_id = self.ids[idx]
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        target = {}
        target["image_id"] = torch.tensor([image_id])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        label_lines = read_txt_file(label_path)
        boxes, labels = _parse_yolo_annotations(label_lines, (w, h), self.class_names)
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
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
    def __init__(self, dataset : YOLODataset):
        self.orig_dataset = dataset
        self.cats = self._create_category_mapping()
        self.imgs = self._create_image_mapping()
        self.anns = self._create_annotation_mapping()
        
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)
        
        for ann in self.anns.values():
            self.imgToAnns[ann['image_id']].append(ann)
            self.catToImgs[ann['category_id']].append(ann['image_id'])
            
        self.dataset = {
            'images': self.imgs,
            'annotations': list(self.anns.values()),
            'categories': list(self.cats.values()),
        }
        
    def _create_category_mapping(self):
        """Create a category mapping similar to COCO format"""
        cats = {}
        for idx, name in enumerate(self.orig_dataset.class_names):
            cat_id = idx
            cats[cat_id] = {
                'id': cat_id,
                'name': name,
                'supercategory': 'none'
            }
        return cats
        
    def _create_image_mapping(self):
        """Create an image mapping similar to COCO format"""
        imgs = []
        for idx, img_path in enumerate(self.orig_dataset.image_paths):
            img = Image.open(img_path)
            width, height = img.size
            imgs.append({
                'id': self.orig_dataset.ids[idx],  
                'file_name': os.path.basename(img_path),
                'width': width,
                'height': height
            })
        return imgs
        
    def _create_annotation_mapping(self):
        """Create an annotation mapping similar to COCO format"""
        anns = {}
        ann_id = 0
        
        for idx, (img_path, label_path) in enumerate(zip(self.orig_dataset.image_paths, self.orig_dataset.label_paths)):
            img = Image.open(img_path)
            width, height = img.size
            
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:  
                        class_id = int(data[0])
                        x_center, y_center, box_width, box_height = map(float, data[1:5])
                        
                        x = (x_center - box_width / 2) * width
                        y = (y_center - box_height / 2) * height
                        w = box_width * width
                        h = box_height * height
                        
                        anns[ann_id] = {
                            'id': ann_id,
                            'image_id': self.orig_dataset.ids[idx],  
                            'category_id': class_id,
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
    