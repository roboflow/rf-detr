"""Synthetic dataset generation with COCO formatting."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Available shapes for synthetic dataset generation
SYNTHETIC_SHAPES = ["square", "triangle", "circle"]
# Available colors for synthetic dataset generation (RGB format)
SYNTHETIC_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def draw_synthetic_shape(img: np.ndarray, shape: str, color: Tuple[int, int, int], center: Tuple[int, int], size: int) -> np.ndarray:
    """Draw a geometric shape on an image.

    Args:
        img: Input image array to draw on.
        shape: Shape to draw ("square", "triangle", or "circle").
        color: RGB color tuple.
        center: Center position (cx, cy).
        size: Size of the shape.

    Returns:
        Image with drawn shape.
    """
    cx, cy = center
    # OpenCV uses BGR, but we receive RGB
    color_bgr = (color[2], color[1], color[0])

    if shape == "square":
        # Draw filled square
        half_size = size // 2
        pt1 = (cx - half_size, cy - half_size)
        pt2 = (cx + half_size, cy + half_size)
        cv2.rectangle(img, pt1, pt2, color_bgr, -1)
    elif shape == "triangle":
        # Draw filled triangle (equilateral pointing up)
        height = int(size * 0.866)  # sqrt(3)/2 for equilateral triangle
        pt1 = (cx, cy - 2 * height // 3)
        pt2 = (cx - size // 2, cy + height // 3)
        pt3 = (cx + size // 2, cy + height // 3)
        pts = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(img, [pts], color_bgr)
    elif shape == "circle":
        # Draw filled circle
        cv2.circle(img, (cx, cy), size // 2, color_bgr, -1)
    return img


def calculate_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2

    x1_min, x1_max = cx1 - w1 / 2, cx1 + w1 / 2
    y1_min, y1_max = cy1 - h1 / 2, cy1 + h1 / 2
    x2_min, x2_max = cx2 - w2 / 2, cx2 + w2 / 2
    y2_min, y2_max = cy2 - h2 / 2, cy2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_boundary_overlap(bbox: Tuple[float, float, float, float], img_size: int) -> float:
    """Calculate how much of a bounding box is outside the image boundaries."""
    cx, cy, w, h = bbox
    # Convert to absolute pixels if input is normalized, but here we use absolute
    bbox_x_min = cx - w / 2
    bbox_y_min = cy - h / 2
    bbox_x_max = cx + w / 2
    bbox_y_max = cy + h / 2

    inside_x_min = max(bbox_x_min, 0)
    inside_y_min = max(bbox_y_min, 0)
    inside_x_max = min(bbox_x_max, img_size)
    inside_y_max = min(bbox_y_max, img_size)

    if inside_x_max > inside_x_min and inside_y_max > inside_y_min:
        inside_area = (inside_x_max - inside_x_min) * (inside_y_max - inside_y_min)
    else:
        inside_area = 0.0

    total_area = w * h
    return 1.0 - (inside_area / total_area) if total_area > 0 else 0.0


def generate_synthetic_sample(
    img_size: int,
    min_objects: int,
    max_objects: int,
    class_mode: Literal["shape", "color"],
    min_size_ratio: float = 0.1,
    max_size_ratio: float = 0.3,
    overlap_threshold: float = 0.3,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a single synthetic image and its COCO annotations."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    annotations = []
    color_names = list(SYNTHETIC_COLORS.keys())
    num_objects = random.randint(min_objects, max_objects)
    
    placed_bboxes = []
    
    for _ in range(num_objects):
        shape = random.choice(SYNTHETIC_SHAPES)
        color_name = random.choice(color_names)
        color = SYNTHETIC_COLORS[color_name]
        
        if class_mode == "shape":
            category_id = SYNTHETIC_SHAPES.index(shape)
        else:
            category_id = color_names.index(color_name)
            
        min_size = max(10, int(img_size * min_size_ratio))
        max_size = max(min_size + 1, int(img_size * max_size_ratio))

        for _ in range(100): # max attempts
            obj_size = random.randint(min_size, max_size)
            cx = random.randint(obj_size // 2, img_size - obj_size // 2)
            cy = random.randint(obj_size // 2, img_size - obj_size // 2)
            
            # Simplified bbox for synthetic shapes (center-based)
            bbox = (float(cx), float(cy), float(obj_size), float(obj_size))
            
            if calculate_boundary_overlap(bbox, img_size) > 0.05:
                continue
            
            if any(calculate_iou(bbox, pb) > overlap_threshold for pb in placed_bboxes):
                continue
                
            img = draw_synthetic_shape(img, shape, color, (cx, cy), obj_size)
            placed_bboxes.append(bbox)
            
            # COCO format: [x, y, width, height]
            coco_bbox = [float(cx - obj_size / 2), float(cy - obj_size / 2), float(obj_size), float(obj_size)]
            annotations.append({
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": float(obj_size * obj_size),
                "iscrowd": 0
            })
            break
            
    return img, annotations


def generate_coco_dataset(
    output_dir: str,
    num_images: int,
    img_size: int = 640,
    class_mode: Literal["shape", "color"] = "shape",
    split_ratios: Dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1},
):
    """Generate a full synthetic dataset in Roboflow COCO format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if class_mode == "shape":
        categories = [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(SYNTHETIC_SHAPES)]
    else:
        categories = [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(SYNTHETIC_COLORS.keys())]
        
    image_id_counter = 1
    anno_id_counter = 1
    
    # Shuffle splits
    all_indices = list(range(num_images))
    random.shuffle(all_indices)
    
    start_idx = 0
    for split, ratio in split_ratios.items():
        num_split = int(num_images * ratio)
        split_indices = all_indices[start_idx : start_idx + num_split]
        start_idx += num_split
        
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        coco_output = {
            "info": {"description": f"Synthetic {class_mode} dataset"},
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        print(f"Generating {split} split with {len(split_indices)} images...")
        for i in tqdm(split_indices):
            img, annos = generate_synthetic_sample(img_size, 1, 10, class_mode)
            
            file_name = f"{i:06d}.jpg"
            cv2.imwrite(str(split_dir / file_name), img)
            
            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": file_name,
                "width": img_size,
                "height": img_size
            })
            
            for anno in annos:
                anno.update({
                    "id": anno_id_counter,
                    "image_id": image_id_counter
                })
                coco_output["annotations"].append(anno)
                anno_id_counter += 1
                
            image_id_counter += 1
            
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(coco_output, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic COCO dataset")
    parser.add_argument("--output", type=str, default="synthetic_dataset", help="Output directory")
    parser.add_argument("--num_images", type=int, default=100, help="Total number of images")
    parser.add_argument("--img_size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--mode", type=str, choices=["shape", "color"], default="shape", help="Classification mode")
    
    args = parser.parse_args()
    generate_coco_dataset(args.output, args.num_images, args.img_size, args.mode)
