"""Synthetic dataset generation with COCO formatting."""

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from typing_extensions import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplitRatios:
    """Dataclass for dataset split ratios.

    Attributes:
        train: Ratio for training set (default: 0.7)
        val: Ratio for validation set (default: 0.2)
        test: Ratio for test set (default: 0.1)

    Raises:
        ValueError: If ratios are negative or sum is not approximately 1.0.
    """
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self):
        """Validate that ratios sum to approximately 1.0 and are non-negative."""
        total = self.train + self.val + self.test
        if any(r < 0 for r in [self.train, self.val, self.test]):
            raise ValueError(f"Split ratios must be non-negative, got train={self.train}, val={self.val}, test={self.test}")
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, filtering out zero ratios."""
        return {k: v for k, v in {"train": self.train, "val": self.val, "test": self.test}.items() if v > 0}


# Default split ratios instance
DEFAULT_SPLIT_RATIOS = DatasetSplitRatios()  # 70/20/10 split


# Type alias for split ratios parameter
SplitRatiosType = Union[DatasetSplitRatios, Tuple[float, ...], Dict[str, float]]


def _normalize_split_ratios(split_ratios: SplitRatiosType) -> Dict[str, float]:
    """Normalize split ratios parameter to a dictionary.

    Args:
        split_ratios: Can be:
            - DatasetSplitRatios dataclass instance
            - Tuple of floats (e.g., (0.7, 0.2, 0.1) for train/val/test)
            - Dictionary (legacy support)

    Returns:
        Dictionary mapping split names to ratios.

    Raises:
        ValueError: If split ratios are invalid.
    """
    if isinstance(split_ratios, DatasetSplitRatios):
        return split_ratios.to_dict()

    if isinstance(split_ratios, tuple):
        if len(split_ratios) == 2:
            result = {"train": split_ratios[0], "val": split_ratios[1]}
        elif len(split_ratios) == 3:
            result = {"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]}
        else:
            raise ValueError(f"Split ratios tuple must have 2 or 3 elements, got {len(split_ratios)}")

        # Validate tuple ratios sum to approximately 1.0
        total = sum(split_ratios)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return result

    if isinstance(split_ratios, dict):
        # Validate that ratios sum to approximately 1.0
        total = sum(split_ratios.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return split_ratios

    raise TypeError(f"split_ratios must be DatasetSplitRatios, tuple, or dict, got {type(split_ratios)}")

# Available shapes for synthetic dataset generation
SYNTHETIC_SHAPES = ["square", "triangle", "circle"]
# Available colors for synthetic dataset generation (RGB format)
SYNTHETIC_COLORS = {
    "red": sv.Color.RED,
    "green": sv.Color.GREEN,
    "blue": sv.Color.BLUE
}


def draw_synthetic_shape(img: np.ndarray, shape: str, color: sv.Color, center: Tuple[int, int], size: int) -> np.ndarray:
    """Draw a geometric shape on an image.

    Args:
        img: Input image array to draw on.
        shape: Shape to draw ("square", "triangle", or "circle").
        color: supervision Color object.
        center: Center position (cx, cy).
        size: Size of the shape.

    Returns:
        Image with drawn shape.
    """
    cx, cy = center
    half_size = size // 2

    if shape == "square":
        rect = sv.Rect(x=cx - half_size, y=cy - half_size, width=size, height=size)
        img = sv.draw_filled_rectangle(scene=img, rect=rect, color=color)
    elif shape == "triangle":
        height = int(size * 0.866)  # sqrt(3)/2 for equilateral triangle
        pt1 = [cx, cy - 2 * height // 3]
        pt2 = [cx - half_size, cy + height // 3]
        pt3 = [cx + half_size, cy + height // 3]
        polygon = np.array([pt1, pt2, pt3], np.int32)
        img = sv.draw_filled_polygon(scene=img, polygon=polygon, color=color)
    elif shape == "circle":
        # Supervision doesn't have a direct filled circle, use cv2 or approximate with polygon
        cv2.circle(img, (cx, cy), half_size, color.as_bgr(), -1)
    return img


def calculate_boundary_overlap(bbox: np.ndarray, img_size: int) -> float:
    """Calculate how much of a bounding box is outside the image boundaries.

    Args:
        bbox: Bounding box in [x_min, y_min, x_max, y_max] format.
        img_size: Size of the image.
    """
    x_min, y_min, x_max, y_max = bbox

    inside_x_min = max(x_min, 0)
    inside_y_min = max(y_min, 0)
    inside_x_max = min(x_max, img_size)
    inside_y_max = min(y_max, img_size)

    if inside_x_max > inside_x_min and inside_y_max > inside_y_min:
        inside_area = (inside_x_max - inside_x_min) * (inside_y_max - inside_y_min)
    else:
        inside_area = 0.0

    total_area = (x_max - x_min) * (y_max - y_min)
    return 1.0 - (inside_area / total_area) if total_area > 0 else 0.0


def generate_synthetic_sample(
    img_size: int,
    min_objects: int,
    max_objects: int,
    class_mode: Literal["shape", "color"],
    min_size_ratio: float = 0.1,
    max_size_ratio: float = 0.3,
    overlap_threshold: float = 0.3,
) -> Tuple[np.ndarray, sv.Detections]:
    """Generate a single synthetic image and its detections."""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    color_names = list(SYNTHETIC_COLORS.keys())
    num_objects = random.randint(min_objects, max_objects)

    xyxys = []
    class_ids = []

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

            # [x_min, y_min, x_max, y_max]
            bbox = np.array([
                float(cx - obj_size / 2),
                float(cy - obj_size / 2),
                float(cx + obj_size / 2),
                float(cy + obj_size / 2)
            ])

            if calculate_boundary_overlap(bbox, img_size) > 0.05:
                continue

            if len(xyxys) > 0:
                ious = sv.box_iou_batch(np.array([bbox]), np.array(xyxys))[0]
                if np.any(ious > overlap_threshold):
                    continue

            img = draw_synthetic_shape(img, shape, color, (cx, cy), obj_size)
            xyxys.append(bbox)
            class_ids.append(category_id)
            break

    detections = sv.Detections(
        xyxy=np.array(xyxys) if xyxys else np.empty((0, 4)),
        class_id=np.array(class_ids) if class_ids else np.empty((0,), dtype=int)
    )
    return img, detections


def generate_coco_dataset(
    output_dir: str,
    num_images: int,
    img_size: int = 640,
    class_mode: Literal["shape", "color"] = "shape",
    split_ratios: SplitRatiosType = DEFAULT_SPLIT_RATIOS,
):
    """Generate a full synthetic dataset in COCO format.

    Args:
        output_dir: Directory where the dataset will be saved.
        num_images: Total number of images to generate.
        img_size: Size of the square images (default: 640).
        class_mode: Classification mode - "shape" or "color" (default: "shape").
        split_ratios: Dataset split ratios. Can be:
            - SplitRatios dataclass instance (default: 70/20/10 split)
            - Tuple of 2 floats for train/val (e.g., (0.8, 0.2))
            - Tuple of 3 floats for train/val/test (e.g., (0.7, 0.2, 0.1))
            - Dictionary (legacy support, e.g., {"train": 0.7, "val": 0.2, "test": 0.1})
    """
    # Normalize split_ratios to dictionary
    split_ratios_dict = _normalize_split_ratios(split_ratios)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if class_mode == "shape":
        classes = SYNTHETIC_SHAPES
    else:
        classes = list(SYNTHETIC_COLORS.keys())

    # Shuffle indices for splits
    all_indices = list(range(num_images))
    random.shuffle(all_indices)

    start_idx = 0
    for split, ratio in split_ratios_dict.items():
        num_split = int(num_images * ratio)
        if num_split == 0 and ratio > 0:
             num_split = 1
        split_indices = all_indices[start_idx : start_idx + num_split]
        start_idx += num_split

        if not split_indices:
            continue

        images_dir = output_path / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_path = output_path / split / "annotations.json"

        images = {}
        annotations = {}

        print(f"Generating {split} split with {len(split_indices)} images...")
        for i in tqdm(split_indices):
            img, detections = generate_synthetic_sample(img_size, 1, 10, class_mode)

            file_name = f"{i:06d}.jpg"
            file_path = str(images_dir / file_name)
            cv2.imwrite(file_path, img)

            images[file_path] = img
            annotations[file_path] = detections

        dataset = sv.DetectionDataset(
            classes=classes,
            images=images,
            annotations=annotations
        )

        dataset.as_coco(
            annotations_path=str(annotations_path)
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic COCO dataset")
    parser.add_argument("--output", type=str, default="synthetic_dataset", help="Output directory")
    parser.add_argument("--num_images", type=int, default=100, help="Total number of images")
    parser.add_argument("--img_size", type=int, default=640, help="Image size (square)")
    parser.add_argument("--mode", type=str, choices=["shape", "color"], default="shape", help="Classification mode")

    args = parser.parse_args()
    generate_coco_dataset(args.output, args.num_images, args.img_size, args.mode)
