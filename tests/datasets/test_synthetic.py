import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rfdetr.datasets.synthetic import (
    calculate_boundary_overlap,
    calculate_iou,
    draw_synthetic_shape,
    generate_coco_dataset,
    generate_synthetic_sample,
)


def test_calculate_iou():
    # Identical boxes
    bbox1 = (100.0, 100.0, 50.0, 50.0)
    assert calculate_iou(bbox1, bbox1) == pytest.approx(1.0)

    # No overlap
    bbox2 = (200.0, 200.0, 50.0, 50.0)
    assert calculate_iou(bbox1, bbox2) == 0.0

    # Partial overlap
    bbox3 = (125.0, 100.0, 50.0, 50.0)
    # intersection: x=[100, 125], y=[75, 125] -> area = 25 * 50 = 1250
    # union: 2500 + 2500 - 1250 = 3750
    # iou: 1250 / 3750 = 1/3
    assert calculate_iou(bbox1, bbox3) == pytest.approx(1 / 3)


def test_calculate_boundary_overlap():
    img_size = 100
    
    # Fully inside
    bbox1 = (50.0, 50.0, 20.0, 20.0)
    assert calculate_boundary_overlap(bbox1, img_size) == 0.0
    
    # Half outside horizontally
    bbox2 = (0.0, 50.0, 20.0, 20.0)
    assert calculate_boundary_overlap(bbox2, img_size) == pytest.approx(0.5)
    
    # Fully outside
    bbox3 = (150.0, 50.0, 20.0, 20.0)
    assert calculate_boundary_overlap(bbox3, img_size) == 1.0


def test_draw_synthetic_shape():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Just check if it runs and modifies the image
    img_modified = draw_synthetic_shape(img.copy(), "square", (255, 0, 0), (50, 50), 20)
    assert not np.array_equal(img, img_modified)
    
    img_modified = draw_synthetic_shape(img.copy(), "triangle", (0, 255, 0), (50, 50), 20)
    assert not np.array_equal(img, img_modified)
    
    img_modified = draw_synthetic_shape(img.copy(), "circle", (0, 0, 255), (50, 50), 20)
    assert not np.array_equal(img, img_modified)


def test_generate_synthetic_sample():
    img_size = 100
    img, annotations = generate_synthetic_sample(
        img_size=img_size,
        min_objects=1,
        max_objects=3,
        class_mode="shape"
    )
    
    assert img.shape == (img_size, img_size, 3)
    assert 1 <= len(annotations) <= 3
    for anno in annotations:
        assert "category_id" in anno
        assert "bbox" in anno
        assert "area" in anno
        assert "iscrowd" in anno
        assert len(anno["bbox"]) == 4


def test_generate_coco_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "test_dataset"
        num_images = 5
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=num_images,
            img_size=100,
            class_mode="shape",
            split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}
        )
        
        assert output_dir.exists()
        for split in ["train", "val", "test"]:
            split_dir = output_dir / split
            assert split_dir.exists()
            assert (split_dir / "_annotations.coco.json").exists()
            
            with open(split_dir / "_annotations.coco.json", "r") as f:
                data = json.load(f)
                assert "images" in data
                assert "annotations" in data
                assert "categories" in data
                
                # Check if images exist
                for img_info in data["images"]:
                    assert (split_dir / img_info["file_name"]).exists()
