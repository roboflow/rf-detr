import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from rfdetr.datasets.synthetic import (
    SYNTHETIC_SHAPES,
    SYNTHETIC_COLORS,
    calculate_boundary_overlap,
    draw_synthetic_shape,
    generate_coco_dataset,
    generate_synthetic_sample,
)


@pytest.mark.parametrize("bbox,expected_overlap", [
    pytest.param(
        np.array([40.0, 40.0, 60.0, 60.0]), 0.0,
        id="fully_inside"
    ),
    pytest.param(
        np.array([-10.0, 40.0, 10.0, 60.0]), 0.5,
        id="half_outside_horizontally"
    ),
    pytest.param(
        np.array([110.0, 40.0, 130.0, 60.0]), 1.0,
        id="fully_outside"
    ),
])
def test_calculate_boundary_overlap(bbox, expected_overlap):
    img_size = 100
    result = calculate_boundary_overlap(bbox, img_size)
    assert result == pytest.approx(expected_overlap)


@pytest.mark.parametrize("shape,color", [
    pytest.param("square", sv.Color.RED, id="square_red"),
    pytest.param("triangle", sv.Color.GREEN, id="triangle_green"),
    pytest.param("circle", sv.Color.BLUE, id="circle_blue"),
])
def test_draw_synthetic_shape(shape, color):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_modified = draw_synthetic_shape(img.copy(), shape, color, (50, 50), 20)
    assert not np.array_equal(img, img_modified)


@pytest.mark.parametrize(
    "img_size,min_objects,max_objects,class_mode",
    [
        pytest.param(100, 1, 3, "shape", id="small_shape_mode"),
        pytest.param(200, 2, 5, "color", id="medium_color_mode"),
        pytest.param(100, 1, 1, "shape", id="single_object"),
    ]
)
def test_generate_synthetic_sample(img_size, min_objects, max_objects, class_mode):
    img, detections = generate_synthetic_sample(
        img_size=img_size,
        min_objects=min_objects,
        max_objects=max_objects,
        class_mode=class_mode
    )

    assert img.shape == (img_size, img_size, 3)
    assert min_objects <= len(detections) <= max_objects
    assert hasattr(detections, 'xyxy')
    assert hasattr(detections, 'class_id')


@pytest.mark.parametrize(
    "num_images,img_size,class_mode,split_ratios",
    [
        pytest.param(
            5, 100, "shape",
            {"train": 0.6, "val": 0.2, "test": 0.2},
            id="shape_mode_all_splits"
        ),
        pytest.param(
            3, 64, "color",
            {"train": 0.5, "val": 0.5},
            id="color_mode_two_splits"
        ),
        pytest.param(
            2, 128, "shape",
            {"train": 1.0},
            id="single_split_only"
        ),
    ]
)
def test_generate_coco_dataset(num_images, img_size, class_mode, split_ratios):
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "test_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=num_images,
            img_size=img_size,
            class_mode=class_mode,
            split_ratios=split_ratios
        )

        assert output_dir.exists()

        for split in split_ratios.keys():
            split_dir = output_dir / split
            assert split_dir.exists()
            assert (split_dir / "annotations.json").exists()

            with open(split_dir / "annotations.json", "r") as f:
                data = json.load(f)
                assert "images" in data
                assert "annotations" in data
                assert "categories" in data

                # Check if images exist
                for img_info in data["images"]:
                    assert (split_dir / "images" / img_info["file_name"]).exists()


# Additional edge case tests
def test_generate_synthetic_sample_zero_objects():
    """Test that sample generation handles cases where no objects are placed."""
    img, detections = generate_synthetic_sample(
        img_size=100,
        min_objects=0,
        max_objects=0,
        class_mode="shape"
    )
    assert img.shape == (100, 100, 3)
    assert len(detections) == 0


def test_calculate_boundary_overlap_edge_cases():
    """Test boundary overlap with edge cases."""
    img_size = 100

    # Exactly at boundary
    bbox_at_boundary = np.array([0.0, 0.0, 50.0, 50.0])
    assert calculate_boundary_overlap(bbox_at_boundary, img_size) == 0.0

    # Exactly at opposite boundary
    bbox_at_max_boundary = np.array([50.0, 50.0, 100.0, 100.0])
    assert calculate_boundary_overlap(bbox_at_max_boundary, img_size) == 0.0
