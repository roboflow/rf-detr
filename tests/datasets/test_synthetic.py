import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from rfdetr.datasets.synthetic import (
    DEFAULT_SPLIT_RATIOS,
    SYNTHETIC_COLORS,
    SYNTHETIC_SHAPES,
    DatasetSplitRatios,
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
    pytest.param(
        np.array([0.0, 0.0, 50.0, 50.0]), 0.0,
        id="exactly_at_boundary"
    ),
    pytest.param(
        np.array([50.0, 50.0, 100.0, 100.0]), 0.0,
        id="exactly_at_max_boundary"
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
        pytest.param(100, 0, 0, "shape", id="zero_objects"),
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
    "num_images,img_size,class_mode,split_ratios,expected_splits",
    [
        # Test with dictionary (legacy support)
        pytest.param(
            5, 100, "shape",
            {"train": 0.6, "val": 0.2, "test": 0.2},
            ["train", "val", "test"],
            id="shape_mode_all_splits_dict"
        ),
        pytest.param(
            3, 64, "color",
            {"train": 0.5, "val": 0.5},
            ["train", "val"],
            id="color_mode_two_splits_dict"
        ),
        pytest.param(
            2, 128, "shape",
            {"train": 1.0},
            ["train"],
            id="single_split_only_dict"
        ),
        # Test with DatasetSplitRatios dataclass
        pytest.param(
            4, 100, "shape",
            DatasetSplitRatios(train=0.7, val=0.2, test=0.1),
            ["train", "val", "test"],
            id="split_ratios_dataclass"
        ),
        pytest.param(
            3, 64, "color",
            DatasetSplitRatios(train=0.8, val=0.2, test=0.0),
            ["train", "val"],
            id="split_ratios_no_test"
        ),
        # Test with tuple
        pytest.param(
            4, 100, "shape",
            (0.7, 0.2, 0.1),
            ["train", "val", "test"],
            id="split_ratios_tuple_three"
        ),
        pytest.param(
            3, 64, "color",
            (0.8, 0.2),
            ["train", "val"],
            id="split_ratios_tuple_two"
        ),
        # Test with default
        pytest.param(
            10, 64, "shape",
            DEFAULT_SPLIT_RATIOS,
            ["train", "val", "test"],
            id="split_ratios_default"
        ),
    ]
)
def test_generate_coco_dataset(num_images, img_size, class_mode, split_ratios, expected_splits, tmp_path):
    output_dir = tmp_path / "test_dataset"
    generate_coco_dataset(
        output_dir=str(output_dir),
        num_images=num_images,
        img_size=img_size,
        class_mode=class_mode,
        split_ratios=split_ratios
    )

    assert output_dir.exists()

    for split in expected_splits:
        split_dir = output_dir / split
        assert split_dir.exists()
        assert (split_dir / "_annotations.coco.json").exists()

        with open(split_dir / "_annotations.coco.json", "r") as f:
            data = json.load(f)
            assert "images" in data
            assert "annotations" in data
            assert "categories" in data

            # Check if images exist (they should be in the split directory, not in a subdirectory)
            for img_info in data["images"]:
                assert (split_dir / img_info["file_name"]).exists()
