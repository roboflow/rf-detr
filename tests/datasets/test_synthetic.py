# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json

import numpy as np
import pytest
import supervision as sv

from rfdetr.datasets.synthetic import (
    DEFAULT_SPLIT_RATIOS,
    SYNTHETIC_SHAPES,
    DatasetSplitRatios,
    _calculate_polygon_area,
    _write_coco_json,
    calculate_boundary_overlap,
    draw_synthetic_shape,
    generate_coco_dataset,
    generate_synthetic_sample,
)


class TestCalculateBoundaryOverlap:
    @pytest.mark.parametrize(
        "bbox,expected_overlap",
        [
            pytest.param(np.array([40.0, 40.0, 60.0, 60.0]), 0.0, id="fully_inside"),
            pytest.param(np.array([-10.0, 40.0, 10.0, 60.0]), 0.5, id="half_outside_horizontally"),
            pytest.param(np.array([110.0, 40.0, 130.0, 60.0]), 1.0, id="fully_outside"),
            pytest.param(np.array([0.0, 0.0, 50.0, 50.0]), 0.0, id="exactly_at_boundary"),
            pytest.param(np.array([50.0, 50.0, 100.0, 100.0]), 0.0, id="exactly_at_max_boundary"),
        ],
    )
    def test_overlap_values(self, bbox, expected_overlap):
        result = calculate_boundary_overlap(bbox, img_size=100)
        assert result == pytest.approx(expected_overlap)


class TestDrawSyntheticShape:
    @pytest.mark.parametrize(
        "shape,color",
        [
            pytest.param("square", sv.Color.RED, id="square_red"),
            pytest.param("triangle", sv.Color.GREEN, id="triangle_green"),
            pytest.param("circle", sv.Color.BLUE, id="circle_blue"),
        ],
    )
    def test_pixels_are_modified(self, shape, color):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_modified, polygon = draw_synthetic_shape(img.copy(), shape, color, (50, 50), 20)
        assert not np.array_equal(img, img_modified)
        assert len(polygon) >= 6
        assert len(polygon) % 2 == 0

    @pytest.mark.parametrize(
        "shape,cx,cy,size",
        [
            pytest.param("square", 50, 50, 20, id="square"),
            pytest.param("triangle", 50, 50, 20, id="triangle"),
            pytest.param("circle", 50, 50, 20, id="circle"),
        ],
    )
    def test_polygon_min_points(self, shape, cx, cy, size):
        """Returned polygon must have at least 3 points (6 values) for COCO."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, shape, sv.Color.WHITE, (cx, cy), size)
        assert len(poly) >= 6, f"{shape} polygon has fewer than 6 values: {poly}"
        assert len(poly) % 2 == 0, f"{shape} polygon has an odd number of values: {poly}"

    @pytest.mark.parametrize(
        "shape,cx,cy,size,expected_n_coords",
        [
            pytest.param("square", 50, 50, 20, 8, id="square_4pts"),
            pytest.param("triangle", 50, 50, 20, 6, id="triangle_3pts"),
            pytest.param("circle", 50, 50, 20, 64, id="circle_32pts"),
        ],
    )
    def test_polygon_coord_count(self, shape, cx, cy, size, expected_n_coords):
        """Each shape must return the expected number of flat coordinate values."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, shape, sv.Color.WHITE, (cx, cy), size)
        assert len(poly) == expected_n_coords

    def test_square_polygon_matches_bbox(self):
        """Square polygon corners must align with the drawn rectangle bounds."""
        cx, cy, size = 60, 40, 30
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, "square", sv.Color.WHITE, (cx, cy), size)
        hs = size // 2
        expected = [
            float(cx - hs),
            float(cy - hs),
            float(cx - hs + size),
            float(cy - hs),
            float(cx - hs + size),
            float(cy - hs + size),
            float(cx - hs),
            float(cy - hs + size),
        ]
        assert poly == pytest.approx(expected)

    def test_unknown_shape_returns_empty_polygon(self):
        """An unrecognised shape name must return an empty polygon without crashing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, "hexagon", sv.Color.WHITE, (50, 50), 20)
        assert poly == []


class TestGenerateSyntheticSample:
    @pytest.mark.parametrize(
        "img_size,min_objects,max_objects,class_mode",
        [
            pytest.param(100, 1, 3, "shape", id="small_shape_mode"),
            pytest.param(200, 2, 5, "color", id="medium_color_mode"),
            pytest.param(100, 1, 1, "shape", id="single_object"),
            pytest.param(100, 0, 0, "shape", id="zero_objects"),
        ],
    )
    def test_output_shape_and_detection_count(self, img_size, min_objects, max_objects, class_mode):
        img, detections = generate_synthetic_sample(
            img_size=img_size, min_objects=min_objects, max_objects=max_objects, class_mode=class_mode
        )
        assert img.shape == (img_size, img_size, 3)
        assert min_objects <= len(detections) <= max_objects
        assert hasattr(detections, "xyxy")
        assert hasattr(detections, "class_id")

    def test_polygon_data_present(self):
        """detections.data must contain a 'polygons' array with one entry per detection."""
        _, detections = generate_synthetic_sample(img_size=100, min_objects=2, max_objects=4, class_mode="shape")
        assert "polygons" in detections.data
        assert len(detections.data["polygons"]) == len(detections)

    def test_polygon_data_non_empty(self):
        """Each stored polygon must be a non-empty list of floats."""
        _, detections = generate_synthetic_sample(img_size=100, min_objects=1, max_objects=3, class_mode="shape")
        for poly in detections.data["polygons"]:
            assert isinstance(poly, list)
            assert len(poly) >= 6

    def test_zero_objects_polygon_data(self):
        """With zero objects the polygon data array must be present but empty."""
        _, detections = generate_synthetic_sample(img_size=100, min_objects=0, max_objects=0, class_mode="shape")
        assert "polygons" in detections.data
        assert len(detections.data["polygons"]) == 0


class TestGenerateCocoDataset:
    @pytest.mark.parametrize(
        "num_images,img_size,class_mode,split_ratios,expected_splits",
        [
            # Test with dictionary (legacy support)
            pytest.param(
                5,
                100,
                "shape",
                {"train": 0.6, "val": 0.2, "test": 0.2},
                ["train", "val", "test"],
                id="shape_mode_all_splits_dict",
            ),
            pytest.param(
                3,
                64,
                "color",
                {"train": 0.5, "val": 0.5},
                ["train", "val"],
                id="color_mode_two_splits_dict",
            ),
            pytest.param(
                2,
                128,
                "shape",
                {"train": 1.0},
                ["train"],
                id="single_split_only_dict",
            ),
            # Test with DatasetSplitRatios dataclass
            pytest.param(
                4,
                100,
                "shape",
                DatasetSplitRatios(train=0.7, val=0.2, test=0.1),
                ["train", "val", "test"],
                id="split_ratios_dataclass",
            ),
            pytest.param(
                3,
                64,
                "color",
                DatasetSplitRatios(train=0.8, val=0.2, test=0.0),
                ["train", "val"],
                id="split_ratios_no_test",
            ),
            # Test with tuple
            pytest.param(
                4,
                100,
                "shape",
                (0.7, 0.2, 0.1),
                ["train", "val", "test"],
                id="split_ratios_tuple_three",
            ),
            pytest.param(
                3,
                64,
                "color",
                (0.8, 0.2),
                ["train", "val"],
                id="split_ratios_tuple_two",
            ),
            # Test with default
            pytest.param(
                10,
                64,
                "shape",
                DEFAULT_SPLIT_RATIOS,
                ["train", "val", "test"],
                id="split_ratios_default",
            ),
        ],
    )
    def test_splits_created(self, num_images, img_size, class_mode, split_ratios, expected_splits, tmp_path):
        output_dir = tmp_path / "test_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=num_images,
            img_size=img_size,
            class_mode=class_mode,
            split_ratios=split_ratios,
        )

        assert output_dir.exists()
        for split in expected_splits:
            split_dir = output_dir / split
            assert split_dir.exists()
            assert (split_dir / "_annotations.coco.json").exists()

            with open(split_dir / "_annotations.coco.json") as f:
                data = json.load(f)
            assert "images" in data
            assert "annotations" in data
            assert "categories" in data
            for img_info in data["images"]:
                assert (split_dir / img_info["file_name"]).exists()

    @pytest.mark.parametrize(
        "split_ratios,error_message",
        [
            pytest.param(
                (1.1, -0.1),
                "Split ratios must be non-negative",
                id="tuple_negative_ratio",
            ),
            pytest.param(
                {"train": 1.1, "val": -0.1},
                "Split ratios must be non-negative",
                id="dict_negative_ratio",
            ),
            pytest.param(
                (0.5, 0.3),
                "Split ratios must sum to 1.0",
                id="tuple_invalid_sum",
            ),
        ],
    )
    def test_invalid_split_ratios(self, split_ratios, error_message, tmp_path):
        output_dir = tmp_path / "test_dataset"
        with pytest.raises(ValueError, match=error_message):
            generate_coco_dataset(
                output_dir=str(output_dir),
                num_images=5,
                img_size=100,
                class_mode="shape",
                split_ratios=split_ratios,
            )


class TestGenerateCocoDatasetWithSegmentation:
    def test_write_coco_json_raises_for_mismatched_inputs(self, tmp_path):
        """Mismatched file/detection list lengths must raise to avoid silent truncation."""
        annotations_path = tmp_path / "_annotations.coco.json"
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=float),
            class_id=np.empty((0,), dtype=int),
            data={"polygons": np.empty(0, dtype=object)},
        )

        with pytest.raises(ValueError, match="file_paths and detections_list must have the same length"):
            _write_coco_json(
                annotations_path=annotations_path,
                classes=["shape"],
                file_paths=["/tmp/a.png", "/tmp/b.png"],
                detections_list=[detections],
                img_size=64,
            )

    def test_creates_files(self, tmp_path):
        """with_segmentation=True must create the same directory/file structure as the default."""
        output_dir = tmp_path / "seg_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=4,
            img_size=64,
            class_mode="shape",
            split_ratios={"train": 0.75, "val": 0.25},
            with_segmentation=True,
        )
        for split in ("train", "val"):
            assert (output_dir / split / "_annotations.coco.json").exists()

    def test_json_structure(self, tmp_path):
        """COCO JSON produced with segmentation must have the required top-level keys."""
        output_dir = tmp_path / "seg_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=4,
            img_size=64,
            class_mode="shape",
            split_ratios={"train": 1.0},
            with_segmentation=True,
        )
        with open(output_dir / "train" / "_annotations.coco.json") as fh:
            data = json.load(fh)
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_has_polygon_field(self, tmp_path):
        """Every annotation must have a non-empty segmentation polygon."""
        output_dir = tmp_path / "seg_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=3,
            img_size=64,
            class_mode="shape",
            min_objects=1,
            max_objects=2,
            split_ratios={"train": 1.0},
            with_segmentation=True,
        )
        with open(output_dir / "train" / "_annotations.coco.json") as fh:
            data = json.load(fh)
        assert len(data["annotations"]) > 0, "Expected at least one annotation"
        for ann in data["annotations"]:
            assert "segmentation" in ann
            assert isinstance(ann["segmentation"], list)
            assert len(ann["segmentation"]) == 1, "Expected exactly one polygon per annotation"
            assert len(ann["segmentation"][0]) >= 6, "Polygon must have at least 3 points"

    def test_area_uses_polygon_when_segmentation_enabled(self, tmp_path):
        """COCO area must match polygon area when segmentation annotations are present."""
        annotations_path = tmp_path / "_annotations.coco.json"
        polygon_data = np.empty(1, dtype=object)
        polygon_data[0] = [0.0, 0.0, 10.0, 0.0, 0.0, 10.0]  # Right triangle area = 50
        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=float),
            class_id=np.array([0], dtype=int),
            data={"polygons": polygon_data},
        )

        _write_coco_json(
            annotations_path=annotations_path,
            classes=["shape"],
            file_paths=["/tmp/synthetic.png"],
            detections_list=[detections],
            img_size=64,
            with_segmentation=True,
        )

        with open(annotations_path) as fh:
            data = json.load(fh)

        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["area"] == pytest.approx(50.0)

    def test_sparse_category_ids(self, tmp_path):
        """Category IDs must use sparse 1-based encoding (1, 3, 5, …)."""
        output_dir = tmp_path / "seg_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=4,
            img_size=64,
            class_mode="shape",
            split_ratios={"train": 1.0},
            with_segmentation=True,
        )
        with open(output_dir / "train" / "_annotations.coco.json") as fh:
            data = json.load(fh)
        cat_ids = {c["id"] for c in data["categories"]}
        expected_ids = {idx * 2 + 1 for idx in range(len(SYNTHETIC_SHAPES))}
        assert cat_ids == expected_ids
        ann_cat_ids = {a["category_id"] for a in data["annotations"]}
        assert ann_cat_ids.issubset(expected_ids)

    def test_images_exist(self, tmp_path):
        """All images referenced in the JSON must exist on disk."""
        output_dir = tmp_path / "seg_dataset"
        generate_coco_dataset(
            output_dir=str(output_dir),
            num_images=3,
            img_size=64,
            class_mode="shape",
            split_ratios={"train": 1.0},
            with_segmentation=True,
        )
        split_dir = output_dir / "train"
        with open(split_dir / "_annotations.coco.json") as fh:
            data = json.load(fh)
        for img_info in data["images"]:
            assert (split_dir / img_info["file_name"]).exists()

    def test_empty_polygon_falls_back_to_empty_segmentation(self, tmp_path):
        """An empty polygon entry silently falls back to ``segmentation=[]``.

        The ``len(polygon_data) < len(detections)`` guard only checks array
        length, not contents.  An element that is an empty list passes the
        guard and takes the ``else`` branch producing ``segmentation=[]``.
        This test documents the existing silent-fallback behaviour.
        """
        annotations_path = tmp_path / "_annotations.coco.json"
        polygon_data = np.empty(1, dtype=object)
        polygon_data[0] = []  # empty polygon — passes length guard
        detections = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=float),
            class_id=np.array([0], dtype=int),
            data={"polygons": polygon_data},
        )
        _write_coco_json(
            annotations_path=annotations_path,
            classes=["shape"],
            file_paths=["/tmp/synthetic.png"],
            detections_list=[detections],
            img_size=64,
            with_segmentation=True,
        )
        with open(annotations_path) as fh:
            data = json.load(fh)
        assert data["annotations"][0]["segmentation"] == []


class TestCalculatePolygonArea:
    @pytest.mark.parametrize(
        "polygon,expected_area",
        [
            pytest.param(
                [0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
                50.0,
                id="right_triangle",
            ),
            pytest.param(
                [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
                100.0,
                id="unit_square_10x10",
            ),
            pytest.param(
                [0.0, 0.0, 5.0, 0.0, 10.0, 0.0],
                0.0,
                id="collinear_points_degenerate",
            ),
            pytest.param(
                [0.0, 0.0, 1.0, 1.0],
                0.0,
                id="fewer_than_3_points",
            ),
            pytest.param(
                [],
                0.0,
                id="empty_polygon",
            ),
        ],
    )
    def test_area(self, polygon, expected_area):
        assert _calculate_polygon_area(polygon) == pytest.approx(expected_area)


class TestDrawSyntheticShapeEdgeCases:
    def test_square_polygon_uses_floor_half_for_odd_size(self):
        """For odd size the square polygon uses ``size // 2`` (floor), which is
        narrower than the bbox half-size ``size / 2``.  This documents the
        known integer-floor behaviour so a future fix is caught by tests.
        """
        cx, cy, size = 50, 50, 21
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, "square", sv.Color.WHITE, (cx, cy), size)

        half_floor = size // 2  # 10
        poly_x_min = min(poly[i] for i in range(0, len(poly), 2))
        poly_x_max = max(poly[i] for i in range(0, len(poly), 2))
        assert poly_x_min == pytest.approx(float(cx - half_floor))
        assert poly_x_max == pytest.approx(float(cx + half_floor))

    def test_triangle_vertex_above_half_size_boundary(self):
        """The triangle apex extends above ``cy - size // 2`` for all sizes ≥ 12.

        ``height = int(size * 0.866)`` and the apex offset is
        ``2 * height // 3``, which exceeds ``size // 2`` because
        ``2/3 * sqrt(3)/2 ≈ 0.577 > 0.5``.  This test documents the known
        geometric overshoot so a future bbox-containment fix is caught.
        """
        cx, cy, size = 50, 50, 20
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, "triangle", sv.Color.WHITE, (cx, cy), size)

        half_size = size // 2  # 10
        poly_y_min = min(poly[i] for i in range(1, len(poly), 2))
        # Apex is at cy - 2*int(20*0.866)//3 = cy - 11, one pixel above cy - 10
        assert poly_y_min < cy - half_size, "Triangle apex should extend above the half_size boundary (known behaviour)"

    @pytest.mark.parametrize(
        "shape,size,expected_n_coords",
        [
            pytest.param("square", 0, 8, id="square_size_0"),
            pytest.param("square", 1, 8, id="square_size_1"),
            pytest.param("circle", 0, 64, id="circle_size_0"),
            pytest.param("circle", 1, 64, id="circle_size_1"),
        ],
    )
    def test_degenerate_size_returns_polygon_without_crashing(self, shape, size, expected_n_coords):
        """draw_synthetic_shape with size=0 or size=1 must not raise and must
        return the expected number of flat coordinate values.
        """
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, poly = draw_synthetic_shape(img, shape, sv.Color.WHITE, (50, 50), size)
        assert len(poly) == expected_n_coords
