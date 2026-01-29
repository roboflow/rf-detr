# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import pytest
import supervision as sv

from rfdetr.datasets.yolo import CocoLikeAPI, _MockSvDataset


class TestCocoLikeAPI:
    """Tests for the CocoLikeAPI class."""

    @pytest.fixture
    def coco_api(self):
        """Fixture to create a test instance of CocoLikeAPI."""
        mock = _MockSvDataset()
        return CocoLikeAPI(mock.classes, mock)

    def test_initialization(self, coco_api):
        """Test that the API initializes correctly."""
        assert coco_api is not None
        assert hasattr(coco_api, "dataset")
        assert hasattr(coco_api, "imgs")
        assert hasattr(coco_api, "anns")
        assert hasattr(coco_api, "cats")
        assert hasattr(coco_api, "imgToAnns")
        assert hasattr(coco_api, "catToImgs")

    def test_dataset_structure(self, coco_api):
        """Test the structure of the COCO dataset."""
        assert "images" in coco_api.dataset
        assert "annotations" in coco_api.dataset
        assert "categories" in coco_api.dataset

    @pytest.mark.parametrize("dataset_part, expected_count", [
        ("images", 2),
        ("categories", 2),
        ("annotations", 2)
    ])
    def test_dataset_counts(self, coco_api, dataset_part, expected_count):
        """Test the number of images, categories, and annotations in the dataset."""
        assert len(coco_api.dataset[dataset_part]) == expected_count

    @pytest.mark.parametrize("img_ids, expected_ids", [
        (None, [0, 1]),
        ([0], [0]),
        ([1], [1]),
        ([0, 1], [0, 1])
    ])
    def test_get_img_ids_by_img_ids(self, coco_api, img_ids, expected_ids):
        """Test getImgIds method with various image ID filters."""
        result = coco_api.getImgIds(imgIds=img_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize("cat_ids, expected_img_ids", [
        (None, [0, 1]),
        ([0], [0]),
        ([1], [1]),
        ([0, 1], [0, 1])
    ])
    def test_get_img_ids_by_cat_ids(self, coco_api, cat_ids, expected_img_ids):
        """Test getImgIds method with various category ID filters."""
        result = coco_api.getImgIds(catIds=cat_ids)
        assert sorted(result) == sorted(expected_img_ids)

    @pytest.mark.parametrize("cat_names, expected_ids", [
        (None, [0, 1]),
        (["cat"], [0]),
        (["dog"], [1]),
        (["cat", "dog"], [0, 1])
    ])
    def test_get_cat_ids_by_names(self, coco_api, cat_names, expected_ids):
        """Test getCatIds method with various category name filters."""
        result = coco_api.getCatIds(catNms=cat_names)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize("cat_ids, expected_ids", [
        (None, [0, 1]),
        ([0], [0]),
        ([1], [1]),
        ([0, 1], [0, 1])
    ])
    def test_get_cat_ids_by_ids(self, coco_api, cat_ids, expected_ids):
        """Test getCatIds method with various category ID filters."""
        result = coco_api.getCatIds(catIds=cat_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize("img_ids, cat_ids, expected_ids", [
        (None, None, [0, 1]),
        ([0], None, [0]),
        (None, [1], [1]),
        ([0], [0], [0])
    ])
    def test_get_ann_ids(self, coco_api, img_ids, cat_ids, expected_ids):
        """Test getAnnIds method with various filter conditions."""
        result = coco_api.getAnnIds(imgIds=img_ids, catIds=cat_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize("ann_ids, expected_length", [
        ([0], 1),
        ([1], 1),
        ([0, 1], 2)
    ])
    def test_load_anns(self, coco_api, ann_ids, expected_length):
        """Test loadAnns method with various annotation IDs."""
        result = coco_api.loadAnns(ann_ids)
        assert len(result) == expected_length
        assert all(ann["id"] in ann_ids for ann in result)

    @pytest.mark.parametrize("cat_ids, expected_length", [
        ([0], 1),
        ([1], 1),
        ([0, 1], 2),
        (None, 2)
    ])
    def test_load_cats(self, coco_api, cat_ids, expected_length):
        """Test loadCats method with various category IDs."""
        result = coco_api.loadCats(cat_ids)
        assert len(result) == expected_length
        if cat_ids is not None:
            assert all(cat["id"] in cat_ids for cat in result)

    @pytest.mark.parametrize("img_ids, expected_length", [
        ([0], 1),
        ([1], 1),
        ([0, 1], 2)
    ])
    def test_load_imgs(self, coco_api, img_ids, expected_length):
        """Test loadImgs method with various image IDs."""
        result = coco_api.loadImgs(img_ids)
        assert len(result) == expected_length
        assert all(img["id"] in img_ids for img in result)

    def test_img_to_anns(self, coco_api):
        """Test the imgToAnns index."""
        assert len(coco_api.imgToAnns[0]) == 1
        assert len(coco_api.imgToAnns[1]) == 1
        assert coco_api.imgToAnns[0][0]["id"] == 0
        assert coco_api.imgToAnns[1][0]["id"] == 1

    def test_cat_to_imgs(self, coco_api):
        """Test the catToImgs index."""
        assert len(coco_api.catToImgs[0]) == 1
        assert len(coco_api.catToImgs[1]) == 1
        assert 0 in coco_api.catToImgs[0]
        assert 1 in coco_api.catToImgs[1]

    @pytest.mark.parametrize("ann_id", [0, 1])
    def test_annotation_format(self, coco_api, ann_id):
        """Test that annotations are in the correct format."""
        ann = coco_api.loadAnns([ann_id])[0]

        # Check required fields
        required_fields = ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]
        for field in required_fields:
            assert field in ann, f"Annotation missing required field: {field}"

        # Check bbox format
        assert len(ann["bbox"]) == 4, "BBox must have 4 coordinates"
        assert all(isinstance(x, (int, float)) for x in ann["bbox"]), "BBox coordinates must be numeric"

        # Check area
        assert isinstance(ann["area"], (int, float)), "Area must be numeric"
        assert ann["area"] > 0, "Area must be positive"

        # Check iscrowd
        assert ann["iscrowd"] in [0, 1], "iscrowd must be 0 or 1"

    @pytest.mark.parametrize("cat_id", [0, 1])
    def test_category_format(self, coco_api, cat_id):
        """Test that categories are in the correct format."""
        cat = coco_api.loadCats([cat_id])[0]

        # Check required fields
        required_fields = ["id", "name", "supercategory"]
        for field in required_fields:
            assert field in cat, f"Category missing required field: {field}"

        # Check field types
        assert isinstance(cat["id"], int), "Category ID must be an integer"
        assert isinstance(cat["name"], str), "Category name must be a string"
        assert isinstance(cat["supercategory"], str), "Supercategory must be a string"

    @pytest.mark.parametrize("img_id", [0, 1])
    def test_image_format(self, coco_api, img_id):
        """Test that images are in the correct format."""
        img = coco_api.loadImgs([img_id])[0]

        # Check required fields
        required_fields = ["id", "file_name", "width", "height"]
        for field in required_fields:
            assert field in img, f"Image missing required field: {field}"

        # Check field types
        assert isinstance(img["id"], int), "Image ID must be an integer"
        assert isinstance(img["file_name"], str), "File name must be a string"
        assert isinstance(img["width"], int), "Width must be an integer"
        assert isinstance(img["height"], int), "Height must be an integer"

    def test_empty_annotations(self):
        """Test handling of images with no annotations."""
        class EmptyMockDataset(_MockSvDataset):
            def __getitem__(self, i):
                det = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([]))
                return f"img_{i}.jpg", np.zeros((100, 100, 3), dtype=np.uint8), det

        api = CocoLikeAPI(["cat"], EmptyMockDataset())
        assert len(api.dataset["annotations"]) == 0
        assert len(api.getAnnIds()) == 0

    def test_images_with_multiple_annotations(self):
        """Test handling of images with multiple annotations per image."""
        class MultiAnnotationMockDataset(_MockSvDataset):
            def __getitem__(self, i):
                if i == 0:
                    det = sv.Detections(
                        xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
                        class_id=np.array([0, 1])
                    )
                else:
                    det = sv.Detections(
                        xyxy=np.array([[15, 25, 35, 45]]),
                        class_id=np.array([0])
                    )
                return f"img_{i}.jpg", np.zeros((100, 100, 3), dtype=np.uint8), det

        api = CocoLikeAPI(["cat", "dog"], MultiAnnotationMockDataset())

        # Verify 3 annotations in total
        assert len(api.dataset["annotations"]) == 3

        # Verify annotations per image
        assert len(api.imgToAnns[0]) == 2
        assert len(api.imgToAnns[1]) == 1

        # Verify image IDs per category
        assert 0 in api.catToImgs[0]
        assert 1 in api.catToImgs[0]
        assert 0 in api.catToImgs[1]
