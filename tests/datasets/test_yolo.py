# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from rfdetr.datasets.yolo import (
    CocoLikeAPI,
    YoloDetection,
    _extract_yolo_class_names,
    _LazyYoloDetectionDataset,
    _MockSvDataset,
    is_valid_yolo_dataset,
)


def _write_yolo_segmentation_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal YOLO segmentation dataset on disk."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / "sample.png"
    Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_path)
    (label_dir / "sample.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names:\n  0: carton\n", encoding="utf-8")
    return image_dir, label_dir, data_file


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
        assert "info" in coco_api.dataset
        assert "images" in coco_api.dataset
        assert "annotations" in coco_api.dataset
        assert "categories" in coco_api.dataset

    @pytest.mark.parametrize(
        "dataset_part, expected_count",
        [
            ("images", 2),
            ("categories", 2),
            ("annotations", 2),
        ],
    )
    def test_dataset_counts(self, coco_api, dataset_part, expected_count):
        """Test the number of images, categories, and annotations in the dataset."""
        assert len(coco_api.dataset[dataset_part]) == expected_count

    @pytest.mark.parametrize(
        "img_ids, expected_ids",
        [
            (None, [0, 1]),
            ([0], [0]),
            ([1], [1]),
            ([0, 1], [0, 1]),
        ],
    )
    def test_get_img_ids_by_img_ids(self, coco_api, img_ids, expected_ids):
        """Test getImgIds method with various image ID filters."""
        result = coco_api.getImgIds(imgIds=img_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "cat_ids, expected_img_ids",
        [
            (None, [0, 1]),
            ([0], [0]),
            ([1], [1]),
            ([0, 1], [0, 1]),
        ],
    )
    def test_get_img_ids_by_cat_ids(self, coco_api, cat_ids, expected_img_ids):
        """Test getImgIds method with various category ID filters."""
        result = coco_api.getImgIds(catIds=cat_ids)
        assert sorted(result) == sorted(expected_img_ids)

    @pytest.mark.parametrize(
        "cat_names, expected_ids",
        [
            (None, [0, 1]),
            (["cat"], [0]),
            (["dog"], [1]),
            (["cat", "dog"], [0, 1]),
        ],
    )
    def test_get_cat_ids_by_names(self, coco_api, cat_names, expected_ids):
        """Test getCatIds method with various category name filters."""
        result = coco_api.getCatIds(catNms=cat_names)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "cat_ids, expected_ids",
        [
            (None, [0, 1]),
            ([0], [0]),
            ([1], [1]),
            ([0, 1], [0, 1]),
        ],
    )
    def test_get_cat_ids_by_ids(self, coco_api, cat_ids, expected_ids):
        """Test getCatIds method with various category ID filters."""
        result = coco_api.getCatIds(catIds=cat_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "img_ids, cat_ids, expected_ids",
        [
            (None, None, [0, 1]),
            ([0], None, [0]),
            (None, [1], [1]),
            ([0], [0], [0]),
        ],
    )
    def test_get_ann_ids(self, coco_api, img_ids, cat_ids, expected_ids):
        """Test getAnnIds method with various filter conditions."""
        result = coco_api.getAnnIds(imgIds=img_ids, catIds=cat_ids)
        assert sorted(result) == sorted(expected_ids)

    @pytest.mark.parametrize(
        "ann_ids, expected_length",
        [
            ([0], 1),
            ([1], 1),
            ([0, 1], 2),
        ],
    )
    def test_load_anns(self, coco_api, ann_ids, expected_length):
        """Test loadAnns method with various annotation IDs."""
        result = coco_api.loadAnns(ann_ids)
        assert len(result) == expected_length
        assert all(ann["id"] in ann_ids for ann in result)

    @pytest.mark.parametrize(
        "cat_ids, expected_length",
        [
            ([0], 1),
            ([1], 1),
            ([0, 1], 2),
            (None, 2),
        ],
    )
    def test_load_cats(self, coco_api, cat_ids, expected_length):
        """Test loadCats method with various category IDs."""
        result = coco_api.loadCats(cat_ids)
        assert len(result) == expected_length
        if cat_ids is not None:
            assert all(cat["id"] in cat_ids for cat in result)

    @pytest.mark.parametrize(
        "img_ids, expected_length",
        [
            ([0], 1),
            ([1], 1),
            ([0, 1], 2),
        ],
    )
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
                    det = sv.Detections(xyxy=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]), class_id=np.array([0, 1]))
                else:
                    det = sv.Detections(xyxy=np.array([[15, 25, 35, 45]]), class_id=np.array([0]))
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


class TestBuildRoboflowFromYoloAugConfig:
    """Regression tests for #769: aug_config forwarded to transform builders."""

    def _make_args(self, square_resize_div_64: bool, aug_config=None) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            dataset_dir="/fake/dataset",
            square_resize_div_64=square_resize_div_64,
            aug_config=aug_config,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=None,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
        )

    @pytest.mark.parametrize(
        "square_resize_div_64,transform_fn,aug_config",
        [
            pytest.param(
                True,
                "make_coco_transforms_square_div_64",
                {"HorizontalFlip": {"p": 0.5}},
                id="square_div_64_with_config",
            ),
            pytest.param(False, "make_coco_transforms", {"HorizontalFlip": {"p": 0.5}}, id="standard_with_config"),
            pytest.param(True, "make_coco_transforms_square_div_64", None, id="square_div_64_none"),
            pytest.param(False, "make_coco_transforms", None, id="standard_none"),
        ],
    )
    def test_aug_config_forwarded_to_transform(
        self, square_resize_div_64: bool, transform_fn: str, aug_config: object
    ) -> None:
        """Regression test for #769: aug_config is forwarded to transform builders for all code paths."""
        args = self._make_args(square_resize_div_64=square_resize_div_64, aug_config=aug_config)

        with (
            patch("rfdetr.datasets.yolo.Path") as mock_path,
            patch(f"rfdetr.datasets.yolo.{transform_fn}") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
        ):
            mock_path.return_value.exists.return_value = True
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_transform.call_args
        assert kwargs.get("aug_config") == aug_config, (
            f"{transform_fn} was not called with aug_config={aug_config!r}; got {kwargs}"
        )

    def test_data_yml_selected_when_data_yaml_missing(self, tmp_path: Path) -> None:
        """Regression test: build_roboflow_from_yolo picks data.yml when data.yaml is not present."""
        (tmp_path / "data.yml").touch()
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.dataset_dir = str(tmp_path)

        with (
            patch("rfdetr.datasets.yolo.make_coco_transforms") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
        ):
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert kwargs["data_file"] == str(tmp_path / "data.yml")


class TestIsValidYoloDataset:
    """Tests for the is_valid_yolo_dataset function."""

    def _create_valid_yolo_dataset(self, tmp_path: Path, yaml_filename: str) -> str:
        """Create a minimal valid YOLO dataset directory structure."""
        (tmp_path / yaml_filename).touch()
        for split in ["train", "valid"]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        return str(tmp_path)

    @pytest.mark.parametrize(
        "yaml_filename",
        [
            pytest.param("data.yaml", id="data_yaml"),
            pytest.param("data.yml", id="data_yml"),
        ],
    )
    def test_valid_dataset_with_yaml_variants(self, tmp_path: Path, yaml_filename: str) -> None:
        """Regression test: both data.yaml and data.yml are accepted as valid YOLO datasets."""
        dataset_dir = self._create_valid_yolo_dataset(tmp_path, yaml_filename)
        assert is_valid_yolo_dataset(dataset_dir) is True

    def test_invalid_dataset_missing_yaml(self, tmp_path: Path) -> None:
        """Dataset without any YAML file should be invalid."""
        for split in ["train", "valid"]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        assert is_valid_yolo_dataset(str(tmp_path)) is False

    def test_invalid_dataset_missing_split_dirs(self, tmp_path: Path) -> None:
        """Dataset without required split directories should be invalid."""
        (tmp_path / "data.yaml").touch()
        assert is_valid_yolo_dataset(str(tmp_path)) is False


class TestYoloDetectionLazyMasks:
    """Segmentation masks should stay lightweight until a sample is fetched."""

    def test_segmentation_init_builds_coco_metadata_without_cv2_loading(self, tmp_path: Path) -> None:
        """Dataset construction should not call cv2.imread for every image."""
        image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(tmp_path)

        with patch("cv2.imread", side_effect=AssertionError("cv2.imread should not run during init")):
            dataset = YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_masks=True,
            )

        sample = dataset.sv_dataset.get_image_info(0)
        assert sample.width == 8
        assert sample.height == 6
        assert sample.xyxy.shape == (1, 4)
        assert len(sample.polygons) == 1
        assert dataset.coco.dataset["images"] == [
            {"id": 0, "file_name": str(image_dir / "sample.png"), "height": 6, "width": 8}
        ]
        assert dataset.coco.dataset["annotations"][0]["segmentation"] == []

    def test_segmentation_masks_are_materialized_per_sample_fetch(self, tmp_path: Path) -> None:
        """Fetching a sample should create the dense boolean mask tensor expected downstream."""
        image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(tmp_path)
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=True,
        )

        _, target = dataset[0]

        assert target["masks"].dtype == torch.bool
        assert target["masks"].shape == (1, 6, 8)
        assert torch.count_nonzero(target["masks"]) > 0
        assert target["boxes"][0].tolist() == pytest.approx([2.0, 1.5, 6.0, 4.5])

    def test_segmentation_image_with_no_label_produces_empty_sample(self, tmp_path: Path) -> None:
        """Image with no matching .txt label file should produce an empty sample."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "unlabeled.png")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - carton\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=True,
        )

        sample = dataset.sv_dataset.get_image_info(0)
        assert sample.xyxy.shape == (0, 4)
        assert sample.class_id.shape == (0,)
        assert sample.polygons == ()

        _, target = dataset[0]
        assert target["masks"].shape == (0, 6, 8)
        assert target["boxes"].shape == (0, 4)

    def test_segmentation_multi_instance_polygons_stack_correctly(self, tmp_path: Path) -> None:
        """Two polygon annotations per image should produce masks with shape (2, H, W)."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "two_instances.png")
        # Two distinct non-overlapping polygons
        (label_dir / "two_instances.txt").write_text(
            "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n",
            encoding="utf-8",
        )
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - cat\n  - dog\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=True,
        )

        _, target = dataset[0]
        assert target["masks"].shape == (2, 6, 8), f"Expected (2, 6, 8), got {target['masks'].shape}"
        assert target["masks"].dtype == torch.bool

    @pytest.mark.parametrize(
        "label_content, match_pattern",
        [
            pytest.param("0\n", "Malformed label", id="only_class_id"),
            pytest.param("0 0.1 0.2 0.3\n", "Malformed label", id="too_few_fields"),
            pytest.param(
                "0 0.1 0.2 0.3 0.4 0.5\n",
                "Malformed polygon",
                id="odd_polygon_coords",
            ),
        ],
    )
    def test_malformed_label_line_raises_clear_error(
        self, tmp_path: Path, label_content: str, match_pattern: str
    ) -> None:
        """Malformed label lines should raise a descriptive ValueError with file context."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "bad.png")
        (label_dir / "bad.txt").write_text(label_content, encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - carton\n", encoding="utf-8")

        with pytest.raises(ValueError, match=match_pattern):
            YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_masks=True,
            )

    def test_lazy_dataset_polygon_storage_is_smaller_than_eager_masks(self, tmp_path: Path) -> None:
        """Lazy dataset retains polygon coords, not dense masks — footprint is orders of magnitude smaller."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        n_images = 20
        width, height = 256, 256
        for i in range(n_images):
            Image.new("RGB", (width, height)).save(image_dir / f"img_{i:03d}.png")
            # One quadrilateral polygon per image
            (label_dir / f"img_{i:03d}.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - obj\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=True,
        )

        # Bytes actually retained in the lazy samples (polygon coords + bbox + class id)
        lazy_bytes = sum(
            dataset.sv_dataset.get_image_info(i).xyxy.nbytes
            + dataset.sv_dataset.get_image_info(i).class_id.nbytes
            + sum(p.nbytes for p in dataset.sv_dataset.get_image_info(i).polygons)
            for i in range(len(dataset.sv_dataset))
        )

        # Bytes that eager rasterization would have retained (one bool mask per image)
        eager_mask_bytes = n_images * height * width * np.dtype(bool).itemsize

        assert lazy_bytes < eager_mask_bytes / 10, (
            f"Lazy storage ({lazy_bytes} B) should be at least 10× smaller than eager mask cost ({eager_mask_bytes} B)."
        )

    def test_out_of_range_class_id_raises_clear_error(self, tmp_path: Path) -> None:
        """A label with a class ID beyond the class count should raise ValueError at init."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "sample.png")
        # Dataset defines 1 class (ID 0); label references class ID 5 — out of range
        (label_dir / "sample.txt").write_text("5 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - carton\n", encoding="utf-8")

        with pytest.raises(ValueError, match="out of range"):
            YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_masks=True,
            )

    def test_include_masks_false_uses_supervision_dataset_path(self, tmp_path: Path) -> None:
        """include_masks=False must use supervision's DetectionDataset, not the lazy path."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "sample.png")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - carton\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=False,
        )

        assert not isinstance(dataset.sv_dataset, _LazyYoloDetectionDataset)
        assert len(dataset) == 1
        _, target = dataset[0]
        assert "boxes" in target
        assert "masks" not in target

    def test_lazy_getitem_cv2_returns_none_raises_value_error(self, tmp_path: Path) -> None:
        """When cv2.imread returns None (missing/corrupted file), __getitem__ must raise ValueError."""
        image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(tmp_path)
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=True,
        )

        with patch("cv2.imread", return_value=None):
            with pytest.raises(ValueError, match="Could not read image"):
                dataset[0]

    def test_non_integer_class_id_in_label_raises_value_error(self, tmp_path: Path) -> None:
        """A label line with a non-integer class ID must raise ValueError during init."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "sample.png")
        # "cat" is not a valid integer class ID
        (label_dir / "sample.txt").write_text("cat 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - carton\n", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid class ID"):
            YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_masks=True,
            )


class TestExtractYoloClassNames:
    """Tests for _extract_yolo_class_names with different YAML formats."""

    @pytest.mark.parametrize(
        "yaml_content, expected_names",
        [
            pytest.param(
                "names:\n  - cat\n  - dog\n",
                ["cat", "dog"],
                id="list_format",
            ),
            pytest.param(
                "names:\n  0: cat\n  1: dog\n",
                ["cat", "dog"],
                id="dict_format_sorted_keys",
            ),
            pytest.param(
                "names:\n  1: dog\n  0: cat\n",
                ["cat", "dog"],
                id="dict_format_unsorted_keys",
            ),
        ],
    )
    def test_class_names_formats(self, tmp_path: Path, yaml_content: str, expected_names: list[str]) -> None:
        """Both list and dict YAML formats for class names should be supported."""
        data_file = tmp_path / "data.yaml"
        data_file.write_text(yaml_content, encoding="utf-8")
        assert _extract_yolo_class_names(str(data_file)) == expected_names

    @pytest.mark.parametrize(
        "yaml_content",
        [
            pytest.param(
                "names:\n  0: cat\n  2: dog\n",
                id="dict_format_sparse_keys",
            ),
            pytest.param(
                "names:\n  10: cat\n  20: dog\n",
                id="dict_format_large_numeric_keys",
            ),
        ],
    )
    def test_class_names_dict_non_contiguous_raises(self, tmp_path: Path, yaml_content: str) -> None:
        """Dict 'names' with non-contiguous or non-zero-based keys must raise ValueError.

        The downstream range check in _parse_yolo_label_line assumes class IDs
        are a contiguous 0..N-1 range.  Silently accepting sparse keys would
        cause valid label files to be rejected during parsing (e.g. class ID 2
        in a 2-class dataset built from {0: cat, 2: dog} would exceed the
        num_classes bound).
        """
        data_file = tmp_path / "data.yaml"
        data_file.write_text(yaml_content, encoding="utf-8")
        with pytest.raises(ValueError, match="contiguous"):
            _extract_yolo_class_names(str(data_file))
