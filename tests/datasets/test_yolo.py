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
import torch
from PIL import Image
from pycocotools.coco import COCO

from rfdetr.datasets.yolo import (
    YoloDetection,
    _extract_yolo_class_names,
    _LazyYoloDetectionDataset,
    _resolve_yolo_split_dirs,
    is_valid_yolo_dataset,
)


def _write_minimal_roboflow_yolo_dataset(tmp_path: Path) -> None:
    """Create a minimal Roboflow YOLO dataset root."""
    (tmp_path / "data.yaml").write_text("names:\n  - person\n", encoding="utf-8")
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(tmp_path / split / "images" / "sample.png")
        (tmp_path / split / "labels" / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


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


def _write_yolo_pose_dataset(tmp_path: Path, *, keypoint_dim: int = 3) -> tuple[Path, Path, Path]:
    """Create a minimal YOLO pose dataset on disk."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "sample.png")
    if keypoint_dim == 3:
        label = "0 0.5 0.5 0.5 0.5 0.25 0.25 2 0 0 0\n"
        yaml_text = """
names:
  0: person
kpt_shape: [2, 3]
kpt_names:
  0: [left_eye, right_eye]
flip_idx: [1, 0]
"""
    else:
        label = "0 0.5 0.5 0.5 0.5 0.25 0.25 0 0\n"
        yaml_text = """
names:
  0: person
kpt_shape: [2, 2]
kpt_names:
  0: [left_eye, right_eye]
"""
    (label_dir / "sample.txt").write_text(label, encoding="utf-8")
    data_file = tmp_path / "data.yaml"
    data_file.write_text(yaml_text, encoding="utf-8")
    return image_dir, label_dir, data_file


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

        fake_dirs = (Path("/fake/dataset/train/images"), Path("/fake/dataset/train/labels"))
        with (
            patch("rfdetr.datasets.yolo.Path") as mock_path,
            patch(f"rfdetr.datasets.yolo.{transform_fn}") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
            patch("rfdetr.datasets.yolo._resolve_yolo_split_dirs", return_value=fake_dirs),
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

    def test_auto_no_cuda_sets_gpu_postprocess_false(self) -> None:
        """Auto + no CUDA must keep CPU normalize by passing gpu_postprocess=False."""
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.augmentation_backend = "auto"
        fake_dirs = (Path("/fake/dataset/train/images"), Path("/fake/dataset/train/labels"))
        with (
            patch("rfdetr.datasets.yolo.Path") as mock_path,
            patch("rfdetr.datasets.yolo.make_coco_transforms") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False),
            patch("rfdetr.datasets.yolo._resolve_yolo_split_dirs", return_value=fake_dirs),
        ):
            mock_path.return_value.exists.return_value = True
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_transform.call_args
        assert kwargs["gpu_postprocess"] is False

    def test_keypoint_mode_rejects_detection_only_yolo_format(self, tmp_path: Path) -> None:
        """Keypoint preview training should fail clearly for YOLO datasets without pose metadata."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.dataset_dir = str(tmp_path)
        args.use_grouppose_keypoints = True

        from rfdetr.datasets import build_roboflow

        with pytest.raises(ValueError, match="YOLO keypoint"):
            build_roboflow("train", args, resolution=64)

    def test_keypoint_mode_accepts_yolo_pose_format(self, tmp_path: Path) -> None:
        """Keypoint preview training should build YOLO pose datasets when kpt_shape is present."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        (tmp_path / "data.yaml").write_text(
            "names:\n  - person\nkpt_shape: [1, 3]\n",
            encoding="utf-8",
        )
        (tmp_path / "train" / "labels" / "sample.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.5 0.5 2\n",
            encoding="utf-8",
        )
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.dataset_dir = str(tmp_path)
        args.use_grouppose_keypoints = True
        args.num_keypoints_per_class = [1]

        from rfdetr.datasets import build_roboflow

        dataset = build_roboflow("train", args, resolution=64)

        _, target = dataset[0]
        assert target["keypoints"].shape == (1, 1, 3)
        assert target["keypoints"][0, 0, 2].item() == pytest.approx(2.0)


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

    def test_segmentation_init_builds_coco_metadata_without_pixel_loading(self, tmp_path: Path) -> None:
        """Dataset construction must not decode pixel data for every image (only metadata is needed at init)."""
        image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(tmp_path)

        # ``Image.open`` is allowed during init to read header metadata (``image.size``),
        # but ``Image.Image.convert`` decodes the full pixel buffer and must not run until
        # ``__getitem__`` is invoked on the lazy dataset.
        with patch.object(
            Image.Image,
            "convert",
            side_effect=AssertionError("Image.convert should not run during init"),
        ):
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
        assert isinstance(dataset.coco, COCO)

    def test_init_raises_when_masks_and_keypoints_both_enabled(self, tmp_path: Path) -> None:
        """YoloDetection must reject include_masks=True + include_keypoints=True before any I/O."""
        with pytest.raises(ValueError, match="at the same time"):
            YoloDetection(
                img_folder=str(tmp_path / "images"),
                lb_folder=str(tmp_path / "labels"),
                data_file=str(tmp_path / "data.yaml"),
                transforms=None,
                include_masks=True,
                include_keypoints=True,
            )

    def test_detection_init_exposes_real_coco_api_indexes(self, tmp_path: Path) -> None:
        """`dataset.coco` should be a real pycocotools.COCO object with working indexes."""
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

        assert isinstance(dataset.coco, COCO)
        assert dataset.coco.getCatIds() == [0]
        assert dataset.coco.getImgIds() == [0]
        assert dataset.coco.getAnnIds(imgIds=[0], catIds=[0]) == [0]

    def test_pose_init_exposes_keypoint_coco_metadata_without_pixel_loading(self, tmp_path: Path) -> None:
        """YOLO pose construction should synthesize COCO keypoint metadata lazily."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)

        with patch.object(
            Image.Image,
            "convert",
            side_effect=AssertionError("Image.convert should not run during init"),
        ):
            dataset = YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_keypoints=True,
                num_keypoints_per_class=[2],
            )

        sample = dataset.sv_dataset.get_image_info(0)
        assert sample.keypoints.shape == (1, 2, 3)
        assert dataset.coco.dataset["categories"] == [
            {
                "id": 0,
                "name": "person",
                "supercategory": "none",
                "keypoints": ["left_eye", "right_eye"],
                "skeleton": [],
            }
        ]
        assert dataset.coco.dataset["annotations"][0]["num_keypoints"] == 1
        assert dataset.coco.dataset["annotations"][0]["keypoints"] == pytest.approx([2.0, 1.5, 2.0, 0.0, 0.0, 0.0])

    def test_pose_getitem_returns_keypoint_targets_with_visibility(self, tmp_path: Path) -> None:
        """YOLO pose labels with visibility should become RF-DETR keypoint targets."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[2],
        )

        _, target = dataset[0]

        assert target["boxes"][0].tolist() == pytest.approx([2.0, 1.5, 6.0, 4.5])
        assert target["labels"].tolist() == [0]
        assert target["keypoints"].shape == (1, 2, 3)
        torch.testing.assert_close(
            target["keypoints"][0],
            torch.tensor([[2.0, 1.5, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        assert "masks" not in target

    def test_pose_2d_keypoints_synthesize_visibility(self, tmp_path: Path) -> None:
        """YOLO pose labels without visibility should mark nonzero points visible and zero points absent."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=2)
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[2],
        )

        _, target = dataset[0]

        torch.testing.assert_close(
            target["keypoints"][0],
            torch.tensor([[2.0, 1.5, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        )

    def test_pose_background_image_has_empty_keypoint_tensor(self, tmp_path: Path) -> None:
        """YOLO pose background images should keep an empty keypoint tensor with the schema width."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (label_dir / "sample.txt").unlink()
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[2],
        )

        _, target = dataset[0]

        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        assert target["keypoints"].shape == (0, 2, 3)

    def test_pose_multi_instance_keypoints_stack_correctly(self, tmp_path: Path) -> None:
        """Multiple YOLO pose rows should stack boxes, labels, and keypoints per instance."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (label_dir / "sample.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.25 0.25 2 0 0 0\n0 0.25 0.25 0.25 0.25 0.5 0.5 1 0.75 0.75 2\n",
            encoding="utf-8",
        )
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[2],
        )

        _, target = dataset[0]

        assert target["boxes"].shape == (2, 4)
        assert target["labels"].tolist() == [0, 0]
        assert target["keypoints"].shape == (2, 2, 3)
        torch.testing.assert_close(target["keypoints"][1, :, 2], torch.tensor([1.0, 2.0]))

    def test_pose_malformed_keypoint_count_raises_clear_error(self, tmp_path: Path) -> None:
        """YOLO pose rows must match kpt_shape exactly."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 0.25 0.25\n", encoding="utf-8")

        with pytest.raises(ValueError, match="kpt_shape"):
            YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_keypoints=True,
                num_keypoints_per_class=[2],
            )

    @pytest.mark.parametrize(
        "bad_label, expected_match",
        [
            pytest.param(
                "0 0.5 0.5 0.5 0.5 0.5 0.5 3.0 0.5 0.5 2.0\n",
                "visibility values must be",
                id="visibility_out_of_range",
            ),
            pytest.param(
                "0 0.5 0.5 0.5 0.5 nan 0.5 2.0 0.5 0.5 2.0\n",
                "non-finite",
                id="nan_keypoint",
            ),
        ],
    )
    def test_pose_malformed_label_value_raises_clear_error(
        self, tmp_path: Path, bad_label: str, expected_match: str
    ) -> None:
        """Out-of-range visibility or NaN keypoints raise ValueError."""
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (label_dir / "sample.txt").write_text(bad_label, encoding="utf-8")

        with pytest.raises(ValueError, match=expected_match):
            YoloDetection(
                img_folder=str(image_dir),
                lb_folder=str(label_dir),
                data_file=str(data_file),
                transforms=None,
                include_keypoints=True,
            )

    def test_pose_dim3_out_of_bounds_coord_clamped_to_image_edge(self, tmp_path: Path) -> None:
        """Dim-3: OOB keypoint coords are clamped to [0, 1]; visibility flag unchanged.

        Roboflow exports sometimes annotate keypoints slightly outside the image frame. Clamping maps them to the
        nearest edge so training proceeds without crashing. Image fixture is 8 × 6 px.
        """
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        # kpt0: x=1.5 (OOB right), y=0.5, v=2 → clamp x to 1.0 → pixel x = 8.0
        # kpt1: x=-0.1 (OOB left), y=0.5, v=1 → clamp x to 0.0 → pixel x = 0.0; v=1 preserved
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 1.5 0.5 2.0 -0.1 0.5 1.0\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[2],
        )

        _, target = dataset[0]
        kpts = target["keypoints"]
        assert kpts.shape == (1, 2, 3)
        assert kpts[0, 0, 0].item() == pytest.approx(8.0)
        assert kpts[0, 0, 2].item() == pytest.approx(2.0)
        assert kpts[0, 1, 0].item() == pytest.approx(0.0)
        assert kpts[0, 1, 2].item() == pytest.approx(1.0)

    def test_pose_dim2_negative_coord_treated_as_absent(self, tmp_path: Path) -> None:
        """Dim-2: negative coordinate is the Ultralytics absent-keypoint sentinel.

        Per the YOLO pose spec, a negative x or y signals the keypoint is not labeled.  The parser must detect absence
        BEFORE clamping so that a keypoint like (-0.1, 0.5) is not clamped to (0.0, 0.5) and mistaken for a present
        keypoint at the left image edge. Image fixture is 8 × 6 px.
        """
        image_dir, label_dir, data_file = _write_yolo_pose_dataset(tmp_path, keypoint_dim=2)
        # kpt0: x=1.5 (OOB, positive) → clamp to 1.0 → pixel x=8; present → v=2
        # kpt1: x=-0.1 (negative absent signal), y=0.5 → absent → coords zeroed, v=0
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 1.5 0.5 -0.1 0.5\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[2],
        )

        _, target = dataset[0]
        kpts = target["keypoints"]
        assert kpts.shape == (1, 2, 3)
        # kpt0: present, clamped to right edge
        assert kpts[0, 0, 0].item() == pytest.approx(8.0)
        assert kpts[0, 0, 2].item() == pytest.approx(2.0)
        # kpt1: absent — coords zeroed, visibility 0
        assert kpts[0, 1, 0].item() == pytest.approx(0.0)
        assert kpts[0, 1, 1].item() == pytest.approx(0.0)
        assert kpts[0, 1, 2].item() == pytest.approx(0.0)

    def test_build_dataset_accepts_explicit_yolo_pose_file(self, tmp_path: Path) -> None:
        """dataset_file='yolo' should use the same pose path as Roboflow auto-detection."""
        root = tmp_path / "dataset"
        root.mkdir()
        _write_minimal_roboflow_yolo_dataset(root)
        (root / "data.yaml").write_text("names:\n  - person\nkpt_shape: [1, 3]\n", encoding="utf-8")
        (root / "train" / "labels" / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 0.5 0.5 2\n", encoding="utf-8")
        args = types.SimpleNamespace(
            dataset_dir=str(root),
            dataset_file="yolo",
            square_resize_div_64=False,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            aug_config={},
            augmentation_backend="cpu",
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[1],
            keypoint_flip_pairs=[],
        )

        from rfdetr.datasets import build_dataset

        dataset = build_dataset("train", args, resolution=64)

        _, target = dataset[0]
        assert target["keypoints"].shape == (1, 1, 3)

    def test_rfdetr_aligns_keypoint_schema_from_yolo_pose_yaml(self, tmp_path: Path) -> None:
        """Training setup should infer RF-DETR keypoint schema and flip pairs from YOLO pose metadata."""
        root = tmp_path / "dataset"
        root.mkdir()
        _write_minimal_roboflow_yolo_dataset(root)
        (root / "data.yaml").write_text(
            "names:\n  - person\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n",
            encoding="utf-8",
        )

        from rfdetr.config import KeypointTrainConfig, RFDETRKeypointPreviewConfig
        from rfdetr.detr import RFDETR

        model = object.__new__(RFDETR)
        model.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None)
        model.model = types.SimpleNamespace(args=types.SimpleNamespace(num_keypoints_per_class=[]))
        train_config = KeypointTrainConfig(dataset_dir=str(root), dataset_file="yolo", tensorboard=False)

        model._align_keypoint_schema_from_dataset(train_config)

        assert model.model_config.num_keypoints_per_class == [2]
        assert model.model.args.num_keypoints_per_class == [2]
        assert train_config.keypoint_flip_pairs == [0, 1]

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
    @pytest.mark.parametrize("include_masks", [pytest.param(True, id="masks"), pytest.param(False, id="no_masks")])
    def test_malformed_label_line_raises_clear_error(
        self, tmp_path: Path, label_content: str, match_pattern: str, include_masks: bool
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
                include_masks=include_masks,
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

    @pytest.mark.parametrize("include_masks", [pytest.param(True, id="masks"), pytest.param(False, id="no_masks")])
    def test_out_of_range_class_id_raises_clear_error(self, tmp_path: Path, include_masks: bool) -> None:
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
                include_masks=include_masks,
            )

    def test_include_masks_false_uses_lazy_detection_dataset(self, tmp_path: Path) -> None:
        """include_masks=False must use the lazy detection backend (not supervision's DetectionDataset)."""
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

        assert isinstance(dataset.sv_dataset, _LazyYoloDetectionDataset)
        assert len(dataset) == 1
        _, target = dataset[0]
        assert "boxes" in target
        assert "masks" not in target

    def test_detection_image_with_no_label_produces_empty_sample(self, tmp_path: Path) -> None:
        """Detection path: image without a .txt label file should produce an empty sample (background image)."""
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
            include_masks=False,
        )

        assert len(dataset) == 1
        sample = dataset.sv_dataset.get_image_info(0)
        assert sample.xyxy.shape == (0, 4)
        assert sample.class_id.shape == (0,)

        _, target = dataset[0]
        assert target["boxes"].shape == (0, 4)
        assert "masks" not in target

    def test_detection_background_and_labeled_images_counted_together(self, tmp_path: Path) -> None:
        """Detection path: dataset length includes both labeled and background images."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "labeled.png")
        Image.new("RGB", (8, 6), color=(0, 0, 0)).save(image_dir / "unlabeled.png")
        (label_dir / "labeled.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - carton\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=False,
        )

        assert len(dataset) == 2

        targets = [dataset[i][1] for i in range(2)]
        box_counts = sorted(t["boxes"].shape[0] for t in targets)
        assert box_counts == [0, 1], f"Expected one background and one annotated sample, got: {box_counts}"

    def test_detection_multi_instance_boxes_stack_correctly(self, tmp_path: Path) -> None:
        """Two bbox annotations per image should produce a (2, 4) boxes tensor with correct class IDs."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "two_boxes.png")
        # Two distinct non-overlapping bounding boxes
        (label_dir / "two_boxes.txt").write_text(
            "0 0.2 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n",
            encoding="utf-8",
        )
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - cat\n  - dog\n", encoding="utf-8")

        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=False,
        )

        _, target = dataset[0]
        assert target["boxes"].shape == (2, 4), f"Expected (2, 4), got {target['boxes'].shape}"
        assert set(target["labels"].tolist()) == {0, 1}

    def test_lazy_getitem_unreadable_image_raises_value_error(self, tmp_path: Path) -> None:
        """Lazy mask loading should raise ValueError when PIL cannot decode the image."""
        image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(tmp_path)
        dataset = YoloDetection(
            img_folder=str(image_dir),
            lb_folder=str(label_dir),
            data_file=str(data_file),
            transforms=None,
            include_masks=True,
        )

        # Replace the on-disk image with non-decodable bytes after dataset init has
        # already captured width/height from the original PNG header.
        (image_dir / "sample.png").write_bytes(b"not a valid image file")

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


def _write_ultralytics_yolo_dataset(tmp_path: Path) -> None:
    """Create a minimal Ultralytics-style YOLO dataset with val/ (not valid/) and yaml paths."""
    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(tmp_path / split / "images" / "sample.png")
        (tmp_path / split / "labels" / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


class TestResolveYoloSplitDirs:
    """Tests for _resolve_yolo_split_dirs: yaml-aware split path resolution (#1177)."""

    def test_roboflow_layout_without_yaml_paths(self, tmp_path: Path) -> None:
        """Roboflow export layout (valid/, no yaml paths) must keep working unchanged."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert img == tmp_path / "valid" / "images"
        assert lb == tmp_path / "valid" / "labels"

    def test_ultralytics_layout_val_from_yaml(self, tmp_path: Path) -> None:
        """Ultralytics layout (val/, yaml declares paths) must resolve from yaml."""
        _write_ultralytics_yolo_dataset(tmp_path)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert img == tmp_path / "val" / "images"
        assert lb == tmp_path / "val" / "labels"

    def test_ultralytics_layout_train_from_yaml(self, tmp_path: Path) -> None:
        """Train split resolved from yaml paths."""
        _write_ultralytics_yolo_dataset(tmp_path)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "train")
        assert img == tmp_path / "train" / "images"
        assert lb == tmp_path / "train" / "labels"

    def test_ultralytics_layout_test_from_yaml(self, tmp_path: Path) -> None:
        """Test split resolved from yaml paths."""
        _write_ultralytics_yolo_dataset(tmp_path)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "test")
        assert img == tmp_path / "test" / "images"
        assert lb == tmp_path / "test" / "labels"

    def test_yaml_path_key_resolves_relative_to_root(self, tmp_path: Path) -> None:
        """Ultralytics 'path' key in yaml is used as base for relative split dirs."""
        sub = tmp_path / "datasets" / "coco128"
        sub.mkdir(parents=True)
        for split in ("train", "val"):
            (sub / split / "images").mkdir(parents=True)
            (sub / split / "labels").mkdir(parents=True)
        data_file = tmp_path / "data.yaml"
        data_file.write_text(
            f"path: {sub}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert img == sub / "val" / "images"
        assert lb == sub / "val" / "labels"

    def test_images_suffix_swapped_to_labels(self, tmp_path: Path) -> None:
        """Yaml path ending with /images must have labels dir derived by swapping last component."""
        _write_ultralytics_yolo_dataset(tmp_path)
        data_file = tmp_path / "data.yaml"
        _, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert lb.name == "labels"
        assert lb.parent == (tmp_path / "val")

    def test_fallback_val_dir_when_no_yaml_paths_and_no_valid(self, tmp_path: Path) -> None:
        """When yaml has no path keys and valid/ does not exist but val/ does, use val/."""
        (tmp_path / "data.yaml").write_text("names:\n  0: person\n", encoding="utf-8")
        for split in ("train", "val"):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "labels").mkdir(parents=True)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert img == tmp_path / "val" / "images"
        assert lb == tmp_path / "val" / "labels"

    def test_yaml_declared_path_absent_on_disk_falls_back_to_roboflow(self, tmp_path: Path) -> None:
        """Yaml declares a split path that does not exist; must fall back to Roboflow convention."""
        (tmp_path / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: nonexistent/images\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        (tmp_path / "valid" / "images").mkdir(parents=True)
        (tmp_path / "valid" / "labels").mkdir(parents=True)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert img == tmp_path / "valid" / "images"
        assert lb == tmp_path / "valid" / "labels"

    def test_test_split_falls_back_when_yaml_has_no_test_key(self, tmp_path: Path) -> None:
        """When yaml exists but has no 'test' key, return root/test/images convention."""
        (tmp_path / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: val/images\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, "test")
        assert img == tmp_path / "test" / "images"
        assert lb == tmp_path / "test" / "labels"

    def test_path_traversal_in_yaml_does_not_escape_root(self, tmp_path: Path) -> None:
        """Traversal sequence in yaml split path must not escape dataset root."""
        outside = tmp_path.parent / "outside_dataset" / "images"
        outside_labels = tmp_path.parent / "outside_dataset" / "labels"
        outside.mkdir(parents=True)
        outside_labels.mkdir(parents=True)
        (tmp_path / "data.yaml").write_text(
            "path: .\nval: ../outside_dataset/images\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        data_file = tmp_path / "data.yaml"
        img, _ = _resolve_yolo_split_dirs(tmp_path, data_file, "val")
        assert not str(img).startswith(str(tmp_path.parent / "outside_dataset"))


class TestIsValidYoloDatasetUltralytics:
    """is_valid_yolo_dataset must accept both val/ (Ultralytics) and valid/ (Roboflow) layouts."""

    def test_ultralytics_val_dir_accepted(self, tmp_path: Path) -> None:
        """Dataset with val/ instead of valid/ should be recognized as valid YOLO."""
        (tmp_path / "data.yaml").touch()
        for split in ["train", "val"]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        assert is_valid_yolo_dataset(str(tmp_path)) is True

    def test_roboflow_valid_dir_still_accepted(self, tmp_path: Path) -> None:
        """Existing Roboflow layout (valid/) must keep working."""
        (tmp_path / "data.yaml").touch()
        for split in ["train", "valid"]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        assert is_valid_yolo_dataset(str(tmp_path)) is True


class TestBuildRoboflowFromYoloUltralytics:
    """build_roboflow_from_yolo must handle Ultralytics YOLO layout (#1177)."""

    def _make_args(self, dataset_dir: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            dataset_dir=dataset_dir,
            square_resize_div_64=False,
            aug_config=None,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=None,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            augmentation_backend="cpu",
            use_grouppose_keypoints=False,
            num_keypoints_per_class=[],
            keypoint_flip_pairs=[],
        )

    def test_ultralytics_val_split_resolves_correctly(self, tmp_path: Path) -> None:
        """Val split on Ultralytics layout (val/, yaml paths) should not raise."""
        _write_ultralytics_yolo_dataset(tmp_path)
        args = self._make_args(str(tmp_path))

        with (
            patch("rfdetr.datasets.yolo.make_coco_transforms") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
        ):
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("val", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert kwargs["img_folder"] == str(tmp_path / "val" / "images")

    def test_roboflow_layout_still_works(self, tmp_path: Path) -> None:
        """Roboflow export layout (valid/) must keep working after the change."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        args = self._make_args(str(tmp_path))

        with (
            patch("rfdetr.datasets.yolo.make_coco_transforms") as mock_transform,
            patch("rfdetr.datasets.yolo.YoloDetection") as mock_dataset,
        ):
            mock_transform.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()

            from rfdetr.datasets.yolo import build_roboflow_from_yolo

            build_roboflow_from_yolo("val", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert "valid" in kwargs["img_folder"]


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

        The downstream range check in _parse_yolo_label_line assumes class IDs are a contiguous 0..N-1 range.  Silently
        accepting sparse keys would cause valid label files to be rejected during parsing (e.g. class ID 2 in a 2-class
        dataset built from {0: cat, 2: dog} would exceed the num_classes bound).
        """
        data_file = tmp_path / "data.yaml"
        data_file.write_text(yaml_content, encoding="utf-8")
        with pytest.raises(ValueError, match="0..N-1"):
            _extract_yolo_class_names(str(data_file))
