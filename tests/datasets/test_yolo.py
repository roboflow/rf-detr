# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO

from rfdetr.config import KeypointTrainConfig, RFDETRKeypointPreviewConfig
from rfdetr.datasets import build_dataset
from rfdetr.datasets.yolo import (
    YoloDetection,
    YoloSplitUnavailableError,
    _extract_yolo_class_names,
    _LazyYoloDetectionDataset,
    _parse_yaml_split_dirs,
    _resolve_yolo_split_dirs,
    build_roboflow_from_yolo,
    is_valid_yolo_dataset,
)
from rfdetr.detr import RFDETR


def _write_minimal_roboflow_yolo_dataset(tmp_path: Path) -> None:
    """Create a minimal Roboflow YOLO dataset root.

    Example:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> _write_minimal_roboflow_yolo_dataset(root)
        >>> (root / "data.yaml").exists()
        True
    """
    (tmp_path / "data.yaml").write_text("names:\n  - person\n", encoding="utf-8")
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        Image.new("RGB", (8, 6), color=(255, 255, 255)).save(tmp_path / split / "images" / "sample.png")
        (tmp_path / split / "labels" / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def _write_yolo_segmentation_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal YOLO segmentation dataset on disk.

    Example:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> image_dir, label_dir, data_file = _write_yolo_segmentation_dataset(root)
        >>> image_dir.name, label_dir.name, data_file.name
        ('images', 'labels', 'data.yaml')
    """
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
    """Create a minimal YOLO pose dataset on disk.

    Example:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> _, _, data_file = _write_yolo_pose_dataset(root)
        >>> data_file.exists()
        True
    """
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
        """Build the builder namespace for a fake dataset with the given resize mode and ``aug_config``."""
        return types.SimpleNamespace(
            dataset_dir="/fake/dataset",
            square_resize_div_64=square_resize_div_64,
            aug_config=aug_config,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=None,
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
        self, mocker, square_resize_div_64: bool, transform_fn: str, aug_config: object
    ) -> None:
        """Regression test for #769: aug_config is forwarded to transform builders for all code paths."""
        args = self._make_args(square_resize_div_64=square_resize_div_64, aug_config=aug_config)
        fake_dirs = (Path("/fake/dataset/train/images"), Path("/fake/dataset/train/labels"))
        mock_path = mocker.patch("rfdetr.datasets.yolo.Path")
        mock_transform = mocker.patch(f"rfdetr.datasets.yolo.{transform_fn}")
        mocker.patch("rfdetr.datasets.yolo.YoloDetection")
        mocker.patch("rfdetr.datasets.yolo._resolve_yolo_split_dirs_with_notes", return_value=(fake_dirs, ()))
        mock_path.return_value.exists.return_value = True

        build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_transform.call_args
        assert kwargs.get("aug_config") == aug_config, (
            f"{transform_fn} was not called with aug_config={aug_config!r}; got {kwargs}"
        )

    def test_data_yml_selected_when_data_yaml_missing(self, mocker, tmp_path: Path) -> None:
        """Regression test: build_roboflow_from_yolo picks data.yml when data.yaml is not present."""
        (tmp_path / "data.yml").touch()
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.dataset_dir = str(tmp_path)
        mock_dataset = mocker.patch("rfdetr.datasets.yolo.YoloDetection")
        mocker.patch("rfdetr.datasets.yolo.make_coco_transforms")

        build_roboflow_from_yolo("train", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert kwargs["data_file"] == str(tmp_path / "data.yml")

    def test_auto_no_cuda_sets_gpu_postprocess_false(self, mocker) -> None:
        """Auto + no CUDA must keep CPU normalize by passing gpu_postprocess=False."""
        args = self._make_args(square_resize_div_64=False, aug_config=None)
        args.augmentation_backend = "auto"
        fake_dirs = (Path("/fake/dataset/train/images"), Path("/fake/dataset/train/labels"))
        mock_path = mocker.patch("rfdetr.datasets.yolo.Path")
        mock_transform = mocker.patch("rfdetr.datasets.yolo.make_coco_transforms")
        mocker.patch("rfdetr.datasets.yolo.YoloDetection")
        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False)
        mocker.patch("rfdetr.datasets.yolo._resolve_yolo_split_dirs_with_notes", return_value=(fake_dirs, ()))
        mock_path.return_value.exists.return_value = True

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

        with pytest.raises(ValueError, match="kpt_shape"):
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


def _write_yolo_detection_dataset(
    tmp_path: Path, label_text: str | None = "0 0.5 0.5 0.5 0.5\n", names: tuple[str, ...] = ("carton",)
) -> tuple[Path, Path, Path]:
    """Create a one-image (8 x 6 white ``sample.png``) YOLO dataset.

    Args:
        tmp_path: Directory receiving ``images/``, ``labels/`` and ``data.yaml``.
        label_text: Content of ``labels/sample.txt``; ``None`` writes no label file (a background image).
        names: Class names written to ``data.yaml``.

    Returns:
        ``(image_dir, label_dir, data_file)``.

    Examples:
        >>> import tempfile
        >>> image_dir, label_dir, data_file = _write_yolo_detection_dataset(Path(tempfile.mkdtemp()), None)
        >>> list(label_dir.iterdir())
        []
        >>> data_file.read_text(encoding="utf-8").splitlines()
        ['names:', '  - carton']
    """
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / "sample.png")
    if label_text is not None:
        (label_dir / "sample.txt").write_text(label_text, encoding="utf-8")
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names:\n" + "".join(f"  - {name}\n" for name in names), encoding="utf-8")
    return image_dir, label_dir, data_file


def _build_yolo_dataset(paths: tuple[Path, Path, Path], **kwargs: object) -> YoloDetection:
    """Open the ``(image_dir, label_dir, data_file)`` triple written by a ``_write_yolo_*`` helper.

    Examples:
        >>> import contextlib, io, tempfile
        >>> paths = _write_yolo_detection_dataset(Path(tempfile.mkdtemp()))
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     dataset = _build_yolo_dataset(paths, include_masks=False)
        >>> len(dataset)
        1
    """
    image_dir, label_dir, data_file = paths
    return YoloDetection(
        img_folder=str(image_dir), lb_folder=str(label_dir), data_file=str(data_file), transforms=None, **kwargs
    )


class TestYoloDetectionLazyMasks:
    """Segmentation masks and pose keypoints stay lightweight until a sample is fetched."""

    def test_segmentation_init_reads_metadata_without_decoding_pixels(self, mocker, tmp_path: Path) -> None:
        """Init reads only image headers; ``Image.convert`` (full pixel decode) must wait for ``__getitem__``."""
        paths = _write_yolo_segmentation_dataset(tmp_path)
        mocker.patch.object(Image.Image, "convert", side_effect=AssertionError("Image.convert should not run in init"))

        dataset = _build_yolo_dataset(paths, include_masks=True)

        sample = dataset.sv_dataset.get_image_info(0)
        assert sample.width == 8
        assert sample.height == 6
        assert sample.xyxy.shape == (1, 4)
        assert len(sample.polygons) == 1
        assert dataset.coco.dataset["images"] == [
            {"id": 0, "file_name": str(paths[0] / "sample.png"), "height": 6, "width": 8}
        ]
        assert dataset.coco.dataset["annotations"][0]["segmentation"] == []
        assert isinstance(dataset.coco, COCO)

    def test_masks_and_keypoints_together_raise_before_io(self, tmp_path: Path) -> None:
        """``include_masks=True`` with ``include_keypoints=True`` is rejected before any file is read."""
        paths = (tmp_path / "images", tmp_path / "labels", tmp_path / "data.yaml")

        with pytest.raises(ValueError) as exc_info:
            _build_yolo_dataset(paths, include_masks=True, include_keypoints=True)

        assert "at the same time" in str(exc_info.value)

    def test_detection_init_exposes_real_coco_api_indexes(self, tmp_path: Path) -> None:
        """``dataset.coco`` is a real pycocotools ``COCO`` object with working indexes."""
        dataset = _build_yolo_dataset(_write_yolo_detection_dataset(tmp_path), include_masks=False)

        assert isinstance(dataset.coco, COCO)
        assert dataset.coco.getCatIds() == [0]
        assert dataset.coco.getImgIds() == [0]
        assert dataset.coco.getAnnIds(imgIds=[0], catIds=[0]) == [0]

    def test_pose_init_exposes_keypoint_coco_metadata_without_decoding_pixels(self, mocker, tmp_path: Path) -> None:
        """Pose construction synthesizes COCO keypoint metadata from label files without decoding pixels."""
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        mocker.patch.object(Image.Image, "convert", side_effect=AssertionError("Image.convert should not run in init"))

        dataset = _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

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

    @pytest.mark.parametrize("keypoint_dim", [3, 2])
    def test_pose_getitem_returns_keypoint_targets(self, tmp_path: Path, keypoint_dim: int) -> None:
        """Pose labels become keypoint targets; without a visibility column nonzero points are visible, zeros absent."""
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=keypoint_dim)
        dataset = _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = dataset[0]

        assert target["boxes"][0].tolist() == pytest.approx([2.0, 1.5, 6.0, 4.5])
        assert target["labels"].tolist() == [0]
        assert target["keypoints"].shape == (1, 2, 3)
        torch.testing.assert_close(
            target["keypoints"][0], torch.tensor([[2.0, 1.5, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        )
        assert "masks" not in target

    def test_pose_background_image_has_empty_keypoint_tensor(self, tmp_path: Path) -> None:
        """A pose background image keeps an empty keypoint tensor with the schema width."""
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (paths[1] / "sample.txt").unlink()
        dataset = _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = dataset[0]

        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        assert target["keypoints"].shape == (0, 2, 3)

    def test_pose_multi_instance_keypoints_stack(self, tmp_path: Path) -> None:
        """Multiple pose rows stack boxes, labels and keypoints per instance."""
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (paths[1] / "sample.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.25 0.25 2 0 0 0\n0 0.25 0.25 0.25 0.25 0.5 0.5 1 0.75 0.75 2\n", encoding="utf-8"
        )
        dataset = _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = dataset[0]

        assert target["boxes"].shape == (2, 4)
        assert target["labels"].tolist() == [0, 0]
        assert target["keypoints"].shape == (2, 2, 3)
        torch.testing.assert_close(target["keypoints"][1, :, 2], torch.tensor([1.0, 2.0]))

    def test_pose_malformed_keypoint_count_raises(self, tmp_path: Path) -> None:
        """A pose row whose keypoint count differs from ``kpt_shape`` raises a ValueError naming ``kpt_shape``."""
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (paths[1] / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 0.25 0.25\n", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

        assert "kpt_shape" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("bad_label", "expected_message"),
        [
            pytest.param(
                "0 0.5 0.5 0.5 0.5 0.5 0.5 3.0 0.5 0.5 2.0\n", "visibility values must be", id="visibility-out-of-range"
            ),
            pytest.param("0 0.5 0.5 0.5 0.5 nan 0.5 2.0 0.5 0.5 2.0\n", "non-finite", id="nan-keypoint"),
        ],
    )
    def test_pose_malformed_label_value_raises(self, tmp_path: Path, bad_label: str, expected_message: str) -> None:
        """Out-of-range visibility or NaN keypoints raise ValueError."""
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (paths[1] / "sample.txt").write_text(bad_label, encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            _build_yolo_dataset(paths, include_keypoints=True)

        assert expected_message in str(exc_info.value)

    def test_pose_dim3_out_of_bounds_coord_clamped_to_image_edge(self, tmp_path: Path) -> None:
        """Dim-3: out-of-frame keypoint coords clamp to the nearest image edge and the visibility flag is unchanged.

        Roboflow exports sometimes annotate keypoints slightly outside the frame; clamping lets training proceed. The 8
        x 6 px image maps kpt0 x=1.5 to pixel 8.0 (v=2) and kpt1 x=-0.1 to pixel 0.0 (v=1 preserved).
        """
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=3)
        (paths[1] / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 1.5 0.5 2.0 -0.1 0.5 1.0\n", encoding="utf-8")
        dataset = _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = dataset[0]

        kpts = target["keypoints"]
        assert kpts.shape == (1, 2, 3)
        assert kpts[0, 0, 0].item() == pytest.approx(8.0)
        assert kpts[0, 0, 2].item() == pytest.approx(2.0)
        assert kpts[0, 1, 0].item() == pytest.approx(0.0)
        assert kpts[0, 1, 2].item() == pytest.approx(1.0)

    def test_pose_dim2_negative_coord_treated_as_absent(self, tmp_path: Path) -> None:
        """Dim-2: a negative coordinate is the Ultralytics absent-keypoint sentinel, detected before clamping.

        Clamping first would turn (-0.1, 0.5) into a present keypoint on the left edge. Here kpt0 x=1.5 clamps to pixel
        8 (v=2) while kpt1 x=-0.1 is absent (coords zeroed, v=0). The image is 8 x 6 px.
        """
        paths = _write_yolo_pose_dataset(tmp_path, keypoint_dim=2)
        (paths[1] / "sample.txt").write_text("0 0.5 0.5 0.5 0.5 1.5 0.5 -0.1 0.5\n", encoding="utf-8")
        dataset = _build_yolo_dataset(paths, include_keypoints=True, num_keypoints_per_class=[2])

        _, target = dataset[0]

        kpts = target["keypoints"]
        assert kpts.shape == (1, 2, 3)
        assert kpts[0, 0, 0].item() == pytest.approx(8.0)
        assert kpts[0, 0, 2].item() == pytest.approx(2.0)
        assert kpts[0, 1, 0].item() == pytest.approx(0.0)
        assert kpts[0, 1, 1].item() == pytest.approx(0.0)
        assert kpts[0, 1, 2].item() == pytest.approx(0.0)

    def test_build_dataset_accepts_explicit_yolo_pose_file(self, tmp_path: Path) -> None:
        """``dataset_file='yolo'`` takes the same pose path as Roboflow auto-detection."""
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
            patch_size=16,
            num_windows=4,
            aug_config={},
            augmentation_backend="cpu",
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[1],
            keypoint_flip_pairs=[],
        )

        _, target = build_dataset("train", args, resolution=64)[0]

        assert target["keypoints"].shape == (1, 1, 3)

    def test_rfdetr_aligns_keypoint_schema_from_yolo_pose_yaml(self, tmp_path: Path) -> None:
        """Training setup infers the RF-DETR keypoint schema and flip pairs from YOLO pose metadata."""
        root = tmp_path / "dataset"
        root.mkdir()
        _write_minimal_roboflow_yolo_dataset(root)
        (root / "data.yaml").write_text(
            "names:\n  - person\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n",
            encoding="utf-8",
        )
        model = object.__new__(RFDETR)
        model.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None)
        model.model = types.SimpleNamespace(args=types.SimpleNamespace(num_keypoints_per_class=[]))
        train_config = KeypointTrainConfig(dataset_dir=str(root), dataset_file="yolo", tensorboard=False)

        model._align_keypoint_schema_from_dataset(train_config)

        assert model.model_config.num_keypoints_per_class == [2]
        assert model.model.args.num_keypoints_per_class == [2]
        assert train_config.keypoint_flip_pairs == [0, 1]

    def test_segmentation_masks_are_materialized_per_sample_fetch(self, tmp_path: Path) -> None:
        """Fetching a sample creates the dense boolean mask tensor expected downstream."""
        dataset = _build_yolo_dataset(_write_yolo_segmentation_dataset(tmp_path), include_masks=True)

        _, target = dataset[0]

        assert target["masks"].dtype == torch.bool
        assert target["masks"].shape == (1, 6, 8)
        assert torch.count_nonzero(target["masks"]) > 0
        assert target["boxes"][0].tolist() == pytest.approx([2.0, 1.5, 6.0, 4.5])

    def test_segmentation_image_without_label_yields_empty_sample(self, tmp_path: Path) -> None:
        """An image with no matching ``.txt`` label file yields an empty sample and empty mask and box tensors."""
        dataset = _build_yolo_dataset(_write_yolo_detection_dataset(tmp_path, None), include_masks=True)

        sample = dataset.sv_dataset.get_image_info(0)
        assert sample.xyxy.shape == (0, 4)
        assert sample.class_id.shape == (0,)
        assert sample.polygons == ()
        _, target = dataset[0]
        assert target["masks"].shape == (0, 6, 8)
        assert target["boxes"].shape == (0, 4)

    def test_segmentation_multi_instance_polygons_stack(self, tmp_path: Path) -> None:
        """Two non-overlapping polygons in one image produce masks of shape (2, H, W)."""
        label_text = "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
        dataset = _build_yolo_dataset(
            _write_yolo_detection_dataset(tmp_path, label_text, names=("cat", "dog")), include_masks=True
        )

        _, target = dataset[0]

        assert target["masks"].shape == (2, 6, 8), f"Expected (2, 6, 8), got {target['masks'].shape}"
        assert target["masks"].dtype == torch.bool

    @pytest.mark.parametrize(
        ("label_text", "expected_message"),
        [
            pytest.param("0\n", "Malformed label", id="only-class-id"),
            pytest.param("0 0.1 0.2 0.3\n", "Malformed label", id="too-few-fields"),
            pytest.param("0 0.1 0.2 0.3 0.4 0.5\n", "Malformed polygon", id="odd-polygon-coords"),
            # Dataset defines one class (id 0); the label references class 5.
            pytest.param("5 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", "out of range", id="class-id-out-of-range"),
            pytest.param("cat 0.5 0.5 0.25 0.25\n", "invalid class ID", id="non-integer-class-id"),
        ],
    )
    @pytest.mark.parametrize("include_masks", [True, False])
    def test_invalid_label_raises_value_error(
        self, tmp_path: Path, label_text: str, expected_message: str, include_masks: bool
    ) -> None:
        """Malformed lines, out-of-range class ids and non-integer class ids raise a descriptive ValueError."""
        paths = _write_yolo_detection_dataset(tmp_path, label_text)

        with pytest.raises(ValueError) as exc_info:
            _build_yolo_dataset(paths, include_masks=include_masks)

        assert expected_message in str(exc_info.value)

    def test_lazy_dataset_polygon_storage_is_smaller_than_eager_masks(self, tmp_path: Path) -> None:
        """The lazy dataset retains polygon coords, not dense masks, so its footprint is orders of magnitude smaller."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        n_images = 20
        width, height = 256, 256
        for i in range(n_images):
            Image.new("RGB", (width, height)).save(image_dir / f"img_{i:03d}.png")
            (label_dir / f"img_{i:03d}.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  - obj\n", encoding="utf-8")
        dataset = _build_yolo_dataset((image_dir, label_dir, data_file), include_masks=True)

        # Bytes retained in the lazy samples (polygon coords + bbox + class id) vs one eager bool mask per image.
        lazy_bytes = sum(
            dataset.sv_dataset.get_image_info(i).xyxy.nbytes
            + dataset.sv_dataset.get_image_info(i).class_id.nbytes
            + sum(p.nbytes for p in dataset.sv_dataset.get_image_info(i).polygons)
            for i in range(len(dataset.sv_dataset))
        )
        eager_mask_bytes = n_images * height * width * np.dtype(bool).itemsize

        assert lazy_bytes < eager_mask_bytes / 10, (
            f"Lazy storage ({lazy_bytes} B) should be at least 10× smaller than eager mask cost ({eager_mask_bytes} B)."
        )

    def test_include_masks_false_uses_lazy_detection_dataset(self, tmp_path: Path) -> None:
        """``include_masks=False`` uses the lazy detection backend, not supervision's ``DetectionDataset``."""
        dataset = _build_yolo_dataset(_write_yolo_detection_dataset(tmp_path), include_masks=False)

        _, target = dataset[0]

        assert isinstance(dataset.sv_dataset, _LazyYoloDetectionDataset)
        assert len(dataset) == 1
        assert "boxes" in target
        assert "masks" not in target

    def test_detection_image_without_label_yields_empty_sample(self, tmp_path: Path) -> None:
        """Detection: an image without a ``.txt`` label file is a background image with an empty sample."""
        dataset = _build_yolo_dataset(_write_yolo_detection_dataset(tmp_path, None), include_masks=False)

        sample = dataset.sv_dataset.get_image_info(0)
        _, target = dataset[0]

        assert len(dataset) == 1
        assert sample.xyxy.shape == (0, 4)
        assert sample.class_id.shape == (0,)
        assert target["boxes"].shape == (0, 4)
        assert "masks" not in target

    def test_detection_background_and_labeled_images_counted_together(self, tmp_path: Path) -> None:
        """Detection: dataset length includes both labeled and background images."""
        paths = _write_yolo_detection_dataset(tmp_path)
        Image.new("RGB", (8, 6), color=(0, 0, 0)).save(paths[0] / "unlabeled.png")
        dataset = _build_yolo_dataset(paths, include_masks=False)

        box_counts = sorted(dataset[i][1]["boxes"].shape[0] for i in range(len(dataset)))

        assert len(dataset) == 2
        assert box_counts == [0, 1], f"Expected one background and one annotated sample, got: {box_counts}"

    def test_detection_multi_instance_boxes_stack(self, tmp_path: Path) -> None:
        """Two non-overlapping boxes in one image produce a (2, 4) boxes tensor with both class ids."""
        label_text = "0 0.2 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n"
        dataset = _build_yolo_dataset(
            _write_yolo_detection_dataset(tmp_path, label_text, names=("cat", "dog")), include_masks=False
        )

        _, target = dataset[0]

        assert target["boxes"].shape == (2, 4), f"Expected (2, 4), got {target['boxes'].shape}"
        assert set(target["labels"].tolist()) == {0, 1}

    def test_lazy_getitem_jpeg_matches_pillow(self, tmp_path: Path) -> None:
        """A JPEG sample decodes to the same uint8 RGB array Pillow produces, whichever decoder handled it."""
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        image_path = image_dir / "sample.jpg"
        noise = np.random.default_rng(0).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
        Image.fromarray(noise).save(image_path, quality=90)
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        data_file = tmp_path / "data.yaml"
        data_file.write_text("names:\n  0: carton\n", encoding="utf-8")
        with Image.open(image_path) as image:
            expected = np.asarray(image.convert("RGB"))
        dataset = _build_yolo_dataset((image_dir, label_dir, data_file))

        _, rgb_image, _ = dataset.sv_dataset[0]

        assert rgb_image.dtype == np.uint8
        np.testing.assert_array_equal(rgb_image, expected)

    def test_lazy_getitem_unreadable_image_raises_value_error(self, tmp_path: Path) -> None:
        """Lazy mask loading raises ValueError when PIL cannot decode the image."""
        paths = _write_yolo_segmentation_dataset(tmp_path)
        dataset = _build_yolo_dataset(paths, include_masks=True)
        # Init already captured width/height from the PNG header; swap in non-decodable bytes afterwards.
        (paths[0] / "sample.png").write_bytes(b"not a valid image file")

        with pytest.raises(ValueError) as exc_info:
            dataset[0]

        assert "Could not read image" in str(exc_info.value)


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

    @pytest.mark.parametrize("split", ["val", "train", "test"])
    def test_ultralytics_layout_split_from_yaml(self, tmp_path: Path, split: str) -> None:
        """Ultralytics layout (val/, yaml declares paths) resolves each split's images/labels dirs from yaml."""
        _write_ultralytics_yolo_dataset(tmp_path)
        data_file = tmp_path / "data.yaml"
        img, lb = _resolve_yolo_split_dirs(tmp_path, data_file, split)
        assert img == tmp_path / split / "images"
        assert lb == tmp_path / split / "labels"

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

    def test_labels_dir_swaps_last_images_segment(self, tmp_path: Path) -> None:
        """An ``images`` directory above the dataset root must not be the segment swapped to ``labels``."""
        root = tmp_path / "images" / "coco8"
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
        data_file = root / "data.yaml"
        data_file.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        img, lb = _resolve_yolo_split_dirs(root, data_file, "val")
        assert img == root / "images" / "val"
        assert lb == root / "labels" / "val"

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

    def test_split_labels_symlink_outside_root_rejected(self, tmp_path: Path) -> None:
        """A labels directory symlinked outside root must not be returned."""
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        (root / "val" / "images").mkdir(parents=True)
        outside.mkdir(parents=True)
        (outside / "leak.txt").write_text("secret", encoding="utf-8")
        try:
            (root / "val" / "labels").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"cannot create symlinks in this environment: {exc}")
        data_file = root / "data.yaml"
        data_file.write_text("path: .\nval: val/images\nnames:\n  0: person\n", encoding="utf-8")
        assert _parse_yaml_split_dirs(root, data_file, "val")[0] is None

    def test_nested_split_with_no_images_segment_resolves_via_subdirectories(self, tmp_path: Path) -> None:
        """A split path with no ``images`` segment under an images-named ancestor still resolves."""
        root = tmp_path / "images" / "coco8"
        (root / "splits" / "val" / "images").mkdir(parents=True)
        (root / "splits" / "val" / "labels").mkdir(parents=True)
        data_file = root / "data.yaml"
        data_file.write_text("path: .\nval: splits/val\nnames:\n  0: person\n", encoding="utf-8")
        img, lb = _resolve_yolo_split_dirs(root, data_file, "val")
        assert img == root / "splits" / "val" / "images"
        assert lb == root / "splits" / "val" / "labels"

    def test_three_images_occurrences_swaps_rightmost(self, tmp_path: Path) -> None:
        """With three ``images`` segments, the rightmost one is swapped, not an ancestor."""
        root = tmp_path / "images" / "images" / "coco8"
        (root / "val" / "images").mkdir(parents=True)
        (root / "val" / "labels").mkdir(parents=True)
        data_file = root / "data.yaml"
        data_file.write_text("path: .\nval: val/images\nnames:\n  0: person\n", encoding="utf-8")
        img, lb = _resolve_yolo_split_dirs(root, data_file, "val")
        assert img == root / "val" / "images"
        assert lb == root / "val" / "labels"

    def test_capitalized_images_segment_is_not_matched(self, tmp_path: Path) -> None:
        """The ``images`` segment match is case-exact; a capitalized ancestor is not swapped."""
        root = tmp_path / "coco8"
        (root / "Images" / "val").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)
        data_file = root / "data.yaml"
        data_file.write_text("path: .\nval: Images/val\nnames:\n  0: person\n", encoding="utf-8")
        assert _parse_yaml_split_dirs(root, data_file, "val")[0] is None

    def test_bare_leaf_images_split_with_empty_tail(self, tmp_path: Path) -> None:
        """A split path that is itself a bare ``images`` leaf swaps with an empty tail."""
        root = tmp_path / "images" / "coco8"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        data_file = root / "data.yaml"
        data_file.write_text("path: .\nval: images\nnames:\n  0: person\n", encoding="utf-8")
        img, lb = _resolve_yolo_split_dirs(root, data_file, "val")
        assert img == root / "images"
        assert lb == root / "labels"


class TestIsValidYoloDatasetUltralytics:
    """is_valid_yolo_dataset must accept both val/ (Ultralytics) and valid/ (Roboflow) layouts."""

    @pytest.mark.parametrize("val_dir", ["val", "valid"])
    def test_validation_dir_name_accepted(self, tmp_path: Path, val_dir: str) -> None:
        """Both the Ultralytics ``val/`` and the Roboflow ``valid/`` validation directory are recognized as valid."""
        (tmp_path / "data.yaml").touch()
        for split in ["train", val_dir]:
            for subdir in ["images", "labels"]:
                (tmp_path / split / subdir).mkdir(parents=True)
        assert is_valid_yolo_dataset(str(tmp_path)) is True


class TestBuildRoboflowFromYoloUltralytics:
    """build_roboflow_from_yolo must handle Ultralytics YOLO layout (#1177)."""

    @staticmethod
    def _write_yolo_dataset(
        tmp_path: Path,
        *,
        images: bool = True,
        labels: bool = True,
        image_name: str = "sample.png",
        label_content: str | None = None,
        test_yaml_path: str | None = None,
        create_test_images_dir: bool = True,
        image_symlink_target: Path | None = None,
    ) -> tuple[Path, Path]:
        """Create the YOLO dataset layout used by builder regression tests.

        Examples:
            >>> import tempfile
            >>> root = Path(tempfile.mkdtemp())
            >>> image_dir, label_dir = TestBuildRoboflowFromYoloUltralytics._write_yolo_dataset(root, images=False)
            >>> image_dir.is_dir(), label_dir.is_dir()
            (True, True)
        """
        yaml_text = "names:\n  0: person\n"
        if test_yaml_path is not None:
            yaml_text = f"path: .\nval: val/images\ntest: {test_yaml_path}\nnames:\n  0: person\n"
            (tmp_path / "val" / "images").mkdir(parents=True)
            (tmp_path / "val" / "labels").mkdir()
        (tmp_path / "data.yaml").write_text(yaml_text, encoding="utf-8")

        image_dir = tmp_path / (test_yaml_path or "test/images")
        label_dir = image_dir.parent / "labels"
        if image_symlink_target is not None:
            image_dir.parent.mkdir(parents=True)
            image_dir.symlink_to(image_symlink_target, target_is_directory=True)
        elif create_test_images_dir:
            image_dir.mkdir(parents=True)
        if images:
            Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_dir / image_name)
        if labels:
            label_dir.mkdir(parents=True)
            if label_content is not None:
                label_dir.joinpath(Path(image_name).with_suffix(".txt")).write_text(label_content, encoding="utf-8")
        return image_dir, label_dir

    def _make_args(self, dataset_dir: str) -> types.SimpleNamespace:
        """Build the builder namespace for a CPU-backend, detection-only run on ``dataset_dir``."""
        return types.SimpleNamespace(
            dataset_dir=dataset_dir,
            square_resize_div_64=False,
            aug_config=None,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=None,
            patch_size=16,
            num_windows=4,
            augmentation_backend="cpu",
            use_grouppose_keypoints=False,
            num_keypoints_per_class=[],
            keypoint_flip_pairs=[],
        )

    def test_ultralytics_val_split_resolves_correctly(self, mocker, tmp_path: Path) -> None:
        """Val split on Ultralytics layout (val/, yaml paths) should not raise."""
        _write_ultralytics_yolo_dataset(tmp_path)
        args = self._make_args(str(tmp_path))
        mock_dataset = mocker.patch("rfdetr.datasets.yolo.YoloDetection")
        mocker.patch("rfdetr.datasets.yolo.make_coco_transforms")

        build_roboflow_from_yolo("val", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert kwargs["img_folder"] == str(tmp_path / "val" / "images")

    def test_roboflow_layout_still_works(self, mocker, tmp_path: Path) -> None:
        """Roboflow export layout (valid/) must keep working after the change."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        args = self._make_args(str(tmp_path))
        mock_dataset = mocker.patch("rfdetr.datasets.yolo.YoloDetection")
        mocker.patch("rfdetr.datasets.yolo.make_coco_transforms")

        build_roboflow_from_yolo("val", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        assert "valid" in kwargs["img_folder"]

    def test_missing_test_images_signal_unavailable_split(self, tmp_path: Path) -> None:
        """A missing test image directory is the explicit val-fallback signal."""
        (tmp_path / "data.yaml").write_text("names:\n  0: person\n", encoding="utf-8")
        args = self._make_args(str(tmp_path))

        with pytest.raises(YoloSplitUnavailableError, match="test images directory"):
            build_roboflow_from_yolo("test", args, resolution=64)

    def test_broken_test_images_symlink_is_not_treated_as_missing(self, tmp_path: Path) -> None:
        """A broken test image symlink is invalid data, not an unavailable split."""
        self._write_yolo_dataset(
            tmp_path,
            images=False,
            labels=False,
            image_symlink_target=tmp_path / "deleted-images",
        )

        args = self._make_args(str(tmp_path))
        with pytest.raises(ValueError, match="invalid symlink"):
            build_roboflow_from_yolo("test", args, resolution=64)

    @pytest.mark.parametrize(
        "images, labels, expected_message",
        [
            pytest.param(False, True, "contains no supported images", id="empty-images"),
            pytest.param(True, False, "labels directory not found", id="missing-labels"),
        ],
    )
    def test_invalid_existing_test_split_propagates(
        self, tmp_path: Path, images: bool, labels: bool, expected_message: str
    ) -> None:
        """Existing but unusable test data must not fall back to validation data."""
        self._write_yolo_dataset(tmp_path, images=images, labels=labels)

        args = self._make_args(str(tmp_path))
        with pytest.raises(ValueError, match=expected_message):
            build_roboflow_from_yolo("test", args, resolution=64)

    def test_declared_broken_test_path_propagates(self, tmp_path: Path) -> None:
        """A declared but unusable YAML test path must not become a val fallback."""
        self._write_yolo_dataset(
            tmp_path,
            test_yaml_path="missing-test/images",
            images=False,
            labels=False,
            create_test_images_dir=False,
        )
        args = self._make_args(str(tmp_path))

        with pytest.raises(ValueError, match="test split declared"):
            build_roboflow_from_yolo("test", args, resolution=64)

    def test_declared_test_path_with_missing_labels_propagates(self, tmp_path: Path) -> None:
        """A declared test image path with missing labels must not become a val fallback."""
        self._write_yolo_dataset(tmp_path, test_yaml_path="custom/images", labels=False)
        args = self._make_args(str(tmp_path))

        with pytest.raises(ValueError, match="test split declared"):
            build_roboflow_from_yolo("test", args, resolution=64)

    def test_valid_test_split_builds_real_dataset(self, mocker, tmp_path: Path) -> None:
        """A non-empty labelled test split builds and preserves its sample."""
        mocker.patch("rfdetr.datasets.yolo.make_coco_transforms", return_value=None)
        self._write_yolo_dataset(tmp_path, label_content="0 0.5 0.5 0.5 0.5\n")
        args = self._make_args(str(tmp_path))

        dataset = build_roboflow_from_yolo("test", args, resolution=64)

        assert len(dataset) == 1
        _, target = dataset[0]
        assert target["labels"].tolist() == [0]

    def test_background_only_test_split_is_valid_with_labels_directory(self, mocker, tmp_path: Path) -> None:
        """A true-negative test image remains valid when its labels directory exists."""
        mocker.patch("rfdetr.datasets.yolo.make_coco_transforms", return_value=None)
        self._write_yolo_dataset(tmp_path, image_name="background.png")
        args = self._make_args(str(tmp_path))

        dataset = build_roboflow_from_yolo("test", args, resolution=64)

        assert len(dataset) == 1
        _, target = dataset[0]
        assert target["labels"].numel() == 0


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


def _write_yolo_names_data_file(tmp_path: Path, names: list[str] | dict[int | str, str]) -> Path:
    """Write a ``data.yaml`` with an arbitrary ``names`` value and a minimal ``kpt_shape``.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> data_file = _write_yolo_names_data_file(root, ["person"])
        >>> data_file.name
        'data.yaml'
    """
    data_file = tmp_path / "data.yaml"
    data_file.write_text(yaml.safe_dump({"names": names, "kpt_shape": [1, 3]}, sort_keys=False), encoding="utf-8")
    return data_file


#: Numeric-key ``names`` mappings (and the equivalent list form) that must all resolve to the
#: same 12 class names in label-ID order, regardless of key type, ordering, or zero-padding.
_VALID_NUMERIC_NAMES_CASES = [
    pytest.param([f"class_{idx}" for idx in range(12)], id="list"),
    pytest.param({idx: f"class_{idx}" for idx in reversed(range(12))}, id="integer"),
    pytest.param({str(idx): f"class_{idx}" for idx in reversed(range(12))}, id="quoted"),
    pytest.param({idx if idx % 2 else str(idx): f"class_{idx}" for idx in reversed(range(12))}, id="mixed"),
    pytest.param({f"{idx:02}": f"class_{idx}" for idx in reversed(range(12))}, id="zero-padded"),
]

#: ``names`` mappings that must all be rejected as not forming a contiguous 0..N-1 ID space.
_INVALID_CLASS_ID_CASES = [
    pytest.param({0: "cat", "0": "dog"}, id="duplicate-mixed"),
    pytest.param({"0": "cat", "00": "dog"}, id="duplicate-quoted"),
    pytest.param({0: "cat", 2: "dog"}, id="sparse-integer"),
    pytest.param({"0": "cat", "2": "dog"}, id="sparse-quoted"),
    pytest.param({1: "cat", 2: "dog"}, id="nonzero-start"),
    pytest.param({}, id="empty"),
    pytest.param({"1.0": "cat"}, id="non-integer-decimal-string"),
    pytest.param({"1e1": "cat"}, id="non-integer-exponent-string"),
    pytest.param({" 0": "cat"}, id="whitespace-padded-leading"),
    pytest.param({"1 ": "cat"}, id="whitespace-padded-trailing"),
    pytest.param({-1: "cat"}, id="negative-integer-key"),
    pytest.param({"-1": "cat"}, id="negative-string-key"),
    pytest.param({"x": "cat"}, id="non-numeric-key"),
    pytest.param({True: "cat"}, id="bool-key"),
    pytest.param({"²": "cat"}, id="unicode-digit-superscript"),
    pytest.param({"１": "cat"}, id="unicode-digit-fullwidth"),
    pytest.param({"١": "cat"}, id="unicode-digit-arabic-indic"),
]


class TestYoloClassNamesEntryPoints:
    """Keep detection, pose, and facade class names aligned with numeric label IDs.

    Each behavior is checked once per entry point (direct detection construction, pose construction, and the ``RFDETR``
    facade) in its own test method: the facade returns a plain list while ``YoloDetection`` exposes
    ``.classes``/``.coco.cats``, so the three entry points need different assertions rather than a single branching test
    body.
    """

    @pytest.mark.parametrize("names", _VALID_NUMERIC_NAMES_CASES)
    def test_class_names_follow_numeric_ids_detection(
        self, tmp_path: Path, names: list[str] | dict[int | str, str]
    ) -> None:
        """Numeric YAML keys retain label identity for a direct detection-mode dataset.

        Builds a ``YoloDetection`` without keypoints from each numeric ``names`` variant and checks that class 10's name
        and the parsed label survive key type, ordering, and zero-padding differences.
        """
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        data_file = _write_yolo_names_data_file(tmp_path, names)
        (tmp_path / "train" / "labels" / "sample.txt").write_text("10 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        dataset = YoloDetection(
            str(tmp_path / "train" / "images"),
            str(tmp_path / "train" / "labels"),
            str(data_file),
            include_keypoints=False,
        )

        expected_names = [f"class_{idx}" for idx in range(12)]
        assert dataset.classes == expected_names
        _, target = dataset[0]
        assert target is not None
        assert target["labels"].tolist() == [10]
        assert dataset.coco.cats[10]["name"] == "class_10"

    @pytest.mark.parametrize("names", _VALID_NUMERIC_NAMES_CASES)
    def test_class_names_follow_numeric_ids_pose(self, tmp_path: Path, names: list[str] | dict[int | str, str]) -> None:
        """Numeric YAML keys retain label identity for a pose-mode dataset.

        Same check as the detection case, but with keypoints enabled and a label line carrying one keypoint, so the pose
        entry point is verified independently of detection.
        """
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        data_file = _write_yolo_names_data_file(tmp_path, names)
        (tmp_path / "train" / "labels" / "sample.txt").write_text("10 0.5 0.5 0.5 0.5 0.25 0.25 2\n", encoding="utf-8")
        dataset = YoloDetection(
            str(tmp_path / "train" / "images"),
            str(tmp_path / "train" / "labels"),
            str(data_file),
            include_keypoints=True,
        )

        expected_names = [f"class_{idx}" for idx in range(12)]
        assert dataset.classes == expected_names
        _, target = dataset[0]
        assert target is not None
        assert target["labels"].tolist() == [10]
        assert dataset.coco.cats[10]["name"] == "class_10"

    @pytest.mark.parametrize("names", _VALID_NUMERIC_NAMES_CASES)
    def test_class_names_follow_numeric_ids_facade(
        self, tmp_path: Path, names: list[str] | dict[int | str, str]
    ) -> None:
        """Numeric YAML keys retain label identity through the ``RFDETR`` facade.

        ``RFDETR._load_classes`` returns a plain list rather than a dataset object, so this checks the facade's own
        return value instead of ``.classes``/``.coco.cats``.
        """
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        _write_yolo_names_data_file(tmp_path, names)

        expected_names = [f"class_{idx}" for idx in range(12)]
        assert RFDETR._load_classes(str(tmp_path)) == expected_names

    @pytest.mark.parametrize("names", _INVALID_CLASS_ID_CASES)
    def test_invalid_class_ids_raise_detection(self, tmp_path: Path, names: dict[int | str, str]) -> None:
        """Ambiguous or non-contiguous IDs raise for a direct detection-mode dataset."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        data_file = _write_yolo_names_data_file(tmp_path, names)
        with pytest.raises(ValueError, match="0..N-1"):
            YoloDetection(
                str(tmp_path / "train" / "images"),
                str(tmp_path / "train" / "labels"),
                str(data_file),
                include_keypoints=False,
            )

    @pytest.mark.parametrize("names", _INVALID_CLASS_ID_CASES)
    def test_invalid_class_ids_raise_pose(self, tmp_path: Path, names: dict[int | str, str]) -> None:
        """Ambiguous or non-contiguous IDs raise for a pose-mode dataset.

        The real cause must appear directly in the raised message (not only in ``__cause__``), so pose must not report a
        different, misleading label space from detection/facade.
        """
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        data_file = _write_yolo_names_data_file(tmp_path, names)
        with pytest.raises(ValueError, match="0..N-1"):
            YoloDetection(
                str(tmp_path / "train" / "images"),
                str(tmp_path / "train" / "labels"),
                str(data_file),
                include_keypoints=True,
            )

    @pytest.mark.parametrize("names", _INVALID_CLASS_ID_CASES)
    def test_invalid_class_ids_raise_facade(self, tmp_path: Path, names: dict[int | str, str]) -> None:
        """Ambiguous or non-contiguous IDs raise through the ``RFDETR`` facade."""
        _write_minimal_roboflow_yolo_dataset(tmp_path)
        _write_yolo_names_data_file(tmp_path, names)
        with pytest.raises(ValueError, match="0..N-1"):
            RFDETR._load_classes(str(tmp_path))
