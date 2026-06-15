# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for torchvision-native default dataset transforms."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.torchvision_transforms import RandomHorizontalFlip
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize


class TestDefaultTorchvisionTransforms:
    """Default COCO transform builders use torchvision without Albumentations."""

    @pytest.mark.parametrize("builder", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_default_train_uses_torchvision_flip(self, builder) -> None:
        """Default train transforms include torchvision HorizontalFlip and no Albumentations wrappers."""
        pipeline = builder("train", 640)

        assert any(isinstance(step, RandomHorizontalFlip) for step in pipeline.transforms)
        assert not any(isinstance(step, AlbumentationsWrapper) for step in pipeline.transforms)

    @pytest.mark.parametrize("builder", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_empty_aug_config_disables_default_flip(self, builder) -> None:
        """An empty aug_config disables default augmentation."""
        pipeline = builder("train", 640, aug_config={})

        assert not any(isinstance(step, RandomHorizontalFlip) for step in pipeline.transforms)
        assert not any(isinstance(step, AlbumentationsWrapper) for step in pipeline.transforms)

    @pytest.mark.parametrize("split", ["val", "test", "val_speed"])
    def test_eval_splits_do_not_use_albumentations(self, split: str) -> None:
        """Evaluation transforms use torchvision resize and normalization only."""
        pipeline = make_coco_transforms(split, 640, aug_config={"HorizontalFlip": {"p": 1.0}})

        assert not any(isinstance(step, AlbumentationsWrapper) for step in pipeline.transforms)
        assert any(isinstance(step, Normalize) for step in pipeline.transforms)

    def test_custom_aug_config_missing_albumentations_raises_extra_hint(self) -> None:
        """Custom Albumentations configs require the augmentation extra."""
        with (
            patch("rfdetr.datasets.transforms.alb", None),
            pytest.raises(ImportError, match=r"rfdetr\[augmentation\]"),
        ):
            make_coco_transforms("train", 640, aug_config={"HorizontalFlip": {"p": 1.0}})

    def test_gpu_postprocess_custom_aug_config_does_not_require_albumentations(self) -> None:
        """GPU augmentation uses torchvision CPU resize even with custom aug_config."""
        with patch("rfdetr.datasets.transforms.alb", None):
            pipeline = make_coco_transforms(
                "train",
                640,
                aug_config={"HorizontalFlip": {"p": 1.0}},
                gpu_postprocess=True,
            )

        assert not any(isinstance(step, AlbumentationsWrapper) for step in pipeline.transforms)
        assert not any(isinstance(step, Normalize) for step in pipeline.transforms)


class TestTorchvisionTransformOutputs:
    """Default torchvision transforms preserve RF-DETR target semantics."""

    def test_square_val_resizes_boxes_masks_and_normalizes_boxes(self) -> None:
        """Square val transform resizes masks and converts boxes to normalized cxcywh."""
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.ones((1, 50, 100), dtype=torch.bool),
            "orig_size": torch.tensor([50, 100]),
            "size": torch.tensor([50, 100]),
        }
        transform = make_coco_transforms_square_div_64("val", 200)

        tensor, transformed = transform(image, target)

        assert tensor.shape[-2:] == (200, 200)
        assert transformed["masks"].shape == (1, 200, 200)
        torch.testing.assert_close(
            transformed["boxes"],
            torch.tensor([[0.2, 0.3, 0.2, 0.4]]),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_horizontal_flip_updates_boxes_and_keypoint_pairs(self) -> None:
        """Default flip mirrors boxes, keypoints, and configured left/right pairs."""
        image = Image.new("RGB", (100, 50))
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[10.0, 5.0, 2.0], [30.0, 25.0, 2.0]]]),
            "orig_size": torch.tensor([50, 100]),
            "size": torch.tensor([50, 100]),
        }
        flip = RandomHorizontalFlip(p=1.0, keypoint_flip_pairs=[0, 1])

        _, transformed = flip(image, target)

        torch.testing.assert_close(transformed["boxes"], torch.tensor([[70.0, 5.0, 90.0, 25.0]]))
        torch.testing.assert_close(
            transformed["keypoints"],
            torch.tensor([[[69.0, 25.0, 2.0], [89.0, 5.0, 2.0]]]),
        )


class TestTrainExtraIsMinimal:
    """Dependency metadata keeps advanced augmentations outside the train extra."""

    def test_train_extra_does_not_include_albumentations_or_kornia(self) -> None:
        """The train extra remains minimal; augmentation libraries are separate extras."""
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)

        optional = data["project"]["optional-dependencies"]
        train_deps = "\n".join(optional["train"])
        assert "albumentations" not in train_deps
        assert "kornia" not in train_deps
        assert any(dep.startswith("albumentations") for dep in optional["augmentation"])
        assert any(dep.startswith("kornia") for dep in optional["kornia"])
