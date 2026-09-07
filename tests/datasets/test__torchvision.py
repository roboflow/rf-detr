# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for torchvision-native default dataset transforms."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from rfdetr.datasets._torchvision import (
    Compose,
    RandomChoice,
    RandomHorizontalFlip,
    RandomResize,
    RandomSizedCrop,
    Resize,
    crop,
)
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
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

    @pytest.mark.parametrize("builder", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_keypoint_pipeline_without_flip_pairs_disables_default_flip(self, builder: Callable[..., Compose]) -> None:
        """A keypoint pipeline (list sentinel) with no flip pairs disables the default flip.

        Regression test: RandomHorizontalFlip's default construction previously ran the flip
        unconditionally whenever keypoint_flip_pairs was falsy, mirroring the mirror-without-
        relabel it performs internally (see TestRandomHorizontalFlipEdgeCases below) at the
        pipeline level too — the flip must be dropped entirely, not partially applied, matching
        AlbumentationsWrapper.from_config's filter_keypoint_hflip_augmentations for the same
        sentinel (keypoint_flip_pairs is not None and not keypoint_flip_pairs).
        """
        pipeline = builder("train", 640, keypoint_flip_pairs=[])

        assert not any(isinstance(step, RandomHorizontalFlip) for step in pipeline.transforms)

    @pytest.mark.parametrize("builder", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_keypoint_pipeline_without_flip_pairs_warns(self, builder: Callable[..., Compose]) -> None:
        """Disabling the default flip for an unpaired keypoint pipeline logs a warning.

        The warning must not tell the user to remove the transform "from your augmentation config" — the torchvision-
        native default path (``aug_config=None``) has no such config object; the only actionable remedy here is
        providing ``keypoint_flip_pairs``.
        """
        with patch("rfdetr.datasets.coco.logger") as mock_logger:
            builder("train", 640, keypoint_flip_pairs=[])

        mock_logger.warning.assert_called_once()
        message = str(mock_logger.warning.call_args)
        assert "RandomHorizontalFlip" in message
        assert "provide keypoint_flip_pairs" in message.lower()
        assert "augmentation config" not in message.lower()

    @pytest.mark.parametrize("builder", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_keypoint_pipeline_with_flip_pairs_keeps_default_flip(self, builder: Callable[..., Compose]) -> None:
        """A keypoint pipeline with real flip pairs keeps the default flip (control, no regression)."""
        pipeline = builder("train", 640, keypoint_flip_pairs=[0, 1])

        assert any(isinstance(step, RandomHorizontalFlip) for step in pipeline.transforms)

    @pytest.mark.parametrize("builder", [make_coco_transforms, make_coco_transforms_square_div_64])
    def test_detection_only_pipeline_keeps_default_flip(self, builder: Callable[..., Compose]) -> None:
        """A detection-only pipeline (None sentinel) keeps the default flip (control, no regression)."""
        pipeline = builder("train", 640, keypoint_flip_pairs=None)

        assert any(isinstance(step, RandomHorizontalFlip) for step in pipeline.transforms)

    @pytest.mark.parametrize("split", ["val", "test", "val_speed"])
    def test_eval_splits_do_not_use_albumentations(self, split: str) -> None:
        """Evaluation transforms use torchvision resize and normalization only."""
        pipeline = make_coco_transforms(split, 640, aug_config={"HorizontalFlip": {"p": 1.0}})

        assert not any(isinstance(step, AlbumentationsWrapper) for step in pipeline.transforms)
        assert any(isinstance(step, Normalize) for step in pipeline.transforms)

    @patch("rfdetr.datasets.transforms.alb", None)
    def test_custom_aug_config_missing_albumentations_raises_extra_hint(self) -> None:
        """Custom Albumentations configs require the augmentation extra."""
        with pytest.raises(ImportError, match=r"rfdetr\[augment\]"):
            make_coco_transforms("train", 640, aug_config={"HorizontalFlip": {"p": 1.0}})

    @patch("rfdetr.datasets.transforms.alb", None)
    def test_gpu_postprocess_custom_aug_config_does_not_require_albumentations(self) -> None:
        """GPU augmentation uses torchvision CPU resize even with custom aug_config."""
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

        toml_path = Path(__file__).parents[2] / "pyproject.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        optional = data["project"]["optional-dependencies"]
        train_deps = "\n".join(optional["train"])
        assert "albumentations" not in train_deps
        assert "kornia" not in train_deps
        assert any(dep.startswith("albumentations") for dep in optional["augment"])
        assert any(dep.startswith("kornia") for dep in optional["augment"])


class TestCropFunction:
    """Crop() correctly filters degenerate boxes and synchronises per-instance fields."""

    def test_all_boxes_outside_crop_produces_empty_target(self) -> None:
        """Boxes fully outside the crop are removed and every per-instance field is filtered to match.

        ``labels``, ``iscrowd``, and ``masks`` are per-instance fields filtered down to zero rows alongside ``boxes``,
        so box/label correspondence is preserved.
        """
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
            "area": torch.tensor([100.0]),
            "iscrowd": torch.tensor([0]),
            "masks": torch.ones((1, 100, 100), dtype=torch.bool),
        }

        _, out = crop(image, target, top=50, left=50, height=50, width=50)

        assert out["boxes"].shape == (0, 4), "all boxes removed"
        # labels is per-instance — filtered in lockstep with boxes so lengths stay equal
        assert out["labels"].shape[0] == out["boxes"].shape[0], "labels length matches boxes length"
        assert out["labels"].shape == (0,), "labels filtered to zero alongside boxes"
        # iscrowd is a per-instance field — filtered to match surviving boxes
        assert out["iscrowd"].shape == (0,), "per-instance field filtered to zero"
        assert out["masks"].shape == (0, 50, 50)

    def test_one_box_survives_one_removed(self) -> None:
        """Box inside crop survives; per-instance fields ``labels`` and ``area`` are filtered in sync.

        ``labels`` and ``area`` are per-instance fields filtered to match the surviving boxes, so all three stay the
        same length after the crop.
        """
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[60.0, 60.0, 90.0, 90.0], [0.0, 0.0, 5.0, 5.0]]),
            "labels": torch.tensor([2, 3]),
            "area": torch.tensor([900.0, 25.0]),
        }

        _, out = crop(image, target, top=50, left=50, height=50, width=50)

        assert out["boxes"].shape[0] == 1, "one surviving box"
        # labels is a per-instance field; it must stay the same length as boxes
        assert out["labels"].shape[0] == out["boxes"].shape[0], "labels filtered in sync with boxes"
        # area is a per-instance field; only the surviving box's area is kept
        assert out["area"].shape[0] == 1, "area filtered to one surviving instance"

    def test_surviving_label_value_matches_kept_box(self) -> None:
        """The surviving label VALUE tracks the surviving box, not merely its count.

        With boxes ``[[0,0,10,10], [60,60,90,90]]`` and labels ``[5, 7]``, a crop at ``(50, 50)`` of size ``50x50``
        keeps only the second box, so the surviving label must be ``7`` (index 1), never ``5``.
        """
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [60.0, 60.0, 90.0, 90.0]]),
            "labels": torch.tensor([5, 7]),
        }

        _, out = crop(image, target, top=50, left=50, height=50, width=50)

        assert out["boxes"].shape[0] == 1, "only the second box survives"
        assert out["labels"].tolist() == [7], "surviving label value tracks the kept box, not the first index"

    def test_all_boxes_survive_full_crop(self) -> None:
        """A crop equal to the full image leaves all boxes intact."""
        image = Image.new("RGB", (80, 60))
        boxes = torch.tensor([[5.0, 5.0, 20.0, 20.0], [30.0, 30.0, 60.0, 50.0]])
        target = {"boxes": boxes.clone(), "labels": torch.tensor([1, 2])}

        _, out = crop(image, target, top=0, left=0, height=60, width=80)

        assert out["boxes"].shape[0] == 2


class TestRandomHorizontalFlipEdgeCases:
    """RandomHorizontalFlip boundary and skip-path behaviour."""

    def test_p_zero_returns_input_unchanged(self) -> None:
        """p=0.0 always skips the flip; image and target returned unmodified."""
        image = Image.new("RGB", (100, 50))
        boxes = torch.tensor([[10.0, 5.0, 30.0, 25.0]])
        target = {"boxes": boxes.clone(), "labels": torch.tensor([1])}
        flip = RandomHorizontalFlip(p=0.0)

        _, out = flip(image, target)

        torch.testing.assert_close(out["boxes"], boxes)

    def test_odd_length_keypoint_flip_pairs_raises(self) -> None:
        """Odd-length keypoint_flip_pairs raises ValueError at construction time."""
        with pytest.raises(ValueError, match="even"):
            RandomHorizontalFlip(p=1.0, keypoint_flip_pairs=[0, 1, 2])


class TestRandomSelectBoundaries:
    """RandomSelect always routes to the correct transform at boundary probabilities."""

    def test_p_zero_always_selects_transform2(self) -> None:
        """p=0.0 always picks transform2."""
        from rfdetr.datasets._torchvision import RandomSelect

        def t1(img, tgt):
            return img, {"selected": 1}

        def t2(img, tgt):
            return img, {"selected": 2}

        sel = RandomSelect(t1, t2, p=0.0)
        image = Image.new("RGB", (10, 10))
        for _ in range(10):
            _, out = sel(image, {})
            assert out["selected"] == 2

    def test_p_one_always_selects_transform1(self) -> None:
        """p=1.0 always picks transform1."""
        from rfdetr.datasets._torchvision import RandomSelect

        def t1(img, tgt):
            return img, {"selected": 1}

        def t2(img, tgt):
            return img, {"selected": 2}

        sel = RandomSelect(t1, t2, p=1.0)
        image = Image.new("RGB", (10, 10))
        for _ in range(10):
            _, out = sel(image, {})
            assert out["selected"] == 1


class TestRandomSizedCropBoundary:
    """RandomSizedCrop handles degenerate and near-degenerate image sizes."""

    @pytest.mark.parametrize(
        "img_size,crop_range,output_size",
        [
            pytest.param((1, 1), (1, 10), (640, 640), id="single_pixel"),
            pytest.param((8, 8), (100, 200), (640, 640), id="small_image_larger_crop_range"),
        ],
    )
    def test_does_not_crash_on_degenerate_size(
        self,
        img_size: tuple[int, int],
        crop_range: tuple[int, int],
        output_size: tuple[int, int],
    ) -> None:
        """RandomSizedCrop does not crash when min_crop >= image dimension after clamping."""
        image = Image.new("RGB", img_size)
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
        transform = RandomSizedCrop(crop_range, output_size)

        img_out, _ = transform(image, target)

        assert img_out is not None


class TestResizeDtypeGuard:
    """Resize correctly handles integer-dtype boxes without silently zeroing them."""

    def test_int64_boxes_are_not_zeroed_on_downscale(self) -> None:
        """Integer-dtype boxes survive downscale resize with correct non-zero coordinates."""
        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[10, 5, 30, 25]], dtype=torch.int64),
            "labels": torch.tensor([1]),
        }
        transform = Resize((50, 50))

        _, out = transform(image, target)

        assert out["boxes"].sum() > 0, "boxes should not be zeroed by dtype truncation"
        assert out["boxes"].dtype == torch.float32


class TestEdgeCaseCoverage:
    """Edge cases: empty constructors, unknown splits, zero-spatial masks, max_size cap."""

    def test_random_choice_empty_raises(self) -> None:
        """RandomChoice with no transforms raises ValueError."""
        with pytest.raises(ValueError, match="at least one transform"):
            RandomChoice([])

    def test_random_resize_empty_raises(self) -> None:
        """RandomResize with no sizes raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            RandomResize([])

    @pytest.mark.parametrize(
        "builder",
        [
            pytest.param(make_coco_transforms, id="standard"),
            pytest.param(make_coco_transforms_square_div_64, id="square"),
        ],
    )
    def test_unknown_image_set_raises(self, builder) -> None:
        """Unknown image_set raises ValueError."""
        with pytest.raises(ValueError, match="unknown"):
            builder("predict", 640)

    def test_resize_masks_zero_spatial_dim(self) -> None:
        """_apply_to_masks handles (0, H, W) masks returning (0, new_H, new_W) without error."""
        from torchvision.transforms.v2 import functional

        from rfdetr.datasets._torchvision import _apply_to_masks

        empty_masks = torch.zeros((0, 50, 100), dtype=torch.bool)
        result = _apply_to_masks(empty_masks, lambda masks: functional.resize(masks, [200, 200]))

        assert result.shape == (0, 200, 200)
        assert result.dtype == torch.bool

    def test_random_resize_max_size_cap(self) -> None:
        """RandomResize clips the long side to max_size when it would exceed the cap."""
        # Image 100x200 (w x h): requesting short-side=100, max_size=150.
        # Without cap: long side would become 200; with cap it is clamped to 150.
        image = Image.new("RGB", (100, 200))
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
        transform = RandomResize([100], max_size=150)

        img_out, _ = transform(image, target)

        height, width = img_out.height, img_out.width
        assert max(height, width) <= 150, f"long side {max(height, width)} exceeds max_size=150"

    def test_multi_scale_train_torchvision_produces_valid_shape(self) -> None:
        """make_coco_transforms with multi_scale=True produces a valid output tensor."""
        image = Image.new("RGB", (640, 480))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
            "labels": torch.tensor([1]),
        }
        transform = make_coco_transforms("train", 640, multi_scale=True)

        tensor, out = transform(image, target)

        assert tensor.ndim == 3, "output should be CHW tensor"
        assert tensor.shape[0] == 3, "should have 3 channels"

    def test_tensor_image_branch_resize(self) -> None:
        """Resize handles a torch.Tensor image as input (not PIL)."""
        tensor_image = torch.rand(3, 50, 100)
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 30.0, 25.0]]),
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([50, 100]),
            "size": torch.tensor([50, 100]),
        }
        transform = Resize((200, 200))

        img_out, out = transform(tensor_image, target)

        assert img_out.shape[-2:] == (200, 200)

    def test_mark_invisible_keypoints_empty_no_crash(self) -> None:
        """_mark_invisible_keypoints returns immediately for empty keypoints tensor."""
        from rfdetr.datasets._torchvision import _mark_invisible_keypoints

        empty_kps = torch.zeros((0, 17, 3))
        result = _mark_invisible_keypoints(empty_kps, height=480, width=640)

        assert result.shape == (0, 17, 3)


class TestNonUniformMaskParity:
    """Non-uniform masks are correctly resized and flipped."""

    def test_horizontal_flip_non_uniform_mask(self) -> None:
        """Horizontal flip mirrors a non-uniform mask correctly."""
        image = Image.new("RGB", (100, 50))
        mask = torch.zeros((1, 50, 100), dtype=torch.bool)
        mask[0, :, :50] = True  # left half True, right half False
        target = {
            "boxes": torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
            "masks": mask,
        }
        flip = RandomHorizontalFlip(p=1.0)

        _, out = flip(image, target)

        assert out["masks"][0, :, 50:].all(), "right half should be True after flip"
        assert not out["masks"][0, :, :50].any(), "left half should be False after flip"


class TestDefaultAugBackendWarning:
    """make_coco_transforms emits UserWarning when aug_config=None (backend changed in v1.8)."""

    def test_default_train_emits_backend_change_warning(self) -> None:
        """aug_config=None on train split emits a UserWarning about the backend switch."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_coco_transforms("train", 640)

        user_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "torchvision" in str(w.message).lower()
        ]
        assert len(user_warnings) >= 1, "Expected UserWarning about aug backend change"

    def test_empty_aug_config_does_not_warn(self) -> None:
        """aug_config={} does not trigger the backend-switch warning (augmentation disabled)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_coco_transforms("train", 640, aug_config={})

        backend_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "torchvision" in str(w.message).lower()
        ]
        assert len(backend_warnings) == 0
