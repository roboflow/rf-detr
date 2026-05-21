# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Kornia GPU augmentation pipeline builder and bbox utilities.

All tests in this module are CPU-compatible — Kornia operates on CPU tensors identically to GPU tensors, so no
``@pytest.mark.gpu`` is needed.
"""

import pytest
import torch

from rfdetr.datasets.aug_config import (
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONSERVATIVE,
    AUG_INDUSTRIAL,
)

# ---------------------------------------------------------------------------
# TestBuildKorniaPipeline — validates the factory that translates aug_config
# dicts into a Kornia AugmentationSequential pipeline.
# ---------------------------------------------------------------------------


class TestBuildKorniaPipeline:
    """build_kornia_pipeline returns a valid pipeline for every preset and rejects unknown transform keys with a clear
    error."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        pytest.importorskip("kornia")

    @pytest.mark.parametrize(
        "config,config_name",
        [
            pytest.param(AUG_CONSERVATIVE, "AUG_CONSERVATIVE", id="conservative"),
            pytest.param(AUG_AGGRESSIVE, "AUG_AGGRESSIVE", id="aggressive"),
            pytest.param(AUG_AERIAL, "AUG_AERIAL", id="aerial"),
            pytest.param(AUG_INDUSTRIAL, "AUG_INDUSTRIAL", id="industrial"),
        ],
    )
    def test_each_preset_config(self, config, config_name):
        """Each named preset builds a pipeline without errors."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline(config, 560)
        assert pipeline is not None, f"build_kornia_pipeline({config_name}, 560) must return a non-None pipeline"

    def test_unknown_key_raises_value_error(self):
        """An unrecognised transform key raises ValueError immediately."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        with pytest.raises(ValueError, match="FooBarTransform"):
            build_kornia_pipeline({"FooBarTransform": {"p": 0.5}}, 560)

    def test_empty_config_returns_pipeline(self):
        """An empty config dict returns a valid (no-op) pipeline, not None."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({}, 560)
        assert pipeline is not None, "Empty config must still return a pipeline object"

    def test_known_plus_unknown_raises(self):
        """Mixing a valid key with an unknown key still raises ValueError."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        mixed = {"HorizontalFlip": {"p": 0.5}, "BogusTransform": {"p": 0.3}}
        with pytest.raises(ValueError, match="BogusTransform"):
            build_kornia_pipeline(mixed, 560)


# ---------------------------------------------------------------------------
# TestCollateBoxes — validates packing of variable-length per-image boxes
# into a zero-padded [B, N_max, 4] tensor with a boolean validity mask.
# ---------------------------------------------------------------------------


class TestCollateBoxes:
    """collate_boxes packs variable-length boxes into [B, N_max, 4] with mask."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        pytest.importorskip("kornia")

    def _make_targets(self, box_counts):
        """Build a list of target dicts with the given per-image box counts.

        Each box is a valid xyxy rectangle within a 100x100 image.
        """
        targets = []
        for n in box_counts:
            boxes = (
                torch.tensor([[10.0, 10.0, 50.0, 50.0]] * n, dtype=torch.float32)
                if n > 0
                else torch.zeros(0, 4, dtype=torch.float32)
            )
            targets.append({"boxes": boxes})
        return targets

    def test_normal_batch(self):
        """Batch of 2 images: output shape is [2, N_max, 4] with valid mask [2, N_max]."""
        from rfdetr.datasets.kornia_transforms import collate_boxes

        targets = self._make_targets([2, 3])
        boxes_padded, valid = collate_boxes(targets, torch.device("cpu"))

        assert boxes_padded.shape == (2, 3, 4), f"Expected shape (2, 3, 4), got {boxes_padded.shape}"
        assert valid.shape == (2, 3), f"Expected valid shape (2, 3), got {valid.shape}"
        assert valid.dtype == torch.bool

    def test_b_zero(self):
        """Empty target list produces shape [0, 0, 4] and valid [0, 0]."""
        from rfdetr.datasets.kornia_transforms import collate_boxes

        boxes_padded, valid = collate_boxes([], torch.device("cpu"))

        assert boxes_padded.shape == (0, 0, 4), f"Expected (0, 0, 4) for empty batch, got {boxes_padded.shape}"
        assert valid.shape == (0, 0), f"Expected valid (0, 0) for empty batch, got {valid.shape}"

    def test_n_zero_per_image(self):
        """One image with 0 boxes: shape [1, 0, 4], valid all-False."""
        from rfdetr.datasets.kornia_transforms import collate_boxes

        targets = self._make_targets([0])
        boxes_padded, valid = collate_boxes(targets, torch.device("cpu"))

        assert boxes_padded.shape == (1, 0, 4), f"Expected (1, 0, 4), got {boxes_padded.shape}"
        assert valid.shape == (1, 0), f"Expected (1, 0), got {valid.shape}"

    def test_single_image(self):
        """B=1 with 3 boxes: output shape is [1, 3, 4]."""
        from rfdetr.datasets.kornia_transforms import collate_boxes

        targets = self._make_targets([3])
        boxes_padded, valid = collate_boxes(targets, torch.device("cpu"))

        assert boxes_padded.shape == (1, 3, 4)
        assert valid.shape == (1, 3)

    def test_valid_mask_matches_box_count(self):
        """The valid mask has True for real boxes and False for padding."""
        from rfdetr.datasets.kornia_transforms import collate_boxes

        targets = self._make_targets([1, 3])
        _, valid = collate_boxes(targets, torch.device("cpu"))

        # Image 0: 1 real box, 2 padding → [True, False, False]
        assert valid[0].tolist() == [True, False, False], f"Image 0 valid mask wrong: {valid[0].tolist()}"
        # Image 1: 3 real boxes, 0 padding → [True, True, True]
        assert valid[1].tolist() == [True, True, True], f"Image 1 valid mask wrong: {valid[1].tolist()}"


# ---------------------------------------------------------------------------
# TestUnpackBoxes — validates the inverse: writing augmented boxes back into
# per-image target dicts with clamping, zero-area removal, and label sync.
# ---------------------------------------------------------------------------


class TestUnpackBoxes:
    """unpack_boxes writes augmented boxes back and removes zero-area entries."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        pytest.importorskip("kornia")

    def _make_inputs(
        self,
        boxes_aug,
        valid_mask,
        original_targets,
        image_height=100,
        image_width=100,
    ):
        """Return tensors suitable for unpack_boxes."""
        boxes_tensor = torch.tensor(boxes_aug, dtype=torch.float32)
        valid_tensor = torch.tensor(valid_mask, dtype=torch.bool)
        return boxes_tensor, valid_tensor, original_targets, image_height, image_width

    def test_all_boxes_removed_after_aug(self):
        """When all augmented boxes are zero-area, output targets have empty boxes."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        # B=1, N=2: both boxes are zero-area (x1==x2 or y1==y2)
        boxes_aug = [[[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]]]
        valid = [[True, True]]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]]),
                "labels": torch.tensor([1, 2]),
                "area": torch.tensor([1600.0, 1600.0]),
                "iscrowd": torch.tensor([0, 0]),
            }
        ]
        boxes_t, valid_t, tgts, image_height, image_width = self._make_inputs(boxes_aug, valid, targets)
        result = unpack_boxes(boxes_t, valid_t, tgts, image_height, image_width)

        assert result[0]["boxes"].shape[0] == 0, (
            f"Expected 0 boxes after zero-area removal, got {result[0]['boxes'].shape[0]}"
        )
        assert result[0]["labels"].shape[0] == 0

    def test_partial_removal(self):
        """Some boxes survive, some removed; labels/area/iscrowd synced."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        # Box 0: valid, non-zero area; Box 1: zero-area
        boxes_aug = [[[10.0, 10.0, 50.0, 50.0], [30.0, 30.0, 30.0, 30.0]]]
        valid = [[True, True]]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [30.0, 30.0, 70.0, 70.0]]),
                "labels": torch.tensor([1, 2]),
                "area": torch.tensor([1600.0, 1600.0]),
                "iscrowd": torch.tensor([0, 1]),
            }
        ]
        boxes_t, valid_t, tgts, image_height, image_width = self._make_inputs(boxes_aug, valid, targets)
        result = unpack_boxes(boxes_t, valid_t, tgts, image_height, image_width)

        assert result[0]["boxes"].shape[0] == 1, f"Expected 1 surviving box, got {result[0]['boxes'].shape[0]}"
        assert result[0]["labels"].tolist() == [1]

    def test_labels_area_iscrowd_sync(self):
        """When boxes are removed, labels/area/iscrowd entries are also removed."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        # Box 0: zero-area (removed), Box 1: valid
        boxes_aug = [[[5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 40.0, 40.0]]]
        valid = [[True, True]]
        targets = [
            {
                "boxes": torch.tensor([[5.0, 5.0, 30.0, 30.0], [10.0, 10.0, 40.0, 40.0]]),
                "labels": torch.tensor([7, 9]),
                "area": torch.tensor([625.0, 900.0]),
                "iscrowd": torch.tensor([0, 1]),
            }
        ]
        boxes_t, valid_t, tgts, image_height, image_width = self._make_inputs(boxes_aug, valid, targets)
        result = unpack_boxes(boxes_t, valid_t, tgts, image_height, image_width)

        assert result[0]["labels"].tolist() == [9], (
            f"Expected label [9] after removal of box 0, got {result[0]['labels'].tolist()}"
        )
        assert result[0]["area"].shape[0] == 1
        assert result[0]["iscrowd"].tolist() == [1]

    def test_boxes_clamped_to_image_bounds(self):
        """Boxes outside [0,W]x[0,H] are clamped to image bounds."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        # Box extends beyond 100x100 image
        boxes_aug = [[[-10.0, -5.0, 120.0, 110.0]]]
        valid = [[True]]
        targets = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 90.0, 90.0]]),
                "labels": torch.tensor([1]),
                "area": torch.tensor([8100.0]),
                "iscrowd": torch.tensor([0]),
            }
        ]
        image_height, image_width = 100, 100
        boxes_t, valid_t, tgts, image_height, image_width = self._make_inputs(
            boxes_aug,
            valid,
            targets,
            image_height,
            image_width,
        )
        result = unpack_boxes(boxes_t, valid_t, tgts, image_height, image_width)

        result_boxes = result[0]["boxes"]
        assert result_boxes.shape[0] == 1, "Clamped box should survive (non-zero area)"
        # Verify clamping: x1>=0, y1>=0, x2<=W, y2<=H
        assert result_boxes[0, 0].item() >= 0.0, "x1 not clamped to >= 0"
        assert result_boxes[0, 1].item() >= 0.0, "y1 not clamped to >= 0"
        assert result_boxes[0, 2].item() <= image_width, f"x2 not clamped to <= {image_width}"
        assert result_boxes[0, 3].item() <= image_height, f"y2 not clamped to <= {image_height}"


# ---------------------------------------------------------------------------
# TestRotateFactory — validates the Rotate parameter translation from
# Albumentations-style limit (scalar or tuple) to Kornia RandomRotation.
# ---------------------------------------------------------------------------


class TestRotateFactory:
    """Rotate factory translates limit (scalar or tuple) to K.RandomRotation(degrees=...)."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        pytest.importorskip("kornia")

    def test_limit_as_scalar(self):
        """Rotate(limit=45) produces K.RandomRotation(degrees=(-45, 45))."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        # Build a pipeline with just Rotate(limit=45)
        pipeline = build_kornia_pipeline({"Rotate": {"limit": 45, "p": 1.0}}, 560)
        assert pipeline is not None

        # Inspect the pipeline's children to find the RandomRotation and check degrees
        import kornia.augmentation as kornia_augmentation

        rotation_augs = [
            child for child in pipeline.children() if isinstance(child, kornia_augmentation.RandomRotation)
        ]
        assert len(rotation_augs) == 1, f"Expected exactly 1 RandomRotation, found {len(rotation_augs)}"
        degrees = rotation_augs[0].flags["degrees"]
        # degrees should be a tensor representing (-45, 45)
        assert float(degrees[0]) == pytest.approx(-45.0, abs=0.1)
        assert float(degrees[1]) == pytest.approx(45.0, abs=0.1)

    def test_limit_as_tuple(self):
        """Rotate(limit=(90, 90)) produces K.RandomRotation(degrees=(90, 90))."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"Rotate": {"limit": (90, 90), "p": 1.0}}, 560)
        assert pipeline is not None

        import kornia.augmentation as kornia_augmentation

        rotation_augs = [
            child for child in pipeline.children() if isinstance(child, kornia_augmentation.RandomRotation)
        ]
        assert len(rotation_augs) == 1
        degrees = rotation_augs[0].flags["degrees"]
        assert float(degrees[0]) == pytest.approx(90.0, abs=0.1)
        assert float(degrees[1]) == pytest.approx(90.0, abs=0.1)

    def test_flags_include_degrees(self):
        """Rotate factory keeps a legacy degrees entry in Kornia flags for compatibility."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"Rotate": {"limit": 30, "p": 1.0}}, 560)
        assert pipeline is not None

        import kornia.augmentation as kornia_augmentation

        rotation_augs = [
            child for child in pipeline.children() if isinstance(child, kornia_augmentation.RandomRotation)
        ]
        assert len(rotation_augs) == 1
        assert "degrees" in rotation_augs[0].flags
        assert rotation_augs[0].flags["degrees"] == (-30, 30)


# ---------------------------------------------------------------------------
# TestGpuPostprocessFlag — validates that make_coco_transforms respects the
# gpu_postprocess flag to omit augmentation and normalization from CPU path.
# ---------------------------------------------------------------------------


class TestGpuPostprocessFlag:
    """gpu_postprocess flag controls whether aug + normalize appear in CPU pipeline."""

    def test_gpu_postprocess_true_omits_aug_and_normalize_from_train(self):
        """gpu_postprocess=True: train pipeline has no Normalize; fewer AlbumentationsWrappers (no aug_wrappers)."""
        from rfdetr.datasets.coco import make_coco_transforms
        from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize

        pipeline_gpu = make_coco_transforms("train", 560, gpu_postprocess=True)
        pipeline_cpu = make_coco_transforms("train", 560, gpu_postprocess=False)

        steps_gpu = pipeline_gpu.transforms
        steps_cpu = pipeline_cpu.transforms

        normalize_gpu = [s for s in steps_gpu if isinstance(s, Normalize)]
        assert len(normalize_gpu) == 0, "gpu_postprocess=True must omit Normalize from train pipeline"

        # Resize wrappers (AlbumentationsWrapper) remain; aug wrappers are removed.
        # Default AUG_CONFIG adds 1 aug wrapper, so gpu version must have fewer wrappers.
        n_alb_gpu = sum(isinstance(s, AlbumentationsWrapper) for s in steps_gpu)
        n_alb_cpu = sum(isinstance(s, AlbumentationsWrapper) for s in steps_cpu)
        assert n_alb_gpu < n_alb_cpu, "gpu_postprocess=True must remove aug AlbumentationsWrappers from train pipeline"

    def test_gpu_postprocess_false_includes_aug_and_normalize_from_train(self):
        """gpu_postprocess=False (default): train pipeline includes Normalize."""
        from rfdetr.datasets.coco import make_coco_transforms
        from rfdetr.datasets.transforms import Normalize

        pipeline = make_coco_transforms("train", 560, gpu_postprocess=False)
        steps = pipeline.transforms

        normalize_steps = [s for s in steps if isinstance(s, Normalize)]
        assert len(normalize_steps) > 0, "gpu_postprocess=False must include Normalize in train pipeline"

    def test_val_path_unaffected_by_gpu_postprocess(self):
        """Val pipeline is unchanged regardless of gpu_postprocess value."""
        from rfdetr.datasets.coco import make_coco_transforms
        from rfdetr.datasets.transforms import Normalize

        pipeline_default = make_coco_transforms("val", 560, gpu_postprocess=False)
        pipeline_gpu = make_coco_transforms("val", 560, gpu_postprocess=True)

        # Both should have Normalize (val is never stripped)
        norm_default = [s for s in pipeline_default.transforms if isinstance(s, Normalize)]
        norm_gpu = [s for s in pipeline_gpu.transforms if isinstance(s, Normalize)]

        assert len(norm_default) > 0, "Val pipeline (default) must include Normalize"
        assert len(norm_gpu) > 0, "Val pipeline (gpu_postprocess=True) must include Normalize"

        # Same number of pipeline steps
        assert len(pipeline_default.transforms) == len(pipeline_gpu.transforms), (
            "Val pipeline step count must be identical regardless of gpu_postprocess"
        )


# ---------------------------------------------------------------------------
# TestGaussianBlurMinKernel — validates that blur_limit < 3 is clamped so
# Kornia never receives an invalid kernel_size < 3.
# ---------------------------------------------------------------------------


class TestGaussianBlurMinKernel:
    """_make_gaussian_blur enforces kernel_size >= 3 regardless of blur_limit."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        pytest.importorskip("kornia")

    @pytest.mark.parametrize(
        "blur_limit",
        [pytest.param(1, id="blur_limit_1"), pytest.param(2, id="blur_limit_2")],
    )
    def test_small_blur_limit_produces_valid_kernel(self, blur_limit):
        """blur_limit below 3 must be clamped so the resulting kernel_size >= 3."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        # Should not raise; previously blur_limit=1 produced kernel_size=(3,1)
        pipeline = build_kornia_pipeline({"GaussianBlur": {"blur_limit": blur_limit, "p": 1.0}}, 560)
        assert pipeline is not None

        import kornia.augmentation as kornia_augmentation

        blur_augs = [c for c in pipeline.children() if isinstance(c, kornia_augmentation.RandomGaussianBlur)]
        assert len(blur_augs) == 1
        ks = blur_augs[0].flags["kernel_size"]
        assert int(ks[0]) >= 3, f"kernel_size[0]={int(ks[0])} must be >= 3"
        assert int(ks[1]) >= 3, f"kernel_size[1]={int(ks[1])} must be >= 3"

    def test_blur_limit_3_unchanged(self):
        """blur_limit=3 (default) passes through without modification."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"GaussianBlur": {"blur_limit": 3, "p": 1.0}}, 560)
        import kornia.augmentation as kornia_augmentation

        blur_augs = [c for c in pipeline.children() if isinstance(c, kornia_augmentation.RandomGaussianBlur)]
        ks = blur_augs[0].flags["kernel_size"]
        assert int(ks[0]) == 3
        assert int(ks[1]) == 3


# ---------------------------------------------------------------------------
# TestKorniaPipelineForwardPass — validates that a built pipeline produces
# output of the correct shape and dtype on CPU tensors.
# ---------------------------------------------------------------------------


class TestKorniaPipelineForwardPass:
    """build_kornia_pipeline output passes through without shape/dtype errors."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        pytest.importorskip("kornia")

    def test_forward_pass_shape_and_dtype(self):
        """Pipeline output images have same shape as input; boxes shape is [B, N, 4]."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=64)

        batch_size, channels, image_height, image_width = 2, 3, 64, 64
        img = torch.rand(batch_size, channels, image_height, image_width)
        boxes = torch.tensor([[[0.0, 0.0, 32.0, 32.0]], [[10.0, 10.0, 50.0, 50.0]]], dtype=torch.float32)

        img_out, boxes_out = pipeline(img, boxes)

        assert img_out.shape == (batch_size, channels, image_height, image_width), (
            f"Image shape changed: {img_out.shape}"
        )
        assert img_out.dtype == torch.float32
        assert boxes_out.shape == (batch_size, 1, 4), f"Boxes shape wrong: {boxes_out.shape}"

    def test_forward_pass_empty_boxes(self):
        """Pipeline handles a batch where N_max=0 (no boxes) without error."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=32)

        batch_size, channels, image_height, image_width = 2, 3, 32, 32
        img = torch.rand(batch_size, channels, image_height, image_width)
        # [B, 0, 4] — no boxes
        boxes = torch.zeros(batch_size, 0, 4, dtype=torch.float32)

        img_out, boxes_out = pipeline(img, boxes)

        assert img_out.shape == (batch_size, channels, image_height, image_width)
        assert boxes_out.shape == (batch_size, 0, 4)


# ---------------------------------------------------------------------------
# TestCollateMasks — validates packing of variable-length per-image masks
# into a zero-padded [B, N_max, H, W] float32 tensor.
# ---------------------------------------------------------------------------


class TestCollateMasks:
    """collate_masks packs [N_i, H, W] instance masks into [B, N_max, H, W]."""

    def _make_targets_with_masks(self, mask_counts, h=16, w=16):
        """Build target dicts with boolean mask tensors for given instance counts."""
        targets = []
        for n in mask_counts:
            masks = torch.ones(n, h, w, dtype=torch.bool) if n > 0 else torch.zeros(0, h, w, dtype=torch.bool)
            targets.append({"masks": masks, "boxes": torch.zeros(n, 4)})
        return targets

    def test_normal_batch(self):
        """Batch of [2 masks, 3 masks] → shape [2, 3, H, W] float32."""
        from rfdetr.datasets.kornia_transforms import collate_masks

        targets = self._make_targets_with_masks([2, 3])
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=3, image_height=16, image_width=16)

        assert masks_padded.shape == (2, 3, 16, 16), f"Expected (2, 3, 16, 16), got {masks_padded.shape}"
        assert masks_padded.dtype == torch.float32, f"Expected float32, got {masks_padded.dtype}"

    def test_padding_is_zero(self):
        """Padded slots (beyond real instance count) are filled with zeros."""
        from rfdetr.datasets.kornia_transforms import collate_masks

        targets = self._make_targets_with_masks([1, 3])  # image 0 padded to 3
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=3, image_height=16, image_width=16)

        # Image 0: slot 0 real (ones), slots 1-2 zero-padded
        assert masks_padded[0, 0].min() == pytest.approx(1.0), "Real mask slot must be all ones"
        assert masks_padded[0, 1].max() == pytest.approx(0.0), "Padded slot 1 must be all zeros"
        assert masks_padded[0, 2].max() == pytest.approx(0.0), "Padded slot 2 must be all zeros"

    def test_n_max_zero_returns_empty(self):
        """n_max=0 → shape [B, 0, H, W]."""
        from rfdetr.datasets.kornia_transforms import collate_masks

        targets = self._make_targets_with_masks([0, 0])
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=0, image_height=16, image_width=16)

        assert masks_padded.shape == (2, 0, 16, 16), f"Expected (2, 0, 16, 16), got {masks_padded.shape}"

    def test_empty_target_list(self):
        """Empty target list → shape [0, 0, H, W]."""
        from rfdetr.datasets.kornia_transforms import collate_masks

        masks_padded = collate_masks([], torch.device("cpu"), n_max=0, image_height=16, image_width=16)

        assert masks_padded.shape == (0, 0, 16, 16), f"Expected (0, 0, 16, 16), got {masks_padded.shape}"

    def test_targets_without_masks_key(self):
        """Targets without 'masks' key produce all-zero rows."""
        from rfdetr.datasets.kornia_transforms import collate_masks

        targets = [{"boxes": torch.zeros(2, 4)}, {"boxes": torch.zeros(1, 4)}]
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=2, image_height=8, image_width=8)

        assert masks_padded.shape == (2, 2, 8, 8)
        assert masks_padded.max() == pytest.approx(0.0), "Targets without masks key must produce all-zero output"


# ---------------------------------------------------------------------------
# TestBuildKorniaPipelineWithMasks — validates that with_masks=True produces
# a pipeline with mask data_key included.
# ---------------------------------------------------------------------------


class TestBuildKorniaPipelineWithMasks:
    """build_kornia_pipeline(with_masks=True) includes mask in data_keys."""

    @pytest.fixture(autouse=True)
    def _require_kornia(self):
        """Skip when Kornia is unavailable (optional extra not installed in CPU CI)."""
        pytest.importorskip("kornia")

    def test_with_masks_false_is_default(self):
        """with_masks defaults to False; pipeline returns (img, boxes) on call."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=32)
        img = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])
        result = pipeline(img, boxes)
        assert len(result) == 2, f"Detection pipeline must return 2 values, got {len(result)}"

    def test_with_masks_true_returns_three_values(self):
        """with_masks=True: pipeline(img, boxes, masks) returns (img, boxes, masks)."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=32, with_masks=True)
        img = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])
        masks = torch.ones(1, 1, 32, 32, dtype=torch.float32)
        result = pipeline(img, boxes, masks)
        assert len(result) == 3, f"Segmentation pipeline must return 3 values, got {len(result)}"

    def test_with_masks_true_preserves_mask_shape(self):
        """Mask shape [B, N, H, W] is preserved after pipeline pass."""
        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline

        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 0.0}}, resolution=32, with_masks=True)
        img = torch.rand(2, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]], [[8.0, 8.0, 24.0, 24.0]]])
        masks = torch.ones(2, 1, 32, 32, dtype=torch.float32)
        _, _, masks_aug = pipeline(img, boxes, masks)
        assert masks_aug.shape == (2, 1, 32, 32), f"Mask shape must be preserved: {masks_aug.shape}"


# ---------------------------------------------------------------------------
# TestUnpackBoxesWithMasks — validates that unpack_boxes propagates the same
# keep filter to masks when masks_aug is provided.
# ---------------------------------------------------------------------------


class TestUnpackBoxesWithMasks:
    """unpack_boxes with masks_aug keeps/removes masks in sync with boxes."""

    def test_masks_filtered_same_as_boxes(self):
        """Box removed → corresponding mask also removed from output."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        # B=1, N=2: box 0 valid, box 1 zero-area (will be removed)
        boxes_aug = torch.tensor([[[5.0, 5.0, 25.0, 25.0], [30.0, 30.0, 30.0, 30.0]]])
        valid = torch.tensor([[True, True]])
        targets = [
            {
                "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0], [30.0, 30.0, 60.0, 60.0]]),
                "labels": torch.tensor([1, 2]),
            }
        ]
        # 2 masks: instance 0 = all ones, instance 1 = all twos (distinguishable)
        masks_aug = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
        masks_aug[0, 0] = 1.0
        masks_aug[0, 1] = 1.0  # will be removed with box 1

        result = unpack_boxes(boxes_aug, valid, targets, 100, 100, masks_aug=masks_aug)

        assert "masks" in result[0], "masks key must be present in output target"
        assert result[0]["masks"].shape[0] == 1, f"Expected 1 surviving mask, got {result[0]['masks'].shape[0]}"

    def test_masks_converted_to_bool(self):
        """Float masks > 0.5 threshold converted to bool in output."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        boxes_aug = torch.tensor([[[5.0, 5.0, 25.0, 25.0]]])
        valid = torch.tensor([[True]])
        targets = [{"boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]), "labels": torch.tensor([1])}]
        masks_aug = torch.full((1, 1, 8, 8), 0.8, dtype=torch.float32)  # float, all 0.8

        result = unpack_boxes(boxes_aug, valid, targets, 100, 100, masks_aug=masks_aug)

        assert result[0]["masks"].dtype == torch.bool, f"masks must be bool, got {result[0]['masks'].dtype}"
        assert result[0]["masks"].all(), "All values > 0.5 should be True after thresholding"

    def test_no_masks_aug_leaves_masks_key_unchanged(self):
        """When masks_aug=None, existing masks key in target is preserved as-is."""
        from rfdetr.datasets.kornia_transforms import unpack_boxes

        boxes_aug = torch.tensor([[[5.0, 5.0, 25.0, 25.0]]])
        valid = torch.tensor([[True]])
        original_mask = torch.ones(1, 8, 8, dtype=torch.bool)
        targets = [
            {
                "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
                "labels": torch.tensor([1]),
                "masks": original_mask,
            }
        ]

        result = unpack_boxes(boxes_aug, valid, targets, 100, 100, masks_aug=None)

        assert "masks" in result[0], "masks key must still be present when masks_aug=None"
        assert result[0]["masks"] is original_mask, "Original masks object must be preserved unchanged"
