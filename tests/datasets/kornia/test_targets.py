# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the targets the Kornia GPU augmentation pipeline carries: box and mask collation, unpacking and keypoints.

All tests in this module are CPU-compatible — Kornia operates on CPU tensors identically to GPU tensors, so no
``@pytest.mark.gpu`` is needed.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rfdetr.datasets._torchvision import RandomHorizontalFlip
from rfdetr.datasets.kornia_transforms import (
    build_kornia_pipeline,
    collate_boxes,
    collate_keypoints,
    collate_masks,
    keypoint_horizontal_flip_mask,
    unpack_boxes,
)
from tests._markers import _requires_kornia


@_requires_kornia
class TestCollateBoxes:
    """collate_boxes packs variable-length boxes into [B, N_max, 4] with mask."""

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

    @pytest.mark.parametrize(
        ("box_counts", "expected_boxes_shape", "expected_valid_shape"),
        [
            pytest.param([2, 3], (2, 3, 4), (2, 3), id="two-images-padded-to-max"),
            pytest.param([], (0, 0, 4), (0, 0), id="empty-batch"),
            pytest.param([0], (1, 0, 4), (1, 0), id="image-without-boxes"),
            pytest.param([3], (1, 3, 4), (1, 3), id="single-image"),
        ],
    )
    def test_pads_to_max_box_count(self, box_counts, expected_boxes_shape, expected_valid_shape):
        """Boxes pack into [B, N_max, 4] with a boolean validity mask [B, N_max], including empty batches."""
        boxes_padded, valid = collate_boxes(self._make_targets(box_counts), torch.device("cpu"))

        assert boxes_padded.shape == expected_boxes_shape, f"Expected {expected_boxes_shape}, got {boxes_padded.shape}"
        assert valid.shape == expected_valid_shape, f"Expected valid {expected_valid_shape}, got {valid.shape}"
        assert valid.dtype == torch.bool

    def test_valid_mask_matches_box_count(self):
        """The valid mask has True for real boxes and False for padding."""
        targets = self._make_targets([1, 3])
        _, valid = collate_boxes(targets, torch.device("cpu"))

        # Image 0: 1 real box, 2 padding → [True, False, False]
        assert valid[0].tolist() == [True, False, False], f"Image 0 valid mask wrong: {valid[0].tolist()}"
        # Image 1: 3 real boxes, 0 padding → [True, True, True]
        assert valid[1].tolist() == [True, True, True], f"Image 1 valid mask wrong: {valid[1].tolist()}"


@_requires_kornia
class TestUnpackBoxes:
    """unpack_boxes writes augmented boxes back and removes zero-area entries."""

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

    def test_boxes_follow_masks_and_empty_masks_drop_instances(self):
        """Boxes become the tight mask extent; an instance whose mask vanished is dropped; box-only stays."""
        boxes_aug = [[[0.0, 0.0, 90.0, 90.0], [10.0, 10.0, 40.0, 40.0], [50.0, 50.0, 70.0, 70.0]]]
        valid = [[True, True, True]]
        masks_in = torch.zeros(3, 100, 100, dtype=torch.bool)
        masks_in[0, 20:30, 20:60] = True
        masks_in[1, 10:40, 10:40] = True
        masks_aug = torch.zeros(1, 3, 100, 100)
        masks_aug[0, 0, 20:30, 20:60] = 1.0
        targets = [
            {
                "boxes": torch.tensor([[20.0, 20.0, 60.0, 30.0], [10.0, 10.0, 40.0, 40.0], [50.0, 50.0, 70.0, 70.0]]),
                "labels": torch.tensor([1, 2, 3]),
                "area": torch.tensor([400.0, 900.0, 400.0]),
                "iscrowd": torch.tensor([0, 0, 1]),
                "masks": masks_in,
            }
        ]
        boxes_t, valid_t, tgts, height, width = self._make_inputs(boxes_aug, valid, targets)

        result = unpack_boxes(boxes_t, valid_t, tgts, height, width, masks_aug=masks_aug)

        assert result[0]["labels"].tolist() == [1, 3]
        assert result[0]["iscrowd"].tolist() == [0, 1]
        assert result[0]["boxes"].tolist() == [[20.0, 20.0, 60.0, 30.0], [50.0, 50.0, 70.0, 70.0]]
        assert result[0]["masks"].shape == (2, 100, 100)

    def test_all_boxes_removed_after_aug(self):
        """When all augmented boxes are zero-area, output targets have empty boxes."""
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


class TestCollateMasks:
    """collate_masks packs [N_i, H, W] instance masks into [B, N_max, H, W]."""

    def _make_targets_with_masks(self, mask_counts, h=16, w=16):
        """Build target dicts with boolean mask tensors for given instance counts."""
        targets = []
        for n in mask_counts:
            masks = torch.ones(n, h, w, dtype=torch.bool) if n > 0 else torch.zeros(0, h, w, dtype=torch.bool)
            targets.append({"masks": masks, "boxes": torch.zeros(n, 4)})
        return targets

    @pytest.mark.parametrize(
        ("mask_counts", "n_max", "expected_shape"),
        [
            pytest.param([2, 3], 3, (2, 3, 16, 16), id="two-images"),
            pytest.param([0, 0], 0, (2, 0, 16, 16), id="n-max-zero"),
            pytest.param([], 0, (0, 0, 16, 16), id="empty-target-list"),
        ],
    )
    def test_packs_masks_into_float32_batch(self, mask_counts, n_max, expected_shape):
        """Masks pack into a [B, N_max, H, W] float32 tensor, including n_max=0 and an empty target list."""
        targets = self._make_targets_with_masks(mask_counts)
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=n_max, image_height=16, image_width=16)

        assert masks_padded.shape == expected_shape, f"Expected {expected_shape}, got {masks_padded.shape}"
        assert masks_padded.dtype == torch.float32, f"Expected float32, got {masks_padded.dtype}"

    def test_padding_is_zero(self):
        """Padded slots (beyond real instance count) are filled with zeros."""
        targets = self._make_targets_with_masks([1, 3])  # image 0 padded to 3
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=3, image_height=16, image_width=16)

        # Image 0: slot 0 real (ones), slots 1-2 zero-padded
        assert masks_padded[0, 0].min() == pytest.approx(1.0), "Real mask slot must be all ones"
        assert masks_padded[0, 1].max() == pytest.approx(0.0), "Padded slot 1 must be all zeros"
        assert masks_padded[0, 2].max() == pytest.approx(0.0), "Padded slot 2 must be all zeros"

    def test_targets_without_masks_key(self):
        """Targets without 'masks' key produce all-zero rows."""
        targets = [{"boxes": torch.zeros(2, 4)}, {"boxes": torch.zeros(1, 4)}]
        masks_padded = collate_masks(targets, torch.device("cpu"), n_max=2, image_height=8, image_width=8)

        assert masks_padded.shape == (2, 2, 8, 8)
        assert masks_padded.max() == pytest.approx(0.0), "Targets without masks key must produce all-zero output"


@_requires_kornia
class TestBuildKorniaPipelineWithMasks:
    """build_kornia_pipeline(with_masks=True) includes mask in data_keys."""

    def test_with_masks_false_is_default(self):
        """with_masks defaults to False; pipeline returns (img, boxes) on call."""
        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=32)
        img = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])
        result = pipeline(img, boxes)
        assert len(result) == 2, f"Detection pipeline must return 2 values, got {len(result)}"

    def test_with_masks_true_returns_three_values(self):
        """with_masks=True: pipeline(img, boxes, masks) returns (img, boxes, masks)."""
        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=32, with_masks=True)
        img = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])
        masks = torch.ones(1, 1, 32, 32, dtype=torch.float32)
        result = pipeline(img, boxes, masks)
        assert len(result) == 3, f"Segmentation pipeline must return 3 values, got {len(result)}"

    def test_with_masks_true_preserves_mask_shape(self):
        """Mask shape [B, N, H, W] is preserved after pipeline pass."""
        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 0.0}}, resolution=32, with_masks=True)
        img = torch.rand(2, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]], [[8.0, 8.0, 24.0, 24.0]]])
        masks = torch.ones(2, 1, 32, 32, dtype=torch.float32)
        _, _, masks_aug = pipeline(img, boxes, masks)
        assert masks_aug.shape == (2, 1, 32, 32), f"Mask shape must be preserved: {masks_aug.shape}"

    def test_keypoint_pipeline_flips_points_with_boxes_and_masks(self) -> None:
        """A keypoint pipeline transports every geometric target in the same draw."""
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}},
            16,
            with_masks=True,
            include_keypoints=True,
            with_keypoints=True,
            keypoint_flip_pairs=[0, 1],
        )
        image = torch.zeros(1, 3, 16, 16)
        boxes = torch.tensor([[[2.0, 2.0, 8.0, 8.0]]])
        masks = torch.zeros(1, 1, 16, 16)
        targets = [
            {
                "boxes": boxes[0],
                "labels": torch.tensor([1]),
                "keypoints": torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 1.0], [0.0, 0.0, 2.0]]]),
            }
        ]
        points, visibility = collate_keypoints(targets, torch.device("cpu"), n_max=1)

        _, boxes_out, _, points_out = pipeline(image, boxes - 0.5, masks, points - 0.5)
        result = unpack_boxes(
            boxes_out + 0.5,
            torch.tensor([[True]]),
            targets,
            16,
            16,
            keypoints_aug=points_out + 0.5,
            keypoint_visibility=visibility,
            keypoint_flip_pairs=[0, 1],
            keypoint_flip_mask=torch.tensor([True]),
        )

        torch.testing.assert_close(
            result[0]["keypoints"], torch.tensor([[[10.0, 7.0, 1.0], [13.0, 4.0, 2.0], [16.0, 0.0, 2.0]]])
        )


@_requires_kornia
class TestKeypointTargetTransport:
    """Keypoints retain visibility and instance alignment through GPU augmentation."""

    def test_horizontal_flip_matches_torchvision_continuous_coordinates_on_boundary(self) -> None:
        """Kornia matches the default CPU backend's continuous keypoint convention."""
        target = {
            "boxes": torch.tensor([[0.0, 2.0, 8.0, 12.0]]),
            "labels": torch.tensor([0]),
            "keypoints": torch.tensor([[[0.0, 8.0, 2.0], [3.0, 8.0, 2.0]]]),
        }
        cpu_flip = RandomHorizontalFlip(p=1.0, keypoint_flip_pairs=[0, 1])
        _, cpu_target = cpu_flip(torch.zeros(3, 16, 16), target)
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}}, 16, with_masks=True, with_keypoints=True, keypoint_flip_pairs=[0, 1]
        )
        points, visibility = collate_keypoints([target], torch.device("cpu"), 1)
        _, boxes, _, points = pipeline(
            torch.zeros(1, 3, 16, 16), target["boxes"].unsqueeze(0) - 0.5, torch.zeros(1, 1, 16, 16), points - 0.5
        )
        gpu_target = unpack_boxes(
            boxes + 0.5,
            torch.tensor([[True]]),
            [target],
            16,
            16,
            keypoints_aug=points + 0.5,
            keypoint_visibility=visibility,
            keypoint_flip_pairs=[0, 1],
            keypoint_flip_mask=keypoint_horizontal_flip_mask(pipeline, 1, torch.device("cpu")),
        )[0]
        torch.testing.assert_close(cpu_target["keypoints"], torch.tensor([[[13.0, 8.0, 2.0], [16.0, 8.0, 2.0]]]))
        torch.testing.assert_close(gpu_target["keypoints"], cpu_target["keypoints"])

    def test_flip_mask_tracks_each_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A batch can flip one image without relabeling joints in another.

        The pipeline is a real one-flip ``AugmentationSequential`` whose sampled draws are replaced by a known per-image
        pattern, so the mask must follow the draw of each image.
        """
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 0.5}}, 16, with_masks=True, with_keypoints=True, keypoint_flip_pairs=[0, 1]
        )
        draw = SimpleNamespace(name="RandomHorizontalFlip_0", data={"batch_prob": torch.tensor([1.0, 0.0])})
        monkeypatch.setattr(pipeline, "_params", [draw], raising=False)

        assert keypoint_horizontal_flip_mask(pipeline, 2, torch.device("cpu")).tolist() == [True, False]

    def test_flip_mask_tracks_each_image_on_a_real_two_image_kornia_batch(self) -> None:
        """A real ``AugmentationSequential`` batch call swaps joints only for the image it actually flipped."""
        target = {
            "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 1.0]]]),
        }
        targets = [target, target]
        points, visibility = collate_keypoints(targets, torch.device("cpu"), n_max=1)
        points = points - 0.5
        boxes = torch.stack([target["boxes"], target["boxes"]]) - 0.5
        image = torch.zeros(2, 3, 16, 16)
        masks = torch.zeros(2, 1, 16, 16)

        mask = None
        for seed in range(10):
            torch.manual_seed(seed)
            pipeline = build_kornia_pipeline(
                {"HorizontalFlip": {"p": 0.5}},
                16,
                with_masks=True,
                include_keypoints=True,
                with_keypoints=True,
                keypoint_flip_pairs=[0, 1],
            )
            _, boxes_aug, _, points_aug = pipeline(image.clone(), boxes.clone(), masks.clone(), points.clone())
            mask = keypoint_horizontal_flip_mask(pipeline, 2, torch.device("cpu"))
            if bool(mask[0]) != bool(mask[1]):
                break
        else:
            raise AssertionError("no seed in range(10) produced differing per-image Kornia flip draws")

        boxes_aug = boxes_aug + 0.5
        points_aug = points_aug + 0.5
        flipped_idx, identity_idx = (0, 1) if mask[0] else (1, 0)
        flipped_target = unpack_boxes(
            boxes_aug[flipped_idx : flipped_idx + 1],
            torch.tensor([[True]]),
            [target],
            16,
            16,
            keypoints_aug=points_aug[flipped_idx : flipped_idx + 1],
            keypoint_visibility=visibility[flipped_idx : flipped_idx + 1],
            keypoint_flip_pairs=[0, 1],
            keypoint_flip_mask=mask[flipped_idx : flipped_idx + 1],
        )[0]
        identity_target = unpack_boxes(
            boxes_aug[identity_idx : identity_idx + 1],
            torch.tensor([[True]]),
            [target],
            16,
            16,
            keypoints_aug=points_aug[identity_idx : identity_idx + 1],
            keypoint_visibility=visibility[identity_idx : identity_idx + 1],
            keypoint_flip_pairs=[0, 1],
            keypoint_flip_mask=mask[identity_idx : identity_idx + 1],
        )[0]

        torch.testing.assert_close(flipped_target["keypoints"], torch.tensor([[[10.0, 7.0, 1.0], [13.0, 4.0, 2.0]]]))
        torch.testing.assert_close(identity_target["keypoints"], target["keypoints"])

    def test_flip_draws_fail_closed_when_kornia_params_are_unavailable(self) -> None:
        """A changed Kornia parameter API cannot silently leave paired joints unswapped.

        A real pipeline holding a horizontal flip that has not run exposes no sampled draws, which is exactly what a
        Kornia release hiding its parameters looks like to the caller.
        """
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}}, 16, with_masks=True, with_keypoints=True, keypoint_flip_pairs=[0, 1]
        )
        with pytest.raises(RuntimeError) as exc_info:
            keypoint_horizontal_flip_mask(pipeline, 1, torch.device("cpu"))

        assert "horizontal-flip draws" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param({"VerticalFlip": {"p": 1.0}}, (3.0, 12.0), id="vertical-flip"),
            pytest.param({"Rotate": {"limit": (90, 90), "p": 1.0}}, (4.0, 13.0), id="rotate-90"),
        ],
    )
    def test_geometric_transforms_use_continuous_coordinates(
        self, config: dict[str, dict[str, Any]], expected: tuple[float, float]
    ) -> None:
        """Flip and rotation agree with the default CPU path's continuous frame."""
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0]]),
                "labels": torch.tensor([1]),
                "keypoints": torch.tensor([[[3.0, 4.0, 2.0]]]),
            }
        ]
        points, visibility = collate_keypoints(targets, torch.device("cpu"), 1)
        pipeline = build_kornia_pipeline(config, 16, with_masks=True, include_keypoints=True, with_keypoints=True)
        _, boxes_aug, _, points_aug = pipeline(
            torch.zeros(1, 3, 16, 16), targets[0]["boxes"].unsqueeze(0) - 0.5, torch.zeros(1, 1, 16, 16), points - 0.5
        )

        result = unpack_boxes(
            boxes_aug + 0.5,
            torch.tensor([[True]]),
            targets,
            16,
            16,
            keypoints_aug=points_aug + 0.5,
            keypoint_visibility=visibility,
        )

        torch.testing.assert_close(result[0]["keypoints"], torch.tensor([[[*expected, 2.0]]]))

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"Affine": {"rotate": (90, 90), "p": 1.0}}, id="affine"),
            pytest.param(
                {"ShiftScaleRotate": {"shift_limit": 0.0, "scale_limit": 0.0, "rotate_limit": (90, 90), "p": 1.0}},
                id="shift-scale-rotate",
            ),
        ],
    )
    def test_fixed_rotation_moves_keypoint_to_analytic_position(self, config: dict[str, dict[str, Any]]) -> None:
        """A 90-degree Affine or ShiftScaleRotate turns a joint about the image center.

        With rotation pinned to ``(90, 90)`` and no shift or scale, the joint at ``(10, 12)`` on a 32-pixel image sits
        at ``(-6, -4)`` from the center ``(16, 16)``. Kornia's ``RandomAffine`` maps ``(dx, dy)`` to ``(-dy, dx)``, so
        the joint must land at ``(20, 10)``; an untransformed joint stays at ``(10, 12)``.
        """
        pipeline = build_kornia_pipeline(config, 32, with_masks=True, with_keypoints=True)
        boxes = torch.tensor([[[4.0, 4.0, 20.0, 20.0]]]) - 0.5
        point = torch.tensor([[[10.0, 12.0]]]) - 0.5

        _, _, _, point_out = pipeline(torch.zeros(1, 3, 32, 32), boxes, torch.zeros(1, 1, 32, 32), point)

        torch.testing.assert_close(point_out + 0.5, torch.tensor([[[20.0, 10.0]]]))

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {"Affine": {"rotate": (0, 0), "translate_percent": (0.25, 0.25), "scale": (1.0, 1.0), "p": 1.0}},
                id="affine",
            ),
            pytest.param(
                {"ShiftScaleRotate": {"shift_limit": 0.25, "scale_limit": 0.0, "rotate_limit": 0, "p": 1.0}},
                id="shift-scale-rotate",
            ),
        ],
    )
    def test_translation_moves_keypoint_by_the_same_offset_as_its_box(self, config: dict[str, dict[str, Any]]) -> None:
        """A pure translation shifts a joint by exactly the offset it shifts the box, within the configured bound.

        The shift is sampled, so the offset is read off the box corner: the joint must move by the same nonzero
        ``(dx, dy)``, and that offset can never exceed ``0.25 * 32 = 8`` pixels on either axis.
        """
        pipeline = build_kornia_pipeline(config, 32, with_masks=True, with_keypoints=True)
        boxes = torch.tensor([[[4.0, 4.0, 20.0, 20.0]]]) - 0.5
        point = torch.tensor([[[10.0, 12.0]]]) - 0.5

        _, boxes_out, _, point_out = pipeline(torch.zeros(1, 3, 32, 32), boxes, torch.zeros(1, 1, 32, 32), point)

        box_offset = boxes_out[0, 0, :2] - boxes[0, 0, :2]
        torch.testing.assert_close(point_out[0, 0] - point[0, 0], box_offset)
        assert 0.5 < box_offset.abs().max() <= 8.0 + 1e-4, f"sampled offset {box_offset.tolist()} is not a real shift"

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"Perspective": {"scale": (0.5, 0.5), "p": 1.0}}, id="perspective"),
            pytest.param({"Affine": {"rotate": (90, 90), "p": 1.0}}, id="affine-rotate"),
        ],
    )
    def test_keypoint_lands_on_the_warped_image_marker(self, config: dict[str, dict[str, Any]]) -> None:
        """A one-hot marker pixel and the joint placed on it end up at the same warped position.

        The marker sits at row 26, column 26 where the joint is. After the warp, the brightest pixel of the output image
        must be within one pixel of the transported joint, and that joint must have moved well away from its start so an
        untransformed joint cannot pass.
        """
        pipeline = build_kornia_pipeline(config, 32, with_keypoints=True)
        image = torch.zeros(1, 3, 32, 32)
        image[:, :, 26, 26] = 1.0
        boxes = torch.tensor([[[4.0, 4.0, 28.0, 28.0]]])
        point = torch.tensor([[[26.0, 26.0]]])

        image_out, _, point_out = pipeline(image, boxes, point)

        peak = image_out[0].mean(dim=0).flatten().argmax().item()
        marker_xy = torch.tensor([peak % 32, peak // 32], dtype=torch.float32)
        assert (point_out[0, 0] - point[0, 0]).abs().max() > 2.0, "joint barely moved; the check would be vacuous"
        assert (marker_xy - point_out[0, 0]).abs().max() <= 1.0, (
            f"joint {point_out[0, 0].tolist()} is not on the warped marker pixel {marker_xy.tolist()}"
        )

    def test_collate_keypoints_copies_coordinates_without_frame_shift(self) -> None:
        """Collated joints keep the caller's coordinate frame; any half-pixel shift is the caller's job.

        The data module shifts boxes and joints by ``-0.5`` around the pipeline itself, so the collate step must hand
        over the annotation coordinates untouched and park the visibility values in a separate tensor.
        """
        targets = [
            {"keypoints": torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 0.0]], [[11.0, 12.0, 2.0], [13.0, 14.0, 1.0]]])}
        ]

        points, visibility = collate_keypoints(targets, torch.device("cpu"), n_max=2)

        torch.testing.assert_close(points, torch.tensor([[[3.0, 4.0], [6.0, 7.0], [11.0, 12.0], [13.0, 14.0]]]))
        torch.testing.assert_close(visibility, torch.tensor([[[2.0, 0.0], [2.0, 1.0]]]))

    def test_dropped_leading_instance_leaves_the_trailing_survivor_keypoints(self) -> None:
        """Dropping the first instance keeps the second instance's own joints, not the first's.

        Instance 0 collapses to zero width while instance 1 stays visible and in bounds, so an unpacker that keeps the
        first ``n_kept`` rows instead of the rows selected by the keep mask returns the wrong joints.
        """
        targets = [
            {
                "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0], [10.0, 10.0, 14.0, 14.0]]),
                "labels": torch.tensor([1, 2]),
                "keypoints": torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 2.0]], [[11.0, 12.0, 2.0], [13.0, 14.0, 1.0]]]),
            }
        ]
        _, visibility = collate_keypoints(targets, torch.device("cpu"), n_max=2)
        boxes_aug = torch.tensor([[[1.0, 1.0, 1.0, 8.0], [10.0, 10.0, 14.0, 14.0]]])
        points_aug = torch.tensor([[[3.0, 4.0], [6.0, 7.0], [11.0, 12.0], [13.0, 14.0]]])

        result = unpack_boxes(
            boxes_aug,
            torch.tensor([[True, True]]),
            targets,
            16,
            16,
            keypoints_aug=points_aug,
            keypoint_visibility=visibility,
        )

        torch.testing.assert_close(result[0]["boxes"], torch.tensor([[10.0, 10.0, 14.0, 14.0]]))
        torch.testing.assert_close(result[0]["keypoints"], torch.tensor([[[11.0, 12.0, 2.0], [13.0, 14.0, 1.0]]]))

    @pytest.mark.parametrize(
        ("xy", "visibility"),
        [
            pytest.param((5.0, 6.0), 0.0, id="invisible-but-inside"),
            pytest.param((16.5, 6.0), 2.0, id="visible-but-right-of-image"),
            pytest.param((-0.5, 6.0), 2.0, id="visible-but-left-of-image"),
            pytest.param((5.0, 16.5), 2.0, id="visible-but-below-image"),
            pytest.param((5.0, -0.5), 2.0, id="visible-but-above-image"),
        ],
    )
    def test_invisible_or_outside_joint_is_zeroed_while_its_neighbor_survives(
        self, xy: tuple[float, float], visibility: float
    ) -> None:
        """A joint that is invisible or ends up outside the image is zeroed entirely, x, y and v.

        The instance and its second joint stay valid, so the zeroing must be per joint and must not touch the
        neighboring joint or drop the instance.
        """
        targets = [
            {
                "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]),
                "labels": torch.tensor([1]),
                "keypoints": torch.tensor([[[*xy, visibility], [6.0, 7.0, 2.0]]]),
            }
        ]
        _, visibility_tensor = collate_keypoints(targets, torch.device("cpu"), n_max=1)
        points_aug = torch.tensor([[[*xy], [6.0, 7.0]]])

        result = unpack_boxes(
            torch.tensor([[[1.0, 1.0, 8.0, 8.0]]]),
            torch.tensor([[True]]),
            targets,
            16,
            16,
            keypoints_aug=points_aug,
            keypoint_visibility=visibility_tensor,
        )

        torch.testing.assert_close(result[0]["keypoints"], torch.tensor([[[0.0, 0.0, 0.0], [6.0, 7.0, 2.0]]]))

    def test_joint_on_the_image_border_is_kept(self) -> None:
        """A visible joint exactly on ``x == W`` and ``y == 0`` is inside the closed image and is kept."""
        targets = [{"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]), "labels": torch.tensor([1])}]

        result = unpack_boxes(
            torch.tensor([[[1.0, 1.0, 8.0, 8.0]]]),
            torch.tensor([[True]]),
            targets,
            16,
            16,
            keypoints_aug=torch.tensor([[[16.0, 0.0]]]),
            keypoint_visibility=torch.tensor([[[2.0]]]),
        )

        torch.testing.assert_close(result[0]["keypoints"], torch.tensor([[[16.0, 0.0, 2.0]]]))

    def test_all_instances_dropped_leaves_empty_keypoints(self) -> None:
        """When every box collapses, the target keeps no boxes and no keypoint rows but still reports K joints."""
        targets = [
            {
                "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0], [10.0, 10.0, 14.0, 14.0]]),
                "labels": torch.tensor([1, 2]),
                "keypoints": torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 2.0]], [[11.0, 12.0, 2.0], [13.0, 14.0, 1.0]]]),
            }
        ]
        points_aug, visibility = collate_keypoints(targets, torch.device("cpu"), n_max=2)
        boxes_aug = torch.tensor([[[1.0, 1.0, 1.0, 8.0], [10.0, 10.0, 14.0, 10.0]]])

        result = unpack_boxes(
            boxes_aug,
            torch.tensor([[True, True]]),
            targets,
            16,
            16,
            keypoints_aug=points_aug,
            keypoint_visibility=visibility,
        )

        assert result[0]["boxes"].shape == (0, 4)
        assert result[0]["keypoints"].shape == (0, 2, 3)

    def test_unequal_instance_counts_do_not_leak_padded_points(self) -> None:
        """Joint padding follows box padding for a mixed-size batch."""
        targets = [
            {
                "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0]]),
                "labels": torch.tensor([1]),
                "keypoints": torch.tensor([[[3.0, 3.0, 2.0]]]),
            },
            {
                "boxes": torch.tensor([[2.0, 2.0, 6.0, 6.0], [8.0, 8.0, 12.0, 12.0]]),
                "labels": torch.tensor([2, 3]),
                "keypoints": torch.tensor([[[4.0, 4.0, 1.0]], [[10.0, 10.0, 2.0]]]),
            },
        ]
        boxes, valid = collate_boxes(targets, torch.device("cpu"))
        points, visibility = collate_keypoints(targets, torch.device("cpu"), valid.shape[1])
        result = unpack_boxes(boxes, valid, targets, 16, 16, keypoints_aug=points, keypoint_visibility=visibility)

        assert [item["keypoints"].shape[0] for item in result] == [1, 2]
        torch.testing.assert_close(result[0]["keypoints"], targets[0]["keypoints"])
        torch.testing.assert_close(result[1]["keypoints"], targets[1]["keypoints"])


class TestUnpackBoxesWithMasks:
    """unpack_boxes with masks_aug keeps/removes masks in sync with boxes."""

    def test_masks_filtered_same_as_boxes(self):
        """Box removed → corresponding mask also removed from output."""
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
        boxes_aug = torch.tensor([[[5.0, 5.0, 25.0, 25.0]]])
        valid = torch.tensor([[True]])
        targets = [{"boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]), "labels": torch.tensor([1])}]
        masks_aug = torch.full((1, 1, 8, 8), 0.8, dtype=torch.float32)  # float, all 0.8

        result = unpack_boxes(boxes_aug, valid, targets, 100, 100, masks_aug=masks_aug)

        assert result[0]["masks"].dtype == torch.bool, f"masks must be bool, got {result[0]['masks'].dtype}"
        assert result[0]["masks"].all(), "All values > 0.5 should be True after thresholding"

    def test_no_masks_aug_leaves_masks_key_unchanged(self):
        """When masks_aug=None, existing masks key in target is preserved as-is."""
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


@pytest.fixture
def flip_pipeline():
    """Always-flip Kornia pipeline carrying keypoints with one flip pair."""
    return build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, 16, with_keypoints=True, keypoint_flip_pairs=[0, 1])


@_requires_kornia
class TestBuildKorniaPipelineFlipPairs:
    """``build_kornia_pipeline`` accepts flip pairs only when the pipeline actually transports keypoints."""

    @pytest.mark.parametrize("include_keypoints", [False, True])
    def test_pairs_without_keypoint_data_key_raise(self, include_keypoints):
        """Flip pairs without ``with_keypoints=True`` raise instead of keeping flips that never relabel joints.

        ``include_keypoints`` alone only filters horizontal flips; without the keypoints data key the caller has no
        augmented joints to swap, so pairs would silently keep flips that corrupt left/right labels.
        """
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline(
                {"HorizontalFlip": {"p": 1.0}}, 16, include_keypoints=include_keypoints, keypoint_flip_pairs=[0, 1]
            )

        assert "requires with_keypoints=True" in str(exc_info.value)

    def test_pairs_with_keypoint_data_key_build(self, flip_pipeline):
        """Even flip pairs with ``with_keypoints=True`` build a pipeline that keeps its horizontal flip.

        This is the supported keypoint configuration: the caller relabels joint slots after each flip.
        """
        assert len(list(flip_pipeline.children())) == 1


@_requires_kornia
class TestKeypointHorizontalFlipMask:
    """``keypoint_horizontal_flip_mask`` detects horizontal flips from the pipeline's own transforms."""

    def test_reads_flip_draws_after_forward(self, flip_pipeline):
        """After a forward pass through an always-flip pipeline, every image is reported as flipped.

        This is the normal training use: the DataModule runs the pipeline, then asks which images to relabel.
        """
        flip_pipeline(torch.zeros(2, 3, 16, 16), torch.zeros(2, 1, 4), torch.zeros(2, 1, 2))

        assert keypoint_horizontal_flip_mask(flip_pipeline, 2, torch.device("cpu")).tolist() == [True, True]

    def test_missing_flip_draws_raise(self, flip_pipeline):
        """A pipeline holding a horizontal flip but no sampled draws raises instead of skipping relabeling.

        Before any forward pass Kornia has sampled nothing, which is indistinguishable from a Kornia release that
        stopped exposing the draws; silently returning "not flipped" would leave left/right joints swapped.
        """
        with pytest.raises(RuntimeError) as exc_info:
            keypoint_horizontal_flip_mask(flip_pipeline, 1, torch.device("cpu"))

        assert "did not expose horizontal-flip draws" in str(exc_info.value)
