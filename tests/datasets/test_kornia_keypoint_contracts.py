# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Input contracts of the Kornia keypoint augmentation helpers."""

import pytest
import torch

from rfdetr.config import AugmentationBackend
from rfdetr.datasets.kornia_transforms import build_kornia_pipeline, keypoint_horizontal_flip_mask

kornia_only = pytest.mark.skipif(
    not AugmentationBackend.KORNIA._is_available(), reason="kornia not installed, run: pip install rfdetr[augment]"
)


@kornia_only
class TestBuildKorniaPipelineFlipPairs:
    """``build_kornia_pipeline`` accepts flip pairs only when the pipeline actually transports keypoints."""

    @pytest.mark.parametrize("include_keypoints", [False, True])
    def test_pairs_without_keypoint_data_key_raise(self, include_keypoints):
        """Flip pairs without ``with_keypoints=True`` raise instead of keeping flips that never relabel joints.

        ``include_keypoints`` alone only filters horizontal flips; without the keypoints data key the caller has no
        augmented joints to swap, so pairs would silently keep flips that corrupt left/right labels.
        """
        with pytest.raises(ValueError, match="requires with_keypoints=True"):
            build_kornia_pipeline(
                {"HorizontalFlip": {"p": 1.0}}, 16, include_keypoints=include_keypoints, keypoint_flip_pairs=[0, 1]
            )

    def test_pairs_with_keypoint_data_key_build(self):
        """Even flip pairs with ``with_keypoints=True`` build a pipeline that keeps its horizontal flip.

        This is the supported keypoint configuration: the caller relabels joint slots after each flip.
        """
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}}, 16, with_keypoints=True, keypoint_flip_pairs=[0, 1]
        )

        assert len(list(pipeline.children())) == 1


@kornia_only
class TestKeypointHorizontalFlipMask:
    """``keypoint_horizontal_flip_mask`` detects horizontal flips from the pipeline's own transforms."""

    def test_reads_flip_draws_after_forward(self):
        """After a forward pass through an always-flip pipeline, every image is reported as flipped.

        This is the normal training use: the DataModule runs the pipeline, then asks which images to relabel.
        """
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}}, 16, with_keypoints=True, keypoint_flip_pairs=[0, 1]
        )
        pipeline(torch.zeros(2, 3, 16, 16), torch.zeros(2, 1, 4), torch.zeros(2, 1, 2))

        assert keypoint_horizontal_flip_mask(pipeline, 2, torch.device("cpu")).tolist() == [True, True]

    def test_missing_flip_draws_raise(self):
        """A pipeline holding a horizontal flip but no sampled draws raises instead of skipping relabeling.

        Before any forward pass Kornia has sampled nothing, which is indistinguishable from a Kornia release that
        stopped exposing the draws; silently returning "not flipped" would leave left/right joints swapped.
        """
        pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}}, 16, with_keypoints=True, keypoint_flip_pairs=[0, 1]
        )

        with pytest.raises(RuntimeError, match="did not expose horizontal-flip draws"):
            keypoint_horizontal_flip_mask(pipeline, 1, torch.device("cpu"))
