# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Input contracts of the Kornia keypoint augmentation helpers."""

import pytest
import torch

from rfdetr.datasets.kornia_transforms import build_kornia_pipeline, keypoint_horizontal_flip_mask
from tests._markers import _requires_kornia


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
