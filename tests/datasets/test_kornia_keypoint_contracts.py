# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Input contracts of the Kornia keypoint augmentation helpers."""

import pytest

from rfdetr.datasets.kornia_transforms import build_kornia_pipeline
from rfdetr.utilities.imports import _IS_KORNIA_INSTALLED

kornia_only = pytest.mark.skipif(not _IS_KORNIA_INSTALLED, reason="kornia not installed")


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
