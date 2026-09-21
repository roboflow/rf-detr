# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for MultiScaleProjector's extra-pool marker scale factor (0.25)."""

from __future__ import annotations

import torch

from rfdetr.models.backbone.projector import MultiScaleProjector


class TestMultiScaleProjectorExtraPoolMarker:
    """``scale_factors`` entries of ``0.25`` mark an extra max-pool, not a pyramid stage."""

    def test_extra_pool_marker_does_not_build_an_empty_stage(self) -> None:
        """A 0.25 entry in ``scale_factors`` must not create its own pyramid stage.

        Regression test for a bug where the extra-pool marker's ``continue`` only skipped the per-input-channel layer
        construction (an inner loop) instead of the whole scale (the outer loop), leaving an empty ``stages_sampling``
        entry that crashed ``forward()`` with ``IndexError`` on ``feat_fuse_list[0]``.
        """
        in_channels = [8, 16]
        projector = MultiScaleProjector(
            in_channels=in_channels,
            out_channels=4,
            scale_factors=[1.0, 0.25],
            num_blocks=1,
        )

        assert len(projector.stages) == 1
        assert len(projector.stages_sampling) == 1
        assert projector.use_extra_pool is True

    def test_extra_pool_marker_appends_a_pooled_feature_map_in_forward(self) -> None:
        """``forward()`` with a 0.25 marker returns one extra, spatially-halved feature map.

        Exercises the full construction-to-forward path that previously raised ``IndexError`` before any output was
        produced.
        """
        in_channels = [8, 16]
        projector = MultiScaleProjector(
            in_channels=in_channels,
            out_channels=4,
            scale_factors=[1.0, 0.25],
            num_blocks=1,
        )
        features = [torch.zeros(1, c, 8, 8) for c in in_channels]

        results = projector(features)

        assert len(results) == 2
        assert results[-1].shape[-2:] == (4, 4)
