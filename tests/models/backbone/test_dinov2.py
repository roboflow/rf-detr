# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for windowed-attention block index derivation in the DINOv2 backbone.

See issue #1223: `out_feature_indexes` (1-indexed HF stage numbers) is reused to derive `window_block_indexes`
(0-indexed encoder block positions) without renumbering, so the computed routing does not match a naive 1:1 reading of
the RF-DETR paper's Figure 2 for every config. These tests pin the current, checkpoint-compatible routing so any future
change to `compute_window_block_indexes` is a deliberate, visible decision rather than a silent drift.
"""

import pytest

from rfdetr.config import RFDETRBaseConfig, RFDETRSmallConfig
from rfdetr.models.backbone.dinov2 import compute_window_block_indexes


class TestComputeWindowBlockIndexes:
    """Tests for compute_window_block_indexes default-derivation and override paths."""

    @pytest.mark.parametrize(
        "out_feature_indexes, expected",
        [
            pytest.param(
                RFDETRBaseConfig.model_fields["out_feature_indexes"].default,
                [0, 1, 3, 4, 6, 7, 9, 10],
                id="base-config-2-5-8-11",
            ),
            pytest.param(
                RFDETRSmallConfig.model_fields["out_feature_indexes"].default,
                [0, 1, 2, 4, 5, 7, 8, 10, 11],
                id="small-config-3-6-9-12",
            ),
        ],
    )
    def test_default_derivation_matches_pinned_official_config_routing(self, out_feature_indexes, expected):
        """Derived windowed-block set for each distinct official out_feature_indexes must not drift."""
        result = compute_window_block_indexes(out_feature_indexes)

        assert result == expected

    def test_default_derivation_is_complement_of_out_feature_indexes(self):
        """Every block up to the last exported stage is either windowed or in out_feature_indexes."""
        out_feature_indexes = [3, 6, 9, 12]

        result = compute_window_block_indexes(out_feature_indexes)

        assert set(result).isdisjoint(out_feature_indexes)
        assert set(result) | set(out_feature_indexes) == set(range(out_feature_indexes[-1] + 1))

    def test_explicit_override_returned_unchanged(self):
        """An explicit window_block_indexes bypasses derivation entirely."""
        override = [0, 1, 3, 4, 6, 7, 9, 10]

        result = compute_window_block_indexes([3, 6, 9, 12], window_block_indexes=override)

        assert result == override
