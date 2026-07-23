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
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2, compute_window_block_indexes


class TestComputeWindowBlockIndexes:
    """Tests for compute_window_block_indexes default-derivation and override paths."""

    @pytest.mark.parametrize(
        "out_feature_indexes, expected",
        [
            pytest.param(
                # `RFDETRBaseConfig` here is a drift-guard introspection target: it reads the config's live
                # default value so this test fails loudly if that default ever changes, not a usage example.
                # AGENTS.md's guidance against RFDETRBaseConfig/RFDETRBase is scoped to models/examples/tests
                # that *run* the deprecated model — it does not apply to introspecting a pydantic field default.
                RFDETRBaseConfig.model_fields["out_feature_indexes"].default,
                [0, 1, 3, 4, 6, 7, 9, 10],
                id="base-config-2-5-8-11",
            ),
            pytest.param(
                # Verified via `grep out_feature_indexes src/rfdetr/config.py` (and rfdetr_plus/models/detection.py):
                # every released non-Base config shares this exact default — Nano, Small, Medium, Large, XLarge, and
                # 2XLarge detection configs, every RFDETRSeg* size (Preview/Nano/Small/Medium/Large/XLarge/2XLarge),
                # and RFDETRKeypointPreviewConfig. `compute_window_block_indexes` is a pure function of its input
                # list with no per-variant branching, so this single case already covers all of them — pinning each
                # variant separately would add rows with zero new failure-mode coverage.
                RFDETRSmallConfig.model_fields["out_feature_indexes"].default,
                [0, 1, 2, 4, 5, 7, 8, 10, 11],
                id="small-config-3-6-9-12-shared-by-all-released-non-base-variants",
            ),
        ],
    )
    def test_default_derivation_matches_pinned_official_config_routing(self, out_feature_indexes, expected):
        """Derived windowed-block set for each distinct official out_feature_indexes must not drift.

        The second case (`small-config-...`) represents every released non-Base RF-DETR config, not just Small — see the
        inline comment on that `pytest.param` for the verified list of variants sharing this default.
        """
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


class TestWindowBlockIndexesReachesEncoderConfig:
    """Regression tests that `window_block_indexes` actually reaches the encoder config end-to-end.

    `TestComputeWindowBlockIndexes` above only exercises the pure `compute_window_block_indexes` helper; it never
    constructs `DinoV2`/`Backbone`, so it cannot catch a regression in the wiring that forwards `window_block_indexes`
    from `Backbone.__init__` into `DinoV2(...)` (previously silently dropped). `out_feature_indexes=[12]` is used below
    (rather than a multi-stage default) to keep construction weightless/offline: `load_dinov2_weights=False` builds the
    windowed encoder directly instead of downloading a checkpoint, but the local `dinov2_configs/*.json` files pin a
    single-stage `out_indices: [12]` that a multi-stage `out_feature_indexes` override would leave stale, tripping
    transformers' `verify_out_features_out_indices` — an unrelated, pre-existing config bug in `DinoV2.__init__`, not
    something these tests are meant to cover.
    """

    def test_dinov2_window_block_indexes_override_reaches_encoder_config(self):
        """DinoV2 constructed with an explicit window_block_indexes override stores it on encoder.config."""
        override = [0, 1, 3, 4]

        dino = DinoV2(
            out_feature_indexes=[12],
            window_block_indexes=override,
            use_windowed_attn=True,
            load_dinov2_weights=False,
            size="small",
        )

        assert dino.encoder.config.window_block_indexes == override

    def test_dinov2_default_window_block_indexes_reaches_encoder_config(self):
        """DinoV2 constructed without an override still derives window_block_indexes onto encoder.config."""
        out_feature_indexes = [12]

        dino = DinoV2(
            out_feature_indexes=out_feature_indexes,
            use_windowed_attn=True,
            load_dinov2_weights=False,
            size="small",
        )

        assert dino.encoder.config.window_block_indexes == compute_window_block_indexes(out_feature_indexes)

    def test_backbone_window_block_indexes_override_reaches_dinov2_encoder_config(self):
        """Backbone forwards its window_block_indexes override through DinoV2 into the encoder config.

        This is the actual regression guard for `backbone.py`'s `DinoV2(...)` call: reverting the
        `window_block_indexes=window_block_indexes` forwarding line there makes only this test fail, while the
        DinoV2-direct tests above (and all of `TestComputeWindowBlockIndexes`) keep passing.
        """
        override = [0, 1, 3, 4]

        backbone = Backbone(
            name="dinov2_registers_windowed_small",
            out_feature_indexes=[12],
            window_block_indexes=override,
            projector_scale=["P3"],
            load_dinov2_weights=False,
        )

        assert backbone.encoder.encoder.config.window_block_indexes == override
