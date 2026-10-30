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

import copy
import logging
from unittest import mock

import pytest
import torch
from torch import nn

from rfdetr.config import RFDETRBaseConfig, RFDETRNanoConfig, RFDETRSmallConfig
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2, compute_window_block_indexes
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
    WindowedDinov2WithRegistersEmbeddings,
)
from rfdetr.models.lwdetr import build_model_from_config


def test_compiled_embeddings_preserve_interpolated_outputs_and_gradients() -> None:
    """Dynamic compilation must preserve positional gradients across resized inputs without suppressed errors."""
    torch.manual_seed(0)
    eager = WindowedDinov2WithRegistersEmbeddings(
        WindowedDinov2WithRegistersConfig(image_size=32, patch_size=16, hidden_size=16)
    )
    model = copy.deepcopy(eager)
    compiled = torch.compile(model, backend="aot_eager", dynamic=True)
    with torch._dynamo.config.patch(suppress_errors=False):
        for height, width in ((48, 48), (64, 48)):
            inputs = torch.randn(1, 3, height, width)
            expected = eager(inputs)
            actual = compiled(inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            expected.square().mean().backward()
            actual.square().mean().backward()
            torch.testing.assert_close(
                model.position_embeddings.grad, eager.position_embeddings.grad, rtol=1e-4, atol=1e-6
            )
            eager.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)


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


class TestWindowBlockIndexesValidation:
    """Validation guards in `DinoV2.__init__` for empty `out_feature_indexes` and invalid overrides."""

    def test_empty_out_feature_indexes_raises(self):
        """An empty out_feature_indexes raises ValueError instead of an unhelpful IndexError."""
        with pytest.raises(ValueError, match="out_feature_indexes must be non-empty"):
            DinoV2(
                out_feature_indexes=[],
                use_windowed_attn=True,
                load_dinov2_weights=False,
                size="small",
            )

    @pytest.mark.parametrize(
        "window_block_indexes",
        [
            pytest.param([0, 0, 1], id="duplicate-entry"),
            pytest.param([-1, 0, 1], id="negative-entry"),
            pytest.param([0, 1, 12], id="out-of-range-entry"),
        ],
    )
    def test_invalid_window_block_indexes_override_raises(self, window_block_indexes):
        """An override with duplicate, negative, or out-of-range entries raises ValueError.

        `size="small"` has `num_hidden_layers=12` (valid range `[0, 12)`), so `12` in the out-of-range case and `-1` in
        the negative case are both one step outside that range.
        """
        with pytest.raises(ValueError, match="unique and within"):
            DinoV2(
                out_feature_indexes=[12],
                window_block_indexes=window_block_indexes,
                use_windowed_attn=True,
                load_dinov2_weights=False,
                size="small",
            )


class TestSkippedDinov2WeightsWarning:
    """The "not loading DINOv2 backbone weights" warnings fire only when those weights were requested.

    The model builder passes `load_dinov2_weights=False` when RF-DETR pretrained weights overwrite the backbone.
    """

    _SKIPPED_WEIGHTS = "not loading DINOv2 backbone weights"

    @pytest.fixture(autouse=True)
    def _propagate_rfdetr_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Let `caplog` see the `rf-detr` logger, which does not propagate to the root logger.

        Examples:
            Pytest fixtures cannot be called directly, so this runs only through the tests below.

            >>> TestSkippedDinov2WeightsWarning()._propagate_rfdetr_logs(monkeypatch)  # doctest: +SKIP
        """
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

    def _skipped_weights_messages(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        """Return the captured messages that report skipped DINOv2 weights.

        Examples:
            >>> from types import SimpleNamespace
            >>> def record(msg):
            ...     return logging.LogRecord("rf-detr", logging.WARNING, __file__, 0, msg, None, None)
            >>> caplog = SimpleNamespace(records=[record("not loading DINOv2 backbone weights"), record("unrelated")])
            >>> TestSkippedDinov2WeightsWarning()._skipped_weights_messages(caplog)
            ['not loading DINOv2 backbone weights']
        """
        return [record.getMessage() for record in caplog.records if self._SKIPPED_WEIGHTS in record.getMessage()]

    @pytest.mark.parametrize(
        "patch_size, positional_encoding_size",
        [
            (16, 24),
            (14, 24),
        ],
    )
    def test_no_warning_when_dinov2_weights_not_requested(
        self, caplog: pytest.LogCaptureFixture, patch_size: int, positional_encoding_size: int
    ) -> None:
        """Building with `load_dinov2_weights=False`, as for pretrained RF-DETR models, logs no skip warning."""
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            DinoV2(
                out_feature_indexes=[12],
                size="small",
                load_dinov2_weights=False,
                patch_size=patch_size,
                positional_encoding_size=positional_encoding_size,
            )

        assert self._skipped_weights_messages(caplog) == []

    @pytest.mark.parametrize(
        "patch_size, positional_encoding_size, expected_count",
        [
            (16, 24, 2),
            (14, 24, 1),
        ],
    )
    def test_warning_kept_when_dinov2_weights_requested(
        self,
        caplog: pytest.LogCaptureFixture,
        patch_size: int,
        positional_encoding_size: int,
        expected_count: int,
    ) -> None:
        """Requesting DINOv2 weights with an incompatible shape still warns that they are skipped.

        The mismatch disables the hub download before it starts, so this stays offline.
        """
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            DinoV2(
                out_feature_indexes=[12],
                size="small",
                load_dinov2_weights=True,
                patch_size=patch_size,
                positional_encoding_size=positional_encoding_size,
            )

        assert len(self._skipped_weights_messages(caplog)) == expected_count

    def test_no_warning_when_requested_weights_match_dinov2_shape(self, caplog: pytest.LogCaptureFixture) -> None:
        """Requesting DINOv2 weights with the DINOv2 patch size and positional encodings stays silent and loads them.

        Guards against over-gating: with patch size 14 and 37 positional encodings the shapes match, so the hub loader
        must still run and no skipped-weights warning may appear. The hub call is the network boundary, so it is
        replaced.
        """
        with (
            mock.patch.object(
                WindowedDinov2WithRegistersBackbone, "from_pretrained", return_value=nn.Identity()
            ) as hub,
            caplog.at_level(logging.WARNING, logger="rf-detr"),
        ):
            DinoV2(
                out_feature_indexes=[12],
                size="small",
                load_dinov2_weights=True,
                patch_size=14,
                positional_encoding_size=37,
            )

        assert self._skipped_weights_messages(caplog) == []
        hub.assert_called_once()

    @pytest.mark.parametrize(
        "pretrain_weights, expected_count",
        [
            ("rf-detr-nano.pth", 0),
            (None, 2),
        ],
    )
    def test_builder_ties_warnings_to_pretrain_weights(
        self, caplog: pytest.LogCaptureFixture, pretrain_weights: str | None, expected_count: int
    ) -> None:
        """Building a Nano model warns about skipped DINOv2 weights only when no RF-DETR pretrain weights are set.

        The builder passes `load_dinov2_weights=args.pretrain_weights is None`; a regression that always passes True
        would bring the warnings back for every default load while the direct `DinoV2` tests above still pass.
        Construction downloads nothing, so this stays offline.
        """
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            build_model_from_config(RFDETRNanoConfig(pretrain_weights=pretrain_weights))

        assert len(self._skipped_weights_messages(caplog)) == expected_count
