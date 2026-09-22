# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the feature-pyramid projector building blocks and ``MultiScaleProjector``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from rfdetr.models.backbone.projector import (
    Bottleneck,
    C2f,
    ConvX,
    LayerNorm,
    MultiScaleProjector,
    get_activation,
    get_norm,
)


def _levels(channels: int, count: int, height: int = 4, width: int = 4) -> list[Tensor]:
    """Build ``count`` distinct constant feature maps, one per pyramid input level.

    Each level is filled with its own scalar (1.0, 2.0, ...) so that concatenation and
    zeroing effects on individual levels remain distinguishable downstream.

    Examples:
        >>> levels = _levels(channels=2, count=2, height=1, width=1)
        >>> [level.flatten()[0].item() for level in levels]
        [1.0, 2.0]
    """
    return [torch.full((1, channels, height, width), float(index + 1)) for index in range(count)]


class TestGetNorm:
    """``get_norm`` resolves a norm spec (``None``, empty string, or a name) to a module or ``None``."""

    def test_none_returns_none(self) -> None:
        """A ``None`` spec means no normalization layer at all."""
        assert get_norm(None, 8) is None

    def test_empty_string_returns_none(self) -> None:
        """An empty string spec is treated the same as ``None``."""
        assert get_norm("", 8) is None

    def test_ln_returns_layernorm_instance(self) -> None:
        """The ``"LN"`` name resolves to this module's channels-first ``LayerNorm``."""
        norm = get_norm("LN", 8)
        assert isinstance(norm, LayerNorm)
        assert norm.normalized_shape == (8,)


class TestGetActivation:
    """``get_activation`` resolves an activation name to a module, or raises for an unknown one."""

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("relu", nn.ReLU),
            ("silu", nn.SiLU),
            ("leakyrelu", nn.LeakyReLU),
            (None, nn.Identity),
        ],
    )
    def test_known_name_returns_expected_module_type(self, name: str | None, expected_type: type[nn.Module]) -> None:
        """Every registered name (and ``None``) resolves to its documented module type."""
        assert isinstance(get_activation(name), expected_type)

    def test_unsupported_name_raises_attribute_error(self) -> None:
        """An unregistered activation name raises rather than silently falling back."""
        with pytest.raises(AttributeError, match="Unsupported act type: bogus"):
            get_activation("bogus")


class TestConvX:
    """``ConvX`` picks its normalization layer from the ``layer_norm``/``rms_norm`` flags."""

    def test_default_uses_batchnorm(self) -> None:
        """With both flags left at their defaults, the norm layer is ``BatchNorm2d``."""
        conv = ConvX(4, 8)
        assert isinstance(conv.bn, nn.BatchNorm2d)

    def test_layer_norm_flag_uses_custom_layernorm(self) -> None:
        """``layer_norm=True`` selects this module's channels-first ``LayerNorm``."""
        conv = ConvX(4, 8, layer_norm=True)
        assert isinstance(conv.bn, LayerNorm)

    def test_rms_norm_flag_uses_rmsnorm(self) -> None:
        """``rms_norm=True`` takes priority and selects ``nn.RMSNorm``."""
        conv = ConvX(4, 8, rms_norm=True)
        assert isinstance(conv.bn, nn.RMSNorm)

    def test_forward_produces_expected_output_channels_and_spatial_size(self) -> None:
        """A stride-2 convolution halves spatial size and remaps to ``out_planes`` channels."""
        conv = ConvX(4, 8, kernel=3, stride=2)
        out = conv(torch.rand(2, 4, 8, 8))
        assert out.shape == (2, 8, 4, 4)


class TestBottleneck:
    """``Bottleneck`` adds its residual only when input and output channel counts match."""

    def test_adds_residual_when_channels_match(self) -> None:
        """Equal in/out channels with ``shortcut=True`` enables the residual add."""
        block = Bottleneck(8, 8, shortcut=True)
        assert block.add is True
        out = block(torch.rand(1, 8, 4, 4))
        assert out.shape == (1, 8, 4, 4)

    def test_skips_residual_when_channels_differ(self) -> None:
        """Differing in/out channels disable the residual add even with ``shortcut=True``."""
        block = Bottleneck(8, 16, shortcut=True)
        assert block.add is False
        out = block(torch.rand(1, 8, 4, 4))
        assert out.shape == (1, 16, 4, 4)


class TestC2f:
    """``C2f`` fuses its split branches and bottleneck outputs back into ``c2`` channels."""

    def test_forward_produces_expected_output_channels(self) -> None:
        """The module's output channel count always matches its configured ``c2``, regardless of depth."""
        block = C2f(12, 8, n=2)
        out = block(torch.rand(1, 12, 4, 4))
        assert out.shape == (1, 8, 4, 4)


class TestMultiScaleProjectorInit:
    """Each supported ``scale_factors`` entry builds a distinct per-level sampling path."""

    def test_scale_4_upsamples_spatial_size_by_four_and_quarters_channels(self) -> None:
        """The 4x branch chains two transpose convolutions, matching the C2f fan-in it feeds."""
        proj = MultiScaleProjector(in_channels=[64], out_channels=8, scale_factors=[4.0], num_blocks=1)
        out = proj([torch.rand(2, 64, 4, 4)])
        assert out[0].shape == (2, 8, 16, 16)

    def test_scale_2_upsamples_spatial_size_by_two_and_halves_channels(self) -> None:
        """The 2x branch chains a single transpose convolution."""
        proj = MultiScaleProjector(in_channels=[64], out_channels=8, scale_factors=[2.0], num_blocks=1)
        out = proj([torch.rand(2, 64, 4, 4)])
        assert out[0].shape == (2, 8, 8, 8)

    def test_scale_0_5_downsamples_spatial_size_by_two_and_keeps_channels(self) -> None:
        """The 0.5x branch is a strided ``ConvX`` that halves spatial size without changing channels."""
        proj = MultiScaleProjector(in_channels=[64], out_channels=8, scale_factors=[0.5], num_blocks=1)
        out = proj([torch.rand(2, 64, 4, 4)])
        assert out[0].shape == (2, 8, 2, 2)

    def test_unsupported_scale_factor_raises_not_implemented_error(self) -> None:
        """A scale factor outside the documented set is rejected at construction time."""
        with pytest.raises(NotImplementedError, match="Unsupported scale_factor:3.0"):
            MultiScaleProjector(in_channels=[64], out_channels=8, scale_factors=[3.0])


class TestMultiScaleProjectorExtraPool:
    """``forward`` appends an extra max-pooled feature map when ``use_extra_pool`` is set."""

    def test_forward_appends_a_max_pooled_copy_of_the_last_stage_output(self) -> None:
        """Verifies the extra-pooling step independently of marker construction.

        The flag is set directly on an otherwise ordinary projector so this test isolates ``forward()``;
        ``TestMultiScaleProjectorExtraPoolMarker`` covers construction via ``scale_factors=[..., 0.25]``.
        """
        proj = MultiScaleProjector(in_channels=[4], out_channels=4, scale_factors=[1.0])
        proj.use_extra_pool = True

        out = proj([torch.rand(1, 4, 8, 8)])

        assert len(out) == 2
        assert out[1].shape == (1, 4, 4, 4)
        assert torch.equal(out[1], F.max_pool2d(out[0], kernel_size=1, stride=2, padding=0))


class TestMultiScaleProjectorSurvivalProb:
    """``forward`` stochastically zeroes non-first feature levels while training, per ``survival_prob``."""

    def test_drops_only_the_level_whose_threshold_the_draw_exceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With 3 levels and ``survival_prob=0.4``, a fixed draw of 0.5 clears only the last level.

        Thresholds are ``critical_drop_prob = i * (0.6 / 2)`` for ``i in (1, 2)`` -> ``0.3`` and ``0.6``.
        A draw of ``0.5`` is below the second threshold only, so level 0 (never eligible) and level 1
        stay intact while level 2 is zeroed. The stage's C2f+LN fusion is replaced with ``nn.Identity``
        so the per-level channel blocks stay distinguishable in the concatenated output.
        """
        proj = MultiScaleProjector(in_channels=[4, 4, 4], out_channels=4, scale_factors=[1.0], survival_prob=0.4)
        proj.train()
        proj.stages[0] = nn.Identity()
        levels = _levels(channels=4, count=3, height=2, width=2)
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.5))

        out = proj(levels)[0]

        assert torch.equal(out[:, 0:4], levels[0])
        assert torch.equal(out[:, 4:8], levels[1])
        assert torch.equal(out[:, 8:12], torch.zeros_like(levels[2]))

    def test_inactive_outside_training_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eval mode skips the dropping branch entirely, even with a draw that would otherwise trigger it."""
        proj = MultiScaleProjector(in_channels=[4, 4], out_channels=4, scale_factors=[1.0], survival_prob=0.1)
        proj.eval()
        proj.stages[0] = nn.Identity()
        levels = _levels(channels=4, count=2, height=2, width=2)
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))

        out = proj(levels)[0]

        assert torch.equal(out[:, 0:4], levels[0])
        assert torch.equal(out[:, 4:8], levels[1])

    def test_leaves_the_callers_input_list_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``forward`` copies its input list before zeroing entries, per the in-code invariant comment."""
        proj = MultiScaleProjector(in_channels=[4, 4], out_channels=4, scale_factors=[1.0], survival_prob=0.1)
        proj.train()
        proj.stages[0] = nn.Identity()
        levels = _levels(channels=4, count=2, height=2, width=2)
        original_ids = [id(level) for level in levels]
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))

        proj(levels)

        assert [id(level) for level in levels] == original_ids
        assert torch.equal(levels[1], torch.full((1, 4, 2, 2), 2.0))


class TestMultiScaleProjectorForceDropLastNFeatures:
    """``forward`` deterministically zeroes the trailing ``force_drop_last_n_features`` levels."""

    def test_zeroes_exactly_the_trailing_n_levels(self) -> None:
        """With ``force_drop_last_n_features=2`` over 3 levels, only the first level survives untouched."""
        proj = MultiScaleProjector(
            in_channels=[4, 4, 4], out_channels=4, scale_factors=[1.0], force_drop_last_n_features=2
        )
        proj.stages[0] = nn.Identity()
        levels = _levels(channels=4, count=3, height=2, width=2)

        out = proj(levels)[0]

        assert torch.equal(out[:, 0:4], levels[0])
        assert torch.equal(out[:, 4:8], torch.zeros_like(levels[1]))
        assert torch.equal(out[:, 8:12], torch.zeros_like(levels[2]))


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
        features = [torch.arange(c * 8 * 8, dtype=torch.float32).reshape(1, c, 8, 8) for c in in_channels]

        results = projector(features)

        assert len(results) == 2
        expected_pooled_feature = torch.nn.functional.max_pool2d(results[0], kernel_size=1, stride=2)
        assert results[-1].shape == (1, 4, 4, 4)
        torch.testing.assert_close(results[-1], expected_pooled_feature)
