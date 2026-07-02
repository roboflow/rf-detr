# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.position_encoding.build_position_encoding.

Covers:
- Supported aliases ``"sine"`` / ``"v2"`` return a ``PositionEmbeddingSine`` instance.
- Unsupported but previously accepted aliases ``"learned"`` / ``"v3"`` now raise
  ``ValueError`` with a message that names supported alternatives.
- Fully unsupported values raise ``ValueError`` with the same pattern.
"""

import pytest

from rfdetr.models.position_encoding import (
    PositionEmbeddingSine,
    build_position_encoding,
)


class TestBuildPositionEncodingSupportedValues:
    """build_position_encoding returns valid modules for supported aliases."""

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param("sine", id="sine"),
            pytest.param("v2", id="v2"),
        ],
    )
    def test_returns_sine_embedding(self, alias: str) -> None:
        """Supported aliases produce a PositionEmbeddingSine with normalized=True."""
        enc = build_position_encoding(hidden_dim=256, position_embedding=alias)
        assert isinstance(enc, PositionEmbeddingSine)
        assert enc.normalize is True

    @pytest.mark.parametrize(
        "hidden_dim, expected_num_pos_feats",
        [
            pytest.param(256, 128, id="dim256"),
            pytest.param(512, 256, id="dim512"),
        ],
    )
    def test_num_pos_feats_is_half_hidden_dim(self, hidden_dim: int, expected_num_pos_feats: int) -> None:
        """The sine encoding uses hidden_dim // 2 positional feature dimensions."""
        enc = build_position_encoding(hidden_dim=hidden_dim, position_embedding="sine")
        assert enc.num_pos_feats == expected_num_pos_feats


class TestBuildPositionEncodingUnsupportedValues:
    """build_position_encoding raises ValueError for broken or unknown aliases."""

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param("learned", id="learned"),
            pytest.param("v3", id="v3"),
        ],
    )
    def test_learned_raises_value_error(self, alias: str) -> None:
        """'learned' and 'v3' are doubly broken and must raise ValueError immediately.

        The PositionEmbeddingLearned class has two bugs:
        1. forward() signature is incompatible with Joiner.forward() (no align_dim_orders param).
        2. h, w = x.shape[:2] unpacks batch and channels instead of height and width.
        Rejecting them at build time is preferable to a silent or confusing runtime failure.
        """
        with pytest.raises(ValueError, match="not supported"):
            build_position_encoding(hidden_dim=256, position_embedding=alias)

    def test_unknown_value_raises_value_error(self) -> None:
        """A fully unknown alias raises ValueError naming the supported alternatives."""
        with pytest.raises(ValueError, match="not supported"):
            build_position_encoding(hidden_dim=256, position_embedding="unknown_variant")

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param("learned", id="learned"),
            pytest.param("v3", id="v3"),
        ],
    )
    def test_error_message_mentions_supported_alternatives(self, alias: str) -> None:
        """Error message for 'learned'/'v3' mentions at least one supported alternative."""
        with pytest.raises(ValueError, match="sine"):
            build_position_encoding(hidden_dim=256, position_embedding=alias)
