# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``(height, width)`` size helper."""

from __future__ import annotations

from typing import Any

import pytest

from rfdetr.utilities.shapes import as_pair


class TestAsPair:
    """``as_pair`` reads a side length as a square and keeps the order of a pair."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(384, (384, 384), id="int"),
            pytest.param((320, 448), (320, 448), id="tuple"),
            pytest.param([448, 320], (448, 320), id="list"),
        ],
    )
    def test_returns_height_width_tuple(self, value: Any, expected: tuple[int, int]) -> None:
        """A side length becomes a square; a pair stays in ``(height, width)`` order."""
        assert as_pair(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param((320,), id="one-item"),
            pytest.param((320, 448, 3), id="three-items"),
            pytest.param("64", id="two-character-string"),
        ],
    )
    def test_rejects_anything_but_an_int_or_a_pair(self, value: Any) -> None:
        """An iterable that is not a pair of sizes is refused, a string of two digits included."""
        with pytest.raises(ValueError, match="Expected an int or a"):
            as_pair(value)

    @pytest.mark.parametrize("value", [0, -16, (320, 0), (-320, 448)])
    def test_rejects_non_positive_sizes(self, value: Any) -> None:
        """Every side must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            as_pair(value)
