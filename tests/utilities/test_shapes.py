# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``(height, width)`` size helper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from rfdetr.utilities.shapes import as_pair


class TestAsPair:
    """``as_pair`` reads a side length as a square and keeps the order of a pair."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(384, (384, 384), id="int"),
            pytest.param((320, 448), (320, 448), id="tuple"),
            pytest.param([448, 320], (448, 320), id="list"),
            pytest.param(np.int64(384), (384, 384), id="numpy-int"),
            pytest.param((np.int32(320), torch.tensor(448)), (320, 448), id="numpy-and-torch-ints"),
        ],
    )
    def test_returns_height_width_tuple(self, value: Any, expected: tuple[int, int]) -> None:
        """A side length becomes a square; a pair stays in ``(height, width)`` order, as plain ints."""
        pair = as_pair(value)

        assert pair == expected
        assert [type(side) for side in pair] == [int, int]

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param((320.75, 448), id="float-item"),
            pytest.param((320.0, 448), id="integral-float-item"),
            pytest.param(("320", "448"), id="string-items"),
            pytest.param((True, 448), id="bool-item"),
            pytest.param((320, False), id="false-item"),
            pytest.param((320, None), id="none-item"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_rejects_sides_that_are_not_integers(self, value: Any) -> None:
        """A float is not truncated and a bool is not read as 0 or 1: only integers are sizes."""
        with pytest.raises(ValueError, match="Expected an int or a"):
            as_pair(value)

    @pytest.mark.parametrize("value", [448.0, 320.75, None])
    def test_rejects_a_scalar_that_is_not_an_integer(self, value: Any) -> None:
        """A scalar that is not an integer is refused, an integral float included."""
        with pytest.raises(ValueError, match="Expected an int or a"):
            as_pair(value)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param((320,), id="one-item"),
            pytest.param((320, 448, 3), id="three-items"),
            pytest.param("64", id="two-character-string"),
            pytest.param(b" @", id="two-bytes"),
            pytest.param({32: 0, 64: 0}, id="dict-with-two-integer-keys"),
            pytest.param({32, 64}, id="set-with-two-integers"),
            pytest.param(np.array([320, 448]), id="numpy-array"),
        ],
    )
    def test_rejects_anything_but_an_int_or_a_pair(self, value: Any) -> None:
        """Only a tuple or list can supply two side lengths."""
        with pytest.raises(ValueError, match="Expected an int or a"):
            as_pair(value)

    @pytest.mark.parametrize("value", [0, -16, (320, 0), (-320, 448)])
    def test_rejects_non_positive_sizes(self, value: Any) -> None:
        """Every side must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            as_pair(value)
