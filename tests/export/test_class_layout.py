# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for explicit exported class-layout handling."""

from __future__ import annotations

import numpy as np
import pytest

from rfdetr.export._class_layout import _exclude_background_class


def test_no_background_class_preserves_every_exported_slot() -> None:
    """``None`` keeps the score grid and every original class ID."""
    scores_all = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    scores, class_ids = _exclude_background_class(scores_all, None)

    np.testing.assert_array_equal(scores, scores_all)
    assert class_ids.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("background_class_id", "expected_scores", "expected_class_ids"),
    [
        pytest.param(0, [[0.2, 0.3]], [1, 2], id="background-first"),
        pytest.param(-1, [[0.1, 0.2]], [0, 1], id="background-last"),
    ],
)
def test_background_exclusion_preserves_original_class_ids(
    background_class_id: int,
    expected_scores: list[list[float]],
    expected_class_ids: list[int],
) -> None:
    """Removing a slot does not renumber the remaining exported classes."""
    scores_all = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    scores, class_ids = _exclude_background_class(scores_all, background_class_id)

    np.testing.assert_array_equal(scores, np.array(expected_scores, dtype=np.float32))
    assert class_ids.tolist() == expected_class_ids


@pytest.mark.parametrize("background_class_id", [2, -3], ids=["positive", "negative"])
def test_out_of_range_background_class_is_rejected(background_class_id: int) -> None:
    """Background IDs outside the exported class axis fail with a clear error."""
    with pytest.raises(ValueError, match="background_class_id"):
        _exclude_background_class(np.ones((1, 2), dtype=np.float32), background_class_id)


def test_non_matrix_scores_are_rejected() -> None:
    """The layout helper rejects a score tensor without query and class axes."""
    with pytest.raises(ValueError, match=r"shape \(Q, C\)"):
        _exclude_background_class(np.ones(2, dtype=np.float32), None)


@pytest.mark.parametrize(
    "background_class_id",
    [pytest.param(0.0, id="non-integral-float"), pytest.param(True, id="bool")],
)
def test_non_integer_background_class_is_rejected(background_class_id: object) -> None:
    """A float or bool background_class_id fails fast instead of silently mis-selecting a slot.

    ``bool`` is an ``int`` subclass in Python, so it would otherwise act as index 0/1; a float would compare unequal to
    every integer class ID and silently keep every slot instead of excluding one.
    """
    with pytest.raises(TypeError, match="background_class_id"):
        _exclude_background_class(np.ones((1, 2), dtype=np.float32), background_class_id)
