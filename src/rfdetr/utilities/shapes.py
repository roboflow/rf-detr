# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Helpers for spatial sizes given either as one side length or as a ``(height, width)`` pair."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["as_pair"]


def as_pair(value: int | Iterable[int]) -> tuple[int, int]:
    """Normalize a size to an explicit ``(height, width)`` pair.

    Args:
        value: A single ``int``, read as the square ``(value, value)``, or an iterable of exactly two ints ordered
            ``(height, width)``.

    Returns:
        The size as a ``(height, width)`` tuple of ints.

    Raises:
        TypeError: If *value* is neither an ``int`` nor an iterable.
        ValueError: If *value* is an iterable whose length is not 2, or if a side is not positive.

    Examples:
        >>> as_pair(384)
        (384, 384)
        >>> as_pair([320, 448])
        (320, 448)
        >>> as_pair((320, 448, 3))
        Traceback (most recent call last):
            ...
        ValueError: Expected an int or a (height, width) pair of ints, got (320, 448, 3).
    """
    if isinstance(value, int):
        pair = (value, value)
    else:
        items = () if isinstance(value, (str, bytes)) else tuple(value)
        if len(items) != 2:
            raise ValueError(f"Expected an int or a (height, width) pair of ints, got {value!r}.")
        pair = (int(items[0]), int(items[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"Sizes must be positive, got {value!r}.")
    return pair
