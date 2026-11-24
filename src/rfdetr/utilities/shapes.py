# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Helpers for spatial sizes given either as one side length or as a ``(height, width)`` pair."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

__all__ = ["as_pair"]


def _as_int(side: Any) -> int | None:
    """Return *side* as an ``int`` when it is an integer (``operator.index``) and not a ``bool``, else ``None``.

    Examples:
        >>> _as_int(448), _as_int(448.0), _as_int(True), _as_int("448")
        (448, None, None, None)
    """
    if isinstance(side, bool):
        return None
    try:
        return operator.index(side)
    except TypeError:
        return None


def as_pair(value: int | Iterable[int]) -> tuple[int, int]:
    """Normalize a size to an explicit ``(height, width)`` pair.

    Integers are read with ``operator.index``, so NumPy and PyTorch integer scalars are accepted; a ``bool``, a
    ``float`` (integral or not) and a string are not.

    Args:
        value: A single integer, read as the square ``(value, value)``, or an iterable of exactly two integers ordered
            ``(height, width)``.

    Returns:
        The size as a ``(height, width)`` tuple of ints.

    Raises:
        TypeError: If *value* is neither an integer nor an iterable.
        ValueError: If *value* is a ``bool``, an iterable whose length is not 2 or whose items are not integers, or if
            a side is not positive.

    Examples:
        >>> as_pair(384)
        (384, 384)
        >>> as_pair([320, 448])
        (320, 448)
        >>> as_pair((320, 448, 3))
        Traceback (most recent call last):
            ...
        ValueError: Expected an int or a (height, width) pair of ints, got (320, 448, 3).
        >>> as_pair((320.75, 448))
        Traceback (most recent call last):
            ...
        ValueError: Expected an int or a (height, width) pair of ints, got (320.75, 448).
    """
    message = f"Expected an int or a (height, width) pair of ints, got {value!r}."
    side = _as_int(value)
    if side is not None:
        pair = (side, side)
    else:
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(message)
        try:
            items = tuple(value)
        except TypeError:
            raise TypeError(message) from None
        sides = [_as_int(item) for item in items]
        if len(sides) != 2 or sides[0] is None or sides[1] is None:
            raise ValueError(message)
        pair = (sides[0], sides[1])
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"Sizes must be positive, got {value!r}.")
    return pair
