# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Helpers for spatial sizes given either as one side length or as a ``(height, width)`` pair."""

from __future__ import annotations

import operator
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


def as_pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize a size to an explicit ``(height, width)`` pair.

    Integers are read with ``operator.index``, so NumPy and PyTorch integer scalars are accepted; a ``bool``, a
    ``float`` (integral or not) and a string are not.

    Args:
        value: A positive side length for a square, or two positive side lengths in ``(height, width)`` order
            supplied as a tuple or list.

    Returns:
        The size as a ``(height, width)`` tuple of ints.

    Raises:
        ValueError: If *value* has a different form or a side is not a positive integer. Booleans are invalid.

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
        >>> as_pair(384.0)
        Traceback (most recent call last):
            ...
        ValueError: Expected an int or a (height, width) pair of ints, got 384.0.
    """
    pair = (value, value) if _as_int(value) is not None else value
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(f"Expected an int or a (height, width) pair of ints, got {value!r}.")
    height, width = _as_int(pair[0]), _as_int(pair[1])
    if height is None or width is None:
        raise ValueError(f"Expected an int or a (height, width) pair of ints, got {value!r}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"Sizes must be positive, got {value!r}.")
    return height, width
