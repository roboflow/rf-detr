# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""OpenVINO input-shape test double shared by the export loader tests."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace


class FakePartialShape(list):
    """Stand in for an OpenVINO ``PartialShape`` in loader tests that do not import ``openvino``.

    Indexing and ``len()`` behave like the real binding for a static rank. Each item is a dimension exposing
    ``is_static`` and ``get_length()``; a ``None`` size gives a dynamic dimension. ``rank`` reports a static
    dimension of the shape's length, or a dynamic one when *static_rank* is false.

    Examples:
        >>> shape = FakePartialShape([1, 3, None, 8])
        >>> shape.rank.is_static, len(shape), shape[3].get_length(), shape[2].is_static
        (True, 4, 8, False)
        >>> FakePartialShape([], static_rank=False).rank.is_static
        False
    """

    def __init__(self, sizes: Sequence[int | None], *, static_rank: bool = True) -> None:
        super().__init__(
            SimpleNamespace(is_static=size is not None, get_length=lambda length=size: length) for size in sizes
        )
        self.rank = SimpleNamespace(is_static=static_rank, get_length=lambda: len(self))
