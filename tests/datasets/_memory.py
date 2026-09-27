# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Allocation measurement shared by the dataset decode tests."""

import tracemalloc
from collections.abc import Callable
from typing import Any


def peak_traced_bytes(function: Callable[..., Any], *args: Any) -> int:
    """Return the peak Python-traced allocation of one ``function(*args)`` call, after a warm-up call.

    ``tracemalloc`` sees NumPy's array buffers but not Pillow's own image memory, so a decode that copies a frame
    through NumPy shows up here at the frame's size, while a Pillow-only decode stays at its read buffers. The warm-up
    call keeps one-time work, such as Pillow registering its format plugins on first open, out of the measurement.

    Examples:
        >>> import numpy as np
        >>> peak_traced_bytes(np.ones, 100_000) >= 800_000
        True
    """
    function(*args)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        function(*args)
        return tracemalloc.get_traced_memory()[1] - baseline
    finally:
        if not was_tracing:
            tracemalloc.stop()
