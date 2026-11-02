# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fixtures shared by the inference tests."""

from collections.abc import Generator

import pytest
import torch


@pytest.fixture
def nondeterministic_algorithms() -> Generator[None, None, None]:
    """Turn deterministic algorithms off for one test, restoring the previous mode afterwards.

    The autouse ``reset_random_seeds`` fixture enables them, which routes CUDA ``F.interpolate`` through a
    decomposition that fails under ``torch.jit.trace`` (its sizes become CPU tensors). A segmentation head traced on
    CUDA hits it.

    Examples:
        >>> # Applied with ``@pytest.mark.usefixtures("nondeterministic_algorithms")``; a fixture cannot run alone.
        >>> nondeterministic_algorithms()  # doctest: +SKIP
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
