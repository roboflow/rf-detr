# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Reproducibility helpers: seed all RNGs consistently."""

from __future__ import annotations

import random

import numpy as np
import torch

from rfdetr.utilities.logger import get_logger


def seed_all(seed: int = 7) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for Python's ``random`` module, NumPy, and PyTorch (CPU and all CUDA devices).  Also configures cuDNN to
    use deterministic algorithms and disables its auto-tuner, then escalates to
    :func:`torch.use_deterministic_algorithms` with ``warn_only=True`` so every op that offers a deterministic
    implementation uses it, and the ones that do not (e.g. some scatter / ``grid_sample`` CUDA kernels) emit a warning
    at execution time instead of raising.  Results are reproducible across runs at the cost of a possible slight
    performance decrease.

    Under distributed data-parallel, ``random`` and NumPy are offset by the process rank so stochastic augmentations
    differ across ranks, while PyTorch is seeded identically so model initialization stays in sync.

    Args:
        seed: Integer seed value.  Defaults to ``7``.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # ``warn_only=True`` keeps determinism best-effort for ops without a deterministic kernel;
    # the call itself is guarded so a failure to enable deterministic algorithms degrades
    # to a warning rather than propagating out of ``seed_all``.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError as error:
        get_logger().warning("Could not enable deterministic algorithms: %s", error)
