# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Drop-path / dropout schedule utilities."""

from typing import Literal

import numpy as np


def drop_scheduler(
    drop_rate: float,
    epochs: int,
    niter_per_ep: int,
    cutoff_epoch: int = 0,
    mode: Literal["standard", "early", "late"] = "standard",
    schedule: Literal["constant", "linear"] = "constant",
) -> np.ndarray:
    """Build a per-iteration drop-path or dropout rate schedule.

    ``"standard"`` mode: every iteration uses the same ``drop_rate``; ignored ``cutoff_epoch``, ``schedule``.
    ``"early"`` mode: drop rate applies during the first ``cutoff_epoch`` epochs; remaining epochs use zero drop rate.
        optional: schedule ``linear`` decay to zero over the first ``cutoff_epoch`` epochs.
    ``"late"`` mode: the first ``cutoff_epoch`` epochs use zero drop rate; remaining epochs use ``drop_rate``.

    Args:
        drop_rate: Target drop probability.
        epochs: Total number of training epochs.
        niter_per_ep: Number of optimizer steps per epoch.
        cutoff_epoch: Number of epochs in the initial schedule phase. Phases split at cutoff_epoch * niter_per_ep steps
            Ignored when ``mode`` is ``"standard"``.
        mode: Scheduling strategy: ``"standard"``, ``"early"``, or ``"late"``.
        schedule: Shape of the initial schedule phase in ``"early"`` mode: ``"constant"`` or ``"linear"``.
            Ignored when ``mode`` is ``"standard"`` or ``"late"``.

    Returns:
        One-dimensional array of length ``epochs * niter_per_ep`` containing the drop rate per iteration.
    """
    assert mode in ["standard", "early", "late"]
    if mode == "standard":
        return np.full(epochs * niter_per_ep, drop_rate)

    early_iters = cutoff_epoch * niter_per_ep
    late_iters = (epochs - cutoff_epoch) * niter_per_ep

    if mode == "early":
        assert schedule in ["constant", "linear"]
        if schedule == "constant":
            early_schedule = np.full(early_iters, drop_rate)
        elif schedule == "linear":
            early_schedule = np.linspace(drop_rate, 0, early_iters)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, 0)))
    elif mode == "late":
        assert schedule in ["constant"]
        early_schedule = np.full(early_iters, 0)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, drop_rate)))

    assert len(final_schedule) == epochs * niter_per_ep
    return final_schedule
