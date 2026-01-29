# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""util for drop scheduler."""
from typing import Literal

import numpy as np


def drop_scheduler(
    drop_rate: float,
    epochs: int,
    niter_per_ep: int,
    cutoff_epoch: int = 0,
    mode: Literal['standard', 'early', 'late'] = 'standard',
    schedule: Literal['constant', 'linear'] = 'constant',
) -> np.ndarray:
    """drop scheduler"""
    assert mode in ['standard', 'early', 'late']
    if mode == 'standard':
        return np.full(epochs * niter_per_ep, drop_rate)

    early_iters = cutoff_epoch * niter_per_ep
    late_iters = (epochs - cutoff_epoch) * niter_per_ep

    if mode == 'early':
        assert schedule in ['constant', 'linear']
        if schedule == 'constant':
            early_schedule = np.full(early_iters, drop_rate)
        elif schedule == 'linear':
            early_schedule = np.linspace(drop_rate, 0, early_iters)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, 0)))
    elif mode == 'late':
        assert schedule in ['constant']
        early_schedule = np.full(early_iters, 0)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, drop_rate)))

    assert len(final_schedule) == epochs * niter_per_ep
    return final_schedule
