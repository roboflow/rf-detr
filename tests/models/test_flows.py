# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.models.flows import RealNVP


def test_realnvp_log_prob_shape() -> None:
    """RealNVP should return a per-sample log-probability vector for unconditioned inputs."""
    torch.manual_seed(0)
    flow = RealNVP()
    x = torch.randn(10, 2)

    log_prob = flow.log_prob(x)

    assert log_prob.shape == (10,)
    assert torch.isfinite(log_prob).all()


def test_realnvp_conditional_shape() -> None:
    """RealNVP should accept conditional vectors and preserve sample shape in outputs."""
    torch.manual_seed(0)
    flow = RealNVP(hidden_dim=8)
    x = torch.randn(7, 2)
    cond = torch.randn(7, 8)

    log_prob = flow.log_prob(x, cond=cond)

    assert log_prob.shape == (7,)
    assert torch.isfinite(log_prob).all()
