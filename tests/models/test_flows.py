# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
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


def test_realnvp_log_prob_masks_invalid_inverse_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid latent rows should not make the whole RLE batch raise or return non-finite values."""
    flow = RealNVP()
    x = torch.zeros(2, 2)

    def invalid_inverse(inputs: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = inputs.clone()
        z[0, 0] = float("nan")
        return z, inputs.new_zeros(inputs.shape[0])

    monkeypatch.setattr(flow, "_inverse", invalid_inverse)

    log_prob = flow.log_prob(x)

    assert log_prob.shape == (2,)
    assert torch.isfinite(log_prob).all()
    assert log_prob[0] == 0.0
