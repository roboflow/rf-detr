# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RealNVP normalizing flow utilities for keypoint residual modeling."""

import numpy as np
import torch
import torch.distributions as distributions
from torch import nn

FLOW_HIDDEN = 64


def _build_scale_net(input_dim: int) -> nn.Sequential:
    """Build the scale network used by affine coupling blocks.

    Args:
        input_dim: Input feature dimension to the network.

    Returns:
        A small two-layer MLP with tanh output bounds.
    """

    return nn.Sequential(
        nn.Linear(input_dim, FLOW_HIDDEN),
        nn.LeakyReLU(),
        nn.Linear(FLOW_HIDDEN, FLOW_HIDDEN),
        nn.LeakyReLU(),
        nn.Linear(FLOW_HIDDEN, 2),
        nn.Tanh(),
    )


def _build_translate_net(input_dim: int) -> nn.Sequential:
    """Build the translation network used by affine coupling blocks.

    Args:
        input_dim: Input feature dimension to the network.

    Returns:
        A small two-layer MLP with linear output.
    """

    return nn.Sequential(
        nn.Linear(input_dim, FLOW_HIDDEN),
        nn.LeakyReLU(),
        nn.Linear(FLOW_HIDDEN, FLOW_HIDDEN),
        nn.LeakyReLU(),
        nn.Linear(FLOW_HIDDEN, 2),
    )


class RealNVP(nn.Module):  # type: ignore[misc]
    """RealNVP for 2D keypoint residual vectors with optional conditioning.

    The module uses six affine coupling layers with alternating masks
    ``[0, 1], [1, 0]`` repeated three times. The prior is a standard 2D Gaussian
    and weights are initialized with small Xavier gain to keep the transform near-identity.
    """

    def __init__(self, hidden_dim: int | None = None) -> None:
        """Create a conditional (optional) RealNVP flow.

        Args:
            hidden_dim: Optional conditioning feature size. If provided, conditioning is injected by projecting
                keypoint hidden states to ``FLOW_HIDDEN`` and concatenating into the scale/translation nets.
        """

        super().__init__()
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3, dtype=np.float32))
        self.register_buffer("masks", masks)

        if hidden_dim is not None:
            self.cond_proj = nn.Linear(hidden_dim, FLOW_HIDDEN)
            coupling_input_dim = 2 + FLOW_HIDDEN
        else:
            self.cond_proj = None
            coupling_input_dim = 2

        self.s = nn.ModuleList([_build_scale_net(coupling_input_dim) for _ in range(len(masks))])
        self.t = nn.ModuleList([_build_translate_net(coupling_input_dim) for _ in range(len(masks))])
        self.prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights near-identity for stable first-iteration behavior."""

        for network in [*self.s, *self.t]:
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        if self.cond_proj is not None:
            nn.init.xavier_uniform_(self.cond_proj.weight, gain=0.01)
            nn.init.zeros_(self.cond_proj.bias)

    def _inverse(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the inverse RealNVP transformation.

        Args:
            x: Residual tensor of shape ``(N, 2)``.
            cond: Optional projected conditioning tensor of shape ``(N, FLOW_HIDDEN)``.

        Returns:
            Tuple ``(z, log_det_jacobian)`` where ``z`` is latent and ``log_det_jacobian``
            is the change-of-variables term.
        """

        log_det_jacobian = x.new_zeros(x.shape[0])
        z = x

        for i in reversed(range(len(self.s))):
            mask = self.masks[i]
            z_masked = mask * z
            scale_translate_input = torch.cat((z_masked, cond), dim=-1) if cond is not None else z_masked
            scale = self.s[i](scale_translate_input) * (1 - mask)
            translation = self.t[i](scale_translate_input) * (1 - mask)
            z = (1 - mask) * (z - translation) * torch.exp(-scale) + z_masked
            log_det_jacobian -= scale.sum(dim=1)

        return z, log_det_jacobian

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Compute ``log p(x | cond)`` under the flow model.

        Args:
            x: Residual tensor of shape ``(N, 2)``.
            cond: Optional conditioning tensor of shape ``(N, hidden_dim)``.

        Returns:
            Log probabilities with shape ``(N,)``.
        """

        projected_cond = self.cond_proj(cond) if (cond is not None and self.cond_proj is not None) else None
        z, log_det_jacobian = self._inverse(x, cond=projected_cond)

        if self.prior.loc.device != z.device:
            self.prior = distributions.MultivariateNormal(
                self.prior.loc.to(z.device), self.prior.covariance_matrix.to(z.device)
            )

        return self.prior.log_prob(z) + log_det_jacobian

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Alias for :meth:`log_prob`.
        """

        return self.log_prob(x, cond=cond)


__all__ = ["RealNVP"]
