# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Utilities for oriented (rotated) bounding box manipulation, IoU, and losses."""

import math

import torch


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [0, pi) with pi-periodicity.

    Args:
        angle: Angles in radians, arbitrary range.

    Returns:
        Angles clamped to [0, pi).
    """
    return angle - torch.floor(angle / math.pi) * math.pi


def box_cxcywha_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert oriented boxes from center format to four corner points.

    Args:
        boxes: Tensor of shape ``(..., 5)`` as ``[cx, cy, w, h, angle]``
            where angle is in radians.

    Returns:
        Tensor of shape ``(..., 4, 2)`` with four corner coordinates.
    """
    cx, cy, w, h, angle = boxes.unbind(-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    hw = w / 2
    hh = h / 2

    dx_w = hw * cos_a
    dy_w = hw * sin_a
    dx_h = hh * -sin_a
    dy_h = hh * cos_a

    c1 = torch.stack([cx - dx_w - dx_h, cy - dy_w - dy_h], dim=-1)
    c2 = torch.stack([cx + dx_w - dx_h, cy + dy_w - dy_h], dim=-1)
    c3 = torch.stack([cx + dx_w + dx_h, cy + dy_w + dy_h], dim=-1)
    c4 = torch.stack([cx - dx_w + dx_h, cy - dy_w + dy_h], dim=-1)

    return torch.stack([c1, c2, c3, c4], dim=-2)


def corners_to_cxcywha(corners: torch.Tensor) -> torch.Tensor:
    """Convert four corner points to oriented box center format.

    Uses the first edge (corner0 -> corner1) as the width direction to
    derive the rotation angle.

    Args:
        corners: Tensor of shape ``(..., 4, 2)`` with four corner points
            ordered sequentially around the box.

    Returns:
        Tensor of shape ``(..., 5)`` as ``[cx, cy, w, h, angle]``.
    """
    cx = corners[..., :, 0].mean(dim=-1)
    cy = corners[..., :, 1].mean(dim=-1)

    edge_w = corners[..., 1, :] - corners[..., 0, :]
    edge_h = corners[..., 3, :] - corners[..., 0, :]

    w = torch.linalg.norm(edge_w, dim=-1)
    h = torch.linalg.norm(edge_h, dim=-1)

    angle = torch.atan2(edge_w[..., 1], edge_w[..., 0])
    angle = normalize_angle(angle)

    return torch.stack([cx, cy, w, h, angle], dim=-1)


def _obb_to_gaussian(
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert oriented boxes to 2D Gaussian distributions.

    Each box ``[cx, cy, w, h, angle]`` maps to a Gaussian with mean
    ``(cx, cy)`` and covariance ``R @ diag(w^2/4, h^2/4) @ R^T``.

    Args:
        boxes: Tensor of shape ``(..., 5)``.

    Returns:
        Tuple of ``(mu, sigma)`` where ``mu`` has shape ``(..., 2)`` and
        ``sigma`` has shape ``(..., 2, 2)``.
    """
    cx, cy, w, h, angle = boxes.unbind(-1)
    mu = torch.stack([cx, cy], dim=-1)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    var_w = (w * w) / 4
    var_h = (h * h) / 4

    a = var_w * cos_a * cos_a + var_h * sin_a * sin_a
    b = (var_w - var_h) * cos_a * sin_a
    d = var_w * sin_a * sin_a + var_h * cos_a * cos_a

    sigma = torch.stack([a, b, b, d], dim=-1).reshape(*boxes.shape[:-1], 2, 2)

    return mu, sigma


def gwd_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Gaussian Wasserstein Distance between paired oriented boxes.

    Uses the closed-form 2nd Wasserstein distance between two 2D Gaussians
    derived from the box parameters.

    Args:
        pred: Predicted boxes ``(..., 5)`` as ``[cx, cy, w, h, angle]``.
        target: Target boxes ``(..., 5)``, same shape as pred.
        tau: Temperature parameter for loss normalization.

    Returns:
        Per-box GWD loss, same leading shape as inputs.
    """
    mu_p, sigma_p = _obb_to_gaussian(pred)
    mu_t, sigma_t = _obb_to_gaussian(target)

    diff_mu = mu_p - mu_t
    term_center = (diff_mu * diff_mu).sum(dim=-1)

    trace_p = sigma_p[..., 0, 0] + sigma_p[..., 1, 1]
    trace_t = sigma_t[..., 0, 0] + sigma_t[..., 1, 1]

    product = torch.bmm(sigma_p.reshape(-1, 2, 2), sigma_t.reshape(-1, 2, 2)).reshape(*sigma_p.shape)
    trace_product = product[..., 0, 0] + product[..., 1, 1]
    det_p = sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] * sigma_p[..., 1, 0]
    det_t = sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] * sigma_t[..., 1, 0]
    det_sqrt = (det_p.clamp(min=1e-8) * det_t.clamp(min=1e-8)).sqrt()
    trace_sqrt = (trace_product + 2 * det_sqrt).clamp(min=1e-8).sqrt()

    w2 = (term_center + trace_p + trace_t - 2 * trace_sqrt).clamp(min=0)

    return 1 - 1 / (tau + torch.log1p(w2))


def kld_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL Divergence loss between oriented boxes modeled as 2D Gaussians.

    Scale-invariant and aspect-ratio adaptive: elongated objects receive
    stronger angular gradients.

    Args:
        pred: Predicted boxes ``(..., 5)`` as ``[cx, cy, w, h, angle]``.
        target: Target boxes ``(..., 5)``, same shape as pred.

    Returns:
        Per-box KLD loss, same leading shape as inputs.
    """
    mu_p, sigma_p = _obb_to_gaussian(pred)
    mu_t, sigma_t = _obb_to_gaussian(target)

    det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2).clamp(min=1e-8)
    det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2).clamp(min=1e-8)

    inv_t00 = sigma_t[..., 1, 1] / det_t
    inv_t01 = -sigma_t[..., 0, 1] / det_t
    inv_t11 = sigma_t[..., 0, 0] / det_t

    trace_term = inv_t00 * sigma_p[..., 0, 0] + 2 * inv_t01 * sigma_p[..., 0, 1] + inv_t11 * sigma_p[..., 1, 1]

    diff = mu_p - mu_t
    mahal_term = inv_t00 * diff[..., 0] ** 2 + 2 * inv_t01 * diff[..., 0] * diff[..., 1] + inv_t11 * diff[..., 1] ** 2

    log_det_term = torch.log(det_t) - torch.log(det_p)

    kld = 0.5 * (trace_term + mahal_term + log_det_term - 2)

    return torch.log1p(kld)


def probiou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Probabilistic IoU via Bhattacharyya coefficient between Gaussian-encoded boxes.

    Returns a similarity score in ``[0, 1]`` where 1 means identical boxes.

    Args:
        pred: Predicted boxes ``(..., 5)`` as ``[cx, cy, w, h, angle]``.
        target: Target boxes ``(..., 5)``, same shape as pred.

    Returns:
        Per-box ProbIoU similarity, same leading shape as inputs.
    """
    mu_p, sigma_p = _obb_to_gaussian(pred)
    mu_t, sigma_t = _obb_to_gaussian(target)

    sigma_avg = (sigma_p + sigma_t) / 2
    det_avg = (sigma_avg[..., 0, 0] * sigma_avg[..., 1, 1] - sigma_avg[..., 0, 1] ** 2).clamp(min=1e-8)
    det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2).clamp(min=1e-8)
    det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2).clamp(min=1e-8)

    inv_avg00 = sigma_avg[..., 1, 1] / det_avg
    inv_avg01 = -sigma_avg[..., 0, 1] / det_avg
    inv_avg11 = sigma_avg[..., 0, 0] / det_avg

    diff = mu_p - mu_t
    mahal = inv_avg00 * diff[..., 0] ** 2 + 2 * inv_avg01 * diff[..., 0] * diff[..., 1] + inv_avg11 * diff[..., 1] ** 2

    log_coeff = 0.5 * (torch.log(det_avg) - 0.5 * (torch.log(det_p) + torch.log(det_t)))
    bd = 0.125 * mahal + log_coeff

    hd_squared = (1 - torch.exp(-bd.clamp(max=50))).clamp(min=0)
    return 1 - hd_squared


def gwd_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Pairwise GWD cost matrix for Hungarian matching.

    Args:
        boxes1: Predicted boxes of shape ``(N, 5)``.
        boxes2: Target boxes of shape ``(M, 5)``.
        tau: Temperature parameter.

    Returns:
        Cost matrix of shape ``(N, M)``.
    """
    mu_p, sigma_p = _obb_to_gaussian(boxes1)
    mu_t, sigma_t = _obb_to_gaussian(boxes2)

    diff_mu = mu_p[:, None, :] - mu_t[None, :, :]
    term_center = (diff_mu * diff_mu).sum(dim=-1)

    trace_p = sigma_p[..., 0, 0] + sigma_p[..., 1, 1]
    trace_t = sigma_t[..., 0, 0] + sigma_t[..., 1, 1]

    n, m = boxes1.shape[0], boxes2.shape[0]

    sp_exp = sigma_p[:, None, :, :].expand(n, m, 2, 2).reshape(n * m, 2, 2)
    st_exp = sigma_t[None, :, :, :].expand(n, m, 2, 2).reshape(n * m, 2, 2)

    product = torch.bmm(sp_exp, st_exp).reshape(n, m, 2, 2)
    trace_product = product[..., 0, 0] + product[..., 1, 1]

    det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2).clamp(min=1e-8)
    det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2).clamp(min=1e-8)

    det_sqrt = (det_p[:, None] * det_t[None, :]).sqrt()
    trace_sqrt = (trace_product + 2 * det_sqrt).clamp(min=1e-8).sqrt()

    w2 = (term_center + trace_p[:, None] + trace_t[None, :] - 2 * trace_sqrt).clamp(min=0)

    return 1 - 1 / (tau + torch.log1p(w2))
