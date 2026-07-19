# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Utilities for oriented (rotated) bounding box manipulation, IoU, and losses."""

import math

import torch

# Lower bound on box side length before it is treated as degenerate.  Sizes are
# normalised to [0, 1] during training, so 1e-4 is a tenth of a pixel at 1024px.
_SIZE_EPS = 1e-4
# Pure division guard for covariance determinants.  This must stay far below any
# legitimate value: a 10px box at 1024px normalises to a determinant of ~6e-10, so
# a larger floor silently rescales the inverse covariance and collapses the
# Mahalanobis term — making distant small boxes score as near-identical.
_DET_EPS = 1e-20


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [0, pi) with pi-periodicity.

    Args:
        angle: Angles in radians, arbitrary range.

    Returns:
        Angles in [0, pi).
    """
    out = angle - torch.floor(angle / math.pi) * math.pi
    # Float32 rounding can push the result outside the half-open interval: a tiny
    # negative input (which atan2 produces for near-axis-aligned boxes) rounds up to
    # exactly pi, and large-magnitude inputs lose enough precision in the subtraction
    # to land slightly below zero.  Fold both back in — pi is equivalent to 0 here.
    out = torch.where(out >= math.pi, torch.zeros_like(out), out)
    return out.clamp(min=0.0)


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

    ``w`` is taken from the first listed edge and ``h`` from the second, so the
    result follows the annotation's own corner order rather than a canonical
    convention.

    Known limitation: this makes the parameterisation depend on corner order and
    winding. The same physical box yields ``(w=10, h=4, angle=0)`` or
    ``(w=4, h=10, angle=pi/2)`` depending on which corner the file lists first, and
    DOTA annotations contain both windings. ProbIoU is invariant to the difference,
    but the L1 term regresses ``w`` and ``h`` directly, so this injects some noise
    into ``loss_bbox``. Canonicalising to a long-edge convention (``w >= h``) would
    fix it, but it re-parameterises ~90% of DOTA ground truth and therefore requires
    training from scratch — see ``test_winding_changes_parameterisation``.

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

    angle = normalize_angle(torch.atan2(edge_w[..., 1], edge_w[..., 0]))

    return torch.stack([cx, cy, w, h, angle], dim=-1)


def obb_to_aabb(boxes: torch.Tensor) -> torch.Tensor:
    """Reduce oriented boxes to their axis-aligned bounding envelope.

    Needed wherever a 5D oriented box has to be compared against a 4D
    axis-aligned one — notably the two-stage encoder, whose proposals carry no
    angle. Slicing ``boxes[..., :4]`` instead is wrong: it reuses the rotated
    side lengths as if they were axis-aligned extents, so a 40x10 box at 45
    degrees is compared as 40x10 rather than its true 35.36x35.36 envelope.

    Args:
        boxes: Oriented boxes of shape ``(..., 5)`` as ``[cx, cy, w, h, angle]``.

    Returns:
        Tensor of shape ``(..., 4)`` as ``[cx, cy, w, h]``, centre preserved.
    """
    cx, cy, w, h, angle = boxes.unbind(-1)
    cos_a = torch.cos(angle).abs()
    sin_a = torch.sin(angle).abs()
    return torch.stack([cx, cy, w * cos_a + h * sin_a, w * sin_a + h * cos_a], dim=-1)


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

    w = w.clamp(min=_SIZE_EPS)
    h = h.clamp(min=_SIZE_EPS)
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

    Not wired into training: SetCriterion uses :func:`probiou` for the oriented
    regression term. Kept as a validated alternative for experimentation.

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
    det_sqrt = (det_p.clamp(min=_DET_EPS) * det_t.clamp(min=_DET_EPS)).sqrt()
    trace_sqrt = (trace_product + 2 * det_sqrt).clamp(min=_DET_EPS).sqrt()

    w2 = (term_center + trace_p + trace_t - 2 * trace_sqrt).clamp(min=0)

    return 1 - 1 / (tau + torch.log1p(w2))


def kld_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL Divergence loss between oriented boxes modeled as 2D Gaussians.

    Scale-invariant and aspect-ratio adaptive: elongated objects receive
    stronger angular gradients.

    Not wired into training, despite the name of the ``"loss_kld"`` key that
    SetCriterion reports — that key carries :func:`probiou`. Kept as a validated
    alternative for experimentation.

    Args:
        pred: Predicted boxes ``(..., 5)`` as ``[cx, cy, w, h, angle]``.
        target: Target boxes ``(..., 5)``, same shape as pred.

    Returns:
        Per-box KLD loss, same leading shape as inputs.
    """
    mu_p, sigma_p = _obb_to_gaussian(pred)
    mu_t, sigma_t = _obb_to_gaussian(target)

    det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2).clamp(min=_DET_EPS)
    det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2).clamp(min=_DET_EPS)

    inv_t00 = sigma_t[..., 1, 1] / det_t
    inv_t01 = -sigma_t[..., 0, 1] / det_t
    inv_t11 = sigma_t[..., 0, 0] / det_t

    trace_term = inv_t00 * sigma_p[..., 0, 0] + 2 * inv_t01 * sigma_p[..., 0, 1] + inv_t11 * sigma_p[..., 1, 1]

    diff = mu_p - mu_t
    mahal_term = inv_t00 * diff[..., 0] ** 2 + 2 * inv_t01 * diff[..., 0] * diff[..., 1] + inv_t11 * diff[..., 1] ** 2

    log_det_term = torch.log(det_t) - torch.log(det_p)

    kld = 0.5 * (trace_term + mahal_term + log_det_term - 2)

    # Clamp before log1p: raw KLD can go slightly negative from floating-point
    # error when the predicted covariance is near-degenerate.
    return torch.log1p(kld.clamp(min=0))


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
    det_avg = (sigma_avg[..., 0, 0] * sigma_avg[..., 1, 1] - sigma_avg[..., 0, 1] ** 2).clamp(min=_DET_EPS)
    det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2).clamp(min=_DET_EPS)
    det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2).clamp(min=_DET_EPS)

    inv_avg00 = sigma_avg[..., 1, 1] / det_avg
    inv_avg01 = -sigma_avg[..., 0, 1] / det_avg
    inv_avg11 = sigma_avg[..., 0, 0] / det_avg

    diff = mu_p - mu_t
    mahal = inv_avg00 * diff[..., 0] ** 2 + 2 * inv_avg01 * diff[..., 0] * diff[..., 1] + inv_avg11 * diff[..., 1] ** 2

    log_coeff = 0.5 * (torch.log(det_avg) - 0.5 * (torch.log(det_p) + torch.log(det_t)))
    bd = 0.125 * mahal + log_coeff

    hd_squared = (1 - torch.exp(-bd.clamp(max=50))).clamp(min=0)
    return 1 - hd_squared


def probiou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise ProbIoU cost matrix for Hungarian matching.

    Args:
        boxes1: Predicted boxes of shape ``(N, 5)`` as ``[cx, cy, w, h, angle]``.
        boxes2: Target boxes of shape ``(M, 5)``.

    Returns:
        Cost matrix of shape ``(N, M)`` with values in ``[0, 1]``;
        0 = identical boxes, 1 = no overlap.
    """
    return 1 - probiou(boxes1[:, None, :], boxes2[None, :, :])


def gwd_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Pairwise GWD cost matrix for Hungarian matching.

    Not wired into training: HungarianMatcher uses :func:`probiou_pairwise` for
    the oriented cost. Kept as a validated alternative for experimentation,
    alongside :func:`gwd_loss` and :func:`kld_loss`.

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

    det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2).clamp(min=_DET_EPS)
    det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2).clamp(min=_DET_EPS)

    det_sqrt = (det_p[:, None] * det_t[None, :]).sqrt()
    trace_sqrt = (trace_product + 2 * det_sqrt).clamp(min=_DET_EPS).sqrt()

    w2 = (term_center + trace_p[:, None] + trace_t[None, :] - 2 * trace_sqrt).clamp(min=0)

    return 1 - 1 / (tau + torch.log1p(w2))
