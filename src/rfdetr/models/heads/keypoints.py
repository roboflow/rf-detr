# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint helper primitives for GroupPose-style decoding.

Keypoint predictions are dense ``[N, K, KEYPOINT_PRED_DIM]`` tensors where the trailing channel layout is fixed:

================  ===========  ==========================================================
Slot index        Name         Meaning
================  ===========  ==========================================================
``0``             ``x``        Normalized x coordinate (image-relative).
``1``             ``y``        Normalized y coordinate (image-relative).
``2``             ``findable`` Logit for "annotator could find this keypoint" (``v > 0``).
``3``             ``visible``  Logit for "fully visible" (``v == 2``).
``4`` – ``6``     ``L_*``      Lower-triangular Cholesky parameters ``Lxx``, ``Lxy``, ``Lyy``.
``7``             ``class``    Per-keypoint class-logit contribution aggregated into detection-class logits.
================  ===========  ==========================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if TYPE_CHECKING:
    from rfdetr.models.flows import RealNVP


# Number of channels in a keypoint prediction slot — see module docstring for layout.
KEYPOINT_PRED_DIM: int = 8


def modulate(features: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation to a feature tensor.

    Args:
        features: Input feature map to modulate.
        scale: Per-feature scale terms.
        shift: Per-feature shift terms.

    Returns:
        Modulated features.
    """

    return (scale + 1.0) * features + shift


class ConditionalQueryInitializer(nn.Module):  # type: ignore[misc]
    """Initialize keypoint query tokens with adaptive layer-normalization style modulation."""

    def __init__(self, dim: int, num_queries: int, out_dim: int | None = None) -> None:
        """Create the initializer.

        Args:
            dim: Query conditioning dimensionality.
            num_queries: Number of query tokens to instantiate.
            out_dim: Output embedding size. Defaults to ``dim``.
        """

        super().__init__()
        out_dim = out_dim or dim

        self.queries = nn.Parameter(torch.randn(num_queries, out_dim))
        self.query_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim * 3),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """Return modulated query embeddings.

        Args:
            query_features: Tensor of shape ``(B, dim)`` containing conditioning features.

        Returns:
            Tensor of shape ``(B, num_queries, out_dim)`` with initialized keypoint queries.
        """

        normed_query_features = self.query_norm(self.queries)
        scale, shift, gate = self.adaLN_modulation(query_features.unsqueeze(-2)).chunk(3, dim=-1)
        modulated_query_features = self.out_proj(modulate(normed_query_features, scale, shift)) * gate + self.queries
        return modulated_query_features


def compute_l1_keypoint_loss(
    all_pred_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    target_classes: torch.Tensor,
    target_areas: torch.Tensor,
    num_keypoints_per_class: Sequence[int],
    flow: "RealNVP | None" = None,
    keypoint_hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the five-component keypoint loss vector per matched target.

    The tensor layout follows GroupPose-style keypoints where each target class defines how many
    valid keypoints it owns.

    Args:
        all_pred_keypoints: Predicted keypoints with shape ``(N, K_total, >=7)``.
        target_keypoints: Ground truth keypoints with shape ``(N, K_max, 3)``.
        target_classes: Class ids per target with shape ``(N,)``.
        target_areas: Target box areas with shape ``(N,)``.
        num_keypoints_per_class: Number of keypoints per class.
        flow: Optional RLE flow module.
        keypoint_hidden_states: Optional keypoint hidden states matching ``all_pred_keypoints``.

    Returns:
        Tuple of location, findable BCE, visible BCE, Gaussian NLL, and flow-based RLE loss.
    """

    n_targets, total_padded_num_keypoints, pred_dim = all_pred_keypoints.shape
    assert pred_dim >= 7, "Expected all_pred_keypoints last dim >= 7 (x,y,2 logits + 3 chol params)."
    num_classes = len(num_keypoints_per_class)

    if num_classes == 0:
        empty = torch.zeros(n_targets, device=all_pred_keypoints.device, dtype=all_pred_keypoints.dtype)
        return empty, empty, empty, empty, empty

    kpad = total_padded_num_keypoints // num_classes
    split_pred_keypoints = all_pred_keypoints.view(n_targets, num_classes, kpad, pred_dim)
    selected_pred_keypoints = split_pred_keypoints[
        torch.arange(n_targets, device=split_pred_keypoints.device), target_classes
    ]

    active_keypoints_mask = torch.zeros(num_classes, kpad, dtype=torch.bool, device=split_pred_keypoints.device)
    for class_idx, num_keypoints in enumerate(num_keypoints_per_class):
        active_keypoints_mask[class_idx, :num_keypoints] = True

    keypoints_loss_mask = active_keypoints_mask[target_classes]
    keypoints_per_target = keypoints_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
    area = target_areas.to(torch.float32)
    area_eps = torch.finfo(area.dtype).eps
    valid_area = torch.isfinite(area) & (area > area_eps)
    valid_xy = torch.isfinite(selected_pred_keypoints[:, :, :2]).all(dim=-1) & torch.isfinite(
        target_keypoints[:, :, :2]
    ).all(dim=-1)
    valid_visibility = torch.isfinite(target_keypoints[:, :, 2]) & (target_keypoints[:, :, 2] > 0)
    location_loss_mask = keypoints_loss_mask & valid_visibility & valid_xy & valid_area.unsqueeze(1)
    location_count = location_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
    valid_count = location_count.clamp(min=1)
    denom_keypoints = keypoints_per_target.clamp(min=1).to(dtype=selected_pred_keypoints.dtype)
    safe_area_sqrt = area.clamp_min(area_eps).sqrt()

    scaled_masked_l1 = (
        F.l1_loss(selected_pred_keypoints[:, :, :2], target_keypoints[:, :, :2], reduction="none").sum(-1)
        * location_loss_mask.to(selected_pred_keypoints.dtype)
        / safe_area_sqrt.unsqueeze(1)
    )
    location_loss = scaled_masked_l1.sum(-1) / valid_count

    findable_loss = (
        F.binary_cross_entropy_with_logits(
            selected_pred_keypoints[:, :, 2],
            (target_keypoints[:, :, 2] > 0).to(selected_pred_keypoints.dtype),
            reduction="none",
        )
        * keypoints_loss_mask.to(selected_pred_keypoints.dtype)
    ).sum(-1) / denom_keypoints

    visible_loss = (
        F.binary_cross_entropy_with_logits(
            selected_pred_keypoints[:, :, 3],
            (target_keypoints[:, :, 2] > 1).to(selected_pred_keypoints.dtype),
            reduction="none",
        )
        * keypoints_loss_mask.to(selected_pred_keypoints.dtype)
    ).sum(-1) / denom_keypoints

    dxdy = (selected_pred_keypoints[:, :, :2] - target_keypoints[:, :, :2]).to(torch.float32)
    dx = dxdy[:, :, 0]
    dy = dxdy[:, :, 1]

    raw_log_l11 = selected_pred_keypoints[:, :, 4].to(torch.float32)
    raw_l21 = selected_pred_keypoints[:, :, 5].to(torch.float32)
    raw_log_l22 = selected_pred_keypoints[:, :, 6].to(torch.float32)
    finite_uncertainty = torch.isfinite(raw_log_l11) & torch.isfinite(raw_l21) & torch.isfinite(raw_log_l22)
    gaussian_loss_mask = location_loss_mask & finite_uncertainty

    log_l11 = raw_log_l11.clamp(min=-20.0, max=20.0)
    l21 = raw_l21.clamp(min=-1.0e4, max=1.0e4)
    log_l22 = raw_log_l22.clamp(min=-20.0, max=20.0)

    l11 = log_l11.exp()
    l22 = log_l22.exp()
    u0 = l11 * dx + l21 * dy
    u1 = l22 * dy
    maha2 = u0 * u0 + u1 * u1
    gaussian_loss_mask = gaussian_loss_mask & torch.isfinite(u0) & torch.isfinite(u1) & torch.isfinite(maha2)
    gaussian_count = gaussian_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
    gaussian_valid_count = gaussian_count.clamp(min=1)
    nll_raw = 0.5 * (maha2 / area.clamp_min(area_eps).unsqueeze(1)) - (log_l11 + log_l22)
    nll_raw = torch.nan_to_num(nll_raw, nan=0.0, posinf=0.0, neginf=0.0)
    nll_keypoints = nll_raw.masked_fill(~gaussian_loss_mask, 0.0)
    nll_loss = nll_keypoints.sum(-1) / gaussian_valid_count
    no_valid = gaussian_count <= 0
    nll_loss = torch.where(no_valid, torch.zeros_like(nll_loss), nll_loss)

    if flow is not None:
        area_sqrt = safe_area_sqrt.unsqueeze(1).unsqueeze(-1)
        whitened_residuals = torch.stack([u0, u1], dim=-1) / area_sqrt
        selected_kp_hs = None
        if keypoint_hidden_states is not None and n_targets > 0:
            hidden_dim = keypoint_hidden_states.shape[-1]
            keypoint_hs_split = keypoint_hidden_states.view(
                n_targets,
                num_classes,
                kpad,
                hidden_dim,
            )
            selected_kp_hs = keypoint_hs_split[
                torch.arange(n_targets, device=keypoint_hidden_states.device),
                target_classes,
            ]

        flow_loss_mask = gaussian_loss_mask & torch.isfinite(whitened_residuals).all(dim=-1)
        cond_valid = None
        if selected_kp_hs is not None:
            finite_cond = torch.isfinite(selected_kp_hs).all(dim=-1)
            flow_loss_mask = flow_loss_mask & finite_cond
            cond_valid = selected_kp_hs[flow_loss_mask]

        z_valid = whitened_residuals[flow_loss_mask]
        if z_valid.numel() > 0:
            rle_log_prob = flow.log_prob(
                z_valid.to(torch.float32),
                cond=cond_valid.to(torch.float32) if cond_valid is not None else None,
            )
            rle_log_prob = torch.nan_to_num(rle_log_prob, nan=0.0, posinf=0.0, neginf=0.0)
            log_det_l = (log_l11 + log_l22).to(dtype=selected_pred_keypoints.dtype)
            rle_values = torch.zeros(
                (n_targets, kpad),
                device=selected_pred_keypoints.device,
                dtype=selected_pred_keypoints.dtype,
            )
            rle_values[flow_loss_mask] = (-rle_log_prob).to(selected_pred_keypoints.dtype)
            rle_values = rle_values - log_det_l
            rle_values = rle_values.masked_fill(~flow_loss_mask, 0.0)
            flow_count = flow_loss_mask.sum(-1).to(dtype=selected_pred_keypoints.dtype)
            flow_valid_count = flow_count.clamp(min=1)
            rle_loss = rle_values.sum(-1) / flow_valid_count
            rle_loss = torch.where(flow_count <= 0, torch.zeros_like(rle_loss), rle_loss)
        else:
            rle_loss = torch.zeros(n_targets, device=all_pred_keypoints.device, dtype=selected_pred_keypoints.dtype)
    else:
        rle_loss = torch.zeros(n_targets, device=all_pred_keypoints.device, dtype=selected_pred_keypoints.dtype)

    # NaN/Inf protection is upstream (see ``nan_to_num`` calls earlier in this function);
    # per-step hot-path assertions removed for performance (5–15% step latency).
    return location_loss, findable_loss, visible_loss, nll_loss, rle_loss


def _cdist_bce_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise BCE-with-logits summed along the last dim."""

    y_float = y.to(dtype=x.dtype)
    softplus = F.softplus(x).sum(dim=1, keepdim=True)
    dot = torch.matmul(x, y_float.t())
    return softplus - dot


def compute_keypoint_matching_cost(
    all_pred_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    target_classes: torch.Tensor,
    target_areas: torch.Tensor,
    num_keypoints_per_class: Sequence[int],
    flow: RealNVP | None = None,
    keypoint_hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute many-to-many keypoint matching costs.

    Args:
        all_pred_keypoints: Predicted keypoints of shape ``(B, Q, K_total, >=7)``.
        target_keypoints: Ground truth keypoints with shape ``(N, Kmax, 3)``.
        target_classes: Class ids for each target with shape ``(N,)``.
        target_areas: Target box areas with shape ``(N,)``.
        num_keypoints_per_class: Number of keypoints per class.
        flow: Optional RLE flow module.
        keypoint_hidden_states: Optional keypoint hidden states of shape ``(B, Q, K_total, hidden_dim)``.

    Returns:
        Tuple ``(cost_l1, cost_findable, cost_visible, cost_nll, cost_rle)`` each of shape ``(B, Q, N)``.
    """

    b, num_queries, total_num_keypoints, pred_dim = all_pred_keypoints.shape
    assert pred_dim >= 7, "Expected all_pred_keypoints last dim >= 7 (x,y,2 logits + 3 chol params)."
    n_targets = target_keypoints.shape[0]
    num_classes = len(num_keypoints_per_class)

    if n_targets == 0 or num_classes == 0:
        zeros = torch.zeros(
            (b, num_queries, n_targets),
            device=all_pred_keypoints.device,
            dtype=all_pred_keypoints.dtype,
        )
        return zeros, zeros, zeros, zeros, zeros

    kpad = total_num_keypoints // num_classes
    pred = all_pred_keypoints.view(b, num_queries, num_classes, kpad, pred_dim)
    hidden_states_split = (
        keypoint_hidden_states.view(b, num_queries, num_classes, kpad, -1)
        if keypoint_hidden_states is not None
        else None
    )

    cost_l1 = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_findable = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_visible = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_nll = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )
    cost_rle = torch.zeros(
        (b, num_queries, n_targets),
        device=all_pred_keypoints.device,
        dtype=all_pred_keypoints.dtype,
    )

    flat_bq = b * num_queries
    for class_idx in range(num_classes):
        target_indices = (target_classes == class_idx).nonzero().squeeze(1)
        if target_indices.numel() == 0:
            continue
        num_kpts = num_keypoints_per_class[class_idx]
        if num_kpts == 0:
            continue

        pred_by_class = pred[:, :, class_idx, :num_kpts, :]
        hidden_by_class = (
            hidden_states_split[:, :, class_idx, :num_kpts, :] if hidden_states_split is not None else None
        )
        target_by_class = target_keypoints.index_select(0, target_indices)[:, :num_kpts, :]
        n_targets_by_class = target_by_class.shape[0]

        areas = target_areas.index_select(0, target_indices).to(torch.float32)
        area_eps = torch.finfo(areas.dtype).eps
        valid_area = torch.isfinite(areas) & (areas > area_eps)
        target_xy = target_by_class[:, :, :2]
        visible = (
            torch.isfinite(target_by_class[:, :, 2])
            & (target_by_class[:, :, 2] > 0)
            & torch.isfinite(target_xy).all(dim=-1)
            & valid_area.unsqueeze(1)
        )
        valid_per_target = visible.sum(dim=1).to(torch.float32)
        nll_denom = valid_per_target.clamp(min=1)
        has_visible = valid_per_target > 0

        area_sqrt = areas.clamp_min(area_eps).sqrt()

        pred_xy = pred_by_class[:, :, :, :2]
        scaled_loc = torch.zeros(
            (flat_bq, n_targets_by_class),
            device=all_pred_keypoints.device,
            dtype=torch.float32,
        )
        for keypoint_idx in range(num_kpts):
            visibility = visible[:, keypoint_idx].unsqueeze(0)
            distance = torch.cdist(
                pred_xy[:, :, keypoint_idx].reshape(flat_bq, 2).to(torch.float32),
                target_xy[:, keypoint_idx, :].to(torch.float32),
                p=1.0,
            )
            distance.masked_fill_(~visibility, 0.0)
            scaled_loc.add_(distance)
        loc_cost = (scaled_loc / nll_denom.unsqueeze(0)).div(area_sqrt.unsqueeze(0))
        loc_cost = torch.nan_to_num(loc_cost, nan=0.0, posinf=0.0, neginf=0.0)
        cost_l1[:, :, target_indices] = loc_cost.reshape(
            b,
            num_queries,
            n_targets_by_class,
        ).to(all_pred_keypoints.dtype)

        nll_sum = torch.zeros((flat_bq, n_targets_by_class), device=all_pred_keypoints.device, dtype=torch.float32)
        rle_sum = (
            torch.zeros(
                (flat_bq, n_targets_by_class),
                device=all_pred_keypoints.device,
                dtype=torch.float32,
            )
            if flow is not None
            else None
        )

        for keypoint_idx in range(num_kpts):
            visibility_k = visible[:, keypoint_idx]
            x_pred = pred_by_class[:, :, keypoint_idx, 0].reshape(flat_bq, 1).to(torch.float32)
            y_pred = pred_by_class[:, :, keypoint_idx, 1].reshape(flat_bq, 1).to(torch.float32)
            x_tgt = target_xy[:, keypoint_idx, 0].reshape(1, n_targets_by_class).to(torch.float32)
            y_tgt = target_xy[:, keypoint_idx, 1].reshape(1, n_targets_by_class).to(torch.float32)

            dx = x_pred - x_tgt
            dy = y_pred - y_tgt

            raw_log_l11 = pred_by_class[:, :, keypoint_idx, 4].reshape(flat_bq).to(torch.float32)
            raw_l21 = pred_by_class[:, :, keypoint_idx, 5].reshape(flat_bq).to(torch.float32)
            raw_log_l22 = pred_by_class[:, :, keypoint_idx, 6].reshape(flat_bq).to(torch.float32)
            finite_pred = (
                torch.isfinite(x_pred.squeeze(1))
                & torch.isfinite(y_pred.squeeze(1))
                & torch.isfinite(raw_log_l11)
                & torch.isfinite(raw_l21)
                & torch.isfinite(raw_log_l22)
            )

            log_l11 = raw_log_l11.clamp(min=-20.0, max=20.0)
            l21 = raw_l21.clamp(min=-1.0e4, max=1.0e4)
            log_l22 = raw_log_l22.clamp(min=-20.0, max=20.0)

            l11 = log_l11.exp()
            l22 = log_l22.exp()
            dx.mul_(l11.unsqueeze(1)).addcmul_(dy, l21.unsqueeze(1))
            dy.mul_(l22.unsqueeze(1))
            keypoint_mask = (
                visibility_k.unsqueeze(0) & finite_pred.unsqueeze(1) & torch.isfinite(dx) & torch.isfinite(dy)
            )

            if flow is not None:
                z = torch.stack([dx, dy], dim=-1) / area_sqrt.unsqueeze(0).unsqueeze(-1)
                cond_kp = None
                if hidden_by_class is not None:
                    hc_k = (
                        hidden_by_class[:, :, keypoint_idx, :]
                        .reshape(
                            flat_bq,
                            -1,
                        )
                        .to(torch.float32)
                    )
                    cond_kp = (
                        hc_k.unsqueeze(1)
                        .expand(
                            -1,
                            n_targets_by_class,
                            -1,
                        )
                        .reshape(-1, hc_k.shape[-1])
                    )
                flat_flow_mask = (keypoint_mask & torch.isfinite(z).all(dim=-1)).reshape(-1)
                if cond_kp is not None:
                    flat_flow_mask = flat_flow_mask & torch.isfinite(cond_kp).all(dim=-1)
                flow_lp = z.new_zeros(flat_bq * n_targets_by_class)
                if flat_flow_mask.any():
                    cond_valid = cond_kp[flat_flow_mask] if cond_kp is not None else None
                    flow_lp_valid = flow.log_prob(z.reshape(-1, 2)[flat_flow_mask], cond=cond_valid)
                    flow_lp[flat_flow_mask] = torch.nan_to_num(
                        flow_lp_valid,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                rle_keypoint = (-flow_lp).reshape(flat_bq, n_targets_by_class) - ((log_l11 + log_l22).unsqueeze(1))
                rle_keypoint.masked_fill_(~flat_flow_mask.reshape(flat_bq, n_targets_by_class), 0.0)
                if rle_sum is not None:
                    rle_sum.add_(rle_keypoint)

            maha2 = dx.square_().add_(dy.square_())
            keypoint_mask = keypoint_mask & torch.isfinite(maha2)
            nll_k = 0.5 * (maha2 / areas.clamp_min(area_eps).unsqueeze(0)) - (log_l11 + log_l22).unsqueeze(1)
            nll_k = torch.nan_to_num(nll_k, nan=0.0, posinf=0.0, neginf=0.0)
            nll_k.masked_fill_(~keypoint_mask, 0.0)
            nll_sum.add_(nll_k)

        mean_nll = (nll_sum / nll_denom.unsqueeze(0)).reshape(b, num_queries, n_targets_by_class)
        mean_nll.masked_fill_(~has_visible.unsqueeze(0), 0.0)
        cost_nll[:, :, target_indices] = mean_nll.to(all_pred_keypoints.dtype)

        if flow is not None and rle_sum is not None:
            mean_rle = (rle_sum / nll_denom.unsqueeze(0)).reshape(b, num_queries, n_targets_by_class)
            mean_rle.masked_fill_(~has_visible.unsqueeze(0), 0.0)
            cost_rle[:, :, target_indices] = mean_rle.to(all_pred_keypoints.dtype)

        pred_findable = pred_by_class[:, :, :, 2].reshape(flat_bq, num_kpts)
        target_findable = (
            (target_by_class[:, :, 2] > 0)
            .to(all_pred_keypoints.dtype)
            .reshape(
                n_targets_by_class,
                num_kpts,
            )
        )
        pred_visible = pred_by_class[:, :, :, 3].reshape(flat_bq, num_kpts)
        target_visible = (
            (target_by_class[:, :, 2] > 1)
            .to(all_pred_keypoints.dtype)
            .reshape(
                n_targets_by_class,
                num_kpts,
            )
        )
        cost_findable[:, :, target_indices] = _cdist_bce_with_logits(pred_findable, target_findable).reshape(
            b,
            num_queries,
            n_targets_by_class,
        ) / float(num_kpts)
        cost_visible[:, :, target_indices] = _cdist_bce_with_logits(pred_visible, target_visible).reshape(
            b,
            num_queries,
            n_targets_by_class,
        ) / float(num_kpts)

    # NaN/Inf protection is upstream (see ``nan_to_num`` calls earlier); per-step
    # matching-cost assertions removed for performance (5–15% step latency).
    return cost_l1, cost_findable, cost_visible, cost_nll, cost_rle


__all__ = [
    "ConditionalQueryInitializer",
    "compute_keypoint_matching_cost",
    "compute_l1_keypoint_loss",
]
