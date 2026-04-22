# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared pretrained-weight loading helpers.

This module provides a config-native weight loader for tests and internal
callers that operate on a ``ModelConfig`` instance directly rather than on the
legacy namespace used by ``rfdetr.detr``.
"""

from __future__ import annotations

import math
import os
from typing import List

import torch
import torch.nn.functional as F

from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import _ckpt_args_get, validate_checkpoint_compatibility

logger = get_logger()

__all__ = ["load_pretrain_weights"]

_PE_KEY_SUFFIX = "embeddings.position_embeddings"


def _interpolate_position_embeddings(
    checkpoint_state: dict,
    pe_size: int,
) -> None:
    """Interpolate DINOv2 positional embeddings in *checkpoint_state* to match *pe_size*.

    When the model is configured with a custom ``resolution`` that differs from the
    checkpoint's training resolution, the DINOv2 backbone's ``position_embeddings``
    parameter has an incompatible shape.  ``load_state_dict(strict=False)`` does **not**
    skip shape mismatches on matching keys — it raises ``RuntimeError``.

    This function bicubic-interpolates every PE tensor in the checkpoint whose shape
    differs from the target grid, modifying *checkpoint_state* in-place before
    ``load_state_dict`` is called.

    Args:
        checkpoint_state: The ``"model"`` sub-dict from a loaded checkpoint.
        pe_size: Target grid side length in patches (number of patches per spatial
            dimension, assuming a square grid).  Typically
            ``model_config.positional_encoding_size``.
    """
    n_target = pe_size * pe_size  # target number of patch tokens

    pe_keys = [k for k in checkpoint_state if k.endswith(_PE_KEY_SUFFIX)]
    for key in pe_keys:
        ckpt_pe = checkpoint_state[key]  # [1, N_src+1, dim]
        n_source = ckpt_pe.shape[1] - 1  # exclude class token
        if n_source == n_target:
            continue  # no mismatch — skip

        h_src = int(math.isqrt(n_source))
        h_tgt = int(math.isqrt(n_target))
        if h_src * h_src != n_source or h_tgt * h_tgt != n_target:
            logger.warning(
                f"Skipping PE interpolation for {key}:"
                f" grid size is not a perfect square (source {n_source}, target {n_target}).",
            )
            continue

        dim = ckpt_pe.shape[-1]
        class_token = ckpt_pe[:, :1]  # [1, 1, dim] — keeps the sequence dimension
        patch_pe = ckpt_pe[:, 1:]  # [1, N_src, dim]

        patch_pe = patch_pe.reshape(1, h_src, h_src, dim).permute(0, 3, 1, 2)  # [1, dim, H, W]
        patch_pe = F.interpolate(
            patch_pe.float(),
            size=(h_tgt, h_tgt),
            mode="bicubic",
            align_corners=False,
            antialias=patch_pe.device.type != "mps",
        ).to(ckpt_pe.dtype)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, n_target, dim)  # [1, N_tgt, dim]

        checkpoint_state[key] = torch.cat([class_token, patch_pe], dim=1)
        logger.debug(
            "Interpolated positional embeddings %s: %s → %s.",
            key,
            tuple(ckpt_pe.shape),
            tuple(checkpoint_state[key].shape),
        )


def load_pretrain_weights(nn_model: torch.nn.Module, model_config: ModelConfig) -> List[str]:
    """Load pretrained checkpoint weights into ``nn_model`` in-place.

    Args:
        nn_model: The model to update.
        model_config: Model configuration describing the checkpoint and target
            architecture.

    Returns:
        Class names extracted from the checkpoint ``args`` block, or an empty
        list when the checkpoint does not embed them.
    """
    pretrain_weights = model_config.pretrain_weights
    if pretrain_weights is None:
        return []

    class_names: List[str] = []

    download_pretrain_weights(pretrain_weights)
    if not os.path.isfile(pretrain_weights):
        logger.warning("Pretrain weights not found after initial download; retrying without MD5 validation.")
        download_pretrain_weights(pretrain_weights, redownload=True, validate_md5=False)
    validate_pretrain_weights(pretrain_weights, strict=False)

    try:
        checkpoint = torch.load(pretrain_weights, map_location="cpu", weights_only=False)
    except Exception:
        logger.info("Failed to load pretrain weights, re-downloading")
        download_pretrain_weights(pretrain_weights, redownload=True, validate_md5=False)
        checkpoint = torch.load(pretrain_weights, map_location="cpu", weights_only=False)

    if "args" in checkpoint:
        class_names = _ckpt_args_get(checkpoint["args"], "class_names") or []

    validate_checkpoint_compatibility(checkpoint, model_config)

    user_set_num_classes = "num_classes" in getattr(model_config, "model_fields_set", set())
    default_num_classes = type(model_config).model_fields["num_classes"].default
    num_classes = model_config.num_classes
    user_overrode_default_num_classes = user_set_num_classes and num_classes != default_num_classes

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    configured_num_classes_plus_bg = num_classes + 1
    if checkpoint_num_classes != configured_num_classes_plus_bg:
        if checkpoint_num_classes < configured_num_classes_plus_bg and not user_overrode_default_num_classes:
            num_classes = checkpoint_num_classes - 1
            configured_num_classes_plus_bg = checkpoint_num_classes
            model_config.num_classes = num_classes
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    num_desired_queries = model_config.num_queries * model_config.group_detr
    query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(param_name) for param_name in query_param_names):
            checkpoint["model"][name] = checkpoint["model"][name][:num_desired_queries]

    _interpolate_position_embeddings(checkpoint["model"], model_config.positional_encoding_size)
    nn_model.load_state_dict(checkpoint["model"], strict=False)

    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode_default_num_classes:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names
