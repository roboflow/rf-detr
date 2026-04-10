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

import os
from typing import List

import torch

from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import _ckpt_args_get, validate_checkpoint_compatibility

logger = get_logger()

__all__ = ["load_pretrain_weights"]


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

    nn_model.load_state_dict(checkpoint["model"], strict=False)

    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode_default_num_classes:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names
