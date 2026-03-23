# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared weight-loading and LoRA application utilities.

Provides the canonical implementations of pretrained checkpoint loading and
LoRA adapter injection, used by both the L1 inference facade (``rfdetr.detr``)
and the L2 LightningModule (``rfdetr.training.module_model``).

The weight-loading logic is taken from ``RFDETRModelModule._load_pretrain_weights``
in ``module_model.py`` (more complete: Pydantic-aware user-override detection,
auto-alignment for fine-tuned checkpoints) and augmented with class-name
extraction from ``detr.py:_load_pretrain_weights_into``.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import List

import torch

from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.utilities.decorators import deprecated
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import _ckpt_args_get, validate_checkpoint_compatibility

logger = get_logger()

__all__ = ["load_pretrain_weights", "apply_lora"]


@deprecated(
    target=True,
    args_mapping={"train_config": None},
    deprecated_in="1.8",
    remove_in="1.9",
    num_warns=-1,
    stream=functools.partial(warnings.warn, category=DeprecationWarning),
)
def load_pretrain_weights(
    nn_model: torch.nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig | None = None,
) -> List[str]:
    """Load pretrained checkpoint weights into *nn_model* in-place.

    Canonical implementation shared by the L1 facade (``_build_model_context``
    in ``rfdetr.detr``) and the L2 LightningModule (``RFDETRModelModule.__init__``
    in ``rfdetr.training.module_model``).

    Uses the Pydantic-aware logic from ``module_model.py``:

    - When the user did **not** explicitly override ``num_classes`` (left at the
      ModelConfig default), the checkpoint class count is treated as authoritative
      and the model head is auto-aligned to it.
    - When the user **did** explicitly override ``num_classes`` to a value larger
      than the checkpoint provides, the head is temporarily aligned to the
      checkpoint for loading, then expanded back to the configured size.
    - When the checkpoint has more classes than configured (backbone-pretrain
      scenario), both reinitializations are applied: expand to checkpoint size for
      loading, then trim to configured size.

    Class names stored in the checkpoint ``args`` are extracted and returned.

    Args:
        nn_model: The model whose weights will be updated in-place.
        model_config: Pydantic ``ModelConfig`` instance. Must have
            ``pretrain_weights``, ``num_classes``, ``num_queries``, and
            ``group_detr`` attributes.
        train_config: Deprecated since v1.8 — no longer used internally.
            Passing a non-``None`` value emits a ``DeprecationWarning``.
            Omit the argument; it will be removed in v1.9.

    Returns:
        List of class name strings from the checkpoint, or an empty list if none
        are present or if ``model_config.pretrain_weights`` is ``None``.

    Raises:
        Exception: If the checkpoint file cannot be loaded even after a re-download.
    """
    mc = model_config
    pretrain_weights = mc.pretrain_weights
    if pretrain_weights is None:
        return []
    class_names: List[str] = []

    # Download first (no-op if already present and hash is valid).
    download_pretrain_weights(pretrain_weights)
    # If the first download attempt didn't produce the file (e.g. stale MD5
    # caused an earlier ValueError that was silently swallowed), retry with
    # MD5 validation disabled so a stale registry hash can't block training.
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

    # Extract class_names from the checkpoint if available (ported from detr.py).
    if "args" in checkpoint:
        raw_class_names = _ckpt_args_get(checkpoint["args"], "class_names")
        if raw_class_names:
            # Normalize to a new List[str] to avoid leaking mutable references and
            # to respect the annotated return type.
            if isinstance(raw_class_names, str):
                class_names = [raw_class_names]
            else:
                try:
                    iterator = iter(raw_class_names)
                except TypeError:
                    # Non-iterable, ignore and keep the default empty list.
                    class_names = []
                else:
                    class_names = [name for name in iterator if isinstance(name, str)]

    validate_checkpoint_compatibility(checkpoint, mc)

    # Determine whether the user explicitly set num_classes on the ModelConfig,
    # and whether that explicit value differs from the model default.
    user_set_num_classes = False
    if hasattr(mc, "model_fields_set"):
        user_set_num_classes = "num_classes" in getattr(mc, "model_fields_set", set())
    default_num_classes = type(mc).model_fields["num_classes"].default
    num_classes = mc.num_classes
    # True only when the user explicitly set num_classes to a non-default value.
    user_overrode_default_num_classes = user_set_num_classes and num_classes != default_num_classes

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    configured_num_classes_plus_bg = num_classes + 1
    if checkpoint_num_classes != configured_num_classes_plus_bg:
        # Align model head size before loading checkpoint weights.
        if checkpoint_num_classes < configured_num_classes_plus_bg:
            # Checkpoint has FEWER classes than configured.
            if not user_overrode_default_num_classes:
                # Auto-align to the checkpoint when the user did NOT provide a
                # non-default override for num_classes (i.e., left it at the
                # ModelConfig default): treat the checkpoint as authoritative.
                num_classes = checkpoint_num_classes - 1
                configured_num_classes_plus_bg = checkpoint_num_classes
                mc.num_classes = num_classes
        # In all mismatch cases we need the head to match the checkpoint's
        # class count so load_state_dict succeeds without size mismatches.
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    # Trim query embeddings to the configured query count.
    num_desired_queries = mc.num_queries * mc.group_detr
    query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(x) for x in query_param_names):
            checkpoint["model"][name] = checkpoint["model"][name][:num_desired_queries]

    nn_model.load_state_dict(checkpoint["model"], strict=False)

    # If the user explicitly set a class count larger than the checkpoint,
    # expand/reinitialize the head back to the configured size after load.
    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode_default_num_classes:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    # Only trim back down when loading a larger pretrain checkpoint into a
    # smaller configured task-specific class count.
    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names


def apply_lora(nn_model: torch.nn.Module) -> None:
    """Apply LoRA adapters to the backbone encoder of *nn_model*.

    Replaces ``nn_model.backbone[0].encoder`` in-place with a PEFT-wrapped
    encoder using DoRA with rank 16 and alpha 16.

    Args:
        nn_model: LWDETR model whose backbone encoder will receive LoRA adapters.

    Raises:
        ImportError: If ``peft`` is not installed.
            Install via the RF-DETR extras, for example::

                pip install "rfdetr[lora]"
                # or
                pip install "rfdetr[train]"
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the 'peft' dependency. "
            "Install it via RF-DETR extras, e.g.: "
            'pip install "rfdetr[lora]" or pip install "rfdetr[train]".'
        ) from exc

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        use_dora=True,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "qkv",
            "query",
            "key",
            "value",
            "cls_token",
            "register_tokens",
        ],
    )
    nn_model.backbone[0].encoder = get_peft_model(nn_model.backbone[0].encoder, lora_config)
