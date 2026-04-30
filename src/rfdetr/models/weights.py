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
import math
import os
import warnings
from typing import List

import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.utilities.decorators import deprecated
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import _ckpt_args_get, validate_checkpoint_compatibility

logger = get_logger()

__all__ = ["load_pretrain_weights", "apply_lora", "interpolate_position_embeddings"]

_PE_KEY_SUFFIX = "embeddings.position_embeddings"

_QUERY_PARAM_SUFFIXES: tuple[str, ...] = ("refpoint_embed.weight", "query_feat.weight")


def _slice_query_param_per_group(
    tensor: torch.Tensor,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> torch.Tensor:
    """Slice a ``refpoint_embed`` / ``query_feat`` weight preserving per-group structure.

    ``LWDETR`` packs query embeddings as ``nn.Embedding(num_queries * group_detr, ...)``
    where group ``g`` occupies the contiguous slot range
    ``[g * num_queries, (g + 1) * num_queries)`` (see ``LWDETR.__init__`` and
    ``LWDETR.forward`` in ``models/lwdetr.py``).  When ``num_queries`` decreases
    and ``group_detr > 1``, a flat ``tensor[: target_num_queries * target_group_detr]``
    slice silently scrambles groups: the tail of group 0 winds up in what should
    be group 1's slots, and so on.  At inference only group 0 is read so the
    bug is invisible, but for training-resume it corrupts groups 1+.

    This helper does the right thing per group:

    * ``num_queries`` decrease (``target_num_queries < ckpt_num_queries``) →
      keep the first ``target_num_queries`` slots of each retained group.
    * ``group_detr`` decrease (``target_group_detr < ckpt_group_detr``) →
      drop tail groups; retained groups stay pretrained.
    * Either dimension expands → return the per-group sub-tensor anyway.
      ``load_state_dict`` will raise a shape mismatch immediately, since the
      checkpoint has fewer slots than the model expects.

    When the tensor's flat length disagrees with
    ``ckpt_num_queries * ckpt_group_detr`` (corrupt or unexpected checkpoint
    shape), fall back to the legacy flat slice so loading continues with the
    same behavior the codebase had before this fix.

    Args:
        tensor: The checkpoint tensor for ``refpoint_embed.weight`` or
            ``query_feat.weight``.
        ckpt_num_queries: ``num_queries`` recorded in the checkpoint's training args.
        ckpt_group_detr: ``group_detr`` recorded in the checkpoint's training args.
        target_num_queries: ``num_queries`` configured for the model.
        target_group_detr: ``group_detr`` configured for the model.

    Returns:
        A tensor whose layout matches the model's configured packing for the
        decrease-or-equal cases, or a smaller per-group sub-tensor for the
        expansion case (which ``load_state_dict`` will then reject).
    """
    expected_total = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_total:
        # Args inconsistent with tensor — fall back to legacy flat slice.
        return tensor[: target_num_queries * target_group_detr]

    if target_num_queries == ckpt_num_queries and target_group_detr == ckpt_group_detr:
        return tensor

    keep_groups = min(target_group_detr, ckpt_group_detr)
    keep_per_group = min(target_num_queries, ckpt_num_queries)
    pieces = [tensor[g * ckpt_num_queries : g * ckpt_num_queries + keep_per_group] for g in range(keep_groups)]
    return torch.cat(pieces, dim=0)


def interpolate_position_embeddings(
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

    # Normalize PyTorch Lightning native .ckpt format to the expected {"model": {...}}
    # structure.  PTL stores model weights in "state_dict" with keys prefixed by
    # "model." (matching the attribute path inside RFDETRModelModule).  Legacy and
    # BestModelCallback checkpoints already have a top-level "model" key.
    if "model" not in checkpoint and "state_dict" in checkpoint:
        logger.debug("Normalizing PTL .ckpt checkpoint format (state_dict -> model)")
        prefix = "model."
        # When the model was wrapped with torch.compile, PTL stores weights with keys
        # like "model._orig_mod.<param>".  Strip the extra "_orig_mod." segment so the
        # resulting keys match the expected bare parameter names.
        compile_prefix = "_orig_mod."
        model_state = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith(prefix):
                stripped = k[len(prefix) :]
                if stripped.startswith(compile_prefix):
                    stripped = stripped[len(compile_prefix) :]
                model_state[stripped] = v
        if not model_state:
            raise ValueError(
                f"The checkpoint at {pretrain_weights!r} appears to be in PyTorch Lightning "
                "format ('state_dict' key present, 'model' key absent), but 'state_dict' "
                "contains no keys with the expected 'model.' prefix. "
                "The checkpoint may be corrupt or in an unsupported format."
            )
        checkpoint["model"] = model_state
        # PTL stores training hyper-parameters under "hyper_parameters".  Map them
        # to the "args" key expected by class-name extraction and compatibility checks
        # (only when "args" is not already present).
        if "args" not in checkpoint and "hyper_parameters" in checkpoint:
            checkpoint["args"] = checkpoint["hyper_parameters"]

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

    # Reshape query embeddings to the configured query count, preserving per-group
    # structure when the checkpoint records its training-time num_queries / group_detr.
    # See _slice_query_param_per_group for why a flat slice is wrong with group_detr > 1.
    ckpt_args = checkpoint.get("args")
    ckpt_num_queries = _ckpt_args_get(ckpt_args, "num_queries") if ckpt_args is not None else None
    ckpt_group_detr = _ckpt_args_get(ckpt_args, "group_detr") if ckpt_args is not None else None
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(x) for x in _QUERY_PARAM_SUFFIXES):
            tensor = checkpoint["model"][name]
            if ckpt_num_queries is not None and ckpt_group_detr is not None:
                checkpoint["model"][name] = _slice_query_param_per_group(
                    tensor,
                    ckpt_num_queries=ckpt_num_queries,
                    ckpt_group_detr=ckpt_group_detr,
                    target_num_queries=mc.num_queries,
                    target_group_detr=mc.group_detr,
                )
            else:
                # Legacy checkpoint with no num_queries/group_detr in args:
                # preserve the original flat slice for backward compatibility.
                checkpoint["model"][name] = tensor[: mc.num_queries * mc.group_detr]

    interpolate_position_embeddings(checkpoint["model"], mc.positional_encoding_size)
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
