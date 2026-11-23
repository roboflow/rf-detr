# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared weight-loading and LoRA application utilities.

Provides the canonical implementations of pretrained checkpoint loading and LoRA adapter injection, used by both the L1
inference facade (``rfdetr.detr``) and the L2 LightningModule (``rfdetr.training.module_model``).

The weight-loading logic is taken from ``RFDETRModelModule._load_pretrain_weights`` in ``module_model.py`` (more
complete: Pydantic-aware user-override detection, auto-alignment for fine-tuned checkpoints) and augmented with class-
name extraction from ``detr.py:_load_pretrain_weights_into``.
"""

from __future__ import annotations

import copy
import math
import os
import re
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from rfdetr.assets.model_weights import ModelWeights, download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import ModelConfig
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import (
    _LORA_ENCODER_KEY_PREFIX,
    _ckpt_args_get,
    remap_projector_to_cross_attn,
    validate_checkpoint_compatibility,
)

logger = get_logger()

__all__ = ["adapt_input_channels", "apply_lora", "interpolate_position_embeddings", "load_pretrain_weights"]

_PE_KEY_SUFFIX = "embeddings.position_embeddings"
_STEM_KEY_SUFFIX = "patch_embeddings.projection.weight"

# Query-related parameters that LWDETR packs as nn.Embedding(num_queries * group_detr, ...).
# Any new query parameter packed the same way must be added here.
_QUERY_PARAM_SUFFIXES: tuple[str, ...] = ("refpoint_embed.weight", "query_feat.weight")


def _is_query_param(name: str) -> bool:
    """Return whether *name* is a state-dict key for a packed query parameter.

    Single source of truth for the ``_QUERY_PARAM_SUFFIXES`` suffix match, so every call site agrees on which keys are
    reshaped per group.

    Args:
        name: A checkpoint / model ``state_dict`` key.

    Returns:
        ``True`` when the key ends with one of ``_QUERY_PARAM_SUFFIXES``.

    Examples:
        >>> _is_query_param("transformer.refpoint_embed.weight")
        True
        >>> _is_query_param("class_embed.weight")
        False
    """
    return name.endswith(_QUERY_PARAM_SUFFIXES)


def _slice_query_param_per_group(
    tensor: Tensor,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> Tensor:
    """Slice a ``refpoint_embed`` / ``query_feat`` weight preserving per-group structure.

    ``LWDETR`` packs query embeddings as ``nn.Embedding(num_queries * group_detr, ...)`` where group ``g`` occupies the
    contiguous slot range ``[g * num_queries, (g + 1) * num_queries)`` (see ``LWDETR.__init__`` and ``LWDETR.forward``
    in ``models/lwdetr.py``).  When ``num_queries`` decreases and ``group_detr > 1``, a flat ``tensor[:
    target_num_queries * target_group_detr]`` slice silently scrambles groups: the tail of group 0 winds up in what
    should be group 1's slots, and so on.  At inference only group 0 is read so the bug is invisible, but for
    training-resume it corrupts groups 1+.

    This helper does the right thing per group:

    * ``num_queries`` decrease (``target_num_queries < ckpt_num_queries``) →
      keep the first ``target_num_queries`` slots of each retained group.
    * ``group_detr`` decrease (``target_group_detr < ckpt_group_detr``) →
      drop tail groups; retained groups stay pretrained.
    * Either dimension expands, or one shrinks while the other expands →
      return whatever per-group sub-tensor can be built (``min(target, ckpt)`` along each axis). The result has fewer
      rows than the model expects; ``load_pretrain_weights`` then pads a ``num_queries`` increase with
      :func:`_pad_query_param_per_group` and rejects anything else.

    When the tensor's flat length disagrees with ``ckpt_num_queries * ckpt_group_detr`` (corrupt or unexpected
    checkpoint shape), fall back to the legacy flat slice so loading continues with the same behavior the codebase had
    before this fix.

    Args:
        tensor: The checkpoint tensor for ``refpoint_embed.weight`` or
            ``query_feat.weight``.
        ckpt_num_queries: ``num_queries`` recorded in the checkpoint's training args.
        ckpt_group_detr: ``group_detr`` recorded in the checkpoint's training args.
        target_num_queries: ``num_queries`` configured for the model.
        target_group_detr: ``group_detr`` configured for the model.

    Returns:
        A tensor whose layout matches the model's configured packing for the decrease-or-equal cases, or a per-group
        sub-tensor built from ``min(target, ckpt)`` along each axis for the expansion case (which ``load_state_dict``
        will then reject on shape mismatch).

    Raises:
        ValueError: If any of ``ckpt_num_queries``, ``ckpt_group_detr``,
            ``target_num_queries``, or ``target_group_detr`` is ≤ 0.
    """
    if ckpt_num_queries <= 0 or ckpt_group_detr <= 0 or target_num_queries <= 0 or target_group_detr <= 0:
        raise ValueError(
            f"_slice_query_param_per_group: all dimension args must be positive; "
            f"got ckpt_num_queries={ckpt_num_queries}, ckpt_group_detr={ckpt_group_detr}, "
            f"target_num_queries={target_num_queries}, target_group_detr={target_group_detr}."
        )

    expected_total = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_total:
        # Args inconsistent with tensor shape — fall back to legacy flat slice.
        logger.warning(
            "_slice_query_param_per_group: checkpoint args claim %d × %d = %d rows "
            "but tensor has %d rows; falling back to flat slice. Per-group structure "
            "may be scrambled if group_detr > 1.",
            ckpt_num_queries,
            ckpt_group_detr,
            expected_total,
            tensor.shape[0],
        )
        return tensor[: target_num_queries * target_group_detr]

    if target_num_queries == ckpt_num_queries and target_group_detr == ckpt_group_detr:
        return tensor

    keep_groups = min(target_group_detr, ckpt_group_detr)
    keep_per_group = min(target_num_queries, ckpt_num_queries)
    pieces = [tensor[g * ckpt_num_queries : g * ckpt_num_queries + keep_per_group] for g in range(keep_groups)]
    return torch.cat(pieces, dim=0)


def _pad_query_param_per_group(
    tensor: torch.Tensor,
    ckpt_num_queries: int,
    target_num_queries: int,
    target_group_detr: int,
    fresh: torch.Tensor,
) -> torch.Tensor:
    """Grow a ``refpoint_embed`` / ``query_feat`` weight to more queries per group, keeping every pretrained row.

    Each group of the result starts with the checkpoint's rows for that group; the remaining rows of the group come from
    ``fresh``, the tensor the model initialised for itself. Appending the fresh rows flat would instead push the
    checkpoint's second group into the new slots of the first, which is the group inference reads.

    Args:
        tensor: Checkpoint rows already cut to ``target_group_detr`` groups of ``ckpt_num_queries`` rows each.
        ckpt_num_queries: Queries per group held by ``tensor``.
        target_num_queries: Queries per group the model needs; at least ``ckpt_num_queries``.
        target_group_detr: Number of groups in both ``tensor`` and the result.
        fresh: The model's own tensor of shape ``(target_num_queries * target_group_detr, ...)``.

    Returns:
        A new tensor shaped like ``fresh``; ``fresh`` itself is not modified.

    Raises:
        ValueError: If the shapes of ``tensor`` and ``fresh`` do not match the stated layout.

    Examples:
        >>> pretrained = torch.tensor([[1.0], [2.0]])
        >>> fresh = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0]])
        >>> _pad_query_param_per_group(pretrained, 1, 2, 2, fresh).squeeze(-1).tolist()
        [1.0, -2.0, 2.0, -4.0]
    """
    if ckpt_num_queries > target_num_queries:
        raise ValueError(f"cannot pad {ckpt_num_queries} queries per group down to {target_num_queries}.")
    if (
        tensor.shape[0] != ckpt_num_queries * target_group_detr
        or fresh.shape[0] != target_num_queries * target_group_detr
    ):
        raise ValueError(
            f"cannot pad: got {tensor.shape[0]} pretrained and {fresh.shape[0]} fresh rows for "
            f"{target_group_detr} groups of {ckpt_num_queries} -> {target_num_queries} queries."
        )
    padded = fresh.clone()
    for group in range(target_group_detr):
        start = group * target_num_queries
        padded[start : start + ckpt_num_queries] = tensor[group * ckpt_num_queries : (group + 1) * ckpt_num_queries]
    return padded


def _filter_intentional_keys(keys: list[str]) -> list[str]:
    """Return *keys* with intentional-reinit/trim entries removed.

    Matching is boundary-aware: a pattern matches a key when the pattern appears at the start of the key or immediately
    after a module separator (``.``).  This prevents substring collisions where a pattern like ``"class_embed."`` would
    inadvertently match a key belonging to an unrelated module (e.g. ``"class_embed_projection.weight"`` is safe because
    ``class_embed_projection.`` ≠ ``class_embed.``, but using a plain ``in`` check against longer ambiguous strings is
    fragile by design).
    """
    # Substrings identifying state-dict keys that ``load_pretrain_weights`` is
    # *expected* to have to reconcile (head reinitialisation and per-group query
    # trimming).  Keys matching any of these are filtered from the partial-load
    # warning so it only fires on *unexpected* mismatches that indicate a real
    # config / checkpoint incompatibility.
    intentional_patterns: tuple[str, ...] = (
        "class_embed.",
        "bbox_embed.",
        *_QUERY_PARAM_SUFFIXES,
        "enc_out_class_embed.",
        "enc_out_bbox_embed.",
        # The preview keypoint checkpoint still stores the old standalone
        # MLP projection head, but the current GroupPose inference path no
        # longer consumes it.
        "keypoint_head.keypoint_proj.",
    )

    def _is_intentional(key: str) -> bool:
        return any(key.startswith(pat) or f".{pat}" in key for pat in intentional_patterns)

    return [k for k in keys if not _is_intentional(k)]


def _warn_on_partial_load(incompatible: Any, pretrain_weights_path: str) -> None:
    """Emit a ``logger.warning`` when ``load_state_dict`` left non-trivial gaps.

    ``load_state_dict(strict=False)`` silently ignores keys that the model has but the checkpoint does not
    (``missing_keys``) and keys present in the checkpoint but absent from the model (``unexpected_keys``).  When this
    happens for parameters outside the head / query embeddings — which the loader intentionally reinitialises or trims —
    the corresponding model weights were left at their random initial values and the user is silently getting a much
    weaker model. This helper surfaces that condition with a single, actionable warning. Same-key shape mismatches do
    not reach this function — they raise :class:`RuntimeError` directly from ``load_state_dict`` and are therefore
    impossible to miss.

    Two separate exclusions are applied before deciding whether to warn, for two different reasons:

    * :func:`_filter_intentional_keys` drops head / query-embedding keys in *both* directions, because the loader
      itself reinitialises or trims them.  Matching there is module-prefix based.
    * ``_kp_active_mask`` is dropped from ``missing_keys`` only, because it is a deterministic schema buffer rather
      than a learned parameter — the model always rebuilds it from the configured keypoint schema, so a checkpoint
      predating keypoint support is not actually missing anything.  Matching is on the exact terminal key so a
      similarly-named real parameter still warns, and the ``unexpected_keys`` direction is deliberately left alone.

    Args:
        incompatible: The ``_IncompatibleKeys`` namedtuple returned by
            :meth:`torch.nn.Module.load_state_dict`.
        pretrain_weights_path: Path to the checkpoint that was loaded; included
            in the warning so the user can identify which load partially succeeded.
    """
    missing_keys_raw = getattr(incompatible, "missing_keys", None)
    unexpected_keys_raw = getattr(incompatible, "unexpected_keys", None)
    try:
        missing_keys = [str(k) for k in missing_keys_raw] if missing_keys_raw else []
        unexpected_keys = [str(k) for k in unexpected_keys_raw] if unexpected_keys_raw else []
    except TypeError:
        # Result wasn't iterable (e.g. a MagicMock in unit tests) — quietly skip.
        return
    # `_kp_active_mask` is a deterministic schema buffer, not a learned
    # parameter. Detection checkpoints published before keypoint support do not
    # contain it; the current model correctly reconstructs it from the configured
    # keypoint schema (an empty mask for detection-only variants).
    missing: list[str] = []
    for key in _filter_intentional_keys(missing_keys):
        if key == "_kp_active_mask" or key.endswith("._kp_active_mask"):
            # Keep the breadcrumb at debug level: harmless for detection-only variants, but a keypoint model
            # loading a mask-less checkpoint keeps its config-derived schema, and this is the only remaining trace.
            logger.debug(
                "Checkpoint %r has no %r — rebuilding it from the configured keypoint schema.",
                pretrain_weights_path,
                key,
            )
            continue
        missing.append(key)
    unexpected = _filter_intentional_keys(unexpected_keys)
    if not missing and not unexpected:
        return

    parts: list[str] = []
    if missing:
        sample = ", ".join(missing[:5])
        if len(missing) > 5:
            sample += ", ..."
        parts.append(f"{len(missing)} model parameter(s) not in checkpoint (left at random init): [{sample}]")
    if unexpected:
        sample = ", ".join(unexpected[:5])
        if len(unexpected) > 5:
            sample += ", ..."
        parts.append(f"{len(unexpected)} checkpoint key(s) not consumed by model: [{sample}]")

    logger.warning(
        "Pretrained weights at %r loaded only partially — this typically produces "
        "lower accuracy. %s. Check that the model configuration (encoder, hidden_dim, "
        "out_feature_indexes, projector_scale, ...) matches the architecture the "
        "checkpoint was trained with.",
        pretrain_weights_path,
        " ".join(parts),
    )


def interpolate_position_embeddings(
    checkpoint_state: dict[str, Any],
    pe_size: int,
) -> None:
    """Interpolate DINOv2 positional embeddings in *checkpoint_state* to match *pe_size*.

    When the model is configured with a custom ``resolution`` that differs from the checkpoint's training resolution,
    the DINOv2 backbone's ``position_embeddings`` parameter has an incompatible shape.
    ``load_state_dict(strict=False)`` does **not** skip shape mismatches on matching keys — it raises ``RuntimeError``.

    This function bicubic-interpolates every PE tensor in the checkpoint whose shape differs from the target grid,
    modifying *checkpoint_state* in-place before ``load_state_dict`` is called.

    Args:
        checkpoint_state: The ``"model"`` sub-dict from a loaded checkpoint.
        pe_size: Target grid side length in patches (number of patches per spatial
            dimension, assuming a square grid).  Typically ``model_config.positional_encoding_size``.
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


def load_pretrain_weights(
    nn_model: LWDETR,
    model_config: ModelConfig,
    *,
    trust: bool = False,
) -> list[str]:
    """Load pretrained checkpoint weights into *nn_model* in-place.

    Canonical implementation shared by the L1 facade (``_build_model_context`` in ``rfdetr.detr``) and the L2
    LightningModule (``RFDETRModelModule.__init__`` in ``rfdetr.training.module_model``).

    Uses the Pydantic-aware logic from ``module_model.py``:

    - When the user did **not** explicitly set ``num_classes`` (it is left unset),
      the checkpoint class count is treated as authoritative and the model head is auto-aligned to it.
    - When the user **did** explicitly set ``num_classes`` (to any value, including the
      class default) larger than the checkpoint provides, the head is temporarily aligned to the checkpoint for
      loading, then expanded back to the configured size.
    - When the checkpoint has more classes than configured (backbone-pretrain
      scenario), both reinitializations are applied: expand to checkpoint size for loading, then trim to configured
      size.

    A checkpoint saved by a ``backbone_lora=True`` run is loaded into a LoRA-wrapped encoder. With
    ``model_config.backbone_lora`` set the encoder stays wrapped with the trained adapters; otherwise the adapters are
    merged into the encoder weights, unless the caller had already wrapped the encoder (see :func:`_load_model_state`).

    Class names stored in the checkpoint ``args`` are extracted and returned.

    Args:
        nn_model: The model whose weights will be updated in-place.
        model_config: Pydantic ``ModelConfig`` instance. Must have
            ``pretrain_weights``, ``num_classes``, ``num_queries``, and ``group_detr`` attributes.
        trust: Forwarded to :func:`rfdetr.utilities.io._safe_torch_load` as its ``trust``
            argument. Set ``True`` only when ``model_config.pretrain_weights`` points to a
            checkpoint the caller explicitly trusts (mirrors ``RFDETR.from_checkpoint(...,
            trust_checkpoint=True)``); the safe-load default otherwise applies here too.

    Returns:
        List of class name strings from the checkpoint, or an empty list if none are present or if
        ``model_config.pretrain_weights`` is ``None``.

    Raises:
        Exception: If the checkpoint file cannot be loaded.
        ImportError: If the checkpoint was saved by a ``backbone_lora=True`` run and ``peft`` is not installed.
    """
    mc = model_config
    if mc.pretrain_weights is None:
        return []
    # `expand_path`/`_coerce_resume_path` pydantic validators on ModelConfig already normalize
    # this field to a `str` at runtime; the `str()` here just satisfies the static `PathLikeStr` type.
    pretrain_weights = str(mc.pretrain_weights)
    class_names: list[str] = []

    from rfdetr.utilities.io import _safe_torch_load

    # Download first (no-op if already present and hash is valid).
    download_pretrain_weights(pretrain_weights)
    # If the first download attempt didn't produce the file (e.g. stale MD5
    # caused an earlier ValueError that was silently swallowed), retry once.
    # MD5 validation is kept on the retry — if it fails again the error is real.
    if not os.path.isfile(pretrain_weights):
        logger.warning("Pretrain weights not found after initial download; retrying.")
        download_pretrain_weights(pretrain_weights, redownload=True)
    validated_weights = validate_pretrain_weights(pretrain_weights, strict=False)

    # A failed read does not prove the file is a disposable published weight: it may be a user's checkpoint.
    checkpoint = _safe_torch_load(pretrain_weights, trust=trust)

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

    # Determine whether the user explicitly set num_classes on the ModelConfig.
    # "num_classes" in model_fields_set is True only when the field was explicitly provided
    # (or assigned afterwards, e.g. by dataset alignment); an explicit value is honored
    # regardless of whether it equals the class default, so an intentional num_classes always
    # wins over the checkpoint's class count.
    # Capture BEFORE any mc.num_classes assignment below — those re-add "num_classes" to
    # model_fields_set, which would falsely appear as user-set to the second guard (~line 524).
    user_overrode = "num_classes" in getattr(mc, "model_fields_set", set())
    num_classes = mc.num_classes

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    configured_num_classes_plus_bg = num_classes + 1
    if checkpoint_num_classes != configured_num_classes_plus_bg:
        # Align model head size before loading checkpoint weights.
        if checkpoint_num_classes < configured_num_classes_plus_bg:
            # Checkpoint has FEWER classes than configured.
            if not user_overrode:
                # Auto-align to the checkpoint when the user did NOT explicitly set
                # num_classes (i.e., left it unset): treat the checkpoint as authoritative.
                num_classes = checkpoint_num_classes - 1
                configured_num_classes_plus_bg = checkpoint_num_classes
                mc.num_classes = num_classes
                # The assignment above records ``num_classes`` in Pydantic's ``model_fields_set``
                # as if the user had set it. ``RFDETR._align_num_classes_from_dataset`` reads that
                # set to decide whether the training dataset may change the class count, so a
                # checkpoint-derived count left there is kept even when the dataset has MORE
                # classes: the head and the criterion stay at the checkpoint's size and the new
                # labels index past them at the first training step (a CUDA device-side assert;
                # on Windows WDDM a display-driver reset). Clear the flag, as ``from_checkpoint``
                # does for every checkpoint-derived field, so the dataset count can still raise it.
                fields_set = getattr(mc, "model_fields_set", None)
                if fields_set is None:
                    fields_set = getattr(mc, "__pydantic_fields_set__", None)
                if fields_set is not None:
                    fields_set.discard("num_classes")
        # In all mismatch cases we need the head to match the checkpoint's
        # class count so load_state_dict succeeds without size mismatches.
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    # Reshape query embeddings to the configured query count, preserving per-group
    # structure when the checkpoint records its training-time num_queries / group_detr.
    # See _slice_query_param_per_group for why a flat slice is wrong with group_detr > 1.
    ckpt_args = checkpoint.get("args")
    saved_config = checkpoint.get("model_config")
    layout_source = (
        saved_config
        if isinstance(saved_config, dict) and {"num_queries", "group_detr"} <= saved_config.keys()
        else ckpt_args
    )
    ckpt_num_queries_raw = _ckpt_args_get(layout_source, "num_queries") if layout_source is not None else None
    ckpt_group_detr_raw = _ckpt_args_get(layout_source, "group_detr") if layout_source is not None else None
    try:
        ckpt_num_queries = int(ckpt_num_queries_raw) if ckpt_num_queries_raw is not None else None
        ckpt_group_detr = int(ckpt_group_detr_raw) if ckpt_group_detr_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint query layout must contain integer num_queries and group_detr.") from exc

    query_rows = [t.shape[0] for n, t in checkpoint["model"].items() if _is_query_param(n)]
    # Only a checksum-verified published checkpoint establishes that its encoder heads are complete. Arbitrary
    # checkpoints can omit trailing heads, so matching contiguous indices alone cannot prove the group count.
    asset = ModelWeights.from_filename(os.path.basename(pretrain_weights))
    known_published_weights = asset is not None and asset.md5_hash is not None and validated_weights is True
    if ckpt_num_queries is None and ckpt_group_detr is None and query_rows and known_published_weights:
        head_groups = {
            int(match.group(1))
            for name in checkpoint["model"]
            if (match := re.search(r"(?:^|\.)enc_out_class_embed\.(\d+)\.weight$", name))
        }
        box_groups = {
            int(match.group(1))
            for name in checkpoint["model"]
            if (match := re.search(r"(?:^|\.)enc_out_bbox_embed\.(\d+)\.layers\.0\.weight$", name))
        }
        if (
            head_groups
            and head_groups == box_groups == set(range(len(head_groups)))
            and all(rows % len(head_groups) == 0 for rows in query_rows)
        ):
            ckpt_group_detr = len(head_groups)
            ckpt_num_queries = query_rows[0] // ckpt_group_detr

    if (ckpt_num_queries is None) != (ckpt_group_detr is None) and query_rows:
        known = ckpt_num_queries if ckpt_num_queries is not None else ckpt_group_detr
        if known is not None and known > 0 and query_rows[0] % known == 0:
            if ckpt_num_queries is None:
                ckpt_num_queries = query_rows[0] // known
            else:
                ckpt_group_detr = query_rows[0] // known

    if ckpt_num_queries is not None and ckpt_group_detr is not None:
        if (
            ckpt_num_queries <= 0
            or ckpt_group_detr <= 0
            or any(rows != ckpt_num_queries * ckpt_group_detr for rows in query_rows)
        ):
            raise ValueError("Checkpoint query layout does not match the query tensor shapes.")

    target_query_rows = mc.num_queries * mc.group_detr
    default_queries = type(mc).model_fields["num_queries"].default
    default_groups = type(mc).model_fields["group_detr"].default
    if (ckpt_num_queries is None or ckpt_group_detr is None) and query_rows:
        # A row total has many possible factorizations. Do not silently reinterpret a custom checkpoint when growth
        # requires padding, including when fewer target groups make the total appear to shrink.
        if any(rows < target_query_rows for rows in query_rows) or (
            mc.num_queries != default_queries or mc.group_detr != default_groups
        ):
            raise ValueError(
                "Cannot change num_queries or group_detr because the checkpoint layout is unknown; "
                "save model_config or use pretrain_weights=None."
            )
        if mc.group_detr > 1 and any(rows > target_query_rows for rows in query_rows):
            logger.warning(
                "load_pretrain_weights: checkpoint lacks query layout; falling back to flat slice. "
                "With group_detr=%d this may scramble per-group query structure.",
                mc.group_detr,
            )
    # Growing num_queries keeps the pretrained rows of every group and takes the extra rows from the model's own
    # initialisation; group_detr cannot grow, because the extra groups would need encoder heads the checkpoint lacks.
    can_pad_queries = (
        ckpt_num_queries is not None
        and ckpt_group_detr is not None
        and ckpt_num_queries < mc.num_queries
        and mc.group_detr <= ckpt_group_detr
    )
    fresh_state = nn_model.state_dict() if can_pad_queries else {}
    padded_any = False
    for name in list(checkpoint["model"].keys()):
        if _is_query_param(name):
            tensor = checkpoint["model"][name]
            if ckpt_num_queries is not None and ckpt_group_detr is not None:
                checkpoint["model"][name] = _slice_query_param_per_group(
                    tensor,
                    ckpt_num_queries=ckpt_num_queries,
                    ckpt_group_detr=ckpt_group_detr,
                    target_num_queries=mc.num_queries,
                    target_group_detr=mc.group_detr,
                )
                fresh = fresh_state.get(name)
                if (
                    can_pad_queries
                    and isinstance(fresh, torch.Tensor)
                    and fresh.shape[0] == target_query_rows
                    and fresh.shape[1:] == tensor.shape[1:]
                ):
                    checkpoint["model"][name] = _pad_query_param_per_group(
                        checkpoint["model"][name], ckpt_num_queries, mc.num_queries, mc.group_detr, fresh
                    )
                    padded_any = True
            else:
                # Checkpoint without args.num_queries/args.group_detr (published legacy files
                # and BestModelCallback output, whose args value is a TrainConfig dump lacking
                # both fields):
                # preserve the original flat slice for backward compatibility.
                # This is only a data identity when the row total already matches;
                # without metadata, a different (num_queries, group_detr) split
                # is still reinterpreted under the target partition undetectably.
                checkpoint["model"][name] = tensor[:target_query_rows]

    if padded_any:
        logger.warning(
            "load_pretrain_weights: num_queries=%d exceeds the %d the checkpoint was trained with; the %d extra query "
            "slots per group are untrained (initialised like a fresh model) and need fine-tuning before they add "
            "useful detections.",
            mc.num_queries,
            ckpt_num_queries,
            mc.num_queries - (ckpt_num_queries or 0),
        )
    # Anything still short could not be padded (group_detr growth, or a layout that cannot be recovered), and would
    # surface later as a bare ``load_state_dict`` size mismatch. Fail here with the setting to change instead.
    short_rows = {
        name: tensor.shape[0]
        for name, tensor in checkpoint["model"].items()
        if _is_query_param(name) and tensor.shape[0] < target_query_rows
    }
    if short_rows:
        raise ValueError(
            f"Pretrained weights hold {min(short_rows.values())} query rows but the config needs "
            f"{target_query_rows} (num_queries={mc.num_queries} x group_detr={mc.group_detr}). Pretrained weights "
            "cannot grow group_detr, and num_queries only when the checkpoint layout is known: keep the defaults and "
            "raise only num_select, or pass pretrain_weights=None to train with the larger layout from scratch."
        )

    checkpoint["model"] = remap_projector_to_cross_attn(checkpoint["model"], nn_model)

    # Auto-align num_keypoints_per_class from checkpoint when the user did not explicitly set it.
    # Without this, loading a bg-first checkpoint ([0, 17]) into a model configured with the
    # active-first default ([17]) causes a semantic mismatch: the detection head is trimmed to 2
    # rows but keeps background weights at row 0 — the slot active-first inference expects to be
    # person — producing AP ≈ 0.0 on pretrained keypoint models.
    # Mirrors the existing num_classes auto-align pattern (lines ~385-392).
    _user_overrode_kp_schema = "num_keypoints_per_class" in getattr(mc, "model_fields_set", set())
    if (
        not _user_overrode_kp_schema
        and getattr(mc, "use_grouppose_keypoints", False)
        and hasattr(mc, "num_keypoints_per_class")
    ):
        _early_kp_mask = checkpoint["model"].get("_kp_active_mask")
        if isinstance(_early_kp_mask, Tensor) and _early_kp_mask.ndim == 2:
            _ckpt_kp_schema = [int(n) for n in _early_kp_mask.sum(dim=1).tolist()]
            _cfg_kp_schema = list(getattr(mc, "num_keypoints_per_class", []) or [])
            if not any(n > 0 for n in _ckpt_kp_schema):
                logger.warning(
                    "load_pretrain_weights: _kp_active_mask in checkpoint has no active slots "
                    "(schema=%s) — skipping auto-align to avoid overwriting config with empty schema.",
                    _ckpt_kp_schema,
                )
            elif _ckpt_kp_schema != _cfg_kp_schema:
                logger.debug(
                    "load_pretrain_weights: auto-aligning num_keypoints_per_class %s → %s "
                    "(inferred from checkpoint _kp_active_mask; user did not set explicitly).",
                    _cfg_kp_schema,
                    _ckpt_kp_schema,
                )
                mc.num_keypoints_per_class = _ckpt_kp_schema
        elif isinstance(_early_kp_mask, Tensor):
            logger.warning(
                "load_pretrain_weights: _kp_active_mask has unexpected shape %s (expected 2-D) "
                "— skipping auto-align; schema mismatch may cause AP≈0 on keypoint models.",
                tuple(_early_kp_mask.shape),
            )

    # Detection checkpoints/configs may omit keypoint schema fields; absence means no keypoint reconciliation.
    configured_keypoint_schema = list(getattr(mc, "num_keypoints_per_class", []) or [])
    checkpoint_keypoint_schema = None
    should_restore_config_keypoint_schema = False
    if getattr(mc, "use_grouppose_keypoints", False) and hasattr(
        nn_model, "get_num_keypoints_per_class_from_checkpoint"
    ):
        checkpoint_keypoint_schema = nn_model.get_num_keypoints_per_class_from_checkpoint(checkpoint["model"])
        if checkpoint_keypoint_schema:
            get_model_schema = getattr(nn_model, "get_num_keypoints_per_class", None)
            model_keypoint_schema = get_model_schema() if callable(get_model_schema) else configured_keypoint_schema
            if checkpoint_keypoint_schema != model_keypoint_schema and hasattr(nn_model, "reinitialize_keypoint_head"):
                logger.warning(
                    "load_pretrain_weights: temporarily resizing keypoint schema from %s to checkpoint schema %s.",
                    model_keypoint_schema,
                    checkpoint_keypoint_schema,
                )
                nn_model.reinitialize_keypoint_head(checkpoint_keypoint_schema)
                should_restore_config_keypoint_schema = checkpoint_keypoint_schema != configured_keypoint_schema

    # `_kp_active_mask` is a deterministic buffer derived from
    # `num_keypoints_per_class` in the current config. Some checkpoints store a
    # shape that reflects an earlier schema (for example [2, 17] vs [1, 17]).
    # Dropping only this key avoids hard load failures while preserving all
    # learned weights.
    ckpt_kp_active_mask = checkpoint["model"].get("_kp_active_mask")
    model_state_dict = nn_model.state_dict() if hasattr(nn_model, "state_dict") else {}
    model_kp_active_mask = model_state_dict.get("_kp_active_mask") if isinstance(model_state_dict, dict) else None
    if (
        isinstance(ckpt_kp_active_mask, Tensor)
        and isinstance(model_kp_active_mask, Tensor)
        and ckpt_kp_active_mask.shape != model_kp_active_mask.shape
        and not should_restore_config_keypoint_schema
    ):
        logger.warning(
            "load_pretrain_weights: dropping checkpoint _kp_active_mask with shape %s "
            "because current model expects %s (derived from current keypoint schema).",
            tuple(ckpt_kp_active_mask.shape),
            tuple(model_kp_active_mask.shape),
        )
        checkpoint["model"].pop("_kp_active_mask", None)
    interpolate_position_embeddings(checkpoint["model"], mc.positional_encoding_size)
    _widen_stem_for_checkpoint(nn_model, checkpoint["model"], mc)
    incompatible = _load_model_state(nn_model, checkpoint["model"], backbone_lora=mc.backbone_lora)
    _warn_on_partial_load(incompatible, pretrain_weights)

    if should_restore_config_keypoint_schema and hasattr(nn_model, "reinitialize_keypoint_head"):
        logger.warning(
            "load_pretrain_weights: restoring configured keypoint schema %s after checkpoint load.",
            configured_keypoint_schema,
        )
        nn_model.reinitialize_keypoint_head(configured_keypoint_schema)

    # If the user explicitly set a class count larger than the checkpoint,
    # expand/reinitialize the head back to the configured size after load.
    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    # Only trim back down when loading a larger pretrain checkpoint into a
    # smaller configured task-specific class count.
    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names


def _adapt_input_conv(num_channels: int, conv_weight: Tensor) -> Tensor:
    """Adapt a 3-channel pretrained conv weight tensor to *num_channels* input channels.

    When ``num_channels == 3``: returns the weight unchanged. When ``num_channels == 1``: averages weights across the
    original 3 channels.
    Otherwise (``num_channels != 1`` and ``num_channels != 3``): tiles the 3-channel
    pattern and scales by ``3 / num_channels`` to preserve activation magnitude.

    Args:
        num_channels: Target number of input channels.
        conv_weight: Original weight tensor of shape ``[out_ch, 3, H, W]``.

    Returns:
        Adapted weight tensor of shape ``[out_ch, num_channels, H, W]``.
    """
    if num_channels == 3:
        return conv_weight
    if num_channels == 1:
        return conv_weight.mean(dim=1, keepdim=True)
    # General case: tile and scale
    repeats = (num_channels + 2) // 3
    weight_out = torch.cat([conv_weight] * repeats, dim=1)[:, :num_channels]
    weight_out = weight_out * (3.0 / num_channels)
    return weight_out


def _patch_embeddings(nn_model: LWDETR) -> Any:
    """Return the DINOv2 patch-embedding module of *nn_model* (the model's input stem).

    Args:
        nn_model: Model with a DINOv2 backbone, possibly wrapped by PEFT.

    Returns:
        The patch-embedding module, whose ``projection`` is the stem convolution.
    """
    backbone = cast(Backbone, nn_model.backbone[0])
    # A PEFT wrapper forwards attribute access to the wrapped DinoV2.
    return cast(DinoV2, backbone.encoder).encoder.embeddings.patch_embeddings


def adapt_input_channels(nn_model: LWDETR, num_channels: int, encoder: str) -> None:
    """Rebuild the DINOv2 patch embedding of *nn_model* for *num_channels* input channels, in-place.

    Shared by the prediction build and the training module, so both models take the same input.  A stem that already
    has *num_channels* inputs is left alone, which makes a second call a no-op.

    Args:
        nn_model: Model whose patch embedding is adapted.
        num_channels: Input channel count the model is configured for.
        encoder: Configured encoder name (``ModelConfig.encoder``).

    Raises:
        ValueError: If *num_channels* is not 3 and *encoder* is not a DINOv2 encoder.
    """
    if num_channels == 3:
        return
    # Dispatch on the encoder name, as Backbone._build_encoder does: with backbone_lora the encoder is a PEFT
    # wrapper, not a DinoV2 instance.
    if encoder.split("_")[0] != "dinov2":
        raise ValueError(f"num_channels={num_channels} is supported for DINOv2 encoders only, not encoder={encoder!r}.")
    patch_embeddings = _patch_embeddings(nn_model)
    proj = patch_embeddings.projection
    if proj.in_channels == num_channels:
        return
    new_proj = copy.deepcopy(proj)
    new_proj.in_channels = num_channels
    new_proj.weight = torch.nn.Parameter(_adapt_input_conv(num_channels, proj.weight))
    new_proj.weight.requires_grad = proj.weight.requires_grad
    patch_embeddings.projection = new_proj
    patch_embeddings.num_channels = num_channels


def _widen_stem_for_checkpoint(nn_model: LWDETR, model_state: dict[str, Tensor], model_config: ModelConfig) -> None:
    """Widen the model's stem before loading a checkpoint that was saved from an already widened model.

    A 3-channel checkpoint needs nothing here: it loads into the 3-channel stem and is adapted afterwards.  A checkpoint
    whose stem already has ``model_config.num_channels`` inputs comes from an earlier multi-channel run, so the model
    is widened first and the trained filters of the extra channels load with the rest.

    Args:
        nn_model: Model about to receive *model_state*.
        model_state: Checkpoint weights.
        model_config: Configuration of *nn_model*.

    Raises:
        ValueError: If the checkpoint's stem is neither 3 channels wide nor ``model_config.num_channels`` wide.
    """
    stem = next((value for key, value in model_state.items() if key.endswith(_STEM_KEY_SUFFIX)), None)
    if stem is None or stem.shape[1] == 3:
        return
    if stem.shape[1] != model_config.num_channels:
        raise ValueError(
            f"The checkpoint was saved from a model with num_channels={stem.shape[1]}, but this model is configured "
            f"with num_channels={model_config.num_channels}. Only a 3-channel checkpoint can be adapted to another "
            f"channel count; pass num_channels={stem.shape[1]} to load this checkpoint."
        )
    adapt_input_channels(nn_model, model_config.num_channels, model_config.encoder)


def _load_model_state(nn_model: LWDETR, model_state: dict[str, Tensor], *, backbone_lora: bool) -> Any:
    """Load *model_state* into *nn_model* with the encoder in the layout the checkpoint was saved from.

    A ``backbone_lora=True`` run saves the encoder under PEFT names (``backbone.0.encoder.base_model.model.…``). A plain
    encoder consumes none of them, so ``load_state_dict(strict=False)`` would skip them all and leave the encoder at
    random init (#1540). The encoder is therefore wrapped with LoRA before such a checkpoint loads. A model configured
    without LoRA then gets the loaded adapters merged into its encoder weights, and the trainable flags that peft froze
    when wrapping are restored. An encoder the caller had already wrapped is loaded as is and stays wrapped.

    Args:
        nn_model: Model to load into, in-place.
        model_state: Checkpoint weights, already aligned to the model's heads and query counts.
        backbone_lora: Whether the model is configured to keep LoRA adapters on its encoder.

    Returns:
        The ``load_state_dict`` result listing missing and unexpected keys.

    Raises:
        ImportError: If *model_state* comes from a ``backbone_lora=True`` run and ``peft`` is not installed.
    """
    if not any(key.startswith(_LORA_ENCODER_KEY_PREFIX) for key in model_state):
        return nn_model.load_state_dict(model_state, strict=False)

    backbone = cast(Backbone, nn_model.backbone[0])
    encoder = backbone.encoder
    encoder_requires_grad = {name: param.requires_grad for name, param in encoder.named_parameters()}
    try:
        apply_lora(nn_model)
    except ImportError as err:
        raise ImportError(
            "This checkpoint was saved by a backbone_lora=True run, and loading it needs the 'peft' dependency. "
            'Install it with pip install "rfdetr[lora]".'
        ) from err
    incompatible = nn_model.load_state_dict(model_state, strict=False)
    # Merge only a wrap made here: an encoder the caller had already wrapped stays as the caller left it.
    if not backbone_lora and backbone.encoder is not encoder:
        from peft import PeftModel  # optional dependency; apply_lora above has already required it

        backbone.encoder = cast(DinoV2, cast(PeftModel, backbone.encoder).merge_and_unload())
        for name, param in backbone.encoder.named_parameters():
            param.requires_grad = encoder_requires_grad[name]
        logger.info("Merged the checkpoint's LoRA adapters into the backbone encoder, since backbone_lora=False.")
    return incompatible


def apply_lora(nn_model: LWDETR) -> None:
    """Apply LoRA adapters to the backbone encoder of *nn_model*.

    Replaces ``nn_model.backbone[0].encoder`` in-place with a PEFT-wrapped encoder using DoRA with rank 16 and alpha 16.
    Does nothing when the encoder is already wrapped, which :func:`load_pretrain_weights` does itself to load a
    checkpoint saved by a ``backbone_lora=True`` run.

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
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requires the 'peft' dependency. "
            "Install it via RF-DETR extras, e.g.: "
            'pip install "rfdetr[lora]" or pip install "rfdetr[train]".'
        ) from exc

    backbone = cast(Backbone, nn_model.backbone[0])
    if isinstance(backbone.encoder, PeftModel):
        return

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
    # PEFT's type signature requires a transformers PreTrainedModel, but DinoV2 is a compatible nn.Module
    # wrapper at runtime. Cast both sides of this dynamic wrapper boundary instead of relying
    # on an environment-sensitive ignore for PEFT's evolving type annotations.
    peft_encoder = get_peft_model(cast(Any, backbone.encoder), lora_config)
    backbone.encoder = cast(DinoV2, peft_encoder)
