# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Convert PyTorch RF-DETR weights to MLX format.

Handles the key remapping between PyTorch HuggingFace-style naming and the flat MLX module naming, plus Conv2d weight
transposition (PyTorch [out,in,kH,kW] -> MLX [out,kH,kW,in]).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    import torch

import mlx.nn as nn
import mlx.utils
import numpy as np

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# ============================================================
# Backbone key mapping (HuggingFace DINOv2 -> MLX)
# ============================================================

_BACKBONE_PREFIX_MAP = {
    "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection": "backbone.patch_embed.proj",
    "backbone.0.encoder.encoder.embeddings.cls_token": "backbone.cls_token",
    "backbone.0.encoder.encoder.embeddings.register_tokens": "backbone.register_tokens",
    "backbone.0.encoder.encoder.embeddings.position_embeddings": "backbone.pos_embed",
    "backbone.0.encoder.encoder.layernorm": "backbone.norm",
}

_LAYER_SUBKEY_MAP = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    # attn.q/k/v.* are staging names, not MLX parameter paths — the backbone's Attention builds one fused
    # `qkv` Linear, so convert_state_dict collects these three and concatenates them into attn.qkv.*.
    "attention.attention.query.weight": "attn.q.weight",
    "attention.attention.query.bias": "attn.q.bias",
    "attention.attention.key.weight": "attn.k.weight",
    "attention.attention.key.bias": "attn.k.bias",
    "attention.attention.value.weight": "attn.v.weight",
    "attention.attention.value.bias": "attn.v.bias",
    "attention.output.dense.weight": "attn.out.weight",
    "attention.output.dense.bias": "attn.out.bias",
    "mlp.fc1.weight": "mlp.fc1.weight",
    "mlp.fc1.bias": "mlp.fc1.bias",
    "mlp.fc2.weight": "mlp.fc2.weight",
    "mlp.fc2.bias": "mlp.fc2.bias",
    "layer_scale1.lambda1": "ls1.gamma",
    "layer_scale2.lambda1": "ls2.gamma",
}

_LAYER_PATTERN = re.compile(r"^backbone\.0\.encoder\.encoder\.encoder\.layer\.(\d+)\.(.+)$")

# Matches the staging keys emitted for the three HuggingFace attention projections, which are fused
# into one `attn.qkv` parameter. Order is fixed at q, k, v — the same convention the decoder's
# self_attn.in_proj_weight split in convert_state_dict unpacks.
_BACKBONE_QKV_PATTERN = re.compile(r"^blocks\.(\d+)\.attn\.([qkv])\.(weight|bias)$")
_QKV_ORDER = ("q", "k", "v")
# Role letter -> HuggingFace projection name, so a missing-projection error names the key the
# caller can actually search for in their checkpoint.
_QKV_HF_NAMES = {"q": "query", "k": "key", "v": "value"}


def _remap_backbone_key(key: str) -> Optional[str]:
    """Map a PyTorch backbone key to its MLX equivalent.

    Args:
        key: PyTorch state_dict key.

    Returns:
        MLX parameter path, or None if the key should be skipped. Attention projections return the
        ``attn.q/k/v.*`` staging names, which :func:`convert_state_dict` fuses into ``attn.qkv.*``.
    """
    for hf_prefix, mlx_prefix in _BACKBONE_PREFIX_MAP.items():
        if key == hf_prefix or key.startswith(hf_prefix + "."):
            suffix = key[len(hf_prefix) :]
            return mlx_prefix + suffix

    m = _LAYER_PATTERN.match(key)
    if m:
        layer_idx = m.group(1)
        subkey = m.group(2)
        mlx_subkey = _LAYER_SUBKEY_MAP.get(subkey)
        if mlx_subkey is None:
            return None
        return f"backbone.blocks.{layer_idx}.{mlx_subkey}"

    return None


# ============================================================
# Decoder key mapping
# ============================================================


def _remap_decoder_key(key: str) -> Optional[str]:
    """Map a PyTorch decoder key to its MLX equivalent.

    Args:
        key: PyTorch state_dict key (without 'decoder.' prefix).

    Returns:
        MLX parameter path relative to the decoder module, or None if skipped.
    """
    parts = key.split(".")
    try:
        idx = next(i for i, p in enumerate(parts) if p in ("transformer", "decoder"))
        return ".".join(parts[idx + 1 :])
    except StopIteration:
        return key


def _map_decoder_weight_key(st_key: str) -> Optional[str]:
    """Map a safetensors/state_dict key to the MLX decoder parameter path.

    Args:
        st_key: Key from the converted state dict (decoder-prefixed keys).

    Returns:
        MLX parameter path, or None if skipped.
    """
    if st_key.startswith("projector."):
        return st_key

    if st_key == "query_feat.weight":
        return "query_feat"
    if st_key == "refpoint_embed.weight":
        return "refpoint_embed"
    if st_key.startswith("class_embed."):
        return st_key
    if st_key.startswith("bbox_embed.layers."):
        return st_key.replace("bbox_embed.layers.", "bbox_embed_layers.")

    if st_key.startswith("decoder.enc_output.0."):
        return st_key.replace("decoder.enc_output.0.", "enc_output.0.")
    if st_key.startswith("decoder.enc_output_norm.0."):
        return st_key.replace("decoder.enc_output_norm.0.", "enc_output_norm.0.")
    if st_key.startswith("decoder.enc_out_class_embed.0."):
        return st_key.replace("decoder.enc_out_class_embed.0.", "enc_out_class_embed.0.")
    if st_key.startswith("decoder.enc_out_bbox_embed.0.layers."):
        rest = st_key[len("decoder.enc_out_bbox_embed.0.layers.") :]
        return "enc_out_bbox_embed.0." + rest

    if st_key.startswith("decoder.decoder.layers."):
        rest = st_key[len("decoder.decoder.") :]
        return "decoder." + rest
    if st_key.startswith("decoder.decoder.norm."):
        rest = st_key[len("decoder.decoder.") :]
        return "decoder." + rest
    if st_key.startswith("decoder.decoder.ref_point_head."):
        rest = st_key[len("decoder.decoder.") :]
        return "decoder." + rest

    return None


# ============================================================
# Weight conversion
# ============================================================


def _transpose_conv_weight(w: np.ndarray) -> np.ndarray:
    """Transpose conv weight from PyTorch to MLX layout.

    Args:
        w: (out, in, kH, kW) PyTorch conv weight.

    Returns:
        (out, kH, kW, in) MLX conv weight.
    """
    return w.transpose(0, 2, 3, 1)


def convert_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Convert a PyTorch RF-DETR state_dict to MLX backbone and decoder weights.

    Args:
        state_dict: PyTorch model state_dict (from model.model.state_dict()).

    Returns:
        Tuple of (backbone_weights, decoder_weights) as numpy arrays.

    Raises:
        ValueError: If a backbone layer supplies only some of its q/k/v projections, leaving the
            fused ``attn.qkv`` parameter impossible to build.
    """
    backbone_weights: Dict[str, np.ndarray] = {}
    decoder_weights: Dict[str, np.ndarray] = {}
    decoder_in_proj: Dict[str, np.ndarray] = {}
    dropped: Dict[str, list] = {"backbone": [], "decoder": []}
    backbone_qkv: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().float().numpy()

        if key.startswith("backbone.0.projector."):
            mlx_key = key[len("backbone.0.") :]
            if arr.ndim == 4:
                arr = _transpose_conv_weight(arr)
            decoder_weights[mlx_key] = arr

        elif "backbone" in key:
            mlx_key = _remap_backbone_key(key)
            if mlx_key is None:
                dropped["backbone"].append(key)
                continue
            # Strip "backbone." prefix for loading into backbone module
            bare_key = mlx_key[len("backbone.") :] if mlx_key.startswith("backbone.") else mlx_key

            # Hold the separate q/k/v projections back; they are fused after the scan completes.
            qkv_match = _BACKBONE_QKV_PATTERN.match(bare_key)
            if qkv_match is not None:
                layer_idx, role, param_type = qkv_match.groups()
                backbone_qkv.setdefault((layer_idx, param_type), {})[role] = arr
                continue

            if arr.ndim == 4:
                arr = _transpose_conv_weight(arr)
            backbone_weights[bare_key] = arr

        elif "transformer" in key or "decoder" in key:
            # Remap decoder keys
            remapped = "decoder." + _remap_decoder_key(key)
            mlx_key = _map_decoder_weight_key(remapped)
            if mlx_key is None:
                dropped["decoder"].append(key)
                continue

            # Track in_proj_weight/bias for later splitting
            if "self_attn.in_proj_weight" in key or "self_attn.in_proj_bias" in key:
                decoder_in_proj[key] = arr
                continue

            if arr.ndim == 4:
                arr = _transpose_conv_weight(arr)
            decoder_weights[mlx_key] = arr

        else:
            # Top-level keys (class_embed, bbox_embed, query_feat, refpoint_embed)
            mlx_key = _map_decoder_weight_key(key)
            if mlx_key is None:
                dropped["decoder"].append(key)
                continue
            decoder_weights[mlx_key] = arr

    # Fuse the backbone's three separate q/k/v projections into one `attn.qkv` matrix. This is the
    # mirror image of the decoder's in_proj split below and uses the same q, k, v row order; axis 0
    # holds output features for both the (3*dim, dim) weight and the (3*dim,) bias.
    for (layer_idx, param_type), parts in backbone_qkv.items():
        missing = [_QKV_HF_NAMES[role] for role in _QKV_ORDER if role not in parts]
        if missing:
            raise ValueError(
                f"backbone layer {layer_idx} is missing the {', '.join(missing)} projection "
                f"{param_type} — cannot build the fused attn.qkv.{param_type}"
            )
        fused = np.concatenate([parts[role] for role in _QKV_ORDER], axis=0)
        backbone_weights[f"blocks.{layer_idx}.attn.qkv.{param_type}"] = fused

    # Split self_attn in_proj_weight/bias into query/key/value projections
    for key, arr in decoder_in_proj.items():
        # Extract layer index from key like "decoder.decoder.layers.0.self_attn.in_proj_weight"
        layer_match = re.search(r"layers\.(\d+)\.self_attn\.(in_proj_\w+)", key)
        if layer_match:
            layer_idx = layer_match.group(1)
            param_type = layer_match.group(2)
            d = arr.shape[0] // 3

            if param_type == "in_proj_weight":
                names = ["query_proj.weight", "key_proj.weight", "value_proj.weight"]
            else:
                names = ["query_proj.bias", "key_proj.bias", "value_proj.bias"]

            for i, name in enumerate(names):
                mlx_key = f"decoder.layers.{layer_idx}.self_attn.{name}"
                decoder_weights[mlx_key] = arr[i * d : (i + 1) * d]

    for group, keys in dropped.items():
        if keys:
            logger.info(f"convert_state_dict: dropped {len(keys)} unmapped {group} key(s)")
            logger.debug(f"convert_state_dict: dropped {group} keys: {_format_keys(sorted(keys))}")

    return backbone_weights, decoder_weights


# ============================================================
# Segmentation head weight conversion
# ============================================================


def _remap_seg_head_key(bare_key: str) -> str:
    """Remap a PyTorch segmentation head subkey to its MLX subkey.

    Args:
        bare_key: Key with the ``segmentation_head.`` prefix already stripped.

    Returns:
        Remapped key for loading into the MLX ``SegHead`` module.
    """
    bare_key = re.sub(r"^(blocks\.\d+\.)pwconv1\.", r"\1pwconv.", bare_key)
    bare_key = re.sub(r"^query_features_block\.layers\.0\.", "query_features_block.fc1.", bare_key)
    bare_key = re.sub(r"^query_features_block\.layers\.2\.", "query_features_block.fc2.", bare_key)
    return bare_key


def convert_seg_weights(
    state_dict: Dict[str, "torch.Tensor"],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Extract and convert segmentation head weights from a PyTorch state dict.

    Scans for all ``segmentation_head.*`` keys, transposes any conv weights,
    and remaps PyTorch naming to MLX naming.

    Args:
        state_dict: PyTorch model state dict (from ``model.model.state_dict()``).

    Returns:
        Tuple of (seg_weights dict with bare MLX keys as numpy arrays, num_blocks).

    Raises:
        ValueError: If no ``blocks.<i>`` keys were found, which means the scan or the remapping
            above it failed rather than that the head genuinely has no refinement blocks.
    """
    seg_weights: Dict[str, np.ndarray] = {}
    block_indices: set = set()

    for key, tensor in state_dict.items():
        if not key.startswith("segmentation_head."):
            continue
        arr = tensor.detach().cpu().float().numpy()
        bare = key[len("segmentation_head.") :]
        if "blocks." in bare:
            block_indices.add(int(bare.split("blocks.")[1].split(".")[0]))
        mlx_bare = _remap_seg_head_key(bare)
        if arr.ndim == 4:
            arr = _transpose_conv_weight(arr)
        seg_weights[mlx_bare] = arr

    if not seg_weights:
        logger.warning(
            "No segmentation_head.* keys found in the state dict; the MLX segmentation head has no weights to load."
        )
        num_blocks = 4
    elif not block_indices:
        raise ValueError(
            f"convert_seg_weights found no 'blocks.<i>' keys among {len(seg_weights)} segmentation_head.* "
            "entries — segmentation head weight conversion failed"
        )
    else:
        num_blocks = max(block_indices) + 1
    return seg_weights, num_blocks


# ============================================================
# Post-load audit
# ============================================================


def _format_keys(keys: list, limit: int = 10) -> str:
    """Render a bounded, comma-separated preview of parameter keys.

    Args:
        keys: Sorted parameter paths to display.
        limit: Maximum number of keys to spell out before summarising the remainder.

    Returns:
        Comma-separated keys, with a trailing count when the list was truncated.
    """
    preview = ", ".join(keys[:limit])
    return preview if len(keys) <= limit else f"{preview}, ... (+{len(keys) - limit} more)"


def audit_loaded_weights(module: nn.Module, weights: Dict[str, np.ndarray], module_name: str) -> None:
    """Verify that a ``strict=False`` weight load actually populated every parameter.

    ``load_weights(..., strict=False)`` silently tolerates both directions of mismatch. A
    parameter left unmatched keeps its random initialisation and produces confidently wrong
    predictions, so it raises; a supplied key matching no parameter only wastes data, so it warns.

    Args:
        module: MLX module the weights were loaded into.
        weights: The same mapping that was handed to ``load_weights``.
        module_name: Human-readable module name used in the messages.

    Raises:
        RuntimeError: If any module parameter received no corresponding weight.
    """
    param_keys = {key for key, _ in mlx.utils.tree_flatten(module.parameters())}
    supplied_keys = set(weights)

    missing = sorted(param_keys - supplied_keys)
    if missing:
        raise RuntimeError(
            f"{module_name}: {len(missing)} of {len(param_keys)} parameters received no weights and would keep "
            f"their random initialisation: {_format_keys(missing)}"
        )

    extra = sorted(supplied_keys - param_keys)
    if extra:
        logger.warning(
            f"{module_name}: {len(extra)} supplied weight(s) matched no parameter and were dropped: "
            f"{_format_keys(extra)}"
        )
