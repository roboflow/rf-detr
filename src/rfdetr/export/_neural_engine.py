# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Graph rewrites that keep an exported RF-DETR on the Apple Neural Engine.

- :func:`split_einsum_encoder` runs the windowed DINOv2 attention as one einsum pair per head, on a channels-first
  layout, in query chunks (Apple's ``SPLIT_EINSUM_V2``). The Neural Engine runs the fused attention op of the original
  graph about 10x slower per FLOP than its linear layers.
- The Neural Engine has no ``topk`` and no ``gather``, so Core ML runs the two-stage query selection on the CPU, between
  two Neural Engine segments. :func:`one_hot_top_rows` selects the same rows with comparisons, a sum and a matmul.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput

from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersEncoder,
    WindowedDinov2WithRegistersLayer,
)
from rfdetr.models.transformer import Transformer
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# The selected ranks stay below the query count, and float16 holds every integer up to 2048 exactly.
_MAX_EXACT_FLOAT16_RANK = 2048


def one_hot_top_rows(scores: Tensor, k: int) -> Callable[[Tensor], Tensor]:
    """Rank ``(batch, tokens)`` scores like a stable descending sort, and select rows with a one-hot matmul.

    ``rank_i`` counts the scores that beat score ``i``: larger, or equal at a lower index. Row ``r`` of the one-hot
    matrix ``P`` has its 1 where ``rank_i == r``, so ``P @ rows`` equals the gather of the top ``k`` rows. Equal scores
    keep the lower index first; ``torch.topk`` leaves that order unspecified.

    The ranks of the selected rows stay below ``k``, so they are exact in float16 for ``k <= 2048``, whatever the
    token count. A row that is not selected multiplies by 0, so the selected rows are exact only when every row is
    finite: ``0 * inf`` is NaN.

    Args:
        scores: One score per token.
        k: Number of rows to keep, best first.

    Returns:
        A function from a ``(batch, tokens, C)`` tensor to its ``(batch, k, C)`` selected rows.

    Examples:
        >>> rows = torch.arange(4.0).reshape(1, 4, 1)
        >>> one_hot_top_rows(torch.tensor([[0.1, 0.9, 0.5, 0.9]]), 3)(rows).flatten().tolist()
        [1.0, 3.0, 2.0]
    """
    num_tokens = scores.shape[-1]
    score_i, score_j = scores[..., :, None], scores[..., None, :]
    lower_index = torch.ones(num_tokens, num_tokens, dtype=scores.dtype, device=scores.device).tril(-1)
    beats = (score_j > score_i).to(scores.dtype) + (score_j == score_i).to(scores.dtype) * lower_index
    rank = beats.sum(dim=-1)
    positions = torch.arange(k, dtype=scores.dtype, device=scores.device)[:, None]
    one_hot = torch.relu(1 - (rank[..., None, :] - positions).abs())  # (batch, k, tokens)
    return lambda rows: torch.matmul(one_hot.to(rows.dtype), rows)


def _conv_from_linear(linear: nn.Linear, scale: Tensor | None = None) -> nn.Conv2d:
    weight = linear.weight.detach().clone()
    bias = linear.bias.detach().clone() if linear.bias is not None else torch.zeros(linear.out_features)
    if scale is not None:
        weight, bias = weight * scale[:, None], bias * scale
    conv = nn.Conv2d(linear.in_features, linear.out_features, 1)
    conv.weight, conv.bias = nn.Parameter(weight[:, :, None, None]), nn.Parameter(bias)
    return conv


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over dim 1 of a ``(batch, C, rows, tokens)`` tensor, with the weights of an ``nn.LayerNorm``."""

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.eps = norm.eps
        self.weight = nn.Parameter(norm.weight.detach().clone()[None, :, None, None])
        self.bias = nn.Parameter(norm.bias.detach().clone()[None, :, None, None])

    def forward(self, x: Tensor) -> Tensor:
        """Normalize every token over its channels.

        Args:
            x: Activations of shape ``(batch, C, rows, tokens)``.

        Returns:
            The normalized activations, same shape.
        """
        centered = x - x.mean(dim=1, keepdim=True)
        normalized: Tensor = centered * torch.rsqrt(centered.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return normalized * self.weight + self.bias


class _SplitEinsumLayer(nn.Module):
    """One windowed DINOv2 layer on a ``(batch, C, windows, tokens)`` layout, with the weights of the original.

    The attention scale goes into the query weights and the layer scales go into the output weights.
    """

    def __init__(self, layer: WindowedDinov2WithRegistersLayer, num_heads: int, query_chunk: int) -> None:
        super().__init__()
        attention = layer.attention.attention
        self.num_heads, self.head_dim, self.query_chunk = num_heads, attention.attention_head_size, query_chunk
        self.norm1, self.norm2 = _ChannelLayerNorm(layer.norm1), _ChannelLayerNorm(layer.norm2)
        query_scale = torch.full((attention.query.out_features,), self.head_dim**-0.5)
        self.query = _conv_from_linear(attention.query, query_scale)
        self.key = _conv_from_linear(attention.key)
        self.value = _conv_from_linear(attention.value)
        self.output = _conv_from_linear(layer.attention.output.dense, layer.layer_scale1.lambda1.detach())
        self.fc1 = _conv_from_linear(cast(nn.Linear, layer.mlp.fc1))
        self.fc2 = _conv_from_linear(cast(nn.Linear, layer.mlp.fc2), layer.layer_scale2.lambda1.detach())
        self.activation = cast(Callable[[Tensor], Tensor], layer.mlp.activation)

    def _attention(self, x: Tensor) -> Tensor:
        query, value = self.query(x), self.value(x)
        key = self.key(x).transpose(1, 3)  # (batch, tokens, rows, C)
        num_tokens = x.shape[3]
        # Equal chunks: a short remainder chunk (for example 256 + 1 tokens) is slow on the Neural Engine.
        chunk = -(-num_tokens // -(-num_tokens // self.query_chunk))
        heads = []
        for head in range(self.num_heads):
            channels = slice(head * self.head_dim, (head + 1) * self.head_dim)
            head_key, head_value = key[..., channels], value[:, channels]
            outputs = []
            for start in range(0, num_tokens, chunk):
                head_query = query[:, channels, :, start : start + chunk]
                weights = torch.einsum("bchq,bkhc->bkhq", head_query, head_key).softmax(dim=1)
                outputs.append(torch.einsum("bkhq,bchk->bchq", weights, head_value))
            heads.append(torch.cat(outputs, dim=3) if len(outputs) > 1 else outputs[0])
        attended: Tensor = self.output(torch.cat(heads, dim=1))
        return attended

    def forward(self, x: Tensor, run_full_attention: bool) -> Tensor:
        """Apply the layer.

        Args:
            x: Activations of shape ``(batch, C, windows, tokens)``.
            run_full_attention: Attend over the tokens of all windows, not over each window.

        Returns:
            The layer output, same shape.
        """
        batch, channels, windows, tokens = x.shape
        normed = self.norm1(x)
        if run_full_attention:
            # The original layer merges the windows window-major, which is this row-major reshape.
            normed = normed.reshape(batch, channels, 1, windows * tokens)
        x = x + self._attention(normed).reshape(batch, channels, windows, tokens)
        output: Tensor = x + self.fc2(self.activation(self.fc1(self.norm2(x))))
        return output


class _SplitEinsumEncoder(nn.Module):
    """Drop-in replacement for :class:`WindowedDinov2WithRegistersEncoder` in inference."""

    def __init__(self, encoder: WindowedDinov2WithRegistersEncoder, query_chunk: int) -> None:
        super().__init__()
        self.config = encoder.config
        self.layer = nn.ModuleList(
            _SplitEinsumLayer(cast(WindowedDinov2WithRegistersLayer, layer), self.config.num_attention_heads, query_chunk)
            for layer in encoder.layer
        )

    def forward(
        self,
        hidden_states: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, ...]] | BaseModelOutput:
        """Run all layers, and return every hidden state in the original ``(batch * windows, tokens, C)`` layout.

        Args:
            hidden_states: Windowed embeddings of shape ``(batch * windows, tokens, C)``.
            output_hidden_states: Ignored: the hidden states are always returned, as the backbone reads them.
            output_attentions: Must be ``False``.
            return_dict: Return a ``BaseModelOutput`` instead of a tuple.

        Returns:
            The last hidden state and all hidden states, embeddings first.
        """
        assert not output_attentions, "the split-einsum encoder does not return attention weights"
        batch_windows, tokens, channels = hidden_states.shape
        windows = self.config.num_windows**2
        batch = batch_windows // windows
        x = hidden_states.reshape(batch, windows, tokens, channels).permute(0, 3, 1, 2)
        all_hidden_states: tuple[Tensor, ...] = (hidden_states,)
        for index, layer in enumerate(self.layer):
            x = layer(x, run_full_attention=index not in self.config.window_block_indexes)
            all_hidden_states += (x.permute(0, 2, 3, 1).reshape(batch_windows, tokens, channels),)
        if not return_dict:
            return all_hidden_states[-1], all_hidden_states
        return BaseModelOutput(
            last_hidden_state=cast("torch.FloatTensor", all_hidden_states[-1]),
            hidden_states=cast("tuple[torch.FloatTensor, ...]", all_hidden_states),
        )


def split_einsum_encoder(encoder: WindowedDinov2WithRegistersEncoder, query_chunk: int = 256) -> nn.Module:
    """Rebuild a windowed DINOv2 encoder for the Neural Engine, with the same weights, for inference.

    Args:
        encoder: The encoder to rebuild. It is not modified.
        query_chunk: Largest number of query tokens in one attention chunk. Chunks are equal in size.

    Returns:
        An eval-mode module with the interface of *encoder*.

    Raises:
        NotImplementedError: For an encoder with SwiGLU MLPs or stochastic depth, which the rebuild does not copy.

    Examples:
        >>> from rfdetr.models.backbone.dinov2_with_windowed_attn import (
        ...     WindowedDinov2WithRegistersBackbone,
        ...     WindowedDinov2WithRegistersConfig,
        ... )
        >>> config = WindowedDinov2WithRegistersConfig(
        ...     image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, out_indices=[2]
        ... )
        >>> encoder = WindowedDinov2WithRegistersBackbone(config).encoder.eval()
        >>> hidden_states = torch.randn(1, 5, 32)
        >>> rebuilt = split_einsum_encoder(encoder)(hidden_states).last_hidden_state
        >>> torch.allclose(rebuilt, encoder(hidden_states).last_hidden_state, atol=1e-5)
        True
    """
    if encoder.config.use_swiglu_ffn or encoder.config.drop_path_rate > 0:
        raise NotImplementedError("the split-einsum encoder copies plain-MLP layers without stochastic depth only")
    return _SplitEinsumEncoder(encoder, query_chunk).eval()


def neural_engine_model(model: nn.Module) -> nn.Module:
    """Return a copy of an export-ready RF-DETR with the Neural Engine rewrites, for tracing.

    - Every windowed DINOv2 encoder becomes a :func:`split_einsum_encoder`.
    - The two-stage query selection uses :func:`one_hot_top_rows`. A transformer keeps ``topk`` when its proposals
      can be infinite (``bbox_reparam=False``) or its query count is too large for exact float16 ranks.

    Args:
        model: The model to rewrite. It is not modified.

    Returns:
        The rewritten copy.

    Examples:
        >>> from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersBackbone
        >>> from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersConfig
        >>> config = WindowedDinov2WithRegistersConfig(
        ...     image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, out_indices=[2]
        ... )
        >>> rewritten = neural_engine_model(WindowedDinov2WithRegistersBackbone(config))
        >>> type(rewritten.encoder).__name__
        '_SplitEinsumEncoder'
    """
    model = copy.deepcopy(model)
    for module in list(model.modules()):
        encoder = getattr(module, "encoder", None)
        if isinstance(encoder, WindowedDinov2WithRegistersEncoder):
            module.encoder = split_einsum_encoder(encoder)
        if isinstance(module, Transformer) and module.two_stage:
            if module.bbox_reparam and module.num_queries <= _MAX_EXACT_FLOAT16_RANK:
                module.select_top_rows = one_hot_top_rows
            else:
                logger.warning(
                    "Keeping top-k query selection on the CPU: one-hot selection needs bbox_reparam=True and at most "
                    f"{_MAX_EXACT_FLOAT16_RANK} queries (got bbox_reparam={module.bbox_reparam}, "
                    f"num_queries={module.num_queries})."
                )
    return model
