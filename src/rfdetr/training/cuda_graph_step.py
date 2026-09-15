# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-signature CUDA graph capture for RF-DETR's training forward."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
from torch import Tensor, nn

from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()

_GraphedCallable = Callable[[Tensor, Tensor], dict[str, Any]]
_ExecutionKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype,
    torch.dtype,
    torch.device,
    bool,
    bool,
    bool,
    torch.dtype | None,
]


def _cuda_autocast_enabled() -> bool:
    """Return CUDA autocast state across supported PyTorch versions."""
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch 2.2 accepts no device argument.
        return torch.is_autocast_enabled()


def _cuda_autocast_dtype() -> torch.dtype:
    """Return CUDA autocast dtype across supported PyTorch versions."""
    get_dtype = getattr(torch, "get_autocast_dtype", None)
    return get_dtype("cuda") if get_dtype is not None else torch.get_autocast_gpu_dtype()


class _GraphableForward(nn.Module):
    """Adapt ``LWDETR`` inputs to the Tensor-only CUDA graph API.

    ``LWDETR.forward`` does not read ``targets``; the variable-length targets remain
    outside capture and enter the unchanged eager criterion after this call.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, tensors: Tensor, mask: Tensor) -> dict[str, Any]:
        """Forward Tensor inputs through the wrapped detector."""
        return cast(dict[str, Any], self.inner(NestedTensor(tensors, mask), None))


class CudaGraphTrainingRunner:
    """Capture and replay a training forward for each static execution signature.

    This deliberately is not an :class:`~torch.nn.Module`: the original model remains
    registered on the Lightning module, so checkpoint keys, optimizers, and EMA keep the
    same parameter ownership and names. Capture failures are memoized per signature and
    fall back to eager execution.

    Args:
        inner: Detection model to execute.
        num_warmup_iters: Warmup iterations used by ``make_graphed_callables``.
    """

    def __init__(self, inner: nn.Module, num_warmup_iters: int = 3) -> None:
        self.inner = inner
        self.num_warmup_iters = num_warmup_iters
        self._graphed_cache: dict[_ExecutionKey, _GraphedCallable] = {}
        self._unsupported_keys: set[_ExecutionKey] = set()

        transformer = getattr(inner, "transformer", None)
        enable_capture = getattr(transformer, "enable_cuda_graph_capture", None)
        if callable(enable_capture):
            enable_capture()

    def __call__(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Any]:
        """Use a captured graph when available, otherwise capture or run eagerly."""
        tensors, mask = samples.decompose()
        if not self.inner.training or mask is None:
            return cast(dict[str, Any], self.inner(samples, targets))

        autocast_enabled = _cuda_autocast_enabled()
        key: _ExecutionKey = (
            tuple(tensors.shape),
            tuple(mask.shape),
            tensors.dtype,
            mask.dtype,
            tensors.device,
            tensors.requires_grad,
            mask.requires_grad,
            autocast_enabled,
            _cuda_autocast_dtype() if autocast_enabled else None,
        )
        if key in self._unsupported_keys:
            return cast(dict[str, Any], self.inner(samples, targets))

        graphed = self._graphed_cache.get(key)
        if graphed is None:
            graphed = self._try_capture(tensors, mask, key)
            if graphed is None:
                self._unsupported_keys.add(key)
                return cast(dict[str, Any], self.inner(samples, targets))
        return graphed(tensors, mask)

    def _try_capture(self, tensors: Tensor, mask: Tensor, key: _ExecutionKey) -> _GraphedCallable | None:
        """Capture one signature without modifying live accumulated gradients."""
        graphable = _GraphableForward(self.inner)
        autocast_cache_enabled = torch.is_autocast_cache_enabled()
        disable_autocast_cache = _cuda_autocast_enabled() and autocast_cache_enabled
        try:
            if disable_autocast_cache:
                # make_graphed_callables rejects autocast's weight cache because its
                # pointer lifetime is incompatible with capture. Preserve caller state.
                torch.set_autocast_cache_enabled(False)
            graphed = cast(
                _GraphedCallable,
                torch.cuda.make_graphed_callables(
                    graphable,
                    (tensors, mask),
                    num_warmup_iters=self.num_warmup_iters,
                    allow_unused_input=True,
                ),
            )
        except Exception:
            logger.warning(
                "CUDA graph capture failed for execution signature %s; falling back to eager for this signature.",
                key,
                exc_info=True,
            )
            return None
        finally:
            if disable_autocast_cache:
                torch.set_autocast_cache_enabled(autocast_cache_enabled)

        self._graphed_cache[key] = graphed
        return graphed
