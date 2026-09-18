# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-signature CUDA graph capture for RF-DETR's training forward."""

from __future__ import annotations

import inspect
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

    ``LWDETR.forward`` does not read ``targets``; the variable-length targets remain outside capture and enter the
    unchanged eager criterion after this call.
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
    same parameter ownership and names. Capture failures stop training because an
    invalidated capture can leave CUDA state unsafe for an eager retry.

    Args:
        inner: Detection model to execute.
        num_warmup_iters: Warmup iterations used by ``make_graphed_callables``.
        fp8_recipe: Lightning's active Transformer Engine recipe. When supplied, use
            Transformer Engine's FP8-aware capture API for one fixed execution signature.
    """

    def __init__(self, inner: nn.Module, num_warmup_iters: int = 3, *, fp8_recipe: Any | None = None) -> None:
        """Select and validate the capture backend without replacing registered parameters."""
        self.inner = inner
        self.num_warmup_iters = num_warmup_iters
        self.fp8_recipe = fp8_recipe
        self._te_capture: Callable[..., Any] | None = None
        self._graphed_cache: dict[_ExecutionKey, _GraphedCallable] = {}

        if fp8_recipe is not None:
            # Optional CUDA extension: ordinary BF16 capture must not import Transformer Engine.
            from transformer_engine.pytorch import make_graphed_callables  # type: ignore[import-not-found]

            required = {"enabled", "recipe", "cache_quantized_params", "clone_param_grads_on_return"}
            if not required <= set(inspect.signature(make_graphed_callables).parameters):
                raise RuntimeError(
                    "FP8 CUDA graphs require the Transformer Engine 2.19 capture API, including "
                    "clone_param_grads_on_return. Install a compatible Transformer Engine or set cuda_graphs=False."
                )
            self._te_capture = make_graphed_callables

        transformer = getattr(inner, "transformer", None)
        enable_capture = getattr(transformer, "enable_cuda_graph_capture", None)
        if callable(enable_capture):
            enable_capture()

    def __call__(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Any]:
        """Capture or replay training inputs; leave evaluation inputs eager.

        Raises:
            RuntimeError: If CUDA capture fails; restart the process before retrying.
        """
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
        graphed = self._graphed_cache.get(key)
        if graphed is None:
            if self.fp8_recipe is not None:
                # A second capture would mutate FP8 scaling state shared with the first graph.
                # Keep the initial integration fixed-shape until multi-signature parity is verified.
                if self._graphed_cache:
                    # Exactly one capture exists here: FP8 mode never admits a second signature.
                    cached_key = next(iter(self._graphed_cache))
                    raise RuntimeError(
                        f"FP8 CUDA graphs require one fixed execution signature; captured {cached_key}, got {key}. "
                        "Keep batch size, resolution, dtype and autocast fixed, or set cuda_graphs=False."
                    )
                if any(parameter.grad is not None for parameter in self.inner.parameters()):
                    raise RuntimeError("FP8 CUDA graph capture requires no existing parameter gradients.")
            graphed = self._try_capture(tensors, mask, key)
        if self.fp8_recipe is None:
            self._own_accumulated_gradients()
        return graphed(tensors, mask)

    def _own_accumulated_gradients(self) -> None:
        """Move live gradients out of the static buffers a previous graphed backward handed to autograd.

        ``make_graphed_callables`` returns its static gradient buffers from every backward. When ``.grad`` was ``None``,
        autograd adopts the buffer itself instead of copying it, and the next replay overwrites that buffer before
        autograd accumulates into it, turning ``g0 + g1`` into ``2 * g1``. Copying before the replay costs one gradient-
        sized copy per microbatch after the first of an accumulation window and nothing when
        ``zero_grad(set_to_none=True)`` ran in between.
        """
        for parameter in self.inner.parameters():
            if parameter.grad is not None:
                parameter.grad = parameter.grad.clone()

    def _try_capture(self, tensors: Tensor, mask: Tensor, key: _ExecutionKey) -> _GraphedCallable:
        """Capture one signature without modifying live accumulated gradients."""
        graphable = _GraphableForward(self.inner)
        autocast_cache_enabled = torch.is_autocast_cache_enabled()
        disable_autocast_cache = _cuda_autocast_enabled() and autocast_cache_enabled
        try:
            if disable_autocast_cache:
                # make_graphed_callables rejects autocast's weight cache because its
                # pointer lifetime is incompatible with capture. Preserve caller state.
                torch.set_autocast_cache_enabled(False)
            if self._te_capture is not None:
                # Transformer Engine owns scale/amax updates during capture and replay. Returned gradients
                # must own their storage; otherwise the next replay can overwrite .grad aliases.
                graphed = cast(
                    _GraphedCallable,
                    self._te_capture(
                        graphable,
                        (tensors, mask),
                        num_warmup_iters=self.num_warmup_iters,
                        allow_unused_input=True,
                        enabled=True,
                        recipe=self.fp8_recipe,
                        cache_quantized_params=False,
                        clone_param_grads_on_return=True,
                    ),
                )
            else:
                graphed = cast(
                    _GraphedCallable,
                    torch.cuda.make_graphed_callables(
                        graphable,
                        (tensors, mask),
                        num_warmup_iters=self.num_warmup_iters,
                        allow_unused_input=True,
                    ),
                )
        except Exception as exc:
            raise RuntimeError(
                f"CUDA graph capture failed for execution signature {key}. "
                "Training cannot safely continue after failed capture. Restart the process "
                "and set cuda_graphs=False to run eagerly."
            ) from exc
        finally:
            if disable_autocast_cache:
                torch.set_autocast_cache_enabled(autocast_cache_enabled)

        self._graphed_cache[key] = graphed
        tensor_shape, mask_shape, *_rest, autocast_enabled, autocast_dtype = key
        logger.info(
            "Captured CUDA graph %d for input shape %s (mask %s, autocast %s); "
            "later batches with this signature replay it. Backend: %s.",
            len(self._graphed_cache),
            tensor_shape,
            mask_shape,
            autocast_dtype if autocast_enabled else "off",
            "Transformer Engine FP8" if self.fp8_recipe is not None else "PyTorch",
        )
        return graphed
