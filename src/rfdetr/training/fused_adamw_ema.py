# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Combined CUDA update for RF-DETR's compiled BF16 training path."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, overload

import torch
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel

#: AdamW options the combined kernel does not implement; requesting one keeps the standard optimizer.
UNSUPPORTED_ADAMW_OPTIONS = ("amsgrad", "maximize", "capturable", "differentiable")

#: ``triton.language.extra.cuda.libdevice`` functions the combined kernels call; a Triton lacking any of them keeps the
#: standard optimizer.
LIBDEVICE_FUNCTIONS = ("fma", "pow", "sqrt")


class FusedAdamWEMA(torch.optim.AdamW):
    """Fuse global-norm clipping, AdamW, and EMA across RF-DETR parameters.

    Triton is imported lazily on the first CUDA step, so importing RF-DETR on a CPU-only or non-Linux installation
    retains the existing dependency surface.

    Args:
        params: Parameters or parameter groups to optimize, as for :class:`torch.optim.AdamW`.
        named_parameters: Every model parameter by name, used to find its counterpart in the EMA copy.
        model_buffers: Every model buffer by name. ``float32`` buffers are averaged by the kernel, integer and boolean
            ones with PyTorch operations.
        max_grad_norm: Global gradient-norm clip; a non-positive value disables clipping.
        ema_decay: Base EMA decay.
        ema_tau: EMA warm-up time constant in updates; ``0`` disables the warm-up.
        lr: Learning rate.
        betas: AdamW moment decay coefficients.
        eps: AdamW denominator term.
        weight_decay: Decoupled weight decay.
        **kwargs: Remaining :class:`torch.optim.AdamW` options. ``fused`` is ignored; ``amsgrad``, ``maximize``,
            ``capturable`` and ``differentiable`` are rejected.

    Raises:
        ValueError: If an AdamW option the combined kernel does not implement is requested.
    """

    _BLOCK = 2048
    _NUM_WARPS = 4
    _fuses_ema = True

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        named_parameters: Mapping[str, nn.Parameter],
        model_buffers: Mapping[str, Tensor],
        max_grad_norm: float,
        ema_decay: float,
        ema_tau: int,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        unsupported = [key for key in UNSUPPORTED_ADAMW_OPTIONS if kwargs.get(key, False)]
        if unsupported:
            options = ", ".join(sorted(unsupported))
            raise ValueError(f"Combined AdamW + EMA does not support: {options}.")
        kwargs.pop("fused", None)
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=False,
            **kwargs,
        )
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm > 0 else math.inf
        self._ema_decay = float(ema_decay)
        self._ema_tau = int(ema_tau)
        self._parameter_names: dict[Tensor, str] = {parameter: name for name, parameter in named_parameters.items()}
        self._model_buffers = dict(model_buffers)
        self._average_model: AveragedModel | None = None
        self.ema_update_step = 0
        self.fused_ema_applied = False
        self._cache: dict[str, Any] | None = None

    def attach_ema_model(self, average_model: AveragedModel) -> None:
        """Attach the callback-owned EMA copy after precision/device setup.

        Args:
            average_model: The EMA copy whose parameters and buffers the combined kernels update.

        Raises:
            RuntimeError: If a different EMA copy is already attached.
        """
        if self._average_model is not None and self._average_model is not average_model:
            raise RuntimeError("Combined AdamW + EMA supports exactly one RFDETREMACallback.")
        self._average_model = average_model
        self.ema_update_step = int(average_model.n_averaged.item())
        self.fused_ema_applied = False
        self._cache = None

    def set_max_grad_norm(self, value: float | None) -> None:
        """Refresh clipping from Lightning's runtime-owned trainer value.

        Args:
            value: Clipping norm from the trainer; ``None`` or a non-positive value disables clipping.
        """
        self.max_grad_norm = float(value) if value is not None and value > 0 else math.inf

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer state marked as fused AdamW's, as the standard route would write it.

        The combined route is only selected when fused AdamW was requested, and ``Optimizer.load_state_dict`` replaces
        a resuming optimizer's group options with the saved ones. A ``fused=False`` written by this optimizer's own
        groups would therefore switch fused AdamW off when the checkpoint is resumed on the standard route.

        Returns:
            The optimizer state with ``fused=True`` on every parameter group.
        """
        state = super().state_dict()
        for group in state["param_groups"]:
            group["fused"] = True
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore AdamW state on each parameter's device and invalidate pointer metadata.

        ``Optimizer.load_state_dict`` only relocates ``step`` for ``fused`` or ``capturable`` groups, and Lightning's
        restore leaves it where the checkpoint was loaded (the host). The kernels read ``step`` through a raw device
        pointer, so it is moved here, as ``_initialize_state`` would have created it. The saved ``fused`` option is
        discarded because this optimizer's own groups are never ``fused``: a fallback step keeps the unfused update
        whichever route wrote the checkpoint.

        Args:
            state_dict: State returned by :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group["fused"] = False
        for parameter, state in self.state.items():
            if "step" in state:
                state["step"] = state["step"].to(device=parameter.device, dtype=torch.float32)
        self._cache = None

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one combined update, falling back if its strict layout changes.

        ``fused_ema_applied`` is ``True`` afterwards only when the combined kernels also applied the EMA update, so the
        EMA callback can tell that step from a fallback step it must average itself.

        Args:
            closure: Optional closure that re-evaluates the model and returns the loss.

        Returns:
            The closure's loss, or ``None`` when no closure was given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.fused_ema_applied = False
        if self._average_model is None:
            self._standard_step()
            return loss

        all_parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if not self._supported(all_parameters):
            self._standard_step()
            return loss

        parameters = [parameter for parameter in all_parameters if parameter.grad is not None]
        inactive_parameters = [parameter for parameter in all_parameters if parameter.grad is None]
        self._initialize_state(parameters)
        cache = self._get_cache(parameters, inactive_parameters)
        gradients = [parameter.grad for parameter in parameters]
        concrete_gradients = [gradient for gradient in gradients if gradient is not None]
        gradient_addresses = tuple(gradient.data_ptr() for gradient in concrete_gradients)
        if cache.get("gradient_addresses") != gradient_addresses:
            cache["grad_ptrs"] = self._pointers(concrete_gradients, parameters[0].device)
            cache["gradient_addresses"] = gradient_addresses
        self._refresh_group_options(cache)

        step_tensors = [self.state[parameter]["step"] for parameter in parameters]
        torch._foreach_add_(step_tensors, 1)
        cache["norm_sq"].zero_()
        global_norm_kernel, fused_kernel, ema_kernel = self._kernels()
        grid = (cache["tensor_ids"].numel(),)
        global_norm_kernel[grid](
            cache["grad_ptrs"],
            cache["tensor_ids"],
            cache["block_starts"],
            cache["lengths"],
            cache["norm_sq"],
            BLOCK=self._BLOCK,
            num_warps=self._NUM_WARPS,
        )
        next_update = self.ema_update_step + 1
        decay = self._effective_decay(next_update)
        fused_kernel[grid](
            cache["param_ptrs"],
            cache["grad_ptrs"],
            cache["exp_avg_ptrs"],
            cache["exp_avg_sq_ptrs"],
            cache["step_ptrs"],
            cache["ema_ptrs"],
            cache["tensor_group_ids"],
            cache["tensor_ids"],
            cache["block_starts"],
            cache["lengths"],
            cache["group_lrs"],
            cache["group_weight_decays"],
            cache["group_beta1s"],
            cache["group_beta2s"],
            cache["group_epsilons"],
            cache["norm_sq"],
            self.max_grad_norm,
            decay,
            BLOCK=self._BLOCK,
            num_warps=self._NUM_WARPS,
        )
        if cache["buffer_tensor_ids"].numel():
            buffer_grid = (cache["buffer_tensor_ids"].numel(),)
            ema_kernel[buffer_grid](
                cache["buffer_source_ptrs"],
                cache["buffer_ema_ptrs"],
                cache["buffer_tensor_ids"],
                cache["buffer_block_starts"],
                cache["buffer_lengths"],
                decay,
                BLOCK=self._BLOCK,
                num_warps=self._NUM_WARPS,
            )
        for source, target in cache["integer_buffers"]:
            target.copy_(target * decay + source * (1.0 - decay))
        self._average_model.n_averaged.add_(1)
        self.ema_update_step = next_update
        self.fused_ema_applied = True
        return loss

    def _standard_step(self) -> None:
        """Apply clipped, non-fused AdamW to a step the combined kernels cannot take.

        The module hook leaves clipping to this optimizer, so the fallback clips here. Torch's fused AdamW would reject
        some of the layouts that land here (a gradient whose strides differ from its parameter's), so the fallback keeps
        the unfused implementation. State is created on each parameter's device first: AdamW would otherwise keep a host
        ``step`` tensor that a later combined step would hand to the GPU kernel as a raw pointer.
        """
        parameters = [
            parameter for group in self.param_groups for parameter in group["params"] if parameter.grad is not None
        ]
        self._initialize_state(parameters)
        if parameters and math.isfinite(self.max_grad_norm):
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        super().step()

    def _supported(self, parameters: list[Tensor]) -> bool:
        active_parameters = [parameter for parameter in parameters if parameter.grad is not None]
        parameters_supported = bool(active_parameters) and all(
            parameter.device.type == "cuda"
            and parameter.dtype == torch.float32
            and parameter.is_contiguous()
            and (
                parameter.grad is None
                or (
                    not parameter.grad.is_sparse
                    and parameter.grad.dtype == torch.float32
                    and parameter.grad.is_contiguous()
                )
            )
            for parameter in parameters
        )
        buffers_supported = all(
            buffer.device.type == "cuda"
            and buffer.is_contiguous()
            and (not buffer.is_floating_point() or buffer.dtype == torch.float32)
            for buffer in self._model_buffers.values()
        )
        return parameters_supported and buffers_supported

    def _initialize_state(self, parameters: list[Tensor]) -> None:
        for parameter in parameters:
            state = self.state[parameter]
            if state:
                continue
            state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

    def _get_cache(self, parameters: list[Tensor], inactive_parameters: list[Tensor]) -> dict[str, Any]:
        signature = tuple(
            (parameter.data_ptr(), self.state[parameter]["exp_avg"].data_ptr()) for parameter in parameters
        ) + tuple((parameter.data_ptr(), 0) for parameter in inactive_parameters)
        if self._cache is not None and self._cache["signature"] == signature:
            return self._cache
        if self._average_model is None:
            raise RuntimeError("EMA model must be attached before building fused metadata.")

        average_module = self._average_model.module
        average_inner = getattr(average_module, "model", average_module)
        average_inner = getattr(average_inner, "_orig_mod", average_inner)
        average_parameters = dict(average_inner.named_parameters())
        ema_parameters = [average_parameters[self._parameter_names[parameter]] for parameter in parameters]
        device = parameters[0].device
        parameter_group_ids = {
            parameter: group_id for group_id, group in enumerate(self.param_groups) for parameter in group["params"]
        }
        tensor_group_ids = []
        for parameter in parameters:
            tensor_group_ids.append(parameter_group_ids[parameter])
        tensor_ids, block_starts = self._blocks(parameters)

        floating_buffer_sources: list[Tensor] = list(inactive_parameters)
        floating_buffer_targets: list[Tensor] = [
            average_parameters[self._parameter_names[parameter]] for parameter in inactive_parameters
        ]
        integer_buffers: list[tuple[Tensor, Tensor]] = []
        average_buffers = dict(average_inner.named_buffers())
        for name, source in self._model_buffers.items():
            target = average_buffers[name]
            if source.is_floating_point() and source.dtype == torch.float32 and source.is_contiguous():
                floating_buffer_sources.append(source)
                floating_buffer_targets.append(target)
            else:
                integer_buffers.append((source, target))
        buffer_tensor_ids, buffer_block_starts = self._blocks(floating_buffer_sources)

        self._cache = {
            "signature": signature,
            "param_ptrs": self._pointers(parameters, device),
            "exp_avg_ptrs": self._pointers([self.state[p]["exp_avg"] for p in parameters], device),
            "exp_avg_sq_ptrs": self._pointers([self.state[p]["exp_avg_sq"] for p in parameters], device),
            "step_ptrs": self._pointers([self.state[p]["step"] for p in parameters], device),
            "ema_ptrs": self._pointers(ema_parameters, device),
            "tensor_group_ids": torch.tensor(tensor_group_ids, dtype=torch.int32, device=device),
            "tensor_ids": torch.tensor(tensor_ids, dtype=torch.int32, device=device),
            "block_starts": torch.tensor(block_starts, dtype=torch.int32, device=device),
            "lengths": torch.tensor([p.numel() for p in parameters], dtype=torch.int32, device=device),
            "norm_sq": torch.zeros((), device=device),
            "buffer_source_ptrs": self._pointers(floating_buffer_sources, device),
            "buffer_ema_ptrs": self._pointers(floating_buffer_targets, device),
            "buffer_tensor_ids": torch.tensor(buffer_tensor_ids, dtype=torch.int32, device=device),
            "buffer_block_starts": torch.tensor(buffer_block_starts, dtype=torch.int32, device=device),
            "buffer_lengths": torch.tensor(
                [buffer.numel() for buffer in floating_buffer_sources], dtype=torch.int32, device=device
            ),
            "integer_buffers": integer_buffers,
            "group_options": None,
        }
        self._refresh_group_options(self._cache)
        return self._cache

    def _refresh_group_options(self, cache: dict[str, Any]) -> None:
        options = tuple(
            (
                float(group["lr"]),
                float(group["weight_decay"]),
                float(group["betas"][0]),
                float(group["betas"][1]),
                float(group["eps"]),
            )
            for group in self.param_groups
        )
        if cache["group_options"] == options:
            return
        device = cache["param_ptrs"].device
        columns = list(zip(*options, strict=True))
        names = ("group_lrs", "group_weight_decays", "group_beta1s", "group_beta2s", "group_epsilons")
        for name, values in zip(names, columns, strict=True):
            value = torch.tensor(values, dtype=torch.float32, device=device)
            if name in cache:
                cache[name].copy_(value)
            else:
                cache[name] = value
        cache["group_options"] = options

    def _effective_decay(self, update: int) -> float:
        if update == 1:
            return 0.0
        if self._ema_tau > 0:
            return self._ema_decay * (1.0 - math.exp(-update / self._ema_tau))
        return self._ema_decay

    @classmethod
    def _blocks(cls, tensors: list[Tensor]) -> tuple[list[int], list[int]]:
        tensor_ids: list[int] = []
        starts: list[int] = []
        for tensor_id, tensor in enumerate(tensors):
            for start in range(0, tensor.numel(), cls._BLOCK):
                tensor_ids.append(tensor_id)
                starts.append(start)
        return tensor_ids, starts

    @staticmethod
    def _pointers(tensors: list[Tensor], device: torch.device) -> Tensor:
        return torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.uint64, device=device)

    @staticmethod
    def _kernels() -> tuple[Any, Any, Any]:
        from rfdetr.training._fused_adamw_ema_triton import (
            ema_only_kernel,
            fused_adamw_ema_kernel,
            global_norm_kernel,
        )

        return global_norm_kernel, fused_adamw_ema_kernel, ema_only_kernel
