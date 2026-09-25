# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Parity tests for the combined CUDA training update."""

from __future__ import annotations

import copy
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.fused_adamw_ema import FusedAdamWEMA
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor


class _CompiledModelHolder(nn.Module):
    """Mirror ``RFDETRModelModule.model._orig_mod`` without compiling a test graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model._orig_mod = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model.

        Args:
            inputs: Model input.

        Returns:
            The wrapped model's output.

        Examples:
            >>> _CompiledModelHolder(nn.Identity())(torch.ones(2)).tolist()
            [1.0, 1.0]
        """
        return self.model._orig_mod(inputs)


class _AlternatingHeads(nn.Module):
    """Two heads that receive gradients on alternating steps, like a branch some batches never reach."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor, use_first: bool) -> torch.Tensor:
        """Run one head, leaving the other without a gradient.

        Args:
            inputs: Model input.
            use_first: Whether the first or the second head runs.

        Returns:
            The selected head's output.

        Examples:
            >>> heads = _AlternatingHeads()
            >>> inputs = torch.ones(1, 4)
            >>> torch.equal(heads(inputs, use_first=True), heads.first(inputs))
            True
            >>> torch.equal(heads(inputs, use_first=False), heads.second(inputs))
            True
        """
        return (self.first if use_first else self.second)(inputs)


class _TwoBatches(Dataset[tuple[NestedTensor, list[dict[str, torch.Tensor]]]]):
    """Two fixed single-image detection batches at the model's training resolution."""

    def __init__(self, resolution: int) -> None:
        self.resolution = resolution

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[NestedTensor, list[dict[str, torch.Tensor]]]:
        generator = torch.Generator().manual_seed(index)
        samples = NestedTensor(
            torch.randn(1, 3, self.resolution, self.resolution, generator=generator),
            torch.zeros(1, self.resolution, self.resolution, dtype=torch.bool),
        )
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2 + 0.05 * index, 0.2]]), "labels": torch.tensor([1])}]
        return samples, targets


def _optimizer(model: nn.Module, average_model: AveragedModel) -> FusedAdamWEMA:
    """Build the combined optimizer for ``model`` with ``average_model`` attached as its EMA copy.

    Examples:
        >>> model = nn.Linear(2, 1)
        >>> optimizer = _optimizer(model, AveragedModel(model))
        >>> optimizer.max_grad_norm, optimizer.ema_update_step
        (0.1, 0)
    """
    optimizer = FusedAdamWEMA(
        model.parameters(),
        named_parameters=dict(model.named_parameters()),
        model_buffers=dict(model.named_buffers()),
        max_grad_norm=0.1,
        ema_decay=0.993,
        ema_tau=100,
        lr=1e-3,
        weight_decay=0.01,
    )
    optimizer.attach_ema_model(average_model)
    return optimizer


def _fail_on_fallback(self: FusedAdamWEMA) -> None:
    """Stand in for ``FusedAdamWEMA._standard_step`` so a fallback step fails the test instead of passing silently.

    Args:
        self: The optimizer whose step fell back.

    Examples:
        >>> _fail_on_fallback(None)
        Traceback (most recent call last):
            ...
        AssertionError: the combined kernels must take every optimizer step of this test
    """
    raise AssertionError("the combined kernels must take every optimizer step of this test")


def _build_arm(
    model_config: RFDETRNanoConfig, train_config: TrainConfig, combined: bool, state: dict[str, Any] | None
) -> tuple[RFDETRModelModule, SimpleNamespace, torch.optim.Optimizer, RFDETREMACallback, list[torch.Tensor]]:
    """Build one arm of the production comparison: module, trainer stand-in, optimizer, EMA callback, parameters.

    Args:
        model_config: Model configuration shared by both arms.
        train_config: Training configuration shared by both arms.
        combined: Whether the arm may select the combined optimizer.
        state: Module weights to start from, or ``None`` for a fresh initialization.

    Returns:
        The CUDA module, its trainer stand-in, the optimizer ``configure_optimizers`` built, the EMA callback attached
        to it, and the optimizer's parameters.

    Examples:
        A CUDA device is required, so this example is skipped where none is available.
        >>> model_config = RFDETRNanoConfig(num_classes=3, pretrain_weights=None)
        >>> _build_arm(model_config, TrainConfig(dataset_dir="."), True, None)  # doctest: +SKIP
    """
    module = RFDETRModelModule(model_config, train_config).cuda().train()
    if state is not None:
        module.load_state_dict(state)
    module._compile_active = combined
    trainer = SimpleNamespace(estimated_stepping_batches=10, precision="bf16-mixed", optimizers=[], global_step=0)
    module._trainer = trainer
    optimizer = module.configure_optimizers()["optimizer"]
    trainer.optimizers = [optimizer]
    callback = RFDETREMACallback(
        decay=train_config.ema_decay,
        tau=train_config.ema_tau,
        update_interval_steps=train_config.ema_update_interval,
    )
    callback.on_fit_start(trainer, module)
    return module, trainer, optimizer, callback, [p for group in optimizer.param_groups for p in group["params"]]


def _fit(
    model_config: RFDETRNanoConfig, train_config: TrainConfig, max_epochs: int, resume_from: Path | None
) -> tuple[pl.Trainer, RFDETREMACallback]:
    """Fit the production module with a real Lightning ``Trainer``, optionally resuming a checkpoint.

    Two micro-batches per epoch under ``accumulate_grad_batches=2`` are one optimizer step per epoch. The trainer's
    clipping value differs from ``TrainConfig.clip_max_norm`` so the runtime-owned value is observable.

    Args:
        model_config: Model configuration.
        train_config: Training configuration.
        max_epochs: Epoch count the fit runs to, counting epochs already in the checkpoint.
        resume_from: Checkpoint to resume, or ``None`` for a fresh fit.

    Returns:
        The fitted trainer and the EMA callback it ran.

    Examples:
        A CUDA device is required, so this example is skipped where none is available.
        >>> model_config = RFDETRNanoConfig(num_classes=3, pretrain_weights=None)
        >>> _fit(model_config, TrainConfig(dataset_dir="."), 1, None)  # doctest: +SKIP
    """
    callback = RFDETREMACallback()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        accumulate_grad_batches=2,
        gradient_clip_val=0.3,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    loader = DataLoader(_TwoBatches(model_config.resolution), batch_size=None)
    trainer.fit(RFDETRModelModule(model_config, train_config), train_dataloaders=loader, ckpt_path=resume_from)
    return trainer, callback


class _BlockEdges(nn.Module):
    """Parameters straddling the kernel's block size: ragged tail, exact multiple, and one element either side."""

    def __init__(self) -> None:
        super().__init__()
        block = FusedAdamWEMA._BLOCK
        self.matrix = nn.Parameter(torch.randn(77, 64))  # 4928 elements: two full blocks and a ragged tail
        self.exact = nn.Parameter(torch.randn(2 * block))
        self.one_past = nn.Parameter(torch.randn(block + 1))
        self.one_short = nn.Parameter(torch.randn(block - 1))
        self.small = nn.Parameter(torch.randn(5))


#: Hyperparameters that differ in every field, so a group whose options are read from another group is observable.
_GROUP_OPTIONS = [
    {"lr": 1e-2, "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8},
    {"lr": 3e-3, "weight_decay": 0.05, "betas": (0.8, 0.99), "eps": 5e-4},
    {"lr": 1e-4, "weight_decay": 0.2, "betas": (0.95, 0.9995), "eps": 1e-6},
]


def _split_into_groups(model: nn.Module, options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deal ``model``'s parameters round-robin into one parameter group per entry of ``options``.

    Examples:
        >>> groups = _split_into_groups(nn.Linear(2, 2), [{"lr": 1.0}, {"lr": 2.0}])
        >>> [(group["lr"], len(group["params"])) for group in groups]
        [(1.0, 1), (2.0, 1)]
    """
    parameters = list(model.parameters())
    return [{**option, "params": parameters[index :: len(options)]} for index, option in enumerate(options)]


def _assert_matches_torch_update(model: nn.Module, options: list[dict[str, Any]], steps: int = 8) -> None:
    """Step the combined optimizer and torch's clip, fused AdamW, and EMA sequence on the same gradients and compare.

    The gradients come from a fixed generator and are large enough that clipping is active on every step, and the
    learning rates are halved midway so the cached per-group options are refreshed.

    Examples:
        Needs a CUDA device for the combined kernels.
        >>> _assert_matches_torch_update(nn.Linear(2, 2), [{"lr": 1e-2}])  # doctest: +SKIP
    """
    baseline = model.cuda()
    candidate = copy.deepcopy(baseline)
    callback = RFDETREMACallback(decay=0.9, tau=0)
    baseline_ema = AveragedModel(
        baseline, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    candidate_ema = AveragedModel(
        candidate, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    baseline_optimizer = torch.optim.AdamW(_split_into_groups(baseline, options), fused=True)
    candidate_optimizer = FusedAdamWEMA(
        _split_into_groups(candidate, options),
        named_parameters=dict(candidate.named_parameters()),
        model_buffers=dict(candidate.named_buffers()),
        max_grad_norm=0.1,
        ema_decay=0.9,
        ema_tau=0,
    )
    candidate_optimizer.attach_ema_model(candidate_ema)

    generator = torch.Generator(device="cuda").manual_seed(2024)
    for step in range(steps):
        for baseline_parameter, candidate_parameter in zip(baseline.parameters(), candidate.parameters(), strict=True):
            gradient = torch.randn(baseline_parameter.shape, device="cuda", generator=generator)
            baseline_parameter.grad = gradient
            candidate_parameter.grad = gradient.clone()
        assert float(torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.1)) > 0.1, "clipping must be active"
        baseline_optimizer.step()
        baseline_ema.update_parameters(baseline)
        candidate_optimizer.step()
        if step == steps // 2:
            for optimizer in (baseline_optimizer, candidate_optimizer):
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5

    torch.cuda.synchronize()
    assert int(candidate_ema.n_averaged) == steps
    for baseline_tensor, candidate_tensor in zip(
        [*baseline.parameters(), *baseline_ema.parameters()],
        [*candidate.parameters(), *candidate_ema.parameters()],
        strict=True,
    ):
        torch.testing.assert_close(candidate_tensor, baseline_tensor, atol=2e-6, rtol=2e-6)


def test_import_does_not_eagerly_import_triton_kernel_module() -> None:
    """CPU-only imports must not load the CUDA kernel implementation."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rfdetr.training.fused_adamw_ema; "
            "assert 'rfdetr.training._fused_adamw_ema_triton' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_rejects_adamw_modes_not_implemented_by_combined_kernel() -> None:
    """Opt-in AdamW variants must fail instead of silently changing their arithmetic."""
    model = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="amsgrad"):
        FusedAdamWEMA(
            model.parameters(),
            named_parameters=dict(model.named_parameters()),
            model_buffers={},
            max_grad_norm=0.1,
            ema_decay=0.993,
            ema_tau=100,
            amsgrad=True,
        )


def test_rejects_a_second_ema_owner() -> None:
    """Two EMA callbacks cannot silently redirect one optimizer's fused writes."""
    model = nn.Linear(2, 1)
    first = AveragedModel(model)
    second = AveragedModel(model)
    optimizer = _optimizer(model, first)

    with pytest.raises(RuntimeError, match="exactly one RFDETREMACallback"):
        optimizer.attach_ema_model(second)


def test_cpu_step_falls_back_to_standard_adamw() -> None:
    """CPU compatibility retains AdamW and leaves EMA for the callback-owned fallback."""
    model = nn.Linear(2, 1)
    original = model.weight.detach().clone()
    average_model = AveragedModel(model)
    optimizer = _optimizer(model, average_model)

    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    assert not torch.equal(model.weight, original)
    assert optimizer.ema_update_step == 0
    assert int(average_model.n_averaged) == 0


def test_checkpoint_marks_groups_fused_like_the_standard_route_without_changing_the_live_groups() -> None:
    """The checkpoint carries the ``fused`` option the standard route writes; the live groups stay unfused."""
    model = nn.Linear(2, 1)
    optimizer = _optimizer(model, AveragedModel(model))

    saved = optimizer.state_dict()

    assert [group["fused"] for group in saved["param_groups"]] == [True]
    assert [group["fused"] for group in optimizer.param_groups] == [False]


@pytest.mark.parametrize("saved_fused", [True, False])
def test_resumed_groups_stay_unfused_whichever_route_wrote_the_checkpoint(saved_fused: bool) -> None:
    """A fallback step after a resume keeps the unfused update, even from a checkpoint saved with ``fused=True``."""
    model = nn.Linear(2, 1)
    optimizer = _optimizer(model, AveragedModel(model))
    saved = optimizer.state_dict()
    for group in saved["param_groups"]:
        group["fused"] = saved_fused

    optimizer.load_state_dict(saved)

    assert [group["fused"] for group in optimizer.param_groups] == [False]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_combined_checkpoint_resumes_on_the_standard_route_with_fused_adamw() -> None:
    """A checkpoint written by the combined route keeps the standard route's fused AdamW on when resumed there."""
    model = nn.Linear(4, 2).cuda()
    callback = RFDETREMACallback()
    average_model = AveragedModel(
        model, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    optimizer = _optimizer(model, average_model)
    model(torch.randn(3, 4, device="cuda")).square().mean().backward()
    optimizer.step()
    assert optimizer.fused_ema_applied, "the checkpoint must come from a combined-kernel step"
    checkpoint = io.BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)

    resumed_model = copy.deepcopy(model)
    standard = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3, weight_decay=0.01, fused=True)
    standard.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    resumed_model(torch.randn(3, 4, device="cuda")).square().mean().backward()
    standard.step()

    assert [group["fused"] for group in standard.param_groups] == [True]
    assert {int(state["step"].item()) for state in standard.state.values()} == {2}


@pytest.mark.parametrize(
    "attach_ema",
    [pytest.param(True, id="unsupported-layout"), pytest.param(False, id="no-ema-attached")],
)
def test_fallback_step_keeps_gradient_clipping(attach_ema: bool) -> None:
    """A fallback step clips with the optimizer's norm, as the existing clip-then-AdamW path does."""
    torch.manual_seed(3)
    baseline = nn.Linear(6, 4)
    candidate = copy.deepcopy(baseline)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-2, weight_decay=0.01)
    candidate_optimizer = FusedAdamWEMA(
        candidate.parameters(),
        named_parameters=dict(candidate.named_parameters()),
        model_buffers=dict(candidate.named_buffers()),
        max_grad_norm=0.5,
        ema_decay=0.99,
        ema_tau=0,
        lr=1e-2,
        weight_decay=0.01,
    )
    if attach_ema:
        candidate_optimizer.attach_ema_model(AveragedModel(candidate))

    pre_clip_norms = []
    for scale in (0.1, 1.0, 50.0, 2.0):
        inputs = torch.randn(8, 6) * scale
        baseline(inputs).square().mean().backward()
        candidate(inputs).square().mean().backward()
        pre_clip_norms.append(float(torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.5)))
        baseline_optimizer.step()
        candidate_optimizer.step()
        baseline_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)

    # Adam is invariant to a uniform gradient rescale, so only a mix of clipped and unclipped steps can tell the
    # configured clip norm from a wrong one.
    assert min(pre_clip_norms) < 0.5 < max(pre_clip_norms), "both clipped and unclipped steps must occur"
    for baseline_parameter, candidate_parameter in zip(baseline.parameters(), candidate.parameters(), strict=True):
        assert torch.allclose(baseline_parameter, candidate_parameter, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fallback_step_matches_the_standard_clip_then_fused_adamw_route() -> None:
    """A fallback step gives the update of the route it replaces, ``clip_grad_norm_`` then ``AdamW(fused=True)``.

    A ``channels_last`` weight is a layout fused AdamW accepts and the combined kernels decline, so both routes can take
    every step. They differ only in float32 rounding between the fused and the unfused implementation, which stays
    orders of magnitude below the ``lr`` of a single step.
    """
    torch.manual_seed(5)
    baseline = nn.Conv2d(3, 8, 3).cuda().to(memory_format=torch.channels_last)
    candidate = copy.deepcopy(baseline)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-4, weight_decay=1e-4, fused=True)
    candidate_optimizer = FusedAdamWEMA(
        candidate.parameters(),
        named_parameters=dict(candidate.named_parameters()),
        model_buffers=dict(candidate.named_buffers()),
        max_grad_norm=0.1,
        ema_decay=0.99,
        ema_tau=0,
        lr=1e-4,
        weight_decay=1e-4,
    )
    candidate_optimizer.attach_ema_model(AveragedModel(candidate, device=torch.device("cuda"), use_buffers=True))
    assert not candidate.weight.is_contiguous(), "the layout must be one the combined kernels decline"

    pre_clip_norms = []
    for scale in (0.1, 1.0, 50.0, 2.0) * 3:
        inputs = (torch.randn(4, 3, 6, 6, device="cuda") * scale).contiguous(memory_format=torch.channels_last)
        baseline(inputs).square().mean().backward()
        candidate(inputs).square().mean().backward()
        pre_clip_norms.append(float(torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.1)))
        baseline_optimizer.step()
        candidate_optimizer.step()
        assert candidate_optimizer.fused_ema_applied is False, "every step of this test must take the fallback"
        baseline_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)

    # Adam is invariant to a uniform gradient rescale, so only a mix of clipped and unclipped steps can tell the
    # configured clip norm from a wrong one.
    assert min(pre_clip_norms) < 0.1 < max(pre_clip_norms), "both clipped and unclipped steps must occur"
    for baseline_parameter, candidate_parameter in zip(baseline.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(candidate_parameter, baseline_parameter, atol=1e-5, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_first_step_fallback_leaves_state_usable_by_combined_step() -> None:
    """A fallback before any combined step must not leave host ``step`` tensors for the GPU kernel to read."""
    model = nn.Linear(6, 4).cuda()
    average_model = AveragedModel(model, device=torch.device("cuda"), use_buffers=True)
    optimizer = _optimizer(model, average_model)

    model(torch.randn(8, 6, device="cuda")).square().mean().backward()
    model.weight.grad = model.weight.grad.t().contiguous().t()  # non-contiguous gradient: unsupported layout
    optimizer.step()

    assert {state["step"].device.type for state in optimizer.state.values()} == {"cuda"}
    assert optimizer.fused_ema_applied is False
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(8, 6, device="cuda")).square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert optimizer.fused_ema_applied is True
    assert optimizer.ema_update_step == 1
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {2}
    assert torch.isfinite(model.weight).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matches_baseline_when_parameters_have_no_gradient_on_some_steps() -> None:
    """Parameters without a gradient keep their AdamW state but still advance their EMA copy."""
    torch.manual_seed(11)
    baseline = _AlternatingHeads().cuda()
    candidate = copy.deepcopy(baseline)
    callback = RFDETREMACallback(decay=0.9, tau=0)
    baseline_ema = AveragedModel(
        baseline, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    candidate_ema = AveragedModel(
        candidate, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-2, weight_decay=0.01, fused=True)
    candidate_optimizer = FusedAdamWEMA(
        candidate.parameters(),
        named_parameters=dict(candidate.named_parameters()),
        model_buffers=dict(candidate.named_buffers()),
        max_grad_norm=0.1,
        ema_decay=0.9,
        ema_tau=0,
        lr=1e-2,
        weight_decay=0.01,
    )
    candidate_optimizer.attach_ema_model(candidate_ema)

    for step in range(8):
        inputs = torch.randn(4, 4, device="cuda")
        use_first = step % 2 == 0
        baseline(inputs, use_first).square().mean().backward()
        candidate(inputs, use_first).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.1)
        baseline_optimizer.step()
        baseline_ema.update_parameters(baseline)
        candidate_optimizer.step()
        baseline_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    assert int(candidate_ema.n_averaged) == 8
    assert {int(state["step"].item()) for state in candidate_optimizer.state.values()} == {4}
    for baseline_parameter, candidate_parameter in zip(baseline.parameters(), candidate.parameters(), strict=True):
        assert torch.allclose(baseline_parameter, candidate_parameter, atol=2e-6, rtol=2e-6)
    for baseline_parameter, candidate_parameter in zip(
        baseline_ema.parameters(), candidate_ema.parameters(), strict=True
    ):
        assert torch.allclose(baseline_parameter, candidate_parameter, atol=2e-6, rtol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matches_clipped_fused_adamw_and_ema_across_steps() -> None:
    """The combined kernel stays close to the current three-operation CUDA path."""
    torch.manual_seed(7642)
    baseline = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 3)).cuda()
    candidate = copy.deepcopy(baseline)
    callback = RFDETREMACallback()
    baseline_ema = AveragedModel(
        baseline, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    candidate_ema = AveragedModel(
        candidate, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=0.01, fused=True)
    candidate_optimizer = _optimizer(candidate, candidate_ema)

    for step in range(12):
        inputs = torch.randn(4, 8, device="cuda")
        baseline(inputs).square().mean().backward()
        candidate(inputs).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.1)
        baseline_optimizer.step()
        baseline_ema.update_parameters(baseline)
        candidate_optimizer.step()
        baseline_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        if step == 5:
            for group in baseline_optimizer.param_groups:
                group["lr"] = 2e-4
            for group in candidate_optimizer.param_groups:
                group["lr"] = 2e-4

    torch.cuda.synchronize()
    assert int(candidate_ema.n_averaged) == 12
    for baseline_parameter, candidate_parameter in zip(baseline.parameters(), candidate.parameters(), strict=True):
        assert torch.allclose(baseline_parameter, candidate_parameter, atol=2e-6, rtol=2e-6)
    for baseline_parameter, candidate_parameter in zip(
        baseline_ema.parameters(), candidate_ema.parameters(), strict=True
    ):
        assert torch.allclose(baseline_parameter, candidate_parameter, atol=2e-6, rtol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matches_baseline_with_distinct_hyperparameters_per_group() -> None:
    """Each parameter group keeps its own learning rate, weight decay, betas, and epsilon, as production groups do."""
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 8))

    _assert_matches_torch_update(model, _GROUP_OPTIONS)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "options",
    [pytest.param(_GROUP_OPTIONS[:1], id="one-group"), pytest.param(_GROUP_OPTIONS, id="three-groups")],
)
def test_matches_baseline_for_tensors_spanning_several_blocks(options: list[dict[str, Any]]) -> None:
    """Tensors larger than one kernel block, with ragged, exact, and off-by-one tails, are updated end to end.

    With several groups the later blocks of a tensor must still read that tensor's group, which a model whose tensors
    each fit one block cannot show.
    """
    torch.manual_seed(6)
    model = _BlockEdges()
    assert max(parameter.numel() for parameter in model.parameters()) > 2 * FusedAdamWEMA._BLOCK

    _assert_matches_torch_update(model, options)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_updates_floating_and_integer_ema_buffers() -> None:
    """EMA buffer semantics are retained alongside fused parameter updates."""
    baseline = nn.BatchNorm1d(4).cuda()
    candidate = copy.deepcopy(baseline)
    callback = RFDETREMACallback(decay=0.5, tau=0)
    baseline_ema = AveragedModel(
        baseline, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    candidate_ema = AveragedModel(
        candidate, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3, fused=True)
    candidate_optimizer = FusedAdamWEMA(
        candidate.parameters(),
        named_parameters=dict(candidate.named_parameters()),
        model_buffers=dict(candidate.named_buffers()),
        max_grad_norm=0.1,
        ema_decay=0.5,
        ema_tau=0,
        lr=1e-3,
    )
    candidate_optimizer.attach_ema_model(candidate_ema)

    for _ in range(3):
        inputs = torch.randn(8, 4, device="cuda")
        baseline(inputs).square().mean().backward()
        candidate(inputs).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 0.1)
        baseline_optimizer.step()
        baseline_ema.update_parameters(baseline)
        candidate_optimizer.step()
        baseline_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)

    for baseline_buffer, candidate_buffer in zip(
        baseline_ema.module.buffers(), candidate_ema.module.buffers(), strict=True
    ):
        assert torch.allclose(baseline_buffer, candidate_buffer, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_maps_parameters_through_compiled_model_wrapper() -> None:
    """EMA parameter lookup unwraps the ``OptimizedModule`` shape used by compiled training."""
    inner = nn.Linear(4, 2).cuda()
    holder = _CompiledModelHolder(inner).cuda()
    callback = RFDETREMACallback()
    average_holder = AveragedModel(
        holder, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    optimizer = _optimizer(inner, average_holder)

    holder(torch.randn(3, 4, device="cuda")).square().mean().backward()
    optimizer.step()

    assert optimizer.ema_update_step == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_optimizer_and_ema_resume_continue_counters() -> None:
    """Restored Adam moments, step tensors, and EMA count remain usable by the fused pointer cache.

    The optimizer state is reloaded onto the CPU, as Lightning reads a checkpoint: ``Optimizer.load_state_dict`` then
    leaves ``step`` on the host for a non-``fused`` group, and the kernel reads it through a device pointer.
    """
    model = nn.Linear(4, 2).cuda()
    callback = RFDETREMACallback()
    average_model = AveragedModel(
        model, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    optimizer = _optimizer(model, average_model)
    model(torch.randn(3, 4, device="cuda")).square().mean().backward()
    optimizer.step()

    resumed_model = copy.deepcopy(model)
    resumed_average = AveragedModel(
        resumed_model, device=torch.device("cuda"), use_buffers=True, multi_avg_fn=callback._multi_avg_fn
    )
    resumed_average.load_state_dict(average_model.state_dict())
    resumed_optimizer = _optimizer(resumed_model, resumed_average)
    checkpoint = io.BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)
    resumed_optimizer.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    assert {state["step"].device.type for state in resumed_optimizer.state.values()} == {"cuda"}
    resumed_optimizer.attach_ema_model(resumed_average)
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_model(torch.randn(3, 4, device="cuda")).square().mean().backward()
    resumed_optimizer.step()

    assert resumed_optimizer.ema_update_step == 2
    assert int(resumed_average.n_averaged) == 2
    assert {int(state["step"].item()) for state in resumed_optimizer.state.values()} == {2}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_production_compiled_module_runs_combined_update(tmp_path: Path) -> None:
    """The production constructor routes a real compiled detection step through the combined optimizer."""
    model_config = RFDETRNanoConfig(
        num_classes=3,
        pretrain_weights=None,
        compile=True,
        cuda_graphs=False,
    )
    train_config = TrainConfig(
        dataset_dir=tmp_path,
        output_dir=tmp_path / "output",
        multi_scale=False,
        expanded_scales=False,
        use_ema=True,
        ema_update_interval=1,
        accelerator="gpu",
    )
    module = RFDETRModelModule(model_config, train_config).cuda().train()
    trainer = SimpleNamespace(
        estimated_stepping_batches=10,
        precision="bf16-mixed",
        optimizers=[],
        global_step=0,
    )
    module._trainer = trainer
    optimizer = module.configure_optimizers()["optimizer"]
    trainer.optimizers = [optimizer]
    callback = RFDETREMACallback()
    callback.on_fit_start(trainer, module)

    resolution = model_config.resolution
    samples = NestedTensor(
        torch.randn(1, 3, resolution, resolution, device="cuda"),
        torch.zeros(1, resolution, resolution, dtype=torch.bool, device="cuda"),
    )
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device="cuda"),
            "labels": torch.tensor([1], device="cuda"),
        }
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        outputs = module.model(samples, targets)
        loss_dict = module.criterion(outputs, targets)
        loss = torch.stack(
            [
                loss_dict[key] * module.criterion.weight_dict[key]
                for key in loss_dict
                if key in module.criterion.weight_dict
            ]
        ).sum()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert isinstance(optimizer, FusedAdamWEMA)
    assert optimizer.ema_update_step == 1
    assert callback._average_model is not None
    assert int(callback._average_model.n_averaged) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_production_parameter_groups_match_torch_update_on_identical_gradients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RF-DETR Nano's real parameter groups follow the existing clip, fused AdamW, and EMA path on identical gradients.

    Both arms run the production module and EMA callback hooks and differ only in the optimizer ``configure_optimizers``
    builds, so the comparison isolates the update arithmetic from the forward/backward noise that separates two full
    training runs.
    """
    monkeypatch.setattr(FusedAdamWEMA, "_standard_step", _fail_on_fallback)
    model_config = RFDETRNanoConfig(num_classes=3, pretrain_weights=None, compile=True, cuda_graphs=False)
    train_config = TrainConfig(
        dataset_dir=tmp_path,
        output_dir=tmp_path / "output",
        multi_scale=False,
        expanded_scales=False,
        use_ema=True,
        ema_update_interval=1,
        accelerator="gpu",
    )

    torch.manual_seed(0)
    candidate = _build_arm(model_config, train_config, True, None)
    baseline = _build_arm(model_config, train_config, False, candidate[0].state_dict())
    candidate_optimizer, baseline_optimizer = candidate[2], baseline[2]
    assert isinstance(candidate_optimizer, FusedAdamWEMA)
    assert type(baseline_optimizer) is torch.optim.AdamW
    # The comparison only discriminates when the groups differ and tensors span several kernel blocks.
    assert len({group["lr"] for group in candidate_optimizer.param_groups}) > 1
    assert max(parameter.numel() for parameter in candidate[4]) > 2 * FusedAdamWEMA._BLOCK

    steps = 6
    clip = train_config.clip_max_norm
    generator = torch.Generator(device="cuda").manual_seed(2024)
    for step in range(1, steps + 1):
        gradients = [0.02 * torch.randn(p.shape, device="cuda", generator=generator) for p in candidate[4]]
        assert float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))) > clip, "clip must be active"
        for module, trainer, optimizer, callback, parameters in (candidate, baseline):
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient.clone()
            module.clip_gradients(optimizer, gradient_clip_val=clip, gradient_clip_algorithm="norm")
            optimizer.step()
            trainer.global_step = step
            callback.on_train_batch_end(trainer, module, None, None, step - 1)
    torch.cuda.synchronize()

    assert int(candidate[3]._average_model.n_averaged) == int(baseline[3]._average_model.n_averaged) == steps
    for candidate_tensor, baseline_tensor in zip(
        [*candidate[0].parameters(), *candidate[3]._average_model.module.parameters()],
        [*baseline[0].parameters(), *baseline[3]._average_model.module.parameters()],
        strict=True,
    ):
        torch.testing.assert_close(candidate_tensor, baseline_tensor, atol=2e-6, rtol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_resume_with_accumulation_continues_combined_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real fit with gradient accumulation, then a checkpoint resume, keeps the kernels, counters, and EMA in step."""
    monkeypatch.setattr(FusedAdamWEMA, "_standard_step", _fail_on_fallback)
    model_config = RFDETRNanoConfig(num_classes=3, pretrain_weights=None, compile=True, cuda_graphs=False)
    train_config = TrainConfig(
        dataset_dir=tmp_path,
        output_dir=tmp_path / "output",
        multi_scale=False,
        expanded_scales=False,
        use_ema=True,
        ema_update_interval=1,
        accelerator="gpu",
    )
    checkpoint_path = tmp_path / "resume.ckpt"

    trainer, _ = _fit(model_config, train_config, max_epochs=1, resume_from=None)
    trainer.save_checkpoint(checkpoint_path)
    trainer, callback = _fit(model_config, train_config, max_epochs=2, resume_from=checkpoint_path)
    torch.cuda.synchronize()

    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, FusedAdamWEMA)
    assert optimizer.max_grad_norm == pytest.approx(0.3)
    # Two micro-batches per epoch under accumulation of 2 are one optimizer step per epoch.
    assert trainer.global_step == 2
    assert {int(state["step"].item()) for state in optimizer.state.values() if "step" in state} == {2}
    assert callback._average_model is not None
    assert int(callback._average_model.n_averaged) == 2
    assert optimizer.ema_update_step == 2
    assert all(torch.isfinite(parameter).all() for parameter in callback._average_model.module.parameters())
