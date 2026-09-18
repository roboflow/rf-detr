# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the opt-in CUDA-graph training forward."""

from __future__ import annotations

import copy
import multiprocessing
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner
from rfdetr.utilities.tensors import NestedTensor

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # torch < 2.3 has no public SDPA backend selector.
    sdpa_kernel = None  # type: ignore[assignment]


class _TinyGraphableModel(nn.Module):
    """Small NestedTensor model whose outputs and gradients depend on each input."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3, device=device)

    def forward(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Project the spatial mean while accepting RF-DETR's training signature."""
        del targets
        pooled = samples.tensors.mean(dim=(-2, -1))
        return {"pred": self.projection(pooled)}


class _CaptureUnsupportedModel(_TinyGraphableModel):
    """Model with a host-to-device construction forbidden during CUDA capture."""

    def forward(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Add a CUDA tensor constructed from host data."""
        output = super().forward(samples, targets)
        output["pred"] = output["pred"] + torch.tensor(1.0, device=samples.tensors.device)
        return output


class _StaticGradProjection(torch.autograd.Function):
    """Linear projection whose backward hands out the same gradient buffers on every call.

    This mirrors the ``Graphed`` function inside ``torch.cuda.make_graphed_callables``: the backward graph writes into
    static buffers and returns ``buffer.detach()``, so a replay overwrites whatever autograd received before.
    """

    @staticmethod
    def forward(ctx: object, static_grads: list[Tensor], tensors: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Project the spatial mean and remember what backward needs."""
        pooled = tensors.mean(dim=(-2, -1))
        ctx.static_grads = static_grads  # type: ignore[attr-defined]
        ctx.save_for_backward(pooled)  # type: ignore[attr-defined]
        return torch.nn.functional.linear(pooled, weight, bias)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: object, grad_output: Tensor) -> tuple[None, None, Tensor, Tensor]:
        """Overwrite the static buffers in place and return them, as a graph replay does."""
        (pooled,) = ctx.saved_tensors  # type: ignore[attr-defined]
        weight_grad, bias_grad = ctx.static_grads  # type: ignore[attr-defined]
        weight_grad.copy_(grad_output.t() @ pooled)
        bias_grad.copy_(grad_output.sum(dim=0))
        return None, None, weight_grad.detach(), bias_grad.detach()


def _samples(value: float = 1.0, *, device: torch.device | str = "cpu", batch_size: int = 2) -> NestedTensor:
    """Build a fixed-shape RF-DETR input.

    Examples:
        >>> tuple(_samples(batch_size=1).tensors.shape)
        (1, 4, 3, 3)
    """
    tensors = torch.full((batch_size, 4, 3, 3), value, device=device)
    mask = torch.zeros((batch_size, 3, 3), dtype=torch.bool, device=device)
    return NestedTensor(tensors, mask)


def test_cuda_graphs_and_compile_can_be_combined() -> None:
    """Both graph runtimes may be requested together; the module routes replay through Inductor.

    The combination used to be rejected at config validation. Measured on an RTX PRO 6000 the combined path is 1.20x
    faster than compile alone at batch 4, so the config must accept it.
    """
    config = RFDETRNanoConfig(compile=True, cuda_graphs=True)

    assert (config.compile, config.cuda_graphs) == (True, True)


def test_capture_preserves_accumulated_gradients() -> None:
    """First-seen resolutions must not clear gradients from earlier microbatches."""
    model = _TinyGraphableModel()
    model.projection.weight.grad = torch.full_like(model.projection.weight, 7.0)
    original_grad = model.projection.weight.grad.clone()

    def _fake_capture(module: nn.Module, _sample_args: tuple[Tensor, Tensor], **_kwargs: object) -> nn.Module:
        torch.testing.assert_close(model.projection.weight.grad, original_grad)
        return module

    with patch("torch.cuda.make_graphed_callables", side_effect=_fake_capture):
        CudaGraphTrainingRunner(model)(_samples())

    torch.testing.assert_close(model.projection.weight.grad, original_grad)


def test_replay_accumulates_onto_previous_microbatch_gradients() -> None:
    """A replayed backward adds to the previous microbatch instead of doubling itself.

    ``make_graphed_callables`` returns its static gradient buffers from every backward. When ``.grad`` was ``None``,
    autograd adopts that buffer as the parameter gradient, and the next replay overwrites it before accumulating, so
    ``g0 + g1`` silently becomes ``2 * g1`` for every microbatch after the first of an accumulation window.
    """
    model = _TinyGraphableModel()
    reference = copy.deepcopy(model)
    static_grads = [torch.zeros_like(parameter) for parameter in model.parameters()]

    def _fake_capture(module: nn.Module, _sample_args: tuple[Tensor, Tensor], **_kwargs: object) -> object:
        def _replay(tensors: Tensor, _mask: Tensor) -> dict[str, Tensor]:
            return {"pred": _StaticGradProjection.apply(static_grads, tensors, *module.inner.parameters())}

        return _replay

    with patch("torch.cuda.make_graphed_callables", side_effect=_fake_capture):
        runner = CudaGraphTrainingRunner(model)
        runner(_samples(1.0))["pred"].sum().backward()
        runner(_samples(3.0))["pred"].sum().backward()
    reference(_samples(1.0))["pred"].sum().backward()
    reference(_samples(3.0))["pred"].sum().backward()

    torch.testing.assert_close(model.projection.weight.grad, reference.projection.weight.grad)
    torch.testing.assert_close(model.projection.bias.grad, reference.projection.bias.grad)


def test_runner_does_not_change_registered_model_state() -> None:
    """The runtime helper must not prefix checkpoint keys or replace parameter owners."""
    model = _TinyGraphableModel()
    state_keys = tuple(model.state_dict())
    parameters = tuple(model.parameters())

    CudaGraphTrainingRunner(model)

    assert tuple(model.state_dict()) == state_keys
    assert tuple(model.parameters()) == parameters


def test_capture_restores_autocast_cache_setting() -> None:
    """Capture may disable autocast's weight cache temporarily but cannot leak that global setting."""
    cache_settings: list[bool] = []
    model = _TinyGraphableModel()
    with (
        patch("torch.is_autocast_enabled", return_value=True),
        patch("torch.is_autocast_cache_enabled", return_value=True),
        patch("torch.set_autocast_cache_enabled", side_effect=cache_settings.append),
        patch("torch.cuda.make_graphed_callables", side_effect=lambda module, _args, **_kwargs: module),
    ):
        CudaGraphTrainingRunner(model)(_samples())

    assert cache_settings == [False, True]


def test_capture_logs_signature_once() -> None:
    """A successful capture announces its signature once; replays stay silent.

    Without this line an active graph run is indistinguishable from eager in the console: the only
    other messages on this path are fallback warnings and capture errors. The module logger is
    patched rather than read through ``caplog`` so the count reflects emissions, not how many
    handlers the shared ``rf-detr`` logger happens to carry at this point of the session.
    """
    model = _TinyGraphableModel()
    runner = CudaGraphTrainingRunner(model)

    with (
        patch("torch.cuda.make_graphed_callables", side_effect=lambda module, _args, **_kwargs: module),
        patch("rfdetr.training.cuda_graph_step.logger") as logger,
    ):
        runner(_samples())
        runner(_samples())

    logger.info.assert_called_once()
    message = logger.info.call_args.args[0] % logger.info.call_args.args[1:]
    assert message.startswith("Captured CUDA graph 1 for input shape (2, 4, 3, 3)")


def test_failed_capture_raises_without_eager_retry() -> None:
    """A capture error must stop training, not execute against potentially damaged CUDA state."""
    model = _TinyGraphableModel()
    runner = CudaGraphTrainingRunner(model)
    with patch("torch.cuda.make_graphed_callables", side_effect=RuntimeError("unsupported")) as capture:
        with pytest.raises(RuntimeError, match="CUDA graph capture failed") as exc:
            runner(_samples())

    capture.assert_called_once()
    assert str(exc.value.__cause__) == "unsupported"


def _assert_real_capture_failure() -> None:
    """Check capture failure in a disposable process because CUDA RNG state may be invalidated.

    Examples:
        >>> _assert_real_capture_failure()  # doctest: +SKIP
        # Requires CUDA and deliberately invalidates capture state.
    """
    torch.manual_seed(0)
    model = _CaptureUnsupportedModel("cuda")
    runner = CudaGraphTrainingRunner(model)
    with pytest.raises(RuntimeError, match="Restart the process"):
        runner(_samples(device="cuda"))


@pytest.mark.gpu
def test_real_capture_failure_isolated_from_test_worker() -> None:
    """An intentionally invalid CUDA capture cannot poison later tests in this worker."""
    process = multiprocessing.get_context("spawn").Process(target=_assert_real_capture_failure)
    process.start()
    try:
        process.join(timeout=120)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    torch.manual_seed(0)
    assert torch.isfinite(torch.rand(4, device="cuda")).all()


@pytest.mark.gpu
def test_cuda_graph_replay_matches_eager_outputs_and_accumulated_gradients() -> None:
    """Real capture must preserve outputs and accumulation across changing input values."""
    torch.manual_seed(0)
    eager = _TinyGraphableModel("cuda")
    graphed = _TinyGraphableModel("cuda")
    graphed.load_state_dict(eager.state_dict())

    eager_grad = torch.full_like(eager.projection.weight, 0.25)
    graph_grad = eager_grad.clone()
    eager.projection.weight.grad = eager_grad
    graphed.projection.weight.grad = graph_grad

    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        eager_output = eager(_samples(1.5, device="cuda"))["pred"]
    eager_output.square().sum().backward()

    runner = CudaGraphTrainingRunner(graphed)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        graph_output = runner(_samples(1.5, device="cuda"))["pred"]
        graph_output.square().sum().backward()

    torch.testing.assert_close(graph_output.float(), eager_output.float(), rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        graphed.projection.weight.grad,
        eager.projection.weight.grad,
        rtol=5e-3,
        atol=5e-3,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        replayed = runner(_samples(2.5, device="cuda"))["pred"]
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        expected = eager(_samples(2.5, device="cuda"))["pred"]
    torch.testing.assert_close(replayed.float(), expected.float(), rtol=5e-3, atol=5e-3)
    assert len(runner._graphed_cache) == 1


@pytest.mark.gpu
@pytest.mark.skipif(sdpa_kernel is None, reason="torch.nn.attention.sdpa_kernel needs torch>=2.3")
def test_nano_capture_replay_matches_eager_loss_gradients_and_optimizer_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Nano capture must compose with its eager criterion, accumulation, and parameter updates.

    Parity is checked in strict fp32: capture changes kernel selection and reduction order against eager
    (cuBLAS workspace, SDPA backward, cuDNN conv algorithm), and under bf16 autocast that per-op noise
    compounds through cancellation-heavy gradient sums (attention ``in_proj_bias``, the top-k gathered
    ``enc_out_bbox_embed`` head) well past bf16 precision, so a bf16 bound cannot separate a capture bug
    from rounding. TF32 is pinned off for the same reason. bf16 replay is covered by the tiny-model test.
    """
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(0)
    model_config = RFDETRNanoConfig(pretrain_weights=None, num_classes=3, device="cuda")
    train_config = TrainConfig(dataset_dir="unused", drop_path=0.0)
    eager = build_model_from_config(model_config, train_config).cuda().train()
    graphed = copy.deepcopy(eager)
    criterion, _ = build_criterion_from_config(model_config, train_config)
    criterion = criterion.cuda()
    runner = CudaGraphTrainingRunner(graphed)
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=1e-3)
    graph_optimizer = torch.optim.SGD(graphed.parameters(), lr=1e-3)
    eager_parameters = dict(eager.named_parameters())
    targets = [
        {"labels": torch.tensor([1], device="cuda"), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device="cuda")}
    ]

    fp32 = {"rtol": 1e-4, "atol": 1e-5}
    # Pin one attention kernel for eager and capture alike: capture may otherwise select a different SDPA
    # backend whose fp32 backward differs at ~1e-4, which is indistinguishable from a capture bug here.
    with sdpa_kernel(SDPBackend.MATH):
        # A repeated signature exercises replay; the last capture must preserve accumulated gradients.
        for resolution in (384, 384, 416):
            samples = NestedTensor(
                torch.randn(1, 3, resolution, resolution, device="cuda"),
                torch.zeros(1, resolution, resolution, dtype=torch.bool, device="cuda"),
            )
            expected = eager(samples, targets)
            actual = runner(samples, targets)
            eager_losses = criterion(expected, targets)
            graph_losses = criterion(actual, targets)
            eager_loss = sum(
                eager_losses[key] * weight for key, weight in criterion.weight_dict.items() if key in eager_losses
            )
            graph_loss = sum(
                graph_losses[key] * weight for key, weight in criterion.weight_dict.items() if key in graph_losses
            )
            torch.testing.assert_close(actual["pred_logits"], expected["pred_logits"], **fp32)
            torch.testing.assert_close(actual["pred_boxes"], expected["pred_boxes"], **fp32)
            torch.testing.assert_close(graph_loss, eager_loss, **fp32)
            assert torch.isfinite(graph_loss)
            eager_loss.backward()
            graph_loss.backward()
            for name, parameter in graphed.named_parameters():
                reference = eager_parameters[name]
                assert (parameter.grad is None) == (reference.grad is None), name
                if parameter.grad is not None:
                    # Scale-aware bound: an element-wise atol cannot separate kernel noise on a large gradient
                    # from a wrong gradient on a small one. A capture bug (stale input, dropped or doubled
                    # accumulation) shows as a relative error of 1e-2 or more; kernel noise stays below 1e-5.
                    # The absolute floor covers analytically zero gradients (softmax attention key biases,
                    # ~1e-12 of rounding residue) whose relative error is meaningless.
                    difference_norm = (parameter.grad - reference.grad).norm()
                    reference_norm = reference.grad.norm()
                    assert difference_norm <= 1e-4 * reference_norm + 1e-9, (
                        f"{name}: ||diff|| {difference_norm:.3e} vs ||reference|| {reference_norm:.3e}; "
                        f"max|reference| {reference.grad.abs().max():.3e}"
                    )

    assert len(runner._graphed_cache) == 2
    before = graphed.class_embed.weight.detach().clone()
    eager_optimizer.step()
    graph_optimizer.step()
    assert not torch.equal(graphed.class_embed.weight, before)
    for actual, expected in zip(graphed.parameters(), eager.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
