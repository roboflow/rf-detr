# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.training import auto_batch
from rfdetr.training.auto_batch import AutoBatchResult


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))


def test_recommend_grad_accum_steps_rounds_up():
    assert auto_batch.recommend_grad_accum_steps(3, 16) == 6


def test_probe_max_micro_batch_uses_exponential_then_binary_search():
    model = _TinyModule()
    criterion = _TinyModule()
    threshold = 7

    def _fake_probe(*args, **kwargs):
        micro_batch_size = args[2]
        return micro_batch_size <= threshold

    with (
        patch("rfdetr.training.auto_batch._probe_step", side_effect=_fake_probe),
        patch("rfdetr.training.auto_batch.torch.cuda.empty_cache"),
    ):
        safe = auto_batch.probe_max_micro_batch(
            model=model,
            criterion=criterion,
            resolution=64,
            device=torch.device("cuda"),
            num_classes=5,
            amp=False,
            safety_margin=1.0,
            max_micro_batch=32,
        )
    assert safe == threshold


def test_probe_max_micro_batch_raises_if_one_is_not_safe():
    model = _TinyModule()
    criterion = _TinyModule()

    with (
        patch("rfdetr.training.auto_batch._probe_step", return_value=False),
        patch("rfdetr.training.auto_batch.torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="micro_batch_size=1"),
    ):
        auto_batch.probe_max_micro_batch(
            model=model,
            criterion=criterion,
            resolution=64,
            device=torch.device("cuda"),
            num_classes=5,
            amp=False,
        )


def test_probe_max_micro_batch_reprobes_batch_one_with_state_resident():
    """shadow_optimizer.step() allocates AdamW's exp_avg/exp_avg_sq state lazily on its first-ever call, so a lone
    success at micro_batch_size=1 only proves the batch fits *before* that state exists -- not once it's resident, which
    is what every real training step from the second one onward looks like.

    probe_max_micro_batch must re-probe batch=1 a second time (with the state now resident) and fail loudly if that re-
    probe doesn't hold, instead of silently returning a "safe" batch size that was never actually validated under
    steady-state memory.
    """
    model = _TinyModule()
    criterion = _TinyModule()
    calls: list[int] = []

    def _fake_probe(*args, **kwargs):
        micro_batch_size = args[2]
        calls.append(micro_batch_size)
        # First-ever call (state not yet resident) succeeds; the re-probe (state resident) fails.
        return len(calls) == 1

    with (
        patch("rfdetr.training.auto_batch._probe_step", side_effect=_fake_probe),
        patch("rfdetr.training.auto_batch.torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="optimizer state is resident"),
    ):
        auto_batch.probe_max_micro_batch(
            model=model,
            criterion=criterion,
            resolution=64,
            device=torch.device("cuda"),
            num_classes=5,
            amp=False,
        )

    assert calls == [1, 1]


def test_probe_max_micro_batch_warms_up_state_before_exponential_search():
    """Once batch=1 survives both the state-allocating call and the state-resident re-probe, the exponential search must
    start from candidate=2 -- not re-probe 1 a third time, and not skip probing larger candidates under the now-resident
    state."""
    model = _TinyModule()
    criterion = _TinyModule()
    calls: list[int] = []

    def _fake_probe(*args, **kwargs):
        micro_batch_size = args[2]
        calls.append(micro_batch_size)
        return micro_batch_size <= 4

    with (
        patch("rfdetr.training.auto_batch._probe_step", side_effect=_fake_probe),
        patch("rfdetr.training.auto_batch.torch.cuda.empty_cache"),
    ):
        safe = auto_batch.probe_max_micro_batch(
            model=model,
            criterion=criterion,
            resolution=64,
            device=torch.device("cuda"),
            num_classes=5,
            amp=False,
            safety_margin=1.0,
        )

    assert safe == 4
    assert calls[:2] == [1, 1]
    assert calls.count(1) == 2


def test_probe_step_raises_when_loss_keys_do_not_overlap_weight_keys():
    """_probe_step must fail fast when weighted loss would be empty."""

    class _DummyCriterion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_dict = {"loss_bbox": 1.0}

        def forward(self, outputs, targets):
            return {"loss_ce": torch.tensor(1.0)}

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, samples, targets):
            return {}

    model = _DummyModel()
    criterion = _DummyCriterion()
    trainable_params = list(model.parameters())
    shadow_optimizer, shadow_params = auto_batch._build_shadow_optimizer(trainable_params, lr=1e-4, weight_decay=1e-4)

    with (
        patch(
            "rfdetr.training.auto_batch._make_synthetic_batch",
            return_value=(MagicMock(), []),
        ),
        pytest.raises(RuntimeError, match="no overlap between criterion loss_dict and weight_dict keys"),
    ):
        auto_batch._probe_step(
            model=model,
            criterion=criterion,
            micro_batch_size=1,
            resolution=64,
            device=torch.device("cpu"),
            num_classes=5,
            amp=False,
            trainable_params=trainable_params,
            shadow_optimizer=shadow_optimizer,
            shadow_params=shadow_params,
        )


class _WorkingDummyModel(torch.nn.Module):
    """A minimal model whose output is differentiable w.r.t.

    its own parameter, so a full forward+backward+shadow-optimizer-step probe cycle can run without a real RF-DETR
    model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(3))

    def forward(self, samples, targets):
        return {"pred": self.w.sum() * len(targets)}


class _WorkingDummyCriterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_dict = {"loss_ce": 1.0}

    def forward(self, outputs, targets):
        return {"loss_ce": outputs["pred"] ** 2}


class TestProbeStepShadowOptimizer:
    """_probe_step's shadow-optimizer step must allocate real AdamW state (so the probe accounts for the memory
    training's first real optimizer.step() will need) without ever writing to the real model's own parameters (which may
    already hold a loaded pretrained checkpoint)."""

    def _run_one_probe_step(self) -> tuple[_WorkingDummyModel, torch.optim.Optimizer, list[torch.Tensor]]:
        """Run one real (non-mocked) forward+backward+shadow-optimizer-step probe cycle on ``_WorkingDummyModel``.

        Shared setup for the tests in this class: builds a fresh model/criterion/shadow optimizer, patches only
        ``_make_synthetic_batch`` (batch construction, not the forward/backward/step mechanic under test), and
        asserts the probe step itself succeeded before handing the state back for each test to inspect.

        Returns:
            The model, the shadow optimizer, and its shadow parameters, all post-step.

        Examples:
            >>> model, shadow_optimizer, shadow_params = TestProbeStepShadowOptimizer()._run_one_probe_step()
            >>> len(shadow_optimizer.state) == len(shadow_params) > 0
            True
        """
        model = _WorkingDummyModel()
        criterion = _WorkingDummyCriterion()
        trainable_params = list(model.parameters())
        shadow_optimizer, shadow_params = auto_batch._build_shadow_optimizer(
            trainable_params, lr=1e-2, weight_decay=0.0
        )
        with patch(
            "rfdetr.training.auto_batch._make_synthetic_batch",
            return_value=(MagicMock(), [{}]),
        ):
            ok = auto_batch._probe_step(
                model=model,
                criterion=criterion,
                micro_batch_size=1,
                resolution=64,
                device=torch.device("cpu"),
                num_classes=5,
                amp=False,
                trainable_params=trainable_params,
                shadow_optimizer=shadow_optimizer,
                shadow_params=shadow_params,
            )
        assert ok is True
        return model, shadow_optimizer, shadow_params

    def test_does_not_mutate_real_model_parameters(self) -> None:
        """The shadow optimizer step must update the shadow tensors, not the real model -- a probe that corrupted a
        loaded pretrained checkpoint's weights before training even started would be far worse than the memory-
        underestimation bug it fixes."""
        model = _WorkingDummyModel()
        before = model.w.detach().clone()
        criterion = _WorkingDummyCriterion()
        trainable_params = list(model.parameters())
        shadow_optimizer, shadow_params = auto_batch._build_shadow_optimizer(
            trainable_params, lr=1e-2, weight_decay=0.0
        )

        with patch(
            "rfdetr.training.auto_batch._make_synthetic_batch",
            return_value=(MagicMock(), [{}]),
        ):
            auto_batch._probe_step(
                model=model,
                criterion=criterion,
                micro_batch_size=1,
                resolution=64,
                device=torch.device("cpu"),
                num_classes=5,
                amp=False,
                trainable_params=trainable_params,
                shadow_optimizer=shadow_optimizer,
                shadow_params=shadow_params,
            )

        assert torch.equal(model.w.detach(), before), "the real model parameter must be unchanged"

    def test_populates_shadow_optimizer_state(self) -> None:
        """AdamW allocates exp_avg/exp_avg_sq lazily on the first step() -- pins that the shadow step actually triggers
        that allocation instead of being a no-op (e.g. from a grad-aliasing bug that left shadow_params.grad as None,
        which AdamW silently skips)."""
        _, shadow_optimizer, shadow_params = self._run_one_probe_step()

        assert len(shadow_optimizer.state) == len(shadow_params) > 0
        for shadow_p in shadow_params:
            state = shadow_optimizer.state[shadow_p]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state
            assert state["exp_avg"].shape == shadow_p.shape

    def test_shadow_optimizer_state_reused_not_reallocated_across_iterations(self) -> None:
        """A second probe iteration (the next candidate batch size in the same search) must reuse the already-allocated
        exp_avg/exp_avg_sq tensors, not allocate fresh ones -- otherwise every probe iteration would pay the allocation
        cost this fix exists to move to the first iteration only."""
        model, shadow_optimizer, shadow_params = self._run_one_probe_step()
        criterion = _WorkingDummyCriterion()
        exp_avg_id_before = id(shadow_optimizer.state[shadow_params[0]]["exp_avg"])

        with patch(
            "rfdetr.training.auto_batch._make_synthetic_batch",
            return_value=(MagicMock(), [{}]),
        ):
            auto_batch._probe_step(
                model=model,
                criterion=criterion,
                micro_batch_size=2,
                resolution=64,
                device=torch.device("cpu"),
                num_classes=5,
                amp=False,
                trainable_params=list(model.parameters()),
                shadow_optimizer=shadow_optimizer,
                shadow_params=shadow_params,
            )

        exp_avg_id_after = id(shadow_optimizer.state[shadow_params[0]]["exp_avg"])
        assert exp_avg_id_after == exp_avg_id_before

    def test_shadow_grad_cleared_after_step(self) -> None:
        """shadow_p.grad must not outlive the shadow_optimizer.step() that consumes it: leaving the alias set keeps this
        iteration's real-gradient tensor resident (via shadow_p.grad, its only remaining reference once
        model.zero_grad() clears real_p.grad) through the next iteration's backward() -- two gradient buffers alive at
        once instead of one."""
        _, _, shadow_params = self._run_one_probe_step()

        for shadow_p in shadow_params:
            assert shadow_p.grad is None

    def test_shadow_grad_cleared_when_step_itself_ooms(self) -> None:
        """A CUDA OOM raised by shadow_optimizer.step() itself (not by forward/backward) must still clear
        shadow_p.grad -- otherwise the alias is the only remaining reference keeping that iteration's real
        gradient tensor alive, unreachable by torch.cuda.empty_cache(), for the rest of the search."""

        class _OomOnStep:
            def step(self) -> None:
                raise RuntimeError("CUDA out of memory.")

        model = _WorkingDummyModel()
        criterion = _WorkingDummyCriterion()
        trainable_params = list(model.parameters())
        _, shadow_params = auto_batch._build_shadow_optimizer(trainable_params, lr=1e-2, weight_decay=0.0)

        with patch(
            "rfdetr.training.auto_batch._make_synthetic_batch",
            return_value=(MagicMock(), [{}]),
        ):
            ok = auto_batch._probe_step(
                model=model,
                criterion=criterion,
                micro_batch_size=1,
                resolution=64,
                device=torch.device("cpu"),
                num_classes=5,
                amp=False,
                trainable_params=trainable_params,
                shadow_optimizer=_OomOnStep(),
                shadow_params=shadow_params,
            )

        assert ok is False
        for shadow_p in shadow_params:
            assert shadow_p.grad is None


def test_build_shadow_optimizer_forwards_optimizer_kwargs():
    """train_config.optimizer_kwargs (e.g. amsgrad=True, which adds a third max_exp_avg_sq state buffer per
    parameter -- see torch.optim.AdamW) must reach the shadow AdamW, the same way configure_optimizers forwards
    them to the real one (module_model.py). Without this, the probe silently under-counts optimizer-state memory
    for any AdamW variant beyond the plain defaults, which is the dangerous direction: it can report a batch size
    that OOMs on the real optimizer.step()."""
    params = [torch.nn.Parameter(torch.zeros(3))]
    optimizer, shadow_params = auto_batch._build_shadow_optimizer(
        params, lr=1e-3, weight_decay=0.0, optimizer_kwargs={"amsgrad": True}
    )
    shadow_params[0].grad = torch.ones_like(shadow_params[0])
    optimizer.step()

    assert "max_exp_avg_sq" in optimizer.state[shadow_params[0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW requires CUDA")
def test_build_shadow_optimizer_forwards_fused():
    """Fused must reach the shadow AdamW so its step-time temporary memory matches the real optimizer's fused vs.

    foreach/single-tensor implementation (see RFDETRModelModule._use_fused_optimizer).
    """
    params = [torch.nn.Parameter(torch.zeros(3, device="cuda"))]
    optimizer, _ = auto_batch._build_shadow_optimizer(params, lr=1e-3, weight_decay=0.0, fused=True)

    assert optimizer.defaults["fused"] is True


def test_probe_max_micro_batch_restores_train_mode_when_shadow_optimizer_build_fails():
    """If _build_shadow_optimizer itself raises (e.g. OOM allocating the shadow parameter tensors for a very
    large model), the model/criterion train-mode flip from the start of probe_max_micro_batch must still be
    undone -- a probe that mutates the caller's eval-mode model and then crashes without restoring it would leave
    the model silently in the wrong mode for whatever runs next."""
    model = _TinyModule()
    criterion = _TinyModule()
    model.eval()
    criterion.eval()

    with (
        patch("rfdetr.training.auto_batch._build_shadow_optimizer", side_effect=RuntimeError("boom")),
        patch("rfdetr.training.auto_batch.torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="boom"),
    ):
        auto_batch.probe_max_micro_batch(
            model=model,
            criterion=criterion,
            resolution=64,
            device=torch.device("cuda"),
            num_classes=5,
            amp=False,
        )

    assert model.training is False
    assert criterion.training is False


def test_resolve_auto_batch_config_requires_cuda():
    model_context = SimpleNamespace(device=torch.device("cpu"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=False, segmentation_head=False)
    train_config = SimpleNamespace(batch_size="auto", auto_batch_target_effective=16)

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="requires a CUDA device"),
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)


def test_resolve_auto_batch_config_returns_expected_values():
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=False, segmentation_head=True)
    train_config = SimpleNamespace(batch_size="auto", auto_batch_target_effective=16, lr=1e-4, weight_decay=1e-4)
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5),
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        result = auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert isinstance(result, AutoBatchResult)
    assert result.safe_micro_batch == 5
    assert result.recommended_grad_accum_steps == 4
    assert result.effective_batch_size == 20
    assert result.device_name == "Fake GPU"


def test_resolve_auto_batch_config_warns_when_optimizer_is_not_builtin_adamw():
    """The shadow optimizer probe always models AdamW's state memory (see _build_shadow_optimizer); a
    train_config.optimizer other than the built-in "adamw" (e.g. "sgd", a pytorch-optimizer name, a custom
    callable/dotted path -- all supported by configure_optimizers) makes that estimate unreliable and must be surfaced,
    not silently assumed correct."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=False, segmentation_head=True)
    train_config = SimpleNamespace(
        batch_size="auto", auto_batch_target_effective=16, lr=1e-4, weight_decay=1e-4, optimizer="sgd"
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5),
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
        patch("rfdetr.training.auto_batch.logger.warning") as mock_warning,
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert mock_warning.call_count == 1
    assert mock_warning.call_args.args[1] == "sgd"


def test_resolve_auto_batch_config_does_not_warn_for_builtin_adamw():
    """The default (and explicit "adamw") config is exactly what the shadow optimizer models, so no warning should
    fire -- a spurious warning on the common path would train users to ignore it."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=False, segmentation_head=True)
    train_config = SimpleNamespace(
        batch_size="auto", auto_batch_target_effective=16, lr=1e-4, weight_decay=1e-4, optimizer="adamw"
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5),
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
        patch("rfdetr.training.auto_batch.logger.warning") as mock_warning,
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    mock_warning.assert_not_called()


def test_resolve_auto_batch_config_forwards_optimizer_kwargs_for_builtin_adamw():
    """train_config.optimizer_kwargs (e.g. amsgrad=True) must reach the shadow optimizer when optimizer="adamw", the
    same kwargs configure_optimizers forwards to the real AdamW -- otherwise the probe silently ignores an optimizer-
    state-affecting setting the real training run actually uses."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=False, segmentation_head=True)
    train_config = SimpleNamespace(
        batch_size="auto",
        auto_batch_target_effective=16,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        optimizer_kwargs={"amsgrad": True},
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5) as mock_probe,
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert mock_probe.call_args.kwargs["optimizer_kwargs"] == {"amsgrad": True}


def test_resolve_auto_batch_config_does_not_forward_optimizer_kwargs_for_non_adamw():
    """A non-"adamw" optimizer already gets the mismatched-optimizer warning; forwarding its optimizer_kwargs (which are
    specific to that other optimizer's constructor, e.g. Lion's) into the AdamW-only shadow optimizer would raise a
    TypeError instead of just producing an imprecise estimate."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=False, segmentation_head=True)
    train_config = SimpleNamespace(
        batch_size="auto",
        auto_batch_target_effective=16,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="sgd",
        optimizer_kwargs={"momentum": 0.9},
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5) as mock_probe,
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
        patch("rfdetr.training.auto_batch.logger.warning"),
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert mock_probe.call_args.kwargs["optimizer_kwargs"] is None


def test_resolve_auto_batch_config_forwards_fused_when_bf16_and_builtin_adamw():
    """When the real run would use bf16-mixed precision with the built-in adamw optimizer and
    model_config.fused_optimizer=True (RFDETRModelModule._use_fused_optimizer's own conditions), the shadow optimizer
    must be built fused too -- otherwise its step-time temporary memory (foreach/single-tensor) can diverge from what
    the real fused AdamW actually needs."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=True, segmentation_head=True, fused_optimizer=True)
    train_config = SimpleNamespace(
        batch_size="auto",
        auto_batch_target_effective=16,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        amp_dtype="auto",
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.torch.cuda.is_bf16_supported", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5) as mock_probe,
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert mock_probe.call_args.kwargs["optimizer_fused"] is True


def test_resolve_auto_batch_config_does_not_force_fused_for_fp16():
    """amp_dtype='fp16' resolves to '16-mixed' precision, not a bf16 variant -- fused AdamW is only safe under
    bf16-mixed (see RFDETRModelModule._use_fused_optimizer), so the shadow optimizer must not be forced fused."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(resolution=64, num_classes=5, amp=True, segmentation_head=True, fused_optimizer=True)
    train_config = SimpleNamespace(
        batch_size="auto",
        auto_batch_target_effective=16,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        amp_dtype="fp16",
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.torch.cuda.is_bf16_supported", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5) as mock_probe,
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert mock_probe.call_args.kwargs["optimizer_fused"] is False


def test_resolve_auto_batch_config_does_not_force_fused_when_model_config_disables_it():
    """model_config.fused_optimizer=False must be honored the same way RFDETRModelModule._use_fused_optimizer honors it
    for the real optimizer."""
    model_context = SimpleNamespace(device=torch.device("cuda"), model=MagicMock())
    model_config = SimpleNamespace(
        resolution=64, num_classes=5, amp=True, segmentation_head=True, fused_optimizer=False
    )
    train_config = SimpleNamespace(
        batch_size="auto",
        auto_batch_target_effective=16,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        amp_dtype="auto",
    )
    criterion = MagicMock()
    criterion.to.return_value = criterion

    with (
        patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True),
        patch("rfdetr.training.auto_batch.torch.cuda.is_bf16_supported", return_value=True),
        patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None)),
        patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5) as mock_probe,
        patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        auto_batch.resolve_auto_batch_config(model_context, model_config, train_config)

    assert mock_probe.call_args.kwargs["optimizer_fused"] is False


@patch("rfdetr.detr.is_main_process", return_value=False)
@patch("rfdetr.training.auto_batch.resolve_auto_batch_config")
@patch("rfdetr.training.build_trainer")
@patch("rfdetr.training.RFDETRDataModule")
@patch("rfdetr.training.RFDETRModelModule")
@patch("rfdetr.detr._move_model_context_to_device")
def test_train_auto_batch_ensures_model_on_device_before_resolve(
    mock_move: MagicMock,
    _mock_module: MagicMock,
    _mock_data_module: MagicMock,
    _mock_build_trainer: MagicMock,
    mock_resolve: MagicMock,
    _mock_is_main: MagicMock,
    tmp_path,
) -> None:
    """Model weights must be moved before resolve_auto_batch_config when batch_size='auto'."""
    auto_result = SimpleNamespace(safe_micro_batch=4, recommended_grad_accum_steps=1, effective_batch_size=4)
    call_order: list[str] = []

    def _move_side_effect(model: object) -> None:
        call_order.append("ensure")

    def _resolve_side_effect(**_kwargs: object) -> object:
        call_order.append("resolve")
        return auto_result

    mock_move.side_effect = _move_side_effect
    mock_resolve.side_effect = _resolve_side_effect

    # Real config objects: RFDETR.train() rebuilds self.model.args via _namespace_from_configs
    # after trainer.fit() (#1199 fix), which requires genuine ModelConfig/TrainConfig instances.
    train_config = TrainConfig(
        dataset_dir=None,
        output_dir=str(tmp_path / "out"),
        batch_size="auto",
        grad_accum_steps=99,
        tensorboard=False,
    )
    mock_self = MagicMock()
    mock_self.model_config = RFDETRBaseConfig(pretrain_weights=None, num_classes=3, device="cpu")
    mock_self.get_train_config.return_value = train_config

    RFDETR.train(mock_self)

    assert train_config.batch_size == 4
    assert train_config.grad_accum_steps == 1
    mock_move.assert_called_once_with(mock_self.model)
    mock_resolve.assert_called_once_with(
        model_context=mock_self.model,
        model_config=mock_self.model_config,
        train_config=train_config,
    )
    assert call_order == ["ensure", "resolve"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for segmentation probe")
def test_probe_step_with_real_segmentation_criterion(tmp_path):
    """Run one probe step with real segmentation model and criterion so loss_masks and t['masks'] are exercised."""
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors, build_model

    mc = RFDETRSegNanoConfig(pretrain_weights=None, device="cuda", num_classes=2)
    tc = SegmentationTrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        batch_size=2,
        grad_accum_steps=1,
        tensorboard=False,
    )
    args = _namespace_from_configs(mc, tc)
    model = build_model(args)
    criterion, _ = build_criterion_and_postprocessors(args)
    device = torch.device("cuda")
    model = model.to(device)
    criterion = criterion.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    shadow_optimizer, shadow_params = auto_batch._build_shadow_optimizer(
        trainable_params, lr=tc.lr, weight_decay=tc.weight_decay
    )

    ok = auto_batch._probe_step(
        model=model,
        criterion=criterion,
        micro_batch_size=1,
        resolution=mc.resolution,
        device=device,
        num_classes=mc.num_classes,
        amp=False,
        trainable_params=trainable_params,
        shadow_optimizer=shadow_optimizer,
        shadow_params=shadow_params,
        segmentation_head=True,
    )
    assert ok is True
