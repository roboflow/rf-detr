# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for automatic batch sizing: the shadow-optimizer memory probe and ``resolve_auto_batch_config``."""

import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.training import auto_batch, build_trainer
from rfdetr.training.auto_batch import AutoBatchResult


class _TinyModule(torch.nn.Module):
    """A module with one trainable parameter, standing in for both model and criterion in mocked probe searches."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))


class _WorkingDummyModel(torch.nn.Module):
    """A minimal model whose output is differentiable w.r.t.

    its own parameter.     A full forward+backward+shadow-optimizer-step probe cycle can therefore run without a real
    RF-DETR model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(3))

    def forward(self, samples, targets):
        """Return a prediction that scales with the number of targets."""
        return {"pred": self.w.sum() * len(targets)}


class _WorkingDummyCriterion(torch.nn.Module):
    """A criterion whose single weighted loss is the square of ``_WorkingDummyModel``'s prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_dict = {"loss_ce": 1.0}

    def forward(self, outputs, targets):
        """Return the squared prediction as the only loss."""
        return {"loss_ce": outputs["pred"] ** 2}


class _PartiallyUsedDummyModel(torch.nn.Module):
    """A model with two trainable parameters where only one participates in the forward pass.

    The other's ``.grad`` stays ``None`` after ``backward()`` -- the scenario ``warn_missing_grads`` exists to report.
    """

    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Parameter(torch.ones(3))
        self.unused = torch.nn.Parameter(torch.ones(2))

    def forward(self, samples, targets):
        """Return a prediction that only depends on ``used``."""
        return {"pred": self.used.sum() * len(targets)}


class _CpuAsCuda(str):
    """A ``str`` subclass equal to ``"cpu"`` whose ``.type`` reports ``"cuda"``.

    ``probe_max_micro_batch`` hard-requires ``device.type == "cuda"`` before it will run at all, but every
    downstream tensor factory call (``torch.randn(..., device=...)`` and friends) accepts any string that names a
    real device -- so this lets a CPU-only test satisfy the guard while every actual tensor op still runs on CPU,
    driving the real search loop without a GPU.

    Examples:
        >>> device = _CpuAsCuda("cpu")
        >>> device.type
        'cuda'
        >>> torch.zeros(1, device=device).device.type
        'cpu'
    """

    @property
    def type(self) -> str:
        """Report ``"cuda"`` regardless of the underlying device string."""
        return "cuda"


class _OomOnStep:
    """A shadow optimizer whose ``step()`` raises a CUDA out-of-memory error."""

    def step(self) -> None:
        """Raise the error CUDA raises when the optimizer state does not fit."""
        raise RuntimeError("CUDA out of memory.")


def _model_context(device: torch.device | None = None) -> SimpleNamespace:
    """Return a model context on a CUDA device (``cuda`` by default) wrapping a mock model.

    Examples:
        >>> _model_context().device.type, _model_context(torch.device("cuda", 1)).device.index
        ('cuda', 1)
    """
    return SimpleNamespace(device=device or torch.device("cuda"), model=MagicMock())


def _model_config(**overrides: Any) -> SimpleNamespace:
    """Return a model config namespace (64px, 5 classes, segmentation head, no AMP) with ``overrides`` applied.

    Examples:
        >>> config = _model_config(amp=True)
        >>> config.resolution, config.amp, config.segmentation_head
        (64, True, True)
    """
    return SimpleNamespace(**{"resolution": 64, "num_classes": 5, "amp": False, "segmentation_head": True} | overrides)


def _train_config(**overrides: Any) -> SimpleNamespace:
    """Return an automatic-batch train config namespace (target 16, no AMP dtype) with ``overrides`` applied.

    Examples:
        >>> config = _train_config(auto_batch_target_effective=17)
        >>> config.batch_size, config.auto_batch_target_effective, config.amp_dtype
        ('auto', 17, None)
    """
    defaults = {
        "amp_dtype": None,
        "batch_size": "auto",
        "auto_batch_target_effective": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    }
    return SimpleNamespace(**defaults | overrides)


def _capability_lookup(capabilities: dict[int, tuple[int, int]]) -> Callable[..., tuple[int, int]]:
    """Return a ``torch.cuda.get_device_capability`` stand-in reading ``capabilities``; ``None`` is the current device.

    Examples:
        >>> lookup = _capability_lookup({0: (7, 5), 1: (8, 0)})
        >>> lookup(), lookup(torch.device("cuda", 1)), lookup(0)
        ((7, 5), (8, 0), (7, 5))
    """

    def device_capability(device: torch.device | int | None = None) -> tuple[int, int]:
        """Return the mocked compute capability of ``device``; ``None`` is the current device, cuda:0."""
        index = device.index if isinstance(device, torch.device) else device
        return capabilities[0 if index is None else index]

    return device_capability


def _probe_max_kwargs(model: torch.nn.Module, criterion: torch.nn.Module, **overrides: Any) -> dict[str, Any]:
    """Return ``probe_max_micro_batch`` keyword arguments for a 64px, 5-class, non-AMP probe on ``cuda``.

    Examples:
        >>> kwargs = _probe_max_kwargs(None, None, resolution=8)
        >>> kwargs["resolution"], kwargs["num_classes"], kwargs["device"].type
        (8, 5, 'cuda')
    """
    kwargs = {
        "model": model,
        "criterion": criterion,
        "resolution": 64,
        "device": torch.device("cuda"),
        "num_classes": 5,
        "amp": False,
    }
    return kwargs | overrides


def _probe_step_kwargs(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    *,
    shadow_lr: float = 1e-2,
    shadow_weight_decay: float = 0.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Return ``_probe_step`` keyword arguments for one 64px CPU sample, with a fresh shadow optimizer for ``model``.

    Examples:
        >>> kwargs = _probe_step_kwargs(_WorkingDummyModel(), _WorkingDummyCriterion(), micro_batch_size=2)
        >>> kwargs["micro_batch_size"], len(kwargs["shadow_params"]), kwargs["device"].type
        (2, 1, 'cpu')
    """
    trainable_params = list(model.parameters())
    shadow_optimizer, shadow_params = auto_batch._build_shadow_optimizer(
        trainable_params, lr=shadow_lr, weight_decay=shadow_weight_decay
    )
    kwargs = {
        "model": model,
        "criterion": criterion,
        "micro_batch_size": 1,
        "resolution": 64,
        "device": torch.device("cpu"),
        "num_classes": 5,
        "amp": False,
        "trainable_params": trainable_params,
        "shadow_optimizer": shadow_optimizer,
        "shadow_params": shadow_params,
    }
    return kwargs | overrides


@pytest.fixture
def mock_empty_cache(mocker: MockerFixture) -> MagicMock:
    """Patch ``torch.cuda.empty_cache`` so probe searches run on hosts without CUDA.

    Examples:
        >>> mock_empty_cache(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    return mocker.patch("rfdetr.training.auto_batch.torch.cuda.empty_cache")


@pytest.fixture
def synthetic_batch(mocker: MockerFixture) -> MagicMock:
    """Patch ``_make_synthetic_batch`` to return one empty target, keeping batch construction out of the probe step.

    Examples:
        >>> synthetic_batch(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    return mocker.patch("rfdetr.training.auto_batch._make_synthetic_batch", return_value=(MagicMock(), [{}]))


@pytest.fixture
def mock_probe(mocker: MockerFixture) -> MagicMock:
    """Patch the CUDA, criterion and device-name boundaries of ``resolve_auto_batch_config``; return the mocked probe.

    The probe reports a safe micro batch of 5; tests set ``return_value`` when they need another.

    Examples:
        >>> mock_probe(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    criterion = MagicMock()
    criterion.to.return_value = criterion
    mocker.patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=True)
    mocker.patch("rfdetr.training.auto_batch.build_criterion_from_config", return_value=(criterion, None))
    mocker.patch("rfdetr.training.auto_batch.torch.cuda.get_device_name", return_value="Fake GPU")
    return mocker.patch("rfdetr.training.auto_batch.probe_max_micro_batch", return_value=5)


def test_recommend_grad_accum_steps_rounds_up():
    """Gradient accumulation steps round up so the effective batch reaches the target."""
    assert auto_batch.recommend_grad_accum_steps(3, 16) == 6


@pytest.mark.usefixtures("mock_empty_cache")
class TestProbeMaxMicroBatch:
    """``probe_max_micro_batch`` finds the largest micro batch that fits and leaves the caller's modules untouched."""

    @pytest.mark.parametrize(
        ("threshold", "search_kwargs"),
        [
            pytest.param(7, {"max_micro_batch": 32}, id="threshold-7-capped-at-32"),
            pytest.param(4, {}, id="threshold-4-default-cap"),
        ],
    )
    def test_warms_up_then_searches_exponentially_and_by_bisection(
        self, mocker: MockerFixture, threshold: int, search_kwargs: dict[str, int]
    ):
        """The probe returns the largest micro batch the step probe reports as safe, after the batch-1 warm-up.

        Once batch=1 survives both the state-allocating call and the state-resident re-probe, the exponential search
        starts from candidate=2 -- it does not re-probe 1 a third time, and does not skip larger candidates under the
        now-resident state.
        """
        probe = mocker.patch(
            "rfdetr.training.auto_batch._probe_step", side_effect=lambda *args, **kwargs: args[2] <= threshold
        )

        safe = auto_batch.probe_max_micro_batch(
            **_probe_max_kwargs(_TinyModule(), _TinyModule(), safety_margin=1.0, **search_kwargs)
        )

        calls = [call.args[2] for call in probe.call_args_list]
        assert safe == threshold
        assert calls[:2] == [1, 1]
        assert calls.count(1) == 2

    def test_raises_if_one_is_not_safe(self, mocker: MockerFixture):
        """The probe raises when even micro_batch_size=1 does not fit."""
        mocker.patch("rfdetr.training.auto_batch._probe_step", return_value=False)

        with pytest.raises(RuntimeError, match="micro_batch_size=1"):
            auto_batch.probe_max_micro_batch(**_probe_max_kwargs(_TinyModule(), _TinyModule()))

    def test_reprobes_batch_one_with_state_resident(self, mocker: MockerFixture):
        """A lone success at batch 1 is re-probed once optimizer state is resident, and failing that fails loudly.

        shadow_optimizer.step() allocates AdamW's exp_avg/exp_avg_sq state lazily on its first-ever call, so a lone
        success at micro_batch_size=1 only proves the batch fits *before* that state exists -- not once it's resident,
        which is what every real training step from the second one onward looks like. Returning a "safe" batch size
        never validated under steady-state memory would be silently wrong.
        """
        # First-ever call (state not yet resident) succeeds; the re-probe (state resident) fails.
        probe = mocker.patch("rfdetr.training.auto_batch._probe_step", side_effect=[True, False])

        with pytest.raises(RuntimeError, match="optimizer state is resident"):
            auto_batch.probe_max_micro_batch(**_probe_max_kwargs(_TinyModule(), _TinyModule()))

        assert [call.args[2] for call in probe.call_args_list] == [1, 1]

    def test_raises_before_building_shadow_optimizer_when_no_trainable_params(self):
        """A fully-frozen model fails fast with a dedicated RuntimeError and still restores eval mode.

        The real (unmocked) guard fires before ``_build_shadow_optimizer``'s ``torch.optim.AdamW([])`` call is reached.
        """
        model = _TinyModule()
        model.w.requires_grad_(False)
        criterion = _TinyModule()
        model.eval()
        criterion.eval()

        with pytest.raises(RuntimeError) as exc_info:
            auto_batch.probe_max_micro_batch(**_probe_max_kwargs(model, criterion))

        assert "requires at least one trainable parameter;" in str(exc_info.value)
        assert "requires_grad=False." in str(exc_info.value)
        assert model.training is False
        assert criterion.training is False

    def test_restores_train_mode_when_shadow_optimizer_build_fails(self, mocker: MockerFixture):
        """A failing ``_build_shadow_optimizer`` (e.g. OOM on a very large model) still restores eval mode.

        A probe that flips the caller's eval-mode model to train mode and then crashes without restoring it would leave
        the model silently in the wrong mode for whatever runs next.
        """
        model = _TinyModule()
        criterion = _TinyModule()
        model.eval()
        criterion.eval()
        mocker.patch("rfdetr.training.auto_batch._build_shadow_optimizer", side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            auto_batch.probe_max_micro_batch(**_probe_max_kwargs(model, criterion))

        assert model.training is False
        assert criterion.training is False

    def test_drives_real_search_loop_end_to_end(self, mocker: MockerFixture):
        """A CPU-only run of the unmocked search loop drives every candidate through the real probe step.

        Real _probe_step and _build_shadow_optimizer calls (not mocked stand-ins) must drive at least two candidate
        batch sizes past the warm-up re-probe and produce the safe batch size the exponential/bisection math predicts.
        """
        spy_probe_step = mocker.patch("rfdetr.training.auto_batch._probe_step", wraps=auto_batch._probe_step)
        kwargs = _probe_max_kwargs(
            _WorkingDummyModel(),
            _WorkingDummyCriterion(),
            resolution=8,
            device=_CpuAsCuda("cpu"),
            safety_margin=1.0,
            max_micro_batch=4,
        )

        safe = auto_batch.probe_max_micro_batch(**kwargs)

        assert safe == 4
        # 1 (warm-up), 1 (state-resident re-probe), 2, 4 -- four real candidates, driven by the real loop.
        assert [call.args[2] for call in spy_probe_step.call_args_list] == [1, 1, 2, 4]

    def test_builds_shadow_optimizer_exactly_once(self, mocker: MockerFixture):
        """The shadow optimizer is built once for the whole search, not once per candidate batch size.

        _build_shadow_optimizer allocates one extra copy of the trainable parameters and (via the first real step())
        AdamW's exp_avg/exp_avg_sq state; rebuilding it per candidate would pay both costs on every iteration.
        """
        spy_build_shadow_optimizer = mocker.patch(
            "rfdetr.training.auto_batch._build_shadow_optimizer", wraps=auto_batch._build_shadow_optimizer
        )
        kwargs = _probe_max_kwargs(
            _WorkingDummyModel(),
            _WorkingDummyCriterion(),
            resolution=8,
            device=_CpuAsCuda("cpu"),
            safety_margin=1.0,
            max_micro_batch=4,
        )

        safe = auto_batch.probe_max_micro_batch(**kwargs)

        assert safe == 4
        assert spy_build_shadow_optimizer.call_count == 1


class TestProbeStep:
    """``_probe_step`` runs one forward/backward/shadow-step cycle and validates its inputs."""

    def test_raises_when_loss_keys_do_not_overlap_weight_keys(self, synthetic_batch: MagicMock):
        """_probe_step must fail fast when weighted loss would be empty."""

        class _DummyCriterion(torch.nn.Module):
            """A criterion whose loss key is absent from its weight dict."""

            def __init__(self) -> None:
                super().__init__()
                self.weight_dict = {"loss_bbox": 1.0}

            def forward(self, outputs, targets):
                """Return a loss under a key ``weight_dict`` does not weight."""
                return {"loss_ce": torch.tensor(1.0)}

        class _DummyModel(torch.nn.Module):
            """A model with one trainable parameter and an empty output."""

            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.ones(1))

            def forward(self, samples, targets):
                """Return no outputs."""
                return {}

        synthetic_batch.return_value = (MagicMock(), [])
        kwargs = _probe_step_kwargs(_DummyModel(), _DummyCriterion(), shadow_lr=1e-4, shadow_weight_decay=1e-4)

        with pytest.raises(RuntimeError, match="no overlap between criterion loss_dict and weight_dict keys"):
            auto_batch._probe_step(**kwargs)

    def test_rejects_trainable_params_shadow_args_passed_positionally(self):
        """trainable_params/shadow_optimizer/shadow_params (and every parameter after them) are keyword-only.

        A caller still passing them positionally must get a TypeError, not have them silently bound to the wrong
        parameter.
        """
        model = _TinyModule()
        criterion = _TinyModule()

        with pytest.raises(TypeError):
            auto_batch._probe_step(model, criterion, 1, 64, torch.device("cpu"), 5, False, [], None, [])

    def test_warns_once_with_missing_grad_count_when_armed(self, mocker: MockerFixture, synthetic_batch: MagicMock):
        """warn_missing_grads=True must report exactly the trainable params the synthetic forward pass never reached.

        Here 1 of 2: AdamW only allocates state for params that carry a gradient, so an unreported miss would silently
        under-count the probe's optimizer-state memory estimate.
        """
        mock_warning = mocker.patch("rfdetr.training.auto_batch.logger.warning")
        kwargs = _probe_step_kwargs(_PartiallyUsedDummyModel(), _WorkingDummyCriterion(), warn_missing_grads=True)

        ok = auto_batch._probe_step(**kwargs)

        assert ok is True
        mock_warning.assert_called_once()
        assert mock_warning.call_args.args[1:] == (1, 2)


class TestProbeStepShadowOptimizer:
    """_probe_step's shadow-optimizer step allocates real AdamW state without writing to the real model's parameters.

    The state memory is what training's first real optimizer.step() will need, and the real model may already hold a
    loaded pretrained checkpoint that the probe must not touch.
    """

    @pytest.fixture
    def probed(
        self, synthetic_batch: MagicMock
    ) -> tuple[_WorkingDummyModel, torch.optim.Optimizer, list[torch.Tensor]]:
        """Run one real forward+backward+shadow-optimizer-step probe cycle on ``_WorkingDummyModel``.

        Only ``_make_synthetic_batch`` is patched (batch construction, not the mechanic under test); the probe step
        itself must succeed before the post-step state is handed to the test.

        Returns:
            The model, the shadow optimizer, and its shadow parameters, all post-step.

        Examples:
            >>> probed(synthetic_batch)  # doctest: +SKIP
            # Needs pytest's mocker-backed synthetic_batch fixture; cannot run standalone.
        """
        model = _WorkingDummyModel()
        kwargs = _probe_step_kwargs(model, _WorkingDummyCriterion())
        assert auto_batch._probe_step(**kwargs) is True
        return model, kwargs["shadow_optimizer"], kwargs["shadow_params"]

    def test_does_not_mutate_real_model_parameters(self, probed):
        """The shadow optimizer step must update the shadow tensors, not the real model.

        A probe that corrupted a loaded pretrained checkpoint's weights before training even started would be far worse
        than the memory-underestimation bug it fixes.
        """
        model, _, _ = probed

        assert torch.equal(model.w.detach(), torch.ones(3)), "the real model parameter must be unchanged"

    def test_populates_shadow_optimizer_state(self, probed):
        """AdamW allocates exp_avg/exp_avg_sq lazily on the first step(); the shadow step must trigger that allocation.

        Pins that the step is not a no-op, e.g. from a grad-aliasing bug that left shadow_params.grad as None, which
        AdamW silently skips.
        """
        _, shadow_optimizer, shadow_params = probed

        assert len(shadow_optimizer.state) == len(shadow_params) > 0
        for shadow_p in shadow_params:
            state = shadow_optimizer.state[shadow_p]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state
            assert state["exp_avg"].shape == shadow_p.shape

    def test_state_reused_not_reallocated_across_iterations(self, probed):
        """A second probe iteration reuses the already-allocated exp_avg/exp_avg_sq tensors instead of allocating anew.

        Otherwise every probe iteration (the next candidate batch size in the same search) would pay the allocation cost
        this fix exists to move to the first iteration only.
        """
        model, shadow_optimizer, shadow_params = probed
        exp_avg_id_before = id(shadow_optimizer.state[shadow_params[0]]["exp_avg"])
        kwargs = _probe_step_kwargs(
            model,
            _WorkingDummyCriterion(),
            micro_batch_size=2,
            shadow_optimizer=shadow_optimizer,
            shadow_params=shadow_params,
        )

        auto_batch._probe_step(**kwargs)

        assert id(shadow_optimizer.state[shadow_params[0]]["exp_avg"]) == exp_avg_id_before

    def test_grad_cleared_after_step(self, probed):
        """shadow_p.grad must not outlive the shadow_optimizer.step() that consumes it.

        Leaving the alias set keeps this iteration's real-gradient tensor resident (via shadow_p.grad, its only
        remaining reference once model.zero_grad() clears real_p.grad) through the next iteration's backward() -- two
        gradient buffers alive at once instead of one.
        """
        _, _, shadow_params = probed

        for shadow_p in shadow_params:
            assert shadow_p.grad is None

    @pytest.mark.usefixtures("synthetic_batch")
    def test_grad_cleared_when_step_itself_ooms(self):
        """A CUDA OOM raised by shadow_optimizer.step() itself (not by forward/backward) must still clear shadow_p.grad.

        Otherwise the alias is the only remaining reference keeping that iteration's real gradient tensor alive,
        unreachable by torch.cuda.empty_cache(), for the rest of the search.
        """
        kwargs = _probe_step_kwargs(_WorkingDummyModel(), _WorkingDummyCriterion(), shadow_optimizer=_OomOnStep())

        ok = auto_batch._probe_step(**kwargs)

        assert ok is False
        for shadow_p in kwargs["shadow_params"]:
            assert shadow_p.grad is None


class TestBuildShadowOptimizer:
    """``_build_shadow_optimizer`` mirrors the real optimizer's configuration so the probe sees its memory footprint."""

    def test_forwards_optimizer_kwargs(self):
        """train_config.optimizer_kwargs (e.g. amsgrad=True) must reach the shadow AdamW, as configure_optimizers does.

        amsgrad adds a third max_exp_avg_sq state buffer per parameter (see torch.optim.AdamW). Without forwarding, the
        probe silently under-counts optimizer-state memory for any AdamW variant beyond the plain defaults, which is
        the dangerous direction: it can report a batch size that OOMs on the real optimizer.step().
        """
        params = [torch.nn.Parameter(torch.zeros(3))]
        optimizer, shadow_params = auto_batch._build_shadow_optimizer(
            params, lr=1e-3, weight_decay=0.0, optimizer_kwargs={"amsgrad": True}
        )
        shadow_params[0].grad = torch.ones_like(shadow_params[0])
        optimizer.step()

        assert "max_exp_avg_sq" in optimizer.state[shadow_params[0]]

    def test_optimizer_kwargs_lr_wins_without_duplicate_keyword_error(self):
        """optimizer_kwargs merge into the lr/weight_decay/fused defaults, so a caller's own lr wins instead of raising.

        A dotted-path config keeps its own lr inside optimizer_kwargs (see the docstring's merge-precedence note);
        splatting both would raise "got multiple values for keyword argument 'lr'".
        """
        params = [torch.nn.Parameter(torch.zeros(3))]

        optimizer, _ = auto_batch._build_shadow_optimizer(
            params, lr=1e-4, weight_decay=1e-4, optimizer_kwargs={"lr": 5e-2}
        )

        assert optimizer.param_groups[0]["lr"] == 5e-2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW requires CUDA")
    def test_forwards_fused(self):
        """Fused reaches the shadow AdamW so its step-time temporary memory matches the real optimizer's.

        The real optimizer's fused vs. foreach/single-tensor choice is made in RFDETRModelModule._use_fused_optimizer.
        """
        params = [torch.nn.Parameter(torch.zeros(3, device="cuda"))]
        optimizer, _ = auto_batch._build_shadow_optimizer(params, lr=1e-3, weight_decay=0.0, fused=True)

        assert optimizer.defaults["fused"] is True


class TestResolveAutoBatchConfig:
    """``resolve_auto_batch_config`` turns a probed safe micro batch into a batch size and accumulation steps."""

    def test_requires_cuda(self, mocker: MockerFixture):
        """Auto batch resolution raises when CUDA is unavailable."""
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.is_available", return_value=False)

        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            auto_batch.resolve_auto_batch_config(_model_context(torch.device("cpu")), _model_config(), _train_config())

    def test_returns_expected_values(self, mock_probe: MagicMock):
        """Auto batch resolution returns the probed micro batch and the accumulation steps reaching the target."""
        result = auto_batch.resolve_auto_batch_config(_model_context(), _model_config(), _train_config())

        assert isinstance(result, AutoBatchResult)
        assert result.safe_micro_batch == 5
        assert result.recommended_grad_accum_steps == 4
        assert result.effective_batch_size == 20
        assert result.device_name == "Fake GPU"

    @pytest.mark.parametrize(
        ("devices", "num_nodes", "target", "safe_micro_batch", "expected_accum_steps", "expected_effective_batch"),
        [
            (2, 1, 16, 8, 1, 8),
            ("2", 1, 16, 8, 1, 8),
            ("auto", 1, 16, 8, 1, 8),
            ("-1", 1, 16, 8, 1, 8),
            ("0,1", 1, 16, 8, 1, 8),
            (2, 2, 16, 2, 2, 4),
            (2, 1, 17, 2, 5, 10),
        ],
    )
    def test_scales_global_target_across_devices(
        self,
        mocker: MockerFixture,
        mock_probe: MagicMock,
        devices: int | str,
        num_nodes: int,
        target: int,
        safe_micro_batch: int,
        expected_accum_steps: int,
        expected_effective_batch: int,
    ):
        """Distribute a global nominal target across PTL device specifications.

        Rows cover an explicit count, a numeric string, ``"auto"`` and ``"-1"`` (all CUDA devices, two here), a CSV
        device list, a multi-node run, and a target that does not divide evenly (rounded up).
        """
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.device_count", return_value=2)
        mock_probe.return_value = safe_micro_batch
        train_config = _train_config(auto_batch_target_effective=target, devices=devices, num_nodes=num_nodes)

        result = auto_batch.resolve_auto_batch_config(_model_context(), _model_config(), train_config)

        assert result.recommended_grad_accum_steps == expected_accum_steps
        assert result.effective_batch_size == expected_effective_batch

    @pytest.mark.parametrize(("optimizer", "warned"), [("sgd", True), ("adamw", False)])
    def test_warns_only_when_optimizer_is_not_builtin_adamw(
        self, mocker: MockerFixture, mock_probe: MagicMock, optimizer: str, warned: bool
    ):
        """The shadow optimizer models AdamW's state memory, so any other train_config.optimizer must be surfaced.

        A name other than the built-in "adamw" (e.g. "sgd", a pytorch-optimizer name, a custom callable/dotted path --
        all supported by configure_optimizers) makes that estimate unreliable and must not be silently assumed correct;
        the default and explicit "adamw" are exactly what the probe models, and a spurious warning on the common path
        would train users to ignore it.
        """
        mock_warning = mocker.patch("rfdetr.training.auto_batch.logger.warning")

        auto_batch.resolve_auto_batch_config(_model_context(), _model_config(), _train_config(optimizer=optimizer))

        assert [call.args[1] for call in mock_warning.call_args_list] == ([optimizer] if warned else [])

    @pytest.mark.usefixtures("mock_probe")
    def test_probe_and_trainer_emit_one_legacy_amp_warning(self, tmp_path):
        """Automatic batch probing must defer the legacy AMP warning to trainer construction.

        Both stages resolve AMP from the same configs. The probe needs the resolved dtype for memory sizing, while the
        trainer is the user-facing construction point that owns the deprecation warning.
        """
        model_config = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=5, amp=False)
        train_config = TrainConfig(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            batch_size="auto",
            num_workers=0,
            tensorboard=False,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", FutureWarning)
            auto_batch.resolve_auto_batch_config(_model_context(), model_config, train_config)
            build_trainer(train_config, model_config, accelerator="cpu")

        legacy_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(warning.category, FutureWarning) and "ModelConfig.amp is deprecated" in str(warning.message)
        ]
        assert len(legacy_warnings) == 1

    @pytest.mark.parametrize(
        ("optimizer", "optimizer_kwargs", "expected_kwargs"),
        [
            pytest.param("adamw", {"amsgrad": True}, {"amsgrad": True}, id="builtin-adamw-forwarded"),
            pytest.param("torch.optim.AdamW", {"amsgrad": True}, {"amsgrad": True}, id="dotted-path-adamw-forwarded"),
            pytest.param("sgd", {"momentum": 0.9}, None, id="non-adamw-not-forwarded"),
        ],
    )
    def test_forwards_optimizer_kwargs_only_for_adamw(
        self, mocker: MockerFixture, mock_probe: MagicMock, optimizer, optimizer_kwargs, expected_kwargs
    ):
        """optimizer_kwargs reach the AdamW-shaped shadow optimizer for any AdamW spelling, but never for another
        optimizer.

        Forwarding another optimizer's kwargs (e.g. SGD momentum) into the AdamW-only shadow optimizer would raise a
        TypeError; that optimizer already gets the mismatched-optimizer warning instead of an exact estimate.
        """
        mocker.patch("rfdetr.training.auto_batch.logger.warning")
        train_config = _train_config(optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)

        auto_batch.resolve_auto_batch_config(_model_context(), _model_config(), train_config)

        assert mock_probe.call_args.kwargs["optimizer_kwargs"] == expected_kwargs

    @pytest.mark.parametrize(
        ("fused_optimizer", "amp_dtype", "expected_fused"),
        [
            (True, "auto", True),
            (True, "fp16", False),
            (False, "auto", False),
        ],
    )
    def test_fused_follows_real_optimizer_conditions(
        self, mocker: MockerFixture, mock_probe: MagicMock, fused_optimizer, amp_dtype, expected_fused
    ):
        """The shadow optimizer is fused only when the real run would be: bf16-mixed, builtin adamw,
        fused_optimizer=True.

        Mirrors RFDETRModelModule._use_fused_optimizer: fused AdamW is only safe under bf16-mixed (fp16 resolves to
        '16-mixed') and model_config.fused_optimizer=False is honored, so the probe's step-time memory matches the run.
        """
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.get_device_capability", return_value=(8, 0))
        model_config = _model_config(amp=True, fused_optimizer=fused_optimizer)
        train_config = _train_config(optimizer="adamw", amp_dtype=amp_dtype)

        auto_batch.resolve_auto_batch_config(_model_context(), model_config, train_config)

        assert mock_probe.call_args.kwargs["optimizer_fused"] is expected_fused

    @pytest.mark.parametrize(
        ("amp_dtype", "expected_dtype", "expected_fused"),
        [
            pytest.param("auto", torch.float16, False, id="auto-trains-fp16"),
            pytest.param("bf16", torch.bfloat16, True, id="explicit-bf16-is-kept"),
        ],
    )
    def test_probe_dtype_on_emulated_bf16_gpu(
        self,
        mocker: MockerFixture,
        mock_probe: MagicMock,
        amp_dtype: str,
        expected_dtype: torch.dtype,
        expected_fused: bool,
    ) -> None:
        """On a GPU with only emulated bfloat16 (T4, V100) the probe measures memory in the dtype the run trains in.

        That is fp16 for amp_dtype='auto', and bf16 for an explicit amp_dtype='bf16', which is honoured there.
        ``torch.cuda.is_bf16_supported()`` counts emulation by default and returns True on such a GPU, so it is mocked
        with that real signature rather than a flat return value.
        """
        mocker.patch(
            "rfdetr.training.auto_batch.torch.cuda.is_bf16_supported",
            side_effect=lambda including_emulation=True: including_emulation,
        )
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.get_device_capability", return_value=(7, 5))
        model_config = _model_config(amp=True, fused_optimizer=True)
        train_config = _train_config(optimizer="adamw", amp_dtype=amp_dtype)

        auto_batch.resolve_auto_batch_config(_model_context(), model_config, train_config)

        probed_dtype = mock_probe.call_args.kwargs["autocast_dtype"]
        assert probed_dtype is expected_dtype, (
            f"probe ran under {probed_dtype}; amp_dtype={amp_dtype!r} trains in {expected_dtype}"
        )
        fused = mock_probe.call_args.kwargs["optimizer_fused"]
        assert fused is expected_fused, (
            f"shadow optimizer fused={fused}, expected {expected_fused} for {expected_dtype}"
        )

    @pytest.mark.parametrize(
        ("capabilities", "config_devices", "devices", "expected_dtype"),
        [
            pytest.param({0: (8, 0), 1: (7, 5)}, 2, None, torch.float16, id="a100-and-t4-both-train"),
            pytest.param({0: (8, 0), 1: (7, 5)}, 1, [1], torch.float16, id="trains-on-the-t4-only"),
            pytest.param({0: (7, 5), 1: (8, 0)}, 1, [1], torch.bfloat16, id="trains-on-the-a100-only"),
        ],
    )
    def test_probe_dtype_follows_every_training_gpu(
        self,
        mocker: MockerFixture,
        mock_probe: MagicMock,
        capabilities: dict[int, tuple[int, int]],
        config_devices: int,
        devices: list[int] | None,
        expected_dtype: torch.dtype,
    ) -> None:
        """With amp_dtype='auto' the probe uses bf16 only when every GPU the run trains on has native bf16.

        That is as the trainer decides it, and not just the GPU the probe runs on. The probe model sits on cuda:0 in
        every case. The run trains on ``devices`` when given (what ``train(device="cuda:1")`` forwards) and on
        ``train_config.devices`` otherwise.
        """
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.device_count", return_value=len(capabilities))
        mocker.patch(
            "rfdetr.training.auto_batch.torch.cuda.get_device_capability", side_effect=_capability_lookup(capabilities)
        )
        model_config = _model_config(amp=True, fused_optimizer=True)
        train_config = _train_config(optimizer="adamw", amp_dtype="auto", devices=config_devices)
        # devices=None means the caller did not pass it, so train_config.devices decides.
        device_kwargs = {} if devices is None else {"devices": devices}

        auto_batch.resolve_auto_batch_config(
            _model_context(torch.device("cuda", 0)), model_config, train_config, **device_kwargs
        )

        probed_dtype = mock_probe.call_args.kwargs["autocast_dtype"]
        assert probed_dtype is expected_dtype, f"probe ran under {probed_dtype}, the run trains in {expected_dtype}"
        fused = mock_probe.call_args.kwargs["optimizer_fused"]
        assert fused is (expected_dtype is torch.bfloat16), f"shadow optimizer fused={fused} for {expected_dtype}"

    def test_explicit_bf16_follows_the_training_gpu(self, mocker: MockerFixture, mock_probe: MagicMock) -> None:
        """An explicit amp_dtype='bf16' probes in bf16 on the strength of the training GPU, not the probe's own device.

        The host has a pre-Ampere GPU at index 0 and an A100 at index 1, on a CUDA build without bfloat16 emulation — so
        ``torch.cuda.is_bf16_supported()``, which only ever reads the current device, says no. A run pinned to the A100
        with ``devices=[1]`` trains in bf16, so the probe has to measure memory in bf16 too.
        """
        capabilities = {0: (7, 5), 1: (8, 0)}
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.device_count", return_value=len(capabilities))
        mocker.patch("rfdetr.training.auto_batch.torch.cuda.is_bf16_supported", return_value=False)
        mocker.patch(
            "rfdetr.training.auto_batch.torch.cuda.get_device_capability", side_effect=_capability_lookup(capabilities)
        )
        mocker.patch("rfdetr.training.auto_batch.torch.version.hip", None)
        model_config = _model_config(amp=True, fused_optimizer=True)
        train_config = _train_config(optimizer="adamw", amp_dtype="bf16", devices=1)

        auto_batch.resolve_auto_batch_config(
            _model_context(torch.device("cuda", 0)), model_config, train_config, devices=[1]
        )

        probed_dtype = mock_probe.call_args.kwargs["autocast_dtype"]
        assert probed_dtype is torch.bfloat16, f"probe ran under {probed_dtype}; the run trains on the A100 in bf16"

    @pytest.mark.parametrize(("pad_targets_to", "expected_probe_targets"), [(None, 100), (8, 8)])
    def test_probes_the_padded_target_count(
        self, mock_probe: MagicMock, pad_targets_to: int | None, expected_probe_targets: int
    ):
        """Fixed-size padding turns the per-image target count from a guess into a fact.

        The probe sizes the memory envelope by synthesizing ``max_targets_per_image`` targets per image. Without padding
        (``None``) that has to be a pessimistic ceiling (100 here), but with ``pad_targets_to`` set it is exactly what
        every image will carry -- so probing the ceiling instead would reject batch sizes that actually fit, and would
        understate the envelope for a pad size above the ceiling.
        """
        train_config = _train_config(auto_batch_max_targets_per_image=100, pad_targets_to=pad_targets_to)

        auto_batch.resolve_auto_batch_config(_model_context(), _model_config(segmentation_head=False), train_config)

        assert mock_probe.call_args.kwargs["max_targets_per_image"] == expected_probe_targets


def test_train_auto_batch_ensures_model_on_device_before_resolve(mocker: MockerFixture, tmp_path) -> None:
    """Model weights must be moved before resolve_auto_batch_config when batch_size='auto'."""
    auto_result = SimpleNamespace(safe_micro_batch=4, recommended_grad_accum_steps=1, effective_batch_size=4)
    mocker.patch("rfdetr.detr.is_main_process", return_value=False)
    mock_resolve = mocker.patch("rfdetr.training.auto_batch.resolve_auto_batch_config")
    mocker.patch("rfdetr.training.build_trainer")
    mocker.patch("rfdetr.training.RFDETRDataModule")
    mocker.patch("rfdetr.training.RFDETRModelModule")
    mock_move = mocker.patch("rfdetr.detr._move_model_context_to_device")
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
        devices=None,
    )
    assert call_order == ["ensure", "resolve"]


@pytest.mark.gpu
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
