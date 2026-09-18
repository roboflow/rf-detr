# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Contract tests for Transformer Engine-backed CUDA graph training."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.training import build_trainer
from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor

_TRANSFORMER_ENGINE_INSTALLED = importlib.util.find_spec("transformer_engine") is not None
_FP8_CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)

_skip_without_fp8_cuda = pytest.mark.skipif(
    not _FP8_CUDA_AVAILABLE, reason="requires FP8-capable CUDA hardware (compute capability >= 8.9)"
)
_skip_without_transformer_engine = pytest.mark.skipif(
    not _TRANSFORMER_ENGINE_INSTALLED, reason="requires transformer-engine"
)


class _TinyGraphableModel(nn.Module):
    """Small detector-shaped model with input-dependent output and gradients."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3, device=device)

    def forward(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Project an image mean using RF-DETR's detector call signature."""
        del targets
        return {"pred": self.projection(samples.tensors.mean(dim=(-2, -1)))}


class _Fp8TinyGraphableModel(nn.Module):
    """Small model whose Transformer Engine linear dimensions satisfy FP8 alignment requirements."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.projection = nn.Linear(16, 16, device=device)

    def forward(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Project the 16-channel image mean through an FP8-compatible linear layer."""
        del targets
        return {"pred": self.projection(samples.tensors.mean(dim=(-2, -1)))}


class _FakeTransformerEngine:
    """CPU-only stand-in recording the public Transformer Engine boundary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.capture_error: Exception | None = None

    def make_graphed_callables(
        self,
        modules: nn.Module,
        sample_args: tuple[Tensor, Tensor],
        num_warmup_iters: int = 3,
        allow_unused_input: bool = False,
        sample_kwargs: dict[str, Any] | None = None,
        enabled: bool | None = None,
        calibrating: bool | None = None,
        recipe: object | None = None,
        cache_quantized_params: bool | None = None,
        clone_param_grads_on_return: bool = True,
    ) -> nn.Module:
        """Record a Transformer Engine 2.19 capture call and return its graphable input.

        Examples:
            >>> fake = _FakeTransformerEngine()
            >>> fake.make_graphed_callables(nn.Identity(), (torch.ones(1), torch.ones(1)))
            Identity()
        """
        self.calls.append(
            {
                "modules": modules,
                "sample_args": sample_args,
                "num_warmup_iters": num_warmup_iters,
                "allow_unused_input": allow_unused_input,
                "sample_kwargs": sample_kwargs,
                "enabled": enabled,
                "calibrating": calibrating,
                "recipe": recipe,
                "cache_quantized_params": cache_quantized_params,
                "clone_param_grads_on_return": clone_param_grads_on_return,
            }
        )
        if self.capture_error is not None:
            raise self.capture_error
        return modules


@pytest.fixture
def fake_transformer_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeTransformerEngine:
    """Install a Transformer Engine 2.19-shaped module without its CUDA extension.

    Examples:
        >>> fake_transformer_engine(monkeypatch)  # doctest: +SKIP
        # Needs pytest's monkeypatch fixture; cannot run standalone.
    """
    fake = _FakeTransformerEngine()
    package = ModuleType("transformer_engine")
    package.__path__ = []  # type: ignore[attr-defined]
    pytorch = ModuleType("transformer_engine.pytorch")
    pytorch.make_graphed_callables = fake.make_graphed_callables  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformer_engine", package)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)
    return fake


def _samples(value: float = 1.0, *, device: torch.device | str = "cpu", batch_size: int = 2) -> NestedTensor:
    """Build a fixed-shape detector input without any dataset or network boundary.

    Examples:
        >>> tuple(_samples(batch_size=1).tensors.shape)
        (1, 4, 3, 3)
    """
    return NestedTensor(
        torch.full((batch_size, 4, 3, 3), value, device=device),
        torch.zeros((batch_size, 3, 3), dtype=torch.bool, device=device),
    )


def _fp8_samples(value: float, *, device: torch.device | str, batch_size: int = 16) -> NestedTensor:
    """Build a Transformer Engine-aligned bfloat16 detector input.

    Examples:
        >>> tuple(_fp8_samples(1.0, device="cpu").tensors.shape)
        (16, 16, 1, 1)
    """
    return NestedTensor(
        torch.full((batch_size, 16, 1, 1), value, dtype=torch.bfloat16, device=device),
        torch.zeros((batch_size, 1, 1), dtype=torch.bool, device=device),
    )


class TestTransformerEngineCaptureBoundary:
    """CPU tests for Transformer Engine selection and fatal-capture behavior."""

    def test_runner_fp8_capture_passes_documented_options(
        self, fake_transformer_engine: _FakeTransformerEngine
    ) -> None:
        """FP8 capture passes Transformer Engine's current safe ownership options.

        Prevents deprecated keyword use, cached quantized weights, and static-gradient aliasing.
        """
        recipe = object()
        runner = CudaGraphTrainingRunner(_TinyGraphableModel(), num_warmup_iters=5, fp8_recipe=recipe)

        output = runner(_samples())

        assert output["pred"].shape == (2, 3)
        assert len(fake_transformer_engine.calls) == 1
        call = fake_transformer_engine.calls[0]
        assert isinstance(call["modules"], nn.Module)
        assert tuple(tensor.shape for tensor in call["sample_args"]) == ((2, 4, 3, 3), (2, 3, 3))
        assert {
            name: call[name]
            for name in (
                "num_warmup_iters",
                "allow_unused_input",
                "enabled",
                "recipe",
                "cache_quantized_params",
                "clone_param_grads_on_return",
            )
        } == {
            "num_warmup_iters": 5,
            "allow_unused_input": True,
            "enabled": True,
            "recipe": recipe,
            "cache_quantized_params": False,
            "clone_param_grads_on_return": True,
        }

    @pytest.mark.skipif(
        _TRANSFORMER_ENGINE_INSTALLED,
        reason="transformer_engine is installed in this environment; cannot exercise the absent-package path",
    )
    def test_runner_fp8_without_transformer_engine_raises_module_not_found_error(self) -> None:
        """A genuinely absent Transformer Engine fails construction with a clear import error.

        No ``sys.modules`` fake is installed here (unlike every other test in this class): this checks the real import
        at ``cuda_graph_step.py``'s ``fp8_recipe is not None`` branch, which is otherwise never exercised because every
        other FP8 test fakes the module.
        """
        with pytest.raises(ModuleNotFoundError, match="transformer_engine"):
            CudaGraphTrainingRunner(_TinyGraphableModel(), fp8_recipe=object())

    def test_runner_old_transformer_engine_api_rejects_before_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An old Transformer Engine API fails before CUDA capture begins.

        Prevents a late capture-time ``TypeError`` that cannot safely fall back to eager execution.
        """
        capture_called = False

        def old_make_graphed_callables(
            modules: nn.Module,
            sample_args: tuple[Tensor, Tensor],
            *,
            enabled: bool,
            recipe: object,
        ) -> nn.Module:
            """Represent a version before required safe capture options existed.

            Examples:
                >>> old_make_graphed_callables(nn.Identity(), (torch.ones(1), torch.ones(1)), enabled=True, recipe=1)
                Identity()
            """
            nonlocal capture_called
            capture_called = True
            return modules

        package = ModuleType("transformer_engine")
        package.__path__ = []  # type: ignore[attr-defined]
        pytorch = ModuleType("transformer_engine.pytorch")
        pytorch.make_graphed_callables = old_make_graphed_callables  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformer_engine", package)
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)

        with pytest.raises(RuntimeError, match="Transformer Engine 2.19"):
            CudaGraphTrainingRunner(_TinyGraphableModel(), fp8_recipe=object())

        assert not capture_called

    def test_runner_fp8_first_capture_with_existing_gradients_rejects_before_backend(
        self, fake_transformer_engine: _FakeTransformerEngine
    ) -> None:
        """FP8 initial capture rejects live parameter gradients before entering the backend."""
        model = _TinyGraphableModel()
        model.projection.weight.grad = torch.ones_like(model.projection.weight)
        runner = CudaGraphTrainingRunner(model, fp8_recipe=object())

        with pytest.raises(RuntimeError, match="no existing parameter gradients"):
            runner(_samples())

        assert fake_transformer_engine.calls == []

    def test_runner_fp8_replay_ignores_pending_gradients_unlike_first_capture(
        self, fake_transformer_engine: _FakeTransformerEngine
    ) -> None:
        """KNOWN GAP: replaying a cached FP8 signature has no guard against un-zeroed gradients.

        ``CudaGraphTrainingRunner.__call__`` only checks for existing parameter gradients inside
        its ``if graphed is None`` branch (``cuda_graph_step.py`` around the FP8 capture guard), so
        the check in :meth:`test_runner_fp8_first_capture_with_existing_gradients_rejects_before_backend`
        protects only the *first* capture of a signature. ``_own_accumulated_gradients`` -- which
        clones live gradients before replay for the non-FP8 backend -- is also skipped whenever
        ``fp8_recipe`` is set. So replaying an already-cached FP8 signature with a pending, non-
        zeroed gradient (e.g. multiple microbatches accumulated without an intervening
        ``zero_grad()``) succeeds silently instead of raising, unlike the identical situation before
        the first capture. Production safety today depends entirely on the caller-side check in
        ``RFDETRModelModule.on_train_start`` (``module_model.py``, FP8 + accumulation guard), which
        direct construction of the runner -- as this whole test file does -- bypasses entirely. This
        test documents today's silent, unguarded behavior; a future internal guard should make the
        replay below raise instead.
        """
        runner = CudaGraphTrainingRunner(_TinyGraphableModel(), fp8_recipe=object())
        runner(_samples())  # First call captures and caches this execution signature.

        runner.inner.projection.weight.grad = torch.ones_like(runner.inner.projection.weight)

        # No zero_grad() before this replay of the SAME cached signature: unlike the pre-capture
        # check above, nothing here rejects the pending gradient.
        output = runner(_samples())

        assert output["pred"].shape == (2, 3)
        assert len(runner._graphed_cache) == 1
        assert len(fake_transformer_engine.calls) == 1

    def test_runner_fp8_signature_change_after_first_capture_rejects(
        self, fake_transformer_engine: _FakeTransformerEngine
    ) -> None:
        """Transformer Engine capture remains fixed-signature until scaling parity is proven."""
        runner = CudaGraphTrainingRunner(_TinyGraphableModel(), fp8_recipe=object())
        runner(_samples(batch_size=2))

        with pytest.raises(RuntimeError, match="one fixed execution signature"):
            runner(_samples(batch_size=1))

        assert len(runner._graphed_cache) == 1
        assert len(fake_transformer_engine.calls) == 1

    def test_runner_fp8_evaluation_bypasses_capture(self, fake_transformer_engine: _FakeTransformerEngine) -> None:
        """Evaluation bypasses capture even when the training runner is configured for FP8."""
        model = _TinyGraphableModel()
        model.eval()
        runner = CudaGraphTrainingRunner(model, fp8_recipe=object())

        output = runner(_samples(2.0))

        torch.testing.assert_close(output["pred"], model(_samples(2.0))["pred"])
        assert fake_transformer_engine.calls == []
        assert runner._graphed_cache == {}

    def test_runner_fp8_capture_failure_restores_cache_and_stays_fatal(
        self, fake_transformer_engine: _FakeTransformerEngine
    ) -> None:
        """A Transformer Engine capture error restores caller state and stays fatal."""
        cache_settings: list[bool] = []
        fake_transformer_engine.capture_error = RuntimeError("capture failed")
        runner = CudaGraphTrainingRunner(_TinyGraphableModel(), fp8_recipe=object())

        with (
            pytest.raises(RuntimeError, match="Training cannot safely continue") as exc,
            pytest.MonkeyPatch.context() as patched,
        ):
            patched.setattr(torch, "is_autocast_enabled", lambda *_args: True)
            patched.setattr(torch, "is_autocast_cache_enabled", lambda: True)
            patched.setattr(torch, "set_autocast_cache_enabled", cache_settings.append)
            runner(_samples())

        assert str(exc.value.__cause__) == "capture failed"
        assert cache_settings == [False, True]
        assert len(fake_transformer_engine.calls) == 1
        assert runner._graphed_cache == {}

    def test_runner_bf16_capture_uses_pytorch_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No FP8 recipe preserves PyTorch capture without importing Transformer Engine."""
        pytorch_calls: list[dict[str, Any]] = []

        def fake_pytorch_capture(modules: nn.Module, sample_args: tuple[Tensor, Tensor], **kwargs: Any) -> nn.Module:
            """Record the existing PyTorch capture call for this focused compatibility test.

            Examples:
                >>> fake_pytorch_capture(nn.Identity(), (torch.ones(1), torch.ones(1)))
                Identity()
            """
            pytorch_calls.append({"modules": modules, "sample_args": sample_args, **kwargs})
            return modules

        monkeypatch.setattr(torch.cuda, "make_graphed_callables", fake_pytorch_capture)
        output = CudaGraphTrainingRunner(_TinyGraphableModel())(_samples())

        assert output["pred"].shape == (2, 3)
        assert len(pytorch_calls) == 1
        assert pytorch_calls[0]["allow_unused_input"] is True


class _FakeFp8Dataset(torch.utils.data.Dataset):
    """Synthetic (3, 384, 384) detection samples for a real ``Trainer.fit()`` FP8 run.

    384x384 matches the resolution already proven against the real Nano model's forward in
    :meth:`TestTransformerEngineCaptureGPU.test_runner_fp8_nano_train_eval_train_reuses_one_capture`, so a real
    ``Trainer.fit()`` run gets the same shape a real capture already handles correctly.
    """

    def __init__(self, length: int = 2) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        image = torch.rand(3, 384, 384)
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([384, 384]),
            "size": torch.tensor([384, 384]),
        }
        return image, target


@_skip_without_transformer_engine
@_skip_without_fp8_cuda
class TestTransformerEngineCaptureGPU:
    """Unmocked CUDA checks; intentionally skipped on CPU-only development machines."""

    @pytest.mark.gpu
    def test_runner_fp8_replay_matches_eager_outputs_gradients_and_updates(self) -> None:
        """Real Transformer Engine replay preserves changing outputs, gradients, and updates."""
        from pytorch_lightning.plugins.precision import TransformerEnginePrecision

        torch.manual_seed(0)
        eager = _Fp8TinyGraphableModel("cuda")
        with torch.no_grad():
            eager.projection.weight.fill_(0.125)
            eager.projection.bias.zero_()
        graphed = copy.deepcopy(eager)
        eager_precision = TransformerEnginePrecision(weights_dtype=torch.bfloat16)
        graphed_precision = TransformerEnginePrecision(weights_dtype=torch.bfloat16)
        eager = eager_precision.convert_module(eager)
        graphed = graphed_precision.convert_module(graphed)
        assert any("transformer_engine.pytorch" in type(module).__module__ for module in graphed.modules())
        eager_optimizer = torch.optim.SGD(eager.parameters(), lr=0.1)
        graphed_optimizer = torch.optim.SGD(graphed.parameters(), lr=0.1)
        runner = CudaGraphTrainingRunner(graphed, fp8_recipe=graphed_precision.recipe)

        # This low-precision smoke tolerance is not a scale-state equivalence claim. Controlled initialization and
        # the learning rate produce representable BF16 updates; separately reject a silently skipped optimizer step.
        fp8_close = {"rtol": 2e-2, "atol": 2e-2}
        for amplitude in (1.5, 2.5, 0.25):
            eager_before = eager.projection.weight.detach().clone()
            graphed_before = graphed.projection.weight.detach().clone()
            with eager_precision.forward_context():
                eager_output = eager(_fp8_samples(amplitude, device="cuda"))["pred"]
                eager_loss = eager_output.square().mean()
            with graphed_precision.forward_context():
                graphed_output = runner(_fp8_samples(amplitude, device="cuda"))["pred"]
                graphed_loss = graphed_output.square().mean()
            eager_loss.backward()
            graphed_loss.backward()

            torch.testing.assert_close(graphed_output.float(), eager_output.float(), **fp8_close)
            torch.testing.assert_close(
                graphed.projection.weight.grad.float(), eager.projection.weight.grad.float(), **fp8_close
            )
            eager_optimizer.step()
            graphed_optimizer.step()
            assert not torch.equal(eager.projection.weight, eager_before)
            assert not torch.equal(graphed.projection.weight, graphed_before)
            torch.testing.assert_close(
                (graphed.projection.weight - graphed_before).float(),
                (eager.projection.weight - eager_before).float(),
                **fp8_close,
            )
            eager_optimizer.zero_grad(set_to_none=True)
            graphed_optimizer.zero_grad(set_to_none=True)

        assert len(runner._graphed_cache) == 1

    @pytest.mark.gpu
    def test_runner_fp8_nano_train_eval_train_reuses_one_capture(self) -> None:
        """A real Nano runner evaluates eagerly and resumes Transformer Engine replay without recapture.

        EMA and checkpoint serialization remain outside this runner-level test.
        """
        from pytorch_lightning.plugins.precision import TransformerEnginePrecision

        torch.manual_seed(0)
        model_config = RFDETRNanoConfig(pretrain_weights=None, num_classes=3, device="cuda")
        train_config = TrainConfig(dataset_dir="unused", drop_path=0.0)
        model = build_model_from_config(model_config, train_config).cuda()
        precision = TransformerEnginePrecision(weights_dtype=torch.bfloat16)
        model = precision.convert_module(model)
        assert any("transformer_engine.pytorch" in type(module).__module__ for module in model.modules())
        runner = CudaGraphTrainingRunner(model, fp8_recipe=precision.recipe)
        samples = NestedTensor(
            torch.randn(4, 3, 384, 384, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(4, 384, 384, dtype=torch.bool, device="cuda"),
        )

        model.train()
        with precision.forward_context():
            training_output = runner(samples)
            training_loss = training_output["pred_logits"].square().mean()
        training_loss.backward()
        assert torch.isfinite(training_output["pred_logits"]).all()
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(), precision.forward_context():
            evaluation_output = runner(samples)
        assert torch.isfinite(evaluation_output["pred_logits"]).all()
        model.train()
        with precision.forward_context():
            replayed_output = runner(samples)
            replayed_loss = replayed_output["pred_logits"].square().mean()
        replayed_loss.backward()

        assert torch.isfinite(replayed_output["pred_logits"]).all()
        first_parameter = next(model.parameters())
        assert first_parameter.grad is not None
        assert torch.isfinite(first_parameter.grad).all()
        assert len(runner._graphed_cache) == 1

    @pytest.mark.gpu
    def test_fit_converts_to_transformer_engine_before_first_capture(self, tmp_path: Path) -> None:
        """A real ``Trainer.fit()`` converts modules to Transformer Engine layers before FP8 capture fires.

        Unlike the two tests above, this drives the FULL ``RFDETRModelModule.on_train_start()`` ->
        ``CudaGraphTrainingRunner`` path through a real ``pytorch_lightning.Trainer.fit()`` call -- not manual
        ``precision.convert_module()`` plus direct runner construction -- so it proves Lightning's actual internal hook
        order (convert, then capture) instead of an order this test file merely assumes. A CPU test cannot exercise this
        ordering: ``RFDETRModelModule. on_train_start``'s device-type gate (``module_model.py``) disables CUDA graphs
        entirely -- before ``CudaGraphTrainingRunner`` is ever constructed -- whenever the module is not on CUDA. Only
        dataset I/O is faked here (a filesystem/network boundary this test does not own); the real Nano model, real
        criterion, real Transformer Engine precision plugin, and real capture call all run unmocked, matching this
        class's other real-CUDA tests.

        Authored but unverified in this environment (no GPU available here); pending CI/GPU hardware.
        """
        import transformer_engine.pytorch as te_pytorch

        conversion_seen_before_first_capture: list[bool] = []
        real_make_graphed_callables = te_pytorch.make_graphed_callables

        def spying_make_graphed_callables(modules: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
            """Record whether Transformer Engine layers already replaced plain ``Linear`` at capture time."""
            conversion_seen_before_first_capture.append(
                any("transformer_engine.pytorch" in type(module).__module__ for module in modules.modules())
            )
            return real_make_graphed_callables(modules, *args, **kwargs)

        model_config = RFDETRNanoConfig(pretrain_weights=None, num_classes=3, device="cuda", cuda_graphs=True)
        train_config = TrainConfig(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=1,
            batch_size=1,
            amp_dtype="fp8",
            drop_path=0.0,
            multi_scale=False,
            grad_accum_steps=1,
            square_resize_div_64=True,
            num_workers=0,
            tensorboard=False,
            use_ema=False,
            run_test=False,
        )

        with (
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeFp8Dataset()),
            patch("transformer_engine.pytorch.make_graphed_callables", side_effect=spying_make_graphed_callables),
        ):
            module = RFDETRModelModule(model_config, train_config)
            datamodule = RFDETRDataModule(model_config, train_config)
            trainer = build_trainer(
                train_config,
                model_config,
                accelerator="cuda",
                devices=1,
                fast_dev_run=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(module, datamodule=datamodule)

        assert conversion_seen_before_first_capture == [True]
