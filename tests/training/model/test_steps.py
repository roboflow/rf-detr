# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for RFDETRModelModule training-loop hooks: batch start, training_step, accumulation, clipping and LR
stepping."""

import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn

from rfdetr.config import RFDETRNanoConfig, RFDETRSmallConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.training.cuda_graph_step import CudaGraphTrainingRunner
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from tests.training.helpers import (
    _attach_trainer,
    _base_model_config,
    _base_train_config,
    _build_module,
    _FakeCriterion,
    _FakeDataset,
    _make_batch,
    _make_param_dicts,
    _TinyModel,
)
from tests.training.helpers import _fake_postprocess as _helpers_fake_postprocess


class _StepHookOptimizer:
    """Optimizer double that fires ``on_before_optimizer_step`` from ``step()``, like ``LightningOptimizer``.

    Real ``LightningOptimizer.step()`` routes through ``Precision.optimizer_step``, which invokes the
    ``on_before_optimizer_step`` hook before running the wrapped optimizer's own ``step()``. This double reproduces just
    that ordering so tests can exercise the module's hook wiring without standing up the full Lightning
    precision/strategy stack.
    """

    def __init__(self, module, optimizer):
        self._module = module
        self._optimizer = optimizer

    @property
    def param_groups(self):
        """Proxy the wrapped optimizer's parameter groups."""
        return self._optimizer.param_groups

    def step(self, *args, **kwargs):
        """Fire the before-step hook, then delegate to the wrapped optimizer's ``step()``."""
        self._module.on_before_optimizer_step(self)
        self._optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """Delegate to the wrapped optimizer's ``zero_grad()``."""
        self._optimizer.zero_grad(*args, **kwargs)


class TestMultiScaleBatchStart:
    """on_train_batch_start multi-scale resize picks a step-deterministic scale without clobbering global RNG."""

    def _build_multi_scale_module(self, tmp_path, global_step):
        """Return a module configured for multi-scale with a stubbed trainer at the given global step."""
        tc = _base_train_config(tmp_path, multi_scale="per-batch")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(global_step=global_step)
        return module

    def test_scale_choice_is_deterministic_per_step(self, tmp_path):
        """The same global step must resize the batch to the same scale regardless of batch contents."""
        module = self._build_multi_scale_module(tmp_path, global_step=7)

        batch_a = _make_batch(h=64, w=64)
        module.on_train_batch_start(batch_a, 0)
        size_a = tuple(batch_a[0].tensors.shape[-2:])

        batch_b = _make_batch(h=64, w=64)
        module.on_train_batch_start(batch_b, 0)
        size_b = tuple(batch_b[0].tensors.shape[-2:])

        assert size_a == size_b

    def test_does_not_perturb_global_rng(self, tmp_path):
        """Scale selection must use a step-local generator and leave the process-global RNG untouched."""
        module = self._build_multi_scale_module(tmp_path, global_step=3)

        random.seed(42)
        expected = [random.random() for _ in range(3)]

        random.seed(42)
        module.on_train_batch_start(_make_batch(h=64, w=64), 0)
        actual = [random.random() for _ in range(3)]

        assert actual == expected


class _ScalarLossModel(nn.Module):
    """Tiny model exposing one scalar parameter for gradient-scaling tests."""

    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, samples, targets=None):
        return {"loss_scale": self.value}


class _BoxNormalizedCriterion:
    """Criterion with controllable per-target loss numerators and box counts."""

    weight_dict = {"loss_ce": 1.0}
    supports_loss_normalizer_override: bool = True

    def num_boxes_for_targets(self, outputs, targets):
        return torch.as_tensor(
            sum(int(target["labels"].numel()) for target in targets),
            dtype=torch.float32,
            device=outputs["loss_scale"].device,
        ).clamp(min=1.0)

    def __call__(self, outputs, targets, num_boxes=None):
        denominator = self.num_boxes_for_targets(outputs, targets) if num_boxes is None else num_boxes
        numerator = outputs["loss_scale"] * sum(target["loss_numerator"] for target in targets)
        return {"loss_ce": numerator / denominator}


class TestOnFitStart:
    """Tests for on_fit_start() seeding behavior."""

    @pytest.mark.parametrize("rank, expected_seed", [(0, 7), (2, 9)])
    def test_seed_is_offset_by_rank(self, mocker, base_train_config, build_module, rank, expected_seed):
        """seed_everything(seed + global_rank) is called, so each DDP worker seeds uniquely (rank 0 keeps the seed)."""
        mock_seed = mocker.patch("rfdetr.training.module_model.seed_everything")
        tc = base_train_config(seed=7)
        module, _, _, _ = build_module(train_config=tc)

        mocker.patch.object(type(module), "global_rank", new_callable=PropertyMock, return_value=rank)
        module.on_fit_start()

        mock_seed.assert_called_once_with(expected_seed, workers=True)

    def test_seed_skipped_when_none(self, mocker, base_train_config, build_module):
        """No seed means on_fit_start should not call seed_everything."""
        mock_seed = mocker.patch("rfdetr.training.module_model.seed_everything")
        tc = base_train_config(seed=None)
        module, _, _, _ = build_module(train_config=tc)

        module.on_fit_start()

        mock_seed.assert_not_called()


class TestOnTrainBatchStart:
    """Tests for on_train_batch_start().

    Covers multi-scale interpolation of NestedTensor inputs and verifies regularization scheduling is delegated to
    DropPathCallback.
    """

    def _setup_module(
        self,
        tmp_path,
        multi_scale=False,
    ):
        tc = _base_train_config(tmp_path, multi_scale=multi_scale)
        module, fake_model, _, _ = _build_module(train_config=tc)

        _attach_trainer(module, global_step=0)

        return module, fake_model

    @pytest.mark.parametrize("method_name, step", [("update_drop_path", 1), ("update_dropout", 2)])
    def test_regularization_not_applied_in_module_hook(self, method_name, step, tmp_path):
        """Drop-path and dropout scheduling belong to DropPathCallback, not the module hook."""
        module, fake_model = self._setup_module(tmp_path)
        module._trainer.global_step = step

        module.on_train_batch_start(_make_batch(), batch_idx=step)

        getattr(fake_model, method_name).assert_not_called()

    @pytest.mark.parametrize(
        "method_name",
        [
            pytest.param("update_drop_path", id="drop-path"),
            pytest.param("update_dropout", id="dropout"),
        ],
    )
    def test_update_not_called_when_schedule_is_none(self, method_name, tmp_path):
        """Without a schedule, neither update_drop_path nor update_dropout must be called."""
        module, fake_model = self._setup_module(tmp_path)

        module.on_train_batch_start(_make_batch(), batch_idx=0)

        getattr(fake_model, method_name).assert_not_called()

    def test_multi_scale_resize_mutates_nested_tensor(self, tmp_path):
        """Multi-scale training must resize the input tensor to a square resolution."""
        module, _ = self._setup_module(tmp_path, multi_scale="per-batch")
        module._trainer.global_step = 0
        samples, targets = _make_batch(batch_size=2, h=16, w=16)

        module.on_train_batch_start((samples, targets), batch_idx=0)

        new_h, new_w = samples.tensors.shape[2], samples.tensors.shape[3]
        assert new_h == new_w, "Multi-scale should produce square outputs"

    def test_multi_scale_skipped_when_per_sample(self, tmp_path):
        """Per-sample scaling happens in the dataset, so the batch-level resize must be a no-op."""
        module, _ = self._setup_module(tmp_path, multi_scale="per-sample")
        samples, targets = _make_batch(batch_size=2, h=16, w=16)
        original_shape = samples.tensors.shape

        module.on_train_batch_start((samples, targets), batch_idx=0)

        assert samples.tensors.shape == original_shape


class TestTrainingStep:
    """Tests for training_step().

    Covers weighted loss aggregation, per-loss logging under the train/ prefix, prog_bar visibility, scalar tensor
    output, and that losses absent from weight_dict are excluded from the total.
    """

    def _run_step(
        self,
        tmp_path,
        loss_dict=None,
        weight_dict=None,
        accumulate_grad_batches=1,
        model_config=None,
        train_log_on_step=False,
        compact_train_metrics=True,
    ):
        module, fake_model, fake_criterion, _ = _build_module(
            model_config=model_config,
            train_config=_base_train_config(
                tmp_path,
                grad_accum_steps=accumulate_grad_batches,
                train_log_on_step=train_log_on_step,
                compact_train_metrics=compact_train_metrics,
            ),
            tmp_path=tmp_path,
        )
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        # Provide a real optimizer so param_groups carries a real "lr" key.
        real_param = nn.Parameter(torch.randn(4))
        real_optimizer = torch.optim.SGD([real_param], lr=1e-3)
        module.optimizers = MagicMock(return_value=real_optimizer)
        module.manual_backward = MagicMock()
        module.lr_schedulers = MagicMock(return_value=None)
        _attach_trainer(
            module,
            accumulate_grad_batches=1,
            num_training_batches=1,
            gradient_clip_val=0.0,
            gradient_clip_algorithm="norm",
        )
        return module, samples, targets, fake_model, fake_criterion

    def test_returns_weighted_loss_sum(self, tmp_path):
        """Total loss must equal the sum of each loss multiplied by its weight."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "loss_bbox": torch.tensor(2.0), "loss_giou": torch.tensor(3.0)}
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0 + 10.0 + 6.0)

    def test_automatic_optimization_never_touches_trainer_strategy(self, mocker, tmp_path: Path) -> None:
        """Detection models must not fetch ``self.optimizers()`` — Lightning owns the optimizer on that path.

        ``optimizers()`` dereferences ``trainer.strategy._lightning_optimizers``. Only the keypoint manual-optimization
        branch consumes the result, so a trainer stub that carries nothing but ``accumulate_grad_batches`` must be
        enough for a detection ``training_step``; anything else means the automatic path grew a hidden dependency.
        """
        module, fake_model, fake_criterion, _ = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        trainer = SimpleNamespace(accumulate_grad_batches=1)

        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)

    def test_routes_forward_through_cuda_graph_runner(self, tmp_path: Path) -> None:
        """Once configured, training_step uses the graph runner while leaving self.model registered."""
        module, samples, targets, fake_model, fake_criterion = self._run_step(tmp_path)
        runner = MagicMock(return_value={})
        module._cuda_graph_runner = runner
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}

        module.training_step((samples, targets), batch_idx=0)

        runner.assert_called_once_with(samples, targets)
        fake_model.assert_not_called()

    @pytest.mark.parametrize("inductor_cudagraphs", [True, False])
    def test_marks_cudagraph_step_only_on_inductor_path(self, mocker, inductor_cudagraphs, tmp_path):
        """Each step on the Inductor CUDA graph path begins with ``cudagraph_mark_step_begin``; other paths never do.

        Lightning keeps the logged loss tensors alive across steps. Without the mark, cudagraph trees raise "accessing
        tensor output of CUDAGraphs that has been overwritten by a subsequent run" on the next replay.
        """
        mark_step = mocker.patch("rfdetr.training.module_model.torch.compiler.cudagraph_mark_step_begin")
        module, samples, targets, _, _ = self._run_step(tmp_path)
        module._inductor_cudagraphs = inductor_cudagraphs

        module.training_step((samples, targets), batch_idx=0)

        assert mark_step.call_count == (1 if inductor_cudagraphs else 0)

    def test_real_model_composes_through_cuda_graph_runner_on_training_step(self, tmp_path: Path) -> None:
        """A real detector, criterion, backward, and optimizer compose through the graph-runner ``training_step()``.

        Neither the synthetic model ``CudaGraphTrainingRunner`` is unit-tested against in isolation nor a fully-mocked
        runner class, as in the test above, covers this.

        ``torch.cuda.make_graphed_callables`` is the one CUDA-only primitive mocked here (as in
        ``test_capture_preserves_accumulated_gradients``); everything else — model, transformer capture toggle,
        criterion, Lightning's training_step, and the optimizer step — runs for real on CPU.
        """
        mc = RFDETRNanoConfig(pretrain_weights=None, num_classes=3, device="cpu")
        tc = _base_train_config(tmp_path)
        real_model = build_model_from_config(mc, tc)
        real_criterion, real_postprocess = build_criterion_from_config(mc, tc)
        real_model.train()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=real_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(real_criterion, real_postprocess),
            ),
        ):
            module = RFDETRModelModule(mc, tc)

        module._cuda_graph_runner = CudaGraphTrainingRunner(module.model)
        samples, targets = _make_batch(batch_size=1, h=mc.resolution, w=mc.resolution)
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_optimizer = torch.optim.SGD(module.model.parameters(), lr=1e-3)
        module.optimizers = MagicMock(return_value=real_optimizer)
        _attach_trainer(module, accumulate_grad_batches=1, num_training_batches=1)

        with patch("torch.cuda.make_graphed_callables", side_effect=lambda inner, _args, **_kwargs: inner):
            loss = module.training_step((samples, targets), batch_idx=0)

        assert torch.isfinite(loss)
        assert all(p.grad is None for p in module.model.parameters())
        loss.backward()
        assert any(p.grad is not None for p in module.model.parameters())
        assert len(module._cuda_graph_runner._graphed_cache) == 1

    def test_loss_backward_uses_box_normalizer_contract(self, tmp_path):
        """Backward loss for keypoint models is scaled by the criterion box normalizer.

        Manual optimization owns accumulation, so Lightning's ``accumulate_grad_batches`` does not scale it.
        """
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        keypoint_config = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17])
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict,
            weight_dict,
            accumulate_grad_batches=4,
            model_config=keypoint_config,
        )
        module.criterion.num_boxes_for_targets.return_value = torch.tensor(4.0)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)
        backward_loss = module.manual_backward.call_args.args[0]
        assert backward_loss.item() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "grad_accum_steps,num_training_batches,batch_idx,sync_grad",
        [(2, 2, 0, False), (2, 4, 1, True), (4, 2, 1, True)],
    )
    def test_keypoint_accumulation_syncs_only_when_optimizer_steps(
        self, tmp_path, grad_accum_steps, num_training_batches, batch_idx, sync_grad
    ):
        """Keypoint DDP must skip gradient synchronization until an accumulation window closes."""
        keypoint_config = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17])
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            accumulate_grad_batches=grad_accum_steps,
            model_config=keypoint_config,
        )
        optimizer = MagicMock(spec=LightningOptimizer)
        optimizer.param_groups = [{"lr": 1e-3}]
        optimizer.toggle_model.return_value = nullcontext()
        optimizer.zero_grad = MagicMock()
        module.optimizers.return_value = optimizer
        module._trainer.num_training_batches = num_training_batches

        module.training_step((samples, targets), batch_idx=batch_idx)

        optimizer.toggle_model.assert_called_once_with(sync_grad=sync_grad)
        module.manual_backward.assert_called_once()

    def test_detection_loss_is_returned_unscaled_for_lightning_accumulation(self, tmp_path: Path) -> None:
        """Detection (automatic optimization) returns the weighted loss unscaled.

        Lightning's ``ClosureResult.from_training_step_output`` divides the returned loss by
        ``trainer.accumulate_grad_batches`` before ``backward()``. Dividing here as well would scale every accumulated
        gradient by ``1/N**2`` (``tests/training/optim/test_grad_accumulation.py`` holds the gradient-level proof).
        """
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict,
            weight_dict,
            accumulate_grad_batches=1,
        )
        module._trainer.accumulate_grad_batches = 4

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(4.0)
        module.manual_backward.assert_not_called()

    def _make_keypoint_module(self, tmp_path, grad_accum_steps, num_training_batches):
        """Build a keypoint module wired with ``_ScalarLossModel`` and ``_BoxNormalizedCriterion`` for accum tests."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        model = _ScalarLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        module.model = model
        module.criterion = _BoxNormalizedCriterion()
        module.postprocess = MagicMock()
        module.log = MagicMock()
        module.log_dict = MagicMock()
        module.optimizers = MagicMock(return_value=optimizer)
        module.manual_backward = lambda loss: loss.backward()
        module.lr_schedulers = MagicMock(return_value=None)
        _attach_trainer(
            module,
            accumulate_grad_batches=1,
            num_training_batches=num_training_batches,
            gradient_clip_val=0.0,
            gradient_clip_algorithm="norm",
        )
        return module, model

    @pytest.mark.parametrize(
        "grad_accum_steps,box_counts,loss_numerators,expected_value",
        [
            pytest.param(1, (4,), (8.0,), -2.0, id="ga1-single-microbatch"),
            pytest.param(2, (2, 6), (10.0, 6.0), -2.0, id="ga2-balanced"),
            pytest.param(3, (2, 4, 6), (4.0, 8.0, 12.0), -2.0, id="ga3-balanced"),
            pytest.param(4, (1, 1, 1, 1), (2.0, 2.0, 2.0, 2.0), -2.0, id="ga4-uniform"),
            pytest.param(2, (1, 99), (1.0, 99.0), -1.0, id="ga2-skewed-1-vs-99"),
            pytest.param(4, (1, 1, 1, 97), (1.0, 1.0, 1.0, 97.0), -1.0, id="ga4-skewed-1-1-1-97"),
        ],
    )
    def test_box_normalized_accumulation_matches_large_effective_batch(
        self, tmp_path, grad_accum_steps, box_counts, loss_numerators, expected_value
    ):
        """Gradients over ``grad_accum_steps`` microbatches equal one large batch normalized by total boxes.

        This holds regardless of how lopsided the per-microbatch box counts are.
        """
        large_module, large_model = self._make_keypoint_module(tmp_path, grad_accum_steps=1, num_training_batches=1)
        accum_module, accum_model = self._make_keypoint_module(
            tmp_path, grad_accum_steps=grad_accum_steps, num_training_batches=grad_accum_steps
        )

        microbatch_targets = [
            {
                "labels": torch.ones(box_count, dtype=torch.int64),
                "loss_numerator": torch.tensor(loss_numerator),
                "orig_size": torch.tensor([16, 16]),
            }
            for box_count, loss_numerator in zip(box_counts, loss_numerators, strict=True)
        ]
        samples, _ = _make_batch(batch_size=2)

        large_module.training_step((samples, microbatch_targets), batch_idx=0)
        for batch_idx, target in enumerate(microbatch_targets):
            accum_module.training_step((samples, [target]), batch_idx=batch_idx)

        torch.testing.assert_close(accum_model.value, large_model.value)
        assert large_model.value.item() == pytest.approx(expected_value)

    def test_logs_live_train_loss_to_progress_bar(self, tmp_path):
        """Aggregate training loss must be logged every step as a progress-only metric."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        progress_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "loss"]
        assert len(progress_loss_calls) == 1
        assert progress_loss_calls[0].kwargs.get("prog_bar") is True
        assert progress_loss_calls[0].kwargs.get("logger") is False
        assert progress_loss_calls[0].kwargs.get("on_step") is True
        assert progress_loss_calls[0].kwargs.get("on_epoch") is False

    @pytest.mark.parametrize(
        ("progress_name", "scale"),
        [("loss", 2.0), ("loss_cls", 1.0)],
    )
    def test_progress_loss_is_epoch_running_mean_weighted_by_batch_size(
        self, tmp_path: Path, progress_name: str, scale: float
    ) -> None:
        """The live progress value is the sample-weighted mean of the epoch so far, not the latest batch."""
        module, _, _, _, fake_criterion = self._run_step(tmp_path, weight_dict={"loss_ce": 2.0})
        first_samples, first_targets = _make_batch(batch_size=1)
        second_samples, second_targets = _make_batch(batch_size=3)

        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        module.training_step((first_samples, first_targets), batch_idx=0)
        fake_criterion.return_value = {"loss_ce": torch.tensor(4.0)}
        module.training_step((second_samples, second_targets), batch_idx=1)

        progress_calls = [c for c in module.log.call_args_list if c[0][0] == progress_name]
        assert progress_calls[-1].args[1].item() == pytest.approx(scale * (1.0 * 1 + 4.0 * 3) / 4)

    def test_progress_terms_stay_per_batch_when_train_log_on_step(self, tmp_path: Path) -> None:
        """With train_log_on_step=True every progress term keeps its per-batch value, as train/loss_step does."""
        module, samples, targets, _, fake_criterion = self._run_step(tmp_path, train_log_on_step=True)

        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        module.training_step((samples, targets), batch_idx=0)
        second_value = torch.tensor(3.0)
        fake_criterion.return_value = {"loss_ce": second_value}
        module.training_step((samples, targets), batch_idx=1)

        progress_calls = [c for c in module.log.call_args_list if c[0][0] == "loss_cls"]
        assert progress_calls[-1].args[1] is second_value

    @pytest.mark.parametrize("train_log_on_step", [False, True])
    def test_progress_terms_are_progress_bar_only(self, tmp_path: Path, train_log_on_step: bool) -> None:
        """Every progress term goes to the progress bar only, per step, with the same flags in both modes."""
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=train_log_on_step)

        module.training_step((samples, targets), batch_idx=0)

        progress_calls = [c for c in module.log.call_args_list if c[0][0] == "loss_cls"]
        assert progress_calls[-1].kwargs == {
            "prog_bar": True,
            "logger": False,
            "on_step": True,
            "on_epoch": False,
            "batch_size": 2,
        }

    @pytest.mark.parametrize("progress_name", ["loss", "loss_cls"])
    def test_progress_loss_running_mean_restarts_each_epoch(self, tmp_path: Path, progress_name: str) -> None:
        """The running mean restarts at each epoch, so the first step of a new epoch reports its own loss."""
        module, samples, targets, _, fake_criterion = self._run_step(tmp_path)

        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        module.training_step((samples, targets), batch_idx=0)
        module.on_train_epoch_start()
        fake_criterion.return_value = {"loss_ce": torch.tensor(3.0)}
        module.log.reset_mock()
        module.training_step((samples, targets), batch_idx=0)

        progress_calls = [c for c in module.log.call_args_list if c[0][0] == progress_name]
        assert progress_calls[-1].args[1].item() == pytest.approx(3.0)

    def test_progress_loss_running_mean_is_detached_from_autograd(self, tmp_path: Path) -> None:
        """The running sums must not keep the autograd graph of every step alive until the epoch ends."""
        module, samples, targets, _, fake_criterion = self._run_step(tmp_path)
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0, requires_grad=True)}

        module.training_step((samples, targets), batch_idx=0)

        progress_calls = [c for c in module.log.call_args_list if c[0][0] == "loss_cls"]
        assert progress_calls[-1].args[1].requires_grad is False

    def test_progress_loss_running_mean_accumulates_in_float32(self, tmp_path: Path) -> None:
        """A half-precision loss term is summed in float32, so the bar does not drift over a long epoch."""
        module, samples, targets, _, fake_criterion = self._run_step(tmp_path)
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0, dtype=torch.bfloat16)}

        module.training_step((samples, targets), batch_idx=0)

        progress_calls = [c for c in module.log.call_args_list if c[0][0] == "loss_cls"]
        assert progress_calls[-1].args[1].dtype is torch.float32

    @pytest.mark.parametrize(
        "train_log_on_step",
        [
            pytest.param(False, id="epoch-only"),
            pytest.param(True, id="step-and-epoch"),
        ],
    )
    def test_logs_epoch_train_loss_to_progress_bar(self, tmp_path, train_log_on_step):
        """Canonical training loss stays epoch-aggregated, with on_step mirroring train_log_on_step."""
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=train_log_on_step)

        module.training_step((samples, targets), batch_idx=0)

        epoch_loss_calls = [call for call in module.log.call_args_list if call[0][0] == "train/loss"]
        assert len(epoch_loss_calls) == 1
        assert epoch_loss_calls[0].kwargs.get("prog_bar") is True
        assert epoch_loss_calls[0].kwargs.get("on_step") is train_log_on_step
        assert epoch_loss_calls[0].kwargs.get("on_epoch") is True

    @pytest.mark.parametrize(
        "train_log_on_step,expected_live_loss_calls",
        [
            pytest.param(False, 1, id="default-emits-live-loss"),
            pytest.param(True, 0, id="on-step-skips-live-loss"),
        ],
    )
    def test_live_loss_key_gated_on_train_log_on_step(
        self, tmp_path: Path, train_log_on_step: bool, expected_live_loss_calls: int
    ) -> None:
        """Redundant live ``loss`` progress key is emitted only when train_log_on_step is False.

        With train_log_on_step=True the ``train/loss`` call forks into a per-batch ``train/loss_step`` progress entry,
        which already shows the aggregate, so the running-mean ``loss`` scalar is skipped to keep the bar's keys as
        before.
        """
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=train_log_on_step)

        module.training_step((samples, targets), batch_idx=0)

        live_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "loss"]
        assert len(live_loss_calls) == expected_live_loss_calls

    def test_training_step_does_not_log_learning_rate(self, tmp_path):
        """Learning-rate metrics must wait for the optimizer-step boundary."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        lr_calls = [c for c in module.log.call_args_list if c[0][0] == "train/lr"]
        assert not lr_calls

    def test_logs_learning_rate_range_for_multiple_param_groups(self, tmp_path):
        """An automatic optimizer step logs first, minimum, and maximum rates with distinct visibility flags."""
        module, samples, targets, _, _ = self._run_step(tmp_path)
        first_param = nn.Parameter(torch.randn(4))
        second_param = nn.Parameter(torch.randn(4))
        third_param = nn.Parameter(torch.randn(4))
        module.optimizers.return_value = torch.optim.SGD(
            [
                {"params": [first_param], "lr": 0.1},
                {"params": [second_param], "lr": 0.05},
                {"params": [third_param], "lr": 0.2},
            ],
            lr=0.1,
        )

        module.on_before_optimizer_step(module.optimizers.return_value)

        expected_metrics = {
            "train/lr": (0.1, True),
            "train/lr_min": (0.05, False),
            "train/lr_max": (0.2, False),
        }
        for metric_name, (expected_value, expected_prog_bar) in expected_metrics.items():
            metric_calls = [call for call in module.log.call_args_list if call.args[0] == metric_name]
            assert len(metric_calls) == 1
            assert metric_calls[0].args[1] == pytest.approx(expected_value)
            assert metric_calls[0].kwargs.get("prog_bar") is expected_prog_bar
            assert metric_calls[0].kwargs.get("on_step") is True
            assert metric_calls[0].kwargs.get("on_epoch") is False

    def test_compacts_auxiliary_loss_metrics(self, tmp_path):
        """Layer-suffixed loss metrics should be reduced to one aggregate per base loss term."""
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_bbox": torch.tensor(2.0),
            "loss_ce_0": torch.tensor(3.0),
            "loss_bbox_0": torch.tensor(4.0),
            "loss_ce_enc": torch.tensor(5.0),
        }
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict=loss_dict,
            weight_dict={key: 1.0 for key in loss_dict},
        )

        module.training_step((samples, targets), batch_idx=0)

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == {"train/loss_ce", "train/loss_bbox", "train/loss_ce_aux", "train/loss_bbox_aux"}
        assert logged["train/loss_ce_aux"].item() == pytest.approx(8.0)
        assert logged["train/loss_bbox_aux"].item() == pytest.approx(4.0)

    def test_excludes_non_weighted_layer_suffixed_keys_from_aggregation(self, tmp_path):
        """A digit/enc-suffixed key absent from weight_dict must be logged individually, not aggregated.

        ``cardinality_error_0`` is a diagnostic count, never a weighted loss term, so weight_dict never contains it even
        though its name matches the layer-suffix pattern used for real auxiliary terms.
        """
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_ce_0": torch.tensor(2.0),
            "cardinality_error": torch.tensor(3.0),
            "cardinality_error_0": torch.tensor(4.0),
        }
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict=loss_dict,
            weight_dict={"loss_ce": 1.0, "loss_ce_0": 1.0},
        )

        module.training_step((samples, targets), batch_idx=0)

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == {
            "train/loss_ce",
            "train/loss_ce_aux",
            "train/cardinality_error",
            "train/cardinality_error_0",
        }
        assert logged["train/cardinality_error"].item() == pytest.approx(3.0)
        assert logged["train/cardinality_error_0"].item() == pytest.approx(4.0)

    def test_logs_layer_suffixed_keys_individually_when_compact_metrics_disabled(self, tmp_path):
        """With compact_train_metrics=False, every per-layer key is logged individually, honoring on_step."""
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_ce_0": torch.tensor(2.0),
            "loss_ce_enc": torch.tensor(3.0),
        }
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict=loss_dict,
            weight_dict={key: 1.0 for key in loss_dict},
            train_log_on_step=True,
            compact_train_metrics=False,
        )

        module.training_step((samples, targets), batch_idx=0)

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == {"train/loss_ce", "train/loss_ce_0", "train/loss_ce_enc"}
        assert module.log_dict.call_args.kwargs.get("on_step") is True

    def test_compacts_real_criterion_output_gated_by_weight_dict(self, tmp_path):
        """Aggregation of a real SetCriterion's key inventory must fold only weighted per-layer losses.

        ``test_compacts_auxiliary_loss_metrics`` above only exercises synthetic ``loss_ce``/``loss_bbox`` keys, whose
        weight_dict membership and layer-suffix pattern happen to coincide. RF-DETR's real criterion also emits
        ``cardinality_error`` with the same ``_<i>``/``_enc`` suffix pattern (see ``SetCriterion.forward``) despite
        never appearing in ``weight_dict`` (it is a logging-only diagnostic, not a backpropagated loss term). Suffix-
        only aggregation folds it into a spurious ``cardinality_error_aux`` key; correct aggregation must gate on
        ``weight_dict`` membership and leave it out. This uses the real ``RFDETRSmall`` model/criterion (this repo's
        default size) with a tiny synthetic batch so the assertion is checked against RF-DETR's actual key inventory
        instead of a hand-picked synthetic one.
        """
        mc = RFDETRSmallConfig(pretrain_weights=None, num_classes=3, device="cpu")
        tc = _base_train_config(tmp_path)
        real_model = build_model_from_config(mc, tc)
        real_criterion, real_postprocess = build_criterion_from_config(mc, tc)
        real_model.train()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=real_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(real_criterion, real_postprocess),
            ),
        ):
            module = RFDETRModelModule(mc, tc)

        samples, targets = _make_batch(batch_size=1, h=mc.resolution, w=mc.resolution)
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        _attach_trainer(module, accumulate_grad_batches=1, num_training_batches=1)

        # Spy on the criterion to capture its raw (pre-aggregation) per-key loss dict, so the expected
        # aggregate is derived from the same real forward pass training_step() uses internally rather
        # than a second, independently-random forward pass.
        raw_loss_dicts: list[dict[str, torch.Tensor]] = []
        original_forward = module.criterion.forward

        def _spy_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            raw_loss_dicts.append(result)
            return result

        module.criterion.forward = _spy_forward

        module.training_step((samples, targets), batch_idx=0)

        raw_loss_dict = raw_loss_dicts[0]
        weight_dict = module.criterion.weight_dict
        expected_aux_sums: dict[str, torch.Tensor] = {}
        expected_base_keys: set[str] = set()
        for key, value in raw_loss_dict.items():
            base_name, separator, suffix = key.rpartition("_")
            is_layer_suffixed = bool(separator) and (suffix.isdigit() or suffix == "enc")
            if is_layer_suffixed and key in weight_dict:
                aggregate_name = f"train/{base_name}_aux"
                expected_aux_sums[aggregate_name] = expected_aux_sums.get(aggregate_name, torch.zeros(())) + value
            else:
                # Not layer-suffixed, or layer-suffixed but absent from weight_dict (e.g.
                # cardinality_error_0/_1/_enc — a diagnostic count, never a weighted loss term):
                # both cases fall through to individual logging, matching production's else branch.
                expected_base_keys.add(f"train/{key}")

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == expected_base_keys | set(expected_aux_sums)
        assert "train/cardinality_error_aux" not in logged
        for aggregate_name, expected_value in expected_aux_sums.items():
            assert logged[aggregate_name].item() == pytest.approx(expected_value.item())

    def test_logs_loss_components_on_epoch_when_step_logging_is_enabled(self, tmp_path):
        """Per-step total loss logging must not re-enable per-step component metrics."""
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=True)

        module.training_step((samples, targets), batch_idx=0)

        assert module.log_dict.call_args.kwargs.get("on_step") is False
        assert module.log_dict.call_args.kwargs.get("on_epoch") is True

    @pytest.mark.parametrize(
        "grad_accum_steps, num_training_batches, expected_log_count",
        [
            pytest.param(2, 3, 2, id="manual-accumulation-tail-fallback-logs-once-per-window-never-twice"),
            pytest.param(
                1,
                3,
                3,
                id="grad-accum-steps-one-every-batch-closes-its-own-window-and-logs",
            ),
            pytest.param(
                2,
                float("inf"),
                1,
                id="infinite-dataset-logs-only-at-modulo-boundary-tail-fallback-never-fires",
            ),
        ],
    )
    def test_logs_learning_rates_for_manual_optimizer_steps(
        self, tmp_path, grad_accum_steps: int, num_training_batches: float, expected_log_count: int
    ):
        """Manual accumulation logs learning rates once per completed or partial optimizer window, never twice.

        ``_step_optimizer`` must not log learning rates itself — the single emission site is the
        ``on_before_optimizer_step`` hook, which ``_StepHookOptimizer`` fires from ``step()`` the same way Lightning's
        real ``LightningOptimizer.step()`` does on both automatic and manual paths. The finite case
        (``num_training_batches=3``) exercises ``_should_step_optimizer``'s end-of-epoch tail fallback on its third
        batch; the infinite case proves ``batch_idx + 1 >= num_training_batches`` can never trip against infinity, so
        only the modulo-boundary step (batch_idx=1) logs.
        """
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        parameter = nn.Parameter(torch.randn(4))
        optimizer = _StepHookOptimizer(module, torch.optim.SGD([parameter], lr=0.1))
        trainer = MagicMock(
            num_training_batches=num_training_batches, gradient_clip_val=0.0, gradient_clip_algorithm="norm"
        )
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        module.log = MagicMock()

        for batch_idx in range(3):
            if module._should_step_optimizer(batch_idx):
                module._step_optimizer(optimizer)

        for metric_name in ("train/lr", "train/lr_min", "train/lr_max"):
            metric_calls = [call for call in module.log.call_args_list if call.args[0] == metric_name]
            assert len(metric_calls) == expected_log_count

    def test_logs_convergence_components_to_progress_bar(self, tmp_path):
        """Selected detection and keypoint losses should appear as compact progress-only metrics."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_keypoints_l1": torch.tensor(0.4),
            "loss_keypoints_nll": torch.tensor(0.2),
        }
        weight_dict = {key: 1.0 for key in loss_dict}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        module.training_step((samples, targets), batch_idx=0)

        progress_names = {c[0][0] for c in module.log.call_args_list if c.kwargs.get("prog_bar") is True}
        assert {"loss_cls", "loss_box", "kp_l1", "kp_nll"}.issubset(progress_names)

    def test_logs_individual_main_losses_as_dict(self, tmp_path):
        """Each main decoder loss must be logged separately under the train/ prefix."""
        loss_dict = {"loss_ce": torch.tensor(0.5), "loss_bbox": torch.tensor(0.3)}
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        module.training_step((samples, targets), batch_idx=0)

        module.log_dict.assert_called_once()
        logged = module.log_dict.call_args[0][0]
        assert "train/loss_ce" in logged
        assert "train/loss_bbox" in logged

    def test_returns_scalar_tensor(self, tmp_path):
        """Loss must be a 0-dim tensor so Lightning can call .backward() on it."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.dim() == 0

    def test_returns_detached_predictions_when_train_metrics_enabled(self, tmp_path):
        """compute_train_metrics=True should expose detached predictions without changing the Lightning loss key."""
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, fake_postprocess = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        model_output = {"pred_logits": torch.randn(2, 3, requires_grad=True)}
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        fake_postprocess.return_value = [{"boxes": torch.randn(1, 4, requires_grad=True)}]
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        _attach_trainer(module, accumulate_grad_batches=1)

        result = module.training_step((samples, targets), batch_idx=0)

        assert isinstance(result, dict)
        assert result["loss"].dim() == 0
        assert result["results"][0]["boxes"].requires_grad is False
        assert result["targets"] is targets

    def test_ignores_losses_not_in_weight_dict(self, tmp_path):
        """Losses absent from weight_dict (e.g. cardinality_error) must not affect total."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "cardinality_error": torch.tensor(99.0)}
        weight_dict = {"loss_ce": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(2.0)

    def test_train_metrics_slices_to_group0_queries(self, tmp_path):
        """compute_train_metrics postprocess must receive only group-0 queries ([:num_queries]).

        Group DETR emits group_detr×num_queries outputs in train mode. Without the slice, postprocess top-k draws from
        all groups and OKS/mAP reads ~50× below true accuracy. Assert the received pred_logits has shape (B,
        num_queries, C).
        """
        nq = 10
        group_detr = 3
        batch_size = 2
        num_classes = 5
        mc = _base_model_config(num_classes=num_classes, num_queries=nq)
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        full_logits = torch.randn(batch_size, group_detr * nq, num_classes)
        model_output = {
            "pred_logits": full_logits,
            "pred_boxes": torch.randn(batch_size, group_detr * nq, 4),
        }
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        received: dict = {}

        def capture_postprocess(outputs, orig_sizes):
            received.update(outputs)
            return [
                {"boxes": torch.zeros(nq, 4), "scores": torch.ones(nq), "labels": torch.zeros(nq, dtype=torch.long)}
            ]

        module.postprocess = capture_postprocess
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        _attach_trainer(module, accumulate_grad_batches=1, num_training_batches=1)
        samples, targets = _make_batch(batch_size=batch_size)

        module.training_step((samples, targets), batch_idx=0)

        assert "pred_logits" in received
        assert received["pred_logits"].shape == (batch_size, nq, num_classes)
        torch.testing.assert_close(received["pred_logits"], full_logits[:, :nq])

    def test_train_metrics_skips_dict_pred_masks(self, tmp_path):
        """Dict-valued pred_masks (sparse_forward in train mode) must not crash training_step.

        In segmentation train mode lwdetr uses sparse_forward which returns pred_masks as a dict. PostProcess cannot
        handle a dict — it calls .shape[0] on it. The fix filters out non-tensor values so postprocess receives
        pred_masks=None (box path).
        """
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)

        model_output = {
            "pred_logits": torch.randn(2, 10, 5),
            "pred_boxes": torch.randn(2, 10, 4),
            "pred_masks": {"spatial_features": torch.randn(2, 256, 8, 8), "query_features": torch.randn(2, 10, 256)},
        }
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        received: dict = {}

        def capture_postprocess(outputs, orig_sizes):
            received.update(outputs)
            return [{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1, dtype=torch.long)}]

        module.postprocess = capture_postprocess
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        _attach_trainer(module, accumulate_grad_batches=1, num_training_batches=1)
        samples, targets = _make_batch()

        module.training_step((samples, targets), batch_idx=0)

        assert "pred_masks" not in received


class TestOnBeforeOptimizerStepFiresOncePerAccumulationWindow:
    """``on_before_optimizer_step`` must fire once per optimizer update on the automatic-optimization path.

    ``TestTrainingStep.test_logs_learning_rate_range_for_multiple_param_groups`` above calls
    ``on_before_optimizer_step`` directly with a hand-built optimizer, which only verifies the payload it logs — it
    never drives Lightning's own automatic-optimization gradient-accumulation loop, so it cannot catch a regression
    where the hook fired once per microbatch instead of once per completed accumulation window. Detection/segmentation
    models use PTL's automatic optimization (unlike keypoint models' manual ``_should_step_optimizer`` path tested
    elsewhere in this file), so accumulation gating here is entirely internal to Lightning; the only way to exercise
    it faithfully is a real ``Trainer.fit()`` run with ``accumulate_grad_batches > 1``.
    """

    def test_fires_once_per_window_not_once_per_microbatch(self, mocker, tmp_path):
        """4 training batches at accumulate_grad_batches=2 must trigger exactly 2 optimizer-step hook calls."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, num_workers=0)

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_helpers_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        class _CountOptimizerSteps(Callback):
            """Count how many times Lightning invokes the pre-optimizer-step hook."""

            def __init__(self) -> None:
                self.count = 0

            def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
                """Increment the call count for every hook invocation."""
                self.count += 1

        counter = _CountOptimizerSteps()

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model)
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(fake_criterion, fake_postprocess),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset)
        mocker.patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        )
        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = Trainer(
            fast_dev_run=4,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accumulate_grad_batches=2,
            callbacks=[counter],
        )
        trainer.fit(module, datamodule)

        assert counter.count == 2


class TestShouldStepOptimizer:
    """Tests for ``_should_step_optimizer``.

    Covers the modulo path, the end-of-epoch fallback, and the iterable / infinite dataset case where
    ``trainer.num_training_batches`` is ``float('inf')``.
    """

    def _make_module_with_trainer(self, tmp_path, grad_accum_steps, num_training_batches):
        """Build a module with a stub trainer exposing ``num_training_batches`` for the test scenario."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        _attach_trainer(module, num_training_batches=num_training_batches)
        return module

    @pytest.mark.parametrize(
        "grad_accum_steps,num_training_batches,batch_idx,expected",
        [
            pytest.param(1, 10, 0, True, id="ga1-bidx0-steps-every-batch"),
            pytest.param(1, 10, 9, True, id="ga1-bidx9-steps-every-batch"),
            pytest.param(2, 10, 0, False, id="ga2-bidx0-mid-window"),
            pytest.param(2, 10, 1, True, id="ga2-bidx1-closes-window"),
            pytest.param(2, 10, 2, False, id="ga2-bidx2-opens-new-window"),
            pytest.param(2, 10, 9, True, id="ga2-bidx9-closes-final-window"),
            pytest.param(4, 10, 7, True, id="ga4-bidx7-closes-second-window"),
            pytest.param(4, 10, 8, False, id="ga4-bidx8-opens-partial-window"),
            pytest.param(4, 10, 9, True, id="ga4-bidx9-final-batch-flushes-partial"),
            pytest.param(4, 11, 8, False, id="ga4-bidx8-of-11-mid-window"),
            pytest.param(4, 11, 10, True, id="ga4-bidx10-final-batch-flushes-partial"),
        ],
    )
    def test_finite_dataset_steps_at_window_close_and_epoch_end(
        self, tmp_path, grad_accum_steps, num_training_batches, batch_idx, expected
    ):
        """Optimizer steps when the accumulation window closes or when the epoch ends with a partial window."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps, num_training_batches)

        assert module._should_step_optimizer(batch_idx) is expected

    @pytest.mark.parametrize(
        "grad_accum_steps,batch_idx,expected",
        [
            pytest.param(2, 0, False, id="ga2-bidx0-mid-window"),
            pytest.param(2, 1, True, id="ga2-bidx1-closes-window"),
            pytest.param(4, 2, False, id="ga4-bidx2-mid-window"),
            pytest.param(4, 3, True, id="ga4-bidx3-closes-window"),
        ],
    )
    def test_infinite_dataset_uses_modulo_only(self, tmp_path, grad_accum_steps, batch_idx, expected):
        """Iterable datasets report ``num_training_batches=float('inf')``; only the modulo path can close the window."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps, float("inf"))

        assert module._should_step_optimizer(batch_idx) is expected

    def test_none_num_training_batches_uses_modulo_only(self, tmp_path):
        """If trainer.num_training_batches is None (very early in fit), only the modulo path can trigger a step."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps=2, num_training_batches=None)

        assert module._should_step_optimizer(batch_idx=0) is False
        assert module._should_step_optimizer(batch_idx=1) is True


class TestOnTrainEpochStart:
    """Tests for ``on_train_epoch_start`` — must reset the accumulated box normalizer between epochs."""

    def test_reset_clears_stale_accumulator(self, tmp_path):
        """A stale normalizer from a previous epoch must not leak into the new epoch's first microbatch."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        module._accumulated_box_normalizer = torch.tensor(42.0)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None

    def test_is_noop_for_detection_module(self, tmp_path):
        """Detection models never populate _accumulated_box_normalizer; reset must leave it None."""
        module, *_ = _build_module(tmp_path=tmp_path)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None

    def test_zeros_optimizer_grad_on_stale_accumulator(self, tmp_path):
        """When a partial window survived epoch end, optimizer gradients must be zeroed before reset."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        real_param = nn.Parameter(torch.randn(4))
        real_param.grad = torch.ones(4)
        optimizer = torch.optim.SGD([real_param], lr=1.0)
        module.optimizers = MagicMock(return_value=optimizer)
        module._accumulated_box_normalizer = torch.tensor(7.0)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None
        assert real_param.grad is None or real_param.grad.abs().sum().item() == pytest.approx(0.0)

    def test_forwards_current_epoch_to_datamodule_train_batch_sampler(self, tmp_path):
        """A custom batch sampler on the datamodule must receive set_epoch(current_epoch) every epoch.

        PTL never calls epoch hooks on a LightningDataModule itself, so RFDETRModelModule bridges the gap
        from its own on_train_epoch_start: this is the only place WeightedMultiSourceBatchSampler's seeded
        per-epoch reshuffle actually advances instead of reusing the first epoch's shuffle for the whole run.
        """
        module, *_ = _build_module(tmp_path=tmp_path)
        fake_sampler = MagicMock()
        fake_datamodule = MagicMock(train_batch_sampler=fake_sampler)
        fake_trainer = MagicMock(datamodule=fake_datamodule)
        module._trainer = fake_trainer
        with patch.object(type(module), "current_epoch", new=3):
            module.on_train_epoch_start()

        fake_sampler.set_epoch.assert_called_once_with(3)

    def test_is_noop_when_datamodule_has_no_custom_batch_sampler(self, tmp_path):
        """build_train_sampler() returning None must not raise when there is no set_epoch to forward."""
        module, *_ = _build_module(tmp_path=tmp_path)
        fake_datamodule = MagicMock(train_batch_sampler=None)
        fake_trainer = MagicMock(datamodule=fake_datamodule)
        module._trainer = fake_trainer

        module.on_train_epoch_start()  # must not raise

        assert module._accumulated_box_normalizer is None

    def test_is_noop_when_trainer_is_unattached(self, tmp_path):
        """Calling on_train_epoch_start with no Trainer attached must not raise (matches the other cases in this
        class)."""
        module, *_ = _build_module(tmp_path=tmp_path)

        module.on_train_epoch_start()  # must not raise RuntimeError from the unattached self.trainer property

        assert module._accumulated_box_normalizer is None


class TestRescaleAccumulatedGradients:
    """Direct contract tests for _rescale_accumulated_gradients."""

    @pytest.mark.parametrize(
        "in_features, out_features, factor, weight_grad, bias_grad, weight_expected, bias_expected",
        [
            pytest.param(3, 5, 0.5, 4.0, 8.0, 2.0, 4.0, id="factor-half-halves-grads"),
            pytest.param(2, 2, 1.0, 3.0, 7.0, 3.0, 7.0, id="factor-one-is-identity"),
        ],
    )
    def test_scales_parameter_grads_by_factor(
        self, tmp_path, in_features, out_features, factor, weight_grad, bias_grad, weight_expected, bias_expected
    ):
        """Every parameter's .grad is multiplied by the factor; factor 1.0 leaves gradients exactly unchanged."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(in_features, out_features)
        nano_model.weight.grad = torch.full((out_features, in_features), weight_grad)
        nano_model.bias.grad = torch.full((out_features,), bias_grad)
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(factor))

        torch.testing.assert_close(nano_model.weight.grad, torch.full((out_features, in_features), weight_expected))
        torch.testing.assert_close(nano_model.bias.grad, torch.full((out_features,), bias_expected))

    def test_skips_params_with_no_grad(self, tmp_path):
        """Parameters without .grad must remain None after rescaling."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(2, 2)
        # No backward pass — all grads are None
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(0.5))

        assert nano_model.weight.grad is None
        assert nano_model.bias.grad is None


class TestConfigureGradientClipping:
    """Tests for configure_gradient_clipping() — verifies precision gating mirrors configure_optimizers()."""

    def _setup_module(self, tmp_path, precision: str):
        tc = _base_train_config(tmp_path)
        module, _, _, _ = _build_module(train_config=tc)
        _attach_trainer(module, precision=precision)
        return module

    @pytest.mark.parametrize(
        "precision",
        [
            pytest.param("32-true", id="fp32"),
            pytest.param("16-mixed", id="fp16-mixed"),
        ],
    )
    def test_delegates_to_clip_gradients_when_not_bf16(self, mocker, precision, tmp_path):
        """configure_gradient_clipping must delegate to clip_gradients() when trainer precision is not a BF16 variant.

        On Ampere+ GPUs is_bf16_supported() is True regardless of actual precision. The method must check
        trainer.precision before choosing the fused path, mirroring the same gate in configure_optimizers() to prevent
        silent divergence. ``clip_gradients`` itself is never overridden (PTL documents it as "do not override"), so the
        un-overridden base implementation on the module's own class is patched directly.
        """
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        module = self._setup_module(tmp_path, precision=precision)

        mock_clip_gradients = mocker.patch.object(type(module), "clip_gradients")
        module.configure_gradient_clipping(MagicMock(), gradient_clip_val=0.1)

        mock_clip_gradients.assert_called_once()

    def test_configure_gradient_clipping_uses_clip_grad_norm_when_bf16_mixed(self, mocker, tmp_path):
        """configure_gradient_clipping must call clip_grad_norm_ directly when precision is bf16-mixed.

        When fused AdamW is active (BF16, no GradScaler), the standard PTL AMP plugin refuses to clip gradients.
        clip_grad_norm_ is called directly instead, bypassing the scaler-aware path that would otherwise raise.
        """
        mock_clip_grad_norm = mocker.patch("rfdetr.training.module_model.torch.nn.utils.clip_grad_norm_")
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
        mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
        module = self._setup_module(tmp_path, precision="bf16-mixed")

        module.configure_gradient_clipping(MagicMock(), gradient_clip_val=0.5)

        mock_clip_grad_norm.assert_called_once()
        _, _call_kwargs = mock_clip_grad_norm.call_args
        # Positional arg[1] is max_norm
        assert mock_clip_grad_norm.call_args[0][1] == pytest.approx(0.5)


class TestManualOptLRSchedulerStepping:
    """Manual-optimization (keypoint) path steps LR schedulers itself, honoring interval and plateau semantics."""

    def _module(self, tmp_path, **overrides):
        tc = _base_train_config(tmp_path, **overrides)
        module, _, _, _ = _build_module(train_config=tc)
        module.automatic_optimization = False
        return module

    @pytest.mark.parametrize(
        "interval, method_name, assert_scheduler_state",
        [
            pytest.param(
                "step",
                "_step_lr_scheduler",
                lambda scheduler: scheduler.step.assert_called_once_with(),
                id="step-interval-stepped-per-optimizer-step",
            ),
            pytest.param(
                "epoch",
                "_step_lr_scheduler",
                lambda scheduler: scheduler.step.assert_not_called(),
                id="epoch-interval-not-stepped-by-the-per-optimizer-step-hook",
            ),
            pytest.param(
                "epoch",
                "on_train_epoch_end",
                lambda scheduler: scheduler.step.assert_called_once_with(),
                id="epoch-interval-stepped-on-train-epoch-end-manual-path",
            ),
        ],
    )
    def test_lr_scheduler_stepped_according_to_interval(
        self, mocker, tmp_path, interval: str, method_name: str, assert_scheduler_state
    ):
        """The manual LR-scheduler step hooks respect ``_lr_scheduler_interval``.

        A step-interval scheduler is stepped on every optimizer step (``_step_lr_scheduler``); an epoch-interval
        scheduler is not stepped by that hook, only by ``on_train_epoch_end``.
        """
        module = self._module(tmp_path)
        module._lr_scheduler_interval = interval
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)
        mocker.patch.object(module, "lr_schedulers", return_value=scheduler)
        getattr(module, method_name)()

        assert_scheduler_state(scheduler)

    def test_plateau_stepped_from_monitor_metric_on_validation_epoch_end(self, mocker, tmp_path):
        """on_validation_epoch_end steps ReduceLROnPlateau with the monitored metric."""
        module = self._module(tmp_path)
        module._lr_scheduler_monitor = "val/loss"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
        _attach_trainer(module, sanity_checking=False, callback_metrics={"val/loss": torch.tensor(1.23)})
        mocker.patch.object(module, "lr_schedulers", return_value=scheduler)
        module.on_validation_epoch_end()

        scheduler.step.assert_called_once()

    def test_plateau_raises_when_monitor_metric_missing(self, mocker, tmp_path):
        """on_validation_epoch_end fails loud (not silent warn) when the monitored metric is absent."""
        module = self._module(tmp_path)
        module._lr_scheduler_monitor = "val/loss"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
        _attach_trainer(module, sanity_checking=False, callback_metrics={})
        mocker.patch.object(module, "lr_schedulers", return_value=scheduler)
        with pytest.raises(RuntimeError) as exc_info:
            module.on_validation_epoch_end()
        assert "never be reduced" in str(exc_info.value)

        scheduler.step.assert_not_called()

    def test_plateau_not_stepped_during_sanity_check(self, mocker, tmp_path):
        """on_validation_epoch_end must not step plateau during Lightning's pre-training sanity check."""
        module = self._module(tmp_path)
        module._lr_scheduler_monitor = "val/loss"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
        _attach_trainer(module, sanity_checking=True, callback_metrics={"val/loss": torch.tensor(1.23)})
        mocker.patch.object(module, "lr_schedulers", return_value=scheduler)
        module.on_validation_epoch_end()

        scheduler.step.assert_not_called()
