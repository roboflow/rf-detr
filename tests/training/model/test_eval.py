# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for RFDETRModelModule validation, test and predict steps.

``validation_step`` and ``test_step`` get every layer's segmentation masks only when they compute a loss.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch import Tensor

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.training.callbacks.best_model import RFDETREarlyStopping
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor
from tests.models.lwdetr.test_model import _BATCH, _CLASSES, _IMAGE, _QUERIES, _build_seg_lwdetr
from tests.training.helpers import (
    _base_model_config,
    _base_train_config,
    _build_module,
    _FakeCriterion,
    _FakeDataset,
    _make_batch,
    _TinyModel,
)
from tests.training.helpers import _fake_postprocess as _helpers_fake_postprocess


class TestValidationStep:
    """Validation-specific behaviour of validation_step(): loss policy, keypoint loss logging, and eval-model choice.

    The contracts validation_step() shares with test_step() live in :class:`TestEvalStepOutputs`.
    """

    def _run_val_step(
        self,
        tmp_path,
        loss_dict: dict[str, torch.Tensor] | None = None,
        weight_dict: dict[str, float] | None = None,
    ):
        tc = _base_train_config(tmp_path, compute_val_loss=True)
        module, fake_model, fake_criterion, fake_pp = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        result = module.validation_step((samples, targets), batch_idx=0)
        return result, fake_pp, module

    def test_logs_val_keypoint_loss_components_once(self, tmp_path):
        """Validation should expose full keypoint losses without duplicate progress aliases."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_keypoints_l1": torch.tensor(0.4),
            "loss_keypoints_findable": torch.tensor(0.3),
            "loss_keypoints_visible": torch.tensor(0.2),
            "loss_keypoints_nll": torch.tensor(0.1),
        }
        weight_dict = {key: 1.0 for key in loss_dict}

        _, _, module = self._run_val_step(tmp_path, loss_dict=loss_dict, weight_dict=weight_dict)

        module.log_dict.assert_called_once()
        logged = module.log_dict.call_args.args[0]
        assert "val/loss_keypoints_l1" in logged
        assert "val/loss_keypoints_findable" in logged
        logged_names = {c[0][0] for c in module.log.call_args_list}
        assert "val/kp_l1" not in logged_names
        assert "val/kp_find" not in logged_names
        assert "val/kp_vis" not in logged_names
        assert "val/kp_nll" not in logged_names

    def test_val_detection_loss_components_not_relogged_as_aliases(self, tmp_path):
        """Validation component losses should be logged once under canonical ``val/loss_*`` names."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
        }
        weight_dict = {key: 1.0 for key in loss_dict}

        _, _, module = self._run_val_step(tmp_path, loss_dict=loss_dict, weight_dict=weight_dict)

        logged_loss_names = set(module.log_dict.call_args.args[0])
        direct_log_names = {c[0][0] for c in module.log.call_args_list}
        assert "val/loss_giou" in logged_loss_names
        assert "val/loss_giou" not in direct_log_names
        assert "val/giou" not in direct_log_names

    @pytest.mark.parametrize("inductor_cudagraphs", [True, False])
    def test_marks_cudagraph_step_only_on_inductor_path(self, mocker, inductor_cudagraphs, tmp_path):
        """Validation on the Inductor CUDA graph path marks each step, since the compiled model records an eval graph.

        With ``eval_base_model=True`` or ``use_ema=False`` the ``OptimizedModule`` itself runs validation; the results
        kept by ``COCOEvalCallback`` across the epoch must not alias a graph output the next batch rewrites.
        """
        mark_step = mocker.patch("rfdetr.training.module_model.torch.compiler.cudagraph_mark_step_begin")
        tc = _base_train_config(tmp_path, compute_val_loss=False)
        module, fake_model, _, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        module._inductor_cudagraphs = inductor_cudagraphs
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        assert mark_step.call_count == (1 if inductor_cudagraphs else 0)

    def test_auto_val_loss_skips_criterion_without_a_loss_monitor(self, tmp_path):
        """compute_val_loss='auto' skips validation loss when no configured consumer monitors it."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        result = module.validation_step((samples, targets), batch_idx=0)

        fake_criterion.assert_not_called()
        assert "results" in result and "targets" in result

    def test_auto_val_loss_keeps_criterion_for_callback_monitor(self, tmp_path):
        """compute_val_loss='auto' retains validation loss for a callback monitoring val/loss."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.trainer = SimpleNamespace(callbacks=[SimpleNamespace(monitor="val/loss")])
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        fake_criterion.assert_called_once_with({}, targets)
        assert any(call.args[0] == "val/loss" for call in module.log.call_args_list)

    @pytest.mark.parametrize(
        "callbacks, expected",
        [
            pytest.param(
                [RFDETREarlyStopping(monitor_regular="val/loss")],
                True,
                id="real-rfdetr-early-stopping-monitor-lives-in-monitor-regular-not-monitor",
            ),
            pytest.param(
                [SimpleNamespace(monitor="__rfdetr_effective_map__", _monitor_ema="val/loss")],
                True,
                id="ema-monitor-attribute-consuming-val-loss",
            ),
            pytest.param([], False, id="empty-callback-list-resolves-like-the-unattached-case"),
        ],
    )
    def test_auto_val_loss_scan_resolves_policy(self, tmp_path, callbacks: list, expected: bool):
        """compute_val_loss='auto' scans trainer.callbacks for any callback that would consume val/loss.

        Covers three cases a naive scan (monitor attribute only) could miss: ``RFDETREarlyStopping``'s real target lives
        in ``_monitor_regular`` (its ``monitor`` is always the synthetic ``__rfdetr_effective_map__`` key it injects
        itself); RF-DETR's EMA callbacks track their metric via ``_monitor_ema`` instead; and an attached trainer with
        an empty callback list must resolve exactly like the unattached case rather than raising or falling back to
        computing the loss.
        """
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=callbacks)

        assert module._should_compute_val_loss is expected

    def test_explicit_val_loss_disable_rejects_callback_monitor(self, tmp_path):
        """compute_val_loss=False rejects a callback that would consume val/loss."""
        tc = _base_train_config(tmp_path, compute_val_loss=False)
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[SimpleNamespace(monitor="val/loss")])

        with pytest.raises(ValueError) as exc_info:
            module.on_fit_start()
        assert "compute_val_loss=False is incompatible" in str(exc_info.value)

    def test_standalone_validate_skips_fit_start_rejection(self, mocker, tmp_path):
        """A standalone trainer.validate() with compute_val_loss=False runs despite a callback monitoring val/loss.

        The rejection lives in ``on_fit_start``, which PTL invokes only when ``trainer.state.fn`` is ``FITTING``; a bare
        ``validate()`` call sets it to ``VALIDATING`` and skips the hook entirely. The asymmetry is intentional (a
        validation-only run has no scheduler or early-stopping loop to starve), so this test documents the current
        behaviour instead of asserting a raise.
        """
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, compute_val_loss=False, num_workers=0)

        class _ValLossMonitorCallback(Callback):
            """Callback declaring a val/loss monitor, as PTL-native checkpoint and early-stopping callbacks do."""

            monitor = "val/loss"

        fake_postprocess = MagicMock(side_effect=_helpers_fake_postprocess)

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), fake_postprocess),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4))
        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = Trainer(
            limit_val_batches=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            callbacks=[_ValLossMonitorCallback()],
        )
        trainer.validate(module, datamodule)

        # The validation batch reached postprocess, so the loop ran to completion instead of raising the guard.
        fake_postprocess.assert_called_once()

    def test_forwards_through_ema_model_not_base_by_default(self, tmp_path):
        """Validation must forward through the EMA-averaged model, not the base model, by default.

        Regression for #416: this single forward replaces the duplicate base+EMA pair COCOEvalCallback would otherwise
        run, and is the ~3-3.5%-of-epoch saving behind the eval_base_model default.
        """
        tc = _base_train_config(tmp_path, use_ema=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {"base": True}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        ema_model = MagicMock(name="ema_model", return_value={"ema": True})
        fake_ema_callback = SimpleNamespace(_average_model=SimpleNamespace(module=SimpleNamespace(model=ema_model)))
        module.trainer = SimpleNamespace(callbacks=[fake_ema_callback])
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        ema_model.assert_called_once()
        fake_model.assert_not_called()

    def test_falls_back_to_base_model_when_ema_not_warmed_up(self, tmp_path):
        """Validation must fall back to the base model if the EMA callback has no averaged model yet.

        The very first validation can run before RFDETREMACallback has built its averaged model; forwarding through a
        missing EMA model would crash instead of degrading to the base weights.
        """
        tc = _base_train_config(tmp_path, use_ema=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {"base": True}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        fake_ema_callback = SimpleNamespace(_average_model=None)
        module.trainer = SimpleNamespace(callbacks=[fake_ema_callback])
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once()

    def test_eval_base_model_does_not_touch_trainer(self, tmp_path):
        """eval_base_model=True must not access self.trainer at all — it returns the base model unconditionally.

        validation_step has to stay usable on a module with no Trainer attached, as every other test in this class
        relies on; the opt-in path must short-circuit before any trainer lookup.
        """
        tc = _base_train_config(tmp_path, eval_base_model=True)
        module, fake_model, _, _ = _build_module(train_config=tc, tmp_path=tmp_path)

        assert module._resolve_eval_model() is fake_model

    def test_falls_back_to_base_model_when_trainer_unattached(self, tmp_path):
        """Validation must fall back to the base model — not raise — when the module isn't attached to a Trainer at all.

        ``LightningModule.trainer`` raises ``RuntimeError`` (not ``None``) when unattached, so a naive
        ``getattr(self.trainer, ...)`` would crash instead of falling back (regression: module never has a Trainer wired
        up in this test module, matching the direct/standalone validation_step-call scenario).
        """
        tc = _base_train_config(tmp_path, use_ema=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {"base": True}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once()


class TestValidationLossSkipNotice:
    """on_validation_epoch_start() announces an 'auto' policy that resolved to skipping validation-loss computation.

    The consumer scan cannot see a human reading a ``val/loss`` curve from ``metrics.csv``, TensorBoard, or Weights &
    Biases, so the resolution is stated once instead of the curve vanishing silently.
    """

    def _build_module_with_trainer(self, tmp_path, **trainer_state):
        """Return an 'auto'-policy module whose stub trainer carries the given validation-loop state."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        state = dict(callbacks=[], sanity_checking=False, is_global_zero=True)
        state.update(trainer_state)
        module.trainer = SimpleNamespace(**state)
        return module

    def test_notice_is_emitted_once_across_validation_epochs(self, mocker, tmp_path):
        """The skip notice is logged on the first validation epoch only, not once per epoch.

        Every validation epoch re-enters the hook, so an unguarded notice would repeat for the whole run and drown the
        per-epoch metric lines it sits next to.
        """
        mock_logger = mocker.patch("rfdetr.training.module_model.logger")
        module = self._build_module_with_trainer(tmp_path)

        module.on_validation_epoch_start()
        module.on_validation_epoch_start()

        mock_logger.info.assert_called_once()

    def test_no_notice_when_a_callback_consumes_val_loss(self, mocker, tmp_path):
        """No notice is logged while a callback monitors val/loss, because the loss is still computed."""
        mock_logger = mocker.patch("rfdetr.training.module_model.logger")
        module = self._build_module_with_trainer(tmp_path, callbacks=[SimpleNamespace(monitor="val/loss")])

        module.on_validation_epoch_start()

        mock_logger.info.assert_not_called()

    def test_sanity_check_defers_notice(self, mocker, tmp_path):
        """The pre-training sanity-check validation pass does not consume the one-shot notice.

        Sanity checking runs before training starts and is invisible in most run logs; emitting the notice there would
        spend the single announcement on a pass the user is least likely to be watching.
        """
        mock_logger = mocker.patch("rfdetr.training.module_model.logger")
        module = self._build_module_with_trainer(tmp_path, sanity_checking=True)

        module.on_validation_epoch_start()

        mock_logger.info.assert_not_called()


class TestValidationLossPolicyCaching:
    """_should_compute_val_loss reuses the resolution that on_validation_epoch_start cached for the running epoch."""

    def test_per_batch_reads_reuse_epoch_resolution(self, tmp_path):
        """A callback attached mid-epoch does not change the policy the running validation epoch already resolved.

        The resolution walks every configured callback, so ``validation_step`` must not redo it per batch. Mutating
        ``trainer.callbacks`` after the epoch hook ran is the observable proxy: an unchanged answer proves the batch
        read came from the cached resolution rather than a fresh scan.
        """
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[], sanity_checking=False, is_global_zero=True)
        module.on_validation_epoch_start()

        module.trainer.callbacks.append(SimpleNamespace(monitor="val/loss"))

        assert module._should_compute_val_loss is False

    def test_resolution_is_refreshed_at_every_validation_epoch(self, tmp_path):
        """Each validation epoch re-resolves the policy, so a fit -> validate transition cannot serve a stale answer."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[], sanity_checking=False, is_global_zero=True)
        module.on_validation_epoch_start()

        module.trainer.callbacks.append(SimpleNamespace(monitor="val/loss"))
        module.on_validation_epoch_start()

        assert module._should_compute_val_loss is True


@pytest.mark.parametrize(
    ("step", "prefix", "loss_flag"),
    [("validation_step", "val", "compute_val_loss"), ("test_step", "test", "compute_test_loss")],
)
class TestEvalStepOutputs:
    """validation_step() and test_step() share one forward+postprocess contract and differ in the metric prefix."""

    @staticmethod
    def _run(tmp_path: Path, step: str, loss_flag: str, compute_loss: bool = True) -> SimpleNamespace:
        """Run one eval step on a fake module and return the pieces the tests inspect."""
        tc = _base_train_config(tmp_path, **{loss_flag: compute_loss})
        module, fake_model, fake_criterion, fake_pp = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        result = getattr(module, step)((samples, targets), batch_idx=0)
        return SimpleNamespace(
            result=result,
            module=module,
            model=fake_model,
            criterion=fake_criterion,
            postprocess=fake_pp,
            samples=samples,
        )

    @pytest.mark.parametrize("key", ["results", "targets"])
    def test_returns_dict_with_required_key(self, tmp_path, step, prefix, loss_flag, key):
        """The output dict carries both 'results' and 'targets' for downstream metric computation."""
        run = self._run(tmp_path, step, loss_flag)

        assert key in run.result

    def test_postprocess_called_with_orig_sizes(self, tmp_path, step, prefix, loss_flag):
        """The postprocessor receives original image sizes to rescale predictions."""
        run = self._run(tmp_path, step, loss_flag)

        run.postprocess.assert_called_once()
        assert run.postprocess.call_args[0][1].shape == (2, 2)

    def test_logs_loss_under_own_prefix_only(self, tmp_path, step, prefix, loss_flag):
        """The loss is logged once as '<prefix>/loss' and never under the other step's namespace."""
        run = self._run(tmp_path, step, loss_flag)

        logged = [c[0][0] for c in run.module.log.call_args_list]
        assert logged.count(f"{prefix}/loss") == 1
        assert ({"val/loss", "test/loss"} - {f"{prefix}/loss"}).isdisjoint(logged)

    def test_model_called_with_samples_only(self, tmp_path, step, prefix, loss_flag):
        """The step passes only samples (not targets) to the model forward."""
        run = self._run(tmp_path, step, loss_flag)

        run.model.assert_called_once_with(run.samples)

    def test_can_disable_loss_computation(self, tmp_path, step, prefix, loss_flag):
        """Disabling the loss flag skips the criterion call and the '<prefix>/loss' log."""
        run = self._run(tmp_path, step, loss_flag, compute_loss=False)

        run.criterion.assert_not_called()
        assert f"{prefix}/loss" not in [c[0][0] for c in run.module.log.call_args_list]
        assert "results" in run.result and "targets" in run.result


class TestPredictStep:
    """Tests for predict_step().

    Verifies that only samples (not targets) are passed to the model, that postprocess receives the correct original
    sizes, and that the postprocessor output is returned directly to the caller.
    """

    def test_calls_postprocess_with_orig_sizes(self, build_module):
        """Postprocessor must receive a (batch, 2) tensor of original image sizes."""
        module, fake_model, _, fake_pp = build_module()
        samples, targets = _make_batch(batch_size=3)
        fake_model.return_value = {}

        module.predict_step((samples, targets), batch_idx=0)

        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (3, 2)

    def test_returns_postprocess_output(self, build_module):
        """predict_step must return the postprocessor output directly to the caller."""
        module, fake_model, _, fake_pp = build_module()
        samples, targets = _make_batch()
        fake_model.return_value = {}
        expected_output = [{"boxes": torch.zeros(1, 4)}]
        fake_pp.return_value = expected_output

        assert module.predict_step((samples, targets), batch_idx=0) is expected_output

    def test_model_called_with_samples_only(self, build_module):
        """Inference must pass only samples (not targets) to the model forward."""
        module, fake_model, _, _ = build_module()
        samples, targets = _make_batch()
        fake_model.return_value = {}

        module.predict_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once_with(samples)

    def test_default_dataloader_idx_is_zero(self, build_module):
        """predict_step must work with the default dataloader_idx without errors."""
        module, fake_model, _, _ = build_module()
        fake_model.return_value = {}

        # Should not raise with default dataloader_idx.
        module.predict_step(_make_batch(), batch_idx=0)


_MASK_LOSS_WEIGHTS = {"loss_mask_ce": 1.0, "loss_mask_dice": 1.0}


def _seg_module(tmp_path: Path | None, **train_overrides: Any) -> RFDETRModelModule:
    """Build a module around a real tiny segmentation ``LWDETR`` and a real mask criterion; postprocess stays fake.

    Args:
        tmp_path: Output directory for the train config, or ``None``.
        **train_overrides: Train config fields to override.

    Returns:
        The module, with its model in eval mode.

    Examples:
        >>> module = _seg_module(None)
        >>> isinstance(module.criterion, SetCriterion)
        True
    """
    module, *_ = _build_module(
        model_config=_base_model_config(segmentation_head=True),
        train_config=_base_train_config(tmp_path, **train_overrides),
    )
    module.model = _build_seg_lwdetr().eval()
    module.criterion = SetCriterion(
        num_classes=_CLASSES,
        matcher=HungarianMatcher(),
        weight_dict=_MASK_LOSS_WEIGHTS,
        focal_alpha=0.25,
        losses=["masks"],
        group_detr=1,
        mask_point_sample_ratio=16,
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return module


def _seg_batch() -> tuple[NestedTensor, list[dict[str, Tensor]]]:
    """Build a batch matching ``_build_seg_lwdetr``'s input size, with one mask target per image.

    Returns:
        The padded images and one target dict per image.

    Examples:
        >>> samples, targets = _seg_batch()
        >>> tuple(samples.tensors.shape), sorted(targets[0])
        ((2, 3, 8, 8), ['boxes', 'labels', 'masks', 'orig_size'])
    """
    samples = NestedTensor(
        torch.zeros(_BATCH, 3, _IMAGE, _IMAGE), torch.zeros(_BATCH, _IMAGE, _IMAGE, dtype=torch.bool)
    )
    generator = torch.Generator().manual_seed(0)
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "masks": (torch.rand(1, _IMAGE, _IMAGE, generator=generator) > 0.5).float(),
            "orig_size": torch.tensor([_IMAGE, _IMAGE]),
        }
        for _ in range(_BATCH)
    ]
    return samples, targets


class _ImagesOnlyDetector(torch.nn.Module):
    """A detection model stand-in whose forward takes the images alone, as a registered backbone's model may.

    Examples:
        >>> tuple(_ImagesOnlyDetector()(torch.zeros(1, 3, 8, 8))["pred_logits"].shape)
        (2, 3, 5)
    """

    def forward(self, samples: Any) -> dict[str, Tensor]:
        """Return fixed logits for a batch of ``_BATCH`` images, whatever the input."""
        return {"pred_logits": torch.zeros(_BATCH, _QUERIES, _CLASSES)}


class TestEvalStepSegmentationMasks:
    """The loss path keeps every layer's mask loss; the metric-only path skips the masks nothing reads."""

    @pytest.mark.parametrize(
        ("step", "overrides"),
        [
            pytest.param("validation_step", {"compute_val_loss": True}, id="validation"),
            pytest.param("test_step", {"compute_test_loss": True}, id="test"),
        ],
    )
    def test_loss_includes_every_layers_mask_loss(self, tmp_path: Path, step: str, overrides: dict[str, Any]) -> None:
        """The criterion gets the auxiliary-layer and encoder-proposal masks it computes the mask losses of."""
        module = _seg_module(tmp_path, **overrides)
        loss_dicts: list[dict[str, Tensor]] = []
        module.criterion.register_forward_hook(lambda _module, _inputs, output: loss_dicts.append(output))

        with torch.no_grad():
            getattr(module, step)(_seg_batch(), batch_idx=0)

        assert len(loss_dicts) == 1, f"{step} must run the criterion once when its loss is enabled"
        expected = {"loss_mask_ce", "loss_mask_ce_0", "loss_mask_ce_1", "loss_mask_ce_enc"}
        assert expected <= set(loss_dicts[0]), f"missing mask losses: {sorted(expected - set(loss_dicts[0]))}"

    @pytest.mark.parametrize(
        ("step", "overrides"),
        [
            pytest.param("validation_step", {"compute_val_loss": False}, id="validation"),
            pytest.param("test_step", {"compute_test_loss": False}, id="test"),
        ],
    )
    def test_step_without_loss_skips_auxiliary_masks(
        self, tmp_path: Path, step: str, overrides: dict[str, Any]
    ) -> None:
        """With the loss off, only the final layer's masks are computed and handed to postprocess."""
        module = _seg_module(tmp_path, **overrides)

        with torch.no_grad():
            getattr(module, step)(_seg_batch(), batch_idx=0)

        outputs = module.postprocess.call_args[0][0]
        assert outputs["pred_masks"].shape == (_BATCH, _QUERIES, _IMAGE, _IMAGE)
        assert all("pred_masks" not in aux for aux in outputs["aux_outputs"]), "aux masks computed without a loss"
        assert "pred_masks" not in outputs["enc_outputs"], "encoder masks computed without a loss"

    @pytest.mark.parametrize("segmentation_config", [False, True])
    def test_model_without_segmentation_head_is_called_with_images_alone(
        self, tmp_path: Path, segmentation_config: bool
    ) -> None:
        """A model without a segmentation head gets no ``all_layer_masks`` keyword with a loss, whatever the config.

        Detection and keypoint models are called as before, and so is a stand-in whose forward takes the images alone.
        """
        module, *_ = _build_module(
            model_config=_base_model_config(segmentation_head=segmentation_config),
            train_config=_base_train_config(tmp_path, compute_val_loss=True),
        )
        module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
        module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]
        module.model = _ImagesOnlyDetector()

        result = module.validation_step(_seg_batch(), batch_idx=0)

        assert len(result["results"]) == 1, "postprocess output must reach the callback"

    def test_criterion_on_default_eval_outputs_names_the_keyword(self, tmp_path: Path) -> None:
        """A mask loss on eval outputs without the auxiliary masks raises a ``ValueError`` that says how to get them."""
        module = _seg_module(tmp_path)
        samples, targets = _seg_batch()
        with torch.no_grad():
            outputs = module.model(samples)

        with pytest.raises(ValueError) as exc_info:
            module.criterion(outputs, targets)

        assert "all_layer_masks=True" in str(exc_info.value)

    def test_criterion_without_mask_targets_takes_default_eval_outputs(self, tmp_path: Path) -> None:
        """Targets without masks need no auxiliary masks, so default eval outputs go through every layer's loss."""
        module = _seg_module(tmp_path)
        module.criterion = SetCriterion(
            num_classes=_CLASSES,
            matcher=HungarianMatcher(),
            weight_dict={"loss_ce": 1.0},
            focal_alpha=0.25,
            losses=["labels"],
            group_detr=1,
        )
        samples, targets = _seg_batch()
        box_targets = [{key: value for key, value in target.items() if key != "masks"} for target in targets]
        with torch.no_grad():
            outputs = module.model(samples)

        assert "loss_ce_enc" in module.criterion(outputs, box_targets)
