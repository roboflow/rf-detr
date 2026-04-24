# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests: Trainer(fast_dev_run=2).fit(module, datamodule) — T7.

Verifies that the PTL training loop runs end-to-end without error for both
detection and segmentation configurations.  All heavy operations (build_model,
build_criterion_and_postprocessors, build_dataset, get_param_dict) are patched
so no real dataset or GPU is required.

Chapter 1 gate: these must pass before Chapter 2 begins.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Trainer

from rfdetr.config import SegmentationTrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from .helpers import (
    _fake_postprocess,
    _FakeCriterion,
    _FakeDataset,
    _FakeDatasetWithMasks,
    _FakePostProcess,
    _make_param_dicts,
    _TinyModel,
)

# ---------------------------------------------------------------------------
# Private helpers unique to smoke tests
# ---------------------------------------------------------------------------


def _make_trainer() -> Trainer:
    """Create a Trainer configured for minimal smoke-test runs."""
    return Trainer(
        fast_dev_run=2,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )


# ---------------------------------------------------------------------------
# Smoke test classes
# ---------------------------------------------------------------------------


class TestDetectionSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for detection."""

    def test_fit_runs_without_error(self, base_model_config, base_train_config):
        """Full PTL fit loop runs 2 train + 2 val batches without raising."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_training_step_called_expected_times(self, base_model_config, base_train_config):
        """fast_dev_run=2 must run exactly 2 training steps."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            original_training_step = module.training_step
            call_count = []

            def _counting_training_step(batch, batch_idx):
                call_count.append(1)
                return original_training_step(batch, batch_idx)

            module.training_step = _counting_training_step
            _make_trainer().fit(module, datamodule)

        assert sum(call_count) == 2

    def test_val_step_called_expected_times(self, base_model_config, base_train_config):
        """fast_dev_run=2 must run exactly 2 validation steps."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            original_validation_step = module.validation_step
            call_count = []

            def _counting_val_step(batch, batch_idx):
                call_count.append(1)
                return original_validation_step(batch, batch_idx)

            module.validation_step = _counting_val_step
            _make_trainer().fit(module, datamodule)

        assert sum(call_count) == 2

    def test_loss_decreases_or_is_finite(self, base_model_config, base_train_config):
        """Training loss must be finite (not NaN/inf) for the run to be valid."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        losses = []

        def _recording_criterion(outputs, targets):
            dummy = outputs.get("dummy", torch.zeros(1))
            loss = dummy.mean()
            losses.append(loss.detach().item())
            return {"loss_ce": loss}

        fake_criterion = MagicMock(side_effect=_recording_criterion)
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

        assert len(losses) > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in losses)


class TestSegmentationSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for segmentation."""

    def test_fit_runs_without_error(self, base_model_config, seg_train_config):
        """Full PTL fit loop runs 2 train + 2 val batches without raising."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDatasetWithMasks(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_segmentation_config_accepted(self, base_model_config, seg_train_config):
        """SegmentationTrainConfig must be accepted by both module and datamodule."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDatasetWithMasks()),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            assert isinstance(module.train_config, SegmentationTrainConfig)
            assert isinstance(datamodule.train_config, SegmentationTrainConfig)


class TestBuildTrainerSmoke:
    """Smoke tests for the ``build_trainer()`` public factory.

    Verifies that the full callback stack wired by ``build_trainer`` runs
    end-to-end with ``fast_dev_run``, using mocked internals so no real
    dataset or GPU is required.
    """

    def test_fit_via_build_trainer(self, base_model_config, base_train_config):
        """build_trainer() + trainer.fit(module, datamodule=datamodule) must not raise."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
            trainer.fit(module, datamodule=datamodule)


class _DDPModule(RFDETRModelModule):
    """RFDETRModelModule subclass for ddp_spawn smoke tests.

    Overrides ``configure_optimizers`` so ``get_param_dict`` is never called
    in child processes.  ``ddp_spawn`` forks child processes that unpack a
    pickled copy of this module; patches applied in the parent process are not
    visible in children, so the real ``get_param_dict`` would be called and
    would fail on ``_TinyModel`` (no ``.backbone`` attribute).

    Must be defined at module level so ``pickle`` can look up the class by
    qualified name when deserialising in the child process.
    """

    def configure_optimizers(self):
        """Minimal single-group AdamW — bypasses get_param_dict."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class _MultiScaleCheckDDPModule(RFDETRModelModule):
    """DDP-safe module that asserts on_train_batch_start mutation reaches training_step.

    With multi_scale=True and _FakeDataset's 32×32 images, on_train_batch_start
    interpolates samples.tensors to a multi-scale resolution (≥392 for
    RFDETRBaseConfig resolution=560).  This module raises AssertionError in
    training_step if the tensor height is still 32, meaning the in-place
    NestedTensor mutation did not propagate through the PTL batch-hook chain.

    Must be defined at module level so pickle can look up the class by qualified
    name when ddp_spawn deserialises it in the child process.

    Regression guard for issue #952.
    """

    def configure_optimizers(self):
        """Minimal single-group AdamW — bypasses get_param_dict."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        """Assert resize from on_train_batch_start propagated before calling super."""
        samples, _ = batch
        h = samples.tensors.shape[2]
        if h == 32:
            raise AssertionError(
                f"training_step received images at original 32-px height (h={h}). "
                "on_train_batch_start's in-place NestedTensor mutation did not "
                "propagate through the PTL hook chain. "
                "Regression of issue #952: resize bypass in DDP batch-hook chain."
            )
        return super().training_step(batch, batch_idx)


# ---------------------------------------------------------------------------
# Multi-scale hook propagation tests (issue #952 regression)
# ---------------------------------------------------------------------------


class TestMultiScaleHookPropagation:
    """on_train_batch_start resize must propagate to training_step via NestedTensor mutation.

    _FakeDataset emits 32×32 images.  With multi_scale=True and
    RFDETRBaseConfig(resolution=560, patch_size=14, num_windows=4) the computed
    scales start at 392, so none equal 32.  _MultiScaleCheckDDPModule raises
    AssertionError in training_step if h==32, making trainer.fit() fail when
    the in-place mutation does not propagate.
    """

    def test_mutation_persists_to_training_step(self, base_model_config, base_train_config):
        """Single-process: training_step must see resized tensors, not original 32×32."""
        mc = base_model_config()
        tc = base_train_config(multi_scale=True, use_ema=False, run_test=False)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), _FakePostProcess()),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
        ):
            module = _MultiScaleCheckDDPModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
            trainer.fit(module, datamodule=datamodule)


# Windows CI currently cannot run this smoke test because gloo DDP spawn fails
# with makeDeviceForHostname unsupported-device errors.
@pytest.mark.skipif(sys.platform == "win32", reason="gloo DDP spawn unsupported on Windows CI")
def test_ddp_spawn_fit_runs_without_error(base_model_config, base_train_config):
    """ddp_spawn with 2 CPU workers must run fast_dev_run=2 without error.

    ``ddp_spawn`` forks child processes, so all objects passed to
    ``trainer.fit()`` must be picklable.  ``MagicMock`` is NOT picklable;
    this test uses ``_FakePostProcess``, plain dataset instances, and
    ``_DDPModule`` (module-level class) instead.
    """
    mc = base_model_config()
    tc = base_train_config(use_ema=False, run_test=False, devices=2, strategy="ddp_spawn")

    fake_dataset = _FakeDataset(length=20)

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), _FakePostProcess()),
        ),
    ):
        module = _DDPModule(mc, tc)

    datamodule = RFDETRDataModule(mc, tc)
    # Pre-set datasets: build_dataset mock doesn't survive the spawn boundary.
    datamodule._dataset_train = fake_dataset
    datamodule._dataset_val = fake_dataset

    trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
    trainer.fit(module, datamodule=datamodule)


@pytest.mark.skipif(sys.platform == "win32", reason="gloo DDP spawn unsupported on Windows CI")
def test_ddp_spawn_multi_scale_mutation_propagates(base_model_config, base_train_config):
    """ddp_spawn with multi_scale=True must propagate on_train_batch_start resize to training_step.

    _MultiScaleCheckDDPModule raises AssertionError in training_step when the
    NestedTensor height is still 32 (original _FakeDataset size).  If trainer.fit()
    completes without error the PTL batch-hook reference chain is intact in DDP,
    i.e. the in-place mutation in on_train_batch_start is visible in training_step
    on both workers.

    Regression test for issue #952 on CPU DDP (non-Windows): confirms the
    transforms/resize propagation is not a Windows-only concern.
    """
    mc = base_model_config()
    tc = base_train_config(multi_scale=True, use_ema=False, run_test=False, devices=2, strategy="ddp_spawn")

    fake_dataset = _FakeDataset(length=20)

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), _FakePostProcess()),
        ),
    ):
        module = _MultiScaleCheckDDPModule(mc, tc)

    datamodule = RFDETRDataModule(mc, tc)
    # Pre-set datasets: build_dataset mock doesn't survive the spawn boundary.
    datamodule._dataset_train = fake_dataset
    datamodule._dataset_val = fake_dataset

    trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
    trainer.fit(module, datamodule=datamodule)
