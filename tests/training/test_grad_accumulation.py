# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Gradient-level contract of gradient accumulation on the automatic-optimization path.

Lightning normalises the loss returned by ``training_step`` by ``accumulate_grad_batches`` itself
(``ClosureResult.from_training_step_output``) before calling ``backward()``. ``RFDETRModelModule`` must therefore
return the loss unscaled; dividing it again scales every accumulated gradient by ``1/N**2``. Only a real
``Trainer.fit()`` exercises that closure, so this cannot be a stub-trainer unit test.

``_TinyModel`` with ``_FakeCriterion`` gives ``loss = dummy.mean()``, so every microbatch contributes
``d(loss)/d(dummy) = 1`` and the gradient handed to the optimizer after ``N`` accumulated microbatches is exactly
``1.0`` when the normalisation is applied once, and ``1/N`` when it is applied twice.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Callback, Trainer

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel


class _CaptureGradient(Callback):
    """Record the gradient on ``_TinyModel.dummy`` at the first optimizer step, before the optimizer consumes it."""

    def __init__(self) -> None:
        self.grad: float | None = None

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: RFDETRModelModule, optimizer: object) -> None:
        """Capture the accumulated gradient once."""
        if self.grad is None:
            self.grad = pl_module.model.dummy.grad.item()


class TestAutomaticOptimizationAccumulationScaling:
    """The accumulated gradient must be the mean of the microbatch gradients, not the mean divided by ``N`` again."""

    @pytest.mark.parametrize("accumulate_grad_batches", [1, 2, 4])
    def test_accumulated_gradient_is_mean_of_microbatch_gradients(
        self, tmp_path: Path, accumulate_grad_batches: int
    ) -> None:
        """``N`` microbatches at ``accumulate_grad_batches=N`` must leave a gradient of ``1.0`` on ``dummy``."""
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=3)
        tc = TrainConfig(
            dataset_dir=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            num_workers=0,
            grad_accum_steps=accumulate_grad_batches,
            tensorboard=False,
        )
        capture = _CaptureGradient()

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
            trainer = Trainer(
                fast_dev_run=accumulate_grad_batches,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=0.0,
                callbacks=[capture],
            )
            trainer.fit(module, datamodule)

        assert capture.grad == pytest.approx(1.0)
