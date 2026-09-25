# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Gradient clipping must act on the true gradient on both optimization paths.

Under fp16 mixed precision the backward pass produces gradients multiplied by the ``GradScaler`` scale, and
Lightning unscales them inside ``MixedPrecision.optimizer_step`` right before the optimizer step. Clipping has to
happen after that unscale; clipping the scaled gradients and unscaling afterwards hands the optimizer
``clip_max_norm / scale`` instead of ``clip_max_norm``. Detection/segmentation models clip through Lightning's
automatic optimization, keypoint models clip inside ``RFDETRModelModule`` under manual optimization, so both paths
are run through a real ``Trainer.fit()``.

Lightning rewrites ``precision="16-mixed"`` to bf16 on CPU, so the fp16 cases pass a ``MixedPrecision`` plugin with
an explicit CPU ``GradScaler``, which is the plugin Lightning builds for fp16 runs on CUDA and MPS.

``_TinyModel`` with ``_FakeCriterion`` gives ``loss = dummy.mean()``, so the true gradient on ``dummy`` is ``1.0``
(the mean over an accumulation window is also ``1.0``) and the gradient the optimizer must consume after clipping
is exactly ``clip_max_norm``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.plugins.precision import MixedPrecision

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

_CLIP_MAX_NORM = 0.1
_INIT_SCALE = 2.0**16


class _KeypointCriterion(_FakeCriterion):
    """``_FakeCriterion`` that accepts the ``num_boxes`` override the manual-optimization path requires."""

    supports_loss_normalizer_override = True


class _CaptureConsumedGradient(Callback):
    """Record ``_TinyModel.dummy.grad`` each time the optimizer's own ``step()`` completes.

    A step post-hook on the underlying optimizer runs after the closure (fp32 automatic optimization runs backward
    inside ``step()``), after Lightning's unscale and after clipping on every path, and before the next ``zero_grad()``,
    so it sees exactly the gradient the update consumed. Steps the ``GradScaler`` skips are not recorded.
    """

    def __init__(self) -> None:
        """Start with no recorded gradients."""
        self.grads: list[float] = []

    def on_train_start(self, trainer: Trainer, pl_module: RFDETRModelModule) -> None:
        """Register the step post-hook once the optimizers exist.

        Examples:
            Requires a Trainer whose optimizers were built by ``Trainer.fit()``.
            >>> callable(_CaptureConsumedGradient.on_train_start)  # doctest: +SKIP
            True
        """

        def _record(optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
            """Store the current gradient on ``dummy``."""
            self.grads.append(pl_module.model.dummy.grad.item())

        for optimizer in trainer.optimizers:
            optimizer.register_step_post_hook(_record)


class TestClipBeforeOptimizerStep:
    """The optimizer must consume the true gradient clipped to ``clip_max_norm``, whatever the precision plugin."""

    @pytest.mark.parametrize(
        "keypoints",
        [
            pytest.param(False, id="automatic-detection"),
            pytest.param(True, id="manual-keypoint"),
        ],
    )
    @pytest.mark.parametrize(
        "fp16_grad_scaler",
        [
            pytest.param(False, id="fp32"),
            pytest.param(True, id="fp16-mixed-gradscaler"),
        ],
    )
    @pytest.mark.parametrize(
        "grad_accum_steps",
        [
            pytest.param(1, id="no-accumulation"),
            pytest.param(2, id="accumulate-2"),
        ],
    )
    def test_optimizer_consumes_clipped_true_gradient(
        self, tmp_path: Path, keypoints: bool, fp16_grad_scaler: bool, grad_accum_steps: int
    ) -> None:
        """One optimizer step per accumulation window must see ``dummy.grad == clip_max_norm``.

        With the clip applied to scaled gradients the fp16 keypoint case sees ``clip_max_norm / 2**16`` instead.
        """
        keypoint_kwargs: dict[str, Any] = (
            {"use_grouppose_keypoints": True, "num_keypoints_per_class": [17]} if keypoints else {}
        )
        mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=3, **keypoint_kwargs)
        tc = TrainConfig(
            dataset_dir=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            epochs=1,
            batch_size=2,
            num_workers=0,
            grad_accum_steps=grad_accum_steps,
            clip_max_norm=_CLIP_MAX_NORM,
            tensorboard=False,
            use_ema=False,
        )
        # Mirror build_trainer(): Lightning owns accumulation and clipping on the automatic path only.
        trainer_kwargs: dict[str, Any] = (
            {} if keypoints else {"accumulate_grad_batches": grad_accum_steps, "gradient_clip_val": tc.clip_max_norm}
        )
        if fp16_grad_scaler:
            scaler = torch.amp.GradScaler("cpu", init_scale=_INIT_SCALE)
            trainer_kwargs["plugins"] = [MixedPrecision("16-mixed", "cpu", scaler=scaler)]
        criterion = _KeypointCriterion() if keypoints else _FakeCriterion()
        capture = _CaptureConsumedGradient()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(criterion, MagicMock(side_effect=_fake_postprocess)),
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
                fast_dev_run=grad_accum_steps,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[capture],
                **trainer_kwargs,
            )
            trainer.fit(module, datamodule)

        assert capture.grads == [pytest.approx(_CLIP_MAX_NORM, rel=1e-4)], (
            f"optimizer consumed {capture.grads}; expected one step on the true gradient 1.0 clipped to "
            f"{_CLIP_MAX_NORM}"
        )
        if fp16_grad_scaler:
            # Guard against the scaler silently disabling itself, which would turn the fp16 cases into fp32 ones.
            assert scaler.is_enabled(), "GradScaler was disabled, so the fp16 case no longer exercises scaling"
            assert scaler.get_scale() == _INIT_SCALE, (
                f"GradScaler scale changed to {scaler.get_scale()}; expected it to stay at {_INIT_SCALE} after one "
                "finite step"
            )
