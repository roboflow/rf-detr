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
from rfdetr.utilities.imports import _IS_TORCH_XLA_INSTALLED

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

torch_xla_only = pytest.mark.skipif(not _IS_TORCH_XLA_INSTALLED, reason="torch_xla not installed")

_CLIP_MAX_NORM = 0.1
_INIT_SCALE = 2.0**16

#: ``torch.amp.GradScaler`` is the device-agnostic API added in torch 2.3; on torch 2.2 (the project's own floor)
#: ``torch.amp`` has no ``GradScaler`` attribute at all, so constructing one unconditionally raises ``AttributeError``.
_HAS_DEVICE_AGNOSTIC_GRAD_SCALER = hasattr(torch.amp, "GradScaler")


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
            A minimal stand-in trainer/module (no real ``Trainer.fit()`` needed) proves the hook fires and
            records the gradient exactly once per optimizer step.
            >>> from unittest.mock import MagicMock
            >>> optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
            >>> stub_trainer = MagicMock(optimizers=[optimizer])
            >>> stub_module = MagicMock()
            >>> stub_module.model.dummy.grad.item.return_value = 1.0
            >>> callback = _CaptureConsumedGradient()
            >>> callback.on_train_start(stub_trainer, stub_module)
            >>> _ = optimizer.step()
            >>> callback.grads
            [1.0]
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
        "precision_mode",
        [
            "fp32",
            pytest.param(
                "fp16-mixed-gradscaler",
                marks=pytest.mark.skipif(
                    not _HAS_DEVICE_AGNOSTIC_GRAD_SCALER,
                    reason="torch.amp.GradScaler(device, ...) needs torch>=2.3",
                ),
            ),
            "bf16-mixed",
        ],
    )
    @pytest.mark.parametrize("grad_accum_steps", [1, 2])
    def test_optimizer_consumes_clipped_true_gradient(
        self, tmp_path: Path, keypoints: bool, precision_mode: str, grad_accum_steps: int
    ) -> None:
        """One optimizer step per accumulation window must see ``dummy.grad == clip_max_norm``.

        With the clip applied to scaled gradients the fp16 keypoint case sees ``clip_max_norm / 2**16`` instead.
        ``bf16-mixed`` takes the ``scaler=None`` branch Lightning's ``MixedPrecision`` uses for BF16, structurally
        distinct from the FP16 GradScaler-present inline path.
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
        if precision_mode == "fp16-mixed-gradscaler":
            scaler = torch.amp.GradScaler("cpu", init_scale=_INIT_SCALE)
            trainer_kwargs["plugins"] = [MixedPrecision("16-mixed", "cpu", scaler=scaler)]
        elif precision_mode == "bf16-mixed":
            trainer_kwargs["plugins"] = [MixedPrecision("bf16-mixed", "cpu")]
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
        if precision_mode == "fp16-mixed-gradscaler":
            # Guard against the scaler silently disabling itself, which would turn the fp16 cases into fp32 ones.
            assert scaler.is_enabled(), "GradScaler was disabled, so the fp16 case no longer exercises scaling"
            assert scaler.get_scale() == _INIT_SCALE, (
                f"GradScaler scale changed to {scaler.get_scale()}; expected it to stay at {_INIT_SCALE} after one "
                "finite step"
            )


class _CaptureRawGradient(Callback):
    """Record ``_TinyModel.dummy.grad`` from ``on_before_optimizer_step``, before RF-DETR's own hook clips it.

    Lightning calls every registered callback's ``on_before_optimizer_step`` before the ``LightningModule``'s own hook
    of the same name, so a callback here sees the true, unclipped gradient — the opposite end of the gradient's
    lifecycle from :class:`_CaptureConsumedGradient`'s post-clip, post-step view.
    """

    def __init__(self) -> None:
        """Start with no recorded gradients."""
        self.grads: list[float] = []

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: RFDETRModelModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """Store the gradient as callbacks see it, ahead of the module's own clip.

        Examples:
            >>> callback = _CaptureRawGradient()
            >>> module = MagicMock()
            >>> module.model.dummy.grad.item.return_value = 1.0
            >>> callback.on_before_optimizer_step(MagicMock(), module, MagicMock())
            >>> callback.grads
            [1.0]
        """
        self.grads.append(pl_module.model.dummy.grad.item())


def test_callback_sees_unclipped_gradient_before_manual_clip(tmp_path: Path) -> None:
    """A callback's ``on_before_optimizer_step`` observes the true gradient, not the value clipping produces.

    Lightning fires every callback's ``on_before_optimizer_step`` before the ``LightningModule``'s own hook of the same
    name, so on the manual-optimization (keypoint) path a callback must see ``dummy.grad == 1.0`` (the true gradient)
    even though the optimizer itself ultimately consumes the clipped ``clip_max_norm``.
    """
    mc = RFDETRBaseConfig(
        pretrain_weights=None,
        device="cpu",
        num_classes=3,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    tc = TrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        clip_max_norm=_CLIP_MAX_NORM,
        tensorboard=False,
        use_ema=False,
    )
    raw_capture = _CaptureRawGradient()
    consumed_capture = _CaptureConsumedGradient()

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_KeypointCriterion(), MagicMock(side_effect=_fake_postprocess)),
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
            fast_dev_run=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[raw_capture, consumed_capture],
        )
        trainer.fit(module, datamodule)

    assert raw_capture.grads == [pytest.approx(1.0, rel=1e-4)], (
        f"callback observed {raw_capture.grads}; expected the true unclipped gradient 1.0"
    )
    assert consumed_capture.grads == [pytest.approx(_CLIP_MAX_NORM, rel=1e-4)], (
        f"optimizer consumed {consumed_capture.grads}; expected the clipped gradient {_CLIP_MAX_NORM}"
    )


class _OverflowCriterion(_KeypointCriterion):
    """Criterion producing a non-finite loss, to exercise the ``GradScaler``'s skip-step path."""

    def __call__(
        self, outputs: dict[str, Any], targets: list[dict[str, Any]], num_boxes: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the base keypoint loss scaled to ``inf``."""
        return {key: value * float("inf") for key, value in super().__call__(outputs, targets, num_boxes).items()}


@pytest.mark.skipif(
    not _HAS_DEVICE_AGNOSTIC_GRAD_SCALER,
    reason="torch.amp.GradScaler(device, ...) needs torch>=2.3",
)
def test_overflowing_gradient_is_safely_skipped_by_scaler(tmp_path: Path) -> None:
    """An overflowing (non-finite) loss may make the ``GradScaler`` skip the parameter update.

    This may happen after unscale and clipping.

    Every other case in this suite uses a clean finite gradient, so none of them exercise the scaler's own overflow-
    detection path. A forced-inf loss must make ``GradScaler`` skip the optimizer's parameter update (the post-hook that
    records a consumed gradient must never fire), back off its scale, and leave the model's parameters finite.
    """
    mc = RFDETRBaseConfig(
        pretrain_weights=None,
        device="cpu",
        num_classes=3,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    tc = TrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        clip_max_norm=_CLIP_MAX_NORM,
        tensorboard=False,
        use_ema=False,
    )
    scaler = torch.amp.GradScaler("cpu", init_scale=_INIT_SCALE)
    capture = _CaptureConsumedGradient()

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_OverflowCriterion(), MagicMock(side_effect=_fake_postprocess)),
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
            fast_dev_run=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[capture],
            plugins=[MixedPrecision("16-mixed", "cpu", scaler=scaler)],
        )
        trainer.fit(module, datamodule)

    assert capture.grads == [], (
        f"optimizer consumed {capture.grads}; expected the overflowing step to be skipped entirely, never reaching "
        "the post-clip optimizer.step() hook"
    )
    assert scaler.get_scale() < _INIT_SCALE, (
        f"GradScaler scale stayed at {scaler.get_scale()}; expected it to back off after a non-finite gradient"
    )
    assert all(torch.isfinite(param).all() for param in module.model.parameters()), (
        "model parameters must stay finite after a skipped (non-finite) optimizer step"
    )


def test_zero_clip_max_norm_disables_clipping_on_manual_path(tmp_path: Path) -> None:
    """``clip_max_norm=0`` on the manual-optimization (keypoint) path leaves the gradient unclipped.

    ``module_model.py``'s manual-clip helper only calls ``clip_grad_norm_`` when the resolved clip value is strictly
    positive, so the true gradient (``1.0``) must reach the optimizer unchanged when the configured value is exactly
    ``0`` — the disabling case the suite's other cases (``clip_max_norm=0.1``) never cover.
    """
    mc = RFDETRBaseConfig(
        pretrain_weights=None,
        device="cpu",
        num_classes=3,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    tc = TrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        clip_max_norm=0,
        tensorboard=False,
        use_ema=False,
    )
    capture = _CaptureConsumedGradient()

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_KeypointCriterion(), MagicMock(side_effect=_fake_postprocess)),
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
            fast_dev_run=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[capture],
        )
        trainer.fit(module, datamodule)

    assert capture.grads == [pytest.approx(1.0, rel=1e-4)], (
        f"optimizer consumed {capture.grads}; expected the unclipped true gradient 1.0 with clip_max_norm=0"
    )


@pytest.mark.xla
@torch_xla_only
def test_xla_reduces_gradients_before_manual_clipping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The XLA precision closure must reduce a local gradient before RF-DETR clips it.

    CPU PJRT provides the ``torch_xla`` runtime used by the XLA CI lane but cannot launch Lightning's TPU trainer,
    so this drives the real ``XLAPrecision.optimizer_step`` closure composition directly. The mocked collective is the
    external boundary: it scales a local gradient of ``1.0`` to ``0.25``. Correct ordering clips that reduced gradient
    to ``0.1``; clipping first and then reducing would leave ``0.025``. This proves the Lightning-plugin-to-RF-DETR
    hook ordering, not the PJRT collective's numerical implementation.
    """
    import torch_xla.core.xla_model as xm
    from pytorch_lightning.plugins.precision import XLAPrecision

    mc = RFDETRBaseConfig(
        pretrain_weights=None,
        device="cpu",
        num_classes=3,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    tc = TrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        clip_max_norm=_CLIP_MAX_NORM,
        tensorboard=False,
        use_ema=False,
    )

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_KeypointCriterion(), MagicMock(side_effect=_fake_postprocess)),
        ),
        patch("rfdetr.training.module_model.get_param_dict", side_effect=lambda args, model: _make_param_dicts(model)),
    ):
        module = RFDETRModelModule(mc, tc)

    precision = XLAPrecision()
    trainer = MagicMock(
        callbacks=[],
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        precision_plugin=precision,
    )
    trainer.lightning_module = module
    module._trainer = trainer
    module.log = MagicMock()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    reduce_gradients = MagicMock(
        side_effect=lambda received_optimizer: torch._foreach_mul_(
            [
                parameter.grad
                for group in received_optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ],
            0.25,
        )
    )
    monkeypatch.setattr(xm, "reduce_gradients", reduce_gradients)
    monkeypatch.setattr(xm, "mark_step", MagicMock())

    precision.optimizer_step(
        optimizer,
        module,
        closure=lambda: setattr(module.model.dummy, "grad", torch.ones_like(module.model.dummy)),
    )

    reduce_gradients.assert_called_once_with(optimizer)
    torch.testing.assert_close(module.model.dummy.grad, torch.full_like(module.model.dummy, _CLIP_MAX_NORM))
