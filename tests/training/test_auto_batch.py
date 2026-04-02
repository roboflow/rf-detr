# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

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


def test_probe_step_raises_when_loss_keys_do_not_overlap_weight_keys():
    """_probe_step must fail fast when weighted loss would be empty."""

    class _DummyCriterion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight_dict = {"loss_bbox": 1.0}

        def forward(self, outputs, targets):
            return {"loss_ce": torch.tensor(1.0)}

    class _DummyModel(torch.nn.Module):
        def forward(self, samples, targets):
            return {}

    model = _DummyModel()
    criterion = _DummyCriterion()

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
        )


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
    train_config = SimpleNamespace(batch_size="auto", auto_batch_target_effective=16)
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

    ok = auto_batch._probe_step(
        model=model,
        criterion=criterion,
        micro_batch_size=1,
        resolution=mc.resolution,
        device=device,
        num_classes=mc.num_classes,
        amp=False,
        segmentation_head=True,
    )
    assert ok is True
