# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

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


@patch("rfdetr.detr.is_main_process", return_value=False)
@patch("rfdetr.training.auto_batch.resolve_auto_batch_config")
@patch("rfdetr.training.build_trainer")
@patch("rfdetr.training.RFDETRDataModule")
@patch("rfdetr.training.RFDETRModelModule")
@patch("rfdetr.detr._ensure_model_on_device")
def test_train_auto_batch_ensures_model_on_device_before_resolve(
    mock_ensure: MagicMock,
    _mock_module: MagicMock,
    _mock_data_module: MagicMock,
    _mock_build_trainer: MagicMock,
    mock_resolve: MagicMock,
    _mock_is_main: MagicMock,
) -> None:
    """_ensure_model_on_device must be called before resolve_auto_batch_config when batch_size='auto'."""
    auto_result = SimpleNamespace(safe_micro_batch=4, recommended_grad_accum_steps=1, effective_batch_size=4)
    call_order: list[str] = []

    def _ensure_side_effect(model: object) -> None:
        call_order.append("ensure")

    def _resolve_side_effect(**_kwargs: object) -> object:
        call_order.append("resolve")
        return auto_result

    mock_ensure.side_effect = _ensure_side_effect
    mock_resolve.side_effect = _resolve_side_effect

    train_config = SimpleNamespace(
        batch_size="auto",
        grad_accum_steps=99,
        dataset_dir=None,
        resume=None,
        class_names=None,
        save_dataset_grids=False,
    )
    mock_self = MagicMock()
    mock_self.model_config = SimpleNamespace(model_name=None)
    mock_self.get_train_config.return_value = train_config

    RFDETR.train(mock_self)

    assert train_config.batch_size == 4
    assert train_config.grad_accum_steps == 1
    mock_ensure.assert_called_once_with(mock_self.model)
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
