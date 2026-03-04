# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from rfdetr import engine
from rfdetr.engine import _get_cuda_autocast_dtype, _is_cuda, evaluate, train_one_epoch
from rfdetr.util.misc import NestedTensor


def test_get_cuda_autocast_dtype_prefers_bfloat16_when_supported(monkeypatch) -> None:
    """Use bfloat16 when CUDA reports BF16 support."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert _get_cuda_autocast_dtype() == torch.bfloat16


def test_get_cuda_autocast_dtype_falls_back_to_float16_when_bfloat16_unsupported(monkeypatch) -> None:
    """Use float16 on CUDA devices that do not support BF16 (e.g. T4)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert _get_cuda_autocast_dtype() == torch.float16


class _DummyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, _targets):
        return {"pred": self.weight * samples.tensors.mean()}

    def update_drop_path(self, _value, _layers):
        return None

    def update_dropout(self, _value):
        return None


class _DummyEvalModel(nn.Module):
    def forward(self, samples):
        batch_size = samples.tensors.shape[0]
        return {
            "pred_boxes": torch.zeros((batch_size, 1, 4), dtype=samples.tensors.dtype),
            "pred_logits": torch.zeros((batch_size, 1, 2), dtype=samples.tensors.dtype),
        }


class _DummyCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_dict = {"loss_bbox": 1.0, "class_error": 1.0}

    def forward(self, outputs, _targets):
        base = outputs["pred"] if "pred" in outputs else outputs["pred_boxes"].sum()
        return {"loss_bbox": base * 0 + 1.0, "class_error": base * 0 + 0.5}


def _single_batch_data_loader():
    samples = NestedTensor(torch.ones((1, 3, 4, 4)), torch.zeros((1, 4, 4), dtype=torch.bool))
    targets = [{"image_id": torch.tensor(1), "orig_size": torch.tensor([4, 4])}]
    return [(samples, targets)]


@pytest.mark.parametrize(
    ("is_main_process", "epoch", "epochs"),
    [
        pytest.param(True, 0, 5, id="main-process"),
        pytest.param(False, 1, 3, id="non-main-process"),
    ],
)
def test_train_one_epoch_progress_bar_creation_and_metrics(
    monkeypatch, is_main_process: bool, epoch: int, epochs: int
) -> None:
    created = []

    def fake_tqdm(iterable, **kwargs):
        bar = MagicMock()
        bar.kwargs = kwargs
        bar.postfixes = []
        bar.__iter__.return_value = iter(iterable)
        bar.set_postfix.side_effect = lambda values: bar.postfixes.append(values)
        created.append(bar)
        return bar

    scaler = MagicMock()
    scaler.scale.side_effect = lambda loss: loss
    scaler.unscale_.return_value = None
    scaler.step.side_effect = lambda optimizer: optimizer.step()
    scaler.update.return_value = None

    monkeypatch.setattr(engine, "tqdm", fake_tqdm)
    monkeypatch.setattr(engine, "GradScaler", lambda *_args, **_kwargs: scaler)
    monkeypatch.setattr(engine, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(engine.utils, "is_main_process", lambda: is_main_process)

    model = _DummyTrainModel()
    criterion = _DummyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    args = SimpleNamespace(
        print_freq=10,
        grad_accum_steps=1,
        amp=False,
        distributed=False,
        multi_scale=False,
        do_random_resize_via_padding=False,
        resolution=4,
        expanded_scales=False,
        patch_size=1,
        num_windows=1,
        progress_bar=True,
        epochs=epochs,
    )

    train_one_epoch(
        model=model,
        criterion=criterion,
        lr_scheduler=scheduler,
        data_loader=_single_batch_data_loader(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=epoch,
        batch_size=1,
        num_training_steps_per_epoch=1,
        vit_encoder_num_layers=1,
        args=args,
        callbacks=defaultdict(list),
    )

    assert len(created) == 1
    assert created[0].kwargs["desc"] == f"Epoch: [{epoch + 1}/{epochs}]"
    assert created[0].kwargs["disable"] is (not is_main_process)
    assert created[0].postfixes
    assert set(created[0].postfixes[-1]).issuperset({"lr", "class_loss", "box_loss", "loss"})
    assert "max_mem" not in created[0].postfixes[-1]


def test_evaluate_progress_bar_creation_and_metrics(monkeypatch) -> None:
    created = []

    def fake_tqdm(iterable, **kwargs):
        bar = MagicMock()
        bar.kwargs = kwargs
        bar.postfixes = []
        bar.__iter__.return_value = iter(iterable)
        bar.set_postfix.side_effect = lambda values: bar.postfixes.append(values)
        created.append(bar)
        return bar

    coco_eval = MagicMock()
    coco_eval.stats = np.zeros(12, dtype=float)
    coco_evaluator = MagicMock()
    coco_evaluator.coco_eval = {"bbox": coco_eval}

    monkeypatch.setattr(engine, "tqdm", fake_tqdm)
    monkeypatch.setattr(engine, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(engine.utils, "is_main_process", lambda: True)
    monkeypatch.setattr(engine, "CocoEvaluator", lambda *_args, **_kwargs: coco_evaluator)
    monkeypatch.setattr(engine, "coco_extended_metrics", lambda _coco: {"class_map": [], "map": 0.0})

    model = _DummyEvalModel()
    criterion = _DummyCriterion()
    args = SimpleNamespace(
        fp16_eval=False,
        segmentation_head=False,
        eval_max_dets=500,
        print_freq=10,
        progress_bar=True,
        amp=False,
    )

    def postprocess(_outputs, _orig_sizes):
        return [{"boxes": torch.zeros((1, 4)), "scores": torch.ones(1), "labels": torch.ones(1, dtype=torch.int64)}]

    evaluate(
        model=model,
        criterion=criterion,
        postprocess=postprocess,
        data_loader=_single_batch_data_loader(),
        base_ds=object(),
        device=torch.device("cpu"),
        args=args,
        header="Test",
    )

    assert len(created) == 1
    assert created[0].kwargs["desc"] == "Test"
    assert created[0].postfixes
    assert set(created[0].postfixes[-1]).issuperset({"class_loss", "box_loss", "loss"})
    assert "max_mem" not in created[0].postfixes[-1]


class TestIsCuda:
    def test_returns_false_for_cpu_device(self) -> None:
        assert _is_cuda(torch.device("cpu")) is False

    def test_returns_false_when_cuda_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        assert _is_cuda(torch.device("cuda")) is False

    def test_returns_false_when_cuda_not_initialized(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        assert _is_cuda(torch.device("cuda")) is False

    def test_returns_true_for_active_cuda_device(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        assert _is_cuda(torch.device("cuda")) is True


def test_train_one_epoch_max_mem_present_with_cuda_device(monkeypatch) -> None:
    """max_mem appears in the progress bar postfix when running on a CUDA device."""
    created = []

    def fake_tqdm(iterable, **kwargs):
        bar = MagicMock()
        bar.kwargs = kwargs
        bar.postfixes = []
        bar.__iter__.return_value = iter(iterable)
        bar.set_postfix.side_effect = lambda values: bar.postfixes.append(values)
        created.append(bar)
        return bar

    scaler = MagicMock()
    scaler.scale.side_effect = lambda loss: loss
    scaler.unscale_.return_value = None
    scaler.step.side_effect = lambda optimizer: optimizer.step()
    scaler.update.return_value = None

    monkeypatch.setattr(engine, "tqdm", fake_tqdm)
    monkeypatch.setattr(engine, "GradScaler", lambda *_args, **_kwargs: scaler)
    monkeypatch.setattr(engine, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(engine.utils, "is_main_process", lambda: True)
    monkeypatch.setattr(engine, "_is_cuda", lambda _device: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 123 * 1024 * 1024)

    model = _DummyTrainModel()
    criterion = _DummyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    args = SimpleNamespace(
        print_freq=10,
        grad_accum_steps=1,
        amp=False,
        distributed=False,
        multi_scale=False,
        do_random_resize_via_padding=False,
        resolution=4,
        expanded_scales=False,
        patch_size=1,
        num_windows=1,
        progress_bar=True,
        epochs=1,
    )

    train_one_epoch(
        model=model,
        criterion=criterion,
        lr_scheduler=scheduler,
        data_loader=_single_batch_data_loader(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        batch_size=1,
        num_training_steps_per_epoch=1,
        vit_encoder_num_layers=1,
        args=args,
        callbacks=defaultdict(list),
    )

    assert created[0].postfixes
    assert created[0].postfixes[-1]["max_mem"] == "123 MB"


def test_train_one_epoch_max_mem_absent_when_cuda_unavailable(monkeypatch) -> None:
    """max_mem is absent from the progress bar when CUDA is not available."""
    created = []

    def fake_tqdm(iterable, **kwargs):
        bar = MagicMock()
        bar.kwargs = kwargs
        bar.postfixes = []
        bar.__iter__.return_value = iter(iterable)
        bar.set_postfix.side_effect = lambda values: bar.postfixes.append(values)
        created.append(bar)
        return bar

    scaler = MagicMock()
    scaler.scale.side_effect = lambda loss: loss
    scaler.unscale_.return_value = None
    scaler.step.side_effect = lambda optimizer: optimizer.step()
    scaler.update.return_value = None

    monkeypatch.setattr(engine, "tqdm", fake_tqdm)
    monkeypatch.setattr(engine, "GradScaler", lambda *_args, **_kwargs: scaler)
    monkeypatch.setattr(engine, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(engine.utils, "is_main_process", lambda: True)
    monkeypatch.setattr(engine, "_is_cuda", lambda _device: False)

    model = _DummyTrainModel()
    criterion = _DummyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    args = SimpleNamespace(
        print_freq=10,
        grad_accum_steps=1,
        amp=False,
        distributed=False,
        multi_scale=False,
        do_random_resize_via_padding=False,
        resolution=4,
        expanded_scales=False,
        patch_size=1,
        num_windows=1,
        progress_bar=True,
        epochs=1,
    )

    train_one_epoch(
        model=model,
        criterion=criterion,
        lr_scheduler=scheduler,
        data_loader=_single_batch_data_loader(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        batch_size=1,
        num_training_steps_per_epoch=1,
        vit_encoder_num_layers=1,
        args=args,
        callbacks=defaultdict(list),
    )

    assert created[0].postfixes
    assert "max_mem" not in created[0].postfixes[-1]


def test_train_one_epoch_max_mem_absent_when_cuda_not_initialized(monkeypatch) -> None:
    """max_mem is absent from the progress bar when CUDA has not been initialized."""
    created = []

    def fake_tqdm(iterable, **kwargs):
        bar = MagicMock()
        bar.kwargs = kwargs
        bar.postfixes = []
        bar.__iter__.return_value = iter(iterable)
        bar.set_postfix.side_effect = lambda values: bar.postfixes.append(values)
        created.append(bar)
        return bar

    scaler = MagicMock()
    scaler.scale.side_effect = lambda loss: loss
    scaler.unscale_.return_value = None
    scaler.step.side_effect = lambda optimizer: optimizer.step()
    scaler.update.return_value = None

    monkeypatch.setattr(engine, "tqdm", fake_tqdm)
    monkeypatch.setattr(engine, "GradScaler", lambda *_args, **_kwargs: scaler)
    monkeypatch.setattr(engine, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(engine.utils, "is_main_process", lambda: True)
    monkeypatch.setattr(engine, "_is_cuda", lambda _device: False)

    model = _DummyTrainModel()
    criterion = _DummyCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    args = SimpleNamespace(
        print_freq=10,
        grad_accum_steps=1,
        amp=False,
        distributed=False,
        multi_scale=False,
        do_random_resize_via_padding=False,
        resolution=4,
        expanded_scales=False,
        patch_size=1,
        num_windows=1,
        progress_bar=True,
        epochs=1,
    )

    train_one_epoch(
        model=model,
        criterion=criterion,
        lr_scheduler=scheduler,
        data_loader=_single_batch_data_loader(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
        batch_size=1,
        num_training_steps_per_epoch=1,
        vit_encoder_num_layers=1,
        args=args,
        callbacks=defaultdict(list),
    )

    assert created[0].postfixes
    assert "max_mem" not in created[0].postfixes[-1]


def test_evaluate_max_mem_present_with_cuda_device(monkeypatch) -> None:
    """max_mem appears in the progress bar postfix when running on a CUDA device."""
    created = []

    def fake_tqdm(iterable, **kwargs):
        bar = MagicMock()
        bar.kwargs = kwargs
        bar.postfixes = []
        bar.__iter__.return_value = iter(iterable)
        bar.set_postfix.side_effect = lambda values: bar.postfixes.append(values)
        created.append(bar)
        return bar

    coco_eval = MagicMock()
    coco_eval.stats = np.zeros(12, dtype=float)
    coco_evaluator = MagicMock()
    coco_evaluator.coco_eval = {"bbox": coco_eval}

    monkeypatch.setattr(engine, "tqdm", fake_tqdm)
    monkeypatch.setattr(engine, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(engine.utils, "is_main_process", lambda: True)
    monkeypatch.setattr(engine, "CocoEvaluator", lambda *_args, **_kwargs: coco_evaluator)
    monkeypatch.setattr(engine, "coco_extended_metrics", lambda _coco: {"class_map": [], "map": 0.0})
    monkeypatch.setattr(engine, "_is_cuda", lambda _device: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 456 * 1024 * 1024)

    model = _DummyEvalModel()
    criterion = _DummyCriterion()
    args = SimpleNamespace(
        fp16_eval=False,
        segmentation_head=False,
        eval_max_dets=500,
        print_freq=10,
        progress_bar=True,
        amp=False,
    )

    def postprocess(_outputs, _orig_sizes):
        return [{"boxes": torch.zeros((1, 4)), "scores": torch.ones(1), "labels": torch.ones(1, dtype=torch.int64)}]

    evaluate(
        model=model,
        criterion=criterion,
        postprocess=postprocess,
        data_loader=_single_batch_data_loader(),
        base_ds=object(),
        device=torch.device("cpu"),
        args=args,
        header="Test",
    )

    assert created[0].postfixes
    assert created[0].postfixes[-1]["max_mem"] == "456 MB"
