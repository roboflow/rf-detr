# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared test helpers for the rfdetr.training test suite.

Plain classes and functions (not pytest fixtures) shared across multiple test modules to avoid verbatim duplication.
Import with an absolute import::

    from tests.training.helpers import _FakeCriterion, _FakeDataset, _TinyModel

``_build_module`` and its config/fake builders serve ``tests/training/model/``; ``_build_datamodule``,
``_datamodule_train_config`` and ``_fake_dataset`` serve ``tests/training/datamodule/``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.utils.data

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.tensors import NestedTensor


class _TinyModel(nn.Module):
    """Minimal real nn.Module satisfying the RFDETRModule model contract.

    Has a single trainable parameter so the optimizer has something to update and the loss has a gradient path back
    through the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, samples, targets=None):
        return {"dummy": self.dummy}

    def update_drop_path(self, *args, **kwargs) -> None:
        pass

    def update_dropout(self, *args, **kwargs) -> None:
        pass

    def reinitialize_detection_head(self, *args, **kwargs) -> None:
        pass


class _FakeCriterion:
    """Callable criterion that returns a loss connected to the model output.

    Keeps a gradient path from the loss back to _TinyModel.dummy so that ``loss.backward()`` does not error when the
    Trainer calls it.
    """

    weight_dict = {"loss_ce": 1.0}

    def num_boxes_for_targets(self, outputs, targets):
        dummy = outputs.get("dummy", torch.zeros(1))
        return torch.ones((), dtype=dummy.dtype, device=dummy.device)

    def __call__(self, outputs, targets, num_boxes=None):
        dummy = outputs.get("dummy", torch.zeros(1))
        denominator = self.num_boxes_for_targets(outputs, targets) if num_boxes is None else num_boxes
        return {"loss_ce": dummy.mean() / denominator}


class _FakeDataset(torch.utils.data.Dataset):
    """Dataset with ``(image, target)`` pairs for detection.

    The image is a ``(3, 32, 32)`` float tensor; the target dict includes the fields expected by RFDETRModule:
    ``boxes``, ``labels``, ``image_id``, ``orig_size``, ``size``.
    """

    def __init__(self, length: int = 20) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([32, 32]),
            "size": torch.tensor([32, 32]),
        }
        return image, target


class _FakeDatasetWithMasks(_FakeDataset):
    """Like _FakeDataset but includes binary instance masks (for segmentation)."""

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        target["masks"] = torch.zeros(1, 32, 32, dtype=torch.bool)
        return image, target


class _FakePostProcess:
    """Picklable postprocessor for ddp_spawn tests.

    ``MagicMock`` is not picklable and cannot survive the subprocess boundary that ``ddp_spawn`` creates.  This plain
    class is a drop-in replacement.

    Delegates to ``_fake_postprocess``; keep both in sync if the fake output format changes.
    """

    def __call__(self, outputs, orig_sizes):
        return _fake_postprocess(outputs, orig_sizes)


def _fake_postprocess(outputs, orig_sizes):
    """Return one non-empty prediction per image so COCOEvalCallback has something to score.

    Examples:
        >>> import torch
        >>> batch = _fake_postprocess({}, torch.zeros(2, 2))
        >>> len(batch)
        2
        >>> sorted(batch[0])
        ['boxes', 'labels', 'scores']
    """
    n = orig_sizes.shape[0]
    return [
        {
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
        for _ in range(n)
    ]


def _make_param_dicts(model: nn.Module) -> list[dict]:
    """Build a minimal param-dict list for AdamW from all trainable parameters.

    Examples:
        >>> model = _TinyModel()
        >>> groups = _make_param_dicts(model)
        >>> len(groups)
        1
        >>> groups[0]["lr"]
        0.0001
    """
    return [{"params": p, "lr": 1e-4} for p in model.parameters() if p.requires_grad]


def _base_model_config(**overrides):
    """Return a minimal RFDETRBaseConfig with pretrain_weights disabled.

    Examples:
        >>> config = _base_model_config(num_classes=7)
        >>> config.device, config.num_classes, config.pretrain_weights
        ('cpu', 7, None)
    """
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _base_train_config(tmp_path=None, **overrides):
    """Return a minimal TrainConfig suitable for unit tests.

    Examples:
        >>> config = _base_train_config(batch_size=4)
        >>> config.batch_size, config.dataset_dir.endswith("dataset"), config.output_dir.endswith("output")
        (4, True, True)
    """
    dataset_dir = str(tmp_path / "dataset") if tmp_path else "/nonexistent/dataset"
    output_dir = str(tmp_path / "output") if tmp_path else "/nonexistent/output"
    defaults = dict(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        grad_accum_steps=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _fake_model():
    """Return a MagicMock that behaves enough like an LWDETR model.

    Examples:
        >>> model = _fake_model()
        >>> isinstance(next(model.parameters()), nn.Parameter)
        True
    """
    model = MagicMock(spec=nn.Module)
    real_param = nn.Parameter(torch.randn(4, 4))
    model.parameters.return_value = iter([real_param])
    model.named_parameters.return_value = iter([("weight", real_param)])
    model.update_drop_path = MagicMock()
    model.update_dropout = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    return model


def _fake_criterion():
    """Return a MagicMock criterion with a realistic weight_dict.

    Examples:
        >>> criterion = _fake_criterion()
        >>> sorted(criterion.weight_dict)
        ['loss_bbox', 'loss_ce', 'loss_giou']
    """
    criterion = MagicMock()
    criterion.weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion.num_boxes_for_targets.return_value = torch.tensor(1.0)
    return criterion


def _mock_postprocess():
    """Return a callable MagicMock for postprocess.

    Unlike :func:`_fake_postprocess`, which computes predictions, this returns a mock whose calls can be asserted on.

    Examples:
        >>> import torch
        >>> postprocess = _mock_postprocess()
        >>> sorted(postprocess({}, torch.zeros(1, 2))[0])
        ['boxes', 'labels', 'scores']
    """
    return MagicMock(return_value=[{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1)}])


def _build_module(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRModelModule with build_model_from_config and build_criterion_from_config mocked.

    Examples:
        >>> module, model, criterion, postprocess = _build_module()
        >>> module.model is model and module.criterion is criterion and module.postprocess is postprocess
        True
    """
    mc = model_config or _base_model_config()
    tc = train_config or _base_train_config(tmp_path)
    fake_model = _fake_model()
    fake_criterion = _fake_criterion()
    fake_postprocess = _mock_postprocess()
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(fake_criterion, fake_postprocess),
        ),
    ):
        from rfdetr.training.module_model import RFDETRModelModule

        module = RFDETRModelModule(mc, tc)
    return module, fake_model, fake_criterion, fake_postprocess


def _make_batch(batch_size: int = 2, channels: int = 3, h: int = 16, w: int = 16):
    """Build a (NestedTensor, targets) tuple for testing.

    Examples:
        >>> samples, targets = _make_batch(batch_size=2, h=8, w=8)
        >>> samples.tensors.shape, len(targets)
        (torch.Size([2, 3, 8, 8]), 2)
    """
    tensors = torch.randn(batch_size, channels, h, w)
    mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
    samples = NestedTensor(tensors, mask)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(i),
            "orig_size": torch.tensor([h, w]),
        }
        for i in range(batch_size)
    ]
    return samples, targets


def _attach_trainer(module, **attributes):
    """Attach a MagicMock trainer carrying ``attributes`` to ``module`` and return it.

    Examples:
        >>> module, *_ = _build_module()
        >>> trainer = _attach_trainer(module, global_step=3)
        >>> module.trainer is trainer, trainer.global_step
        (True, 3)
    """
    trainer = MagicMock()
    for name, value in attributes.items():
        setattr(trainer, name, value)
    module._trainer = trainer
    type(module).trainer = property(lambda self: self._trainer)
    return trainer


def _datamodule_train_config(tmp_path=None, **overrides):
    """Return a minimal TrainConfig for RFDETRDataModule unit tests.

    Same as :func:`_base_train_config` plus ``lr_scheduler_kwargs={"lr_drop": 8}``.

    Examples:
        >>> config = _datamodule_train_config(batch_size=4)
        >>> config.batch_size, config.dataset_dir.endswith("dataset"), config.output_dir.endswith("output")
        (4, True, True)
    """
    dataset_dir = str(tmp_path / "dataset") if tmp_path else "/nonexistent/dataset"
    output_dir = str(tmp_path / "output") if tmp_path else "/nonexistent/output"
    defaults = dict(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        lr_scheduler_kwargs={"lr_drop": 8},
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        grad_accum_steps=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


class _LengthOnlyDataset(torch.utils.data.Dataset):
    """Minimal dataset stub with a controllable length whose items cannot be read.

    Args:
        length: Number of items to report via ``__len__``.
        with_coco: If True, attach a mock ``.coco`` attribute with ``cats``
            so ``class_names`` can be tested.
    """

    def __init__(self, length: int = 100, with_coco: bool = False) -> None:
        self._length = length
        if with_coco:
            coco = MagicMock()
            coco.cats = {1: {"name": "cat"}, 2: {"name": "dog"}}
            self.coco = coco
        else:
            self.coco = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        raise NotImplementedError


def _fake_dataset(length: int = 100, with_coco: bool = False) -> _LengthOnlyDataset:
    """Return a minimal ``_LengthOnlyDataset`` with a controllable length.

    Examples:
        >>> dataset = _fake_dataset(length=3, with_coco=True)
        >>> len(dataset), dataset.coco.cats[1]["name"]
        (3, 'cat')
    """
    return _LengthOnlyDataset(length, with_coco)


def _build_datamodule(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRDataModule (build_dataset is not called at init time).

    Examples:
        >>> datamodule = _build_datamodule()
        >>> datamodule.model_config.device, datamodule.train_config.batch_size
        ('cpu', 2)
    """
    mc = model_config or _base_model_config()
    tc = train_config or _datamodule_train_config(tmp_path)
    return RFDETRDataModule(mc, tc)
