# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Comprehensive unit tests for RFDETRDataModule (LightningDataModule wrapper)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from rfdetr.config import RFDETRBaseConfig, TrainConfig

# ---------------------------------------------------------------------------
# Private helpers — used by both module-level fixtures and class-level _setup_*
# methods (which cannot inject pytest fixtures directly).
# Only define a private helper when it is called from more than one site;
# single-use logic belongs directly in the fixture body.
# ---------------------------------------------------------------------------


def _base_model_config(**overrides):
    """Return a minimal RFDETRBaseConfig with pretrain_weights disabled."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _base_train_config(tmp_path=None, **overrides):
    """Return a minimal TrainConfig suitable for unit tests."""
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
        lr_drop=8,
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


class _FakeDataset(torch.utils.data.Dataset):
    """Minimal dataset stub with a controllable length.

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


def _fake_dataset(length: int = 100, with_coco: bool = False) -> _FakeDataset:
    """Return a minimal ``_FakeDataset`` with a controllable length."""
    return _FakeDataset(length, with_coco)


def _build_datamodule(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRDataModule (build_dataset is not called at init time)."""
    mc = model_config or _base_model_config()
    tc = train_config or _base_train_config(tmp_path)
    from rfdetr.training.datamodule import RFDETRDataModule

    return RFDETRDataModule(mc, tc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def build_datamodule(tmp_path):
    """Factory fixture — returns a constructed RFDETRDataModule.

    build_dataset is mocked automatically.
    tmp_path is injected automatically so test methods do not need to declare it.
    """
    return lambda model_config=None, train_config=None: _build_datamodule(model_config, train_config, tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInit:
    """RFDETRDataModule.__init__ stores configs and initialises dataset slots."""

    def test_stores_model_config(self, build_datamodule, base_model_config):
        """model_config is accessible as an attribute after construction."""
        mc = base_model_config(num_classes=3)
        dm = build_datamodule(model_config=mc)
        assert dm.model_config is mc

    def test_stores_train_config(self, build_datamodule, base_train_config):
        """train_config is accessible as an attribute after construction."""
        tc = base_train_config(epochs=42)
        dm = build_datamodule(train_config=tc)
        assert dm.train_config is tc

    def test_builds_args_namespace(self, build_datamodule):
        """_args Namespace is created during __init__."""
        dm = build_datamodule()
        assert dm._args is not None

    def test_datasets_start_as_none(self, build_datamodule):
        """All three dataset slots are None before setup() is called."""
        dm = build_datamodule()
        assert dm._dataset_train is None
        assert dm._dataset_val is None
        assert dm._dataset_test is None

    def test_prefetch_factor_defaults_to_two_when_workers_enabled(self, build_datamodule, base_train_config):
        """prefetch_factor defaults to 2 for worker-based DataLoaders."""
        tc = base_train_config(num_workers=2, prefetch_factor=None)
        dm = build_datamodule(train_config=tc)
        assert dm._prefetch_factor == 2

    def test_prefetch_factor_honors_train_config(self, build_datamodule, base_train_config):
        """prefetch_factor from TrainConfig is forwarded when workers are enabled."""
        tc = base_train_config(num_workers=2, prefetch_factor=5)
        dm = build_datamodule(train_config=tc)
        assert dm._prefetch_factor == 5

    def test_prefetch_factor_none_when_workers_disabled(self, build_datamodule, base_train_config):
        """prefetch_factor is None when num_workers == 0."""
        tc = base_train_config(num_workers=0, prefetch_factor=5)
        dm = build_datamodule(train_config=tc)
        assert dm._prefetch_factor is None

    def test_pin_memory_override_is_respected(self, build_datamodule, base_train_config):
        """pin_memory can be explicitly overridden from TrainConfig."""
        tc = base_train_config(pin_memory=False)
        dm = build_datamodule(train_config=tc)
        assert dm._pin_memory is False

    def test_persistent_workers_override_is_respected(self, build_datamodule, base_train_config):
        """persistent_workers can be explicitly overridden from TrainConfig."""
        tc = base_train_config(num_workers=2, persistent_workers=False)
        dm = build_datamodule(train_config=tc)
        assert dm._persistent_workers is False


class TestSetup:
    """setup(stage) builds the correct dataset(s) for each PTL stage."""

    def _setup_with_mock(self, tmp_path, stage, dataset_file="roboflow", **train_overrides):
        """Helper: construct DataModule and call setup(stage) with build_dataset mocked."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, dataset_file=dataset_file, **train_overrides)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        fake_train = _fake_dataset(100)
        fake_val = _fake_dataset(20)
        fake_test = _fake_dataset(10)
        datasets = {"train": fake_train, "val": fake_val, "test": fake_test}

        def _build(image_set, args, resolution):
            return datasets[image_set]

        with patch("rfdetr.training.datamodule.build_dataset", side_effect=_build):
            dm.setup(stage)
        return dm, fake_train, fake_val, fake_test

    def test_fit_builds_train_and_val(self, tmp_path):
        """setup('fit') populates both _dataset_train and _dataset_val."""
        dm, fake_train, fake_val, _ = self._setup_with_mock(tmp_path, "fit")
        assert dm._dataset_train is fake_train
        assert dm._dataset_val is fake_val
        assert dm._dataset_test is None

    def test_validate_builds_only_val(self, tmp_path):
        """setup('validate') populates only _dataset_val."""
        dm, _, fake_val, _ = self._setup_with_mock(tmp_path, "validate")
        assert dm._dataset_train is None
        assert dm._dataset_val is fake_val
        assert dm._dataset_test is None

    def test_test_stage_roboflow_uses_test_split(self, tmp_path):
        """setup('test') requests 'test' split when dataset_file=='roboflow'."""
        dm, _, _, fake_test = self._setup_with_mock(tmp_path, "test", dataset_file="roboflow")
        assert dm._dataset_test is fake_test

    def test_test_stage_non_roboflow_uses_val_split(self, tmp_path):
        """setup('test') falls back to 'val' split for non-roboflow datasets."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, dataset_file="coco")
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        requested_splits = []

        def _build(image_set, args, resolution):
            requested_splits.append(image_set)
            return _fake_dataset(10)

        with patch("rfdetr.training.datamodule.build_dataset", side_effect=_build):
            dm.setup("test")

        assert "val" in requested_splits
        assert "test" not in requested_splits

    def test_fit_does_not_rebuild_if_already_set(self, tmp_path):
        """setup('fit') skips building if datasets are already populated."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        existing_train = _fake_dataset(50)
        existing_val = _fake_dataset(10)
        dm._dataset_train = existing_train
        dm._dataset_val = existing_val

        with patch("rfdetr.training.datamodule.build_dataset") as mock_build:
            dm.setup("fit")
            mock_build.assert_not_called()

        assert dm._dataset_train is existing_train
        assert dm._dataset_val is existing_val

    def test_predict_stage_builds_nothing(self, tmp_path):
        """setup('predict') does not build any dataset."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        with patch("rfdetr.training.datamodule.build_dataset") as mock_build:
            dm.setup("predict")
            mock_build.assert_not_called()


class TestTrainDataloader:
    """train_dataloader() returns the correct DataLoader for large and small datasets."""

    def _setup_dm_with_train(self, tmp_path, dataset_length, batch_size=2, grad_accum_steps=1, num_workers=0):
        """Construct DataModule and inject a fake _dataset_train of given length."""
        mc = _base_model_config()
        tc = _base_train_config(
            tmp_path,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            num_workers=num_workers,
        )
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(dataset_length)
        return dm

    def test_returns_dataloader(self, tmp_path):
        """train_dataloader() returns a DataLoader instance."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)

    def test_large_dataset_uses_batch_sampler(self, tmp_path):
        """A large dataset uses a BatchSampler (drop_last=True, no replacement)."""
        # 200 samples > 2*1*5=10 threshold → large path
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200, batch_size=2, grad_accum_steps=1)
        loader = dm.train_dataloader()
        assert loader.batch_sampler is not None
        assert isinstance(loader.batch_sampler, torch.utils.data.BatchSampler)
        assert loader.batch_sampler.drop_last is True

    def test_small_dataset_uses_replacement_sampler(self, tmp_path):
        """A small dataset (< effective_batch * min_batches) uses a replacement sampler."""
        # 3 samples < 2*1*5=10 threshold → small path
        dm = self._setup_dm_with_train(tmp_path, dataset_length=3, batch_size=2, grad_accum_steps=1)
        loader = dm.train_dataloader()
        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
        assert loader.sampler.replacement is True

    def test_small_dataset_replacement_sampler_num_samples(self, tmp_path):
        """Replacement sampler has num_samples == effective_batch_size * _MIN_TRAIN_BATCHES."""
        from rfdetr.training.datamodule import _MIN_TRAIN_BATCHES

        batch_size = 2
        grad_accum_steps = 3
        dm = self._setup_dm_with_train(
            tmp_path,
            dataset_length=3,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
        loader = dm.train_dataloader()
        expected = batch_size * grad_accum_steps * _MIN_TRAIN_BATCHES
        assert loader.sampler.num_samples == expected

    def test_batch_size_forwarded(self, tmp_path):
        """The DataLoader's batch size matches the train config."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200, batch_size=8)
        loader = dm.train_dataloader()
        assert loader.batch_sampler.batch_size == 8

    def test_num_workers_forwarded(self, tmp_path):
        """The DataLoader's num_workers matches the train config."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200, num_workers=0)
        loader = dm.train_dataloader()
        assert loader.num_workers == 0

    def test_threshold_exact_boundary_uses_batch_sampler(self, tmp_path):
        """Dataset of exactly effective_batch_size * _MIN_TRAIN_BATCHES is NOT small."""
        from rfdetr.training.datamodule import _MIN_TRAIN_BATCHES

        batch_size = 2
        grad_accum = 1
        length = batch_size * grad_accum * _MIN_TRAIN_BATCHES  # exactly at threshold
        dm = self._setup_dm_with_train(tmp_path, dataset_length=length, batch_size=batch_size)
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, torch.utils.data.BatchSampler)


class TestValDataloader:
    """val_dataloader() returns a SequentialSampler with drop_last=False."""

    def _setup_dm_with_val(self, tmp_path, dataset_length=50, batch_size=2, num_workers=0):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, batch_size=batch_size, num_workers=num_workers)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_val = _fake_dataset(dataset_length)
        return dm

    def test_returns_dataloader(self, tmp_path):
        """val_dataloader() returns a DataLoader instance."""
        dm = self._setup_dm_with_val(tmp_path)
        loader = dm.val_dataloader()
        assert isinstance(loader, DataLoader)

    def test_uses_sequential_sampler(self, tmp_path):
        """val_dataloader uses a SequentialSampler."""
        dm = self._setup_dm_with_val(tmp_path)
        loader = dm.val_dataloader()
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_drop_last_false(self, tmp_path):
        """val_dataloader does not drop the last incomplete batch."""
        dm = self._setup_dm_with_val(tmp_path)
        loader = dm.val_dataloader()
        assert loader.drop_last is False

    def test_batch_size_forwarded(self, tmp_path):
        """The DataLoader's batch size matches the train config."""
        dm = self._setup_dm_with_val(tmp_path, batch_size=6)
        loader = dm.val_dataloader()
        assert loader.batch_size == 6

    def test_num_workers_forwarded(self, tmp_path):
        """The DataLoader's num_workers matches the train config."""
        dm = self._setup_dm_with_val(tmp_path, num_workers=0)
        loader = dm.val_dataloader()
        assert loader.num_workers == 0


class TestTestDataloader:
    """test_dataloader() returns a SequentialSampler with drop_last=False."""

    def _setup_dm_with_test(self, tmp_path, dataset_length=30, batch_size=2, num_workers=0):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, batch_size=batch_size, num_workers=num_workers)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_test = _fake_dataset(dataset_length)
        return dm

    def test_returns_dataloader(self, tmp_path):
        """test_dataloader() returns a DataLoader instance."""
        dm = self._setup_dm_with_test(tmp_path)
        loader = dm.test_dataloader()
        assert isinstance(loader, DataLoader)

    def test_uses_sequential_sampler(self, tmp_path):
        """test_dataloader uses a SequentialSampler."""
        dm = self._setup_dm_with_test(tmp_path)
        loader = dm.test_dataloader()
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_drop_last_false(self, tmp_path):
        """test_dataloader does not drop the last incomplete batch."""
        dm = self._setup_dm_with_test(tmp_path)
        loader = dm.test_dataloader()
        assert loader.drop_last is False

    def test_batch_size_forwarded(self, tmp_path):
        """The DataLoader's batch size matches the train config."""
        dm = self._setup_dm_with_test(tmp_path, batch_size=4)
        loader = dm.test_dataloader()
        assert loader.batch_size == 4


class TestClassNames:
    """class_names property extracts names from COCO dataset annotations."""

    def test_returns_none_before_setup(self, build_datamodule):
        """class_names is None when no dataset has been set up."""
        dm = build_datamodule()
        assert dm.class_names is None

    def test_returns_names_from_train_dataset(self, tmp_path):
        """class_names reads from _dataset_train.coco.cats when available."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(50, with_coco=True)
        assert dm.class_names == ["cat", "dog"]

    def test_returns_names_from_val_dataset_when_train_missing(self, tmp_path):
        """class_names falls back to _dataset_val when _dataset_train has no COCO."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(50, with_coco=False)
        dm._dataset_val = _fake_dataset(20, with_coco=True)
        assert dm.class_names == ["cat", "dog"]

    def test_returns_none_when_no_coco_attribute(self, tmp_path):
        """class_names returns None when no dataset has a coco attribute."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(50, with_coco=False)
        dm._dataset_val = _fake_dataset(20, with_coco=False)
        assert dm.class_names is None

    def test_class_names_sorted_by_category_id(self, tmp_path):
        """class_names are sorted by COCO category ID."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dataset = _fake_dataset(50)
        coco = MagicMock()
        # Deliberately out of order IDs
        coco.cats = {3: {"name": "zebra"}, 1: {"name": "ant"}, 2: {"name": "bee"}}
        dataset.coco = coco
        dm._dataset_train = dataset
        assert dm.class_names == ["ant", "bee", "zebra"]


class TestSegmentationSupport:
    """DataModule accepts SegmentationTrainConfig without errors."""

    def test_init_with_seg_train_config(self, base_model_config, seg_train_config):
        """RFDETRDataModule can be constructed with a SegmentationTrainConfig."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        assert dm.train_config is tc
        assert dm._args.segmentation_head is True

    def test_seg_args_have_mask_loss_coefs(self, base_model_config, seg_train_config):
        """Segmentation-specific loss coefficients are forwarded to _args."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        from rfdetr.training.datamodule import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        assert dm._args.mask_ce_loss_coef == pytest.approx(5.0)
        assert dm._args.mask_dice_loss_coef == pytest.approx(5.0)
