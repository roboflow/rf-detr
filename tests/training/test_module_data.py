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
from rfdetr.utilities.tensors import NestedTensor

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


def _make_batch(batch_size: int = 2, channels: int = 3, h: int = 16, w: int = 16):
    """Build a ``(NestedTensor, targets)`` tuple for transfer_batch_to_device tests."""
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


def _build_datamodule(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRDataModule (build_dataset is not called at init time)."""
    mc = model_config or _base_model_config()
    tc = train_config or _base_train_config(tmp_path)
    from rfdetr.training.module_data import RFDETRDataModule

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

    @patch("rfdetr.config.DEVICE", "cuda")
    def test_pin_memory_defaults_to_false_when_accelerator_is_cpu(self, build_datamodule, base_train_config):
        """Default pin_memory stays off when training is explicitly CPU-only."""
        tc = base_train_config(pin_memory=None, accelerator="cpu")
        dm = build_datamodule(train_config=tc)
        assert dm._pin_memory is False

    def test_persistent_workers_override_is_respected(self, build_datamodule, base_train_config):
        """persistent_workers can be explicitly overridden from TrainConfig."""
        tc = base_train_config(num_workers=2, persistent_workers=False)
        dm = build_datamodule(train_config=tc)
        assert dm._persistent_workers is False

    def test_ddp_notebook_preserves_num_workers(self, build_datamodule, base_train_config):
        """ddp_notebook keeps num_workers as configured (spawn-based DDP
        children initialise CUDA fresh; DataLoader fork workers are CPU-only
        and never touch CUDA, so nested forks are safe)."""
        tc = base_train_config(num_workers=4, strategy="ddp_notebook")
        dm = build_datamodule(train_config=tc)
        assert dm._num_workers == 4
        assert dm._prefetch_factor == 2

    def test_other_strategy_preserves_num_workers(self, build_datamodule, base_train_config):
        """Non-ddp_notebook strategies also keep num_workers as configured."""
        tc = base_train_config(num_workers=4, strategy="ddp")
        dm = build_datamodule(train_config=tc)
        assert dm._num_workers == 4
        assert dm._prefetch_factor == 2  # default prefetch_factor for num_workers>0


class TestSetup:
    """setup(stage) builds the correct dataset(s) for each PTL stage."""

    def _setup_with_mock(self, tmp_path, stage, dataset_file="roboflow", **train_overrides):
        """Helper: construct DataModule and call setup(stage) with build_dataset mocked."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, dataset_file=dataset_file, **train_overrides)
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        fake_train = _fake_dataset(100)
        fake_val = _fake_dataset(20)
        fake_test = _fake_dataset(10)
        datasets = {"train": fake_train, "val": fake_val, "test": fake_test}

        def _build(image_set, args, resolution):
            return datasets[image_set]

        with patch("rfdetr.training.module_data.build_dataset", side_effect=_build):
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
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        requested_splits = []

        def _build(image_set, args, resolution):
            requested_splits.append(image_set)
            return _fake_dataset(10)

        with patch("rfdetr.training.module_data.build_dataset", side_effect=_build):
            dm.setup("test")

        assert "val" in requested_splits
        assert "test" not in requested_splits

    def test_fit_does_not_rebuild_if_already_set(self, tmp_path):
        """setup('fit') skips building if datasets are already populated."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        existing_train = _fake_dataset(50)
        existing_val = _fake_dataset(10)
        dm._dataset_train = existing_train
        dm._dataset_val = existing_val

        with patch("rfdetr.training.module_data.build_dataset") as mock_build:
            dm.setup("fit")
            mock_build.assert_not_called()

        assert dm._dataset_train is existing_train
        assert dm._dataset_val is existing_val

    def test_predict_stage_builds_val_dataset(self, tmp_path):
        """setup('predict') populates _dataset_val with the 'val' split."""
        dm, _, fake_val, _ = self._setup_with_mock(tmp_path, "predict")
        assert dm._dataset_val is fake_val
        assert dm._dataset_train is None
        assert dm._dataset_test is None

    def test_predict_stage_does_not_rebuild_existing_val(self, tmp_path):
        """setup('predict') skips building when _dataset_val is already set."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        existing_val = _fake_dataset(20)
        dm._dataset_val = existing_val

        with patch("rfdetr.training.module_data.build_dataset") as mock_build:
            dm.setup("predict")
            mock_build.assert_not_called()

        assert dm._dataset_val is existing_val


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
        from rfdetr.training.module_data import RFDETRDataModule

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
        from rfdetr.training.module_data import _MIN_TRAIN_BATCHES

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
        from rfdetr.training.module_data import _MIN_TRAIN_BATCHES

        batch_size = 2
        grad_accum = 1
        length = batch_size * grad_accum * _MIN_TRAIN_BATCHES  # exactly at threshold
        dm = self._setup_dm_with_train(tmp_path, dataset_length=length, batch_size=batch_size)
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, torch.utils.data.BatchSampler)

    @pytest.mark.parametrize(
        "dataset_length, batch_size, grad_accum_steps",
        [
            pytest.param(100, 2, 1, id="already_aligned_ga1"),
            pytest.param(96, 2, 4, id="already_aligned_ga4"),
            pytest.param(101, 2, 4, id="unaligned_one_extra"),
            pytest.param(50, 2, 8, id="unaligned_ga8"),
            pytest.param(59143, 2, 8, id="large_unaligned_coco_like"),
            pytest.param(100, 3, 3, id="non_power_of_two_ga"),
        ],
    )
    def test_train_dataloader_length_is_multiple_of_grad_accum(
        self, tmp_path, dataset_length, batch_size, grad_accum_steps
    ):
        """len(train_dataloader()) is always a multiple of grad_accum_steps.

        Verifies the workaround for
        https://github.com/Lightning-AI/pytorch-lightning/issues/19987:
        the training DataLoader must never present a partial accumulation
        window to PTL.
        """
        dm = self._setup_dm_with_train(
            tmp_path,
            dataset_length=dataset_length,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
        loader = dm.train_dataloader()
        assert len(loader) % grad_accum_steps == 0, (
            f"len(loader)={len(loader)} is not a multiple of grad_accum_steps={grad_accum_steps}"
        )

    def test_train_dataloader_respects_trainer_world_size(self, tmp_path):
        """Large-dataset path aligns wrapped dataset length to effective_batch_size * world_size."""
        dm = self._setup_dm_with_train(
            tmp_path,
            dataset_length=101,
            batch_size=2,
            grad_accum_steps=4,
        )
        dm.trainer = MagicMock(world_size=3)

        loader = dm.train_dataloader()

        assert len(loader.dataset) % (2 * 4 * 3) == 0
        assert len(loader.dataset) == 120


class TestGradAccumAlignedDataset:
    """Unit tests for the GradAccumAlignedDataset wrapper."""

    def _make_dataset(self, length: int) -> torch.utils.data.TensorDataset:
        """Return a simple TensorDataset of given length."""
        return torch.utils.data.TensorDataset(torch.arange(length))

    def test_aligned_length_is_multiple_of_pad_unit(self):
        """Padded length is always a multiple of effective_batch_size * world_size."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        ds = self._make_dataset(50)
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=16, world_size=1)
        assert len(wrapped) % 16 == 0

    def test_no_padding_needed_when_already_aligned(self):
        """If len(dataset) % pad_unit == 0, length is unchanged."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        ds = self._make_dataset(64)
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=16, world_size=1)
        assert len(wrapped) == 64

    def test_padding_adds_correct_count(self):
        """Exactly (pad_unit - remainder) % pad_unit samples are added."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        ds = self._make_dataset(50)  # 50 % 16 = 2 → pad 14
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=16, world_size=1)
        assert len(wrapped) == 64

    def test_getitem_forwards_to_original_dataset(self):
        """Items in the original range map directly to the underlying dataset."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        ds = self._make_dataset(10)
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=4, world_size=1)
        for i in range(10):
            (val,) = wrapped[i]
            assert val.item() == i

    def test_padded_indices_are_valid(self):
        """All padded indices point to valid positions in the original dataset."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        n = 10
        ds = self._make_dataset(n)
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=4, world_size=1)
        for i in range(len(wrapped)):
            (val,) = wrapped[i]
            assert 0 <= val.item() < n

    @pytest.mark.parametrize(
        "n, eff_bs, world_size",
        [
            pytest.param(100, 4, 1, id="aligned_single_gpu"),
            pytest.param(101, 4, 1, id="unaligned_single_gpu"),
            pytest.param(100, 4, 2, id="aligned_ddp2"),
            pytest.param(97, 4, 2, id="unaligned_ddp2"),
        ],
    )
    def test_length_always_multiple_of_pad_unit(self, n, eff_bs, world_size):
        """len(wrapped) % (eff_bs * world_size) == 0 for all inputs."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        ds = self._make_dataset(n)
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=eff_bs, world_size=world_size)
        assert len(wrapped) % (eff_bs * world_size) == 0

    @pytest.mark.parametrize(
        "effective_batch_size, world_size",
        [
            pytest.param(0, 1, id="zero_effective_batch_size"),
            pytest.param(-1, 1, id="negative_effective_batch_size"),
            pytest.param(2, 0, id="zero_world_size"),
            pytest.param(2, -1, id="negative_world_size"),
        ],
    )
    def test_raises_for_non_positive_alignment_inputs(self, effective_batch_size, world_size):
        """Non-positive alignment inputs fail with a clear ValueError."""
        from rfdetr.training.module_data import GradAccumAlignedDataset

        ds = self._make_dataset(10)
        with pytest.raises(ValueError, match="must be >= 1"):
            GradAccumAlignedDataset(
                ds,
                effective_batch_size=effective_batch_size,
                world_size=world_size,
            )


class TestValDataloader:
    """val_dataloader() returns a SequentialSampler with drop_last=False."""

    def _setup_dm_with_val(self, tmp_path, dataset_length=50, batch_size=2, num_workers=0):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, batch_size=batch_size, num_workers=num_workers)
        from rfdetr.training.module_data import RFDETRDataModule

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
        from rfdetr.training.module_data import RFDETRDataModule

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


class TestPredictDataloader:
    """predict_dataloader() reuses the validation dataset with sequential sampling."""

    def _setup_dm_with_val(self, tmp_path, dataset_length=50, batch_size=2, num_workers=0):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, batch_size=batch_size, num_workers=num_workers)
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_val = _fake_dataset(dataset_length)
        return dm

    def test_returns_dataloader(self, tmp_path):
        """predict_dataloader() returns a DataLoader instance."""
        dm = self._setup_dm_with_val(tmp_path)
        loader = dm.predict_dataloader()
        assert isinstance(loader, DataLoader)

    def test_uses_sequential_sampler(self, tmp_path):
        """predict_dataloader uses a SequentialSampler (deterministic ordering)."""
        dm = self._setup_dm_with_val(tmp_path)
        loader = dm.predict_dataloader()
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_drop_last_false(self, tmp_path):
        """predict_dataloader does not drop the last incomplete batch."""
        dm = self._setup_dm_with_val(tmp_path)
        loader = dm.predict_dataloader()
        assert loader.drop_last is False

    def test_batch_size_forwarded(self, tmp_path):
        """The DataLoader's batch size matches the train config."""
        dm = self._setup_dm_with_val(tmp_path, batch_size=6)
        loader = dm.predict_dataloader()
        assert loader.batch_size == 6

    def test_num_workers_forwarded(self, tmp_path):
        """The DataLoader's num_workers matches the train config."""
        dm = self._setup_dm_with_val(tmp_path, num_workers=0)
        loader = dm.predict_dataloader()
        assert loader.num_workers == 0


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
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(50, with_coco=True)
        assert dm.class_names == ["cat", "dog"]

    def test_returns_names_from_val_dataset_when_train_missing(self, tmp_path):
        """class_names falls back to _dataset_val when _dataset_train has no COCO."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(50, with_coco=False)
        dm._dataset_val = _fake_dataset(20, with_coco=True)
        assert dm.class_names == ["cat", "dog"]

    def test_returns_none_when_no_coco_attribute(self, tmp_path):
        """class_names returns None when no dataset has a coco attribute."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(50, with_coco=False)
        dm._dataset_val = _fake_dataset(20, with_coco=False)
        assert dm.class_names is None

    def test_class_names_sorted_by_category_id(self, tmp_path):
        """class_names are sorted by COCO category ID."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)
        from rfdetr.training.module_data import RFDETRDataModule

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
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        assert dm.train_config is tc
        assert dm.model_config.segmentation_head is True

    def test_seg_args_have_mask_loss_coefs(self, base_model_config, seg_train_config):
        """Segmentation-specific loss coefficients are present on train_config."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        from rfdetr.training.module_data import RFDETRDataModule

        dm = RFDETRDataModule(mc, tc)
        assert dm.train_config.mask_ce_loss_coef == pytest.approx(5.0)
        assert dm.train_config.mask_dice_loss_coef == pytest.approx(5.0)


class TestTransferBatchToDevice:
    """Tests for RFDETRDataModule.transfer_batch_to_device().

    Verifies that NestedTensor samples and all target-dict tensors are correctly
    moved to the target device without unwrapping the NestedTensor into plain tensors.
    """

    def test_samples_transferred_to_target_device(self, build_datamodule):
        """Both tensors and mask in NestedTensor must land on the target device."""
        dm = build_datamodule()
        samples, targets = _make_batch()
        device = torch.device("cpu")

        result_samples, _ = dm.transfer_batch_to_device((samples, targets), device, dataloader_idx=0)

        assert result_samples.tensors.device == device
        assert result_samples.mask.device == device

    def test_targets_transferred_to_target_device(self, build_datamodule):
        """All tensor values in every target dict must be moved to the target device."""
        dm = build_datamodule()
        samples, targets = _make_batch()
        device = torch.device("cpu")

        _, result_targets = dm.transfer_batch_to_device((samples, targets), device, dataloader_idx=0)

        for t in result_targets:
            for v in t.values():
                assert v.device == device

    def test_returns_tuple_of_correct_length(self, build_datamodule):
        """Return value must be a (samples, targets) tuple to match batch contract."""
        dm = build_datamodule()
        result = dm.transfer_batch_to_device(_make_batch(), torch.device("cpu"), dataloader_idx=0)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preserves_nested_tensor_type(self, build_datamodule):
        """Device transfer must not unwrap NestedTensor into plain tensors."""
        dm = build_datamodule()
        samples, targets = _make_batch()

        result_samples, _ = dm.transfer_batch_to_device((samples, targets), torch.device("cpu"), dataloader_idx=0)

        assert isinstance(result_samples, NestedTensor)


# ---------------------------------------------------------------------------
# TestBackendResolution — validates augmentation_backend logic in setup("fit")
# ---------------------------------------------------------------------------


class TestBackendResolution:
    """Backend resolution selects Kornia, CPU, or raises depending on environment.

    All tests run on CPU CI by mocking fork-safe CUDA detection and the
    ``kornia`` import as needed.
    """

    def _build_dm_with_backend(self, tmp_path, augmentation_backend="cpu"):
        """Construct a DataModule with the given augmentation_backend."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, augmentation_backend=augmentation_backend)
        from rfdetr.training.module_data import RFDETRDataModule

        return RFDETRDataModule(mc, tc)

    def _setup_with_mock_build(self, dm):
        """Call setup('fit') with build_dataset mocked to avoid real I/O."""
        fake_train = _fake_dataset(100)
        fake_val = _fake_dataset(20)

        def _build(image_set, args, resolution):
            return fake_train if image_set == "train" else fake_val

        with patch("rfdetr.training.module_data.build_dataset", side_effect=_build):
            dm.setup("fit")
        return dm

    def test_auto_no_cuda_falls_back_to_cpu(self, tmp_path):
        """auto + no CUDA: _kornia_pipeline stays None, no error."""
        dm = self._build_dm_with_backend(tmp_path, "auto")
        with patch("rfdetr.training.module_data._has_cuda_device", return_value=False):
            dm = self._setup_with_mock_build(dm)
        assert getattr(dm, "_kornia_pipeline", None) is None, (
            "auto backend with no CUDA must not build a Kornia pipeline"
        )

    def test_auto_no_kornia_falls_back_to_cpu(self, tmp_path):
        """auto + CUDA available but kornia not installed: fallback to CPU."""
        dm = self._build_dm_with_backend(tmp_path, "auto")

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name, *args, **kwargs):
            if name == "kornia" or name.startswith("kornia."):
                raise ImportError("No module named 'kornia'")
            return original_import(name, *args, **kwargs)

        with (
            patch("rfdetr.training.module_data._has_cuda_device", return_value=True),
            patch("builtins.__import__", side_effect=_mock_import),
        ):
            dm = self._setup_with_mock_build(dm)

        assert getattr(dm, "_kornia_pipeline", None) is None, (
            "auto backend with kornia missing must fall back to CPU (pipeline=None)"
        )

    def test_gpu_no_cuda_raises_runtime_error(self, tmp_path):
        """gpu + no CUDA: must raise RuntimeError."""
        dm = self._build_dm_with_backend(tmp_path, "gpu")
        with (
            patch("rfdetr.training.module_data._has_cuda_device", return_value=False),
            pytest.raises(RuntimeError, match="CUDA"),
        ):
            self._setup_with_mock_build(dm)

    def test_gpu_no_kornia_raises_import_error(self, tmp_path):
        """gpu + CUDA but no kornia: must raise ImportError with install hint."""
        dm = self._build_dm_with_backend(tmp_path, "gpu")

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name, *args, **kwargs):
            if name == "kornia" or name.startswith("kornia."):
                raise ImportError("No module named 'kornia'")
            return original_import(name, *args, **kwargs)

        with (
            patch("rfdetr.training.module_data._has_cuda_device", return_value=True),
            patch("builtins.__import__", side_effect=_mock_import),
            pytest.raises(ImportError, match="rfdetr\\[kornia\\]"),
        ):
            self._setup_with_mock_build(dm)

    def test_cpu_backend_builds_no_pipeline(self, tmp_path):
        """Default cpu backend: _kornia_pipeline stays None."""
        dm = self._build_dm_with_backend(tmp_path, "cpu")
        dm = self._setup_with_mock_build(dm)
        assert getattr(dm, "_kornia_pipeline", None) is None, "cpu backend must never build a Kornia pipeline"

    def test_gpu_path_uses_aug_config_fallback(self, tmp_path):
        """When aug_config=None (default), GPU path passes AUG_CONFIG to build_kornia_pipeline."""
        import sys
        from unittest.mock import MagicMock, patch

        from rfdetr.datasets.aug_config import AUG_CONFIG

        dm = self._build_dm_with_backend(tmp_path, "auto")
        assert dm.train_config.aug_config is None, "precondition: aug_config must be None for this test"

        captured = {}

        def _fake_build_kornia(aug_cfg, resolution, with_masks=False):
            captured["aug_config"] = aug_cfg
            return MagicMock()

        with (
            patch("rfdetr.training.module_data._has_cuda_device", return_value=True),
            patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10)),
            patch.dict(sys.modules, {"kornia": MagicMock(), "kornia.augmentation": MagicMock()}),
            patch("rfdetr.datasets.kornia_transforms.build_kornia_pipeline", side_effect=_fake_build_kornia),
            patch("rfdetr.datasets.kornia_transforms.build_normalize", return_value=MagicMock()),
        ):
            dm.setup("fit")

        assert captured.get("aug_config") is AUG_CONFIG, (
            "GPU path must fall back to AUG_CONFIG when train_config.aug_config is None"
        )

    def test_auto_no_cuda_does_not_strip_cpu_normalize(self, tmp_path):
        """auto + no CUDA: gpu_postprocess must be False so CPU Normalize is retained."""
        dm = self._build_dm_with_backend(tmp_path, "auto")
        captured_gpu_postprocess = {}

        def _spy_build(image_set, args, resolution):
            captured_gpu_postprocess[image_set] = getattr(args, "augmentation_backend", "cpu")
            return _fake_dataset(10)

        with (
            patch("rfdetr.training.module_data._has_cuda_device", return_value=False),
            patch("rfdetr.training.module_data.build_dataset", side_effect=_spy_build),
        ):
            dm.setup("fit")

        # When CUDA is unavailable, resolved backend must be 'cpu' so datasets are
        # built with gpu_postprocess=False and CPU Normalize is not stripped.
        assert captured_gpu_postprocess.get("train") == "cpu", (
            "auto + no CUDA must resolve to cpu before dataset build to preserve CPU Normalize"
        )

    def test_resolve_augmentation_backend_auto_no_cuda(self):
        """_resolve_augmentation_backend returns 'cpu' for auto when CUDA is absent."""
        from rfdetr.training.module_data import _resolve_augmentation_backend

        with patch("rfdetr.training.module_data._has_cuda_device", return_value=False):
            assert _resolve_augmentation_backend("auto") == "cpu"

    def test_resolve_augmentation_backend_cpu_passthrough(self):
        """_resolve_augmentation_backend passes 'cpu' through unchanged."""
        from rfdetr.training.module_data import _resolve_augmentation_backend

        assert _resolve_augmentation_backend("cpu") == "cpu"

    def test_resolve_augmentation_backend_gpu_passthrough(self):
        """_resolve_augmentation_backend passes 'gpu' through unchanged."""
        from rfdetr.training.module_data import _resolve_augmentation_backend

        assert _resolve_augmentation_backend("gpu") == "gpu"


# ---------------------------------------------------------------------------
# TestOnAfterBatchTransfer — validates GPU-side augmentation hook
# ---------------------------------------------------------------------------


class TestOnAfterBatchTransfer:
    """on_after_batch_transfer applies Kornia augmentation only during training.

    Uses CPU tensors with a mocked pipeline — no real GPU or Kornia needed.
    """

    def _build_dm(self, tmp_path, segmentation_head=False):
        """Construct a DataModule for on_after_batch_transfer tests."""
        mc = _base_model_config(segmentation_head=segmentation_head)
        tc = _base_train_config(tmp_path)
        from rfdetr.training.module_data import RFDETRDataModule

        return RFDETRDataModule(mc, tc)

    def _attach_mock_trainer(self, dm, training=True):
        """Attach a mock trainer with the given training state to the DataModule."""
        mock_trainer = MagicMock(training=training)
        type(dm).trainer = property(lambda self: mock_trainer)
        return dm

    def _make_kornia_batch(self, batch_size=2, h=16, w=16):
        """Build a batch with xyxy boxes suitable for on_after_batch_transfer.

        Returns (NestedTensor, targets) where boxes are in absolute xyxy format
        and pixel values are in [0, 1] (pre-normalization).
        """
        tensors = torch.rand(batch_size, 3, h, w)  # [0, 1] range
        mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
        samples = NestedTensor(tensors, mask)
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32),
                "labels": torch.tensor([1]),
                "area": torch.tensor([64.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor(i),
                "orig_size": torch.tensor([h, w]),
            }
            for i in range(batch_size)
        ]
        return samples, targets

    def _make_kornia_batch_with_masks(self, batch_size=2, h=16, w=16):
        """Build a batch with xyxy boxes and instance masks for segmentation tests.

        Returns (NestedTensor, targets) where each target includes a 'masks' key
        with one [N, H, W] bool mask tensor per instance.
        """
        tensors = torch.rand(batch_size, 3, h, w)
        mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
        samples = NestedTensor(tensors, mask)
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32),
                "labels": torch.tensor([1]),
                "area": torch.tensor([64.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor(i),
                "orig_size": torch.tensor([h, w]),
                "masks": torch.ones(1, h, w, dtype=torch.bool),
            }
            for i in range(batch_size)
        ]
        return samples, targets

    def test_training_true_applies_augmentation(self, tmp_path):
        """When training=True and _kornia_pipeline is set, image/box outputs match CPU Normalize contract."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch()
        img_aug = samples.tensors.clone()
        # Mock pipeline returns (augmented_images, augmented_boxes)
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        mock_pipeline = MagicMock(return_value=(img_aug, boxes_padded))
        dm._kornia_pipeline = mock_pipeline

        # Normalize adds +1 so we can assert the normalization step is applied.
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x + 1.0)

        result_samples, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        mock_pipeline.assert_called_once()
        dm._kornia_normalize.assert_called_once()
        assert torch.allclose(result_samples.tensors, img_aug + 1.0)
        assert len(result_targets) == 2
        for target in result_targets:
            boxes = target["boxes"]
            assert boxes.shape == (1, 4)
            assert torch.all(boxes >= 0.0)
            assert torch.all(boxes <= 1.0)
            torch.testing.assert_close(
                boxes[0], torch.tensor([0.375, 0.375, 0.5, 0.5], dtype=torch.float32), rtol=1e-4, atol=1e-6
            )

    def test_training_false_skips_augmentation(self, tmp_path):
        """When training=False, batch is returned unchanged."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=False)

        samples, targets = self._make_kornia_batch()
        mock_pipeline = MagicMock()
        dm._kornia_pipeline = mock_pipeline
        dm._kornia_normalize = MagicMock()

        result = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        mock_pipeline.assert_not_called()
        # Batch returned as-is
        result_samples, result_targets = result
        assert result_samples is samples
        assert result_targets is targets

    def test_segmentation_model_applies_augmentation_with_masks(self, tmp_path):
        """Phase 2: segmentation_head=True now calls pipeline with image, boxes, and masks."""
        dm = self._build_dm(tmp_path, segmentation_head=True)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch_with_masks()
        img_aug = samples.tensors.clone()
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        masks_aug = torch.ones(2, 1, 16, 16, dtype=torch.float32)

        mock_pipeline = MagicMock(return_value=(img_aug, boxes_padded, masks_aug))
        dm._kornia_pipeline = mock_pipeline
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        result_samples, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        mock_pipeline.assert_called_once()
        call_args, call_kwargs = mock_pipeline.call_args
        assert len(call_args) == 3, "segmentation augmentation must call pipeline with image, boxes, and masks"
        assert not call_kwargs, "segmentation augmentation should not pass unexpected keyword arguments"

        masks_arg = call_args[2]
        assert isinstance(masks_arg, torch.Tensor), "third pipeline argument must be a masks tensor"
        assert masks_arg.dtype == torch.float32, "masks passed to pipeline must be float32"
        assert masks_arg.shape == (2, 1, 16, 16), "masks passed to pipeline must have shape [B, N_max, H, W]"
        assert "masks" in result_targets[0], "masks key must be present in output targets for segmentation"

    def test_segmentation_masks_stay_in_sync_with_boxes(self, tmp_path):
        """Masks are filtered in sync with boxes: one instance removed → one mask removed."""
        dm = self._build_dm(tmp_path, segmentation_head=True)
        dm = self._attach_mock_trainer(dm, training=True)

        h, w = 16, 16
        tensors = torch.rand(1, 3, h, w)
        mask_nt = torch.zeros(1, h, w, dtype=torch.bool)
        from rfdetr.utilities.tensors import NestedTensor

        samples = NestedTensor(tensors, mask_nt)
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0], [10.0, 10.0, 14.0, 14.0]]),
                "labels": torch.tensor([1, 2]),
                "area": torch.tensor([36.0, 16.0]),
                "iscrowd": torch.tensor([0, 0]),
                "image_id": torch.tensor(0),
                "orig_size": torch.tensor([h, w]),
                "masks": torch.ones(2, h, w, dtype=torch.bool),
            }
        ]
        # Augmented: box 0 survives, box 1 becomes zero-area
        boxes_aug_out = torch.tensor([[[2.0, 2.0, 8.0, 8.0], [5.0, 5.0, 5.0, 5.0]]])
        masks_aug_out = torch.ones(1, 2, h, w, dtype=torch.float32)
        mock_pipeline = MagicMock(return_value=(tensors, boxes_aug_out, masks_aug_out))
        dm._kornia_pipeline = mock_pipeline
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        _, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert result_targets[0]["masks"].shape[0] == 1, (
            f"Expected 1 surviving mask (matching box), got {result_targets[0]['masks'].shape[0]}"
        )

    def test_returns_nested_tensor_in_batch(self, tmp_path):
        """Output batch still has NestedTensor as first element after augmentation."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch()
        img_aug = samples.tensors.clone()
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        dm._kornia_pipeline = MagicMock(return_value=(img_aug, boxes_padded))
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        result_samples, _ = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert isinstance(result_samples, NestedTensor), f"Expected NestedTensor, got {type(result_samples).__name__}"


# ---------------------------------------------------------------------------
# TestKorniaSetupDoneSentinel — validates the _kornia_setup_done guard
# ---------------------------------------------------------------------------


class TestKorniaSetupDoneSentinel:
    """_kornia_setup_done prevents _setup_kornia_pipeline re-running on repeated setup('fit') calls."""

    def _build_dm(self, tmp_path, augmentation_backend="auto"):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, augmentation_backend=augmentation_backend)
        from rfdetr.training.module_data import RFDETRDataModule

        return RFDETRDataModule(mc, tc)

    def _setup_fit_with_mocks(self, dm):
        """Call setup('fit') with build_dataset and cuda mocked (no CUDA → fallback)."""
        fake_train = _fake_dataset(100)
        fake_val = _fake_dataset(20)

        def _build(image_set, args, resolution):
            return fake_train if image_set == "train" else fake_val

        with (
            patch("rfdetr.training.module_data.build_dataset", side_effect=_build),
            patch("rfdetr.training.module_data._has_cuda_device", return_value=False),
        ):
            dm.setup("fit")
        return dm

    def test_sentinel_starts_false(self, tmp_path):
        """_kornia_setup_done is False immediately after __init__."""
        dm = self._build_dm(tmp_path)
        assert dm._kornia_setup_done is False

    def test_sentinel_set_after_fit(self, tmp_path):
        """_kornia_setup_done becomes True after the first setup('fit')."""
        dm = self._build_dm(tmp_path)
        dm = self._setup_fit_with_mocks(dm)
        assert dm._kornia_setup_done is True

    def test_setup_kornia_pipeline_not_called_twice(self, tmp_path):
        """Calling setup('fit') twice only calls _setup_kornia_pipeline once."""
        dm = self._build_dm(tmp_path)
        call_count = 0
        original_setup = dm._setup_kornia_pipeline

        def _counting_setup():
            nonlocal call_count
            call_count += 1
            original_setup()

        dm._setup_kornia_pipeline = _counting_setup

        fake_train = _fake_dataset(100)
        fake_val = _fake_dataset(20)

        def _build(image_set, args, resolution):
            return fake_train if image_set == "train" else fake_val

        with (
            patch("rfdetr.training.module_data.build_dataset", side_effect=_build),
            patch("rfdetr.training.module_data._has_cuda_device", return_value=False),
        ):
            dm.setup("fit")
            dm.setup("fit")

        assert call_count == 1, f"_setup_kornia_pipeline called {call_count} times; expected exactly 1"
