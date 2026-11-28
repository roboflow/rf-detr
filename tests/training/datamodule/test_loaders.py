# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for RFDETRDataModule train dataloader construction, grad-accum dataset alignment and worker init."""

import logging
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from rfdetr.config import KeypointTrainConfig
from rfdetr.datasets.multi_source import WeightedMultiSourceBatchSampler
from rfdetr.training.module_data import _MIN_TRAIN_BATCHES, GradAccumAlignedDataset, RFDETRDataModule, _worker_init_fn
from rfdetr.utilities.tensors import (
    PackedTargets,
)
from tests.training.helpers import _base_model_config, _fake_dataset
from tests.training.helpers import _datamodule_train_config as _base_train_config

_EVAL_LOADERS = ["val_dataloader", "test_dataloader", "predict_dataloader"]


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

        Verifies the workaround for https://github.com/Lightning-AI/pytorch-lightning/issues/19987: the training
        DataLoader must never present a partial accumulation window to PTL.
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
        dm.trainer = MagicMock(world_size=3, accumulate_grad_batches=4)

        loader = dm.train_dataloader()

        assert len(loader.dataset) % (2 * 4 * 3) == 0
        assert len(loader.dataset) == 120

    def test_build_train_sampler_default_returns_none(self, tmp_path):
        """The base build_train_sampler() hook returns None, leaving default sampling untouched."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        assert dm.build_train_sampler(dm._dataset_train) is None

    def test_custom_batch_sampler_is_used_as_is(self, tmp_path):
        """A non-None build_train_sampler() override is passed straight through as batch_sampler."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200, num_workers=0)
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)

        loader = dm.train_dataloader()

        assert loader.batch_sampler is custom_sampler
        dm.build_train_sampler.assert_called_once_with(dm._dataset_train)

    def test_custom_batch_sampler_preserves_loader_kwargs(self, tmp_path):
        """The custom-sampler branch keeps collate_fn/num_workers/pin_memory from the base config."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200, num_workers=0)
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)

        loader = dm.train_dataloader()

        assert loader.collate_fn is dm._collate_fn
        assert loader.num_workers == dm._num_workers
        assert loader.pin_memory == dm._pin_memory

    def test_custom_batch_sampler_skips_grad_accum_alignment(self, tmp_path):
        """The custom-sampler branch does not wrap the dataset in GradAccumAlignedDataset."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=101, batch_size=2, grad_accum_steps=4)
        original_dataset = dm._dataset_train
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(original_dataset), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)

        loader = dm.train_dataloader()

        assert loader.dataset is original_dataset

    def test_warns_when_custom_sampler_leaves_a_short_accumulation_window(self, tmp_path, caplog, monkeypatch):
        """A batch-sampler epoch that is not a multiple of grad_accum_steps under-scales the last optimizer step."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=101, batch_size=2, grad_accum_steps=4)
        # 101 samples in batches of 4 is 26 batches, and 26 % 4 == 2.
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)
        # get_logger() sets propagate=False on the "rf-detr" logger, so caplog's handler only sees its
        # records while propagation is re-enabled.
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            dm.train_dataloader()

        assert "not a multiple of grad_accum_steps" in caplog.text

    def test_no_warning_when_custom_sampler_epoch_is_already_aligned(self, tmp_path, caplog, monkeypatch):
        """An aligned epoch must stay quiet, so the warning keeps its signal."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=96, batch_size=2, grad_accum_steps=4)
        # 96 samples in batches of 4 is exactly 24 batches, and 24 % 4 == 0.
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            dm.train_dataloader()

        assert "not a multiple of grad_accum_steps" not in caplog.text

    def test_no_warning_without_gradient_accumulation(self, tmp_path, caplog, monkeypatch):
        """With grad_accum_steps=1 every window is one micro-batch, so no epoch length can be short."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=101, batch_size=2, grad_accum_steps=1)
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            dm.train_dataloader()

        assert "not a multiple of grad_accum_steps" not in caplog.text

    def test_sampler_without_len_is_not_checked(self, tmp_path):
        """A sampler declaring no epoch length must not break loader construction."""

        class _UnsizedBatchSampler(torch.utils.data.Sampler):
            def __iter__(self):
                yield [0, 1]

        dm = self._setup_dm_with_train(tmp_path, dataset_length=101, batch_size=2, grad_accum_steps=4)
        # No constructor argument: torch dropped Sampler's deprecated `data_source` parameter, so on newer
        # versions this subclass inherits object.__init__ and any argument raises TypeError.
        dm.build_train_sampler = MagicMock(return_value=_UnsizedBatchSampler())

        assert dm.train_dataloader().batch_sampler is not None

    def test_batch_multiple_gives_the_loader_whole_accumulation_windows(self, tmp_path):
        """The documented fix: batch_multiple=grad_accum_steps makes len(loader) an exact number of windows."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=101, batch_size=2, grad_accum_steps=4)
        sampler = WeightedMultiSourceBatchSampler(
            [len(dm._dataset_train)], [1.0], batch_size=2, num_replicas=1, rank=0, batch_multiple=4
        )
        dm.build_train_sampler = MagicMock(return_value=sampler)

        loader = dm.train_dataloader()

        assert len(loader) % 4 == 0

    def test_check_custom_sampler_owns_ddp_noop_when_trainer_is_unset(self, tmp_path):
        """No Trainer attached (e.g. the guard called standalone) never raises."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm._check_custom_sampler_owns_ddp(custom_sampler)

    def test_check_custom_sampler_owns_ddp_noop_for_single_device(self, tmp_path):
        """A non-distributed Trainer (distributed_sampler_kwargs is None) never raises."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        dm.trainer = MagicMock(distributed_sampler_kwargs=None)
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm._check_custom_sampler_owns_ddp(custom_sampler)

    def test_check_custom_sampler_owns_ddp_noop_when_use_distributed_sampler_is_false(self, tmp_path):
        """The correct configuration (use_distributed_sampler=False) never raises."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        trainer = MagicMock()
        trainer.distributed_sampler_kwargs = {"num_replicas": 2, "rank": 0}
        trainer._accelerator_connector.use_distributed_sampler = False
        dm.trainer = trainer
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm._check_custom_sampler_owns_ddp(custom_sampler)

    def test_check_custom_sampler_owns_ddp_raises_when_lightning_would_rewrap_the_sampler(self, tmp_path):
        """A distributed Trainer still set to use_distributed_sampler=True raises before Lightning's TypeError."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        trainer = MagicMock()
        trainer.distributed_sampler_kwargs = {"num_replicas": 2, "rank": 0}
        trainer._accelerator_connector.use_distributed_sampler = True
        trainer._accelerator_connector.is_distributed = True
        dm.trainer = trainer
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )

        with pytest.raises(RuntimeError, match="use_distributed_sampler=False"):
            dm._check_custom_sampler_owns_ddp(custom_sampler)

    def test_train_dataloader_raises_for_custom_sampler_under_misconfigured_ddp(self, tmp_path):
        """train_dataloader() surfaces the DDP guard's RuntimeError before constructing the DataLoader."""
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        trainer = MagicMock()
        trainer.distributed_sampler_kwargs = {"num_replicas": 2, "rank": 0}
        trainer._accelerator_connector.use_distributed_sampler = True
        trainer._accelerator_connector.is_distributed = True
        dm.trainer = trainer
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )
        dm.build_train_sampler = MagicMock(return_value=custom_sampler)

        with pytest.raises(RuntimeError, match="use_distributed_sampler=False"):
            dm.train_dataloader()

    def test_check_custom_sampler_owns_ddp_pins_ptl_private_attributes_on_a_real_trainer(self, tmp_path):
        """The guard's private PyTorch Lightning attributes still exist on a real (non-mocked) Trainer.

        Every other test in this group attaches a ``MagicMock`` trainer, which auto-vivifies any attribute name — so
        `getattr(x, attr, None)` never reaches its `None` default the way it would for a genuinely renamed or removed
        attribute. A real single-device Trainer pins the actual surface, so a PyTorch Lightning upgrade that renames it
        fails this test loudly instead of silently turning the guard into a no-op.
        """
        from pytorch_lightning import Trainer

        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)
        trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
        dm.trainer = trainer
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )

        assert hasattr(trainer._accelerator_connector, "use_distributed_sampler")
        assert hasattr(trainer._accelerator_connector, "is_distributed")
        dm._check_custom_sampler_owns_ddp(custom_sampler)  # single device: never raises

    def test_check_custom_sampler_owns_ddp_noop_when_flag_attribute_is_genuinely_absent(self, tmp_path):
        """A truly missing attribute (not a ``MagicMock`` auto-vivified one) still falls back safely.

        ``MagicMock`` returns a truthy sub-mock for any attribute name, so a MagicMock trainer can never exercise the
        ``getattr(..., None)`` fallback the way a real missing attribute would.
        """
        dm = self._setup_dm_with_train(tmp_path, dataset_length=200)

        class _AccelConnectorMissingUseDistributedSampler:
            is_distributed = True

        class _TrainerWithIncompleteConnector:
            distributed_sampler_kwargs = {"num_replicas": 2, "rank": 0}
            _accelerator_connector = _AccelConnectorMissingUseDistributedSampler()

        dm.trainer = _TrainerWithIncompleteConnector()
        custom_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dm._dataset_train), batch_size=4, drop_last=False
        )

        dm._check_custom_sampler_owns_ddp(custom_sampler)  # missing use_distributed_sampler -> no raise

    @pytest.mark.parametrize(
        ("dataset_length", "grad_accum_steps", "trainer_grad_accum_steps", "expected_samples"),
        [
            pytest.param(3, 1, 1, 20, id="far_below_threshold"),
            pytest.param(12, 1, 1, 20, id="between_single_and_ddp_thresholds"),
            pytest.param(19, 1, 1, 20, id="threshold_minus_one"),
            pytest.param(20, 1, 1, 20, id="exact_threshold"),
            pytest.param(21, 1, 1, 24, id="threshold_plus_one"),
            pytest.param(3, 3, 3, 60, id="configured_gradient_accumulation"),
            pytest.param(3, 1, 2, 40, id="trainer_gradient_accumulation_override"),
        ],
    )
    def test_ddp_preserves_minimum_effective_batches_per_rank(
        self,
        tmp_path: Path,
        dataset_length: int,
        grad_accum_steps: int,
        trainer_grad_accum_steps: int,
        expected_samples: int,
    ) -> None:
        """DDP keeps five complete optimizer steps per rank across the small-dataset threshold."""
        batch_size = 2
        world_size = 2
        dm = self._setup_dm_with_train(
            tmp_path,
            dataset_length=dataset_length,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
        dm.trainer = MagicMock(
            world_size=world_size,
            accumulate_grad_batches=trainer_grad_accum_steps,
        )

        loader = dm.train_dataloader()

        assert len(loader.dataset) == expected_samples

    def test_ddp_keypoint_uses_manually_owned_gradient_accumulation(self, tmp_path: Path) -> None:
        """Keypoint padding follows TrainConfig rather than Lightning's forced accumulation value of one."""
        model_config = _base_model_config(use_grouppose_keypoints=True)
        train_config = KeypointTrainConfig(
            **_base_train_config(tmp_path, batch_size=2, grad_accum_steps=3).model_dump()
        )
        dm = RFDETRDataModule(model_config, train_config)
        dm._dataset_train = _fake_dataset(3)
        dm.trainer = MagicMock(world_size=2, accumulate_grad_batches=1)

        loader = dm.train_dataloader()

        assert len(loader.dataset) == 60

    @staticmethod
    def _raw_sample(h: int = 16, w: int = 16) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build one (image, target) pair as a dataset __getitem__ would return it, for collate_fn input.

        Examples:
            >>> image, target = TestTrainDataloader._raw_sample()
            >>> image.shape, sorted(target)
            (torch.Size([3, 16, 16]), ['boxes', 'image_id', 'labels', 'orig_size'])
        """
        image = torch.randn(3, h, w)
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(0),
            "orig_size": torch.tensor([h, w]),
        }
        return image, target

    @pytest.mark.parametrize(
        ("loader_name", "dataset_attribute"),
        [
            pytest.param("train_dataloader", "_dataset_train", id="train"),
            pytest.param("val_dataloader", "_dataset_val", id="validation"),
            pytest.param("test_dataloader", "_dataset_test", id="test"),
            pytest.param("predict_dataloader", "_dataset_val", id="predict"),
        ],
    )
    def test_pack_targets_default_makes_every_loader_collate_packed_targets(
        self, tmp_path, loader_name, dataset_attribute
    ):
        """The default must make each public DataLoader's collate function return PackedTargets."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path))
        setattr(dm, dataset_attribute, _fake_dataset(200))

        loader = getattr(dm, loader_name)()
        _, targets = loader.collate_fn([self._raw_sample(), self._raw_sample()])

        assert isinstance(targets, PackedTargets)

    def test_pack_targets_false_keeps_collate_fn_output_a_tuple_of_dicts(self, tmp_path):
        """An explicit TrainConfig.pack_targets=False leaves train_dataloader().collate_fn output unpacked."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, pack_targets=False))
        dm._dataset_train = _fake_dataset(200)

        loader = dm.train_dataloader()
        _, targets = loader.collate_fn([self._raw_sample(), self._raw_sample()])

        assert not isinstance(targets, PackedTargets)
        assert all(isinstance(t, dict) for t in targets)

    def test_pad_targets_to_composes_with_pack_for_the_train_loader(self, tmp_path):
        """pad_targets_to runs before pack in the collate seam (see make_collate_fn): once every sample shares one row
        count, a batch that previously packed still packs, at the padded shape."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, pack_targets=True, pad_targets_to=4))
        dm._dataset_train = _fake_dataset(200)
        image, one_box = self._raw_sample()
        three_boxes = {**one_box, "boxes": torch.rand(3, 4), "labels": torch.arange(3)}

        loader = dm.train_dataloader()
        _, targets = loader.collate_fn([(image, one_box), (image, three_boxes)])

        assert isinstance(targets, PackedTargets)
        rebuilt = list(targets)
        assert [t["boxes"].shape[0] for t in rebuilt] == [4, 4]
        assert rebuilt[0]["valid"].tolist() == [True, False, False, False]
        assert rebuilt[1]["valid"].tolist() == [True, True, True, False]

    def test_pad_targets_to_only_reaches_the_train_loader(self, tmp_path):
        """Padding the eval loaders would feed filler rows to COCO matching as real ground truth, so only
        train_dataloader() may pad; val/test/predict keep the real, variable-length targets."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, pack_targets=False, pad_targets_to=4))
        dm._dataset_train = _fake_dataset(200)
        dm._dataset_val = _fake_dataset(200)

        _, train_targets = dm.train_dataloader().collate_fn([self._raw_sample()])
        _, val_targets = dm.val_dataloader().collate_fn([self._raw_sample()])

        assert train_targets[0]["boxes"].shape[0] == 4
        assert "valid" in train_targets[0]
        assert val_targets[0]["boxes"].shape[0] == 1
        assert "valid" not in val_targets[0]

    @patch("rfdetr.training.module_data.build_webdataset_loader")
    def test_webdataset_loader_pads_only_the_fixed_epoch_train_call(self, mock_build_webdataset_loader, tmp_path):
        """_webdataset_loader is shared by train (fixed_epoch=True) and eval; only the train call may collate through
        the padded/packed self._collate_fn_train."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, pack_targets=True, pad_targets_to=4))
        captured = {}

        def _fake_build_webdataset_loader(dataset, *, collate_fn, **kwargs):
            captured["collate_fn"] = collate_fn
            return MagicMock()

        mock_build_webdataset_loader.side_effect = _fake_build_webdataset_loader

        dm._webdataset_loader(MagicMock(), batch_size=2, fixed_epoch=True)
        assert captured["collate_fn"] is dm._collate_fn_train

        dm._webdataset_loader(MagicMock(), batch_size=2, fixed_epoch=False)
        assert captured["collate_fn"] is dm._collate_fn

    @staticmethod
    def _raw_segmentation_sample(
        h: int = 16, w: int = 16, num_instances: int = 1
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build one (image, target) pair shaped exactly as the COCO reader produces for a
        segmentation model: ``masks`` is ``torch.bool`` of shape ``(num_instances, H, W)``,
        alongside the ``area``/``iscrowd``/``size`` fields ``rfdetr.datasets.coco.py`` always
        attaches next to it. ``image_id`` is rank-1 (``torch.as_tensor([image_id])``,
        ``coco.py:653``), not the rank-0 scalar ``ConvertCoco``'s own docstring calls it
        (``coco.py:615``) -- matched to the actual runtime shape here.

        Args:
            h: Image and mask height. Pass distinct values across samples in the same
                batch to exercise ``pack_targets``'s per-sample shape bookkeeping --
                ``RandomResize`` preserves each source image's own aspect ratio, so a real
                collated batch routinely mixes segmentation masks of different spatial shape.
            w: Image and mask width, independent of ``h`` for the same reason.
            num_instances: Number of instances in the sample; ``0`` produces an
                empty-but-shaped ``masks`` tensor.

        Returns:
            The synthetic image tensor and its matching target dict.

        Examples:
            >>> image, target = TestTrainDataloader._raw_segmentation_sample(num_instances=2)
            >>> target["masks"].shape, target["masks"].dtype
            (torch.Size([2, 16, 16]), torch.bool)
        """
        image = torch.randn(3, h, w)
        target = {
            "boxes": torch.rand(num_instances, 4),
            "labels": torch.arange(num_instances, dtype=torch.int64),
            "image_id": torch.as_tensor([0]),
            "area": torch.rand(num_instances),
            "iscrowd": torch.zeros(num_instances, dtype=torch.int64),
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
            "masks": torch.rand(num_instances, h, w) > 0.5,
        }
        return image, target

    def test_pack_targets_round_trips_segmentation_masks_bit_identically(self, tmp_path):
        """#1399's own body flagged this as unmeasured: the packer handles any same-keyed field, including ``masks``,
        but no parity run existed for it.

        This pins correctness through the real DataModule collate seam, using ``to_list()`` -- the exact method
        ``transfer_batch_to_device`` calls on the real training path -- rather than a related but different
        iteration method. The middle, zero-instance sample mirrors
        ``TestPackedTargets.test_a_sample_with_no_instances_survives_the_round_trip`` in
        ``tests/utilities/test_tensors.py`` -- that test pins the same "must not collapse into a neighbour's rows"
        invariant for ``boxes``/``labels``, but never through a real segmentation batch's ``masks`` field.
        """
        model_config = _base_model_config(segmentation_head=True)
        dm = RFDETRDataModule(model_config, _base_train_config(tmp_path, pack_targets=True))
        dm._dataset_train = _fake_dataset(200)

        loader = dm.train_dataloader()
        # Distinct, non-transposed H/W per sample: RandomResize keeps aspect ratios, so real batches mix mask
        # shapes; same-shape samples would round-trip even if pack_targets reused one sample's spatial shape.
        sample_a = self._raw_segmentation_sample(h=14, w=22, num_instances=1)
        sample_zero = self._raw_segmentation_sample(h=20, w=10, num_instances=0)
        sample_b = self._raw_segmentation_sample(h=18, w=16, num_instances=3)
        _, packed = loader.collate_fn([sample_a, sample_zero, sample_b])

        assert isinstance(packed, PackedTargets), "a real segmentation batch must still pack"
        rebuilt = packed.to_list(torch.device("cpu"))
        for (_, expected), actual in zip([sample_a, sample_zero, sample_b], rebuilt, strict=True):
            assert actual["masks"].dtype == torch.bool
            assert actual["masks"].shape == expected["masks"].shape
            assert torch.equal(actual["masks"], expected["masks"]), "masks must round-trip bit-identically"
            assert torch.equal(actual["boxes"], expected["boxes"])
            assert torch.equal(actual["labels"], expected["labels"])
            assert torch.equal(actual["area"], expected["area"])
            assert torch.equal(actual["iscrowd"], expected["iscrowd"])
        assert rebuilt[1]["masks"].shape == (0, 20, 10), "the zero-instance sample must not collapse into a neighbour"


class TestGradAccumAlignedDataset:
    """Unit tests for the GradAccumAlignedDataset wrapper."""

    def _make_dataset(self, length: int) -> torch.utils.data.TensorDataset:
        """Return a simple TensorDataset of given length."""
        return torch.utils.data.TensorDataset(torch.arange(length))

    @pytest.mark.parametrize(
        "length, effective_batch_size, world_size, minimum_length, expected_length",
        [
            pytest.param(64, 16, 1, None, 64, id="no-padding-needed-when-already-aligned"),
            pytest.param(50, 16, 1, None, 64, id="padding-adds-correct-count"),  # 50 % 16 = 2 -> pad 14
            pytest.param(3, 2, 2, 20, 20, id="minimum-length-repeats-to-requested-aligned-size"),
        ],
    )
    def test_wrapped_length(self, length, effective_batch_size, world_size, minimum_length, expected_length):
        """Len(wrapped) reflects padding to the alignment unit, or to minimum_length when it is larger."""
        ds = self._make_dataset(length)
        wrapped = GradAccumAlignedDataset(
            ds, effective_batch_size=effective_batch_size, world_size=world_size, minimum_length=minimum_length
        )
        assert len(wrapped) == expected_length

    def test_getitem_forwards_to_original_dataset(self):
        """Items in the original range map directly to the underlying dataset."""
        ds = self._make_dataset(10)
        wrapped = GradAccumAlignedDataset(ds, effective_batch_size=4, world_size=1)
        for i in range(10):
            (val,) = wrapped[i]
            assert val.item() == i

    def test_padded_indices_are_valid(self):
        """All padded indices point to valid positions in the original dataset."""
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
        """Len(wrapped) % (eff_bs * world_size) == 0 for all inputs."""
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
        ds = self._make_dataset(10)
        with pytest.raises(ValueError) as exc_info:
            GradAccumAlignedDataset(ds, effective_batch_size=effective_batch_size, world_size=world_size)
        assert "must be >= 1" in str(exc_info.value)


class TestWorkerInitFn:
    """DataLoaders seed NumPy/random per worker so augmentation streams are not duplicated across workers."""

    def test_worker_init_fn_seeds_from_torch_initial_seed(self, monkeypatch):
        """_worker_init_fn derives a reproducible NumPy/random seed from ``torch.initial_seed``."""
        monkeypatch.setattr(torch, "initial_seed", lambda: 12345)
        _worker_init_fn(0)
        first = (float(np.random.rand()), random.random())

        # worker_id is irrelevant; the seed is derived from torch's per-worker seed.
        monkeypatch.setattr(torch, "initial_seed", lambda: 12345)
        _worker_init_fn(3)
        second = (float(np.random.rand()), random.random())

        assert first == second

    @pytest.mark.parametrize("loader_name", _EVAL_LOADERS)
    def test_eval_dataloaders_set_worker_init_fn(self, fixture_training_setup, loader_name):
        """Validation/test/predict DataLoaders wire the module-level worker seeding hook."""
        _, _, dm = fixture_training_setup
        dm._dataset_val = _fake_dataset()
        dm._dataset_test = _fake_dataset()

        loader = getattr(dm, loader_name)()

        assert loader.worker_init_fn is _worker_init_fn

    def test_train_dataloader_sets_worker_init_fn(self, fixture_training_setup):
        """The training DataLoader wires the module-level worker seeding hook."""
        _, _, dm = fixture_training_setup
        dm._dataset_train = _fake_dataset()

        loader = dm.train_dataloader()

        assert loader.worker_init_fn is _worker_init_fn
