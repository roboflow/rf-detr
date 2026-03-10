# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""LightningDataModule for RF-DETR dataset construction and loaders (Phase 2)."""

from typing import Any, List, Optional

import torch
import torch.utils.data
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rfdetr._namespace import build_namespace
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets import build_dataset
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import collate_fn

logger = get_logger()

_MIN_TRAIN_BATCHES = 5


class RFDETRDataModule(LightningDataModule):
    """LightningDataModule wrapping RF-DETR dataset construction and data loading.

    Migrates ``Model.train()`` dataset construction and DataLoader setup from
    ``main.py`` into PTL lifecycle hooks.  Coexists with the existing code until
    Chapter 4 removes the legacy path.

    Args:
        model_config: Architecture configuration (used for resolution, patch_size, etc.).
        train_config: Training hyperparameter configuration (used for dataset params).
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        # TODO(Chapter 6): remove _args; read from model_config / train_config directly.
        self._args = self._build_args()

        self._dataset_train: Optional[torch.utils.data.Dataset] = None
        self._dataset_val: Optional[torch.utils.data.Dataset] = None
        self._dataset_test: Optional[torch.utils.data.Dataset] = None

        num_workers = self._args.num_workers
        self._pin_memory: bool = (
            torch.cuda.is_available() if self.train_config.pin_memory is None else bool(self.train_config.pin_memory)
        )
        self._persistent_workers: bool = (
            num_workers > 0
            if self.train_config.persistent_workers is None
            else bool(self.train_config.persistent_workers)
        )
        if num_workers > 0:
            self._prefetch_factor = (
                self.train_config.prefetch_factor if self.train_config.prefetch_factor is not None else 2
            )
        else:
            self._prefetch_factor = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # TODO(Chapter 6): delete _build_args() when _args.py / populate_args() are removed.
    def _build_args(self) -> Any:
        """Map Pydantic configs to the legacy argparse.Namespace.

        Returns:
            Namespace compatible with ``build_dataset``.
        """
        return build_namespace(self.model_config, self.train_config)

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Build datasets for the requested stage.

        PTL calls this on every process before the corresponding
        dataloader method.  Datasets are built lazily — a dataset is
        only constructed once even if ``setup`` is called multiple times.

        Args:
            stage: PTL stage identifier — one of ``"fit"``, ``"validate"``,
                ``"test"``, or ``"predict"``.
        """
        args = self._args
        if stage == "fit":
            if self._dataset_train is None:
                self._dataset_train = build_dataset("train", args, args.resolution)
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", args, args.resolution)
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", args, args.resolution)
        elif stage == "test":
            if self._dataset_test is None:
                split = "test" if args.dataset_file == "roboflow" else "val"
                self._dataset_test = build_dataset(split, args, args.resolution)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Uses a replacement sampler when the dataset is too small to fill
        ``_MIN_TRAIN_BATCHES`` effective batches (matching legacy behaviour in
        ``main.py``).  Otherwise uses a ``BatchSampler`` with
        ``drop_last=True`` to avoid incomplete batches.

        Returns:
            DataLoader for the training dataset.
        """
        args = self._args
        dataset = self._dataset_train
        batch_size = args.batch_size
        effective_batch_size = batch_size * args.grad_accum_steps

        if len(dataset) < effective_batch_size * _MIN_TRAIN_BATCHES:
            logger.info(
                "Training with uniform sampler because dataset is too small: %d < %d",
                len(dataset),
                effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            sampler = torch.utils.data.RandomSampler(
                dataset,
                replacement=True,
                num_samples=effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
            )

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset),
            batch_size,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        args = self._args
        return DataLoader(
            self._dataset_val,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_val),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Returns:
            DataLoader for the test dataset with sequential sampling.
        """
        args = self._args
        return DataLoader(
            self._dataset_test,
            batch_size=args.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_test),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> Optional[List[str]]:
        """Class names from the training or validation dataset annotation file.

        Reads category names from the first available COCO-style dataset.
        Returns ``None`` if no dataset has been set up yet or the dataset
        does not expose COCO-style category information.

        Returns:
            Sorted list of class name strings, or ``None``.
        """
        for dataset in (self._dataset_train, self._dataset_val):
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                return [coco.cats[k]["name"] for k in sorted(coco.cats.keys())]
        return None
