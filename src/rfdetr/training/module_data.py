# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""LightningDataModule for RF-DETR dataset construction and loaders."""

from typing import Any, List, Optional, Tuple

import torch
import torch.utils.data
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets import build_dataset
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import collate_fn

logger = get_logger()

_MIN_TRAIN_BATCHES = 5


class GradAccumAlignedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that pads length to a multiple of ``effective_batch_size * world_size``.

    Workaround for https://github.com/Lightning-AI/pytorch-lightning/issues/19987:
    PTL fires the optimizer on partial accumulation windows at the tail of the
    dataset, causing the last optimizer step to be under-scaled.  Padding the
    dataset to a multiple of ``effective_batch_size * world_size`` ensures that
    ``drop_last=True`` on the DataLoader becomes a true no-op — every
    accumulation window is always complete.

    Padding indices are drawn randomly from the original dataset.  Because RF-DETR
    uses online augmentation, each padded sample receives a fresh random
    augmentation at ``__getitem__`` time, so it behaves like a new training
    example rather than a true duplicate.

    This wrapper can be removed once the upstream PTL issue is resolved.

    Args:
        dataset: The underlying dataset to wrap.
        effective_batch_size: ``batch_size * grad_accum_steps``.
        world_size: Number of DDP processes (default 1 for single-GPU/CPU).
            The alignment unit is ``effective_batch_size * world_size`` so that
            after PTL's ``DistributedSampler`` splits samples across ranks each
            rank still receives an exact multiple of ``effective_batch_size``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        effective_batch_size: int,
        world_size: int = 1,
    ) -> None:
        if effective_batch_size < 1:
            raise ValueError(f"effective_batch_size must be >= 1, got {effective_batch_size}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")

        self._dataset = dataset
        self._dataset_length = len(dataset)  # type: ignore[arg-type]
        pad_unit = effective_batch_size * world_size
        remainder = self._dataset_length % pad_unit
        pad_count = (pad_unit - remainder) % pad_unit
        pad_index_generator = torch.Generator()
        pad_index_generator.manual_seed(0)
        self._pad_indices: list[int] = (
            torch.randint(
                0,
                self._dataset_length,
                (pad_count,),
                generator=pad_index_generator,
            ).tolist()
            if pad_count > 0
            else []
        )
        self._length = self._dataset_length + pad_count

    def __len__(self) -> int:
        """Return the padded dataset length (always a multiple of the alignment unit)."""
        return self._length

    def __getitem__(self, idx: int) -> Any:
        """Return the item at the (possibly remapped) index."""
        # pad_indices are fixed at __init__ time; same indices reused every epoch
        # (different augmentations per epoch due to online augmentation)
        dataset_idx = idx if idx < self._dataset_length else self._pad_indices[idx - self._dataset_length]
        return self._dataset[dataset_idx]


class RFDETRDataModule(LightningDataModule):
    """LightningDataModule wrapping RF-DETR dataset construction and data loading.

    Args:
        model_config: Architecture configuration (used for resolution, patch_size, etc.).
        train_config: Training hyperparameter configuration (used for dataset params).
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config

        self._dataset_train: Optional[torch.utils.data.Dataset] = None
        self._dataset_val: Optional[torch.utils.data.Dataset] = None
        self._dataset_test: Optional[torch.utils.data.Dataset] = None

        self._num_workers: int = self.train_config.num_workers

        # Use the fork-safe DEVICE constant instead of torch.cuda.is_available(),
        # which creates a CUDA driver context that breaks fork-based DDP.
        from rfdetr.config import DEVICE

        accelerator = str(self.train_config.accelerator).lower()
        uses_cuda_accelerator = accelerator in {"auto", "gpu", "cuda"}
        self._pin_memory: bool = (
            (DEVICE == "cuda" and uses_cuda_accelerator)
            if self.train_config.pin_memory is None
            else bool(self.train_config.pin_memory)
        )
        self._persistent_workers: bool = (
            self._num_workers > 0
            if self.train_config.persistent_workers is None
            else bool(self.train_config.persistent_workers)
        )
        if self._num_workers > 0:
            self._prefetch_factor = (
                self.train_config.prefetch_factor if self.train_config.prefetch_factor is not None else 2
            )
        else:
            self._prefetch_factor = None

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
        resolution = self.model_config.resolution
        ns = _namespace_from_configs(self.model_config, self.train_config)
        if stage == "fit":
            if self._dataset_train is None:
                self._dataset_train = build_dataset("train", ns, resolution)
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
        elif stage == "test":
            if self._dataset_test is None:
                split = "test" if self.train_config.dataset_file == "roboflow" else "val"
                self._dataset_test = build_dataset(split, ns, resolution)
        elif stage == "predict":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Uses a replacement sampler when the dataset is too small to fill
        ``_MIN_TRAIN_BATCHES`` effective batches (matching legacy behaviour in
        ``main.py``).  Otherwise wraps the dataset with
        :class:`GradAccumAlignedDataset` to ensure its length is an exact
        multiple of ``effective_batch_size * world_size`` (workaround for
        https://github.com/Lightning-AI/pytorch-lightning/issues/19987) and
        then uses ``shuffle=True, drop_last=True`` so that PTL can
        auto-inject ``DistributedSampler`` in DDP mode.

        Returns:
            DataLoader for the training dataset.
        """
        dataset = self._dataset_train
        batch_size = self.train_config.batch_size
        effective_batch_size = batch_size * self.train_config.grad_accum_steps
        num_workers = self._num_workers

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
                num_workers=num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
            )

        # Pad the dataset to a multiple of effective_batch_size * world_size so
        # that drop_last=True below becomes a true no-op and PTL never fires the
        # optimizer on a partial accumulation window.
        # See https://github.com/Lightning-AI/pytorch-lightning/issues/19987
        world_size: int = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        dataset = GradAccumAlignedDataset(dataset, effective_batch_size, world_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # no-op after alignment, but keeps intent explicit
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        return DataLoader(
            self._dataset_val,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_val),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Returns:
            DataLoader for the test dataset with sequential sampling.
        """
        return DataLoader(
            self._dataset_test,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_test),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the predict DataLoader (reuses the validation dataset, no augmentation).

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        return DataLoader(
            self._dataset_val,
            batch_size=self.train_config.batch_size,
            sampler=torch.utils.data.SequentialSampler(self._dataset_val),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
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

    def transfer_batch_to_device(self, batch: Tuple, device: torch.device, dataloader_idx: int) -> Tuple:
        """Move a ``(NestedTensor, targets)`` batch to *device*.

        PTL's default iterates tuple elements and calls ``.to(device)``; that
        works for plain tensors but ``NestedTensor`` must be moved explicitly.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            device: Target device.
            dataloader_idx: Index of the dataloader providing this batch.

        Returns:
            Batch with all tensors on ``device``.
        """
        samples, targets = batch
        non_blocking = device.type == "cuda"
        samples = samples.to(device, non_blocking=non_blocking)
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
        return samples, targets
