# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningDataModule for RF-DETR dataset construction and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.utils.data
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import AugmentationBackend, ModelConfig, TrainConfig
from rfdetr.datasets import build_dataset
from rfdetr.datasets.aug_configs import AUG_CONFIG
from rfdetr.datasets.yolo import YoloSplitUnavailableError
from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import make_collate_fn

logger = get_logger()

_MIN_TRAIN_BATCHES = 5

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and the ``random`` module per DataLoader worker.

    PyTorch seeds ``torch``'s RNG per worker automatically but leaves NumPy and the stdlib ``random`` module unseeded,
    so without this hook every worker would draw identical NumPy/``random`` sequences (a well-known augmentation
    duplication footgun). Deriving the seed from ``torch.initial_seed()`` keeps augmentation reproducible while still
    giving each worker a distinct stream.

    Args:
        worker_id: Index of the DataLoader worker (unused; seed is derived from the per-worker torch seed).
    """
    import random

    import numpy as np

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _has_cuda_device() -> bool:
    """Return ``True`` when the runtime has a CUDA accelerator available.

    Uses the fork-safe global ``DEVICE`` constant instead of direct ``torch.cuda.is_available()`` calls to avoid
    creating a CUDA context in fork-based notebook/DDP workflows.
    """
    from rfdetr.config import DEVICE

    return str(DEVICE).startswith("cuda")


class GradAccumAlignedDataset(torch.utils.data.Dataset[Any]):
    """Dataset wrapper that pads length to a multiple of ``effective_batch_size * world_size``.

    Workaround for https://github.com/Lightning-AI/pytorch-lightning/issues/19987: PTL fires the optimizer on partial
    accumulation windows at the tail of the dataset, causing the last optimizer step to be under-scaled.  Padding the
    dataset to a multiple of ``effective_batch_size * world_size`` ensures that ``drop_last=True`` on the DataLoader
    becomes a true no-op — every accumulation window is always complete.

    Padding indices are drawn randomly from the original dataset.  Because RF-DETR uses online augmentation, each padded
    sample receives a fresh random augmentation at ``__getitem__`` time, so it behaves like a new training example
    rather than a true duplicate.

    This wrapper can be removed once the upstream PTL issue is resolved.

    Args:
        dataset: The underlying dataset to wrap.
        effective_batch_size: ``batch_size * grad_accum_steps``.
        world_size: Number of DDP processes (default 1 for single-GPU/CPU).
            The alignment unit is ``effective_batch_size * world_size`` so that after PTL's ``DistributedSampler``
            splits samples across ranks each rank still receives an exact multiple of ``effective_batch_size``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Any],
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

        # Backbone divisibility requirement: inputs with windowed attention must
        # have H and W divisible by patch_size * num_windows. The collate_fn
        # below rounds batch-max H/W up to this value so the mask accurately
        # marks every pad pixel.
        block_size = model_config.patch_size * model_config.num_windows
        if block_size <= 0:
            raise ValueError(
                "Computed collate block_size must be > 0, got "
                f"{block_size} from patch_size={model_config.patch_size} "
                f"and num_windows={model_config.num_windows}."
            )
        self._collate_fn = make_collate_fn(
            block_size=block_size,
        )

        self._dataset_train: torch.utils.data.Dataset[Any] | None = None
        self._dataset_val: torch.utils.data.Dataset[Any] | None = None
        self._dataset_test: torch.utils.data.Dataset[Any] | None = None

        # GPU augmentation pipeline (Kornia); built lazily in setup("fit").
        self._kornia_pipeline: Any | None = None
        self._kornia_normalize: Any | None = None
        # Resolved backend for the pending/most recent _setup_kornia_pipeline() call; set by
        # setup("fit") just before that call so _setup_kornia_pipeline can stay a zero-arg method.
        self._resolved_augmentation_backend: AugmentationBackend | None = None
        # Sentinel: True once _setup_kornia_pipeline has run (even on fallback paths
        # where _kornia_pipeline stays None), preventing redundant re-runs on repeated
        # setup("fit") calls (e.g. during validation loops in some PTL strategies).
        self._kornia_setup_done: bool = False

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
        self._prefetch_factor: int | None
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

        PTL calls this on every process before the corresponding dataloader method.  Datasets are built lazily — a
        dataset is only constructed once even if ``setup`` is called multiple times.

        Args:
            stage: PTL stage identifier — one of ``"fit"``, ``"validate"``,
                ``"test"``, or ``"predict"``.
        """
        resolution = self.model_config.resolution
        ns = _namespace_from_configs(self.model_config, self.train_config)
        if stage == "fit":
            requested_backend = self.train_config.augmentation_backend
            # Keypoint transforms are incompatible with the Kornia GPU pipeline outright, so an
            # explicit 'kornia'/'gpu' request is rejected here -- before the CUDA readiness check
            # below -- so a keypoint model without CUDA still sees the keypoint error, not an
            # unrelated "no CUDA" one.
            if self.model_config.use_grouppose_keypoints and requested_backend in (
                AugmentationBackend.KORNIA,
                "kornia",
                "gpu",
            ):
                raise ValueError(
                    f"augmentation_backend={requested_backend!r} does not support keypoint transforms. "
                    "Set augmentation_backend='cpu' or 'albumentations' when use_grouppose_keypoints=True."
                )
            from rfdetr.datasets.kornia_transforms import (
                is_gpu_postprocess,
                require_gpu_backend_ready,
                resolve_augmentation_backend,
            )

            require_gpu_backend_ready(requested_backend, has_cuda=_has_cuda_device())
            # Resolve 'auto' to an actual backend before building datasets so that
            # gpu_postprocess in dataset builders always matches what the DataModule
            # will actually do in on_after_batch_transfer.  Without this, 'auto' on
            # a machine without CUDA/kornia would strip CPU Normalize from datasets
            # while _kornia_pipeline stays None, leaving training inputs unnormalized.
            resolved = resolve_augmentation_backend(requested_backend, has_cuda=_has_cuda_device())
            if resolved != requested_backend:
                # 'auto'/'cpu' (and legacy aliases) were resolved above only to decide the
                # DataModule-side Kornia pipeline and AUG_CONFIG injection below. Dataset
                # builders do their own environment-aware CPU sub-backend pick (Albumentations
                # vs. torchvision), so forward the simplified sentinel -- not e.g. the specific
                # TV/ALBU pick -- to stay consistent with the gpu_postprocess flag they compute.
                ns.augmentation_backend = "kornia" if is_gpu_postprocess(resolved) else "cpu"
            # ALBU forces Albumentations even when aug_config is None
            if resolved == AugmentationBackend.ALBU and ns.aug_config is None:
                ns.aug_config = AUG_CONFIG
            if self.model_config.use_grouppose_keypoints and is_gpu_postprocess(resolved):
                raise ValueError(
                    f"augmentation_backend='{resolved}' does not support keypoint transforms. "
                    "Set augmentation_backend='cpu' or 'albumentations' when use_grouppose_keypoints=True."
                )
            if self._dataset_train is None:
                self._dataset_train = build_dataset("train", ns, resolution)
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
            # Build Kornia pipeline (once); use _kornia_setup_done so fallback paths
            # (pipeline stays None) do not re-run on repeated setup("fit") calls.
            if not self._kornia_setup_done:
                self._resolved_augmentation_backend = resolved
                self._setup_kornia_pipeline()
                self._kornia_setup_done = True
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)
        elif stage == "test":
            if self._dataset_test is None:
                self._dataset_test = self._build_test_dataset(ns, resolution)
        elif stage == "predict":
            if self._dataset_val is None:
                self._dataset_val = build_dataset("val", ns, resolution)

    def _build_test_dataset(self, ns: Any, resolution: int) -> torch.utils.data.Dataset[Any]:
        """Build the dataset backing ``setup("test")``, preferring a real test split.

        ``roboflow`` and ``yolo`` datasets carry a labelled ``test`` split, so it is used directly.  A YOLO-format
        dataset is not required to declare one: a ``roboflow`` dataset whose detected format is YOLO-style routes
        through the same builder as ``dataset_file="yolo"`` (:func:`rfdetr.datasets.build_roboflow` ->
        ``build_roboflow_from_yolo``), and Roboflow's export UI does not require a test split any more than a
        hand-built YOLO dataset does. When that YOLO-style split cannot be resolved, the ``val`` split is used
        instead and the substitution is logged, because falling back silently reports validation numbers under
        the name "test".  A ``roboflow`` export in COCO format still has to ship ``test/_annotations.coco.json``:
        its absence raises a plain ``FileNotFoundError`` from the COCO builder, which this fallback does not catch.

        ``coco`` falls back to ``val`` because its ``test`` split is unlabelled COCO test-dev and cannot be scored
        locally.  ``o365`` also falls back because its dataset builder exposes only ``train`` and ``val`` splits.

        Args:
            ns: Merged model/train config namespace forwarded to :func:`build_dataset`.
            resolution: Target square resolution in pixels.

        Returns:
            The dataset to evaluate during the ``test`` stage.
        """
        dataset_file = self.train_config.dataset_file
        if dataset_file in ("roboflow", "yolo"):
            try:
                return build_dataset("test", ns, resolution)
            except YoloSplitUnavailableError as exc:
                dataset_label = {"roboflow": "Roboflow", "yolo": "YOLO"}[dataset_file]
                logger.warning(
                    "No resolvable 'test' split for this %s dataset (%s); evaluating the 'val' split instead.",
                    dataset_label,
                    str(exc),
                )
        return build_dataset("val", ns, resolution)

    def _resolve_batch_size(self) -> int:
        """Return the concrete training batch size.

        Raises:
            RuntimeError: If ``train_config.batch_size == "auto"`` was never resolved. Auto-batch sizing
                probes the live model and is only performed by ``RFDETR.train()``
                (:func:`rfdetr.training.auto_batch.resolve_auto_batch_config`); the PTL CLI path does not
                resolve it automatically.
        """
        batch_size = self.train_config.batch_size
        if batch_size == "auto":
            raise RuntimeError(
                "train_config.batch_size='auto' was not resolved before dataloader construction. "
                "Auto-batch sizing requires probing the live model and is only performed by "
                "RFDETR.train(); set batch_size to a concrete integer when using the PTL CLI path directly."
            )
        return batch_size

    @staticmethod
    def _require_dataset(dataset: torch.utils.data.Dataset[Any] | None, split: str) -> torch.utils.data.Dataset[Any]:
        """Return *dataset*, raising if ``setup()`` has not built it yet."""
        if dataset is None:
            raise RuntimeError(f"{split} dataset was not built; call setup({split!r}) before requesting a dataloader.")
        return dataset

    def build_train_sampler(self, dataset: torch.utils.data.Dataset[Any]) -> torch.utils.data.Sampler[list[int]] | None:
        """Return a custom training batch sampler, or ``None`` to use the default sampling strategy.

        Override this hook to plug in a batch sampler such as
        :class:`~rfdetr.datasets.multi_source.WeightedMultiSourceBatchSampler` without reimplementing
        :meth:`train_dataloader`. When this returns a sampler, ``train_dataloader`` passes it as the
        DataLoader's ``batch_sampler`` and keeps every other DataLoader kwarg from the base configuration
        (``num_workers``, ``pin_memory``, ``persistent_workers``, ``prefetch_factor``, ``worker_init_fn``,
        ``collate_fn``) — only the batching strategy changes. The default ``batch_size``/``shuffle``/
        ``drop_last``/small-dataset replacement-sampler logic and the :class:`GradAccumAlignedDataset`
        padding wrapper are all skipped in that case, since a batch sampler already fully owns which
        indices make up each batch and ``GradAccumAlignedDataset`` only pads a length that a custom batch
        sampler never consults.

        A returned sampler is responsible for its own DDP sharding (for example via ``num_replicas``/
        ``rank`` constructor arguments), so a distributed run must be started with
        ``Trainer(use_distributed_sampler=False)``. Otherwise Lightning tries to rebuild the batch sampler
        around its own ``DistributedSampler`` and fails with a bare ``TypeError`` about missing ``__init__``
        arguments; :meth:`train_dataloader` raises a ``RuntimeError`` naming the fix before that happens.

        Args:
            dataset: The training dataset built by :meth:`setup`, i.e. ``self._dataset_train``.

        Returns:
            A batch sampler yielding lists of dataset indices, or ``None`` to use the default strategy.

        Example:
            >>> # class MyDataModule(RFDETRDataModule):
            >>> #     def build_train_sampler(self, dataset):
            >>> #         return WeightedMultiSourceBatchSampler.from_concat_dataset(
            >>> #             dataset, weights=[0.6, 0.3, 0.1], batch_size=self.train_config.batch_size,
            >>> #         )
        """
        return None

    def _check_custom_sampler_owns_ddp(self, batch_sampler: torch.utils.data.Sampler[list[int]]) -> None:
        """Raise when Lightning would try to rebuild a custom batch sampler for distributed training.

        A batch sampler returned by :meth:`build_train_sampler` shards itself, so Lightning must be told to
        keep its hands off with ``Trainer(use_distributed_sampler=False)``. Without that flag Lightning
        re-instantiates the batch sampler with ``(sampler, batch_size, drop_last)`` to inject a
        ``DistributedSampler`` and surfaces only the resulting ``TypeError`` about missing ``__init__``
        arguments, which says nothing about the actual fix.

        Detection uses the public ``Trainer.distributed_sampler_kwargs`` property (a dict only for parallel
        strategies) together with the Lightning-internal accelerator connector. The internal flags are read
        defensively: when a Lightning version no longer exposes them, no error is raised and the user falls
        back to Lightning's own failure mode rather than a false alarm on a correct configuration.

        Args:
            batch_sampler: The sampler returned by :meth:`build_train_sampler`, used only for its class name.

        Raises:
            RuntimeError: If a distributed strategy is active and Lightning is still set to wrap samplers.
        """
        trainer = self.trainer
        if trainer is None:
            return
        if not isinstance(getattr(trainer, "distributed_sampler_kwargs", None), dict):
            return
        connector = getattr(trainer, "_accelerator_connector", None)
        if getattr(connector, "use_distributed_sampler", None) is not True:
            return
        if getattr(connector, "is_distributed", None) is not True:
            return
        raise RuntimeError(
            f"build_train_sampler() returned a custom batch sampler ({type(batch_sampler).__name__}), but the "
            "Trainer runs a distributed strategy with use_distributed_sampler=True. Lightning would try to "
            "rebuild that batch sampler around its own DistributedSampler and fail with a bare TypeError about "
            "missing __init__ arguments. Pass Trainer(use_distributed_sampler=False) and let the batch sampler "
            "shard itself (WeightedMultiSourceBatchSampler does this via num_replicas/rank, which default to the "
            "live torch.distributed process group)."
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training DataLoader.

        Consults :meth:`build_train_sampler` first: when it returns a sampler, that sampler is used as the
        DataLoader's ``batch_sampler`` and the logic below is skipped entirely (see that method's docstring
        for exactly which kwargs still apply). Otherwise, uses a replacement sampler when the dataset is too
        small to fill ``_MIN_TRAIN_BATCHES`` effective batches (matching legacy behaviour in ``main.py``).
        Otherwise wraps the dataset with :class:`GradAccumAlignedDataset` to ensure its length is an exact
        multiple of ``effective_batch_size * world_size`` (workaround for
        https://github.com/Lightning-AI/pytorch-lightning/issues/19987) and then uses ``shuffle=True, drop_last=True``
        so that PTL can auto-inject ``DistributedSampler`` in DDP mode.

        Returns:
            DataLoader for the training dataset.

        Raises:
            RuntimeError: If :meth:`build_train_sampler` returns a sampler while Lightning is still set to wrap
                samplers for a distributed strategy; see :meth:`_check_custom_sampler_owns_ddp`.
        """
        dataset: torch.utils.data.Dataset[Any] = self._require_dataset(self._dataset_train, "fit")
        num_workers = self._num_workers

        custom_batch_sampler = self.build_train_sampler(dataset)
        if custom_batch_sampler is not None:
            self._check_custom_sampler_owns_ddp(custom_batch_sampler)
            return DataLoader(
                dataset,
                batch_sampler=custom_batch_sampler,
                collate_fn=self._collate_fn,
                num_workers=num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
                worker_init_fn=_worker_init_fn,
            )

        batch_size = self._resolve_batch_size()
        effective_batch_size = batch_size * self.train_config.grad_accum_steps

        dataset_length = len(dataset)  # type: ignore[arg-type]
        if dataset_length < effective_batch_size * _MIN_TRAIN_BATCHES:
            logger.info(
                "Training with uniform sampler because dataset is too small: %d < %d",
                dataset_length,
                effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            sampler = torch.utils.data.RandomSampler(
                dataset,  # type: ignore[arg-type]
                replacement=True,
                num_samples=effective_batch_size * _MIN_TRAIN_BATCHES,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self._collate_fn,
                num_workers=num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers,
                prefetch_factor=self._prefetch_factor,
                worker_init_fn=_worker_init_fn,
            )

        # Pad the dataset to a multiple of effective_batch_size * world_size so
        # that drop_last=True below becomes a true no-op and PTL never fires the
        # optimizer on a partial accumulation window.
        # See https://github.com/Lightning-AI/pytorch-lightning/issues/19987
        world_size: int = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        aligned_dataset = GradAccumAlignedDataset(dataset, effective_batch_size, world_size)

        return DataLoader(
            aligned_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # no-op after alignment, but keeps intent explicit
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation DataLoader.

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        dataset = self._require_dataset(self._dataset_val, "validate")
        return DataLoader(
            dataset,
            batch_size=self._resolve_batch_size(),
            sampler=torch.utils.data.SequentialSampler(dataset),  # type: ignore[arg-type]
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test DataLoader.

        Returns:
            DataLoader for the test dataset with sequential sampling.
        """
        dataset = self._require_dataset(self._dataset_test, "test")
        return DataLoader(
            dataset,
            batch_size=self._resolve_batch_size(),
            sampler=torch.utils.data.SequentialSampler(dataset),  # type: ignore[arg-type]
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return the predict DataLoader (reuses the validation dataset, no augmentation).

        Returns:
            DataLoader for the validation dataset with sequential sampling.
        """
        dataset = self._require_dataset(self._dataset_val, "predict")
        return DataLoader(
            dataset,
            batch_size=self._resolve_batch_size(),
            sampler=torch.utils.data.SequentialSampler(dataset),  # type: ignore[arg-type]
            drop_last=False,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )

    def _show_samples(
        self,
        count: int,
        split: Literal["train", "val", "test"] = "train",
        *,
        columns: int = 3,
        figure_size: tuple[float, float] | None = None,
    ) -> "Figure":
        """Build a private diagnostic figure for transformed dataset samples.

        Samples the dataset after RF-DETR dataset transforms, so boxes and
        keypoints match the model input tensors rather than raw annotation JSON.

        Args:
            count: Maximum number of samples to render.
            split: Dataset split to visualize.
            columns: Number of subplot columns.
            figure_size: Optional Matplotlib figure size ``(width, height)`` in
                inches. When omitted, the size is derived from the grid shape.

        Returns:
            Matplotlib figure containing the annotated sample grid. When the
            dataset includes instance masks, they are rendered as coloured
            overlays before bounding boxes and labels.

        Raises:
            ValueError: If ``count`` or ``columns`` is not positive.

        Example:
            >>> # dm = RFDETRDataModule(model_config, train_config)
            >>> # figure = dm._show_samples(3, split="train")
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import supervision as sv
            import torchvision.transforms as T  # noqa: N812
        except ImportError as err:
            raise ImportError(
                "RFDETRDataModule._show_samples() requires visualization dependencies. "
                "Install them with `pip install 'rfdetr[visual]'`."
            ) from err

        from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy

        if count <= 0:
            raise ValueError(f"count must be positive, got {count}.")
        if columns <= 0:
            raise ValueError(f"columns must be positive, got {columns}.")
        if figure_size is not None:
            if len(figure_size) != 2:
                raise ValueError(f"figure_size must contain two values, got {figure_size}.")
            if figure_size[0] <= 0 or figure_size[1] <= 0:
                raise ValueError(f"figure_size values must be positive, got {figure_size}.")

        dataset = self._get_dataset_for_visualization(split)
        if dataset is None:
            raise RuntimeError(f"Could not build dataset split {split!r} for visualization.")

        inv_normalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        dataset_length = len(dataset)  # type: ignore[arg-type]
        rows = max(1, (min(count, dataset_length) + columns - 1) // columns)
        figure, axes = plt.subplots(rows, columns, figsize=figure_size or (5 * columns, 5 * rows))
        axes_array = np.asarray(axes, dtype=object).reshape(-1)
        for axis in axes_array:
            axis.axis("off")

        class_names = self.class_names
        for axis, sample_index in zip(axes_array, range(min(count, dataset_length)), strict=False):
            image_tensor, target = dataset[sample_index]
            image_path = self._source_image_path(dataset, sample_index)
            image = inv_normalize(image_tensor)
            image_array = image.detach().cpu().numpy()
            scene = np.ascontiguousarray((np.clip(image_array.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8))

            size = target.get("size")
            if isinstance(size, torch.Tensor):
                height, width = int(size[0]), int(size[1])
            else:
                height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])

            boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            labels = target.get("labels", torch.zeros((0,), dtype=torch.int64))
            if boxes.numel() > 0:
                scale = torch.tensor([width, height, width, height], dtype=torch.float32)
                xyxy = box_cxcywh_to_xyxy(boxes.detach().cpu()) * scale
                class_ids = labels.detach().cpu().numpy().astype(int)
                masks_tensor = target.get("masks")
                mask_array = (
                    masks_tensor.detach().cpu().numpy()
                    if masks_tensor is not None and masks_tensor.numel() > 0
                    else None
                )
                detections = sv.Detections(xyxy=xyxy.numpy().astype(np.float32), class_id=class_ids, mask=mask_array)
                labels_text = [
                    class_names[class_id] if class_names is not None and class_id < len(class_names) else str(class_id)
                    for class_id in class_ids
                ]
                if mask_array is not None:
                    scene = sv.MaskAnnotator().annotate(scene=scene, detections=detections)
                scene = sv.BoxAnnotator(thickness=1).annotate(scene=scene, detections=detections)
                # LabelAnnotator.annotate() is typed as PIL-only, but its @ensure_cv2_image_for_class_method
                # decorator accepts and returns ndarray at runtime too (see its docstring).
                label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=2)
                scene = cast(
                    np.ndarray[Any, np.dtype[Any]],
                    label_annotator.annotate(
                        scene=cast(Any, scene),
                        detections=detections,
                        labels=labels_text,
                    ),
                )

            keypoints = target.get("keypoints")
            if keypoints is not None and keypoints.numel() > 0:
                keypoints_array = keypoints.detach().cpu().numpy().astype(np.float32)
                keypoint_xy = keypoints_array[..., :2] * np.asarray([width, height], dtype=np.float32)
                keypoint_visibility = keypoints_array[..., 2] > 0
                key_points = sv.KeyPoints(
                    xy=keypoint_xy,
                    keypoint_confidence=keypoint_visibility.astype(np.float32),
                    class_id=labels.detach().cpu().numpy().astype(int),
                    visible=keypoint_visibility,
                    data={"visible": keypoint_visibility},
                )
                scene = sv.VertexAnnotator(radius=3).annotate(scene=scene, key_points=key_points)

            axis.imshow(scene)
            title = image_path.name if image_path is not None else f"{split}[{sample_index}]"
            axis.set_title(self._ellipsize_sample_title(title), fontsize=10)
            axis.axis("off")

        figure.tight_layout()
        return figure

    @staticmethod
    def _ellipsize_sample_title(title: str, max_length: int = 48) -> str:
        """Shorten long sample titles so subplot grids do not overflow.

        Args:
            title: Raw title text, usually an image file name.
            max_length: Maximum returned character count including the ellipsis.

        Returns:
            Original title when it already fits, otherwise a middle-ellipsized
            string that preserves the start and file suffix.
        """
        if len(title) <= max_length:
            return title
        if max_length <= 3:
            return "." * max_length
        keep_left = max(1, (max_length - 3) // 2)
        keep_right = max_length - 3 - keep_left
        return f"{title[:keep_left]}...{title[-keep_right:]}"

    def _get_dataset_for_visualization(
        self,
        split: Literal["train", "val", "test"],
    ) -> torch.utils.data.Dataset[Any] | None:
        """Return a built dataset split for private visualization."""
        if split == "train":
            self.setup("fit")
            return self._dataset_train
        if split == "val":
            self.setup("validate")
            return self._dataset_val
        if split == "test":
            self.setup("test")
            return self._dataset_test
        raise ValueError(f"Unsupported split {split!r}.")

    @staticmethod
    def _source_image_path(dataset: torch.utils.data.Dataset[Any], sample_index: int) -> Path | None:
        """Return a source image path for common COCO-style datasets."""
        image_folder = getattr(dataset, "root", None)
        image_ids = getattr(dataset, "ids", None)
        coco = getattr(dataset, "coco", None)
        if image_folder is None or image_ids is None or coco is None:
            return None
        image_id = image_ids[sample_index]
        image_info = coco.loadImgs(image_id)[0]
        return Path(image_folder) / image_info["file_name"]  # type: ignore[no-any-return]

    def _setup_kornia_pipeline(self) -> None:
        """Build the Kornia pipeline for ``self._resolved_augmentation_backend``.

        ``TV`` and ``ALBU`` are no-ops.  ``KORNIA`` validates that kornia is installed then builds the pipeline on
        whatever device the batch arrives on.

        ``self._resolved_augmentation_backend`` (set by :meth:`setup` just before this call) is the concrete backend
        returned by :func:`rfdetr.datasets.kornia_transforms.resolve_augmentation_backend`.
        """
        from rfdetr.datasets.kornia_transforms import is_gpu_postprocess

        resolved = self._resolved_augmentation_backend
        if resolved is None or not is_gpu_postprocess(resolved):
            return

        if not AugmentationBackend._is_kornia_available():
            raise ImportError("Kornia augmentation requires kornia. Install with: pip install 'rfdetr[augment]'")

        from rfdetr.datasets.kornia_transforms import build_kornia_pipeline, build_normalize

        self._kornia_pipeline = build_kornia_pipeline(
            self.train_config.aug_config if self.train_config.aug_config is not None else AUG_CONFIG,
            self.model_config.resolution,
            # The padding mask must receive every geometric warp, even for detection-only batches.
            with_masks=True,
        )
        self._kornia_normalize = build_normalize()
        logger.info("Kornia augmentation pipeline built (resolved=%s)", resolved)

    def on_after_batch_transfer(self, batch: tuple[Any, Any], dataloader_idx: int) -> tuple[Any, Any]:
        """Apply Kornia GPU augmentation after the batch is transferred to device.

        When ``_kornia_pipeline`` is set and the trainer is in training mode, augmentation and normalization are applied
        on the GPU.  Validation and test batches pass through unchanged.

        The pipeline carries the ``NestedTensor`` padding mask with every batch, so geometric transforms keep valid
        image regions and padding aligned. Segmentation batches concatenate their instance masks before that final
        padding-mask channel, then split it back out before target unpacking.

        Args:
            batch: Tuple of ``(NestedTensor, list[dict])`` already on device.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The (possibly augmented) batch.
        """
        kornia_pipeline = self._kornia_pipeline
        kornia_normalize = self._kornia_normalize
        if self.trainer is None or not self.trainer.training or kornia_pipeline is None or kornia_normalize is None:
            return batch

        from rfdetr.datasets.kornia_transforms import collate_boxes, collate_masks, unpack_boxes
        from rfdetr.utilities.tensors import NestedTensor

        samples, targets = batch
        img = samples.tensors  # [B, C, H, W]
        # Move Kornia modules to the batch device (no-op if already there).
        # nn.Module.to() is in-place; no reassignment needed.
        kornia_pipeline.to(img.device)
        kornia_normalize.to(img.device)
        boxes_padded, valid = collate_boxes(targets, img.device)
        padding_mask = samples.mask
        if padding_mask is None:
            padding_mask = torch.zeros(img.shape[0], *img.shape[-2:], dtype=torch.bool, device=img.device)
        padding_masks = padding_mask.unsqueeze(1).to(torch.float32)

        if self.model_config.segmentation_head:
            image_height, image_width = img.shape[-2:]
            masks_padded = collate_masks(
                targets, img.device, n_max=valid.shape[1], image_height=image_height, image_width=image_width
            )
            auxiliary_masks = torch.cat((masks_padded, padding_masks), dim=1)
            img_aug, boxes_aug, auxiliary_masks_aug = kornia_pipeline(img, boxes_padded, auxiliary_masks)
            masks_aug = auxiliary_masks_aug[:, : valid.shape[1]]
            padding_mask_aug = auxiliary_masks_aug[:, valid.shape[1]] > 0.5
            img_aug = kornia_normalize(img_aug)
            aug_height, aug_width = img_aug.shape[-2:]
            targets = unpack_boxes(boxes_aug, valid, targets, aug_height, aug_width, masks_aug=masks_aug)
        else:
            img_aug, boxes_aug, padding_masks_aug = kornia_pipeline(img, boxes_padded, padding_masks)
            padding_mask_aug = padding_masks_aug[:, 0] > 0.5
            img_aug = kornia_normalize(img_aug)
            aug_height, aug_width = img_aug.shape[-2:]
            targets = unpack_boxes(boxes_aug, valid, targets, aug_height, aug_width)

        height, width = img_aug.shape[-2:]
        for target in targets:
            boxes = target["boxes"]
            if boxes.numel() == 0:
                continue
            scale = boxes.new_tensor([width, height, width, height])
            target["boxes"] = box_xyxy_to_cxcywh(boxes) / scale
        batch = (NestedTensor(img_aug, padding_mask_aug), targets)
        return batch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str] | None:
        """Class names from the training or validation dataset annotation file.

        Reads category names from the first available COCO-style dataset. Returns ``None`` if no dataset has been set up
        yet or the dataset does not expose COCO-style category information.

        Returns:
            Sorted list of class name strings, or ``None``.
        """
        for dataset in (self._dataset_train, self._dataset_val, self._dataset_test):
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                label2cat = getattr(dataset, "label2cat", None)
                if label2cat is None:
                    label2cat = getattr(coco, "label2cat", None)
                if isinstance(label2cat, dict) and label2cat:
                    max_label = max(label2cat)
                    names = [""] * (max_label + 1)
                    for label, category_id in sorted(label2cat.items()):
                        category = coco.cats.get(category_id)
                        if category is not None:
                            names[label] = category["name"]
                    return names
                return [coco.cats[k]["name"] for k in sorted(coco.cats.keys())]
        return None

    def transfer_batch_to_device(
        self, batch: tuple[Any, Any], device: torch.device, dataloader_idx: int
    ) -> tuple[Any, Any]:
        """Move a ``(NestedTensor, targets)`` batch to *device*.

        PTL's default iterates tuple elements and calls ``.to(device)``; that works for plain tensors but
        ``NestedTensor`` must be moved explicitly.

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
