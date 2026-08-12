# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Weighted batch sampling across several datasets.

Training on a mix of datasets (for example hand-labelled, synthetic, and public data) with a plain
:class:`~torch.utils.data.ConcatDataset` samples each source in proportion to its size, so a large public set dominates
every batch. :class:`WeightedMultiSourceBatchSampler` instead fixes the per-source composition of *every* batch, which
keeps the gradient signal from a small high-quality source from being drowned out.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import pairwise
from math import ceil
from typing import Any, Literal

import torch
from torch.utils.data import ConcatDataset, Sampler

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# Weights are converted to integers on this scale so the largest-remainder allocation below is exact. Float remainders
# would make ties resolve by rounding error (e.g. ``0.6 * 16 - 9`` compares as slightly less than ``0.1 * 16 - 1``).
_WEIGHT_SCALE = 1_000_000


def compute_source_batch_sizes(batch_size: int, weights: Sequence[float]) -> list[int]:
    """Split a batch across sources in proportion to ``weights``.

    Uses the largest-remainder (Hamilton) method, so the returned counts always sum to ``batch_size`` while staying as
    close as possible to the requested ratios. Every source receives at least one slot when ``batch_size`` is at least
    the number of sources; otherwise the lowest-weighted sources receive zero and are absent from each batch.

    Args:
        batch_size: Number of samples in one batch.
        weights: Relative sampling weight per source. Weights need not sum to 1; they are normalised internally. All
            weights must be strictly positive.

    Returns:
        Per-source sample counts, in the same order as ``weights``, summing to ``batch_size``.

    Raises:
        ValueError: If ``batch_size`` is not positive, ``weights`` is empty, or any weight is not strictly positive.

    Example:
        >>> compute_source_batch_sizes(16, [0.6, 0.3, 0.1])
        [10, 5, 1]
        >>> compute_source_batch_sizes(8, [0.9, 0.04, 0.03, 0.03])  # the minimum of one slot each costs the leader
        [5, 1, 1, 1]
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(weights) == 0:
        raise ValueError("weights must contain at least one source")
    non_positive = [(index, weight) for index, weight in enumerate(weights) if not weight > 0]
    if non_positive:
        raise ValueError(
            f"weights must be strictly positive, got non-positive entries at (index, value): {non_positive}. "
            "To exclude a source, drop it from the sampler instead of giving it zero weight."
        )

    scaled = [round(float(weight) * _WEIGHT_SCALE) for weight in weights]
    total = sum(scaled)
    if total <= 0:
        raise ValueError(f"weights are too small to be represented at 1e-6 precision: {list(weights)}")

    counts: list[int] = []
    remainders: list[int] = []
    for weight in scaled:
        numerator = batch_size * weight
        counts.append(numerator // total)
        remainders.append(numerator % total)

    if batch_size >= len(weights):
        # Guarantee representation: a source rounded down to zero would never appear in a batch.
        counts = [max(count, 1) for count in counts]

    shortfall = batch_size - sum(counts)
    if shortfall > 0:
        # Hand out the leftover slots to the largest remainders first.
        order = sorted(range(len(weights)), key=lambda index: (-remainders[index], -scaled[index], index))
        for index in order[:shortfall]:
            counts[index] += 1
    elif shortfall < 0:
        # The guarantee above over-allocated; reclaim from the smallest remainders, never dropping a source to zero. A
        # single pass frees at most one slot per source, which is not always enough (a dominant weight can be clamped
        # up by more slots than there are other sources), so keep passing until the counts balance. Every pass either
        # reaches the target or shrinks a count towards 1, and ``batch_size >= len(weights)`` whenever the clamp fires,
        # so enough slack always exists; the empty-``reclaimable`` guard only keeps the loop bounded.
        order = sorted(range(len(weights)), key=lambda index: (remainders[index], scaled[index], index))
        while shortfall < 0:
            reclaimable = [index for index in order if counts[index] > 1]
            if not reclaimable:
                break
            for index in reclaimable:
                if shortfall == 0:
                    break
                counts[index] -= 1
                shortfall += 1

    assert sum(counts) == batch_size, f"allocation {counts} does not sum to batch_size={batch_size}"
    return counts


class WeightedMultiSourceBatchSampler(Sampler[list[int]]):
    """Batch sampler that enforces a fixed per-source ratio in every batch.

    Each yielded batch is a flat list of indices into a :class:`~torch.utils.data.ConcatDataset` whose sub-datasets are
    the sources, composed of :func:`compute_source_batch_sizes` slots per source. Indices are shuffled within a source
    every epoch; a source whose samples run out mid-epoch is reshuffled and reused, so the ratio holds even when the
    sources differ wildly in size.

    Note:
        Unlike torch's ``drop_last``, this sampler never emits a short final batch. Setting ``drop_last=False`` appends
        one extra full batch, recycling source samples, after the driving source has been covered once.

    Under DDP each rank consumes a disjoint stride of the global batch stream (``rank``, ``rank + num_replicas``, ...),
    mirroring :class:`~torch.utils.data.distributed.DistributedSampler`. The number of global batches is truncated to a
    multiple of ``num_replicas`` so every rank runs the same number of steps and no rank stalls in gradient all-reduce.

    Like :class:`~torch.utils.data.distributed.DistributedSampler`, :meth:`set_epoch` must be called at the start of
    each epoch, otherwise every epoch reuses the same shuffle.

    .. note::
        Every rank is assumed to build the sampler over the same sources: the same number of sources, the same size
        per source, in the same order. Ranks exchange nothing; each one replays the whole global batch stream from
        ``source_sizes``, ``weights``, ``seed`` and the epoch, and yields only its own stride of it. A rank that
        disagrees about the source layout therefore builds a *different* stream rather than a disjoint share of one
        stream, and the mismatch is silent — no error, no hang. Deriving the sources from an unsorted directory
        listing, a rank-dependent subsample, or a size limit applied only on rank 0 breaks this assumption.

    Args:
        source_sizes: Number of samples in each source, in ``ConcatDataset`` order. Every source must be non-empty.
        weights: Relative sampling weight per source, in the same order. Normalised internally; must be positive.
        batch_size: Number of samples per batch on a single rank. This is the mini-batch the model sees per forward
            pass, so it is neither multiplied by gradient-accumulation steps nor by the world size.
        drop_last: Whether to omit the extra full batch needed to cover the driving source's remainder. At least one
            batch is always produced (even when the driving source is smaller than its per-batch slot count), so
            ``drop_last=True`` will not reduce an epoch to zero batches for tiny datasets.
        shuffle: Whether to shuffle indices within each source.
        num_replicas: Number of DDP processes participating.
        rank: Rank of the current process, in ``[0, num_replicas)``.
        seed: Base seed shared by all ranks; combined with the epoch to shuffle identically across ranks.
        epoch_length: Which source defines the length of an epoch — ``"largest"`` (every sample of the biggest source
            is seen roughly once per epoch, smaller sources cycle), ``"smallest"`` (the smallest source is seen once
            and larger sources are sub-sampled), or the integer index of a specific source.

    Raises:
        ValueError: If the arguments are inconsistent, a source is empty, or ``epoch_length`` is invalid.

    Example:
        >>> import torch
        >>> from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
        >>> sources = [TensorDataset(torch.zeros(n)) for n in (500, 200, 40)]
        >>> dataset = ConcatDataset(sources)
        >>> sampler = WeightedMultiSourceBatchSampler.from_concat_dataset(dataset, [0.6, 0.3, 0.1], batch_size=16)
        >>> sampler.source_batch_sizes
        [10, 5, 1]
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        source_sizes: Sequence[int],
        weights: Sequence[float],
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        epoch_length: Literal["largest", "smallest"] | int = "largest",
    ) -> None:
        if len(source_sizes) != len(weights):
            raise ValueError(
                f"source_sizes and weights must have the same length, got {len(source_sizes)} and {len(weights)}"
            )
        if len(source_sizes) == 0:
            raise ValueError("at least one source is required")
        empty = [index for index, size in enumerate(source_sizes) if size <= 0]
        if empty:
            raise ValueError(f"every source must be non-empty, got empty sources at indices {empty}")
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.source_sizes = [int(size) for size in source_sizes]
        self.weights = [float(weight) for weight in weights]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        self.source_batch_sizes = compute_source_batch_sizes(self.batch_size, self.weights)
        starved = [index for index, count in enumerate(self.source_batch_sizes) if count == 0]
        if starved:
            logger.warning(
                "batch_size=%d is too small to represent every source; sources %s contribute no samples.",
                self.batch_size,
                starved,
            )

        offsets = [0]
        for size in self.source_sizes:
            offsets.append(offsets[-1] + size)
        self._source_offsets = offsets

        self.driving_source = self._resolve_driving_source(epoch_length)
        driving_slots = self.source_batch_sizes[self.driving_source]
        global_batches = self.source_sizes[self.driving_source] // driving_slots
        if not self.drop_last and self.source_sizes[self.driving_source] % driving_slots:
            global_batches += 1
        # Truncate to a whole number of rounds so every rank yields the same count; at least one batch is always
        # produced, even for datasets smaller than a single batch.
        self._batches_per_rank = max(1, global_batches // self.num_replicas)
        self._global_batches = self._batches_per_rank * self.num_replicas

        self._warn_on_ratio_divergence()
        self._warn_on_source_imbalance()

    @classmethod
    def from_concat_dataset(
        cls,
        dataset: ConcatDataset[Any],
        weights: Sequence[float],
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        epoch_length: Literal["largest", "smallest"] | int = "largest",
    ) -> "WeightedMultiSourceBatchSampler":
        """Build a sampler for a :class:`~torch.utils.data.ConcatDataset`.

        .. note::
            Under DDP every rank must pass an identically laid-out ``dataset``: the same sub-datasets, in the same
            order, with the same lengths (see the cross-rank assumption in the class docstring). Only that layout is
            read here, so sub-datasets that agree in length but differ in content across ranks stay undetected and
            silently break the split of the batch stream.

        Args:
            dataset: Concatenated dataset whose sub-datasets are the sources.
            weights: Relative sampling weight per source, ordered like ``dataset.datasets``.
            batch_size: Number of samples per batch on a single rank.
            drop_last: Whether to drop the final partial batch of the driving source.
            shuffle: Whether to shuffle indices within each source.
            num_replicas: Number of DDP processes participating.
            rank: Rank of the current process, in ``[0, num_replicas)``.
            seed: Base seed shared by all ranks.
            epoch_length: Source that defines the epoch length; see the class docstring.

        Returns:
            A sampler whose index space matches ``dataset``.

        Raises:
            ValueError: If ``weights`` does not have one entry per sub-dataset.
        """
        if len(weights) != len(dataset.datasets):
            raise ValueError(
                f"weights must have one entry per sub-dataset, got {len(weights)} for {len(dataset.datasets)} sources"
            )
        boundaries = [0, *dataset.cumulative_sizes]
        source_sizes = [end - start for start, end in pairwise(boundaries)]
        return cls(
            source_sizes,
            weights,
            batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            epoch_length=epoch_length,
        )

    def _resolve_driving_source(self, epoch_length: Literal["largest", "smallest"] | int) -> int:
        """Return the index of the source whose exhaustion ends an epoch.

        Args:
            epoch_length: ``"largest"``, ``"smallest"``, or an explicit source index.

        Returns:
            Index of the driving source, adjusted when the requested source has no slots in a batch.

        Raises:
            ValueError: If ``epoch_length`` is neither a known keyword nor a valid source index.
        """
        indices = range(len(self.source_sizes))
        if epoch_length == "largest":
            driving = max(indices, key=lambda index: self.source_sizes[index])
        elif epoch_length == "smallest":
            driving = min(indices, key=lambda index: self.source_sizes[index])
        elif isinstance(epoch_length, int) and not isinstance(epoch_length, bool) and epoch_length in indices:
            driving = epoch_length
        else:
            raise ValueError(
                f"epoch_length must be 'largest', 'smallest', or a source index in [0, {len(self.source_sizes)}), "
                f"got {epoch_length!r}"
            )

        if self.source_batch_sizes[driving] == 0:
            fallback = next(index for index, count in enumerate(self.source_batch_sizes) if count > 0)
            logger.warning(
                "Source %d contributes no samples per batch and cannot define the epoch length; using source %d.",
                driving,
                fallback,
            )
            return fallback
        return driving

    def _warn_on_ratio_divergence(self) -> None:
        """Warn when a source's realised share of every batch departs sharply from its requested weight.

        Guaranteeing every source at least one slot (see :func:`compute_source_batch_sizes`) lifts tiny sources above
        their requested share and squeezes the dominant one, so the composition the caller asked for can be rewritten
        without any other signal. A divergence is only reported when it is large both relatively and absolutely, which
        keeps the unavoidable ``1 / batch_size`` granularity quiet for weights that are merely small. Starved sources
        (zero slots per batch) are skipped because the constructor already warns about them.
        """
        relative_tolerance = 2.0
        absolute_tolerance = 0.10
        total_weight = sum(self.weights)

        divergent: list[tuple[int, float, float]] = []
        for index, count in enumerate(self.source_batch_sizes):
            if count == 0:
                continue
            requested = self.weights[index] / total_weight
            realised = count / self.batch_size
            larger, smaller = max(requested, realised), min(requested, realised)
            if larger - smaller > absolute_tolerance and larger > relative_tolerance * smaller:
                divergent.append((index, requested, realised))
        if not divergent:
            return

        logger.warning(
            "batch_size=%d cannot hold the requested source weights: %s. Every source is guaranteed one slot per "
            "batch, which rewrites the requested ratio; a batch_size of at least %d would leave room for the "
            "smallest weight.",
            self.batch_size,
            "; ".join(
                f"source {index} requested {requested:.1%} but gets {realised:.1%}"
                for index, requested, realised in divergent
            ),
            ceil(total_weight / min(self.weights)),
        )

    def _warn_on_source_imbalance(self) -> None:
        """Warn when a source is recycled many times per epoch, which risks overfitting it.

        Only sources that actually contribute samples are considered. A starved source (zero slots per batch, possible
        when ``batch_size`` is below the number of sources) is never recycled, so including it would hide a genuinely
        over-recycled contributor behind a pass count of zero.
        """
        recycle_threshold = 10
        passes = {
            index: (self._global_batches * count) / self.source_sizes[index]
            for index, count in enumerate(self.source_batch_sizes)
            if count > 0
        }
        most_recycled = max(passes, key=lambda index: passes[index])
        if passes[most_recycled] < recycle_threshold:
            return
        logger.warning(
            "Source %d (%d samples) is repeated ~%.1f times per epoch to fill its %d slot(s) in every batch, "
            "which risks overfitting it. Consider epoch_length='smallest' or early stopping.",
            most_recycled,
            self.source_sizes[most_recycled],
            passes[most_recycled],
            self.source_batch_sizes[most_recycled],
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed shuffling.

        Call this before every epoch so the within-source shuffle changes, and changes identically on all ranks.

        Args:
            epoch: Zero-based epoch number.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Return the number of batches this rank yields per epoch."""
        return self._batches_per_rank

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of dataset indices honouring the per-source ratio.

        Yields:
            Lists of exactly ``batch_size`` indices into the concatenated dataset.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        streams = [self._shuffled_indices(index, generator) for index in range(len(self.source_sizes))]
        cursors = [0] * len(self.source_sizes)

        def draw(source: int, count: int) -> list[int]:
            picked: list[int] = []
            stream = streams[source]
            while len(picked) < count:
                if cursors[source] >= len(stream):
                    # The source ran out before the epoch ended; reshuffle and keep going so the ratio is preserved.
                    stream = self._shuffled_indices(source, generator)
                    streams[source] = stream
                    cursors[source] = 0
                take = min(count - len(picked), len(stream) - cursors[source])
                picked.extend(stream[cursors[source] : cursors[source] + take])
                cursors[source] += take
            return picked

        for batch_index in range(self._global_batches):
            # Every rank builds every batch so the shared RNG stream stays aligned across ranks, and each rank yields
            # only its own stride of the global sequence.
            batch = [
                index for source, count in enumerate(self.source_batch_sizes) if count for index in draw(source, count)
            ]
            if self.shuffle:
                order = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[position] for position in order]
            if batch_index % self.num_replicas == self.rank:
                yield batch

    def _shuffled_indices(self, source: int, generator: torch.Generator) -> list[int]:
        """Return the source's indices in the concatenated index space, shuffled when ``shuffle`` is set.

        Args:
            source: Index of the source.
            generator: RNG shared by all ranks for this epoch.

        Returns:
            One full pass over the source's global indices.
        """
        offset = self._source_offsets[source]
        size = self.source_sizes[source]
        if not self.shuffle:
            return list(range(offset, offset + size))
        return (torch.randperm(size, generator=generator) + offset).tolist()
