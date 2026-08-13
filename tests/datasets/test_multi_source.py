# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for weighted multi-source batch sampling."""

import logging
from collections import Counter
from contextlib import contextmanager
from typing import Iterator

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from rfdetr.datasets.multi_source import WeightedMultiSourceBatchSampler, compute_source_batch_sizes
from rfdetr.utilities.logger import get_logger


@contextmanager
def _capture_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Capture ``rf-detr`` warnings, which the shared logger does not propagate by default.

    Examples:
        Needs pytest's ``caplog`` fixture, so the live call is covered by tests rather than doctest.

        >>> callable(_capture_warnings)  # doctest: +SKIP
        True
    """
    rf_logger = get_logger()
    previous = rf_logger.propagate
    rf_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            yield
    finally:
        rf_logger.propagate = previous


def _recycling_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the emitted over-recycling warnings.

    Examples:
        Needs pytest's ``caplog`` fixture, so the live call is covered by tests rather than doctest.

        >>> callable(_recycling_warnings)  # doctest: +SKIP
        True
    """
    return [record.getMessage() for record in caplog.records if "is repeated" in record.getMessage()]


def _ratio_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the emitted requested-versus-realised ratio warnings.

    Examples:
        Needs pytest's ``caplog`` fixture, so the live call is covered by tests rather than doctest.

        >>> callable(_ratio_warnings)  # doctest: +SKIP
        True
    """
    return [record.getMessage() for record in caplog.records if "cannot hold" in record.getMessage()]


def _source_of(index: int, source_sizes: list[int]) -> int:
    """Return the source a concatenated-dataset index belongs to.

    Examples:
        >>> _source_of(7, [5, 3])
        1
    """
    for source, size in enumerate(source_sizes):
        if index < size:
            return source
        index -= size
    raise AssertionError(f"index {index} is out of range for sources {source_sizes}")


def _batch_composition(batch: list[int], source_sizes: list[int]) -> list[int]:
    """Return the number of samples drawn from each source in a batch.

    Examples:
        >>> _batch_composition([0, 1, 6], [5, 3])
        [2, 1]
    """
    counts = Counter(_source_of(index, source_sizes) for index in batch)
    return [counts[source] for source in range(len(source_sizes))]


# Weight sets skewed enough that guaranteeing every source one slot over-allocates the batch, which is where the
# largest-remainder allocation has to reclaim slots again.
_SKEWED_WEIGHT_SETS = {
    "dominant_of_two": [0.99, 0.01],
    "dominant_of_four": [0.9, 0.04, 0.03, 0.03],
    "long_tail_of_five": [0.8, 0.1, 0.05, 0.03, 0.02],
    "thirds_of_three": [1 / 3, 1 / 3, 1 / 3],
    "unnormalised_of_three": [97.0, 2.0, 1.0],
}
_SUM_INVARIANT_BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8, 16, 64)
_SUM_INVARIANT_CASES = [
    pytest.param(batch_size, weights, id=f"{name}-batch{batch_size}")
    for name, weights in _SKEWED_WEIGHT_SETS.items()
    for batch_size in _SUM_INVARIANT_BATCH_SIZES
]
# Below batch_size >= len(weights), a source is legitimately starved (zero slots; see TestRecyclingWarning), so
# min(counts) >= 1 only holds once every source is guaranteed a slot.
_MIN_COUNT_CASES = [
    pytest.param(batch_size, weights, id=f"{name}-batch{batch_size}")
    for name, weights in _SKEWED_WEIGHT_SETS.items()
    for batch_size in _SUM_INVARIANT_BATCH_SIZES
    if batch_size >= len(weights)
]


class TestComputeSourceBatchSizes:
    """Allocation of batch slots across weighted sources."""

    @pytest.mark.parametrize(
        ("batch_size", "weights", "expected"),
        [
            pytest.param(16, [0.6, 0.3, 0.1], [10, 5, 1], id="documented_example"),
            pytest.param(8, [0.5, 0.5], [4, 4], id="even_split"),
            pytest.param(9, [1 / 3, 1 / 3, 1 / 3], [3, 3, 3], id="thirds_divide_evenly"),
            pytest.param(10, [1 / 3, 1 / 3, 1 / 3], [4, 3, 3], id="thirds_with_remainder"),
            pytest.param(4, [0.97, 0.02, 0.01], [2, 1, 1], id="tiny_weights_still_represented"),
            pytest.param(2, [0.6, 0.3, 0.1], [1, 1, 0], id="batch_smaller_than_source_count"),
            # The dominant source is clamped up by two slots at once, so one reclaim pass per source is not enough.
            pytest.param(8, [0.9, 0.04, 0.03, 0.03], [5, 1, 1, 1], id="dominant_weight_gives_back_several_slots"),
            pytest.param(4, [0.97, 0.01, 0.01, 0.01], [1, 1, 1, 1], id="minimum_slots_consume_the_whole_batch"),
        ],
    )
    def test_allocation(self, batch_size: int, weights: list[float], expected: list[int]) -> None:
        assert compute_source_batch_sizes(batch_size, weights) == expected

    def test_counts_sum_to_batch_size(self) -> None:
        assert sum(compute_source_batch_sizes(37, [0.55, 0.25, 0.15, 0.05])) == 37

    @pytest.mark.parametrize(("batch_size", "weights"), _SUM_INVARIANT_CASES)
    def test_sum_invariant_holds_for_skewed_weights(self, batch_size: int, weights: list[float]) -> None:
        """Skewed weights across many batch sizes never allocate more or fewer slots than the batch holds."""
        assert sum(compute_source_batch_sizes(batch_size, weights)) == batch_size

    @pytest.mark.parametrize(("batch_size", "weights"), _MIN_COUNT_CASES)
    def test_min_count_at_least_one_when_batch_covers_every_source(self, batch_size: int, weights: list[float]) -> None:
        """Every source keeps at least one slot across skewed weights once batch_size >= the number of sources."""
        assert min(compute_source_batch_sizes(batch_size, weights)) >= 1

    def test_weights_need_not_be_normalised(self) -> None:
        assert compute_source_batch_sizes(16, [6, 3, 1]) == compute_source_batch_sizes(16, [0.6, 0.3, 0.1])

    def test_reclaims_only_the_over_allocated_slots(self) -> None:
        # Guaranteeing one slot each pushes the total to 6 for a batch of 5, so exactly one slot is reclaimed from a
        # source that has more than one.
        counts = compute_source_batch_sizes(5, [0.4, 0.4, 0.1, 0.1])
        assert sum(counts) == 5
        assert min(counts) >= 1

    @pytest.mark.parametrize(
        "weights",
        [
            pytest.param([0.5, 0.5, 0.0], id="zero_weight"),
            pytest.param([0.5, 0.5, -0.1], id="negative_weight"),
            pytest.param([0.5, 0.5, float("nan")], id="nan_weight"),
            pytest.param([0.5, 0.5, float("inf")], id="infinite_weight"),
        ],
    )
    def test_rejects_non_positive_weight(self, weights: list[float]) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            compute_source_batch_sizes(8, weights)

    def test_rejects_empty_weights(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            compute_source_batch_sizes(8, [])

    def test_rejects_weights_below_representable_precision(self) -> None:
        with pytest.raises(ValueError, match="too small to be represented"):
            compute_source_batch_sizes(8, [1e-9, 1e-9])

    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            compute_source_batch_sizes(0, [0.5, 0.5])


class TestSamplerValidation:
    """Constructor argument checking."""

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            WeightedMultiSourceBatchSampler([100, 50], [0.5, 0.3, 0.2], batch_size=8)

    def test_rejects_empty_source(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            WeightedMultiSourceBatchSampler([100, 0], [0.5, 0.5], batch_size=8)

    def test_rejects_missing_sources(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            WeightedMultiSourceBatchSampler([], [], batch_size=8)

    def test_rejects_non_positive_num_replicas(self) -> None:
        with pytest.raises(ValueError, match="num_replicas must be >= 1"):
            WeightedMultiSourceBatchSampler([100, 50], [0.5, 0.5], batch_size=8, num_replicas=0)

    def test_rejects_rank_outside_world(self) -> None:
        with pytest.raises(ValueError, match="rank must be in"):
            WeightedMultiSourceBatchSampler([100, 50], [0.5, 0.5], batch_size=8, num_replicas=2, rank=2)

    @pytest.mark.parametrize(
        "epoch_length",
        [
            pytest.param("biggest", id="unknown_keyword"),
            # bool is an int subclass; without an explicit exclusion True/False would silently alias source 1/0.
            pytest.param(True, id="bool_true"),
            pytest.param(False, id="bool_false"),
        ],
    )
    def test_rejects_unknown_epoch_length(self, epoch_length: object) -> None:
        with pytest.raises(ValueError, match="epoch_length"):
            WeightedMultiSourceBatchSampler([100, 50], [0.5, 0.5], batch_size=8, epoch_length=epoch_length)

    def test_rejects_out_of_range_source_index(self) -> None:
        with pytest.raises(ValueError, match="epoch_length"):
            WeightedMultiSourceBatchSampler([100, 50], [0.5, 0.5], batch_size=8, epoch_length=2)


class TestBatchComposition:
    """Per-batch source ratios."""

    def test_every_batch_matches_requested_ratio(self) -> None:
        source_sizes = [500, 200, 40]
        sampler = WeightedMultiSourceBatchSampler(source_sizes, [0.6, 0.3, 0.1], batch_size=16)
        compositions = {tuple(_batch_composition(batch, source_sizes)) for batch in sampler}
        assert compositions == {(10, 5, 1)}

    def test_every_batch_has_full_batch_size(self) -> None:
        sampler = WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16)
        assert {len(batch) for batch in sampler} == {16}

    def test_indices_stay_inside_their_source(self) -> None:
        source_sizes = [50, 30]
        sampler = WeightedMultiSourceBatchSampler(source_sizes, [0.5, 0.5], batch_size=10)
        assert all(0 <= index < sum(source_sizes) for batch in sampler for index in batch)

    def test_small_source_is_recycled_within_an_epoch(self) -> None:
        # Source 1 holds 5 samples but must supply 5 per batch across many batches, so it has to repeat.
        source_sizes = [500, 5]
        sampler = WeightedMultiSourceBatchSampler(source_sizes, [0.5, 0.5], batch_size=10)
        drawn = [index for batch in sampler for index in batch if _source_of(index, source_sizes) == 1]
        assert len(drawn) > len(set(drawn))

    def test_large_source_is_not_repeated_within_an_epoch(self) -> None:
        source_sizes = [500, 5]
        sampler = WeightedMultiSourceBatchSampler(source_sizes, [0.5, 0.5], batch_size=10)
        drawn = [index for batch in sampler for index in batch if _source_of(index, source_sizes) == 0]
        assert len(drawn) == len(set(drawn))


class TestEpochLength:
    """Which source defines an epoch."""

    @pytest.mark.parametrize(
        ("epoch_length", "expected_batches"),
        [
            # source 0 has 500 samples and 5 slots per batch; source 1 has 100 samples and 5 slots.
            pytest.param("largest", 100, id="largest_drives"),
            pytest.param("smallest", 20, id="smallest_drives"),
            pytest.param(1, 20, id="explicit_index_drives"),
        ],
    )
    def test_length(self, epoch_length: str | int, expected_batches: int) -> None:
        sampler = WeightedMultiSourceBatchSampler([500, 100], [0.5, 0.5], batch_size=10, epoch_length=epoch_length)
        assert len(sampler) == expected_batches

    def test_len_matches_number_of_yielded_batches(self) -> None:
        sampler = WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16)
        assert len(list(sampler)) == len(sampler)

    @pytest.mark.parametrize(
        ("drop_last", "expected_batches"),
        [
            # The driving source has 503 samples and 5 slots per batch, leaving 3 samples over 100 whole batches.
            pytest.param(True, 100, id="partial_batch_dropped"),
            pytest.param(False, 101, id="partial_batch_kept"),
        ],
    )
    def test_drop_last_controls_the_trailing_partial_batch(self, drop_last: bool, expected_batches: int) -> None:
        sampler = WeightedMultiSourceBatchSampler([503, 100], [0.5, 0.5], batch_size=10, drop_last=drop_last)
        assert len(list(sampler)) == expected_batches

    def test_trailing_partial_batch_is_still_full_size(self) -> None:
        # Keeping the partial batch tops the driving source up by recycling it, so the model never sees a short batch.
        sampler = WeightedMultiSourceBatchSampler([503, 100], [0.5, 0.5], batch_size=10, drop_last=False)
        assert len(list(sampler)[-1]) == 10

    def test_tiny_dataset_still_yields_one_batch(self) -> None:
        sampler = WeightedMultiSourceBatchSampler([3, 2], [0.5, 0.5], batch_size=16)
        assert len(list(sampler)) == 1


class TestRecyclingWarning:
    """Over-recycling warnings.

    The scenario below has ``batch_size=3`` over four sources, so the two lowest-weighted sources are starved (zero
    slots per batch). Source 3 is the smallest overall but never sampled, while source 1 is recycled 25x per epoch.
    """

    STARVED_SIZES = [1000, 20, 50, 5]
    STARVED_WEIGHTS = [0.5, 0.3, 0.15, 0.05]
    STARVED_BATCH_SIZE = 3

    def _build_starved_sampler(self) -> WeightedMultiSourceBatchSampler:
        return WeightedMultiSourceBatchSampler(
            self.STARVED_SIZES, self.STARVED_WEIGHTS, batch_size=self.STARVED_BATCH_SIZE
        )

    def test_warns_about_the_most_recycled_contributing_source(self, caplog: pytest.LogCaptureFixture) -> None:
        with _capture_warnings(caplog):
            self._build_starved_sampler()
        assert "Source 1" in "".join(_recycling_warnings(caplog))

    def test_starved_source_does_not_suppress_the_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # Source 3 is the smallest but contributes no samples, so it must not be reported as over-recycled.
        with _capture_warnings(caplog):
            self._build_starved_sampler()
        assert "Source 3" not in "".join(_recycling_warnings(caplog))

    def test_reports_the_actual_recycling_factor(self, caplog: pytest.LogCaptureFixture) -> None:
        with _capture_warnings(caplog):
            self._build_starved_sampler()
        assert "~25.0 times" in "".join(_recycling_warnings(caplog))

    def test_no_warning_when_sources_are_balanced(self, caplog: pytest.LogCaptureFixture) -> None:
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([1000, 900], [0.5, 0.5], batch_size=10)
        assert _recycling_warnings(caplog) == []

    def test_starved_driving_source_falls_back_to_a_contributing_source(self) -> None:
        # epoch_length=3 points at a starved source, which cannot define the epoch length.
        sampler = WeightedMultiSourceBatchSampler(
            self.STARVED_SIZES, self.STARVED_WEIGHTS, batch_size=self.STARVED_BATCH_SIZE, epoch_length=3
        )
        assert sampler.source_batch_sizes[sampler.driving_source] > 0


class TestRatioDivergenceWarning:
    """Warnings about batches that cannot hold the requested weights.

    ``batch_size=4`` over four sources leaves exactly one slot each, so the dominant source realises 25% of every batch
    instead of the requested 97% and the three tiny sources realise 25x their requested share.
    """

    INVERTED_SIZES = [1000, 500, 500, 500]
    INVERTED_WEIGHTS = [0.97, 0.01, 0.01, 0.01]
    INVERTED_BATCH_SIZE = 4

    def _build_inverted_sampler(self) -> WeightedMultiSourceBatchSampler:
        return WeightedMultiSourceBatchSampler(
            self.INVERTED_SIZES, self.INVERTED_WEIGHTS, batch_size=self.INVERTED_BATCH_SIZE
        )

    def test_warns_when_the_guaranteed_slot_inverts_the_requested_share(self, caplog: pytest.LogCaptureFixture) -> None:
        """The dominant source is reported with both its requested and its realised share."""
        with _capture_warnings(caplog):
            self._build_inverted_sampler()
        assert "source 0 requested 97.0% but gets 25.0%" in "".join(_ratio_warnings(caplog))

    def test_suggests_a_batch_size_that_fits_the_smallest_weight(self, caplog: pytest.LogCaptureFixture) -> None:
        """A weight of 1% needs 100 slots before it stops being rounded up to a whole one."""
        with _capture_warnings(caplog):
            self._build_inverted_sampler()
        assert "at least 100" in "".join(_ratio_warnings(caplog))

    def test_no_warning_when_the_batch_holds_the_requested_ratio(self, caplog: pytest.LogCaptureFixture) -> None:
        """Weights of [0.6, 0.3, 0.1] fit a batch of 16 closely enough to stay quiet."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16)
        assert _ratio_warnings(caplog) == []

    def test_no_warning_when_only_the_relative_share_diverges(self, caplog: pytest.LogCaptureFixture) -> None:
        # A source requested at 3% gets one of 16 slots (6.25%): double its share, but only 3 points of the batch.
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([1000, 500, 500], [0.94, 0.03, 0.03], batch_size=16)
        assert _ratio_warnings(caplog) == []

    def test_starved_sources_are_left_to_the_dedicated_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # Source 2 is requested at 15% and realises 0%, which the starved-source warning already reports.
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([1000, 20, 50, 5], [0.5, 0.3, 0.15, 0.05], batch_size=3)
        assert _ratio_warnings(caplog) == []


class TestShufflingDeterminism:
    """Epoch-seeded shuffling."""

    def test_same_epoch_reproduces_batches(self) -> None:
        sampler = WeightedMultiSourceBatchSampler([200, 100], [0.5, 0.5], batch_size=8, seed=7)
        sampler.set_epoch(3)
        first = list(sampler)
        sampler.set_epoch(3)
        assert list(sampler) == first

    def test_different_epochs_reshuffle(self) -> None:
        sampler = WeightedMultiSourceBatchSampler([200, 100], [0.5, 0.5], batch_size=8, seed=7)
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        assert list(sampler) != first

    def test_different_seeds_reshuffle(self) -> None:
        first = list(WeightedMultiSourceBatchSampler([200, 100], [0.5, 0.5], batch_size=8, seed=0))
        second = list(WeightedMultiSourceBatchSampler([200, 100], [0.5, 0.5], batch_size=8, seed=1))
        assert first != second

    def test_shuffle_disabled_yields_sequential_indices(self) -> None:
        sampler = WeightedMultiSourceBatchSampler([100, 100], [0.5, 0.5], batch_size=8, shuffle=False)
        first_batch = next(iter(sampler))
        assert sorted(first_batch) == [0, 1, 2, 3, 100, 101, 102, 103]


class TestDistributedSharding:
    """Behaviour across DDP replicas."""

    @pytest.mark.parametrize("rank", [pytest.param(0, id="rank0"), pytest.param(1, id="rank1")])
    def test_each_rank_yields_the_same_batch_count(self, rank: int) -> None:
        sampler = WeightedMultiSourceBatchSampler(
            [500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=2, rank=rank
        )
        assert len(list(sampler)) == len(sampler)

    def test_ranks_receive_disjoint_batches(self) -> None:
        def batches_for(rank: int) -> list[list[int]]:
            sampler = WeightedMultiSourceBatchSampler(
                [500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=2, rank=rank, seed=11
            )
            sampler.set_epoch(0)
            return list(sampler)

        rank0 = {tuple(batch) for batch in batches_for(0)}
        rank1 = {tuple(batch) for batch in batches_for(1)}
        assert rank0.isdisjoint(rank1)

    def test_sharding_reproduces_the_single_process_stream(self) -> None:
        def batches_for(num_replicas: int, rank: int) -> list[tuple[int, ...]]:
            sampler = WeightedMultiSourceBatchSampler(
                [500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=num_replicas, rank=rank, seed=11
            )
            sampler.set_epoch(0)
            return [tuple(batch) for batch in sampler]

        single = batches_for(1, 0)
        interleaved = [batch for pair in zip(batches_for(2, 0), batches_for(2, 1), strict=True) for batch in pair]
        assert interleaved == single[: len(interleaved)]

    def test_clamped_length_still_shards_disjoint_batches(self) -> None:
        # The driving source (size 3) can't fill even one rank's 16-slot allocation, so len(sampler) clamps to 1 per
        # rank on both sides of the num_replicas=2 split, not just in the single-process case.
        def batches_for(rank: int) -> list[list[int]]:
            sampler = WeightedMultiSourceBatchSampler(
                [3, 2], [0.5, 0.5], batch_size=16, num_replicas=2, rank=rank, seed=11
            )
            sampler.set_epoch(0)
            return list(sampler)

        rank0_batches = batches_for(0)
        rank1_batches = batches_for(1)
        assert len(rank0_batches) == 1
        assert len(rank1_batches) == 1
        rank0 = {tuple(batch) for batch in rank0_batches}
        rank1 = {tuple(batch) for batch in rank1_batches}
        assert rank0.isdisjoint(rank1)

    def test_truncates_to_a_whole_multiple_of_num_replicas(self) -> None:
        # epoch_length=1 drives the epoch off source 1 (17 samples, 5 slots/batch): floor(17/5)=3 global batches
        # single-process, which is not a multiple of num_replicas=2 and must truncate to 2 rather than round up to 4.
        single_process = WeightedMultiSourceBatchSampler(
            [500, 17], [0.5, 0.5], batch_size=10, num_replicas=1, rank=0, epoch_length=1
        )
        assert len(single_process) == 3

        rank0 = WeightedMultiSourceBatchSampler(
            [500, 17], [0.5, 0.5], batch_size=10, num_replicas=2, rank=0, epoch_length=1
        )
        rank1 = WeightedMultiSourceBatchSampler(
            [500, 17], [0.5, 0.5], batch_size=10, num_replicas=2, rank=1, epoch_length=1
        )
        assert len(rank0) == len(rank1) == 1
        assert len(list(rank0)) + len(list(rank1)) == 2


class TestPublicReExports:
    """Import-path parity between ``rfdetr.datasets`` and the internal ``multi_source`` module."""

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("WeightedMultiSourceBatchSampler", id="sampler_class"),
            pytest.param("compute_source_batch_sizes", id="allocation_function"),
        ],
    )
    def test_public_import_resolves_to_the_same_object(self, name: str) -> None:
        """The documented ``rfdetr.datasets`` import path returns the identical object as the internal module."""
        import rfdetr.datasets as public_module
        import rfdetr.datasets.multi_source as internal_module

        assert getattr(public_module, name) is getattr(internal_module, name)


class TestDataLoaderIntegration:
    """Use as a ``DataLoader`` batch sampler."""

    def test_dataloader_yields_requested_batch_size(self) -> None:
        sources = [TensorDataset(torch.arange(size, dtype=torch.float32)) for size in (200, 100, 20)]
        dataset = ConcatDataset(sources)
        sampler = WeightedMultiSourceBatchSampler.from_concat_dataset(dataset, [0.6, 0.3, 0.1], batch_size=16)
        loader = DataLoader(dataset, batch_sampler=sampler)
        assert next(iter(loader))[0].shape[0] == 16

    def test_from_concat_dataset_derives_source_sizes(self) -> None:
        sources = [TensorDataset(torch.zeros(size)) for size in (200, 100, 20)]
        sampler = WeightedMultiSourceBatchSampler.from_concat_dataset(
            ConcatDataset(sources), [0.6, 0.3, 0.1], batch_size=16
        )
        assert sampler.source_sizes == [200, 100, 20]

    def test_from_concat_dataset_rejects_weight_count_mismatch(self) -> None:
        dataset = ConcatDataset([TensorDataset(torch.zeros(size)) for size in (200, 100)])
        with pytest.raises(ValueError, match="one entry per sub-dataset"):
            WeightedMultiSourceBatchSampler.from_concat_dataset(dataset, [0.6, 0.3, 0.1], batch_size=16)
