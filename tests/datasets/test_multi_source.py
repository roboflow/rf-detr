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


def _layout_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the emitted num_replicas=1-hides-a-live-process-group warnings.

    Examples:
        Needs pytest's ``caplog`` fixture, so the live call is covered by tests rather than doctest.

        >>> callable(_layout_warnings)  # doctest: +SKIP
        True
    """
    return [record.getMessage() for record in caplog.records if "world size of" in record.getMessage()]


def _ddp_truncation_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the emitted DDP epoch-truncation warnings (driving-source recycle, under-coverage, drop_last no-op).

    Examples:
        Needs pytest's ``caplog`` fixture, so the live call is covered by tests rather than doctest.

        >>> callable(_ddp_truncation_warnings)  # doctest: +SKIP
        True
    """
    markers = ("recycling the driving source", "is only", "drop_last=False has no effect")
    return [
        record.getMessage() for record in caplog.records if any(marker in record.getMessage() for marker in markers)
    ]


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

    def test_allocates_very_large_unnormalised_weights(self) -> None:
        # Weights are documented as relative, so a 2:1 ratio must allocate the same whatever its absolute magnitude.
        assert compute_source_batch_sizes(6, [1e308, 5e307]) == compute_source_batch_sizes(6, [2.0, 1.0])

    def test_weight_dwarfed_by_a_huge_one_raises_valueerror_not_overflow(self) -> None:
        # 1e308 is finite and passes the positivity check; scaling it must not raise a bare OverflowError.
        with pytest.raises(ValueError, match="too small to be represented"):
            compute_source_batch_sizes(8, [1e308, 1.0])

    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            compute_source_batch_sizes(0, [0.5, 0.5])

    def test_single_source_receives_the_full_batch(self) -> None:
        """A single-source allocation (``len(weights) == 1``) is a legitimate, untested shape.

        A ``ConcatDataset`` of one dataset before expansion hits this path; every other allocation test in this class
        uses 2+ sources.
        """
        assert compute_source_batch_sizes(8, [1.0]) == [8]

    def test_clamped_source_excluded_from_leftover_remainder_ranking(self) -> None:
        """A source the one-slot guarantee bumps from 0 must not also win a leftover slot by remainder.

        batch_size=8, weights=[0.45, 0.45, 0.1]: floor counts are [3, 3, 0]; the guarantee clamps index 2 up to 1,
        giving [3, 3, 1] with shortfall=1. True Hamilton distributes the one leftover slot by remainder among every
        source, landing on index 2 again ([3, 3, 2]) only if the clamp is not excluded — the realised ratio would then
        be [0.375, 0.375, 0.25] against a requested [0.45, 0.45, 0.1], starving both dominant sources. Excluding the
        clamped index from the leftover ranking gives the leftover slot to the highest-remainder *non-clamped* source
        (index 0), matching true Hamilton apportionment: [4, 3, 1].
        """
        assert compute_source_batch_sizes(8, [0.45, 0.45, 0.1]) == [4, 3, 1]


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


class TestBatchMultipleAlignment:
    """``batch_multiple`` keeps an epoch a whole number of gradient-accumulation windows."""

    @pytest.mark.parametrize("multiple", [pytest.param(2, id="two"), pytest.param(4, id="four")])
    def test_epoch_length_is_a_multiple(self, multiple: int) -> None:
        sampler = WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=4, batch_multiple=multiple)
        assert len(sampler) % multiple == 0

    def test_alignment_only_truncates(self) -> None:
        """Rounding down must never invent batches beyond the unaligned epoch."""
        unaligned = WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=4)
        aligned = WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=4, batch_multiple=4)
        assert len(aligned) <= len(unaligned)

    def test_iteration_yields_the_declared_batch_count(self) -> None:
        """``__len__`` must match what ``__iter__`` actually produces once aligned."""
        sampler = WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=4, batch_multiple=4)
        assert sum(1 for _ in sampler) == len(sampler)

    def test_default_is_unaligned(self) -> None:
        """Omitting the argument must leave the existing epoch length untouched."""
        explicit = WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=4, batch_multiple=1)
        assert len(explicit) == len(WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=4))

    def test_alignment_applies_independently_with_a_kept_trailing_batch(self) -> None:
        """``batch_multiple`` rounds correctly whether or not ``drop_last=False`` kept the trailing batch.

        This class and ``TestEpochLength``'s ``drop_last`` cases each exercise ``batch_multiple`` and ``drop_last`` on
        their own; a kept trailing partial batch folding into ``batch_multiple`` rounding was untested.
        """
        # Driving source (101 samples, 5 slots/batch): 20 whole batches + 1 remainder batch.
        kept = WeightedMultiSourceBatchSampler([101, 50], [0.5, 0.5], batch_size=10, drop_last=False, batch_multiple=3)
        dropped = WeightedMultiSourceBatchSampler(
            [101, 50], [0.5, 0.5], batch_size=10, drop_last=True, batch_multiple=3
        )
        assert len(dropped) == 18  # 20 rounded down to a multiple of 3
        assert len(kept) == 21  # 21 (20 + the kept remainder batch) is already a multiple of 3
        assert all(len(batch) == 10 for batch in kept)

    def test_rejects_multiple_larger_than_the_epoch(self) -> None:
        """An alignment the epoch cannot satisfy must fail loudly rather than yield zero batches."""
        with pytest.raises(ValueError, match="batch_multiple"):
            WeightedMultiSourceBatchSampler([8, 8], [0.5, 0.5], batch_size=4, batch_multiple=64)

    def test_rejects_non_positive_multiple(self) -> None:
        with pytest.raises(ValueError, match="batch_multiple must be >= 1"):
            WeightedMultiSourceBatchSampler([100, 50], [0.5, 0.5], batch_size=4, batch_multiple=0)


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
        """Build the sampler whose two lowest-weighted sources get no slots in a batch.

        Returns:
            Sampler over ``STARVED_SIZES`` allocating ``[2, 1, 0, 0]`` slots per batch, so sources 2 and 3 are
            starved and source 1 (20 samples, one slot per batch) is the most-recycled contributor.

        Examples:
            >>> TestRecyclingWarning()._build_starved_sampler().source_batch_sizes
            [2, 1, 0, 0]
        """
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

    def test_reports_every_source_above_the_threshold_not_only_the_worst(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two sources recycled above the threshold (37.5x and 20.0x) must both appear, not only the worst one.

        A ``max()``-based warning would report only the 37.5x source while the 20.0x one reads clean, hiding a second
        genuinely over-recycled contributor from the log.
        """
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([1000, 20, 25, 40], [0.4, 0.3, 0.2, 0.1], batch_size=10, epoch_length=0)
        message = "".join(_recycling_warnings(caplog))
        assert "Source 1" in message
        assert "Source 2" in message

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

    def test_single_source_never_triggers_the_recycling_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """``_warn_on_source_imbalance``'s ``max()`` over a one-entry dict must not crash or false-warn.

        A single source is always its own driving source, so it can never be recycled — every other test in this class
        uses 3+ sources.
        """
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([20], [1.0], batch_size=4)
        assert _recycling_warnings(caplog) == []


class TestDDPTruncationWarnings:
    """Warnings about the three ways many DDP ranks can silently shrink or misrepresent an epoch.

    ``max(1, global_batches // num_replicas)`` guarantees every rank at least one batch, but on a small driving source
    it can recycle that source instead of covering it once; stacked floor-divisions can drop a meaningful fraction of
    the driving source with no other signal; and ``drop_last=False``'s extra batch can be floored away by the same per-
    rank division that adds the guarantee, at every replica count that divides evenly without it.
    """

    def test_warns_when_the_driving_source_is_recycled_across_ranks(self, caplog: pytest.LogCaptureFixture) -> None:
        """2 true global batches over 8 replicas forces every rank to replay the driving source, not see it once."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([5, 5], [0.5, 0.5], batch_size=4, num_replicas=8, rank=0)
        message = "".join(_ddp_truncation_warnings(caplog))
        assert "recycling the driving source" in message

    def test_no_recycle_warning_at_a_single_replica(self, caplog: pytest.LogCaptureFixture) -> None:
        """The same driving source at num_replicas=1 has enough global batches and must stay quiet."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([1000, 500], [0.6, 0.4], batch_size=16, num_replicas=1, rank=0)
        assert "recycling the driving source" not in "".join(_ddp_truncation_warnings(caplog))

    def test_warns_when_stacked_rounding_under_covers_the_driving_source(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """8 replicas + batch_multiple=4 draws only 320 of the driving source's 500 samples (64%), unwarned
        otherwise."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler(
                [500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=8, rank=0, batch_multiple=4
            )
        message = "".join(_ddp_truncation_warnings(caplog))
        assert "is only" in message
        assert "64%" in message

    def test_no_coverage_warning_when_rounding_stays_above_the_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The same sources at num_replicas=1 keep coverage above 90% and must stay quiet."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=1, rank=0)
        assert "is only" not in "".join(_ddp_truncation_warnings(caplog))

    def test_warns_when_drop_last_false_is_floored_away_at_an_even_replica_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """505 samples over 50-slot batches need drop_last=False's extra batch (51 vs 50), but 2 replicas floor it
        away."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler(
                [505, 200, 40], [0.6, 0.3, 0.1], batch_size=16, drop_last=False, num_replicas=2, rank=0
            )
        assert "drop_last=False has no effect" in "".join(_ddp_truncation_warnings(caplog))

    def test_no_no_op_warning_when_drop_last_false_is_honoured(self, caplog: pytest.LogCaptureFixture) -> None:
        """The same sources at num_replicas=1 keep the extra batch, so drop_last=False must stay quiet."""
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler(
                [505, 200, 40], [0.6, 0.3, 0.1], batch_size=16, drop_last=False, num_replicas=1, rank=0
            )
        assert "drop_last=False has no effect" not in "".join(_ddp_truncation_warnings(caplog))


class TestRatioDivergenceWarning:
    """Warnings about batches that cannot hold the requested weights.

    ``batch_size=4`` over four sources leaves exactly one slot each, so the dominant source realises 25% of every batch
    instead of the requested 97% and the three tiny sources realise 25x their requested share.
    """

    INVERTED_SIZES = [1000, 500, 500, 500]
    INVERTED_WEIGHTS = [0.97, 0.01, 0.01, 0.01]
    INVERTED_BATCH_SIZE = 4

    def _build_inverted_sampler(self) -> WeightedMultiSourceBatchSampler:
        """Build the sampler whose one-slot-per-source guarantee inverts the requested 97/1/1/1 split.

        Returns:
            Sampler over ``INVERTED_SIZES`` allocating ``[1, 1, 1, 1]`` slots per batch, so the dominant source
            realises 25% of each batch instead of the 97% it asked for.

        Examples:
            >>> TestRatioDivergenceWarning()._build_inverted_sampler().source_batch_sizes
            [1, 1, 1, 1]
        """
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

    def test_weights_near_float_max_do_not_raise_overflowerror(self) -> None:
        """Two weights near float64 max pass validation individually but overflow ``sum()`` to inf.

        ``_validate_and_scale_weights`` rescales its own internal copy when the *largest* weight alone would overflow,
        but ``_warn_on_ratio_divergence`` reads the raw, unrescaled ``self.weights`` — so ``sum([1e308, 1e308])`` (>
        float64 max) still overflows to ``inf`` there, making ``ceil(total_weight / min(self.weights))`` raise
        ``OverflowError`` unconditionally, regardless of log level, because logging arguments are evaluated eagerly. The
        guard must make construction succeed.
        """
        sampler = WeightedMultiSourceBatchSampler([50, 50], [1e308, 1e308], batch_size=8)
        assert sampler.source_batch_sizes == [4, 4]


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


class TestDistributedLayoutAutoDetect:
    """``num_replicas``/``rank`` default to the live ``torch.distributed`` process group.

    Mocks ``get_world_size``/``get_rank`` at the ``multi_source`` module boundary rather than starting a real process
    group, mirroring how ``_resolve_distributed_layout`` consumes them.
    """

    def test_auto_detects_world_size_and_rank_from_the_process_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting num_replicas/rank reads them from a live process group instead of defaulting to (1, 0)."""
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_world_size", lambda: 4)
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_rank", lambda: 2)
        sampler = WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16)
        assert (sampler.num_replicas, sampler.rank) == (4, 2)

    def test_explicit_values_override_the_process_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit num_replicas/rank win even when a live process group reports a different layout."""
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_world_size", lambda: 4)
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_rank", lambda: 2)
        sampler = WeightedMultiSourceBatchSampler(
            [500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=1, rank=0
        )
        assert (sampler.num_replicas, sampler.rank) == (1, 0)

    def test_warns_when_explicit_num_replicas_of_one_hides_a_live_process_group(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Passing num_replicas=1 under a world_size=4 process group would silently duplicate every rank's data."""
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_world_size", lambda: 4)
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_rank", lambda: 0)
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16, num_replicas=1, rank=0)
        assert "world size of 4" in "".join(_layout_warnings(caplog))

    def test_no_warning_when_auto_detected_layout_matches_the_process_group(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single-process group (world_size=1) never triggers the mismatch warning."""
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_world_size", lambda: 1)
        monkeypatch.setattr("rfdetr.datasets.multi_source.get_rank", lambda: 0)
        with _capture_warnings(caplog):
            WeightedMultiSourceBatchSampler([500, 200, 40], [0.6, 0.3, 0.1], batch_size=16)
        assert _layout_warnings(caplog) == []


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

    def test_single_source_shards_across_ranks(self) -> None:
        """A single-source sampler (``len(weights) == 1``) still shards disjointly under DDP.

        Every other sharding test in this class uses 3+ sources.
        """

        def batches_for(rank: int) -> list[list[int]]:
            sampler = WeightedMultiSourceBatchSampler([1000], [1.0], batch_size=10, num_replicas=2, rank=rank)
            sampler.set_epoch(0)
            return list(sampler)

        rank0 = {tuple(batch) for batch in batches_for(0)}
        rank1 = {tuple(batch) for batch in batches_for(1)}
        assert rank0 and rank1
        assert rank0.isdisjoint(rank1)

    def test_starved_source_configuration_shards_correctly_under_ddp(self) -> None:
        """A starved source (zero slots per batch) combined with ``num_replicas > 1`` shards cleanly.

        ``TestRecyclingWarning``'s starved-source cases are single-process only, and every case here gives each source
        at least one slot, so starved x DDP was never combined.
        """
        weights = [0.9, 0.04, 0.03, 0.03]
        source_sizes = [1000, 1000, 1000, 1000]

        def batches_for(rank: int) -> tuple[WeightedMultiSourceBatchSampler, list[list[int]]]:
            sampler = WeightedMultiSourceBatchSampler(source_sizes, weights, batch_size=3, num_replicas=2, rank=rank)
            sampler.set_epoch(0)
            return sampler, list(sampler)

        rank0, batches0 = batches_for(0)
        _, batches1 = batches_for(1)
        assert rank0.source_batch_sizes == [3, 0, 0, 0]  # sources 1-3 starved at this batch_size
        assert len(batches0) == len(batches1) == len(rank0)
        # Only source 0 (offsets [0, 1000)) contributes samples, on either rank.
        assert all(index < 1000 for batch in batches0 + batches1 for index in batch)


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
