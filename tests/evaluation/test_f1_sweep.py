# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any

import numpy as np
import pytest

from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds


def _reference_sweep(
    per_class_data: list[dict[str, Any]],
    conf_thresholds: Any,
    classes_with_gt: list[int],
) -> list[dict[str, Any]]:
    """O(T*N) reference implementation: the exact algorithm sweep_confidence_thresholds used before
    this PR -- rescans every detection at every threshold instead of sorting once per class. Kept
    standalone here (not the module under test) as the ground truth for the equivalence tests below,
    so a bug shared between the two implementations can't hide a real divergence.

    Examples:
        >>> data = [{"scores": np.array([0.9, 0.4]), "matches": np.array([1, 0]),
        ...          "ignore": np.array([False, False]), "total_gt": 1}]
        >>> results = _reference_sweep(data, [0.5], [0])
        >>> round(results[0]["macro_precision"], 3), round(results[0]["macro_recall"], 3)
        (1.0, 1.0)
    """
    num_classes = len(per_class_data)
    results: list[dict[str, Any]] = []

    for conf_thresh in conf_thresholds:
        per_class_precisions: list[float] = []
        per_class_recalls: list[float] = []
        per_class_f1s: list[float] = []

        for k in range(num_classes):
            data = per_class_data[k]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore
            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = float(np.mean([per_class_precisions[k] for k in classes_with_gt]))
            macro_recall = float(np.mean([per_class_recalls[k] for k in classes_with_gt]))
            macro_f1 = float(np.mean([per_class_f1s[k] for k in classes_with_gt]))
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )

    return results


def _random_per_class_data(seed: int, num_classes: int, max_detections: int) -> list[dict[str, Any]]:
    """Build random per-class matching data with tied scores and a mix of TP/FP/ignored detections,
    for equivalence-testing sweep_confidence_thresholds against _reference_sweep.

    ``matches`` and ``total_gt`` are drawn independently, so ``sum(matches) > total_gt`` is possible
    here even though it can't happen in real matching output -- both implementations under test treat
    ``tp``/``fp``/``total_gt`` as opaque counts and never compare them against each other, so this
    doesn't weaken the equivalence check, only broaden the input space it covers. It also doesn't
    exercise the real detector -> matching -> sweep integration; that lives in
    ``tests/benchmarks/test_inference_coco.py`` (``coco17``-marked, downloads COCO val2017).

    Examples:
        >>> data = _random_per_class_data(seed=0, num_classes=1, max_detections=3)
        >>> sorted(data[0].keys())
        ['ignore', 'matches', 'scores', 'total_gt']
    """
    rng = np.random.default_rng(seed)
    per_class_data = []
    for _ in range(num_classes):
        n = int(rng.integers(0, max_detections + 1))
        # Round scores to encourage exact ties, which is where searchsorted's "side" behavior most
        # needs to match the original inclusive `scores >= conf_thresh` semantics.
        scores = np.round(rng.uniform(0.0, 1.0, size=n), 1)
        matches = rng.integers(0, 2, size=n)  # 0 == unmatched (FP candidate), 1 == matched (TP candidate)
        ignore = rng.random(size=n) < 0.2
        total_gt = int(rng.integers(0, max_detections + 1))
        per_class_data.append({"scores": scores, "matches": matches, "ignore": ignore, "total_gt": total_gt})
    return per_class_data


def _assert_results_equal(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    """Assert two sweep_confidence_thresholds-shaped result lists are numerically identical.

    Examples:
        >>> r = [{"confidence_threshold": 0.5, "macro_f1": 1.0, "macro_precision": 1.0,
        ...       "macro_recall": 1.0, "per_class_prec": np.array([1.0]),
        ...       "per_class_rec": np.array([1.0]), "per_class_f1": np.array([1.0])}]
        >>> _assert_results_equal(r, r)
    """
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a["confidence_threshold"] == pytest.approx(e["confidence_threshold"])
        assert a["macro_f1"] == pytest.approx(e["macro_f1"])
        assert a["macro_precision"] == pytest.approx(e["macro_precision"])
        assert a["macro_recall"] == pytest.approx(e["macro_recall"])
        np.testing.assert_allclose(a["per_class_prec"], e["per_class_prec"])
        np.testing.assert_allclose(a["per_class_rec"], e["per_class_rec"])
        np.testing.assert_allclose(a["per_class_f1"], e["per_class_f1"])


class TestSweepConfidenceThresholdsMatchesReference:
    """sweep_confidence_thresholds must return numerically identical results to the O(T*N) reference
    it replaced, across random data, empty classes, all-ignored classes, and threshold/score ties.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4], ids=lambda s: f"seed={s}")
    def test_matches_reference_on_random_data(self, seed: int) -> None:
        per_class_data = _random_per_class_data(seed=seed, num_classes=5, max_detections=40)
        classes_with_gt = [k for k, d in enumerate(per_class_data) if d["total_gt"] > 0]
        conf_thresholds = np.linspace(0, 1, 101)

        actual = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt)
        expected = _reference_sweep(per_class_data, conf_thresholds, classes_with_gt)

        _assert_results_equal(actual, expected)

    def test_matches_reference_with_no_detections_in_any_class(self) -> None:
        per_class_data = [
            {"scores": np.array([]), "matches": np.array([]), "ignore": np.array([], dtype=bool), "total_gt": 0}
            for _ in range(3)
        ]
        conf_thresholds = [0.0, 0.5, 1.0]

        actual = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt=[])
        expected = _reference_sweep(per_class_data, conf_thresholds, classes_with_gt=[])

        _assert_results_equal(actual, expected)

    def test_matches_reference_when_every_detection_is_ignored(self) -> None:
        per_class_data = [
            {
                "scores": np.array([0.9, 0.6, 0.3]),
                "matches": np.array([1, 0, 1]),
                "ignore": np.array([True, True, True]),
                "total_gt": 2,
            }
        ]
        conf_thresholds = np.linspace(0, 1, 11)

        actual = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt=[0])
        expected = _reference_sweep(per_class_data, conf_thresholds, classes_with_gt=[0])

        _assert_results_equal(actual, expected)
        assert actual[0]["macro_precision"] == 0.0  # tp+fp==0 when everything is ignored

    def test_matches_reference_with_scores_exactly_on_threshold_boundaries(self) -> None:
        """`scores >= conf_thresh` is inclusive; a detection scored exactly at a threshold must count
        as above it in both implementations, and thresholds are chosen to land exactly on scores."""
        per_class_data = [
            {
                "scores": np.array([0.3, 0.3, 0.7, 0.7, 1.0]),
                "matches": np.array([1, 0, 1, 1, 0]),
                "ignore": np.array([False, False, False, False, False]),
                "total_gt": 3,
            }
        ]
        conf_thresholds = [0.0, 0.3, 0.7, 1.0]

        actual = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt=[0])
        expected = _reference_sweep(per_class_data, conf_thresholds, classes_with_gt=[0])

        _assert_results_equal(actual, expected)

    def test_matches_reference_with_nan_scores(self) -> None:
        """A NaN score fails `scores >= conf_thresh` for every threshold, so the O(T*N) reference
        always excludes it. `np.argsort` sorts NaN to the end of ascending order, which would put it
        in the "above every threshold" suffix if not masked out the same way `ignore` is -- this
        regresses `macro_f1` from 0.0 to 1.0 for the exact case below if the mask is dropped."""
        per_class_data = [
            {
                "scores": np.array([np.nan]),
                "matches": np.array([1]),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        ]
        conf_thresholds = [0.5]

        actual = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt=[0])
        expected = _reference_sweep(per_class_data, conf_thresholds, classes_with_gt=[0])

        _assert_results_equal(actual, expected)
        assert actual[0]["macro_f1"] == 0.0

    def test_matches_reference_with_generator_conf_thresholds(self) -> None:
        """conf_thresholds is documented as any iterable, including a one-shot generator -- pins that
        sweep_confidence_thresholds materializes it exactly once instead of consuming it partway
        through (e.g. once to measure its length, again to iterate results), which would silently
        return fewer thresholds than requested for a genuinely single-pass iterable."""
        per_class_data = [
            {
                "scores": np.array([0.9, 0.4]),
                "matches": np.array([1, 0]),
                "ignore": np.array([False, False]),
                "total_gt": 1,
            }
        ]
        thresholds_list = [0.0, 0.5, 1.0]

        actual = sweep_confidence_thresholds(
            per_class_data, (t for t in thresholds_list), classes_with_gt=[0]
        )

        assert len(actual) == len(thresholds_list)
        assert [r["confidence_threshold"] for r in actual] == pytest.approx(thresholds_list)


class TestSweepConfidenceThresholdsBasicCorrectness:
    """A few hand-computed cases pin the actual precision/recall/F1 values, independent of the
    reference-equivalence tests above (which would pass even if both implementations shared the same
    bug)."""

    def test_all_correct_detections_gives_perfect_scores(self) -> None:
        per_class_data = [
            {
                "scores": np.array([0.9, 0.8]),
                "matches": np.array([1, 1]),
                "ignore": np.array([False, False]),
                "total_gt": 2,
            }
        ]

        result = sweep_confidence_thresholds(per_class_data, [0.5], classes_with_gt=[0])[0]

        assert result["macro_precision"] == pytest.approx(1.0)
        assert result["macro_recall"] == pytest.approx(1.0)
        assert result["macro_f1"] == pytest.approx(1.0)

    def test_raising_threshold_above_a_false_positives_score_removes_it(self) -> None:
        per_class_data = [
            {
                "scores": np.array([0.9, 0.4]),  # 0.9 is a TP, 0.4 is a FP
                "matches": np.array([1, 0]),
                "ignore": np.array([False, False]),
                "total_gt": 1,
            }
        ]

        low_thresh = sweep_confidence_thresholds(per_class_data, [0.0], classes_with_gt=[0])[0]
        high_thresh = sweep_confidence_thresholds(per_class_data, [0.5], classes_with_gt=[0])[0]

        assert low_thresh["macro_precision"] == pytest.approx(0.5)  # 1 TP, 1 FP
        assert high_thresh["macro_precision"] == pytest.approx(1.0)  # FP dropped below threshold
        assert high_thresh["macro_recall"] == pytest.approx(1.0)  # the TP is still above threshold
