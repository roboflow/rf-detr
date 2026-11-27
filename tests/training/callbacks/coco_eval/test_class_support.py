# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for COCOEvalCallback minimum class support gating and class-agnostic evaluation."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.evaluation.matching import merge_matching_data
from rfdetr.training.callbacks import coco_eval as coco_eval_module
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision
from tests._markers import _requires_ufcoco, _requires_vernier
from tests.training.callbacks.helpers import (
    _cpu_module,
    _detection_preds,
    _detection_targets,
    _ema_callback,
    _make_trainer,
    _matching_entry,
    _metrics_with_per_class,
    _minimal_metrics,
)


def _run_compute_and_log(
    cb: COCOEvalCallback, merged: dict, metrics: dict, split: str = "val", agnostic: dict | None = None
) -> tuple[MagicMock, MagicMock]:
    """Drive ``_compute_and_log`` on the base-model track with the heavy collaborators stubbed out.

    ``agnostic`` is the result a stubbed class-agnostic metric computes (one that holds updates); without it the
    callback's own agnostic metric, if any, is left as ``setup`` built it.

    Examples:
        >>> cb = COCOEvalCallback(min_class_support=2)
        >>> trainer, _ = _run_compute_and_log(cb, {0: _matching_entry(5)}, _metrics_with_per_class({0: 0.5}))
        >>> float(trainer.callback_metrics["val/mAP_50_95_gated"])
        0.5
    """
    trainer = _make_trainer()
    module = _cpu_module()
    cb.setup(trainer, module, stage="fit")
    trainer.callback_metrics = {}
    cb.map_metric = MagicMock(name="map_metric")
    cb.map_metric.has_updates = True
    cb.map_metric.compute.return_value = metrics
    if agnostic is not None:
        cb.map_metric_agnostic = MagicMock(name="map_metric_agnostic")
        cb.map_metric_agnostic.has_updates = True
        cb.map_metric_agnostic.compute.return_value = agnostic

    # Seeded rather than patched in: outside a process group the merge returns the local accumulator unchanged. The
    # table printer is replaced on this callback instance alone, which goes away with the test.
    merge_matching_data(cb._f1_local, merged)
    cb._print_metrics_tables = MagicMock(name="print_metrics_tables")
    cb._compute_and_log(trainer, module, split)
    return trainer, module


def _logged(module: MagicMock, key: str) -> float:
    """Return the value ``module.log`` received for ``key``.

    Examples:
        >>> module = MagicMock()
        >>> _ = module.log("val/F1", 0.25)
        >>> _logged(module, "val/F1")
        0.25
    """
    return float(next(c for c in module.log.call_args_list if c.args[0] == key).args[1])


def _two_class_batches() -> list[dict]:
    """Two validation batches: class 0 found twice (AP 1), class 1 present once and missed (AP 0).

    Examples:
        >>> [len(batch["targets"][0]["labels"]) for batch in _two_class_batches()]
        [1, 2]
    """
    hit = {"boxes": torch.tensor([[90.0, 45.0, 110.0, 55.0]]), "scores": torch.tensor([0.9])}
    return [
        {
            "results": [{**hit, "labels": torch.tensor([0])}],
            "targets": _detection_targets(label=0),
        },
        {
            "results": [{**hit, "labels": torch.tensor([0])}],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]]),
                    "labels": torch.tensor([0, 1]),
                    "orig_size": torch.tensor([100, 200]),
                }
            ],
        },
    ]


def _crowd_only_batches() -> list[dict]:
    """Two validation batches: class 0 found once (AP 1); class 1 present only as a crowd region with one detection.

    Examples:
        >>> [target["labels"].tolist() for batch in _crowd_only_batches() for target in batch["targets"]]
        [[0], [1]]
        >>> _crowd_only_batches()[1]["targets"][0]["iscrowd"].tolist()
        [1]
    """
    hit = {"boxes": torch.tensor([[90.0, 45.0, 110.0, 55.0]]), "scores": torch.tensor([0.9])}
    on_crowd = {"boxes": torch.tensor([[30.0, 15.0, 50.0, 25.0]]), "scores": torch.tensor([0.8])}
    return [
        {"results": [{**hit, "labels": torch.tensor([0])}], "targets": _detection_targets(label=0)},
        {
            "results": [{**on_crowd, "labels": torch.tensor([1])}],
            "targets": [
                {
                    "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]]),
                    "labels": torch.tensor([1]),
                    "iscrowd": torch.tensor([1]),
                    "orig_size": torch.tensor([100, 200]),
                }
            ],
        },
    ]


class TestMinClassSupport:
    """Support floor keeps thinly-annotated classes out of the macro-averaged headline."""

    def test_gated_map_excludes_low_support_classes(self) -> None:
        """A class under the floor is dropped from the gated mean even though it has GT."""
        cb = COCOEvalCallback(min_class_support=10)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        trainer, module = _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6, 1: 0.0}))

        assert _logged(module, "val/mAP_50_95_gated") == pytest.approx(0.6)
        assert float(trainer.callback_metrics["val/mAP_50_95_gated"]) == pytest.approx(0.6)

    def test_all_class_map_is_left_untouched(self) -> None:
        """The ungated key keeps reporting the metric's own all-class mean."""
        cb = COCOEvalCallback(min_class_support=10)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        _, module = _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6, 1: 0.0}))

        assert _logged(module, "val/mAP_50_95") == pytest.approx(0.4)

    def test_falls_back_to_all_class_map_when_none_qualify(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A floor no class reaches must not leave the monitored metric NaN, nor the F1 sweep without classes.

        Both the gated mAP and the sweep fall back to every class holding ground truth, and the callback says so. The
        ``rf-detr`` logger does not propagate to the root logger ``caplog`` listens on.
        """
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        sweep = mocker.spy(coco_eval_module, "sweep_confidence_thresholds")
        cb = COCOEvalCallback(min_class_support=10_000)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            _, module = _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6, 1: 0.0}))

        assert _logged(module, "val/mAP_50_95_gated") == pytest.approx(0.4)
        assert sweep.call_args.args[2] == [0, 1]
        assert any("min_class_support" in record.getMessage() for record in caplog.records)

    def test_classes_with_no_ground_truth_are_ignored(self) -> None:
        """The COCO -1 sentinel must not be averaged in as a real score."""
        cb = COCOEvalCallback(min_class_support=1)
        merged = {0: _matching_entry(100), 1: _matching_entry(50)}
        _, module = _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6, 1: -1.0}))

        assert _logged(module, "val/mAP_50_95_gated") == pytest.approx(0.6)

    def test_no_gated_key_when_floor_is_zero(self) -> None:
        """Default configuration logs exactly the keys it logged before the floor existed."""
        cb = COCOEvalCallback()
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        trainer, module = _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6, 1: 0.0}))

        assert not [c.args[0] for c in module.log.call_args_list if "gated" in c.args[0] or "agnostic" in c.args[0]]
        assert not [key for key in trainer.callback_metrics if "gated" in key or "agnostic" in key]

    def test_f1_sweep_macro_averages_over_gated_classes(self, mocker: MockerFixture) -> None:
        """The swept threshold is chosen for the classes that have enough GT to measure."""
        cb = COCOEvalCallback(min_class_support=10)
        merged = {0: _matching_entry(100), 1: _matching_entry(3), 2: _matching_entry(60)}
        sweep_result = [
            {
                "macro_f1": 0.5,
                "macro_precision": 0.5,
                "macro_recall": 0.5,
                "per_class_f1": np.zeros(3),
                "per_class_prec": np.zeros(3),
                "per_class_rec": np.zeros(3),
            }
        ]
        sweep = mocker.patch(
            "rfdetr.training.callbacks.coco_eval.sweep_confidence_thresholds", return_value=sweep_result
        )

        _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6, 1: 0.0, 2: 0.5}))

        # sorted_ids == [0, 1, 2]; class 1 holds 3 instances, under the floor of 10.
        assert sweep.call_args.args[2] == [0, 2]

    def test_floor_forces_per_class_ap_when_per_class_logging_is_off(self) -> None:
        """The gated mean needs per-class AP even with the per-class table disabled."""
        cb = COCOEvalCallback(min_class_support=5, log_per_class_metrics=False)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")

        assert cb.map_metric.class_metrics is True

    @pytest.mark.parametrize(("min_class_support", "expected"), [(10, [100, 3]), (0, None)])
    def test_per_class_rows_carry_ground_truth_count(
        self, min_class_support: int, expected: list[int] | None, mocker: MockerFixture
    ) -> None:
        """Under a floor every per-class row reports its ground-truth count; without one the rows are unchanged."""
        cb = COCOEvalCallback(min_class_support=min_class_support, log_per_class_metrics=True)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        trainer.callback_metrics = {}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = True
        cb.map_metric.compute.return_value = _metrics_with_per_class({0: 0.6, 1: 0.0})

        print_tables = mocker.patch.object(cb, "_print_metrics_tables")
        mocker.patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value=merged)

        cb._compute_and_log(trainer, module, "val")

        rows = print_tables.call_args.args[3]
        assert len(rows) == 2
        if expected is None:
            assert not any("support" in row for row in rows)
        else:
            assert [row["support"] for row in rows] == expected
        # Low-support classes stay in the table and in the per-class AP log.
        assert "val/AP/1" in {c.args[0] for c in module.log.call_args_list}

    def test_table_shows_count_and_gated_score(self, mocker: MockerFixture) -> None:
        """The printed summary carries the per-class support column and the gated and agnostic column groups."""
        from rich.console import Console

        cb = COCOEvalCallback(min_class_support=10, in_notebook=False)
        overall = {
            "mAP 50:95": 0.4,
            "mAP 50": 0.6,
            "mAP 75": 0.3,
            "mAR @500": 0.5,
            "F1": 0.5,
            "Precision": 0.5,
            "Recall": 0.5,
            "gated mAP 50:95": 0.61,
            "agnostic mAP 50:95": 0.72,
            "agnostic mAP 50": 0.83,
        }
        row = {"name": "cod", "support": 3, "ap": 0.0, "ar": 0.5, "f1": 0.1, "precision": 0.1, "recall": 0.1}
        console = Console(width=200, color_system=None)

        mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console", return_value=console)

        with console.capture() as capture:
            cb._print_metrics_tables(_make_trainer(), "val", overall, [row])

        printed = capture.get()
        assert "mAP N≥10" in printed
        assert "agnostic mAP" in printed
        assert all(value in printed for value in ("0.6100", "0.7200", "0.8300"))
        assert "Support" in printed

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param("vernier", marks=_requires_vernier),
            "faster_coco_eval",
            "hotcoco",
            "hotcoco_streaming",
            pytest.param("ufcoco", marks=_requires_ufcoco),
        ],
    )
    def test_gated_and_agnostic_scores_from_real_backend(self, backend: str, mocker: MockerFixture) -> None:
        """Every evaluation backend supplies the per-class AP the gate reads and a second, agnostic accumulation.

        The matching data is merged across ranks exactly once per epoch: a second collective in the same position would
        be one more call every rank has to issue in the same order.
        """
        cb = COCOEvalCallback(min_class_support=2, class_agnostic=True, eval_backend=backend, in_notebook=False)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        module = _cpu_module()
        module.model_config.num_classes = 2
        cb.setup(trainer, module, stage="fit")
        cb.on_validation_epoch_start(trainer, module)
        for batch_idx, outputs in enumerate(_two_class_batches()):
            cb.on_validation_batch_end(trainer, module, outputs, None, batch_idx)
        merge = mocker.spy(coco_eval_module, "distributed_merge_matching_data")
        mocker.patch.object(cb, "_print_metrics_tables")

        cb.on_validation_epoch_end(trainer, module)

        assert merge.call_count == 1
        # Class 0 (2 instances, AP 1) clears the floor; class 1 (1 instance, AP 0) does not.
        assert float(trainer.callback_metrics["val/mAP_50_95"]) == pytest.approx(0.5, abs=1e-4)
        assert float(trainer.callback_metrics["val/mAP_50_95_gated"]) == pytest.approx(1.0, abs=1e-4)
        # Collapsed to one class: 2 of 3 ground truths found by 2 perfect detections.
        assert 0.6 < float(trainer.callback_metrics["val/mAP_50_95_agnostic"]) < 0.7
        assert 0.6 < float(trainer.callback_metrics["val/mAP_50_agnostic"]) < 0.7

    def test_ema_only_validation_mirrors_gated_score_onto_primary_key(self, mocker: MockerFixture) -> None:
        """Default eval policy (EMA forward only): the gated EMA score reaches both monitor keys.

        The base metric never updates on this path, so the epoch end merges the matching data ahead of the EMA metrics,
        in the early-return branch; it must still be the one and only merge of the epoch.
        """
        cb = COCOEvalCallback(
            min_class_support=2,
            class_agnostic=True,
            log_per_class_metrics=True,
            eval_backend="faster_coco_eval",
            in_notebook=False,
        )
        trainer = _make_trainer(callbacks=[_ema_callback(MagicMock(name="ema_underlying_model"))])
        trainer.callback_metrics = {}
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.on_validation_epoch_start(trainer, module)
        for batch_idx, outputs in enumerate(_two_class_batches()):
            cb.on_validation_batch_end(trainer, module, outputs, None, batch_idx)
        merge = mocker.spy(coco_eval_module, "distributed_merge_matching_data")

        print_tables = mocker.patch.object(cb, "_print_metrics_tables")

        cb.on_validation_epoch_end(trainer, module)

        assert merge.call_count == 1
        assert float(trainer.callback_metrics["val/ema_mAP_50_95_gated"]) == pytest.approx(1.0, abs=1e-4)
        assert float(trainer.callback_metrics["val/mAP_50_95_gated"]) == pytest.approx(1.0, abs=1e-4)
        assert float(trainer.callback_metrics["val/mAP_50_95"]) == pytest.approx(0.5, abs=1e-4)
        assert "val/mAP_50_95_agnostic" in trainer.callback_metrics
        overall, rows = print_tables.call_args.args[2:4]
        assert overall["gated mAP 50:95"] == pytest.approx(1.0, abs=1e-4)
        assert "agnostic mAP 50:95" in overall
        assert sorted(row["support"] for row in rows) == [1, 2]

    def test_segmentation_logs_gated_mask_map(self) -> None:
        """Segmentation gates the mask mAP too, since that is what best-model selection monitors there."""
        cb = COCOEvalCallback(min_class_support=10, segmentation=True)
        metrics = _minimal_metrics(pfx="bbox_")
        metrics.update(
            {
                "segm_map": torch.tensor(0.2),
                "segm_map_50": torch.tensor(0.3),
                "classes": torch.tensor([0, 1]),
                "bbox_map_per_class": torch.tensor([0.6, 0.0]),
                "segm_map_per_class": torch.tensor([0.5, 0.1]),
                "bbox_mar_500_per_class": torch.tensor([0.5, 0.5]),
            }
        )
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        _, module = _run_compute_and_log(cb, merged, metrics)

        assert _logged(module, "val/mAP_50_95_gated") == pytest.approx(0.6)
        assert _logged(module, "val/segm_mAP_50_95_gated") == pytest.approx(0.5)

    def test_crowd_only_class_is_not_counted_by_a_real_backend(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A class annotated only as a crowd region has no ground truth to measure: AP -1, support 0, never averaged.

        Counting its crowd rows as support would put an all-zero class into the macro F1 (0.5 here, not 1.0), and
        averaging its -1 into the headline would pull ``val/mAP_50_95`` under 1.0.
        """
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        cb = COCOEvalCallback(min_class_support=1, eval_backend="faster_coco_eval", in_notebook=False)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        module = _cpu_module()
        module.model_config.num_classes = 2
        cb.setup(trainer, module, stage="fit")
        cb.on_validation_epoch_start(trainer, module)
        for batch_idx, outputs in enumerate(_crowd_only_batches()):
            cb.on_validation_batch_end(trainer, module, outputs, None, batch_idx)
        mocker.patch.object(cb, "_print_metrics_tables")

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            cb.on_validation_epoch_end(trainer, module)

        assert float(trainer.callback_metrics["val/mAP_50_95"]) == pytest.approx(1.0, abs=1e-4)
        assert float(trainer.callback_metrics["val/mAP_50_95_gated"]) == pytest.approx(1.0, abs=1e-4)
        assert float(trainer.callback_metrics["val/F1"]) == pytest.approx(1.0, abs=1e-4)
        # Class 0 clears the floor, so nothing fell back.
        assert not any("min_class_support" in record.getMessage() for record in caplog.records)

    def test_segmentation_ema_only_epoch_mirrors_gated_mask_map(self, mocker: MockerFixture) -> None:
        """EMA-only segmentation epoch: the gated mask mAP reaches both the ``ema_`` and the primary monitor keys."""
        cb = COCOEvalCallback(max_dets=500, min_class_support=10, segmentation=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        ema_metrics = _minimal_metrics(pfx="bbox_")
        ema_metrics.update(
            {
                "segm_map": torch.tensor(0.2),
                "segm_map_50": torch.tensor(0.3),
                "classes": torch.tensor([0, 1]),
                "bbox_map_per_class": torch.tensor([0.6, 0.0]),
                "segm_map_per_class": torch.tensor([0.5, 0.1]),
                "bbox_mar_500_per_class": torch.tensor([0.5, 0.5]),
            }
        )
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        mocker.patch.object(cb, "_print_metrics_tables")
        mocker.patch(
            "rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data",
            return_value={0: _matching_entry(100), 1: _matching_entry(3)},
        )

        cb.on_validation_epoch_end(trainer, module)

        for key in ("val/ema_segm_mAP_50_95_gated", "val/segm_mAP_50_95_gated"):
            assert float(trainer.callback_metrics[key]) == pytest.approx(0.5), key
        for key in ("val/ema_mAP_50_95_gated", "val/mAP_50_95_gated"):
            assert float(trainer.callback_metrics[key]) == pytest.approx(0.6), key

    def test_test_split_logs_gated_and_agnostic_scores_under_its_own_prefix(self) -> None:
        """The test loop gates and scores agnostically like validation, without a ``val/`` key bleeding in."""
        cb = COCOEvalCallback(min_class_support=10, class_agnostic=True)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        agnostic = {"map": torch.tensor(0.7), "map_50": torch.tensor(0.8)}
        trainer, module = _run_compute_and_log(
            cb, merged, _metrics_with_per_class({0: 0.6, 1: 0.0}), split="test", agnostic=agnostic
        )

        assert _logged(module, "test/mAP_50_95_gated") == pytest.approx(0.6)
        assert float(trainer.callback_metrics["test/mAP_50_95_gated"]) == pytest.approx(0.6)
        assert float(trainer.callback_metrics["test/mAP_50_95_agnostic"]) == pytest.approx(0.7)
        assert float(trainer.callback_metrics["test/mAP_50_agnostic"]) == pytest.approx(0.8)
        assert not [c.args[0] for c in module.log.call_args_list if c.args[0].startswith("val/")]


class TestClassAgnosticEval:
    """Label-collapsed evaluation scores localisation independently of classification."""

    @pytest.mark.parametrize("segmentation", [False, True])
    def test_metric_is_created_when_enabled(self, segmentation: bool) -> None:
        """The agnostic metric exists after setup and scores boxes only, for segmentation models too."""
        cb = COCOEvalCallback(class_agnostic=True, segmentation=segmentation)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        assert isinstance(cb.map_metric_agnostic, OnePassCocoMeanAveragePrecision)
        assert cb.map_metric_agnostic.iou_type == ("bbox",)

    def test_metric_is_absent_by_default(self) -> None:
        """Without class_agnostic there is no second metric to accumulate or compute."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        assert cb.map_metric_agnostic is None

    @pytest.mark.parametrize("hook", ["on_validation_batch_end", "on_test_batch_end"])
    def test_batch_end_updates_the_agnostic_metric(self, hook: str) -> None:
        """Each val/test batch feeds the agnostic metric alongside the class-aware one."""
        cb = COCOEvalCallback(class_agnostic=True)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_agnostic = MagicMock(name="map_metric_agnostic")
        outputs = {"results": _detection_preds(2), "targets": _detection_targets()}

        getattr(cb, hook)(_make_trainer(), _cpu_module(), outputs, None, 0)

        cb.map_metric_agnostic.update.assert_called_once()
        _, targets = cb.map_metric_agnostic.update.call_args.args
        assert targets[0]["labels"].tolist() == [0]

    def test_query_scored_under_two_labels_counts_once(self, mocker: MockerFixture) -> None:
        """A query the top-k emits under two labels is one agnostic detection, not a hit plus a false positive."""
        cb = COCOEvalCallback(class_agnostic=True, eval_backend="faster_coco_eval", in_notebook=False)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        module = _cpu_module()
        module.model_config.num_classes = 2
        cb.setup(trainer, module, stage="fit")
        cb.on_validation_epoch_start(trainer, module)
        first, second = [90.0, 45.0, 110.0, 55.0], [30.0, 15.0, 50.0, 25.0]
        outputs = {
            # The first query's box is selected under both labels; its weaker row outranks the second query.
            "results": [
                {
                    "boxes": torch.tensor([first, first, second]),
                    "scores": torch.tensor([0.9, 0.8, 0.7]),
                    "labels": torch.tensor([0, 1, 1]),
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]]),
                    "labels": torch.tensor([0, 1]),
                    "orig_size": torch.tensor([100, 200]),
                }
            ],
        }
        cb.on_validation_batch_end(trainer, module, outputs, None, 0)
        mocker.patch.object(cb, "_print_metrics_tables")

        cb.on_validation_epoch_end(trainer, module)

        assert float(trainer.callback_metrics["val/mAP_50_95_agnostic"]) == pytest.approx(1.0, abs=1e-4)
        # The class-aware metric still sees every row: class 0 AP 1, class 1 AP 0.5 (the repeat ranks above its hit).
        assert float(trainer.callback_metrics["val/mAP_50_95"]) == pytest.approx(0.75, abs=1e-2)

    def test_repeated_boxes_keep_their_best_row(self) -> None:
        """Only a box's highest-scoring row survives, whatever order the rows arrive in; other rows are untouched."""
        box, other = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]
        preds = [
            {
                "boxes": torch.tensor([box, other, box, box]),
                "scores": torch.tensor([0.3, 0.5, 0.9, 0.9]),
                "labels": torch.tensor([0, 1, 2, 3]),
            },
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)},
        ]

        kept = COCOEvalCallback._keep_best_row_per_box(preds)

        assert kept[0]["scores"].tolist() == pytest.approx([0.5, 0.9])
        assert kept[0]["labels"].tolist() == [1, 2]
        assert kept[0]["boxes"].tolist() == [other, box]
        assert kept[1]["boxes"].shape == (0, 4)
        assert preds[0]["boxes"].shape == (4, 4)

    def test_no_agnostic_key_logged_without_state(self) -> None:
        """An epoch that accumulated nothing must not log a -1 sentinel as a score."""
        cb = COCOEvalCallback(class_agnostic=True)
        merged = {0: _matching_entry(100)}
        _, module = _run_compute_and_log(cb, merged, _metrics_with_per_class({0: 0.6}))

        assert "val/mAP_50_95_agnostic" not in {c.args[0] for c in module.log.call_args_list}

    def test_train_split_leaves_agnostic_state_alone(self, mocker: MockerFixture) -> None:
        """Train-split metrics must not consume or log the val/test agnostic accumulation."""
        cb = COCOEvalCallback(max_dets=500, class_agnostic=True)
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric_agnostic = MagicMock(name="map_metric_agnostic")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        cb.map_metric_train.has_updates = True
        cb.map_metric_train.compute.return_value = _minimal_metrics()

        mocker.patch.object(cb, "_print_metrics_tables")

        cb._compute_and_log(trainer, module, "train", metric=cb.map_metric_train)

        cb.map_metric_agnostic.compute.assert_not_called()
        cb.map_metric_agnostic.reset.assert_not_called()

    def test_locally_empty_rank_still_merges_when_a_peer_has_updates(self, all_reduce, mocker: MockerFixture) -> None:
        """The agnostic merge is entered on a vote, so a rank with an empty shard stays in step with its peers."""

        def _peer_has_updates(flag, op=None):
            if op is torch.distributed.ReduceOp.MAX:
                flag.fill_(1)

        all_reduce.side_effect = _peer_has_updates
        cb = COCOEvalCallback(max_dets=500, class_agnostic=True)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb.map_metric_agnostic = MagicMock(name="map_metric_agnostic")
        cb.map_metric_agnostic.has_updates = False
        cb.map_metric_agnostic.compute.return_value = _minimal_metrics()

        mocker.patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value={})

        cb._compute_and_log(_make_trainer(), _cpu_module(), "val")

        cb.map_metric_agnostic.merge_distributed_state.assert_called_once()

    def test_collapse_labels_zeroes_every_label(self) -> None:
        """Collapsing rewrites all labels to a single class id."""
        collapsed = COCOEvalCallback._collapse_labels([{"boxes": torch.zeros(3, 4), "labels": torch.tensor([4, 7, 2])}])
        assert torch.equal(collapsed[0]["labels"], torch.zeros(3, dtype=torch.long))

    def test_collapse_labels_leaves_the_source_untouched(self) -> None:
        """Other accumulators read the same dicts, so they must not be mutated."""
        source = [{"boxes": torch.zeros(2, 4), "labels": torch.tensor([5, 6])}]
        COCOEvalCallback._collapse_labels(source)
        assert torch.equal(source[0]["labels"], torch.tensor([5, 6]))

    def test_collapse_labels_drops_masks(self) -> None:
        """The agnostic metric is bbox-only; carrying masks would invite a segm IoU pass."""
        collapsed = COCOEvalCallback._collapse_labels(
            [{"boxes": torch.zeros(1, 4), "labels": torch.tensor([1]), "masks": torch.zeros(1, 8, 8)}]
        )
        assert "masks" not in collapsed[0]

    def test_collapse_labels_keeps_scores_and_crowd_flags(self) -> None:
        """Scores drive the PR curve and crowd flags the ignore rule; both must survive the collapse."""
        collapsed = COCOEvalCallback._collapse_labels(
            [
                {
                    "boxes": torch.zeros(1, 4),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.75]),
                    "iscrowd": torch.tensor([1]),
                }
            ]
        )
        assert torch.equal(collapsed[0]["scores"], torch.tensor([0.75]))
        assert torch.equal(collapsed[0]["iscrowd"], torch.tensor([1]))
