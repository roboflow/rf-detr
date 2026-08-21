# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for COCOEvalCallback."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from rfdetr.evaluation.matching import build_matching_data, merge_matching_data
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pl_module() -> MagicMock:
    """Return a minimal mock LightningModule."""
    return MagicMock(name="pl_module")


def _make_trainer(datamodule=None, callbacks: list[object] | None = None) -> MagicMock:
    """Return a minimal mock Trainer with an optional DataModule."""
    trainer = MagicMock(name="trainer")
    trainer.datamodule = datamodule
    trainer.callbacks = callbacks or []
    return trainer


class _TQDMProgressBar:
    """Minimal progress-bar stand-in for callback detection tests."""


def _detection_preds(n: int = 0) -> list[dict]:
    """Return a list with one per-image prediction dict."""
    return [
        {
            "boxes": torch.zeros(n, 4),
            "scores": torch.zeros(n),
            "labels": torch.zeros(n, dtype=torch.long),
        }
    ]


def _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1, label=1) -> list[dict]:
    """Return a single-image target dict with one box in normalised CxCyWH."""
    return [
        {
            "boxes": torch.tensor([[cx, cy, w, h]]),
            "labels": torch.tensor([label]),
            "orig_size": torch.tensor([100, 200]),  # H=100, W=200
        }
    ]


def _minimal_metrics(pfx: str = "", max_dets: int = 500) -> dict:
    """Return a minimal torchmetrics-style metrics dict."""
    return {
        f"{pfx}map": torch.tensor(0.4),
        f"{pfx}map_50": torch.tensor(0.6),
        f"{pfx}map_75": torch.tensor(0.3),
        f"{pfx}mar_{max_dets}": torch.tensor(0.5),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetup:
    """Setup() creates map_metric with correct configuration."""

    def test_init_defaults_notebook_flag_to_false_without_ipython(self) -> None:
        """Constructor sets _in_notebook=False when IPython import is unavailable."""
        original_import = __import__

        def _import_with_missing_ipython(name: str, *args, **kwargs):
            if name == "IPython":
                raise ImportError("IPython not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_with_missing_ipython):
            cb = COCOEvalCallback(in_notebook=None)

        assert cb._in_notebook is False

    def test_detection_iou_type_is_bbox(self) -> None:
        """Detection mode uses iou_type='bbox'."""
        cb = COCOEvalCallback(max_dets=300, segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert isinstance(cb.map_metric, OnePassCocoMeanAveragePrecision)
        assert isinstance(cb.map_metric_train, OnePassCocoMeanAveragePrecision)
        assert "bbox" in cb.map_metric.iou_type
        assert "segm" not in cb.map_metric.iou_type

    def test_detection_max_detection_thresholds(self) -> None:
        """max_dets is forwarded to max_detection_thresholds."""
        cb = COCOEvalCallback(max_dets=300, segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert 300 in cb.map_metric.max_detection_thresholds

    def test_segmentation_iou_type_includes_segm(self) -> None:
        """Segmentation mode uses iou_type=['bbox','segm']."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert "segm" in cb.map_metric.iou_type

    def test_map_metric_created_on_every_setup_call(self) -> None:
        """Repeated setup() calls replace map_metric (idempotent)."""
        cb = COCOEvalCallback()
        trainer, module = _make_trainer(), _make_pl_module()
        cb.setup(trainer, module, stage="fit")
        first = cb.map_metric
        cb.setup(trainer, module, stage="validate")
        assert cb.map_metric is not first

    def test_detection_uses_faster_coco_eval_backend(self) -> None:
        """Detection mode always uses faster_coco_eval backend to avoid map=-1 bug."""
        cb = COCOEvalCallback(segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric._coco_backend.backend == "faster_coco_eval"

    def test_segmentation_uses_faster_coco_eval_backend(self) -> None:
        """Segmentation mode always uses faster_coco_eval backend."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric._coco_backend.backend == "faster_coco_eval"

    def test_log_per_class_metrics_false_disables_class_metrics_compute(self) -> None:
        """log_per_class_metrics=False must disable torchmetrics per-class computation, not just logging.

        Regression test for #416: skips the expensive per-class MeanAveragePrecision compute (not merely its table
        rendering at epoch end) on both the regular and train-split metrics.
        """
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric.class_metrics is False
        assert cb.map_metric_train.class_metrics is False

    def test_log_per_class_metrics_default_keeps_class_metrics_compute(self) -> None:
        """Default log_per_class_metrics=True keeps per-class AP computation on for both metrics.

        Mirrors the disable-path test's symmetric map_metric_train assertion.
        """
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric.class_metrics is True
        assert cb.map_metric_train.class_metrics is True

    def test_log_per_class_metrics_false_disables_class_metrics_on_ema_metric(self) -> None:
        """The EMA metric mirrors the per-class computation flag."""
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        with patch.object(cb, "_get_ema_callback", return_value=MagicMock()):
            cb._prepare_ema_metric(_make_trainer())
        assert cb.map_metric_ema is not None
        assert cb.map_metric_ema.class_metrics is False

    def test_ema_metric_stays_on_cpu_when_module_uses_an_accelerator(self) -> None:
        """The EMA metric itself remains on CPU because its accumulated state is CPU-only."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")

        with patch.object(cb, "_get_ema_callback", return_value=MagicMock()):
            cb._prepare_ema_metric(_make_trainer())

        assert cb.map_metric_ema is not None
        assert cb.map_metric_ema.device.type == "cpu"

    def test_keypoint_mode_does_not_enable_torchmetrics_keypoint_iou(self) -> None:
        """Keypoint mode must keep torchmetrics on bbox-only iou_type."""
        cb = COCOEvalCallback(segmentation=True)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        cb.setup(_make_trainer(), module, stage="fit")
        assert "bbox" in cb.map_metric.iou_type
        assert "segm" not in cb.map_metric.iou_type
        assert "keypoints" not in cb.map_metric.iou_type

    def test_log_per_class_metrics_true_enables_class_metrics_compute(self) -> None:
        """log_per_class_metrics=True (default) keeps class_metrics compute enabled."""
        cb = COCOEvalCallback(log_per_class_metrics=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric.class_metrics is True


class TestOnFitStart:
    """on_fit_start() populates class names from the datamodule."""

    def test_class_names_loaded_from_datamodule(self) -> None:
        """Class names are taken from trainer.datamodule.class_names."""
        dm = MagicMock()
        dm.class_names = ["cat", "dog"]
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._class_names == ["cat", "dog"]

    def test_no_datamodule_leaves_class_names_empty(self) -> None:
        """Absent datamodule keeps class_names as empty list."""
        trainer = _make_trainer(datamodule=None)
        cb = COCOEvalCallback()
        cb.on_fit_start(trainer, _make_pl_module())
        assert cb._class_names == []

    def test_datamodule_without_class_names_attr_leaves_empty(self) -> None:
        """DataModule without class_names attr keeps class_names empty."""
        dm = MagicMock(spec=[])  # no attributes
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._class_names == []

    def test_cat_id_to_name_uses_label2cat_when_available(self) -> None:
        """When coco.label2cat is present (remap_category_ids=True) the mapping uses 0-based remapped label IDs so class
        names align with predictions."""
        coco = MagicMock()
        coco.cats = {1: {"name": "fish"}, 2: {"name": "shark"}}
        # label2cat: remapped_label → original_cat_id  (cat2label inverse)
        coco.label2cat = {0: 1, 1: 2}
        dataset = MagicMock()
        dataset.coco = coco
        dm = MagicMock()
        dm.class_names = ["fish", "shark"]
        dm._dataset_val = dataset
        dm._dataset_train = None
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        # 0-based label indices must map to names, not original cat IDs
        assert cb._cat_id_to_name == {0: "fish", 1: "shark"}

    def test_cat_id_to_name_falls_back_to_raw_cats_without_label2cat(self) -> None:
        """Without coco.label2cat (standard COCO), original category IDs are used."""
        coco = MagicMock(spec=["cats"])  # no label2cat attribute
        coco.cats = {1: {"name": "fish"}, 2: {"name": "shark"}}
        dataset = MagicMock()
        dataset.coco = coco
        dm = MagicMock()
        dm.class_names = ["fish", "shark"]
        dm._dataset_val = dataset
        dm._dataset_train = None
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._cat_id_to_name == {1: "fish", 2: "shark"}


@pytest.mark.parametrize(
    "hook,stage",
    [
        pytest.param("on_validation_batch_end", "fit", id="val"),
        pytest.param("on_test_batch_end", "test", id="test"),
    ],
)
class TestBatchEndCommon:
    """map_metric accumulation shared by on_validation_batch_end and on_test_batch_end."""

    def test_map_metric_update_called_once_per_batch(self, hook, stage) -> None:
        """map_metric.update is called exactly once per batch."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)

        assert cb.map_metric.update.call_count == 1

    def test_map_metric_update_receives_normalized_inputs(self, hook, stage) -> None:
        """The callback must pass normalized detection inputs to the adapter that owns CPU-state conversion."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)

        called_preds, called_targets = cb.map_metric.update.call_args.args
        assert called_preds[0].keys() == {"boxes", "scores", "labels"}
        assert called_targets[0].keys() == {"boxes", "labels"}
        assert "orig_size" not in called_targets[0]

    def test_f1_accumulator_grows_across_batches(self, hook, stage) -> None:
        """Calling the batch-end hook twice accumulates more GT in F1 state."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets(label=1)}
        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)
        total_after_1 = sum(v["total_gt"] for v in cb._f1_local.values())

        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 1)
        total_after_2 = sum(v["total_gt"] for v in cb._f1_local.values())

        assert total_after_2 == total_after_1 * 2

    def test_targets_converted_before_update(self, hook, stage) -> None:
        """map_metric.update receives targets with absolute xyxy boxes."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        captured = {}

        def _capture_update(preds, targets):
            captured["targets"] = targets

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.update.side_effect = _capture_update

        outputs = {
            "results": _detection_preds(0),
            "targets": _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1),
        }
        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)

        # Expected: CxCyWH(0.5,0.5,0.1,0.1) × scale(W=200,H=100) → xyxy(90,45,110,55)
        boxes = captured["targets"][0]["boxes"]
        assert boxes.shape == (1, 4)
        assert boxes[0, 0].item() == pytest.approx(90.0)
        assert boxes[0, 1].item() == pytest.approx(45.0)
        assert boxes[0, 2].item() == pytest.approx(110.0)
        assert boxes[0, 3].item() == pytest.approx(55.0)


class TestOnTestBatchEnd:
    """Test-loop-specific behaviour of on_test_batch_end."""

    def test_dataloader_idx_param_has_default(self) -> None:
        """on_test_batch_end must accept calls with an explicit dataloader_idx."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        # Must not raise with explicit dataloader_idx=0
        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0, dataloader_idx=0)


class TestValidationBatchEndDeviceRouting:
    """Device split between the CPU-resident mAP metric state and the GPU-original F1/keypoint inputs."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_build_matching_data_still_receives_cuda_tensors_while_map_metric_state_is_cpu(self) -> None:
        """build_matching_data must keep receiving the original CUDA tensors while map_metric's stored state is CPU-
        resident — no CPU-only test can encode this device split.

        A future refactor that passes the CPU-converted ``metric_preds``/``metric_targets`` into ``build_matching_data``
        instead of the GPU originals would silently move ``[N, H, W]`` mask/box IoU onto the accelerator's slower CPU
        sibling, a regression far larger than this PR's sync-removal win, with green CPU-only tests and zero numerical
        change.
        """
        cb = COCOEvalCallback()
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        outputs = {
            "results": [
                {
                    "boxes": torch.zeros(1, 4, device="cuda"),
                    "scores": torch.zeros(1, device="cuda"),
                    "labels": torch.zeros(1, dtype=torch.long, device="cuda"),
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], device="cuda"),
                    "labels": torch.tensor([1], device="cuda"),
                    "orig_size": torch.tensor([100, 200]),
                }
            ],
        }

        with patch(
            "rfdetr.training.callbacks.coco_eval.build_matching_data", wraps=build_matching_data
        ) as matching_spy:
            cb.on_validation_batch_end(trainer, module, outputs, None, 0)

        matching_spy.assert_called_once()
        matching_preds, matching_targets = matching_spy.call_args[0][:2]
        assert all(value.is_cuda for item in matching_preds + matching_targets for value in item.values())
        assert all(t.device.type == "cpu" for t in cb.map_metric.detection_box)


class TestOnTrainBatchEnd:
    """Train-loop-specific behaviour for optional train mAP logging."""

    def test_train_metrics_update_only_when_enabled(self) -> None:
        """on_train_batch_end should accumulate train predictions only with compute_train_metrics=True."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        cb.map_metric_train.update.assert_called_once()

    def test_train_batch_end_delegates_normalized_inputs_to_adapter(self) -> None:
        """Train accumulation must delegate normalized predictions and targets to the adapter exactly once."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        called_preds, called_targets = cb.map_metric_train.update.call_args.args
        assert called_preds[0].keys() == {"boxes", "scores", "labels"}
        assert called_targets[0].keys() == {"boxes", "labels"}

    def test_train_metrics_do_not_use_test_hook(self) -> None:
        """Train mAP must be logged under train/* via the train epoch hook, not through test/* hooks."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        cb.map_metric_train.compute.return_value = _minimal_metrics()
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)

        cb.on_train_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "train/mAP_50_95" in logged_keys
        assert "test/mAP_50_95" not in logged_keys

    def test_train_epoch_end_skips_compute_when_no_train_updates(self) -> None:
        """Train mAP should not call torchmetrics compute() when no train batches updated it."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        cb.map_metric_train.has_updates = False
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)

        cb.on_train_epoch_end(_make_trainer(), module)

        cb.map_metric_train.compute.assert_not_called()
        cb.map_metric_train.reset.assert_called_once()

    def test_validation_start_does_not_clear_train_metric_state(self) -> None:
        """In-fit validation should reset only validation accumulators, leaving train metrics isolated."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="val_map_metric")
        cb.map_metric_train = MagicMock(name="train_map_metric")

        cb.on_validation_epoch_start(_make_trainer(), _make_pl_module())

        cb.map_metric.reset.assert_called_once()
        cb.map_metric_train.reset.assert_not_called()

    def test_train_batch_end_segm_without_masks_skips_metric_update(self) -> None:
        """Segm callback skips map_metric_train.update when preds lack a masks key."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        # _detection_preds returns preds without "masks" — mimics sparse_forward training mode
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        cb.map_metric_train.update.assert_not_called()

    def test_train_batch_end_segm_with_masks_calls_metric_update(self) -> None:
        """Segm callback calls map_metric_train.update when preds include a masks key."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        preds_with_masks = [
            {
                "boxes": torch.zeros(1, 4),
                "scores": torch.zeros(1),
                "labels": torch.zeros(1, dtype=torch.long),
                "masks": torch.zeros(1, 16, 16, dtype=torch.bool),
            }
        ]
        outputs = {"results": preds_with_masks, "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        cb.map_metric_train.update.assert_called_once()

    def test_train_batch_end_segm_empty_preds_falls_through(self) -> None:
        """Segm callback with empty preds list falls through guard and calls update."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        # Empty preds: `preds and ...` short-circuits to False → no early return → update called
        outputs = {"results": [], "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        cb.map_metric_train.update.assert_called_once()

    def test_train_batch_end_skips_accumulation_on_non_matching_epoch(self) -> None:
        """With eval_interval>1, batches on non-interval epochs must not pay accumulation cost.

        Regression guard: previously every train batch ran the CPU-syncing accumulation block even
        on epochs whose accumulated state on_train_epoch_end discards unread, wasting a blocking
        .cpu() sync (and matching-data work) on every skipped epoch.
        """
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = 0  # epoch 1 (1-based) is not divisible by 3
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(trainer, module, outputs, None, 0)

        cb.map_metric_train.update.assert_not_called()

    def test_train_batch_end_accumulates_on_matching_epoch(self) -> None:
        """With eval_interval>1, batches on interval-aligned epochs still accumulate normally."""
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = 2  # epoch 3 (1-based) is divisible by 3
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(trainer, module, outputs, None, 0)

        cb.map_metric_train.update.assert_called_once()


class TestMetricsTablePrinting:
    """Metric table terminal/notebook rendering behavior.

    Covers: terminal (console.print path), Rich-missing warning, teardown
    cleanup, RichProgressBar console routing, notebook in-place updates.
    """

    @pytest.mark.parametrize(
        "split,title_pfx",
        [
            pytest.param("val", "Val", id="val"),
            pytest.param("test", "Test", id="test"),
        ],
    )
    def test_terminal_metrics_tables_print_to_console(self, split: str, title_pfx: str) -> None:
        """Terminal metric tables print directly through the Rich console each epoch."""
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True
        console = MagicMock(name="console")

        with (
            patch("rfdetr.training.callbacks.coco_eval._get_rich_console", return_value=console),
            patch(
                "rfdetr.training.callbacks.coco_eval._render_overall_merged",
                side_effect=["overall-1", "overall-2"],
            ),
            patch("rfdetr.training.callbacks.coco_eval._render_summary_tables") as render_tables,
        ):
            cb._print_metrics_tables(trainer, split, {"mAP": 0.1}, [])
            cb._print_metrics_tables(trainer, split, {"mAP": 0.2}, [])

        assert render_tables.call_count == 2
        assert render_tables.call_args_list[0].args[0] is console
        assert render_tables.call_args_list[0].args[1].startswith(title_pfx)
        assert "(Epoch" in render_tables.call_args_list[0].args[1]
        assert render_tables.call_args_list[0].args[2] == "overall-1"

    def test_missing_rich_warns_once_and_skips_metric_tables(self) -> None:
        """Missing Rich emits one warning and skips noisy table rendering."""
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True

        with (
            patch("rfdetr.training.callbacks.coco_eval._IS_RICH_AVAILABLE", False),
            patch("rfdetr.training.callbacks.coco_eval.logger.warning") as warning,
            patch("rfdetr.training.callbacks.coco_eval._get_rich_console") as get_console,
        ):
            cb._print_metrics_tables(trainer, "val", {"mAP": 0.1}, [])
            cb._print_metrics_tables(trainer, "val", {"mAP": 0.2}, [])

        warning.assert_called_once_with(
            "Rich is not installed; skipping metric table rendering. Install `rich` to enable tables."
        )
        assert cb._missing_rich_warning_emitted is True
        get_console.assert_not_called()

    def test_teardown_releases_notebook_widget(self) -> None:
        """Teardown clears the notebook output widget reference."""
        cb = COCOEvalCallback(in_notebook=True)
        cb._output_widget = MagicMock(name="output_widget")

        cb.teardown(_make_trainer(), _make_pl_module(), "fit")

        assert cb._output_widget is None

    @pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
    def test_teardown_no_op_when_no_widget(self, stage: str) -> None:
        """Teardown does not raise when no output widget was created."""
        cb = COCOEvalCallback(in_notebook=False)
        assert cb._output_widget is None

        cb.teardown(_make_trainer(), _make_pl_module(), stage)

        assert cb._output_widget is None

    def test_terminal_prints_through_rich_progress_bar_console(self) -> None:
        """Metric tables route through RichProgressBar._console when active."""
        # Create a fake callback whose class name is RichProgressBar so
        # _get_rich_console picks it up without importing PTL.
        rich_progress_bar_fake = type("RichProgressBar", (), {})
        rich_console = MagicMock(name="rich_console")
        fake_pb = rich_progress_bar_fake()
        fake_pb._console = rich_console  # type: ignore[attr-defined]

        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer(callbacks=[fake_pb])
        trainer.is_global_zero = True

        with patch(
            "rfdetr.training.callbacks.coco_eval._render_overall_merged",
            return_value="overall",
        ):
            cb._print_metrics_tables(trainer, "val", {"mAP": 0.5}, [])

        rich_console.print.assert_called_once()

    def test_notebook_metrics_tables_reuse_and_clear_output_widget(self) -> None:
        """Notebook metric tables update one output widget instead of appending one table block per epoch."""

        class FakeOutput:
            """Minimal ipywidgets.Output stand-in."""

            def __init__(self) -> None:
                self.clear_output = MagicMock(name="clear_output")
                self.enter_count = 0

            def __enter__(self) -> "FakeOutput":
                self.enter_count += 1
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
                return False

        output_widget = FakeOutput()
        display = MagicMock(name="display")
        widgets_module = ModuleType("ipywidgets")
        widgets_module.Output = MagicMock(return_value=output_widget)
        ipython_module = ModuleType("IPython")
        ipython_module.__path__ = []
        display_module = ModuleType("IPython.display")
        display_module.display = display

        cb = COCOEvalCallback(in_notebook=True)
        trainer = _make_trainer()
        trainer.is_global_zero = True

        with (
            patch.dict(
                sys.modules,
                {
                    "ipywidgets": widgets_module,
                    "IPython": ipython_module,
                    "IPython.display": display_module,
                },
            ),
            patch("rfdetr.training.callbacks.coco_eval._render_overall_merged", side_effect=["overall-1", "overall-2"]),
            patch("rfdetr.training.callbacks.coco_eval._render_summary_tables") as render_summary_tables,
        ):
            cb._print_metrics_tables(trainer, "val", {"mAP": 0.1}, [])
            cb._print_metrics_tables(trainer, "val", {"mAP": 0.2}, [])

        widgets_module.Output.assert_called_once()
        display.assert_called_once_with(output_widget)
        assert cb._output_widget is output_widget
        assert [call.kwargs for call in output_widget.clear_output.call_args_list] == [
            {"wait": True},
            {"wait": True},
        ]
        assert output_widget.enter_count == 2
        assert render_summary_tables.call_count == 2


@pytest.mark.parametrize(
    "stage,hook,prefix",
    [
        pytest.param("fit", "on_validation_epoch_end", "val/", id="val"),
        pytest.param("test", "on_test_epoch_end", "test/", id="test"),
    ],
)
class TestEpochEndCommon:
    """Metric logging and state reset shared by on_validation_epoch_end and on_test_epoch_end."""

    def test_detection_core_metrics_are_logged(self, stage, hook, prefix) -> None:
        """mAP_50_95, mAP_50, mAP_75, mAR are always logged under the correct prefix."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}mAP_50_95" in logged_keys
        assert f"{prefix}mAP_50" in logged_keys
        assert f"{prefix}mAP_75" in logged_keys
        assert f"{prefix}mAR" in logged_keys

    def test_f1_metrics_logged_when_gt_present(self, stage, hook, prefix) -> None:
        """F1, precision, recall are logged when GT exists."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb._f1_local = {
            0: {
                "scores": np.array([0.9], dtype=np.float32),
                "matches": np.array([1], dtype=np.int64),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        }
        module = _make_pl_module()
        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}F1" in logged_keys
        assert f"{prefix}precision" in logged_keys
        assert f"{prefix}recall" in logged_keys

    def test_f1_metrics_zero_when_no_gt(self, stage, hook, prefix) -> None:
        """F1 == 0.0 when no predictions were accumulated (empty epoch)."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        f1_call = next(c for c in module.log.call_args_list if c.args[0] == f"{prefix}F1")
        assert f1_call.args[1] == pytest.approx(0.0)

    def test_state_reset_after_epoch(self, stage, hook, prefix) -> None:
        """map_metric.reset() is called and _f1_local is cleared after epoch end."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb._f1_local = {
            0: {
                "scores": np.array([0.9], dtype=np.float32),
                "matches": np.array([1], dtype=np.int64),
                "ignore": np.array([False]),
                "total_gt": 1,
            }
        }

        getattr(cb, hook)(_make_trainer(), _make_pl_module())

        cb.map_metric.reset.assert_called_once()
        assert cb._f1_local == {}

    def test_segmentation_extra_metrics_logged(self, stage, hook, prefix) -> None:
        """segm_mAP_50_95 and segm_mAP_50 are logged in segmentation mode."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        segm_metrics = _minimal_metrics(pfx="bbox_")
        segm_metrics["segm_map"] = torch.tensor(0.35)
        segm_metrics["segm_map_50"] = torch.tensor(0.55)
        cb.map_metric.compute.return_value = segm_metrics
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}segm_mAP_50_95" in logged_keys
        assert f"{prefix}segm_mAP_50" in logged_keys


class TestKeypointCocoEvalRouting:
    """Tests for keypoint COCO evaluation routing in keypoint mode."""

    def test_coco_evaluator_accepts_keypoint_predictions(self) -> None:
        """Keypoint mode should forward keypoint predictions to COCO evaluator update()."""
        cb = COCOEvalCallback(max_dets=500)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        trainer = _make_trainer()
        cb.setup(trainer, module, stage="fit")

        evaluator = MagicMock(name="keypoint_coco_eval")
        cb._get_or_create_keypoint_oks_metric = MagicMock(return_value=evaluator)  # type: ignore[method-assign]
        outputs = {
            "results": [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                    "keypoints": torch.tensor([[[1.0, 2.0, 0.8]]], dtype=torch.float32),
                }
            ],
            "targets": [{"image_id": torch.tensor([12])}],
        }

        cb._update_keypoint_oks_metric(trainer, outputs, split="val")

        evaluator.update.assert_called_once()
        predictions = evaluator.update.call_args.args[0]
        assert 12 in predictions
        assert "keypoints" in predictions[12]
        assert predictions[12]["keypoints"].shape == (1, 1, 3)

    def test_keypoint_coco_eval_exposes_keypoint_ap_and_ar_metrics(self) -> None:
        """Epoch-end logging should expose keypoint AP and AR metrics from MetricKeypointOKS.compute()."""
        cb = COCOEvalCallback(max_dets=500)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()

        keypoint_metric = MagicMock(name="keypoint_oks_metric")
        keypoint_metric.has_updates = True
        keypoint_metric.compute.return_value = {"map": 0.42, "map_50": 0.72, "map_75": 0.31, "mar": 0.55}
        cb._keypoint_oks_metrics["val"] = keypoint_metric

        cb.on_validation_epoch_end(trainer, module)

        logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        assert "val/keypoint_map_50_95" in logged
        assert "val/keypoint_map_50" in logged
        assert "val/keypoint_map_75" in logged
        assert "val/keypoint_mAR" in logged
        assert float(logged["val/keypoint_map_50_95"]) == pytest.approx(0.42)
        assert float(logged["val/keypoint_map_50"]) == pytest.approx(0.72)
        assert float(logged["val/keypoint_map_75"]) == pytest.approx(0.31)
        assert float(logged["val/keypoint_mAR"]) == pytest.approx(0.55)
        keypoint_log_calls = [call for call in module.log.call_args_list if call.args[0] == "val/keypoint_map_50_95"]
        assert keypoint_log_calls[0].kwargs.get("prog_bar") is True
        assert trainer.callback_metrics["val/keypoint_map_50_95"].item() == pytest.approx(0.42)
        assert trainer.callback_metrics["val/keypoint_map_50"].item() == pytest.approx(0.72)
        assert trainer.callback_metrics["val/keypoint_map_75"].item() == pytest.approx(0.31)
        assert trainer.callback_metrics["val/keypoint_mAR"].item() == pytest.approx(0.55)
        keypoint_metric.compute.assert_called_once()

    def test_keypoint_coco_eval_exposes_ema_keypoint_ap_and_ar_metrics(self) -> None:
        """EMA keypoint epoch-end logging should expose val/ema_keypoint_* metrics."""
        cb = COCOEvalCallback(max_dets=500)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, module, stage="fit")

        keypoint_metric = MagicMock(name="ema_keypoint_oks_metric")
        keypoint_metric.has_updates = True
        keypoint_metric.compute.return_value = {"map": 0.25, "map_50": 0.5, "map_75": 0.2, "mar": 0.45}
        cb._keypoint_oks_metrics["val_ema"] = keypoint_metric

        cb._compute_and_log_keypoint_map("val_ema", module, trainer, log_split="val", metric_prefix="ema_")

        logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        assert "val/ema_keypoint_map_50_95" in logged
        assert "val/ema_keypoint_map_50" in logged
        assert "val/ema_keypoint_map_75" in logged
        assert "val/ema_keypoint_mAR" in logged
        assert float(logged["val/ema_keypoint_map_50_95"]) == pytest.approx(0.25)
        assert float(logged["val/ema_keypoint_map_50"]) == pytest.approx(0.5)
        assert float(logged["val/ema_keypoint_map_75"]) == pytest.approx(0.2)
        assert float(logged["val/ema_keypoint_mAR"]) == pytest.approx(0.45)
        keypoint_log_calls = [
            call for call in module.log.call_args_list if call.args[0] == "val/ema_keypoint_map_50_95"
        ]
        assert keypoint_log_calls[0].kwargs.get("prog_bar") is True
        assert trainer.callback_metrics["val/ema_keypoint_map_50_95"].item() == pytest.approx(0.25)
        assert trainer.callback_metrics["val/ema_keypoint_map_50"].item() == pytest.approx(0.5)
        assert trainer.callback_metrics["val/ema_keypoint_map_75"].item() == pytest.approx(0.2)
        assert trainer.callback_metrics["val/ema_keypoint_mAR"].item() == pytest.approx(0.45)
        keypoint_metric.compute.assert_called_once()

    def test_keypoint_oks_metric_created_with_correct_args(self) -> None:
        """_get_or_create_keypoint_oks_metric must construct MetricKeypointOKS with coco_api and sigmas."""
        cb = COCOEvalCallback(max_dets=500, keypoint_oks_sigmas=[0.05])
        dataset = MagicMock(name="dataset")
        datamodule = MagicMock()
        datamodule._dataset_val = dataset
        datamodule._dataset_test = None
        datamodule._dataset_train = None
        trainer = _make_trainer(datamodule=datamodule)
        coco_api = MagicMock(name="coco_api")

        with (
            patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", return_value=coco_api),
            patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS") as oks_metric_cls,
        ):
            result = cb._get_or_create_keypoint_oks_metric(trainer, split="val")

        assert result is oks_metric_cls.return_value
        oks_metric_cls.assert_called_once_with(coco_api, keypoint_oks_sigmas=[0.05], max_dets=500)

    def test_keypoint_train_eval_uses_train_dataset(self) -> None:
        """Train keypoint mAP must construct MetricKeypointOKS from the train dataset."""
        cb = COCOEvalCallback(max_dets=500, keypoint_oks_sigmas=[0.05])
        train_dataset = MagicMock(name="train_dataset")
        val_dataset = MagicMock(name="val_dataset")
        datamodule = MagicMock()
        datamodule._dataset_train = train_dataset
        datamodule._dataset_val = val_dataset
        datamodule._dataset_test = None
        trainer = _make_trainer(datamodule=datamodule)
        train_coco_api = MagicMock(name="train_coco_api")
        val_coco_api = MagicMock(name="val_coco_api")

        def _get_coco_api(dataset):
            if dataset is train_dataset:
                return train_coco_api
            if dataset is val_dataset:
                return val_coco_api
            return None

        with (
            patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", side_effect=_get_coco_api),
            patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS") as oks_metric_cls,
        ):
            cb._get_or_create_keypoint_oks_metric(trainer, split="train")

        assert oks_metric_cls.call_args.args[0] is train_coco_api

    def test_keypoint_ema_eval_uses_validation_dataset(self) -> None:
        """EMA keypoint mAP must construct MetricKeypointOKS from the validation dataset."""
        cb = COCOEvalCallback(max_dets=500, keypoint_oks_sigmas=[0.05])
        train_dataset = MagicMock(name="train_dataset")
        val_dataset = MagicMock(name="val_dataset")
        datamodule = MagicMock()
        datamodule._dataset_train = train_dataset
        datamodule._dataset_val = val_dataset
        datamodule._dataset_test = None
        trainer = _make_trainer(datamodule=datamodule)
        train_coco_api = MagicMock(name="train_coco_api")
        val_coco_api = MagicMock(name="val_coco_api")

        def _get_coco_api(dataset):
            if dataset is train_dataset:
                return train_coco_api
            if dataset is val_dataset:
                return val_coco_api
            return None

        with (
            patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", side_effect=_get_coco_api),
            patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS") as oks_metric_cls,
        ):
            cb._get_or_create_keypoint_oks_metric(trainer, split="val_ema")

        assert oks_metric_cls.call_args.args[0] is val_coco_api

    def test_mixed_keypoint_counts_create_keypoint_oks_metric(self) -> None:
        """Mixed keypoint counts should be handled by MetricKeypointOKS instead of being skipped."""
        cb = COCOEvalCallback(max_dets=500)
        dataset = MagicMock(name="dataset")
        datamodule = MagicMock()
        datamodule._dataset_train = dataset
        datamodule._dataset_val = None
        datamodule._dataset_test = None
        trainer = _make_trainer(datamodule=datamodule)

        with (
            patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", return_value=MagicMock()),
            patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS") as oks_metric_cls,
            patch("rfdetr.training.callbacks.coco_eval.logger.warning") as warning,
        ):
            result = cb._get_or_create_keypoint_oks_metric(trainer, split="train")

        assert result is oks_metric_cls.return_value
        oks_metric_cls.assert_called_once()
        warning.assert_not_called()


@pytest.mark.parametrize(
    "stage,hook,prefix",
    [
        pytest.param("fit", "on_validation_epoch_end", "val/", id="val"),
        pytest.param("test", "on_test_epoch_end", "test/", id="test"),
    ],
)
class TestPerClassAPLogging:
    """Per-class AP logging behavior for validation and test loops."""

    def test_per_class_ap_logged_when_classes_present(self, stage, hook, prefix) -> None:
        """AP/<name> is logged for each class when class metrics are present."""
        cb = COCOEvalCallback()
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}AP/cat" in logged_keys
        assert f"{prefix}AP/dog" in logged_keys

    def test_per_class_ap_falls_back_to_str_id_when_no_class_names(self, stage, hook, prefix) -> None:
        """AP/<id> is logged when class_names is empty."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5])
        metrics["classes"] = torch.tensor([3])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}AP/3" in logged_keys


class TestOnValidationEpochEnd:
    """Validation-specific behaviour of on_validation_epoch_end."""

    def test_ema_metrics_logged_when_map_metric_ema_populated(self) -> None:
        """val/ema_* metrics are logged when map_metric_ema has accumulated data.

        EMA metrics are now computed from a separate map_metric_ema that is populated during on_validation_batch_end
        (not aliased from base metrics).
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        # Simulate map_metric_ema being populated by on_validation_batch_end.
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        cb.map_metric_ema.reset.assert_called_once()

    def test_eval_ema_only_still_logs_ema_metrics_when_base_metric_is_empty(self) -> None:
        """Regression for #1285: under eval_ema_only, on_validation_batch_end routes every prediction to map_metric_ema
        and map_metric never accumulates a single update this epoch.

        The empty-state guard in _compute_and_log must not also suppress the EMA metrics — before the fix it returned
        before ever reaching the EMA compute block, so eval_ema_only runs logged no validation output at all.
        """
        cb = COCOEvalCallback(max_dets=500, eval_ema_only=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: eval_ema_only never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        cb.map_metric.compute.assert_not_called()
        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        assert not any(k in {"val/mAP_50_95", "val/mAP_50", "val/mAP_75", "val/mAR"} for k in logged_keys)
        cb.map_metric_ema.reset.assert_called_once()
        cb.map_metric.reset.assert_called_once()

    def test_eval_ema_only_computes_f1_from_accumulated_matches_when_base_metric_is_empty(self) -> None:
        """Regression for #1285: on_validation_batch_end merges matching data into f1_local unconditionally (outside the
        used_ema_forward branch), independent of which mAP track a batch's predictions were routed to.

        Under eval_ema_only, map_metric never accumulates, so the empty-state guard in _compute_and_log used to discard
        f1_local via _reset_f1_local before ever computing val/F1 — silently dropping it even though real matching data
        (from the EMA-quality predictions) had been collected.
        """
        cb = COCOEvalCallback(max_dets=500, eval_ema_only=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: eval_ema_only never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True
        # A perfect match (pred == target box/label), mirroring what the unconditional
        # merge_matching_data call in on_validation_batch_end accumulates every batch.
        preds = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]
        merge_matching_data(cb._f1_local, build_matching_data(preds, targets))
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        f1_call = next(c for c in module.log.call_args_list if c.args[0] == "val/F1")
        assert f1_call.args[1] == pytest.approx(1.0)

    def test_eval_ema_only_prints_summary_table_when_base_metric_is_empty(self) -> None:
        """Regression for #1285: the console summary table is also part of "validation output".

        Before this fix, ``_print_metrics_tables`` was only ever called from the branch that reads the base ``metric`` —
        which never has data under ``eval_ema_only`` — so no table was ever printed for the entire run, even after the
        scalar/F1 logging gaps were closed. Per-class AP must be logged under ``ema_``-prefixed keys, mirroring
        ``val/ema_mAP_50_95`` staying separate from ``val/mAP_50_95`` elsewhere, so it never collides with a base-track
        key.
        """
        cb = COCOEvalCallback(max_dets=500, eval_ema_only=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: eval_ema_only never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        ema_metrics = _minimal_metrics()
        ema_metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6, 0.3])
        ema_metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        with patch.object(cb, "_print_metrics_tables") as print_metrics_tables:
            cb.on_validation_epoch_end(_make_trainer(), module)

        print_metrics_tables.assert_called_once()
        _trainer_arg, title_arg, overall_arg, per_class_arg = print_metrics_tables.call_args.args
        assert title_arg == "val (ema)"
        assert overall_arg["mAP 50:95"] == pytest.approx(0.4)
        assert {row["name"] for row in per_class_arg} == {"cat", "dog"}

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_AP/cat" in logged_keys
        assert "val/ema_AP/dog" in logged_keys
        assert "val/AP/cat" not in logged_keys
        assert "val/AP/dog" not in logged_keys

    def test_eval_ema_only_resets_val_ema_keypoints_when_ema_not_yet_warmed_up(self) -> None:
        """Regression for #1289 review: under eval_ema_only, when the base metric is empty AND the EMA metric has not
        accumulated any updates this epoch (e.g. EMA not yet warmed up), _should_compute_ema returns False and the
        early-return branch must fall back to resetting the val_ema keypoint split instead of computing it."""
        cb = COCOEvalCallback(max_dets=500, eval_ema_only=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: eval_ema_only never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = False  # EMA metric also has no updates this epoch
        module = _make_pl_module()

        with (
            patch.object(cb, "_compute_and_log_keypoint_map") as compute_keypoint_map,
            patch.object(cb, "_reset_keypoint_split") as reset_keypoint_split,
        ):
            cb.on_validation_epoch_end(_make_trainer(), module)

        compute_keypoint_map.assert_not_called()
        assert call("val_ema") in reset_keypoint_split.call_args_list
        cb.map_metric_ema.compute.assert_not_called()

    def test_eval_ema_only_summary_table_includes_segm_metrics(self) -> None:
        """Regression for #1289 review: _print_ema_only_summary must include segm mAP entries in the overall table when
        segmentation metrics are enabled, mirroring the base-metric path's segm handling."""
        cb = COCOEvalCallback(max_dets=500, segmentation=True, eval_ema_only=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb._class_names = ["cat"]
        cb._cat_id_to_name = {0: "cat"}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: eval_ema_only never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        ema_metrics = _minimal_metrics(pfx="bbox_")
        ema_metrics["map_per_class"] = torch.tensor([0.5])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6])
        ema_metrics["classes"] = torch.tensor([0])
        ema_metrics["segm_map"] = torch.tensor(0.2)
        ema_metrics["segm_map_50"] = torch.tensor(0.35)
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        with patch.object(cb, "_print_metrics_tables") as print_metrics_tables:
            cb.on_validation_epoch_end(_make_trainer(), module)

        print_metrics_tables.assert_called_once()
        _trainer_arg, title_arg, overall_arg, _per_class_arg = print_metrics_tables.call_args.args
        assert title_arg == "val (ema)"
        assert overall_arg["segm mAP 50:95"] == pytest.approx(0.2)
        assert overall_arg["segm mAP 50"] == pytest.approx(0.35)

    def test_eval_interval_skips_non_matching_epochs(self) -> None:
        """Validation metric computation is skipped on non-interval epochs."""
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = 0  # epoch 1 (1-based) is not divisible by 3
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        cb.map_metric.compute.assert_not_called()
        cb.map_metric.reset.assert_called_once()
        module.log.assert_not_called()

    def test_eval_interval_runs_on_matching_epochs(self) -> None:
        """Validation metric computation runs on interval-aligned epochs."""
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = 2  # epoch 3 (1-based) is divisible by 3
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        cb.map_metric.compute.assert_called_once()
        module.log.assert_called()

    def test_progress_bar_suppresses_duplicate_pycocotools_output(self, capsys) -> None:
        """Progress-bar training suppresses duplicate pycocotools stdout but still prints metric tables."""
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer(callbacks=[_TQDMProgressBar()])
        trainer.callback_metrics = {}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = None
        module = _make_pl_module()

        def _compute_with_terminal_summary() -> dict:
            print("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = 0.000")
            return _minimal_metrics()

        cb.map_metric.compute.side_effect = _compute_with_terminal_summary

        with patch.object(cb, "_print_metrics_tables") as print_metrics_tables:
            cb._compute_and_log(trainer, module, "val")

        assert "Average Precision" not in capsys.readouterr().out
        print_metrics_tables.assert_called_once()
        cb.map_metric.compute.assert_called_once()
        module.log.assert_called()

    def test_per_class_ap_can_be_disabled(self) -> None:
        """log_per_class_metrics=False suppresses val/AP/<class> logging."""
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("val/AP/") for k in logged_keys)

    def test_per_class_ap_disabled_does_not_crash_on_torchmetrics_faithful_ar_shape(self) -> None:
        """Regression test for H1: log_per_class_metrics=False must not crash epoch-end validation.

        torchmetrics still emits ``mar_{max_dets}_per_class`` as a 0-dim ``tensor(-1.)`` even when
        ``class_metrics=False`` (only ``map_per_class`` is actually suppressed), while ``classes`` stays 1-d for >=2
        classes. Before the fix, the per-class AR block was gated on key-presence only (not on the flag), so
        ``zip(classes, mar_..._per_class)`` iterated a 0-d tensor and raised ``TypeError: iteration over a 0-d tensor``.
        ``_minimal_metrics()`` alone (used by the sibling ``test_per_class_ap_can_be_disabled``) omits this key entirely
        and would not catch the regression.
        """
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["classes"] = torch.tensor([0, 1])
        metrics["mar_500_per_class"] = torch.tensor(-1.0)  # torchmetrics' real 0-d shape when class_metrics=False
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)  # must not raise TypeError

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("val/AP/") for k in logged_keys)

    def test_callback_metrics_updated_for_model_checkpoint(self) -> None:
        """Core metrics written to trainer.callback_metrics each epoch so ModelCheckpoint / BestModelCallback detect
        improvement.

        pl_module.log() from a callback's on_validation_epoch_end goes only to logged_metrics (external loggers), not
        callback_metrics.
        """
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()

        cb.on_validation_epoch_end(trainer, _make_pl_module())

        assert "val/mAP_50_95" in trainer.callback_metrics
        assert "val/mAP_50" in trainer.callback_metrics
        assert "val/mAP_75" in trainer.callback_metrics
        assert "val/mAR" in trainer.callback_metrics
        assert trainer.callback_metrics["val/mAP_50_95"].item() == pytest.approx(0.4)
        assert trainer.callback_metrics["val/mAP_50"].item() == pytest.approx(0.6)

    def test_callback_metrics_updated_with_ema_when_map_metric_ema_populated(self) -> None:
        """EMA metrics are written to callback_metrics when map_metric_ema has data."""
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True

        cb.on_validation_epoch_end(trainer, _make_pl_module())

        assert "val/ema_mAP_50_95" in trainer.callback_metrics
        assert "val/ema_mAP_50" in trainer.callback_metrics
        assert "val/ema_mAR" in trainer.callback_metrics

    def test_ema_segm_metrics_use_ema_values_not_base(self) -> None:
        """EMA segmentation metrics must come from map_metric_ema, not the base map_metric.

        Regression test for #978.
        """
        cb = COCOEvalCallback(max_dets=500, segmentation=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")

        # Base metrics: segm_map=0.35
        base_metrics = _minimal_metrics(pfx="bbox_")
        base_metrics["segm_map"] = torch.tensor(0.35)
        base_metrics["segm_map_50"] = torch.tensor(0.55)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = base_metrics

        # EMA metrics: segm_map=0.45 (deliberately different)
        ema_metrics = _minimal_metrics(pfx="bbox_")
        ema_metrics["segm_map"] = torch.tensor(0.45)
        ema_metrics["segm_map_50"] = torch.tensor(0.65)
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        # EMA segm values must differ from base
        assert trainer.callback_metrics["val/ema_segm_mAP_50_95"].item() == pytest.approx(0.45)
        assert trainer.callback_metrics["val/ema_segm_mAP_50"].item() == pytest.approx(0.65)
        # Base segm values unchanged
        assert trainer.callback_metrics["val/segm_mAP_50_95"].item() == pytest.approx(0.35)
        assert trainer.callback_metrics["val/segm_mAP_50"].item() == pytest.approx(0.55)
        # pl_module.log() must also receive EMA values (covers both changed code paths)
        logged = {c.args[0]: c.args[1] for c in module.log.call_args_list if len(c.args) >= 2}
        assert logged["val/ema_segm_mAP_50_95"].item() == pytest.approx(0.45)
        assert logged["val/ema_segm_mAP_50"].item() == pytest.approx(0.65)

    def test_ghost_class_with_negative_ar_sentinel_is_filtered(self) -> None:
        """A class where both ap=-1 and ar=-1 (negative sentinels, not NaN) must be excluded from the per-class table.

        The old filter checked for NaN only, so ar=-1 (a valid float) escaped the guard.
        """
        cb = COCOEvalCallback()
        cb._cat_id_to_name = {0: "fish"}
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        # class 0 is a real class; class 8 is a ghost with both sentinels = -1
        metrics["map_per_class"] = torch.tensor([0.5, -1.0])
        metrics["classes"] = torch.tensor([0, 8])
        # ar=-1 for ghost (negative sentinel, not NaN)
        metrics["mar_500_per_class"] = torch.tensor([0.6, -1.0])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        # real class logged, ghost class suppressed
        assert "val/AP/fish" in logged_keys
        assert "val/AP/8" not in logged_keys


# ---------------------------------------------------------------------------
# Test-epoch-end-only behaviour
# ---------------------------------------------------------------------------


class TestOnTestEpochEnd:
    """Test-loop-specific behaviour of on_test_epoch_end."""

    def test_no_ema_aliases_for_test(self) -> None:
        """test/ema_* aliases are NOT logged — test always runs with EMA weights via the RFDETREMACallback swap so
        test/mAP_50 is already the EMA result."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("test/ema_") for k in logged_keys)

    def test_val_prefix_not_logged(self) -> None:
        """test_epoch_end must not emit val/ keys — prefixes must not bleed across loops."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("val/") for k in logged_keys)


class TestValidationBatchEndEvalEmaOnly:
    """eval_ema_only=True must skip the duplicate EMA forward pass in on_validation_batch_end — validation_step already
    forwarded through the EMA model directly (regression for #416)."""

    @staticmethod
    def _ema_callback_with_underlying(ema_underlying: MagicMock) -> MagicMock:
        """Return an EMA callback mock wired so _get_ema_inner_module(cb).model is ema_underlying."""
        ema_cb = MagicMock(name="ema_callback")
        ema_cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
        ema_cb._average_model = SimpleNamespace(module=SimpleNamespace(model=ema_underlying))
        return ema_cb

    def test_eval_ema_only_true_skips_duplicate_ema_forward(self) -> None:
        """eval_ema_only=True must never call the EMA model's forward a second time, and must route the single forward's
        predictions to the EMA metric/checkpoint track (not the regular one, which never ran a base-model forward this
        batch — regression for the val/mAP_50_95 metric/checkpoint-weights mismatch)."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_ema_only=True)
        trainer = _make_trainer(callbacks=[self._ema_callback_with_underlying(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_not_called()
        cb.map_metric_ema.update.assert_called_once()
        cb.map_metric.update.assert_not_called()
        assert cb._ema_has_updates is True

    def test_eval_ema_only_true_falls_back_to_regular_track_when_ema_not_warmed_up(self) -> None:
        """eval_ema_only=True with no averaged EMA model yet available must route predictions to the regular map_metric,
        mirroring RFDETRModelModule._resolve_eval_model's own base-model fallback — otherwise the EMA track would record
        an update that never actually came from an EMA forward pass."""
        cb = COCOEvalCallback(eval_ema_only=True)
        ema_cb = MagicMock(name="ema_callback")
        ema_cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
        ema_cb._average_model = None
        trainer = _make_trainer(callbacks=[ema_cb])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        cb.map_metric.update.assert_called_once()
        cb.map_metric_ema.update.assert_not_called()
        assert cb._ema_has_updates is False

    def test_eval_ema_only_false_still_runs_duplicate_ema_forward(self) -> None:
        """eval_ema_only=False (default) must preserve the existing independent base+EMA forward behaviour."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_ema_only=False)
        trainer = _make_trainer(callbacks=[self._ema_callback_with_underlying(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_called_once()
        cb.map_metric_ema.update.assert_called_once()

    def test_eval_ema_only_true_routes_normalized_inputs_to_ema_track(self) -> None:
        """The used-EMA route must delegate normalized inputs only to the EMA adapter."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_ema_only=True)
        trainer = _make_trainer(callbacks=[self._ema_callback_with_underlying(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        cb.map_metric.update.assert_not_called()
        called_preds, called_targets = cb.map_metric_ema.update.call_args.args
        assert called_preds[0].keys() == {"boxes", "scores", "labels"}
        assert called_targets[0].keys() == {"boxes", "labels"}

    def test_eval_ema_only_false_routes_each_forward_to_its_metric_adapter(self) -> None:
        """Independent base and EMA forwards must each update their corresponding adapter once."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_ema_only=False)
        trainer = _make_trainer(callbacks=[self._ema_callback_with_underlying(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        cb.map_metric.update.assert_called_once()
        cb.map_metric_ema.update.assert_called_once()

    def test_on_validation_batch_end_resizes_native_gt_directly_to_prediction_grid(self) -> None:
        """Native-head evaluation must not round-trip transformed GT masks through ``orig_size``.

        A 4x4 model-input mask with one pixel at ``[1, 1]`` is empty after direct nearest downsampling to 2x2. The
        historical 4x4 -> 5x5 -> 2x2 path instead retains that pixel, changing the metric target for the same model
        output.
        """
        cb = COCOEvalCallback(segmentation=True)
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")

        pred_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        gt_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt_masks[0, 1, 1] = True
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": pred_masks,
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([5, 5]),
                    "masks": gt_masks,
                }
            ],
        }

        cb.on_validation_batch_end(trainer, module, outputs, None, 0)

        called_targets = cb.map_metric.update.call_args[0][1]
        assert torch.equal(called_targets[0]["masks"], torch.zeros_like(pred_masks))

    def test_ema_validation_resizes_gt_to_ema_prediction_grid(self) -> None:
        """EMA segmentation metrics must use the EMA predictions' grid, not the base model's grid."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(segmentation=True)
        trainer = _make_trainer(callbacks=[self._ema_callback_with_underlying(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        base_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        ema_masks = torch.zeros(1, 3, 3, dtype=torch.bool)
        gt_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt_masks[0, 1, 1] = True
        module.postprocess.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.zeros(1, 4),
                "masks": ema_masks,
            }
        ]
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": base_masks,
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([5, 5]),
                    "masks": gt_masks,
                }
            ],
        }

        cb.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        expected_masks = torch.zeros_like(ema_masks)
        expected_masks[0, 1, 1] = True
        called_targets = cb.map_metric_ema.update.call_args[0][1]
        assert torch.equal(called_targets[0]["masks"], expected_masks)

    def test_eval_ema_only_and_native_resolution_masks_combined(self) -> None:
        """eval_ema_only=True and native-resolution mask predictions active together (the #416 commenter's actual
        workaround) must route to map_metric_ema with GT masks aligned to the native pred resolution — neither flag's
        handling should interfere with the other."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_ema_only=True, segmentation=True)
        trainer = _make_trainer(callbacks=[self._ema_callback_with_underlying(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        pred_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt_masks = torch.ones(1, 8, 8, dtype=torch.bool)
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": pred_masks,
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([8, 8]),
                    "masks": gt_masks,
                }
            ],
        }
        batch = (torch.zeros(1), None)

        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_not_called()
        cb.map_metric.update.assert_not_called()
        called_targets = cb.map_metric_ema.update.call_args[0][1]
        assert called_targets[0]["masks"].shape[-2:] == (4, 4)


class TestValidationBatchEndTargetConversion:
    """Independent base and EMA updates convert ``outputs["targets"]`` once when both conversions must agree."""

    @staticmethod
    def _trainer_with_ema(ema_underlying: MagicMock) -> MagicMock:
        """Return a trainer whose EMA callback exposes ``ema_underlying`` as the averaged model.

        Examples:
            >>> underlying = MagicMock(name="ema_underlying_model")
            >>> trainer = TestValidationBatchEndTargetConversion._trainer_with_ema(underlying)
            >>> trainer.callbacks[0]._average_model.module.model is underlying
            True
        """
        ema_cb = MagicMock(name="ema_callback")
        ema_cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
        ema_cb._average_model = SimpleNamespace(module=SimpleNamespace(model=ema_underlying))
        return _make_trainer(callbacks=[ema_cb])

    def test_detection_hands_one_converted_target_list_to_both_metrics(self) -> None:
        """Detection validation converts the batch's targets once and reuses that list for the EMA update.

        Both call sites pass ``preds=None`` outside segmentation, so the second conversion recomputed a
        byte-identical result: the same box rescale and the same ``orig_size`` host transfer, per validation batch,
        on every epoch that evaluates both models. Reuse is safe because the accumulators store detached CPU copies
        of what they are handed rather than the caller's tensors.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback()
        trainer = self._trainer_with_ema(ema_underlying)
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        cb.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        assert cb.map_metric_ema.update.call_args.args[1] is cb.map_metric.update.call_args.args[1]

    def test_segmentation_converts_targets_separately_for_the_ema_grid(self) -> None:
        """Segmentation validation must keep converting targets twice, once per prediction grid.

        Target masks are resampled onto the paired prediction's mask resolution, and the base and EMA heads can emit
        different resolutions for the same image. Sharing the base conversion here would silently score the EMA masks
        against a grid they were never produced on.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(segmentation=True)
        trainer = self._trainer_with_ema(ema_underlying)
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        module.postprocess.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.zeros(1, 4),
                "masks": torch.zeros(1, 3, 3, dtype=torch.bool),
            }
        ]
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": torch.zeros(1, 2, 2, dtype=torch.bool),
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([5, 5]),
                    "masks": torch.zeros(1, 4, 4, dtype=torch.bool),
                }
            ],
        }

        cb.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        ema_targets = cb.map_metric_ema.update.call_args.args[1]
        assert ema_targets is not cb.map_metric.update.call_args.args[1]
        assert ema_targets[0]["masks"].shape[-2:] == (3, 3)


class TestEvalEmaOnlyBatchToEpochEndToEnd:
    """Full batch -> epoch pipeline for eval_ema_only, with real (non-mocked) accumulators (#1285).

    TestValidationBatchEndEvalEmaOnly and TestOnValidationEpochEnd each replace map_metric / map_metric_ema with
    MagicMock and drive a single hook in isolation. Neither would catch a wiring bug between the two (e.g.
    `_ema_has_updates` set by batch-time routing never reaching epoch-end, or `_prepare_ema_metric` creating a
    `map_metric_ema` that batch-time routing never populates). This test keeps the real MeanAveragePrecision instances
    created by setup() / on_validation_epoch_start() and drives on_validation_batch_end then on_validation_epoch_end in
    sequence on the same callback instance.
    """

    def test_eval_ema_only_real_batch_then_epoch_end_logs_ema_metrics(self) -> None:
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        ema_cb = MagicMock(name="ema_callback")
        ema_cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
        ema_cb._average_model = SimpleNamespace(module=SimpleNamespace(model=ema_underlying))
        cb = COCOEvalCallback(max_dets=500, eval_ema_only=True)
        trainer = _make_trainer(callbacks=[ema_cb])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.on_validation_epoch_start(trainer, module)  # real map_metric_ema creation (_prepare_ema_metric)

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_not_called()  # eval_ema_only routes validation_step's own forward, no duplicate
        assert cb._ema_has_updates is True
        assert cb.map_metric._update_count == 0  # regular track never accumulated this epoch

        cb.on_validation_epoch_end(trainer, module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        assert not any(k in {"val/mAP_50_95", "val/mAP_50", "val/mAP_75", "val/mAR"} for k in logged_keys)


class TestConvertPreds:
    """_convert_preds() normalizes prediction dicts for metric consumers."""

    @pytest.mark.parametrize(
        ("boxes", "expected_kept_idxs"),
        [
            pytest.param(
                # Degenerate first -> keep original index 1 (non-zero keep idx).
                [[2.0, 2.0, 2.0, 4.0], [0.0, 0.0, 3.0, 3.0], [5.0, 5.0, 5.0, 7.0]],
                [1],
                id="degenerate-first-keeps-index-1",
            ),
            pytest.param(
                # Degenerate between valid boxes -> keep non-contiguous original indices.
                [[0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 2.0, 4.0], [4.0, 4.0, 6.0, 6.0]],
                [0, 2],
                id="degenerate-middle-keeps-noncontiguous",
            ),
        ],
    )
    def test_masks_remain_aligned_with_original_indices_after_degenerate_filtering(
        self,
        boxes: list[list[float]],
        expected_kept_idxs: list[int],
    ) -> None:
        """Filtering degenerate boxes must preserve mask alignment via original indices.

        Regression context: when a degenerate box is not last, keep indices are non-zero/non-contiguous. Downstream
        filtering must keep masks from the same original prediction indices.
        """
        cb = COCOEvalCallback()

        # Distinct one-hot masks so index/mask misalignment is easy to detect.
        masks = torch.zeros(3, 1, 2, 2, dtype=torch.bool)
        masks[0, 0, 0, 0] = True
        masks[1, 0, 0, 1] = True
        masks[2, 0, 1, 0] = True

        preds = [
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                "labels": torch.tensor([0, 0, 0], dtype=torch.int64),
                "masks": masks,
            }
        ]

        out = cb._convert_preds(preds)
        out_boxes = out[0]["boxes"]
        out_masks = out[0]["masks"]
        assert out_masks.shape == (3, 2, 2)

        keep = torch.where((out_boxes[:, 2] > out_boxes[:, 0]) & (out_boxes[:, 3] > out_boxes[:, 1]))[0]
        assert keep.tolist() == expected_kept_idxs
        assert torch.equal(out_masks[keep], masks.squeeze(1)[keep])


class TestConvertTargets:
    """_convert_targets() converts normalised CxCyWH to absolute xyxy."""

    def test_box_conversion_known_values(self) -> None:
        """CxCyWH(0.5,0.5,0.4,0.6) × (W=100,H=200) → xyxy(30,40,70,160)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([200, 100]),  # H=200, W=100
            }
        ]
        out = cb._convert_targets(targets)
        boxes = out[0]["boxes"]
        # cx=0.5*100=50, cy=0.5*200=100, w=0.4*100=40, h=0.6*200=120
        # → x1=50-20=30, y1=100-60=40, x2=50+20=70, y2=100+60=160
        assert boxes[0, 0].item() == pytest.approx(30.0)
        assert boxes[0, 1].item() == pytest.approx(40.0)
        assert boxes[0, 2].item() == pytest.approx(70.0)
        assert boxes[0, 3].item() == pytest.approx(160.0)

    def test_labels_passed_through(self) -> None:
        """Labels tensor is preserved unchanged."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert out[0]["labels"][0].item() == 7

    def test_masks_passed_through_as_bool(self) -> None:
        """Masks tensor is cast to bool and included in output."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([8, 8]),
                "masks": torch.ones(1, 8, 8, dtype=torch.uint8),
            }
        ]
        out = cb._convert_targets(targets)
        assert "masks" in out[0]
        assert out[0]["masks"].dtype == torch.bool

    def test_masks_resize_to_original_grid_without_predictions(self) -> None:
        """Default full-resolution evaluation keeps the original-image mask grid."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([5, 5]),
                "masks": torch.ones(1, 4, 4, dtype=torch.uint8),
            }
        ]

        out = cb._convert_targets(targets)

        assert out[0]["masks"].shape == (1, 5, 5)

    def test_iscrowd_passed_through(self) -> None:
        """Iscrowd tensor is included when present."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
                "iscrowd": torch.tensor([1]),
            }
        ]
        out = cb._convert_targets(targets)
        assert "iscrowd" in out[0]
        assert out[0]["iscrowd"][0].item() == 1

    def test_no_masks_no_iscrowd_keys_absent(self) -> None:
        """Output dict contains exactly boxes and labels when extras are absent."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert set(out[0].keys()) == {"boxes", "labels"}

    @pytest.mark.parametrize(
        ("num_targets", "expected_calls"),
        [
            pytest.param(3, [(3, 2)], id="multi-target-batch"),
            pytest.param(0, [], id="empty-targets"),
        ],
    )
    def test_orig_size_is_read_once_per_batch(self, num_targets: int, expected_calls: list[tuple[int, ...]]) -> None:
        """A non-empty target batch reads every orig_size together in one device-to-host synchronization instead of one
        ``tolist()`` call per target inside the loop."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
            }
            for _ in range(num_targets)
        ]
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        with patch.object(torch.Tensor, "tolist", tracked_tolist):
            cb._convert_targets(targets)

        assert calls == expected_calls

    def test_batch_pairs_non_square_orig_sizes_with_their_target_rows(self) -> None:
        """Each target's box uses its own non-square orig_size after batched size conversion.

        This prevents swapped ``orig_size.tolist()`` rows and height/width transposition from assigning a sibling
        image's scale to the wrong target.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.4]]),
                "labels": torch.tensor([3]),
                "orig_size": torch.tensor([100, 300]),  # H=100, W=300
            },
            {
                "boxes": torch.tensor([[0.5, 0.25, 0.4, 0.2]]),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([240, 80]),  # H=240, W=80
            },
        ]

        out = cb._convert_targets(targets)

        assert out[0]["boxes"].shape == (1, 4)
        assert out[1]["boxes"].shape == (1, 4)
        assert out[0]["labels"].item() == 3
        assert out[1]["labels"].item() == 7
        torch.testing.assert_close(out[0]["boxes"], torch.tensor([[45.0, 30.0, 105.0, 70.0]]))
        torch.testing.assert_close(out[1]["boxes"], torch.tensor([[24.0, 36.0, 56.0, 84.0]]))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_orig_size_is_read_once_per_batch_cuda(self) -> None:
        """Same call-count guarantee as ``test_orig_size_is_read_once_per_batch`` above, with every target tensor on
        CUDA — the actual site of the device-to-host synchronization this PR collapses from one per target to one per
        batch."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4, device="cuda"),
                "labels": torch.tensor([0], device="cuda"),
                "orig_size": torch.tensor([10, 10], device="cuda"),
            }
            for _ in range(3)
        ]
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        with patch.object(torch.Tensor, "tolist", tracked_tolist):
            cb._convert_targets(targets)

        assert calls == [(3, 2)]


class TestConvertTargetsWithPreds:
    """_convert_targets(targets, preds) resizes each target's masks to its own paired prediction's grid (preds[index]),
    falling back to orig_size when preds is None or an entry lacks 'masks'."""

    @pytest.mark.parametrize(
        ("pred_masks_0", "pred_masks_1", "expected_shape_0", "expected_shape_1"),
        [
            pytest.param(
                torch.zeros(1, 4, 4, dtype=torch.bool),
                torch.zeros(1, 3, 3, dtype=torch.bool),
                (4, 4),
                (3, 3),
                id="both-images-resize-to-distinct-pred-grids",
            ),
            pytest.param(
                torch.zeros(1, 10, 10, dtype=torch.bool),
                torch.zeros(1, 3, 3, dtype=torch.bool),
                (10, 10),
                (3, 3),
                id="first-image-pred-grid-matches-orig-size-second-needs-resize",
            ),
        ],
    )
    def test_batch_resizes_each_target_to_its_own_paired_pred_grid(
        self,
        pred_masks_0: torch.Tensor,
        pred_masks_1: torch.Tensor,
        expected_shape_0: tuple[int, int],
        expected_shape_1: tuple[int, int],
    ) -> None:
        """2-image batch with differing orig_size AND differing pred grids: target[i]'s masks must resize to preds[i]'s
        own grid, never the other image's — verifies index-based pairing, not a swap."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
                "masks": torch.ones(1, 10, 10, dtype=torch.bool),
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "orig_size": torch.tensor([6, 6]),
                "masks": torch.ones(1, 6, 6, dtype=torch.bool),
            },
        ]
        preds = [{"masks": pred_masks_0}, {"masks": pred_masks_1}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == expected_shape_0
        assert out[1]["masks"].shape[-2:] == expected_shape_1

    def test_batch_with_one_image_missing_masks_key_leaves_it_absent(self) -> None:
        """Mixed batch: an image without a GT 'masks' key stays maskless while its sibling still resizes to its own
        paired pred grid."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
                # no "masks" key -> detection-only target in an otherwise-segmentation batch.
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "orig_size": torch.tensor([6, 6]),
                "masks": torch.ones(1, 6, 6, dtype=torch.bool),
            },
        ]
        preds = [{"boxes": torch.zeros(1, 4)}, {"masks": torch.zeros(1, 3, 3, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert "masks" not in out[0]
        assert out[1]["masks"].shape[-2:] == (3, 3)

    def test_upsize_via_preds_branch_preserves_extent_and_dtype(self) -> None:
        """GT masks at 2x2 resized through the preds-aware branch (NOT the preds=None/orig_size path) to a larger 4x4
        pred grid must nearest-upsize, staying bool and preserving the filled region's extent."""
        cb = COCOEvalCallback()
        gt_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        gt_masks[0, 0, 0] = True  # top-left cell
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([2, 2]),  # matches GT already -> orig_size alone would be a no-op
                "masks": gt_masks,
            }
        ]
        preds = [{"masks": torch.zeros(1, 4, 4, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == (4, 4)
        assert out[0]["masks"].dtype == torch.bool
        # nearest-neighbor upsize must preserve the filled top-left quadrant's extent.
        assert out[0]["masks"][0, :2, :2].all()
        assert not out[0]["masks"][0, 2:, :].any()
        assert not out[0]["masks"][0, :, 2:].any()

    def test_preds_shorter_than_targets_raises_assertion_error(self) -> None:
        """Preds and targets must be 1:1 per-image; a shorter preds list must fail loudly instead of silently
        misaligning images to the wrong prediction."""
        cb = COCOEvalCallback()
        targets = [
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "orig_size": torch.tensor([10, 10])},
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "orig_size": torch.tensor([10, 10])},
        ]
        preds = [{"masks": torch.zeros(1, 4, 4, dtype=torch.bool)}]

        with pytest.raises(AssertionError):
            cb._convert_targets(targets, preds)

    def test_empty_targets_and_empty_preds_returns_empty_list(self) -> None:
        """An empty batch (no images) through the preds-aware branch is a no-op, not an error."""
        cb = COCOEvalCallback()

        out = cb._convert_targets([], [])

        assert out == []

    def test_pred_entry_without_masks_key_falls_back_to_orig_size(self) -> None:
        """A prediction dict lacking 'masks' (e.g. detection-only fallback) must not crash; GT resizes to orig_size as
        if preds were None for that image."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([5, 5]),
                "masks": torch.ones(1, 4, 4, dtype=torch.bool),
            }
        ]
        preds = [{"boxes": torch.zeros(1, 4)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == (5, 5)

    def test_empty_mask_tensor_through_preds_path_yields_zero_instances_at_pred_grid(self) -> None:
        """K=0 GT masks (an image with no annotated instances) must resize through the preds-aware branch without
        raising, yielding shape (0, pred_h, pred_w)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "orig_size": torch.tensor([10, 10]),
                "masks": torch.zeros(0, 10, 10, dtype=torch.bool),
            }
        ]
        preds = [{"masks": torch.zeros(3, 4, 4, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape == (0, 4, 4)


def _ema_callback() -> MagicMock:
    """Return a mock that ``_get_ema_callback`` recognises (has ``get_ema_model_state_dict``)."""
    cb = MagicMock(name="ema_callback")
    cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
    return cb


def _cpu_module() -> MagicMock:
    """Mock LightningModule whose ``device`` is a real string so ``metric.to(device)`` works."""
    module = MagicMock(name="pl_module")
    module.device = "cpu"
    return module


class TestEmaCollectiveSymmetry:
    """DDP-deadlock fix: the EMA metric's cross-rank sync must be issued symmetrically (#931/#449)."""

    def test_ema_metric_created_on_val_epoch_start_when_ema_active(self) -> None:
        """map_metric_ema is created on validation start whenever the EMA callback is present.

        This makes the EMA ``compute()`` collective rank-invariant — created on every rank regardless of how many (or
        zero) val batches that rank later processes — rather than lazily per-batch.
        """
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        assert cb.map_metric_ema is None  # not created yet at setup

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is not None

    def test_ema_metric_not_created_without_ema_callback(self) -> None:
        """No EMA callback → map_metric_ema stays None (no EMA collective is ever issued)."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is None

    def test_should_compute_ema_false_when_metric_has_no_updates(self) -> None:
        """A rank whose EMA metric saw no updates votes against computing (avoids empty-state divergence)."""
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = False

        assert cb._should_compute_ema(_cpu_module()) is False

    def test_should_compute_ema_true_when_metric_has_updates_single_process(self) -> None:
        """With updates and no distributed group, the EMA compute proceeds."""
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = True

        assert cb._should_compute_ema(_cpu_module()) is True

    @patch("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", return_value=True)
    @patch("rfdetr.training.callbacks.coco_eval.dist.all_reduce")
    def test_unanimous_gate_skips_when_a_peer_lacks_ema(self, mock_all_reduce, _mock_init) -> None:
        """Even with local updates, the gate returns False if any peer voted 0 (all_reduce MIN → 0)."""

        def _peer_voted_zero(flag, op=None):  # simulate a rank with no EMA data
            flag.zero_()

        mock_all_reduce.side_effect = _peer_voted_zero
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = True

        assert cb._should_compute_ema(_cpu_module()) is False
        mock_all_reduce.assert_called_once()

    @patch("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", return_value=True)
    @patch("rfdetr.training.callbacks.coco_eval.dist.all_reduce")
    def test_unanimous_gate_runs_when_all_ranks_have_ema(self, mock_all_reduce, _mock_init) -> None:
        """When every rank has EMA updates (all_reduce MIN leaves the vote at 1), the gate returns True."""
        mock_all_reduce.side_effect = lambda flag, op=None: None  # vote tensor stays [1]
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = True

        assert cb._should_compute_ema(_cpu_module()) is True
        mock_all_reduce.assert_called_once()


class TestMapCollectiveSymmetry:
    """DDP-deadlock fix: the base/train mAP path's has_updates gate must be a collective vote.

    Unlike ``_should_compute_ema``'s unanimous ``all_reduce(MIN)`` (skip whenever any rank is empty), this path votes
    ``all_reduce(MAX)``: a rank with no local updates must still enter ``merge_distributed_state()`` when a peer has
    data, or the ranks diverge on which collectives they issue next and deadlock validation.
    """

    def test_any_rank_has_updates_false_without_distributed(self) -> None:
        """No distributed group and no local updates: the vote is False."""
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = False

        assert cb._any_rank_has_updates(metric, _cpu_module()) is False

    def test_any_rank_has_updates_true_without_distributed(self) -> None:
        """No distributed group and local updates present: the vote is True."""
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = True

        assert cb._any_rank_has_updates(metric, _cpu_module()) is True

    @patch("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", return_value=True)
    @patch("rfdetr.training.callbacks.coco_eval.dist.all_reduce")
    def test_max_vote_true_when_a_peer_has_updates(self, mock_all_reduce, _mock_init) -> None:
        """A locally-empty rank still votes True when the MAX reduction surfaces a peer's update."""

        def _peer_voted_one(flag, op=None):  # simulate a peer rank with data
            flag.fill_(1)

        mock_all_reduce.side_effect = _peer_voted_one
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = False

        assert cb._any_rank_has_updates(metric, _cpu_module()) is True
        assert mock_all_reduce.call_args.kwargs["op"] is torch.distributed.ReduceOp.MAX

    @patch("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", return_value=True)
    @patch("rfdetr.training.callbacks.coco_eval.dist.all_reduce")
    def test_max_vote_false_when_no_rank_has_updates(self, mock_all_reduce, _mock_init) -> None:
        """When every rank is empty (MAX reduction leaves the vote at 0), the gate returns False."""
        mock_all_reduce.side_effect = lambda flag, op=None: None  # vote tensor stays [0]
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = False

        assert cb._any_rank_has_updates(metric, _cpu_module()) is False

    @patch("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", return_value=True)
    @patch("rfdetr.training.callbacks.coco_eval.dist.all_reduce")
    def test_compute_and_log_enters_merge_when_only_a_peer_has_updates(self, mock_all_reduce, _mock_init) -> None:
        """A locally-empty rank still calls merge_distributed_state(), staying symmetric with its peer.

        Reproduces the deadlock scenario directly at the callback level: rank 0 has no local
        updates but a peer does. Before the fix, this rank took the early-return path and never
        issued ``merge_distributed_state()``'s collectives while its peer did — desynchronising the
        DDP collective sequence. After the fix, the collective vote makes both ranks agree to merge.
        """

        def _peer_has_map_updates_only(flag, op=None):
            # A peer has base mAP data (MAX vote → 1) but no EMA data (MIN vote stays at local 0),
            # isolating the has_updates gate under test from the unrelated EMA collective this
            # code path also issues for split="val".
            if op is torch.distributed.ReduceOp.MAX:
                flag.fill_(1)

        mock_all_reduce.side_effect = _peer_has_map_updates_only
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        cb.map_metric.compute.return_value = _minimal_metrics()

        cb._compute_and_log(_make_trainer(), _cpu_module(), "val")

        cb.map_metric.merge_distributed_state.assert_called_once()


class TestOnTestEpochStart:
    """on_test_epoch_start resets _ema_has_updates before test to prevent stale val state."""

    def test_map_metric_ema_stays_none_without_ema_callback(self) -> None:
        """No EMA callback → map_metric_ema stays None after test hook fires."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_test_epoch_start(trainer, module)

        assert cb.map_metric_ema is None

    def test_resets_ema_has_updates_to_false(self) -> None:
        """on_test_epoch_start resets _ema_has_updates to False even when stale True from validation."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb._ema_has_updates = True  # simulate stale value from a preceding validation epoch

        cb.on_test_epoch_start(trainer, module)

        assert cb._ema_has_updates is False


class TestPrepareEmaMetricSecondEpoch:
    """_prepare_ema_metric resets (not re-creates) the metric on subsequent epochs."""

    def test_resets_not_recreates_metric(self) -> None:
        """Calling on_validation_epoch_start twice resets the metric rather than replacing it."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_validation_epoch_start(trainer, module)
        assert cb.map_metric_ema is not None

        # Replace with a spy mock so reset() calls are trackable on the second epoch
        spy_metric = MagicMock(name="map_metric_ema")
        cb.map_metric_ema = spy_metric

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is spy_metric  # same object, not replaced
        spy_metric.reset.assert_called_once()


class TestComputeAndLogEmaResetPath:
    """Elif branch in _compute_and_log: gate False + metric not None → reset() fires."""

    def test_resets_ema_metric(self) -> None:
        """EMA not computed this epoch but metric exists → reset() clears state for the next epoch."""
        cb = COCOEvalCallback()
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        trainer.callback_metrics = {}

        # EMA metric exists but no batch updated it this epoch → gate returns False → elif fires
        mock_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema = mock_ema
        cb._ema_has_updates = False

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb.map_metric.has_updates = True

        with (
            patch.object(cb, "_build_per_class_rows", return_value=[]),
            patch.object(cb, "_print_metrics_tables"),
            patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value={}),
        ):
            cb._compute_and_log(trainer, module, "val")

        mock_ema.reset.assert_called_once()

    def test_clears_ema_has_updates_after_gate_true_compute(self) -> None:
        """Regression for #1289 review: after a gate-True EMA compute, `_ema_has_updates` must be cleared to False.

        Before this fix, `_compute_and_log_ema_metrics` reset `map_metric_ema` in both branches but never cleared
        `_ema_has_updates`, so a stale `True` left over from a validation epoch could reach a later
        `_compute_and_log(..., "train", ...)` call (when `compute_train_metrics=True`) and drive a spurious `compute()`
        on the freshly emptied EMA metric.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True

        should_compute_ema, _ema_metrics = cb._compute_and_log_ema_metrics(
            _make_trainer(), _make_pl_module(), "val", "", "mar_500"
        )

        assert should_compute_ema is True
        assert cb._ema_has_updates is False
