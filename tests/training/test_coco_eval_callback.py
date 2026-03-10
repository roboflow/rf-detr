# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for COCOEvalCallback (PTL Ch3/T3)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

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
    """setup() creates map_metric with correct configuration."""

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
        """When coco.label2cat is present (remap_category_ids=True) the mapping
        uses 0-based remapped label IDs so class names align with predictions."""
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


# ---------------------------------------------------------------------------
# Batch-end — shared behaviour for both validation and test loops
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Epoch-end — shared behaviour for both validation and test loops
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation-epoch-end-only behaviour
# ---------------------------------------------------------------------------


class TestOnValidationEpochEnd:
    """Validation-specific behaviour of on_validation_epoch_end."""

    def test_ema_metrics_logged_when_map_metric_ema_populated(self) -> None:
        """val/ema_* metrics are logged when map_metric_ema has accumulated data.

        EMA metrics are now computed from a separate map_metric_ema that is
        populated during on_validation_batch_end (not aliased from base metrics).
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        # Simulate map_metric_ema being populated by on_validation_batch_end.
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        cb.map_metric_ema.reset.assert_called_once()

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

    def test_callback_metrics_updated_for_model_checkpoint(self) -> None:
        """Core metrics written to trainer.callback_metrics each epoch so
        ModelCheckpoint / BestModelCallback detect improvement.

        pl_module.log() from a callback's on_validation_epoch_end goes only to
        logged_metrics (external loggers), not callback_metrics.
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

        cb.on_validation_epoch_end(trainer, _make_pl_module())

        assert "val/ema_mAP_50_95" in trainer.callback_metrics
        assert "val/ema_mAP_50" in trainer.callback_metrics
        assert "val/ema_mAR" in trainer.callback_metrics

    def test_ghost_class_with_negative_ar_sentinel_is_filtered(self) -> None:
        """A class where both ap=-1 and ar=-1 (negative sentinels, not NaN) must
        be excluded from the per-class table.  The old filter checked for NaN
        only, so ar=-1 (a valid float) escaped the guard."""
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
        """test/ema_* aliases are NOT logged — test always runs with EMA weights
        via the RFDETREMACallback swap so test/mAP_50 is already the EMA result."""
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


# ---------------------------------------------------------------------------
# Target conversion
# ---------------------------------------------------------------------------


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
        """labels tensor is preserved unchanged."""
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
        """masks tensor is cast to bool and included in output."""
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

    def test_iscrowd_passed_through(self) -> None:
        """iscrowd tensor is included when present."""
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
