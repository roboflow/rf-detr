# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for COCOEvalCallback epoch-end metric computation, logging and table printing.

Tests for the gated-mAP fallback warnings and the console header of ``COCOEvalCallback``.
"""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.evaluation.matching import build_matching_data, merge_matching_data
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from tests.training.callbacks.helpers import (
    _cpu_module,
    _make_pl_module,
    _make_trainer,
    _matching_entry,
    _metrics_with_per_class,
    _minimal_metrics,
)


class _TQDMProgressBar:
    """Minimal progress-bar stand-in for callback detection tests."""


def _one_match_f1_state() -> dict:
    """Return per-class F1 accumulator state holding one matched ground-truth box.

    Examples:
        >>> _one_match_f1_state()[0]["total_gt"]
        1
    """
    return {
        0: {
            "scores": np.array([0.9], dtype=np.float32),
            "matches": np.array([1], dtype=np.int64),
            "ignore": np.array([False]),
            "total_gt": 1,
        }
    }


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
    def test_terminal_metrics_tables_print_to_console(self, split: str, title_pfx: str, mocker: MockerFixture) -> None:
        """Terminal metric tables print directly through the Rich console each epoch."""
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True
        console = MagicMock(name="console")

        mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console", return_value=console)
        mocker.patch(
            "rfdetr.training.callbacks.coco_eval._render_overall_merged",
            side_effect=["overall-1", "overall-2"],
        )
        render_tables = mocker.patch("rfdetr.training.callbacks.coco_eval._render_summary_tables")
        cb._print_metrics_tables(trainer, split, {"mAP": 0.1}, [])
        cb._print_metrics_tables(trainer, split, {"mAP": 0.2}, [])

        assert render_tables.call_count == 2
        assert render_tables.call_args_list[0].args[0] is console
        assert render_tables.call_args_list[0].args[1].startswith(title_pfx)
        assert "(Epoch" in render_tables.call_args_list[0].args[1]
        assert render_tables.call_args_list[0].args[2] == "overall-1"

    @pytest.mark.parametrize("name", ["sign[/]", "helmet[red]", "price:dollar:"])
    def test_terminal_prints_hostile_class_name_through_real_console(self, mocker: MockerFixture, name: str) -> None:
        """A class name that looks like Rich markup prints verbatim through a real console without aborting.

        ``sign[/]`` used to raise ``MarkupError`` out of ``_print_metrics_tables``, which nothing between the callback
        and Lightning catches, so the existing mocked-console tests could not see the training abort.
        """
        import io

        from rich.console import Console

        mocker.patch("rfdetr.training.callbacks.coco_eval._render_overall_merged", return_value="overall")

        console = Console(file=io.StringIO(), width=200, color_system=None)
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True
        per_class = [{"name": name, "ap": 0.5, "ar": 0.6, "f1": 0.55, "precision": 0.6, "recall": 0.5}]

        mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console", return_value=console)
        cb._print_metrics_tables(trainer, "val", {"mAP": 0.1}, per_class)

        assert name in console.file.getvalue()

    def test_missing_rich_warns_once_and_skips_metric_tables(self, mocker: MockerFixture) -> None:
        """Missing Rich emits one warning and skips noisy table rendering."""
        get_console = mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console")
        warning = mocker.patch("rfdetr.training.callbacks.coco_eval.logger.warning")
        mocker.patch("rfdetr.training.callbacks.coco_eval._IS_RICH_AVAILABLE", False)
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True

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

    def test_terminal_prints_through_rich_progress_bar_console(self, mocker: MockerFixture) -> None:
        """Metric tables route through RichProgressBar._console when active."""
        mocker.patch("rfdetr.training.callbacks.coco_eval._render_overall_merged", return_value="overall")
        # Create a fake callback whose class name is RichProgressBar so
        # _get_rich_console picks it up without importing PTL.
        rich_progress_bar_fake = type("RichProgressBar", (), {})
        rich_console = MagicMock(name="rich_console")
        fake_pb = rich_progress_bar_fake()
        fake_pb._console = rich_console  # type: ignore[attr-defined]

        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer(callbacks=[fake_pb])
        trainer.is_global_zero = True

        cb._print_metrics_tables(trainer, "val", {"mAP": 0.5}, [])

        rich_console.print.assert_called_once()

    def test_notebook_metrics_tables_reuse_and_clear_output_widget(self, mocker: MockerFixture) -> None:
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

        mocker.patch.dict(
            sys.modules,
            {
                "ipywidgets": widgets_module,
                "IPython": ipython_module,
                "IPython.display": display_module,
            },
        )
        mocker.patch(
            "rfdetr.training.callbacks.coco_eval._render_overall_merged", side_effect=["overall-1", "overall-2"]
        )
        render_summary_tables = mocker.patch("rfdetr.training.callbacks.coco_eval._render_summary_tables")
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
        cb._f1_local = _one_match_f1_state()
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
        cb._f1_local = _one_match_f1_state()

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

    def test_keypoint_oks_metric_created_with_correct_args(self, mocker: MockerFixture) -> None:
        """_get_or_create_keypoint_oks_metric must construct MetricKeypointOKS with coco_api and sigmas."""
        cb = COCOEvalCallback(max_dets=500, keypoint_oks_sigmas=[0.05])
        dataset = MagicMock(name="dataset")
        datamodule = MagicMock()
        datamodule._dataset_val = dataset
        datamodule._dataset_test = None
        datamodule._dataset_train = None
        trainer = _make_trainer(datamodule=datamodule)
        coco_api = MagicMock(name="coco_api")

        mocker.patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", return_value=coco_api)
        oks_metric_cls = mocker.patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS")
        result = cb._get_or_create_keypoint_oks_metric(trainer, split="val")

        assert result is oks_metric_cls.return_value
        oks_metric_cls.assert_called_once_with(coco_api, keypoint_oks_sigmas=[0.05], max_dets=500)

    @pytest.mark.parametrize(("split", "expected_dataset"), [("train", "train"), ("val_ema", "val")])
    def test_keypoint_oks_metric_uses_dataset_of_split(
        self, mocker: MockerFixture, split: str, expected_dataset: str
    ) -> None:
        """Train keypoint mAP is built from the train dataset and EMA keypoint mAP from the validation dataset."""
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

        mocker.patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", side_effect=_get_coco_api)
        oks_metric_cls = mocker.patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS")
        cb._get_or_create_keypoint_oks_metric(trainer, split=split)

        assert oks_metric_cls.call_args.args[0] is {"train": train_coco_api, "val": val_coco_api}[expected_dataset]

    def test_mixed_keypoint_counts_create_keypoint_oks_metric(self, mocker: MockerFixture) -> None:
        """Mixed keypoint counts should be handled by MetricKeypointOKS instead of being skipped."""
        mocker.patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", return_value=MagicMock())
        oks_metric_cls = mocker.patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS")
        warning = mocker.patch("rfdetr.training.callbacks.coco_eval.logger.warning")
        cb = COCOEvalCallback(max_dets=500)
        dataset = MagicMock(name="dataset")
        datamodule = MagicMock()
        datamodule._dataset_train = dataset
        datamodule._dataset_val = None
        datamodule._dataset_test = None
        trainer = _make_trainer(datamodule=datamodule)

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

    @pytest.fixture
    def cb(self) -> COCOEvalCallback:
        """Return a callback whose base metric accumulated nothing while the EMA metric holds data.

        The base model is not evaluated: ``has_updates`` is False because the default policy never routes batches
        there, and every prediction went to ``map_metric_ema`` instead.

        Examples:
            >>> cb  # doctest: +SKIP
            A pytest fixture.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True
        return cb

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
        assert "val/ema_mAP_75" in logged_keys
        assert "val/ema_mAR" in logged_keys
        cb.map_metric_ema.reset.assert_called_once()

    def test_still_logs_ema_metrics_when_base_metric_is_empty(self, cb: COCOEvalCallback) -> None:
        """Regression for #1285: EMA metrics are still logged when the base metric accumulated nothing this epoch.

        With the base model not evaluated, on_validation_batch_end routes every prediction to map_metric_ema.

        The empty-state guard in _compute_and_log must not also suppress the EMA metrics — before the fix it returned
        before ever reaching the EMA compute block, so such runs logged no validation output at all.
        """
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        cb.map_metric.compute.assert_not_called()
        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        cb.map_metric_ema.reset.assert_called_once()
        cb.map_metric.reset.assert_called_once()

    def test_ema_score_mirrored_onto_primary_keys_when_base_metric_empty(self, cb: COCOEvalCallback) -> None:
        """The EMA score must also reach val/mAP_50_95 when the base model was not evaluated this epoch.

        Leaving the primary key unpopulated — what the deprecated eval_ema_only flag did — is safe for an opt-in
        but not for a default: every scheduler, early-stopping hook, checkpoint monitor and dashboard watching
        val/mAP_50_95 would silently see nothing. The mirrored value must be the EMA one, not a stale or zero
        reading, so BestModelCallback's EMA track and external monitors agree on the epoch's score.
        """
        trainer = _make_trainer()
        trainer.callback_metrics = {}  # the mirror republishes whatever the EMA block wrote here
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        logged = {c.args[0]: c.args[1] for c in module.log.call_args_list}
        assert logged["val/mAP_50_95"] == logged["val/ema_mAP_50_95"]
        assert trainer.callback_metrics["val/mAP_50_95"] == trainer.callback_metrics["val/ema_mAP_50_95"]
        assert logged["val/mAP_75"] == logged["val/ema_mAP_75"]
        assert logged["val/mAR"] == logged["val/ema_mAR"]

    def test_per_class_ap_mirrored_onto_primary_keys(self, cb: COCOEvalCallback) -> None:
        """EMA-only validation publishes per-class AP under both EMA and primary metric namespaces.

        Per-class AP is logged through pl_module.log alone, so the primary alias must be emitted explicitly alongside
        the EMA row when no base model was evaluated.
        """
        cb._class_names = ["cat"]
        cb._cat_id_to_name = {0: "cat"}
        ema_metrics = _minimal_metrics()
        ema_metrics["map_per_class"] = torch.tensor([0.5])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6])
        ema_metrics["classes"] = torch.tensor([0])
        cb.map_metric_ema.compute.return_value = ema_metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_AP/cat" in logged_keys
        assert "val/AP/cat" in logged_keys

        logged = {c.args[0]: c.args[1] for c in module.log.call_args_list}
        assert logged["val/AP/cat"] == logged["val/ema_AP/cat"]

    def test_computes_f1_from_accumulated_matches_when_base_metric_is_empty(self, cb: COCOEvalCallback) -> None:
        """Regression for #1285: val/F1 is computed from accumulated matches even when the base metric is empty.

        on_validation_batch_end merges matching data into f1_local unconditionally, whichever mAP track a batch was
        routed to.

        With the base model not evaluated, map_metric never accumulates, so the empty-state guard in _compute_and_log
        used to discard f1_local via _reset_f1_local before ever computing val/F1 — silently dropping it even though
        real matching data (from the EMA-quality predictions) had been collected.
        """
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

    def test_prints_summary_table_when_base_metric_is_empty(self, mocker: MockerFixture) -> None:
        """Regression for #1285: the console summary table is also part of "validation output".

        Before this fix, ``_print_metrics_tables`` was only ever called from the branch that reads the base ``metric`` —
        which never has data when the base model is not evaluated — so no table was ever printed for the run, even after
        the scalar/F1 logging gaps were closed. Per-class AP must be logged under ``ema_``-prefixed keys, mirroring
        ``val/ema_mAP_50_95`` staying separate from ``val/mAP_50_95`` elsewhere, so it never collides with a base-track
        key.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: the default policy never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        ema_metrics = _minimal_metrics()
        ema_metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6, 0.3])
        ema_metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        print_metrics_tables = mocker.patch.object(cb, "_print_metrics_tables")
        cb.on_validation_epoch_end(_make_trainer(), module)

        print_metrics_tables.assert_called_once()
        _trainer_arg, title_arg, overall_arg, per_class_arg = print_metrics_tables.call_args.args
        assert title_arg == "val (ema)"
        assert overall_arg["mAP 50:95"] == pytest.approx(0.4)
        assert {row["name"] for row in per_class_arg} == {"cat", "dog"}

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_AP/cat" in logged_keys
        assert "val/ema_AP/dog" in logged_keys
        assert "val/AP/cat" in logged_keys
        assert "val/AP/dog" in logged_keys

    def test_resets_val_ema_keypoints_when_ema_not_yet_warmed_up(self, mocker: MockerFixture) -> None:
        """Regression for #1289: an un-warmed EMA with an empty base metric resets the val_ema keypoint split.

        The split is reset, not computed.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: the default policy never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = False  # EMA metric also has no updates this epoch
        module = _make_pl_module()

        compute_keypoint_map = mocker.patch.object(cb, "_compute_and_log_keypoint_map")
        reset_keypoint_split = mocker.patch.object(cb, "_reset_keypoint_split")
        cb.on_validation_epoch_end(_make_trainer(), module)

        compute_keypoint_map.assert_not_called()
        assert call("val_ema") in reset_keypoint_split.call_args_list
        cb.map_metric_ema.compute.assert_not_called()

    def test_summary_table_includes_segm_metrics(self, mocker: MockerFixture) -> None:
        """Regression for #1289: the EMA-only summary table includes segm mAP entries when segmentation is enabled."""
        cb = COCOEvalCallback(max_dets=500, segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb._class_names = ["cat"]
        cb._cat_id_to_name = {0: "cat"}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: the default policy never routes here
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

        print_metrics_tables = mocker.patch.object(cb, "_print_metrics_tables")
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

    def test_progress_bar_suppresses_duplicate_pycocotools_output(self, capsys, mocker: MockerFixture) -> None:
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

        print_metrics_tables = mocker.patch.object(cb, "_print_metrics_tables")
        cb._compute_and_log(trainer, module, "val")

        assert "Average Precision" not in capsys.readouterr().out
        print_metrics_tables.assert_called_once()
        cb.map_metric.compute.assert_called_once()
        module.log.assert_called()

    def test_per_class_ap_disabled_logs_no_class_keys(self) -> None:
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

    def test_per_class_ap_disabled_tolerates_zero_dim_ar(self) -> None:
        """Regression test for H1: log_per_class_metrics=False must not crash epoch-end validation.

        torchmetrics still emits ``mar_{max_dets}_per_class`` as a 0-dim ``tensor(-1.)`` even when
        ``class_metrics=False`` (only ``map_per_class`` is actually suppressed), while ``classes`` stays 1-d for >=2
        classes. Before the fix, the per-class AR block was gated on key-presence only (not on the flag), so
        ``zip(classes, mar_..._per_class)`` iterated a 0-d tensor and raised ``TypeError: iteration over a 0-d tensor``.
        ``_minimal_metrics()`` alone (used by the sibling ``test_per_class_ap_disabled_logs_no_class_keys``) omits this
        key entirely and would not catch the regression.
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
        """Core metrics land in trainer.callback_metrics each epoch so improvement is detected.

        ModelCheckpoint and BestModelCallback read this dict to detect improvement.

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


class TestOnTestEpochEnd:
    """Test-loop-specific behaviour of on_test_epoch_end."""

    @pytest.mark.parametrize("forbidden_prefix", ["test/ema_", "val/"])
    def test_does_not_log_forbidden_prefix(self, forbidden_prefix: str) -> None:
        """No test/ema_* aliases (test already runs with EMA weights) and no val/ keys bleeding across loops."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith(forbidden_prefix) for k in logged_keys)


_WARNING = "rfdetr.training.callbacks.coco_eval.logger.warning"


def _compute(
    cb: COCOEvalCallback, merged: dict, metrics: dict, trainer: MagicMock | None = None
) -> tuple[MagicMock, MagicMock]:
    """Run one base-track ``_compute_and_log`` with the metric and the table printer stubbed out.

    Examples:
        >>> cb = COCOEvalCallback(min_class_support=2)
        >>> trainer, print_tables = _compute(cb, {0: _matching_entry(5)}, _metrics_with_per_class({0: 0.5}))
        >>> print_tables.call_args.args[2]["gated applied"]
        1.0
    """
    trainer = trainer if trainer is not None else _make_trainer()
    module = _cpu_module()
    cb.setup(trainer, module, stage="fit")
    trainer.callback_metrics = {}
    cb.map_metric = MagicMock(name="map_metric")
    cb.map_metric.has_updates = True
    cb.map_metric.compute.return_value = metrics
    with (
        patch.object(cb, "_print_metrics_tables") as print_tables,
        patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value=merged),
    ):
        cb._compute_and_log(trainer, module, "val")
    return trainer, print_tables


def _messages(warning: MagicMock) -> list[str]:
    """Return the formatted text of every logger warning recorded on ``warning``.

    Examples:
        >>> warning = MagicMock()
        >>> _ = warning("%s: n=%d", "val", 3)
        >>> _messages(warning)
        ['val: n=3']
    """
    return [call.args[0] % call.args[1:] for call in warning.call_args_list]


class TestNoEligibleClassWarning:
    """An empty eligible set is warned about once per split per fit, on rank zero, naming mAP and F1."""

    def test_warns_once_per_split_across_epochs(self, mocker: MockerFixture) -> None:
        """Two evaluations of the same split produce one warning, not one per epoch and track."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10_000)
        merged = {0: _matching_entry(100), 1: _matching_entry(3)}
        metrics = _metrics_with_per_class({0: 0.6, 1: 0.0})

        _compute(cb, merged, metrics)
        _compute(cb, merged, metrics)

        assert len(_messages(warning)) == 1

    def test_message_names_gated_map_and_f1_sweep(self, mocker: MockerFixture) -> None:
        """The single warning covers both quantities the floor gates."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10_000)

        _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: 0.6}))

        (text,) = _messages(warning)
        assert "mAP_50_95_gated" in text
        assert "F1 sweep" in text

    def test_on_fit_start_rearms_the_warning(self, mocker: MockerFixture) -> None:
        """A new fit warns again."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10_000)
        merged = {0: _matching_entry(100)}
        metrics = _metrics_with_per_class({0: 0.6})

        _compute(cb, merged, metrics)
        cb.on_fit_start(_make_trainer(), _cpu_module())
        _compute(cb, merged, metrics)

        assert len(_messages(warning)) == 2

    def test_silent_off_rank_zero(self, mocker: MockerFixture) -> None:
        """A non-zero rank logs nothing for an empty eligible set."""
        warning = mocker.patch(_WARNING)
        mocker.patch("rfdetr.training.callbacks.coco_eval.is_main_process", return_value=False)
        trainer = _make_trainer()
        trainer.is_global_zero = False
        cb = COCOEvalCallback(min_class_support=10_000)

        _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: 0.6}), trainer)

        assert not _messages(warning)

    def test_no_warning_when_a_class_qualifies(self, mocker: MockerFixture) -> None:
        """An eligible class means neither fallback message fires."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10)

        _, print_tables = _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: 0.6}))

        assert not _messages(warning)
        assert print_tables.call_args.args[2]["gated applied"] == 1.0


class TestNoValidPerClassAp:
    """Eligible classes without a valid AP (no detections) fall back with their own message and flag."""

    def test_sentinel_aps_use_the_short_message_and_mark_gate_unapplied(self, mocker: MockerFixture) -> None:
        """Zero detections give every class AP -1: not an 'empty eligible set', and the gate did not apply."""
        warning = mocker.patch(_WARNING)
        cb = COCOEvalCallback(min_class_support=10)
        metrics = _metrics_with_per_class({0: -1.0, 1: -1.0})

        trainer, print_tables = _compute(cb, {0: _matching_entry(100), 1: _matching_entry(50)}, metrics)

        (text,) = _messages(warning)
        assert "no valid per-class" in text
        assert "min_class_support" not in text
        assert print_tables.call_args.args[2]["gated applied"] == 0.0
        assert float(trainer.callback_metrics["val/mAP_50_95_gated"]) == pytest.approx(0.4)

    def test_sentinel_message_is_silent_off_rank_zero(self, mocker: MockerFixture) -> None:
        """The all-sentinel message is rank-zero only."""
        warning = mocker.patch(_WARNING)
        mocker.patch("rfdetr.training.callbacks.coco_eval.is_main_process", return_value=False)
        cb = COCOEvalCallback(min_class_support=10)

        _compute(cb, {0: _matching_entry(100)}, _metrics_with_per_class({0: -1.0}))

        assert not _messages(warning)

    def test_applied_flag_is_reported_through_gated_map(self) -> None:
        """``_gated_map`` keeps returning a float and reports whether the gate applied through ``applied``."""
        cb = COCOEvalCallback(min_class_support=10)
        applied: list[bool] = []
        metrics: dict[str, Any] = _metrics_with_per_class({0: 0.6, 1: 0.0})

        value = cb._gated_map(metrics, "", {0}, "val", applied)

        assert value == pytest.approx(0.6)
        assert applied == [True]
        assert isinstance(cb._gated_map(metrics, "", {0}, "val"), float)
