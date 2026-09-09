# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Optional evaluator parity through the production one-pass metric lifecycle."""

import copy
import pickle
import sys
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision


@pytest.fixture
def metric_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return tied predictions, crowd GT, and images missing predictions or GT.

    Examples:
        >>> metric_inputs()  # doctest: +SKIP
        # Pytest creates this fixture; direct fixture calls are unsupported.
    """
    boxes = torch.tensor([[1.0, 2.0, 11.0, 12.0], [1.0, 2.0, 11.0, 12.0], [20.0, 20.0, 25.0, 25.0]])
    masks = torch.zeros((3, 32, 32), dtype=torch.bool)
    masks[:2, 2:12, 1:11] = True
    masks[2, 20:25, 20:25] = True
    predictions = [
        dict(boxes=boxes, masks=masks, labels=torch.tensor([3, 3, 17]), scores=torch.tensor([0.8, 0.8, 0.4])),
        dict(boxes=boxes[:0], masks=masks[:0], labels=torch.empty(0, dtype=torch.long), scores=torch.empty(0)),
        dict(boxes=boxes[:1], masks=masks[:1], labels=torch.tensor([29]), scores=torch.tensor([0.5])),
    ]
    targets = [
        dict(
            boxes=boxes[[0, 2]],
            masks=masks[[0, 2]],
            labels=torch.tensor([3, 17]),
            iscrowd=torch.tensor([0, 1]),
            area=torch.tensor([100.0, 25.0]),
        ),
        dict(
            boxes=boxes[:1],
            masks=masks[:1],
            labels=torch.tensor([3]),
            iscrowd=torch.tensor([0]),
            area=torch.tensor([100.0]),
        ),
        dict(
            boxes=boxes[:0],
            masks=masks[:0],
            labels=torch.empty(0, dtype=torch.long),
            iscrowd=torch.empty(0, dtype=torch.long),
            area=torch.empty(0),
        ),
    ]
    return predictions, targets


class TestUltrafastMetric:
    @pytest.mark.parametrize("iou_type", ["bbox", "segm", pytest.param(("bbox", "segm"), id="both")])
    @pytest.mark.parametrize("max_dets", [100, 500])
    def test_metrics_arrays_pickle_and_reset(
        self,
        metric_inputs: tuple[list[dict[str, Any]], list[dict[str, Any]]],
        iou_type: Any,
        max_dets: int,
    ) -> None:
        """Match all metric tensors and oracle arrays across incremental update and reuse."""
        predictions, targets = metric_inputs
        reference = OnePassCocoMeanAveragePrecision(
            iou_type=iou_type, class_metrics=True, max_detection_thresholds=[1, 10, max_dets]
        )
        actual = OnePassCocoMeanAveragePrecision(
            iou_type=iou_type, class_metrics=True, max_detection_thresholds=[1, 10, max_dets], backend="ultrafast"
        )
        assert actual._coco_backend.coco.__module__.startswith("ultrafast_pycocotools")
        actual = pickle.loads(pickle.dumps(actual))
        for _ in range(2):
            for metric in (reference, actual):
                for index in range(len(predictions)):
                    metric.update(
                        copy.deepcopy(predictions[index : index + 1]), copy.deepcopy(targets[index : index + 1])
                    )
                metric.merge_distributed_state()
            expected, observed = reference.compute(), actual.compute()
            assert observed.keys() == expected.keys()
            for key in expected:
                torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True)
            prediction_data, target_data = actual._coco_datasets(actual._observed_classes())
            oracle_gt, oracle_dt = COCO(), COCO()
            oracle_gt.dataset, oracle_dt.dataset = (
                copy.deepcopy(target_data.dataset),
                copy.deepcopy(prediction_data.dataset),
            )
            oracle_gt.createIndex()
            oracle_dt.createIndex()
            for kind in actual.iou_type:
                if len(actual.iou_type) > 1:
                    for dataset in (prediction_data, oracle_dt):
                        for ann in dataset.dataset["annotations"]:
                            ann["area"] = ann["area_" + kind]
                evaluator = actual._coco_backend.cocoeval(target_data, prediction_data, iouType=kind)
                oracle = COCOeval(oracle_gt, oracle_dt, iouType=kind)
                for instance in (evaluator, oracle):
                    instance.params.maxDets = [1, 10, max_dets]
                    instance.evaluate()
                    instance.accumulate()
                for key in ("precision", "recall", "scores"):
                    assert evaluator.eval[key].shape == oracle.eval[key].shape
                    assert np.asarray(evaluator.eval[key]).tobytes() == np.asarray(oracle.eval[key]).tobytes()
            reference.reset()
            actual.reset()

    def test_missing_extra_has_install_guidance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Selecting a missing optional evaluator explains which extra to install."""
        monkeypatch.setitem(sys.modules, "ultrafast_pycocotools.integrations.rfdetr", None)
        with pytest.raises(ModuleNotFoundError, match=r"rfdetr\[ultrafast\]"):
            OnePassCocoMeanAveragePrecision(backend="ultrafast")

    def test_backend_dependency_error_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Preserve a backend dependency failure instead of suggesting the optional extra."""
        module_name = "ultrafast_pycocotools.integrations.rfdetr"
        module = ModuleType(module_name)
        original_error = ModuleNotFoundError("No module named 'numpy'", name="numpy")
        monkeypatch.setattr(module, "__getattr__", Mock(side_effect=original_error), raising=False)
        monkeypatch.setitem(sys.modules, module_name, module)

        with pytest.raises(ModuleNotFoundError) as caught:
            OnePassCocoMeanAveragePrecision(backend="ultrafast")

        assert caught.value is original_error
