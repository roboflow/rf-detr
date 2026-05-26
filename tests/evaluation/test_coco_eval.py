# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for the local COCO evaluator wrapper."""

import json
from pathlib import Path

import numpy as np
import torch
from faster_coco_eval import COCO

from rfdetr.evaluation.coco_eval import CocoEvaluator


def _write_person_keypoint_coco(path: Path) -> None:
    """Write a minimal COCO keypoint annotation file."""
    keypoints = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    coords = []
    for idx in range(len(keypoints)):
        coords.extend([20.0 + idx, 30.0 + idx, 2.0])
    payload = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "image.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 50.0, 60.0],
                "area": 3000.0,
                "iscrowd": 0,
                "num_keypoints": len(keypoints),
                "keypoints": coords,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": keypoints,
                "skeleton": [],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_coco_evaluator_keypoints_uses_faster_evaluate_without_deprecated_evaluate_img(tmp_path: Path) -> None:
    """Keypoint evaluation should not call faster-coco-eval's deprecated ``evaluateImg`` shim."""
    annotation_path = tmp_path / "person_keypoints_val2017.json"
    _write_person_keypoint_coco(annotation_path)
    coco_gt = COCO(str(annotation_path))
    coco_gt.label2cat = {0: 1}
    evaluator = CocoEvaluator(coco_gt, ["keypoints"])
    keypoints = np.asarray(coco_gt.anns[1]["keypoints"], dtype=np.float32).reshape(1, 17, 3)

    evaluator.update(
        {
            1: {
                "boxes": torch.tensor([[10.0, 20.0, 60.0, 80.0]], dtype=torch.float32),
                "scores": torch.tensor([0.99], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
                "keypoints": torch.as_tensor(keypoints, dtype=torch.float32),
            }
        }
    )
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    stats = evaluator.coco_eval["keypoints"].stats
    assert np.isfinite(stats[0])


def test_coco_evaluator_handles_empty_keypoint_predictions(tmp_path: Path) -> None:
    """Keypoint evaluation should handle images with no detections."""
    annotation_path = tmp_path / "person_keypoints_val2017.json"
    _write_person_keypoint_coco(annotation_path)
    coco_gt = COCO(str(annotation_path))
    evaluator = CocoEvaluator(coco_gt, ["keypoints"])

    evaluator.update(
        {
            1: {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "keypoints": torch.zeros((0, 17, 3), dtype=torch.float32),
            }
        }
    )
    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    stats = evaluator.coco_eval["keypoints"].stats
    assert stats.shape == (10,)
