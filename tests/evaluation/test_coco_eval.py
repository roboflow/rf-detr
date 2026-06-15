# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for the local COCO evaluator wrapper."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pycocotools.coco as pycoco
import pytest
import torch
from faster_coco_eval import COCO

from rfdetr.evaluation import coco_eval as coco_eval_module
from rfdetr.evaluation.coco_eval import CocoEvaluator


def _write_person_keypoint_coco(path: Path, *, include_num_keypoints: bool = True, keypoint_count: int = 17) -> None:
    """Write a minimal COCO keypoint annotation file."""
    if keypoint_count == 17:
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
    else:
        keypoints = [f"point_{idx}" for idx in range(keypoint_count)]
    coords = []
    for idx in range(len(keypoints)):
        coords.extend([20.0 + idx, 30.0 + idx, 2.0])
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10.0, 20.0, 50.0, 60.0],
        "area": 3000.0,
        "iscrowd": 0,
        "keypoints": coords,
    }
    if include_num_keypoints:
        annotation["num_keypoints"] = len(keypoints)
    payload = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "image.jpg"}],
        "annotations": [annotation],
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


def _write_mixed_keypoint_coco(path: Path) -> None:
    """Write a COCO keypoint file with two categories using different keypoint counts."""
    categories = [
        {
            "id": 1,
            "name": "dart",
            "supercategory": "object",
            "keypoints": [f"dart_{idx}" for idx in range(4)],
            "skeleton": [],
        },
        {
            "id": 2,
            "name": "person",
            "supercategory": "person",
            "keypoints": [f"person_{idx}" for idx in range(21)],
            "skeleton": [],
        },
    ]
    annotations = []
    for annotation_id, (category_id, keypoint_count, x0, y0) in enumerate(
        [(1, 4, 10.0, 20.0), (2, 21, 50.0, 60.0)],
        start=1,
    ):
        keypoints = []
        for idx in range(keypoint_count):
            keypoints.extend([x0 + idx, y0 + idx, 2.0])
        annotations.append(
            {
                "id": annotation_id,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [x0, y0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": keypoint_count,
            }
        )

    payload = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "image.jpg"}],
        "annotations": annotations,
        "categories": categories,
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


def test_coco_evaluator_keypoints_log_summary_false_suppresses_summary_rows(tmp_path: Path) -> None:
    """Keypoint accumulation should compute stats without AP/AR logger spam when summaries are disabled."""
    annotation_path = tmp_path / "person_keypoints_val2017.json"
    _write_person_keypoint_coco(annotation_path)
    coco_gt = COCO(str(annotation_path))
    coco_gt.label2cat = {0: 1}
    evaluator = CocoEvaluator(coco_gt, ["keypoints"], log_summary=False)
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

    with patch.object(coco_eval_module.logger, "info") as info:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()

    info.assert_not_called()
    stats = evaluator.coco_eval["keypoints"].stats
    assert np.isfinite(stats[0])


def test_coco_evaluator_keypoints_accepts_pycocotools_coco_api(tmp_path: Path) -> None:
    """Keypoint evaluation should accept COCO APIs returned by torchvision datasets."""
    annotation_path = tmp_path / "person_keypoints_val2017.json"
    _write_person_keypoint_coco(annotation_path)
    coco_gt = pycoco.COCO(str(annotation_path))
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


def test_coco_evaluator_keypoints_infers_custom_oks_sigmas(tmp_path: Path) -> None:
    """Custom keypoint-count datasets should not use COCO's fixed 17-keypoint OKS sigmas."""
    annotation_path = tmp_path / "custom_keypoints_val.json"
    _write_person_keypoint_coco(annotation_path, keypoint_count=25)
    coco_gt = COCO(str(annotation_path))
    coco_gt.label2cat = {0: 1}
    evaluator = CocoEvaluator(coco_gt, ["keypoints"])
    keypoints = np.asarray(coco_gt.anns[1]["keypoints"], dtype=np.float32).reshape(1, 25, 3)

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


def test_coco_evaluator_warns_once_per_custom_keypoint_count(tmp_path: Path) -> None:
    """Repeated evaluator construction should not spam the same custom OKS fallback warning."""
    annotation_path = tmp_path / "custom_keypoints_val.json"
    _write_person_keypoint_coco(annotation_path, keypoint_count=25)
    coco_gt = COCO(str(annotation_path))

    coco_eval_module._WARNED_CUSTOM_KEYPOINT_OKS_COUNTS.clear()
    try:
        with patch.object(coco_eval_module.logger, "warning") as warning:
            CocoEvaluator(coco_gt, ["keypoints"])
            CocoEvaluator(coco_gt, ["keypoints"])
    finally:
        coco_eval_module._WARNED_CUSTOM_KEYPOINT_OKS_COUNTS.clear()

    warning.assert_called_once()


def test_coco_evaluator_rejects_mismatched_custom_oks_sigmas(tmp_path: Path) -> None:
    """Explicit OKS sigmas must match the dataset keypoint count."""
    annotation_path = tmp_path / "custom_keypoints_val.json"
    _write_person_keypoint_coco(annotation_path, keypoint_count=25)
    coco_gt = COCO(str(annotation_path))

    with pytest.raises(ValueError, match="keypoint_oks_sigmas length 17 does not match dataset keypoint count 25"):
        CocoEvaluator(coco_gt, ["keypoints"], keypoint_oks_sigmas=[0.05] * 17)


def test_coco_evaluator_keypoints_handles_mixed_counts_and_multi_instance_image(tmp_path: Path) -> None:
    """Mixed keypoint-count categories should evaluate by group instead of being skipped."""
    annotation_path = tmp_path / "mixed_keypoints_val.json"
    _write_mixed_keypoint_coco(annotation_path)
    coco_gt = COCO(str(annotation_path))
    coco_gt.label2cat = {0: 1, 1: 2}
    evaluator = CocoEvaluator(coco_gt, ["keypoints"], keypoint_oks_sigmas=[0.05] * 21)

    padded_keypoints = np.zeros((2, 21, 3), dtype=np.float32)
    for detection_idx, annotation in enumerate(coco_gt.dataset["annotations"]):
        keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        padded_keypoints[detection_idx, : keypoints.shape[0]] = keypoints

    evaluator.update(
        {
            1: {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]], dtype=torch.float32),
                "scores": torch.tensor([0.99, 0.98], dtype=torch.float32),
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "keypoints": torch.as_tensor(padded_keypoints, dtype=torch.float32),
            }
        }
    )

    results = evaluator.coco_results["keypoints"]
    assert len(results) == 2
    assert len(results[0]["keypoints"]) == 4 * 3
    assert len(results[1]["keypoints"]) == 21 * 3

    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    grouped_eval = evaluator.coco_eval["keypoints"]
    assert len(grouped_eval.evals) == 2
    stats = grouped_eval.stats
    assert stats.shape == (10,)
    assert np.isfinite(stats[0])


def test_coco_evaluator_backfills_missing_num_keypoints(tmp_path: Path) -> None:
    """Keypoint GT without `num_keypoints` should not be ignored during OKS evaluation."""
    annotation_path = tmp_path / "person_keypoints_val2017.json"
    _write_person_keypoint_coco(annotation_path, include_num_keypoints=False)
    coco_gt = COCO(str(annotation_path))
    assert "num_keypoints" not in coco_gt.anns[1]

    evaluator = CocoEvaluator(coco_gt, ["keypoints"])

    assert evaluator.coco_gt.anns[1]["num_keypoints"] == 17


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


class TestSynchronizeBetweenProcesses:
    """synchronize_between_processes() deduplicates DT when DDP padding repeats image_ids."""

    def _make_evaluator(self, tmp_path: Path) -> CocoEvaluator:
        """Return a single-annotation evaluator with label2cat identity mapping."""
        annotation_path = tmp_path / "kp.json"
        _write_person_keypoint_coco(annotation_path)
        coco_gt = COCO(str(annotation_path))
        coco_gt.label2cat = {0: 1}
        return CocoEvaluator(coco_gt, ["keypoints"])

    def _pred(self, image_id: int, score: float = 0.99) -> dict:
        """Single detection prediction dict for image_id."""
        kp = np.zeros((1, 17, 3), dtype=np.float32)
        return {
            image_id: {
                "boxes": torch.tensor([[10.0, 20.0, 60.0, 80.0]]),
                "scores": torch.tensor([score]),
                "labels": torch.tensor([0], dtype=torch.long),
                "keypoints": torch.as_tensor(kp),
            }
        }

    def test_single_gpu_no_dedup_needed(self, tmp_path: Path) -> None:
        """Single-GPU path (world_size=1): all_gather returns one-element list; all results preserved."""
        ev = self._make_evaluator(tmp_path)
        ev.update(self._pred(1))

        with patch("rfdetr.evaluation.coco_eval.all_gather", side_effect=lambda x: [x]):
            ev.synchronize_between_processes()

        assert ev.img_ids == [1]
        assert len(ev.coco_results["keypoints"]) == 1

    def test_no_overlap_across_ranks_all_results_kept(self, tmp_path: Path) -> None:
        """When image_ids are disjoint across ranks, all predictions are preserved."""
        ev = self._make_evaluator(tmp_path)
        # Simulate rank 0 has already called update() with image_id=1
        ev.img_ids = [1]
        ev.coco_results["keypoints"] = [{"image_id": 1, "category_id": 1, "keypoints": [], "score": 0.9}]

        # all_gather returns rank-0 list + rank-1 list (no overlap)
        rank1_ids = [2]
        rank1_results = [{"image_id": 2, "category_id": 1, "keypoints": [], "score": 0.8}]
        call_count = [0]

        def _all_gather(x: list) -> list:
            call_count[0] += 1
            if call_count[0] == 1:
                return [x, rank1_ids]
            return [x, rank1_results]

        with patch("rfdetr.evaluation.coco_eval.all_gather", side_effect=_all_gather):
            ev.synchronize_between_processes()

        assert sorted(ev.img_ids) == [1, 2]
        image_ids_in_results = [r["image_id"] for r in ev.coco_results["keypoints"]]
        assert sorted(image_ids_in_results) == [1, 2]

    def test_ddp_padding_duplicate_image_id_deduped(self, tmp_path: Path) -> None:
        """DDP DistributedSampler padding: same image_id on two ranks → only rank-0 results kept."""
        ev = self._make_evaluator(tmp_path)
        # image_id=1 on BOTH ranks (padding), image_id=2 only on rank-1
        rank0_ids = [1]
        rank1_ids = [1, 2]
        rank0_results = [{"image_id": 1, "category_id": 1, "keypoints": [0.9], "score": 0.9}]
        rank1_results = [
            {"image_id": 1, "category_id": 1, "keypoints": [0.9], "score": 0.9},  # duplicate
            {"image_id": 2, "category_id": 1, "keypoints": [0.5], "score": 0.8},
        ]
        call_count = [0]

        def _all_gather(x: list) -> list:
            call_count[0] += 1
            if call_count[0] == 1:
                return [rank0_ids, rank1_ids]
            return [rank0_results, rank1_results]

        with patch("rfdetr.evaluation.coco_eval.all_gather", side_effect=_all_gather):
            ev.synchronize_between_processes()

        assert sorted(ev.img_ids) == [1, 2]
        image_ids_in_results = [r["image_id"] for r in ev.coco_results["keypoints"]]
        # image_id=1 from rank-0 only (not duplicated), image_id=2 from rank-1
        assert image_ids_in_results.count(1) == 1, "image_id=1 must appear exactly once (no DDP duplicate)"
        assert image_ids_in_results.count(2) == 1

    def test_ddp_padding_rank0_predictions_chosen_over_rank1(self, tmp_path: Path) -> None:
        """When image_id appears on rank-0 and rank-1, rank-0's prediction is kept (first-wins)."""
        ev = self._make_evaluator(tmp_path)
        rank0_ids = [1]
        rank1_ids = [1]
        rank0_results = [{"image_id": 1, "category_id": 1, "keypoints": [], "score": 0.9}]
        rank1_results = [{"image_id": 1, "category_id": 1, "keypoints": [], "score": 0.5}]
        call_count = [0]

        def _all_gather(x: list) -> list:
            call_count[0] += 1
            if call_count[0] == 1:
                return [rank0_ids, rank1_ids]
            return [rank0_results, rank1_results]

        with patch("rfdetr.evaluation.coco_eval.all_gather", side_effect=_all_gather):
            ev.synchronize_between_processes()

        assert len(ev.coco_results["keypoints"]) == 1
        assert ev.coco_results["keypoints"][0]["score"] == pytest.approx(0.9), "rank-0 prediction must win"

    def test_multiple_detections_same_image_all_kept(self, tmp_path: Path) -> None:
        """Multiple DT per image (multi-instance) on the owning rank are all preserved."""
        ev = self._make_evaluator(tmp_path)
        # rank-0 has 3 detections for image_id=1 (3 distinct instances)
        rank0_ids = [1]
        rank0_results = [
            {"image_id": 1, "category_id": 1, "keypoints": [], "score": 0.9},
            {"image_id": 1, "category_id": 1, "keypoints": [], "score": 0.8},
            {"image_id": 1, "category_id": 1, "keypoints": [], "score": 0.7},
        ]
        call_count = [0]

        def _all_gather(x: list) -> list:
            call_count[0] += 1
            if call_count[0] == 1:
                return [rank0_ids]
            return [rank0_results]

        with patch("rfdetr.evaluation.coco_eval.all_gather", side_effect=_all_gather):
            ev.synchronize_between_processes()

        assert len(ev.coco_results["keypoints"]) == 3, "all 3 per-image detections must be kept"


def test_coco_evaluator_skips_unmapped_labels_when_label2cat_is_present(tmp_path: Path) -> None:
    """A non-identity label2cat map should not fall back to raw category IDs for unmapped labels."""
    annotation_path = tmp_path / "person_keypoints_val2017.json"
    _write_person_keypoint_coco(annotation_path)
    coco_gt = COCO(str(annotation_path))
    coco_gt.label2cat = {1: 1}
    evaluator = CocoEvaluator(coco_gt, ["keypoints"])
    keypoints = np.asarray(coco_gt.anns[1]["keypoints"], dtype=np.float32).reshape(1, 17, 3)

    evaluator.update(
        {
            1: {
                "boxes": torch.tensor(
                    [[10.0, 20.0, 60.0, 80.0], [10.0, 20.0, 60.0, 80.0]],
                    dtype=torch.float32,
                ),
                "scores": torch.tensor([0.99, 0.98], dtype=torch.float32),
                "labels": torch.tensor([0, 1], dtype=torch.int64),
                "keypoints": torch.as_tensor(np.concatenate([keypoints, keypoints], axis=0), dtype=torch.float32),
            }
        }
    )

    results = evaluator.coco_results["keypoints"]
    assert len(results) == 1
    assert results[0]["category_id"] == 1
