# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""COCO evaluator for ONNX/TRT export benchmarking.

Provides :class:`CocoEvaluator` used by :mod:`rfdetr.export.benchmark` to compute mAP during ONNX and TensorRT
inference benchmarks.

Implementation mirrors torchvision's evaluator structure but uses ``faster_coco_eval`` as the runtime backend.
"""

import contextlib
import copy
import os
from typing import Any

import faster_coco_eval.core.mask as mask_util
import numpy as np
from faster_coco_eval import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval

from rfdetr.utilities.distributed import all_gather
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _ensure_faster_coco(coco_gt: Any) -> COCO:
    """Return a faster-coco-eval COCO object for evaluator construction."""
    if isinstance(coco_gt, COCO) and hasattr(coco_gt, "cat_img_map"):
        return coco_gt

    faster_coco = COCO()
    faster_coco.dataset = copy.deepcopy(coco_gt.dataset)
    faster_coco.createIndex()
    label2cat = getattr(coco_gt, "label2cat", None)
    if label2cat is not None:
        setattr(faster_coco, "label2cat", copy.deepcopy(label2cat))
    return faster_coco


def _backfill_num_keypoints(coco_gt: COCO) -> None:
    """Populate missing COCO ``num_keypoints`` fields from visibility flags."""
    annotations_by_id: dict[int, dict[str, Any]] = {}
    for annotation in coco_gt.dataset.get("annotations", []):
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, int):
            annotations_by_id[annotation_id] = annotation
        keypoints = annotation.get("keypoints")
        if "num_keypoints" not in annotation and isinstance(keypoints, list):
            annotation["num_keypoints"] = sum(1 for visibility in keypoints[2::3] if visibility > 0)

    for annotation_id, annotation in coco_gt.anns.items():
        keypoints = annotation.get("keypoints")
        if "num_keypoints" not in annotation and isinstance(keypoints, list):
            annotation["num_keypoints"] = sum(1 for visibility in keypoints[2::3] if visibility > 0)
        dataset_annotation = annotations_by_id.get(annotation_id)
        if dataset_annotation is not None and "num_keypoints" in annotation:
            dataset_annotation["num_keypoints"] = annotation["num_keypoints"]


def _load_coco_results(coco_gt: COCO, results: list[dict[str, Any]]) -> COCO:
    """Build a COCO detections object, including the empty-result case."""
    if results:
        return COCO.loadRes(coco_gt, results)

    coco_dt = COCO()
    coco_dt.dataset["info"] = copy.deepcopy(coco_gt.dataset.get("info", {}))
    coco_dt.dataset["images"] = copy.deepcopy(coco_gt.dataset.get("images", []))
    coco_dt.dataset["categories"] = copy.deepcopy(coco_gt.dataset.get("categories", []))
    coco_dt.dataset["annotations"] = []
    coco_dt.createIndex()
    return coco_dt


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [x1, y1, w, h]."""
    boxes = boxes.copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


class CocoEvaluator:
    """COCO evaluator that works in distributed mode."""

    def __init__(self, coco_gt: Any, iou_types: list[str], max_dets: int = 100) -> None:
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(_ensure_faster_coco(coco_gt))
        if "keypoints" in iou_types:
            _backfill_num_keypoints(coco_gt)
        self.coco_gt = coco_gt
        self.max_dets = max_dets
        # label2cat maps contiguous model label indices back to original COCO category_ids.
        # Set by CocoDetection when cat2label remapping is active; None otherwise.
        self.label2cat: dict[int, int] | None = getattr(coco_gt, "label2cat", None)

        self.iou_types = iou_types
        self.coco_eval: dict[str, COCOeval] = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].params.maxDets = [20] if iou_type == "keypoints" else [1, 10, max_dets]

        self.img_ids: list[int] = []
        self.coco_results: dict[str, list[dict[str, Any]]] = {k: [] for k in iou_types}
        self.cat_ids = set(coco_gt.cats.keys())
        self._prefer_raw_category_ids = False

    def _resolve_category_id(self, label: int, use_raw_category_ids: bool) -> int | None:
        """Resolve a predicted label to a COCO category_id."""
        if use_raw_category_ids:
            return label if label in self.cat_ids else None
        if self.label2cat is not None:
            category_id = self.label2cat.get(label)
            return category_id if category_id in self.cat_ids else None
        if label in self.cat_ids:
            return label
        return None

    def _should_use_raw_category_ids(self, labels: list[int]) -> bool:
        """Detect whether model predictions are already raw COCO category IDs."""
        if self.label2cat is None:
            return True
        if self._prefer_raw_category_ids:
            return True
        uses_raw_ids = list(self.label2cat.keys()) == list(self.label2cat.values())
        if uses_raw_ids:
            self._prefer_raw_category_ids = True
            return True
        return False

    def update(self, predictions: dict[int, Any]) -> None:
        """Accumulate per-image predictions."""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            self.coco_results[iou_type].extend(results)

    def synchronize_between_processes(self) -> None:
        """Merge image IDs and COCO result records across distributed processes."""
        gathered_img_ids = all_gather(self.img_ids)
        self.img_ids = sorted({image_id for rank_img_ids in gathered_img_ids for image_id in rank_img_ids})
        for iou_type in self.iou_types:
            gathered_results = all_gather(self.coco_results[iou_type])
            self.coco_results[iou_type] = [result for rank_results in gathered_results for result in rank_results]

    def accumulate(self) -> None:
        """Accumulate per-image evaluation results into mean metrics."""
        for iou_type, coco_eval in self.coco_eval.items():
            self._evaluate(iou_type, coco_eval)
            coco_eval.accumulate()
            patched_pycocotools_summarize(coco_eval)

    def summarize(self) -> None:
        """Print and log COCO summary statistics."""
        for iou_type, coco_eval in self.coco_eval.items():
            logger.info("IoU metric: {}".format(iou_type))
            patched_pycocotools_summarize(coco_eval)

    def _evaluate(self, iou_type: str, coco_eval: COCOeval) -> None:
        """Run faster-coco-eval evaluation for accumulated COCO result records."""
        results = self.coco_results[iou_type]
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = _load_coco_results(self.coco_gt, results)
                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = list(np.unique(self.img_ids))
                coco_eval.evaluate()

    def prepare(self, predictions: dict[int, Any], iou_type: str) -> list[dict[str, Any]]:
        """Convert predictions to COCO format for the given iou_type."""
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format bounding-box predictions as COCO result dicts."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _xyxy_to_xywh(boxes.cpu().numpy()).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)
            for k, box in enumerate(boxes):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "bbox": box,
                        "score": scores[k],
                    }
                )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format segmentation mask predictions as COCO result dicts."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)

            rles = [
                mask_util.encode(np.array(mask.cpu()[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            for k, rle in enumerate(rles):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "segmentation": rle,
                        "score": scores[k],
                    }
                )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Format keypoint predictions as COCO result dicts."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _xyxy_to_xywh(boxes.cpu().numpy()).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()
            use_raw_category_ids = self._should_use_raw_category_ids(labels)
            for k, keypoint in enumerate(keypoints):
                category_id = self._resolve_category_id(labels[k], use_raw_category_ids)
                if category_id is None:
                    continue
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": category_id,
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                )
        return coco_results


#################################################################
# From pycocotools, patched first _summarize() call to use
# maxDets[-1] instead of hardcoded 100.
#################################################################
def patched_pycocotools_summarize(self: COCOeval) -> None:
    """Compute and display summary metrics for evaluation results."""

    def _summarize(ap: int = 1, iou_thr: float | None = None, area_rng: str = "all", max_dets: int = 100) -> float:
        p = self.params
        log_template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        title_str = "Average Precision" if ap == 1 else "Average Recall"
        type_str = "(AP)" if ap == 1 else "(AR)"
        iou_str = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iou_thr is None else "{:0.2f}".format(iou_thr)
        )

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
        if ap == 1:
            s = self.eval["precision"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        mean_s = -1 if len(s[s > -1]) == 0 else float(np.mean(s[s > -1]))
        logger.info(log_template.format(title_str, type_str, iou_str, area_rng, max_dets, mean_s))
        return mean_s

    def _summarizeDets() -> np.ndarray:  # noqa: N802
        stats = np.zeros((12,))
        stats[0] = _summarize(1, max_dets=self.params.maxDets[2])
        stats[1] = _summarize(1, iou_thr=0.5, max_dets=self.params.maxDets[2])
        stats[2] = _summarize(1, iou_thr=0.75, max_dets=self.params.maxDets[2])
        stats[3] = _summarize(1, area_rng="small", max_dets=self.params.maxDets[2])
        stats[4] = _summarize(1, area_rng="medium", max_dets=self.params.maxDets[2])
        stats[5] = _summarize(1, area_rng="large", max_dets=self.params.maxDets[2])
        stats[6] = _summarize(0, max_dets=self.params.maxDets[0])
        stats[7] = _summarize(0, max_dets=self.params.maxDets[1])
        stats[8] = _summarize(0, max_dets=self.params.maxDets[2])
        stats[9] = _summarize(0, area_rng="small", max_dets=self.params.maxDets[2])
        stats[10] = _summarize(0, area_rng="medium", max_dets=self.params.maxDets[2])
        stats[11] = _summarize(0, area_rng="large", max_dets=self.params.maxDets[2])
        return stats

    def _summarizeKps() -> np.ndarray:  # noqa: N802
        stats = np.zeros((10,))
        stats[0] = _summarize(1, max_dets=20)
        stats[1] = _summarize(1, max_dets=20, iou_thr=0.5)
        stats[2] = _summarize(1, max_dets=20, iou_thr=0.75)
        stats[3] = _summarize(1, max_dets=20, area_rng="medium")
        stats[4] = _summarize(1, max_dets=20, area_rng="large")
        stats[5] = _summarize(0, max_dets=20)
        stats[6] = _summarize(0, max_dets=20, iou_thr=0.5)
        stats[7] = _summarize(0, max_dets=20, iou_thr=0.75)
        stats[8] = _summarize(0, max_dets=20, area_rng="medium")
        stats[9] = _summarize(0, max_dets=20, area_rng="large")
        return stats

    if not self.eval:
        raise Exception("Please run accumulate() first")
    iou_type = self.params.iouType
    if iou_type == "segm" or iou_type == "bbox":
        summarize = _summarizeDets
    elif iou_type == "keypoints":
        summarize = _summarizeKps
    self.stats = summarize()
