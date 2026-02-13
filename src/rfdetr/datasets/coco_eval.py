# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import inspect
from typing import List

from faster_coco_eval.utils.pytorch import FasterCocoEvaluator


class CocoEvaluator(FasterCocoEvaluator):
    """Compatibility wrapper for FasterCocoEvaluator across versions."""

    def __init__(self, coco_gt, iou_types: List[str], max_dets: int = 100) -> None:
        """
        Initialize the evaluator with optional max detections support.

        Args:
            coco_gt: COCO ground truth dataset.
            iou_types: IoU types to evaluate (e.g., ["bbox", "segm"]).
            max_dets: Max detections per image if supported by the evaluator.
        """
        init_sig = inspect.signature(FasterCocoEvaluator.__init__)
        if "max_dets" in init_sig.parameters:
            super().__init__(coco_gt, iou_types, max_dets=max_dets)
        elif "max_detections" in init_sig.parameters:
            super().__init__(coco_gt, iou_types, max_detections=max_dets)
        else:
            super().__init__(coco_gt, iou_types)
            if hasattr(self, "max_dets"):
                self.max_dets = max_dets
