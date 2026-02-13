# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from typing import List

from faster_coco_eval.utils.pytorch import FasterCocoEvaluator


class CocoEvaluator(FasterCocoEvaluator):
    def __init__(self, coco_gt, iou_types: List[str], max_dets: int = 100) -> None:
        super().__init__(coco_gt, iou_types, max_dets=max_dets)
