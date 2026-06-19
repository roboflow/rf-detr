from __future__ import annotations

import sys as sys

from faster_coco_eval.core import coco, mask
from faster_coco_eval.core import faster_eval_api as cocoeval
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster

__all__: list = [
    "init_as_pycocotools",
    "mask",
    "coco",
    "cocoeval",
    "COCO",
    "COCOeval_faster",
    "__author__",
    "__version__",
]

def init_as_pycocotools(): ...

__author__: str = "MiXaiLL76"
__version__: str = "1.7.2"
