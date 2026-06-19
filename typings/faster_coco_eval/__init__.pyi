from __future__ import annotations
from faster_coco_eval.core import coco
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core import faster_eval_api as cocoeval
from faster_coco_eval.core.faster_eval_api import COCOeval_faster
from faster_coco_eval.core import mask
import sys as sys
from . import core
from . import faster_eval_api_cpp
from . import mask_api_new_cpp
from . import version
__all__: list = ['init_as_pycocotools', 'mask', 'coco', 'cocoeval', 'COCO', 'COCOeval_faster', '__author__', '__version__']
def init_as_pycocotools():
    ...
__author__: str = 'MiXaiLL76'
__version__: str = '1.7.2'
