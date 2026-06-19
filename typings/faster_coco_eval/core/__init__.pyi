from __future__ import annotations
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster
from . import coco
from . import cocoeval
from . import faster_eval_api
from . import mask
__all__: list = ['COCO', 'COCOeval_faster']
