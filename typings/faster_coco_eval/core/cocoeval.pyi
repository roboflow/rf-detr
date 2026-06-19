from __future__ import annotations
from collections import defaultdict
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core import mask as maskUtils
from faster_coco_eval import faster_eval_api_cpp as _C
import logging as logging
import numpy
import numpy as np
import os as os
import sys as sys
import typing
__all__: list[str] = ['COCO', 'COCOeval', 'Params', 'defaultdict', 'iouTypeT', 'logger', 'logging', 'maskUtils', 'np', 'os', 'sys']
class COCOeval:
    def __init__(self, cocoGt: typing.Optional[faster_coco_eval.core.coco.COCO] = None, cocoDt: typing.Optional[faster_coco_eval.core.coco.COCO] = None, iouType: typing.Literal['segm', 'bbox', 'keypoints', 'keypoints_crowd', 'boundary'] = 'segm', ranges: typing.Optional[dict] = {'small': [0, 1024], 'medium': [1024, 9216], 'large': [9216, 10000000000.0]}, print_function: typing.Callable = logging.Logger.info, extra_calc: bool = False, kpt_oks_sigmas: typing.Optional[typing.List[float]] = None, use_area: typing.Optional[bool] = True, lvis_style: bool = False, separate_eval: bool = False, boundary_dilation_ratio: float = 0.02, boundary_cpu_count: int = 4):
        """
        Initialize CocoEval using coco APIs for gt and dt.
        
                Args:
                    cocoGt (Optional[COCO]): Object with ground truth annotations.
                    cocoDt (Optional[COCO]): Object with detection annotations.
                    iouType (iouTypeT): Type of the intersection over union, defaults to "segm".
                    ranges (Optional[dict]): Dictionary of area ranges, defaults to predefined ranges.
                    print_function (Callable): Function to print output, defaults to logger.info.
                    extra_calc (bool): Whether to perform extra calculations, defaults to False.
                    kpt_oks_sigmas (Optional[List[float]]): List of sigmas for keypoint evaluation, defaults to None.
                    use_area (Optional[bool]): If gt annotations (eg. CrowdPose) do not have 'area', set use_area=False.
                    lvis_style (bool): Whether to use LVIS style evaluation, defaults to False.
                    separate_eval (bool): Whether to perform separate evaluation, defaults to False.
                    boundary_dilation_ratio (float): Ratio for boundary dilation, defaults to 0.02.
                    boundary_cpu_count (int): Number of CPUs for boundary computation, defaults to min(os.cpu_count(), 4).
                
        """
    def __repr__(self) -> str:
        """
        
                Returns:
                    str: Representation of the class with author and version info.
                
        """
    def __str__(self) -> str:
        """
        
                Returns:
                    str: String representation after summarization.
                
        """
    def _prepare(self):
        """
        Prepare self.gt_dataset and self.dt_dataset for evaluation based on
                params.
        
                Populates datasets with annotations, computes RLEs and
                boundaries, and applies LVIS filtering if necessary.
                
        """
    def _prepare_freq_group(self) -> list:
        """
        Prepare frequency group for LVIS evaluation.
        
                Returns:
                    list: Frequency groups, grouping category indices by frequency label.
                
        """
    def _summarize(self, ap = 1, iouThr = None, areaRng = 'all', maxDets = 100, freq_group_idx = None, catIds = None):
        """
        Summarize evaluation results.
        
                Args:
                    ap (int): 1 for average precision, 0 for average recall.
                    iouThr (float, optional): Specific IoU threshold.
                    areaRng (str): Area range label.
                    maxDets (int): Maximum detections.
                    freq_group_idx (int, optional): Frequency group index (for LVIS).
                    catIds (list, optional): Category IDs to summarize.
        
                Returns:
                    float: Summary metric.
                
        """
    def accumulate(self, p = None):
        """
        Deprecated. Use COCOeval_faster.accumulate instead.
        
                Args:
                    p: Unused.
        
                Raises:
                    DeprecationWarning: Always.
                
        """
    def computeIoU(self, imgId: int, catId: int) -> typing.Union[typing.List[float], numpy.ndarray]:
        """
        Compute IoU between ground truth and detection for a given image and
                category.
        
                Args:
                    imgId (int): Image ID.
                    catId (int): Category ID.
        
                Returns:
                    Union[List[float], np.ndarray]: IoUs between gt and dt for the given image and category.
                
        """
    def computeOks(self, imgId: int, catId: int) -> numpy.ndarray:
        """
        Compute OKS between ground truth and detection for a given image and
                category.
        
                Args:
                    imgId (int): Image ID.
                    catId (int): Category ID.
        
                Returns:
                    np.ndarray: OKS between gt and dt for the given image and category.
                
        """
    def evaluate(self):
        """
        Deprecated. Use COCOeval_faster.evaluate instead.
        
                Raises:
                    DeprecationWarning: Always.
                
        """
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        Deprecated. Use COCOeval_faster.evaluateImg instead.
        
                Args:
                    imgId: Image ID.
                    catId: Category ID.
                    aRng: Area range.
                    maxDet: Maximum detections.
        
                Raises:
                    DeprecationWarning: Always.
                
        """
    def get_type_result(self, first: float = 0.01, second: float = 0.85) -> list:
        """
        Calculate type results for easy, medium, and hard splits.
        
                Args:
                    first (float): Threshold for 'easy' crowdIndex.
                    second (float): Threshold for 'medium' crowdIndex.
        
                Returns:
                    list: List of scores for [easy, medium, hard].
                
        """
    def split(self, gt_file: str, first: float = 0.01, second: float = 0.85):
        """
        Split images into easy, medium, hard according to 'crowdIndex'.
        
                Args:
                    gt_file (str): Path to the ground truth file.
                    first (float): Threshold for 'easy' crowdIndex.
                    second (float): Threshold for 'medium' crowdIndex.
        
                Returns:
                    tuple: Lists of image ids for (easy, medium, hard).
                
        """
    def summarize(self):
        """
        Compute and display summary metrics for evaluation results. After
                calling this method, self.all_stats will contain the **full results**
                including all metrics while self.stats contains a subset of the most
                commonly used metrics.
        
                Note:
                    This function can *only* be applied on the default parameter setting.
                
        """
    @property
    def print_function(self) -> typing.Callable:
        """
        
                Returns:
                    Callable: The function used for printing/logging.
                
        """
    @print_function.setter
    def print_function(self, value: typing.Callable):
        """
        Set the print function.
        
                Args:
                    value (Callable): The new print function.
                
        """
class Params:
    """
    Params for coco evaluation api.
    """
    def __init__(self, iouType: typing.Literal['segm', 'bbox', 'keypoints', 'keypoints_crowd', 'boundary'] = 'segm', kpt_sigmas: typing.Optional[typing.List[float]] = None, ranges: typing.Optional[dict] = {'small': [0, 1024], 'medium': [1024, 9216], 'large': [9216, 10000000000.0]}):
        """
        Initialize Params for COCO evaluation API.
        
                Args:
                    iouType (iouTypeT): Either "segm", "bbox", "boundary", "keypoints", or "keypoints_crowd".
                    kpt_sigmas (Optional[List[float]]): List of keypoint sigma values.
                    ranges (Optional[dict]): Dictionary defining area ranges with labels as keys and [min, max] as values.
                
        """
    def setDetParams(self, ranges: dict):
        """
        Set parameters for detection evaluation.
        """
    def setKpParams(self):
        """
        Set parameters for keypoint evaluation.
        """
    @property
    def area_rng(self) -> list:
        """
        
                Returns:
                    list: Area ranges.
                
        """
    @property
    def area_rng_lbl(self) -> list:
        """
        
                Returns:
                    list: Area range labels.
                
        """
    @property
    def cat_ids(self) -> list:
        """
        
                Returns:
                    list: Category IDs.
                
        """
    @property
    def img_count_lbl(self) -> list:
        """
        
                Returns:
                    list: Image count frequency labels.
                
        """
    @property
    def img_ids(self) -> list:
        """
        
                Returns:
                    list: Image IDs.
                
        """
    @property
    def iou_thrs(self) -> numpy.ndarray:
        """
        
                Returns:
                    np.ndarray: IOU thresholds.
                
        """
    @property
    def iou_type(self) -> typing.Literal['segm', 'bbox', 'keypoints', 'keypoints_crowd', 'boundary']:
        """
        
                Returns:
                    iouTypeT: IOU type.
                
        """
    @property
    def max_dets(self) -> list:
        """
        
                Returns:
                    list: Maximum number of detections.
                
        """
    @property
    def rec_thrs(self) -> numpy.ndarray:
        """
        
                Returns:
                    np.ndarray: Recall thresholds.
                
        """
    @property
    def useSegm(self) -> int:
        """
        
                Returns:
                    int: 1 if iouType is "segm", else 0.
                
        """
    @useSegm.setter
    def useSegm(self, value: int):
        """
        Set segmentation mode and issue deprecation warning.
        
                Args:
                    value (int): 1 for segm, 0 for bbox.
                
        """
    @property
    def use_cats(self) -> int:
        """
        
                Returns:
                    int: Whether to use categories.
                
        """
__author__: str = 'MiXaiLL76'
__version__: str = '1.7.2'
iouTypeT: typing._LiteralGenericAlias  # value = typing.Literal['segm', 'bbox', 'keypoints', 'keypoints_crowd', 'boundary']
logger: logging.Logger  # value = <Logger faster_coco_eval.core.cocoeval (INFO)>
