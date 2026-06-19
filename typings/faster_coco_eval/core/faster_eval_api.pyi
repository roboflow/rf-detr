from __future__ import annotations
import copy as copy
import faster_coco_eval.core.cocoeval
from faster_coco_eval.core.cocoeval import COCOeval as COCOevalBase
from faster_coco_eval import faster_eval_api_cpp as _C
import itertools as itertools
import logging as logging
import numpy as np
import time as time
__all__: list[str] = ['COCOeval', 'COCOevalBase', 'COCOeval_faster', 'copy', 'itertools', 'logger', 'logging', 'np', 'time']
class COCOeval(COCOeval_faster):
    @property
    def print_function(self):
        """
        Return the print function.
        
                Returns:
                    Callable: The built-in print function.
                
        """
class COCOeval_faster(faster_coco_eval.core.cocoeval.COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the
        functions evaluateImg() and accumulate() are implemented in C++ to speedup
        evaluation.
    """
    @staticmethod
    def calc_auc(recall_list: typing.Union[typing.List[float], numpy.ndarray], precision_list: typing.Union[typing.List[float], numpy.ndarray], method: str = 'c++'):
        """
        Calculate area under precision recall curve.
        
                Args:
                    recall_list (Union[List[float], np.ndarray]): List or array of recall values.
                    precision_list (Union[List[float], np.ndarray]): List or array of precision values.
                    method (str, optional): Method to calculate auc. Defaults to "c++".
        
                Returns:
                    float: Area under precision recall curve.
                
        """
    def accumulate(self):
        """
        Accumulate per image evaluation results and store the result in
                self.eval.
        
                Does not support changing parameter settings from those used by
                self.evaluate()
                
        """
    def compute_mAUC(self) -> float:
        """
        Compute the mean Area Under Curve (mAUC) metric.
        
                Returns:
                    float: Mean AUC across all categories and area ranges.
                
        """
    def compute_mIoU(self) -> float:
        """
        Compute the mean Intersection over Union (mIoU) metric.
        
                Returns:
                    float: Mean IoU across all matched detections and ground truths.
                
        """
    def evaluate(self):
        """
        Run per image evaluation on given images and store results in
                self.evalImgs_cpp, a datastructure that isn't readable from Python but
                is used by a c++ implementation of accumulate().
        
                Unlike the original COCO PythonAPI, we don't populate the
                datastructure self.evalImgs because this datastructure is a
                computational bottleneck.
                
        """
    def math_matches(self):
        """
        Analyze matched detections and ground truths to assign true
                positive, false positive, and false negative flags, and update
                detection and ground truth annotations in-place.
        
                Returns:
                    None
                
        """
    def run(self):
        """
        Wrapper function which runs the evaluation.
        
                Calls evaluate(), accumulate(), and summarize() in sequence.
        
                Returns:
                    None
                
        """
    def summarize(self):
        """
        Summarize and finalize the statistics of the evaluation.
        
                Returns:
                    None
                
        """
    @property
    def extended_metrics(self):
        """
        Computes extended evaluation metrics for object detection results.
        
                Calculates per-class and overall (macro) metrics such as mean average precision (mAP) at IoU thresholds,
                precision, recall, and F1-score. Results are computed using evaluation results stored in the object.
                For each class, if categories are used, metrics are reported separately and for the overall dataset.
        
                Returns:
                    dict: A dictionary with the following keys:
                        - 'class_map' (list of dict): List of per-class and overall metrics, each as a dictionary containing:
                            - 'class' (str): Class name or "all" for macro metrics.
                            - 'map@50:95' (float): Mean average precision at IoU 0.50:0.95.
                            - 'map@50' (float): Mean average precision at IoU 0.50.
                            - 'precision' (float): Macro-averaged precision.
                            - 'recall' (float): Macro-averaged recall.
                        - 'map' (float): Overall mean average precision at IoU 0.50.
                        - 'precision' (float): Macro-averaged precision for the best F1-score.
                        - 'recall' (float): Macro-averaged recall for the best F1-score.
        
                Notes:
                    - Uses COCO-style evaluation results (precision and scores arrays).
                    - Filters out classes with NaN results in any metric.
                    - The best F1-score across confidence thresholds is used to select macro precision and recall.
                    - Precision and recall are computed from actual (non-interpolated) detection data to avoid
                      over-estimating precision when false positives exist below the recall ceiling.
                
        """
    @property
    def stats_as_dict(self):
        """
        Return the evaluation statistics as a dictionary with descriptive
                labels.
        
                Returns:
                    dict[str, float]: Dictionary mapping metric names to their values.
                
        """
logger: logging.Logger  # value = <Logger faster_coco_eval.core.faster_eval_api (INFO)>
