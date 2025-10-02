"""
Early stopping callback for RF-DETR training
"""

from logging import getLogger

logger = getLogger(__name__)

class EarlyStoppingCallback:
    """
    Early stopping callback that monitors mAP and stops training if no improvement 
    over a threshold is observed for a specified number of epochs.
    
    Args:
        patience (int): Number of epochs with no improvement to wait before stopping
        min_delta (float): Minimum change in mAP to qualify as improvement
        use_ema (bool): Whether to use EMA model metrics for early stopping
        verbose (bool): Whether to print early stopping messages
    """
    
    def __init__(self, model, patience=5, min_delta=0.001, use_ema=False, verbose=True, segmentation_head=False):
        self.patience = patience
        self.min_delta = min_delta
        self.use_ema = use_ema
        self.verbose = verbose
        self.best_map = 0.0
        self.counter = 0
        self.model = model
        self.segmentation_head = segmentation_head
        
    def update(self, log_stats):
        """Update early stopping state based on epoch validation metrics"""
        regular_map = None
        ema_map = None
        
        if 'test_coco_eval_bbox' in log_stats:
            if not self.segmentation_head:
                regular_map = log_stats['test_coco_eval_bbox'][0]
            else:
                regular_map = log_stats['test_coco_eval_masks'][0]
        
        if 'ema_test_coco_eval_bbox' in log_stats:
            if not self.segmentation_head:
                ema_map = log_stats['ema_test_coco_eval_bbox'][0]
            else:
                ema_map = log_stats['ema_test_coco_eval_masks'][0]
        
        current_map = None
        if regular_map is not None and ema_map is not None:
            if self.use_ema:
                current_map = ema_map
                metric_source = "EMA"
            else:
                current_map = max(regular_map, ema_map)
                metric_source = "max(regular, EMA)"
        elif ema_map is not None:
            current_map = ema_map
            metric_source = "EMA"
        elif regular_map is not None:
            current_map = regular_map
            metric_source = "regular"
        else:
            if self.verbose:
                raise ValueError("No valid mAP metric found!")
            return
        
        if self.verbose:
            print(f"Early stopping: Current mAP ({metric_source}): {current_map:.4f}, Best: {self.best_map:.4f}, Diff: {current_map - self.best_map:.4f}, Min delta: {self.min_delta}")
        
        if current_map > self.best_map + self.min_delta:
            self.best_map = current_map
            self.counter = 0
            logger.info(f"Early stopping: mAP improved to {current_map:.4f} using {metric_source} metric")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: No improvement in mAP for {self.counter} epochs (best: {self.best_map:.4f}, current: {current_map:.4f})")
            
        if self.counter >= self.patience:
            print(f"Early stopping triggered: No improvement above {self.min_delta} threshold for {self.patience} epochs")
            if self.model:
                self.model.request_early_stop()