# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

plt.ioff()

PLOT_FILE_NAME = "metrics_plot.png"

_T = TypeVar("_T")


def safe_index(arr: Sequence[_T], idx: int) -> Optional[_T]:
    return arr[idx] if 0 <= idx < len(arr) else None


class MetricsPlotSink:
    """
    The MetricsPlotSink class records training metrics and saves them to a plot.

    Args:
        output_dir (str): Directory where the plot will be saved.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.history: List[Dict[str, Any]] = []

    def update(self, values: Dict[str, Any]) -> None:
        self.history.append(values)

    def save(self) -> None:
        if not self.history:
            print("No data to plot.")
            return

        def get_array(key: str) -> np.ndarray:
            return np.array([h[key] for h in self.history if key in h])

        epochs = get_array('epoch')
        train_loss = get_array('train_loss')
        test_loss = get_array('test_loss')
        test_coco_eval_bbox = [h['test_coco_eval_bbox'] for h in self.history if 'test_coco_eval_bbox' in h]
        ap50_90_bbox = np.array([safe_index(x, 0) for x in test_coco_eval_bbox if x is not None], dtype=np.float32)
        ap50_bbox = np.array([safe_index(x, 1) for x in test_coco_eval_bbox if x is not None], dtype=np.float32)
        ar50_90_bbox = np.array([safe_index(x, 8) for x in test_coco_eval_bbox if x is not None], dtype=np.float32)

        ema_coco_eval_bbox = [h['ema_test_coco_eval_bbox'] for h in self.history if 'ema_test_coco_eval_bbox' in h]
        ema_ap50_90_bbox = np.array([safe_index(x, 0) for x in ema_coco_eval_bbox if x is not None], dtype=np.float32)
        ema_ap50_bbox = np.array([safe_index(x, 1) for x in ema_coco_eval_bbox if x is not None], dtype=np.float32)
        ema_ar50_90_bbox = np.array([safe_index(x, 8) for x in ema_coco_eval_bbox if x is not None], dtype=np.float32)

        test_coco_eval_masks = [h['test_coco_eval_masks'] for h in self.history if 'test_coco_eval_masks' in h]
        ap50_90_masks = np.array([safe_index(x, 0) for x in test_coco_eval_masks if x is not None], dtype=np.float32)
        ap50_masks = np.array([safe_index(x, 1) for x in test_coco_eval_masks if x is not None], dtype=np.float32)
        ar50_90_masks = np.array([safe_index(x, 8) for x in test_coco_eval_masks if x is not None], dtype=np.float32)

        ema_coco_eval_masks = [h['ema_test_coco_eval_masks'] for h in self.history if 'ema_test_coco_eval_masks' in h]
        ema_ap50_90_masks = np.array([safe_index(x, 0) for x in ema_coco_eval_masks if x is not None], dtype=np.float32)
        ema_ap50_masks = np.array([safe_index(x, 1) for x in ema_coco_eval_masks if x is not None], dtype=np.float32)
        ema_ar50_90_masks = np.array([safe_index(x, 8) for x in ema_coco_eval_masks if x is not None], dtype=np.float32)

        if len(test_coco_eval_masks) or len(ema_coco_eval_masks):
            fig, axes = plt.subplots(4, 2, figsize=(18, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Training and Validation Loss
        r, c = 0, 0
        if len(epochs) > 0:
            if len(train_loss):
                axes[r][c].plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')
            if len(test_loss):
                axes[r][c].plot(epochs, test_loss, label='Validation Loss', marker='o', linestyle='--')
            axes[r][c].set_title('Training and Validation Loss')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('Loss Value')
            axes[r][c].legend()
            axes[r][c].grid(True)

        # BBox Average Precision @0.50
        r, c = 0, 1
        if ap50_bbox.size > 0 or ema_ap50_bbox.size > 0:
            if ap50_bbox.size > 0:
                axes[r][c].plot(epochs[:len(ap50_bbox)], ap50_bbox, marker='o', linestyle='-', label='Base Model')
            if ema_ap50_bbox.size > 0:
                axes[r][c].plot(epochs[:len(ema_ap50_bbox)], ema_ap50_bbox, marker='o', linestyle='--', label='EMA Model')
            axes[r][c].set_title('Average Precision @0.50 (BBox)')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('AP50')
            axes[r][c].legend()
            axes[r][c].grid(True)

        # BBox Average Precision @0.50:0.95
        r, c = 1, 0
        if ap50_90_bbox.size > 0 or ema_ap50_90_bbox.size > 0:
            if ap50_90_bbox.size > 0:
                axes[r][c].plot(epochs[:len(ap50_90_bbox)], ap50_90_bbox, marker='o', linestyle='-', label='Base Model')
            if ema_ap50_90_bbox.size > 0:
                axes[r][c].plot(epochs[:len(ema_ap50_90_bbox)], ema_ap50_90_bbox, marker='o', linestyle='--', label='EMA Model')
            axes[r][c].set_title('Average Precision @0.50:0.95 (BBox)')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('AP')
            axes[r][c].legend()
            axes[r][c].grid(True)

        # BBox Average Recall @0.50:0.95
        r, c = 1, 1
        if ar50_90_bbox.size > 0 or ema_ar50_90_bbox.size > 0:
            if ar50_90_bbox.size > 0:
                axes[r][c].plot(epochs[:len(ar50_90_bbox)], ar50_90_bbox, marker='o', linestyle='-', label='Base Model')
            if ema_ar50_90_bbox.size > 0:
                axes[r][c].plot(epochs[:len(ema_ar50_90_bbox)], ema_ar50_90_bbox, marker='o', linestyle='--', label='EMA Model')
            axes[r][c].set_title('Average Recall @0.50:0.95 (BBox)')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('AR')
            axes[r][c].legend()
            axes[r][c].grid(True)

        # Masks Average Precision @0.50
        r, c = 2, 0
        if ap50_masks.size > 0 or ema_ap50_masks.size > 0:
            if ap50_masks.size > 0:
                axes[r][c].plot(epochs[:len(ap50_masks)], ap50_masks, marker='o', linestyle='-', label='Base Model')
            if ema_ap50_masks.size > 0:
                axes[r][c].plot(epochs[:len(ema_ap50_masks)], ema_ap50_masks, marker='o', linestyle='--', label='EMA Model')
            axes[r][c].set_title('Average Precision @0.50 (Masks)')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('AP50')
            axes[r][c].legend()
            axes[r][c].grid(True)

        # Masks Average Precision @0.50:0.95
        r, c = 2, 1
        if ap50_90_masks.size > 0 or ema_ap50_90_masks.size > 0:
            if ap50_90_masks.size > 0:
                axes[r][c].plot(epochs[:len(ap50_90_masks)], ap50_90_masks, marker='o', linestyle='-', label='Base Model')
            if ema_ap50_90_masks.size > 0:
                axes[r][c].plot(epochs[:len(ema_ap50_90_masks)], ema_ap50_90_masks, marker='o', linestyle='--', label='EMA Model')
            axes[r][c].set_title('Average Precision @0.50:0.95 (Masks)')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('AP')
            axes[r][c].legend()
            axes[r][c].grid(True)

        # Masks Average Recall @0.50:0.95
        r, c = 3, 0
        if ar50_90_masks.size > 0 or ema_ar50_90_masks.size > 0:
            if ar50_90_masks.size > 0:
                axes[r][c].plot(epochs[:len(ar50_90_masks)], ar50_90_masks, marker='o', linestyle='-', label='Base Model')
            if ema_ar50_90_masks.size > 0:
                axes[r][c].plot(epochs[:len(ema_ar50_90_masks)], ema_ar50_90_masks, marker='o', linestyle='--', label='EMA Model')
            axes[r][c].set_title('Average Recall @0.50:0.95 (Masks)')
            axes[r][c].set_xlabel('Epoch Number')
            axes[r][c].set_ylabel('AR')
            axes[r][c].legend()
            axes[r][c].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{PLOT_FILE_NAME}")
        plt.close(fig)
        print(f"Results saved to {self.output_dir}/{PLOT_FILE_NAME}")


class MetricsTensorBoardSink:
    """
    Training metrics via TensorBoard.

    Args:
        output_dir (str): Directory where TensorBoard logs will be written.
    """

    def __init__(self, output_dir: str) -> None:
        if SummaryWriter:
            self.writer = SummaryWriter(log_dir=output_dir)
            print(f"TensorBoard logging initialized. To monitor logs, use 'tensorboard --logdir {output_dir}' and open http://localhost:6006/ in browser.")
        else:
            self.writer = None
            print("Unable to initialize TensorBoard. Logging is turned off for this session.  Run 'pip install tensorboard' to enable logging.")

    def update(self, values: Dict[str, Any]) -> None:
        if not self.writer:
            return

        epoch = values['epoch']

        if 'train_loss' in values:
            self.writer.add_scalar("Loss/Train", values['train_loss'], epoch)
        if 'test_loss' in values:
            self.writer.add_scalar("Loss/Test", values['test_loss'], epoch)

        if 'test_coco_eval_bbox' in values:
            coco_eval = values['test_coco_eval_bbox']
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                self.writer.add_scalar("Metrics/Base/AP50_90", ap50_90, epoch)
            if ap50 is not None:
                self.writer.add_scalar("Metrics/Base/AP50", ap50, epoch)
            if ar50_90 is not None:
                self.writer.add_scalar("Metrics/Base/AR50_90", ar50_90, epoch)

        if 'ema_test_coco_eval_bbox' in values:
            ema_coco_eval = values['ema_test_coco_eval_bbox']
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                self.writer.add_scalar("Metrics/EMA/AP50_90", ema_ap50_90, epoch)
            if ema_ap50 is not None:
                self.writer.add_scalar("Metrics/EMA/AP50", ema_ap50, epoch)
            if ema_ar50_90 is not None:
                self.writer.add_scalar("Metrics/EMA/AR50_90", ema_ar50_90, epoch)

        self.writer.flush()

    def close(self):
        if not self.writer:
            return

        self.writer.close()

class MetricsWandBSink:
    """
    Training metrics via W&B.

    Args:
        output_dir (str): Directory where W&B logs will be written locally.
        project (str, optional): Associate this training run with a W&B project. If None, W&B will generate a name based on the git repo name.
        run (str, optional): W&B run name. If None, W&B will generate a random name.
        config (dict, optional): Input parameters, like hyperparameters or data preprocessing settings for the run for later comparison.
    """

    def __init__(self, output_dir: str, project: Optional[str] = None, run: Optional[str] = None, config: Optional[dict] = None):
        self.output_dir = output_dir
        if wandb:
            self.run = wandb.init(
                project=project,
                name=run,
                config=config,
                dir=output_dir
            )
            print(f"W&B logging initialized. To monitor logs, open {wandb.run.url}.")
        else:
            self.run = None
            print("Unable to initialize W&B. Logging is turned off for this session. Run 'pip install wandb' to enable logging.")

    def update(self, values: dict):
        if not wandb or not self.run:
            return

        epoch = values['epoch']
        log_dict = {"epoch": epoch}

        if 'train_loss' in values:
            log_dict["Loss/Train"] = values['train_loss']
        if 'test_loss' in values:
            log_dict["Loss/Test"] = values['test_loss']

        if 'test_coco_eval_bbox' in values:
            coco_eval = values['test_coco_eval_bbox']
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                log_dict["Metrics/Base/AP50_90"] = ap50_90
            if ap50 is not None:
                log_dict["Metrics/Base/AP50"] = ap50
            if ar50_90 is not None:
                log_dict["Metrics/Base/AR50_90"] = ar50_90

        if 'ema_test_coco_eval_bbox' in values:
            ema_coco_eval = values['ema_test_coco_eval_bbox']
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                log_dict["Metrics/EMA/AP50_90"] = ema_ap50_90
            if ema_ap50 is not None:
                log_dict["Metrics/EMA/AP50"] = ema_ap50
            if ema_ar50_90 is not None:
                log_dict["Metrics/EMA/AR50_90"] = ema_ar50_90

        wandb.log(log_dict)

    def close(self):
        if not wandb or not self.run:
            return

        self.run.finish()
