# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MetricsPlotCallback — generate a metrics plot from CSVLogger output."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
from pytorch_lightning import Callback

PLOT_FILE_NAME = "metrics_plot.png"

# CSVLogger metric keys written by COCOEvalCallback
_BBOX_KEYS = {
    "ap50": "val/mAP_50",
    "ap50_95": "val/mAP_50_95",
    "ar": "val/mAR",
    "ema_ap50": "val/ema_mAP_50",
    "ema_ap50_95": "val/ema_mAP_50_95",
    "ema_ar": "val/ema_mAR",
}
_SEGM_KEYS = {
    "ap50": "val/segm_mAP_50",
    "ap50_95": "val/segm_mAP_50_95",
    "ema_ap50": "val/ema_segm_mAP_50",
    "ema_ap50_95": "val/ema_segm_mAP_50_95",
}
_LOSS_KEYS = {
    "train": "train/loss",
    "val": "val/loss",
}


def _col(history: list[dict], key: str) -> np.ndarray:
    """Extract a column from CSV history, skipping empty/missing cells."""
    vals = []
    for row in history:
        v = row.get(key, "").strip()
        if v:
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return np.array(vals, dtype=np.float32)


def _epoch_col(history: list[dict], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs, values) for rows that have a non-empty metric value."""
    epochs, vals = [], []
    for row in history:
        v = row.get(metric_key, "").strip()
        e = row.get("epoch", "").strip()
        if v and e:
            try:
                vals.append(float(v))
                epochs.append(float(e))
            except ValueError:
                pass
    return np.array(epochs, dtype=np.float32), np.array(vals, dtype=np.float32)


class MetricsPlotCallback(Callback):
    """Read ``metrics.csv`` written by CSVLogger and save a metrics figure.

    Generates a 2×2 grid for detection-only models, or a 3×2 grid when
    segmentation metrics are present in the log.

    The plot is saved to ``{output_dir}/metrics_plot.png`` after training ends.
    """

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        """Save the metrics plot when training completes."""
        csv_path = self._find_csv(trainer)
        if csv_path is None:
            return
        history = self._read_csv(csv_path)
        if not history:
            return
        output_dir = Path(trainer.default_root_dir)
        self._save_plot(history, output_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_csv(self, trainer: Any) -> "Path | None":
        """Locate the CSVLogger metrics file inside the trainer's root dir."""
        csv_path = Path(trainer.default_root_dir) / "metrics.csv"
        if csv_path.exists():
            return csv_path
        return None

    def _read_csv(self, path: Path) -> list[dict]:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def _save_plot(self, history: list[dict], output_dir: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        e_loss, train_loss = _epoch_col(history, _LOSS_KEYS["train"])
        _, val_loss = _epoch_col(history, _LOSS_KEYS["val"])

        e_bbox, ap50 = _epoch_col(history, _BBOX_KEYS["ap50"])
        _, ap50_95 = _epoch_col(history, _BBOX_KEYS["ap50_95"])
        _, ar = _epoch_col(history, _BBOX_KEYS["ar"])
        _, ema_ap50 = _epoch_col(history, _BBOX_KEYS["ema_ap50"])
        _, ema_ap50_95 = _epoch_col(history, _BBOX_KEYS["ema_ap50_95"])
        _, ema_ar = _epoch_col(history, _BBOX_KEYS["ema_ar"])

        e_segm, segm_ap50 = _epoch_col(history, _SEGM_KEYS["ap50"])
        _, segm_ap50_95 = _epoch_col(history, _SEGM_KEYS["ap50_95"])
        _, ema_segm_ap50 = _epoch_col(history, _SEGM_KEYS["ema_ap50"])
        _, ema_segm_ap50_95 = _epoch_col(history, _SEGM_KEYS["ema_ap50_95"])

        has_segm = segm_ap50.size > 0 or segm_ap50_95.size > 0

        if has_segm:
            fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        def _plot(ax, epochs, base, ema, title, ylabel):
            if base.size > 0:
                ax.plot(epochs[: len(base)], base, marker="o", linestyle="-", label="Base Model")
            if ema.size > 0:
                ax.plot(epochs[: len(ema)], ema, marker="o", linestyle="--", label="EMA Model")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        # Row 0: Loss | BBox AP50
        ax = axes[0][0]
        if train_loss.size > 0:
            ax.plot(e_loss[: len(train_loss)], train_loss, marker="o", linestyle="-", label="Train Loss")
        if val_loss.size > 0:
            ax.plot(e_loss[: len(val_loss)], val_loss, marker="o", linestyle="--", label="Val Loss")
        ax.set_title("Training and Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

        _plot(axes[0][1], e_bbox, ap50, ema_ap50, "BBox Average Precision @0.50", "AP50")

        # Row 1: BBox AP50:95 | BBox AR
        _plot(axes[1][0], e_bbox, ap50_95, ema_ap50_95, "BBox Average Precision @0.50:0.95", "AP")
        _plot(axes[1][1], e_bbox, ar, ema_ar, "BBox Average Recall", "AR")

        # Row 2 (segmentation only): Segm AP50 | Segm AP50:95
        if has_segm:
            _plot(axes[2][0], e_segm, segm_ap50, ema_segm_ap50, "Segm Average Precision @0.50", "AP50")
            _plot(axes[2][1], e_segm, segm_ap50_95, ema_segm_ap50_95, "Segm Average Precision @0.50:0.95", "AP")

        plt.tight_layout()
        out = output_dir / PLOT_FILE_NAME
        plt.savefig(out)
        plt.close(fig)
