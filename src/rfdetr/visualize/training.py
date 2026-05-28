# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Post-training metrics plotting utilities.

Reads the ``metrics.csv`` written by PTL's ``CSVLogger`` (always present after a ``build_trainer``-based run) and saves
a seaborn figure grouped by metric type (Loss, AP@0.50, AP@0.50:0.95, AR).

Loss panel shows only the aggregate ``train/loss`` and ``val/loss`` scalars. AP/AR panels show all ``val/`` columns for
each group — both the base and EMA series when EMA is enabled, so both are visible in the legend.

Usage::

    from rfdetr.visualize.training import plot_metrics
    fig = plot_metrics("output/rfdetr_base/metrics.csv", "output/rfdetr_base/metrics_plot.png")
    plt.show(fig)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _build_metric_groups(df: Any) -> dict[str, list[str]]:
    """Build plot groups from numeric PTL CSVLogger metrics.

    Args:
        df: DataFrame-like object with metric columns.

    Returns:
        Non-empty metric groups keyed by subplot title.

    Raises:
        AttributeError: If ``df`` does not provide DataFrame-like columns.
    """

    def _cols(*names: str) -> list[str]:
        """Return columns from ``names`` that contain at least one non-NaN value."""
        return [name for name in names if name in df.columns and df[name].notna().any()]

    def _val_cols(*patterns: str) -> list[str]:
        """Return val/ columns whose name contains any of the given patterns."""
        return [c for c in df.columns if c.startswith("val/") and any(p in c for p in patterns) and df[c].notna().any()]

    loss_cols = _cols("train/loss", "val/loss", "test/loss")
    detection_map_50 = [c for c in _val_cols("mAP_50", "ema_mAP_50") if "mAP_50_95" not in c]
    detection_map_50_95 = _val_cols("mAP_50_95", "ema_mAP_50_95")
    detection_mar = [c for c in _val_cols("mAR", "ema_mAR") if "keypoint_" not in c]
    keypoint_map_50 = [c for c in _val_cols("keypoint_map_50", "ema_keypoint_map_50") if "map_50_95" not in c]
    keypoint_map_50_95 = _val_cols("keypoint_map_50_95", "ema_keypoint_map_50_95")
    keypoint_mar = _val_cols("keypoint_mAR", "ema_keypoint_mAR")
    f1_precision_recall = _val_cols("F1", "precision", "recall")

    metric_groups: dict[str, list[str]] = {
        "Loss": loss_cols,
        "Detection AP@0.50": detection_map_50,
        "Detection AP@0.50:0.95": detection_map_50_95,
        "Detection AR": detection_mar,
        "Keypoint AP@0.50": keypoint_map_50,
        "Keypoint AP@0.50:0.95": keypoint_map_50_95,
        "Keypoint AR": keypoint_mar,
        "F1 / Precision / Recall": f1_precision_recall,
    }
    return {name: columns for name, columns in metric_groups.items() if columns}


def plot_metrics(
    metrics_csv: str,
    output_path: Optional[str] = None,
    loss_log_scale: bool = False,
) -> Figure:
    """Read a PTL ``CSVLogger`` metrics file and build a seaborn training plot.

    The figure contains one subplot per metric group (loss, detection metrics,
    keypoint metrics, and F1/precision/recall), arranged in a 2-column grid.
    Only groups with at least one non-NaN column are shown.

    Args:
        metrics_csv: Path to the ``metrics.csv`` file produced by
            ``CSVLogger``.
        output_path: Destination for the PNG file.  Defaults to
            ``metrics_plot.png`` next to ``metrics_csv``.

    Returns:
        The matplotlib figure. The figure is left open so notebook cells can
        display it inline.

    Raises:
        ImportError: If ``matplotlib``, ``pandas``, or ``seaborn`` are not
            installed.
        FileNotFoundError: If ``metrics_csv`` does not exist.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_metrics(). Install it with: pip install matplotlib") from exc

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for plot_metrics(). Install it with: pip install pandas") from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("seaborn is required for plot_metrics(). Install it with: pip install seaborn") from exc

    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    if output_path is None:
        output_path = str(csv_path.parent / "metrics_plot.png")

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError("metrics.csv does not contain an 'epoch' column.")
    # CSVLogger writes one row per step; aggregate to one row per epoch.
    df = df.groupby("epoch").mean(numeric_only=True).reset_index()

    metric_groups = _build_metric_groups(df)
    if not metric_groups:
        raise ValueError("metrics.csv does not contain any supported non-empty metric columns.")

    n_groups = len(metric_groups)
    n_cols = 2
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    melted = df.melt(id_vars="epoch", var_name="metric", value_name="value")

    for idx, (title, metric_list) in enumerate(metric_groups.items()):
        ax = axes_flat[idx]
        group_data = melted[melted["metric"].isin(metric_list)]
        sns.lineplot(data=group_data, x="epoch", y="value", hue="metric", marker="o", ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        if title == "Loss" and loss_log_scale:
            ax.set_yscale("log")

    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("RF-DETR Training Metrics", fontsize=14)
    fig.tight_layout()
    return fig
