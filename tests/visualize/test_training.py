# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF-DETR training metric visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfdetr.visualize.training import (
    _build_metric_groups,
    _plot_map_columns,
    plot_loss_metrics,
    plot_map_metrics,
    plot_metrics,
)


class _FakeSeries:
    """Minimal series object for metric grouping tests."""

    def __init__(self, values: list[float | None]) -> None:
        """Store values for ``notna().any()`` checks."""
        self._values = values

    def notna(self) -> "_FakeSeries":
        """Return values interpreted as non-null booleans."""
        return _FakeSeries([value is not None for value in self._values])

    def any(self) -> bool:
        """Return whether any value is truthy."""
        return any(bool(value) for value in self._values)


class _FakeDataFrame:
    """Minimal DataFrame object for metric grouping tests."""

    def __init__(self, data: dict[str, list[float | None]]) -> None:
        """Store column data for ``_build_metric_groups``."""
        self._data = data
        self.columns = list(data)

    def __getitem__(self, key: str) -> _FakeSeries:
        """Return fake series by column name."""
        return _FakeSeries(self._data[key])


class _FakePlotDataFrame:
    """Minimal DataFrame-like object for matplotlib plot calls."""

    def __init__(self, data: dict[str, list[float]]) -> None:
        """Store list-valued metric columns."""
        self._data = data

    def __getitem__(self, key: str) -> list[float]:
        """Return fake plot column by name."""
        return self._data[key]


def test_build_metric_groups_includes_detection_and_keypoint_metrics() -> None:
    """Metric grouping should include both detection and keypoint validation series."""
    metrics = _FakeDataFrame(
        {
            "epoch": [0, 1],
            "train/loss": [2.0, 1.5],
            "train/loss_cls": [0.8, 0.6],
            "train/loss_cls_0": [0.9, 0.7],
            "train/kp_nll": [-1.0, -2.0],
            "train/kp_nll_1": [-0.8, -1.8],
            "val/loss": [2.2, 1.6],
            "val/loss_keypoints_visible": [0.4, 0.3],
            "val/loss_keypoints_visible_0": [0.5, 0.4],
            "train/mAP_50": [0.08, 0.18],
            "train/mAP_50_95": [0.04, 0.09],
            "val/mAP_50": [0.1, 0.2],
            "val/mAP_50_95": [0.05, 0.1],
            "val/mAR": [0.2, 0.3],
            "train/keypoint_map_50": [0.008, 0.018],
            "train/keypoint_map_50_95": [0.004, 0.009],
            "val/keypoint_map_50": [0.01, 0.02],
            "val/keypoint_map_50_95": [0.005, 0.01],
            "val/keypoint_mAR": [0.03, 0.04],
            "val/F1": [0.4, 0.5],
            "val/precision": [0.6, 0.7],
            "val/recall": [0.3, 0.4],
        }
    )

    groups = _build_metric_groups(metrics)

    assert groups["Loss"] == ["train/loss", "train/loss_cls", "train/kp_nll", "val/loss", "val/loss_keypoints_visible"]
    assert groups["Detection AP@0.50"] == ["train/mAP_50", "val/mAP_50"]
    assert groups["Detection AP@0.50:0.95"] == ["train/mAP_50_95", "val/mAP_50_95"]
    assert groups["Detection AR"] == ["val/mAR"]
    assert groups["Keypoint AP@0.50"] == ["train/keypoint_map_50", "val/keypoint_map_50"]
    assert groups["Keypoint AP@0.50:0.95"] == ["train/keypoint_map_50_95", "val/keypoint_map_50_95"]
    assert groups["Keypoint AR"] == ["val/keypoint_mAR"]
    assert groups["F1 / Precision / Recall"] == ["val/F1", "val/precision", "val/recall"]


def test_plot_metrics_writes_keypoint_metrics_figure(tmp_path: Path) -> None:
    """plot_metrics should write a figure for CSVLogger files containing keypoint metrics."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    metrics_csv = tmp_path / "metrics.csv"
    output_path = tmp_path / "metrics.png"
    pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1],
            "step": [0, 1, 2, 3],
            "train/loss": [2.0, None, 1.5, None],
            "train/loss_cls": [0.7, None, 0.6, None],
            "train/kp_nll": [-1.0, None, -2.0, None],
            "val/loss": [None, 2.2, None, 1.6],
            "val/loss_keypoints_visible": [None, 0.4, None, 0.3],
            "train/keypoint_map_50": [None, 0.008, None, 0.018],
            "train/keypoint_map_50_95": [None, 0.004, None, 0.009],
            "val/keypoint_map_50": [None, 0.01, None, 0.02],
            "val/keypoint_map_50_95": [None, 0.005, None, 0.01],
            "val/keypoint_mAR": [None, 0.03, None, 0.04],
        }
    ).to_csv(metrics_csv, index=False)

    figure = plot_metrics(str(metrics_csv), str(output_path), loss_log_scale=True)

    assert isinstance(figure, Figure)
    assert plt.fignum_exists(figure.number)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(figure)


def test_split_loss_and_map_plots_return_separate_figures(tmp_path: Path) -> None:
    """Loss and mAP plot helpers should build separate notebook-displayable figures."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "epoch": [0, 1],
            "train/loss": [2.0, 1.5],
            "val/loss": [2.2, 1.6],
            "train/mAP_50_95": [0.04, 0.09],
            "val/mAP_50_95": [0.05, 0.1],
            "train/keypoint_map_50_95": [0.004, 0.009],
            "val/keypoint_map_50_95": [0.005, 0.01],
        }
    ).to_csv(metrics_csv, index=False)

    loss_figure = plot_loss_metrics(str(metrics_csv))
    map_figure = plot_map_metrics(str(metrics_csv))

    assert isinstance(loss_figure, Figure)
    assert isinstance(map_figure, Figure)
    assert loss_figure is not map_figure
    assert any("Loss" in ax.get_title() for ax in loss_figure.axes)
    assert len(map_figure.axes) == 1
    assert map_figure.axes[0].get_title() == "RF-DETR mAP Metrics"
    plt.close(loss_figure)
    plt.close(map_figure)


def test_map_plot_uses_line_style_for_train_and_val_splits(tmp_path: Path) -> None:
    """MAP plot should use one axes with dashed train lines and solid val lines."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    from matplotlib import pyplot as plt

    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "epoch": [0, 1],
            "train/mAP_50_95": [0.04, 0.09],
            "val/mAP_50_95": [0.05, 0.1],
            "train/keypoint_map_50_95": [0.004, 0.009],
            "val/keypoint_map_50_95": [0.005, 0.01],
        }
    ).to_csv(metrics_csv, index=False)

    figure = plot_map_metrics(str(metrics_csv))

    assert len(figure.axes) == 1
    linestyles = {line.get_label(): line.get_linestyle() for line in figure.axes[0].lines}
    assert linestyles["train/mAP_50_95"] == "--"
    assert linestyles["val/mAP_50_95"] == "-"
    assert linestyles["train/keypoint_map_50_95"] == "--"
    assert linestyles["val/keypoint_map_50_95"] == "-"
    plt.close(figure)


def test_map_renderer_uses_line_style_for_train_and_val_splits() -> None:
    """MAP renderer should use dashed train lines and solid val lines on one axes."""
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    df = _FakePlotDataFrame(
        {
            "epoch": [0, 1],
            "train/mAP_50_95": [0.04, 0.09],
            "val/mAP_50_95": [0.05, 0.1],
            "train/keypoint_map_50_95": [0.004, 0.009],
            "val/keypoint_map_50_95": [0.005, 0.01],
        }
    )

    figure = _plot_map_columns(
        df,
        ["train/mAP_50_95", "val/mAP_50_95", "train/keypoint_map_50_95", "val/keypoint_map_50_95"],
        output_path=None,
    )

    assert len(figure.axes) == 1
    linestyles = {line.get_label(): line.get_linestyle() for line in figure.axes[0].lines}
    assert linestyles["train/mAP_50_95"] == "--"
    assert linestyles["val/mAP_50_95"] == "-"
    assert linestyles["train/keypoint_map_50_95"] == "--"
    assert linestyles["val/keypoint_map_50_95"] == "-"
    plt.close(figure)


def test_map_renderer_hides_negative_coco_metric_sentinels() -> None:
    """MAP renderer should not plot COCO -1 sentinel values as real metric values."""
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    df = _FakePlotDataFrame(
        {
            "epoch": [0, 1, 2],
            "val/keypoint_map_50_95": [-1.0, 0.15, -0.5],
        }
    )

    figure = _plot_map_columns(df, ["val/keypoint_map_50_95"], output_path=None)

    y_values = figure.axes[0].lines[0].get_ydata()
    assert np.isnan(y_values[0])
    assert y_values[1] == pytest.approx(0.15)
    assert np.isnan(y_values[2])
    plt.close(figure)


def test_plot_metrics_warns_when_log_loss_has_non_positive_values(tmp_path: Path) -> None:
    """Loss log scale should fall back to linear scale when component losses are non-positive."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from matplotlib import pyplot as plt

    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "epoch": [0, 1],
            "train/loss": [1.0, 0.5],
            "train/kp_nll": [-1.0, -2.0],
        }
    ).to_csv(metrics_csv, index=False)

    with pytest.warns(UserWarning, match="non-positive"):
        figure = plot_metrics(str(metrics_csv), loss_log_scale=True)

    assert not (tmp_path / "metrics_plot.png").exists()
    plt.close(figure)
