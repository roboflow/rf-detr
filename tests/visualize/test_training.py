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
    _plot_metric_groups,
    _read_metrics_csv,
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
            "val/mAP_75": [0.07, 0.15],
            "val/mAR": [0.2, 0.3],
            "train/keypoint_map_50": [0.008, 0.018],
            "train/keypoint_map_50_95": [0.004, 0.009],
            "val/keypoint_map_50": [0.01, 0.02],
            "val/keypoint_map_50_95": [0.005, 0.01],
            "val/keypoint_map_75": [0.006, 0.012],
            "val/keypoint_mAR": [0.03, 0.04],
            "val/AP/small": [0.05, 0.1],
            "val/F1": [0.4, 0.5],
            "val/precision": [0.6, 0.7],
            "val/recall": [0.3, 0.4],
        }
    )

    groups = _build_metric_groups(metrics)

    assert groups["Loss"] == ["train/loss", "train/loss_cls", "train/kp_nll", "val/loss", "val/loss_keypoints_visible"]
    assert groups["Detection AP@0.50"] == ["train/mAP_50", "val/mAP_50"]
    assert groups["Detection AP@0.50:0.95"] == ["train/mAP_50_95", "val/mAP_50_95", "val/AP/small"]
    assert groups["Detection AP@0.75"] == ["val/mAP_75"]
    assert groups["Detection AR"] == ["val/mAR"]
    assert groups["Keypoint AP@0.50"] == ["train/keypoint_map_50", "val/keypoint_map_50"]
    assert groups["Keypoint AP@0.50:0.95"] == ["train/keypoint_map_50_95", "val/keypoint_map_50_95"]
    assert groups["Keypoint AP@0.75"] == ["val/keypoint_map_75"]
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
    loss_legend = loss_figure.axes[0].get_legend()
    assert loss_legend is not None
    assert getattr(loss_legend, "_ncols") == 2
    loss_lines = {line.get_label(): line for line in loss_figure.axes[0].lines}
    assert loss_lines["train/loss"].get_linestyle() == ":"
    assert loss_lines["val/loss"].get_linestyle() == "-"
    assert loss_lines["train/loss"].get_color() == loss_lines["val/loss"].get_color()
    assert {line.get_marker() for line in loss_lines.values()} == {"None"}
    assert len(map_figure.axes) == 1
    assert map_figure.axes[0].get_title() == "RF-DETR mAP Metrics"
    map_lines = {line.get_label(): line for line in map_figure.axes[0].lines}
    assert {line.get_marker() for line in map_lines.values()} == {"None"}
    plt.close(loss_figure)
    plt.close(map_figure)


def test_metrics_reader_drops_trailing_post_fit_validation_epoch(tmp_path: Path) -> None:
    """Post-fit ``trainer.validate()`` rows should not appear as training-curve epochs."""
    pd = pytest.importorskip("pandas")

    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "epoch": [0, 0, 1, 1, 2],
            "step": [0, 1, 2, 3, 4],
            "train/loss": [2.0, None, 1.5, None, None],
            "val/loss": [None, 2.2, None, 1.6, 1.6],
            "val/mAP_50_95": [None, 0.1, None, 0.2, 0.99],
        }
    ).to_csv(metrics_csv, index=False)

    _, epoch_df = _read_metrics_csv(str(metrics_csv))

    assert epoch_df["epoch"].tolist() == [0, 1]
    assert epoch_df["val/mAP_50_95"].tolist() == pytest.approx([0.1, 0.2])


def test_map_plot_uses_line_style_for_train_and_val_splits(tmp_path: Path) -> None:
    """MAP plot should use one axes with dotted train lines and solid val lines."""
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
    assert linestyles["train/mAP_50_95"] == ":"
    assert linestyles["val/mAP_50_95"] == "-"
    assert linestyles["train/keypoint_map_50_95"] == ":"
    assert linestyles["val/keypoint_map_50_95"] == "-"
    plt.close(figure)


def test_map_renderer_uses_line_style_for_train_and_val_splits() -> None:
    """MAP renderer should pair train/val lines by color and distinguish split by style."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    from matplotlib import pyplot as plt

    df = pd.DataFrame(
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
        df,
        ["train/mAP_50_95", "val/mAP_50_95", "train/keypoint_map_50_95", "val/keypoint_map_50_95"],
        output_path=None,
    )

    assert len(figure.axes) == 1
    lines = {line.get_label(): line for line in figure.axes[0].lines if not line.get_label().startswith("_")}
    assert lines["train/mAP_50_95"].get_linestyle() == ":"
    assert lines["val/mAP_50_95"].get_linestyle() == "-"
    assert lines["train/keypoint_map_50_95"].get_linestyle() == ":"
    assert lines["val/keypoint_map_50_95"].get_linestyle() == "-"
    assert {line.get_marker() for line in lines.values()} == {"None"}
    assert lines["train/mAP_50_95"].get_color() == lines["val/mAP_50_95"].get_color()
    assert lines["train/keypoint_map_50_95"].get_color() == lines["val/keypoint_map_50_95"].get_color()
    assert lines["train/mAP_50_95"].get_color() != lines["train/keypoint_map_50_95"].get_color()
    plt.close(figure)


def test_map_renderer_preserves_negative_values() -> None:
    """MAP renderer should plot raw metric values from the CSV without sentinel masking."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    from matplotlib import pyplot as plt

    df = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "val/keypoint_map_50_95": [-1.0, 0.15, -0.5],
        }
    )

    figure = _plot_map_columns(df, df, ["val/keypoint_map_50_95"], output_path=None)

    lines = {line.get_label(): line for line in figure.axes[0].lines if not line.get_label().startswith("_")}
    y_values = lines["val/keypoint_map_50_95"].get_ydata()
    assert y_values[0] == pytest.approx(-1.0)
    assert y_values[1] == pytest.approx(0.15)
    assert y_values[2] == pytest.approx(-0.5)
    plt.close(figure)


def test_loss_renderer_preserves_negative_component_losses() -> None:
    """Loss renderer should plot negative NLL values rather than treating them as COCO sentinels."""
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    from matplotlib import pyplot as plt

    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "train/kp_nll": [-1.0, -2.0],
        }
    )

    figure = _plot_metric_groups(
        df,
        df,
        {"Loss": ["train/kp_nll"]},
        title="RF-DETR Loss Metrics",
        output_path=None,
        loss_log_scale=False,
    )

    lines = {line.get_label(): line for line in figure.axes[0].lines if not line.get_label().startswith("_")}
    np.testing.assert_allclose(lines["train/kp_nll"].get_ydata(), [-1.0, -2.0])
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

    lines = {line.get_label(): line for line in figure.axes[0].lines}
    np.testing.assert_allclose(lines["train/kp_nll"].get_ydata(), [-1.0, -2.0])
    assert not (tmp_path / "metrics_plot.png").exists()
    plt.close(figure)


class TestPlotMetricsNoSeaborn:
    """Verify plot_metrics falls back gracefully when seaborn is unavailable."""

    def test_plot_metrics_succeeds_without_seaborn(self, tmp_path: Path) -> None:
        """plot_metrics returns a Figure when _IS_SEABORN_AVAILABLE is False (matplotlib-only fallback).

        Scenario: seaborn flag patched to False; plot_metrics called with a minimal DataFrame
        containing epoch and one metric column.  Expected outcome: call succeeds, returns a
        matplotlib Figure, raises no ImportError.
        """
        pytest.importorskip("matplotlib")
        pd = pytest.importorskip("pandas")
        from unittest.mock import patch

        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        metrics_csv = tmp_path / "metrics.csv"
        pd.DataFrame(
            {
                "epoch": [0, 1],
                "val/mAP_50": [0.1, 0.2],
            }
        ).to_csv(metrics_csv, index=False)

        with patch("rfdetr.visualize.training._IS_SEABORN_AVAILABLE", False):
            figure = plot_metrics(str(metrics_csv))

        assert isinstance(figure, Figure), "plot_metrics must return a matplotlib Figure when seaborn is absent"
        plt.close(figure)


class TestSeabornErrorBands:
    """Error band rendering when seaborn is available."""

    def test_multi_step_epoch_produces_error_band_on_train_metrics(self, tmp_path: Path) -> None:
        """Train metrics logged at multiple steps per epoch produce a shaded ±1-std band."""
        pytest.importorskip("matplotlib")
        pd = pytest.importorskip("pandas")
        pytest.importorskip("seaborn")
        from matplotlib import pyplot as plt
        from matplotlib.collections import PolyCollection

        metrics_csv = tmp_path / "metrics.csv"
        pd.DataFrame(
            {
                "epoch": [0, 0, 1, 1],
                "step": [0, 1, 2, 3],
                "train/loss": [2.0, 3.0, 1.0, 1.5],
                "val/loss": [None, 2.2, None, 1.6],
            }
        ).to_csv(metrics_csv, index=False)

        figure = plot_metrics(str(metrics_csv))
        loss_ax = figure.axes[0]

        poly_collections = [c for c in loss_ax.collections if isinstance(c, PolyCollection)]
        assert len(poly_collections) >= 1, "Expected error-band patch for multi-step train/loss"
        plt.close(figure)
