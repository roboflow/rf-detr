# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for RF-DETR training metric visualization helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfdetr.visualize.training import _build_metric_groups, plot_metrics


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
            "val/loss": [2.2, 1.6],
            "val/mAP_50": [0.1, 0.2],
            "val/mAP_50_95": [0.05, 0.1],
            "val/mAR": [0.2, 0.3],
            "val/keypoint_map_50": [0.01, 0.02],
            "val/keypoint_map_50_95": [0.005, 0.01],
            "val/keypoint_mAR": [0.03, 0.04],
            "val/F1": [0.4, 0.5],
            "val/precision": [0.6, 0.7],
            "val/recall": [0.3, 0.4],
        }
    )

    groups = _build_metric_groups(metrics)

    assert groups["Loss"] == ["train/loss", "val/loss"]
    assert groups["Detection AP@0.50"] == ["val/mAP_50"]
    assert groups["Detection AP@0.50:0.95"] == ["val/mAP_50_95"]
    assert groups["Detection AR"] == ["val/mAR"]
    assert groups["Keypoint AP@0.50"] == ["val/keypoint_map_50"]
    assert groups["Keypoint AP@0.50:0.95"] == ["val/keypoint_map_50_95"]
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
            "val/loss": [None, 2.2, None, 1.6],
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
