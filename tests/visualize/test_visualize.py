# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for RF-DETR training metric visualization helpers.

Tests for the RF-DETR ground-truth/prediction visualization helper.

Tests for private RF-DETR keypoint visualization helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.utilities.imports import _IS_MATPLOTLIB_INSTALLED, _IS_PANDAS_INSTALLED, _IS_SEABORN_INSTALLED
from rfdetr.utilities.keypoints import precision_cholesky_to_pixel_covariance
from rfdetr.visualize.data import save_gt_predictions_visualization
from rfdetr.visualize.keypoints import _key_points_for_display, _keypoint_prediction_records
from rfdetr.visualize.training import (
    _build_metric_groups,
    _plot_map_columns,
    _plot_metric_groups,
    _read_metrics_csv,
    plot_loss_metrics,
    plot_map_metrics,
    plot_metrics,
)

_requires_pandas = pytest.mark.skipif(
    not _IS_PANDAS_INSTALLED, reason="pandas not installed; skip metrics DataFrame tests"
)


_requires_plotting = pytest.mark.skipif(
    not (_IS_MATPLOTLIB_INSTALLED and _IS_PANDAS_INSTALLED),
    reason="matplotlib/pandas not installed; skip metric plotting tests",
)


_requires_seaborn_plotting = pytest.mark.skipif(
    not (_IS_MATPLOTLIB_INSTALLED and _IS_PANDAS_INSTALLED and _IS_SEABORN_INSTALLED),
    reason="matplotlib/pandas/seaborn not installed; skip seaborn plotting tests",
)


class _FakeSeries:
    """Minimal series object for metric grouping tests."""

    def __init__(self, values: list[float | None]) -> None:
        """Store values for ``notna().any()`` checks.

        Examples:
            >>> _FakeSeries([1.0, None])._values
            [1.0, None]
        """
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
        """Store column data for ``_build_metric_groups``.

        Examples:
            >>> frame = _FakeDataFrame({'metric': [1.0, None]})
            >>> frame.columns
            ['metric']
        """
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


@_requires_seaborn_plotting
def test_plot_metrics_writes_keypoint_metrics_figure(tmp_path: Path) -> None:
    """plot_metrics should write a figure for CSVLogger files containing keypoint metrics."""
    import pandas as pd
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


@_requires_seaborn_plotting
def test_split_loss_and_map_plots_return_separate_figures(tmp_path: Path) -> None:
    """Loss and mAP plot helpers should build separate notebook-displayable figures."""
    import pandas as pd
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


@_requires_pandas
def test_metrics_reader_drops_trailing_post_fit_validation_epoch(tmp_path: Path) -> None:
    """Post-fit ``trainer.validate()`` rows should not appear as training-curve epochs."""
    import pandas as pd

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


@_requires_plotting
def test_map_plot_uses_line_style_for_train_and_val_splits(tmp_path: Path) -> None:
    """MAP plot should use one axes with dotted train lines and solid val lines."""
    import pandas as pd
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


@_requires_plotting
def test_map_renderer_uses_line_style_for_train_and_val_splits() -> None:
    """MAP renderer should pair train/val lines by color and distinguish split by style."""
    import pandas as pd
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


@_requires_plotting
def test_map_renderer_preserves_negative_values() -> None:
    """MAP renderer should plot raw metric values from the CSV without sentinel masking."""
    import pandas as pd
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


@_requires_plotting
def test_loss_renderer_preserves_negative_component_losses() -> None:
    """Loss renderer should plot negative NLL values rather than treating them as COCO sentinels."""
    import pandas as pd
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


@_requires_seaborn_plotting
def test_plot_metrics_warns_when_log_loss_has_non_positive_values(tmp_path: Path) -> None:
    """Loss log scale should fall back to linear scale when component losses are non-positive."""
    import pandas as pd
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

    @_requires_plotting
    def test_plot_metrics_succeeds_without_seaborn(self, tmp_path: Path, mocker) -> None:
        """plot_metrics returns a Figure when _IS_SEABORN_AVAILABLE is False (matplotlib-only fallback).

        Scenario: seaborn flag patched to False; plot_metrics called with a minimal DataFrame
        containing epoch and one metric column.  Expected outcome: call succeeds, returns a
        matplotlib Figure, raises no ImportError.
        """
        import pandas as pd
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        metrics_csv = tmp_path / "metrics.csv"
        pd.DataFrame(
            {
                "epoch": [0, 1],
                "val/mAP_50": [0.1, 0.2],
            }
        ).to_csv(metrics_csv, index=False)

        mocker.patch("rfdetr.visualize.training._IS_SEABORN_AVAILABLE", False)

        figure = plot_metrics(str(metrics_csv))

        assert isinstance(figure, Figure), "plot_metrics must return a matplotlib Figure when seaborn is absent"
        plt.close(figure)


class TestSeabornErrorBands:
    """Error band rendering when seaborn is available."""

    @_requires_seaborn_plotting
    def test_multi_step_epoch_produces_error_band_on_train_metrics(self, tmp_path: Path) -> None:
        """Train metrics logged at multiple steps per epoch produce a shaded ±1-std band."""
        import pandas as pd
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


@_requires_plotting
def test_plot_metrics_adds_legends_to_all_populated_subplots(tmp_path: Path) -> None:
    """Ensure that every populated metric has a legend."""
    import pandas as pd
    from matplotlib import pyplot as plt

    metrics_csv = tmp_path / "metrics.csv"

    pd.DataFrame(
        {
            "epoch": [0, 1],
            "train/loss": [2.0, 1.5],
            "val/mAP_50": [0.10, 0.20],
        }
    ).to_csv(metrics_csv, index=False)

    figure = plot_metrics(str(metrics_csv))

    visible_axes = [ax for ax in figure.axes if ax.get_visible()]

    axes_by_title = {ax.get_title(): ax for ax in visible_axes}

    assert set(axes_by_title) == {"Loss", "Detection AP@0.50"}

    loss_ax = axes_by_title["Loss"]
    detection_ax = axes_by_title["Detection AP@0.50"]

    assert loss_ax.get_legend() is not None, "Loss subplot should have legend"
    assert detection_ax.get_legend() is not None, "Detection AP@0.50 subplot should have legend"

    plt.close(figure)


class TestSaveGtPredictionsVisualization:
    """Tests for `save_gt_predictions_visualization`."""

    def test_saves_png_with_gt_and_pred_boxes(self, tmp_path: Path) -> None:
        """A scenario with both ground-truth and prediction boxes saves a PNG named after it.

        This is the function's primary contract: given at least one box on each side, it renders and persists
        `<scenario_name>.png` under `save_dir` without raising.
        """
        save_gt_predictions_visualization(
            scenario_name="scenario_a",
            image_width=8,
            image_height=8,
            gt_boxes=[[1.0, 1.0, 2.0, 2.0]],
            gt_class_ids=[1],
            pred_boxes=[[2.0, 2.0, 2.0, 2.0]],
            pred_class_ids=[1],
            pred_confidences=[0.9],
            pred_ious=[0.5],
            save_dir=tmp_path,
        )

        assert (tmp_path / "scenario_a.png").is_file()

    def test_creates_missing_save_directory(self, tmp_path: Path) -> None:
        """A `save_dir` that does not exist yet is created rather than raising.

        Mirrors a fresh training run's first visualization call, where the target directory has not been created by any
        earlier step.
        """
        save_dir = tmp_path / "viz"

        save_gt_predictions_visualization(
            scenario_name="scenario_b",
            image_width=4,
            image_height=4,
            gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
            gt_class_ids=[1],
            pred_boxes=[[0.0, 0.0, 1.0, 1.0]],
            pred_class_ids=[1],
            pred_confidences=[0.5],
            pred_ious=[None],
            save_dir=save_dir,
        )

        assert save_dir.is_dir()
        assert (save_dir / "scenario_b.png").is_file()

    def test_output_image_includes_top_padding(self, tmp_path: Path) -> None:
        """Saved image height equals the requested height plus the fixed 60px label strip.

        The top padding reserves room for labels above the boxes; a regression here would silently crop or misplace
        every annotation.
        """
        save_gt_predictions_visualization(
            scenario_name="scenario_c",
            image_width=10,
            image_height=20,
            gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
            gt_class_ids=[1],
            pred_boxes=[[0.0, 0.0, 1.0, 1.0]],
            pred_class_ids=[1],
            pred_confidences=[0.5],
            pred_ious=[None],
            save_dir=tmp_path,
        )

        with Image.open(tmp_path / "scenario_c.png") as image:
            assert image.size == (10, 80)

    @pytest.mark.parametrize(
        ("pred_iou", "expected_label"),
        [
            pytest.param(0.512, "c2\nconf=0.876\niou=0.512", id="iou-known"),
            pytest.param(None, "c2\nconf=0.876", id="iou-unknown"),
        ],
    )
    def test_pred_label_reflects_iou_availability(
        self, mocker: MockerFixture, tmp_path: Path, pred_iou: float | None, expected_label: str
    ) -> None:
        """Prediction labels append the IoU segment only when an IoU value is known.

        An unmatched false-positive prediction carries `iou=None`; the label loop must drop the IoU segment in that case
        instead of formatting `None` into the string.
        """
        mocker.patch("rfdetr.visualize.data.BoxAnnotator")
        mock_label_annotator = mocker.patch("rfdetr.visualize.data.LabelAnnotator")

        save_gt_predictions_visualization(
            scenario_name="scenario_d",
            image_width=6,
            image_height=6,
            gt_boxes=[[0.0, 0.0, 1.0, 1.0]],
            gt_class_ids=[1],
            pred_boxes=[[0.0, 0.0, 2.0, 2.0]],
            pred_class_ids=[2],
            pred_confidences=[0.876],
            pred_ious=[pred_iou],
            save_dir=tmp_path,
        )

        pred_labels = mock_label_annotator.return_value.annotate.call_args_list[-1].kwargs["labels"]
        assert pred_labels == [expected_label]

    def test_gt_and_pred_boxes_offset_by_top_padding(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Both GT and prediction boxes are shifted down by the 60px top padding before annotation.

        Verifies the xywh->xyxy conversion sees the offset y-coordinate, and that each side's class ids (and the
        prediction's confidence) reach the annotator unchanged.
        """
        mock_box_annotator = mocker.patch("rfdetr.visualize.data.BoxAnnotator")
        mocker.patch("rfdetr.visualize.data.LabelAnnotator")

        save_gt_predictions_visualization(
            scenario_name="scenario_e",
            image_width=8,
            image_height=6,
            gt_boxes=[[0.0, 0.0, 4.0, 4.0]],
            gt_class_ids=[1],
            pred_boxes=[[1.0, 1.0, 2.0, 2.0]],
            pred_class_ids=[2],
            pred_confidences=[0.75],
            pred_ious=[None],
            save_dir=tmp_path,
        )

        gt_call, pred_call = mock_box_annotator.return_value.annotate.call_args_list
        np.testing.assert_array_equal(gt_call.kwargs["detections"].xyxy, np.array([[0.0, 60.0, 4.0, 64.0]]))
        np.testing.assert_array_equal(gt_call.kwargs["detections"].class_id, np.array([1]))
        np.testing.assert_array_equal(pred_call.kwargs["detections"].xyxy, np.array([[1.0, 61.0, 3.0, 63.0]]))
        np.testing.assert_array_equal(pred_call.kwargs["detections"].class_id, np.array([2]))
        np.testing.assert_array_equal(pred_call.kwargs["detections"].confidence, np.array([0.75]))


class TestSaveGtPredictionsVisualizationEmptyBoxes:
    """An empty ``gt_boxes`` or ``pred_boxes`` list must be a no-op for that side, not a crash."""

    @pytest.mark.parametrize(
        ("num_gt", "num_pred"),
        [
            (0, 1),  # no ground truth: former IndexError, 1-D np.array([]) indexed as 2-D by xywh_to_xyxy
            (1, 0),  # no predictions: same IndexError on the prediction side
            (0, 0),  # neither side: blank visualization
        ],
    )
    def test_saves_png_when_a_side_is_empty(self, tmp_path: Path, num_gt: int, num_pred: int) -> None:
        """A scenario with an empty GT and/or prediction side still saves its visualization.

        Regression test for an ``IndexError``: an empty box list produced a 1-D ``np.array([])`` that ``xywh_to_xyxy``
        then indexed as 2-D.
        """
        scenario_name = f"gt{num_gt}_pred{num_pred}"

        save_gt_predictions_visualization(
            scenario_name=scenario_name,
            image_width=32,
            image_height=32,
            gt_boxes=[[4.0, 4.0, 8.0, 8.0]] * num_gt,
            gt_class_ids=[1] * num_gt,
            pred_boxes=[[4.0, 4.0, 8.0, 8.0]] * num_pred,
            pred_class_ids=[1] * num_pred,
            pred_confidences=[0.9] * num_pred,
            pred_ious=[None] * num_pred,
            save_dir=tmp_path,
        )

        assert (tmp_path / f"{scenario_name}.png").exists()


class TestPrecisionCholeskyToPixelCovariance:
    """Tests for `precision_cholesky_to_pixel_covariance`."""

    def test_identity_precision_scales_to_pixel_variance(self) -> None:
        """Identity normalized precision should scale to width/height pixel variance."""
        precision_cholesky = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        source_shape = np.array([[10.0, 20.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=precision_cholesky, source_shape=source_shape
        )

        expected = np.array([[[[400.0, 0.0], [0.0, 100.0]]]], dtype=np.float32)
        np.testing.assert_allclose(covariance, expected, rtol=1e-4, atol=1e-6)

    def test_does_not_clamp_log_cholesky(self) -> None:
        """Covariance display should use raw RF-DETR precision parameters."""
        precision_cholesky = np.array([[[25.0, 0.0, 0.0]]], dtype=np.float32)
        source_shape = np.array([[1.0, 1.0]], dtype=np.float32)

        covariance = precision_cholesky_to_pixel_covariance(
            precision_cholesky=precision_cholesky, source_shape=source_shape
        )

        np.testing.assert_allclose(covariance[0, 0, 0, 0], np.exp(-50.0), rtol=1e-4, atol=1e-28)

    @pytest.mark.parametrize(
        ("precision_shape", "source_shape", "message"),
        [
            ((1, 2, 4), (1, 2), "precision_cholesky must have shape"),
            ((2, 1, 3), (1, 2), "source_shape must have shape"),
        ],
    )
    def test_rejects_bad_shape(self, precision_shape: tuple, source_shape: tuple, message: str) -> None:
        """Invalid precision and source shapes should fail before annotation."""
        with pytest.raises(ValueError) as exc_info:
            precision_cholesky_to_pixel_covariance(
                precision_cholesky=np.zeros(precision_shape, dtype=np.float32),
                source_shape=np.zeros(source_shape, dtype=np.float32),
            )

        assert message in str(exc_info.value)


class TestKeyPointsForDisplay:
    """Tests for `_key_points_for_display`."""

    def test_builds_keypoints_with_covariance_and_masks_low_confidence(self) -> None:
        """RF-DETR keypoints become annotator-ready keypoints with covariance and a low-confidence visibility mask."""
        predictions = sv.KeyPoints(
            xy=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            keypoint_confidence=np.array([[0.9, 0.1]], dtype=np.float32),
            detection_confidence=np.array([0.95], dtype=np.float32),
            class_id=np.array([3], dtype=int),
            data={
                "keypoint_precision_cholesky": np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
                "source_shape": np.array([[10, 20]], dtype=np.int64),
                "xyxy": np.array([[0, 0, 10, 10]], dtype=np.float32),
            },
        )

        key_points = _key_points_for_display(predictions, keypoint_threshold=0.2)

        np.testing.assert_allclose(
            key_points.xy, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            key_points.keypoint_confidence, np.array([[0.9, 0.1]], dtype=np.float32), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_array_equal(key_points.visible, np.array([[True, False]]))
        np.testing.assert_array_equal(key_points.class_id, np.array([3]))
        assert "covariance" in key_points.data
        assert key_points.data["covariance"].shape == (1, 2, 2, 2)

    def test_accepts_keypoints_directly(self) -> None:
        """RF-DETR KeyPoints should be annotator-ready without converting through Detections."""
        predictions = sv.KeyPoints(
            xy=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            keypoint_confidence=np.array([[0.9, 0.1]], dtype=np.float32),
            detection_confidence=np.array([0.95], dtype=np.float32),
            data={
                "keypoint_precision_cholesky": np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
                "source_shape": np.array([[10, 20]], dtype=np.int64),
                "xyxy": np.array([[0, 0, 10, 10]], dtype=np.float32),
            },
        )

        key_points = _key_points_for_display(predictions, keypoint_threshold=0.2)

        np.testing.assert_allclose(
            key_points.xy, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), rtol=1e-4, atol=1e-6
        )
        np.testing.assert_array_equal(key_points.visible, np.array([[True, False]]))
        np.testing.assert_array_equal(key_points.data["xyxy"], predictions.data["xyxy"])
        np.testing.assert_array_equal(key_points.detection_confidence, predictions.detection_confidence)
        assert "covariance" in key_points.data

    def test_preserves_existing_covariance(self) -> None:
        """Display preparation should not overwrite covariance emitted by prediction."""
        covariance = np.array([[[[1.0, 0.0], [0.0, 2.0]]]], dtype=np.float32)
        predictions = sv.KeyPoints(
            xy=np.array([[[1.0, 2.0]]], dtype=np.float32),
            keypoint_confidence=np.array([[0.9]], dtype=np.float32),
            data={
                "covariance": covariance,
                "keypoint_precision_cholesky": np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32),
                "source_shape": np.array([[10, 20]], dtype=np.int64),
            },
        )

        key_points = _key_points_for_display(predictions)

        np.testing.assert_array_equal(key_points.data["covariance"], covariance)

    def test_rejects_keypoints_without_confidence_channel(self) -> None:
        """RF-DETR display helper expects per-keypoint confidence."""
        key_points = sv.KeyPoints(xy=np.array([[[1.0, 2.0]]], dtype=np.float32))

        with pytest.raises(ValueError, match="Expected RF-DETR keypoints"):
            _key_points_for_display(key_points)

    def test_returns_empty_keypoints_unchanged(self) -> None:
        """Empty KeyPoints (zero detections) should be returned unchanged without raising."""
        empty_predictions = sv.KeyPoints.empty()

        result = _key_points_for_display(empty_predictions)

        assert len(result) == 0, f"Expected empty KeyPoints, got len={len(result)}"


def test_keypoint_prediction_records_flattens_visible_keypoints() -> None:
    """Prediction records should expose detection and keypoint confidence for visible non-zero points."""
    key_points = sv.KeyPoints(
        xy=np.array([[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]], dtype=np.float32),
        keypoint_confidence=np.array([[0.9, 0.99, 0.1]], dtype=np.float32),
        detection_confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([2], dtype=int),
        visible=np.array([[True, True, False]]),
        data={"class_name": np.array(["dartboard"], dtype=object)},
    )

    records = _keypoint_prediction_records(key_points, image=Path("/tmp/sample.jpg"), keypoint_threshold=0.2)

    assert records == [
        {
            "image": "sample.jpg",
            "detection_index": 0,
            "class_id": 2,
            "class_name": "dartboard",
            "detection_confidence": pytest.approx(0.95),
            "keypoint_index": 0,
            "x": pytest.approx(1.0),
            "y": pytest.approx(2.0),
            "keypoint_confidence": pytest.approx(0.9),
        }
    ]
