# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the benchmark chart generator in ``scripts/render_benchmark_charts.py``."""

from pathlib import Path

import pytest

from scripts import render_benchmark_charts

_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS = _ROOT / "docs" / "learn" / "benchmarks.md"


class TestCommittedCharts:
    """The SVGs in docs/assets match what the generator renders from the benchmark tables."""

    @pytest.mark.parametrize(
        "name", ["benchmark-detection.svg", "benchmark-segmentation.svg", "benchmark-keypoints.svg"]
    )
    def test_chart_matches_tables(self, name: str) -> None:
        """The committed chart equals a fresh render of the Benchmarks page tables.

        A table edit without regenerating the charts fails here; rerun ``python scripts/render_benchmark_charts.py``.
        """
        rendered = render_benchmark_charts.render_charts(_BENCHMARKS.read_text(encoding="utf-8"))
        assert rendered[name] == (_ROOT / "docs" / "assets" / name).read_text(encoding="utf-8")


class TestReadTable:
    """Parsing one Markdown table under a heading."""

    def test_heading_without_table_raises(self) -> None:
        """A heading with no table under it raises instead of drawing an empty chart."""
        with pytest.raises(ValueError, match="No table rows"):
            render_benchmark_charts.read_table("## Detection\n\nNo table here.\n", "Detection")

    def test_missing_heading_raises(self) -> None:
        """A renamed section heading raises a ValueError that names it."""
        with pytest.raises(ValueError, match="'## Detection' not found"):
            render_benchmark_charts.read_table("## Segmentation\n\n| A |\n| - |\n| 1 |\n", "Detection")


class TestMain:
    """Command-line entry point."""

    def test_writes_every_chart(self, tmp_path: Path) -> None:
        """The default run writes the three chart files into the output directory."""
        render_benchmark_charts.main(["--output-dir", str(tmp_path)])
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "benchmark-detection.svg",
            "benchmark-keypoints.svg",
            "benchmark-segmentation.svg",
        ]
