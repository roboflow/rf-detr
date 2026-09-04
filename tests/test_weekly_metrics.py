# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for weekly project metrics SVG generation."""

import io
import json
import runpy
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import update_weekly_metrics


class TestCompletedWeek:
    """Tests for selecting stable Monday-to-Sunday download periods."""

    def test_thursday_selects_previous_completed_week(self) -> None:
        """A midweek run must exclude every day from the in-progress week."""
        start, end = update_weekly_metrics.completed_week(date(2026, 9, 3))

        assert start == date(2026, 8, 24)
        assert end == date(2026, 8, 30)


class TestWeeklyDownloads:
    """Tests for aggregation of complete PyPI Stats periods."""

    def test_sums_exact_completed_week(self) -> None:
        """Aggregation must use all seven dates and exclude adjacent data."""
        daily_downloads = {
            date(2026, 8, 23): 100,
            date(2026, 8, 24): 1,
            date(2026, 8, 25): 2,
            date(2026, 8, 26): 3,
            date(2026, 8, 27): 4,
            date(2026, 8, 28): 5,
            date(2026, 8, 29): 6,
            date(2026, 8, 30): 7,
            date(2026, 8, 31): 200,
        }

        total = update_weekly_metrics.weekly_downloads(
            daily_downloads,
            date(2026, 8, 24),
            date(2026, 8, 30),
        )

        assert total == 28

    def test_rejects_incomplete_week(self) -> None:
        """Missing daily data must stop update instead of publishing a partial week."""
        daily_downloads = {
            date(2026, 8, 24): 1,
            date(2026, 8, 25): 2,
        }

        with pytest.raises(update_weekly_metrics.MetricsError, match="2026-08-26"):
            update_weekly_metrics.weekly_downloads(
                daily_downloads,
                date(2026, 8, 24),
                date(2026, 8, 30),
            )

    def test_rejects_non_calendar_week(self) -> None:
        """Aggregation must not silently accept partial or shifted periods."""
        daily_downloads = {date(2026, 8, 25): 1}

        with pytest.raises(update_weekly_metrics.MetricsError, match="Monday-to-Sunday"):
            update_weekly_metrics.weekly_downloads(
                daily_downloads,
                date(2026, 8, 25),
                date(2026, 8, 25),
            )


class TestRecordWeek:
    """Tests for checkpoint deltas and bounded history updates."""

    def test_first_observation_establishes_zero_delta_baseline(self) -> None:
        """First star count must become baseline because no earlier checkpoint exists."""
        state = update_weekly_metrics.MetricsState(
            repository="roboflow/rf-detr",
            package="rfdetr",
            history=(),
        )

        updated = update_weekly_metrics.record_week(
            state,
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_000,
            downloads=70_000,
            history_limit=52,
        )

        assert updated.history == (
            update_weekly_metrics.WeeklyMetric(
                week_start=date(2026, 8, 24),
                week_end=date(2026, 8, 30),
                stars_total=12_000,
                new_stars=0,
                downloads=70_000,
            ),
        )

    def test_next_observation_records_star_delta(self) -> None:
        """New period must calculate star growth from previous cumulative checkpoint."""
        previous = update_weekly_metrics.WeeklyMetric(
            week_start=date(2026, 8, 17),
            week_end=date(2026, 8, 23),
            stars_total=12_000,
            new_stars=75,
            downloads=70_000,
        )
        state = update_weekly_metrics.MetricsState("roboflow/rf-detr", "rfdetr", (previous,))

        updated = update_weekly_metrics.record_week(
            state,
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_125,
            downloads=80_000,
            history_limit=52,
        )

        assert updated.history[-1].new_stars == 125

    def test_same_period_replaces_observation_without_resetting_delta(self) -> None:
        """A rerun must update one period relative to its preceding checkpoint."""
        first = update_weekly_metrics.WeeklyMetric(
            week_start=date(2026, 8, 17),
            week_end=date(2026, 8, 23),
            stars_total=12_000,
            new_stars=75,
            downloads=70_000,
        )
        current = update_weekly_metrics.WeeklyMetric(
            week_start=date(2026, 8, 24),
            week_end=date(2026, 8, 30),
            stars_total=12_100,
            new_stars=100,
            downloads=79_000,
        )
        state = update_weekly_metrics.MetricsState("roboflow/rf-detr", "rfdetr", (first, current))

        updated = update_weekly_metrics.record_week(
            state,
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_125,
            downloads=80_000,
            history_limit=52,
        )

        assert len(updated.history) == 2
        assert updated.history[-1].new_stars == 125
        assert updated.history[-1].downloads == 80_000

    def test_rejects_period_older_than_checkpoint(self) -> None:
        """Clock or input regressions must not corrupt ordered checkpoint history."""
        current = update_weekly_metrics.WeeklyMetric(
            week_start=date(2026, 8, 24),
            week_end=date(2026, 8, 30),
            stars_total=12_100,
            new_stars=100,
            downloads=79_000,
        )
        state = update_weekly_metrics.MetricsState("roboflow/rf-detr", "rfdetr", (current,))

        with pytest.raises(update_weekly_metrics.MetricsError, match="older than latest checkpoint"):
            update_weekly_metrics.record_week(
                state,
                start=date(2026, 8, 17),
                end=date(2026, 8, 23),
                stars_total=12_125,
                downloads=80_000,
                history_limit=52,
            )

    def test_retains_only_configured_latest_history(self) -> None:
        """Bounded history must keep latest checkpoints and their recorded deltas."""
        history = tuple(
            update_weekly_metrics.WeeklyMetric(
                week_start=date(2026, 8, 3) + timedelta(weeks=index),
                week_end=date(2026, 8, 9) + timedelta(weeks=index),
                stars_total=12_000 + index * 100,
                new_stars=0 if index == 0 else 100,
                downloads=70_000 + index * 1_000,
            )
            for index in range(3)
        )
        state = update_weekly_metrics.MetricsState("roboflow/rf-detr", "rfdetr", history)

        updated = update_weekly_metrics.record_week(
            state,
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_300,
            downloads=73_000,
            history_limit=2,
        )

        assert [metric.week_start for metric in updated.history] == [date(2026, 8, 17), date(2026, 8, 24)]
        assert [metric.new_stars for metric in updated.history] == [100, 100]


class TestSvgState:
    """Tests for self-contained checkpoint serialization."""

    def test_rendered_metadata_round_trips_complete_state(self) -> None:
        """SVG metadata must remain sole machine-readable checkpoint source."""
        state = update_weekly_metrics.MetricsState(
            repository="roboflow/rf-detr",
            package="rfdetr",
            history=(
                update_weekly_metrics.WeeklyMetric(
                    week_start=date(2026, 8, 24),
                    week_end=date(2026, 8, 30),
                    stars_total=12_125,
                    new_stars=125,
                    downloads=80_000,
                ),
            ),
        )

        svg = update_weekly_metrics.render_svg(state, window_weeks=12)
        parsed = update_weekly_metrics.parse_state(svg, "roboflow/rf-detr", "rfdetr")

        assert parsed == state

    def test_parser_rejects_malformed_checkpoint_json(self) -> None:
        """Corrupt embedded state must fail closed before replacing SVG."""
        svg = '<svg><metadata id="weekly-metrics-state">{broken</metadata></svg>'

        with pytest.raises(update_weekly_metrics.MetricsError, match="Invalid weekly metrics metadata"):
            update_weekly_metrics.parse_state(svg, "roboflow/rf-detr", "rfdetr")

    def test_parser_rejects_negative_downloads(self) -> None:
        """Invalid upstream counts in embedded state must fail validation."""
        svg = """<svg><metadata id="weekly-metrics-state">{
            "schema_version": 1,
            "repository": "roboflow/rf-detr",
            "package": "rfdetr",
            "history": [{
                "week_start": "2026-08-24",
                "week_end": "2026-08-30",
                "stars_total": 12125,
                "new_stars": 125,
                "downloads": -1
            }]
        }</metadata></svg>"""

        with pytest.raises(update_weekly_metrics.MetricsError, match="downloads must be a non-negative integer"):
            update_weekly_metrics.parse_state(svg, "roboflow/rf-detr", "rfdetr")

    def test_metadata_escapes_xml_characters_and_round_trips(self) -> None:
        """Source identifiers must not break XML or change after extraction."""
        state = update_weekly_metrics.MetricsState(
            repository="owner/repo&mirror",
            package="package<nightly",
            history=(),
        )

        svg = update_weekly_metrics.render_svg(state, window_weeks=12)

        assert "repo&amp;mirror" in svg
        assert "package&lt;nightly" in svg
        assert update_weekly_metrics.parse_state(svg, "owner/repo&mirror", "package<nightly") == state


class TestSvgRendering:
    """Tests for deterministic dual-axis chart output."""

    def test_chart_uses_display_window_and_latest_summary(self) -> None:
        """Chart marks must show only configured recent periods plus latest totals."""
        state = update_weekly_metrics.MetricsState(
            repository="roboflow/rf-detr",
            package="rfdetr",
            history=(
                update_weekly_metrics.WeeklyMetric(
                    week_start=date(2026, 8, 17),
                    week_end=date(2026, 8, 23),
                    stars_total=12_000,
                    new_stars=75,
                    downloads=70_000,
                ),
                update_weekly_metrics.WeeklyMetric(
                    week_start=date(2026, 8, 24),
                    week_end=date(2026, 8, 30),
                    stars_total=12_125,
                    new_stars=125,
                    downloads=80_000,
                ),
            ),
        )

        svg = update_weekly_metrics.render_svg(state, window_weeks=1)

        assert 'id="weekly-stars-bars"' in svg
        assert 'id="weekly-downloads-line"' in svg
        assert "+125 new stars" in svg
        assert "80,000 weekly downloads" in svg
        assert "Aug 30" in svg
        assert "Aug 23" not in svg


class TestMetricsApis:
    """Tests for upstream API boundary parsing."""

    def test_fetch_star_count_uses_repository_api_and_token_header(self) -> None:
        """GitHub request must authenticate through header and return validated count."""
        response = io.BytesIO(json.dumps({"stargazers_count": 12_125}).encode())

        with patch("scripts.update_weekly_metrics.urlopen", return_value=response) as urlopen:
            stars = update_weekly_metrics.fetch_star_count("roboflow/rf-detr", "secret-token")

        request = urlopen.call_args.args[0]
        assert request.full_url == "https://api.github.com/repos/roboflow/rf-detr"
        assert request.get_header("Authorization") == "Bearer secret-token"
        assert stars == 12_125

    def test_fetch_daily_downloads_reads_without_mirrors_series(self) -> None:
        """PyPI Stats response must become validated UTC daily counts."""
        payload = {
            "package": "rfdetr",
            "type": "overall_downloads",
            "data": [
                {"category": "without_mirrors", "date": "2026-08-24", "downloads": 10_000},
                {"category": "without_mirrors", "date": "2026-08-25", "downloads": 11_000},
            ],
        }
        response = io.BytesIO(json.dumps(payload).encode())

        with patch("scripts.update_weekly_metrics.urlopen", return_value=response) as urlopen:
            downloads = update_weekly_metrics.fetch_daily_downloads("rfdetr")

        request = urlopen.call_args.args[0]
        assert request.full_url == "https://pypistats.org/api/packages/rfdetr/overall?mirrors=false"
        assert downloads == {date(2026, 8, 24): 10_000, date(2026, 8, 25): 11_000}

    def test_fetch_daily_downloads_rejects_duplicate_dates(self) -> None:
        """Duplicate upstream days must fail instead of inflating weekly totals."""
        row = {"category": "without_mirrors", "date": "2026-08-24", "downloads": 10_000}
        response = io.BytesIO(
            json.dumps({"package": "rfdetr", "type": "overall_downloads", "data": [row, row]}).encode()
        )

        with (
            patch("scripts.update_weekly_metrics.urlopen", return_value=response),
            pytest.raises(update_weekly_metrics.MetricsError, match="duplicate data"),
        ):
            update_weekly_metrics.fetch_daily_downloads("rfdetr")


class TestMetricsUpdate:
    """Tests for complete local SVG update behavior."""

    def test_first_update_writes_fetch_results_as_embedded_baseline(self, tmp_path: Path) -> None:
        """Missing output must produce one self-contained baseline observation."""
        output = tmp_path / "weekly-metrics.svg"
        daily_downloads = {
            date(2026, 8, 24): 10_000,
            date(2026, 8, 25): 11_000,
            date(2026, 8, 26): 12_000,
            date(2026, 8, 27): 13_000,
            date(2026, 8, 28): 14_000,
            date(2026, 8, 29): 15_000,
            date(2026, 8, 30): 16_000,
        }

        with (
            patch("scripts.update_weekly_metrics.fetch_star_count", return_value=12_125),
            patch("scripts.update_weekly_metrics.fetch_daily_downloads", return_value=daily_downloads),
        ):
            update_weekly_metrics.update_metrics(
                output=output,
                repository="roboflow/rf-detr",
                package="rfdetr",
                window_weeks=12,
                history_limit=104,
                today=date(2026, 9, 3),
                github_token=None,
            )

        state = update_weekly_metrics.parse_state(output.read_text(), "roboflow/rf-detr", "rfdetr")
        assert state.history[0].stars_total == 12_125
        assert state.history[0].new_stars == 0
        assert state.history[0].downloads == 91_000
        assert output.stat().st_mode & stat.S_IROTH

    def test_cli_writes_custom_output(self, tmp_path: Path) -> None:
        """Command entry point must run complete generator with explicit output path."""
        output = tmp_path / "metrics.svg"
        start, _ = update_weekly_metrics.completed_week(datetime.now(timezone.utc).date())
        daily_downloads = {start + timedelta(days=offset): 100 for offset in range(7)}

        with (
            patch("scripts.update_weekly_metrics.fetch_star_count", return_value=12_125),
            patch("scripts.update_weekly_metrics.fetch_daily_downloads", return_value=daily_downloads),
        ):
            exit_code = update_weekly_metrics.main(["--output", str(output), "--weeks", "4"])

        assert exit_code == 0
        assert output.read_text().startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_same_week_rerun_is_byte_identical(self, tmp_path: Path) -> None:
        """Unchanged API observations must not create SVG content changes."""
        output = tmp_path / "metrics.svg"
        start = date(2026, 8, 24)
        daily_downloads = {start + timedelta(days=offset): 100 for offset in range(7)}

        with (
            patch("scripts.update_weekly_metrics.fetch_star_count", return_value=12_125),
            patch("scripts.update_weekly_metrics.fetch_daily_downloads", return_value=daily_downloads),
        ):
            first = update_weekly_metrics.update_metrics(
                output, "roboflow/rf-detr", "rfdetr", 12, 104, date(2026, 9, 3), None
            )
            first_svg = output.read_bytes()
            second = update_weekly_metrics.update_metrics(
                output, "roboflow/rf-detr", "rfdetr", 12, 104, date(2026, 9, 3), None
            )

        assert second == first
        assert output.read_bytes() == first_svg

    def test_malformed_existing_svg_is_not_overwritten(self, tmp_path: Path) -> None:
        """Invalid embedded state must stop before fetching or replacing artifact."""
        output = tmp_path / "metrics.svg"
        malformed = b'<svg><metadata id="weekly-metrics-state">{broken</metadata></svg>'
        output.write_bytes(malformed)

        with (
            patch("scripts.update_weekly_metrics.fetch_star_count") as fetch_stars,
            patch("scripts.update_weekly_metrics.fetch_daily_downloads") as fetch_downloads,
            pytest.raises(update_weekly_metrics.MetricsError, match="Invalid weekly metrics metadata"),
        ):
            update_weekly_metrics.update_metrics(output, "roboflow/rf-detr", "rfdetr", 12, 104, date(2026, 9, 3), None)

        fetch_stars.assert_not_called()
        fetch_downloads.assert_not_called()
        assert output.read_bytes() == malformed

    def test_script_entry_point_executes_after_all_helpers_are_defined(self, tmp_path: Path) -> None:
        """Direct script execution must not call main before later definitions exist."""
        output = tmp_path / "metrics.svg"
        start, _ = update_weekly_metrics.completed_week(datetime.now(timezone.utc).date())
        github_response = io.BytesIO(json.dumps({"stargazers_count": 12_125}).encode())
        pypi_response = io.BytesIO(
            json.dumps(
                {
                    "package": "rfdetr",
                    "type": "overall_downloads",
                    "data": [
                        {
                            "category": "without_mirrors",
                            "date": (start + timedelta(days=offset)).isoformat(),
                            "downloads": 100,
                        }
                        for offset in range(7)
                    ],
                }
            ).encode()
        )

        with (
            patch("urllib.request.urlopen", side_effect=[github_response, pypi_response]),
            patch("sys.argv", ["update_weekly_metrics.py", "--output", str(output)]),
            pytest.raises(SystemExit) as exit_info,
        ):
            runpy.run_path(Path(update_weekly_metrics.__file__), run_name="__main__")

        assert exit_info.value.code == 0
        assert output.exists()
