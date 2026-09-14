# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for weekly project metrics SVG generation."""

import http.client
import io
import json
import runpy
import stat
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from scripts import update_weekly_metrics


def _metric(week_start: date, stars_total: int, new_stars: int | None, downloads: int) -> object:
    """Build one checkpoint for a Monday-to-Sunday week starting at ``week_start``.

    ``week_end`` is the only derived field; every value a test asserts on stays explicit at the call site. A case that
    needs a deliberately invalid (non-calendar) period constructs ``WeeklyMetric`` inline instead.

    Examples:
        >>> _metric(date(2026, 8, 24), stars_total=12_000, new_stars=None, downloads=70_000).week_end
        datetime.date(2026, 8, 30)
    """
    return update_weekly_metrics.WeeklyMetric(
        week_start=week_start,
        week_end=week_start + timedelta(days=6),
        stars_total=stars_total,
        new_stars=new_stars,
        downloads=downloads,
    )


def _state(
    *history: object,
    repository: str = "roboflow/rf-detr",
    package: str = "rfdetr",
) -> object:
    """Build metrics state over ``history``, defaulting to this repository's own source identifiers.

    Examples:
        >>> _state().repository
        'roboflow/rf-detr'
        >>> len(_state(_metric(date(2026, 8, 24), 12_000, None, 70_000)).history)
        1
    """
    return update_weekly_metrics.MetricsState(repository=repository, package=package, history=tuple(history))


def _body_response(body: bytes) -> MagicMock:
    """Build a urlopen stand-in whose context-managed ``read`` returns ``body``.

    Examples:
        >>> _body_response(b"{}").__enter__().read(4)
        b'{}'
    """
    response = MagicMock()
    response.__enter__.return_value.read.return_value = body
    return response


def _failing_response(read_error: Exception) -> MagicMock:
    """Build a urlopen stand-in that opens normally and then fails during ``read``.

    Examples:
        >>> _failing_response(TimeoutError("timed out")).__enter__().read  # doctest: +ELLIPSIS
        <MagicMock ...>
    """
    response = MagicMock()
    response.__enter__.return_value.read.side_effect = read_error
    return response


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

    def test_first_observation_establishes_unknown_delta_baseline(self) -> None:
        """First star count must not claim growth without an earlier checkpoint."""
        updated = update_weekly_metrics.record_week(
            _state(),
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_000,
            downloads=70_000,
            history_limit=52,
        )

        assert updated.history == (_metric(date(2026, 8, 24), stars_total=12_000, new_stars=None, downloads=70_000),)

    def test_next_observation_records_star_delta(self) -> None:
        """New period must calculate star growth from previous cumulative checkpoint."""
        previous = _metric(date(2026, 8, 17), stars_total=12_000, new_stars=75, downloads=70_000)

        updated = update_weekly_metrics.record_week(
            _state(previous),
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_125,
            downloads=80_000,
            history_limit=52,
        )

        assert updated.history[-1].new_stars == 125

    def test_gap_establishes_unknown_delta_baseline(self) -> None:
        """A skipped checkpoint must not be reported as one week's star growth."""
        previous = _metric(date(2026, 8, 17), stars_total=12_000, new_stars=75, downloads=70_000)

        updated = update_weekly_metrics.record_week(
            _state(previous),
            start=date(2026, 8, 31),
            end=date(2026, 9, 6),
            stars_total=12_250,
            downloads=80_000,
            history_limit=52,
        )

        assert updated.history[-1].new_stars is None

    def test_same_period_replaces_observation_without_resetting_delta(self) -> None:
        """A rerun must refresh one period's downloads while holding its star checkpoint.

        A manually dispatched rerun reads a cumulative star count that already includes the in-progress week. Recording
        it against the completed week would move that growth backwards and leave the next scheduled run undercounting by
        the same amount.
        """
        first = _metric(date(2026, 8, 17), stars_total=12_000, new_stars=75, downloads=70_000)
        current = _metric(date(2026, 8, 24), stars_total=12_100, new_stars=100, downloads=79_000)

        updated = update_weekly_metrics.record_week(
            _state(first, current),
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_125,
            downloads=80_000,
            history_limit=52,
        )

        assert updated.history == (
            first,
            _metric(date(2026, 8, 24), stars_total=12_100, new_stars=100, downloads=80_000),
        )

    def test_same_period_rerun_keeps_delta_at_minimum_history_limit(self) -> None:
        """A rerun under ``--history-limit 1`` must keep the delta its predecessor established.

        One retained checkpoint is the smallest configuration the CLI accepts, and it is the only one where the rerun's
        predecessor has already been truncated away. Recomputing the delta there has nothing to measure against, so a
        week of recorded growth would be published as an unknown baseline instead.
        """
        current = _metric(date(2026, 8, 24), stars_total=12_100, new_stars=100, downloads=79_000)

        updated = update_weekly_metrics.record_week(
            _state(current),
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_250,
            downloads=80_000,
            history_limit=1,
        )

        assert updated.history == (_metric(date(2026, 8, 24), stars_total=12_100, new_stars=100, downloads=80_000),)

    def test_rejects_period_older_than_checkpoint(self) -> None:
        """Clock or input regressions must not corrupt ordered checkpoint history."""
        current = _metric(date(2026, 8, 24), stars_total=12_100, new_stars=100, downloads=79_000)

        with pytest.raises(update_weekly_metrics.MetricsError, match="older than latest checkpoint"):
            update_weekly_metrics.record_week(
                _state(current),
                start=date(2026, 8, 17),
                end=date(2026, 8, 23),
                stars_total=12_125,
                downloads=80_000,
                history_limit=52,
            )

    def test_next_observation_records_zero_star_delta(self) -> None:
        """A stalled star count between contiguous checkpoints must record zero, not baseline.

        Only the missing-previous-checkpoint path (tested above) is expected to render '-- baseline'; a real contiguous
        pair with no star movement must compute an explicit 0, which is a materially different claim.
        """
        previous = _metric(date(2026, 8, 17), stars_total=12_000, new_stars=75, downloads=70_000)

        updated = update_weekly_metrics.record_week(
            _state(previous),
            start=date(2026, 8, 24),
            end=date(2026, 8, 30),
            stars_total=12_000,
            downloads=80_000,
            history_limit=52,
        )

        assert updated.history[-1].new_stars == 0

    def test_retains_only_configured_latest_history(self) -> None:
        """Bounded history must keep latest checkpoints and their recorded deltas."""
        history = [
            _metric(
                date(2026, 8, 3) + timedelta(weeks=index),
                stars_total=12_000 + index * 100,
                new_stars=0 if index == 0 else 100,
                downloads=70_000 + index * 1_000,
            )
            for index in range(3)
        ]

        updated = update_weekly_metrics.record_week(
            _state(*history),
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

    def test_checked_in_svg_contains_post_correction_backfill(self) -> None:
        """Published history must contain a validly shaped baseline checkpoint.

        The checked-in SVG is periodically regenerated by the scheduled workflow, so this test asserts the structural
        invariants ``_validate_state`` and ``record_week`` guarantee rather than any one run's literal counts, which
        would go stale on every automated regeneration.
        """
        output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "weekly-metrics.svg"

        state = update_weekly_metrics.load_state(output, "roboflow/rf-detr", "rfdetr")

        assert len(state.history) >= 1
        for index, checkpoint in enumerate(state.history):
            assert checkpoint.week_start.weekday() == 0
            assert checkpoint.week_end.weekday() == 6
            assert checkpoint.stars_total >= 0
            assert checkpoint.downloads >= 0
            if index == 0:
                assert checkpoint.new_stars is None
            else:
                previous = state.history[index - 1]
                assert checkpoint.week_start == previous.week_end + timedelta(days=1)
                assert checkpoint.new_stars == checkpoint.stars_total - previous.stars_total

    def test_load_state_rejects_oversized_existing_svg(self, tmp_path: Path) -> None:
        """An existing artifact past the byte cap must fail closed instead of being parsed unbounded.

        The cap exists to bound how much untrusted on-disk content ``load_state`` will read before parsing; this drives
        that guard directly rather than only relying on it never being reached in practice.
        """
        output = tmp_path / "weekly-metrics.svg"
        output.write_text("<svg>" + "x" * update_weekly_metrics.MAX_SVG_BYTES + "</svg>")

        with pytest.raises(update_weekly_metrics.MetricsError, match="exceeds"):
            update_weekly_metrics.load_state(output, "roboflow/rf-detr", "rfdetr")

    def test_rendered_metadata_round_trips_complete_state(self) -> None:
        """SVG metadata must remain sole machine-readable checkpoint source."""
        state = _state(_metric(date(2026, 8, 24), stars_total=12_125, new_stars=125, downloads=80_000))

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
        state = _state(repository="owner/repo&mirror", package="package<nightly")

        svg = update_weekly_metrics.render_svg(state, window_weeks=12)

        assert "repo&amp;mirror" in svg
        assert "package&lt;nightly" in svg
        assert update_weekly_metrics.parse_state(svg, "owner/repo&mirror", "package<nightly") == state


class TestSvgRendering:
    """Tests for deterministic dual-axis chart output."""

    def test_baseline_does_not_claim_zero_weekly_star_growth(self) -> None:
        """Missing prior checkpoint must render as baseline rather than measured zero growth."""
        state = _state(_metric(date(2026, 8, 24), stars_total=12_125, new_stars=None, downloads=80_000))

        svg = update_weekly_metrics.render_svg(state, window_weeks=12)

        assert "— baseline" in svg
        assert "+0 new stars" not in svg

    def test_baseline_week_draws_no_download_mark(self) -> None:
        """A baseline week must draw no download point either, so both series start together.

        ``new_stars is None`` means the week has no delta context; plotting its download total anyway would leave a lone
        blue dot with no matching purple bar, which reads as a rendering glitch rather than a chart baseline.
        """
        state = _state(
            _metric(date(2026, 8, 17), stars_total=12_000, new_stars=None, downloads=70_000),
            _metric(date(2026, 8, 24), stars_total=12_125, new_stars=125, downloads=80_000),
        )

        svg = update_weekly_metrics.render_svg(state, window_weeks=12)

        # r="5" is the data-point radius; the legend swatch circle uses r="4" and must not be counted.
        assert svg.count('r="5"') == 1
        # rx="4" is the bar corner radius; summary-card (rx="10") and legend (rx="2") rects must not be counted.
        assert svg.count('rx="4"') == 1

    def test_renders_zero_downloads_week_without_dividing_by_zero(self) -> None:
        """A week with zero downloads must not collapse the download axis scale.

        The axis-scaling ``download_max`` guard clamps to a minimum of 1 so a genuinely empty week does not divide the
        y-position of every mark by zero.
        """
        state = _state(_metric(date(2026, 8, 24), stars_total=12_125, new_stars=None, downloads=0))

        svg = update_weekly_metrics.render_svg(state, window_weeks=12)

        assert "0 weekly downloads" in svg

    def test_chart_uses_display_window_and_latest_summary(self) -> None:
        """Chart marks must show only configured recent periods plus latest totals."""
        state = _state(
            _metric(date(2026, 8, 17), stars_total=12_000, new_stars=75, downloads=70_000),
            _metric(date(2026, 8, 24), stars_total=12_125, new_stars=125, downloads=80_000),
        )

        svg = update_weekly_metrics.render_svg(state, window_weeks=1)

        assert 'id="weekly-stars-bars"' in svg
        assert 'id="weekly-downloads-line"' in svg
        assert "+125 new stars" in svg
        assert "80,000 weekly downloads" in svg
        assert "Aug 30" in svg
        assert "Aug 23" not in svg


class TestFetchJson:
    """Tests for the shared bounded JSON API request helper.

    ``_fetch_json`` is the single chokepoint every upstream call passes through, so its failure branches are tested
    directly here rather than only indirectly through ``fetch_star_count``/``fetch_daily_downloads``.
    """

    @pytest.mark.parametrize(
        ("error", "match"),
        [
            pytest.param(
                HTTPError("https://example.invalid/api", 404, "Not Found", None, None),
                "HTTP 404",
                id="http-error",
            ),
            pytest.param(URLError("Name or service not known"), "Name or service not known", id="url-error"),
        ],
    )
    def test_reports_connection_failure_as_metrics_error(self, error: Exception, match: str) -> None:
        """A failure raised by urlopen itself must surface as MetricsError, not a raw urllib traceback.

        Both a non-2xx status and a DNS/connection-refused failure reach the caller from urlopen, so each must be
        translated with the detail (status code, resolver reason) that makes a scheduled-run log actionable.
        """
        with (
            patch("scripts.update_weekly_metrics.urlopen", side_effect=error),
            pytest.raises(update_weekly_metrics.MetricsError, match=match),
        ):
            update_weekly_metrics._fetch_json("https://example.invalid/api", {})

    @pytest.mark.parametrize(
        "read_error",
        [
            pytest.param(TimeoutError("timed out"), id="timeout"),
            pytest.param(http.client.IncompleteRead(b"partial"), id="incomplete-read"),
        ],
    )
    def test_reports_body_read_failure_as_metrics_error(self, read_error: Exception) -> None:
        """A connection lost after the response opens must fail closed, not escape as a traceback.

        urlopen returns normally and only the body read fails, so the failure misses both the HTTPError and URLError
        branches. main() reports MetricsError alone, so any other exception reaches the scheduled workflow log as an
        unhandled traceback.
        """
        with (
            patch("scripts.update_weekly_metrics.urlopen", return_value=_failing_response(read_error)),
            pytest.raises(update_weekly_metrics.MetricsError, match="API request failed"),
        ):
            update_weekly_metrics._fetch_json("https://example.invalid/api", {})

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            pytest.param(b"x" * 11, "exceeds 10 bytes", id="over-byte-cap"),
            pytest.param(b"not json", "invalid JSON", id="not-json"),
            pytest.param(b"[1, 2, 3]", "not an object", id="json-array"),
        ],
    )
    def test_reports_invalid_body_as_metrics_error(self, body: bytes, match: str) -> None:
        """A body that is oversized, unparsable, or not a JSON object must fail closed.

        The byte cap is lowered to 10 for every case so the oversized body stays small; the two well-formed-length
        bodies sit under that cap and therefore still reach the parsing branches they target.
        """
        with (
            patch("scripts.update_weekly_metrics.urlopen", return_value=_body_response(body)),
            patch("scripts.update_weekly_metrics.MAX_API_RESPONSE_BYTES", 10),
            pytest.raises(update_weekly_metrics.MetricsError, match=match),
        ):
            update_weekly_metrics._fetch_json("https://example.invalid/api", {})


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

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            pytest.param(
                {"package": "other-pkg", "type": "overall_downloads", "data": []},
                "does not match requested package",
                id="mismatched-package-field",
            ),
            pytest.param(
                {"package": "rfdetr", "type": "python_major", "data": []},
                "does not match requested package",
                id="mismatched-type-field",
            ),
            pytest.param(
                {"package": "rfdetr", "type": "overall_downloads", "data": "not-a-list"},
                "missing daily data",
                id="non-list-data",
            ),
            pytest.param(
                {"package": "rfdetr", "type": "overall_downloads", "data": ["not-a-dict"]},
                "invalid download row",
                id="non-dict-row",
            ),
            pytest.param(
                {
                    "package": "rfdetr",
                    "type": "overall_downloads",
                    "data": [{"category": "with_mirrors", "date": "2026-08-24", "downloads": 1}],
                },
                "invalid download row",
                id="wrong-category",
            ),
            pytest.param(
                {
                    "package": "rfdetr",
                    "type": "overall_downloads",
                    "data": [{"category": "without_mirrors", "downloads": 1}],
                },
                "invalid download row",
                id="missing-date-key",
            ),
            pytest.param(
                {
                    "package": "rfdetr",
                    "type": "overall_downloads",
                    "data": [{"category": "without_mirrors", "date": "2026-08-24"}],
                },
                "invalid download row",
                id="missing-downloads-key",
            ),
        ],
    )
    def test_fetch_daily_downloads_rejects_malformed_response(self, payload: dict, match: str) -> None:
        """Every deviation from the documented PyPI Stats response shape must fail closed.

        A schema drift upstream (renamed field, dropped category, truncated row) must surface as a MetricsError the
        scheduled workflow can report, not an unhandled KeyError/TypeError deep in aggregation.
        """
        response = io.BytesIO(json.dumps(payload).encode())

        with (
            patch("scripts.update_weekly_metrics.urlopen", return_value=response),
            pytest.raises(update_weekly_metrics.MetricsError, match=match),
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
        assert state.history[0].new_stars is None
        assert state.history[0].downloads == 91_000
        assert output.stat().st_mode & stat.S_IROTH

    def test_cli_writes_custom_output(self, tmp_path: Path) -> None:
        """Command entry point must run complete generator with explicit output path.

        ``main`` reads the current instant internally rather than accepting one as an argument, so the test freezes
        ``datetime.now`` instead of computing its own ``start`` from a second, independent call — a UTC-midnight
        crossing between the two would otherwise desynchronise them and flake.
        """
        output = tmp_path / "metrics.svg"
        frozen_now = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
        start, _ = update_weekly_metrics.completed_week(frozen_now.date())
        daily_downloads = {start + timedelta(days=offset): 100 for offset in range(7)}

        with (
            patch("scripts.update_weekly_metrics.datetime") as mock_datetime,
            patch("scripts.update_weekly_metrics.fetch_star_count", return_value=12_125),
            patch("scripts.update_weekly_metrics.fetch_daily_downloads", return_value=daily_downloads),
        ):
            mock_datetime.now.return_value = frozen_now
            exit_code = update_weekly_metrics.main(["--output", str(output), "--weeks", "4"])

        assert exit_code == 0
        assert output.read_text().startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_same_week_rerun_is_byte_identical(self, tmp_path: Path) -> None:
        """Unchanged download data must not create SVG content changes as the star count moves.

        Stars accrue continuously, so a rerun minutes later reads a larger total. Only a rerun that leaves the artifact
        untouched keeps the scheduled workflow from opening an empty pull request every time it is dispatched manually.
        """
        output = tmp_path / "metrics.svg"
        start = date(2026, 8, 24)
        daily_downloads = {start + timedelta(days=offset): 100 for offset in range(7)}

        with (
            patch("scripts.update_weekly_metrics.fetch_star_count", side_effect=[12_125, 12_310]),
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

    def test_failed_write_removes_its_temporary_file(self, tmp_path: Path) -> None:
        """A write that fails part way must not leave a temporary file beside the artifact.

        The temporary is created with delete=False so the completed file can be renamed into place atomically. A failure
        between creation and rename therefore strands a dotfile in docs/assets/, which the next automated commit would
        publish alongside the real SVG.
        """
        output = tmp_path / "metrics.svg"
        stranded = tmp_path / ".metrics.svg.stranded.tmp"
        stranded.write_text("partial")
        temporary = MagicMock()
        temporary.__enter__.return_value.name = str(stranded)
        temporary.__enter__.return_value.write.side_effect = OSError("no space left on device")

        with (
            patch("scripts.update_weekly_metrics.tempfile.NamedTemporaryFile", return_value=temporary),
            pytest.raises(update_weekly_metrics.MetricsError, match="Could not write generated SVG"),
        ):
            update_weekly_metrics.write_svg(output, "<svg/>")

        assert not stranded.exists()

    def test_script_entry_point_executes_after_all_helpers_are_defined(self, tmp_path: Path) -> None:
        """Direct script execution must not call main before later definitions exist.

        ``runpy`` re-executes the module fresh under ``__main__``, so its own internal ``datetime.now()`` call cannot be
        reached through the already-imported ``update_weekly_metrics`` module — the global ``datetime.datetime`` class
        is patched instead, which the freshly re-executed script's own ``from datetime import datetime`` resolves to.
        Freezing it (rather than computing ``start`` from a second, independent ``datetime.now()`` call in this test)
        removes a UTC-midnight desync flake risk.
        """
        output = tmp_path / "metrics.svg"
        frozen_now = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
        start, _ = update_weekly_metrics.completed_week(frozen_now.date())
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
            patch("datetime.datetime") as mock_datetime_cls,
            patch("urllib.request.urlopen", side_effect=[github_response, pypi_response]),
            patch("sys.argv", ["update_weekly_metrics.py", "--output", str(output)]),
            pytest.raises(SystemExit) as exit_info,
        ):
            mock_datetime_cls.now.return_value = frozen_now
            runpy.run_path(Path(update_weekly_metrics.__file__), run_name="__main__")

        assert exit_info.value.code == 0
        assert output.exists()
