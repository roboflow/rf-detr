# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Generate RF-DETR's self-contained weekly metrics SVG."""

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import escape, unescape
from pathlib import Path
from typing import Mapping, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

STATE_SCHEMA_VERSION = 1
STATE_PATTERN = re.compile(r'<metadata id="weekly-metrics-state">(?P<payload>.*?)</metadata>', re.DOTALL)
MAX_API_RESPONSE_BYTES = 5_000_000
MAX_SVG_BYTES = 1_000_000


class MetricsError(RuntimeError):
    """Raised when metrics cannot be fetched, validated, or rendered."""


@dataclass(frozen=True)
class WeeklyMetric:
    """Metrics captured for one completed calendar week."""

    week_start: date
    week_end: date
    stars_total: int
    new_stars: int
    downloads: int


@dataclass(frozen=True)
class MetricsState:
    """Validated checkpoint state embedded in generated SVG."""

    repository: str
    package: str
    history: tuple[WeeklyMetric, ...]


def _require_nonnegative_int(value: object, name: str) -> int:
    """Validate one count from checkpoint or upstream data.

    Args:
        value: Candidate count.
        name: Field name used in validation errors.

    Returns:
        Validated integer.

    Raises:
        MetricsError: If value is a boolean, non-integer, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a non-negative integer"
        raise MetricsError(msg)
    return value


def _fetch_json(url: str, headers: Mapping[str, str]) -> dict[str, object]:
    """Fetch and decode one bounded JSON API response.

    Args:
        url: Public API endpoint without credentials.
        headers: HTTP headers, including optional authentication.

    Returns:
        Decoded JSON object.

    Raises:
        MetricsError: If request fails or response is invalid or too large.
    """
    request = Request(url, headers=dict(headers))
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read(MAX_API_RESPONSE_BYTES + 1)
    except HTTPError as error:
        msg = f"API request failed with HTTP {error.code}: {url}"
        raise MetricsError(msg) from error
    except URLError as error:
        msg = f"API request failed: {url}: {error.reason}"
        raise MetricsError(msg) from error
    if len(body) > MAX_API_RESPONSE_BYTES:
        msg = f"API response exceeds {MAX_API_RESPONSE_BYTES:,} bytes: {url}"
        raise MetricsError(msg)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"API returned invalid JSON: {url}"
        raise MetricsError(msg) from error
    if not isinstance(payload, dict):
        msg = f"API returned JSON that is not an object: {url}"
        raise MetricsError(msg)
    return cast(dict[str, object], payload)


def fetch_star_count(repository: str, token: str | None = None) -> int:
    """Fetch current GitHub stargazer count.

    Args:
        repository: GitHub repository in ``owner/name`` form.
        token: Optional GitHub token sent only in authorization header.

    Returns:
        Current cumulative stargazer count.

    Raises:
        MetricsError: If repository identifier or API response is invalid.
    """
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        msg = "repository must use owner/name format"
        raise MetricsError(msg)
    owner, name = (quote(part, safe="") for part in parts)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "rf-detr-weekly-metrics",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = _fetch_json(f"https://api.github.com/repos/{owner}/{name}", headers)
    try:
        return _require_nonnegative_int(payload["stargazers_count"], "stargazers_count")
    except KeyError as error:
        msg = "GitHub response is missing stargazers_count"
        raise MetricsError(msg) from error


def fetch_daily_downloads(package: str) -> dict[date, int]:
    """Fetch PyPI Stats daily downloads excluding known mirrors.

    PyPI does not expose a lifetime cumulative counter. Daily observations are
    retained by PyPI Stats and summed into completed calendar weeks instead.

    Args:
        package: PyPI package name.

    Returns:
        Download counts indexed by UTC date.

    Raises:
        MetricsError: If package name or API response is invalid.
    """
    if not isinstance(package, str) or not package.strip():
        msg = "package must be a non-empty string"
        raise MetricsError(msg)
    url = f"https://pypistats.org/api/packages/{quote(package, safe='')}/overall?mirrors=false"
    payload = _fetch_json(url, {"Accept": "application/json", "User-Agent": "rf-detr-weekly-metrics"})
    if payload.get("package") != package or payload.get("type") != "overall_downloads":
        msg = "PyPI Stats response source does not match requested package"
        raise MetricsError(msg)
    rows = payload.get("data")
    if not isinstance(rows, list):
        msg = "PyPI Stats response is missing daily data"
        raise MetricsError(msg)

    downloads: dict[date, int] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("category") != "without_mirrors":
            msg = "PyPI Stats response contains an invalid download row"
            raise MetricsError(msg)
        try:
            day = date.fromisoformat(cast(str, row["date"]))
            count = _require_nonnegative_int(row["downloads"], "downloads")
        except (KeyError, TypeError, ValueError) as error:
            msg = "PyPI Stats response contains an invalid download row"
            raise MetricsError(msg) from error
        if day in downloads:
            msg = f"PyPI Stats response contains duplicate data for {day.isoformat()}"
            raise MetricsError(msg)
        downloads[day] = count
    return downloads


def load_state(output: Path, repository: str, package: str) -> MetricsState:
    """Load embedded state or create empty state for missing output.

    Args:
        output: SVG artifact path.
        repository: Expected GitHub repository identifier.
        package: Expected PyPI package name.

    Returns:
        Existing validated state or empty initial state.

    Raises:
        MetricsError: If existing SVG is too large or contains invalid state.
    """
    empty = MetricsState(repository=repository, package=package, history=())
    _validate_state(empty)
    if not output.exists():
        return empty
    if output.stat().st_size > MAX_SVG_BYTES:
        msg = f"Existing SVG exceeds {MAX_SVG_BYTES:,} bytes: {output}"
        raise MetricsError(msg)
    try:
        svg = output.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"Could not read existing SVG: {output}"
        raise MetricsError(msg) from error
    return parse_state(svg, repository, package)


def write_svg(output: Path, svg: str) -> None:
    """Atomically write generated SVG beside existing artifact.

    Args:
        output: Destination SVG path.
        svg: Complete generated SVG document.

    Raises:
        MetricsError: If directory creation or atomic replacement fails.
    """
    temporary_path: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary.write(svg)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        msg = f"Could not write generated SVG: {output}"
        raise MetricsError(msg) from error


def update_metrics(
    output: Path,
    repository: str,
    package: str,
    window_weeks: int,
    history_limit: int,
    today: date,
    github_token: str | None,
) -> MetricsState:
    """Fetch current metrics and update self-contained SVG artifact.

    Args:
        output: SVG artifact path.
        repository: GitHub repository in ``owner/name`` form.
        package: PyPI package name.
        window_weeks: Number of recent periods displayed.
        history_limit: Maximum embedded periods retained.
        today: Current UTC date used to select completed week.
        github_token: Optional GitHub token passed only through request header.

    Returns:
        Updated metrics state written to SVG.

    Raises:
        MetricsError: If existing state, upstream data, or output is invalid.
    """
    if history_limit < window_weeks:
        msg = "history_limit must be greater than or equal to window_weeks"
        raise MetricsError(msg)
    state = load_state(output, repository, package)
    start, end = completed_week(today)
    stars_total = fetch_star_count(repository, github_token)
    daily_downloads = fetch_daily_downloads(package)
    downloads = weekly_downloads(daily_downloads, start, end)
    updated = record_week(state, start, end, stars_total, downloads, history_limit)
    write_svg(output, render_svg(updated, window_weeks))
    return updated


def _validate_state(state: MetricsState) -> None:
    """Validate source identifiers and ordered weekly checkpoints.

    Args:
        state: State loaded from metadata or built from current metrics.

    Raises:
        MetricsError: If state violates checkpoint schema invariants.
    """
    if not isinstance(state.repository, str) or state.repository.count("/") != 1:
        msg = "repository must use owner/name format"
        raise MetricsError(msg)
    if not all(state.repository.split("/")):
        msg = "repository must use owner/name format"
        raise MetricsError(msg)
    if not isinstance(state.package, str) or not state.package.strip():
        msg = "package must be a non-empty string"
        raise MetricsError(msg)

    previous: WeeklyMetric | None = None
    for metric in state.history:
        if not isinstance(metric.week_start, date) or not isinstance(metric.week_end, date):
            msg = "week_start and week_end must be ISO dates"
            raise MetricsError(msg)
        if metric.week_start.weekday() != 0 or metric.week_end != metric.week_start + timedelta(days=6):
            msg = "Each history item must cover one Monday-to-Sunday week"
            raise MetricsError(msg)
        _require_nonnegative_int(metric.stars_total, "stars_total")
        _require_nonnegative_int(metric.downloads, "downloads")
        if isinstance(metric.new_stars, bool) or not isinstance(metric.new_stars, int):
            msg = "new_stars must be an integer"
            raise MetricsError(msg)
        if previous is not None:
            if metric.week_start <= previous.week_start:
                msg = "Weekly metrics history must be strictly chronological"
                raise MetricsError(msg)
            is_contiguous = metric.week_start == previous.week_end + timedelta(days=1)
            expected_new_stars = metric.stars_total - previous.stars_total if is_contiguous else 0
            if metric.new_stars != expected_new_stars:
                msg = "new_stars does not match its checkpoint interval"
                raise MetricsError(msg)
        previous = metric


def _serialize_state(state: MetricsState) -> str:
    """Serialize metrics state as deterministic compact JSON.

    Args:
        state: Validated state to serialize.

    Returns:
        Stable JSON representation for SVG metadata.
    """
    _validate_state(state)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "repository": state.repository,
        "package": state.package,
        "history": [
            {
                "week_start": metric.week_start.isoformat(),
                "week_end": metric.week_end.isoformat(),
                "stars_total": metric.stars_total,
                "new_stars": metric.new_stars,
                "downloads": metric.downloads,
            }
            for metric in state.history
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_state(svg: str, repository: str, package: str) -> MetricsState:
    """Extract metrics checkpoint JSON from generated SVG.

    Args:
        svg: Existing SVG document text.
        repository: Expected GitHub repository identifier.
        package: Expected PyPI package name.

    Returns:
        Parsed metrics state.

    Raises:
        MetricsError: If metadata is missing, invalid, or belongs to another source.
    """
    matches = list(STATE_PATTERN.finditer(svg))
    if len(matches) != 1:
        msg = "Expected exactly one weekly-metrics-state metadata element"
        raise MetricsError(msg)
    try:
        payload = json.loads(unescape(matches[0].group("payload")))
        state = MetricsState(
            repository=payload["repository"],
            package=payload["package"],
            history=tuple(
                WeeklyMetric(
                    week_start=date.fromisoformat(metric["week_start"]),
                    week_end=date.fromisoformat(metric["week_end"]),
                    stars_total=metric["stars_total"],
                    new_stars=metric["new_stars"],
                    downloads=metric["downloads"],
                )
                for metric in payload["history"]
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        msg = "Invalid weekly metrics metadata"
        raise MetricsError(msg) from error
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        msg = f"Unsupported weekly metrics schema: {payload.get('schema_version')!r}"
        raise MetricsError(msg)
    if state.repository != repository or state.package != package:
        msg = "Embedded metrics sources do not match requested repository and package"
        raise MetricsError(msg)
    _validate_state(state)
    return state


def render_svg(state: MetricsState, window_weeks: int) -> str:
    """Render metrics state as one self-contained SVG document.

    Args:
        state: Metrics state embedded into output.
        window_weeks: Maximum recent periods displayed in chart.

    Returns:
        Deterministic SVG text.

    Raises:
        MetricsError: If display window is not positive.
    """
    if isinstance(window_weeks, bool) or not isinstance(window_weeks, int) or window_weeks < 1:
        msg = "Display window must be at least one week"
        raise MetricsError(msg)
    _validate_state(state)
    metadata = escape(_serialize_state(state), quote=False)
    display = state.history[-window_weeks:]
    latest = display[-1] if display else None
    if latest is None:
        description = "Weekly GitHub star growth and PyPI download history. No completed periods recorded."
        latest_period = "Awaiting first completed week"
        stars_total = "—"
        new_stars = "— new stars"
        downloads = "— weekly downloads"
    else:
        description = (
            f"For week ending {latest.week_end.isoformat()}, RF-DETR had {latest.new_stars:+,} new GitHub stars "
            f"and {latest.downloads:,} PyPI downloads."
        )
        start_label = latest.week_start.strftime("%b %d").replace(" 0", " ")
        end_label = latest.week_end.strftime("%b %d, %Y").replace(" 0", " ")
        latest_period = f"Latest completed week · {start_label}–{end_label}"
        stars_total = f"{latest.stars_total:,}"
        new_stars = f"{latest.new_stars:+,} new stars"
        downloads = f"{latest.downloads:,} weekly downloads"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="640" viewBox="0 0 1200 640" '
        'role="img" aria-labelledby="title description">',
        '  <title id="title">RF-DETR weekly project metrics</title>',
        f'  <desc id="description">{escape(description)}</desc>',
        f'  <metadata id="weekly-metrics-state">{metadata}</metadata>',
        '  <rect width="1200" height="640" rx="18" fill="#ffffff"/>',
        '  <text x="56" y="58" fill="#0b0b0b" font-family="DejaVu Sans, Arial, sans-serif" '
        'font-size="30" font-weight="700">RF-DETR weekly growth</text>',
        f'  <text x="56" y="86" fill="#6b6a66" font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-size="15">{escape(latest_period)}</text>',
        '  <g font-family="DejaVu Sans, Arial, sans-serif">',
        '    <rect x="56" y="112" width="336" height="82" rx="10" fill="#f5f1fa"/>',
        '    <text x="76" y="141" fill="#6b6a66" font-size="14">GITHUB STARS</text>',
        f'    <text x="76" y="176" fill="#0b0b0b" font-size="28" font-weight="700">{stars_total}</text>',
        '    <rect x="410" y="112" width="336" height="82" rx="10" fill="#f5f1fa"/>',
        '    <text x="430" y="141" fill="#6b6a66" font-size="14">WEEKLY STAR CHANGE</text>',
        f'    <text x="430" y="176" fill="#8315f9" font-size="28" font-weight="700">{new_stars}</text>',
        '    <rect x="764" y="112" width="380" height="82" rx="10" fill="#f5f1fa"/>',
        '    <text x="784" y="141" fill="#6b6a66" font-size="14">PYPI · EXCLUDING MIRRORS</text>',
        f'    <text x="784" y="176" fill="#2a78d6" font-size="28" font-weight="700">{downloads}</text>',
        "  </g>",
    ]

    plot_left = 100.0
    plot_top = 240.0
    plot_width = 1000.0
    plot_height = 290.0
    plot_bottom = plot_top + plot_height
    star_values = [metric.new_stars for metric in display] or [0]
    star_min = min(0, min(star_values))
    star_max = max(0, max(star_values))
    if star_min == star_max:
        star_max = star_min + 1
    download_max = max([metric.downloads for metric in display] or [1])
    download_max = max(download_max, 1)

    lines.extend(
        [
            '  <g aria-hidden="true" fill="none" stroke="#e1e0d9" stroke-width="1">',
        ]
    )
    for step in range(3):
        fraction = step / 2
        y = plot_bottom - fraction * plot_height
        lines.append(f'    <path d="M{plot_left:.0f} {y:.1f}H{plot_left + plot_width:.0f}"/>')
    lines.extend(
        [
            "  </g>",
            f'  <path d="M{plot_left:.0f} {plot_top:.0f}V{plot_bottom:.0f}H{plot_left + plot_width:.0f}" '
            'fill="none" stroke="#c3c2b7" stroke-width="1.5"/>',
            '  <g fill="#6b6a66" font-family="DejaVu Sans, Arial, sans-serif" font-size="13">',
        ]
    )
    for step in range(3):
        fraction = step / 2
        y = plot_bottom - fraction * plot_height
        star_tick = round(star_min + fraction * (star_max - star_min))
        download_tick = round(fraction * download_max)
        lines.append(f'    <text x="88" y="{y + 5:.1f}" text-anchor="end">{star_tick:+,}</text>')
        lines.append(f'    <text x="1112" y="{y + 5:.1f}" text-anchor="start">{download_tick:,}</text>')
    lines.extend(
        [
            '    <text x="100" y="221" fill="#8315f9" font-size="14" font-weight="700">NEW STARS</text>',
            '    <text x="1100" y="221" fill="#2a78d6" font-size="14" font-weight="700" '
            'text-anchor="end">WEEKLY DOWNLOADS</text>',
            "  </g>",
        ]
    )

    if display:
        spacing = plot_width / len(display)
        bar_width = min(44.0, spacing * 0.5)
        zero_y = plot_bottom - ((0 - star_min) / (star_max - star_min)) * plot_height
        points = []
        lines.append('  <g id="weekly-stars-bars">')
        for index, metric in enumerate(display):
            x = plot_left + (index + 0.5) * spacing
            value_y = plot_bottom - ((metric.new_stars - star_min) / (star_max - star_min)) * plot_height
            bar_y = min(value_y, zero_y)
            bar_height = max(abs(zero_y - value_y), 1.0)
            color = "#8315f9" if metric.new_stars >= 0 else "#e85d75"
            lines.append(
                f'    <rect x="{x - bar_width / 2:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="4" fill="{color}"/>'
            )
            download_y = plot_bottom - (metric.downloads / download_max) * plot_height
            points.append(f"{x:.1f},{download_y:.1f}")
        lines.append("  </g>")
        lines.append('  <g id="weekly-downloads-line">')
        if len(points) > 1:
            lines.append(
                f'    <polyline points="{" ".join(points)}" fill="none" stroke="#2a78d6" stroke-width="4" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        for point in points:
            x, y = point.split(",")
            lines.append(f'    <circle cx="{x}" cy="{y}" r="5" fill="#2a78d6" stroke="#ffffff" stroke-width="2"/>')
        lines.append("  </g>")
        lines.append(
            '  <g fill="#6b6a66" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" text-anchor="middle">'
        )
        for index, metric in enumerate(display):
            x = plot_left + (index + 0.5) * spacing
            label = metric.week_end.strftime("%b %d").replace(" 0", " ")
            lines.append(f'    <text x="{x:.1f}" y="552">{label}</text>')
        lines.append("  </g>")
    else:
        lines.extend(
            [
                '  <g id="weekly-stars-bars"/>',
                '  <g id="weekly-downloads-line"/>',
                '  <text x="600" y="395" fill="#898781" font-family="DejaVu Sans, Arial, sans-serif" '
                'font-size="18" text-anchor="middle">Metrics will appear after first scheduled update.</text>',
            ]
        )

    lines.extend(
        [
            '  <g font-family="DejaVu Sans, Arial, sans-serif" font-size="13" fill="#52514e">',
            '    <rect x="396" y="593" width="18" height="12" rx="2" fill="#8315f9"/>',
            '    <text x="424" y="604">New GitHub stars</text>',
            '    <path d="M582 599H610" fill="none" stroke="#2a78d6" stroke-width="4" stroke-linecap="round"/>',
            '    <circle cx="596" cy="599" r="4" fill="#2a78d6"/>',
            '    <text x="620" y="604">PyPI downloads</text>',
            "  </g>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def completed_week(today: date) -> tuple[date, date]:
    """Return latest completed Monday-to-Sunday period.

    Args:
        today: Current UTC calendar date.

    Returns:
        Inclusive start and end dates for latest completed week.

    Example:
        >>> completed_week(date(2026, 9, 3))
        (datetime.date(2026, 8, 24), datetime.date(2026, 8, 30))
    """
    days_since_sunday = today.weekday() + 1
    end = today - timedelta(days=days_since_sunday)
    return end - timedelta(days=6), end


def weekly_downloads(daily_downloads: Mapping[date, int], start: date, end: date) -> int:
    """Sum daily downloads across an inclusive period.

    Args:
        daily_downloads: Download count indexed by UTC date.
        start: First date included in period.
        end: Last date included in period.

    Returns:
        Total downloads between ``start`` and ``end``.

    Example:
        >>> days = {date(2026, 8, 24) + timedelta(days=i): i + 1 for i in range(7)}
        >>> weekly_downloads(days, date(2026, 8, 24), date(2026, 8, 30))
        28
    """
    if start.weekday() != 0 or end != start + timedelta(days=6):
        msg = "Download period must cover one Monday-to-Sunday week"
        raise MetricsError(msg)
    total = 0
    current = start
    while current <= end:
        try:
            total += daily_downloads[current]
        except KeyError as error:
            msg = f"PyPI Stats response is missing download data for {current.isoformat()}"
            raise MetricsError(msg) from error
        current += timedelta(days=1)
    return total


def record_week(
    state: MetricsState,
    start: date,
    end: date,
    stars_total: int,
    downloads: int,
    history_limit: int,
) -> MetricsState:
    """Append one completed week, resetting star baseline after any checkpoint gap.

    Args:
        state: Existing embedded metrics state.
        start: First date in completed week.
        end: Last date in completed week.
        stars_total: Current GitHub stargazer count.
        downloads: PyPI downloads during completed week.
        history_limit: Maximum number of weekly records retained.

    Returns:
        Updated state with bounded history.
    """
    _validate_state(state)
    _require_nonnegative_int(stars_total, "stars_total")
    _require_nonnegative_int(downloads, "downloads")
    if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit < 1:
        msg = "history_limit must be a positive integer"
        raise MetricsError(msg)
    if start.weekday() != 0 or end != start + timedelta(days=6):
        msg = "Recorded period must cover one Monday-to-Sunday week"
        raise MetricsError(msg)

    history = state.history
    if history and start < history[-1].week_start:
        msg = f"Week {start.isoformat()} is older than latest checkpoint {history[-1].week_start.isoformat()}"
        raise MetricsError(msg)
    if history and history[-1].week_start == start:
        history = history[:-1]
    previous_stars = stars_total
    if history and start == history[-1].week_end + timedelta(days=1):
        previous_stars = history[-1].stars_total
    metric = WeeklyMetric(
        week_start=start,
        week_end=end,
        stars_total=stars_total,
        new_stars=stars_total - previous_stars,
        downloads=downloads,
    )
    history = (*history, metric)[-history_limit:]
    updated = MetricsState(repository=state.repository, package=state.package, history=history)
    _validate_state(updated)
    return updated


def _positive_int(value: str) -> int:
    """Parse positive integer command-line argument.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed < 1:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run weekly metrics SVG generator.

    Args:
        argv: Optional command-line arguments, excluding executable name.

    Returns:
        Process exit code. Zero means SVG was updated successfully.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Update a self-contained SVG with GitHub star deltas and PyPI Stats downloads "
            "for latest completed Monday-to-Sunday week."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("docs/assets/weekly-metrics.svg"))
    parser.add_argument("--repository", default="roboflow/rf-detr")
    parser.add_argument("--package", default="rfdetr")
    parser.add_argument("--weeks", type=_positive_int, default=12, help="recent weeks displayed (default: 12)")
    parser.add_argument(
        "--history-limit",
        type=_positive_int,
        default=104,
        help="weekly checkpoints retained in SVG metadata (default: 104)",
    )
    args = parser.parse_args(argv)
    try:
        update_metrics(
            output=args.output,
            repository=args.repository,
            package=args.package,
            window_weeks=args.weeks,
            history_limit=args.history_limit,
            today=datetime.now(timezone.utc).date(),
            github_token=os.environ.get("GITHUB_TOKEN"),
        )
    except MetricsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
