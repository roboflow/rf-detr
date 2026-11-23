# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Generate RF-DETR's self-contained weekly metrics SVG."""

import argparse
import http.client
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
DEPENDENTS_PATTERN = re.compile(r"(?P<count>\d[\d,]*)\s+Repositories\b")
MAX_API_RESPONSE_BYTES = 5_000_000
MAX_SVG_BYTES = 1_000_000
PLOT_LEFT = 100.0
PLOT_TOP = 240.0
PLOT_WIDTH = 1000.0
PLOT_HEIGHT = 290.0
PLOT_BOTTOM = PLOT_TOP + PLOT_HEIGHT
# One colour per metric, shared by its headline card and its chart series so the eye can follow it.
STARS_COLOR = "#b8860b"
USED_BY_COLOR = "#8315f9"
DOWNLOADS_COLOR = "#2a78d6"
CARD_FILL = "#f6f5f0"


class MetricsError(RuntimeError):
    """Raised when metrics cannot be fetched, validated, or rendered."""


@dataclass(frozen=True)
class WeeklyMetric:
    """Metrics captured for one completed calendar week."""

    week_start: date
    week_end: date
    stars_total: int
    new_stars: int | None
    downloads: int
    # Dependent repositories GitHub lists for the package at the last run of this week; None when never captured.
    used_by: int | None = None


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


def _fetch_bytes(url: str, headers: Mapping[str, str]) -> bytes:
    """Fetch one bounded HTTP response body.

    Args:
        url: Public endpoint without credentials.
        headers: HTTP headers, including optional authentication.

    Returns:
        Raw response body.

    Raises:
        MetricsError: If request fails or response is too large.
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
    except (OSError, http.client.HTTPException) as error:
        # Timeouts, TLS failures, and truncated bodies surface from read() rather than from urlopen().
        # HTTPException is not an OSError, so both bases are needed to keep every transport failure inside
        # the MetricsError contract that main() reports instead of an unhandled traceback.
        msg = f"API request failed: {url}: {error!r}"
        raise MetricsError(msg) from error
    if len(body) > MAX_API_RESPONSE_BYTES:
        msg = f"API response exceeds {MAX_API_RESPONSE_BYTES:,} bytes: {url}"
        raise MetricsError(msg)
    return cast(bytes, body)


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
    body = _fetch_bytes(url, headers)
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


def parse_used_by(page: str) -> int:
    """Extract the dependent-repository count from a GitHub dependents page.

    GitHub exposes no API for the "Used by" figure, so the count is read from the visible
    ``N Repositories`` tab label, the same number the repository sidebar shows.

    Args:
        page: HTML of ``/network/dependents``.

    Returns:
        Number of dependent repositories.

    Raises:
        MetricsError: If the page carries no ``N Repositories`` label.

    Example:
        >>> parse_used_by('<a>\\n  <svg></svg>\\n  1,209\\n  Repositories\\n</a><a> 52 Packages</a>')
        1209
    """
    text = re.sub(r"<[^>]*>", " ", page)
    match = DEPENDENTS_PATTERN.search(text)
    if match is None:
        msg = "GitHub dependents page has no repository count"
        raise MetricsError(msg)
    return int(match.group("count").replace(",", ""))


def fetch_used_by(repository: str) -> int:
    """Fetch how many repositories GitHub lists as depending on the package.

    Args:
        repository: GitHub repository in ``owner/name`` form.

    Returns:
        Dependent-repository count for the repository's default package.

    Raises:
        MetricsError: If the repository identifier or the page is invalid.
    """
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        msg = "repository must use owner/name format"
        raise MetricsError(msg)
    owner, name = (quote(part, safe="") for part in parts)
    body = _fetch_bytes(
        f"https://github.com/{owner}/{name}/network/dependents",
        {"Accept": "text/html", "User-Agent": "rf-detr-weekly-metrics"},
    )
    return parse_used_by(body.decode("utf-8", errors="replace"))


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
        MetricsError: If directory creation, buffered write, or atomic replacement fails.
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
            # Record the path before writing: the file already exists on disk with delete=False, so a
            # failed write would otherwise leave it behind with no handle for the cleanup branch.
            temporary_path = Path(temporary.name)
            temporary.write(svg)
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
    try:
        used_by: int | None = fetch_used_by(repository)
    except MetricsError as error:
        # Scraped, unofficial source: a layout change or outage must not block the star and download update.
        print(f"warning: used-by count unavailable: {error}", file=sys.stderr)
        used_by = None
    updated = record_week(state, start, end, stars_total, downloads, history_limit, used_by)
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
        if metric.used_by is not None:
            _require_nonnegative_int(metric.used_by, "used_by")
        if metric.new_stars is not None and (
            isinstance(metric.new_stars, bool) or not isinstance(metric.new_stars, int)
        ):
            msg = "new_stars must be an integer or null"
            raise MetricsError(msg)
        if previous is not None:
            if metric.week_start <= previous.week_start:
                msg = "Weekly metrics history must be strictly chronological"
                raise MetricsError(msg)
            is_contiguous = metric.week_start == previous.week_end + timedelta(days=1)
            expected_new_stars = metric.stars_total - previous.stars_total if is_contiguous else None
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
        # ``used_by`` is omitted while unknown, so checkpoints written before it existed stay byte-identical.
        "history": [
            {
                "week_start": metric.week_start.isoformat(),
                "week_end": metric.week_end.isoformat(),
                "stars_total": metric.stars_total,
                "new_stars": metric.new_stars,
                "downloads": metric.downloads,
                **({} if metric.used_by is None else {"used_by": metric.used_by}),
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
                    used_by=metric.get("used_by"),
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


def format_count(value: int) -> str:
    """Shorten a count for display, with ``k`` for thousands and ``M`` for millions.

    Only the drawn labels use this. The embedded checkpoint state and the accessible description keep every exact
    value, so the history loses nothing.

    Args:
        value: Non-negative count.

    Returns:
        The count as is below 1,000. From there one decimal below 100k and none from 100k up, with a trailing
        ``.0`` dropped. The digits are cut, never rounded up, so a label never claims more than was counted.

    Example:
        >>> [format_count(count) for count in (209, 999, 1_000, 10_135, 99_960, 114_513, 999_600, 2_400_000)]
        ['209', '999', '1k', '10.1k', '99.9k', '114k', '999k', '2.4M']
    """
    if value < 1_000:
        return f"{value:,}"
    if value < 100_000:
        return f"{value // 100 / 10:.1f}".removesuffix(".0") + "k"
    if value < 1_000_000:
        return f"{value // 1_000}k"
    return f"{value // 100_000 / 10:.1f}".removesuffix(".0") + "M"


def _render_summary_cards(metadata: str, latest: WeeklyMetric | None) -> list[str]:
    """Render the document header and the three headline metric cards: stars, downloads and used by.

    Args:
        metadata: Serialized checkpoint state, already escaped for XML text content.
        latest: Most recent displayed week, or None when no week has been recorded yet.

    Returns:
        Opening SVG lines, from the XML declaration through the card group.
    """
    if latest is None:
        description = "Weekly GitHub star growth and PyPI download history. No completed periods recorded."
        latest_period = "Awaiting first completed week"
        stars_total = "—"
        downloads = "— weekly downloads"
    elif latest.new_stars is None:
        description = (
            f"For week ending {latest.week_end.isoformat()}, RF-DETR recorded a GitHub star baseline of "
            f"{latest.stars_total:,} and {latest.downloads:,} PyPI downloads."
        )
        start_label = latest.week_start.strftime("%b %d").replace(" 0", " ")
        end_label = latest.week_end.strftime("%b %d, %Y").replace(" 0", " ")
        latest_period = f"Latest completed week · {start_label}–{end_label}"
        stars_total = format_count(latest.stars_total)
        downloads = f"{format_count(latest.downloads)} weekly downloads"
    else:
        description = (
            f"For week ending {latest.week_end.isoformat()}, RF-DETR had {latest.new_stars:+,} new GitHub stars "
            f"and {latest.downloads:,} PyPI downloads."
        )
        start_label = latest.week_start.strftime("%b %d").replace(" 0", " ")
        end_label = latest.week_end.strftime("%b %d, %Y").replace(" 0", " ")
        latest_period = f"Latest completed week · {start_label}–{end_label}"
        stars_total = format_count(latest.stars_total)
        downloads = f"{format_count(latest.downloads)} weekly downloads"

    used_by_value = "—" if latest is None or latest.used_by is None else f"{format_count(latest.used_by)} repos"
    if latest is not None and latest.used_by is not None:
        description += f" {latest.used_by:,} repositories depend on the package."

    return [
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
        *_render_cards(
            [
                ("GITHUB STARS", stars_total, STARS_COLOR),
                ("PYPI · EXCLUDING MIRRORS", downloads, DOWNLOADS_COLOR),
                ("USED BY", used_by_value, USED_BY_COLOR),
            ]
        ),
        "  </g>",
    ]


def _render_cards(cards: Sequence[tuple[str, str, str]]) -> list[str]:
    """Render a row of metric cards whose widths follow the length of their text.

    Each card needs room for its label and its value; the row then splits its fixed width in proportion to
    those needs, so a short count gets a narrow card and a long one a wide card. The label sits at the left
    edge of its card and the value is centred.

    Args:
        cards: One ``(label, value, value colour)`` triple per card, left to right.

    Returns:
        SVG lines, one background rectangle and two text lines per card.

    Example:
        >>> lines = _render_cards([("A", "1", "#000000"), ("B", "1,000,000 downloads", "#000000")])
        >>> [line.split('width="')[1].split('"')[0] for line in lines if "<rect" in line]
        ['137.2', '932.8']
    """
    row_left, row_width, gap = 56.0, 1088.0, 18.0
    needs = [max(len(label) * 9.0, len(value) * 19.0) + 40.0 for label, value, _ in cards]
    scale = (row_width - gap * (len(cards) - 1)) / sum(needs)
    lines = []
    x = row_left
    for (label, value, color), need in zip(cards, needs, strict=True):
        width = need * scale
        lines.extend(
            [
                f'    <rect x="{x:.1f}" y="112" width="{width:.1f}" height="82" rx="10" fill="{CARD_FILL}"/>',
                f'    <text x="{x + 20:.1f}" y="141" fill="#6b6a66" font-size="14">{escape(label)}</text>',
                f'    <text x="{x + width / 2:.1f}" y="176" text-anchor="middle" fill="{color}" font-size="28" '
                f'font-weight="700">{escape(value)}</text>',
            ]
        )
        x += width + gap
    return lines


def used_by_changes(history: Sequence[WeeklyMetric]) -> list[int | None]:
    """Derive the weekly change of the used-by count, aligned with ``history``.

    A change exists only between two directly consecutive weeks that both carry a reading, the same
    rule that gives a star baseline week no star delta.

    Args:
        history: Recorded weeks, oldest first.

    Returns:
        One entry per week: the used-by count minus the previous week's, or None when unknown.

    Example:
        >>> week = WeeklyMetric(date(2026, 11, 9), date(2026, 11, 15), 10, None, 5, 200)
        >>> following = WeeklyMetric(date(2026, 11, 16), date(2026, 11, 22), 12, 2, 6, 209)
        >>> used_by_changes([week, following])
        [None, 9]
    """
    changes: list[int | None] = []
    for index, metric in enumerate(history):
        previous = history[index - 1] if index else None
        known = (
            previous is not None
            and previous.used_by is not None
            and metric.used_by is not None
            and metric.week_start == previous.week_end + timedelta(days=1)
        )
        changes.append(metric.used_by - previous.used_by if known else None)  # type: ignore[operator, union-attr]
    return changes


def _render_axes(star_min: int, star_max: int, download_max: int, show_used_by: bool = False) -> list[str]:
    """Render the chart gridlines, frame, and the two opposing value axes.

    Star counts read against the left axis and download counts against the right one, so both ranges
    are drawn against the same three gridlines.

    Args:
        star_min: Lowest weekly change the left axis must reach, never above zero.
        star_max: Highest weekly change the left axis must reach, never below zero.
        download_max: Highest weekly download total the right axis must reach.
        show_used_by: Name the used-by change in the left axis caption, because its bars share that axis.

    Returns:
        SVG lines for the gridlines, the axis frame, the tick labels, and the axis captions.
    """
    lines = ['  <g aria-hidden="true" fill="none" stroke="#e1e0d9" stroke-width="1">']
    for step in range(3):
        fraction = step / 2
        y = PLOT_BOTTOM - fraction * PLOT_HEIGHT
        lines.append(f'    <path d="M{PLOT_LEFT:.0f} {y:.1f}H{PLOT_LEFT + PLOT_WIDTH:.0f}"/>')
    lines.extend(
        [
            "  </g>",
            f'  <path d="M{PLOT_LEFT:.0f} {PLOT_TOP:.0f}V{PLOT_BOTTOM:.0f}H{PLOT_LEFT + PLOT_WIDTH:.0f}" '
            'fill="none" stroke="#c3c2b7" stroke-width="1.5"/>',
            '  <g fill="#6b6a66" font-family="DejaVu Sans, Arial, sans-serif" font-size="13">',
        ]
    )
    for step in range(3):
        fraction = step / 2
        y = PLOT_BOTTOM - fraction * PLOT_HEIGHT
        star_tick = round(star_min + fraction * (star_max - star_min))
        download_tick = round(fraction * download_max)
        lines.append(f'    <text x="88" y="{y + 5:.1f}" text-anchor="end">{star_tick:+,}</text>')
        lines.append(f'    <text x="1112" y="{y + 5:.1f}" text-anchor="start">{format_count(download_tick)}</text>')
    lines.extend(
        [
            f'    <text x="100" y="221" fill="{STARS_COLOR}" font-size="14" font-weight="700">NEW STARS'
            + (f'<tspan fill="{USED_BY_COLOR}"> · NEW USED BY</tspan>' if show_used_by else "")
            + "</text>",
            f'    <text x="1100" y="221" fill="{DOWNLOADS_COLOR}" font-size="14" font-weight="700" '
            'text-anchor="end">WEEKLY DOWNLOADS</text>',
            "  </g>",
        ]
    )
    return lines


def _render_marks(
    display: Sequence[WeeklyMetric],
    used_by: Sequence[int | None],
    star_min: int,
    star_max: int,
    download_max: int,
) -> list[str]:
    """Render the star bar, the used-by bar and the download point of each displayed week, plus the week labels.

    The two bars share the left axis and sit side by side. While no displayed week has a used-by change the
    star bar stays centred on its week, as before used-by tracking.

    A week with no known star delta (``new_stars is None``, a checkpoint gap or the chain's first
    entry) is a baseline used only to compute the next week's delta: it draws neither a star bar nor
    a download point, so both series visually start at the same first fully-known week.

    Args:
        display: Weeks shown in the chart, oldest first.
        used_by: Weekly used-by change of each displayed week, from `used_by_changes`; None where unknown.
        star_min: Lowest weekly change the left axis reaches, as used by `_render_axes`.
        star_max: Highest weekly change the left axis reaches, as used by `_render_axes`.
        download_max: Highest weekly download total the right axis reaches.

    Returns:
        SVG lines for the bars, the download line, and the week labels, or empty placeholder groups
        and a waiting notice when no week has been recorded yet.
    """
    if not display:
        return [
            '  <g id="weekly-stars-bars"/>',
            '  <g id="weekly-downloads-line"/>',
            '  <text x="600" y="395" fill="#898781" font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="18" text-anchor="middle">Metrics will appear after first scheduled update.</text>',
        ]
    spacing = PLOT_WIDTH / len(display)
    paired = any(change is not None for change in used_by)
    bar_width = min(22.0, spacing * 0.25) if paired else min(44.0, spacing * 0.5)
    zero_y = PLOT_BOTTOM - ((0 - star_min) / (star_max - star_min)) * PLOT_HEIGHT

    def bar(center: float, value: int, color: str) -> str:
        """Draw one bar from the zero line to ``value`` on the shared left axis."""
        value_y = PLOT_BOTTOM - ((value - star_min) / (star_max - star_min)) * PLOT_HEIGHT
        return (
            f'    <rect x="{center - bar_width / 2:.1f}" y="{min(value_y, zero_y):.1f}" width="{bar_width:.1f}" '
            f'height="{max(abs(zero_y - value_y), 1.0):.1f}" rx="4" fill="{color}"/>'
        )

    points = []
    lines = ['  <g id="weekly-stars-bars">']
    for index, metric in enumerate(display):
        x = PLOT_LEFT + (index + 0.5) * spacing
        if metric.new_stars is not None:
            star_x = x - bar_width / 2 - 1 if paired else x
            lines.append(bar(star_x, metric.new_stars, STARS_COLOR if metric.new_stars >= 0 else "#e85d75"))
            download_y = PLOT_BOTTOM - (metric.downloads / download_max) * PLOT_HEIGHT
            points.append(f"{x:.1f},{download_y:.1f}")
        change = used_by[index]
        if change is not None:
            lines.append(bar(x + bar_width / 2 + 1, change, USED_BY_COLOR if change >= 0 else "#e85d75"))
    lines.append("  </g>")
    lines.append('  <g id="weekly-downloads-line">')
    if len(points) > 1:
        lines.append(
            f'    <polyline points="{" ".join(points)}" fill="none" stroke="{DOWNLOADS_COLOR}" stroke-width="4" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    for point in points:
        point_x, point_y = point.split(",")
        lines.append(
            f'    <circle cx="{point_x}" cy="{point_y}" r="5" fill="{DOWNLOADS_COLOR}" '
            'stroke="#ffffff" stroke-width="2"/>'
        )
    lines.append("  </g>")
    lines.append(
        '  <g fill="#6b6a66" font-family="DejaVu Sans, Arial, sans-serif" font-size="12" text-anchor="middle">'
    )
    for index, metric in enumerate(display):
        x = PLOT_LEFT + (index + 0.5) * spacing
        label = metric.week_end.strftime("%b %d").replace(" 0", " ")
        lines.append(f'    <text x="{x:.1f}" y="552">{label}</text>')
    lines.append("  </g>")
    return lines


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
    # Weeks opening a checkpoint gap carry no delta, and a flat range would divide by zero below.
    used_by = used_by_changes(state.history)[len(state.history) - len(display) :]
    star_values = [metric.new_stars for metric in display if metric.new_stars is not None]
    star_values = [*star_values, *(change for change in used_by if change is not None)] or [0]
    star_min = min(0, min(star_values))
    star_max = max(0, max(star_values))
    if star_min == star_max:
        star_max = star_min + 1
    # A baseline week draws no download mark either (see _render_marks), so its downloads must not
    # stretch the axis scale for marks that are never shown.
    plotted_downloads = [metric.downloads for metric in display if metric.new_stars is not None]
    download_max = max(plotted_downloads or [1])
    download_max = max(download_max, 1)

    lines = _render_summary_cards(metadata, display[-1] if display else None)
    paired = any(change is not None for change in used_by)
    lines.extend(_render_axes(star_min, star_max, download_max, paired))
    lines.extend(_render_marks(display, used_by, star_min, star_max, download_max))
    # The used-by entry only appears once a bar for it is drawn; the other two keep their original place until then.
    shift = 0 if not paired else -105
    lines.append('  <g font-family="DejaVu Sans, Arial, sans-serif" font-size="13" fill="#52514e">')
    lines.append(f'    <rect x="{396 + shift}" y="593" width="18" height="12" rx="2" fill="{STARS_COLOR}"/>')
    lines.append(f'    <text x="{424 + shift}" y="604">New GitHub stars</text>')
    if paired:
        lines.append(
            f'    <rect x="{396 + shift + 150}" y="593" width="18" height="12" rx="2" fill="{USED_BY_COLOR}"/>'
        )
        lines.append(f'    <text x="{424 + shift + 150}" y="604">New used-by repos</text>')
        shift += 150
    lines.extend(
        [
            f'    <path d="M{582 + shift} 599H{610 + shift}" fill="none" stroke="{DOWNLOADS_COLOR}" stroke-width="4" '
            'stroke-linecap="round"/>',
            f'    <circle cx="{596 + shift}" cy="599" r="4" fill="{DOWNLOADS_COLOR}"/>',
            f'    <text x="{620 + shift}" y="604">PyPI downloads</text>',
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
    used_by: int | None = None,
) -> MetricsState:
    """Append one completed week, resetting star baseline after any checkpoint gap.

    Rerunning an already recorded week refreshes only its download total and used-by count. ``stars_total``
    is a cumulative reading taken at run time, so a later reading for the same week already counts stars
    earned during the in-progress week; keeping the first reading holds each week's growth attribution where
    it belongs. ``used_by`` is a plain current reading with no delta, so the latest one wins, and a rerun
    that could not read it keeps the value already recorded.

    Args:
        state: Existing embedded metrics state.
        start: First date in completed week.
        end: Last date in completed week.
        stars_total: Current GitHub stargazer count. Ignored when this week is already recorded.
        downloads: PyPI downloads during completed week.
        history_limit: Maximum number of weekly records retained.
        used_by: Dependent-repository count read at run time, or None when it could not be read.

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
    replaced = history[-1] if history and history[-1].week_start == start else None
    if replaced is not None:
        history = history[:-1]
        recorded_stars, new_stars = replaced.stars_total, replaced.new_stars
        if used_by is None:
            used_by = replaced.used_by
    elif history and start == history[-1].week_end + timedelta(days=1):
        recorded_stars, new_stars = stars_total, stars_total - history[-1].stars_total
    else:
        recorded_stars, new_stars = stars_total, None
    metric = WeeklyMetric(
        week_start=start,
        week_end=end,
        stars_total=recorded_stars,
        new_stars=new_stars,
        downloads=downloads,
        used_by=used_by,
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
