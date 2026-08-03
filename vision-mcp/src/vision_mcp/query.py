"""Time-window grammar shared by every historical query."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from vision_mcp.errors import ErrorCode, VisionError

#: Refuse to build more buckets than this; protects both memory and response size.
MAX_BUCKETS = 500
MAX_WINDOW_SECONDS = 7 * 24 * 3600
MIN_INTERVAL_SECONDS = 1

_DURATION = re.compile(r"^\s*(?P<value>\d+)\s*(?P<unit>[smhd])\s*$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
SUPPORTED_WINDOWS = frozenset({"5m", "15m", "1h", "6h", "24h", "7d"})
SUPPORTED_INTERVALS = frozenset({"1s", "10s", "1m", "5m", "1h", "1d"})


def parse_duration(text: str, field: str) -> int:
    """Parse ``<int><s|m|h|d>`` into seconds.

    Raises:
        VisionError: INVALID_TIME_WINDOW when the string is malformed or zero.
    """
    match = _DURATION.match(text or "")
    if not match:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            f"{field} must look like '30s', '5m', '2h' or '1d'.",
            {"field": field, "value": text},
        )
    seconds = int(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
    if seconds <= 0:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            f"{field} must be greater than zero.",
            {"field": field, "value": text},
        )
    return seconds


@dataclass(frozen=True)
class TimeQuery:
    """A validated window, resolved to absolute epoch bounds and a bucket width."""

    start: float
    end: float
    window_seconds: int
    interval_seconds: int
    bucket_count: int

    @property
    def bucket_starts(self) -> list[float]:
        """Left edge of every bucket, oldest first."""
        return [self.start + i * self.interval_seconds for i in range(self.bucket_count)]


def build_time_query(window: str, interval: str | None = None, now: float | None = None) -> TimeQuery:
    """Validate a window/interval pair and resolve it against the current clock.

    Raises:
        VisionError: INVALID_TIME_WINDOW when the window is too long, the interval exceeds
            the window, or the pair would produce more than ``MAX_BUCKETS`` buckets.
    """
    end = time.time() if now is None else now
    if window not in SUPPORTED_WINDOWS:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            f"time_window must be one of {sorted(SUPPORTED_WINDOWS)}.",
            {"time_window": window},
        )
    if interval is not None and interval not in SUPPORTED_INTERVALS:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            f"interval must be one of {sorted(SUPPORTED_INTERVALS)}.",
            {"interval": interval},
        )
    window_seconds = parse_duration(window, "window")
    if window_seconds > MAX_WINDOW_SECONDS:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            "window exceeds the maximum supported range of 7d.",
            {"window": window, "max_window": "7d"},
        )
    interval_seconds = parse_duration(interval, "interval") if interval else window_seconds
    if interval_seconds < MIN_INTERVAL_SECONDS:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW, "interval must be at least 1s.", {"interval": interval}
        )
    if interval_seconds > window_seconds:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            "interval must not be larger than window.",
            {"window": window, "interval": interval},
        )
    bucket_count = -(-window_seconds // interval_seconds)  # ceil
    if bucket_count > MAX_BUCKETS:
        raise VisionError(
            ErrorCode.INVALID_TIME_WINDOW,
            f"window/interval would produce {bucket_count} buckets; the limit is {MAX_BUCKETS}.",
            {
                "window": window,
                "interval": interval,
                "bucket_count": bucket_count,
                "max_buckets": MAX_BUCKETS,
            },
        )
    return TimeQuery(
        start=end - window_seconds,
        end=end,
        window_seconds=window_seconds,
        interval_seconds=interval_seconds,
        bucket_count=bucket_count,
    )
