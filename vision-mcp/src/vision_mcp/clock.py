"""Time helpers.

Wall clock (`time.time`) is only ever used for timestamps that leave the process; every
duration and latency measurement uses the monotonic clock.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime


def utc_iso(epoch_seconds: float | None = None) -> str:
    """Format an epoch timestamp as UTC ISO-8601 with milliseconds."""
    moment = datetime.fromtimestamp(time.time() if epoch_seconds is None else epoch_seconds, tz=UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_iso_or_none(epoch_seconds: float | None) -> str | None:
    """ISO-8601 for an optional timestamp, preserving ``None``."""
    return None if epoch_seconds is None else utc_iso(epoch_seconds)


def epoch() -> float:
    """Wall-clock epoch seconds, for timestamps stored in SQLite or returned to clients."""
    return time.time()


def monotonic() -> float:
    """Monotonic seconds; safe for durations across clock adjustments."""
    return time.monotonic()
