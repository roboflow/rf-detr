"""Retention cleanup: one periodic task deleting rows and artifact files past their age."""

from __future__ import annotations

import asyncio
import time

from vision_mcp.logging_setup import get_logger
from vision_mcp.storage.artifacts import ArtifactStore
from vision_mcp.storage.database import Database
from vision_mcp.storage.schema import RETENTION_TABLES

logger = get_logger("vision-mcp.retention")


async def purge_older_than(database: Database, artifacts: ArtifactStore, cutoff: float) -> dict[str, int]:
    """Delete every retained row and artifact older than *cutoff*.

    Returns:
        Row counts removed per table, plus an ``artifacts`` entry for unlinked files.
    """
    removed: dict[str, int] = {}
    for table, column in RETENTION_TABLES:
        row = await database.fetch_one(f"SELECT COUNT(*) AS n FROM {table} WHERE {column} < ?", (cutoff,))
        count = int(row["n"]) if row else 0
        if count:
            await database.write([(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))])
        removed[table] = count
    removed["artifacts"] = await artifacts.delete_before(cutoff)
    return removed


async def retention_loop(
    database: Database, artifacts: ArtifactStore, retention_days: int, interval_seconds: float
) -> None:
    """Run cleanup on a fixed interval until cancelled."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            cutoff = time.time() - retention_days * 86400
            removed = await purge_older_than(database, artifacts, cutoff)
            if any(removed.values()):
                logger.info("retention cleanup", extra={"removed": removed, "retention_days": retention_days})
        except asyncio.CancelledError:
            raise
        except Exception:  # process boundary: cleanup must never take the engine down
            logger.exception("retention cleanup failed")
