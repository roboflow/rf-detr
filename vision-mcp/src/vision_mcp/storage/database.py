# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""SQLite access: one writer task, one reader connection, WAL mode.

Every write in the process goes through `Database.write`, which hands the statement to a single writer task
consuming a bounded queue. Reads use a separate connection and run in a worker thread so no coroutine ever
blocks on disk.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.logging_setup import get_logger
from vision_mcp.storage.schema import MIGRATIONS

logger = get_logger("vision-mcp.db")

Params = Sequence[Any]
Statement = tuple[str, Params]


@dataclass(slots=True)
class WriteOp:
    """A batch of statements applied in one transaction."""

    statements: list[Statement]
    done: asyncio.Future[None] | None = None


@dataclass(slots=True)
class WriterStats:
    """Counters exposed through worker status."""

    applied: int = 0
    dropped: int = 0
    failed: int = 0
    queue_capacity: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class Database:
    """Owns both SQLite connections and the writer task."""

    def __init__(self, path: Path, queue_size: int = 4096) -> None:
        self._path = path
        self._queue: asyncio.Queue[WriteOp | None] = asyncio.Queue(maxsize=queue_size)
        self._write_conn: sqlite3.Connection | None = None
        self._read_conn: sqlite3.Connection | None = None
        self._read_lock = threading.Lock()
        self._writer_task: asyncio.Task[None] | None = None
        self._pending_submits: set[asyncio.Task[None]] = set()
        self.stats = WriterStats(queue_capacity=queue_size)

    @property
    def ok(self) -> bool:
        """True once both connections are open."""
        return self._write_conn is not None and self._read_conn is not None

    @property
    def queue_depth(self) -> int:
        """Pending write batches."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Open connections, apply migrations and start the writer task."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_conn = await asyncio.to_thread(self._connect)
        self._read_conn = await asyncio.to_thread(self._connect)
        await asyncio.to_thread(self._migrate, self._write_conn)
        self._writer_task = asyncio.create_task(self._writer_loop(), name="db-writer")
        logger.info("database ready", extra={"path": str(self._path)})

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = int(row["v"]) if row and row["v"] is not None else 0
        for index, statements in enumerate(MIGRATIONS[current:], start=current + 1):
            for statement in statements:
                conn.execute(statement)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (index,))
            logger.info("applied migration", extra={"version": index})
        conn.commit()

    async def stop(self) -> None:
        """Drain the queue, stop the writer and close both connections."""
        if self._writer_task is not None:
            if self._pending_submits:
                await asyncio.gather(*tuple(self._pending_submits), return_exceptions=True)
            await self._queue.put(None)
            await self._writer_task
            self._writer_task = None
        for conn in (self._write_conn, self._read_conn):
            if conn is not None:
                await asyncio.to_thread(conn.close)
        self._write_conn = None
        self._read_conn = None
        logger.info("database closed")

    def submit(self, statements: list[Statement]) -> None:
        """Queue a write without waiting, spilling to an async put when full."""
        try:
            self._queue.put_nowait(WriteOp(statements=statements))
        except asyncio.QueueFull:
            task = asyncio.create_task(self._enqueue(statements), name="db-write-backpressure")
            self._pending_submits.add(task)
            task.add_done_callback(self._pending_submits.discard)
            logger.warning("write queue full; preserving batch", extra={"statements": len(statements)})

    async def _enqueue(self, statements: list[Statement]) -> None:
        """Wait for queue space so historical data is never silently discarded."""
        await self._queue.put(WriteOp(statements=statements))

    async def write(self, statements: list[Statement]) -> None:
        """Queue a write and wait for it to be applied.

        Raises:
            VisionError: DATABASE_UNAVAILABLE when the write fails.
        """
        if not self.ok:
            raise VisionError(ErrorCode.DATABASE_UNAVAILABLE, "Database is not open.")
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(WriteOp(statements=statements, done=done))
        await done

    async def fetch_all(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        """Run a read query in a worker thread."""
        return await asyncio.to_thread(self._fetch_all, sql, params)

    async def fetch_one(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        """Run a read query returning at most one row."""
        rows = await self.fetch_all(sql, params)
        return rows[0] if rows else None

    def _fetch_all(self, sql: str, params: Params) -> list[sqlite3.Row]:
        conn = self._read_conn
        if conn is None:
            raise VisionError(ErrorCode.DATABASE_UNAVAILABLE, "Database is not open.")
        try:
            with self._read_lock:
                return list(conn.execute(sql, params).fetchall())
        except sqlite3.Error as exc:
            logger.exception("read failed")
            raise VisionError(ErrorCode.DATABASE_UNAVAILABLE, f"Database read failed: {exc}") from exc

    async def _writer_loop(self) -> None:
        """Single writer.

        Applies one batch per transaction until it receives the stop sentinel.
        """
        while True:
            op = await self._queue.get()
            if op is None:
                return
            try:
                await asyncio.to_thread(self._apply, op.statements)
            except sqlite3.Error as exc:  # process boundary: a bad row must not kill the writer
                with self.stats.lock:
                    self.stats.failed += 1
                logger.exception("write failed")
                if op.done is not None and not op.done.done():
                    op.done.set_exception(
                        VisionError(ErrorCode.DATABASE_UNAVAILABLE, f"Database write failed: {exc}")
                    )
                continue
            with self.stats.lock:
                self.stats.applied += 1
            if op.done is not None and not op.done.done():
                op.done.set_result(None)

    def _apply(self, statements: list[Statement]) -> None:
        conn = self._write_conn
        if conn is None:
            raise sqlite3.OperationalError("database is closed")
        with conn:
            for sql, params in statements:
                conn.execute(sql, params)
