# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Ownership of every configured stream.

The manager is the only thing that starts and stops `StreamRuntime` objects. Sources are never accepted from a caller: a
stream can only be started by the id it was given in the config file, so no tool call can point the engine at a new URL
or path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

from vision_mcp.api_contract import LiveSnapshot, StreamStatus, StreamSummary
from vision_mcp.config import EngineConfig
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.inference.detector import Detector
from vision_mcp.logging_setup import get_logger
from vision_mcp.streams.runtime import StreamObserver, StreamRuntime

logger = get_logger("vision-mcp.streams")

ObserverFactory = Callable[[str], StreamObserver]
"""Builds the analytics chain for a stream id.

Supplied by the engine.
"""


class StreamManager:
    """Starts, stops and reports on the streams named in the config."""

    def __init__(
        self,
        config: EngineConfig,
        detector: Detector,
        observer_factory: ObserverFactory | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._observer_factory = observer_factory
        self._lock = asyncio.Lock()
        self._runtimes: dict[str, StreamRuntime] = {
            stream_id: StreamRuntime(
                stream_id=stream_id,
                entry=entry,
                config=config,
                detector=detector,
                observer=None if observer_factory is None else observer_factory(stream_id),
            )
            for stream_id, entry in config.streams.items()
        }

    def __iter__(self) -> Iterator[StreamRuntime]:
        """Iterate every configured runtime, running or not."""
        return iter(self._runtimes.values())

    def get(self, stream_id: str) -> StreamRuntime:
        """Look up a stream.

        Raises:
            VisionError: STREAM_NOT_FOUND when the id is not in the config.
        """
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            raise VisionError(
                ErrorCode.STREAM_NOT_FOUND,
                f"Unknown stream {stream_id!r}.",
                {"stream_id": stream_id, "known": sorted(self._runtimes)},
            )
        return runtime

    async def start_all(self) -> None:
        """Start every configured stream, tolerating individual failures."""

        async with self._lock:
            for runtime in self._runtimes.values():
                try:
                    await runtime.start()
                except Exception:  # one bad source must not stop the engine
                    logger.exception("stream failed to start", extra={"stream_id": runtime.stream_id})

    async def stop_all(self) -> None:
        """Stop every stream and release every capture source."""

        async with self._lock:
            await asyncio.gather(
                *(runtime.stop() for runtime in self._runtimes.values()),
                return_exceptions=True,
            )

    async def start(self, stream_id: str) -> StreamStatus:
        """Start one configured stream.

        Starting a running stream is a no-op.
        """
        runtime = self.get(stream_id)
        async with self._lock:
            await runtime.start()
        return runtime.status()

    async def stop(self, stream_id: str) -> StreamStatus:
        """Stop one stream without forgetting it; it can be started again."""
        runtime = self.get(stream_id)
        async with self._lock:
            await runtime.stop()
        return runtime.status()

    def summaries(self) -> list[StreamSummary]:
        """One row per configured stream, in config order."""
        return [runtime.summary() for runtime in self._runtimes.values()]

    def status(self, stream_id: str) -> StreamStatus:
        """Full status for one stream."""
        return self.get(stream_id).status()

    def snapshot(self, stream_id: str) -> LiveSnapshot:
        """Cheap live snapshot for one stream."""
        return self.get(stream_id).snapshot()

    @property
    def running_count(self) -> int:
        """How many streams currently have a live worker."""
        return sum(1 for runtime in self._runtimes.values() if runtime.running)
