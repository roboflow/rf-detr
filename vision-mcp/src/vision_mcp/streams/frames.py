# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The bounded hand-off between a capture thread and its inference worker.

The queue is deliberately tiny (config `queue_size`, default 2). A live camera produces frames whether or not
anyone is ready for them, so the only two honest options are to drop frames or to grow latency without limit.
This drops, and counts every drop.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_mcp.api_contract import ImageSize


@dataclass(slots=True)
class Frame:
    """One captured frame, already downscaled to the size inference will run on."""

    array: np.ndarray[Any, Any]
    size: ImageSize
    index: int
    captured_at: float
    captured_monotonic: float


class LatestFrameQueue:
    """A drop-oldest queue sized in frames, safe to push to from a plain thread.

    `push` is called from the capture thread and never blocks or raises. `get` is awaited by the stream worker
    on the event loop. The two are bridged with a mutex plus an `asyncio.Event` scheduled onto the loop,
    because `asyncio.Queue` is not thread-safe.
    """

    def __init__(self, capacity: int, loop: asyncio.AbstractEventLoop) -> None:
        if capacity < 1:
            raise ValueError("queue capacity must be at least 1")
        self._capacity = capacity
        self._loop = loop
        self._mutex = threading.Lock()
        self._items: list[Frame] = []
        self._ready = asyncio.Event()
        self._closed = False
        self._dropped = 0
        self._high_water = 0

    @property
    def capacity(self) -> int:
        """Maximum frames held before the oldest is dropped."""
        return self._capacity

    @property
    def depth(self) -> int:
        """Frames waiting right now."""
        with self._mutex:
            return len(self._items)

    @property
    def dropped(self) -> int:
        """Frames discarded because the worker could not keep up."""
        with self._mutex:
            return self._dropped

    @property
    def high_water(self) -> int:
        """Deepest the queue has ever been; proves the bound holds."""
        with self._mutex:
            return self._high_water

    def push(self, frame: Frame) -> bool:
        """Offer a frame from the capture thread.

        Returns False if a frame was dropped.
        """
        with self._mutex:
            if self._closed:
                return False
            dropped = False
            while len(self._items) >= self._capacity:
                self._items.pop(0)
                self._dropped += 1
                dropped = True
            self._items.append(frame)
            self._high_water = max(self._high_water, len(self._items))
        self._signal()
        return not dropped

    async def get(self) -> Frame | None:
        """Await the next frame.

        Returns None once the queue is closed and drained.
        """
        while True:
            with self._mutex:
                if self._items:
                    frame = self._items.pop(0)
                    if not self._items:
                        self._ready.clear()
                    return frame
                if self._closed:
                    return None
                self._ready.clear()
            await self._ready.wait()

    def close(self) -> None:
        """Wake any waiter and refuse further pushes."""
        with self._mutex:
            self._closed = True
        self._signal()

    def _signal(self) -> None:
        """Wake the worker from whichever thread we are on, tolerating a stopped loop."""
        with contextlib.suppress(RuntimeError):  # loop already closed during shutdown
            self._loop.call_soon_threadsafe(self._ready.set)
