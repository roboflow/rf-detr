# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model lifecycle: lazy load once, keep warm, run every inference on the model's own thread.

RF-DETR modules are not thread-safe and MPS dislikes concurrent submission from many threads, so each model owns a
single-thread executor and all of its work is serialised through that thread.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from vision_mcp.api_contract import ModelInfo, ModelList, ModelStatus, ModelSummary
from vision_mcp.clock import monotonic, utc_iso_or_none
from vision_mcp.config import ARCHITECTURES, EngineConfig, ModelEntry, Task
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.inference.device import fallback_device, resolve_device
from vision_mcp.logging_setup import get_logger
from vision_mcp.security import redact, validate_local_path

logger = get_logger("vision-mcp.models")


@dataclass(slots=True)
class LoadedModel:
    """A warm model and the counters that prove it stayed warm."""

    name: str
    entry: ModelEntry
    device: str
    model: Any
    class_names: list[str]
    resolution: int | None
    loaded_at: float
    load_seconds: float
    executor: ThreadPoolExecutor
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_inference_at: float | None = None
    queue_depth: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def mean_latency_ms(self) -> float | None:
        """Mean inference latency since load, or None before the first inference."""
        if self.inference_count == 0:
            return None
        return self.total_latency_ms / self.inference_count


class ModelManager:
    """Owns every model instance in the engine."""

    def __init__(self, config: EngineConfig) -> None:
        self._entries = config.models
        self._security = config.security
        self._loaded: dict[str, LoadedModel] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}

    def entry(self, name: str) -> ModelEntry:
        """Configured entry for *name*.

        Raises:
            VisionError: MODEL_NOT_FOUND when the model is not in the config.
        """
        try:
            return self._entries[name]
        except KeyError:
            raise VisionError(
                ErrorCode.MODEL_NOT_FOUND,
                f"No model named {name!r} is configured.",
                {"available": sorted(self._entries)},
            ) from None

    def is_loaded(self, name: str) -> bool:
        """Whether the model currently holds weights in memory."""
        return name in self._loaded

    def summary(self, name: str) -> ModelSummary:
        """Static description of one configured model."""
        entry = self.entry(name)
        return ModelSummary(
            name=name,
            architecture=entry.architecture,
            task=entry.task,
            device=entry.device,
            confidence=entry.confidence,
            checkpoint=entry.checkpoint,
            loaded=self.is_loaded(name),
        )

    def list_models(self) -> ModelList:
        """Every configured model, loaded or not."""
        return ModelList(models=[self.summary(name) for name in self._entries])

    def info(self, name: str) -> ModelInfo:
        """Static description plus the label set, which is only known once weights are loaded."""
        loaded = self._loaded.get(name)
        return ModelInfo(
            model=self.summary(name),
            resolution=loaded.resolution if loaded else self.entry(name).effective_resolution,
            class_names=list(loaded.class_names) if loaded else None,
            class_count=len(loaded.class_names) if loaded else None,
        )

    def status(self, name: str) -> ModelStatus:
        """Runtime counters for one model."""
        self.entry(name)
        loaded = self._loaded.get(name)
        if loaded is None:
            return ModelStatus(
                name=name,
                loaded=False,
                device=None,
                loaded_at=None,
                load_seconds=None,
                inference_count=0,
                last_inference_at=None,
                mean_latency_ms=None,
                queue_depth=0,
            )
        return ModelStatus(
            name=name,
            loaded=True,
            device=loaded.device,
            loaded_at=utc_iso_or_none(loaded.loaded_at),
            load_seconds=round(loaded.load_seconds, 3),
            inference_count=loaded.inference_count,
            last_inference_at=utc_iso_or_none(loaded.last_inference_at),
            mean_latency_ms=None if loaded.mean_latency_ms is None else round(loaded.mean_latency_ms, 2),
            queue_depth=loaded.queue_depth,
        )

    def statuses(self) -> list[ModelStatus]:
        """Runtime counters for every configured model."""
        return [self.status(name) for name in self._entries]

    def require_task(self, name: str, task: Task) -> ModelEntry:
        """Assert that *name* serves *task*.

        Raises:
            VisionError: MODEL_TASK_MISMATCH when the configured task differs.
        """
        entry = self.entry(name)
        if entry.task != task:
            raise VisionError(
                ErrorCode.MODEL_TASK_MISMATCH,
                f"Model {name!r} serves {entry.task!r}, not {task!r}.",
                {"model": name, "configured_task": entry.task, "required_task": task},
            )
        return entry

    async def acquire(self, name: str) -> LoadedModel:
        """Return the warm model, loading it on first use.

        Concurrent callers for the same model wait on one load; different models load in parallel.
        """
        loaded = self._loaded.get(name)
        if loaded is not None:
            return loaded
        entry = self.entry(name)
        lock = self._load_locks.setdefault(name, asyncio.Lock())
        async with lock:
            loaded = self._loaded.get(name)
            if loaded is not None:
                return loaded
            loaded = await self._load(name, entry)
            self._loaded[name] = loaded
            return loaded

    async def preload(self, names: list[str]) -> None:
        """Warm models at startup so the first tool call is not the one that pays for weights."""
        for name in dict.fromkeys(names):
            try:
                await self.acquire(name)
            except VisionError as exc:
                logger.warning("preload failed", extra={"model": name, "error": exc.message})

    async def run(self, name: str, call: Any) -> tuple[Any, float]:
        """Execute ``call(loaded_model)`` on the model's thread and record latency.

        Returns:
            The call result and the elapsed inference time in milliseconds.
        """
        loaded = await self.acquire(name)
        loaded.queue_depth += 1
        try:
            async with loaded.lock:
                started = monotonic()
                try:
                    result = await asyncio.get_running_loop().run_in_executor(loaded.executor, call, loaded)
                except VisionError:
                    raise
                except Exception as exc:
                    if loaded.device == "mps" and loaded.inference_count == 0:
                        result = await self._retry_first_inference_on_cpu(name, loaded, call, exc)
                    else:
                        logger.exception("inference failed", extra={"model": name})
                        raise VisionError(
                            ErrorCode.INFERENCE_FAILED,
                            "Model inference failed.",
                            {"model": name, "error": redact(exc)},
                        ) from exc
                elapsed_ms = (monotonic() - started) * 1000.0
        finally:
            loaded.queue_depth -= 1
        loaded.inference_count += 1
        loaded.total_latency_ms += elapsed_ms
        loaded.last_inference_at = time.time()
        return result, elapsed_ms

    async def _retry_first_inference_on_cpu(
        self, name: str, loaded: LoadedModel, call: Any, failure: Exception
    ) -> Any:
        """Rebuild after a first-inference MPS failure and retry exactly once on CPU."""
        logger.warning(
            "first MPS inference failed; rebuilding model on CPU",
            extra={"model": name, "error": redact(failure)},
        )
        loop = asyncio.get_running_loop()
        started = monotonic()
        model = await loop.run_in_executor(loaded.executor, self._build, loaded.entry, "cpu")
        loaded.model = model
        loaded.device = "cpu"
        loaded.loaded_at = time.time()
        loaded.load_seconds = monotonic() - started
        loaded.class_names = list(model.class_names)
        try:
            result = await loop.run_in_executor(loaded.executor, call, loaded)
        except Exception as exc:
            logger.exception("CPU retry failed", extra={"model": name})
            raise VisionError(
                ErrorCode.INFERENCE_FAILED,
                "Model inference failed after MPS-to-CPU fallback.",
                {"model": name, "error": redact(exc)},
            ) from exc
        return result

    async def shutdown(self) -> None:
        """Drop every model and stop its thread."""
        for loaded in list(self._loaded.values()):
            loaded.executor.shutdown(wait=True, cancel_futures=True)
        self._loaded.clear()

    async def _load(self, name: str, entry: ModelEntry) -> LoadedModel:
        """Build one model on its own thread, falling back to CPU when the accelerator refuses."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"model-{name}")
        loop = asyncio.get_running_loop()
        device = resolve_device(entry.device)
        started = monotonic()
        while True:
            try:
                model = await loop.run_in_executor(executor, self._build, entry, device)
                break
            except VisionError:
                executor.shutdown(wait=False)
                raise
            except Exception as exc:
                retry = fallback_device(device)
                if retry is None:
                    executor.shutdown(wait=False)
                    logger.exception("model load failed", extra={"model": name})
                    raise VisionError(
                        ErrorCode.MODEL_NOT_LOADED,
                        f"Could not load model {name!r}.",
                        {"model": name, "device": device, "error": redact(exc)},
                    ) from exc
                logger.warning(
                    "model load failed; retrying on CPU",
                    extra={"model": name, "device": device, "error": redact(exc)},
                )
                device = retry
        load_seconds = monotonic() - started
        class_names = list(model.class_names)
        logger.info(
            "model loaded",
            extra={
                "model": name,
                "architecture": entry.architecture,
                "device": device,
                "load_seconds": round(load_seconds, 2),
                "classes": len(class_names),
            },
        )
        return LoadedModel(
            name=name,
            entry=entry,
            device=device,
            model=model,
            class_names=class_names,
            resolution=entry.effective_resolution,
            loaded_at=time.time(),
            load_seconds=load_seconds,
            executor=executor,
        )

    def _build(self, entry: ModelEntry, device: str) -> Any:
        """Construct the RF-DETR wrapper.

        Runs on the model thread; imports torch lazily.
        """
        import rfdetr

        if entry.architecture not in ARCHITECTURES:  # pragma: no cover - config validation covers this
            raise VisionError(ErrorCode.INVALID_ARGUMENT, f"Unknown architecture {entry.architecture!r}.")
        factory = getattr(rfdetr, entry.architecture)
        kwargs: dict[str, Any] = {"device": device}
        if entry.resolution is not None:
            kwargs["resolution"] = entry.resolution
        if entry.checkpoint is not None:
            checkpoint = validate_local_path(self._security.filesystem_roots, entry.checkpoint)
            kwargs["pretrain_weights"] = str(checkpoint)
        return factory(**kwargs)
