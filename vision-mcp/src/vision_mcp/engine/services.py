"""Engine service graph and M1-M6 operation dispatcher."""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict

from vision_mcp import __version__
from vision_mcp.analytics.aggregator import MetricsAggregator
from vision_mcp.analytics.events import EventSink
from vision_mcp.analytics.observer import StreamAnalytics
from vision_mcp.api_contract import (
    ArtifactResult,
    CompareResult,
    CountsByClass,
    CropResult,
    CurrentCounts,
    DetectionResult,
    GpuMetrics,
    ModelList,
    QueueMetrics,
    StreamList,
    SystemStatus,
    WorkerInfo,
    WorkerList,
)
from vision_mcp.clock import epoch, utc_iso
from vision_mcp.config import EngineConfig
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.inference.detector import Detector, count_by_class
from vision_mcp.inference.images import ImageLoader, crop_array, encode_jpeg
from vision_mcp.inference.models import ModelManager
from vision_mcp.storage.artifacts import ArtifactStore
from vision_mcp.storage.database import Database
from vision_mcp.storage.retention import retention_loop
from vision_mcp.streams.annotate import Annotator
from vision_mcp.streams.manager import StreamManager
from vision_mcp.streams.preview import PreviewService

from .queries import HistoricalQueries
from .support import (
    HISTORICAL_TO_QUERY,
    gpu_metrics,
    limit,
    optional_float,
    optional_string,
    optional_strings,
    period,
    required,
    value,
)


class ToolArguments(BaseModel):
    """Validated open argument bag; individual operations validate their exact fields."""

    model_config = ConfigDict(extra="allow")


class EngineServices:
    """Own all long-lived engine components and their shutdown ordering."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.started_at = epoch()
        self.database = Database(config.storage.database)
        self.artifacts = ArtifactStore(
            config.storage.artifacts, self.database, config.storage.max_artifact_bytes
        )
        self.models = ModelManager(config)
        self.images = ImageLoader(config.security)
        self.detector = Detector(self.models, self.images)
        self.events = EventSink(self.database, self.artifacts)
        self.analytics: dict[str, StreamAnalytics] = {}

        def observer(stream_id: str) -> StreamAnalytics:
            result = StreamAnalytics(stream_id, config.streams[stream_id], config, self.database, self.events)
            self.analytics[stream_id] = result
            return result

        self.streams = StreamManager(config, self.detector, observer)
        self.aggregator = MetricsAggregator(
            self.database,
            self.streams,
            self.analytics,
            config.metrics.aggregation_interval_seconds,
            self.events,
        )
        self.preview = PreviewService(config.debug)
        self.queries = HistoricalQueries(self.database, config)
        self._retention: asyncio.Task[None] | None = None
        self._stopped = False

    async def start(self) -> None:
        """Start storage, artifacts, streams and background maintenance."""
        await self.database.start()
        await asyncio.to_thread(self.artifacts.ensure_root)
        await self.streams.start_all()
        await self.aggregator.start()
        self._retention = asyncio.create_task(
            retention_loop(
                self.database,
                self.artifacts,
                self.config.storage.retention_days,
                self.config.storage.cleanup_interval_seconds,
            ),
            name="retention-cleanup",
        )

    async def stop(self) -> None:
        """Release cameras, flush metrics and writes, stop models, then close SQLite."""
        if self._stopped:
            return
        self._stopped = True
        await self.streams.stop_all()
        await self.aggregator.stop()
        if self._retention is not None:
            self._retention.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._retention
            self._retention = None
        await self.models.shutdown()
        await self.database.stop()

    async def system_status(self) -> SystemStatus:
        """Return engine-wide live status."""
        summaries = self.streams.summaries()
        reasons = [
            f"{item.stream_id}: {item.health}"
            for item in summaries
            if item.health in {"unhealthy", "degraded"}
        ]
        health = "unhealthy" if any(item.health == "unhealthy" for item in summaries) else "healthy"
        if not summaries:
            health = "unknown"
        elif reasons and health == "healthy":
            health = "degraded"
        devices = {status.device for status in self.models.statuses() if status.device}
        return SystemStatus(
            version=__version__,
            started_at=utc_iso(self.started_at),
            uptime_seconds=round(epoch() - self.started_at, 2),
            device=",".join(sorted(devices)) or "not_loaded",
            mps_fallback_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            streams_configured=len(summaries),
            streams_running=self.streams.running_count,
            models_loaded=sum(status.loaded for status in self.models.statuses()),
            database_ok=self.database.ok,
            artifacts_bytes=await self.artifacts.total_bytes(),
            health=health,  # type: ignore[arg-type]
            health_reasons=reasons,
        )

    def workers(self) -> WorkerList:
        """Describe each long-lived task or thread."""
        workers = [
            WorkerInfo(name="db-writer", kind="db_writer", alive=self.database.ok),
            WorkerInfo(name="metrics-aggregator", kind="aggregator", alive=self.aggregator.running),
            WorkerInfo(
                name="retention-cleanup",
                kind="cleanup",
                alive=self._retention is not None and not self._retention.done(),
            ),
        ]
        for runtime in self.streams:
            workers.extend(
                [
                    WorkerInfo(
                        name=f"capture:{runtime.stream_id}",
                        kind="capture",
                        alive=runtime.capture_alive,
                    ),
                    WorkerInfo(name=f"stream:{runtime.stream_id}", kind="stream_loop", alive=runtime.running),
                ]
            )
        for status in self.models.statuses():
            if status.loaded:
                workers.append(WorkerInfo(name=f"model:{status.name}", kind="inference", alive=True))
        return WorkerList(workers=workers, thread_count=len(threading.enumerate()))

    async def call(self, name: str, raw: dict[str, Any]) -> BaseModel:
        """Dispatch one named M1-M6 operation and return a contract model."""
        args = ToolArguments.model_validate(raw)
        if name in HISTORICAL_TO_QUERY:
            stream_id = required(args, "stream_id")
            self.streams.get(stream_id)
            query_period = period(args)
            target = getattr(self.queries, HISTORICAL_TO_QUERY[name])
            historical_result: BaseModel = await target(
                stream_id, query_period.time_window, query_period.interval
            )
            return historical_result
        if name == "get_recent_detection_events":
            query_period = period(args)
            optional_stream = optional_string(args, "stream_id")
            if optional_stream is not None:
                self.streams.get(optional_stream)
            return await self.queries.events(
                optional_stream, query_period.time_window, query_period.interval, limit(args)
            )
        if name == "get_recent_errors":
            query_period = period(args)
            return await self.queries.errors(query_period.time_window, query_period.interval, limit(args))
        method = getattr(self, f"tool_{name}", None)
        if method is None:
            raise VisionError(ErrorCode.INVALID_ARGUMENT, f"Unknown operation {name!r}.")
        tool_result: BaseModel = await method(args)
        return tool_result

    async def tool_list_models(self, args: ToolArguments) -> ModelList:
        """List configured models."""
        return self.models.list_models()

    async def tool_get_model_info(self, args: ToolArguments) -> BaseModel:
        """Return static model information."""
        return self.models.info(required(args, "model"))

    async def tool_get_model_status(self, args: ToolArguments) -> BaseModel:
        """Return warm-load and inference counters."""
        return self.models.status(required(args, "model"))

    async def tool_detect_objects(self, args: ToolArguments) -> DetectionResult:
        """Run detection on one image source."""
        return await self._infer(args, "detection")

    async def tool_segment_instances(self, args: ToolArguments) -> DetectionResult:
        """Run instance segmentation on one image source."""
        return await self._infer(args, "segmentation")

    async def tool_detect_keypoints(self, args: ToolArguments) -> DetectionResult:
        """Run keypoint detection on one image source."""
        return await self._infer(args, "keypoints")

    async def tool_count_objects(self, args: ToolArguments) -> CountsByClass:
        """Count detections in one image."""
        output = await self._source_output(args)
        counts = count_by_class(output.detections)
        return CountsByClass(
            model=required(args, "model"),
            source=output.image.label,
            counts=counts,
            total=sum(counts.values()),
        )

    async def tool_find_objects(self, args: ToolArguments) -> DetectionResult:
        """Detect only requested classes in one image."""
        return await self._infer(args, None)

    async def tool_crop_detections(self, args: ToolArguments) -> CropResult:
        """Persist one generated artifact for each matching detection crop."""
        output = await self._source_output(args)
        refs = []
        for detection in output.detections:
            box = detection.box
            crop = crop_array(output.image.array, box.x1, box.y1, box.x2, box.y2)
            refs.append(await self.artifacts.save("crop", encode_jpeg(crop)))
        return CropResult(model=required(args, "model"), source=output.image.label, crops=refs)

    async def tool_compare_detections(self, args: ToolArguments) -> CompareResult:
        """Compare class counts between two image sources."""
        model = required(args, "model")
        left = await self.detector.detect_source(model, required(args, "left"))
        right = await self.detector.detect_source(model, required(args, "right"))
        left_counts, right_counts = count_by_class(left.detections), count_by_class(right.detections)
        keys = sorted(set(left_counts) | set(right_counts))
        return CompareResult(
            model=model,
            left=CountsByClass(
                model=model, source=left.image.label, counts=left_counts, total=sum(left_counts.values())
            ),
            right=CountsByClass(
                model=model, source=right.image.label, counts=right_counts, total=sum(right_counts.values())
            ),
            delta={key: right_counts.get(key, 0) - left_counts.get(key, 0) for key in keys},
        )

    async def tool_get_system_status(self, args: ToolArguments) -> SystemStatus:
        """Return system status."""
        return await self.system_status()

    async def tool_get_stream_status(self, args: ToolArguments) -> BaseModel:
        """Return one stream's status."""
        return self.streams.status(required(args, "stream_id"))

    async def tool_list_active_streams(self, args: ToolArguments) -> StreamList:
        """List configured streams and current state."""
        return StreamList(streams=self.streams.summaries())

    async def tool_get_worker_status(self, args: ToolArguments) -> WorkerList:
        """Return worker task and thread state."""
        return self.workers()

    async def tool_get_current_counts(self, args: ToolArguments) -> CurrentCounts:
        """Return detections in only the latest processed frame."""
        live = self.streams.snapshot(required(args, "stream_id"))
        return CurrentCounts(
            stream_id=live.stream_id,
            current_objects=live.current_objects,
            counts_by_class=live.counts_by_class,
            frame_at=live.last_frame_at,
        )

    async def tool_get_active_tracks(self, args: ToolArguments) -> BaseModel:
        """Return tracks visible in the latest processed frame."""
        return self._analytics(args).active_tracks()

    async def tool_get_zone_occupancy(self, args: ToolArguments) -> BaseModel:
        """Return current zone occupancy."""
        return self._analytics(args).zone_occupancy()

    async def tool_get_queue_metrics(self, args: ToolArguments) -> QueueMetrics:
        """Return bounded queue state."""
        runtime = self.streams.get(required(args, "stream_id"))
        return QueueMetrics(
            stream_id=runtime.stream_id,
            depth=runtime.queue_depth,
            capacity=runtime.queue_capacity,
            high_water=runtime.queue_high_water,
            dropped_frames=runtime.dropped_frames,
            inference_queue_depth=self.models.status(runtime.entry.model).queue_depth,
        )

    async def tool_get_gpu_metrics(self, args: ToolArguments) -> GpuMetrics:
        """Return NVIDIA metrics or a structured unsupported result."""
        return gpu_metrics()

    async def tool_get_latest_annotated_frame(self, args: ToolArguments) -> ArtifactResult:
        """Annotate the current in-memory frame and persist it on demand."""
        stream_id = required(args, "stream_id")
        jpeg = await self.frame_jpeg(stream_id, True)
        return ArtifactResult(artifact=await self.artifacts.save("frame", jpeg, stream_id=stream_id))

    async def tool_get_event_snapshot(self, args: ToolArguments) -> ArtifactResult:
        """Look up the artifact attached to an event."""
        event_id = required(args, "event_id")
        row = await self.database.fetch_one("SELECT artifact_id FROM events WHERE event_id = ?", (event_id,))
        if row is None or row["artifact_id"] is None:
            raise VisionError(ErrorCode.EVENT_NOT_FOUND, "Event has no snapshot.", {"event_id": event_id})
        return ArtifactResult(artifact=await self.artifacts.get_ref(str(row["artifact_id"])))

    async def frame_jpeg(self, stream_id: str, annotate: bool) -> bytes:
        """Encode the latest already-decoded frame, annotating only when requested."""
        runtime = self.streams.get(stream_id)
        frame = runtime.preview_frame()
        if frame is None:
            raise VisionError(
                ErrorCode.STREAM_DISCONNECTED,
                "No frame has been processed for this stream yet.",
                {"stream_id": stream_id},
            )
        if annotate:
            return await asyncio.to_thread(self.preview.render, runtime, frame)
        return await asyncio.to_thread(encode_jpeg, frame.array)

    async def _infer(self, args: ToolArguments, task: str | None) -> DetectionResult:
        """Shared implementation for the three inference task tools."""
        model = required(args, "model")
        if task is not None:
            self.models.require_task(model, task)  # type: ignore[arg-type]
        output = await self._source_output(args)
        artifact = None
        if bool(value(args, "annotate", False)):
            canvas = await asyncio.to_thread(
                Annotator().render, output.image.array, output.raw, output.detections
            )
            artifact = await self.artifacts.save("frame", encode_jpeg(canvas))
        return DetectionResult(
            model=model,
            task=self.models.entry(model).task,
            source=output.image.label,
            image=output.image.size,
            detections=output.detections,
            current_objects=len(output.detections),
            inference_ms=round(output.inference_ms, 2),
            artifact=artifact,
        )

    async def _source_output(self, args: ToolArguments) -> Any:
        """Load and infer with common optional filters."""
        return await self.detector.detect_source(
            required(args, "model"),
            required(args, "source"),
            confidence=optional_float(args, "confidence"),
            classes=optional_strings(args, "classes"),
        )

    def _analytics(self, args: ToolArguments) -> StreamAnalytics:
        """Validate a stream and return its analytics observer."""
        stream_id = required(args, "stream_id")
        self.streams.get(stream_id)
        return self.analytics[stream_id]
