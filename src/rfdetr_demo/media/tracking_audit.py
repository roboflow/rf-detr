# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Confidential interval audit for center-person tracking diagnostics."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from rfdetr_demo.inference.models import build_keypoint_model
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
from rfdetr_demo.inference.temporal_filter import KeypointTemporalFilter, MotionPlausibilitySettings
from rfdetr_demo.inference.video_io import probe_video_size
from rfdetr_demo.paths import CONFIDENTIAL_AUDIT, REPO_ROOT, resolve_default_source
from rfdetr_demo.tracking.detection_stabilizer import (
    TrackDiagnostic,
    detection_bbox,
    detection_confidence,
    nms_detection_indices,
)
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings

logger = logging.getLogger(__name__)

_TRACKING_AUDIT_ROOT = CONFIDENTIAL_AUDIT / "tracking-runs"
_TRACKING_AUDIT_JSONL = CONFIDENTIAL_AUDIT / "tracking-audit.jsonl"

# Center lane for mn1-2.mov (~1280px wide); scaled if frame width differs.
DEFAULT_CENTER_X_FRACTION = (0.28, 0.48)


@dataclass(frozen=True)
class RawDetectionDiagnostic:
    """One raw model detection before stabilization."""

    index: int
    cx: float
    cy: float
    confidence: float
    in_center_lane: bool


@dataclass
class TrackingFrameRecord:
    """Metrics for one processed video frame."""

    frame_index: int
    processed_count: int
    raw_count: int
    nms_count: int
    active_count: int
    ghost_count: int
    center_present_raw: bool
    center_present_stabilized: bool
    center_track_id: int | None
    center_is_ghost: bool
    center_confidence: float | None
    image_saved: bool
    image_relpath: str | None = None
    raw_detections: list[RawDetectionDiagnostic] = field(default_factory=list)
    stabilized_tracks: list[TrackDiagnostic] = field(default_factory=list)


@dataclass
class TrackingAuditSummary:
    """Run-level analysis written under confidential/audit."""

    run_id: str
    source_relpath: str
    frames_processed: int
    frames_logged: int
    images_saved: int
    run_dir_relpath: str
    evaluation: dict[str, Any]
    records: list[TrackingFrameRecord] = field(default_factory=list)


def _relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def center_x_range(frame_width: int) -> tuple[float, float]:
    """Return pixel x-range treated as the center person lane."""
    x_min_frac, x_max_frac = DEFAULT_CENTER_X_FRACTION
    return x_min_frac * frame_width, x_max_frac * frame_width


def _in_center_lane(cx: float, frame_width: int) -> bool:
    x_min, x_max = center_x_range(frame_width)
    return x_min <= cx <= x_max


def _raw_diagnostics(key_points: Any, frame_width: int) -> list[RawDetectionDiagnostic]:
    rows: list[RawDetectionDiagnostic] = []
    for index in range(len(key_points)):
        box = detection_bbox(key_points, index)
        if box is None:
            continue
        cx = float((box[0] + box[2]) / 2.0)
        cy = float((box[1] + box[3]) / 2.0)
        rows.append(
            RawDetectionDiagnostic(
                index=index,
                cx=cx,
                cy=cy,
                confidence=detection_confidence(key_points, index),
                in_center_lane=_in_center_lane(cx, frame_width),
            ),
        )
    return rows


def _find_center_track(
    diagnostics: list[TrackDiagnostic],
    frame_width: int,
) -> TrackDiagnostic | None:
    x_min, x_max = center_x_range(frame_width)
    candidates = [row for row in diagnostics if x_min <= row.cx <= x_max]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(row.cx - (x_min + x_max) / 2.0))


def _annotate_track_labels(
    annotated_bgr: npt.NDArray[np.uint8],
    diagnostics: list[TrackDiagnostic],
) -> npt.NDArray[np.uint8]:
    output = annotated_bgr.copy()
    for track in diagnostics:
        label = f"T{track.track_id}"
        if track.is_ghost:
            label += f" ghost m{track.missed}"
        color = (0, 180, 255) if track.is_ghost else (0, 255, 0)
        cv2.circle(output, (int(track.cx), int(track.cy)), 6, color, 2)
        cv2.putText(
            output,
            label,
            (int(track.cx) + 8, int(track.cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def evaluate_center_tracking(records: list[TrackingFrameRecord]) -> dict[str, Any]:
    """Analyze when the center person is lost or only held as ghost."""
    if not records:
        return {"verdict": "データなし", "issues": ["フレーム記録がありません"]}

    total = len(records)
    center_missing_stabilized = [
        row for row in records if not row.center_present_stabilized
    ]
    center_missing_raw = [row for row in records if not row.center_present_raw]
    center_ghost_only = [
        row
        for row in records
        if row.center_present_stabilized and row.center_is_ghost
    ]
    track_id_changes: list[dict[str, int | None]] = []
    prev_track_id: int | None = None
    for row in records:
        if row.center_track_id is None:
            prev_track_id = None
            continue
        if prev_track_id is not None and row.center_track_id != prev_track_id:
            track_id_changes.append(
                {
                    "frame_index": row.frame_index,
                    "from_track_id": prev_track_id,
                    "to_track_id": row.center_track_id,
                },
            )
        prev_track_id = row.center_track_id

    first_loss = center_missing_stabilized[0].frame_index if center_missing_stabilized else None
    issues: list[str] = []
    notes: list[str] = []

    if center_missing_stabilized:
        issues.append(
            f"中央人物なし（安定化後）: {len(center_missing_stabilized)}/{total} フレーム"
            f"（初回 frame_index={first_loss}）",
        )
    if center_missing_raw:
        notes.append(
            f"生検出で中央欠落: {len(center_missing_raw)}/{total} フレーム",
        )
    if center_ghost_only:
        notes.append(
            f"中央人物がゴーストのみ: {len(center_ghost_only)}/{total} フレーム",
        )
    if track_id_changes:
        issues.append(f"中央トラック ID 切替: {len(track_id_changes)} 回")

    if not center_missing_stabilized and not track_id_changes:
        verdict = "OK - 中央トラックは全フレームで維持"
    elif center_missing_stabilized and len(center_missing_stabilized) <= total * 0.05:
        verdict = "概ね OK - 中央トラックの短時間欠落あり"
    else:
        verdict = "要確認 - 中央トラックが途中で失われています"

    highlights = [
        {
            "frame_index": row.frame_index,
            "processed_count": row.processed_count,
            "raw": row.raw_count,
            "active": row.active_count,
            "center_track_id": row.center_track_id,
            "center_is_ghost": row.center_is_ghost,
            "center_present_stabilized": row.center_present_stabilized,
            "image_relpath": Path(row.image_relpath).name if row.image_relpath else None,
        }
        for row in records
        if (
            not row.center_present_stabilized
            or row.center_is_ghost
            or row.frame_index % 20 == 0
        )
    ][:30]

    return {
        "verdict": verdict,
        "issues": issues,
        "notes": notes,
        "frames_total": total,
        "center_missing_stabilized_count": len(center_missing_stabilized),
        "center_missing_raw_count": len(center_missing_raw),
        "center_ghost_only_count": len(center_ghost_only),
        "track_id_change_count": len(track_id_changes),
        "first_center_loss_frame_index": first_loss,
        "track_id_changes": track_id_changes[:20],
        "frame_highlights": highlights,
    }


def run_center_tracking_audit(
    *,
    source_path: Path | None = None,
    sample_interval: int = 20,
    threshold: float = 0.6,
    keypoint_threshold: float = 0.25,
    max_frames: int | None = None,
    save_on_center_event: bool = True,
    run_id: str | None = None,
) -> TrackingAuditSummary:
    """Process a video and log confidential tracking diagnostics every ``sample_interval`` frames."""
    source = (source_path or resolve_default_source()).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")

    frame_width, frame_height, fps = probe_video_size(source)
    x_lane = center_x_range(frame_width)

    audit_run_id = run_id or (
        datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    )
    run_dir = _TRACKING_AUDIT_ROOT / audit_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_keypoint_model()
    motion = MotionPlausibilitySettings(enabled=True)
    temporal_filter = KeypointTemporalFilter(
        motion,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        frame_stride=1,
    )
    track_pipeline = PersonTrackPipeline.from_env(
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_filter=temporal_filter,
        base_settings=PersonTrackSettings(enabled=True),
    )
    overlay_settings = KeypointOverlaySettings(
        keypoint_threshold=keypoint_threshold,
        uncertainty_enabled=True,
        uncertainty_style="heatmap",
        frame_width=frame_width,
        frame_height=frame_height,
        motion=motion,
    )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    records: list[TrackingFrameRecord] = []
    images_saved = 0
    processed_count = 0
    prev_center_present: bool | None = None
    frame_index = 0

    try:
        while True:
            if max_frames is not None and processed_count >= max_frames:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                break

            processed_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_key_points = model.predict(
                frame_rgb,
                threshold=threshold,
                include_source_image=False,
            )
            raw_rows = _raw_diagnostics(raw_key_points, frame_width)
            nms_keep = nms_detection_indices(raw_key_points, track_pipeline.settings.nms_iou_threshold)

            stabilized = track_pipeline.apply(raw_key_points, frame_index)
            key_points = stabilized.key_points

            center_raw = any(row.in_center_lane for row in raw_rows)
            center_track = _find_center_track(stabilized.diagnostics, frame_width)
            center_present = center_track is not None
            center_is_ghost = bool(center_track and center_track.is_ghost)

            center_changed = (
                prev_center_present is not None and center_present != prev_center_present
            )
            should_save = (
                frame_index % sample_interval == 0
                or (save_on_center_event and center_changed)
                or not center_present
            )

            image_relpath: str | None = None
            if should_save:
                annotated = render_keypoint_overlay(frame_bgr, key_points, overlay_settings)
                annotated = _annotate_track_labels(annotated, stabilized.diagnostics)
                cv2.rectangle(
                    annotated,
                    (int(x_lane[0]), 0),
                    (int(x_lane[1]), frame_height - 1),
                    (255, 128, 0),
                    1,
                )
                cv2.putText(
                    annotated,
                    f"frame={frame_index} raw={len(raw_key_points)} active={len(key_points)}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                image_name = f"frame_{processed_count:04d}_src{frame_index:05d}.jpg"
                image_path = run_dir / image_name
                if cv2.imwrite(str(image_path), annotated):
                    image_relpath = _relpath(image_path)
                    images_saved += 1

            record = TrackingFrameRecord(
                frame_index=frame_index,
                processed_count=processed_count,
                raw_count=len(raw_key_points),
                nms_count=len(nms_keep),
                active_count=len(stabilized.key_points),
                ghost_count=stabilized.stats.ghost_count,
                center_present_raw=center_raw,
                center_present_stabilized=center_present,
                center_track_id=center_track.track_id if center_track else None,
                center_is_ghost=center_is_ghost,
                center_confidence=center_track.confidence if center_track else None,
                image_saved=image_relpath is not None,
                image_relpath=image_relpath,
                raw_detections=raw_rows,
                stabilized_tracks=list(stabilized.diagnostics),
            )
            records.append(record)

            audit_line = {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "classification": "CONFIDENTIAL",
                "run_id": audit_run_id,
                "source_relpath": _relpath(source),
                "frame_index": frame_index,
                "processed_count": processed_count,
                "raw_count": record.raw_count,
                "nms_count": record.nms_count,
                "active_count": record.active_count,
                "ghost_count": record.ghost_count,
                "center_present_raw": center_raw,
                "center_present_stabilized": center_present,
                "center_track_id": record.center_track_id,
                "center_is_ghost": center_is_ghost,
                "center_lane_x": [x_lane[0], x_lane[1]],
                "image_relpath": image_relpath,
                "raw_centroids_x": [round(row.cx, 1) for row in raw_rows],
                "track_diagnostics": [asdict(row) for row in stabilized.diagnostics],
            }
            CONFIDENTIAL_AUDIT.mkdir(parents=True, exist_ok=True)
            with _TRACKING_AUDIT_JSONL.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit_line, ensure_ascii=False) + "\n")

            prev_center_present = center_present
            frame_index += 1
    finally:
        capture.release()

    evaluation = evaluate_center_tracking(records)
    summary = TrackingAuditSummary(
        run_id=audit_run_id,
        source_relpath=_relpath(source),
        frames_processed=len(records),
        frames_logged=len(records),
        images_saved=images_saved,
        run_dir_relpath=_relpath(run_dir),
        evaluation=evaluation,
        records=records,
    )

    summary_path = run_dir / "summary.json"
    payload = {
        "classification": "CONFIDENTIAL",
        "run_id": summary.run_id,
        "source_relpath": summary.source_relpath,
        "sample_interval": sample_interval,
        "threshold": threshold,
        "center_lane_x_fraction": list(DEFAULT_CENTER_X_FRACTION),
        "frames_processed": summary.frames_processed,
        "images_saved": summary.images_saved,
        "evaluation": evaluation,
        "frames": [
            {
                "frame_index": row.frame_index,
                "processed_count": row.processed_count,
                "raw_count": row.raw_count,
                "nms_count": row.nms_count,
                "active_count": row.active_count,
                "ghost_count": row.ghost_count,
                "center_present_raw": row.center_present_raw,
                "center_present_stabilized": row.center_present_stabilized,
                "center_track_id": row.center_track_id,
                "center_is_ghost": row.center_is_ghost,
                "center_confidence": row.center_confidence,
                "image_relpath": row.image_relpath,
                "raw_centroids_x": [round(det.cx, 1) for det in row.raw_detections],
                "track_ids": [track.track_id for track in row.stabilized_tracks],
            }
            for row in records
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "tracking_audit_complete run_id=%s frames=%s images=%s verdict=%s",
        audit_run_id,
        summary.frames_processed,
        summary.images_saved,
        evaluation.get("verdict"),
    )
    return summary
