# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compare person tracking with appearance ReID off vs on over one clip.

The same detections are fed through two tracking pipelines that differ only in
``reid_enabled``, so any change in track-id stability is attributable to ReID.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import supervision as sv

from rfdetr_demo.paths import resolve_default_source
from rfdetr_demo.tracking.keypoints_ops import is_track_ghost, track_ids_from_key_points
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunSummary:
    """Track-id stability metrics for one pipeline variant over a clip."""

    frames: int
    unique_ids: int
    id_starts: int
    mean_active: float
    count_std: float
    max_active: int
    min_active: int

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def summarize_track_ids(per_frame_ids: list[list[int]]) -> RunSummary:
    """Reduce a per-frame list of track ids to stability metrics.

    Args:
        per_frame_ids: One list of active track ids per processed frame.

    Returns:
        Aggregate metrics; fewer ``unique_ids`` and ``id_starts`` for the same
        frames and similar ``mean_active`` indicate more stable identities.
    """
    frames = len(per_frame_ids)
    if frames == 0:
        return RunSummary(0, 0, 0, 0.0, 0.0, 0, 0)

    seen: set[int] = set()
    id_starts = 0
    previous: set[int] = set()
    counts: list[int] = []
    for ids in per_frame_ids:
        current = {track_id for track_id in ids if track_id is not None}
        id_starts += len(current - previous)
        seen |= current
        counts.append(len(current))
        previous = current

    counts_array = np.asarray(counts, dtype=np.float64)
    return RunSummary(
        frames=frames,
        unique_ids=len(seen),
        id_starts=id_starts,
        mean_active=float(counts_array.mean()),
        count_std=float(counts_array.std()),
        max_active=int(counts_array.max()),
        min_active=int(counts_array.min()),
    )


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic, well-spread BGR color for a track id."""
    hue = (track_id * 47) % 180
    import cv2

    pixel = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_tracks(frame_bgr: np.ndarray, key_points: sv.KeyPoints) -> np.ndarray:
    """Draw per-track boxes, joints, and id labels colored consistently by id."""
    import cv2

    annotated = frame_bgr.copy()
    track_ids = track_ids_from_key_points(key_points)
    boxes = key_points.data.get("xyxy") if key_points.data else None
    for index, track_id in enumerate(track_ids):
        if track_id is None:
            continue
        color = color_for_id(track_id)
        ghost = is_track_ghost(key_points, index)
        thickness = 1 if ghost else 2
        if boxes is not None and index < len(boxes):
            x1, y1, x2, y2 = (int(round(float(value))) for value in boxes[index])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"{track_id}*" if ghost else str(track_id)
            cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        xy = key_points.xy[index]
        visible = key_points.visible[index] if key_points.visible is not None else None
        for joint_index in range(len(xy)):
            if visible is not None and not visible[joint_index]:
                continue
            px, py = int(round(float(xy[joint_index, 0]))), int(round(float(xy[joint_index, 1])))
            if px <= 0 and py <= 0:
                continue
            cv2.circle(annotated, (px, py), 2, color, -1)
    return annotated


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``compare-reid`` subcommand."""
    parser = subparsers.add_parser(
        "compare-reid",
        help="Compare tracking with appearance ReID off vs on over one clip",
    )
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (lower = more recall)")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Keypoint model input resolution (higher = better small-person recall; must divide the patch size)",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames")
    parser.add_argument("--reid-weight", type=float, default=0.3, help="Appearance vs IoU cost blend")
    parser.add_argument("--reid-similarity", type=float, default=0.5, help="Gallery revival threshold")
    parser.add_argument("--reid-gallery-frames", type=int, default=60, help="Gallery retention window")
    parser.add_argument(
        "--reid-backend",
        choices=["histogram", "embedding"],
        default="histogram",
        help="Appearance descriptor: histogram (color, no deps) or embedding (ONNX ReID, grayscale-robust)",
    )
    parser.add_argument("--reid-model", type=Path, default=None, help="ONNX ReID model path (embedding backend)")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write metrics JSON")
    parser.add_argument(
        "--write-video",
        action="store_true",
        help="Write id-labeled reid_off.mp4 / reid_on.mp4 for visual comparison",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for --write-video output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.set_defaults(_handler=run)


def _iter_frames(source: Path, *, frame_stride: int, max_frames: int | None):
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    try:
        emitted = 0
        index = -1
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            index += 1
            if index % max(1, frame_stride) != 0:
                continue
            if max_frames is not None and emitted >= max_frames:
                break
            emitted += 1
            yield index, frame_bgr
    finally:
        capture.release()


def _collect_ids(
    source: Path,
    *,
    threshold: float,
    frame_stride: int,
    max_frames: int | None,
    reid_settings: PersonTrackSettings,
    baseline_settings: PersonTrackSettings,
    video_dir: Path | None = None,
    resolution: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Run one model, feed each frame through both pipelines, and collect ids.

    When ``video_dir`` is given, also writes id-labeled ``reid_off.mp4`` and
    ``reid_on.mp4`` there for visual comparison.
    """
    import cv2

    from rfdetr_demo.inference.models import build_keypoint_model

    model = build_keypoint_model(resolution=resolution)
    probe = cv2.VideoCapture(str(source))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    probe.release()

    baseline_pipeline = PersonTrackPipeline(settings=baseline_settings, frame_width=width, frame_height=height)
    reid_pipeline = PersonTrackPipeline(settings=reid_settings, frame_width=width, frame_height=height)

    baseline_writer = None
    reid_writer = None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        out_fps = max(1.0, fps / max(1, frame_stride))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        baseline_writer = cv2.VideoWriter(str(video_dir / "reid_off.mp4"), fourcc, out_fps, (width, height))
        reid_writer = cv2.VideoWriter(str(video_dir / "reid_on.mp4"), fourcc, out_fps, (width, height))

    baseline_ids: list[list[int]] = []
    reid_ids: list[list[int]] = []
    try:
        for index, frame_bgr in _iter_frames(source, frame_stride=frame_stride, max_frames=max_frames):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            key_points = model.predict(frame_rgb, threshold=threshold, include_source_image=False)
            baseline_result = baseline_pipeline.apply(key_points, index, frame_bgr)
            reid_result = reid_pipeline.apply(key_points, index, frame_bgr)
            baseline_ids.append(
                [tid for tid in track_ids_from_key_points(baseline_result.key_points) if tid is not None],
            )
            reid_ids.append([tid for tid in track_ids_from_key_points(reid_result.key_points) if tid is not None])
            if baseline_writer is not None and reid_writer is not None:
                baseline_writer.write(_draw_tracks(frame_bgr, baseline_result.key_points))
                reid_writer.write(_draw_tracks(frame_bgr, reid_result.key_points))
    finally:
        if baseline_writer is not None:
            baseline_writer.release()
        if reid_writer is not None:
            reid_writer.release()
    return baseline_ids, reid_ids


def _print_comparison(baseline: RunSummary, reid: RunSummary) -> None:
    rows = [
        ("frames", baseline.frames, reid.frames),
        ("unique track ids", baseline.unique_ids, reid.unique_ids),
        ("id starts (churn)", baseline.id_starts, reid.id_starts),
        ("mean active", round(baseline.mean_active, 2), round(reid.mean_active, 2)),
        ("active count std", round(baseline.count_std, 2), round(reid.count_std, 2)),
        ("max active", baseline.max_active, reid.max_active),
    ]
    print(f"{'metric':<20}{'reid off':>12}{'reid on':>12}")
    print("-" * 44)
    for name, off_value, on_value in rows:
        print(f"{name:<20}{off_value!s:>12}{on_value!s:>12}")
    if baseline.unique_ids:
        reduction = 100.0 * (baseline.unique_ids - reid.unique_ids) / baseline.unique_ids
        print("-" * 44)
        print(f"unique-id reduction with ReID: {reduction:.1f}%")


def run(args: argparse.Namespace) -> int:
    """Execute the ReID comparison and print a metrics table."""
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    source = args.source or resolve_default_source()

    baseline_settings = PersonTrackSettings(reid_enabled=False)
    reid_settings = dataclasses.replace(
        baseline_settings,
        reid_enabled=True,
        reid_weight=args.reid_weight,
        reid_similarity_threshold=args.reid_similarity,
        reid_max_gallery_frames=args.reid_gallery_frames,
        reid_backend=args.reid_backend,
        reid_model_path=str(args.reid_model) if args.reid_model is not None else None,
    )

    logger.info("Comparing ReID off vs on over %s", source)
    video_dir = args.out_dir if args.write_video else None
    baseline_ids, reid_ids = _collect_ids(
        source,
        threshold=args.threshold,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        reid_settings=reid_settings,
        baseline_settings=baseline_settings,
        video_dir=video_dir,
        resolution=args.resolution,
    )

    baseline_summary = summarize_track_ids(baseline_ids)
    reid_summary = summarize_track_ids(reid_ids)
    _print_comparison(baseline_summary, reid_summary)

    if video_dir is not None:
        print(f"\nWrote videos: {video_dir / 'reid_off.mp4'} and {video_dir / 'reid_on.mp4'} (id* = ghost hold)")

    if args.json is not None:
        payload = {
            "source": str(source),
            "detection": {"threshold": args.threshold, "resolution": args.resolution},
            "reid_off": baseline_summary.as_dict(),
            "reid_on": reid_summary.as_dict(),
            "reid_settings": {
                "reid_backend": args.reid_backend,
                "reid_weight": args.reid_weight,
                "reid_similarity_threshold": args.reid_similarity,
                "reid_max_gallery_frames": args.reid_gallery_frames,
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote metrics: {args.json}")
    return 0
