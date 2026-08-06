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

from rfdetr_demo.paths import resolve_default_source
from rfdetr_demo.tracking.keypoints_ops import track_ids_from_key_points
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


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``compare-reid`` subcommand."""
    parser = subparsers.add_parser(
        "compare-reid",
        help="Compare tracking with appearance ReID off vs on over one clip",
    )
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames")
    parser.add_argument("--reid-weight", type=float, default=0.3, help="Appearance vs IoU cost blend")
    parser.add_argument("--reid-similarity", type=float, default=0.5, help="Gallery revival threshold")
    parser.add_argument("--reid-gallery-frames", type=int, default=60, help="Gallery retention window")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write metrics JSON")
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
) -> tuple[list[list[int]], list[list[int]]]:
    """Run one model, feed each frame through both pipelines, and collect ids."""
    import cv2

    from rfdetr_demo.inference.models import build_keypoint_model

    model = build_keypoint_model()
    probe = cv2.VideoCapture(str(source))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    probe.release()

    baseline_pipeline = PersonTrackPipeline(settings=baseline_settings, frame_width=width, frame_height=height)
    reid_pipeline = PersonTrackPipeline(settings=reid_settings, frame_width=width, frame_height=height)

    baseline_ids: list[list[int]] = []
    reid_ids: list[list[int]] = []
    for index, frame_bgr in _iter_frames(source, frame_stride=frame_stride, max_frames=max_frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        key_points = model.predict(frame_rgb, threshold=threshold, include_source_image=False)
        baseline_result = baseline_pipeline.apply(key_points, index, frame_bgr)
        reid_result = reid_pipeline.apply(key_points, index, frame_bgr)
        baseline_ids.append([tid for tid in track_ids_from_key_points(baseline_result.key_points) if tid is not None])
        reid_ids.append([tid for tid in track_ids_from_key_points(reid_result.key_points) if tid is not None])
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
    )

    logger.info("Comparing ReID off vs on over %s", source)
    baseline_ids, reid_ids = _collect_ids(
        source,
        threshold=args.threshold,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        reid_settings=reid_settings,
        baseline_settings=baseline_settings,
    )

    baseline_summary = summarize_track_ids(baseline_ids)
    reid_summary = summarize_track_ids(reid_ids)
    _print_comparison(baseline_summary, reid_summary)

    if args.json is not None:
        payload = {
            "source": str(source),
            "reid_off": baseline_summary.as_dict(),
            "reid_on": reid_summary.as_dict(),
            "reid_settings": {
                "reid_weight": args.reid_weight,
                "reid_similarity_threshold": args.reid_similarity,
                "reid_max_gallery_frames": args.reid_gallery_frames,
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote metrics: {args.json}")
    return 0
