# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Probe per-frame keypoint person counts (raw / NMS / stabilize)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from rfdetr_demo.inference.models import build_keypoint_model
from rfdetr_demo.paths import REPO_ROOT, resolve_default_source
from rfdetr_demo.tracking.bbox import nms_detection_indices
from rfdetr_demo.tracking.detection_stabilizer import DetectionStabilizer
from rfdetr_demo.tracking.keypoints_ops import subset_key_points
from rfdetr_demo.tracking.types import PersonTrackSettings


def _centroids(key_points: object) -> list[dict[str, float]]:
    xyxy = key_points.data.get("xyxy") if key_points.data else None
    centroids: list[dict[str, float]] = []
    if xyxy is None:
        return centroids
    for box in xyxy:
        x1, y1, x2, y2 = box
        centroids.append(
            {
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "area": float((x2 - x1) * (y2 - y1)),
            },
        )
    centroids.sort(key=lambda centroid: centroid["x"])
    return centroids


def _row_from_keypoints(
    frame_index: int,
    key_points: object,
    *,
    raw_n: int | None = None,
    nms_n: int | None = None,
    ghost_n: int = 0,
) -> dict[str, object]:
    detection_confidence = key_points.detection_confidence
    confidences = (
        [float(value) for value in detection_confidence]
        if detection_confidence is not None
        else []
    )
    centroids = _centroids(key_points)
    row: dict[str, object] = {
        "frame": frame_index,
        "n": len(key_points),
        "confs": [round(confidence, 3) for confidence in sorted(confidences, reverse=True)],
        "centroids_x": [round(centroid["x"], 1) for centroid in centroids],
        "centroids_area": [round(centroid["area"], 0) for centroid in centroids],
    }
    if raw_n is not None:
        row["raw_n"] = raw_n
    if nms_n is not None:
        row["nms_n"] = nms_n
    if ghost_n:
        row["ghost_n"] = ghost_n
    return row


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``probe-count`` subcommand."""
    parser = subparsers.add_parser(
        "probe-count",
        help="Probe per-frame person counts (raw / nms / stabilize)",
    )
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to probe")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection threshold")
    parser.add_argument("--source", type=Path, default=None, help="Video source path")
    parser.add_argument(
        "--mode",
        choices=("raw", "nms", "stabilize"),
        default="stabilize",
        help="raw=predict only, nms=IoU-NMS, stabilize=NMS+track hold",
    )
    parser.add_argument("--nms-iou", type=float, default=0.50, help="IoU threshold for NMS")
    parser.add_argument("--max-missed", type=int, default=2, help="Track hold frames")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "person_count_probe.json",
        help="Output JSON path",
    )
    parser.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute probe-count and write JSON summary."""
    source = args.source if args.source is not None else resolve_default_source()
    capture = cv2.VideoCapture(str(source))
    model = build_keypoint_model()
    stabilizer = DetectionStabilizer(
        settings=PersonTrackSettings(
            enabled=True,
            nms_iou_threshold=args.nms_iou,
            max_missed=args.max_missed,
        ),
    )
    rows: list[dict[str, object]] = []

    for frame_index in range(args.frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key_points = model.predict(rgb, threshold=args.threshold, include_source_image=False)
        raw_n = len(key_points)

        if args.mode == "raw":
            rows.append(_row_from_keypoints(frame_index, key_points, raw_n=raw_n))
            continue

        if args.mode == "nms":
            keep = nms_detection_indices(key_points, args.nms_iou)
            filtered = subset_key_points(key_points, keep)
            rows.append(_row_from_keypoints(frame_index, filtered, raw_n=raw_n, nms_n=len(filtered)))
            continue

        result = stabilizer.apply(key_points, frame_index)
        rows.append(
            _row_from_keypoints(
                frame_index,
                result.key_points,
                raw_n=result.stats.raw_count,
                nms_n=result.stats.nms_count,
                ghost_n=result.stats.ghost_count,
            ),
        )

    capture.release()

    summary = {
        "source": str(source),
        "mode": args.mode,
        "threshold": args.threshold,
        "frames": len(rows),
        "n_min": min(int(row["n"]) for row in rows) if rows else 0,
        "n_max": max(int(row["n"]) for row in rows) if rows else 0,
        "n_values": [int(row["n"]) for row in rows],
    }
    payload = {"summary": summary, "frames": rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        extra = ""
        if "raw_n" in row:
            extra = f" raw={row['raw_n']}"
        if "nms_n" in row:
            extra += f" nms={row['nms_n']}"
        if row.get("ghost_n"):
            extra += f" ghost={row['ghost_n']}"
        print(
            f"frame {row['frame']:2d} n={row['n']}{extra} xs={row['centroids_x']}",
        )
    return 0
