# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Sample a clip and recommend a FlowCount preset (overhead vs eye-level)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from rfdetr_demo.inference.models import build_keypoint_model
from rfdetr_demo.paths import REPO_ROOT, resolve_default_source
from rfdetr_demo.tracking.viewpoint import estimate_camera_viewpoint, preset_for_viewpoint


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``probe-viewpoint`` subcommand."""
    parser = subparsers.add_parser(
        "probe-viewpoint",
        help="Sample a clip and recommend a FlowCount preset (overhead / eye-level)",
    )
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection threshold")
    parser.add_argument("--source", type=Path, default=None, help="Video source path")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "viewpoint_probe.json",
        help="Output JSON path",
    )
    parser.set_defaults(_handler=run)


def run(args: argparse.Namespace) -> int:
    """Sample frames, estimate camera viewpoint, and print the recommended preset."""
    source = args.source if args.source is not None else resolve_default_source()
    if not source.is_file():
        print(f"Error: video not found: {source}")
        print("Place mn1-2.mov under rf-detr/confidential/media/input/ or pass --source")
        return 1

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"Error: cannot open video: {source}")
        return 1

    frame_height = max(1, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    model = build_keypoint_model()
    boxes: list[tuple[float, float, float, float]] = []

    for frame_index in range(args.frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        key_points = model.predict(rgb, threshold=args.threshold, include_source_image=False)
        if isinstance(key_points, list) or not key_points.data:
            continue
        xyxy = key_points.data.get("xyxy")
        if xyxy is None:
            continue
        for x1, y1, x2, y2 in xyxy:
            boxes.append((float(x1), float(y1), float(x2), float(y2)))

    capture.release()

    estimate = estimate_camera_viewpoint(boxes, frame_height=float(frame_height))
    preset = preset_for_viewpoint(estimate)

    summary = {
        "source": str(source),
        "frames_sampled": args.frames,
        "frame_height": frame_height,
        "sample_count": estimate.sample_count,
        "viewpoint": estimate.viewpoint,
        "confidence": estimate.confidence,
        "median_aspect_ratio": round(estimate.median_aspect_ratio, 3),
        "size_position_correlation": round(estimate.size_position_correlation, 3),
        "recommended_preset": preset,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0
