# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLI for the RF-DETR video demo (``rfdetr-demo`` default command)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rfdetr_demo.inference.runner import run_demo
from rfdetr_demo.inference.types import KeypointUncertaintyStyle, TaskName
from rfdetr_demo.paths import SAMPLE_DANCE, default_output_path, resolve_default_source

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the video demo."""
    parser = argparse.ArgumentParser(
        description="Run RF-DETR on a video and export an annotated MP4.",
    )
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "keypoint"],
        default="detect",
    )
    parser.add_argument("--model", choices=["nano", "small", "medium", "large"], default="nano")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Model input resolution (higher = better small-person recall; detect and keypoint tasks)",
    )
    parser.add_argument("--person-only", action="store_true")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Track person detections (detect task): stable ids + live count via the box-IoU tracker",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable appearance ReID on the tracker (reduces id fragmentation across occlusions)",
    )
    parser.add_argument(
        "--reid-model",
        type=Path,
        default=None,
        help="ONNX ReID model path; enables the grayscale-robust embedding backend (else color histogram)",
    )
    parser.add_argument("--reid-similarity", type=float, default=0.6, help="ReID revival threshold (0..1)")
    parser.add_argument(
        "--reid-stride",
        type=int,
        default=1,
        help="Run the ReID embedding every Nth frame (motion-only in between) to recover speed",
    )
    parser.add_argument(
        "--pose-topk",
        type=int,
        default=0,
        help="Two-stage: also pose-estimate the N largest tracked boxes (detect+track everyone, pose a subset)",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-source-seconds", type=float, default=None)
    parser.add_argument("--keypoint-threshold", type=float, default=0.0)
    parser.add_argument("--keypoint-uncertainty", action="store_true")
    parser.add_argument(
        "--keypoint-uncertainty-style",
        choices=["ellipse", "halo", "heatmap", "magnitude", "outline", "cross", "filled"],
        default="heatmap",
    )
    parser.add_argument("--heatmap-opacity", type=float, default=0.38)
    parser.add_argument("--heatmap-decay", type=float, default=3.0)
    parser.add_argument("--vertex-radius", type=int, default=4)
    parser.add_argument("--ellipse-sigma", type=float, default=1.5)
    parser.add_argument("--max-ellipse-axis", type=float, default=None)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the video demo CLI."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    source_path = args.source if args.source is not None else resolve_default_source()
    task: TaskName = args.task
    keypoint_uncertainty_style: KeypointUncertaintyStyle = (
        args.keypoint_uncertainty_style if args.keypoint_uncertainty else "none"
    )
    output_path = (
        args.output
        if args.output is not None
        else default_output_path(
            source_path,
            task,
            keypoint_uncertainty=keypoint_uncertainty_style != "none",
            keypoint_uncertainty_style=keypoint_uncertainty_style,
        )
    )

    person_only = args.person_only
    if not person_only and source_path.resolve() == SAMPLE_DANCE.resolve() and task in {"detect", "segment"}:
        person_only = True
        logger.info("Defaulting to --person-only for sample/mzoo.mov dance demo")

    try:
        summary = run_demo(
            source_path=source_path,
            target_path=output_path,
            task=task,
            model_size=args.model,
            threshold=args.threshold,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            person_only=person_only,
            keypoint_threshold=args.keypoint_threshold,
            keypoint_uncertainty_style=keypoint_uncertainty_style,
            ellipse_sigma=args.ellipse_sigma,
            max_ellipse_axis=args.max_ellipse_axis,
            progress_file=args.progress_file,
            max_source_seconds=args.max_source_seconds,
            heatmap_opacity=args.heatmap_opacity,
            heatmap_decay=args.heatmap_decay,
            vertex_radius=args.vertex_radius,
            keypoint_uncertainty_enabled=args.keypoint_uncertainty,
            model_resolution=args.resolution,
            detect_track=args.track,
            reid_enabled=args.reid or args.reid_model is not None,
            reid_model=str(args.reid_model) if args.reid_model is not None else None,
            reid_similarity=args.reid_similarity,
            reid_stride=args.reid_stride,
            pose_topk=args.pose_topk,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1

    logger.info("Demo complete: %s", summary)
    print(f"Output video: {summary['target']}")
    print(
        f"Task: {summary['task']}, frames: {summary['processed_frames']}, "
        f"instances: {summary['total_detections']}, elapsed: {summary['elapsed_sec']}s",
    )
    if "avg_fps" in summary:
        print(f"Average inference FPS: {summary['avg_fps']}")
    if "unique_track_ids" in summary:
        print(f"Unique track ids over clip: {summary['unique_track_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
