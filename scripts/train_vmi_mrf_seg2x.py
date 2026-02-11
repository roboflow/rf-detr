# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Train RF-DETR segmentation on vmi_mrf with step-based validation callbacks."""

import argparse
from pathlib import Path

from rfdetr import RFDETRSeg2XLarge
from rfdetr.util.logger import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR-Seg-2XL on vmi_mrf")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to Roboflow-style COCO dataset containing train/valid/test folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/vmi_mrf/rfdetr_seg_2xl",
        help="Directory where checkpoints and metrics are saved",
    )
    parser.add_argument("--resolution", type=int, default=768, help="Square train/val resolution")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--val-interval-steps",
        type=int,
        default=200,
        help="Run validation callback every N global training steps",
    )
    parser.add_argument("--run-test", action="store_true", help="Run test split evaluation at train end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution % 12 != 0:
        raise ValueError(
            f"resolution must be divisible by 12 for RFDETRSeg2XLarge, got {args.resolution}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting vmi_mrf segmentation training with RFDETRSeg2XLarge at fixed resolution "
        f"{args.resolution} (no multiscale)."
    )

    model = RFDETRSeg2XLarge(resolution=args.resolution)
    model.train(
        dataset_file="roboflow",
        dataset_dir=args.dataset_dir,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        checkpoint_interval=args.checkpoint_interval,
        val_interval_steps=args.val_interval_steps,
        run_test=args.run_test,
        # Keep train and validation at the same resolution for this first experiment.
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
    )


if __name__ == "__main__":
    main()
