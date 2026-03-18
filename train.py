"""
RF-DETR Training Script

Usage:
    # Standard COCO (train split)
    python train.py --dataset_dir ~/Datasets/coco --dataset_file coco

    # COCO using val split as training data (when train set is too large)
    python train.py --dataset_dir ~/Datasets/coco --dataset_file coco --train_split val

    # Keypoint detection (COCO person keypoints, val split)
    python train.py --dataset_dir ~/Datasets/coco --dataset_file coco --model kpt-base \
        --num_keypoints 17 --train_split val

    # Roboflow / custom COCO-format dataset
    python train.py --dataset_dir /path/to/dataset --model large --epochs 50

Dataset formats supported:
    - coco      : dataset_dir/train2017/ + dataset_dir/annotations/instances_*.json
    - roboflow  : dataset_dir/train/_annotations.coco.json  (default)
    - yolo      : dataset_dir/data.yaml + train/val/test folders
"""

import argparse
import os
import sys

MODEL_MAP = {
    "nano":       ("RFDETRNano",      False, False),
    "small":      ("RFDETRSmall",     False, False),
    "medium":     ("RFDETRMedium",    False, False),
    "base":       ("RFDETRBase",      False, False),
    "large":      ("RFDETRLarge",     False, False),
    "seg-nano":   ("RFDETRSegNano",   True,  False),
    "seg-small":  ("RFDETRSegSmall",  True,  False),
    "seg-medium": ("RFDETRSegMedium", True,  False),
    "seg-large":  ("RFDETRSegLarge",  True,  False),
    "kpt-nano":   ("RFDETRKptNano",   False, True),
    "kpt-small":  ("RFDETRKptSmall",  False, True),
    "kpt-medium": ("RFDETRKptMedium", False, True),
    "kpt-base":   ("RFDETRKptBase",   False, True),
    "kpt-large":  ("RFDETRKptLarge",  False, True),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RF-DETR Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    dataset_group.add_argument(
        "--dataset_file",
        type=str,
        default="roboflow",
        choices=["coco", "roboflow", "yolo"],
        help=(
            "Dataset format. "
            "'coco' expects train2017/val2017 + annotations/ subdirs. "
            "'roboflow' expects train/valid/test + _annotations.coco.json. "
            "'yolo' expects data.yaml."
        ),
    )
    dataset_group.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "val"],
        help=(
            "Which split to use as training data. "
            "Set to 'val' to train on the validation set "
            "(useful when the train set is too large to download)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default="base",
        choices=list(MODEL_MAP.keys()),
        help="Model variant to train",
    )
    model_group.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of object classes. Auto-detected from dataset if not set.",
    )
    model_group.add_argument(
        "--num_keypoints",
        type=int,
        default=17,
        help="Number of keypoints per instance (kpt-* models only)",
    )
    model_group.add_argument(
        "--pretrain_weights",
        type=str,
        default=None,
        help="Path or key for pretrained weights (default: official pretrained)",
    )
    model_group.add_argument(
        "--force_no_pretrain",
        action="store_true",
        default=False,
        help="Train from scratch — disables all pretrained weights even when a model default exists.",
    )

    # ------------------------------------------------------------------ #
    # Training hyperparameters
    # ------------------------------------------------------------------ #
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs",           type=int,   default=100)
    train_group.add_argument("--batch_size",       type=int,   default=4)
    train_group.add_argument("--lr",               type=float, default=1e-4)
    train_group.add_argument("--lr_encoder",       type=float, default=1.5e-4)
    train_group.add_argument("--weight_decay",     type=float, default=1e-4)
    train_group.add_argument("--grad_accum_steps", type=int,   default=4)
    train_group.add_argument("--warmup_epochs",    type=float, default=0.0)

    # ------------------------------------------------------------------ #
    # Output & checkpointing
    # ------------------------------------------------------------------ #
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output_dir",         type=str,   default="output")
    out_group.add_argument("--checkpoint_interval",type=int,   default=10)
    out_group.add_argument("--resume",             type=str,   default=None,
                           help="Path to checkpoint to resume from")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--tensorboard", action="store_true", default=True)
    log_group.add_argument("--wandb",       action="store_true", default=False)
    log_group.add_argument("--project",     type=str, default=None)
    log_group.add_argument("--run",         type=str, default=None)

    # ------------------------------------------------------------------ #
    # Performance
    # ------------------------------------------------------------------ #
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--num_workers", type=int, default=2)
    perf_group.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device override (default: auto-detect)",
    )
    perf_group.add_argument("--seed", type=int, default=42)

    # ------------------------------------------------------------------ #
    # Early stopping
    # ------------------------------------------------------------------ #
    es_group = parser.add_argument_group("Early stopping")
    es_group.add_argument("--early_stopping",         action="store_true", default=False)
    es_group.add_argument("--early_stopping_patience",type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    # Expand ~ in dataset_dir
    args.dataset_dir = os.path.expanduser(args.dataset_dir)

    # Auto-resume: if no explicit --resume, look for last.ckpt in output_dir.
    if args.resume is None:
        last_ckpt = os.path.join(args.output_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            args.resume = last_ckpt

    class_name, is_seg, is_kpt = MODEL_MAP[args.model]

    import rfdetr
    ModelClass = getattr(rfdetr, class_name)

    # ---- Model kwargs ----
    model_kwargs = {}
    if args.num_classes is not None:
        model_kwargs["num_classes"] = args.num_classes
    if args.force_no_pretrain:
        model_kwargs["pretrain_weights"] = None
    elif args.pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = args.pretrain_weights
    if is_kpt:
        model_kwargs["num_keypoints"] = args.num_keypoints

    print("=" * 60)
    print(f"  Model        : {class_name}")
    print(f"  Dataset dir  : {args.dataset_dir}")
    print(f"  Dataset fmt  : {args.dataset_file}")
    print(f"  Train split  : {args.train_split}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    if is_kpt:
        print(f"  Keypoints    : {args.num_keypoints}")
    if args.force_no_pretrain:
        print("  NOTE: Training from scratch (no pretrained weights)")
    if args.train_split == "val":
        print("  NOTE: Training on the val split (train set skipped)")
    if args.resume:
        print(f"  Resuming from  : {args.resume}")
    print("=" * 60)

    model = ModelClass(**model_kwargs)

    # ---- Train kwargs ----
    train_kwargs = dict(
        progress_bar=sys.stdout.isatty(),
        dataset_dir=args.dataset_dir,
        dataset_file=args.dataset_file,
        train_split=args.train_split,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        warmup_epochs=args.warmup_epochs,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        tensorboard=args.tensorboard,
        wandb=args.wandb,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )
    if args.resume is not None:
        train_kwargs["resume"] = args.resume
    if args.project is not None:
        train_kwargs["project"] = args.project
    if args.run is not None:
        train_kwargs["run"] = args.run
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
