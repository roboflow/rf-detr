# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Command-line packing of COCO splits into WebDataset shards.

Run ``python -m rfdetr.cli.webdataset --help`` for arguments. Reusable packing and index handling live in
:mod:`rfdetr.datasets.webdataset`.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from rfdetr.datasets.webdataset.index import DEFAULT_MAX_SHARD_BYTES, index_name
from rfdetr.datasets.webdataset.pack import pack_coco_to_shards


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the packing entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m rfdetr.cli.webdataset",
        description="Pack a COCO-format split into WebDataset tar shards for sequential-I/O training.",
    )
    parser.add_argument("--image-dir", required=True, help="Directory holding the split's image files.")
    parser.add_argument("--annotations", required=True, help="COCO-format JSON annotation file for the split.")
    parser.add_argument("--output-dir", required=True, help="Directory to write shards and the index into.")
    parser.add_argument("--split", default="train", help="Split name used for shard and index file names.")
    parser.add_argument(
        "--max-shard-mb",
        type=float,
        default=DEFAULT_MAX_SHARD_BYTES / (1024 * 1024),
        help="Approximate shard size in MB.",
    )
    parser.add_argument(
        "--category-ids",
        choices=("remap", "raw"),
        default="remap",
        help="'remap' assigns contiguous 0-based labels; 'raw' keeps the source category_id values.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Pack one COCO split into shards from the command line.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = _build_arg_parser().parse_args(argv)
    index = pack_coco_to_shards(
        args.image_dir,
        args.annotations,
        args.output_dir,
        split=args.split,
        max_shard_bytes=int(args.max_shard_mb * 1024 * 1024),
        category_ids=args.category_ids,
    )
    print(
        f"{index.num_samples} samples -> {len(index.shards)} shard(s) in "
        f"{os.fspath(Path(args.output_dir))} ({index_name(index.split)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
