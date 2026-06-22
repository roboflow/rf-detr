# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CLI for destroying orphaned Vast.ai GPU instances."""

from __future__ import annotations

import argparse
import logging
import sys

from rfdetr_demo.vast.api_config import resolve_vast_api_key
from rfdetr_demo.vast.cli import ensure_vast_cli_or_raise
from rfdetr_demo.vast.safety import VastSafetySettings, cleanup_orphan_instances

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Destroy orphaned rf-detr Vast.ai instances."""
    parser = argparse.ArgumentParser(
        description="Destroy orphaned rf-detr Vast.ai GPU instances.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List orphans without destroying")
    parser.add_argument("--api-key", default=None, help="Override Vast API key")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        ensure_vast_cli_or_raise()
        api_key = resolve_vast_api_key(args.api_key)
        settings = VastSafetySettings()
        destroyed = cleanup_orphan_instances(
            api_key=api_key,
            settings=settings,
        )
    except Exception as error:
        logger.error("%s", error)
        return 1

    action = "Would destroy" if args.dry_run else "Destroyed"
    logger.info("%s %d orphan instance(s)", action, len(destroyed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
