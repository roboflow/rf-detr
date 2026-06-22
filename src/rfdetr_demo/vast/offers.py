# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Search Vast.ai GPU offers."""

from __future__ import annotations

import logging
from threading import Event

from rfdetr_demo.vast.api_config import resolve_vast_api_key
from rfdetr_demo.vast.cli import parse_json_output, run_vast_cli
from rfdetr_demo.vast.types import VastGpuOffer, VastRunnerCancelledError, VastRunnerError

logger = logging.getLogger(__name__)


def search_gpu_offers(
    *,
    api_key: str | None = None,
    max_dph: float = 0.80,
    gpu_name: str | None = None,
    min_reliability: float = 0.95,
    limit: int = 20,
    cancel_event: Event | None = None,
) -> list[VastGpuOffer]:
    """Search rentable single-GPU offers on Vast.ai sorted by price."""
    try:
        resolved_key = resolve_vast_api_key(api_key)
    except ValueError as error:
        raise VastRunnerError(str(error)) from error
    if cancel_event is not None and cancel_event.is_set():
        raise VastRunnerCancelledError("Cancelled before searching offers.")

    query_parts = [
        "rentable=true",
        "verified=true",
        "external=false",
        "num_gpus=1",
        f"reliability>{min_reliability:.2f}",
        f"dph_total<{max_dph:.2f}",
    ]
    if gpu_name and gpu_name.strip() and gpu_name.strip().lower() not in {"any", "任意"}:
        query_parts.append(f"gpu_name={gpu_name.strip()}")

    completed = run_vast_cli(
        [
            "search",
            "offers",
            " ".join(query_parts),
            "--order",
            "dph",
            "--limit",
            str(limit),
            "--raw",
        ],
        api_key=resolved_key,
        cancel_event=cancel_event,
    )
    payload = parse_json_output(completed.stdout)
    if not isinstance(payload, list):
        raise VastRunnerError(f"Unexpected search_offers payload type: {type(payload)!r}")

    offers: list[VastGpuOffer] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            offers.append(
                VastGpuOffer(
                    offer_id=int(item["id"]),
                    gpu_name=str(item.get("gpu_name", "GPU")),
                    num_gpus=int(item.get("num_gpus", 1)),
                    gpu_ram_gb=float(item.get("gpu_ram", 0.0)),
                    dph_total=float(item.get("dph_total", item.get("dph", 0.0))),
                    reliability=float(item.get("reliability2", item.get("reliability", 0.0))),
                    cuda_max_good=float(item.get("cuda_max_good", 0.0)),
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            logger.warning("Skipping malformed offer entry: %s (%s)", item, error)
    return offers
