# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Governance for confidential media and Vast.ai transfer allowlists."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from rfdetr_demo.paths import CONFIDENTIAL_AUDIT, CONFIDENTIAL_INPUT, CONFIDENTIAL_ROOT, REPO_ROOT

logger = logging.getLogger(__name__)

ALLOWED_VAST_INPUT_ROOTS: tuple[Path, ...] = (CONFIDENTIAL_INPUT,)

_AUDIT_LOG = CONFIDENTIAL_AUDIT / "vast-transfers.jsonl"


class MediaGuardError(PermissionError):
    """Raised when a media operation violates governance rules."""


def resolve_media_path(path: Path | str) -> Path:
    """Normalize *path* and ensure it resolves to an existing file under the repo."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        msg = f"Media file not found: {resolved}"
        raise FileNotFoundError(msg)
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        msg = (
            f"Media path must be inside the repository: {resolved}. "
            "Place confidential videos under confidential/media/input/."
        )
        raise MediaGuardError(msg) from exc
    return resolved


def is_under_confidential(path: Path) -> bool:
    """Return True when *path* is under ``confidential/``."""
    try:
        path.resolve().relative_to(CONFIDENTIAL_ROOT.resolve())
        return True
    except ValueError:
        return False


def _allow_any_source() -> bool:
    return os.environ.get("RFDETR_VAST_ALLOW_ANY_SOURCE", "").strip() in {"1", "true", "yes"}


def is_vast_transfer_allowed(source: Path) -> bool:
    """Return True when *source* may be uploaded to Vast.ai under current policy."""
    resolved = resolve_media_path(source)
    if _allow_any_source():
        return True
    for root in ALLOWED_VAST_INPUT_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def assert_vast_transfer_allowed(source: Path, *, user_acknowledged: bool) -> None:
    """Enforce Vast transfer policy; raise ``MediaGuardError`` when blocked."""
    resolved = resolve_media_path(source)
    if not user_acknowledged:
        msg = (
            "Vast.ai transfer requires explicit acknowledgement. "
            "Use --vast-ack-transfer (CLI) or confirm in the GUI."
        )
        raise MediaGuardError(msg)
    if is_vast_transfer_allowed(resolved):
        return
    if _allow_any_source():
        logger.warning(
            "RFDETR_VAST_ALLOW_ANY_SOURCE=1: transferring non-allowlisted path %s",
            resolved,
        )
        return
    allowed = ", ".join(str(p) for p in ALLOWED_VAST_INPUT_ROOTS)
    msg = (
        f"Vast transfer blocked for {resolved}. "
        f"Only files under [{allowed}] are allowed. "
        "Move confidential input videos to confidential/media/input/ "
        "or set RFDETR_VAST_ALLOW_ANY_SOURCE=1 with caution."
    )
    raise MediaGuardError(msg)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_transfer_audit(
    event: str,
    source: Path,
    destination: str,
    *,
    instance_id: int | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """Append a JSONL audit record under ``confidential/audit/``."""
    CONFIDENTIAL_AUDIT.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "event": event,
        "source": str(source.resolve()),
        "destination": destination,
        "sha256": _sha256_file(source) if source.is_file() else None,
        "instance_id": instance_id,
    }
    if extra:
        record.update(extra)
    with _AUDIT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Vast audit: %s %s -> %s", event, source, destination)
