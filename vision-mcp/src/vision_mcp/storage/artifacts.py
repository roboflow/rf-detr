# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Artifact store: generated IDs, metadata in SQLite, bytes on disk.

Callers never supply a filename and never receive a filesystem path — only an opaque ID and a `vision://artifacts/<id>`
URI.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from vision_mcp.api_contract import ArtifactRef
from vision_mcp.clock import utc_iso
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.logging_setup import get_logger
from vision_mcp.security import resolve_within, validate_artifact_id
from vision_mcp.storage.database import Database

logger = get_logger("vision-mcp.artifacts")

ArtifactKind = str

#: Only image types are storable; anything else is a bug in the caller.
_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png"}


class ArtifactStore:
    """Writes artifact bytes under a fixed root and records metadata in the database."""

    def __init__(self, root: Path, database: Database, max_bytes: int) -> None:
        self._root = root.expanduser().resolve()
        self._db = database
        self._max_bytes = max_bytes

    def ensure_root(self) -> None:
        """Create the artifact root; called once at engine startup."""
        self._root.mkdir(parents=True, exist_ok=True)

    async def save(
        self, kind: str, data: bytes, media_type: str = "image/jpeg", stream_id: str | None = None
    ) -> ArtifactRef:
        """Store bytes and return the reference clients may use.

        Raises:
            VisionError: INVALID_ARGUMENT for unsupported media types or oversized payloads.
        """
        extension = _EXTENSIONS.get(media_type)
        if extension is None:
            raise VisionError(
                ErrorCode.INVALID_ARGUMENT, "Unsupported artifact media type.", {"type": media_type}
            )
        if len(data) > self._max_bytes:
            raise VisionError(
                ErrorCode.INVALID_ARGUMENT,
                "Artifact exceeds the configured size limit.",
                {"bytes": len(data), "max_bytes": self._max_bytes},
            )
        artifact_id = uuid.uuid4().hex
        created = time.time()
        relative = Path(kind) / time.strftime("%Y%m%d", time.gmtime(created)) / f"{artifact_id}{extension}"
        target = self._root / relative
        await asyncio.to_thread(_write_file, target, data)
        await self._db.write(
            [
                (
                    "INSERT INTO artifacts (artifact_id, kind, media_type, relative_path, bytes, created_at,"
                    " stream_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (artifact_id, kind, media_type, str(relative), len(data), created, stream_id),
                )
            ]
        )
        return ArtifactRef(
            artifact_id=artifact_id,
            uri=f"vision://artifacts/{artifact_id}",
            kind=kind,  # type: ignore[arg-type]
            media_type=media_type,
            bytes=len(data),
            created_at=utc_iso(created),
            stream_id=stream_id,
        )

    async def path_for(self, artifact_id: str) -> tuple[Path, str]:
        """Resolve an artifact ID to a contained path and its media type.

        Raises:
            VisionError: ARTIFACT_NOT_FOUND for unknown IDs or missing files.
        """
        validate_artifact_id(artifact_id)
        row = await self._db.fetch_one(
            "SELECT relative_path, media_type FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        if row is None:
            raise VisionError(
                ErrorCode.ARTIFACT_NOT_FOUND, "Unknown artifact id.", {"artifact_id": artifact_id}
            )
        path = resolve_within(self._root, str(row["relative_path"]))
        if not path.is_file():
            raise VisionError(ErrorCode.ARTIFACT_NOT_FOUND, "Artifact metadata exists but the file is gone.")
        return path, str(row["media_type"])

    async def read(self, artifact_id: str) -> tuple[bytes, str]:
        """Return artifact bytes and media type."""
        path, media_type = await self.path_for(artifact_id)
        return await asyncio.to_thread(path.read_bytes), media_type

    async def total_bytes(self) -> int:
        """Sum of recorded artifact sizes."""
        row = await self._db.fetch_one("SELECT COALESCE(SUM(bytes), 0) AS total FROM artifacts")
        return int(row["total"]) if row else 0

    async def get_ref(self, artifact_id: str) -> ArtifactRef:
        """Look up a stored artifact's reference."""
        validate_artifact_id(artifact_id)
        row = await self._db.fetch_one(
            "SELECT artifact_id, kind, media_type, bytes, created_at, stream_id FROM artifacts"
            " WHERE artifact_id = ?",
            (artifact_id,),
        )
        if row is None:
            raise VisionError(
                ErrorCode.ARTIFACT_NOT_FOUND, "Unknown artifact id.", {"artifact_id": artifact_id}
            )
        return ArtifactRef(
            artifact_id=str(row["artifact_id"]),
            uri=f"vision://artifacts/{row['artifact_id']}",
            kind=str(row["kind"]),  # type: ignore[arg-type]
            media_type=str(row["media_type"]),
            bytes=int(row["bytes"]),
            created_at=utc_iso(float(row["created_at"])),
            stream_id=None if row["stream_id"] is None else str(row["stream_id"]),
        )

    async def delete_before(self, cutoff: float) -> int:
        """Remove artifacts created before *cutoff*; returns how many files were unlinked."""
        rows = await self._db.fetch_all(
            "SELECT artifact_id, relative_path FROM artifacts WHERE created_at < ?", (cutoff,)
        )
        removed = 0
        for row in rows:
            try:
                path = resolve_within(self._root, str(row["relative_path"]))
            except VisionError:
                logger.warning("skipping artifact outside root", extra={"artifact_id": row["artifact_id"]})
                continue
            if await asyncio.to_thread(_unlink, path):
                removed += 1
        if rows:
            await self._db.write([("DELETE FROM artifacts WHERE created_at < ?", (cutoff,))])
        return removed


def _write_file(target: Path, data: bytes) -> None:
    """Blocking write, always called through a worker thread."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _unlink(path: Path) -> bool:
    """Blocking unlink; returns whether a file was actually removed."""
    if not path.is_file():
        return False
    path.unlink()
    return True
