# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared filesystem and archive primitives for WebDataset storage.

Purpose: Keep shard-path validation and tar-size accounting consistent across packing and loading. Scope: trusted local
paths, image-member extensions and POSIX tar member framing. Usage: import these private helpers only from sibling
WebDataset modules. Outputs: verified child paths and archive byte counts. Failure: rejects shard-index paths that leave
their designated directory. Used by: rfdetr.datasets.webdataset.pack and rfdetr.datasets.webdataset.load.
"""

from __future__ import annotations

from pathlib import Path

#: Supported image extensions written to and decoded from shards.
IMAGE_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp")

_TAR_BLOCK_BYTES = 512


def resolve_within(base: Path, name: str) -> Path:
    """Resolve *name* under *base*, rejecting any path that escapes it.

    Shard file names come from ordinary on-disk JSON. Both shard reading and
    stale-shard cleanup trust those entries, so traversal must fail before an
    open or unlink reaches a path outside the shard directory.

    Args:
        base: Directory *name* must resolve inside.
        name: Untrusted path-like string taken from a shard index.

    Returns:
        The resolved absolute path to *name* under *base*.

    Raises:
        ValueError: If *name* is absolute or resolves outside *base*.

    Examples:
        >>> resolve_within(Path("/data/shards"), "train-000000.tar").name
        'train-000000.tar'
        >>> resolve_within(Path("/data/shards"), "../../etc/passwd")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: shard entry '../../etc/passwd' resolves outside ...shards.
    """
    base_resolved = base.resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"shard entry {name!r} resolves outside {base}.")
    return candidate


def tar_member_bytes(payload_len: int) -> int:
    """Return the on-disk bytes occupied by a POSIX tar member payload.

    Each member adds one header block and rounds its content up to the next
    block. Packing uses this rather than raw payload lengths to respect the
    requested shard-size boundary.

    Args:
        payload_len: Content length of the member, in bytes.

    Returns:
        Total bytes occupied in the tar archive, including header and padding.

    Examples:
        >>> tar_member_bytes(0)
        512
        >>> tar_member_bytes(513)
        1536
    """
    return _TAR_BLOCK_BYTES + -(-payload_len // _TAR_BLOCK_BYTES) * _TAR_BLOCK_BYTES
