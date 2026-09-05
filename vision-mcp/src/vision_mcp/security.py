# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Redaction, path containment and URL allowlisting.

Every value that reaches a log line, an error message or the filesystem passes through here.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from vision_mcp.errors import ErrorCode, VisionError

_REDACTED = "***REDACTED***"

#: userinfo in any URL (rtsp://user:pass@host, https://key@host, ...)
_URL_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<userinfo>[^/@\s]+)@")
#: key=value, key: value and "key": "value" forms for secret-looking keys
_SECRET_KV = re.compile(
    r"(?i)(?P<key>\b(?:password|passwd|pwd|secret|token|api[_-]?key|authorization|auth)\b\"?)"
    r"(?P<sep>\s*[=:]\s*\"?)"
    r"(?P<value>[^\s,;\"'&}]+)"
)


def redact(value: object) -> str:
    """Strip credentials from arbitrary text before it is logged or returned."""
    text = str(value)
    text = _URL_USERINFO.sub(lambda m: f"{m.group('scheme')}{_REDACTED}@", text)
    return _SECRET_KV.sub(lambda m: f"{m.group('key')}{m.group('sep')}{_REDACTED}", text)


def redact_data(value: object) -> Any:
    """Recursively redact strings in mappings and sequences."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(key): redact_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, set):
        return sorted((redact_data(item) for item in value), key=str)
    return value


def redact_source(source: str) -> str:
    """Redacted form of a stream source, safe to place in status payloads."""
    return redact(source)


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve *candidate* and confirm it stays inside *root*.

    Raises:
        VisionError: INVALID_ARGUMENT when the resolved path escapes the root.
    """
    root_resolved = root.expanduser().resolve()
    raw = Path(candidate).expanduser()
    target = raw.resolve() if raw.is_absolute() else (root_resolved / raw).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise VisionError(
            ErrorCode.INVALID_ARGUMENT,
            "Path is outside the configured root directory.",
            {"root": str(root_resolved)},
        )
    return target


def validate_local_path(roots: list[Path], candidate: str) -> Path:
    """Return the resolved path if it lives under any configured filesystem root."""
    if not roots:
        raise VisionError(ErrorCode.UNSUPPORTED_SOURCE, "No filesystem roots are configured for local files.")
    for root in roots:
        try:
            path = resolve_within(root, candidate)
        except VisionError:
            continue
        if not path.exists() or not path.is_file():
            raise VisionError(ErrorCode.INVALID_IMAGE, "File does not exist.", {"path": str(path)})
        return path
    raise VisionError(
        ErrorCode.INVALID_ARGUMENT,
        "Path is outside every configured filesystem root.",
        {"roots": [str(r) for r in roots]},
    )


def _is_private_host(host: str) -> bool:
    """True when the host resolves to loopback, link-local, private or reserved address space."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # unresolvable hosts are treated as unsafe
    for info in infos:
        address = ipaddress.ip_address(str(info[4][0]))
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return True
    return False


def _validate_remote_url(
    url: str, schemes: set[str], allowed_hosts: list[str], allow_private_network: bool
) -> str:
    """Validate a remote URL against schemes, allowlist and network policy."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in schemes:
        expected = ", ".join(sorted(schemes))
        raise VisionError(ErrorCode.URL_NOT_ALLOWED, f"URL scheme must be one of: {expected}.")
    host = parts.hostname
    if not host:
        raise VisionError(ErrorCode.URL_NOT_ALLOWED, "URL has no host.")
    if not allowed_hosts:
        raise VisionError(ErrorCode.URL_NOT_ALLOWED, "No URL hosts are allowlisted in the engine config.")
    if not any(host == entry or host.endswith(f".{entry}") for entry in allowed_hosts):
        raise VisionError(
            ErrorCode.URL_NOT_ALLOWED,
            "Host is not in the configured URL allowlist.",
            {"host": host, "allowed_hosts": allowed_hosts},
        )
    if not allow_private_network and _is_private_host(host):
        raise VisionError(
            ErrorCode.URL_NOT_ALLOWED,
            "Host resolves to a private or loopback address; "
            "set security.allow_private_network to permit it.",
            {"host": host},
        )
    return url


def validate_url(url: str, allowed_hosts: list[str], allow_private_network: bool) -> str:
    """Validate a remote image URL against scheme, host allowlist and network policy."""
    return _validate_remote_url(url, {"http", "https"}, allowed_hosts, allow_private_network)


def validate_stream_url(url: str, allowed_hosts: list[str], allow_private_network: bool) -> str:
    """Validate an HTTP(S) or RTSP(S) stream URL without applying image-only rules."""
    return _validate_remote_url(url, {"http", "https", "rtsp", "rtsps"}, allowed_hosts, allow_private_network)


def validate_content_type(content_type: str | None) -> None:
    """Reject downloads that do not declare an image content type."""
    if content_type is None or not content_type.split(";")[0].strip().startswith("image/"):
        raise VisionError(
            ErrorCode.INVALID_IMAGE,
            "Remote resource is not an image.",
            {"content_type": content_type or "missing"},
        )


def validate_pixel_count(width: int, height: int, max_pixels: int) -> None:
    """Reject decoded images larger than the configured pixel budget."""
    if width * height > max_pixels:
        raise VisionError(
            ErrorCode.INVALID_IMAGE,
            "Image exceeds the configured maximum pixel count.",
            {"pixels": width * height, "max_pixels": max_pixels},
        )


ARTIFACT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def validate_artifact_id(artifact_id: str) -> str:
    """Artifact IDs are engine-generated hex; anything else is a traversal attempt."""
    if not ARTIFACT_ID_PATTERN.match(artifact_id):
        raise VisionError(
            ErrorCode.ARTIFACT_NOT_FOUND, "Unknown artifact id.", {"artifact_id": artifact_id[:64]}
        )
    return artifact_id
