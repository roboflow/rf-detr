# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Loading images from the three supported sources, with every security check applied.

A source is a local path under a configured filesystem root, an allowlisted HTTP(S) URL, or an inline
`data:image/...;base64,` URI. Nothing else is accepted.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from vision_mcp.api_contract import ImageSize
from vision_mcp.config import SecurityConfig
from vision_mcp.errors import ErrorCode, VisionError
from vision_mcp.logging_setup import get_logger
from vision_mcp.security import (
    redact,
    validate_content_type,
    validate_local_path,
    validate_pixel_count,
    validate_url,
)

logger = get_logger("vision-mcp.images")

_DATA_PREFIX = "data:image/"
_MAX_DATA_URI_HEADER_CHARS = 256


@dataclass(slots=True)
class LoadedImage:
    """A decoded RGB image plus a log-safe description of where it came from."""

    array: np.ndarray[Any, Any]
    size: ImageSize
    label: str


class ImageLoader:
    """Fetches and decodes single images under the configured security policy."""

    def __init__(self, security: SecurityConfig) -> None:
        self._security = security

    async def load(self, source: str) -> LoadedImage:
        """Resolve *source* to a decoded RGB image.

        Raises:
            VisionError: UNSUPPORTED_SOURCE, URL_NOT_ALLOWED, INVALID_IMAGE or INVALID_ARGUMENT.
        """
        if source.startswith(_DATA_PREFIX):
            data = _decode_data_uri(source, self._security.max_download_bytes)
            return self._decode(data, label="data:image")
        if source.startswith(("http://", "https://")):
            data = await self._download(source)
            return self._decode(data, label=redact(source))
        if source.startswith(("rtsp://", "file://", "ftp://")):
            raise VisionError(
                ErrorCode.UNSUPPORTED_SOURCE,
                "Only local files, http(s) URLs and data URIs are valid image sources.",
                {"source": redact(source)},
            )
        path = validate_local_path(self._security.filesystem_roots, source)
        data = await asyncio.to_thread(path.read_bytes)
        return self._decode(data, label=path.name)

    async def load_many(self, sources: list[str]) -> list[LoadedImage]:
        """Load several images sequentially; decode cost is CPU-bound so there is nothing to overlap."""
        return [await self.load(source) for source in sources]

    async def _download(self, url: str) -> bytes:
        """Fetch a remote image with size, timeout and content-type limits enforced."""
        validate_url(url, self._security.allowed_url_hosts, self._security.allow_private_network)
        limit = self._security.max_download_bytes
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self._security.download_timeout_seconds, follow_redirects=False
                ) as client,
                client.stream("GET", url) as response,
            ):
                response.raise_for_status()
                validate_content_type(response.headers.get("content-type"))
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > limit:
                        raise VisionError(
                            ErrorCode.INVALID_IMAGE,
                            "Remote image exceeds the configured download limit.",
                            {"max_download_bytes": limit},
                        )
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            logger.warning("image download failed", extra={"url": redact(url), "error": redact(exc)})
            raise VisionError(
                ErrorCode.INVALID_IMAGE, "Could not download the image.", {"url": redact(url)}
            ) from exc
        return b"".join(chunks)

    def _decode(self, data: bytes, label: str) -> LoadedImage:
        """Decode bytes to RGB, validating dimensions from the header before paying for pixels."""
        if not data:
            raise VisionError(ErrorCode.INVALID_IMAGE, "Image payload is empty.", {"source": label})
        try:
            with Image.open(io.BytesIO(data)) as image:
                validate_pixel_count(image.width, image.height, self._security.max_pixels)
                rgb = image.convert("RGB")
                array = np.asarray(rgb, dtype=np.uint8)
        except VisionError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise VisionError(
                ErrorCode.INVALID_IMAGE, "Image could not be decoded.", {"source": label}
            ) from exc
        return LoadedImage(
            array=array,
            size=ImageSize(width=int(array.shape[1]), height=int(array.shape[0])),
            label=label,
        )


def _decode_data_uri(source: str, max_bytes: int) -> bytes:
    """Extract bytes from a base64 data URI without allocating beyond *max_bytes*."""
    separator = source.find(",")
    if (
        separator < 0
        or separator > _MAX_DATA_URI_HEADER_CHARS
        or ";base64" not in source[:separator]
        or separator == len(source) - 1
    ):
        raise VisionError(ErrorCode.INVALID_IMAGE, "Data URIs must be base64-encoded image data.")
    encoded_bytes = len(source) - separator - 1
    padding = 2 if source.endswith("==") else int(source.endswith("="))
    decoded_bytes = (encoded_bytes // 4) * 3 - padding
    if decoded_bytes > max_bytes:
        raise VisionError(
            ErrorCode.INVALID_IMAGE,
            "Inline image exceeds the configured payload limit.",
            {"max_download_bytes": max_bytes},
        )
    payload = source[separator + 1 :]
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VisionError(ErrorCode.INVALID_IMAGE, "Data URI payload is not valid base64.") from exc
    if len(decoded) > max_bytes:
        raise VisionError(
            ErrorCode.INVALID_IMAGE,
            "Inline image exceeds the configured payload limit.",
            {"max_download_bytes": max_bytes},
        )
    return decoded


def encode_jpeg(array: np.ndarray[Any, Any], quality: int = 85) -> bytes:
    """Encode an RGB array as JPEG bytes for artifacts and preview frames."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def crop_array(
    array: np.ndarray[Any, Any], x1: float, y1: float, x2: float, y2: float
) -> np.ndarray[Any, Any]:
    """Clamp a box to the image and return that region."""
    height, width = array.shape[:2]
    left = max(0, min(int(x1), width - 1))
    top = max(0, min(int(y1), height - 1))
    right = max(left + 1, min(int(x2), width))
    bottom = max(top + 1, min(int(y2), height))
    return array[top:bottom, left:right]


def image_path_label(path: Path) -> str:
    """File name only; absolute paths never leave the engine."""
    return path.name
