# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private helpers for parsing Ultralytics YOLO data YAML files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

# Per-process memo for _load_yaml_mapping, keyed on (resolved_path, mtime_ns, size) so a rewritten
# file at the same path busts the entry. A handful of dataset YAML files are parsed per process, so
# this never needs eviction; RFDETR._memoized_coco_categories is the closest existing precedent, but
# that one is a per-instance cache passed in by the caller because it is read from a method — this
# memo backs a free function called from detection/pose/facade entry points with no owning instance.
_yaml_mapping_cache: dict[tuple[str, int, int], dict[str, Any]] = {}


def _load_yaml_mapping(yaml_path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping root.

    Training calls this multiple times per ``RFDETR.train()`` invocation (num_classes alignment,
    keypoint schema inference, test-split resolution, ...) for the same ``data.yaml``. The parsed
    mapping is memoized by resolved path plus file mtime/size so repeat calls in one process skip
    reparsing; each caller gets its own deep copy so mutating the return value can never corrupt the
    cached entry for a later caller.

    Args:
        yaml_path: Path to a YAML data file.

    Returns:
        Parsed YAML mapping.

    Raises:
        ValueError: If the YAML root is not a mapping.
        OSError: If the file cannot be read.

    Example:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "data.yaml"
        >>> _ = path.write_text("names: [person]\\nkpt_shape: [1, 3]\\n", encoding="utf-8")
        >>> sorted(_load_yaml_mapping(path))
        ['kpt_shape', 'names']
    """
    resolved_path = Path(yaml_path).resolve()
    file_stat = resolved_path.stat()
    cache_key = (str(resolved_path), file_stat.st_mtime_ns, file_stat.st_size)
    cached = _yaml_mapping_cache.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    import yaml  # type: ignore[import-untyped,unused-ignore]

    with resolved_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in data file {str(yaml_path)!r}, got {type(data).__name__}.")
    _yaml_mapping_cache[cache_key] = data
    return copy.deepcopy(data)


def _ascii_digit_key(key: Any) -> int | None:
    """Parse a YAML mapping key as an ASCII-digit integer, or return ``None``.

    Non-ASCII digit characters (e.g. superscript ``'²'``, fullwidth ``'１'``, Arabic-Indic
    ``'١'``) satisfy ``str.isdigit()`` but are rejected here: ``int()`` either raises on them
    (superscripts) or silently accepts a visually different numbering scheme (fullwidth, Arabic-Indic),
    neither of which is a YOLO-YAML author's intended plain integer key.

    Args:
        key: Raw YAML mapping key (typically an ``int`` or ``str``).

    Returns:
        The parsed integer when ``key`` stringifies to ASCII digits only, otherwise ``None``.

    Example:
        >>> _ascii_digit_key(0)
        0
        >>> _ascii_digit_key("00")
        0
        >>> _ascii_digit_key("²") is None
        True
    """
    key_str = str(key)
    if key_str.isascii() and key_str.isdigit():
        return int(key_str)
    return None


def _extract_yolo_class_names_from_data(data: dict[str, Any], data_file: Path) -> list[str]:
    """Extract contiguous YOLO class names from parsed YAML data."""
    names = data.get("names")
    if isinstance(names, dict):
        names_by_id: dict[int, Any] = {}
        for key, name in names.items():
            numeric_key = _ascii_digit_key(key)
            if numeric_key is not None:
                names_by_id[numeric_key] = name

        sorted_ids = sorted(names_by_id)
        if not sorted_ids or sorted_ids != list(range(len(sorted_ids))) or len(names_by_id) != len(names):
            # This only catches duplicate IDs from distinct keys colliding after numeric conversion
            # (e.g. {0, "0"} or {"0", "00"}) — identical literal YAML keys (e.g. two quoted "0" keys)
            # are already collapsed to one entry by yaml.safe_load before this function ever runs.
            raise ValueError(
                "Unsupported 'names' mapping in data file "
                f"{str(data_file)!r}: expected integer keys 0..N-1 with no gaps and no distinct "
                "keys colliding on the same numeric ID."
            )
        return [str(names_by_id[idx]) for idx in sorted_ids]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Expected 'names' to be a list or dict in {str(data_file)!r}, got {type(names).__name__}.")
