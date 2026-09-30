# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private helpers for parsing Ultralytics YOLO data YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_yaml_mapping(yaml_path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping root.

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
    import yaml  # type: ignore[import-untyped,unused-ignore]

    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in data file {str(yaml_path)!r}, got {type(data).__name__}.")
    return data


def _extract_yolo_class_names_from_data(data: dict[str, Any], data_file: Path) -> list[str]:
    """Extract contiguous YOLO class names from parsed YAML data."""
    names = data.get("names")
    if isinstance(names, dict):
        names_by_id: dict[int, Any] = {}
        for key, name in names.items():
            key_str = str(key)
            if key_str.isdigit():
                names_by_id[int(key_str)] = name

        sorted_ids = sorted(names_by_id)
        if not sorted_ids or sorted_ids != list(range(len(sorted_ids))) or len(names_by_id) != len(names):
            raise ValueError(
                "Unsupported 'names' mapping in data file "
                f"{str(data_file)!r}: expected integer keys 0..N-1 with no gaps or duplicate IDs."
            )
        return [str(names_by_id[idx]) for idx in sorted_ids]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Expected 'names' to be a list or dict in {str(data_file)!r}, got {type(names).__name__}.")
