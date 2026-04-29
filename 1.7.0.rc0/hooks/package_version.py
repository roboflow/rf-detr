# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MkDocs hooks for exposing package metadata to documentation templates."""

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli


def _load_pyproject(pyproject_path: Path) -> dict[str, Any]:
    """Load project metadata from a pyproject file.

    Reads the TOML file in binary mode, matching the parser API. The returned
    dictionary is used by MkDocs hooks to avoid duplicating package metadata.

    Args:
        pyproject_path: Path to the repository's pyproject file.

    Returns:
        Parsed pyproject data.

    Raises:
        FileNotFoundError: If the pyproject file does not exist.
        tomllib.TOMLDecodeError: If the pyproject file is invalid TOML on Python 3.11+.
        tomli.TOMLDecodeError: If the pyproject file is invalid TOML on Python 3.10.

    Example:
        >>> data = _load_pyproject(Path("pyproject.toml"))
        >>> isinstance(data["project"]["version"], str)
        True
    """
    with pyproject_path.open("rb") as pyproject_file:
        if sys.version_info >= (3, 11):
            return tomllib.load(pyproject_file)
        return tomli.load(pyproject_file)


def _read_project_version(pyproject_path: Path) -> str:
    """Read the package version from pyproject project metadata.

    Validates that the version exists and is a string before exposing it to the
    documentation templates.

    Args:
        pyproject_path: Path to the repository's pyproject file.

    Returns:
        Package version from the ``[project]`` table.

    Raises:
        RuntimeError: If ``[project].version`` is missing or not a string.

    Example:
        >>> _read_project_version(Path("pyproject.toml"))
        '1.6.0'
    """
    pyproject = _load_pyproject(pyproject_path)
    version = pyproject.get("project", {}).get("version")
    if not isinstance(version, str):
        msg = f"Missing string [project].version in {pyproject_path}"
        raise RuntimeError(msg)
    return version


def on_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expose the package version to MkDocs templates.

    Adds ``config.extra.software_version`` so schema.org JSON-LD can stay in
    sync with the package metadata in ``pyproject.toml``.

    Args:
        config: MkDocs configuration object.

    Returns:
        Updated MkDocs configuration object.

    Raises:
        RuntimeError: If package metadata cannot provide a valid version.

    Example:
        >>> cfg = {"extra": {}}
        >>> updated = on_config(cfg)
        >>> updated["extra"]["software_version"]
        '1.6.0'
    """
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    extra = config.setdefault("extra", {})
    extra["software_version"] = _read_project_version(pyproject_path)
    return config
