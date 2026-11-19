# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Package version, install-probe, and git-status helpers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version


def is_installed(name: str) -> bool:
    """Report whether a module is installed, without importing it.

    ``find_spec`` answers from the import system's metadata, so an optional package whose import pulls in heavy
    machinery (CUDA libraries, a compiler backend) costs nothing to ask about. Two of its answers fold into ``False``
    here. A spec with no ``origin`` is a namespace package -- a bare directory that happens to carry the name, such as
    an export folder, rather than an install. A ``ValueError`` means ``sys.modules`` already holds an entry whose
    ``__spec__`` is missing or ``None``: a stub left behind by a test or by a library that builds its own module
    object, which is likewise nothing to build against. (``ImportError`` covers the separate case of a missing parent
    package.)

    Args:
        name: Absolute module name, e.g. ``"tensorrt"``.

    Returns:
        Whether *name* resolves to an installed module.

    Examples:
        >>> is_installed("json")
        True
        >>> is_installed("a_module_that_is_not_installed")
        False
    """
    try:
        return getattr(importlib.util.find_spec(name), "origin", None) is not None
    except (ImportError, ValueError):
        return False


def get_version(package_name: str = "rfdetr") -> str | None:
    """Get the current version of the specified package.

    Args:
        package_name: The name of the package to get the version for. Defaults to ``'rfdetr'``.

    Returns:
        The version string of the specified package, or ``None`` if the version cannot be determined.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_sha() -> str:
    """Return a short status string for the current git repo, or 'unknown' if unavailable.

    Returns:
        String describing the current git HEAD, status, and branch.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        diff_result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if diff_result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(
                returncode=diff_result.returncode,
                cmd=["git", "diff-index", "--quiet", "HEAD", "--"],
                output=diff_result.stdout,
                stderr=diff_result.stderr,
            )
        has_diff = diff_result.returncode == 1
        status = "has uncommitted changes" if has_diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return f"sha: {sha}, status: {status}, branch: {branch}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
