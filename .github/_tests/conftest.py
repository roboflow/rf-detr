# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures for the tests covering `.github/` scripts and workflows."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root of the repository checkout these tests run against.

    Examples:
        >>> repo_root  # doctest: +SKIP
        pytest fixture; resolved from the test file's location.
    """
    return REPO_ROOT


@pytest.fixture(scope="session")
def injector() -> ModuleType:
    """The gh-pages banner backfill script, loaded by path.

    It is a standalone CI entry point under `.github/`, not an installed module, so importing
    it by name is not possible.

    Examples:
        >>> injector  # doctest: +SKIP
        pytest fixture; loads .github/scripts/inject_outdated_banner.py by path.
    """
    script = REPO_ROOT / ".github" / "scripts" / "inject_outdated_banner.py"
    spec = importlib.util.spec_from_file_location("inject_outdated_banner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
