# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures for the tests covering `.github/` scripts and workflows."""

from pathlib import Path

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
