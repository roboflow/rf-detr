# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures for model tests."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _prewarm_dinov2_cache() -> None:
    """Download DINOv2 backbone weights once per test session.

    HuggingFace hub uses file-level locking internally, so concurrent xdist
    workers block on each other rather than issuing duplicate network requests.
    After the first worker finishes, all others read from the local disk cache.

    Examples:
        This fixture is autouse — no explicit reference needed in tests.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        "facebook/dinov2-with-registers-base",
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
