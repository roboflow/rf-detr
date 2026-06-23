# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures and test helpers for model tests."""

from types import SimpleNamespace

import pytest

from rfdetr.detr import RFDETR


class _BaseFakeRFDETR(RFDETR):
    """RFDETR test double that skips weight downloads and returns a minimal model config.

    Subclasses must override ``get_model`` to supply the model context appropriate for
    the scenario under test.

    Examples:
        This class is imported directly by test modules that need a weight-free RFDETR.
    """

    def maybe_download_pretrain_weights(self) -> None:
        """Skip weight download in tests."""
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        """Return a minimal config sufficient for most test scenarios."""
        return SimpleNamespace(num_channels=3)


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
