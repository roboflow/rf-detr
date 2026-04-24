# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the legacy checkpoint generation utility."""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import pytorch_utils

from tests.legacy.generate_checkpoint import _install_transformers_pytorch_utils_compat


def test_transformers_compat_installs_missing_prune_helper(monkeypatch) -> None:
    """RF-DETR 1.4 imports the old pruning helper from transformers.

    Args:
        monkeypatch: Pytest fixture used to simulate transformers v5 removing
            the historical public helper.
    """
    monkeypatch.delattr(pytorch_utils, "find_pruneable_heads_and_indices", raising=False)

    _install_transformers_pytorch_utils_compat()

    helper: Callable[[set[int], int, int, set[int]], tuple[set[int], torch.LongTensor]] = getattr(
        pytorch_utils, "find_pruneable_heads_and_indices"
    )
    heads, index = helper({1}, 4, 3, set())

    assert heads == {1}
    assert index.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
