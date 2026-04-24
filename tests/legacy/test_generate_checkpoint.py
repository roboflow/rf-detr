# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the legacy checkpoint generation utility."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from types import ModuleType

import torch
from transformers import pytorch_utils

from tests.legacy.generate_checkpoint import _install_transformers_compat


def _get_transformers_backbone_utils() -> ModuleType:
    """Import deprecated transformers backbone module without test warnings.

    Returns:
        Imported ``transformers.utils.backbone_utils`` module.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing `Backbone.*` from `utils/backbone_utils.py` is deprecated.*",
            category=FutureWarning,
        )
        return importlib.import_module("transformers.utils.backbone_utils")


def test_transformers_compat_installs_missing_prune_helper(monkeypatch) -> None:
    """RF-DETR 1.4 imports the old pruning helper from transformers.

    Args:
        monkeypatch: Pytest fixture used to simulate transformers v5 removing
            the historical public helper.
    """
    monkeypatch.delattr(pytorch_utils, "find_pruneable_heads_and_indices", raising=False)

    _install_transformers_compat()

    helper: Callable[[set[int], int, int, set[int]], tuple[set[int], torch.LongTensor]] = getattr(
        pytorch_utils, "find_pruneable_heads_and_indices"
    )
    heads, index = helper({1}, 4, 3, set())

    assert heads == {1}
    assert index.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]


def test_transformers_compat_installs_missing_backbone_alignment_helper(monkeypatch) -> None:
    """RF-DETR 1.4 imports the old backbone alignment helper from transformers.

    Args:
        monkeypatch: Pytest fixture used to simulate transformers v5 removing
            the historical public helper.
    """
    backbone_utils = _get_transformers_backbone_utils()
    monkeypatch.delattr(backbone_utils, "get_aligned_output_features_output_indices", raising=False)

    _install_transformers_compat()

    helper: Callable[[list[str] | None, list[int] | tuple[int, ...] | None, list[str]], tuple[list[str], list[int]]]
    helper = getattr(backbone_utils, "get_aligned_output_features_output_indices")

    features, indices = helper(None, (1, 2), ["stem", "layer1", "layer2"])

    assert features == ["layer1", "layer2"]
    assert indices == [1, 2]
