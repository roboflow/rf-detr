# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pytest fixtures shared by the callback test modules."""

from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def all_reduce(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Pretend a distributed group is initialized and return the mocked ``dist.all_reduce``.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects ``monkeypatch``):

        >>> all_reduce  # doctest: +SKIP
        <MagicMock name='all_reduce' id='...'>
    """
    mock = MagicMock(name="all_reduce")
    monkeypatch.setattr("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", lambda: True)
    monkeypatch.setattr("rfdetr.training.callbacks.coco_eval.dist.all_reduce", mock)
    return mock


@pytest.fixture
def loaders() -> tuple[DataLoader, DataLoader]:
    """Train and validation loaders over the same 8 random samples, batch size 2.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects its value):

        >>> train_loader, val_loader = loaders  # doctest: +SKIP
        >>> len(train_loader), len(val_loader)  # doctest: +SKIP
        (4, 4)
    """
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    return DataLoader(TensorDataset(x, y), batch_size=2), DataLoader(TensorDataset(x, y), batch_size=2)
