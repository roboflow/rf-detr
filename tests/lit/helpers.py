# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared test helpers for the rfdetr.training test suite.

Plain classes and functions (not pytest fixtures) shared across multiple test
modules to avoid verbatim duplication.  Import with a relative import::

    from .helpers import _FakeCriterion, _FakeDataset, _TinyModel
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.data


class _TinyModel(nn.Module):
    """Minimal real nn.Module satisfying the RFDETRModule model contract.

    Has a single trainable parameter so the optimizer has something to update
    and the loss has a gradient path back through the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, samples, targets=None):
        return {"dummy": self.dummy}

    def update_drop_path(self, *args, **kwargs) -> None:
        pass

    def update_dropout(self, *args, **kwargs) -> None:
        pass

    def reinitialize_detection_head(self, *args, **kwargs) -> None:
        pass


class _FakeCriterion:
    """Callable criterion that returns a loss connected to the model output.

    Keeps a gradient path from the loss back to _TinyModel.dummy so that
    ``loss.backward()`` does not error when the Trainer calls it.
    """

    weight_dict = {"loss_ce": 1.0}

    def __call__(self, outputs, targets):
        dummy = outputs.get("dummy", torch.zeros(1))
        return {"loss_ce": dummy.mean()}


class _FakeDataset(torch.utils.data.Dataset):
    """Dataset with ``(image, target)`` pairs for detection.

    The image is a ``(3, 32, 32)`` float tensor; the target dict includes the
    fields expected by RFDETRModule: ``boxes``, ``labels``, ``image_id``,
    ``orig_size``, ``size``.
    """

    def __init__(self, length: int = 20) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([32, 32]),
            "size": torch.tensor([32, 32]),
        }
        return image, target


class _FakeDatasetWithMasks(_FakeDataset):
    """Like _FakeDataset but includes binary instance masks (for segmentation)."""

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        target["masks"] = torch.zeros(1, 32, 32, dtype=torch.bool)
        return image, target


def _fake_postprocess(outputs, orig_sizes):
    """Return one non-empty prediction per image so COCOEvalCallback has something to score."""
    n = orig_sizes.shape[0]
    return [
        {
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }
        for _ in range(n)
    ]


def _make_param_dicts(model: nn.Module) -> list[dict]:
    """Build a minimal param-dict list for AdamW from all trainable parameters."""
    return [{"params": p, "lr": 1e-4} for p in model.parameters() if p.requires_grad]
