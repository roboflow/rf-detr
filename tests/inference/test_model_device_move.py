# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the lazy device move running under ``torch.inference_mode()``.

``predict()`` stacks ``@torch.inference_mode()`` on top of ``@_ensure_model_on_device``, so the deferred CPU-to-
accelerator move happens while inference mode is active.  Tensors materialised under inference mode are *inference
tensors*: they can never require gradients, so a later ``train()`` / auto-batch probe silently produces no gradients.
The move itself must therefore always run with inference mode disabled.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from rfdetr.detr import _move_model_context_to_device


class _RecordingModule(nn.Module):
    """Module whose ``to()`` records whether inference mode was active at move time."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.inference_mode_at_move: bool | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_RecordingModule":
        """Record the inference-mode state instead of performing a real device move."""
        self.inference_mode_at_move = torch.is_inference_mode_enabled()
        return self


class TestMoveModelContextUnderInferenceMode:
    """The deferred device move must never materialise parameters as inference tensors."""

    def test_moved_params_are_not_inference_tensors(self) -> None:
        """A real ``.to()`` move inside ``torch.inference_mode()`` must not create inference-tensor parameters."""
        ctx = SimpleNamespace(device=torch.device("meta"), model=nn.Linear(2, 2))

        with torch.inference_mode():
            _move_model_context_to_device(ctx)

        assert not any(p.is_inference() for p in ctx.model.parameters())

    def test_move_still_materializes_on_target_device(self) -> None:
        """The inference-mode guard must not suppress the device move itself."""
        ctx = SimpleNamespace(device=torch.device("meta"), model=nn.Linear(2, 2))

        with torch.inference_mode():
            _move_model_context_to_device(ctx)

        assert all(p.device.type == "meta" for p in ctx.model.parameters())

    def test_move_runs_with_inference_mode_disabled(self) -> None:
        """The ``.to()`` call itself must observe inference mode as disabled."""
        module = _RecordingModule()
        ctx = SimpleNamespace(device=torch.device("meta"), model=module)

        with torch.inference_mode():
            _move_model_context_to_device(ctx)

        assert module.inference_mode_at_move is False
