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
from unittest import mock

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


class _FakeParam:
    """Duck-typed stand-in for a parameter: only ``.device`` is read by the guard."""

    def __init__(self, device: torch.device) -> None:
        self.device = device


class _CountingDeviceModule:
    """Duck-typed module stand-in that counts real ``.to()`` calls without touching any accelerator.

    ``_move_model_context_to_device`` only calls ``next(inner.parameters(), None)`` and ``inner.to(target)`` on the
    model context's inner module, so a minimal stand-in exercises the guard logic without requiring a CUDA device to be
    present (this repo's CPU-only CI has none).
    """

    def __init__(self, initial_device: torch.device) -> None:
        self._device = initial_device
        self.to_call_count = 0

    def parameters(self) -> Any:
        """Yield the single fake parameter tracking the module's current device.

        Examples:
            >>> module = _CountingDeviceModule(torch.device("cpu"))
            >>> next(module.parameters()).device
            device(type='cpu')
        """
        yield _FakeParam(self._device)

    def to(self, device: torch.device) -> "_CountingDeviceModule":
        """Record the call and move the fake parameter to *device*."""
        self.to_call_count += 1
        self._device = device
        return self


class TestMoveModelContextIndexNormalization:
    """``torch.device('cuda')`` (no index) must compare equal to the indexed device it resolves to.

    ``model_ctx.device`` is built from a plain ``"cuda"`` string (see ``_build_model_context``), which
    ``torch.device()`` converts to an index-less device. A real parameter's ``.device`` always carries an explicit index
    once placed. Comparing the two with ``!=`` is ``True`` even when they name the same physical GPU, so without index
    normalization the guard below would move the whole model on every single call instead of only the first one that
    actually changes device.
    """

    def test_second_call_skips_redundant_move_when_target_has_no_index(self) -> None:
        """A second call with an index-less target must not re-trigger ``.to()`` once already placed."""
        module = _CountingDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("cuda"), model=module)

        with mock.patch("torch.cuda.current_device", return_value=0):
            _move_model_context_to_device(ctx)
            assert module.to_call_count == 1
            assert next(module.parameters()).device == torch.device("cuda", 0)

            _move_model_context_to_device(ctx)

        assert module.to_call_count == 1

    def test_explicit_index_mismatch_still_triggers_move(self) -> None:
        """An explicit different cuda index (multi-GPU) must still trigger a real move."""
        module = _CountingDeviceModule(initial_device=torch.device("cuda", 0))
        ctx = SimpleNamespace(device=torch.device("cuda", 1), model=module)

        with mock.patch("torch.cuda.current_device", return_value=0):
            _move_model_context_to_device(ctx)

        assert module.to_call_count == 1
        assert next(module.parameters()).device == torch.device("cuda", 1)
