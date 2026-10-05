# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.compiler — version-compatible compile, trace and CUDA autocast predicates."""

import pytest
import torch

from rfdetr.utilities.compiler import cuda_autocast_dtype, is_compiling, is_tracing


def _is_autocast_enabled_without_device(*args: object) -> bool:
    """Stand in for PyTorch 2.2's ``torch.is_autocast_enabled``, which takes no device argument.

    Examples:
        >>> _is_autocast_enabled_without_device()
        True
        >>> _is_autocast_enabled_without_device("cuda")
        Traceback (most recent call last):
        ...
        TypeError: is_autocast_enabled() takes 0 positional arguments but 1 was given
    """
    if args:
        raise TypeError(f"is_autocast_enabled() takes 0 positional arguments but {len(args)} was given")
    return True


# ---------------------------------------------------------------------------
# is_compiling
# ---------------------------------------------------------------------------


class TestIsCompiling:
    """is_compiling reflects torch's current compile-state predicate on every call."""

    def test_returns_false_outside_compiled_graph(self) -> None:
        """Returns False on plain eager execution with no monkeypatching involved."""
        assert is_compiling() is False

    def test_uses_public_predicate_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reflects torch.compiler.is_compiling when that public predicate is present.

        PyTorch >=2.3 exposes the public predicate directly on torch.compiler; when it is present, is_compiling must
        defer to it rather than touching the legacy Dynamo API.
        """
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True, raising=False)

        assert is_compiling() is True

    def test_falls_back_to_dynamo_predicate_when_public_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reflects torch._dynamo.is_compiling when the public predicate is absent.

        Simulates PyTorch 2.2, which lacks torch.compiler.is_compiling: the legacy Dynamo predicate must be consulted
        instead so the module keeps working on that minimum supported version.
        """
        monkeypatch.delattr(torch.compiler, "is_compiling", raising=False)
        monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)

        assert is_compiling() is True


class TestIsTracing:
    """is_tracing reports whether a ``torch.jit.trace`` is recording the current call."""

    def test_returns_true_inside_a_jit_trace(self) -> None:
        """Returns True for code that runs while ``torch.jit.trace`` records it.

        The decoder rewrites step aside under tracing because the ONNX/TorchScript exporters expect the plain ops; a
        predicate that stayed False inside a trace would bake a custom autograd function into the exported graph.
        """
        seen: list[bool] = []

        def record(x: torch.Tensor) -> torch.Tensor:
            seen.append(is_tracing())
            return x * 2

        torch.jit.trace(record, torch.ones(2), check_trace=False)

        assert seen == [True]


class TestCudaAutocastDtype:
    """cuda_autocast_dtype reports CUDA autocast's compute dtype, or None when it is off."""

    def test_returns_the_compute_dtype_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns the dtype from ``torch.get_autocast_dtype("cuda")`` while CUDA autocast is on.

        This is the path every current torch takes; the CUDA graph runner keys captures on it and the decoder emits its
        folded casts in it.
        """
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: True)
        monkeypatch.setattr(torch, "get_autocast_dtype", lambda device: torch.bfloat16, raising=False)

        assert cuda_autocast_dtype() == torch.bfloat16

    def test_falls_back_to_the_torch_2_2_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses the device-less ``is_autocast_enabled`` and ``get_autocast_gpu_dtype`` when torch lacks the new API.

        Simulates PyTorch 2.2, the minimum supported version: ``is_autocast_enabled`` rejects a device argument and
        ``get_autocast_dtype`` does not exist yet.
        """
        monkeypatch.setattr(torch, "is_autocast_enabled", _is_autocast_enabled_without_device)
        monkeypatch.delattr(torch, "get_autocast_dtype", raising=False)
        monkeypatch.setattr(torch, "get_autocast_gpu_dtype", lambda: torch.float16, raising=False)

        assert cuda_autocast_dtype() == torch.float16
