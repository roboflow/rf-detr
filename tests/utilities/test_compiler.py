# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.compiler — version-compatible compile, trace and CUDA autocast predicates."""

import sys
from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.utilities.box_ops import batch_dice_loss
from rfdetr.utilities.compiler import cuda_autocast_dtype, is_compiling, is_tracing, script_or_eager


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


def test_is_tracing_returns_true_inside_a_jit_trace() -> None:
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


# ---------------------------------------------------------------------------
# script_or_eager
# ---------------------------------------------------------------------------


def _sum_variadic(*tensors: torch.Tensor) -> torch.Tensor:
    """Sum tensors through a variadic signature, which the TorchScript frontend rejects.

    Examples:
        >>> _sum_variadic(torch.tensor(1.0), torch.tensor(2.0))
        tensor(3.)
    """
    return torch.stack(tensors).sum(0)


class TestScriptOrEager:
    """script_or_eager compiles where TorchScript works and returns the eager function where it cannot."""

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="TorchScript is not used on Python 3.14+")
    def test_returns_script_function_when_scripting_succeeds(self) -> None:
        """A scriptable function comes back as a TorchScript function before Python 3.14.

        This is the historical contract of the ``*_jit`` loss aliases on every interpreter where TorchScript works.
        """
        assert isinstance(script_or_eager(batch_dice_loss), torch.jit.ScriptFunction)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(RuntimeError("not supported in Torchscript"), id="runtime-error"),
            pytest.param(OSError("Can't get source for <function _wrapped>"), id="source-less-wrapper"),
            pytest.param(
                TypeError("module, class, method, function, traceback, frame, or code object was expected"),
                id="functools-wraps-wrapper",
            ),
            pytest.param(SyntaxError("unexpected EOF while parsing"), id="lambda-wrapper"),
        ],
    )
    def test_returns_eager_function_when_scripting_raises(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        """Any exception raised while scripting falls back to the eager function instead of propagating.

        Hosts that wrap ``warnings.warn`` (Databricks notebooks) make TorchScript fail while compiling torch's own loss
        helpers, previously failing ``import rfdetr``. The failure class depends on the wrapper: ``RuntimeError`` when
        torch saw the wrapper, ``OSError`` when it has no retrievable source, ``TypeError`` for a ``functools.wraps``
        wrapper and ``SyntaxError`` for a lambda whose source line does not parse alone.
        """
        monkeypatch.setattr(sys, "version_info", (3, 13, 0))
        monkeypatch.setattr(torch.jit, "script", MagicMock(side_effect=error))

        assert script_or_eager(batch_dice_loss) is batch_dice_loss

    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="TorchScript is not used on Python 3.14+")
    def test_returns_eager_function_when_frontend_rejects_source(self) -> None:
        """A TorchScript frontend error falls back to the eager function instead of propagating.

        A ``warnings.warn`` wrapper installed after torch was imported is compiled as Python source when its source is
        retrievable, and its variadic signature makes the TorchScript frontend raise; a variadic function reproduces
        that without patching torch.
        """
        assert script_or_eager(_sum_variadic) is _sum_variadic

    def test_skips_scripting_on_python_314(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Python 3.14+ returns the eager function without calling TorchScript at all.

        TorchScript is unsupported there, and compiling it during import emits a deprecation warning.
        """
        monkeypatch.setattr(sys, "version_info", (3, 14, 0))
        script = MagicMock()
        monkeypatch.setattr(torch.jit, "script", script)

        assert script_or_eager(batch_dice_loss) is batch_dice_loss
        script.assert_not_called()
