# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``_safe_torch_load`` helper in ``rfdetr.util.io``.

Covers the three-stage safe-load strategy: strict weights_only, safe-globals fallback, and opt-in pickle fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rfdetr.util.io import _safe_torch_load

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_tensor_only_checkpoint(path: Path) -> None:
    """Save a checkpoint containing only tensors and plain dicts to *path*."""
    ckpt = {"model": {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.0])}}
    torch.save(ckpt, path)


def _write_namespace_checkpoint(path: Path) -> None:
    """Save a checkpoint with an ``argparse.Namespace`` args value to *path*.

    Legacy RF-DETR engine.py checkpoints embed a Namespace; strict ``weights_only=True`` (without safe globals) would
    reject these.
    """
    ckpt = {
        "model": {"weight": torch.tensor([1.0])},
        "args": argparse.Namespace(pretrain_weights="rf-detr-small.pth", num_classes=80),
    }
    torch.save(ckpt, path)


def _write_simple_namespace_checkpoint(path: Path) -> None:
    """Save a checkpoint with a ``types.SimpleNamespace`` to *path*."""
    ckpt = {
        "model": {"weight": torch.tensor([1.0])},
        "args": SimpleNamespace(pretrain_weights="rf-detr-small.pth"),
    }
    torch.save(ckpt, path)


class _ArbitraryObject:
    """Module-level object that torch.save can pickle but weights_only=True rejects.

    Must be defined at module scope so pickle can resolve its fully-qualified name during serialisation (local/nested
    classes cannot be pickled by torch.save).
    """

    value = 42


def _write_arbitrary_pickle_checkpoint(path: Path) -> None:
    """Save a checkpoint that embeds an arbitrary class (requires pickle)."""
    ckpt = {"model": {"weight": torch.tensor([1.0])}, "extra": _ArbitraryObject()}
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Safe path (weights_only=True)
# ---------------------------------------------------------------------------


class TestSafeTorchLoadSafePath:
    """Tensor-only checkpoints load without trust=True."""

    def test_tensor_only_checkpoint_loads(self, tmp_path: Path) -> None:
        """Pure-tensor checkpoint succeeds on the first safe-load attempt."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_tensor_only_checkpoint(ckpt_path)

        result = _safe_torch_load(ckpt_path)

        assert "model" in result
        assert torch.allclose(result["model"]["weight"], torch.tensor([1.0, 2.0]))

    def test_accepts_pathlib_path(self, tmp_path: Path) -> None:
        """Helper accepts a :class:`pathlib.Path` argument without error."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_tensor_only_checkpoint(ckpt_path)

        result = _safe_torch_load(ckpt_path)

        assert "model" in result

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Helper accepts a :class:`str` path argument without error."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_tensor_only_checkpoint(ckpt_path)

        result = _safe_torch_load(str(ckpt_path))

        assert "model" in result


# ---------------------------------------------------------------------------
# Safe-globals fallback (legacy Namespace checkpoints)
# ---------------------------------------------------------------------------


class TestSafeTorchLoadSafeGlobals:
    """Checkpoints with argparse.Namespace / SimpleNamespace load without trust=True."""

    def test_argparse_namespace_loads_without_trust(self, tmp_path: Path) -> None:
        """argparse.Namespace checkpoint succeeds via the safe-globals retry."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_namespace_checkpoint(ckpt_path)

        result = _safe_torch_load(ckpt_path)

        assert isinstance(result["args"], argparse.Namespace)
        assert result["args"].num_classes == 80

    def test_simple_namespace_loads_without_trust(self, tmp_path: Path) -> None:
        """SimpleNamespace checkpoint succeeds via the safe-globals retry."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_simple_namespace_checkpoint(ckpt_path)

        result = _safe_torch_load(ckpt_path)

        assert isinstance(result["args"], SimpleNamespace)


# ---------------------------------------------------------------------------
# Arbitrary pickle — trust=False must raise, trust=True must succeed
# ---------------------------------------------------------------------------


class TestSafeTorchLoadTrustGate:
    """Arbitrary-pickle checkpoints require explicit trust=True."""

    def test_arbitrary_pickle_raises_without_trust(self, tmp_path: Path) -> None:
        """Checkpoint with unknown Python object raises RuntimeError when trust=False."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_arbitrary_pickle_checkpoint(ckpt_path)

        with pytest.raises(RuntimeError, match="trust_checkpoint=True"):
            _safe_torch_load(ckpt_path, trust=False)

    def test_arbitrary_pickle_raises_by_default(self, tmp_path: Path) -> None:
        """Checkpoint with unknown Python object raises RuntimeError when trust omitted (default=False)."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_arbitrary_pickle_checkpoint(ckpt_path)

        with pytest.raises(RuntimeError, match="trust_checkpoint=True"):
            _safe_torch_load(ckpt_path)

    def test_arbitrary_pickle_succeeds_with_trust(self, tmp_path: Path) -> None:
        """Checkpoint with unknown Python object loads when trust=True."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_arbitrary_pickle_checkpoint(ckpt_path)

        result = _safe_torch_load(ckpt_path, trust=True)

        assert "model" in result

    def test_trust_true_emits_warning(self, tmp_path: Path) -> None:
        """Trust=True triggers a UserWarning about unsafe loading."""
        ckpt_path = tmp_path / "ckpt.pth"
        _write_arbitrary_pickle_checkpoint(ckpt_path)

        with pytest.warns(UserWarning, match="weights_only=False"):
            _safe_torch_load(ckpt_path, trust=True)
