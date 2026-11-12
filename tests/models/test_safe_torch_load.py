# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``_safe_torch_load`` helper in ``rfdetr.utilities.io``.

Covers the three-stage safe-load strategy: strict weights_only, safe-globals fallback, and opt-in pickle fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rfdetr.utilities.io import _safe_torch_load


class _ArbitraryObject:
    """Module-level object that torch.save can pickle but weights_only=True rejects.

    Must be defined at module scope so pickle can resolve its fully-qualified name during serialisation (local/nested
    classes cannot be pickled by torch.save).
    """

    value = 42


@pytest.fixture
def tensor_checkpoint(tmp_path: Path) -> Path:
    """Save a checkpoint containing only tensors and plain dicts and return its path.

    Examples:
        Pytest fixtures cannot be called directly outside fixture injection.

        >>> tensor_checkpoint(None)  # doctest: +SKIP
    """
    path = tmp_path / "ckpt.pth"
    torch.save({"model": {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.0])}}, path)
    return path


@pytest.fixture
def pickle_checkpoint(tmp_path: Path) -> Path:
    """Save a checkpoint that embeds an arbitrary class (requires pickle) and return its path.

    Examples:
        Pytest fixtures cannot be called directly outside fixture injection.

        >>> pickle_checkpoint(None)  # doctest: +SKIP
    """
    path = tmp_path / "ckpt.pth"
    torch.save({"model": {"weight": torch.tensor([1.0])}, "extra": _ArbitraryObject()}, path)
    return path


class TestSafeTorchLoadSafePath:
    """Tensor-only checkpoints load without trust=True."""

    @pytest.mark.parametrize("path_type", [Path, str])
    def test_tensor_only_checkpoint_loads(self, tensor_checkpoint: Path, path_type: type) -> None:
        """Pure-tensor checkpoint succeeds on the first safe-load attempt for ``Path`` and ``str`` paths."""
        result = _safe_torch_load(path_type(tensor_checkpoint))

        assert "model" in result
        assert torch.allclose(result["model"]["weight"], torch.tensor([1.0, 2.0]))


class TestSafeTorchLoadSafeGlobals:
    """Checkpoints with argparse.Namespace / SimpleNamespace load without trust=True.

    Legacy RF-DETR engine.py checkpoints embed a Namespace; strict ``weights_only=True`` (without safe globals) would
    reject these.
    """

    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(argparse.Namespace(pretrain_weights="rf-detr-small.pth", num_classes=80), id="argparse"),
            pytest.param(SimpleNamespace(pretrain_weights="rf-detr-small.pth", num_classes=80), id="simple"),
        ],
    )
    def test_namespace_args_load_without_trust(self, tmp_path: Path, args: object) -> None:
        """A checkpoint whose ``args`` is a namespace loads via the safe-globals retry with its fields intact."""
        ckpt_path = tmp_path / "ckpt.pth"
        torch.save({"model": {"weight": torch.tensor([1.0])}, "args": args}, ckpt_path)

        result = _safe_torch_load(ckpt_path)

        assert type(result["args"]) is type(args)
        assert vars(result["args"]) == vars(args)


class TestSafeTorchLoadTrustGate:
    """Arbitrary-pickle checkpoints require explicit trust=True."""

    @pytest.mark.parametrize(
        "trust_kwargs",
        [pytest.param({"trust": False}, id="explicit_false"), pytest.param({}, id="default")],
    )
    def test_arbitrary_pickle_raises_without_trust(self, pickle_checkpoint: Path, trust_kwargs: dict) -> None:
        """A checkpoint with an unknown Python object raises RuntimeError when trust is False or omitted."""
        with pytest.raises(RuntimeError) as exc_info:
            _safe_torch_load(pickle_checkpoint, **trust_kwargs)
        assert "trust_checkpoint=True" in str(exc_info.value)

    def test_arbitrary_pickle_succeeds_with_trust(self, pickle_checkpoint: Path) -> None:
        """A checkpoint with an unknown Python object loads when trust=True."""
        result = _safe_torch_load(pickle_checkpoint, trust=True)

        assert "model" in result

    def test_trust_true_emits_warning(self, pickle_checkpoint: Path) -> None:
        """Trust=True triggers a UserWarning about unsafe loading."""
        with pytest.warns(UserWarning) as record:
            _safe_torch_load(pickle_checkpoint, trust=True)

        assert any("weights_only=False" in str(warning.message) for warning in record)
