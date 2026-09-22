# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.compiler — version-compatible compile-state predicate."""

import pytest
import torch

from rfdetr.utilities.compiler import is_compiling

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
