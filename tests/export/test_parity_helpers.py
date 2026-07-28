# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Error-path coverage for the shared export-parity helpers in ``tests/export/conftest.py``.

These guards only fire when a backend diverges structurally (output count/shape mismatch, an empty forward, or a missing
fixture image), so the green-path end-to-end suites never exercise them. The checks here are backend-agnostic and CPU-
only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.export.conftest import (
    _parity_input_from_image,
    eager_reference_tensors,
    max_abs_output_diffs,
)


class _NoTensorModule(torch.nn.Module):
    """Stub export-mode module whose forward yields no tensor outputs."""

    def forward(self, x: torch.Tensor) -> tuple[()]:
        """Return an empty output tuple regardless of input."""
        return ()


class TestMaxAbsOutputDiffs:
    """Guards on positional pairing of eager and backend output tensors."""

    def test_count_mismatch_raises(self) -> None:
        """A differing number of eager vs backend tensors must raise AssertionError."""
        eager = [torch.zeros(2, 2)]
        other = [torch.zeros(2, 2), torch.zeros(2, 2)]
        with pytest.raises(AssertionError, match="output count"):
            max_abs_output_diffs(eager, other)

    def test_shape_mismatch_raises(self) -> None:
        """Paired outputs with differing shapes must raise AssertionError when check_shape is set."""
        eager = [torch.zeros(2, 2)]
        other = [torch.zeros(2, 3)]
        with pytest.raises(AssertionError, match="shape mismatch"):
            max_abs_output_diffs(eager, other, check_shape=True)


class TestEagerReferenceTensors:
    """Guard on the eager reference forward producing comparable tensors."""

    def test_no_tensor_outputs_raises(self) -> None:
        """A model that yields no tensor outputs must raise AssertionError."""
        with pytest.raises(AssertionError, match="no tensor outputs"):
            eager_reference_tensors(_NoTensorModule(), torch.zeros(1, 3, 8, 8))


class TestParityInputFromImage:
    """Guard on the image-input builder for missing fixture files."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A path that does not exist must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _parity_input_from_image(tmp_path / "missing.png", 64)
