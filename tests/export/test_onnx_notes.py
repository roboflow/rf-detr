# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the ``notes`` parameter in :func:`~rfdetr.export._onnx.exporter.export_onnx`."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx", reason="onnx not installed; skip ONNX notes tests")


from rfdetr.export._onnx.exporter import export_onnx  # noqa: E402


class _TinyModel(nn.Module):
    """Minimal model that can be exported to ONNX."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a trivial identity-like forward pass.

        Args:
            x: Input tensor.

        Returns:
            Input tensor unchanged.
        """
        return x


def _export_tiny_model(tmp_path: Path, notes: object = None) -> str:
    """Export a tiny model to ONNX and return the output file path.

    Args:
        tmp_path: Temporary directory provided by pytest.
        notes: Optional notes to embed in the ONNX file.

    Returns:
        Path to the exported ONNX file.
    """
    model = _TinyModel().eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    return export_onnx(
        output_dir=str(tmp_path),
        model=model,
        input_names=["input"],
        input_tensors=input_tensor,
        output_names=["output"],
        dynamic_axes=None,
        verbose=False,
        notes=notes,
    )


class TestExportOnnxNotes:
    """Verify ``notes`` metadata round-trips through the ONNX export."""

    @pytest.mark.parametrize(
        "notes, expected_value",
        [
            pytest.param("simple string", "simple string", id="string"),
            pytest.param(
                {"date": "2026-01-01", "labeller": "Alice"},
                '{"date": "2026-01-01", "labeller": "Alice"}',
                id="dict",
            ),
            pytest.param(["class_a", "class_b"], '["class_a", "class_b"]', id="list"),
            pytest.param(42, "42", id="int"),
        ],
    )
    def test_notes_embedded_in_onnx_metadata(self, tmp_path: Path, notes: object, expected_value: str) -> None:
        """Notes are stored as the 'notes' metadata_props entry in the ONNX model."""
        output_file = _export_tiny_model(tmp_path, notes=notes)

        model = onnx.load(output_file)
        meta = {prop.key: prop.value for prop in model.metadata_props}
        assert "notes" in meta
        assert meta["notes"] == expected_value

    def test_string_notes_stored_verbatim_without_json_wrapping(self, tmp_path: Path) -> None:
        """Plain string notes must be stored as-is, not double-encoded as JSON."""
        notes = "my run description"
        output_file = _export_tiny_model(tmp_path, notes=notes)

        model = onnx.load(output_file)
        meta = {prop.key: prop.value for prop in model.metadata_props}
        assert meta["notes"] == "my run description"

    def test_dict_notes_round_trip_via_json(self, tmp_path: Path) -> None:
        """Dict notes deserialise back to the original dict via json.loads."""
        notes = {"project": "ceramics", "batch": 7}
        output_file = _export_tiny_model(tmp_path, notes=notes)

        model = onnx.load(output_file)
        meta = {prop.key: prop.value for prop in model.metadata_props}
        assert json.loads(meta["notes"]) == notes

    def test_no_notes_metadata_when_notes_is_none(self, tmp_path: Path) -> None:
        """When notes=None (default), no 'notes' metadata entry is written."""
        output_file = _export_tiny_model(tmp_path, notes=None)

        model = onnx.load(output_file)
        meta = {prop.key: prop.value for prop in model.metadata_props}
        assert "notes" not in meta
