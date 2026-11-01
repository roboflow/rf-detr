# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``notes`` setting on :class:`~rfdetr.export._onnx.exporter.OnnxConfig`, embedded by
``OnnxExporter``."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export.prepare import ExportGraph
from tests._markers import _requires_onnx

# The helper doctest exports a real ONNX model, so it needs the package a class-level skipif cannot gate.
__doctest_requires__ = {("_export_tiny_model", "_embedded_metadata"): ["onnx"]}

#: A list that contains itself: JSON cannot encode it, and ``json.dumps`` reports it as a ``ValueError``.
_CIRCULAR_NOTES: list[object] = []
_CIRCULAR_NOTES.append(_CIRCULAR_NOTES)


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

    Examples:
        ``export_onnx`` logs its success at INFO, and rf-detr's stdout handler re-resolves
        ``sys.stdout`` per record, so without the redirect the log line lands in doctest's captured
        output and is compared against the expected value.

        >>> import contextlib
        >>> import io
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        ...     out = _export_tiny_model(Path(d))
        >>> Path(out).suffix
        '.onnx'
    """
    model = _TinyModel().eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    graph = ExportGraph(
        model=model,
        input_tensors=input_tensor,
        input_names=("input",),
        output_names=("output",),
        dynamic_axes=None,
        shape=(32, 32),
        backbone_only=False,
    )
    return str(OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False, notes=notes))(graph))


def _embedded_metadata(onnx_path: str) -> dict[str, str]:
    """Return the ``metadata_props`` of the ONNX file at *onnx_path* as a key/value dict.

    Args:
        onnx_path: Path to an exported ONNX file.

    Returns:
        Mapping of each metadata property key to its string value.

    Examples:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        ...     meta = _embedded_metadata(_export_tiny_model(Path(d), notes="hi"))
        >>> meta["rfdetr_notes"]
        'hi'
    """
    import onnx

    return {prop.key: prop.value for prop in onnx.load(onnx_path).metadata_props}


@_requires_onnx
class TestExportOnnxNotes:
    """Verify ``notes`` metadata round-trips through the ONNX export."""

    @pytest.mark.parametrize(
        ("notes", "expected_value"),
        [
            pytest.param("simple string", "simple string", id="string"),
            pytest.param(
                {"date": "2026-01-01", "labeller": "Alice"},
                '{"date": "2026-01-01", "labeller": "Alice"}',
                id="dict",
            ),
            pytest.param(["class_a", "class_b"], '["class_a", "class_b"]', id="list"),
            pytest.param(42, "42", id="int"),
            pytest.param("Reviewer: Łukasz · 2026-Q2 · ✅", "Reviewer: Łukasz · 2026-Q2 · ✅", id="unicode"),
        ],
    )
    def test_notes_embedded_in_onnx_metadata(self, tmp_path: Path, notes: object, expected_value: str) -> None:
        """Notes are stored as the 'rfdetr_notes' metadata entry: strings verbatim (unicode intact), the rest as
        JSON."""
        meta = _embedded_metadata(_export_tiny_model(tmp_path, notes=notes))

        assert "rfdetr_notes" in meta
        assert meta["rfdetr_notes"] == expected_value

    def test_dict_notes_round_trip_via_json(self, tmp_path: Path) -> None:
        """Dict notes deserialise back to the original dict via json.loads."""
        notes = {"project": "ceramics", "batch": 7}

        meta = _embedded_metadata(_export_tiny_model(tmp_path, notes=notes))

        assert json.loads(meta["rfdetr_notes"]) == notes

    def test_no_notes_metadata_when_notes_is_none(self, tmp_path: Path) -> None:
        """When notes=None (default), no 'rfdetr_notes' metadata entry is written."""
        meta = _embedded_metadata(_export_tiny_model(tmp_path, notes=None))

        assert "rfdetr_notes" not in meta

    @pytest.mark.parametrize(
        "notes",
        [
            pytest.param("", id="empty_string"),
            pytest.param({}, id="empty_dict"),
            pytest.param([], id="empty_list"),
            pytest.param(0, id="zero"),
            pytest.param(False, id="false"),
        ],
    )
    def test_falsy_notes_still_embedded(self, tmp_path: Path, notes: object) -> None:
        """Falsy but non-None notes are embedded; guard is 'is not None', not truthiness."""
        meta = _embedded_metadata(_export_tiny_model(tmp_path, notes=notes))

        assert "rfdetr_notes" in meta

    def test_embedding_notes_does_not_read_a_module_level_onnx_binding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Notes still reach the file when the exporter module carries a stale ``onnx = None`` binding.

        The exporter module used to bind ``onnx`` at import, to ``None`` without the package, and ``_embed_notes`` read
        it, so a notebook that installed ``onnx`` after the import lost its notes. ``_embed_notes`` now imports ``onnx``
        when it runs and nothing binds the name at module level; the planted binding fails this test only if a read of
        such a binding comes back. The late-install scenario itself is covered by
        ``test_onnx_installed_after_a_refused_check_is_found``.
        """
        monkeypatch.setattr("rfdetr.export._onnx.exporter.onnx", None, raising=False)

        meta = _embedded_metadata(_export_tiny_model(tmp_path, notes={"run": 1}))

        assert meta["rfdetr_notes"] == '{"run": 1}'

    @pytest.mark.parametrize(
        "notes, error",
        [
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            pytest.param({"scores": [float("nan")]}, ValueError, id="nested-nan"),
            pytest.param({1, 2}, TypeError, id="set"),
            pytest.param(_CIRCULAR_NOTES, ValueError, id="circular"),
        ],
    )
    def test_unserializable_notes_are_refused_at_construction(
        self, tmp_path: Path, notes: object, error: type[Exception]
    ) -> None:
        """A value the metadata slot cannot hold is refused when the exporter is built, before any trace."""
        with pytest.raises(error, match="notes"):
            OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False, notes=notes))
