# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for :meth:`rfdetr.detr.RFDETR.export` — the public facade over the per-format exporter classes.

Use cases covered:
- Shape and patch-size validation must reject an unexportable request before any conversion work happens.
- An invalid format setting, ``notes`` value or ``batch_size`` must be refused before the model's forward pass.
- A missing optional package must be reported before any model work, and a refusal must leave no side effects.
- The export-mode switch must be safe to apply to every module an exporter is handed, backbone wrappers included.
- Every symbol these tests monkeypatch must stay on the real call path, not merely remain importable.
"""

import os
import subprocess
import sys
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr import detr as _detr_module
from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._tensorrt.exporter import _BuiltEngine
from rfdetr.export.base import Exporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.export.registry import REGISTRY, resolve_exporter
from tests.export.helpers import (
    _disable_metadata_publication,
    _make_export_model,
    _make_mock_infer_tensor,
    _make_tensorrt_export_model,
)


@pytest.fixture(autouse=True)
def _skip_metadata_publication_for_export_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep facade tests focused on conversion when their fake artifacts have no file bytes.

    The metadata round-trip tests use the real writer in ``tests/export/core/test_metadata.py``.
    This fixture does not replace ``metadata_from_model``: fake RFDETR models still need a valid prediction contract.

    Examples:
        Requires pytest's monkeypatch lifecycle, so this example cannot run standalone:
        >>> _skip_metadata_publication_for_export_stubs(None)  # doctest: +SKIP
    """
    _disable_metadata_publication(monkeypatch)


def test_exporter_check_dependencies_defaults_to_a_no_op() -> None:
    """The base hook does nothing, so a format that needs no optional package has nothing to override.

    ``RFDETR.export()`` and ``Exporter.__call__`` call the hook for every format; each format shipped today overrides it
    with the check its own packages need.
    """
    assert Exporter.check_dependencies() is None


def test_exporter_check_environment_defaults_to_a_no_op(tmp_path: Path) -> None:
    """The base hook does nothing, so a format whose settings need only its packages has nothing to override."""
    assert Exporter.check_environment(OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False))) is None


class TestExportPatchSize:
    """RFDETR.export() patch_size validation and shape-divisibility tests."""

    @staticmethod
    def _scaffold(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_size: int, num_windows: int
    ) -> types.SimpleNamespace:
        """Build a minimal RFDETR-like namespace with controllable patch_size/num_windows."""
        model = _make_export_model(
            resolution=patch_size * num_windows * 2,
            patch_size=patch_size,
            num_windows=num_windows,
        )

        def _fake_make_infer_image(*_a, **_kw):
            return torch.zeros(1, 3, 8, 8)

        def _fake_convert(_self, _graph):
            return str(tmp_path / "inference_model.onnx")

        monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", _fake_convert)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
        return model

    def test_patch_size_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """export(patch_size=X) must raise ValueError when X != model_config.patch_size."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=4)
        with pytest.raises(ValueError, match="patch_size"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=16)

    @pytest.mark.parametrize("bad_patch_size", [0, -1])
    def test_invalid_patch_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_patch_size: int
    ) -> None:
        """Export() must raise ValueError when patch_size is not a positive integer."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=4)
        # Keep model_config.patch_size consistent with the patch_size argument for this test
        model.model_config.patch_size = bad_patch_size
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=bad_patch_size)

    def test_shape_must_be_divisible_by_block_size(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Export() must reject shapes not divisible by patch_size * num_windows."""
        # patch_size=16, num_windows=2 → block_size=32; shape (48, 64): 48 % 32 != 0
        model = self._scaffold(monkeypatch, tmp_path, patch_size=16, num_windows=2)
        with pytest.raises(ValueError, match="divisible by 32"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(48, 64))

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((-64, 64), id="negative_height"),
            pytest.param((64, -64), id="negative_width"),
            pytest.param((0, 64), id="zero_height"),
            pytest.param((64, 0), id="zero_width"),
        ],
    )
    def test_non_positive_shape_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_shape: tuple[int, int]
    ) -> None:
        """Export() must reject non-positive shape dimensions (Python -N % M == 0 wraps silently)."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=16, num_windows=2)
        with pytest.raises(ValueError, match="positive integers"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=bad_shape)

    @pytest.mark.parametrize(
        "patch_size, num_windows, export_kwargs",
        [
            pytest.param(16, 2, {"shape": (64, 64)}, id="shape-divisible-by-block-size"),
            pytest.param(14, 4, {"patch_size": 14}, id="patch-size-matches-config"),
        ],
    )
    def test_accepts_valid_patch_size_and_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        patch_size: int,
        num_windows: int,
        export_kwargs: dict[str, object],
    ) -> None:
        """Export() accepts a shape divisible by patch_size * num_windows and an explicit patch_size matching the
        config.

        patch_size=16, num_windows=2 gives block_size=32, so (64, 64) is valid; patch_size=14, num_windows=4 gives
        block_size=56 and the scaffold's resolution of 112 (56 * 2).
        """
        model = self._scaffold(monkeypatch, tmp_path, patch_size=patch_size, num_windows=num_windows)

        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), **export_kwargs)

    @pytest.mark.parametrize("bad_patch_size", [True, False])
    def test_bool_patch_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_patch_size: bool
    ) -> None:
        """Export() must reject bool values for patch_size (isinstance(True, int) is True)."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=bad_patch_size)

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((14.0, 14.0), id="float_dims"),
            pytest.param((14,), id="wrong_arity_one_element"),
            pytest.param((14, 14, 3), id="wrong_arity_three_elements"),
            pytest.param((True, 14), id="bool_height"),
            pytest.param((14, False), id="bool_width"),
        ],
    )
    def test_invalid_shape_type_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_shape: tuple) -> None:
        """Export() must raise ValueError for float, bool, or wrong-arity shape tuples."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        with pytest.raises(ValueError, match="shape"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=bad_shape)

    @pytest.mark.parametrize("bad_num_windows", [0, -1, True])
    def test_invalid_num_windows_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_num_windows: int
    ) -> None:
        """Export() must raise ValueError when model_config.num_windows is not a positive integer."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        model.model_config.num_windows = bad_num_windows
        with pytest.raises(ValueError, match="num_windows must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path))

    def test_default_resolution_not_divisible_by_block_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Export() with shape=None must raise ValueError when model.resolution % block_size != 0."""
        # patch_size=14, num_windows=3 → block_size=42; scaffold sets resolution=84 (42*2) which is valid
        # Override resolution to 50 (not divisible by 42) to trigger the check
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=3)
        model.model.resolution = 50
        with pytest.raises(ValueError, match="default resolution"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path))


def _make_export_graph(*, backbone_only: bool = False) -> ExportGraph:
    """Build a minimal prepared graph an exporter can be handed without a real model.

    Args:
        backbone_only: Whether the graph stands in for a backbone-only export.

    Returns:
        An :class:`ExportGraph` wrapping a plain ``Sequential``, which — like the real ``_BackboneExport``
        wrapper — exposes no ``export`` method for the export-mode switch to call.

    Examples:
        >>> _make_export_graph(backbone_only=True).backbone_only
        True
    """
    return ExportGraph(
        model=torch.nn.Sequential(torch.nn.Identity()),
        input_tensors=torch.zeros(1, 3, 32, 32),
        input_names=("input",),
        output_names=("dets", "labels"),
        dynamic_axes=None,
        shape=(32, 32),
        backbone_only=backbone_only,
    )


@pytest.mark.parametrize("format", ["coreml", "executorch"])
def test_prepared_backbone_module_reaches_the_exporter(mocker: MockerFixture, tmp_path: Path, format: str) -> None:
    """An exporter accepts a prepared backbone graph, whose module exposes no ``export`` method.

    Backbone-only exports hand over a wrapper rather than the detector, and the wrapper deliberately has no export-mode
    switch to call — a format that assumed otherwise would raise before writing anything.
    """
    graph = _make_export_graph(backbone_only=True)
    backend = "xnnpack" if format == "executorch" else None
    exporter_class = resolve_exporter(format)
    config = exporter_class.build_config(output_dir=tmp_path, variant_name="nano", verbose=False, backend=backend)
    exporter = exporter_class(config)
    output_path = tmp_path / "backbone"

    mocker.patch.object(type(exporter), "check_dependencies")
    convert = mocker.patch.object(type(exporter), "_convert", return_value=output_path)

    result = exporter(graph)

    assert result == output_path
    assert convert.call_args.args[0] is graph


class TestSwitchToExportMode:
    """``_switch_to_export_mode`` — the shared export-mode choke point for the non-ONNX backends.

    RF-DETR's ``export()`` implementations are not all idempotent: ``LWDETR``, ``Backbone`` and
    ``PositionEmbeddingSine`` each stash ``self._forward_origin = self.forward`` before swapping in
    ``forward_export``, so calling ``export()`` twice replaces the saved original with the export
    forward and loses the real one. These tests pin the guard that makes a second switch a no-op.
    """

    class _SwitchCountingModel(torch.nn.Module):
        """Minimal stand-in reproducing the models' unguarded save-then-swap export mode.

        Examples:
            >>> model = TestSwitchToExportMode._SwitchCountingModel()
            >>> model._export
            False
        """

        def __init__(self) -> None:
            super().__init__()
            self._export = False
            self.switches = 0
            self._forward_origin = None

        def export(self) -> None:
            """Record the switch and stash the current forward, exactly as the real models do."""
            self._export = True
            self.switches += 1
            self._forward_origin = self.forward

    @pytest.mark.parametrize(
        "switch",
        [
            pytest.param(_switch_to_export_mode, id="first-call"),
            pytest.param(
                lambda model: (_switch_to_export_mode(model), _switch_to_export_mode(model)), id="second-call"
            ),
        ],
    )
    def test_switches_exactly_once(self, switch: Callable[[torch.nn.Module], object]) -> None:
        """A fresh model is switched by the first call, and a model already in export mode is left alone afterwards.

        One call is the ordinary path of the ExecuTorch, CoreML and OpenVINO dispatchers, which receive a freshly
        deepcopied module from ``RFDETR.export()``. A second ``export()`` would overwrite ``_forward_origin`` with
        ``forward_export``, silently and unrecoverably, and two switches become reachable as soon as one exporter
        composes another (a TFLite or TensorRT export running an ONNX export internally).
        """
        model = self._SwitchCountingModel()
        switch(model)
        assert model.switches == 1

    def test_leaves_a_module_without_export_untouched(self) -> None:
        """A plain ``nn.Module`` with no ``export`` attribute must pass through without raising.

        Backbone-only exports hand the dispatchers a ``_BackboneExport`` wrapper, which exposes no ``export`` method —
        that path must stay a silent no-op rather than an ``AttributeError``.
        """
        model = torch.nn.Linear(2, 2)
        _switch_to_export_mode(model)
        assert not hasattr(model, "_export")

    @pytest.mark.parametrize(
        "prepare",
        [
            pytest.param(_switch_to_export_mode, id="already-switched"),
            pytest.param(lambda _model: None, id="fresh"),
        ],
    )
    def test_onnx_exporter_switches_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare: Callable[[torch.nn.Module], object]
    ) -> None:
        """``OnnxExporter`` shares the guarded choke point: a switched model is not switched again, a fresh one is.

        A TFLite or TensorRT export runs an ONNX export internally after the model was already switched, so an unguarded
        ``model.export()`` here would overwrite ``_forward_origin`` with the export forward.
        """
        model = self._SwitchCountingModel()
        prepare(model)
        monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: None)
        graph = ExportGraph(
            model=model,
            input_tensors=torch.zeros(1, 2),
            input_names=("input",),
            output_names=("output",),
            dynamic_axes=None,
            shape=(2, 2),
            backbone_only=False,
        )

        OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False))(graph)

        assert model.switches == 1


def _stub_export_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, export_format: str
) -> dict[str, MagicMock]:
    """Replace every heavy step of ``RFDETR.export()`` with a recording mock and return them by patch target.

    Covers the seams a *format* needs on top of the format-independent ones, so a caller can run a complete
    ``RFDETR.export()`` without ONNX, TensorRT or onnx2tf installed and then assert on any individual seam.

    Args:
        monkeypatch: Fixture used to install the stubs for the duration of one test.
        tmp_path: Directory the stubbed artifact paths are rooted in.
        export_format: Normalized export format the caller is about to run.

    Returns:
        Mapping of patch target to the mock installed there.

    Examples:
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as monkeypatch:
        ...     stubs = _stub_export_dependencies(monkeypatch, Path(directory), export_format="onnx")
        ...     "rfdetr.export._onnx.exporter.OnnxExporter._convert" in stubs
        True
    """
    onnx_path = str(tmp_path / "inference_model.onnx")
    stubs: dict[str, MagicMock] = {
        "rfdetr.detr.deepcopy": MagicMock(side_effect=lambda module: module),
        "rfdetr.export.prepare.make_infer_image": MagicMock(side_effect=lambda *_a, **_kw: _make_mock_infer_tensor()),
        "rfdetr.export._onnx.exporter.OnnxExporter._convert": MagicMock(return_value=onnx_path),
        "rfdetr.export._backend._resolve_export_backend": MagicMock(return_value=(None, None)),
        "rfdetr.export._onnx.exporter.OnnxExporter.check_dependencies": MagicMock(return_value=None),
    }
    if export_format == "tensorrt":
        stubs["rfdetr.export._tensorrt.exporter.TensorRTExporter._build"] = MagicMock(
            return_value=_BuiltEngine(path=str(tmp_path / "inference_model.trt"), fp16=True, engine_facts=None)
        )
        # The build is stubbed, so the host's TensorRT install (none in CPU CI) must not decide the outcome.
        monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
        stubs["rfdetr.export._tensorrt.exporter.TensorRTExporter.check_dependencies"] = MagicMock(return_value=None)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_POLYGRAPHY_AVAILABLE", True)
    if export_format == "tflite":
        stubs["rfdetr.export._backend.preload_tensorflow_before_onnx"] = MagicMock(return_value=None)
        stubs["rfdetr.export._tflite.exporter.TFLiteExporter.check_dependencies"] = MagicMock(return_value=None)
        stubs["rfdetr.export._tflite.exporter.TFLiteExporter.convert_onnx"] = MagicMock(
            return_value=tmp_path / "inference_model_fp32.tflite"
        )
        # A fake flatbuffer has no signature for the metadata publication step to inspect.
        stubs["rfdetr.export._tflite.exporter.TFLiteExporter._metadata_for_artifact"] = MagicMock(
            side_effect=lambda metadata, _path: metadata
        )
    for target, stub in stubs.items():
        monkeypatch.setattr(target, stub)
    return stubs


class TestExportSeamInventory:
    """Every symbol the export tests monkeypatch must stay on ``RFDETR.export()``'s call path.

    These symbols are patched across the export test files to stub out slow or optional-dependency work. Moving code
    between modules can leave such a symbol importable — so ``mock.patch`` still resolves it and raises nothing — while
    the production call path no longer goes through it. The test that patched it then keeps passing while the real, slow
    implementation runs underneath. Existence is therefore not enough: each case below patches one seam, runs a full
    export, and asserts the mock was actually invoked.
    """

    @pytest.mark.parametrize(
        "seam, export_format",
        [
            ("rfdetr.detr.deepcopy", "onnx"),
            ("rfdetr.export.prepare.make_infer_image", "onnx"),
            ("rfdetr.export._onnx.exporter.OnnxExporter._convert", "onnx"),
            ("rfdetr.export._backend._resolve_export_backend", "onnx"),
            ("rfdetr.export._tensorrt.exporter.TensorRTExporter._build", "tensorrt"),
            ("rfdetr.export._backend.preload_tensorflow_before_onnx", "tflite"),
            ("rfdetr.export._onnx.exporter.OnnxExporter.check_dependencies", "onnx"),
            ("rfdetr.export._tensorrt.exporter.TensorRTExporter.check_dependencies", "tensorrt"),
            ("rfdetr.export._tflite.exporter.TFLiteExporter.check_dependencies", "tflite"),
        ],
    )
    def test_seam_is_invoked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seam: str, export_format: str
    ) -> None:
        """The patched symbol is reached by a real ``RFDETR.export()`` run, not merely importable."""
        stubs = _stub_export_dependencies(monkeypatch, tmp_path, export_format=export_format)

        _detr_module.RFDETR.export(
            _make_tensorrt_export_model(), output_dir=str(tmp_path), format=export_format, shape=(14, 14)
        )

        assert stubs[seam].called, f"{seam} is importable but never called during a format={export_format!r} export"

    @pytest.mark.parametrize(
        "export_format, expected_name",
        [
            ("onnx", "inference_model.onnx"),
            ("tensorrt", "inference_model.trt"),
            ("tflite", "inference_model_fp32.tflite"),
        ],
    )
    def test_returns_the_converter_artifact_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, export_format: str, expected_name: str
    ) -> None:
        """``RFDETR.export()`` returns the path its converter produced, unmodified.

        Users glob the returned filename, so the facade must propagate whatever the format's converter named rather than
        re-deriving a name of its own — which is exactly what a dispatch rewrite can silently start doing.
        """
        _stub_export_dependencies(monkeypatch, tmp_path, export_format=export_format)

        result = _detr_module.RFDETR.export(
            _make_tensorrt_export_model(), output_dir=str(tmp_path), format=export_format, shape=(14, 14)
        )

        assert Path(result).name == expected_name


class TestExportRejectsBeforeForwardPass:
    """An invalid ``RFDETR.export()`` argument is refused before the model's forward pass, not after it.

    ``RFDETR.export`` checks its format-independent arguments before it resolves an exporter, and constructing the
    exporter runs the format's ``_check_capabilities``; both happen before ``prepare_export_graph`` runs the model. A
    check placed where a value is first used, inside ``_convert``, fires only after that forward pass, and for the two-
    stage formats after the whole ONNX stage as well. Calling an exporter directly on a hand-built graph cannot tell the
    two apart, so these cases go through the public method with every converter stubbed and then assert that the example
    input for the forward pass was never built.
    """

    @pytest.mark.parametrize(
        "export_format, settings, error, message",
        [
            pytest.param(
                "tflite", {"quantization": "q4"}, ValueError, "Unsupported quantization", id="tflite-quantization"
            ),
            pytest.param(
                "openvino", {"openvino_precision": "int8"}, ValueError, "precision must be", id="openvino-precision"
            ),
            pytest.param(
                "coreai", {"coreai_precision": "int8"}, ValueError, "precision must be", id="coreai-precision"
            ),
            pytest.param(
                "coreml", {"coreml_precision": "int8"}, ValueError, "compute_precision must be", id="coreml-precision"
            ),
            pytest.param(
                "tensorrt",
                {"trt_timing_cache": ""},
                ValueError,
                "non-empty file path",
                id="tensorrt-timing-cache-empty",
            ),
            pytest.param(
                "tensorrt",
                {"trt_timing_cache": "cache/"},
                ValueError,
                "names a directory",
                id="tensorrt-timing-cache-directory",
            ),
            pytest.param(
                "tensorrt",
                {"quantization": "int8"},
                ValueError,
                "calibration_data",
                id="tensorrt-int8-without-calibration-data",
            ),
            pytest.param("onnx", {"notes": float("nan")}, ValueError, "notes", id="notes-nan"),
            pytest.param("tflite", {"notes": float("nan")}, ValueError, "notes", id="tflite-notes-nan"),
            pytest.param("tensorrt", {"notes": float("nan")}, ValueError, "notes", id="tensorrt-notes-nan"),
            pytest.param("coreai", {"notes": float("nan")}, ValueError, "notes", id="coreai-notes-nan"),
            pytest.param("coreml", {"notes": float("nan")}, ValueError, "notes", id="coreml-notes-nan"),
            pytest.param(
                "tensorrt",
                {"trt_hardware_compatibility": "AMPERE_PLUS"},
                ValueError,
                "trt_hardware_compatibility must be",
                id="tensorrt-hardware-compatibility-uppercase",
            ),
            pytest.param(
                "tensorrt",
                {"trt_hardware_compatibility": "ampere"},
                ValueError,
                "trt_hardware_compatibility must be",
                id="tensorrt-hardware-compatibility-unknown",
            ),
            pytest.param(
                "tensorrt",
                {"trt_version_compatible": "yes"},
                ValueError,
                "trt_version_compatible must be a bool",
                id="tensorrt-version-compatible-not-bool",
            ),
            pytest.param("onnx", {"notes": object()}, TypeError, "notes", id="notes-not-json"),
            pytest.param(
                "litert",
                {"quantization": "int8"},
                NotImplementedError,
                "not supported",
                id="litert-quantization",
            ),
        ],
    )
    def test_invalid_setting_never_reaches_the_forward_pass(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export_format: str,
        settings: dict[str, object],
        error: type[Exception],
        message: str,
    ) -> None:
        """The request is refused, and ``make_infer_image``, which builds the forward pass's input, never runs."""
        stubs = _stub_export_dependencies(monkeypatch, tmp_path, export_format=export_format)
        # With the converter stubbed, a check that still lives inside `_convert` never raises at all.
        mocker.patch.object(resolve_exporter(export_format), "_convert", return_value=tmp_path / "artifact")

        with pytest.raises(error) as exc_info:
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(),
                output_dir=str(tmp_path),
                format=export_format,
                shape=(14, 14),
                **settings,
            )

        assert message in str(exc_info.value)
        stubs["rfdetr.export.prepare.make_infer_image"].assert_not_called()

    def test_notes_a_format_drops_are_not_serialized(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A format with no metadata slot warns and drops ``notes``, so a value it never writes is not refused."""
        _stub_export_dependencies(monkeypatch, tmp_path, export_format="openvino")
        mocker.patch.object(resolve_exporter("openvino"), "_convert", return_value=tmp_path / "model.xml")
        mocker.patch.object(resolve_exporter("openvino"), "check_dependencies", return_value=None)

        with pytest.warns(UserWarning, match="`notes` is not forwarded"):
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(),
                output_dir=str(tmp_path),
                format="openvino",
                shape=(14, 14),
                notes=float("nan"),
            )

    def test_invalid_executorch_backend_never_reaches_the_forward_pass(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """An unsupported ``backend`` for ``format='executorch'`` is refused before the forward pass input is built.

        The refusal happens while resolving the backend, even before an exporter class is resolved. This does not fit
        the table above: that table's default stubbing replaces ``_resolve_export_backend`` wholesale (every other
        row's format is backend-agnostic), which would silently discard the invalid ``backend`` this test needs to
        reach the real validation.
        """
        make_infer_image = mocker.patch("rfdetr.export.prepare.make_infer_image")

        with pytest.raises(ValueError, match="Unsupported backend 'nonesuch'"):
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(),
                output_dir=str(tmp_path),
                format="executorch",
                backend="nonesuch",
                shape=(14, 14),
            )

        make_infer_image.assert_not_called()


class TestExportWarningLocation:
    """Warnings ``RFDETR.export`` raises while it builds the exporter point at the caller, not inside ``rfdetr``.

    ``warnings.filterwarnings(..., module=...)`` matches on that location, so a format whose own ``_check_capabilities``
    override adds a frame must not move it.
    """

    @pytest.mark.parametrize(
        "export_format, backend",
        [
            ("tflite", None),
            ("executorch", "xnnpack"),
            ("coreml", None),
            ("coreai", None),
            ("openvino", None),
            ("litert", None),
        ],
    )
    def test_construction_warnings_point_at_the_caller(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export_format: str,
        backend: str | None,
    ) -> None:
        """The experimental and dropped-``notes`` warnings are reported at the line that called ``export``."""
        stubs = _stub_export_dependencies(monkeypatch, tmp_path, export_format=export_format)
        stubs["rfdetr.export._backend._resolve_export_backend"].return_value = (backend, None)
        exporter_class = resolve_exporter(export_format)
        mocker.patch.object(exporter_class, "_convert", return_value=tmp_path / "artifact")
        mocker.patch.object(exporter_class, "check_dependencies", return_value=None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(),
                output_dir=str(tmp_path),
                format=export_format,
                shape=(14, 14),
                notes="provenance",
            )

        reported = {
            warning.filename
            for warning in caught
            if any(text in str(warning.message) for text in ("experimental", "`notes`"))
        }
        assert reported == {__file__}

    def test_refused_request_does_not_warn_first(self, tmp_path: Path) -> None:
        """A request the checks refuse raises without the experimental or dropped-``notes`` warning in front of it."""
        coreml_exporter_class = resolve_exporter("coreml")
        config = coreml_exporter_class.build_config(output_dir=tmp_path, coreml_precision="int8", notes="provenance")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="compute_precision must be"):
                coreml_exporter_class(config)

        assert [str(warning.message) for warning in caught] == []

    def test_refused_shape_does_not_warn_first(self, tmp_path: Path) -> None:
        """A shape the model cannot take is refused before the exporter that would warn is even built."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="divisible"):
                _detr_module.RFDETR.export(
                    _make_tensorrt_export_model(),
                    output_dir=str(tmp_path),
                    format="coreml",
                    shape=(15, 15),
                    notes="provenance",
                )

        assert [str(warning.message) for warning in caught] == []


class TestExportFormatSpelling:
    """``RFDETR.export`` matches the format name in any case, as it already did ``backend``."""

    def test_upper_case_name_reaches_the_exporter(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``format="ONNX"`` runs the ONNX exporter; the case variants themselves are pinned in
        ``tests/export/core/test_registry.py``."""
        stubs = _stub_export_dependencies(monkeypatch, tmp_path, export_format="onnx")

        _detr_module.RFDETR.export(
            _make_tensorrt_export_model(), output_dir=str(tmp_path), format="ONNX", shape=(14, 14)
        )

        stubs["rfdetr.export._onnx.exporter.OnnxExporter._convert"].assert_called_once()

    @pytest.mark.parametrize(
        "export_format, message",
        [
            (123, "Unsupported export format 123"),
            pytest.param(["onnx"], "Unsupported export format ['onnx']", id="list"),
        ],
    )
    def test_non_string_format_is_refused(
        self, mocker: MockerFixture, tmp_path: Path, export_format: object, message: str
    ) -> None:
        """A non-string ``format`` fails with the "unsupported format" message, and no exporter is resolved.

        Neither an int nor a list may surface as an opaque ``KeyError``/``TypeError`` from deep inside the registry
        (such as a set-membership ``TypeError``).
        """
        resolve_exporter_stub = mocker.patch("rfdetr.export.registry.resolve_exporter")

        with pytest.raises(ValueError) as exc_info:
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(), output_dir=str(tmp_path), format=export_format, shape=(14, 14)
            )

        assert message in str(exc_info.value)
        resolve_exporter_stub.assert_not_called()


class TestExportBatchSize:
    """``batch_size`` sizes the example batch for every format, so ``RFDETR.export`` checks it before anything else."""

    @pytest.mark.parametrize(
        "batch_size",
        [
            0,
            -1,
            True,
            pytest.param(np.bool_(True), id="numpy-bool"),
            pytest.param(torch.tensor(True), id="bool-tensor"),
            2.0,
            "2",
            None,
        ],
    )
    def test_invalid_batch_size_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch_size: object
    ) -> None:
        """A value that is not a positive integer is named back instead of failing while the example batch is built."""
        _stub_export_dependencies(monkeypatch, tmp_path, export_format="onnx")

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(), output_dir=str(tmp_path), shape=(14, 14), batch_size=batch_size
            )

    def test_invalid_batch_size_is_refused_before_any_exporter_is_imported(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """No exporter module, and with it no optional dependency, is imported for a batch size that cannot work."""
        resolve_exporter_stub = mocker.patch("rfdetr.export.registry.resolve_exporter")

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            _detr_module.RFDETR.export(_make_tensorrt_export_model(), output_dir=str(tmp_path), batch_size=0)

        resolve_exporter_stub.assert_not_called()

    @pytest.mark.parametrize(
        "batch_size",
        [1, 3, pytest.param(np.int64(3), id="numpy-int64"), 2**31],
    )
    def test_integer_batch_size_sizes_the_example_batch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch_size: int | np.integer
    ) -> None:
        """Every integer type the caller may hold, numpy's included, reaches the example batch as a plain ``int``.

        ``validate_batch_size`` only rejects a value below 1 (see :func:`rfdetr.export.prepare.validate_batch_size`);
        there is no upper bound. The ``2**31`` case documents that an extreme ``batch_size`` is accepted verbatim —
        Python ``int`` has no width to truncate to, unlike the fixed-width counters this size would overflow in
        other languages — rather than silently clamped or wrapped.
        """
        stubs = _stub_export_dependencies(monkeypatch, tmp_path, export_format="onnx")

        _detr_module.RFDETR.export(
            _make_tensorrt_export_model(), output_dir=str(tmp_path), shape=(14, 14), batch_size=batch_size
        )

        example_batch_size = stubs["rfdetr.export.prepare.make_infer_image"].call_args.args[2]
        assert (type(example_batch_size), example_batch_size) == (int, int(batch_size))

    def test_numpy_batch_size_forms_a_tensorrt_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A numpy ``batch_size`` reaches TensorRT's profile check as a plain ``int``, which refused numpy integers."""
        stubs = _stub_export_dependencies(monkeypatch, tmp_path, export_format="tensorrt")

        _detr_module.RFDETR.export(
            _make_tensorrt_export_model(),
            output_dir=str(tmp_path),
            format="tensorrt",
            dynamic_batch=True,
            batch_size=np.int64(2),
            max_batch_size=4,
            shape=(14, 14),
        )

        stubs["rfdetr.export._tensorrt.exporter.TensorRTExporter._build"].assert_called_once()

    def test_numpy_batch_size_over_numpy_max_batch_size_is_refused(self, tmp_path: Path) -> None:
        """A numpy ``batch_size`` over a numpy ``max_batch_size`` still hits the bound error.

        Both are coerced to plain ``int`` via ``validate_batch_size`` before ``TensorRTConfig`` compares them, so a
        numpy ``max_batch_size`` must not slip past the ``batch_size <= max_batch_size`` check the way an un-coerced one
        previously would have (rejected instead as "not a plain int").
        """
        with pytest.raises(ValueError, match="1 <= batch_size <= max_batch_size"):
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(),
                output_dir=str(tmp_path),
                format="tensorrt",
                dynamic_batch=True,
                batch_size=np.int64(8),
                max_batch_size=np.int64(4),
                shape=(14, 14),
            )


def _run_in_fresh_interpreter(script: str, *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run ``script`` with ``python -c`` in a new interpreter that imports the same ``rfdetr`` as this process.

    Examples:
        >>> _run_in_fresh_interpreter("import rfdetr; print('ok')").stdout.strip().splitlines()[-1]
        'ok'
    """
    source_root = str(Path(_detr_module.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")]))}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=env, timeout=timeout
    )


class TestExportDependencyCheck:
    """A format's missing package is reported before the forward pass, naming the extra that installs it.

    Constructing an exporter never checks: a TensorRT ``dry_run`` and the tests that stub a conversion build exporters
    without the format's packages. ``RFDETR.export`` calls ``check_dependencies`` after construction, then
    ``check_environment`` for what the configuration needs of the installed packages, and ``Exporter.__call__`` calls
    both again before the conversion.
    """

    @pytest.mark.parametrize(
        ("setting", "error"),
        [
            pytest.param({"trt_version_compatible": True}, ImportError, id="missing-lean-runtime"),
            pytest.param({"trt_hardware_compatibility": "ampere_plus"}, ValueError, id="missing-hardware-level"),
        ],
    )
    def test_a_configuration_the_host_cannot_build_never_reaches_the_forward_pass(
        self,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        setting: dict[str, object],
        error: type[Exception],
    ) -> None:
        """A TensorRT without the lean runtime or the hardware level refuses the request before the model runs.

        The check needs the configuration, which the ``check_dependencies`` class method does not see.
        """
        make_infer_image = mocker.patch(
            "rfdetr.export.prepare.make_infer_image", side_effect=lambda *_a, **_kw: _make_mock_infer_tensor()
        )
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: str(tmp_path / "model.onnx")
        )
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda module: module)
        monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
        monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_POLYGRAPHY_AVAILABLE", True)
        tensorrt = types.ModuleType("tensorrt")
        tensorrt.__version__ = "11.3.0.99"
        monkeypatch.setitem(sys.modules, "tensorrt", tensorrt)
        monkeypatch.setitem(sys.modules, "tensorrt_lean_libs", None)
        mocker.patch(
            "rfdetr.export._tensorrt.exporter.ctypes.CDLL",
            side_effect=OSError("libnvinfer_lean.so.11: cannot open shared object file"),
        )

        with pytest.raises(error):
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(), output_dir=str(tmp_path), format="tensorrt", shape=(14, 14), **setting
            )

        make_infer_image.assert_not_called()

    def test_a_missing_tensorrt_is_named_before_the_configuration_is_checked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``check_environment`` may import the format's packages, so it runs only once they were found installed.

        A host without TensorRT is told which extra to install, not shown the import error of a portability probe.
        """
        monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "tensorrt", None)

        with pytest.raises(ImportError) as exc_info:
            _detr_module.RFDETR.export(
                _make_tensorrt_export_model(),
                output_dir=str(tmp_path),
                format="tensorrt",
                shape=(14, 14),
                trt_version_compatible=True,
            )

        assert "rfdetr[tensorrt]" in str(exc_info.value)

    def test_an_exporter_handed_a_graph_checks_its_configuration_before_converting(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """``Exporter.__call__`` repeats the check for a caller that skips ``RFDETR.export``, as for packages."""
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False))
        mocker.patch.object(OnnxExporter, "check_dependencies")
        mocker.patch.object(OnnxExporter, "check_environment", side_effect=ImportError("a library is missing"))
        convert = mocker.patch.object(OnnxExporter, "_convert")

        with pytest.raises(ImportError, match="a library is missing"):
            exporter(_make_export_graph())

        convert.assert_not_called()

    def test_missing_onnx_never_reaches_the_forward_pass(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A base install without ``onnx`` learns which extra to install before the model runs, not after the trace."""
        make_infer_image = mocker.patch("rfdetr.export.prepare.make_infer_image")
        monkeypatch.setitem(sys.modules, "onnx", None)

        with pytest.raises(ImportError) as exc_info:
            _detr_module.RFDETR.export(_make_tensorrt_export_model(), output_dir=str(tmp_path), shape=(14, 14))

        assert 'pip install "rfdetr[onnx]"' in str(exc_info.value)
        make_infer_image.assert_not_called()

    def test_onnx_installed_after_a_refused_check_is_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A retry after ``%pip install`` in the same process passes: nothing remembers the first refusal.

        Hiding ``onnx`` from imports stands in for the host before the install. An availability flag or an ``onnx``
        binding fixed when the module loads, such as the TensorRT exporter keeps for polygraphy, would let the first
        check pass here, and would refuse a notebook user's retry until the kernel restarts.
        """
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False))
        with monkeypatch.context() as before_install:
            before_install.setitem(sys.modules, "onnx", None)
            with pytest.raises(ImportError) as exc_info:
                exporter.check_dependencies()
            assert "rfdetr[onnx]" in str(exc_info.value)

        exporter.check_dependencies()

    @pytest.mark.parametrize(
        "init_source, prefix",
        [
            pytest.param(
                'raise ImportError("DLL load failed while importing onnx_cpp2py_export")',
                "DLL load failed",
                id="native-extension-fails",
            ),
            pytest.param(
                "raise ModuleNotFoundError(\"No module named 'onnx.onnx_cpp2py_export'\", "
                'name="onnx.onnx_cpp2py_export")',
                "No module named 'onnx.onnx_cpp2py_export'",
                id="submodule-missing",
            ),
        ],
    )
    def test_broken_onnx_install_raises_its_own_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, init_source: str, prefix: str
    ) -> None:
        """An installed ``onnx`` that fails to import is not reported as missing: reinstalling it would not help."""
        broken_package = tmp_path / "site" / "onnx"
        broken_package.mkdir(parents=True)
        (broken_package / "__init__.py").write_text(init_source + "\n")
        monkeypatch.delitem(sys.modules, "onnx", raising=False)
        monkeypatch.syspath_prepend(str(tmp_path / "site"))
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, verbose=False))

        with pytest.raises(ImportError) as exc_info:
            exporter.check_dependencies()

        assert str(exc_info.value).startswith(prefix)

    @pytest.mark.parametrize(
        "shape, optimized_inplace, error, message",
        [
            pytest.param((15, 15), False, ValueError, "divisible", id="invalid-shape"),
            pytest.param((14, 14), True, RuntimeError, "after inplace optimization", id="inplace-optimized"),
        ],
    )
    def test_invalid_request_is_refused_before_the_dependency_check(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        shape: tuple[int, int],
        optimized_inplace: bool,
        error: type[Exception],
        message: str,
    ) -> None:
        """A request no install could satisfy is named first: installing a missing extra would not make it valid.

        That covers a shape the model cannot take and a legacy model whose base weights were cleared.
        """
        mocker.patch.object(OnnxExporter, "check_dependencies", side_effect=ImportError("missing package"))
        model = _make_tensorrt_export_model()
        model._optimized_inplace = optimized_inplace

        with pytest.raises(error) as exc_info:
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=shape)

        assert message in str(exc_info.value)

    @pytest.mark.parametrize("export_format", sorted(REGISTRY))
    def test_direct_conversion_checks_dependencies_first(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, export_format: str
    ) -> None:
        """An exporter handed a graph directly runs the same check before its conversion starts.

        ``Exporter.__call__`` runs it, so no format has to remember to: the hook's own error surfaces and ``_convert``
        never runs.
        """
        # TFLite's registry preimport would otherwise load the real TensorFlow.
        monkeypatch.setattr("rfdetr.export._backend.preload_tensorflow_before_onnx", lambda: None)
        exporter_class = resolve_exporter(export_format)
        mocker.patch.object(exporter_class, "check_dependencies", side_effect=ImportError("missing package"))
        convert = mocker.patch.object(exporter_class, "_convert")
        # build_config drops a setting the format does not read, so every format can take ExecuTorch's backend.
        exporter = exporter_class(exporter_class.build_config(output_dir=tmp_path, verbose=False, backend="xnnpack"))

        with pytest.raises(ImportError) as exc_info:
            exporter(_make_export_graph())

        assert str(exc_info.value) == "missing package"

        convert.assert_not_called()

    @pytest.mark.parametrize(
        "export_format, settings, patches, modules, fragments",
        [
            pytest.param(
                "onnx",
                {},
                {},
                {"onnx": None},
                ('missing (onnx). Install with: pip install "rfdetr[onnx]"',),
                id="onnx",
            ),
            pytest.param(
                "tflite",
                {},
                {
                    "rfdetr.export._backend.preload_tensorflow_before_onnx": lambda: None,
                    "rfdetr.export._tflite.exporter._check_tf_keras_available": lambda: None,
                    "rfdetr.export._tflite.exporter._check_onnx_graphsurgeon_available": lambda: None,
                },
                {"tensorflow": types.ModuleType("tensorflow"), "onnx2tf": None},
                ("onnx2tf is not installed", "rfdetr[tflite]"),
                id="tflite-onnx2tf",
            ),
            pytest.param(
                "tflite",
                {},
                {
                    "rfdetr.export._backend.preload_tensorflow_before_onnx": lambda: None,
                    "rfdetr.export._tflite.exporter._check_tf_keras_available": lambda: None,
                },
                {"tensorflow": types.ModuleType("tensorflow"), "onnx_graphsurgeon": None},
                ("requires onnx_graphsurgeon", 'pip install onnx_graphsurgeon ("rfdetr[tflite]" includes it'),
                id="tflite-onnx_graphsurgeon",
            ),
            pytest.param(
                "tflite",
                {},
                {"rfdetr.export._backend.preload_tensorflow_before_onnx": lambda: None},
                {"tensorflow": types.ModuleType("tensorflow"), "tf_keras": None},
                ("requires tf-keras", "rfdetr[tflite]"),
                id="tflite-tf_keras",
            ),
            pytest.param(
                "tflite",
                {},
                {"rfdetr.export._backend.preload_tensorflow_before_onnx": lambda: None},
                {"tensorflow": None, "onnx2tf": types.ModuleType("onnx2tf")},
                ("requires TensorFlow", "rfdetr[tflite]"),
                id="tflite-tensorflow",
            ),
            pytest.param(
                "tflite",
                {},
                {
                    "rfdetr.export._backend.preload_tensorflow_before_onnx": lambda: None,
                    "rfdetr.export._tflite.exporter._check_tf_keras_available": lambda: None,
                },
                {"tensorflow": types.ModuleType("tensorflow"), "onnx": None},
                (
                    'missing (onnx). Install it with: pip install "rfdetr[tflite]" '
                    "(the extra installs on Python 3.12 only)",
                ),
                id="tflite-onnx",
            ),
            pytest.param(
                "tensorrt",
                {},
                {"rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE": False},
                {},
                ("rfdetr[tensorrt]",),
                id="tensorrt",
            ),
            pytest.param(
                "tensorrt",
                {},
                # TensorRT is reported present, so the onnx check is reached on every host: it comes second, after the
                # flags, so that a host without TensorRT is refused before anything imports onnx.
                {
                    "rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE": True,
                    "rfdetr.export._tensorrt.exporter._IS_POLYGRAPHY_AVAILABLE": True,
                },
                {"onnx": None},
                ('missing (onnx). Install with: pip install "rfdetr[tensorrt]"',),
                id="tensorrt-onnx",
            ),
            pytest.param(
                "executorch", {"backend": "xnnpack"}, {}, {"executorch": None}, ("rfdetr[executorch]",), id="executorch"
            ),
            pytest.param(
                "coreml",
                {},
                {"rfdetr.export._coreml.exporter._IS_COREMLTOOLS_AVAILABLE": False},
                {},
                ("rfdetr[coreml]",),
                id="coreml",
            ),
            pytest.param(
                "coreai",
                {},
                {"rfdetr.export._coreai.exporter._IS_COREAI_TORCH_AVAILABLE": False},
                {},
                ("rfdetr[coreai]",),
                id="coreai",
            ),
            pytest.param("openvino", {}, {}, {"openvino": None}, ("rfdetr[openvino]",), id="openvino"),
            pytest.param("litert", {}, {}, {"litert_torch": None}, ("rfdetr[litert]",), id="litert"),
        ],
    )
    def test_missing_package_names_its_extra(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        export_format: str,
        settings: dict[str, object],
        patches: dict[str, object],
        modules: dict[str, types.ModuleType | None],
        fragments: tuple[str, ...],
    ) -> None:
        """Each format's check fails on each package it needs, naming the ``rfdetr[...]`` extra that installs it.

        TFLite and TensorRT export through the ONNX stage, so they check ``onnx`` too, and a missing one names their own
        extra: it installs ``onnx`` along with everything else the format needs, where ``rfdetr[onnx]`` would not.
        """
        for target, value in patches.items():
            monkeypatch.setattr(target, value)
        exporter_class = resolve_exporter(export_format)
        exporter = exporter_class(exporter_class.build_config(output_dir=tmp_path, verbose=False, **settings))
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)

        with pytest.raises(ImportError) as exc_info:
            exporter.check_dependencies()

        assert all(fragment in str(exc_info.value) for fragment in fragments)

    def test_refused_tensorrt_request_does_not_import_onnx(self) -> None:
        """A host without TensorRT is refused before anything imports ``onnx``.

        Runs in a fresh interpreter because this suite has already imported ONNX. A refusal that left ``onnx`` loaded
        without TensorFlow would start a later ``format="tflite"`` export in the same process in the import order that
        hangs TensorFlow's SavedModel restore (issue #1322). The child imports the same ``rfdetr`` as this process.
        """
        script = (
            "import sys\n"
            "sys.modules['tensorrt'] = None\n"
            "from rfdetr.export._tensorrt.exporter import TensorRTExporter\n"
            "try:\n"
            "    TensorRTExporter.check_dependencies()\n"
            "except ImportError:\n"
            "    print('onnx' in sys.modules)\n"
        )
        result = _run_in_fresh_interpreter(script)

        assert (result.returncode, result.stdout.strip().splitlines()[-1:]) == (0, ["False"]), result.stderr

    @pytest.mark.parametrize(
        "module, allowed",
        [
            pytest.param("rfdetr.export._onnx.exporter", [], id="onnx"),
            pytest.param("rfdetr.export._tflite.exporter", [], id="tflite"),
            pytest.param("rfdetr.export._tensorrt.exporter", ["polygraphy"], id="tensorrt"),
        ],
    )
    def test_loading_the_exporter_imports_no_onnx_package(self, module: str, allowed: list[str]) -> None:
        """Loading the ONNX, TFLite or TensorRT exporter module imports no ONNX package beyond ``allowed``.

        The TFLite exporter must load TensorFlow before anything loads ``onnx`` (#1322), and it imports the ONNX
        exporter at module scope, as the TensorRT exporter does. A finder first on ``sys.meta_path`` claims every
        module of these packages, records each one an import executes and refuses it, so the result does not depend on
        which of them the host has installed: CPU CI has no ``onnx_graphsurgeon``, so a test that only looked for
        ``onnx`` in ``sys.modules`` would pass there with a guarded ``import onnx_graphsurgeon`` added. Its specs carry
        an origin, so ``is_installed`` reports them installed and an import guarded by it still runs and is recorded;
        the probe itself executes nothing. ``allowed`` names the packages a module imports on purpose: the TensorRT
        exporter runs a guarded ``polygraphy`` import, and every other package stays refused. Runs in a fresh
        interpreter because this suite has already imported them.
        """
        script = (
            "import importlib.abc, importlib.util, sys\n"
            f"allowed = {allowed!r}\n"
            "class Refuse(importlib.abc.MetaPathFinder, importlib.abc.Loader):\n"
            "    executed = []\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        package = name.partition('.')[0]\n"
            "        if package in {'onnx', 'onnx_graphsurgeon', 'onnxruntime', 'onnxsim', 'polygraphy'}:\n"
            "            return importlib.util.spec_from_loader(name, self, origin='refused')\n"
            "        return None\n"
            "    def create_module(self, spec):\n"
            "        return None\n"
            "    def exec_module(self, module):\n"
            "        self.executed.append(module.__name__)\n"
            "        raise ImportError(module.__name__)\n"
            "assert 'onnx' not in sys.modules, 'onnx was imported before the finder was installed'\n"
            "sys.meta_path.insert(0, Refuse())\n"
            f"import {module}\n"
            "print(sorted(name for name in Refuse.executed if name.partition('.')[0] not in allowed))\n"
        )
        result = _run_in_fresh_interpreter(script)

        assert (result.returncode, result.stdout.strip().splitlines()[-1:]) == (0, ["[]"]), result.stderr


class TestExportRefusalHasNoSideEffects:
    """A request refused before the forward pass leaves the live model, the disk and the exporter imports alone."""

    @pytest.mark.parametrize(
        "shape, dependency_error, error",
        [
            pytest.param((15, 15), None, ValueError, id="invalid-shape"),
            pytest.param((14, 14), ImportError("missing package"), ImportError, id="missing-package"),
        ],
    )
    def test_refused_request_leaves_the_model_and_the_disk_untouched(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        shape: tuple[int, int],
        dependency_error: ImportError | None,
        error: type[Exception],
    ) -> None:
        """The live model is neither moved to the CPU nor copied, and no output directory is created."""
        deepcopy = mocker.patch("rfdetr.detr.deepcopy")
        mocker.patch.object(OnnxExporter, "check_dependencies", side_effect=dependency_error)
        model = _make_tensorrt_export_model(device="original-device")
        output_dir = tmp_path / "out"

        with pytest.raises(error):
            _detr_module.RFDETR.export(model, output_dir=str(output_dir), shape=shape)

        assert (model.model.model.to_calls, deepcopy.called, output_dir.exists()) == ([], False, False)

    @pytest.mark.parametrize(
        "optimized_inplace, shape, error",
        [
            pytest.param(False, (15, 15), ValueError, id="invalid-shape"),
            pytest.param(True, (14, 14), RuntimeError, id="inplace-optimized"),
        ],
    )
    def test_format_independent_refusal_imports_no_exporter(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        optimized_inplace: bool,
        shape: tuple[int, int],
        error: type[Exception],
    ) -> None:
        """A request no format could export is refused before ``resolve_exporter`` imports the format's exporter."""
        resolve_exporter_stub = mocker.patch("rfdetr.export.registry.resolve_exporter")
        model = _make_tensorrt_export_model()
        model._optimized_inplace = optimized_inplace

        with pytest.raises(error):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=shape)

        resolve_exporter_stub.assert_not_called()
