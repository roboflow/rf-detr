# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for :meth:`rfdetr.detr.RFDETR.export` — the public facade over the per-format exporter classes.

Use cases covered:
- Segmentation outputs must be present in both train/eval modes to avoid export crashes.
- Export must not change the original model's training state, and must restore its device even when a converter raises.
- Shape and patch-size validation must reject an unexportable request before any conversion work happens.
- The ONNX artifact's filename must follow the variant/output-name/backbone rules users glob for.
- Every symbol these tests monkeypatch must stay on the real call path, not merely remain importable.
"""

import importlib.util
import inspect
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.jit import TracerWarning

from rfdetr import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
from rfdetr import detr as _detr_module
from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._tensorrt.exporter import TensorRTExporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.export.registry import resolve_exporter
from rfdetr.models.backbone.dinov2 import DinoV2

if TYPE_CHECKING:
    import onnx

_IS_ONNX_INSTALLED = importlib.util.find_spec("onnx") is not None


@contextmanager
def ignore_tracer_warnings() -> Iterator[None]:
    """Suppress torch.jit.TracerWarning during export tests to reduce log spam."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        yield


class _DummyCoreModel:
    """Minimal torch.nn.Module stub shared across export tests.

    Avoids real forward passes; returns synthetic detection (and optionally segmentation) outputs matching the shapes
    expected by RFDETR.export().
    """

    def __init__(self, *, segmentation_head: bool = False) -> None:
        self._segmentation_head = segmentation_head

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def cpu(self):
        return self

    def modules(self):
        return iter(())

    def __call__(self, *_args, **_kwargs):
        out = {"pred_boxes": torch.zeros(1, 1, 4), "pred_logits": torch.zeros(1, 1, 2)}
        if self._segmentation_head:
            out["pred_masks"] = torch.zeros(1, 1, 2, 2)
        return out


def _run_onnx_export(
    *,
    output_dir: str,
    model: torch.nn.Module,
    input_names: Sequence[str],
    input_tensors: torch.Tensor,
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None,
    backbone_only: bool = False,
    verbose: bool = True,
    opset_version: int = 17,
    variant_name: str | None = None,
    output_name: str | None = None,
) -> Path:
    """Run ``OnnxExporter`` over a throwaway graph, the way ``RFDETR.export()`` does.

    Assembles the config and the prepared graph the exporter expects so a test can exercise the real conversion
    path without building an ``RFDETR``. The keyword names mirror what ``RFDETR.export()`` accepts, which is what
    the naming assertions below are actually about.

    Args:
        output_dir: Directory the artifact is written to.
        model: Module to trace.
        input_names: Graph input names.
        input_tensors: Example input to trace with.
        output_names: Graph output names.
        dynamic_axes: Dynamic-axis mapping, or ``None`` for a static graph.
        backbone_only: Whether the graph is a backbone-only export.
        verbose: Whether the exporter logs its progress.
        opset_version: ONNX opset to target.
        variant_name: Model variant identifier used to name the artifact.
        output_name: Full filename override, without extension.

    Returns:
        Path to the exported ``.onnx`` file.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:  # doctest: +ELLIPSIS
        ...     path = _run_onnx_export(
        ...         output_dir=directory,
        ...         model=torch.nn.Identity(),
        ...         input_names=["input"],
        ...         input_tensors=torch.zeros(1, 3, 8, 8),
        ...         output_names=["dets"],
        ...         dynamic_axes=None,
        ...         verbose=False,
        ...     )
        ...     path.name
        [...] [INFO] rf-detr - Successfully exported ONNX model to: ...inference_model.onnx
        'inference_model.onnx'
    """
    config = OnnxConfig(
        output_dir=Path(output_dir),
        output_name=output_name,
        variant_name=variant_name,
        backbone_only=backbone_only,
        dynamic_batch=dynamic_axes is not None,
        verbose=verbose,
        opset_version=opset_version,
    )
    graph = ExportGraph(
        model=model,
        input_tensors=input_tensors,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        dynamic_axes=dynamic_axes,
        shape=tuple(input_tensors.shape[-2:]),
        backbone_only=backbone_only,
    )
    return OnnxExporter(config)(graph)


def test_export_onnx_uses_legacy_exporter_when_dynamo_flag_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`export_onnx` should pass `dynamo=False` when supported by torch.onnx.export."""
    captured_kwargs: dict = {}

    class _ToyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def _fake_onnx_export(*args, **kwargs) -> None:
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

    _run_onnx_export(
        output_dir=str(tmp_path),
        model=_ToyModel(),
        input_names=["images"],
        input_tensors=torch.randn(1, 3, 8, 8),
        output_names=["dets"],
        dynamic_axes={},
        verbose=False,
    )

    has_dynamo_arg = "dynamo" in inspect.signature(torch.onnx.export).parameters
    assert ("dynamo" in captured_kwargs) == has_dynamo_arg
    if has_dynamo_arg:
        assert captured_kwargs["dynamo"] is False


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnx]")
def test_segmentation_model_export_no_crash(tmp_path: Path) -> None:
    """Integration test: exporting a segmentation model should not crash.

    This exercises the full export path to ensure no AttributeError occurs.
    """
    model = RFDETRSegNano()

    # This should not crash with "AttributeError: 'dict' object has no attribute 'shape'"
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), verbose=False)

    # Verify export produced output files
    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnx]")
def test_export_with_rectangular_shape_different_from_resolution_no_crash(tmp_path: Path) -> None:
    """Integration test: exporting with a valid rectangular shape should not crash.

    A mismatched shape forces the DINOv2 backbone to interpolate its position embeddings for the new grid size. This
    must not trace an antialiased bicubic resize (``aten::_upsample_bicubic2d_aa``), which has no ONNX opset-17
    symbolic function.
    """
    model = RFDETRNano()
    block_size = model.model_config.patch_size * model.model_config.num_windows
    native_resolution = model.model.resolution
    export_shape = (native_resolution, native_resolution + block_size)
    assert all(dimension % block_size == 0 for dimension in export_shape)
    assert export_shape != (native_resolution, native_resolution)

    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), shape=export_shape, verbose=False)

    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


def test_dinov2_export_uses_precomputed_positions_for_exact_rectangular_grid() -> None:
    """DINOv2 export must bypass interpolation only for its precomputed rectangular grid."""
    patch_size = 8
    export_shape = (32, 48)
    source_positions = torch.nn.Parameter(torch.randn(1, 17, 2))
    original_calls: list[tuple[int, int]] = []
    fallback_positions = torch.randn(1, 1, 2)

    def original_interpolate_pos_encoding(_embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Record fallback interpolation calls made by the exported backbone."""
        original_calls.append((height, width))
        return fallback_positions

    backbone = DinoV2.__new__(DinoV2)
    torch.nn.Module.__init__(backbone)
    backbone.shape = export_shape
    backbone._export = False
    backbone.encoder = types.SimpleNamespace(
        config=types.SimpleNamespace(patch_size=patch_size),
        embeddings=types.SimpleNamespace(
            position_embeddings=source_positions,
            interpolate_pos_encoding=original_interpolate_pos_encoding,
        ),
    )

    backbone.export()
    patch_count = (export_shape[0] // patch_size) * (export_shape[1] // patch_size)
    embeddings = torch.randn(1, patch_count + 1, 2)

    fixed_positions = backbone.encoder.embeddings.interpolate_pos_encoding(embeddings, *export_shape)
    assert fixed_positions is backbone.encoder.embeddings.position_embeddings
    assert original_calls == []

    transposed_positions = backbone.encoder.embeddings.interpolate_pos_encoding(
        embeddings, export_shape[1], export_shape[0]
    )
    assert transposed_positions is fallback_positions
    assert original_calls == [(export_shape[1], export_shape[0])]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnx]")
def test_export_does_not_change_original_training_state(tmp_path: Path) -> None:
    """Verify that calling export() does not change the original model's train/eval state.

    This ensures that export() puts a deepcopy of the model in eval mode without mutating the underlying training model
    used by RF-DETR.
    """
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model), as in other tests
    torch_model = model.model.model.to("cuda")

    # Ensure the original model is in training mode
    torch_model.train()
    assert torch_model.training is True, "Precondition: original model should start in training mode"

    # Call export() on the high-level model; this should not change the original model's mode
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path))

    # After export, the original underlying model should still be in training mode
    assert torch_model.training is True, "export() should not change the original model's training state"


@pytest.mark.parametrize(
    "dynamic_batch, segmentation_head",
    [
        pytest.param(True, False, id="detection_dynamic"),
        pytest.param(True, True, id="segmentation_dynamic"),
        pytest.param(False, False, id="detection_static"),
    ],
)
def test_rfdetr_export_dynamic_batch_forwards_dynamic_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dynamic_batch: bool,
    segmentation_head: bool,
) -> None:
    """`RFDETR.export(..., dynamic_batch=True)` must pass a non-None `dynamic_axes` dict to `export_onnx`;
    `dynamic_batch=False` must pass `None`."""
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            model=_DummyCoreModel(segmentation_head=segmentation_head), device="cpu", resolution=14
        ),
        model_config=types.SimpleNamespace(
            segmentation_head=segmentation_head,
            use_grouppose_keypoints=False,
            num_channels=3,
        ),
        size=None,
    )

    captured: dict = {}

    def _fake_make_infer_image(*_args, **_kwargs):
        return torch.zeros(1, 3, 14, 14)

    def _fake_convert(_self, graph):
        captured["dynamic_axes"] = graph.dynamic_axes
        return str(tmp_path / "inference_model.onnx")

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", _fake_make_infer_image)
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", _fake_convert)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

    _detr_module.RFDETR.export(model, output_dir=str(tmp_path), dynamic_batch=dynamic_batch, shape=(14, 14))

    dynamic_axes = captured.get("dynamic_axes")
    if not dynamic_batch:
        assert dynamic_axes is None, f"expected None for static export, got {dynamic_axes!r}"
        return

    assert isinstance(dynamic_axes, dict), f"expected dict, got {dynamic_axes!r}"
    for name, axes in dynamic_axes.items():
        assert axes == {0: "batch"}, f"axis spec for {name!r} should be {{0: 'batch'}}, got {axes!r}"

    expected_names = {"input", "dets", "labels", "masks"} if segmentation_head else {"input", "dets", "labels"}
    assert set(dynamic_axes.keys()) == expected_names, f"expected keys {expected_names}, got {set(dynamic_axes.keys())}"


class _DeviceTrackingCoreModel(_DummyCoreModel):
    """`_DummyCoreModel` variant that records every `.to()` call's target device."""

    def __init__(self) -> None:
        super().__init__()
        self.to_calls: list[str] = []

    def to(self, device, *_args, **_kwargs):
        self.to_calls.append(device)
        return self


def _make_tensorrt_export_model(*, device: str = "cpu") -> types.SimpleNamespace:
    """Build the minimal `self`-like fake `RFDETR.export()` needs for the format="tensorrt" branch.

    Examples:
        >>> m = _make_tensorrt_export_model()
        >>> m.model.device
        'cpu'
        >>> m = _make_tensorrt_export_model(device="cuda")
        >>> m.model.device
        'cuda'
    """
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model=_DeviceTrackingCoreModel(), device=device, resolution=14),
        model_config=types.SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False, num_channels=3),
        size=None,
    )


def _make_mock_infer_tensor() -> MagicMock:
    """Mock tensor standing in for `make_infer_image()`'s return value — avoids real-device `.to()`/`.cpu()`.

    Examples:
        >>> t = _make_mock_infer_tensor()
        >>> t.to("cpu") is t
        True
        >>> t.cpu() is t
        True
    """
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    return mock_tensor


@pytest.mark.parametrize(
    "export_format",
    [
        pytest.param("tensorrt", id="canonical"),
        pytest.param("trt", id="alias"),
    ],
)
def test_rfdetr_export_tensorrt_calls_build_engine_with_onnx_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, export_format: str
) -> None:
    """`RFDETR.export(format="tensorrt")` — and its `"trt"` alias — must call `build_engine` once with the ONNX path.

    Covers the public-API wrapper directly (as opposed to the CLI `main()` path, which
    `TestCliExportMain.test_tensorrt_flag_calls_build_engine` already covers).
    """
    model = _make_tensorrt_export_model()
    onnx_output = str(tmp_path / "inference_model.onnx")
    mock_build_engine = MagicMock(return_value=str(tmp_path / "inference_model.trt"))

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: onnx_output)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(
        "rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine",
        lambda _self, *args, **kwargs: mock_build_engine(*args, **kwargs),
    )

    result = _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format=export_format, shape=(14, 14))

    mock_build_engine.assert_called_once()
    assert mock_build_engine.call_args.args == (onnx_output,), (
        f"ONNX path must be passed positionally, got {mock_build_engine.call_args.args!r}"
    )
    assert str(result) == str(tmp_path / "inference_model.trt")


def test_rfdetr_export_warns_when_max_batch_size_used_without_tensorrt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`max_batch_size` outside `format="tensorrt"` is ignored and must warn rather than silently no-op.

    `format="onnx"` with `dynamic_batch=True` accepts a dynamic batch axis but has no optimization-profile concept for
    `max_batch_size` to tune, so passing it there previously vanished with no signal at all.
    """
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(model=_DummyCoreModel(), device="cpu", resolution=14),
        model_config=types.SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False, num_channels=3),
        size=None,
    )
    onnx_output = str(tmp_path / "inference_model.onnx")

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: onnx_output)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

    with pytest.warns(UserWarning, match=r"`max_batch_size`.*ignored"):
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="onnx", dynamic_batch=True, max_batch_size=8, shape=(14, 14)
        )


def test_rfdetr_export_warns_when_max_batch_size_used_without_dynamic_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`max_batch_size` on a static (`dynamic_batch=False`) TensorRT export is ignored and must warn.

    A static export builds the engine with the exact call it always made -- there is no optimization profile for
    `max_batch_size` to bound -- so a value passed there is inert and should not vanish silently.
    """
    model = _make_tensorrt_export_model()
    onnx_output = str(tmp_path / "inference_model.onnx")

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: onnx_output)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(
        "rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine",
        lambda _self, *args, **kwargs: str(tmp_path / "inference_model.trt"),
    )

    with pytest.warns(UserWarning, match=r"`max_batch_size`.*ignored"):
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", dynamic_batch=False, max_batch_size=8, shape=(14, 14)
        )


def test_rfdetr_export_tensorrt_without_tensorrt_fails_before_the_onnx_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``tensorrt`` installed, ``format="tensorrt"`` reports it before paying for the ONNX export.

    ``rfdetr[onnx]`` installs polygraphy but not ``tensorrt``; that host used to write the whole ONNX file first and
    only then fail inside polygraphy.
    """
    model = _make_tensorrt_export_model()

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr(
        "rfdetr.export._onnx.exporter.OnnxExporter._convert",
        lambda *_a, **_kw: pytest.fail("the ONNX export ran before the missing TensorRT was reported"),
    )
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", False)

    with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", shape=(14, 14))


@pytest.mark.parametrize("fp16", [pytest.param(True, id="fp16"), pytest.param(False, id="fp32")])
def test_rfdetr_export_tensorrt_forwards_fp16(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fp16: bool) -> None:
    """`RFDETR.export(format="tensorrt", fp16=...)` must reach the exporter that builds the engine.

    Passing ``fp16=False`` lets callers build an FP32 engine on TensorRT builds that lack the FP16 builder flag, so the
    flag has to survive the trip through the configuration rather than silently reverting to the default.
    """
    model = _make_tensorrt_export_model()
    onnx_output = str(tmp_path / "inference_model.onnx")
    captured: dict = {}

    def _fake_build_engine(self, *_args, **_kwargs) -> str:
        captured["fp16"] = self.config.fp16
        return str(tmp_path / "inference_model.trt")

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: onnx_output)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine", _fake_build_engine)

    _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", fp16=fp16, shape=(14, 14))

    assert captured["fp16"] == fp16


def test_rfdetr_export_tensorrt_failure_restores_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A `build_engine` failure must still restore the live model to its original device.

    Regression test for the try/finally around the CPU-move .. TensorRT-conversion span in `RFDETR.export()` — a build
    failure previously (pre-merge) could strand the model on CPU.
    """
    # Deliberately distinct from the "cpu" staging move inside export() — if this were "cpu" too, the
    # assertion below would pass even with the `finally` restore deleted (both moves would look identical).
    original_device = "original-device"
    model = _make_tensorrt_export_model(device=original_device)
    onnx_output = str(tmp_path / "inference_model.onnx")

    def _raise_build_engine(*_args, **_kwargs):
        raise RuntimeError("engine build failed")

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: onnx_output)
    # Real deepcopy (not identity) — the exported `model` local must be a distinct object from
    # `self.model.model` so only the latter's `.to()` calls are tracked, matching production behavior.
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine", _raise_build_engine)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)

    with pytest.raises(RuntimeError):
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", shape=(14, 14))

    core_model = model.model.model
    assert core_model.to_calls == ["cpu", original_device], (
        f"expected exactly one staging move to 'cpu' then one restore to {original_device!r} even though "
        f"build_engine raised, got device move sequence {core_model.to_calls!r}"
    )


def test_rfdetr_export_tensorrt_dynamic_batch_requires_max_batch_size(tmp_path: Path) -> None:
    """`RFDETR.export(format="tensorrt", dynamic_batch=True)` without `max_batch_size` must raise.

    Covers the public facade users actually call (as opposed to `TestDynamicBatchConfig` in `test_tensorrt_export.py`,
    which only exercises `TensorRTExporter`/`TensorRTConfig` directly). The exporter is constructed — and validated —
    before any ONNX conversion work starts, so no conversion-chain monkeypatching is needed here.
    """
    model = _make_tensorrt_export_model()

    with pytest.raises(ValueError, match="max_batch_size"):
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", dynamic_batch=True, shape=(14, 14)
        )


def test_rfdetr_export_tensorrt_dynamic_batch_rejects_batch_size_over_max(tmp_path: Path) -> None:
    """`RFDETR.export(format="tensorrt", dynamic_batch=True, batch_size=..., max_batch_size=...)` enforces the bound.

    `batch_size > max_batch_size` cannot form a valid optimization profile, and the facade must refuse it at the same
    surface users call rather than only inside `TensorRTConfig` construction tests.
    """
    model = _make_tensorrt_export_model()

    with pytest.raises(ValueError, match="1 <= batch_size <= max_batch_size"):
        _detr_module.RFDETR.export(
            model,
            output_dir=str(tmp_path),
            format="tensorrt",
            dynamic_batch=True,
            batch_size=8,
            max_batch_size=4,
            shape=(14, 14),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", [pytest.param("train", id="train_mode"), pytest.param("eval", id="eval_mode")])
def test_segmentation_outputs_present_in_train_and_eval(mode: Literal["train", "eval"]) -> None:
    """Use case: segmentation outputs are present in both train and eval modes."""
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model)
    torch_model = model.model.model.to("cuda")

    # Use resolution compatible with model's patch size (312 for seg-nano)
    resolution = model.model.resolution
    dummy_input = torch.randn(1, 3, resolution, resolution, device="cuda")

    if mode == "train":
        torch_model.train()
    else:
        torch_model.eval()

    with torch.no_grad():
        output = torch_model(dummy_input)

    assert "pred_boxes" in output
    assert "pred_logits" in output
    assert "pred_masks" in output


class TestExportPatchSize:
    """RFDETR.export() patch_size validation and shape-divisibility tests."""

    @staticmethod
    def _scaffold(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_size: int, num_windows: int
    ) -> types.SimpleNamespace:
        """Build a minimal RFDETR-like namespace with controllable patch_size/num_windows."""
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                model=_DummyCoreModel(),
                device="cpu",
                resolution=patch_size * num_windows * 2,  # always valid
            ),
            model_config=types.SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                patch_size=patch_size,
                num_windows=num_windows,
                num_channels=3,
            ),
            size=None,
        )

        def _fake_make_infer_image(*_a, **_kw):
            return torch.zeros(1, 3, 8, 8)

        def _fake_convert(_self, _graph):
            return str(tmp_path / "inference_model.onnx")

        monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", _fake_convert)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
        return model

    def test_export_patch_size_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """export(patch_size=X) must raise ValueError when X != model_config.patch_size."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=4)
        with pytest.raises(ValueError, match="patch_size"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=16)

    @pytest.mark.parametrize("bad_patch_size", [0, -1])
    def test_export_invalid_patch_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_patch_size: int
    ) -> None:
        """Export() must raise ValueError when patch_size is not a positive integer."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=4)
        # Keep model_config.patch_size consistent with the patch_size argument for this test
        model.model_config.patch_size = bad_patch_size
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=bad_patch_size)

    def test_export_shape_must_be_divisible_by_block_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
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
    def test_export_negative_or_zero_shape_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_shape: tuple[int, int]
    ) -> None:
        """Export() must reject non-positive shape dimensions (Python -N % M == 0 wraps silently)."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=16, num_windows=2)
        with pytest.raises(ValueError, match="positive integers"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=bad_shape)

    def test_export_shape_valid_for_block_size(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Export() accepts shape divisible by patch_size * num_windows without error."""
        # patch_size=16, num_windows=2 → block_size=32; shape (64, 64) is valid
        model = self._scaffold(monkeypatch, tmp_path, patch_size=16, num_windows=2)
        # Should not raise
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(64, 64))

    @pytest.mark.parametrize("bad_patch_size", [True, False])
    def test_export_bool_patch_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_patch_size: bool
    ) -> None:
        """Export() must reject bool values for patch_size (isinstance(True, int) is True)."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=bad_patch_size)

    def test_export_explicit_patch_size_matching_config_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """export(patch_size=X) must succeed when X matches model_config.patch_size."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=4)
        # patch_size=14 matches model_config.patch_size=14; block_size=56; resolution=112 (56*2)
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=14)

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
    def test_export_invalid_shape_type_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_shape: tuple
    ) -> None:
        """Export() must raise ValueError for float, bool, or wrong-arity shape tuples."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        with pytest.raises(ValueError, match="shape"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=bad_shape)

    @pytest.mark.parametrize("bad_num_windows", [0, -1, True])
    def test_export_invalid_num_windows_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_num_windows: int
    ) -> None:
        """Export() must raise ValueError when model_config.num_windows is not a positive integer."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        model.model_config.num_windows = bad_num_windows
        with pytest.raises(ValueError, match="num_windows must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path))

    def test_export_default_resolution_not_divisible_by_block_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Export() with shape=None must raise ValueError when model.resolution % block_size != 0."""
        # patch_size=14, num_windows=3 → block_size=42; scaffold sets resolution=84 (42*2) which is valid
        # Override resolution to 50 (not divisible by 42) to trigger the check
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=3)
        model.model.resolution = 50
        with pytest.raises(ValueError, match="default resolution"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path))


def test_make_infer_image_produces_correct_rectangular_shape() -> None:
    """make_infer_image must produce a (B, C, H, W) tensor for non-square shapes.

    Regression test for the square-resize bug where ``Resize((shape[0], shape[0]))`` was used instead of
    ``Resize((shape[0], shape[1]))``, causing the output width to silently equal the height.
    """
    from rfdetr.export.prepare import make_infer_image

    h, w, b = 112, 224, 2
    tensor = make_infer_image(infer_dir=None, shape=(h, w), batch_size=b, device="cpu")
    assert tensor.shape == (b, 3, h, w), f"Expected shape ({b}, 3, {h}, {w}), got {tensor.shape}"


# ---------------------------------------------------------------------------
# ONNX export variant naming
# ---------------------------------------------------------------------------


class TestExportOnnxVariantNaming:
    """Verify that export_onnx uses variant_name in the output filename."""

    @pytest.mark.parametrize(
        "variant_name, backbone_only, output_name, output_names, expected_suffix",
        [
            pytest.param("rfdetr-medium", False, None, ["dets"], "rfdetr-medium.onnx", id="variant_name"),
            pytest.param(
                "rfdetr-nano", True, None, ["features"], "rfdetr-nano-backbone.onnx", id="variant_name_with_backbone"
            ),
            pytest.param(None, False, None, ["dets"], "inference_model.onnx", id="default_name_without_variant"),
            pytest.param(
                None, True, None, ["features"], "backbone_model.onnx", id="default_backbone_name_without_variant"
            ),
            pytest.param(
                "rfdetr-medium", False, "my-model", ["dets"], "my-model.onnx", id="output_name_overrides_variant_name"
            ),
            pytest.param(
                None,
                True,
                "my-model",
                ["features"],
                "my-model-backbone.onnx",
                id="output_name_with_backbone_keeps_structural_suffix",
            ),
        ],
    )
    def test_onnx_export_filename_naming(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        variant_name: str | None,
        backbone_only: bool,
        output_name: str | None,
        output_names: list[str],
        expected_suffix: str,
    ) -> None:
        """export_onnx's output filename follows the variant/output-name/backbone-suffix naming rules.

        Covers variant_name alone, variant_name + backbone_only (appends '-backbone'), no variant_name (falls back to
        the default name, backbone or not), output_name overriding variant_name, and output_name + backbone_only still
        appending the structural '-backbone' suffix.
        """
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]  # 3rd positional arg is output_file

        monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

        _run_onnx_export(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=output_names,
            dynamic_axes=None,
            backbone_only=backbone_only,
            verbose=False,
            variant_name=variant_name,
            output_name=output_name,
        )

        assert captured["output_file"].endswith(expected_suffix)

    @pytest.mark.parametrize(
        "size, output_name_kwarg, expected_variant_name, expected_output_name",
        [
            pytest.param("rfdetr-medium", None, "rfdetr-medium", None, id="variant_name_from_size"),
            pytest.param(None, None, None, None, id="none_when_size_not_set"),
            pytest.param(
                "rfdetr-medium", "my-model", "rfdetr-medium", "my-model", id="output_name_alongside_variant_name"
            ),
        ],
    )
    def test_rfdetr_export_passes_variant_and_output_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        size: str | None,
        output_name_kwarg: str | None,
        expected_variant_name: str | None,
        expected_output_name: str | None,
    ) -> None:
        """RFDETR.export() forwards self.size as variant_name, and any explicit output_name, to export_onnx.

        size=None must forward variant_name=None (base RFDETR has no size), and an explicit output_name kwarg must reach
        the exporter config alongside variant_name rather than replacing it there.
        """
        captured: dict = {}

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(model=_DummyCoreModel(), device="cpu", resolution=14),
            model_config=types.SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                num_channels=3,
            ),
            size=size,
        )

        def _fake_make_infer_image(*_args, **_kwargs):
            return torch.zeros(1, 3, 14, 14)

        def _fake_convert(self, _graph):
            captured["variant_name"] = self.config.variant_name
            captured["output_name"] = self.config.output_name
            return str(tmp_path / "inference_model.onnx")

        monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", _fake_convert)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

        export_kwargs = {"output_dir": str(tmp_path), "shape": (14, 14)}
        if output_name_kwarg is not None:
            export_kwargs["output_name"] = output_name_kwarg
        _detr_module.RFDETR.export(model, **export_kwargs)

        assert captured["variant_name"] == expected_variant_name
        assert captured["output_name"] == expected_output_name

    @pytest.mark.parametrize(
        "variant_name, expected_suffix",
        [
            pytest.param("", "inference_model.onnx", id="empty_string_falls_back_to_default"),
            pytest.param("foo/bar", "bar.onnx", id="path_separator_stripped_to_basename"),
            pytest.param("/tmp/x", "x.onnx", id="absolute_path_stripped_to_basename"),
        ],
    )
    def test_variant_name_sanitization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        variant_name: str,
        expected_suffix: str,
    ) -> None:
        """variant_name edge cases: empty string falls back to default; path separators are stripped."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]

        monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

        _run_onnx_export(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["dets"],
            dynamic_axes=None,
            verbose=False,
            variant_name=variant_name or None,
        )

        assert captured["output_file"].endswith(expected_suffix)


@pytest.mark.gpu
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnx]")
@pytest.mark.parametrize("model_class", [RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview])
@pytest.mark.parametrize("projector_scale", [["P4"], ["P4", "P5"]])
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_backbone_only_exports_features_without_detector(
    tmp_path: Path,
    model_class: type[RFDETRNano] | type[RFDETRSegNano] | type[RFDETRKeypointPreview],
    projector_scale: list[str],
    dynamic_batch: bool,
) -> None:
    """The public backbone export runs independently of detector heads and preserves every feature level."""
    import numpy as np

    ort = pytest.importorskip("onnxruntime")
    model = model_class(
        pretrain_weights=None,
        device="cpu",
        resolution=96,
        num_queries=4,
        num_select=4,
        num_classes=2,
        projector_scale=projector_scale,
    )
    core = model.model.model
    assert core is not None
    core.eval()
    batch = 2 if dynamic_batch else 1
    inputs = torch.linspace(-1, 1, batch * 3 * 96 * 192).reshape(batch, 3, 96, 192)
    backbone = core.backbone[0]
    with torch.no_grad():
        raw_features = backbone.encoder(inputs)
        expected = backbone.projector(raw_features)
        if backbone.cross_attn_projector is not None:
            expected.extend(backbone.cross_attn_projector(raw_features))
    with ignore_tracer_warnings():
        path = model.export(
            output_dir=str(tmp_path), backbone_only=True, shape=(96, 192), dynamic_batch=dynamic_batch, verbose=False
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    expected_names = ["features" if index == 0 else f"features_{index}" for index in range(len(projector_scale))]
    if backbone.cross_attn_projector is not None:
        expected_names.extend(
            "cross_attn_features" if index == 0 else f"cross_attn_features_{index}"
            for index in range(len(projector_scale))
        )
    assert [output.name for output in session.get_outputs()] == expected_names
    actual = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})
    assert len(actual) == len(expected)
    for exported, reference in zip(actual, expected):
        np.testing.assert_allclose(exported, reference.numpy(), atol=1e-4, rtol=1e-4)
    assert not backbone._export
    assert model.model.model is core


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
def test_prepared_backbone_module_reaches_the_exporter(tmp_path: Path, format: str) -> None:
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

    with patch.object(type(exporter), "_convert", return_value=output_path) as convert:
        result = exporter(graph)

    assert result == output_path
    assert convert.call_args.args[0] is graph


@pytest.mark.parametrize("backbone_only", [False, True])
def test_public_export_preserves_backbone_marker_in_custom_tensorrt_name(
    tmp_path: Path,
    backbone_only: bool,
) -> None:
    """TensorRT receives the ONNX backbone marker when the caller supplies a custom name."""
    obj = _detr_module.RFDETR.__new__(_detr_module.RFDETR)
    obj.model = MagicMock()
    obj.model.resolution = 32
    obj.model.device = "cpu"
    obj.model.model.to.return_value = obj.model.model
    backbone = torch.nn.Identity()
    backbone.export = MagicMock()
    backbone.forward_export = MagicMock(return_value=([torch.zeros(1, 4, 2, 2)], None, None))
    backbone.cross_attn_projector = None
    obj.model.model.backbone = torch.nn.ModuleList([backbone])
    obj.model_config = types.SimpleNamespace(
        segmentation_head=False,
        use_grouppose_keypoints=False,
        patch_size=16,
        num_windows=1,
        num_channels=3,
        projector_scale=["P4"],
    )
    stem = "custom-backbone" if backbone_only else "custom"
    onnx_path = tmp_path / f"{stem}.onnx"
    with (
        patch("rfdetr.export.prepare.make_infer_image", return_value=torch.zeros(1, 3, 32, 32)),
        patch("rfdetr.export._onnx.exporter.OnnxExporter._convert", return_value=onnx_path),
        patch("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True),
        patch.object(TensorRTExporter, "build_engine", return_value=tmp_path / f"{stem}.trt") as build,
    ):
        obj.export(format="tensorrt", backbone_only=backbone_only, output_name="custom", output_dir=str(tmp_path))
    assert build.call_args.kwargs["output_name"] == stem


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

    def test_switches_a_fresh_model(self) -> None:
        """A model that has not been switched yet must have ``export()`` called on it.

        This is the ordinary path taken by the ExecuTorch, CoreML and OpenVINO dispatchers, which receive a freshly
        deepcopied module from ``RFDETR.export()``.
        """
        model = self._SwitchCountingModel()
        _switch_to_export_mode(model)
        assert model.switches == 1

    def test_does_not_switch_twice(self) -> None:
        """A model already in export mode must be left alone on a second call.

        The failure this guards is silent and unrecoverable: the second ``export()`` would overwrite
        ``_forward_origin`` with ``forward_export``, so the module could never be restored. Two
        switches become reachable as soon as one exporter composes another (a TFLite or TensorRT
        export running an ONNX export internally).
        """
        model = self._SwitchCountingModel()
        _switch_to_export_mode(model)
        _switch_to_export_mode(model)
        assert model.switches == 1

    def test_leaves_a_module_without_export_untouched(self) -> None:
        """A plain ``nn.Module`` with no ``export`` attribute must pass through without raising.

        Backbone-only exports hand the dispatchers a ``_BackboneExport`` wrapper, which exposes no ``export`` method —
        that path must stay a silent no-op rather than an ``AttributeError``.
        """
        model = torch.nn.Linear(2, 2)
        _switch_to_export_mode(model)
        assert not hasattr(model, "_export")

    def test_onnx_exporter_routes_through_the_guarded_switch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``OnnxExporter`` must share the choke point, so a composed ONNX export cannot switch twice.

        A TFLite or TensorRT export runs an ONNX export internally after the model was already switched, so an unguarded
        ``model.export()`` here would overwrite ``_forward_origin`` with the export forward.
        """
        model = self._SwitchCountingModel()
        _switch_to_export_mode(model)
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

    def test_onnx_exporter_still_switches_a_fresh_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Routing through the guarded helper must not drop the switch for a fresh model."""
        model = self._SwitchCountingModel()
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


def test_onnx_exporter_creates_a_missing_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ONNX stage must create its own output directory.

    The composed TFLite and TensorRT exporters route their intermediate ONNX through ``OnnxExporter`` before creating
    their own output directory, so it cannot rely on a caller having made it.
    """
    monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: None)
    output_dir = tmp_path / "missing" / "nested"
    graph = ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 2),
        input_names=("input",),
        output_names=("output",),
        dynamic_axes=None,
        shape=(2, 2),
        backbone_only=False,
    )

    OnnxExporter(OnnxConfig(output_dir=output_dir, verbose=False))(graph)

    assert output_dir.is_dir()


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
    }
    if export_format == "tensorrt":
        stubs["rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine"] = MagicMock(
            return_value=str(tmp_path / "inference_model.trt")
        )
        # The build is stubbed, so the host's TensorRT install (none in CPU CI) must not decide the outcome.
        monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    if export_format == "tflite":
        stubs["rfdetr.export._backend.preload_tensorflow_before_onnx"] = MagicMock(return_value=None)
        stubs["rfdetr.export._tflite.exporter.TFLiteExporter.convert_onnx"] = MagicMock(
            return_value=tmp_path / "inference_model_fp32.tflite"
        )
    for target, stub in stubs.items():
        monkeypatch.setattr(target, stub)
    return stubs


class TestExportSeamInventory:
    """Every symbol the export tests monkeypatch must stay on ``RFDETR.export()``'s call path.

    Six symbols across six test files are patched to stub out slow or optional-dependency work. Moving code between
    modules can leave such a symbol importable — so ``mock.patch`` still resolves it and raises nothing — while the
    production call path no longer goes through it. The test that patched it then keeps passing while the real, slow
    implementation runs underneath. Existence is therefore not enough: each case below patches one seam, runs a full
    export, and asserts the mock was actually invoked.
    """

    @pytest.mark.parametrize(
        "seam, export_format",
        [
            pytest.param("rfdetr.detr.deepcopy", "onnx", id="deepcopy"),
            pytest.param("rfdetr.export.prepare.make_infer_image", "onnx", id="make_infer_image"),
            pytest.param("rfdetr.export._onnx.exporter.OnnxExporter._convert", "onnx", id="export_onnx"),
            pytest.param("rfdetr.export._backend._resolve_export_backend", "onnx", id="resolve_export_backend"),
            pytest.param(
                "rfdetr.export._tensorrt.exporter.TensorRTExporter.build_engine", "tensorrt", id="build_engine"
            ),
            pytest.param(
                "rfdetr.export._backend.preload_tensorflow_before_onnx", "tflite", id="preload_tensorflow_before_onnx"
            ),
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
            pytest.param("onnx", "inference_model.onnx", id="onnx"),
            pytest.param("tensorrt", "inference_model.trt", id="tensorrt"),
            pytest.param("tflite", "inference_model_fp32.tflite", id="tflite"),
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


def _keypoint_axis_broadcasts(model: "onnx.ModelProto") -> list[str]:
    """Name every elementwise node that broadcasts one input over axis -2 only, the keypoint axis of the decode.

    That is the pattern onnx2tf mis-transposes: a ``(B, Q, 1, 2)`` box reference broadcast against ``(B, Q, K, 2)``
    keypoint deltas made the TFLite conversion of keypoint models fail (#1514). Shapes are right-aligned, so a
    rank-3 ``(Q, 1, 2)`` input counts too. A full broadcast such as ``(1, 1, 1, 2)``, which onnx2tf converts, does not.

    Args:
        model: The ONNX model; shapes are inferred before the scan.

    Returns:
        The names of the offending nodes, in graph order.

    Raises:
        ValueError: If a possible keypoint-axis broadcast has an unknown keypoint
            dimension, or no eligible node was checked.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> def _elementwise(ref_shape: list[int]) -> "onnx.ModelProto":
        ...     graph = helper.make_graph(
        ...         [helper.make_node("Mul", ["delta", "ref"], ["out"], name="decode")],
        ...         "g",
        ...         [
        ...             helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 4, 3, 2]),
        ...             helper.make_tensor_value_info("ref", TensorProto.FLOAT, ref_shape),
        ...         ],
        ...         [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
        ...     )
        ...     return helper.make_model(graph)
        >>> _keypoint_axis_broadcasts(_elementwise([1, 4, 1, 2]))
        ['decode']
        >>> _keypoint_axis_broadcasts(_elementwise([4, 1, 2]))
        ['decode']
        >>> _keypoint_axis_broadcasts(_elementwise([1, 1, 1, 2]))
        []
        >>> _keypoint_axis_broadcasts(_elementwise([1, 4, 3, 2]))
        []
    """
    from onnx import shape_inference

    graph = shape_inference.infer_shapes(model).graph
    shapes = {
        value.name: [dim.dim_value if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]
        for value in [*graph.input, *graph.value_info, *graph.output]
    }
    shapes.update({initializer.name: list(initializer.dims) for initializer in graph.initializer})
    offenders = []
    checked = 0
    for node in graph.node:
        if node.op_type not in ("Mul", "Add", "Sub", "Div"):
            continue
        output_shape = shapes.get(node.output[0])
        if not output_shape or len(output_shape) != 4 or output_shape[-2] == 1:
            continue
        if output_shape[-2] is None:
            for name in node.input:
                input_shape = shapes.get(name) or []
                if (
                    2 < len(input_shape) <= 4
                    and input_shape[-2] == 1
                    and output_shape[-3] not in (None, 1)
                    and input_shape[-3] == output_shape[-3]
                ):
                    raise ValueError(
                        f"unknown keypoint axis at elementwise node {node.name!r}; shape check is incomplete"
                    )
            continue
        checked += 1
        for name in node.input:
            input_shape = shapes.get(name) or []
            if not 2 < len(input_shape) <= 4:
                continue
            if input_shape[-2] == 1 and output_shape[-3] not in (None, 1) and input_shape[-3] == output_shape[-3]:
                offenders.append(node.name)
                break
    if not checked:
        raise ValueError(
            "no rank-4 elementwise node with a known keypoint axis; shape inference found nothing to check"
        )
    return offenders


def _if_nodes(model: "onnx.ModelProto") -> list[str]:
    """Name every ``If`` node; onnx2tf's onnxsim pass cannot simplify one, and conversion then fails on ``Expand``.

    Args:
        model: The ONNX model.

    Returns:
        The names of the ``If`` nodes, in graph order.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> branch = helper.make_graph([], "branch", [], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])])
        >>> node = helper.make_node("If", ["cond"], ["y"], name="rank_check", then_branch=branch, else_branch=branch)
        >>> graph = helper.make_graph(
        ...     [node],
        ...     "g",
        ...     [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        ... )
        >>> _if_nodes(helper.make_model(graph))
        ['rank_check']
    """
    return [node.name for node in model.graph.node if node.op_type == "If"]


@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnx]")
class TestKeypointOnnxGraphAvoidsOnnx2tfBlockers:
    """The keypoint ONNX graph must avoid the constructs that broke its onnx2tf/TFLite conversion (#1514).

    A real onnx2tf conversion of a keypoint model takes over ten minutes on CPU, so these tests pin the graph constructs
    onnx2tf could not handle instead.
    """

    @pytest.fixture(scope="class")
    def keypoint_onnx(self, tmp_path_factory: pytest.TempPathFactory) -> "onnx.ModelProto":
        """Export a small, randomly initialized keypoint model to ONNX once and load the graph.

        Args:
            tmp_path_factory: Pytest's session-scoped temporary directory factory.

        Returns:
            The exported ONNX model, with a ``dets``/``labels``/``keypoints`` output contract.
        """
        import onnx

        model = RFDETRKeypointPreview(
            pretrain_weights=None,
            device="cpu",
            resolution=96,
            num_queries=4,
            num_classes=2,
            num_keypoints_per_class=[3],
        )
        with ignore_tracer_warnings():
            path = model.export(output_dir=str(tmp_path_factory.mktemp("keypoint_onnx")), verbose=False)
        return onnx.load(str(path))

    @pytest.mark.parametrize(
        "find_blockers",
        [
            pytest.param(_keypoint_axis_broadcasts, id="keypoint_axis_broadcast"),
            pytest.param(_if_nodes, id="if_node"),
        ],
    )
    def test_graph_has_no_onnx2tf_blockers(
        self, keypoint_onnx: "onnx.ModelProto", find_blockers: Callable[["onnx.ModelProto"], list[str]]
    ) -> None:
        """No node in the exported keypoint graph matches a construct onnx2tf fails to convert."""
        assert find_blockers(keypoint_onnx) == []

    def test_unknown_keypoint_axis_does_not_pass_after_checking_an_unrelated_node(self) -> None:
        """A symbolic keypoint dimension must not be hidden by an unrelated eligible elementwise node."""
        from onnx import TensorProto, helper

        checked_node = helper.make_node("Add", ["left", "right"], ["checked"], name="unrelated")
        keypoint_node = helper.make_node("Mul", ["delta", "reference"], ["keypoints"], name="decode")
        graph = helper.make_graph(
            [checked_node, keypoint_node],
            "keypoint_unknown_axis",
            [
                helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 4, 3, 2]),
                helper.make_tensor_value_info("right", TensorProto.FLOAT, [1, 4, 3, 2]),
                helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 4, "keypoints", 2]),
                helper.make_tensor_value_info("reference", TensorProto.FLOAT, [1, 4, 1, 2]),
            ],
            [
                helper.make_tensor_value_info("checked", TensorProto.FLOAT, [1, 4, 3, 2]),
                helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, [1, 4, "keypoints", 2]),
            ],
        )
        model = helper.make_model(graph)

        with pytest.raises(ValueError, match="unknown keypoint axis"):
            _keypoint_axis_broadcasts(model)
