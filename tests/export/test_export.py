# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for model export functionality.

Use cases covered:
- Segmentation outputs must be present in both train/eval modes to avoid export crashes.
- Export should not change the original model's training state.
- CLI export path (deploy.export.main) must include 'masks' in output_names for
  segmentation models, call make_infer_image with the correct individual args, and call export_onnx with args.output_dir
  as the first argument.
"""

import importlib.util
import inspect
import types
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.jit import TracerWarning

from rfdetr import RFDETRNano, RFDETRSegNano
from rfdetr import detr as _detr_module
from rfdetr.export import main as _cli_export_module
from rfdetr.models.backbone.dinov2 import DinoV2

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

    monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

    _cli_export_module.export_onnx(
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

    def _fake_export_onnx(*_args, dynamic_axes=None, **_kw):
        captured["dynamic_axes"] = dynamic_axes
        return str(tmp_path / "inference_model.onnx")

    monkeypatch.setattr("rfdetr.export.main.make_infer_image", _fake_make_infer_image)
    monkeypatch.setattr("rfdetr.export.main.export_onnx", _fake_export_onnx)
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
    """Build the minimal `self`-like fake `RFDETR.export()` needs for the format="tensorrt" branch."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model=_DeviceTrackingCoreModel(), device=device, resolution=14),
        model_config=types.SimpleNamespace(segmentation_head=False, use_grouppose_keypoints=False, num_channels=3),
        size=None,
    )


def _make_mock_infer_tensor() -> MagicMock:
    """Mock tensor standing in for `make_infer_image()`'s return value — avoids real-device `.to()`/`.cpu()`."""
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

    monkeypatch.setattr("rfdetr.export.main.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export.main.export_onnx", lambda *_a, **_kw: onnx_output)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.build_engine", mock_build_engine)

    result = _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format=export_format, shape=(14, 14))

    mock_build_engine.assert_called_once()
    assert mock_build_engine.call_args.args == (onnx_output,), (
        f"ONNX path must be passed positionally, got {mock_build_engine.call_args.args!r}"
    )
    assert str(result) == str(tmp_path / "inference_model.trt")


@pytest.mark.parametrize("fp16", [pytest.param(True, id="fp16"), pytest.param(False, id="fp32")])
def test_rfdetr_export_tensorrt_forwards_fp16(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fp16: bool) -> None:
    """`RFDETR.export(format="tensorrt", fp16=...)` must forward the precision flag to `build_engine`.

    Passing ``fp16=False`` lets callers build an FP32 engine on TensorRT builds that lack the FP16 builder flag.
    """
    model = _make_tensorrt_export_model()
    onnx_output = str(tmp_path / "inference_model.onnx")
    mock_build_engine = MagicMock(return_value=str(tmp_path / "inference_model.trt"))

    monkeypatch.setattr("rfdetr.export.main.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export.main.export_onnx", lambda *_a, **_kw: onnx_output)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.build_engine", mock_build_engine)

    _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", fp16=fp16, shape=(14, 14))

    assert mock_build_engine.call_args.kwargs["fp16"] == fp16


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

    monkeypatch.setattr("rfdetr.export.main.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export.main.export_onnx", lambda *_a, **_kw: onnx_output)
    # Real deepcopy (not identity) — the exported `model` local must be a distinct object from
    # `self.model.model` so only the latter's `.to()` calls are tracked, matching production behavior.
    monkeypatch.setattr("rfdetr.export._tensorrt.build_engine", _raise_build_engine)

    with pytest.raises(RuntimeError):
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", shape=(14, 14))

    core_model = model.model.model
    assert core_model.to_calls == ["cpu", original_device], (
        f"expected exactly one staging move to 'cpu' then one restore to {original_device!r} even though "
        f"build_engine raised, got device move sequence {core_model.to_calls!r}"
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


# --------------------------------------------------------------------------
# Tests for the CLI export path: rfdetr.export.main.main()
# --------------------------------------------------------------------------


class TestCliExportMain:
    """Unit tests for deploy.export.main() (CLI export path).

    Three bugs were present before the fix:
    1. output_names omitted 'masks' for segmentation models.
    2. make_infer_image received the whole args Namespace instead of individual fields.
    3. export_onnx received model/args in the wrong positions (output_dir was missing).
    """

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> str:
        return str(tmp_path)

    @staticmethod
    def _make_args(
        *,
        backbone_only: bool = False,
        segmentation_head: bool = False,
        output_dir: str,
        infer_dir: str | None = None,
        shape: tuple[int, int] = (640, 640),
        batch_size: int = 1,
        verbose: bool = False,
        opset_version: int = 17,
        tensorrt: bool = False,
        profile: bool = False,
        dry_run: bool = False,
        dynamic_batch: bool = False,
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            device="cpu",
            seed=42,
            layer_norm=False,
            resume=None,
            backbone_only=backbone_only,
            segmentation_head=segmentation_head,
            output_dir=output_dir,
            infer_dir=infer_dir,
            shape=shape,
            batch_size=batch_size,
            verbose=verbose,
            opset_version=opset_version,
            tensorrt=tensorrt,
            profile=profile,
            dry_run=dry_run,
            dynamic_batch=dynamic_batch,
        )

    @staticmethod
    def _run(args: types.SimpleNamespace) -> tuple[dict, dict]:
        """Run deploy.export.main(args) with all heavy dependencies mocked.

        Stubs out build_model, make_infer_image, and export_onnx, and injects mock onnx/onnxsim modules so the export
        module can be imported even when those optional packages are not installed.

        Returns (make_infer_image_captured, export_onnx_captured).
        """
        make_infer_image_captured: dict = {}
        export_onnx_captured: dict = {}

        mock_model = MagicMock()
        # parameters() must return an iterable of real objects so sum(p.numel()) works
        mock_model.parameters.return_value = []
        mock_model.backbone.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.projector.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.encoder.parameters.return_value = []
        mock_model.transformer.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model

        if args.backbone_only:
            mock_model.return_value = torch.zeros(1, 512, 20, 20)
        elif args.segmentation_head:
            mock_model.return_value = {
                "pred_boxes": torch.zeros(1, 100, 4),
                "pred_logits": torch.zeros(1, 100, 90),
                "pred_masks": torch.zeros(1, 100, 27, 27),
            }
        else:
            mock_model.return_value = {
                "pred_boxes": torch.zeros(1, 300, 4),
                "pred_logits": torch.zeros(1, 300, 90),
            }

        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor

        def fake_make_infer_image(*pos_args, **kw_args):
            make_infer_image_captured["positional"] = pos_args
            make_infer_image_captured["keyword"] = kw_args
            return mock_tensor

        def fake_export_onnx(output_dir, model, input_names, input_tensors, output_names, dynamic_axes, **kwargs):
            export_onnx_captured["output_dir"] = output_dir
            export_onnx_captured["model"] = model
            export_onnx_captured["output_names"] = output_names
            export_onnx_captured["dynamic_axes"] = dynamic_axes
            export_onnx_captured["kwargs"] = kwargs
            return str(args.output_dir) + "/inference_model.onnx"

        with (
            patch.object(_cli_export_module, "build_model", return_value=(mock_model, MagicMock(), MagicMock())),
            patch.object(_cli_export_module, "make_infer_image", side_effect=fake_make_infer_image),
            patch.object(_cli_export_module, "export_onnx", side_effect=fake_export_onnx),
            patch.object(_cli_export_module, "get_rank", return_value=0),
        ):
            _cli_export_module.main(args)

        return make_infer_image_captured, export_onnx_captured

    @pytest.mark.parametrize(
        "segmentation_head, backbone_only, expected_output_names",
        [
            pytest.param(True, False, ["dets", "labels", "masks"], id="segmentation"),
            pytest.param(False, False, ["dets", "labels"], id="detection"),
            pytest.param(False, True, ["features"], id="backbone_only"),
        ],
    )
    def test_output_names(
        self,
        output_dir: str,
        segmentation_head: bool,
        backbone_only: bool,
        expected_output_names: list[str],
    ) -> None:
        """export_onnx must receive the correct output_names for every model type.

        Before the fix, deploy/export.py line 253 used:

        output_names = ['features'] if args.backbone_only else ['dets', 'labels']

        which always omitted 'masks' for segmentation models.
        """
        args = self._make_args(
            backbone_only=backbone_only,
            segmentation_head=segmentation_head,
            output_dir=output_dir,
        )
        _, export_onnx_captured = self._run(args)

        actual = export_onnx_captured.get("output_names")
        assert actual == expected_output_names, f"expected output_names={expected_output_names}, got {actual!r}"

    def test_make_infer_image_receives_individual_fields(self, output_dir: str) -> None:
        """make_infer_image must be called with (infer_dir, shape, batch_size, device), not with the whole args
        Namespace.

        Before the fix, deploy/export.py line 251 used:

        input_tensors = make_infer_image(args, device)
        """
        shape = (640, 640)
        batch_size = 2
        infer_dir = None
        args = self._make_args(
            output_dir=output_dir,
            infer_dir=infer_dir,
            shape=shape,
            batch_size=batch_size,
        )
        make_infer_image_captured, _ = self._run(args)

        pos = make_infer_image_captured.get("positional", ())
        assert pos[:3] == (infer_dir, shape, batch_size), f"expected (infer_dir, shape, batch_size), got {pos[:3]!r}"

    def test_export_onnx_receives_output_dir_and_kwargs(self, output_dir: str) -> None:
        """export_onnx must be called as export_onnx(output_dir, model, ...) with backbone_only, verbose, and
        opset_version forwarded as keyword args.

        Before the fix, deploy/export.py line 294 used:

        export_onnx(model, args, input_names, input_tensors, output_names, dynamic_axes)

        which swapped output_dir/model and dropped all keyword args.
        """
        args = self._make_args(
            output_dir=output_dir,
            verbose=True,
            opset_version=11,
        )
        _, export_onnx_captured = self._run(args)

        assert "output_dir" in export_onnx_captured, "export_onnx was not called"
        assert export_onnx_captured["output_dir"] == output_dir, (
            f"expected output_dir={output_dir!r}, got {export_onnx_captured['output_dir']!r}"
        )
        kwargs = export_onnx_captured.get("kwargs", {})
        assert kwargs.get("verbose") == args.verbose, (
            f"expected verbose={args.verbose!r}, got {kwargs.get('verbose')!r}"
        )
        assert kwargs.get("opset_version") == args.opset_version, (
            f"expected opset_version={args.opset_version!r}, got {kwargs.get('opset_version')!r}"
        )
        assert "backbone_only" in kwargs, "backbone_only kwarg missing from export_onnx call"

    @pytest.mark.parametrize(
        "dynamic_batch, segmentation_head, backbone_only",
        [
            pytest.param(True, False, False, id="detection_dynamic"),
            pytest.param(True, True, False, id="segmentation_dynamic"),
            pytest.param(True, False, True, id="backbone_only_dynamic"),
            pytest.param(False, False, False, id="detection_static"),
        ],
    )
    def test_dynamic_batch_forwards_dynamic_axes(
        self,
        output_dir: str,
        dynamic_batch: bool,
        segmentation_head: bool,
        backbone_only: bool,
    ) -> None:
        """CLI --dynamic_batch=True must pass {name: {0: 'batch'}} for every I/O name.

        When dynamic_batch=False, dynamic_axes must be None (static export).
        """
        args = self._make_args(
            output_dir=output_dir,
            dynamic_batch=dynamic_batch,
            segmentation_head=segmentation_head,
            backbone_only=backbone_only,
        )
        _, captured = self._run(args)

        dynamic_axes = captured.get("dynamic_axes")
        if not dynamic_batch:
            assert dynamic_axes is None, f"expected None for static export, got {dynamic_axes!r}"
            return

        assert isinstance(dynamic_axes, dict), f"expected dict, got {dynamic_axes!r}"
        for name, axes in dynamic_axes.items():
            assert axes == {0: "batch"}, f"axis spec for {name!r} should be {{0: 'batch'}}, got {axes!r}"

        # Every input/output name must have an entry
        if backbone_only:
            expected_names = {"input", "features"}
        elif segmentation_head:
            expected_names = {"input", "dets", "labels", "masks"}
        else:
            expected_names = {"input", "dets", "labels"}
        assert set(dynamic_axes.keys()) == expected_names, (
            f"expected keys {expected_names}, got {set(dynamic_axes.keys())}"
        )

    def test_tensorrt_flag_calls_build_engine(self, output_dir: str) -> None:
        """When tensorrt=True, main() must call build_engine with the ONNX output path."""
        build_engine_calls: list[str] = []

        def fake_build_engine(onnx_path: str, **_kwargs) -> str:
            build_engine_calls.append(onnx_path)
            return onnx_path.replace(".onnx", ".trt")

        args = self._make_args(output_dir=output_dir, tensorrt=True)
        onnx_output = str(args.output_dir) + "/inference_model.onnx"

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.backbone.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.projector.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.encoder.parameters.return_value = []
        mock_model.transformer.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = {
            "pred_boxes": torch.zeros(1, 300, 4),
            "pred_logits": torch.zeros(1, 300, 90),
        }
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor

        with (
            patch.object(_cli_export_module, "build_model", return_value=(mock_model, MagicMock(), MagicMock())),
            patch.object(_cli_export_module, "make_infer_image", return_value=mock_tensor),
            patch.object(_cli_export_module, "export_onnx", return_value=onnx_output),
            patch.object(_cli_export_module, "build_engine", side_effect=fake_build_engine),
            patch.object(_cli_export_module, "get_rank", return_value=0),
        ):
            _cli_export_module.main(args)

        assert len(build_engine_calls) == 1, "build_engine should be called exactly once"
        assert build_engine_calls[0] == onnx_output, (
            f"build_engine called with {build_engine_calls[0]!r}, expected {onnx_output!r}"
        )

    def test_tensorrt_flag_forwards_verbose_and_dry_run_kwargs(self, output_dir: str) -> None:
        """Main() must forward args.verbose/args.dry_run to build_engine as keyword args, not attribute access on
        build_engine's side — regression test for the latent AttributeError risk when args lacks these attrs."""
        args = self._make_args(output_dir=output_dir, tensorrt=True, verbose=True, dry_run=True)
        onnx_output = str(args.output_dir) + "/inference_model.onnx"

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.backbone.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.projector.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.encoder.parameters.return_value = []
        mock_model.transformer.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = {
            "pred_boxes": torch.zeros(1, 300, 4),
            "pred_logits": torch.zeros(1, 300, 90),
        }
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_build_engine = MagicMock(return_value=str(args.output_dir) + "/inference_model.trt")

        with (
            patch.object(_cli_export_module, "build_model", return_value=(mock_model, MagicMock(), MagicMock())),
            patch.object(_cli_export_module, "make_infer_image", return_value=mock_tensor),
            patch.object(_cli_export_module, "export_onnx", return_value=onnx_output),
            patch.object(_cli_export_module, "build_engine", mock_build_engine),
            patch.object(_cli_export_module, "get_rank", return_value=0),
        ):
            _cli_export_module.main(args)

        mock_build_engine.assert_called_once_with(onnx_output, fp16=True, verbose=True, dry_run=True, output_name=None)

    def test_tensorrt_false_does_not_call_build_engine(self, output_dir: str) -> None:
        """When tensorrt=False (default), main() must not call build_engine."""
        build_engine_calls: list[str] = []

        def fake_build_engine(onnx_path: str, **_kwargs) -> str:
            build_engine_calls.append(onnx_path)
            return onnx_path.replace(".onnx", ".trt")

        args = self._make_args(output_dir=output_dir, tensorrt=False)

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.backbone.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.projector.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.encoder.parameters.return_value = []
        mock_model.transformer.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor

        with (
            patch.object(_cli_export_module, "build_model", return_value=(mock_model, MagicMock(), MagicMock())),
            patch.object(_cli_export_module, "make_infer_image", return_value=mock_tensor),
            patch.object(_cli_export_module, "export_onnx", return_value=str(args.output_dir) + "/model.onnx"),
            patch.object(_cli_export_module, "build_engine", side_effect=fake_build_engine),
            patch.object(_cli_export_module, "get_rank", return_value=0),
        ):
            _cli_export_module.main(args)

        assert len(build_engine_calls) == 0, "build_engine must not be called when tensorrt=False"

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu", id="cpu"),
            pytest.param("cuda", id="cuda"),
        ],
    )
    def test_model_moved_to_correct_device(self, output_dir: str, device: str, monkeypatch) -> None:
        """model.to() and input_tensors.to() must use args.device, not a hard-coded 'cuda'.

        Before the fix, export/main.py line 145 called model.eval().to('cuda') unconditionally, which crashed when
        CUDA_VISIBLE_DEVICES was blank (CPU export).
        """
        import torch

        to_calls = []

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.backbone.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.projector.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.encoder.parameters.return_value = []
        mock_model.transformer.parameters.return_value = []

        # Capture .to() calls
        def _record_to(dev):
            to_calls.append(str(dev))
            return mock_model

        mock_model.to.side_effect = _record_to
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = {"pred_boxes": torch.zeros(1, 300, 4), "pred_logits": torch.zeros(1, 300, 90)}

        mock_tensor = MagicMock()
        tensor_to_calls = []

        def _tensor_to(dev):
            tensor_to_calls.append(str(dev))
            return mock_tensor

        mock_tensor.to.side_effect = _tensor_to
        mock_tensor.cpu.return_value = mock_tensor

        # When testing "cuda" path, pretend CUDA is not available so the
        # fallback-to-cpu branch fires and we can verify the warning without
        # needing a GPU in CI.
        monkeypatch.setattr(torch, "cuda", MagicMock(is_available=MagicMock(return_value=False)))

        args = self._make_args(output_dir=output_dir)
        args.device = device

        with (
            patch.object(_cli_export_module, "build_model", return_value=(mock_model, MagicMock(), MagicMock())),
            patch.object(_cli_export_module, "make_infer_image", return_value=mock_tensor),
            patch.object(_cli_export_module, "export_onnx", return_value=str(output_dir) + "/m.onnx"),
            patch.object(_cli_export_module, "get_rank", return_value=0),
        ):
            _cli_export_module.main(args)

        # The model must NOT have been moved to a hard-coded "cuda" device when
        # device="cpu" — verify the fallback CPU path was taken.
        assert "cuda" not in to_calls, f"model was moved to 'cuda' unexpectedly: {to_calls}"


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

        def _fake_export_onnx(*_a, **_kw):
            return str(tmp_path / "inference_model.onnx")

        monkeypatch.setattr("rfdetr.export.main.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export.main.export_onnx", _fake_export_onnx)
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
    from rfdetr.export.main import make_infer_image

    h, w, b = 112, 224, 2
    tensor = make_infer_image(infer_dir=None, shape=(h, w), batch_size=b, device="cpu")
    assert tensor.shape == (b, 3, h, w), f"Expected shape ({b}, 3, {h}, {w}), got {tensor.shape}"


# ---------------------------------------------------------------------------
# ONNX export variant naming
# ---------------------------------------------------------------------------


class TestExportOnnxVariantNaming:
    """Verify that export_onnx uses variant_name in the output filename."""

    def test_variant_name_in_filename(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When variant_name is provided, the ONNX file is named after the variant."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]  # 3rd positional arg is output_file

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["dets"],
            dynamic_axes=None,
            verbose=False,
            variant_name="rfdetr-medium",
        )

        assert captured["output_file"].endswith("rfdetr-medium.onnx")

    def test_variant_name_with_backbone(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """backbone_only + variant_name produces '{variant}-backbone.onnx'."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["features"],
            dynamic_axes=None,
            backbone_only=True,
            verbose=False,
            variant_name="rfdetr-nano",
        )

        assert captured["output_file"].endswith("rfdetr-nano-backbone.onnx")

    def test_default_name_without_variant(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Without variant_name, falls back to 'inference_model.onnx'."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["dets"],
            dynamic_axes=None,
            verbose=False,
        )

        assert captured["output_file"].endswith("inference_model.onnx")

    def test_default_backbone_name_without_variant(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Without variant_name + backbone_only, falls back to 'backbone_model.onnx'."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["features"],
            dynamic_axes=None,
            backbone_only=True,
            verbose=False,
        )

        assert captured["output_file"].endswith("backbone_model.onnx")

    def test_output_name_overrides_variant_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """output_name takes precedence over variant_name and is used verbatim."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["dets"],
            dynamic_axes=None,
            verbose=False,
            variant_name="rfdetr-medium",
            output_name="my-model",
        )

        assert captured["output_file"].endswith("my-model.onnx")

    def test_output_name_with_backbone_keeps_structural_suffix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """output_name + backbone_only still appends '-backbone' (structural, not a precision detail)."""
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=["features"],
            dynamic_axes=None,
            backbone_only=True,
            verbose=False,
            output_name="my-model",
        )

        assert captured["output_file"].endswith("my-model-backbone.onnx")

    def test_rfdetr_export_passes_variant_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """RFDETR.export() passes self.size as variant_name to export_onnx."""
        captured: dict = {}

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(model=_DummyCoreModel(), device="cpu", resolution=14),
            model_config=types.SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                num_channels=3,
            ),
            size="rfdetr-medium",
        )

        def _fake_make_infer_image(*_args, **_kwargs):
            return torch.zeros(1, 3, 14, 14)

        def _fake_export_onnx(*_args, variant_name=None, **_kw):
            captured["variant_name"] = variant_name
            return str(tmp_path / "rfdetr-medium.onnx")

        monkeypatch.setattr("rfdetr.export.main.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export.main.export_onnx", _fake_export_onnx)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(14, 14))

        assert captured["variant_name"] == "rfdetr-medium"

    def test_rfdetr_export_passes_none_when_size_not_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Base RFDETR (size=None) passes None as variant_name."""
        captured: dict = {}

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(model=_DummyCoreModel(), device="cpu", resolution=14),
            model_config=types.SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                num_channels=3,
            ),
            size=None,
        )

        def _fake_make_infer_image(*_args, **_kwargs):
            return torch.zeros(1, 3, 14, 14)

        def _fake_export_onnx(*_args, variant_name=None, **_kw):
            captured["variant_name"] = variant_name
            return str(tmp_path / "inference_model.onnx")

        monkeypatch.setattr("rfdetr.export.main.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export.main.export_onnx", _fake_export_onnx)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(14, 14))

        assert captured["variant_name"] is None

    def test_rfdetr_export_passes_output_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """RFDETR.export()'s output_name kwarg reaches export_onnx alongside variant_name (output_name wins)."""
        captured: dict = {}

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(model=_DummyCoreModel(), device="cpu", resolution=14),
            model_config=types.SimpleNamespace(
                segmentation_head=False,
                use_grouppose_keypoints=False,
                num_channels=3,
            ),
            size="rfdetr-medium",
        )

        def _fake_make_infer_image(*_args, **_kwargs):
            return torch.zeros(1, 3, 14, 14)

        def _fake_export_onnx(*_args, variant_name=None, output_name=None, **_kw):
            captured["variant_name"] = variant_name
            captured["output_name"] = output_name
            return str(tmp_path / "my-model.onnx")

        monkeypatch.setattr("rfdetr.export.main.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export.main.export_onnx", _fake_export_onnx)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(14, 14), output_name="my-model")

        assert captured["variant_name"] == "rfdetr-medium"
        assert captured["output_name"] == "my-model"

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

        monkeypatch.setattr(_cli_export_module.torch.onnx, "export", _fake_onnx_export)

        _cli_export_module.export_onnx(
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
