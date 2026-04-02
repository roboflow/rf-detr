# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Tests for model export functionality.

Use cases covered:
- Segmentation outputs must be present in both train/eval modes to avoid export crashes.
- Export should not change the original model's training state.
- CLI export path (deploy.export.main) must include 'masks' in output_names for
  segmentation models, call make_infer_image with the correct individual args, and
  call export_onnx with args.output_dir as the first argument.
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

from rfdetr import RFDETRSegNano
from rfdetr import detr as _detr_module
from rfdetr.export import main as _cli_export_module

_IS_ONNX_INSTALLED = importlib.util.find_spec("onnx") is not None


@contextmanager
def ignore_tracer_warnings() -> Iterator[None]:
    """Suppress torch.jit.TracerWarning during export tests to reduce log spam."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        yield


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
    """
    Integration test: exporting a segmentation model should not crash.

    This exercises the full export path to ensure no AttributeError occurs.
    """
    model = RFDETRSegNano()

    # This should not crash with "AttributeError: 'dict' object has no attribute 'shape'"
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), simplify=False, verbose=False)

    # Verify export produced output files
    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnx]")
def test_export_does_not_change_original_training_state(tmp_path: Path) -> None:
    """
    Verify that calling export() does not change the original model's train/eval state.

    This ensures that export() puts a deepcopy of the model in eval mode without
    mutating the underlying training model used by RF-DETR.
    """
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model), as in other tests
    torch_model = model.model.model.to("cuda")

    # Ensure the original model is in training mode
    torch_model.train()
    assert torch_model.training is True, "Precondition: original model should start in training mode"

    # Call export() on the high-level model; this should not change the original model's mode
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), simplify=False)

    # After export, the original underlying model should still be in training mode
    assert torch_model.training is True, "export() should not change the original model's training state"


@pytest.fixture
def _detr_export_scaffold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Shared scaffold for RFDETR.export() deprecated-argument tests."""

    class _DummyCoreModel:
        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        def cpu(self):
            return self

        def __call__(self, *_args, **_kwargs):
            return {
                "pred_boxes": torch.zeros(1, 1, 4),
                "pred_logits": torch.zeros(1, 1, 2),
                "pred_masks": torch.zeros(1, 1, 2, 2),
            }

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            model=_DummyCoreModel(),
            device="cpu",
            resolution=14,
        ),
        model_config=types.SimpleNamespace(segmentation_head=False),
    )

    export_called: dict[str, bool] = {"value": False}

    def _fake_make_infer_image(*_args, **_kwargs):
        return torch.zeros(1, 3, 14, 14)

    def _fake_export_onnx(*_args, **_kwargs):
        export_called["value"] = True
        return str(tmp_path / "inference_model.onnx")

    monkeypatch.setattr("rfdetr.export.main.make_infer_image", _fake_make_infer_image)
    monkeypatch.setattr("rfdetr.export.main.export_onnx", _fake_export_onnx)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

    return model, export_called


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
    """`RFDETR.export(..., dynamic_batch=True)` must pass a non-None `dynamic_axes` dict
    to `export_onnx`; `dynamic_batch=False` must pass `None`.
    """

    class _DummyCoreModel:
        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        def cpu(self):
            return self

        def __call__(self, *_args, **_kwargs):
            if segmentation_head:
                return {
                    "pred_boxes": torch.zeros(1, 1, 4),
                    "pred_logits": torch.zeros(1, 1, 2),
                    "pred_masks": torch.zeros(1, 1, 2, 2),
                }
            return {"pred_boxes": torch.zeros(1, 1, 4), "pred_logits": torch.zeros(1, 1, 2)}

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(model=_DummyCoreModel(), device="cpu", resolution=14),
        model_config=types.SimpleNamespace(segmentation_head=segmentation_head),
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


def test_export_simplify_flag_is_ignored_with_deprecation_warning(_detr_export_scaffold: tuple, tmp_path: Path) -> None:
    """`simplify=True` should not run ONNX simplification and should emit a deprecation warning."""
    model, export_called = _detr_export_scaffold
    with pytest.deprecated_call(match=r".*`export`.*deprecated.*`simplify`.*"):
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), simplify=True, verbose=False, shape=(14, 14))
    assert export_called["value"] is True


def test_export_force_flag_is_ignored_with_deprecation_warning(_detr_export_scaffold: tuple, tmp_path: Path) -> None:
    """`force=True` should be a no-op and emit a deprecation warning."""
    model, export_called = _detr_export_scaffold
    with pytest.deprecated_call(match=r".*`export`.*deprecated.*`force`.*"):
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), force=True, verbose=False, shape=(14, 14))
    assert export_called["value"] is True


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["train", "eval"], ids=["train_mode", "eval_mode"])
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
    """
    Unit tests for deploy.export.main() (CLI export path).

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
        simplify: bool = False,
        tensorrt: bool = False,
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
            simplify=simplify,
            tensorrt=tensorrt,
            dynamic_batch=dynamic_batch,
        )

    @staticmethod
    def _run(args: types.SimpleNamespace) -> tuple[dict, dict]:
        """
        Run deploy.export.main(args) with all heavy dependencies mocked.

        Stubs out build_model, make_infer_image, and export_onnx, and injects
        mock onnx/onnxsim modules so the export module can be imported even when
        those optional packages are not installed.

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
        """
        export_onnx must receive the correct output_names for every model type.

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
        """
        make_infer_image must be called with (infer_dir, shape, batch_size, device),
        not with the whole args Namespace.

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
        """
        export_onnx must be called as export_onnx(output_dir, model, ...) with
        backbone_only, verbose, and opset_version forwarded as keyword args.

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

    def test_simplify_flag_logs_warning_and_continues_export(self, output_dir: str) -> None:
        """CLI --simplify=True must log a deprecation warning and still call export_onnx.

        The flag is now a no-op: the logger emits a warning and export continues
        without running ONNX simplification.
        """
        args = self._make_args(output_dir=output_dir, simplify=True)
        export_onnx_called: dict[str, bool] = {"value": False}

        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.backbone.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.projector.parameters.return_value = []
        mock_model.backbone.__getitem__.return_value.encoder.parameters.return_value = []
        mock_model.transformer.parameters.return_value = []
        mock_model.to.return_value = mock_model
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = {"pred_boxes": torch.zeros(1, 300, 4), "pred_logits": torch.zeros(1, 300, 90)}

        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor

        def fake_export_onnx(*_args, **_kwargs):
            export_onnx_called["value"] = True
            return str(output_dir) + "/inference_model.onnx"

        with (
            patch.object(_cli_export_module, "build_model", return_value=(mock_model, MagicMock(), MagicMock())),
            patch.object(_cli_export_module, "make_infer_image", return_value=mock_tensor),
            patch.object(_cli_export_module, "export_onnx", side_effect=fake_export_onnx),
            patch.object(_cli_export_module, "get_rank", return_value=0),
            patch.object(_cli_export_module, "logger") as mock_logger,
        ):
            _cli_export_module.main(args)

        mock_logger.warning.assert_called_once()
        assert "simplify" in mock_logger.warning.call_args[0][0].lower()
        assert export_onnx_called["value"] is True, "export_onnx should still be called with simplify=True"

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


class TestExportPatchSize:
    """RFDETR.export() patch_size validation and shape-divisibility tests."""

    @staticmethod
    def _scaffold(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, patch_size: int, num_windows: int
    ) -> types.SimpleNamespace:
        """Build a minimal RFDETR-like namespace with controllable patch_size/num_windows."""

        class _DummyCoreModel:
            def to(self, *_a, **_kw):
                return self

            def eval(self):
                return self

            def cpu(self):
                return self

            def __call__(self, *_a, **_kw):
                return {"pred_boxes": torch.zeros(1, 1, 4), "pred_logits": torch.zeros(1, 1, 2)}

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                model=_DummyCoreModel(),
                device="cpu",
                resolution=patch_size * num_windows * 2,  # always valid
            ),
            model_config=types.SimpleNamespace(
                segmentation_head=False,
                patch_size=patch_size,
                num_windows=num_windows,
            ),
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
        """export() must raise ValueError when patch_size is not a positive integer."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=4)
        # Keep model_config.patch_size consistent with the patch_size argument for this test
        model.model_config.patch_size = bad_patch_size
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), patch_size=bad_patch_size)

    def test_export_shape_must_be_divisible_by_block_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """export() must reject shapes not divisible by patch_size * num_windows."""
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
        """export() must reject non-positive shape dimensions (Python -N % M == 0 wraps silently)."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=16, num_windows=2)
        with pytest.raises(ValueError, match="positive integers"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=bad_shape)

    def test_export_shape_valid_for_block_size(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """export() accepts shape divisible by patch_size * num_windows without error."""
        # patch_size=16, num_windows=2 → block_size=32; shape (64, 64) is valid
        model = self._scaffold(monkeypatch, tmp_path, patch_size=16, num_windows=2)
        # Should not raise
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(64, 64))

    @pytest.mark.parametrize("bad_patch_size", [True, False])
    def test_export_bool_patch_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_patch_size: bool
    ) -> None:
        """export() must reject bool values for patch_size (isinstance(True, int) is True)."""
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
        """export() must raise ValueError for float, bool, or wrong-arity shape tuples."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        with pytest.raises(ValueError, match="shape"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=bad_shape)

    @pytest.mark.parametrize("bad_num_windows", [0, -1, True])
    def test_export_invalid_num_windows_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_num_windows: int
    ) -> None:
        """export() must raise ValueError when model_config.num_windows is not a positive integer."""
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=1)
        model.model_config.num_windows = bad_num_windows
        with pytest.raises(ValueError, match="num_windows must be a positive integer"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path))

    def test_export_default_resolution_not_divisible_by_block_size_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """export() with shape=None must raise ValueError when model.resolution % block_size != 0."""
        # patch_size=14, num_windows=3 → block_size=42; scaffold sets resolution=84 (42*2) which is valid
        # Override resolution to 50 (not divisible by 42) to trigger the check
        model = self._scaffold(monkeypatch, tmp_path, patch_size=14, num_windows=3)
        model.model.resolution = 50
        with pytest.raises(ValueError, match="default resolution"):
            _detr_module.RFDETR.export(model, output_dir=str(tmp_path))


def test_make_infer_image_produces_correct_rectangular_shape() -> None:
    """make_infer_image must produce a (B, C, H, W) tensor for non-square shapes.

    Regression test for the square-resize bug where ``Resize((shape[0], shape[0]))``
    was used instead of ``Resize((shape[0], shape[1]))``, causing the output width
    to silently equal the height.
    """
    from rfdetr.export.main import make_infer_image

    h, w, b = 112, 224, 2
    tensor = make_infer_image(infer_dir=None, shape=(h, w), batch_size=b, device="cpu")
    assert tensor.shape == (b, 3, h, w), f"Expected shape ({b}, 3, {h}, {w}), got {tensor.shape}"
