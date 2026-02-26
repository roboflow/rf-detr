# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Tests for model export functionality.

Use cases covered:
- Export should use eval() on the deepcopy (not the original model).
- Segmentation outputs must be present in both train/eval modes to avoid export crashes.
- Segmentation model exports must include 'masks' in output_names (PR #402).
- CLI export path (deploy.export.main) must include 'masks' in output_names for
  segmentation models, call make_infer_image with the correct individual args, and
  call export_onnx with args.output_dir as the first argument.
"""

import importlib.util
import inspect
import sys
import types
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.jit import TracerWarning

from rfdetr import RFDETRSegNano
from rfdetr.deploy import export as _cli_export_module

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
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnxexport]")
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
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnxexport]")
def test_export_calls_eval_on_deepcopy_not_original(tmp_path: Path) -> None:
    """
    Verify that Model.export() calls eval() on the deepcopy, not the original model.

    This test patches deepcopy to track whether eval() is called on the copied
    model during export, ensuring the fix in PR #578 is working correctly.
    """
    model = RFDETRSegNano()

    # Access the underlying torch module and set it to training mode
    torch_model = model.model.model.to("cuda")
    torch_model.train()
    assert torch_model.training is True, "Precondition: original model should start in training mode"

    # Store the original deepcopy function
    original_deepcopy = deepcopy

    # Mock to track eval() calls
    eval_mock = Mock()

    def tracking_deepcopy(obj):
        """Deepcopy wrapper that tracks eval() calls on the copy"""
        copied = original_deepcopy(obj)

        # Only track eval calls on torch.nn.Module objects
        if isinstance(copied, torch.nn.Module):
            # Save reference to original eval before replacing it
            original_eval = copied.eval

            def tracked_eval(*args, **kwargs):
                """Wrapper that tracks calls while delegating to the original eval"""
                eval_mock()
                return original_eval(*args, **kwargs)

            # Replace eval with tracked version
            copied.eval = tracked_eval

        return copied

    # Patch deepcopy in the main module where export is defined
    with patch("rfdetr.main.deepcopy", side_effect=tracking_deepcopy):
        try:
            with ignore_tracer_warnings():
                model.export(output_dir=str(tmp_path), simplify=False)
        except (ImportError, OSError, RuntimeError):
            # Expected failures: missing dependencies, network issues, CUDA errors
            # These are acceptable as we're testing the deepcopy/eval pattern, not the full export
            pass

    # Verify that eval() was called on the deepcopy during export
    assert eval_mock.call_count > 0, (
        "export() should call eval() on the deepcopy. "
        "This ensures the exported model is in eval mode without affecting the original."
    )

    # Verify the original model's training state was not changed
    assert torch_model.training is True, "export() should not change the original model's training state"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
@pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed, run: pip install rfdetr[onnxexport]")
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


def _make_export_mocks(tmp_path: Path, segmentation_head: bool, backbone_only: bool = False) -> tuple:
    """
    Build the mock objects needed to unit-test Model.export() without ONNX or a GPU.

    Returns (mock_export_module, mock_model, mock_model_wrapper, captured_kwargs_store)
    where captured_kwargs_store is a dict that will be populated with the kwargs
    passed to export_onnx when export() is called.
    """
    captured_kwargs: dict = {}

    def fake_export_onnx(**kwargs):
        captured_kwargs.update(kwargs)
        return tmp_path / "inference_model.onnx"

    mock_export_module = types.ModuleType("rfdetr.deploy.export")
    mock_export_module.export_onnx = fake_export_onnx
    mock_export_module.onnx_simplify = MagicMock()

    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_export_module.make_infer_image = MagicMock(return_value=mock_tensor)

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.cpu.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.training = False
    if backbone_only:
        # backbone_only: model(input) returns a tensor and .shape is accessed
        feature_tensor = torch.zeros(1, 512, 20, 20)
        mock_model.return_value = feature_tensor
    elif segmentation_head:
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

    mock_model_wrapper = MagicMock()
    mock_model_wrapper.to.return_value = mock_model_wrapper

    return mock_export_module, mock_model, mock_model_wrapper, captured_kwargs


@pytest.mark.parametrize(
    "segmentation_head, backbone_only, expected_output_names",
    [
        pytest.param(True, False, ["dets", "labels", "masks"], id="segmentation_model"),
        pytest.param(False, False, ["dets", "labels"], id="detection_model"),
        pytest.param(False, True, ["features"], id="backbone_only"),
    ],
)
def test_export_output_names(
    tmp_path: Path,
    segmentation_head: bool,
    backbone_only: bool,
    expected_output_names: list[str],
) -> None:
    """
    Unit test: export() must pass the correct output_names to export_onnx.

    Before PR #402, the export method used this one-liner:

        output_names = ['features'] if backbone_only else ['dets', 'labels']

    This always produced ['dets', 'labels'] for segmentation models, omitting
    'masks'.  The parametrized case segmentation_model would therefore FAIL
    with the old code.

    The fix adds an explicit elif branch:

        elif self.args.segmentation_head:
            output_names = ['dets', 'labels', 'masks']
    """
    mock_export_module, mock_model, mock_model_wrapper, captured_kwargs = _make_export_mocks(
        tmp_path, segmentation_head, backbone_only
    )

    with (
        patch("rfdetr.main.build_model", return_value=mock_model_wrapper),
        patch("rfdetr.main.validate_pretrain_weights"),
        patch("rfdetr.main.deepcopy", return_value=mock_model),
        patch.dict("sys.modules", {"rfdetr.deploy.export": mock_export_module}),
    ):
        from rfdetr.main import Model

        model = Model(
            segmentation_head=segmentation_head,
            resolution=560,
            pretrain_weights=None,
            device="cpu",
        )
        model.export(
            output_dir=str(tmp_path),
            backbone_only=backbone_only,
            simplify=False,
            verbose=False,
        )

    assert "output_names" in captured_kwargs, "export_onnx was not called — check that the mock wiring is correct"
    actual = captured_kwargs["output_names"]
    assert actual == expected_output_names, (
        f"output_names mismatch.\n"
        f"  expected : {expected_output_names}\n"
        f"  actual   : {actual}\n"
        "Before PR #402, segmentation models produced ['dets', 'labels'] "
        "instead of ['dets', 'labels', 'masks']."
    )


# --------------------------------------------------------------------------
# Tests for the CLI export path: rfdetr.deploy.export.main()
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
