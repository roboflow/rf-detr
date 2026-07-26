# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for native CoreML (``.mlpackage``) export.

Covers:
* ``export_coreml()`` — argument/dependency behaviour (``coremltools`` mocked where needed)
* ``format="coreml"`` wiring through ``RFDETR.export()``
* End-to-end convert + numerical parity (``@pytest.mark.coreml_e2e``, opt-in)

Parity inputs are spatially structured (gradient + checkerboard) and
``download_assets(ImageAssets.PEOPLE_WALKING)`` under ``coreml_e2e`` (no committed images).
Random Gaussian noise is intentionally avoided: it can hide export/runtime divergence that
only appears on correlated image structure.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from supervision.assets import ImageAssets, download_assets

from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._coreml.converter import _check_coremltools_available, export_coreml

coreml_only = pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="coremltools not installed")

# FLOAT32 CoreML convert matches eager to ~1e-5 on boxes/logits; masks need a bit more
# headroom (~8e-5 observed on SegNano). Bound stays well under structural-failure scale (>=1e-3).
_COREML_MAX_ABS_DIFF = 1e-4

# ImageNet stats used by RF-DETR preprocess / exported CoreML bundles.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _structured_parity_input(
    batch: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build a deterministic, spatially correlated ``NCHW`` tensor for export parity.

    Combines a smooth spatial gradient with a coarse checkerboard so the backbone sees
    local structure (unlike ``torch.randn``, which can mask export/runtime divergence).

    Args:
        batch: Batch size.
        channels: Channel count (typically 3).
        height: Spatial height.
        width: Spatial width.

    Returns:
        Float tensor shaped ``(batch, channels, height, width)`` in roughly ImageNet-normalized range.
    """
    ys = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    gradient = (0.35 * ys + 0.25 * xs).expand(1, channels, height, width).clone()
    # Per-channel offset so RGB planes are not identical.
    for c in range(channels):
        gradient[:, c] = gradient[:, c] + 0.05 * (c - 1)

    tile = 16
    yy = (torch.arange(height).view(height, 1) // tile) % 2
    xx = (torch.arange(width).view(1, width) // tile) % 2
    checker = ((yy + xx) % 2).to(dtype=torch.float32).view(1, 1, height, width)
    checker = (checker * 0.4 - 0.2).expand(1, channels, height, width)

    sample = gradient + checker
    return sample.expand(batch, channels, height, width).contiguous()


def _parity_input_from_image(path: Path, resolution: int) -> torch.Tensor:
    """Load an RGB image, resize to square ``resolution``, and ImageNet-normalize (predict-style).

    Args:
        path: Path to an RGB image file.
        resolution: Square side length matching the exported model.

    Returns:
        Float tensor shaped ``(1, 3, resolution, resolution)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"parity fixture image not found: {path}")
    with Image.open(path) as img:
        image = img.convert("RGB")
    tensor = TF.to_tensor(image)
    tensor = TF.resize(tensor, [resolution, resolution], antialias=False)
    tensor = TF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    return tensor.unsqueeze(0)


def _coreml_parity_diffs(
    mlpackage_path: Path,
    pytorch_model: torch.nn.Module,
    example_input: torch.Tensor,
) -> list[float]:
    """Run *example_input* through eager export-mode PyTorch and CoreML; return per-output max-abs-diff.

    Export-mode ``forward`` can mutate its input, so each side gets a fresh clone. CoreML outputs are
    taken in ``MLModel`` spec order and paired with the eager tuple in the same order.

    Args:
        mlpackage_path: Path to the exported ``.mlpackage``.
        pytorch_model: Export-mode PyTorch module on CPU.
        example_input: ``(N, C, H, W)`` example tensor.

    Returns:
        One max-abs-diff per output tensor.

    Raises:
        AssertionError: If output counts or shapes disagree.
        ImportError: If ``coremltools`` is not installed.
    """
    import coremltools as ct

    with torch.no_grad():
        eager_out = pytorch_model(example_input.clone())
    if not isinstance(eager_out, tuple):
        raise AssertionError(f"export-mode forward must return a tuple, got {type(eager_out)!r}")
    eager_tensors = [t.detach().float().cpu() for t in eager_out if isinstance(t, torch.Tensor)]

    # CPU_ONLY avoids ANE/GPU fp16 execution drift when validating FLOAT32 bundles.
    mlmodel = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_names = [o.name for o in spec.description.output]
    prediction = mlmodel.predict({input_name: np.ascontiguousarray(example_input.detach().cpu().numpy())})
    coreml_tensors = [torch.from_numpy(np.asarray(prediction[name], dtype=np.float32)) for name in output_names]

    assert len(coreml_tensors) == len(eager_tensors), (
        f"CoreML output count {len(coreml_tensors)} != PyTorch {len(eager_tensors)} (spec names={output_names})"
    )
    diffs: list[float] = []
    for idx, (eager, coreml) in enumerate(zip(eager_tensors, coreml_tensors)):
        assert eager.shape == coreml.shape, (
            f"output[{idx}] shape mismatch: PyTorch {tuple(eager.shape)} vs CoreML {tuple(coreml.shape)} "
            f"(name={output_names[idx]!r})"
        )
        diffs.append((eager - coreml).abs().max().item())
    return diffs


def validate_detection_coreml_vs_pytorch(
    mlpackage_path: Path,
    pytorch_model: torch.nn.Module,
    example_input: torch.Tensor,
) -> None:
    """Compare CoreML detection outputs (boxes, logits) to eager export-mode PyTorch.

    Args:
        mlpackage_path: Path to the exported ``.mlpackage``.
        pytorch_model: Export-mode PyTorch module on CPU.
        example_input: ``(N, C, H, W)`` tensor used for both forwards.

    Raises:
        AssertionError: When output count/shape disagrees or max-abs-diff exceeds tolerance.
    """
    diffs = _coreml_parity_diffs(mlpackage_path, pytorch_model, example_input)
    assert len(diffs) == 2, f"detection export must yield (boxes, logits), got {len(diffs)} outputs"
    assert max(diffs) < _COREML_MAX_ABS_DIFF, (
        f"CoreML detection outputs diverge from PyTorch: max abs diff {max(diffs)} "
        f"(boxes={diffs[0]}, logits={diffs[1]}, bound={_COREML_MAX_ABS_DIFF})"
    )


def validate_segmentation_coreml_vs_pytorch(
    mlpackage_path: Path,
    pytorch_model: torch.nn.Module,
    example_input: torch.Tensor,
) -> None:
    """Compare CoreML segmentation outputs (boxes, logits, masks) to eager export-mode PyTorch.

    Args:
        mlpackage_path: Path to the exported ``.mlpackage``.
        pytorch_model: Export-mode PyTorch segmentation module on CPU.
        example_input: ``(N, C, H, W)`` tensor used for both forwards.

    Raises:
        AssertionError: When output count/shape disagrees or max-abs-diff exceeds tolerance.
    """
    diffs = _coreml_parity_diffs(mlpackage_path, pytorch_model, example_input)
    assert len(diffs) == 3, f"segmentation export must yield (boxes, logits, masks), got {len(diffs)} outputs"
    assert max(diffs) < _COREML_MAX_ABS_DIFF, (
        f"CoreML segmentation outputs diverge from PyTorch: max abs diff {max(diffs)} "
        f"(boxes={diffs[0]}, logits={diffs[1]}, masks={diffs[2]}, bound={_COREML_MAX_ABS_DIFF})"
    )


# ---------------------------------------------------------------------------
# export_coreml() — unit / dependency behaviour
# ---------------------------------------------------------------------------


class TestCheckCoremltoolsAvailable:
    """Tests for ``_check_coremltools_available``."""

    def test_raises_import_error_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing ``coremltools`` must raise ImportError with install hint."""
        if "coremltools" in sys.modules:
            monkeypatch.delitem(sys.modules, "coremltools", raising=False)

        real_import = __import__

        def _block_coremltools(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "coremltools" or name.startswith("coremltools."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _block_coremltools)
        with pytest.raises(ImportError, match="rfdetr\\[coreml\\]"):
            _check_coremltools_available()

    @coreml_only
    def test_succeeds_when_installed(self) -> None:
        """Installed ``coremltools`` must not raise."""
        _check_coremltools_available()


class TestExportCoremlValidation:
    """Argument and dependency behaviour of ``export_coreml()`` (no real convert)."""

    def test_dynamic_batch_raises_not_implemented(self, tmp_path: Path) -> None:
        """``dynamic_batch=True`` must be refused (ANE / static-shape friendly export)."""
        model = torch.nn.Linear(1, 1)
        example = torch.zeros(1, 3, 32, 32)
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            export_coreml(model, example, tmp_path, dynamic_batch=True)

    def test_missing_coremltools_raises_import_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``export_coreml`` must surface the install hint when coremltools is absent."""
        monkeypatch.setattr(
            "rfdetr.export._coreml.converter._check_coremltools_available",
            mock.Mock(side_effect=ImportError("pip install rfdetr[coreml]")),
        )
        model = torch.nn.Linear(1, 1)
        example = torch.zeros(1, 3, 32, 32)
        with pytest.raises(ImportError, match="rfdetr\\[coreml\\]"):
            export_coreml(model, example, tmp_path)


class TestExportFormatParameter:
    """Tests for ``format="coreml"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock heavy export deps so ``RFDETR.export()`` stays fast."""
        self._tmp_path = tmp_path
        mlpackage = tmp_path / "inference_model.mlpackage"
        mlpackage.mkdir()

        self._mock_stack = contextlib.ExitStack()
        self._mock_export_onnx = self._mock_stack.enter_context(
            mock.patch("rfdetr.export.main.export_onnx", return_value=str(tmp_path / "inference_model.onnx"))
        )
        self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export.main.make_infer_image",
                return_value=torch.zeros(1, 3, 560, 560),
            )
        )
        self._mock_export_coreml = self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export._coreml.converter.export_coreml",
                return_value=mlpackage,
            )
        )
        yield
        self._mock_stack.close()

    @staticmethod
    def _make_rfdetr(*, segmentation_head: bool = False) -> Any:
        """Create a minimal RFDETR instance with mocked internals.

        Args:
            segmentation_head: Whether the mocked config reports a seg head.
        """
        from rfdetr.detr import RFDETR

        obj = RFDETR.__new__(RFDETR)
        obj.model = mock.MagicMock()
        obj.model.resolution = 560
        obj.model.device = "cpu"
        obj.model.model.to.return_value = obj.model.model
        obj.model_config = mock.MagicMock()
        obj.model_config.segmentation_head = segmentation_head
        obj.model_config.use_grouppose_keypoints = False
        obj.model_config.patch_size = 14
        obj.model_config.num_windows = 1
        obj.model_config.num_channels = 3
        return obj

    def test_coreml_format_calls_export_coreml(self) -> None:
        """``format="coreml"`` must dispatch to ``export_coreml`` and warn (experimental)."""
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match="experimental"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"))
        self._mock_export_coreml.assert_called_once()

    def test_coreml_format_does_not_call_export_onnx(self) -> None:
        """Native CoreML must not go through the ONNX interchange path."""
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match="experimental"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"))
        self._mock_export_onnx.assert_not_called()

    def test_onnx_format_does_not_call_export_coreml(self) -> None:
        """``format="onnx"`` must not import/call the CoreML converter."""
        obj = self._make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_export_coreml.assert_not_called()

    def test_segmentation_model_still_dispatches_to_coreml(self) -> None:
        """Seg models use the same ``format="coreml"`` path (outputs from model config)."""
        obj = self._make_rfdetr(segmentation_head=True)
        with pytest.warns(UserWarning, match="experimental"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"))
        self._mock_export_coreml.assert_called_once()

    def test_notes_warns_and_is_ignored(self) -> None:
        """``notes`` must warn for CoreML (no ONNX-style metadata slot) but still export."""
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match=r"`notes` is not forwarded to format='coreml'"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"), notes="hello")
        self._mock_export_coreml.assert_called_once()

    def test_dynamic_batch_raises_before_converter(self) -> None:
        """``dynamic_batch=True`` is refused by ``RFDETR.export()`` before the converter is invoked."""
        obj = self._make_rfdetr()
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"), dynamic_batch=True)
        self._mock_export_coreml.assert_not_called()

    def test_invalid_format_raises_value_error(self) -> None:
        """Unknown ``format`` must raise ``ValueError`` listing supported formats."""
        obj = self._make_rfdetr()
        with pytest.raises(ValueError, match="Unsupported export format"):
            obj.export(format="bogus", output_dir=str(self._tmp_path / "out"))


# ---------------------------------------------------------------------------
# End-to-end (gated) — real convert + FLOAT32 CPU parity vs eager PyTorch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coreml_detection_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRNano to a ``.mlpackage`` once for e2e tests."""
    from rfdetr import RFDETRNano

    out_dir = tmp_path_factory.mktemp("coreml_det")
    detector = RFDETRNano(pretrain_weights=None)
    mlpackage_path = detector.export(output_dir=str(out_dir), format="coreml", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return model, example, Path(mlpackage_path)


@pytest.fixture(scope="module")
def coreml_segmentation_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRSegNano to a ``.mlpackage`` once for e2e tests."""
    from rfdetr import RFDETRSegNano

    out_dir = tmp_path_factory.mktemp("coreml_seg")
    detector = RFDETRSegNano(pretrain_weights=None)
    mlpackage_path = detector.export(output_dir=str(out_dir), format="coreml", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return model, example, Path(mlpackage_path)


@coreml_only
@pytest.mark.coreml_e2e
class TestCoreMLEndToEnd:
    """Real CoreML export + FLOAT32 CPU numerical parity (``-m coreml_e2e``)."""

    def test_mlpackage_written_for_detection(self, coreml_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        """Detection export must write a non-empty ``.mlpackage`` directory/bundle."""
        _, _, mlpackage_path = coreml_detection_export
        assert mlpackage_path.exists()
        assert mlpackage_path.suffix == ".mlpackage" or mlpackage_path.name.endswith(".mlpackage")

    def test_mlpackage_written_for_segmentation(
        self, coreml_segmentation_export: tuple[Any, torch.Tensor, Path]
    ) -> None:
        """Segmentation export must write a non-empty ``.mlpackage`` bundle."""
        _, _, mlpackage_path = coreml_segmentation_export
        assert mlpackage_path.exists()
        assert mlpackage_path.suffix == ".mlpackage" or mlpackage_path.name.endswith(".mlpackage")

    def test_detection_outputs_match_pytorch_structured(
        self, coreml_detection_export: tuple[Any, torch.Tensor, Path]
    ) -> None:
        """CoreML detection matches eager on structured (gradient+checkerboard) input."""
        model, example, mlpackage_path = coreml_detection_export
        validate_detection_coreml_vs_pytorch(mlpackage_path, model, example)

    def test_segmentation_outputs_match_pytorch_structured(
        self, coreml_segmentation_export: tuple[Any, torch.Tensor, Path]
    ) -> None:
        """CoreML segmentation matches eager on structured (gradient+checkerboard) input."""
        model, example, mlpackage_path = coreml_segmentation_export
        validate_segmentation_coreml_vs_pytorch(mlpackage_path, model, example)

    def test_detection_outputs_match_pytorch_supervision_image(
        self,
        coreml_detection_export: tuple[Any, torch.Tensor, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CoreML detection matches eager on ``ImageAssets.PEOPLE_WALKING``."""
        model, structured, mlpackage_path = coreml_detection_export
        monkeypatch.chdir(tmp_path)
        example = _parity_input_from_image(
            Path(download_assets(ImageAssets.PEOPLE_WALKING)),
            int(structured.shape[-1]),
        )
        validate_detection_coreml_vs_pytorch(mlpackage_path, model, example)

    def test_segmentation_outputs_match_pytorch_supervision_image(
        self,
        coreml_segmentation_export: tuple[Any, torch.Tensor, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CoreML segmentation matches eager on ``ImageAssets.PEOPLE_WALKING``."""
        model, structured, mlpackage_path = coreml_segmentation_export
        monkeypatch.chdir(tmp_path)
        example = _parity_input_from_image(
            Path(download_assets(ImageAssets.PEOPLE_WALKING)),
            int(structured.shape[-1]),
        )
        validate_segmentation_coreml_vs_pytorch(mlpackage_path, model, example)


class TestCoreMLParityInputHelpers:
    """Unit checks for CoreML-local parity input builders (no coremltools required)."""

    def test_structured_parity_input_shape_and_determinism(self) -> None:
        """Structured tensor must be ``NCHW``, finite, and seed-independent-deterministic."""
        a = _structured_parity_input(1, 3, 64, 64)
        b = _structured_parity_input(1, 3, 64, 64)
        assert a.shape == (1, 3, 64, 64)
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)
        # Spatially varying — not a constant fill.
        assert float(a.std()) > 1e-3

    def test_parity_input_from_image_loads_rgb(self, tmp_path: Path) -> None:
        """``_parity_input_from_image`` must normalize a local RGB file to ``1x3xHxW``."""
        image_path = tmp_path / "tiny.png"
        Image.new("RGB", (32, 24), color=(20, 40, 60)).save(image_path)
        tensor = _parity_input_from_image(image_path, 64)
        assert tensor.shape == (1, 3, 64, 64)
        assert torch.isfinite(tensor).all()
