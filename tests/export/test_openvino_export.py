# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for direct PyTorch -> OpenVINO IR (``.xml``/``.bin``) export.

Covers:
* ``export_openvino()`` — dependency-missing path, path-traversal sanitization, and the internal
  ``ModelWrapper`` dict-output mapping (``openvino.convert_model``/``save_model`` stubbed via
  ``sys.modules`` injection so these run without the real ``openvino`` package installed).
* ``OpenVINOInference.__init__`` — dependency-missing path.
* ``format="openvino"`` wiring through ``RFDETR.export()`` (heavy deps mocked, fast).
* A real end-to-end export + numerical parity check, gated behind ``pytest.importorskip("openvino")``
  so it only runs where the ``openvino`` package is installed.

This repository's CI/dev environment does not install ``openvino`` — the dependency-missing tests below
exercise the real (uninstalled) code path directly rather than mocking an ``ImportError``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from rfdetr.export._openvino.exporter import export_openvino
from rfdetr.export._openvino.inference import OpenVINOInference
from tests.export.conftest import _structured_parity_input, eager_reference_tensors, max_abs_output_diffs


def _stub_openvino_module() -> types.ModuleType:
    """Build a minimal fake ``openvino`` module exposing ``convert_model``/``save_model``.

    Injected into ``sys.modules`` so ``export_openvino()``'s ``from openvino import convert_model,
    save_model`` succeeds without the real package installed, letting the naming/wrapping logic run
    end-to-end while the actual (heavy, unavailable) conversion is a no-op mock.

    Returns:
        A fresh fake module with ``convert_model`` and ``save_model`` as ``MagicMock`` attributes.

    Examples:
        >>> fake = _stub_openvino_module()
        >>> callable(fake.convert_model) and callable(fake.save_model)
        True
    """
    fake = types.ModuleType("openvino")
    fake.convert_model = mock.MagicMock(return_value=mock.MagicMock(name="ov_model"))
    fake.save_model = mock.MagicMock()
    return fake


# ---------------------------------------------------------------------------
# export_openvino() — dependency-missing path (real environment, no openvino installed)
# ---------------------------------------------------------------------------


class TestExportOpenvinoMissingDependency:
    """``export_openvino()``'s ``ImportError`` path, exercised for real (openvino not installed here)."""

    def test_raises_import_error(self, tmp_path: Path) -> None:
        """Missing ``openvino`` must surface an ``ImportError``, not any other exception type."""
        model = torch.nn.Identity()
        example = torch.zeros(1, 3, 32, 32)
        with pytest.raises(ImportError):
            export_openvino(str(tmp_path), model, example)

    def test_logs_pip_install_hint(self, tmp_path: Path) -> None:
        """The logged error must name the ``rfdetr[openvino]`` extra so users know how to fix it.

        ``export_openvino`` re-raises the bare ``ImportError`` from ``from openvino import ...`` (whose message is the
        stdlib's own, e.g. ``"No module named 'openvino'"``), so the actionable install hint lives only in the
        ``logger.error(...)`` call preceding the ``raise`` — not in the exception message itself. Assert on the logged
        call rather than ``pytest.raises(..., match=...)``.
        """
        model = torch.nn.Identity()
        example = torch.zeros(1, 3, 32, 32)
        with mock.patch("rfdetr.export._openvino.exporter.logger.error") as mock_error:
            with pytest.raises(ImportError):
                export_openvino(str(tmp_path), model, example)
        mock_error.assert_called_once()
        logged_message = mock_error.call_args.args[0].replace('"', "")
        assert "rfdetr[openvino]" in logged_message


class TestOpenVINOInferenceMissingDependency:
    """``OpenVINOInference.__init__``'s ``ImportError`` path, exercised for real (openvino not installed)."""

    def test_raises_import_error_before_file_check(self, tmp_path: Path) -> None:
        """A missing ``openvino`` install raises ``ImportError`` even for a nonexistent model path.

        ``OpenVINOInference.__init__`` imports ``openvino`` before checking ``model_path.exists()``, so a nonexistent
        path must still surface ``ImportError`` here (never ``FileNotFoundError`` — that branch is unreachable without
        openvino installed; see ``TestOpenVINOInferenceEndToEnd`` for the gated ``FileNotFoundError`` coverage).
        """
        with pytest.raises(ImportError):
            OpenVINOInference(tmp_path / "does-not-exist.xml")


# ---------------------------------------------------------------------------
# export_openvino() — naming, path safety, and ModelWrapper dict-output mapping
# (openvino.convert_model/save_model stubbed; no real openvino needed)
# ---------------------------------------------------------------------------


class TestExportOpenvinoNaming:
    """Output filename resolution, exercised with a stubbed ``openvino`` module."""

    @pytest.mark.parametrize(
        ("variant_name", "backbone_only", "expected_stem"),
        [
            pytest.param(None, False, "inference_model", id="bare-default-detector"),
            pytest.param(None, True, "backbone_model", id="bare-default-backbone"),
            pytest.param("rfdetr-nano", False, "rfdetr-nano", id="variant-detector"),
            pytest.param("rfdetr-nano", True, "rfdetr-nano-backbone", id="variant-backbone"),
        ],
    )
    def test_resolves_expected_stem(
        self, tmp_path: Path, variant_name: str | None, backbone_only: bool, expected_stem: str
    ) -> None:
        """Backbone and detector exports must resolve distinct, predictable ``.xml`` stems."""
        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            output_xml = export_openvino(
                str(tmp_path),
                torch.nn.Identity(),
                torch.zeros(1, 3, 8, 8),
                backbone_only=backbone_only,
                variant_name=variant_name,
                verbose=False,
            )
        assert output_xml == str(tmp_path / f"{expected_stem}.xml")
        fake_ov.save_model.assert_called_once_with(mock.ANY, output_xml)

    @pytest.mark.parametrize(
        ("variant_name", "expected"),
        [
            pytest.param("../../etc/passwd", "passwd", id="forward-slash-traversal"),
            pytest.param("/absolute/path/rfdetr-nano", "rfdetr-nano", id="absolute-path"),
            pytest.param("rfdetr-nano.xml", "rfdetr-nano", id="strips-extension"),
            pytest.param("rfdetr-nano", "rfdetr-nano", id="plain-name-unchanged"),
        ],
    )
    def test_sanitizes_variant_name_directory_components(
        self, tmp_path: Path, variant_name: str, expected: str
    ) -> None:
        """``variant_name`` must be reduced to a bare filename stem before building the output path.

        Regression coverage for the same ``os.path.splitext(os.path.basename(...))`` mitigation
        ``export_coreml`` applies (see ``tests/export/test_coreml_export.py::TestVariantNamePathSafety``);
        ``export_openvino`` guards its ``variant_name`` the identical way.
        """
        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            output_xml = export_openvino(
                str(tmp_path),
                torch.nn.Identity(),
                torch.zeros(1, 3, 8, 8),
                variant_name=variant_name,
                verbose=False,
            )
        assert output_xml == str(tmp_path / f"{expected}.xml")
        assert ".." not in output_xml.removeprefix(str(tmp_path))


class TestExportOpenvinoModelWrapper:
    """``ModelWrapper`` (built internally by ``export_openvino``) dict-output-to-tuple mapping.

    ``ModelWrapper`` is defined function-local inside ``export_openvino`` (not importable in isolation), so these tests
    capture the live instance via the stubbed ``convert_model`` call and invoke it directly.
    """

    @staticmethod
    def _wrapped_model_call_args(
        tmp_path: Path, dict_output: dict[str, torch.Tensor], output_names: list[str]
    ) -> tuple[torch.Tensor, ...]:
        """Export a model whose ``forward`` returns *dict_output*; return the wrapper's mapped tuple output.

        Examples:
            >>> boxes = torch.zeros(1, 4)
            >>> logits = torch.zeros(1, 2)
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as d:
            ...     out = TestExportOpenvinoModelWrapper._wrapped_model_call_args(
            ...         Path(d), {"pred_boxes": boxes, "pred_logits": logits}, ["dets", "labels"]
            ...     )
            ...     len(out)
            2
        """

        class _DictOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return dict_output

        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            export_openvino(
                str(tmp_path),
                _DictOutputModel(),
                torch.zeros(1, 3, 8, 8),
                output_names=output_names,
                verbose=False,
            )
        wrapped_model = fake_ov.convert_model.call_args.args[0]
        return wrapped_model(torch.zeros(1, 3, 8, 8))

    def test_detection_output_maps_dets_and_labels_in_order(self, tmp_path: Path) -> None:
        """``["dets", "labels"]`` must map to ``(pred_boxes, pred_logits)``, positionally ordered."""
        dets, labels = self._wrapped_model_call_args(
            tmp_path,
            {"pred_boxes": torch.full((1, 4), 1.0), "pred_logits": torch.full((1, 2), 2.0)},
            ["dets", "labels"],
        )
        assert torch.equal(dets, torch.full((1, 4), 1.0))
        assert torch.equal(labels, torch.full((1, 2), 2.0))

    def test_segmentation_output_maps_dets_labels_masks_in_order(self, tmp_path: Path) -> None:
        """``["dets", "labels", "masks"]`` must map to ``(pred_boxes, pred_logits, pred_masks)``."""
        dets, labels, masks = self._wrapped_model_call_args(
            tmp_path,
            {
                "pred_boxes": torch.full((1, 4), 1.0),
                "pred_logits": torch.full((1, 2), 2.0),
                "pred_masks": torch.full((1, 1, 4, 4), 3.0),
            },
            ["dets", "labels", "masks"],
        )
        assert torch.equal(dets, torch.full((1, 4), 1.0))
        assert torch.equal(labels, torch.full((1, 2), 2.0))
        assert torch.equal(masks, torch.full((1, 1, 4, 4), 3.0))

    def test_keypoints_output_maps_dets_labels_keypoints_in_order(self, tmp_path: Path) -> None:
        """``["dets", "labels", "keypoints"]`` must map to ``(pred_boxes, pred_logits, pred_keypoints)``."""
        dets, labels, keypoints = self._wrapped_model_call_args(
            tmp_path,
            {
                "pred_boxes": torch.full((1, 4), 1.0),
                "pred_logits": torch.full((1, 2), 2.0),
                "pred_keypoints": torch.full((1, 17, 2), 4.0),
            },
            ["dets", "labels", "keypoints"],
        )
        assert torch.equal(dets, torch.full((1, 4), 1.0))
        assert torch.equal(labels, torch.full((1, 2), 2.0))
        assert torch.equal(keypoints, torch.full((1, 17, 2), 4.0))

    def test_backbone_only_tensor_output_passes_through_unwrapped(self, tmp_path: Path) -> None:
        """A backbone-only export (bare tensor output, not a dict) must pass through as a 1-tuple."""

        class _TensorOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.full((1, 3, 4, 4), 5.0)

        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            export_openvino(
                str(tmp_path),
                _TensorOutputModel(),
                torch.zeros(1, 3, 8, 8),
                backbone_only=True,
                verbose=False,
            )
        wrapped_model = fake_ov.convert_model.call_args.args[0]
        (features,) = wrapped_model(torch.zeros(1, 3, 8, 8))
        assert torch.equal(features, torch.full((1, 3, 4, 4), 5.0))


# ---------------------------------------------------------------------------
# format="openvino" wiring through RFDETR.export() (heavy deps mocked)
# ---------------------------------------------------------------------------


class TestExportFormatParameter:
    """Tests for ``format="openvino"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock heavy export deps so ``RFDETR.export()`` reaches the format dispatch without real work."""
        self._tmp_path = tmp_path
        xml_out = tmp_path / "inference_model.xml"
        xml_out.write_bytes(b"<xml/>")

        self._mock_stack = mock.patch.multiple(
            "rfdetr.export.main",
            make_infer_image=mock.DEFAULT,
            export_onnx=mock.DEFAULT,
        )
        mocks = self._mock_stack.start()
        mocks["make_infer_image"].return_value = torch.zeros(1, 3, 560, 560)
        self._mock_export_onnx = mocks["export_onnx"]
        self._mock_export_onnx.return_value = str(tmp_path / "inference_model.onnx")

        self._mock_export_openvino = mock.patch(
            "rfdetr.export._openvino.exporter.export_openvino", return_value=str(xml_out)
        ).start()

        yield

        mock.patch.stopall()

    @staticmethod
    def _make_rfdetr(*, segmentation_head: bool = False, use_grouppose_keypoints: bool = False) -> Any:
        """Create a minimal RFDETR instance with mocked internals (mirrors the CoreML/ExecuTorch suites)."""
        from rfdetr.detr import RFDETR

        obj = RFDETR.__new__(RFDETR)
        obj.model = mock.MagicMock()
        obj.model.resolution = 560
        obj.model.device = "cpu"
        obj.model.model.to.return_value = obj.model.model
        obj.model_config = mock.MagicMock()
        obj.model_config.segmentation_head = segmentation_head
        obj.model_config.use_grouppose_keypoints = use_grouppose_keypoints
        obj.model_config.patch_size = 14
        obj.model_config.num_windows = 1
        obj.model_config.num_channels = 3
        obj.size = "rfdetr-nano"
        return obj

    @pytest.mark.parametrize(
        "segmentation_head",
        [pytest.param(False, id="detection"), pytest.param(True, id="segmentation")],
    )
    def test_openvino_format_dispatches_to_export_openvino_not_onnx(self, segmentation_head: bool) -> None:
        """``format="openvino"`` must dispatch to ``export_openvino`` (not ``export_onnx``)."""
        obj = self._make_rfdetr(segmentation_head=segmentation_head)
        output_path = obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))
        self._mock_export_openvino.assert_called_once()
        self._mock_export_onnx.assert_not_called()
        assert output_path.suffix == ".xml"

    def test_onnx_format_does_not_call_export_openvino(self) -> None:
        """``format="onnx"`` must not import/call the OpenVINO converter."""
        obj = self._make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_export_openvino.assert_not_called()

    def test_variant_name_forwarded_to_converter(self) -> None:
        """The model's ``size`` attribute must be forwarded as ``variant_name``."""
        obj = self._make_rfdetr()
        obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_openvino.call_args.kwargs["variant_name"] == "rfdetr-nano"

    def test_detection_output_names_forwarded_to_converter(self) -> None:
        """Plain detection models must forward ``["dets", "labels"]`` as ``output_names``."""
        obj = self._make_rfdetr()
        obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_openvino.call_args.kwargs["output_names"] == ["dets", "labels"]

    def test_segmentation_output_names_forwarded_to_converter(self) -> None:
        """Segmentation models must forward ``["dets", "labels", "masks"]`` as ``output_names``."""
        obj = self._make_rfdetr(segmentation_head=True)
        obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_openvino.call_args.kwargs["output_names"] == ["dets", "labels", "masks"]

    def test_invalid_format_raises_value_error(self) -> None:
        """Unknown ``format`` must raise ``ValueError`` listing supported formats, not reach the converter."""
        obj = self._make_rfdetr()
        with pytest.raises(ValueError, match="Unsupported export format"):
            obj.export(format="bogus", output_dir=str(self._tmp_path / "out"))
        self._mock_export_openvino.assert_not_called()


class TestExportOpenvinoMissingDependencyViaPublicAPI:
    """``RFDETR.export(format="openvino")`` surfaces ``ImportError`` (not the registry ``ValueError``)."""

    @pytest.fixture(autouse=True)
    def _patch_light_export_deps(self, tmp_path: Path) -> Any:
        """Mock only ``make_infer_image``; the OpenVINO converter itself must run for real."""
        self._tmp_path = tmp_path
        self._mock_stack = mock.patch("rfdetr.export.main.make_infer_image", return_value=torch.zeros(1, 3, 560, 560))
        self._mock_stack.start()
        yield
        self._mock_stack.stop()

    @staticmethod
    def _make_rfdetr() -> Any:
        """Create a minimal RFDETR instance with mocked internals."""
        from rfdetr.detr import RFDETR

        obj = RFDETR.__new__(RFDETR)
        obj.model = mock.MagicMock()
        obj.model.resolution = 560
        obj.model.device = "cpu"
        obj.model.model.to.return_value = obj.model.model
        obj.model_config = mock.MagicMock()
        obj.model_config.segmentation_head = False
        obj.model_config.use_grouppose_keypoints = False
        obj.model_config.patch_size = 14
        obj.model_config.num_windows = 1
        obj.model_config.num_channels = 3
        return obj

    def test_raises_import_error_not_registry_value_error(self) -> None:
        """Missing ``openvino`` must surface ``ImportError`` — the format itself must be accepted.

        Regression guard for the format-dispatch registry: ``format="openvino"`` must reach the real
        ``export_openvino()`` call (and fail there, on the missing dependency) rather than being rejected
        upfront by ``_resolve_export_backend``'s ``_EXPORT_FORMATS`` membership check.
        """
        obj = self._make_rfdetr()
        with pytest.raises(ImportError):
            obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))


# ---------------------------------------------------------------------------
# End-to-end (gated) — real convert + parity vs eager PyTorch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openvino_detection_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRNano to OpenVINO IR once, shared across the gated detection e2e tests."""
    pytest.importorskip("openvino")
    import rfdetr

    out_dir = tmp_path_factory.mktemp("openvino_nano")
    detector = rfdetr.RFDETRNano(pretrain_weights=None)
    xml_path = detector.export(output_dir=str(out_dir), format="openvino", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return model, example, Path(xml_path)


@pytest.fixture(scope="module")
def openvino_backbone_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
    """Export RFDETRNano's backbone-only OpenVINO IR once, shared across the gated backbone e2e test."""
    pytest.importorskip("openvino")
    import rfdetr
    from rfdetr.export._backend import _BackboneExport

    out_dir = tmp_path_factory.mktemp("openvino_backbone")
    detector = rfdetr.RFDETRNano(pretrain_weights=None)
    xml_path = detector.export(output_dir=str(out_dir), format="openvino", backbone_only=True, verbose=False)
    backbone = detector.model.model.backbone[0].to("cpu").eval()
    reference_model = _BackboneExport(backbone)
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return reference_model, example, Path(xml_path)


@pytest.mark.e2e_openvino
class TestOpenVINOEndToEnd:
    """Real OpenVINO IR export + CPU numerical parity (``-m e2e_openvino``, requires ``openvino`` installed)."""

    def test_xml_and_bin_written(self, openvino_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        """Export must write both the ``.xml`` graph and its companion ``.bin`` weights file."""
        _, _, xml_path = openvino_detection_export
        assert xml_path.exists()
        assert xml_path.suffix == ".xml"
        assert xml_path.with_suffix(".bin").exists()

    def test_detection_outputs_match_pytorch(self, openvino_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        """OpenVINO detection output (boxes, logits) must match eager PyTorch within a tight tolerance."""
        model, example, xml_path = openvino_detection_export
        eager_tensors = eager_reference_tensors(model, example)

        inference = OpenVINOInference(xml_path)
        ov_outputs = inference(example.numpy())
        ov_tensors = [torch.from_numpy(output) for output in ov_outputs]

        diffs = max_abs_output_diffs(eager_tensors, ov_tensors, check_shape=True, names=["dets", "labels"])
        assert len(diffs) == 2, f"detection export must yield (boxes, logits), got {len(diffs)} outputs"
        assert max(diffs) < 1e-3, f"OpenVINO detection outputs diverge from PyTorch: max abs diff {max(diffs)}"

    def test_backbone_outputs_match_pytorch(
        self, openvino_backbone_export: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """OpenVINO must run every backbone feature-map output from the public backbone-only export."""
        model, example, xml_path = openvino_backbone_export
        assert "backbone" in xml_path.stem
        eager_tensors = eager_reference_tensors(model, example)

        inference = OpenVINOInference(xml_path)
        ov_outputs = inference(example.numpy())
        ov_tensors = [torch.from_numpy(output) for output in ov_outputs]

        diffs = max_abs_output_diffs(eager_tensors, ov_tensors, check_shape=True)
        assert max(diffs) < 1e-3, f"OpenVINO backbone outputs diverge from PyTorch: max abs diff {max(diffs)}"


class TestOpenVINOInferenceEndToEnd:
    """Real ``OpenVINOInference`` behaviour gated behind an actual ``openvino`` install."""

    def test_missing_model_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """A nonexistent ``.xml`` path must raise ``FileNotFoundError`` once ``openvino`` is importable.

        Unreachable without ``openvino`` installed (see ``TestOpenVINOInferenceMissingDependency`` for the ungated
        ``ImportError`` coverage of the same constructor when the import itself fails first).
        """
        pytest.importorskip("openvino")
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            OpenVINOInference(tmp_path / "does-not-exist.xml")
