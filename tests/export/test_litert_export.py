# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for direct PyTorch -> LiteRT (``.tflite``) export via ``litert-torch``.

Covers:
* ``export_litert()`` — dependency-missing path (``litert_torch`` masked out of ``sys.modules``, so it runs the same
  whether or not the package is installed), naming / path-traversal sanitization, error tiers, and the internal
  ``ModelWrapper`` (``litert_torch.convert`` stubbed via ``sys.modules`` injection so these run without the real
  package installed).
* ``format="litert"`` wiring through ``RFDETR.export()`` (heavy deps mocked, fast).
* The single-level deformable-attention core emitting no one-output ``split`` under ``torch.export`` — the only op
  litert-torch 0.9.4 could not lower on RF-DETR's export graph.
* A real end-to-end export + numerical parity check, gated behind the ``e2e_litert`` marker and
  ``pytest.importorskip("litert_torch")`` so it only runs where the ``[litert]`` extra is installed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rfdetr.export._litert.converter import ModelWrapper, export_litert
from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
from tests.export.conftest import (
    _parity_input_from_image,
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)


def _stub_litert_torch_module() -> types.ModuleType:
    """Build a minimal fake ``litert_torch`` module exposing ``convert(...).export(path)``.

    Injected into ``sys.modules`` so ``export_litert()``'s ``import litert_torch`` succeeds without the real package
    installed, letting the naming/wrapping logic run end-to-end while the actual (heavy, unavailable) conversion is a
    stub.  ``convert`` runs the wrapped module once on the example input -- so ``ModelWrapper``'s own errors surface
    the way they do through the real converter -- and ``export`` writes a placeholder file at the requested path.

    Returns:
        A fresh fake module whose ``convert`` is a ``MagicMock`` returning ``fake.edge_model``, whose ``export`` is a
        ``MagicMock`` too (exposed on the module so tests can assert on the path it received).

    Examples:
        >>> import tempfile
        >>> fake = _stub_litert_torch_module()
        >>> edge = fake.convert(torch.nn.Identity(), (torch.zeros(1),))
        >>> edge is fake.edge_model
        True
        >>> with tempfile.TemporaryDirectory() as d:
        ...     edge.export(f"{d}/m.tflite")
        ...     Path(f"{d}/m.tflite").read_bytes()
        b'TFL3'
    """
    fake = types.ModuleType("litert_torch")
    edge_model = mock.MagicMock(name="edge_model")

    def _export(path: str) -> None:
        Path(path).write_bytes(b"TFL3")

    def _convert(module: torch.nn.Module, sample_args: tuple[torch.Tensor, ...]) -> mock.MagicMock:
        module(*sample_args)
        return edge_model

    edge_model.export.side_effect = _export
    fake.convert = mock.MagicMock(side_effect=_convert)
    fake.edge_model = edge_model
    return fake


def _run_litert(tflite_path: Path, input_array: NDArray[Any]) -> tuple[NDArray[Any], ...]:
    """Run *tflite_path* through the ``ai_edge_litert`` interpreter (CPU / XNNPACK) on one input.

    Args:
        tflite_path: Path to an exported ``.tflite`` file with a single input.
        input_array: C-contiguous float32 input, shaped to match the model's input tensor.

    Returns:
        One NumPy array per model output, in the interpreter's output order (the export's positional order).

    Examples:
        Requires a real exported ``.tflite`` and ``ai_edge_litert`` — not runnable standalone.
        See ``TestLiteRTEndToEnd`` for real invocations.

        >>> callable(_run_litert)
        True
    """
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    (input_detail,) = interpreter.get_input_details()
    interpreter.set_tensor(input_detail["index"], input_array)
    interpreter.invoke()
    return tuple(np.copy(interpreter.get_tensor(detail["index"])) for detail in interpreter.get_output_details())


def _confident_query_diffs(
    eager_tensors: list[torch.Tensor],
    other_tensors: list[torch.Tensor],
    top_k: int = 10,
    sigmoid_indices: frozenset[int] = frozenset(),
) -> list[float]:
    """Max-abs-diff per output, restricted to the ``top_k`` highest-confidence queries.

    RF-DETR's two-stage query selection keeps ~300 raw encoder proposals ranked by objectness; on a real photo only a
    handful are genuine detections, and the low-confidence rest sit close enough together that ordinary cross-backend
    floating-point differences flip their relative rank and swap which query slot each lands in.  Comparing every raw
    position would swamp the signal (do the genuine detections match?) with that background reordering, so the check
    is restricted to the confident queries, which stay positionally aligned.  Same rationale as the OpenVINO suite.

    Args:
        eager_tensors: Reference tensors from :func:`eager_reference_tensors`; ``eager_tensors[1]`` must be the
            per-query class-logit tensor (``dets, labels[, ...]`` output order).
        other_tensors: Backend output tensors, same order and shapes as *eager_tensors*.
        top_k: Number of highest-confidence queries (by eager logit max) to compare.
        sigmoid_indices: Output indices to compare in sigmoid (probability) space instead of raw logit space --
            segmentation mask logits span a much wider range than boxes/labels.

    Returns:
        One max-abs-diff per output, computed over the ``top_k`` selected queries only.

    Examples:
        >>> boxes = torch.zeros(1, 3, 4)
        >>> labels = torch.tensor([[[0.1, 0.2], [5.0, 0.1], [0.0, 0.0]]])
        >>> other_boxes = boxes.clone()
        >>> other_boxes[0, 1] += 0.5  # perturb the only confident query (index 1)
        >>> diffs = _confident_query_diffs([boxes, labels], [other_boxes, labels.clone()], top_k=1)
        >>> round(diffs[0], 4)
        0.5
    """
    scores = eager_tensors[1][0].amax(dim=-1)
    top_indices = scores.topk(min(top_k, scores.numel())).indices
    diffs: list[float] = []
    for index, (eager, other) in enumerate(zip(eager_tensors, other_tensors)):
        eager_selected = eager[0, top_indices]
        other_selected = other[0, top_indices].float()
        if index in sigmoid_indices:
            eager_selected = eager_selected.sigmoid()
            other_selected = other_selected.sigmoid()
        diffs.append((eager_selected - other_selected).abs().max().item())
    return diffs


# ---------------------------------------------------------------------------
# export_litert() — dependency-missing path (litert_torch masked out of sys.modules)
# ---------------------------------------------------------------------------


class TestExportLitertMissingDependency:
    """``export_litert()``'s ``ImportError`` path, with ``litert_torch`` masked so it runs on any environment."""

    def test_raises_import_error(self, tmp_path: Path) -> None:
        """Missing ``litert_torch`` must surface an ``ImportError``, not any other exception type."""
        with mock.patch.dict(sys.modules, {"litert_torch": None}), pytest.raises(ImportError):
            export_litert(torch.nn.Identity(), torch.zeros(1, 3, 32, 32), str(tmp_path))

    def test_import_error_names_pip_install_hint(self, tmp_path: Path) -> None:
        """The raised ``ImportError`` must name the ``rfdetr[litert]`` extra so users know how to fix it."""
        with (
            mock.patch.dict(sys.modules, {"litert_torch": None}),
            pytest.raises(ImportError, match=r"rfdetr\[litert\]"),
        ):
            export_litert(torch.nn.Identity(), torch.zeros(1, 3, 32, 32), str(tmp_path))


# ---------------------------------------------------------------------------
# ModelWrapper — output normalization contract
# ---------------------------------------------------------------------------


class TestModelWrapper:
    """``ModelWrapper`` turns the export-mode forward's tuple/list into a tuple and refuses anything else."""

    def test_tuple_output_passes_through_unchanged(self) -> None:
        dets, labels = torch.zeros(1, 300, 4), torch.zeros(1, 300, 91)
        model = mock.MagicMock(spec=torch.nn.Module, return_value=(dets, labels))
        assert ModelWrapper(model)(torch.zeros(1, 3, 32, 32)) == (dets, labels)

    def test_list_output_converted_to_tuple(self) -> None:
        """The backbone-only export graph returns a list; the wrapper hands the converter a tuple."""
        features = torch.zeros(1, 256, 24, 24)
        model = mock.MagicMock(spec=torch.nn.Module, return_value=[features])
        output = ModelWrapper(model)(torch.zeros(1, 3, 32, 32))
        assert isinstance(output, tuple)
        assert output == (features,)

    def test_dict_output_raises_not_implemented(self) -> None:
        """A dict output means the caller skipped ``model.export()``; that is reported, not flattened."""
        model = mock.MagicMock(spec=torch.nn.Module, return_value={"pred_boxes": torch.zeros(1, 300, 4)})
        with pytest.raises(NotImplementedError, match="export mode"):
            ModelWrapper(model)(torch.zeros(1, 3, 32, 32))

    def test_unsupported_output_type_raises_type_error(self) -> None:
        model = mock.MagicMock(spec=torch.nn.Module, return_value=torch.zeros(1, 300, 4))
        with pytest.raises(TypeError, match="Unsupported model output type"):
            ModelWrapper(model)(torch.zeros(1, 3, 32, 32))


# ---------------------------------------------------------------------------
# export_litert() — naming and error tiers (litert_torch stubbed)
# ---------------------------------------------------------------------------


class _TwoOutputs(torch.nn.Module):
    """Stand-in for an export-mode detector: returns a ``(dets, labels)`` tuple."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x.new_zeros(1, 300, 4), x.new_zeros(1, 300, 91)


class TestExportLitertNaming:
    """Filename resolution mirrors the ONNX/CoreML/ExecuTorch/OpenVINO exporters' stem and ``-backbone`` rules."""

    @pytest.mark.parametrize(
        ("variant_name", "output_name", "backbone_only", "expected"),
        [
            pytest.param("rfdetr-nano", None, False, "rfdetr-nano.tflite", id="variant"),
            pytest.param(None, "my-model", False, "my-model.tflite", id="output-name-wins"),
            pytest.param("rfdetr-nano", "my-model", False, "my-model.tflite", id="output-name-over-variant"),
            pytest.param(None, None, False, "inference_model.tflite", id="default"),
            pytest.param("rfdetr-nano", None, True, "rfdetr-nano-backbone.tflite", id="variant-backbone"),
            pytest.param(None, "custom", True, "custom-backbone.tflite", id="output-name-backbone"),
            pytest.param(None, None, True, "backbone_model.tflite", id="default-backbone-no-marker"),
        ],
    )
    def test_resolves_expected_filename(
        self, tmp_path: Path, variant_name: str | None, output_name: str | None, backbone_only: bool, expected: str
    ) -> None:
        fake = _stub_litert_torch_module()
        with mock.patch.dict(sys.modules, {"litert_torch": fake}):
            result = export_litert(
                _TwoOutputs(),
                torch.zeros(1, 3, 32, 32),
                tmp_path,
                backbone_only=backbone_only,
                variant_name=variant_name,
                output_name=output_name,
            )
        assert result == tmp_path / expected
        assert result.read_bytes() == b"TFL3"
        fake.convert.assert_called_once()
        assert isinstance(fake.convert.call_args.args[0], ModelWrapper)
        fake.edge_model.export.assert_called_once_with(str(tmp_path / expected))

    def test_sanitizes_variant_name_directory_components(self, tmp_path: Path) -> None:
        """A variant name carrying path separators/extensions must not escape *output_dir*."""
        with mock.patch.dict(sys.modules, {"litert_torch": _stub_litert_torch_module()}):
            result = export_litert(
                _TwoOutputs(), torch.zeros(1, 3, 32, 32), tmp_path, variant_name="../evil/rfdetr-nano.onnx"
            )
        assert result == tmp_path / "rfdetr-nano.tflite"
        assert result.exists()

    def test_creates_missing_output_dir(self, tmp_path: Path) -> None:
        with mock.patch.dict(sys.modules, {"litert_torch": _stub_litert_torch_module()}):
            result = export_litert(_TwoOutputs(), torch.zeros(1, 3, 32, 32), tmp_path / "nested" / "out")
        assert result == tmp_path / "nested" / "out" / "inference_model.tflite"


class TestExportLitertErrorTiers:
    """Converter failures become ``RuntimeError``; ``ModelWrapper``'s own contract errors pass through unchanged."""

    def test_converter_failure_is_wrapped_as_runtime_error(self, tmp_path: Path) -> None:
        fake = _stub_litert_torch_module()
        fake.convert.side_effect = ValueError("Failed to run converter passes")
        with (
            mock.patch.dict(sys.modules, {"litert_torch": fake}),
            pytest.raises(RuntimeError, match="Failed to export model to LiteRT"),
        ):
            export_litert(_TwoOutputs(), torch.zeros(1, 3, 32, 32), tmp_path)

    def test_dict_output_not_relabeled_as_runtime_error(self, tmp_path: Path) -> None:
        """The wrapper's ``NotImplementedError`` (model not in export mode) must reach the caller as-is."""
        dict_model = mock.MagicMock(spec=torch.nn.Module, return_value={"pred_boxes": torch.zeros(1, 300, 4)})
        dict_model.eval.return_value = dict_model
        dict_model.cpu.return_value = dict_model
        with (
            mock.patch.dict(sys.modules, {"litert_torch": _stub_litert_torch_module()}),
            pytest.raises(NotImplementedError, match="export mode"),
        ):
            export_litert(dict_model, torch.zeros(1, 3, 32, 32), tmp_path)


# ---------------------------------------------------------------------------
# format="litert" wiring through RFDETR.export() (heavy deps mocked)
# ---------------------------------------------------------------------------


def _make_rfdetr(*, segmentation_head: bool = False) -> Any:
    """Create a minimal ``RFDETR`` instance with mocked internals (mirrors the CoreML/ExecuTorch/OpenVINO suites).

    Examples:
        >>> obj = _make_rfdetr()
        >>> obj.size, obj.model.resolution
        ('rfdetr-nano', 560)
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
    obj.size = "rfdetr-nano"
    return obj


class TestExportFormatParameter:
    """Tests for ``format="litert"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock heavy export deps so ``RFDETR.export()`` reaches the format dispatch without real work."""
        self._tmp_path = tmp_path
        tflite_out = tmp_path / "rfdetr-nano.tflite"
        tflite_out.write_bytes(b"TFL3")

        mocks = mock.patch.multiple(
            "rfdetr.export.main",
            make_infer_image=mock.DEFAULT,
            export_onnx=mock.DEFAULT,
        ).start()
        mocks["make_infer_image"].return_value = torch.zeros(1, 3, 560, 560)
        self._mock_make_infer_image = mocks["make_infer_image"]
        self._mock_export_onnx = mocks["export_onnx"]
        self._mock_export_onnx.return_value = str(tmp_path / "inference_model.onnx")

        self._mock_export_litert = mock.patch(
            "rfdetr.export._litert.converter.export_litert", return_value=tflite_out
        ).start()

        yield

        mock.patch.stopall()

    @pytest.mark.parametrize(
        "segmentation_head",
        [pytest.param(False, id="detection"), pytest.param(True, id="segmentation")],
    )
    def test_litert_format_dispatches_to_export_litert_not_onnx(self, segmentation_head: bool) -> None:
        """``format="litert"`` must dispatch to ``export_litert`` (no ONNX step)."""
        obj = _make_rfdetr(segmentation_head=segmentation_head)
        output_path = obj.export(format="litert", output_dir=str(self._tmp_path / "out"))
        self._mock_export_litert.assert_called_once()
        self._mock_export_onnx.assert_not_called()
        assert output_path.suffix == ".tflite"

    def test_onnx_format_does_not_call_export_litert(self) -> None:
        obj = _make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_export_litert.assert_not_called()

    def test_variant_name_forwarded_to_converter(self) -> None:
        obj = _make_rfdetr()
        obj.export(format="litert", output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_litert.call_args.kwargs["variant_name"] == "rfdetr-nano"

    def test_output_name_forwarded_to_converter(self) -> None:
        obj = _make_rfdetr()
        obj.export(format="litert", output_dir=str(self._tmp_path / "out"), output_name="my-model")
        assert self._mock_export_litert.call_args.kwargs["output_name"] == "my-model"

    @pytest.mark.parametrize("quantization", [None, "fp32"])
    def test_fp32_quantization_values_are_accepted(self, quantization: str | None) -> None:
        obj = _make_rfdetr()
        obj.export(format="litert", output_dir=str(self._tmp_path / "out"), quantization=quantization)
        self._mock_export_litert.assert_called_once()

    @pytest.mark.parametrize("quantization", ["fp16", "int8"])
    def test_other_quantization_raises_before_forward_pass(self, quantization: str) -> None:
        """A ``.tflite`` caller may expect ``quantization`` to apply; refusing beats silently writing float32."""
        obj = _make_rfdetr()
        with pytest.raises(NotImplementedError, match="quantization"):
            obj.export(format="litert", output_dir=str(self._tmp_path / "out"), quantization=quantization)
        self._mock_make_infer_image.assert_not_called()
        self._mock_export_litert.assert_not_called()

    def test_dynamic_batch_raises_before_forward_pass(self) -> None:
        """``dynamic_batch=True`` is rejected next to the ExecuTorch/CoreML/OpenVINO fail-fast checks."""
        obj = _make_rfdetr()
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            obj.export(format="litert", output_dir=str(self._tmp_path / "out"), dynamic_batch=True)
        self._mock_make_infer_image.assert_not_called()

    def test_notes_warns_and_is_dropped(self) -> None:
        obj = _make_rfdetr()
        with pytest.warns(UserWarning, match="notes"):
            obj.export(format="litert", output_dir=str(self._tmp_path / "out"), notes="some metadata")

    def test_experimental_warning_is_emitted(self) -> None:
        obj = _make_rfdetr()
        with pytest.warns(UserWarning, match="LiteRT export is experimental"):
            obj.export(format="litert", output_dir=str(self._tmp_path / "out"))


class TestExportLitertMissingDependencyViaPublicAPI:
    """``RFDETR.export(format="litert")`` surfaces ``ImportError`` (not the registry ``ValueError``)."""

    def test_raises_import_error_not_registry_value_error(self, tmp_path: Path) -> None:
        """The format must be accepted by the registry and fail only on the missing dependency."""
        obj = _make_rfdetr()
        with (
            mock.patch("rfdetr.export.main.make_infer_image", return_value=torch.zeros(1, 3, 560, 560)),
            mock.patch.dict(sys.modules, {"litert_torch": None}),
            pytest.raises(ImportError, match=r"rfdetr\[litert\]"),
        ):
            obj.export(format="litert", output_dir=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Deformable-attention core: no one-output split in the exported graph
# ---------------------------------------------------------------------------


class _SingleLevelCore(torch.nn.Module):
    """Call the deformable-attention core with Python ``(H, W)`` pairs, as the export path does."""

    def __init__(self, shapes_hw: list[tuple[int, int]]) -> None:
        super().__init__()
        self.shapes_hw = shapes_hw

    def forward(self, value: torch.Tensor, locations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        spatial_shapes = torch.tensor(self.shapes_hw, dtype=torch.long)
        return ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights, value_spatial_shapes_hw=self.shapes_hw)


class TestDeformableCoreSplit:
    """The single-level core emits no ``split`` op, the one node litert-torch 0.9.4 cannot lower for RF-DETR."""

    @staticmethod
    def _core_inputs(shapes_hw: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(value, sampling_locations, attention_weights)`` for the rank-5 (export) core layout.

        Examples:
            >>> value, locations, weights = TestDeformableCoreSplit._core_inputs([(4, 4)])
            >>> tuple(value.shape), tuple(locations.shape), tuple(weights.shape)
            ((1, 2, 8, 16), (1, 3, 2, 4, 2), (1, 3, 2, 4))
        """
        torch.manual_seed(0)
        batch, heads, head_dim, len_q, points = 1, 2, 8, 3, 4
        total = sum(h * w for h, w in shapes_hw)
        value = torch.randn(batch, heads, head_dim, total)
        locations = torch.rand(batch, len_q, heads, len(shapes_hw) * points, 2)
        weights = torch.softmax(torch.randn(batch, len_q, heads, len(shapes_hw) * points), -1)
        return value, locations, weights

    @pytest.mark.parametrize(
        ("shapes_hw", "expects_split"),
        [
            pytest.param([(4, 4)], False, id="single-level-no-split"),
            pytest.param([(4, 4), (2, 2)], True, id="two-levels-still-split"),
        ],
    )
    def test_split_present_only_for_multiple_levels(
        self, shapes_hw: list[tuple[int, int]], expects_split: bool
    ) -> None:
        inputs = self._core_inputs(shapes_hw)
        program = torch.export.export(_SingleLevelCore(shapes_hw), inputs, strict=False)
        targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
        has_split = any("split" in target for target in targets)
        assert has_split is expects_split, sorted(targets)

    def test_single_level_output_unchanged(self) -> None:
        """Skipping the split must not change the numbers: single-level equals the two-level formula on one piece."""
        shapes_hw = [(4, 4)]
        value, locations, weights = self._core_inputs(shapes_hw)
        spatial_shapes = torch.tensor(shapes_hw, dtype=torch.long)
        with_hw = ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights, value_spatial_shapes_hw=shapes_hw)
        eager = ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights)
        torch.testing.assert_close(with_hw, eager)


# ---------------------------------------------------------------------------
# End-to-end (gated) — real litert-torch convert + parity vs eager PyTorch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def people_walking_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download supervision's ``PEOPLE_WALKING`` asset once, shared across LiteRT e2e tests.

    A real photo gives genuine, well-separated detections whose query positions stay stable across backends; the
    structured synthetic input would leave every two-stage candidate at a similar low objectness and let float noise
    reorder them (see ``_confident_query_diffs``).
    """
    asset_dir = tmp_path_factory.mktemp("litert_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        from supervision.assets import ImageAssets, download_assets

        return Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def litert_detection_export(
    tmp_path_factory: pytest.TempPathFactory, people_walking_image_path: Path
) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRNano to LiteRT once, shared across the gated detection e2e tests."""
    pytest.importorskip("litert_torch")
    import rfdetr

    out_dir = tmp_path_factory.mktemp("litert_nano")
    detector = rfdetr.RFDETRNano()
    tflite_path = detector.export(output_dir=str(out_dir), format="litert", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _parity_input_from_image(people_walking_image_path, resolution)
    return model, example, Path(tflite_path)


@pytest.fixture(scope="module")
def litert_segmentation_export(
    tmp_path_factory: pytest.TempPathFactory, people_walking_image_path: Path
) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRSegNano to LiteRT once, shared across the gated segmentation e2e test."""
    pytest.importorskip("litert_torch")
    import rfdetr

    out_dir = tmp_path_factory.mktemp("litert_seg_nano")
    detector = rfdetr.RFDETRSegNano()
    tflite_path = detector.export(output_dir=str(out_dir), format="litert", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _parity_input_from_image(people_walking_image_path, resolution)
    return model, example, Path(tflite_path)


@pytest.fixture(scope="module")
def litert_backbone_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
    """Export RFDETRNano's backbone-only LiteRT graph once, shared across the gated backbone e2e test."""
    pytest.importorskip("litert_torch")
    import rfdetr
    from rfdetr.export._backend import _BackboneExport

    out_dir = tmp_path_factory.mktemp("litert_backbone")
    detector = rfdetr.RFDETRNano(pretrain_weights=None)
    tflite_path = detector.export(output_dir=str(out_dir), format="litert", backbone_only=True, verbose=False)
    backbone = detector.model.model.backbone[0].to("cpu").eval()
    reference_model = _BackboneExport(backbone)
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return reference_model, example, Path(tflite_path)


@pytest.mark.e2e_litert
class TestLiteRTEndToEnd:
    """Real litert-torch export + CPU (XNNPACK) numerical parity (``-m e2e_litert``, requires ``[litert]``)."""

    def test_tflite_written_with_variant_name(self, litert_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        _, _, tflite_path = litert_detection_export
        assert tflite_path.exists()
        assert tflite_path.name == "rfdetr-nano.tflite"
        assert tflite_path.read_bytes()[4:8] == b"TFL3"

    def test_detection_outputs_match_pytorch(self, litert_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        """LiteRT detection output (boxes, logits) must match eager PyTorch on confident detections.

        Compared over the top-10 highest-confidence queries (see ``_confident_query_diffs``).  Measured maxima on a
        random-init RFDETRNano at this resolution: box ~6e-8, label ~1e-5; tolerances leave generous headroom.
        """
        model, example, tflite_path = litert_detection_export
        eager_tensors = eager_reference_tensors(model, example)
        litert_tensors = [torch.from_numpy(output) for output in _run_litert(tflite_path, example.numpy())]

        assert len(litert_tensors) == 2, f"detection export must yield (boxes, logits), got {len(litert_tensors)}"
        box_diff, label_diff = _confident_query_diffs(eager_tensors, litert_tensors)
        assert box_diff < 1e-3, f"LiteRT detection boxes diverge from PyTorch: max abs diff {box_diff}"
        assert label_diff < 0.1, f"LiteRT detection logits diverge from PyTorch: max abs diff {label_diff}"

    def test_segmentation_outputs_match_pytorch(
        self, litert_segmentation_export: tuple[Any, torch.Tensor, Path]
    ) -> None:
        """LiteRT segmentation output (boxes, logits, masks) must match eager PyTorch on confident detections.

        Masks are compared in sigmoid space (mask logits span a far wider range than boxes/labels).  Measured
        maxima on a random-init RFDETRSegNano: box ~6e-8, label ~1e-5, mask (raw) ~1e-4.
        """
        model, example, tflite_path = litert_segmentation_export
        eager_tensors = eager_reference_tensors(model, example)
        litert_tensors = [torch.from_numpy(output) for output in _run_litert(tflite_path, example.numpy())]

        assert len(litert_tensors) == 3, f"segmentation export must yield (boxes, logits, masks), got {len(litert_tensors)}"
        box_diff, label_diff, mask_diff = _confident_query_diffs(eager_tensors, litert_tensors, sigmoid_indices={2})
        assert box_diff < 1e-3, f"LiteRT segmentation boxes diverge from PyTorch: max abs diff {box_diff}"
        assert label_diff < 0.1, f"LiteRT segmentation logits diverge from PyTorch: max abs diff {label_diff}"
        assert mask_diff < 0.05, f"LiteRT segmentation masks diverge from PyTorch (sigmoid space): {mask_diff}"

    def test_backbone_outputs_match_pytorch(
        self, litert_backbone_export: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """The backbone-only export has no two-stage selection, so a plain positional comparison applies.

        Measured maximum on this fixture: ~2e-5.
        """
        model, example, tflite_path = litert_backbone_export
        assert tflite_path.name == "rfdetr-nano-backbone.tflite"
        eager_tensors = eager_reference_tensors(model, example)
        litert_tensors = [torch.from_numpy(output) for output in _run_litert(tflite_path, example.numpy())]

        diffs = max_abs_output_diffs(eager_tensors, litert_tensors, check_shape=True)
        assert max(diffs) < 1e-3, f"LiteRT backbone outputs diverge from PyTorch: max abs diff {max(diffs)}"

    def test_keypoint_export_is_rejected_by_converter(self, tmp_path: Path) -> None:
        """Pins the documented limitation: litert-torch 0.9.4 rejects the keypoint head's rank-4 ``batch_matmul``.

        When a litert-torch release lowers it, this test fails — that is the cue to drop the keypoint caveat from
        the ``format="litert"`` docs and add a keypoint parity test next to the detection one.
        """
        pytest.importorskip("litert_torch")
        import rfdetr

        detector = rfdetr.RFDETRKeypointPreview(pretrain_weights=None)
        with pytest.raises(RuntimeError, match="Failed to export model to LiteRT"):
            detector.export(output_dir=str(tmp_path), format="litert", verbose=False)
