# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Apple Core AI (``.aimodel``) export.

Covers:
* ``grid_sampler_2d_gather`` — the decomposition that replaces ``aten.grid_sampler_2d`` (no Core AI lowering) with a
  gather-based bilinear sampler. Pure PyTorch, runs everywhere.
* ``CoreAIExporter`` — configuration, naming, precision, notes and the missing-dependency path, with the
  ``coreai-torch`` stack mocked.
* ``format="coreai"`` wiring through ``RFDETR.export()`` (heavy deps mocked, fast).
* End-to-end convert + numerical parity against eager PyTorch (``@pytest.mark.e2e_coreai``, opt-in). Converting only
  needs ``coreai-torch``; running an ``.aimodel`` needs the Core AI runtime, which ships with macOS 27.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from supervision.assets import ImageAssets, download_assets

import rfdetr
from rfdetr.export._backend import _BackboneExport
from rfdetr.export._coreai import _IS_COREAI_TORCH_AVAILABLE
from rfdetr.export._coreai.decompositions import coreai_decomposition_table, grid_sampler_2d_gather, topk_in_float32
from rfdetr.export._coreai.exporter import CoreAIConfig, CoreAIExporter, _check_coreai_torch_available
from rfdetr.export.prepare import ExportGraph, resolve_output_names
from rfdetr.utilities.reproducibility import seed_all
from tests.export.conftest import (
    _parity_input_from_image,
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)


def _coreai_runtime_available() -> bool:
    """Return whether this host can load and run ``.aimodel`` assets (Apple silicon, macOS 27 or later).

    Examples:
        >>> isinstance(_coreai_runtime_available(), bool)
        True
    """
    if sys.platform != "darwin" or platform.machine() != "arm64" or not _IS_COREAI_TORCH_AVAILABLE:
        return False
    return int(platform.mac_ver()[0].split(".")[0] or 0) >= 27


coreai_only = pytest.mark.skipif(not _IS_COREAI_TORCH_AVAILABLE, reason="coreai-torch not installed")
coreai_runtime_only = pytest.mark.skipif(
    not _coreai_runtime_available(), reason="running .aimodel assets needs the Core AI runtime (macOS 27+)"
)

# Float32 Core AI matches eager to ~1e-5 on boxes/logits on the CPU; masks and keypoints get the same headroom the
# CoreML suite uses. The bound stays well under structural-failure scale (>= 1e-3).
_COREAI_MAX_ABS_DIFF = 1e-4
# The GPU executes float32 graphs with reduced-precision accumulation, so it gets its own, looser bound. It still sits
# two orders of magnitude below the ~0.5-1.5 abs diffs a broken sampler (or a swapped query) produces.
_COREAI_GPU_MAX_ABS_DIFF = 5e-3
#: Queries the e2e parity exports select, small enough that their two-stage ranking is well separated.
_COREAI_E2E_NUM_QUERIES = 5
#: Minimum eager gap between neighbouring top-ranked two-stage scores, see ``test_coreml_export.py``.
_MIN_TWO_STAGE_RANK_MARGIN = 1e-4
_COREAI_EXPORT_SEED = 0

_BOOL_PRODUCING_OPS = frozenset(
    {
        "aten.eq",
        "aten.ne",
        "aten.lt",
        "aten.le",
        "aten.gt",
        "aten.ge",
        "aten.logical_and",
        "aten.logical_or",
        "aten.logical_not",
        "aten.bitwise_and",
        "aten.bitwise_or",
        "aten.bitwise_not",
        "aten.where",
    }
)


class _FixedGridSample(torch.nn.Module):
    """Sample a constant grid from a single input, so an ``ExportGraph`` can carry a grid sampler."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grid = torch.zeros(image.shape[0], 2, 2, 2, dtype=image.dtype)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class _GridSample(torch.nn.Module):
    """Wrap ``F.grid_sample`` so a single call can be traced with ``torch.export``."""

    def __init__(self, padding_mode: str, align_corners: bool) -> None:
        super().__init__()
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            value, grid, mode="bilinear", padding_mode=self.padding_mode, align_corners=self.align_corners
        )


def _sampling_case(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a feature map and a grid whose points fall inside, on the border of, and outside the image.

    Examples:
        >>> value, grid = _sampling_case()
        >>> value.shape, grid.shape
        (torch.Size([2, 5, 7, 9]), torch.Size([2, 4, 6, 2]))
    """
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(2, 5, 7, 9, generator=generator)
    grid = torch.rand(2, 4, 6, 2, generator=generator) * 2.8 - 1.4
    # Pin a few exact cases: the four image corners and points one pixel beyond each edge.
    grid[0, 0, :4] = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    grid[1, 0, :4] = torch.tensor([[-1.3, 0.0], [1.3, 0.0], [0.0, -1.3], [0.0, 1.3]])
    return value, grid


class TestGridSamplerDecomposition:
    """``grid_sampler_2d_gather`` must match ``F.grid_sample`` and stay clear of the ops Core AI mishandles."""

    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("padding_mode, padding_code", [("zeros", 0), ("border", 1)])
    def test_matches_grid_sample(self, padding_mode: str, padding_code: int, align_corners: bool) -> None:
        """Every supported mode matches PyTorch's kernel, including out-of-bounds and corner samples."""
        value, grid = _sampling_case()
        expected = F.grid_sample(value, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)
        actual = grid_sampler_2d_gather(value, grid, 0, padding_code, align_corners)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    def test_keeps_half_precision(self) -> None:
        """A float16 feature map is sampled in float16, as the fp16 export traces it."""
        value, grid = _sampling_case()
        actual = grid_sampler_2d_gather(value.half(), grid.half(), 0, 0, False)
        assert actual.dtype == torch.float16
        expected = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=5e-2)

    @pytest.mark.parametrize(
        "interpolation_mode, padding_mode",
        [pytest.param(1, 0, id="nearest"), pytest.param(2, 0, id="bicubic"), pytest.param(0, 2, id="reflection")],
    )
    def test_rejects_unsupported_modes(self, interpolation_mode: int, padding_mode: int) -> None:
        """Modes RF-DETR never exports are refused rather than silently sampled bilinearly."""
        value, grid = _sampling_case()
        with pytest.raises(NotImplementedError, match="grid_sampler_2d"):
            grid_sampler_2d_gather(value, grid, interpolation_mode, padding_mode, False)

    @pytest.mark.parametrize("padding_mode", ["zeros", "border"])
    def test_decomposed_graph_has_no_grid_sampler_and_no_bool_masks(self, padding_mode: str) -> None:
        """The decomposed graph must not contain the unsupported op, nor any comparison-to-bool mask chain.

        coreai-torch has no lowering for ``aten.grid_sampler_2d``, and a comparison -> bool -> float mask chain (the
        textbook in-bounds mask of a gather sampler) makes the Core AI runtime overwrite an unrelated live tensor
        (apple/coreai-torch#11). The decomposition therefore builds its masks with float arithmetic only.

        The table is built on ``torch.export.default_decompositions()``, not ``{}``: ``torch.export.export`` leaves
        ``aten.grid_sampler`` in the graph, and only the default table's own ``CompositeImplicitAutograd`` lowering
        turns it into the ``aten.grid_sampler_2d.default`` this module's override actually matches. An empty base never
        reaches that override, so ``grid_sampler_2d_gather`` never runs and every assertion here would pass vacuously;
        the positive ``aten.gather`` assertion is what catches that regression.
        """
        value, grid = _sampling_case()
        exported = torch.export.export(_GridSample(padding_mode, align_corners=False), (value, grid))
        base = torch.export.default_decompositions()
        decomposed = exported.run_decompositions(coreai_decomposition_table(base))
        targets = {str(node.target).rsplit(".", 1)[0] for node in decomposed.graph.nodes if node.op == "call_function"}
        assert "aten.gather" in targets, f"grid_sampler_2d_gather never ran, decomposition did not fire: {targets}"
        assert "aten.grid_sampler_2d" not in targets
        assert not targets & _BOOL_PRODUCING_OPS, f"bool-producing ops in the sampler: {targets & _BOOL_PRODUCING_OPS}"
        torch.testing.assert_close(
            decomposed.module()(value, grid),
            F.grid_sample(value, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False),
            rtol=0,
            atol=1e-5,
        )

    def test_table_extends_the_base_table_without_mutating_it(self) -> None:
        """The Core AI table is the base table plus the sampler entry; the caller's table is left alone."""
        base = {torch.ops.aten.silu.default: object()}
        table = coreai_decomposition_table(base)
        assert table[torch.ops.aten.grid_sampler_2d.default] is grid_sampler_2d_gather
        assert table[torch.ops.aten.silu.default] is base[torch.ops.aten.silu.default]
        assert torch.ops.aten.grid_sampler_2d.default not in base

    def test_table_also_lowers_the_rank_agnostic_sampler(self) -> None:
        """``aten.grid_sampler`` maps to the same decomposition as its 4-D form.

        ``F.grid_sample`` exports as the rank-agnostic ``aten.grid_sampler``; only a base decomposition table rewrites
        it into ``aten.grid_sampler_2d``. Registering both keeps the Core AI table working with a base table that
        preserves the rank-agnostic op, instead of failing deep inside the converter on an unlowerable node.
        """
        assert coreai_decomposition_table({})[torch.ops.aten.grid_sampler.default] is grid_sampler_2d_gather


class _TopK(torch.nn.Module):
    """Return the indices of the four largest scores, as the two-stage query selection does."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.topk(scores, 4, dim=1)[1].to(torch.int32)


#: fp16 scores whose ranking is exact: ``_TopK`` must return ``[[7, 6, 5, 4]]``.
_RANKED_SCORES = torch.arange(8, dtype=torch.float16)[None]


class TestTopkDecomposition:
    """``topk_in_float32`` must move an fp16 ``topk`` to float32 and leave every other ``topk`` alone."""

    def test_float16_topk_runs_in_float32(self) -> None:
        """An fp16 ranking is cast to float32 before ``topk`` and the indices stay those of eager fp16.

        On the Neural Engine an fp16 ``topk`` returns its indices as 16-bit integers in an int32 buffer, so every
        element packs two indices and the queries the decoder gathers are garbage. A float32 ``topk`` is not placed
        there.
        """
        scores = _RANKED_SCORES
        exported = torch.export.export(_TopK(), (scores,)).run_decompositions(coreai_decomposition_table({}))
        topk_inputs = [
            node.args[0].meta["val"].dtype
            for node in exported.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.aten.topk.default
        ]
        assert topk_inputs == [torch.float32]
        torch.testing.assert_close(exported.module()(scores), _TopK()(scores))

    def test_values_keep_the_input_dtype(self) -> None:
        """The values ``topk`` returns are cast back, so the rest of an fp16 graph stays fp16."""
        values, indices = topk_in_float32(torch.tensor([[0.5, 2.0, 1.0]], dtype=torch.float16), 2, 1)
        assert values.dtype == torch.float16
        assert indices.tolist() == [[1, 2]]

    def test_other_dtypes_are_left_to_the_default(self) -> None:
        """A float32 ``topk`` needs no workaround, so the decomposition declines it."""
        assert topk_in_float32(torch.zeros(1, 4), 2, 1) is NotImplemented

    def test_table_carries_the_topk_entry(self) -> None:
        """The Core AI table registers the workaround for ``aten.topk``."""
        assert coreai_decomposition_table({})[torch.ops.aten.topk.default] is topk_in_float32


# ---------------------------------------------------------------------------
# CoreAIExporter — unit / dependency behaviour (coreai-torch mocked)
# ---------------------------------------------------------------------------


def _make_export_graph(*, backbone_only: bool = False) -> ExportGraph:
    """Build a minimal prepared graph a :class:`CoreAIExporter` can be handed without a real detector.

    Examples:
        >>> _make_export_graph(backbone_only=True).backbone_only
        True
    """
    return ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 3, 32, 32),
        input_names=("input",),
        output_names=("dets", "labels"),
        dynamic_axes=None,
        shape=(32, 32),
        backbone_only=backbone_only,
    )


@contextlib.contextmanager
def _mocked_coreai_stack() -> Any:
    """Stand in for ``coreai_torch`` and ``coreai.runtime`` so the exporter runs without either installed.

    Yields the mocks so a test can inspect how the converter was driven. ``torch.export.export`` is mocked too: tracing
    is the converter's input, not what these tests are about.

    Examples:
        >>> with _mocked_coreai_stack() as mocks:
        ...     import coreai_torch
        ...     coreai_torch.TorchConverter().add_exported_program(None) is mocks.converter
        True
    """
    coreai_torch = mock.MagicMock(name="coreai_torch")
    coreai_torch.get_decomp_table.return_value = {}
    converter = coreai_torch.TorchConverter.return_value
    converter.add_exported_program.return_value = converter
    program = converter.to_coreai.return_value
    runtime = mock.MagicMock(name="coreai.runtime")
    coreai = mock.MagicMock(name="coreai", runtime=runtime)
    exported_program = mock.MagicMock(name="exported_program")
    exported_program.run_decompositions.return_value = exported_program
    with (
        mock.patch.dict(sys.modules, {"coreai_torch": coreai_torch, "coreai": coreai, "coreai.runtime": runtime}),
        mock.patch("rfdetr.export._coreai.exporter._IS_COREAI_TORCH_AVAILABLE", True),
        mock.patch("torch.export.export", return_value=exported_program) as export,
    ):
        yield mock.Mock(
            coreai_torch=coreai_torch,
            converter=converter,
            program=program,
            runtime=runtime,
            export=export,
            exported_program=exported_program,
        )


class TestCheckCoreaiTorchAvailable:
    """Tests for ``_check_coreai_torch_available``."""

    def test_raises_install_hint_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ``coreai-torch`` is reported with the extra that installs it."""
        monkeypatch.setattr("rfdetr.export._coreai.exporter._IS_COREAI_TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"pip install \"rfdetr\[coreai\]\""):
            _check_coreai_torch_available()

    def test_returns_false_without_raising_when_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``raise_error=False`` turns the missing dependency into a plain ``False``."""
        monkeypatch.setattr("rfdetr.export._coreai.exporter._IS_COREAI_TORCH_AVAILABLE", False)
        assert _check_coreai_torch_available(raise_error=False) is False

    @coreai_only
    def test_returns_true_when_installed(self) -> None:
        """An installed ``coreai-torch`` is reported as available."""
        assert _check_coreai_torch_available() is True


class TestCoreAIExporter:
    """Argument handling of ``CoreAIExporter`` with the Core AI stack mocked out."""

    def test_missing_dependency_raises_before_tracing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``coreai-torch`` the export fails with the install hint, before any tracing work."""
        monkeypatch.setattr("rfdetr.export._coreai.exporter._IS_COREAI_TORCH_AVAILABLE", False)
        with mock.patch("torch.export.export") as export, pytest.raises(ImportError, match="rfdetr\\[coreai\\]"):
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, verbose=False))(_make_export_graph())
        export.assert_not_called()

    def test_missing_runtime_package_raises_before_tracing(self, tmp_path: Path) -> None:
        """A missing Core AI runtime distribution is reported before the trace, not after the whole conversion.

        The asset metadata comes from ``coreai.runtime``, a different distribution from the ``coreai_torch`` the
        availability probe covers. Building it up front keeps the failure at the same choke point as every other missing
        dependency, instead of after a full trace and conversion.
        """
        coreai_torch = mock.MagicMock(name="coreai_torch")
        with (
            mock.patch("rfdetr.export._coreai.exporter._IS_COREAI_TORCH_AVAILABLE", True),
            mock.patch.dict(sys.modules, {"coreai_torch": coreai_torch, "coreai": None, "coreai.runtime": None}),
            mock.patch("torch.export.export") as export,
            pytest.raises(ImportError),
        ):
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, verbose=False))(_make_export_graph())
        export.assert_not_called()

    @pytest.mark.parametrize(
        "variant_name, output_name, precision, backbone_only, expected",
        [
            pytest.param("rfdetr-nano", None, None, False, "rfdetr-nano_fp32.aimodel", id="variant"),
            pytest.param("rfdetr-nano", None, "float16", False, "rfdetr-nano_fp16.aimodel", id="variant-fp16"),
            pytest.param(None, None, None, False, "inference_model_fp32.aimodel", id="default"),
            pytest.param("rfdetr-nano", "custom", "float16", False, "custom.aimodel", id="custom-name"),
            pytest.param("rfdetr-nano", None, None, True, "rfdetr-nano_fp32-backbone.aimodel", id="backbone"),
            pytest.param(None, "custom", None, True, "custom-backbone.aimodel", id="custom-backbone"),
        ],
    )
    def test_artifact_name(
        self,
        tmp_path: Path,
        variant_name: str | None,
        output_name: str | None,
        precision: str | None,
        backbone_only: bool,
        expected: str,
    ) -> None:
        """The artifact name follows the shared naming precedence and never lets a backbone overwrite a detector."""
        config = CoreAIConfig(
            output_dir=tmp_path, variant_name=variant_name, output_name=output_name, precision=precision, verbose=False
        )
        with _mocked_coreai_stack() as stack:
            path = CoreAIExporter(config)(_make_export_graph(backbone_only=backbone_only))
        assert path == tmp_path / expected
        stack.program.save_asset.assert_called_once()
        assert stack.program.save_asset.call_args.args[0] == path

    def test_converter_receives_graph_names_and_the_sampler_decomposition(self, tmp_path: Path) -> None:
        """Inputs and outputs keep the graph's names, and the decomposition table carries the sampler entry."""
        with _mocked_coreai_stack() as stack:
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, verbose=False))(_make_export_graph())
        (table,) = stack.exported_program.run_decompositions.call_args.args
        assert table[torch.ops.aten.grid_sampler_2d.default] is grid_sampler_2d_gather
        kwargs = stack.converter.add_exported_program.call_args.kwargs
        assert kwargs["input_names"] == ["input"]
        assert kwargs["output_names"] == ["dets", "labels"]

    def test_surviving_grid_sampler_is_refused_before_the_converter(self, tmp_path: Path) -> None:
        """A grid sampler the table failed to lower is named here rather than failing inside ``coreai-torch``.

        The table registers the decomposition for both grid-sampling ops, so this only fires if a future torch or
        converter release moves the op out from under it. Simulated here with a table that lowers neither.
        """
        coreai_torch = mock.MagicMock(name="coreai_torch")
        coreai_torch.get_decomp_table.return_value = {}
        graph = ExportGraph(
            model=_FixedGridSample(),
            input_tensors=torch.zeros(1, 3, 8, 8),
            input_names=("input",),
            output_names=("dets",),
            dynamic_axes=None,
            shape=(8, 8),
            backbone_only=False,
        )
        exporter = CoreAIExporter(CoreAIConfig(output_dir=tmp_path, verbose=False))
        with (
            mock.patch.dict(sys.modules, {"coreai_torch": coreai_torch}),
            mock.patch("rfdetr.export._coreai.exporter.coreai_decomposition_table", return_value={}),
            pytest.raises(NotImplementedError, match="aten.grid_sampler"),
        ):
            exporter._export_program(graph, torch.float32)

    def test_float16_traces_a_half_precision_graph(self, tmp_path: Path) -> None:
        """``precision="float16"`` traces the model and its example input in float16."""
        with _mocked_coreai_stack() as stack:
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, precision="float16", verbose=False))(_make_export_graph())
        (traced_input,) = stack.export.call_args.args[1]
        assert traced_input.dtype == torch.float16

    def test_float32_is_the_default(self, tmp_path: Path) -> None:
        """No precision means a float32 graph."""
        with _mocked_coreai_stack() as stack:
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, verbose=False))(_make_export_graph())
        (traced_input,) = stack.export.call_args.args[1]
        assert traced_input.dtype == torch.float32

    def test_invalid_precision_is_rejected_before_tracing(self, tmp_path: Path) -> None:
        """An unknown precision is a ``ValueError`` raised before the converter runs."""
        with _mocked_coreai_stack() as stack, pytest.raises(ValueError, match="precision"):
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, precision="int8", verbose=False))(_make_export_graph())
        stack.export.assert_not_called()

    @pytest.mark.parametrize(
        "notes, stored",
        [
            pytest.param("trained on pallets", "trained on pallets", id="string"),
            pytest.param({"run": 3, "classes": ["box"]}, json.dumps({"run": 3, "classes": ["box"]}), id="json"),
        ],
    )
    def test_notes_are_written_to_the_asset_metadata(self, tmp_path: Path, notes: object, stored: str) -> None:
        """``notes`` land in the ``.aimodel`` metadata under the same key the ONNX export uses."""
        with _mocked_coreai_stack() as stack:
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, notes=notes, verbose=False))(_make_export_graph())
        metadata = stack.runtime.AIModelAssetMetadata.return_value
        metadata.set_custom.assert_any_call("rfdetr_notes", stored)
        assert stack.program.save_asset.call_args.kwargs["metadata"] is metadata

    def test_notes_do_not_warn(self, tmp_path: Path, recwarn: pytest.WarningsRecorder) -> None:
        """Core AI has a metadata slot, so ``notes`` must not trigger the dropped-notes warning."""
        CoreAIExporter(CoreAIConfig(output_dir=tmp_path, notes="kept", verbose=False))
        assert not [w for w in recwarn if "notes" in str(w.message)]

    def test_dynamic_batch_is_refused(self, tmp_path: Path) -> None:
        """A dynamic batch is refused at construction, with the reason and what to do instead."""
        with pytest.raises(NotImplementedError, match="one .aimodel per batch size"):
            CoreAIExporter(CoreAIConfig(output_dir=tmp_path, dynamic_batch=True))

    def test_build_config_reads_coreai_precision(self) -> None:
        """``RFDETR.export(coreai_precision=...)`` reaches the configuration; other formats' knobs are dropped."""
        config = CoreAIExporter.build_config(output_dir=Path("out"), coreai_precision="float16", coreml_precision="x")
        assert isinstance(config, CoreAIConfig)
        assert config.precision == "float16"


class TestExportFormatParameter:
    """Tests for ``format="coreai"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock the converter and example-image synthesis so ``RFDETR.export()`` stays fast."""
        self._tmp_path = tmp_path
        aimodel = tmp_path / "inference_model.aimodel"
        aimodel.mkdir()
        with (
            mock.patch("rfdetr.export.prepare.make_infer_image", return_value=torch.zeros(1, 3, 560, 560)),
            mock.patch("rfdetr.export._coreai.exporter.CoreAIExporter._convert", return_value=aimodel) as convert,
            mock.patch("rfdetr.export._onnx.exporter.OnnxExporter._convert") as onnx_convert,
        ):
            self._mock_convert = convert
            self._mock_onnx_convert = onnx_convert
            yield

    @staticmethod
    def _make_rfdetr() -> Any:
        """Create a minimal RFDETR instance with mocked internals.

        Examples:
            >>> TestExportFormatParameter._make_rfdetr().model.resolution
            560
        """
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

    def test_coreai_format_dispatches_to_coreai_exporter(self) -> None:
        """``format="coreai"`` reaches ``CoreAIExporter`` (not ONNX) and warns that the format is experimental."""
        with pytest.warns(UserWarning, match="experimental"):
            self._make_rfdetr().export(format="coreai", output_dir=str(self._tmp_path / "out"))
        self._mock_convert.assert_called_once()
        self._mock_onnx_convert.assert_not_called()

    def test_dynamic_batch_raises_before_converter(self) -> None:
        """``dynamic_batch=True`` is refused by ``RFDETR.export()`` before the converter is invoked."""
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            self._make_rfdetr().export(format="coreai", output_dir=str(self._tmp_path / "out"), dynamic_batch=True)
        self._mock_convert.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end (gated) — real convert + float32 parity vs eager PyTorch
# ---------------------------------------------------------------------------

_COREAI_E2E_VARIANTS = [
    pytest.param(("RFDETRNano", ("dets", "labels")), id="detection"),
    pytest.param(("RFDETRSegNano", ("dets", "labels", "masks")), id="segmentation"),
    pytest.param(("RFDETRKeypointPreview", ("dets", "labels", "keypoints")), id="keypoint"),
]


def _require_compute_unit(unit: str) -> None:
    """Skip the calling test when the host exposes no *unit* (``"gpu"`` or ``"neural_engine"``) to Core AI.

    Loading with a preferred compute unit the host lacks raises instead of falling back; GitHub's virtualized
    macOS 27 runner, for one, exposes only the CPU.

    Examples:
        Requires the Core AI runtime, so documentation only:

        >>> callable(_require_compute_unit)
        True
    """
    if unit in ("cpu", "default"):
        return
    import coreai.runtime as rt

    wanted = str(getattr(rt.ComputeUnitKind, unit)())
    if wanted not in {str(kind) for kind in rt.ComputeUnitKind.available_kinds()}:
        pytest.skip(f"this host exposes no {wanted} to Core AI")


def _run_aimodel(path: Path, inputs: dict[str, np.ndarray], output_names: tuple[str, ...], unit: str) -> list[Any]:
    """Load *path* with the Core AI runtime on *unit* (``"cpu"``, ``"gpu"``, ``"neural_engine"`` or ``"default"``) and
    run once; the outputs keep the dtypes the runtime returns.

    Examples:
        Requires the Core AI runtime (macOS 27) and a real ``.aimodel``, so documentation only:

        >>> callable(_run_aimodel)
        True
    """
    import coreai.runtime as rt

    async def run() -> list[Any]:
        if unit == "cpu":
            options = rt.SpecializationOptions.cpu_only()
        elif unit == "default":
            options = rt.SpecializationOptions.default()
        else:
            options = rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)())
        model = await rt.AIModel.load(path, options)
        function = model.load_function("main")
        outputs = await function({name: rt.NDArray(np.ascontiguousarray(a)) for name, a in inputs.items()})
        return [torch.from_numpy(np.array(outputs[name].numpy())) for name in output_names]

    return asyncio.run(run())


def _coreai_parity_diffs(
    aimodel_path: Path, pytorch_model: torch.nn.Module, example_input: torch.Tensor, output_names: tuple[str, ...], unit
) -> list[float]:
    """Run *example_input* through eager export-mode PyTorch and Core AI; return per-output max-abs-diff by name.

    Examples:
        Requires the Core AI runtime (macOS 27) and a real ``.aimodel``, so documentation only:

        >>> callable(_coreai_parity_diffs)
        True
    """
    eager = eager_reference_tensors(pytorch_model, example_input)
    coreai = _run_aimodel(aimodel_path, {"input": example_input.numpy()}, output_names, unit)
    return max_abs_output_diffs(eager, coreai, check_shape=True, names=list(output_names))


def _two_stage_rank_margin(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Return the smallest gap between the neighbouring scores that decide the two-stage ``torch.topk``.

    Examples:
        >>> class _Ranker(torch.nn.Module):
        ...     def forward(self, x):
        ...         return torch.topk(x, 1, dim=-1)
        >>> _two_stage_rank_margin(_Ranker(), torch.tensor([[1.0, 0.75, 0.0]]))
        0.25
    """
    with mock.patch("torch.topk", wraps=torch.topk) as topk, torch.no_grad():
        model(example_input.clone())
    assert topk.call_count == 1, f"expected one two-stage torch.topk call, got {topk.call_count}"
    scores, k = topk.call_args.args[:2]
    top = scores.sort(dim=-1, descending=True).values[..., : k + 1]
    return float((top[..., :-1] - top[..., 1:]).min())


@pytest.fixture(scope="module")
def people_walking_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download supervision's ``PEOPLE_WALKING`` asset once, shared across Core AI e2e tests."""
    asset_dir = tmp_path_factory.mktemp("coreai_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        return Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module", params=_COREAI_E2E_VARIANTS)
def coreai_export(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Any, torch.Tensor, Path, tuple[str, ...]]:
    """Export each variant with ``_COREAI_E2E_NUM_QUERIES`` queries, so its two-stage ranking is well separated."""
    model_cls_name, output_names = request.param
    out_dir = tmp_path_factory.mktemp(f"coreai_{model_cls_name.lower()}")
    seed_all(_COREAI_EXPORT_SEED)
    detector = getattr(rfdetr, model_cls_name)(pretrain_weights=None, num_queries=_COREAI_E2E_NUM_QUERIES)
    aimodel_path = detector.export(output_dir=str(out_dir), format="coreai", verbose=False)
    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return model, example, Path(aimodel_path), output_names


@coreai_only
@coreai_runtime_only
@pytest.mark.integration
@pytest.mark.e2e_coreai
class TestCoreAIEndToEnd:
    """Real Core AI export + float32 numerical parity against eager PyTorch (``-m e2e_coreai``)."""

    def test_aimodel_written(self, coreai_export: tuple[Any, torch.Tensor, Path, tuple[str, ...]]) -> None:
        """The export writes a non-empty ``.aimodel`` bundle named with its precision."""
        _, _, aimodel_path, _ = coreai_export
        assert aimodel_path.is_dir()
        assert aimodel_path.name.endswith("_fp32.aimodel")
        assert any(aimodel_path.iterdir())

    @pytest.mark.parametrize("unit, bound", [("cpu", _COREAI_MAX_ABS_DIFF), ("gpu", _COREAI_GPU_MAX_ABS_DIFF)])
    def test_outputs_match_pytorch_structured(
        self, coreai_export: tuple[Any, torch.Tensor, Path, tuple[str, ...]], unit: str, bound: float
    ) -> None:
        """Core AI matches eager on structured (gradient + checkerboard) input on the CPU and the GPU."""
        _require_compute_unit(unit)
        model, example, aimodel_path, output_names = coreai_export
        assert _two_stage_rank_margin(model, example) >= _MIN_TWO_STAGE_RANK_MARGIN
        diffs = _coreai_parity_diffs(aimodel_path, model, example, output_names, unit)
        assert max(diffs) < bound, f"Core AI ({unit}) diverges from PyTorch: {dict(zip(output_names, diffs))}"

    def test_outputs_match_pytorch_supervision_image(
        self,
        coreai_export: tuple[Any, torch.Tensor, Path, tuple[str, ...]],
        people_walking_image_path: Path,
    ) -> None:
        """Core AI matches eager on ``ImageAssets.PEOPLE_WALKING``."""
        model, structured, aimodel_path, output_names = coreai_export
        example = _parity_input_from_image(people_walking_image_path, int(structured.shape[-1]))
        assert _two_stage_rank_margin(model, example) >= _MIN_TWO_STAGE_RANK_MARGIN
        diffs = _coreai_parity_diffs(aimodel_path, model, example, output_names, "cpu")
        assert max(diffs) < _COREAI_MAX_ABS_DIFF, f"Core AI diverges from PyTorch: {dict(zip(output_names, diffs))}"

    def test_backbone_outputs_match_pytorch(self, tmp_path: Path) -> None:
        """The public ``backbone_only=True`` export runs every feature-map output on Core AI."""
        seed_all(_COREAI_EXPORT_SEED)
        detector = rfdetr.RFDETRNano(pretrain_weights=None)
        aimodel_path = detector.export(output_dir=str(tmp_path), format="coreai", backbone_only=True, verbose=False)
        assert aimodel_path.name.endswith("-backbone.aimodel")
        backbone = detector.model.model.backbone[0].to("cpu").eval()
        reference = _BackboneExport(backbone)
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        eager = eager_reference_tensors(reference, example)
        names = tuple(resolve_output_names(detector.model_config, backbone_only=True, backbone=backbone))
        coreai = _run_aimodel(aimodel_path, {"input": example.numpy()}, names, "cpu")
        diffs = max_abs_output_diffs(eager, coreai, check_shape=True, names=list(names))
        assert max(diffs) < _COREAI_MAX_ABS_DIFF, f"Core AI backbone diverges from PyTorch: {diffs}"

    def test_default_query_count_runs_with_eager_shapes(self, tmp_path: Path) -> None:
        """The shipped 300-query Nano graph runs with eager's output shapes and finite values."""
        seed_all(_COREAI_EXPORT_SEED)
        detector = rfdetr.RFDETRNano(pretrain_weights=None)
        aimodel_path = detector.export(output_dir=str(tmp_path), format="coreai", verbose=False)
        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        diffs = _coreai_parity_diffs(aimodel_path, model, example, ("dets", "labels"), "cpu")
        assert all(np.isfinite(diffs)), f"Core AI produced non-finite outputs: {diffs}"

    def test_float16_topk_indices_survive_the_neural_engine(self, tmp_path: Path) -> None:
        """An fp16 ``topk`` returns plain indices on the Neural Engine, as RF-DETR's two-stage selection needs.

        iOS and iPadOS pick the Neural Engine for fp16 graphs by default. There an fp16 ``topk`` writes 16-bit indices
        into its int32 output, so ``topk(arange(8), 4)`` returns the whole descending order, two indices per element,
        instead of ``[7, 6, 5, 4]``. An fp16 RF-DETR export then gathers the wrong encoder tokens and detects nothing.
        The scores are exact in fp16, so the expected indices are too.
        """
        _require_compute_unit("neural_engine")
        import coreai.runtime as rt
        from coreai_torch import TorchConverter, get_decomp_table

        exported = torch.export.export(_TopK(), (_RANKED_SCORES,))
        program = (
            TorchConverter()
            .add_exported_program(
                exported.run_decompositions(coreai_decomposition_table(get_decomp_table())),
                input_names=["scores"],
                output_names=["indices"],
            )
            .to_coreai()
        )
        path = tmp_path / "topk.aimodel"
        program.save_asset(path, metadata=rt.AIModelAssetMetadata())
        (indices,) = _run_aimodel(path, {"scores": _RANKED_SCORES.numpy()}, ("indices",), "neural_engine")
        assert indices.tolist() == [[7, 6, 5, 4]], f"corrupt topk indices on the Neural Engine: {indices.tolist()}"

    def test_float16_export_runs(self, tmp_path: Path) -> None:
        """A float16 export loads with the default specialization and returns float16 outputs of the eager shapes."""
        seed_all(_COREAI_EXPORT_SEED)
        detector = rfdetr.RFDETRNano(pretrain_weights=None, num_queries=_COREAI_E2E_NUM_QUERIES)
        aimodel_path = detector.export(
            output_dir=str(tmp_path), format="coreai", coreai_precision="float16", verbose=False
        )
        assert aimodel_path.name.endswith("_fp16.aimodel")
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution).half()
        dets, labels = _run_aimodel(aimodel_path, {"input": example.numpy()}, ("dets", "labels"), "default")
        assert dets.dtype == torch.float16 and labels.dtype == torch.float16
        assert tuple(dets.shape) == (1, _COREAI_E2E_NUM_QUERIES, 4)
        assert labels.shape[:2] == (1, _COREAI_E2E_NUM_QUERIES)
        assert torch.isfinite(dets).all() and torch.isfinite(labels).all()
