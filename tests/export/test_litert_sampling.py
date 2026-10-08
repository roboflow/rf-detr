# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the pixel-row bilinear sampler that ``format="litert"`` captures the deformable attention with."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.export._litert.exporter import LiteRTConfig, LiteRTExporter, ModelWrapper
from rfdetr.export._litert.sampling import pixel_row_grid_sample, pixel_row_sampling
from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
from rfdetr.utilities.tensors import _bilinear_grid_sample


def _value_and_grid(
    batch: int, channels: int, height: int, width: int, points: int, spread: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a value map and a ``(batch, points, 2, 2)`` grid whose coordinates span ``[-spread, spread]``.

    A *spread* above 1 puts samples outside the image, where zeros padding applies.

    Examples:
        >>> value, grid = _value_and_grid(2, 3, 4, 5, 6, 1.5)
        >>> tuple(value.shape), tuple(grid.shape), bool(grid.abs().max() <= 1.5)
        ((2, 3, 4, 5), (2, 6, 2, 2), True)
    """
    generator = torch.Generator().manual_seed(0)
    value = torch.randn(batch, channels, height, width, generator=generator)
    grid = (torch.rand(batch, points, 2, 2, generator=generator) * 2 - 1) * spread
    return value, grid


def _pixel_edge_grid(height: int, width: int) -> torch.Tensor:
    """Build a ``(1, n, 1, 2)`` grid on pixel centres, image borders and points just outside them.

    Examples:
        >>> tuple(_pixel_edge_grid(4, 5).shape)
        (1, 81, 1, 2)
    """
    edges_x = torch.tensor([-3.0, -1.1, -1.0, -1 + 1 / width, 0.0, 1 - 1 / width, 1.0, 1.1, 3.0])
    edges_y = torch.tensor([-3.0, -1.1, -1.0, -1 + 1 / height, 0.0, 1 - 1 / height, 1.0, 1.1, 3.0])
    xx, yy = torch.meshgrid(edges_x, edges_y, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, -1, 1, 2)


def _reference(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``F.grid_sample`` with the settings RF-DETR's deformable attention uses.

    Examples:
        >>> value, grid = _value_and_grid(1, 2, 3, 3, 4, 1.0)
        >>> tuple(_reference(value, grid).shape)
        (1, 2, 4, 2)
    """
    return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class _PixelRowSampler(torch.nn.Module):
    """Run :func:`pixel_row_grid_sample` with its default settings, so ``torch.export`` can capture it."""

    def forward(self, value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return pixel_row_grid_sample(value, grid)


def _sampler_ops(value: torch.Tensor, grid: torch.Tensor) -> set[str]:
    """Name the sampling op ``pixel_row_grid_sample`` captures into a ``torch.export`` graph.

    Returns ``{"embedding"}`` on the pixel-row path and ``{"grid_sampler"}`` when it falls back to the default sampler.
    Capturing only traces the graph, so it also works for dtype pairs the default sampler rejects when it runs.

    Examples:
        >>> value, grid = _value_and_grid(1, 2, 3, 3, 4, 1.0)
        >>> sorted(_sampler_ops(value, grid)), sorted(_sampler_ops(value.double(), grid.double()))
        (['embedding'], ['grid_sampler'])
    """
    program = torch.export.export(_PixelRowSampler(), (value, grid), strict=False)
    targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    return {name for name in ("embedding", "grid_sampler") if any(name in target for target in targets)}


class TestPixelRowGridSample:
    """The pixel-row sampler computes the same values as ``F.grid_sample`` (bilinear, zeros, align_corners=False).

    The comparisons use the default float32 tolerances: the two sum the four corners in a different order, and on x86
    ``F.grid_sample``'s vectorized kernel differs from the pixel-row result by up to about 5e-6 on these shapes.
    """

    @pytest.mark.parametrize(
        ("batch", "channels", "height", "width"),
        [
            pytest.param(16, 16, 24, 24, id="nano-decoder-shape"),
            pytest.param(4, 8, 5, 7, id="non-square"),
            pytest.param(1, 1, 1, 1, id="single-pixel"),
        ],
    )
    def test_matches_grid_sample_inside_and_outside_the_image(
        self, batch: int, channels: int, height: int, width: int
    ) -> None:
        value, grid = _value_and_grid(batch, channels, height, width, points=300, spread=1.5)
        torch.testing.assert_close(pixel_row_grid_sample(value, grid), _reference(value, grid))

    def test_matches_grid_sample_on_pixel_edges(self) -> None:
        value = torch.randn(1, 3, 4, 5, generator=torch.Generator().manual_seed(1))
        grid = _pixel_edge_grid(4, 5)
        torch.testing.assert_close(pixel_row_grid_sample(value, grid), _reference(value, grid))

    @pytest.mark.parametrize(
        ("padding_mode", "align_corners"),
        [("border", False), ("zeros", True)],
    )
    def test_other_modes_use_the_original_sampler(self, padding_mode: str, align_corners: bool) -> None:
        value, grid = _value_and_grid(2, 3, 4, 5, points=10, spread=1.5)
        expected = _bilinear_grid_sample(value, grid, padding_mode=padding_mode, align_corners=align_corners)
        actual = pixel_row_grid_sample(value, grid, padding_mode=padding_mode, align_corners=align_corners)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_float32_uses_pixel_rows(self) -> None:
        value, grid = _value_and_grid(2, 3, 4, 5, points=10, spread=1.5)
        assert _sampler_ops(value, grid) == {"embedding"}

    @pytest.mark.parametrize(
        ("value_dtype", "grid_dtype"),
        [
            (torch.float64, torch.float64),
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float64, torch.float32),
            (torch.float32, torch.float64),
        ],
    )
    def test_other_dtypes_use_the_original_sampler(self, value_dtype: torch.dtype, grid_dtype: torch.dtype) -> None:
        value, grid = _value_and_grid(2, 3, 4, 5, points=10, spread=1.5)
        assert _sampler_ops(value.to(value_dtype), grid.to(grid_dtype)) == {"grid_sampler"}

    @pytest.mark.parametrize(
        ("extra_headroom", "expected_op"),
        [
            (0, "grid_sampler"),
            (1, "embedding"),
        ],
    )
    def test_pixel_row_count_must_stay_below_the_exact_float32_limit(
        self, monkeypatch: pytest.MonkeyPatch, extra_headroom: int, expected_op: str
    ) -> None:
        """A map with exactly ``limit`` padded pixels falls back; one pixel fewer takes the pixel-row path."""
        batch, height, width = 2, 4, 5
        padded_pixels = batch * (height + 2) * (width + 2)
        monkeypatch.setattr("rfdetr.export._litert.sampling._EXACT_FLOAT32_INTEGERS", padded_pixels + extra_headroom)
        value, grid = _value_and_grid(batch, 3, height, width, points=10, spread=1.5)
        assert _sampler_ops(value, grid) == {expected_op}

    def test_fallback_logs_the_inputs_at_debug(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        value, grid = _value_and_grid(2, 3, 4, 5, points=10, spread=1.5)
        # The rf-detr logger does not propagate, so caplog's root handler would see nothing.
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        with caplog.at_level(logging.DEBUG, logger="rf-detr"):
            pixel_row_grid_sample(value, grid, align_corners=True)
        assert "align_corners=True" in caplog.text

    def test_batch_mismatch_raises(self) -> None:
        value, _ = _value_and_grid(2, 3, 4, 5, points=10, spread=1.0)
        _, grid = _value_and_grid(1, 3, 4, 5, points=10, spread=1.0)
        with pytest.raises(RuntimeError):
            pixel_row_grid_sample(value, grid)


class _DeformableCore(torch.nn.Module):
    """Call the deformable-attention core the way the export path does, with the pixel-row sampler.

    Args:
        shapes_hw: Per-level ``(height, width)`` of the flattened value map.

    Examples:
        >>> value, locations, weights = _core_inputs([(4, 4)])
        >>> tuple(_DeformableCore([(4, 4)])(value, locations, weights).shape)
        (1, 3, 16)
    """

    def __init__(self, shapes_hw: list[tuple[int, int]]) -> None:
        super().__init__()
        self.shapes_hw = shapes_hw

    def forward(self, value: torch.Tensor, locations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return ms_deform_attn_core_pytorch(
            value,
            torch.tensor(self.shapes_hw),
            locations,
            weights,
            value_spatial_shapes_hw=self.shapes_hw,
            grid_sample=pixel_row_grid_sample,
        )


def _core_inputs(
    shapes_hw: list[tuple[int, int]], batch: int = 1, heads: int = 2, rank6: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(value, sampling_locations, attention_weights)`` for the deformable-attention core.

    The locations use the rank-5 export layout, or the rank-6 eager layout when *rank6* is set. Both carry the same
    values.

    Examples:
        >>> value, locations, weights = _core_inputs([(4, 4)])
        >>> tuple(value.shape), tuple(locations.shape), tuple(weights.shape)
        ((1, 2, 8, 16), (1, 3, 2, 4, 2), (1, 3, 2, 4))
        >>> value, locations, weights = _core_inputs([(4, 4), (2, 3)], batch=3, heads=4, rank6=True)
        >>> tuple(value.shape), tuple(locations.shape), tuple(weights.shape)
        ((3, 4, 8, 22), (3, 3, 4, 2, 4, 2), (3, 3, 4, 8))
    """
    generator = torch.Generator().manual_seed(0)
    head_dim, len_query, points = 8, 3, 4
    levels = len(shapes_hw)
    value = torch.randn(batch, heads, head_dim, sum(height * width for height, width in shapes_hw), generator=generator)
    locations = torch.rand(batch, len_query, heads, levels * points, 2, generator=generator) * 1.4 - 0.2
    if rank6:
        locations = locations.view(batch, len_query, heads, levels, points, 2)
    weights = torch.softmax(torch.randn(batch, len_query, heads, levels * points, generator=generator), -1)
    return value, locations, weights


class TestDeformableCoreWithPixelRows:
    """The deformable-attention core samples with the ``grid_sample`` it is given."""

    def test_captured_core_has_no_grid_sampler(self) -> None:
        program = torch.export.export(_DeformableCore([(4, 4)]), _core_inputs([(4, 4)]), strict=False)
        targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
        assert not any("grid_sampler" in target for target in targets), sorted(targets)
        assert any("embedding" in target for target in targets), sorted(targets)

    @pytest.mark.parametrize(
        "shapes_hw",
        [
            [(4, 4)],
            [(4, 4), (2, 3)],
            [(4, 4), (2, 3), (3, 2)],
        ],
    )
    def test_core_output_is_unchanged(self, shapes_hw: list[tuple[int, int]]) -> None:
        value, locations, weights = _core_inputs(shapes_hw, batch=3, heads=4)
        expected = ms_deform_attn_core_pytorch(value, torch.tensor(shapes_hw), locations, weights)
        torch.testing.assert_close(_DeformableCore(shapes_hw)(value, locations, weights), expected)

    def test_core_output_is_unchanged_for_rank6_locations(self) -> None:
        shapes_hw = [(4, 4), (2, 3)]
        value, locations, weights = _core_inputs(shapes_hw, batch=3, heads=4, rank6=True)
        expected = ms_deform_attn_core_pytorch(value, torch.tensor(shapes_hw), locations, weights)
        torch.testing.assert_close(_DeformableCore(shapes_hw)(value, locations, weights), expected)


def _attention() -> MSDeformAttn:
    """Build a small single-level deformable-attention module.

    Examples:
        >>> _attention().n_levels
        1
    """
    return MSDeformAttn(d_model=16, n_levels=1, n_heads=2, n_points=2)


class TestPixelRowSampling:
    """``pixel_row_sampling`` sets the pixel-row sampler on the given model's deformable attention only."""

    def test_model_attention_uses_pixel_rows_inside_the_context(self) -> None:
        model = torch.nn.ModuleList([_attention(), _attention()])
        with pixel_row_sampling(model):
            assert [attention.grid_sample for attention in model] == [pixel_row_grid_sample] * 2

    def test_other_model_keeps_the_default_sampler(self) -> None:
        other = _attention()
        with pixel_row_sampling(_attention()):
            assert other.grid_sample is _bilinear_grid_sample

    def test_default_sampler_is_restored_after_the_context(self) -> None:
        attention = _attention()
        with pixel_row_sampling(attention):
            pass
        assert attention.grid_sample is _bilinear_grid_sample

    def test_default_sampler_is_restored_after_an_error(self) -> None:
        attention = _attention()
        with pytest.raises(RuntimeError), pixel_row_sampling(attention):
            raise RuntimeError("conversion failed")
        assert attention.grid_sample is _bilinear_grid_sample

    def test_instance_sampler_is_restored_after_the_context(self) -> None:
        attention = _attention()
        attention.grid_sample = F.grid_sample
        with pixel_row_sampling(attention):
            pass
        assert attention.grid_sample is F.grid_sample

    def test_explicit_none_sampler_is_restored_after_the_context(self) -> None:
        attention = _attention()
        attention.grid_sample = None  # type: ignore[assignment]
        with pixel_row_sampling(attention):
            pass
        assert attention.grid_sample is None

    def test_installed_samplers_are_restored_when_a_later_install_fails(self) -> None:
        model = torch.nn.ModuleList([_attention(), _attention()])
        # A registered submodule cannot be replaced by a function, so the second install raises TypeError.
        model[1].grid_sample = torch.nn.Identity()
        with pytest.raises(TypeError), pixel_row_sampling(model):
            pass
        assert model[0].grid_sample is _bilinear_grid_sample

    def test_sampler_removed_inside_the_context_restores_the_default(self) -> None:
        attention = _attention()
        with pixel_row_sampling(attention):
            del attention.grid_sample
        assert attention.grid_sample is _bilinear_grid_sample

    @pytest.mark.parametrize("export_mode", [False, True])
    def test_attention_forward_matches_the_default_sampler(self, export_mode: bool) -> None:
        """``MSDeformAttn.forward`` gives the same output inside and outside the context, in eager and export mode."""
        shapes_hw = [(4, 4), (2, 3)]
        batch, len_query, d_model = 3, 5, 16
        attention = MSDeformAttn(d_model=d_model, n_levels=2, n_heads=4, n_points=2)
        if export_mode:
            attention.export()
        generator = torch.Generator().manual_seed(0)
        inputs = (
            torch.randn(batch, len_query, d_model, generator=generator),
            torch.rand(batch, len_query, 2, 2, generator=generator),
            torch.randn(batch, sum(height * width for height, width in shapes_hw), d_model, generator=generator),
            torch.tensor(shapes_hw),
            torch.tensor([0, 16]),
        )
        expected = attention(*inputs, input_spatial_shapes_hw=shapes_hw)
        with pixel_row_sampling(attention):
            actual = attention(*inputs, input_spatial_shapes_hw=shapes_hw)
        torch.testing.assert_close(actual, expected)


class TestLiteRTExporterSampling:
    """The LiteRT exporter captures the model with the pixel-row sampler in place."""

    def test_conversion_runs_with_the_pixel_row_sampler(self, tmp_path: Path) -> None:
        samplers_seen = []

        def convert(wrapped_model: torch.nn.Module, *args: object) -> MagicMock:
            samplers_seen.extend(m.grid_sample for m in wrapped_model.modules() if isinstance(m, MSDeformAttn))
            return MagicMock()

        litert_torch = MagicMock()
        litert_torch.convert.side_effect = convert
        exporter = LiteRTExporter(LiteRTConfig(output_dir=tmp_path, verbose=False))
        model = ModelWrapper(_attention())
        exporter._convert_and_save(litert_torch, model, torch.zeros(1, 3, 4, 4), tmp_path / "model.tflite")
        assert samplers_seen == [pixel_row_grid_sample]

    def test_default_sampler_is_restored_when_conversion_fails(self, tmp_path: Path) -> None:
        """A failing ``litert_torch.convert`` surfaces as the exporter's RuntimeError and leaves the sampler
        restored."""
        litert_torch = MagicMock()
        litert_torch.convert.side_effect = ValueError("boom")
        exporter = LiteRTExporter(LiteRTConfig(output_dir=tmp_path, verbose=False))
        attention = _attention()
        with pytest.raises(RuntimeError, match="Failed to export model to LiteRT"):
            exporter._convert_and_save(litert_torch, ModelWrapper(attention), torch.zeros(1, 3, 4, 4), tmp_path / "m")
        assert attention.grid_sample is _bilinear_grid_sample
