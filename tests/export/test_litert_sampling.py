# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the pixel-row bilinear sampler that ``format="litert"`` captures the deformable attention with."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import rfdetr.models.ops.functions.ms_deform_attn_func as deform_attention
from rfdetr.export._litert.exporter import LiteRTConfig, LiteRTExporter, ModelWrapper
from rfdetr.export._litert.sampling import pixel_row_grid_sample, pixel_row_sampling
from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
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


class TestPixelRowGridSample:
    """The pixel-row sampler computes the same values as ``F.grid_sample`` (bilinear, zeros, align_corners=False)."""

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
        torch.testing.assert_close(pixel_row_grid_sample(value, grid), _reference(value, grid), rtol=0, atol=1e-6)

    def test_matches_grid_sample_on_pixel_edges(self) -> None:
        value = torch.randn(1, 3, 4, 5, generator=torch.Generator().manual_seed(1))
        grid = _pixel_edge_grid(4, 5)
        torch.testing.assert_close(pixel_row_grid_sample(value, grid), _reference(value, grid), rtol=0, atol=1e-6)

    @pytest.mark.parametrize(
        ("padding_mode", "align_corners"),
        [("border", False), ("zeros", True)],
    )
    def test_other_modes_use_the_original_sampler(self, padding_mode: str, align_corners: bool) -> None:
        value, grid = _value_and_grid(2, 3, 4, 5, points=10, spread=1.5)
        expected = _bilinear_grid_sample(value, grid, padding_mode=padding_mode, align_corners=align_corners)
        actual = pixel_row_grid_sample(value, grid, padding_mode=padding_mode, align_corners=align_corners)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _DeformableCore(torch.nn.Module):
    """Call the single-level deformable-attention core the way the export path does."""

    def forward(self, value: torch.Tensor, locations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        shapes_hw = [(4, 4)]
        return ms_deform_attn_core_pytorch(
            value, torch.tensor(shapes_hw), locations, weights, value_spatial_shapes_hw=shapes_hw
        )


class TestPixelRowSampling:
    """``pixel_row_sampling`` swaps the deformable-attention sampler for the capture only."""

    @staticmethod
    def _core_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(value, sampling_locations, attention_weights)`` in the rank-5 export layout.

        Examples:
            >>> value, locations, weights = TestPixelRowSampling._core_inputs()
            >>> tuple(value.shape), tuple(locations.shape), tuple(weights.shape)
            ((1, 2, 8, 16), (1, 3, 2, 4, 2), (1, 3, 2, 4))
        """
        generator = torch.Generator().manual_seed(0)
        value = torch.randn(1, 2, 8, 16, generator=generator)
        locations = torch.rand(1, 3, 2, 4, 2, generator=generator) * 1.4 - 0.2
        weights = torch.softmax(torch.randn(1, 3, 2, 4, generator=generator), -1)
        return value, locations, weights

    def test_captured_core_has_no_grid_sampler(self) -> None:
        with pixel_row_sampling():
            program = torch.export.export(_DeformableCore(), self._core_inputs(), strict=False)
        targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
        assert not any("grid_sampler" in target for target in targets), sorted(targets)

    def test_core_output_is_unchanged(self) -> None:
        inputs = self._core_inputs()
        expected = _DeformableCore()(*inputs)
        with pixel_row_sampling():
            actual = _DeformableCore()(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_original_sampler_is_restored_after_the_context(self) -> None:
        original = deform_attention._bilinear_grid_sample
        with pixel_row_sampling():
            pass
        assert deform_attention._bilinear_grid_sample is original

    def test_original_sampler_is_restored_after_an_error(self) -> None:
        original = deform_attention._bilinear_grid_sample
        with pytest.raises(RuntimeError), pixel_row_sampling():
            raise RuntimeError("conversion failed")
        assert deform_attention._bilinear_grid_sample is original


class TestLiteRTExporterSampling:
    """The LiteRT exporter captures the model with the pixel-row sampler in place."""

    def test_conversion_runs_with_the_pixel_row_sampler(self, tmp_path: Path) -> None:
        samplers_seen = []

        def convert(*args: object) -> MagicMock:
            samplers_seen.append(deform_attention._bilinear_grid_sample)
            return MagicMock()

        litert_torch = MagicMock()
        litert_torch.convert.side_effect = convert
        exporter = LiteRTExporter(LiteRTConfig(output_dir=tmp_path, verbose=False))
        model = ModelWrapper(torch.nn.Identity())
        exporter._convert_and_save(litert_torch, model, torch.zeros(1, 3, 4, 4), tmp_path / "model.tflite")
        assert samplers_seen == [pixel_row_grid_sample]
