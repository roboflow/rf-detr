# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Direct numerical and gradient tests for ``ms_deform_attn_core_pytorch``."""

from __future__ import annotations

import torch

from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


class TestMsDeformAttnCorePytorch:
    """Ground-truth numerical and gradient checks for the CPU deformable-attention core.

    ``tests/models/test_transformer.py::TestMSDeformAttnCorePytorch`` already exercises this
    function's shape contracts, the Python-int-pair vs. tensor ``spatial_shapes`` paths, and the
    single-level sample-packing skip, driving it to 100% line coverage. Those tests compare code
    paths against each other rather than against an independently derived expectation, so they add
    two checks no existing test performs: a hand-computed bilinear-sampling value, and a
    finite-difference gradient check through value, locations, and weights.
    """

    def test_single_level_center_sample_averages_the_four_corner_pixels(self) -> None:
        """A sampling location at the exact grid center returns the mean of all four pixels.

        With ``align_corners=False`` and a 2x2 single-level feature map, normalized location (0.5, 0.5) maps to grid
        coordinate (0, 0), which is exactly equidistant from all four pixel centers, so bilinear interpolation must
        weight each pixel by 0.25. The expected value (2.5) is computed by hand here rather than by re-running another
        part of the function under test.
        """
        # value flattens (height, width) row-major: [[1, 2], [3, 4]] -> [1, 2, 3, 4].
        value = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (batch=1, n_heads=1, head_dim=1, hw=4)
        spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)
        sampling_locations = torch.tensor([0.5, 0.5]).view(1, 1, 1, 1, 1, 2)
        attention_weights = torch.ones(1, 1, 1, 1)

        output = ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        assert output.shape == (1, 1, 1)
        torch.testing.assert_close(output, torch.tensor([[[2.5]]]))

    def test_gradients_flow_to_value_locations_and_weights(self) -> None:
        """Autograd propagates finite, numerically correct gradients to every differentiable input.

        ``ms_deform_attn_core_pytorch`` is the differentiable fallback used whenever the CUDA extension is unavailable,
        so its backward pass must match finite-difference estimates, not merely run without raising. Locations are kept
        strictly inside the grid (away from pixel boundaries) so the numerical estimate does not cross a non-smooth
        bilinear seam.
        """
        spatial_shapes = torch.tensor([[3, 3]], dtype=torch.long)
        value = torch.randn(1, 1, 1, 9, dtype=torch.float64, requires_grad=True)
        sampling_locations = (0.2 + 0.6 * torch.rand(1, 1, 1, 1, 1, 2, dtype=torch.float64)).requires_grad_(True)
        attention_weights = torch.rand(1, 1, 1, 1, dtype=torch.float64, requires_grad=True)

        def _core(
            value: torch.Tensor, sampling_locations: torch.Tensor, attention_weights: torch.Tensor
        ) -> torch.Tensor:
            """Bind ``spatial_shapes`` so only the differentiable inputs are handed to ``gradcheck``."""
            return ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        assert torch.autograd.gradcheck(_core, (value, sampling_locations, attention_weights))
