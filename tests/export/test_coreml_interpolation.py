# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The matrices the Core ML export multiplies by in place of ``upsample_bilinear``.

No coremltools needed.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.export._coreml.interpolation import bilinear_interpolation_matrix


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    ("size_in", "size_out"),
    [
        pytest.param((26, 20), (78, 60), id="3x"),
        pytest.param((36, 36), (72, 72), id="2x"),
        pytest.param((5, 7), (1, 1), id="to-one-cell"),
    ],
)
def test_resizes_like_torch(size_in: tuple[int, int], size_out: tuple[int, int], align_corners: bool) -> None:
    """``R_h @ x @ R_w.T`` equals ``F.interpolate(mode="bilinear")``, up by 3 or 2 and down to one cell."""
    x = torch.randn(2, 3, *size_in, generator=torch.Generator().manual_seed(0))
    height = torch.from_numpy(bilinear_interpolation_matrix(size_in[0], size_out[0], align_corners))
    width = torch.from_numpy(bilinear_interpolation_matrix(size_in[1], size_out[1], align_corners))

    expected = F.interpolate(x, size=size_out, mode="bilinear", align_corners=align_corners)
    torch.testing.assert_close(height @ x @ width.T, expected)
