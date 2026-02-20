# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.models.transformer import gen_encoder_output_proposals


def test_gen_encoder_output_proposals_passes_ij_indexing_to_meshgrid(monkeypatch) -> None:
    """`gen_encoder_output_proposals` should call `torch.meshgrid` with explicit ij indexing."""
    original_meshgrid = torch.meshgrid
    call_count = 0

    def _meshgrid_with_indexing_assertion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("indexing") != "ij":
            raise AssertionError("torch.meshgrid must be called with indexing='ij'")
        return original_meshgrid(*args, **kwargs)

    monkeypatch.setattr(torch, "meshgrid", _meshgrid_with_indexing_assertion)

    memory = torch.randn(1, 4, 8)
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)

    output_memory, output_proposals = gen_encoder_output_proposals(
        memory=memory,
        memory_padding_mask=None,
        spatial_shapes=spatial_shapes,
        unsigmoid=True,
    )

    assert call_count == 1
    assert output_memory.shape == memory.shape
    assert output_proposals.shape == (1, 4, 4)
