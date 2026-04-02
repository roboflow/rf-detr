# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.models.postprocess import PostProcess


class TestPostProcessOriented:
    def test_oriented_output_keys(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = pp(outputs, target_sizes)
        assert len(results) == 1
        assert "boxes_obb" in results[0]
        assert "corners" in results[0]
        assert "scores" in results[0]
        assert "labels" in results[0]

    def test_oriented_obb_shape(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 5),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = pp(outputs, target_sizes)
        assert results[0]["boxes_obb"].shape == (5, 5)
        assert results[0]["corners"].shape == (5, 4, 2)

    def test_oriented_scales_spatial_dims(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.full((1, 10, 5), 0.5),
        }
        target_sizes = torch.tensor([[100, 200]])
        results = pp(outputs, target_sizes)
        obb = results[0]["boxes_obb"]
        assert torch.allclose(obb[0, 0], torch.tensor(100.0), atol=1.0)
        assert torch.allclose(obb[0, 1], torch.tensor(50.0), atol=1.0)

    def test_standard_postprocess_unchanged(self) -> None:
        pp = PostProcess(num_select=5, oriented=False)
        outputs = {
            "pred_logits": torch.randn(1, 10, 3),
            "pred_boxes": torch.rand(1, 10, 4),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = pp(outputs, target_sizes)
        assert "boxes" in results[0]
        assert "boxes_obb" not in results[0]

    def test_batch_support(self) -> None:
        pp = PostProcess(num_select=5, oriented=True)
        outputs = {
            "pred_logits": torch.randn(3, 10, 3),
            "pred_boxes": torch.rand(3, 10, 5),
        }
        target_sizes = torch.tensor([[480, 640], [320, 320], [600, 800]])
        results = pp(outputs, target_sizes)
        assert len(results) == 3
