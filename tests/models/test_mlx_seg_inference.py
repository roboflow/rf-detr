# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the MLX segmentation inference backend.

These tests require macOS with Apple Silicon and MLX installed.
They are skipped on other platforms via the ``requires_mlx`` mark.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rfdetr.mlx import is_mlx_available

pytestmark = pytest.mark.mlx

requires_mlx = pytest.mark.skipif(
    not is_mlx_available(),
    reason="MLX not available (requires macOS with Apple Silicon)",
)


class TestConvertSegWeights:
    """Tests for segmentation-head weight conversion."""

    @requires_mlx
    def test_extract_seg_keys(self) -> None:
        """convert_seg_weights only extracts segmentation_head.* keys."""
        from rfdetr.mlx.convert_weights import convert_seg_weights

        state_dict = {
            "segmentation_head.bias": torch.zeros(1),
            "segmentation_head.blocks.0.dwconv.weight": torch.zeros(256, 1, 3, 3),
            "segmentation_head.blocks.0.pwconv1.weight": torch.zeros(256, 256),
            "backbone.0.encoder.encoder.embeddings.cls_token": torch.zeros(1, 1, 384),
            "class_embed.weight": torch.zeros(91, 256),
        }
        seg_weights, num_blocks = convert_seg_weights(state_dict)

        # Only seg keys should appear
        for key in seg_weights:
            assert not key.startswith("backbone")
            assert not key.startswith("class_embed")

        assert "bias" in seg_weights

    @requires_mlx
    def test_pwconv_key_remapping(self) -> None:
        """pwconv1 is remapped to pwconv for each block."""
        from rfdetr.mlx.convert_weights import convert_seg_weights

        state_dict = {
            "segmentation_head.blocks.0.pwconv1.weight": torch.zeros(256, 256),
            "segmentation_head.blocks.1.pwconv1.bias": torch.zeros(256),
        }
        seg_weights, _ = convert_seg_weights(state_dict)

        assert "blocks.0.pwconv.weight" in seg_weights
        assert "blocks.1.pwconv.bias" in seg_weights
        assert "blocks.0.pwconv1.weight" not in seg_weights

    @requires_mlx
    def test_mlp_key_remapping(self) -> None:
        """query_features_block.layers.0 -> fc1, layers.2 -> fc2."""
        from rfdetr.mlx.convert_weights import convert_seg_weights

        state_dict = {
            "segmentation_head.query_features_block.layers.0.weight": torch.zeros(1024, 256),
            "segmentation_head.query_features_block.layers.2.weight": torch.zeros(256, 1024),
        }
        seg_weights, _ = convert_seg_weights(state_dict)

        assert "query_features_block.fc1.weight" in seg_weights
        assert "query_features_block.fc2.weight" in seg_weights
        assert "query_features_block.layers.0.weight" not in seg_weights

    @requires_mlx
    def test_num_blocks_auto_detection(self) -> None:
        """num_blocks is inferred as max block index + 1."""
        from rfdetr.mlx.convert_weights import convert_seg_weights

        state_dict = {
            "segmentation_head.blocks.0.norm.weight": torch.zeros(256),
            "segmentation_head.blocks.3.norm.weight": torch.zeros(256),
        }
        _, num_blocks = convert_seg_weights(state_dict)
        assert num_blocks == 4  # max index 3 + 1

    @requires_mlx
    def test_conv_weight_transposed(self) -> None:
        """4-D conv weights are transposed from OIHW to OHWI layout."""
        from rfdetr.mlx.convert_weights import convert_seg_weights

        state_dict = {
            "segmentation_head.blocks.0.dwconv.weight": torch.ones(256, 1, 3, 3),
        }
        seg_weights, _ = convert_seg_weights(state_dict)
        assert seg_weights["blocks.0.dwconv.weight"].shape == (256, 3, 3, 1)


class TestSegDecoder:
    """Tests for RFDETRDecoder with return_intermediate=True."""

    @requires_mlx
    def test_return_intermediate_false_returns_dict(self) -> None:
        """Without return_intermediate, decoder returns only pred_logits and pred_boxes."""
        import mlx.core as mx

        from rfdetr.mlx.decoder import RFDETRDecoder

        decoder = RFDETRDecoder(
            d_model=256,
            sa_nhead=8,
            ca_nhead=16,
            ca_npoints=2,
            num_layers=2,
            num_queries=100,
            num_classes=91,
            embed_dim=384,
        )
        features = [mx.zeros((1, 26, 26, 384)) for _ in range(4)]
        out = decoder(features, return_intermediate=False)
        mx.eval(out)

        assert set(out.keys()) == {"pred_logits", "pred_boxes"}

    @requires_mlx
    def test_return_intermediate_true_adds_keys(self) -> None:
        """With return_intermediate=True, spatial_features and hs_list are included."""
        import mlx.core as mx

        from rfdetr.mlx.decoder import RFDETRDecoder

        decoder = RFDETRDecoder(
            d_model=256,
            sa_nhead=8,
            ca_nhead=16,
            ca_npoints=2,
            num_layers=2,
            num_queries=100,
            num_classes=91,
            embed_dim=384,
        )
        features = [mx.zeros((1, 26, 26, 384)) for _ in range(4)]
        out = decoder(features, return_intermediate=True)
        mx.eval(out)

        assert "spatial_features" in out
        assert "hs_list" in out
        assert out["spatial_features"].shape == (1, 26, 26, 256)

    @requires_mlx
    def test_hs_list_length_matches_num_layers(self) -> None:
        """hs_list has one entry per decoder layer."""
        import mlx.core as mx

        from rfdetr.mlx.decoder import RFDETRDecoder

        num_layers = 4
        decoder = RFDETRDecoder(
            d_model=256,
            sa_nhead=8,
            ca_nhead=16,
            ca_npoints=2,
            num_layers=num_layers,
            num_queries=100,
            num_classes=91,
            embed_dim=384,
        )
        features = [mx.zeros((1, 26, 26, 384)) for _ in range(4)]
        out = decoder(features, return_intermediate=True)
        mx.eval(out)

        assert len(out["hs_list"]) == num_layers
        for hs in out["hs_list"]:
            assert hs.shape == (1, 100, 256)


class TestSegHead:
    """Tests for the MLX SegHead module."""

    @requires_mlx
    def test_seg_head_output_shape(self) -> None:
        """SegHead returns (N, nQ, H_mask, W_mask) mask logits."""
        import mlx.core as mx

        from rfdetr.mlx.seg_head import SegHead

        seg_head = SegHead(in_dim=256, num_blocks=4)
        spatial = mx.zeros((1, 26, 26, 256))
        hs_list = [mx.zeros((1, 100, 256))] * 4
        masks = seg_head(spatial, hs_list, img_size=312, downsample_ratio=4)
        mx.eval(masks)

        assert masks.shape == (1, 100, 78, 78)

    @requires_mlx
    def test_seg_head_custom_resolution(self) -> None:
        """SegHead respects custom img_size and downsample_ratio."""
        import mlx.core as mx

        from rfdetr.mlx.seg_head import SegHead

        seg_head = SegHead(in_dim=256, num_blocks=2)
        spatial = mx.zeros((1, 24, 24, 256))
        hs_list = [mx.zeros((1, 300, 256))] * 2
        masks = seg_head(spatial, hs_list, img_size=384, downsample_ratio=4)
        mx.eval(masks)

        assert masks.shape == (1, 300, 96, 96)

    @requires_mlx
    def test_build_seg_head_from_dict(self) -> None:
        """build_seg_head loads weights from a numpy dict without errors."""
        import mlx.core as mx

        from rfdetr.mlx.seg_head import SegHead, build_seg_head

        # Build a reference head to get weight structure
        ref_head = SegHead(num_blocks=2)
        import mlx.utils

        raw = dict(mlx.utils.tree_flatten(ref_head.parameters()))
        seg_weights = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

        built = build_seg_head(seg_weights, num_blocks=2)
        assert built is not None

        spatial = mx.zeros((1, 24, 24, 256))
        hs_list = [mx.zeros((1, 100, 256))] * 2
        out = built(spatial, hs_list, img_size=312, downsample_ratio=4)
        mx.eval(out)
        assert out.shape == (1, 100, 78, 78)


class TestMLXSegInferenceModel:
    """Tests for MLXSegInferenceModel construction and postprocessing."""

    @requires_mlx
    def test_seg_postprocess_output_keys(self) -> None:
        """postprocess returns scores, labels, boxes, and masks for each image."""
        import mlx.core as mx

        from rfdetr.mlx.inference import MLXSegInferenceModel

        pred_logits = mx.zeros((1, 100, 91))
        pred_boxes = mx.full((1, 100, 4), 0.5)
        mask_logits = mx.zeros((1, 100, 78, 78))

        model = object.__new__(MLXSegInferenceModel)
        model.num_classes = 91
        model.num_select = 100

        results = model.postprocess((pred_logits, pred_boxes, mask_logits), [(480, 640)])

        assert len(results) == 1
        result = results[0]
        assert "scores" in result
        assert "labels" in result
        assert "boxes" in result
        assert "masks" in result

    @requires_mlx
    def test_seg_postprocess_mask_shape(self) -> None:
        """postprocess resizes masks to original image dimensions."""
        import mlx.core as mx

        from rfdetr.mlx.inference import MLXSegInferenceModel

        pred_logits = mx.zeros((1, 100, 91))
        pred_boxes = mx.full((1, 100, 4), 0.5)
        mask_logits = mx.zeros((1, 100, 78, 78))

        model = object.__new__(MLXSegInferenceModel)
        model.num_classes = 91
        model.num_select = 100

        orig_h, orig_w = 480, 640
        results = model.postprocess((pred_logits, pred_boxes, mask_logits), [(orig_h, orig_w)])

        masks = results[0]["masks"]
        assert masks.shape == (100, orig_h, orig_w)
        assert masks.dtype == np.float32

    @requires_mlx
    def test_seg_postprocess_mask_values_in_range(self) -> None:
        """postprocess applies sigmoid so masks are in [0, 1]."""
        import mlx.core as mx

        from rfdetr.mlx.inference import MLXSegInferenceModel

        pred_logits = mx.zeros((1, 10, 91))
        pred_boxes = mx.full((1, 10, 4), 0.5)
        mask_logits = mx.zeros((1, 10, 78, 78))

        model = object.__new__(MLXSegInferenceModel)
        model.num_classes = 91
        model.num_select = 10

        results = model.postprocess((pred_logits, pred_boxes, mask_logits), [(100, 100)])
        masks = results[0]["masks"]

        assert masks.min() >= 0.0
        assert masks.max() <= 1.0
