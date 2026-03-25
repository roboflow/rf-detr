# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for MLX inference backend.

These tests require macOS with Apple Silicon and MLX installed.
They are skipped on other platforms via the @requires_mlx decorator.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
import torch

from rfdetr.mlx import is_mlx_available

pytestmark = pytest.mark.mlx

requires_mlx = pytest.mark.skipif(
    not is_mlx_available(),
    reason="MLX not available (requires macOS with Apple Silicon)",
)


class TestIsMLXAvailable:
    """Tests for the MLX availability check."""

    def test_returns_bool(self) -> None:
        """Test that is_mlx_available returns a bool."""
        result = is_mlx_available()
        assert isinstance(result, bool)

    @patch.dict(sys.modules, {"mlx.core": None, "mlx": None})
    def test_returns_false_when_mlx_not_installed(self) -> None:
        """Test that is_mlx_available returns False when mlx is not importable."""
        # Reimport to pick up the mocked modules
        from rfdetr.mlx import is_mlx_available as check

        # When mlx is not importable, should return False
        assert check() is False or sys.platform != "darwin"


class TestConvertWeights:
    """Tests for weight conversion from PyTorch to MLX format."""

    @requires_mlx
    def test_backbone_key_remapping(self) -> None:
        """Test that patch embed, cls token, and norm keys remap correctly."""
        from rfdetr.mlx.convert_weights import _remap_backbone_key

        # Patch embedding
        result = _remap_backbone_key("backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight")
        assert result == "backbone.patch_embed.proj.weight"

        # CLS token
        result = _remap_backbone_key("backbone.0.encoder.encoder.embeddings.cls_token")
        assert result == "backbone.cls_token"

        # Position embeddings
        result = _remap_backbone_key("backbone.0.encoder.encoder.embeddings.position_embeddings")
        assert result == "backbone.pos_embed"

        # Layer norm
        result = _remap_backbone_key("backbone.0.encoder.encoder.layernorm.weight")
        assert result == "backbone.norm.weight"

    @requires_mlx
    def test_layer_key_remapping(self) -> None:
        """Test that per-layer attention, scale, and MLP keys remap correctly."""
        from rfdetr.mlx.convert_weights import _remap_backbone_key

        # Attention query weight
        result = _remap_backbone_key("backbone.0.encoder.encoder.encoder.layer.0.attention.attention.query.weight")
        assert result == "backbone.blocks.0.attn.q.weight"

        # Layer scale
        result = _remap_backbone_key("backbone.0.encoder.encoder.encoder.layer.5.layer_scale1.lambda1")
        assert result == "backbone.blocks.5.ls1.gamma"

        # MLP
        result = _remap_backbone_key("backbone.0.encoder.encoder.encoder.layer.11.mlp.fc2.bias")
        assert result == "backbone.blocks.11.mlp.fc2.bias"

    @requires_mlx
    def test_decoder_key_remapping(self) -> None:
        """Test that decoder query, refpoint, class, and bbox keys remap correctly."""
        from rfdetr.mlx.convert_weights import _map_decoder_weight_key

        assert _map_decoder_weight_key("query_feat.weight") == "query_feat"
        assert _map_decoder_weight_key("refpoint_embed.weight") == "refpoint_embed"
        assert _map_decoder_weight_key("class_embed.weight") == "class_embed.weight"
        assert _map_decoder_weight_key("bbox_embed.layers.0.weight") == "bbox_embed_layers.0.weight"

    @requires_mlx
    def test_conv_weight_transposition(self) -> None:
        """Test that conv weights are transposed from PyTorch OIHW to MLX OHWI layout."""
        from rfdetr.mlx.convert_weights import _transpose_conv_weight

        # PyTorch: [out, in, kH, kW] -> MLX: [out, kH, kW, in]
        w = np.random.randn(64, 3, 16, 16).astype(np.float32)
        result = _transpose_conv_weight(w)
        assert result.shape == (64, 16, 16, 3)

    @requires_mlx
    def test_convert_state_dict_splits_in_proj(self) -> None:
        """Verify that self_attn in_proj_weight is split into q/k/v projections."""
        from rfdetr.mlx.convert_weights import convert_state_dict

        # Create minimal state dict with in_proj_weight
        d_model = 256
        state_dict = {
            "decoder.decoder.layers.0.self_attn.in_proj_weight": torch.randn(3 * d_model, d_model),
            "decoder.decoder.layers.0.self_attn.in_proj_bias": torch.randn(3 * d_model),
        }
        _, decoder_weights = convert_state_dict(state_dict)

        assert "decoder.layers.0.self_attn.query_proj.weight" in decoder_weights
        assert "decoder.layers.0.self_attn.key_proj.weight" in decoder_weights
        assert "decoder.layers.0.self_attn.value_proj.weight" in decoder_weights
        assert decoder_weights["decoder.layers.0.self_attn.query_proj.weight"].shape == (d_model, d_model)


class TestBackbone:
    """Tests for the MLX DINOv2 backbone."""

    @requires_mlx
    def test_backbone_output_shapes_nano(self) -> None:
        """Test that the Nano backbone returns 4 features of shape (1, 24, 24, 384)."""
        import mlx.core as mx

        from rfdetr.mlx.backbone import DINOv2Backbone

        backbone = DINOv2Backbone(
            img_size=384,
            patch_size=16,
            embed_dim=384,
            num_heads=6,
            num_windows=2,
            feature_indices=[2, 5, 8, 11],
        )
        x = mx.zeros((1, 384, 384, 3))
        features = backbone(x)

        assert len(features) == 4
        # 384/16 = 24 patches per side
        for feat in features:
            assert feat.shape == (1, 24, 24, 384)

    @requires_mlx
    def test_backbone_output_shapes_base(self) -> None:
        """Test that the Base backbone returns 4 features of shape (1, 40, 40, 384)."""
        import mlx.core as mx

        from rfdetr.mlx.backbone import DINOv2Backbone

        backbone = DINOv2Backbone(
            img_size=560,
            patch_size=14,
            embed_dim=384,
            num_heads=6,
            num_windows=4,
            feature_indices=[1, 4, 7, 10],
        )
        x = mx.zeros((1, 560, 560, 3))
        features = backbone(x)

        assert len(features) == 4
        # 560/14 = 40 patches per side
        for feat in features:
            assert feat.shape == (1, 40, 40, 384)


class TestDecoder:
    """Tests for the MLX RF-DETR decoder."""

    @requires_mlx
    def test_decoder_output_shapes(self) -> None:
        """Test that the decoder returns pred_logits (1, 300, 91) and pred_boxes (1, 300, 4)."""
        import mlx.core as mx

        from rfdetr.mlx.decoder import RFDETRDecoder

        decoder = RFDETRDecoder(
            d_model=256,
            sa_nhead=8,
            ca_nhead=16,
            ca_npoints=2,
            num_layers=2,
            num_queries=300,
            num_classes=91,
            embed_dim=384,
        )

        # Simulate 4 backbone features at 24x24
        features = [mx.zeros((1, 24, 24, 384)) for _ in range(4)]
        output = decoder(features)
        mx.eval(output)

        assert "pred_logits" in output
        assert "pred_boxes" in output
        assert output["pred_logits"].shape == (1, 300, 91)
        assert output["pred_boxes"].shape == (1, 300, 4)


class TestPosEmbedInterpolation:
    """Tests for positional embedding interpolation."""

    @requires_mlx
    def test_no_interpolation_needed(self) -> None:
        """Test that pos_embed is returned unchanged when target matches stored size."""
        from rfdetr.mlx.backbone import interpolate_pos_embed

        pos = np.random.randn(1, 577, 384).astype(np.float32)  # 1 + 576 patches (24x24)
        result = interpolate_pos_embed(pos, 576)
        assert result.shape == (1, 577, 384)
        np.testing.assert_array_equal(result, pos)

    @requires_mlx
    def test_interpolation_changes_size(self) -> None:
        """Test that pos_embed is bicubic-interpolated to the target patch count."""
        from rfdetr.mlx.backbone import interpolate_pos_embed

        # Start with 384px (24x24 = 576 patches), interpolate to 640px (40x40 = 1600)
        pos = np.random.randn(1, 577, 384).astype(np.float32)
        result = interpolate_pos_embed(pos, 1600)
        assert result.shape == (1, 1601, 384)
        # CLS token should be unchanged
        np.testing.assert_array_equal(result[:, :1, :], pos[:, :1, :])


class TestInferencePostprocess:
    """Tests for MLX inference postprocessing."""

    @requires_mlx
    def test_postprocess_output_format(self) -> None:
        """Test that postprocess returns scores, labels, and boxes for each image."""
        import mlx.core as mx

        from rfdetr.mlx.inference import MLXInferenceModel

        # Create mock outputs
        pred_logits = mx.zeros((1, 300, 91))
        pred_boxes = mx.full((1, 300, 4), 0.5)
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        # Create a minimal model instance for postprocessing
        model = object.__new__(MLXInferenceModel)
        model.num_classes = 91
        model.num_select = 300

        results = model.postprocess(outputs, [(480, 640)])

        assert len(results) == 1
        assert "scores" in results[0]
        assert "labels" in results[0]
        assert "boxes" in results[0]
        assert results[0]["scores"].shape[0] == 300
        assert results[0]["boxes"].shape == (300, 4)

    @requires_mlx
    def test_postprocess_box_scaling(self) -> None:
        """Test that postprocess scales boxes from normalised cxcywh to pixel xyxy."""
        import mlx.core as mx

        from rfdetr.mlx.inference import MLXInferenceModel

        # Create outputs with a known box at center
        pred_logits = mx.full((1, 1, 91), 5.0)  # high confidence
        pred_boxes = mx.array([[[0.5, 0.5, 0.2, 0.2]]])  # center box, 20% size
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        model = object.__new__(MLXInferenceModel)
        model.num_classes = 91
        model.num_select = 1

        results = model.postprocess(outputs, [(100, 200)])  # 100h x 200w

        boxes = results[0]["boxes"]
        # cxcywh (0.5, 0.5, 0.2, 0.2) -> xyxy should be scaled by (200, 100, 200, 100)
        # x1 = (0.5-0.1)*200 = 80, y1 = (0.5-0.1)*100 = 40
        # x2 = (0.5+0.1)*200 = 120, y2 = (0.5+0.1)*100 = 60
        np.testing.assert_allclose(boxes[0], [80.0, 40.0, 120.0, 60.0], atol=0.1)


class TestDetrMLXIntegration:
    """Tests for the RFDETR.optimize_for_inference(backend='mlx') integration."""

    def test_optimize_for_inference_raises_without_mlx(self) -> None:
        """Test that MLX backend raises clearly when MLX is not available."""
        if is_mlx_available():
            pytest.skip("MLX is available, cannot test unavailability error")

        from rfdetr.detr import RFDETR

        # Create a mock RFDETR that doesn't download weights
        class _MockRFDETR(RFDETR):
            def maybe_download_pretrain_weights(self) -> None:
                return None

            def get_model_config(self, **kwargs) -> SimpleNamespace:
                return SimpleNamespace()

            def get_model(self, config: SimpleNamespace) -> Any:
                mock = MagicMock()
                mock.inference_model = None
                mock.resolution = 384
                return mock

        model = _MockRFDETR()
        with pytest.raises(RuntimeError, match="MLX is not available"):
            model.optimize_for_inference(backend="mlx")

    @requires_mlx
    def test_predict_mlx_returns_detections(self) -> None:
        """Test that MLX predict returns sv.Detections for a dummy image."""
        from rfdetr.detr import RFDETR

        class _MockRFDETR(RFDETR):
            def maybe_download_pretrain_weights(self) -> None:
                return None

            def get_model_config(self, **kwargs) -> SimpleNamespace:
                return SimpleNamespace()

            def get_model(self, config: SimpleNamespace) -> Any:
                mock = MagicMock()
                mock.inference_model = None
                mock.resolution = 384
                return mock

        model = _MockRFDETR()

        # Set up a mock MLX model
        mock_mlx = MagicMock()
        mock_mlx.resolution = 384
        mock_mlx.forward.return_value = {
            "pred_logits": __import__("mlx.core", fromlist=["core"]).zeros((1, 300, 91)),
            "pred_boxes": __import__("mlx.core", fromlist=["core"]).full((1, 300, 4), 0.5),
        }
        mock_mlx.postprocess.return_value = [
            {
                "scores": np.array([0.9, 0.1]),
                "labels": np.array([1, 2]),
                "boxes": np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32),
            }
        ]

        model._mlx_model = mock_mlx
        model._inference_backend = "mlx"
        model._is_optimized_for_inference = True

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = model.predict(dummy_image, threshold=0.5)

        assert isinstance(detections, sv.Detections)
        assert detections.xyxy.shape[1] == 4

    @requires_mlx
    def test_optimize_for_inference_routes_seg_to_mlx_seg_model(self) -> None:
        """Test that segmentation models route to MLXSegInferenceModel, not MLXInferenceModel."""
        from unittest.mock import patch

        from rfdetr.detr import RFDETR

        class _MockSegRFDETR(RFDETR):
            def maybe_download_pretrain_weights(self) -> None:
                return None

            def get_model_config(self, **kwargs) -> SimpleNamespace:
                return SimpleNamespace(segmentation_head=True)

            def get_model(self, config: SimpleNamespace) -> Any:
                mock = MagicMock()
                mock.inference_model = None
                mock.resolution = 312
                return mock

        model = _MockSegRFDETR()

        with (
            patch("rfdetr.mlx.build_mlx_seg_inference") as mock_seg,
            patch("rfdetr.mlx.build_mlx_inference") as mock_det,
        ):
            mock_seg.return_value = MagicMock()
            model.optimize_for_inference(backend="mlx")

        mock_seg.assert_called_once()
        mock_det.assert_not_called()

    @requires_mlx
    def test_optimize_for_inference_routes_det_to_mlx_inference_model(self) -> None:
        """Test that detection models route to MLXInferenceModel, not MLXSegInferenceModel."""
        from unittest.mock import patch

        from rfdetr.detr import RFDETR

        class _MockDetRFDETR(RFDETR):
            def maybe_download_pretrain_weights(self) -> None:
                return None

            def get_model_config(self, **kwargs) -> SimpleNamespace:
                return SimpleNamespace(segmentation_head=False)

            def get_model(self, config: SimpleNamespace) -> Any:
                mock = MagicMock()
                mock.inference_model = None
                mock.resolution = 384
                return mock

        model = _MockDetRFDETR()

        with (
            patch("rfdetr.mlx.build_mlx_inference") as mock_det,
            patch("rfdetr.mlx.build_mlx_seg_inference") as mock_seg,
        ):
            mock_det.return_value = MagicMock()
            model.optimize_for_inference(backend="mlx")

        mock_det.assert_called_once()
        mock_seg.assert_not_called()
