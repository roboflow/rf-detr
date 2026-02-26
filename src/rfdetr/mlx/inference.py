# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Compiled MLX inference pipeline for RF-DETR.

Builds a FP16 compiled forward pass that includes GPU-side preprocessing
(uint8 -> float16 + ImageNet normalization). Postprocessing converts MLX
outputs to numpy arrays matching the PyTorch PostProcess output format.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

from rfdetr.mlx.backbone import DINOv2Backbone, interpolate_pos_embed
from rfdetr.mlx.convert_weights import convert_state_dict
from rfdetr.mlx.decoder import RFDETRDecoder
from rfdetr.util.logger import get_logger

logger = get_logger()

# Backbone config derived from encoder name
_ENCODER_CONFIGS = {
    "dinov2_windowed_small": {"embed_dim": 384, "num_heads": 6, "depth": 12},
    "dinov2_windowed_base": {"embed_dim": 768, "num_heads": 12, "depth": 12},
}


class MLXInferenceModel:
    """Compiled MLX inference model for RF-DETR.

    Encapsulates the backbone, decoder, and compiled forward pass.
    Handles weight conversion, FP16 casting, and postprocessing.

    Args:
        backbone: MLX DINOv2 backbone.
        decoder: MLX RF-DETR decoder.
        resolution: Model input resolution (square).
        num_classes: Number of output classes (including background).
        num_select: Number of top predictions to return.
    """

    def __init__(
        self,
        backbone: DINOv2Backbone,
        decoder: RFDETRDecoder,
        resolution: int,
        num_classes: int,
        num_select: int = 300,
    ) -> None:
        self.backbone = backbone
        self.decoder = decoder
        self.resolution = resolution
        self.num_classes = num_classes
        self.num_select = num_select

        # ImageNet normalization constants for GPU-side preprocessing
        mean = mx.array([0.485, 0.456, 0.406], dtype=mx.float16).reshape(1, 1, 1, 3)
        std = mx.array([0.229, 0.224, 0.225], dtype=mx.float16).reshape(1, 1, 1, 3)

        def _forward(x_uint8: mx.array) -> Dict[str, mx.array]:
            """Compiled forward: uint8 NHWC RGB -> predictions."""
            x = x_uint8.astype(mx.float16) * (1.0 / 255.0)
            x = (x - mean) / std
            features = backbone(x)
            return decoder(features)

        self._compiled_forward = mx.compile(_forward)

        # Warm up the compiled graph
        dummy = mx.zeros((1, resolution, resolution, 3), dtype=mx.uint8)
        result = self._compiled_forward(dummy)
        mx.eval(result)
        logger.debug("MLX inference model compiled and warmed up")

    @classmethod
    def from_pytorch(
        cls,
        model_config: object,
        pytorch_model: object,
    ) -> "MLXInferenceModel":
        """Build MLX inference model from PyTorch model and config.

        Args:
            model_config: RF-DETR model configuration instance.
            pytorch_model: The rfdetr.main.Model instance.

        Returns:
            Compiled MLX inference model.
        """
        encoder_name = model_config.encoder
        if encoder_name not in _ENCODER_CONFIGS:
            raise ValueError(
                f"Unsupported encoder '{encoder_name}' for MLX backend. "
                f"Supported: {list(_ENCODER_CONFIGS.keys())}"
            )

        enc_cfg = _ENCODER_CONFIGS[encoder_name]
        embed_dim = enc_cfg["embed_dim"]
        num_heads = enc_cfg["num_heads"]
        depth = enc_cfg["depth"]

        patch_size = model_config.patch_size
        num_windows = model_config.num_windows
        resolution = model_config.resolution
        hidden_dim = model_config.hidden_dim
        sa_nheads = model_config.sa_nheads
        ca_nheads = model_config.ca_nheads
        dec_n_points = model_config.dec_n_points
        dec_layers = model_config.dec_layers
        num_classes = model_config.num_classes + 1  # +1 for background
        num_queries = getattr(model_config, "num_queries", 300)
        num_select = getattr(model_config, "num_select", 300)
        group_detr = model_config.group_detr

        # Convert 1-indexed out_feature_indexes to 0-indexed
        feature_indices = [idx - 1 for idx in model_config.out_feature_indexes]

        logger.info(
            f"Building MLX model: {encoder_name}, patch_size={patch_size}, "
            f"num_windows={num_windows}, resolution={resolution}, "
            f"dec_layers={dec_layers}, hidden_dim={hidden_dim}"
        )

        backbone = DINOv2Backbone(
            img_size=resolution,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_windows=num_windows,
            feature_indices=feature_indices,
        )

        decoder = RFDETRDecoder(
            d_model=hidden_dim,
            sa_nhead=sa_nheads,
            ca_nhead=ca_nheads,
            ca_npoints=dec_n_points,
            num_layers=dec_layers,
            num_queries=num_queries,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_features=len(feature_indices),
            group_detr=group_detr,
        )

        state_dict = pytorch_model.model.state_dict()
        backbone_weights, decoder_weights = convert_state_dict(state_dict)

        # Interpolate positional embedding if resolution differs
        if "pos_embed" in backbone_weights:
            target_patches = backbone.patch_embed.num_patches
            pos = backbone_weights["pos_embed"]
            if pos.shape[1] != 1 + target_patches:
                logger.info(
                    f"Interpolating pos_embed: {pos.shape} -> target {target_patches} patches"
                )
                backbone_weights["pos_embed"] = interpolate_pos_embed(pos, target_patches)

        backbone.load_weights([(k, mx.array(v)) for k, v in backbone_weights.items()])
        decoder.load_weights([(k, mx.array(v)) for k, v in decoder_weights.items()], strict=False)

        logger.info(
            f"Loaded {len(backbone_weights)} backbone + {len(decoder_weights)} decoder weights"
        )

        _cast_to_fp16(backbone)
        _cast_to_fp16(decoder)

        return cls(backbone, decoder, resolution, num_classes, num_select)

    def forward(self, x_uint8: mx.array) -> Dict[str, mx.array]:
        """Run compiled forward pass.

        Args:
            x_uint8: (N, H, W, 3) uint8 RGB image tensor.

        Returns:
            Dict with "pred_logits" and "pred_boxes".
        """
        output = self._compiled_forward(x_uint8)
        mx.eval(output)
        return output

    def postprocess(
        self,
        outputs: Dict[str, mx.array],
        orig_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, np.ndarray]]:
        """Postprocess MLX outputs to match PyTorch PostProcess format.

        Args:
            outputs: Dict with "pred_logits" (N, nQ, num_classes) and
                "pred_boxes" (N, nQ, 4) in cxcywh format.
            orig_sizes: List of (height, width) for each image.

        Returns:
            List of dicts with "scores", "labels", "boxes" as numpy arrays.
        """
        logits = np.array(outputs["pred_logits"], dtype=np.float32)
        boxes = np.array(outputs["pred_boxes"], dtype=np.float32)

        prob = 1.0 / (1.0 + np.exp(-logits))

        batch_size = logits.shape[0]
        results = []

        for i in range(batch_size):
            prob_i = prob[i]  # (nQ, num_classes)
            boxes_i = boxes[i]  # (nQ, 4) cxcywh

            # Flatten and select top-K
            flat = prob_i.reshape(-1)
            num_select = min(self.num_select, flat.shape[0])
            topk_idx = np.argpartition(-flat, num_select)[:num_select]
            topk_values = flat[topk_idx]

            sort_order = np.argsort(-topk_values)
            topk_idx = topk_idx[sort_order]
            scores = topk_values[sort_order]

            # Map flat indices to query and class indices
            topk_boxes_idx = topk_idx // prob_i.shape[1]
            labels = topk_idx % prob_i.shape[1]

            # Convert cxcywh -> xyxy
            sel_boxes = boxes_i[topk_boxes_idx]
            cx, cy, w, h = sel_boxes[:, 0], sel_boxes[:, 1], sel_boxes[:, 2], sel_boxes[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            xyxy = np.stack([x1, y1, x2, y2], axis=-1)

            orig_h, orig_w = orig_sizes[i]
            scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
            xyxy = xyxy * scale

            results.append(
                {
                    "scores": scores,
                    "labels": labels,
                    "boxes": xyxy,
                }
            )

        return results


def _cast_to_fp16(model: nn.Module) -> None:
    """Cast all model parameters to float16.

    Args:
        model: MLX module to cast.
    """
    params = [(k, v.astype(mx.float16)) for k, v in mlx.utils.tree_flatten(model.parameters())]
    model.load_weights(params)
