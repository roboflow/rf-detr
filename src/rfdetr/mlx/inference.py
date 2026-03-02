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
from rfdetr.mlx.convert_weights import convert_seg_weights, convert_state_dict
from rfdetr.mlx.decoder import RFDETRDecoder
from rfdetr.mlx.seg_head import SegHead, build_seg_head
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
                f"Unsupported encoder '{encoder_name}' for MLX backend. Supported: {list(_ENCODER_CONFIGS.keys())}"
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
                logger.info(f"Interpolating pos_embed: {pos.shape} -> target {target_patches} patches")
                backbone_weights["pos_embed"] = interpolate_pos_embed(pos, target_patches)

        backbone.load_weights([(k, mx.array(v)) for k, v in backbone_weights.items()])
        decoder.load_weights([(k, mx.array(v)) for k, v in decoder_weights.items()], strict=False)

        logger.info(f"Loaded {len(backbone_weights)} backbone + {len(decoder_weights)} decoder weights")

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
        logits = np.clip(np.array(outputs["pred_logits"], dtype=np.float32), -88.0, 88.0)
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


class MLXSegInferenceModel:
    """Compiled MLX inference model for RF-DETR segmentation.

    Extends the detection pipeline with a native MLX segmentation head that
    produces per-query mask logits at ``resolution // downsample_ratio`` resolution,
    resized to the original image size during postprocessing.

    Args:
        backbone: MLX DINOv2 backbone.
        decoder: MLX RF-DETR decoder.
        seg_head: MLX segmentation head.
        resolution: Model input resolution (square).
        num_classes: Number of output classes (including background).
        downsample_ratio: Mask spatial resolution = resolution // downsample_ratio.
        num_select: Number of top predictions to return.
    """

    def __init__(
        self,
        backbone: DINOv2Backbone,
        decoder: RFDETRDecoder,
        seg_head: SegHead,
        resolution: int,
        num_classes: int,
        downsample_ratio: int = 4,
        num_select: int = 100,
    ) -> None:
        self.backbone = backbone
        self.decoder = decoder
        self.seg_head = seg_head
        self.resolution = resolution
        self.num_classes = num_classes
        self.downsample_ratio = downsample_ratio
        self.num_select = num_select

        # ImageNet normalization constants for GPU-side preprocessing
        mean = mx.array([0.485, 0.456, 0.406], dtype=mx.float16).reshape(1, 1, 1, 3)
        std = mx.array([0.229, 0.224, 0.225], dtype=mx.float16).reshape(1, 1, 1, 3)
        _resolution = resolution
        _downsample_ratio = downsample_ratio

        def _forward(x_uint8: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
            """Compiled forward: uint8 NHWC RGB -> (logits, boxes, masks)."""
            x = x_uint8.astype(mx.float16) * (1.0 / 255.0)
            x = (x - mean) / std
            features = backbone(x)
            out = decoder(features, return_intermediate=True)
            masks = seg_head(
                out["spatial_features"],
                out["hs_list"],
                img_size=_resolution,
                downsample_ratio=_downsample_ratio,
            )
            return out["pred_logits"], out["pred_boxes"], masks

        self._compiled_forward = mx.compile(_forward)

        # Warm up the compiled graph
        dummy = mx.zeros((1, resolution, resolution, 3), dtype=mx.uint8)
        logits, boxes, masks = self._compiled_forward(dummy)
        mx.eval(logits, boxes, masks)
        logger.debug("MLX seg inference model compiled and warmed up")

    @classmethod
    def from_pytorch(
        cls,
        model_config: object,
        pytorch_model: object,
    ) -> "MLXSegInferenceModel":
        """Build MLX seg inference model from a PyTorch RF-DETR seg model.

        Args:
            model_config: RF-DETR segmentation model configuration instance.
            pytorch_model: The rfdetr.main.Model instance with loaded weights.

        Returns:
            Compiled MLX segmentation inference model.

        Raises:
            ValueError: If the encoder name is not recognised.
        """
        encoder_name = model_config.encoder
        if encoder_name not in _ENCODER_CONFIGS:
            raise ValueError(
                f"Unsupported encoder '{encoder_name}' for MLX backend. Supported: {list(_ENCODER_CONFIGS.keys())}"
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
        num_queries = getattr(model_config, "num_queries", 100)
        num_select = getattr(model_config, "num_select", 100)
        group_detr = model_config.group_detr
        downsample_ratio = getattr(model_config, "mask_downsample_ratio", 4)

        # Convert 1-indexed out_feature_indexes to 0-indexed
        feature_indices = [idx - 1 for idx in model_config.out_feature_indexes]

        logger.info(
            f"Building MLX seg model: {encoder_name}, patch_size={patch_size}, "
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
        seg_weights, num_blocks = convert_seg_weights(state_dict)

        # Interpolate positional embedding if resolution differs
        if "pos_embed" in backbone_weights:
            target_patches = backbone.patch_embed.num_patches
            pos = backbone_weights["pos_embed"]
            if pos.shape[1] != 1 + target_patches:
                logger.info(f"Interpolating pos_embed: {pos.shape} -> target {target_patches} patches")
                backbone_weights["pos_embed"] = interpolate_pos_embed(pos, target_patches)

        backbone.load_weights([(k, mx.array(v)) for k, v in backbone_weights.items()])
        decoder.load_weights([(k, mx.array(v)) for k, v in decoder_weights.items()], strict=False)

        seg_head = build_seg_head(seg_weights, num_blocks)

        logger.info(
            f"Loaded {len(backbone_weights)} backbone + {len(decoder_weights)} decoder + "
            f"{len(seg_weights)} seg_head weights ({num_blocks} blocks)"
        )

        _cast_to_fp16(backbone)
        _cast_to_fp16(decoder)
        _cast_to_fp16(seg_head)

        return cls(backbone, decoder, seg_head, resolution, num_classes, downsample_ratio, num_select)

    def forward(self, x_uint8: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Run compiled forward pass.

        Args:
            x_uint8: (N, H, W, 3) uint8 RGB image tensor.

        Returns:
            3-tuple of (pred_logits, pred_boxes, mask_logits) as MLX arrays.
        """
        logits, boxes, masks = self._compiled_forward(x_uint8)
        mx.eval(logits, boxes, masks)
        return logits, boxes, masks

    def postprocess(
        self,
        outputs: Tuple[mx.array, mx.array, mx.array],
        orig_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, np.ndarray]]:
        """Postprocess MLX outputs to supervision-compatible format.

        Args:
            outputs: 3-tuple ``(pred_logits, pred_boxes, mask_logits)`` from
                ``forward()``.
            orig_sizes: List of ``(height, width)`` for each image in the batch.

        Returns:
            List of dicts with ``"scores"``, ``"labels"``, ``"boxes"`` and
            ``"masks"`` as numpy arrays.  ``"masks"`` has shape
            ``(num_select, orig_h, orig_w)`` and contains float32 sigmoid
            probabilities.
        """
        import cv2

        pred_logits, pred_boxes, mask_logits = outputs
        logits = np.clip(np.array(pred_logits, dtype=np.float32), -88.0, 88.0)
        boxes = np.array(pred_boxes, dtype=np.float32)
        masks_np = np.array(mask_logits, dtype=np.float32)

        prob = 1.0 / (1.0 + np.exp(-logits))

        batch_size = logits.shape[0]
        results = []

        for i in range(batch_size):
            prob_i = prob[i]  # (nQ, num_classes)
            boxes_i = boxes[i]  # (nQ, 4) cxcywh
            masks_i = masks_np[i]  # (nQ, H_mask, W_mask)

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

            # Select and resize masks
            sel_masks_logits = masks_i[topk_boxes_idx]  # (num_select, H_mask, W_mask)
            sel_masks_logits = np.clip(sel_masks_logits, -88.0, 88.0)
            sel_masks_prob = 1.0 / (1.0 + np.exp(-sel_masks_logits))  # sigmoid

            resized_masks = np.zeros((num_select, orig_h, orig_w), dtype=np.float32)
            for j in range(num_select):
                resized_masks[j] = cv2.resize(
                    sel_masks_prob[j],
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            results.append(
                {
                    "scores": scores,
                    "labels": labels,
                    "boxes": xyxy,
                    "masks": resized_masks,
                }
            )

        return results
