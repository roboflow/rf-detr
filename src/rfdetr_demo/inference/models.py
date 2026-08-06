# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Model factory helpers for video demo inference."""

from __future__ import annotations

import logging

from rfdetr.detr import RFDETR
from rfdetr_demo.inference.types import ModelSize

logger = logging.getLogger(__name__)

_MODEL_FACTORIES: dict[str, type[RFDETR]] = {}
_SEG_MODEL_FACTORIES: dict[str, type[RFDETR]] = {}


def _register_detection_models() -> None:
    if _MODEL_FACTORIES:
        return
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    _MODEL_FACTORIES.update(
        {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        },
    )


def _register_segmentation_models() -> None:
    if _SEG_MODEL_FACTORIES:
        return
    from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

    _SEG_MODEL_FACTORIES.update(
        {
            "nano": RFDETRSegNano,
            "small": RFDETRSegSmall,
            "medium": RFDETRSegMedium,
            "large": RFDETRSegLarge,
        },
    )


def build_detection_model(model_size: ModelSize, resolution: int | None = None) -> RFDETR:
    """Instantiate a detection model.

    Args:
        model_size: One of nano/small/medium/large.
        resolution: Optional square input resolution (higher improves recall on
            small/distant people). ``None`` uses the model default.

    Returns:
        The detection model.
    """
    _register_detection_models()
    factory = _MODEL_FACTORIES.get(model_size)
    if factory is None:
        supported = ", ".join(sorted(_MODEL_FACTORIES))
        msg = f"Unsupported model size {model_size!r}. Choose from: {supported}"
        raise ValueError(msg)
    if resolution is not None:
        logger.info("Loading RF-DETR detection model: %s at resolution=%d", model_size, resolution)
        return factory(resolution=resolution)
    logger.info("Loading RF-DETR detection model: %s", model_size)
    return factory()


def build_segmentation_model(model_size: ModelSize) -> RFDETR:
    """Instantiate a segmentation model."""
    _register_segmentation_models()
    factory = _SEG_MODEL_FACTORIES.get(model_size)
    if factory is None:
        supported = ", ".join(sorted(_SEG_MODEL_FACTORIES))
        msg = f"Unsupported segmentation model size {model_size!r}. Choose from: {supported}"
        raise ValueError(msg)
    logger.info("Loading RF-DETR segmentation model: %s", model_size)
    return factory()


def build_keypoint_model(resolution: int | None = None) -> RFDETR:
    """Instantiate the keypoint preview model.

    Args:
        resolution: Optional square input resolution (must be divisible by the
            model patch size). A higher value improves recall on small/distant
            people at the cost of speed. ``None`` uses the model default.

    Returns:
        The keypoint preview model.
    """
    from rfdetr import RFDETRKeypointPreview

    if resolution is not None:
        logger.info("Loading RF-DETR keypoint preview model at resolution=%d", resolution)
        return RFDETRKeypointPreview(resolution=resolution)
    logger.info("Loading RF-DETR keypoint preview model")
    return RFDETRKeypointPreview()
