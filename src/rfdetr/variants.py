# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Concrete RF-DETR model variant classes.

All classes inherit from :class:`~rfdetr.detr.RFDETR` which remains defined in
``rfdetr.detr``. Backward-compatible access from ``rfdetr.detr`` is provided
via lazy ``__getattr__`` re-exports, so importing ``rfdetr.variants`` no longer
depends on a fragile eager ``detr -> variants`` import sequence.
"""

from __future__ import annotations

__all__ = [
    "RFDETRBase",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRSeg",
    "RFDETRSegPreview",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]

import warnings

from rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRLargeDeprecatedConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
)
from rfdetr.detr import RFDETR
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """

    size = "rfdetr-base"
    _model_config_class = RFDETRBaseConfig


class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """

    size = "rfdetr-nano"
    _model_config_class = RFDETRNanoConfig


class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """

    size = "rfdetr-small"
    _model_config_class = RFDETRSmallConfig


class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """

    size = "rfdetr-medium"
    _model_config_class = RFDETRMediumConfig


class RFDETRLargeDeprecated(RFDETR):
    """
    Train an RF-DETR Large model.
    """

    size = "rfdetr-large"
    _model_config_class = RFDETRLargeDeprecatedConfig

    def __init__(self, **kwargs):
        warnings.warn(
            "RFDETRLargeDeprecated is deprecated and will be removed in a future version."
            " Please use RFDETRLarge instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class RFDETRLarge(RFDETR):
    size = "rfdetr-large"

    @staticmethod
    def _should_fallback_to_deprecated_config(exc: Exception) -> bool:
        """Return whether initialization should retry with deprecated Large config.

        The fallback is only for known checkpoint/config incompatibilities from
        deprecated Large weights. Runtime issues such as CUDA OOM must fail
        fast and must not trigger a second initialization attempt.

        Args:
            exc: Exception raised by initial ``RFDETR`` initialization.

        Returns:
            ``True`` when retrying with deprecated config is expected to help.
        """
        message = str(exc).lower()
        if "out of memory" in message:
            return False
        if isinstance(exc, ValueError):
            return "patch_size" in message
        if isinstance(exc, RuntimeError):
            incompatible_state_dict_markers = (
                "error(s) in loading state_dict",
                "size mismatch",
                "missing key(s) in state_dict",
                "unexpected key(s) in state_dict",
            )
            return any(marker in message for marker in incompatible_state_dict_markers)
        return False

    def __init__(self, **kwargs):
        self.init_error = None
        self.is_deprecated = False
        try:
            super().__init__(**kwargs)
        except (ValueError, RuntimeError) as exc:
            if not self._should_fallback_to_deprecated_config(exc):
                raise
            self.init_error = exc
            self.is_deprecated = True
            try:
                super().__init__(**kwargs)
                logger.warning(
                    "\n"
                    "=" * 100 + "\n"
                    "WARNING: Automatically switched to deprecated model configuration,"
                    " due to using deprecated weights."
                    " This will be removed in a future version.\n"
                    " Please retrain your model with the new weights and configuration.\n"
                    "=" * 100 + "\n"
                )
            except Exception:
                raise self.init_error

    def get_model_config(self, **kwargs) -> ModelConfig:
        if not self.is_deprecated:
            return RFDETRLargeConfig(**kwargs)
        else:
            return RFDETRLargeDeprecatedConfig(**kwargs)


class RFDETRSeg(RFDETR):
    """Base class for all RF-DETR segmentation models."""

    _train_config_class = SegmentationTrainConfig


class RFDETRSegPreview(RFDETRSeg):
    size = "rfdetr-seg-preview"
    _model_config_class = RFDETRSegPreviewConfig


class RFDETRSegNano(RFDETRSeg):
    size = "rfdetr-seg-nano"
    _model_config_class = RFDETRSegNanoConfig


class RFDETRSegSmall(RFDETRSeg):
    size = "rfdetr-seg-small"
    _model_config_class = RFDETRSegSmallConfig


class RFDETRSegMedium(RFDETRSeg):
    size = "rfdetr-seg-medium"
    _model_config_class = RFDETRSegMediumConfig


class RFDETRSegLarge(RFDETRSeg):
    size = "rfdetr-seg-large"
    _model_config_class = RFDETRSegLargeConfig


class RFDETRSegXLarge(RFDETRSeg):
    size = "rfdetr-seg-xlarge"
    _model_config_class = RFDETRSegXLargeConfig


class RFDETRSeg2XLarge(RFDETRSeg):
    size = "rfdetr-seg-2xlarge"
    _model_config_class = RFDETRSeg2XLargeConfig
