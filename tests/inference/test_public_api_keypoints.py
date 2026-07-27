# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public API tests for the keypoint preview variant."""

from rfdetr import RFDETRKeypointPreview
from rfdetr.config import KeypointTrainConfig, RFDETRKeypointPreviewConfig
from rfdetr.detr import RFDETRKeypointPreview as RFDETRKeypointPreviewFromDetr
from rfdetr.variants import RFDETRKeypointPreview as RFDETRKeypointPreviewFromVariants


def test_keypoint_preview_top_level_import() -> None:
    """RFDETRKeypointPreview must be importable from top-level package and keep shared identity."""
    assert RFDETRKeypointPreview is RFDETRKeypointPreviewFromVariants
    assert RFDETRKeypointPreview is RFDETRKeypointPreviewFromDetr


def test_keypoint_preview_variant_metadata() -> None:
    """RFDETRKeypointPreview exposes the expected variant metadata and config class."""
    assert RFDETRKeypointPreview.size == "rfdetr-keypoint-preview"
    assert RFDETRKeypointPreview._model_config_class is RFDETRKeypointPreviewConfig
    assert RFDETRKeypointPreview._train_config_class is KeypointTrainConfig
    assert RFDETRKeypointPreviewConfig.model_fields["pretrain_weights"].default == "rf-detr-keypoint-preview-xlarge.pth"


def test_predict_docstring_mentions_one_class_keypoint_mapping() -> None:
    """The public predict() contract must document active-first keypoint class-id mapping."""
    doc = RFDETRKeypointPreview.predict.__doc__ or ""
    assert "one-class preview keypoint setup" in doc
    assert "class_id=0" in doc
    assert "class_id=1" in doc
    assert "__background__" in doc
