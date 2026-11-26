# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pixel covariance of the keypoints a model with both a segmentation and a keypoint head returns.

Such a model returns ``sv.Detections`` (``sv.KeyPoints`` has no mask field). It must carry the same derived
``data["covariance"]`` a keypoint-only model's ``sv.KeyPoints`` does, so a caller switching models keeps the key.
"""

import numpy as np
import PIL.Image
import supervision as sv

from rfdetr.utilities.keypoints import precision_cholesky_to_pixel_covariance
from tests.inference.helpers import _DummyModel, _DummyRFDETR

_NUM_KEYPOINTS = 2


def _both_heads_model() -> _DummyRFDETR:
    """Return a weight-free model whose postprocess emits masks, keypoints and precision factors for two rows.

    Examples:
        >>> model = _both_heads_model()
        >>> model.model._include_masks, model.model._include_keypoints
        (True, True)
    """
    model = _DummyRFDETR()
    model.model = _DummyModel(labels=[0, 1], include_keypoints=True, num_keypoints=_NUM_KEYPOINTS, include_masks=True)
    return model


class TestBothHeadsCovariance:
    """``detections.data["covariance"]`` holds the pixel covariance of every kept row's keypoints."""

    def test_covariance_is_derived_from_the_precision_factors(self) -> None:
        """The covariance equals the conversion of the returned precision factors at the source image size."""
        detections = _both_heads_model().predict(PIL.Image.new("RGB", (64, 48)))

        assert isinstance(detections, sv.Detections)
        assert detections.mask is not None
        expected = precision_cholesky_to_pixel_covariance(
            precision_cholesky=detections.data["keypoint_precision_cholesky"],
            source_shape=detections.data["source_shape"],
        )
        assert detections.data["covariance"].shape == (2, _NUM_KEYPOINTS, 2, 2)
        np.testing.assert_allclose(detections.data["covariance"], expected, rtol=1e-6)

    def test_covariance_is_present_when_all_filtered(self) -> None:
        """The key stays, with no rows, when the threshold drops every detection."""
        detections = _both_heads_model().predict(PIL.Image.new("RGB", (64, 48)), threshold=1.1)

        assert isinstance(detections, sv.Detections)
        assert len(detections) == 0
        assert detections.data["covariance"].shape == (0, _NUM_KEYPOINTS, 2, 2)
