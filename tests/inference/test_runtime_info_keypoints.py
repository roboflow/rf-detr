# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``runtime_info["has_keypoints"]`` tells whether ``predict()`` returns keypoints.

``runtime_info["task"]`` gives segmentation precedence, so a model with both a segmentation and a keypoint head reports
``segment``; the flag is how a caller finds the keypoints such a model also returns.
"""

from __future__ import annotations

from typing import Any

import pytest

from rfdetr.detr import RFDETR
from rfdetr.inference import RFDETRInference
from rfdetr.variants import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano

_SMALL = {"pretrain_weights": None, "device": "cpu", "resolution": 64, "num_queries": 4, "num_select": 4}
_KEYPOINT_HEAD = {"num_classes": 1, "use_grouppose_keypoints": True, "num_keypoints_per_class": [2]}


class TestNativeRuntimeInfoKeypoints:
    """The native ``runtime_info`` reports the task and, separately, whether keypoints are returned."""

    @pytest.mark.parametrize(
        ("model_type", "options", "task", "has_keypoints"),
        [
            pytest.param(RFDETRNano, {}, "detect", False, id="detection"),
            pytest.param(RFDETRSegNano, {}, "segment", False, id="segmentation"),
            pytest.param(RFDETRKeypointPreview, {}, "keypoints", True, id="keypoints"),
            pytest.param(RFDETRSegNano, _KEYPOINT_HEAD, "segment", True, id="segmentation-with-keypoints"),
        ],
    )
    def test_reports_keypoints_next_to_the_task(
        self, model_type: type[RFDETR], options: dict[str, Any], task: str, has_keypoints: bool
    ) -> None:
        """A two-head model keeps ``task="segment"`` and reports ``has_keypoints=True``."""
        predictor = RFDETRInference(model_type(**_SMALL, **options))

        info = predictor.runtime_info

        assert (info["task"], info["has_keypoints"]) == (task, has_keypoints)
