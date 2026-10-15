# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the class layout shared by native prediction and export metadata."""

from __future__ import annotations

from types import SimpleNamespace

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from rfdetr.utilities.class_names import PredictionLabels, prediction_labels


class TestPredictionLabels:
    """Verify the one derivation of names, logit slots, keypoint schema, and the class-ID map."""

    def test_training_args_define_the_layout(self) -> None:
        """Logit slots and the keypoint schema come from ``model.args`` when the context carries them.

        A background-first keypoint schema leaves slot 0 unnamed, so the class-ID map must follow the args rather than
        the number of class names or the config.
        """
        args = SimpleNamespace(num_classes=3, num_keypoints_per_class=[0, 17, 4])
        model = SimpleNamespace(
            class_names=["person", "car"],
            model=SimpleNamespace(args=args),
            model_config=SimpleNamespace(num_keypoints_per_class=[]),
        )

        labels = prediction_labels(model)

        assert labels == PredictionLabels(["person", "car"], 3, [0, 17, 4], {1: "person", 2: "car"})

    def test_missing_args_keep_coco_names_zero_indexed(self) -> None:
        """Without ``model.args`` there is one logit slot per name, as native predict uses, so COCO stays 0-indexed.

        Export used to fall back to ``config.num_classes`` here instead, which switched on the sparse COCO map and gave
        the same detection a different ``class_name`` once exported. Both paths now read this one fallback.
        """
        model = SimpleNamespace(
            class_names=list(COCO_CLASS_NAMES),
            model=SimpleNamespace(),
            model_config=SimpleNamespace(num_classes=90, num_keypoints_per_class=[]),
        )

        labels = prediction_labels(model)

        assert labels == PredictionLabels(
            list(COCO_CLASS_NAMES), len(COCO_CLASS_NAMES), [], dict(enumerate(COCO_CLASS_NAMES))
        )

    def test_missing_args_take_the_keypoint_schema_from_the_config(self) -> None:
        """Without ``model.args`` a keypoint model keeps the keypoint schema its config built the head with.

        Export metadata refuses a keypoint task with no keypoint schema, so an empty fallback would break the export of
        such a model; the config also keeps slot 0 unnamed for a background-first schema.
        """
        model = SimpleNamespace(
            class_names=["person"],
            model=SimpleNamespace(),
            model_config=SimpleNamespace(num_keypoints_per_class=[0, 17]),
        )

        labels = prediction_labels(model)

        assert labels == PredictionLabels(["person"], 1, [0, 17], {1: "person"})
