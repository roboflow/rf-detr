# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Mask filtering avoids a full-resolution gather when every row survives."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr._prediction import core
from tests._markers import _cuda_marks
from tests.inference.helpers import _DummyRFDETR


class TestMaskFiltering:
    """Check mask values, ownership, and the allocation-producing gather itself."""

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_cuda_marks)])
    @pytest.mark.parametrize(
        ("score_floor", "class_thresholds", "expected_indices", "empty"),
        [
            pytest.param(0.2, None, [0, 1, 2], False, id="all"),
            pytest.param(0.5, None, [2], False, id="strict-threshold"),
            pytest.param(1.0, None, [], False, id="none"),
            pytest.param(0.2, [0.2, 0.8, float("inf")], [0, 2], False, id="per-class-some"),
            pytest.param(0.2, [0.2, 0.2, float("inf")], [0, 1, 2], False, id="per-class-all"),
            pytest.param(0.2, None, [], True, id="empty"),
        ],
    )
    def test_decode_masks_without_redundant_gather(
        self,
        device: str,
        score_floor: float,
        class_thresholds: list[float] | None,
        expected_indices: list[int],
        empty: bool,
        mocker: MockerFixture,
    ) -> None:
        """Keep strict scalar/per-class filtering and independent result storage on CPU and CUDA."""
        count = 0 if empty else 3
        masks = (torch.arange(count * 20, device=device).reshape(count, 1, 4, 5) % 3) == 0
        rows = {
            "scores": torch.tensor([0.25, 0.5, 0.9][:count], device=device),
            "labels": torch.tensor([0, 1, 0][:count], dtype=torch.int64, device=device),
            "boxes": torch.arange(count * 4, device=device, dtype=torch.float32).reshape(count, 4),
            "masks": masks,
        }
        original_masks = masks.cpu().numpy().copy()
        # A custom postprocessor may ignore score_threshold; decoding still has to filter its rows.
        context = replace(_DummyRFDETR()._prediction_context(), postprocess=mocker.Mock(return_value=[rows]))
        prepared = core._PreparedBatch(
            context=context,
            batch_tensor=torch.empty(1, 3, 4, 5, device=device),
            target_sizes=torch.tensor([[4, 5]], device=device),
            orig_sizes=[(4, 5)],
            source_images=[],
            include_source_image=False,
            single_input=True,
            score_floor=score_floor,
            threshold_by_class_id=None if class_thresholds is None else torch.tensor(class_thresholds),
        )

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
            detections = core._decode_batch(prepared, {})

        indices = np.array(expected_indices, dtype=np.int64)
        np.testing.assert_array_equal(detections.mask, original_masks[indices, 0])
        np.testing.assert_array_equal(detections.confidence, rows["scores"].cpu().numpy()[indices])
        np.testing.assert_array_equal(detections.class_id, rows["labels"].cpu().numpy()[indices])
        np.testing.assert_array_equal(detections.xyxy, rows["boxes"].cpu().numpy()[indices])
        mask_gathers = [
            event
            for event in profile.events()
            if event.name == "aten::index" and event.input_shapes[0] == list(masks.shape)
        ]
        assert len(mask_gathers) == (len(indices) != count)

        # Even a custom postprocessor that caches CPU tensors must not alias the public numpy result.
        detections.mask[:] = ~detections.mask
        np.testing.assert_array_equal(masks.cpu().numpy(), original_masks)
