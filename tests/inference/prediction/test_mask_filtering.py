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
from tests._markers import _cuda_marks, _requires_mps
from tests.inference.helpers import _DummyRFDETR


class TestMaskFiltering:
    """Check mask values, ownership, and the allocation-producing gather itself."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=_cuda_marks), pytest.param("mps", marks=_requires_mps)]
    )
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

    def test_decode_mixed_batch_gathers_only_filtered_image(self, mocker: MockerFixture) -> None:
        """Only the image that drops rows pays for the mask gather; each image keeps its own masks.

        Image 0 keeps every row and image 1 drops its first one, so a flag carried over from one image to the next would
        either gather twice or hand image 1 the unfiltered masks of image 0.
        """
        masks = [(torch.arange(60).reshape(3, 1, 4, 5) % modulus) == 0 for modulus in (3, 4)]
        rows = [
            {
                "scores": torch.tensor(scores),
                "labels": torch.tensor([0, 1, 0], dtype=torch.int64),
                "boxes": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "masks": image_masks,
            }
            for scores, image_masks in zip(([0.6, 0.7, 0.9], [0.25, 0.7, 0.9]), masks)
        ]
        context = replace(_DummyRFDETR()._prediction_context(), postprocess=mocker.Mock(return_value=rows))
        prepared = core._PreparedBatch(
            context=context,
            batch_tensor=torch.empty(2, 3, 4, 5),
            target_sizes=torch.tensor([[4, 5], [4, 5]]),
            orig_sizes=[(4, 5), (4, 5)],
            source_images=[],
            include_source_image=False,
            single_input=False,
            score_floor=0.5,
            threshold_by_class_id=None,
        )

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
            detections = core._decode_batch(prepared, {})

        np.testing.assert_array_equal(detections[0].mask, masks[0].numpy()[:, 0])
        np.testing.assert_array_equal(detections[1].mask, masks[1].numpy()[[1, 2], 0])
        mask_gathers = [
            event for event in profile.events() if event.name == "aten::index" and event.input_shapes[0] == [3, 1, 4, 5]
        ]
        assert len(mask_gathers) == 1

    def test_decode_raises_when_mask_rows_do_not_match_score_rows(self, mocker: MockerFixture) -> None:
        """A postprocessor returning fewer mask rows than score rows still reaches the gather and fails loudly.

        Two of three score rows survive the threshold, which equals the two mask rows, so the kept count alone must
        not be mistaken for an identity selection: pairing score rows 1-2 with mask rows 0-1 would silently misalign.
        """
        rows = {
            "scores": torch.tensor([0.25, 0.7, 0.9]),
            "labels": torch.tensor([0, 1, 0], dtype=torch.int64),
            "boxes": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "masks": torch.zeros(2, 1, 4, 5, dtype=torch.bool),
        }
        context = replace(_DummyRFDETR()._prediction_context(), postprocess=mocker.Mock(return_value=[rows]))
        prepared = core._PreparedBatch(
            context=context,
            batch_tensor=torch.empty(1, 3, 4, 5),
            target_sizes=torch.tensor([[4, 5]]),
            orig_sizes=[(4, 5)],
            source_images=[],
            include_source_image=False,
            single_input=True,
            score_floor=0.5,
            threshold_by_class_id=None,
        )

        with pytest.raises(IndexError):
            core._decode_batch(prepared, {})
