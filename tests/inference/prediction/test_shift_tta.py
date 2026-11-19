# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for sub-patch shift test-time augmentation, ``predict(..., shift_tta=N)`` (issue #1599)."""

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import PIL.Image
import pytest
import supervision as sv
import torch

from rfdetr import ClassThresholds, RFDETRSegNano
from rfdetr._prediction import core
from rfdetr._prediction.shift_tta import (
    fuse_passes,
    map_to_image,
    place_on_canvas,
    shift_offsets,
    validate_shift_tta,
)
from rfdetr.models.postprocess import PostProcess
from tests.inference.helpers import _DummyModel, _DummyRFDETR

_PATCH = 16
_SIDE = 64
_SQUARE = 16


def _detections(xyxy: list[list[float]], confidence: list[float], class_id: list[int]) -> sv.Detections:
    """Build box-only detections."""
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id),
    )


def _square_image(corner: int) -> torch.Tensor:
    """A grey ``_SIDE`` image with one white ``_SQUARE`` square whose top-left corner is at ``(corner, corner)``."""
    image = torch.full((3, _SIDE, _SIDE), 0.5)
    image[:, corner : corner + _SQUARE, corner : corner + _SQUARE] = 1.0
    return image


def _grid_sensitive_run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """A one-query model that finds the white square exactly but scores it by its position within the patch grid.

    The logit is ``3 * cos(2 * pi * x0 / patch)``, where ``x0`` is the square's left edge in input pixels, the same kind
    of grid-phase dependence issue #1599 reports. Mask logits are at a quarter of the input resolution.
    """
    _, _, height, width = batch.shape
    boxes, logits, masks = [], [], []
    for image in batch:
        rows = (image[0] > 0.75).any(dim=1).nonzero().flatten()
        cols = (image[0] > 0.75).any(dim=0).nonzero().flatten()
        y0, y1, x0, x1 = rows[0].item(), rows[-1].item() + 1, cols[0].item(), cols[-1].item() + 1
        boxes.append([(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height])
        logits.append([3 * math.cos(2 * math.pi * x0 / _PATCH)])
        mask = torch.full((height // 4, width // 4), -10.0)
        mask[y0 // 4 : y1 // 4, x0 // 4 : x1 // 4] = 10.0
        masks.append(mask)
    return {
        "pred_logits": torch.tensor(logits).unsqueeze(1),
        "pred_boxes": torch.tensor(boxes).unsqueeze(1),
        "pred_masks": torch.stack(masks).unsqueeze(1),
    }


def _grid_sensitive_context() -> core.PredictionContext:
    """A prediction context around :func:`_grid_sensitive_run`: identity normalization, no resize."""
    return core.PredictionContext(
        config=SimpleNamespace(patch_size=_PATCH, num_windows=1, num_channels=3),
        device=torch.device("cpu"),
        default_shape=(_SIDE, _SIDE),
        means=[0.0, 0.0, 0.0],
        stds=[1.0, 1.0, 1.0],
        class_names=["square"],
        class_id_to_name={0: "square"},
        num_classes=1,
        num_keypoints_per_class=[],
        postprocess=PostProcess(),
        run=_grid_sensitive_run,
        runtime_info={},
    )


def _predict(images: Any, threshold: Any = 0.0, shift_tta: int = 1) -> Any:
    """Predict with the grid-sensitive model."""
    return core._predict_batch(_grid_sensitive_context, images, threshold, None, None, True, False, shift_tta)


class TestShiftTTAHelpers:
    """The pure helpers: argument check, offsets, canvas placement and mapping back."""

    @pytest.mark.parametrize("value", [1, 2, 4, 16])
    def test_validate_accepts_one_to_patch_size(self, value: int) -> None:
        assert validate_shift_tta(value, patch_size=16) == value

    @pytest.mark.parametrize("value", [0, -1, 17, 2.0, True, "4", None])
    def test_validate_rejects_other_values(self, value: object) -> None:
        with pytest.raises(ValueError, match="shift_tta must be an integer from 1 to the patch size"):
            validate_shift_tta(value, patch_size=16)

    @pytest.mark.parametrize(
        ("patch_size", "passes", "expected"),
        [(16, 1, [0]), (16, 2, [0, 8]), (16, 4, [0, 4, 8, 12]), (12, 4, [0, 3, 6, 9]), (14, 4, [0, 4, 7, 10])],
    )
    def test_offsets_are_even_fractions_of_a_patch(self, patch_size: int, passes: int, expected: list[int]) -> None:
        assert shift_offsets(patch_size, passes) == expected

    def test_canvas_holds_the_whole_batch_at_the_offset(self) -> None:
        batch = torch.rand(2, 3, 8, 6)
        canvas = place_on_canvas(batch, offset=3, margin=4)
        assert canvas.shape == (2, 3, 12, 10)
        torch.testing.assert_close(canvas[:, :, 3:11, 3:9], batch)
        assert canvas.sum() == pytest.approx(batch.sum().item())

    @pytest.mark.parametrize("offset", [0, 3, 7])
    def test_boxes_map_back_to_the_image(self, offset: int) -> None:
        """A box at a known place in the image, expressed on the canvas, maps back to that place."""
        h, w, margin = 48, 64, 16
        x0, y0, x1, y1 = 10.0, 5.0, 30.0, 25.0
        canvas_box = [(x0 + x1) / 2 + offset, (y0 + y1) / 2 + offset, x1 - x0, y1 - y0]
        scale = torch.tensor([w + margin, h + margin, w + margin, h + margin])
        outputs = {"pred_boxes": (torch.tensor(canvas_box) / scale).view(1, 1, 4)}
        mapped = map_to_image(outputs, offset, (h, w), margin)["pred_boxes"].view(4)
        expected = torch.tensor([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
        torch.testing.assert_close(mapped, expected)

    def test_masks_map_back_to_the_image_region(self) -> None:
        """Mask logits are resampled from the image's part of the canvas, at the same resolution per pixel."""
        canvas_logits = torch.zeros(1, 1, 20, 20)
        canvas_logits[..., 2:18, 2:18] = 1.0  # the image's region, for offset 8 and image side 64 at 1/4 resolution
        mapped = map_to_image({"pred_boxes": torch.zeros(1, 1, 4), "pred_masks": canvas_logits}, 8, (64, 64), 16)
        assert mapped["pred_masks"].shape == (1, 1, 16, 16)
        torch.testing.assert_close(mapped["pred_masks"], torch.ones(1, 1, 16, 16))


class TestFusePasses:
    """Averaging the detections of the passes."""

    def test_matching_detections_average_box_and_score(self) -> None:
        fused = fuse_passes([_detections([[0, 0, 10, 10]], [0.9], [0]), _detections([[1, 1, 11, 11]], [0.5], [0])])
        np.testing.assert_allclose(fused.xyxy, [[0.5, 0.5, 10.5, 10.5]])
        np.testing.assert_allclose(fused.confidence, [0.7], rtol=1e-6)

    def test_a_missed_pass_counts_as_zero(self) -> None:
        fused = fuse_passes([_detections([[0, 0, 10, 10]], [0.8], [0]), sv.Detections.empty()])
        np.testing.assert_allclose(fused.confidence, [0.4], rtol=1e-6)

    def test_other_class_or_low_overlap_is_another_object(self) -> None:
        first = _detections([[0, 0, 10, 10]], [0.8], [0])
        fused = fuse_passes([first, _detections([[0, 0, 10, 10], [5, 5, 15, 15]], [0.6, 0.4], [1, 0])])
        np.testing.assert_allclose(sorted(fused.confidence), [0.2, 0.3, 0.4], rtol=1e-6)

    def test_prefers_the_highest_scoring_match_over_the_closest_box(self) -> None:
        """DETR often has a low-score query on a near-copy of a box; it must not take the object's real match."""
        first = _detections([[0, 0, 10, 10]], [0.9], [0])
        second = _detections([[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5]], [0.01, 0.9], [0, 0])
        fused = fuse_passes([first, second])
        assert fused.confidence[0] == pytest.approx(0.9)

    def test_each_pass_joins_a_group_once(self) -> None:
        """Two detections of one pass on the same object stay in different groups."""
        first = _detections([[0, 0, 10, 10]], [0.9], [0])
        second = _detections([[0, 0, 10, 10], [0, 0, 10, 10]], [0.7, 0.3], [0, 0])
        np.testing.assert_allclose(fuse_passes([first, second]).confidence, [0.8, 0.15], rtol=1e-6)

    def test_masks_keep_pixels_covered_by_at_least_half(self) -> None:
        masks = np.zeros((3, 1, 4, 4), dtype=bool)
        masks[0, 0, :2] = True
        masks[1, 0, :3] = True
        masks[2, 0, :1] = True
        passes = []
        for k in range(3):
            detections = _detections([[0, 0, 10, 10]], [0.9], [0])
            detections.mask = masks[k]
            passes.append(detections)
        np.testing.assert_array_equal(fuse_passes(passes).mask[0], masks[0, 0])

    def test_keeps_data_of_the_best_detection_and_sorts_by_score(self) -> None:
        first = _detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.3, 0.9], [0, 0])
        second = _detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.3, 0.9], [0, 0])
        for detections in (first, second):
            detections.data["class_name"] = np.array(["low", "high"])
        fused = fuse_passes([first, second])
        assert fused.data["class_name"].tolist() == ["high", "low"]

    def test_no_detections(self) -> None:
        assert len(fuse_passes([sv.Detections.empty(), sv.Detections.empty()])) == 0


class TestShiftTTAPredict:
    """End to end through the prediction pipeline, with a model whose score depends on the patch-grid phase."""

    def test_single_pass_score_swings_with_the_grid_phase(self) -> None:
        scores = [_predict(_square_image(corner)).confidence[0] for corner in range(16, 32)]
        assert max(scores) - min(scores) > 0.8

    def test_four_passes_remove_the_swing_and_keep_the_box(self) -> None:
        """Four passes a quarter patch apart average the periodic score to a constant; the box stays exact."""
        for corner in range(16, 32):
            detections = _predict(_square_image(corner), shift_tta=4)
            assert len(detections) == 1
            assert detections.confidence[0] == pytest.approx(0.5, abs=1e-5)
            np.testing.assert_allclose(detections.xyxy[0], [corner, corner, corner + _SQUARE, corner + _SQUARE])

    def test_masks_come_back_in_place(self) -> None:
        detections = _predict(_square_image(20), shift_tta=2)
        expected = np.zeros((_SIDE, _SIDE), dtype=bool)
        expected[20 : 20 + _SQUARE, 20 : 20 + _SQUARE] = True
        assert detections.mask.shape == (1, _SIDE, _SIDE)
        assert (detections.mask[0] & expected).sum() / (detections.mask[0] | expected).sum() > 0.85

    def test_threshold_applies_to_the_fused_score(self) -> None:
        image = _square_image(16)  # single-pass score ~0.95, fused 0.5
        assert len(_predict(image, threshold=0.6)) == 1
        assert len(_predict(image, threshold=0.6, shift_tta=4)) == 0
        assert len(_predict(image, threshold=0.4, shift_tta=4)) == 1

    def test_class_thresholds_apply_to_the_fused_score(self) -> None:
        image = _square_image(16)
        assert len(_predict(image, threshold=ClassThresholds({"square": 0.4}), shift_tta=4)) == 1
        assert len(_predict(image, threshold=ClassThresholds({"square": 0.6}), shift_tta=4)) == 0

    def test_batch_returns_a_list_with_source_metadata(self) -> None:
        results = _predict([_square_image(16), _square_image(24)], shift_tta=2)
        assert isinstance(results, list) and len(results) == 2
        for result, corner in zip(results, (16, 24)):
            np.testing.assert_allclose(result.xyxy[0], [corner, corner, corner + _SQUARE, corner + _SQUARE])
            assert result.metadata["source_image"].shape == (_SIDE, _SIDE, 3)
            assert result.data["class_name"].tolist() == ["square"]
            assert result.data["source_shape"].tolist() == [[_SIDE, _SIDE]]

    def test_shift_tta_one_is_the_plain_path(self) -> None:
        image = _square_image(20)
        plain, explicit = _predict(image), _predict(image, shift_tta=1)
        np.testing.assert_array_equal(plain.xyxy, explicit.xyxy)
        np.testing.assert_array_equal(plain.confidence, explicit.confidence)


class TestShiftTTAPublicAPI:
    """``RFDETR.predict(..., shift_tta=N)``."""

    def test_rejects_out_of_range_values(self) -> None:
        model = _DummyRFDETR()
        with pytest.raises(ValueError, match="shift_tta must be an integer"):
            model.predict(PIL.Image.new("RGB", (28, 28)), shift_tta=0)

    def test_rejects_keypoint_models(self) -> None:
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0], include_keypoints=True)
        model.model.args = SimpleNamespace(num_keypoints_per_class=[17])
        with pytest.raises(ValueError, match="does not support keypoint models"):
            model.predict(PIL.Image.new("RGB", (28, 28)), shift_tta=2)

    def test_segmentation_model_returns_image_sized_masks(self) -> None:
        model = RFDETRSegNano(pretrain_weights=None)
        detections = model.predict(PIL.Image.new("RGB", (100, 80), (128, 128, 128)), threshold=0.0, shift_tta=2)
        assert isinstance(detections, sv.Detections)
        assert detections.mask is not None and detections.mask.shape[1:] == (80, 100)
        assert detections.xyxy[:, [0, 2]].max() <= 100 and detections.xyxy[:, [1, 3]].max() <= 80
