# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for test-time augmentation views, ``RFDETRInference.predict(..., tta=TTAConfig(...))`` (issue #1599)."""

import math
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import PIL.Image
import pytest
import supervision as sv
import torch
from pytest_mock import MockerFixture
from torchvision.transforms.functional import resize

from rfdetr import ClassThresholds, RFDETRInference, RFDETRNano, RFDETRSegNano
from rfdetr._prediction import core, pipelines
from rfdetr._prediction.tta import (
    DEFAULT_ZOOM,
    TTAConfig,
    _View,
    build_views,
    canvas_shape,
    fuse_passes,
    group_passes,
    map_to_image,
    place_view,
    shift_offsets,
    validate_shift,
    view_count,
    vote_masks,
)
from rfdetr.models.postprocess import PostProcess
from tests._markers import _requires_cuda

_PATCH = 16
_SIDE = 64
_SQUARE = 16


def _detections(xyxy: list[list[float]], confidence: list[float], class_id: list[int]) -> sv.Detections:
    """Build box-only detections.

    Examples:
        >>> _detections([[0, 0, 4, 4]], [0.5], [1]).class_id.tolist()
        [1]
    """
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id),
    )


def _square_image(corner: int) -> torch.Tensor:
    """A grey ``_SIDE`` image with one white ``_SQUARE`` square whose top-left corner is at ``(corner, corner)``.

    Examples:
        >>> image = _square_image(16)
        >>> image[0, 16, 16].item(), image[0, 15, 16].item()
        (1.0, 0.5)
    """
    image = torch.full((3, _SIDE, _SIDE), 0.5)
    image[:, corner : corner + _SQUARE, corner : corner + _SQUARE] = 1.0
    return image


def _grid_sensitive_run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """A one-query model that finds the white square exactly but scores it by its position within the patch grid.

    The logit is ``3 * cos(2 * pi * x0 / patch)``, where ``x0`` is the square's left edge in input pixels, the same kind
    of grid-phase dependence issue #1599 reports. Mask logits are at a quarter of the input resolution.

    Examples:
        >>> outputs = _grid_sensitive_run(_square_image(16).unsqueeze(0))
        >>> tuple(outputs["pred_boxes"].shape), tuple(outputs["pred_masks"].shape)
        ((1, 1, 4), (1, 1, 16, 16))
        >>> outputs["pred_logits"].flatten().tolist()
        [3.0]
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


def _half_patch_run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Like :func:`_grid_sensitive_run`, but score the square 0.9 in the first half of a patch and 0.2 in the second.

    With two passes half a patch apart, one pass scores 0.9 and the other 0.2, so the fused score is 0.55.

    Examples:
        >>> batch = torch.stack([_square_image(16), _square_image(24)])
        >>> [round(p, 4) for p in torch.sigmoid(_half_patch_run(batch)["pred_logits"]).flatten().tolist()]
        [0.9, 0.2]
    """
    outputs = _grid_sensitive_run(batch)
    columns = (batch[:, 0] > 0.75).any(dim=1).float().argmax(dim=1)
    probability = torch.where(columns % _PATCH < _PATCH // 2, 0.9, 0.2)
    outputs["pred_logits"] = torch.logit(probability).view(-1, 1, 1)
    return outputs


def _no_score_run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Like :func:`_grid_sensitive_run`, but every score underflows to 0, so no pass keeps a detection.

    Examples:
        >>> torch.sigmoid(_no_score_run(_square_image(16).unsqueeze(0))["pred_logits"]).tolist()
        [[[0.0]]]
    """
    outputs = _grid_sensitive_run(batch)
    outputs["pred_logits"] = torch.full_like(outputs["pred_logits"], -200.0)
    return outputs


def _two_class_run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """A two-query, two-class model: class ``a`` is :func:`_half_patch_run`'s 0.9 or 0.2, class ``b`` is always 0.4.

    Both queries box the same white square. With two passes half a patch apart, class ``a`` fuses to 0.55 and class
    ``b`` to 0.4, and the lower pass of ``a`` (0.2) is below the lowest of most per-class thresholds.

    Examples:
        >>> outputs = _two_class_run(torch.stack([_square_image(16), _square_image(24)]))
        >>> [[round(p, 4) for p in row] for row in torch.sigmoid(outputs["pred_logits"]).flatten(1).tolist()]
        [[0.9, 0.0, 0.0, 0.4], [0.2, 0.0, 0.0, 0.4]]
    """
    outputs = _half_patch_run(batch)
    a_logit, no_class = outputs["pred_logits"], torch.full_like(outputs["pred_logits"], -200.0)
    b_logit = torch.full_like(a_logit, math.log(0.4 / 0.6))
    query_a, query_b = torch.cat([a_logit, no_class], dim=2), torch.cat([no_class, b_logit], dim=2)
    outputs["pred_logits"] = torch.cat([query_a, query_b], dim=1)
    outputs["pred_boxes"] = outputs["pred_boxes"].repeat(1, 2, 1)
    outputs["pred_masks"] = outputs["pred_masks"].repeat(1, 2, 1, 1)
    return outputs


# Four far-apart objects as image ``xyxy`` and their score logits: score order (1, 3, 0, 2) differs from query order.
_OBJECTS = [[2, 2, 18, 18], [40, 2, 56, 18], [2, 40, 18, 56], [40, 40, 56, 56]]
_OBJECT_LOGITS = [0.5, 3.0, 1.0, 2.0]
# The query that scores 0 in the pass at this offset, so its group has one member fewer than there are passes.
_MISSING_QUERY, _MISSING_OFFSET = 2, 5


def _multi_query_run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """A four-query segmentation model with fixed objects and seeded noise mask logits that differ per pass.

    The image's offset on the canvas is the first row with a non-zero pixel. Query ``q`` finds object ``q`` at the same
    image position in every pass, except that query ``_MISSING_QUERY`` scores 0 in the pass at ``_MISSING_OFFSET``.
    The mask logits are standard normal noise seeded by the offset, at a quarter of the canvas resolution, so the
    members of a group disagree on many pixels.

    Examples:
        >>> batch = torch.full((1, 3, _SIDE, _SIDE), 0.5)
        >>> canvas = place_view(batch, _View(offset=_MISSING_OFFSET), (_SIDE + _PATCH, _SIDE + _PATCH))
        >>> outputs = _multi_query_run(canvas)
        >>> tuple(outputs["pred_masks"].shape), torch.sigmoid(outputs["pred_logits"]).flatten().gt(0).tolist()
        ((1, 4, 20, 20), [True, True, False, True])
    """
    _, _, height, width = batch.shape
    canvas_size = torch.tensor([width, height, width, height])
    boxes, logits, masks = [], [], []
    for image in batch:
        offset = int((image[0] != 0).any(dim=1).float().argmax())
        cxcywh = [[(x0 + x1) / 2 + offset, (y0 + y1) / 2 + offset, x1 - x0, y1 - y0] for x0, y0, x1, y1 in _OBJECTS]
        boxes.append(torch.tensor(cxcywh) / canvas_size)
        missing = offset == _MISSING_OFFSET
        logits.append([[-200.0 if missing and q == _MISSING_QUERY else v] for q, v in enumerate(_OBJECT_LOGITS)])
        generator = torch.Generator().manual_seed(offset)
        masks.append(torch.randn(len(_OBJECTS), height // 4, width // 4, generator=generator))
    return {"pred_logits": torch.tensor(logits), "pred_boxes": torch.stack(boxes), "pred_masks": torch.stack(masks)}


def _grid_sensitive_context(run: Any = _grid_sensitive_run) -> core.PredictionContext:
    """A prediction context around a fake ``run``: identity normalization, no resize.

    Examples:
        >>> _grid_sensitive_context().class_names
        ['square']
    """
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
        run=run,
        runtime_info={},
    )


def _two_class_context() -> core.PredictionContext:
    """A prediction context around :func:`_two_class_run` with classes ``a`` and ``b``.

    Examples:
        >>> _two_class_context().class_names
        ['a', 'b']
    """
    return replace(
        _grid_sensitive_context(_two_class_run),
        class_names=["a", "b"],
        class_id_to_name={0: "a", 1: "b"},
        num_classes=2,
    )


def _shifts(passes: int) -> TTAConfig | None:
    """Return the views of ``passes`` shifted passes, ``None`` for the plain prediction.

    Examples:
        >>> _shifts(1) is None
        True
        >>> _shifts(3)
        TTAConfig(shift=3, zoom=())
    """
    return None if passes == 1 else TTAConfig(shift=passes)


def _predict(images: Any, threshold: Any = 0.0, shifts: int = 1) -> Any:
    """Predict with the grid-sensitive model.

    Examples:
        >>> len(_predict(_square_image(16)))
        1
    """
    return core._predict_batch(_grid_sensitive_context, images, threshold, None, None, True, False, tta=_shifts(shifts))


class TestShiftTTAHelpers:
    """The pure helpers: argument check, offsets, canvas placement and mapping back."""

    @pytest.mark.parametrize("value", [1, 2, 4, 16])
    def test_validate_accepts_one_to_patch_size(self, value: int) -> None:
        assert validate_shift(value, patch_size=16) == value

    @pytest.mark.parametrize("value", [0, -1, 17, 2.0, True, "4", None])
    def test_validate_rejects_other_values(self, value: object) -> None:
        with pytest.raises(ValueError, match="TTAConfig.shift must be an integer from 1 to the patch size"):
            validate_shift(value, patch_size=16)

    @pytest.mark.parametrize(
        ("patch_size", "passes", "expected"),
        [(16, 1, [0]), (16, 2, [0, 8]), (16, 4, [0, 4, 8, 12]), (12, 4, [0, 3, 6, 9]), (14, 4, [0, 4, 7, 10])],
    )
    def test_offsets_are_even_fractions_of_a_patch(self, patch_size: int, passes: int, expected: list[int]) -> None:
        assert shift_offsets(patch_size, passes) == expected

    def test_canvas_holds_the_whole_batch_at_the_offset(self) -> None:
        batch = torch.rand(2, 3, 8, 6)
        canvas = place_view(batch, _View(offset=3), (12, 10))
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
        mapped = map_to_image(outputs, _View(offset=offset), (h, w), (h + margin, w + margin), None)["pred_boxes"].view(
            4
        )
        expected = torch.tensor([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
        torch.testing.assert_close(mapped, expected)

    def test_masks_map_back_to_the_image_region(self) -> None:
        """Mask logits are resampled from the image's part of the canvas, at the same resolution per pixel."""
        canvas_logits = torch.zeros(1, 1, 20, 20)
        canvas_logits[..., 2:18, 2:18] = 1.0  # the image's region, for offset 8 and image side 64 at 1/4 resolution
        outputs = {"pred_boxes": torch.zeros(1, 1, 4), "pred_masks": canvas_logits}
        mapped = map_to_image(outputs, _View(offset=8), (64, 64), (80, 80), (16, 16))
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

    def test_keeps_data_of_the_best_detection_and_sorts_by_score(self) -> None:
        first = _detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.3, 0.9], [0, 0])
        second = _detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.3, 0.9], [0, 0])
        for detections in (first, second):
            detections.data["class_name"] = np.array(["low", "high"])
        fused = fuse_passes([first, second])
        assert fused.data["class_name"].tolist() == ["high", "low"]

    def test_no_detections_keep_the_passes_schema(self) -> None:
        """With every pass empty, the result keeps the data keys and dtypes the passes carry, in a new object."""
        empty = sv.Detections(
            xyxy=np.zeros((0, 4), np.float32),
            confidence=np.zeros(0, np.float32),
            class_id=np.zeros(0, np.int64),
            data={"class_name": np.array([], dtype=str)},
        )
        fused = fuse_passes([empty, empty])
        assert len(fused) == 0
        assert fused is not empty
        assert fused.data.keys() == {"class_name"}
        assert fused.data["class_name"].dtype == empty.data["class_name"].dtype
        assert fused.xyxy.shape == (0, 4)

    def test_seeding_above_threshold_keeps_the_same_kept_groups(self) -> None:
        """Letting only rows above the threshold seed changes no group whose fused score is above it."""
        rng = np.random.default_rng(0)
        objects = rng.uniform(0, 80, size=(8, 2))
        passes = []
        for _ in range(4):
            corners = np.repeat(objects, 3, axis=0) + rng.normal(0, 1.5, size=(24, 2))
            boxes = np.hstack([corners, corners + 20]).tolist()
            passes.append(_detections(boxes, rng.uniform(0, 1, 24).tolist(), rng.integers(0, 2, 24).tolist()))
        threshold = np.float32(0.3)
        everything = fuse_passes(passes)
        expected = everything[everything.confidence > threshold]
        seeds = np.concatenate([p.confidence for p in passes]) > threshold
        restricted = fuse_passes(passes, group_passes(passes, can_seed=seeds))
        kept = restricted[restricted.confidence > threshold]
        assert len(expected) > 0 and len(expected) < len(everything)
        np.testing.assert_array_equal(kept.xyxy, expected.xyxy)
        np.testing.assert_array_equal(kept.confidence, expected.confidence)

    def test_only_allowed_rows_seed_groups(self) -> None:
        """A row that may not seed still joins a group as a member, and starts none of its own."""
        first = _detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.3], [0, 0])
        second = _detections([[1, 1, 11, 11]], [0.2], [0])
        assert group_passes([first, second], can_seed=np.array([True, False, False])) == [[0, 2]]


class TestVoteMasks:
    """Fusing the masks of each group from mask logits."""

    def test_masks_keep_pixels_covered_by_at_least_half(self) -> None:
        masks = torch.zeros(3, 4, 4, dtype=torch.bool)
        masks[0, :2] = True
        masks[1, :3] = True
        masks[2, :1] = True
        logits = [torch.where(mask, 1.0, -1.0).unsqueeze(0) for mask in masks]
        fused = vote_masks(logits, [[(0, 0), (1, 0), (2, 0)]], size=(4, 4))
        torch.testing.assert_close(fused, masks[:1])


class TestShiftTTAMaskFusion:
    """Masks of a four-query segmentation model through ``predict(..., shifts=3)``, end to end."""

    def test_fused_masks_match_the_per_member_majority_vote(self, mocker: MockerFixture) -> None:
        """Each kept detection's mask is the old majority vote over its own members' full-size masks.

        The expected masks are built here from the mapped mask logits with plain torch, one member plane at a time,
        as the per-pass post-processing built them before fusion: bilinear resize to the image, ``> 0``, then
        ``mean >= 0.5`` over the members that found the object.
        """
        # Each pass's outputs after `map_to_image`, recorded as returned: `spy_return_list` needs pytest-mock 3.13.
        passes = []
        real_map = core.map_to_image

        def record(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            passes.append(real_map(*args, **kwargs))
            return passes[-1]

        mocker.patch.object(core, "map_to_image", side_effect=record)
        context, image = partial(_grid_sensitive_context, _multi_query_run), torch.full((3, _SIDE, _SIDE), 0.5)
        detections = core._predict_batch(context, image, 0.0, None, None, False, False, tta=_shifts(3))
        expected, strict = [], []
        for query in range(len(_OBJECTS)):
            members = [
                outputs["pred_masks"][0, query]
                for outputs in passes
                if torch.sigmoid(outputs["pred_logits"][0, query, 0]) > 0
            ]
            planes = [
                torch.nn.functional.interpolate(
                    member[None, None], size=(_SIDE, _SIDE), mode="bilinear", align_corners=False
                )[0, 0]
                > 0.0
                for member in members
            ]
            share = torch.stack(planes).float().mean(dim=0)
            expected.append(share >= 0.5)
            strict.append(share > 0.5)
        queries = [int(np.abs(np.array(_OBJECTS) - box).sum(axis=1).argmin()) for box in detections.xyxy]
        # Rows come back by fused score, not query order, and the two-member group has tie pixels that `>=` keeps.
        assert queries == [1, 3, 0, 2]
        assert not torch.equal(expected[_MISSING_QUERY], strict[_MISSING_QUERY])
        np.testing.assert_array_equal(detections.mask, torch.stack([expected[q] for q in queries]).numpy())

    def test_resizes_one_group_of_mask_planes_at_a_time(self, mocker: MockerFixture) -> None:
        """Each resize call takes one kept group's member planes at mask-head size, never more than the passes."""
        resize = mocker.spy(torch.nn.functional, "interpolate")
        context, image = partial(_grid_sensitive_context, _multi_query_run), torch.full((3, _SIDE, _SIDE), 0.5)
        core._predict_batch(context, image, 0.0, None, None, False, False, tta=_shifts(3))
        mask_inputs = [tuple(call.args[0].shape) for call in resize.call_args_list if call.args[0].shape[1] == 1]
        # Kept groups by fused score: three of 3 members, then the one a pass missed, of 2; 11 planes in all.
        assert mask_inputs == [(3, 1, 16, 16), (3, 1, 16, 16), (3, 1, 16, 16), (2, 1, 16, 16)]

    @pytest.mark.parametrize(("shifts", "called"), [(1, True), (3, False)])
    def test_per_pass_mask_postprocessing_runs_only_on_the_plain_path(
        self, shifts: int, called: bool, mocker: MockerFixture
    ) -> None:
        """The shifted path never builds per-pass source-size masks through ``PostProcess._postprocess_masks``."""
        postprocess_masks = mocker.spy(PostProcess, "_postprocess_masks")
        context, image = partial(_grid_sensitive_context, _multi_query_run), torch.full((3, _SIDE, _SIDE), 0.5)
        core._predict_batch(context, image, 0.0, None, None, False, False, tta=_shifts(shifts))
        assert postprocess_masks.called is called


class TestShiftTTAPredict:
    """End to end through the prediction pipeline, with a model whose score depends on the patch-grid phase."""

    def test_single_pass_score_swings_with_the_grid_phase(self) -> None:
        scores = [_predict(_square_image(corner)).confidence[0] for corner in range(16, 32)]
        assert max(scores) - min(scores) > 0.8

    @pytest.mark.parametrize("corner", range(16, 32))
    def test_four_passes_remove_the_swing_and_keep_the_box(self, corner: int) -> None:
        """Four passes a quarter patch apart average the periodic score to a constant; the box stays exact."""
        detections = _predict(_square_image(corner), shifts=4)
        assert len(detections) == 1
        assert detections.confidence[0] == pytest.approx(0.5, abs=1e-5)
        np.testing.assert_allclose(detections.xyxy[0], [corner, corner, corner + _SQUARE, corner + _SQUARE])

    def test_masks_come_back_in_place(self) -> None:
        detections = _predict(_square_image(20), shifts=2)
        expected = np.zeros((_SIDE, _SIDE), dtype=bool)
        expected[20 : 20 + _SQUARE, 20 : 20 + _SQUARE] = True
        assert detections.mask.shape == (1, _SIDE, _SIDE)
        assert (detections.mask[0] & expected).sum() / (detections.mask[0] | expected).sum() > 0.85

    @pytest.mark.parametrize(("threshold", "shifts", "kept"), [(0.6, 1, 1), (0.6, 4, 0), (0.4, 4, 1)])
    def test_threshold_applies_to_the_fused_score(self, threshold: float, shifts: int, kept: int) -> None:
        """The single-pass score is ~0.95 and the four-pass fused score 0.5."""
        assert len(_predict(_square_image(16), threshold=threshold, shifts=shifts)) == kept

    @pytest.mark.parametrize(("threshold", "kept"), [(0.5, True), (0.54, True), (0.56, False), (0.95, False)])
    def test_fused_score_is_the_plain_mean_thresholded_once(self, threshold: float, kept: bool) -> None:
        """Pass scores 0.9 and 0.2 fuse to 0.55; the bar applies to that mean, not to each pass.

        A bar at 0.5 passes although 0.2 is below 0.5 / 2, and one at 0.56 drops the object although a pass scored 0.9.
        """
        context = partial(_grid_sensitive_context, _half_patch_run)
        detections = core._predict_batch(
            context, _square_image(16), threshold, None, None, False, False, tta=_shifts(2)
        )
        np.testing.assert_allclose(detections.confidence, [0.55] if kept else [], rtol=1e-6)

    @pytest.mark.parametrize(("threshold", "kept"), [(0.4, 1), (0.6, 0)])
    def test_class_thresholds_apply_to_the_fused_score(self, threshold: float, kept: int) -> None:
        detections = _predict(_square_image(16), threshold=ClassThresholds({"square": threshold}), shifts=4)
        assert len(detections) == kept

    def test_batch_returns_a_list_with_source_metadata(self) -> None:
        results = _predict([_square_image(16), _square_image(24)], shifts=2)
        assert isinstance(results, list) and len(results) == 2
        for result, corner in zip(results, (16, 24)):
            np.testing.assert_allclose(result.xyxy[0], [corner, corner, corner + _SQUARE, corner + _SQUARE])
            assert result.metadata["source_image"].shape == (_SIDE, _SIDE, 3)
            assert result.data["class_name"].tolist() == ["square"]
            assert result.data["source_shape"].tolist() == [[_SIDE, _SIDE]]

    @pytest.mark.parametrize(
        ("run", "threshold"),
        [
            pytest.param(_no_score_run, 0.0, id="every-pass-empty"),
            pytest.param(_grid_sensitive_run, 0.99, id="no-group-above-threshold"),
        ],
    )
    def test_no_detection_keeps_the_plain_schema(self, run: Any, threshold: float) -> None:
        """An image with no kept detection still carries ``class_name``, ``source_shape`` and a ``(0, H, W)`` mask."""
        context, image = partial(_grid_sensitive_context, run), _square_image(16)
        detections = core._predict_batch(context, image, threshold, None, None, False, False, tta=_shifts(2))
        plain = core._predict_batch(context, image, threshold, None, None, False, False)
        assert len(detections) == 0
        assert detections.data.keys() == plain.data.keys() == {"class_name", "source_shape"}
        assert detections.data["source_shape"].shape == (0, 2)
        assert detections.mask is not None and detections.mask.shape == plain.mask.shape == (0, _SIDE, _SIDE)

    @pytest.mark.parametrize(("shifts", "passes"), [(1, 1), (2, 2), (4, 4)])
    def test_the_model_runs_once_per_pass_and_one_shift_stays_plain(
        self, shifts: int, passes: int, mocker: MockerFixture
    ) -> None:
        """``shifts=1`` never takes the shifted path and runs the model once; ``N`` runs it ``N`` times."""
        shifted = mocker.spy(core, "_predict_shifted")
        run = mocker.Mock(wraps=_grid_sensitive_run)
        context = partial(_grid_sensitive_context, run)
        core._predict_batch(context, _square_image(20), 0.0, None, None, False, False, tta=_shifts(shifts))
        assert run.call_count == passes
        assert shifted.called is (shifts != 1)


class TestShiftTTAClassThresholds:
    """Per-class thresholds on a two-class model: class ``a`` fuses to 0.55, class ``b`` to 0.4."""

    @pytest.mark.parametrize(
        ("thresholds", "kept"),
        [
            pytest.param(ClassThresholds({"a": 0.5, "b": 0.3}), ["a", "b"], id="both-above-their-own-bar"),
            pytest.param(ClassThresholds({"a": 0.6, "b": 0.3}), ["b"], id="a-below-its-bar-above-the-lowest"),
            pytest.param(ClassThresholds({"a": 0.5, "b": 0.45}), ["a"], id="b-below-its-bar"),
            pytest.param(ClassThresholds({"b": 0.3}, default=0.6), ["b"], id="default-drops-a"),
            pytest.param(ClassThresholds({"a": 0.5}, default=0.45), ["a"], id="default-drops-b"),
            pytest.param(ClassThresholds([0.5, 0.3]), ["a", "b"], id="one-per-class-sequence"),
            pytest.param(ClassThresholds({"a": float("inf")}, default=0.3), ["b"], id="inf-drops-a"),
        ],
    )
    def test_each_class_is_held_to_its_own_bar(self, thresholds: ClassThresholds, kept: list[str]) -> None:
        """Rows come back by fused score, ``a`` (0.55) before ``b`` (0.4); a class under its own bar is dropped."""
        detections = core._predict_batch(
            _two_class_context, _square_image(16), thresholds, None, None, False, False, tta=_shifts(2)
        )
        assert detections.data["class_name"].tolist() == kept

    def test_fused_scores_are_means_over_the_passes_below_the_lowest_bar(self) -> None:
        """A pass of ``a`` scoring 0.2 is under the lowest bar, 0.3, and still counts toward the 0.55 mean."""
        thresholds = ClassThresholds({"a": 0.5, "b": 0.3})
        detections = core._predict_batch(
            _two_class_context, _square_image(16), thresholds, None, None, False, False, tta=_shifts(2)
        )
        np.testing.assert_allclose(detections.confidence, [0.55, 0.4], rtol=1e-6)
        assert detections.class_id.tolist() == [0, 1]


class TestShiftTTAValidation:
    """A request ``TTAConfig`` cannot serve is refused before any image is read."""

    def test_rejects_a_shift_above_the_patch_size_before_any_image_is_read(self, mocker: MockerFixture) -> None:
        decode = mocker.spy(core, "_decode_prefetched_jpegs")
        with pytest.raises(ValueError, match="TTAConfig.shift must be an integer from 1 to the patch size"):
            core._predict_batch(
                _grid_sensitive_context,
                _square_image(16),
                0.5,
                None,
                None,
                False,
                False,
                tta=TTAConfig(shift=_PATCH + 1),
            )
        decode.assert_not_called()

    def test_refuses_keypoint_models_before_any_image_is_read(self, mocker: MockerFixture) -> None:
        context = partial(replace, _grid_sensitive_context(), num_keypoints_per_class=[17])
        decode = mocker.spy(core, "_decode_prefetched_jpegs")
        with pytest.raises(ValueError, match="does not support keypoint models"):
            core._predict_batch(context, _square_image(16), 0.5, None, None, False, False, tta=TTAConfig(shift=2))
        decode.assert_not_called()


class TestTTAPublicAPI:
    """``RFDETRInference(model).predict(..., tta=TTAConfig(shift=N))``."""

    def test_rejects_segmentation_model_with_keypoint_head(self) -> None:
        """Keypoints are not fused across passes, so a model with both heads is refused like a keypoint model."""
        model = RFDETRSegNano(pretrain_weights=None, use_grouppose_keypoints=True, num_keypoints_per_class=[17])
        with pytest.raises(ValueError, match="does not support keypoint models"):
            RFDETRInference(model).predict(PIL.Image.new("RGB", (28, 28)), tta=TTAConfig(shift=2))

    def test_segmentation_model_returns_image_sized_masks(self) -> None:
        torch.manual_seed(0)
        predictor = RFDETRInference(RFDETRSegNano(pretrain_weights=None))
        detections = predictor.predict(
            PIL.Image.new("RGB", (100, 80), (128, 128, 128)), threshold=0.0, tta=TTAConfig(shift=2)
        )
        assert isinstance(detections, sv.Detections)
        assert detections.mask is not None and detections.mask.shape[1:] == (80, 100)
        assert detections.xyxy[:, [0, 2]].max() <= 100 and detections.xyxy[:, [1, 3]].max() <= 80
        assert len(detections) > 0
        assert detections.mask.dtype == np.bool_
        assert len(detections.mask) == len(detections) == len(detections.data["class_name"])
        assert detections.mask.any(axis=(1, 2)).any()


def _predict_tta(images: Any, tta: TTAConfig, threshold: Any = 0.0, run: Any = _grid_sensitive_run) -> Any:
    """Predict with the grid-sensitive model through ``tta=``, one image per call.

    Examples:
        >>> len(_predict_tta(_square_image(16), TTAConfig(shift=2)))
        1
    """
    return core._predict_batch(
        partial(_grid_sensitive_context, run), images, threshold, None, None, True, False, tta=tta
    )


class TestViews:
    """The views a ``TTAConfig`` expands to."""

    @pytest.mark.parametrize(
        ("config", "count"),
        [
            (TTAConfig(zoom=()), 1),
            (TTAConfig(), 4),
            pytest.param(TTAConfig(shift=1), 4, id="explicit-default-shift"),
            (TTAConfig(shift=4), 4),
            (TTAConfig(zoom=(1.25,)), 2),
            (TTAConfig(shift=4, zoom=(0.75, 1.5)), 6),
        ],
    )
    def test_view_count_adds_shifts_and_zooms(self, config: TTAConfig, count: int) -> None:
        assert view_count(config) == count
        assert len(build_views(config, _PATCH, (_SIDE, _SIDE))) == count

    def test_shift_only_views_are_todays_offsets(self) -> None:
        views = build_views(TTAConfig(shift=4), _PATCH, (_SIDE, _SIDE))
        assert [(v.scale, v.offset) for v in views] == [(1.0, o) for o in shift_offsets(_PATCH, 4)]

    def test_empty_zoom_without_shift_is_the_plain_path(self) -> None:
        assert len(build_views(TTAConfig(zoom=()), _PATCH, (_SIDE, _SIDE))) == 1

    def test_default_config_is_the_plain_image_and_the_default_zooms(self) -> None:
        config = TTAConfig()
        assert config.zoom == DEFAULT_ZOOM
        views = build_views(config, _PATCH, (_SIDE, _SIDE))
        assert [(v.scale, v.offset) for v in views] == [(1.0, 0)] + [(z, 0) for z in DEFAULT_ZOOM]

    def test_naming_shift_holds_no_default_zoom(self) -> None:
        assert TTAConfig(shift=2).zoom == ()

    def test_explicit_zoom_replaces_the_default_next_to_shift(self) -> None:
        assert TTAConfig(shift=2, zoom=(1.25,)).zoom == (1.25,)

    def test_zoom_that_places_the_image_at_the_base_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="the size of another view"):
            build_views(TTAConfig(zoom=(0.999,)), _PATCH, (_SIDE, _SIDE))

    def test_shift_above_the_patch_size_is_refused_with_the_shift_message(self) -> None:
        with pytest.raises(ValueError, match="TTAConfig.shift must be an integer from 1 to the patch size"):
            build_views(TTAConfig(shift=_PATCH + 1), _PATCH, (_SIDE, _SIDE))

    def test_canvas_is_one_block_larger_while_the_content_fits_else_the_block_multiple_it_needs(self) -> None:
        assert canvas_shape(_View(offset=5), (_SIDE, _SIDE), _PATCH) == (_SIDE + _PATCH, _SIDE + _PATCH)
        assert canvas_shape(_View(scale=1.25), (_SIDE, _SIDE), _PATCH) == (_SIDE + _PATCH, _SIDE + _PATCH)
        assert canvas_shape(_View(scale=1.5), (_SIDE, _SIDE), _PATCH) == (96, 96)


class TestPlaceAndMapRoundTrip:
    """``map_to_image`` undoes ``place_view`` for every combination of scale and offset."""

    @pytest.mark.parametrize("offset", [0, 5])
    @pytest.mark.parametrize("scale", [0.75, 1.0, 1.25, 1.5])
    def test_boxes_map_back_to_the_image(self, scale: float, offset: int) -> None:
        """A box at a known place in the image, moved forward by hand, maps back to that place."""
        h, w = 48, 64
        view = _View(scale=scale, offset=offset)
        canvas = canvas_shape(view, (h, w), _PATCH)
        ch, cw = round(scale * h), round(scale * w)
        x0, y0, x1, y1 = 10.0, 5.0, 30.0, 25.0
        # Scale to the placed content, then shift by the offset.
        px0, px1, py0, py1 = x0 * cw / w + offset, x1 * cw / w + offset, y0 * ch / h + offset, y1 * ch / h + offset
        canvas_box = torch.tensor(
            [(px0 + px1) / 2 / canvas[1], (py0 + py1) / 2 / canvas[0], (px1 - px0) / canvas[1], (py1 - py0) / canvas[0]]
        )
        mapped = map_to_image({"pred_boxes": canvas_box.view(1, 1, 4)}, view, (h, w), canvas, None)["pred_boxes"]
        expected = torch.tensor([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h])
        torch.testing.assert_close(mapped.view(4), expected, rtol=0, atol=1e-5)

    @pytest.mark.parametrize("offset", [0, 5])
    @pytest.mark.parametrize("scale", [1.0, 1.5])
    def test_an_asymmetric_mask_comes_back_in_place(self, scale: float, offset: int) -> None:
        """A blob in one corner of the image stays in that corner after placing, mask-head pooling and mapping back."""
        batch = torch.zeros(1, 1, _SIDE, _SIDE)
        batch[:, :, 8:24, 4:20] = 1.0
        view = _View(scale=scale, offset=offset)
        canvas = canvas_shape(view, (_SIDE, _SIDE), _PATCH)
        placed = place_view(batch, view, canvas)
        logits = torch.nn.functional.avg_pool2d(placed, 4) * 8 - 4
        mapped = map_to_image(
            {"pred_boxes": torch.zeros(1, 1, 4), "pred_masks": logits}, view, (_SIDE, _SIDE), canvas, (16, 16)
        )
        expected = torch.nn.functional.avg_pool2d(batch, 4) * 8 - 4

        def centroid(logit: torch.Tensor) -> torch.Tensor:
            rows, cols = (logit[0, 0] > 0).nonzero(as_tuple=True)
            return torch.stack([rows.float().mean(), cols.float().mean()])

        torch.testing.assert_close(centroid(mapped["pred_masks"]), centroid(expected), rtol=0, atol=0.75)


class TestGroupedForward:
    """Views of one canvas shape run as one forward pass, and fusion sees them in their original order."""

    @staticmethod
    def _counting_run() -> tuple[Any, list[int]]:
        """A grid-sensitive run that records the number of rows of each call.

        Examples:
            >>> run, rows = TestGroupedForward._counting_run()
            >>> _ = run(_square_image(16).unsqueeze(0))
            >>> rows
            [1]
        """
        rows: list[int] = []

        def run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
            rows.append(batch.shape[0])
            return _grid_sensitive_run(batch)

        return run, rows

    @pytest.mark.parametrize(
        ("config", "groups"),
        [
            (TTAConfig(shift=4), 1),
            (TTAConfig(shift=2, zoom=(1.5,)), 2),
            (TTAConfig(shift=4, zoom=(1.25, 1.5)), 2),
        ],
    )
    def test_one_forward_per_canvas_group_equals_the_per_view_loop(
        self, config: TTAConfig, groups: int, mocker: MockerFixture
    ) -> None:
        image = _square_image(20)
        loop = _predict_tta(image, config)  # CPU runs one view per forward
        run, rows = self._counting_run()
        mocker.patch.object(core, "_views_per_forward", side_effect=lambda _context, _rows, _canvas, views: views)
        batched = _predict_tta(image, config, run=run)
        assert len(rows) == groups
        assert sum(rows) == view_count(config)
        np.testing.assert_array_equal(batched.xyxy, loop.xyxy)
        np.testing.assert_array_equal(batched.confidence, loop.confidence)
        np.testing.assert_array_equal(batched.mask, loop.mask)

    def test_detections_are_decoded_once_per_forward_not_once_per_view(self, mocker: MockerFixture) -> None:
        """One top-k and one host copy serve every view of a forward, so the host sync count does not grow with V."""
        decode = mocker.spy(core, "_shift_candidates")
        run, rows = self._counting_run()
        mocker.patch.object(core, "_views_per_forward", side_effect=lambda _context, _rows, _canvas, views: views)
        _predict_tta(_square_image(20), TTAConfig(shift=8), run=run)
        assert rows == [8]
        assert decode.call_count == 1

    def test_short_chunk_keeps_distinct_images_in_order(self, mocker: MockerFixture) -> None:
        """Restore view and image rows after chunking interleaved canvas groups."""
        images = [_square_image(16), _square_image(26)[:, :56, :52]]
        config = TTAConfig(shift=6, zoom=(1.5,))
        expected = _predict_tta(images, config)
        run, rows = self._counting_run()
        mocker.patch.object(core, "_views_per_forward", return_value=4)
        actual = _predict_tta(images, config, run=run)
        assert rows == [8, 4, 2]
        assert [tuple(result.metadata["source_image"].shape[:2]) for result in actual] == [(64, 64), (56, 52)]
        np.testing.assert_array_equal(actual[0].xyxy, expected[0].xyxy)
        np.testing.assert_array_equal(actual[1].xyxy, expected[1].xyxy)
        np.testing.assert_array_equal(actual[0].confidence, expected[0].confidence)
        np.testing.assert_array_equal(actual[1].confidence, expected[1].confidence)
        np.testing.assert_array_equal(actual[0].mask, expected[0].mask)
        np.testing.assert_array_equal(actual[1].mask, expected[1].mask)

    def test_views_are_fused_in_their_original_order_whichever_group_runs_first(self, mocker: MockerFixture) -> None:
        """A zoom view that shares no canvas with the shifts still counts as the view it is in the config."""
        seen: list[list[sv.Detections]] = []
        original = core.group_passes

        def spy(passes: list[sv.Detections], *args: Any, **kwargs: Any) -> Any:
            """Retain the per-view inputs to fusion."""
            seen.append(passes)
            return original(passes, *args, **kwargs)

        def tag_view(outputs: dict[str, torch.Tensor], view: _View, *_args: Any) -> dict[str, torch.Tensor]:
            """Encode distinct view identities in the x coordinate independently of the model output."""
            center_x = 0.2 + 0.1 * (view.scale - 1) + view.offset / 100
            boxes = torch.tensor([center_x, 0.5, 0.1, 0.1]).expand_as(outputs["pred_boxes"])
            return {"pred_boxes": boxes}

        mocker.patch.object(core, "group_passes", side_effect=spy)
        mocker.patch.object(core, "map_to_image", side_effect=tag_view)
        mocker.patch.object(core, "_views_per_forward", return_value=3)
        _predict_tta(_square_image(20), TTAConfig(shift=2, zoom=(1.5,)))
        (passes,) = seen
        boxes = sv.Detections.merge(passes).xyxy
        np.testing.assert_allclose((boxes[:, 0] + boxes[:, 2]) / (2 * _SIDE), [0.2, 0.28, 0.25])

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        ("model_class", "resolution"),
        [pytest.param(RFDETRNano, 96, id="detection"), pytest.param(RFDETRSegNano, 72, id="segmentation")],
    )
    def test_public_cuda_batches_match_sequential_views(
        self, model_class: type, resolution: int, mocker: MockerFixture
    ) -> None:
        """Compare a CUDA stream's grouped predictions against one forward per view."""
        torch.manual_seed(0)
        model = model_class(
            pretrain_weights=None, device="cuda", resolution=resolution, num_queries=4, num_select=4, num_classes=2
        )
        predictor = RFDETRInference(model)
        images = [_square_image(16), _square_image(26)[:, :56, :52]]
        recipe = TTAConfig(shift=7, zoom=(1.5,))
        block = model.model_config.patch_size * model.model_config.num_windows
        mocker.patch.object(core, "_MAX_PIXELS_PER_FORWARD", 8 * (resolution + block) ** 2)
        forward = mocker.spy(model, "_run_inference_batch")
        actual = list(predictor.predict(images, stream=True, batch=2, tta=recipe, threshold=0.0))
        assert [call.args[0].shape[0] for call in forward.call_args_list] == [8, 6, 2]
        assert forward.call_args_list[0].args[0].shape[-2:] != forward.call_args_list[-1].args[0].shape[-2:]
        mocker.patch.object(core, "_views_per_forward", return_value=1)
        expected = list(predictor.predict(images, stream=True, batch=2, tta=recipe, threshold=0.0))
        assert len(actual) == len(expected) == 2
        self._assert_cuda_result(actual[0], expected[0], masks=model.model_config.segmentation_head)
        self._assert_cuda_result(actual[1], expected[1], masks=model.model_config.segmentation_head)

    @staticmethod
    def _assert_cuda_result(result: sv.Detections, reference: sv.Detections, *, masks: bool) -> None:
        """Allow accelerator rounding without accepting missing detections or changed mask geometry."""
        assert len(reference) > 0
        np.testing.assert_array_equal(result.class_id, reference.class_id)
        np.testing.assert_allclose(result.xyxy, reference.xyxy, rtol=1e-4, atol=5e-3)
        np.testing.assert_allclose(result.confidence, reference.confidence, rtol=1e-4, atol=1e-6)
        if masks:
            assert reference.mask is not None and result.mask is not None
            assert result.mask.shape == reference.mask.shape
            assert (result.mask != reference.mask).mean() < 1e-3


class TestZoomPadding:
    """Predictions must overlap the part of the zoom canvas that contains the source image."""

    def test_padding_only_box_is_dropped_but_clipped_box_survives(self) -> None:
        """Clip partial overlap at the source boundary and discard the zero-width candidate."""

        def run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
            """Return two high-confidence boxes in the smaller view and none in the other."""
            scores = [-float("inf"), -float("inf")] if batch[0, 0, 50, 50] else [2.0, 2.0]
            return {
                "pred_logits": torch.tensor(scores).view(1, 2, 1),
                "pred_boxes": torch.tensor([[[0.75, 0.3, 0.1, 0.2], [0.60, 0.3, 0.2, 0.2]]]),
            }

        actual = _predict_tta(torch.ones(3, 64, 64), TTAConfig(zoom=(0.75,)), threshold=0.1, run=run)
        np.testing.assert_allclose(actual.xyxy, [[160 / 3, 64 / 3, 64, 128 / 3]], atol=1e-5)

    def test_box_touching_padding_boundary_does_not_round_into_image(self) -> None:
        """Reject a box whose exact left edge touches the content boundary on an 80-pixel canvas."""

        def run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
            """Detect a box only in the 50-pixel zoom view."""
            score = -float("inf") if batch[0, 0, 51, 51] else 2.0
            return {
                "pred_logits": torch.tensor([[[score]]]),
                "pred_boxes": torch.tensor([[[0.626953125, 0.3, 0.00390625, 0.2]]]),
            }

        actual = _predict_tta(torch.ones(3, 64, 64), TTAConfig(zoom=(0.78125,)), threshold=0.1, run=run)
        assert len(actual) == 0


class TestChunkedForward:
    """A forward that runs out of memory is retried with half the views, down to one view."""

    @staticmethod
    def _oom_run(limit: int, rows: list[int]) -> Any:
        """A grid-sensitive run that records its rows and raises out of memory above ``limit`` rows.

        Examples:
            >>> rows: list[int] = []
            >>> run = TestChunkedForward._oom_run(2, rows)
            >>> _ = run(_square_image(16).unsqueeze(0))
            >>> rows
            [1]
        """

        def run(batch: torch.Tensor) -> dict[str, torch.Tensor]:
            rows.append(batch.shape[0])
            if batch.shape[0] > limit:
                raise torch.cuda.OutOfMemoryError("out of memory")
            return _grid_sensitive_run(batch)

        return run

    @pytest.fixture
    def all_views_in_one_forward(self, mocker: MockerFixture) -> None:
        """Make every group run as one forward first, the CUDA policy with an ample budget."""
        mocker.patch.object(core, "_views_per_forward", side_effect=lambda _context, _rows, _canvas, views: views)

    @pytest.mark.usefixtures("all_views_in_one_forward")
    def test_halves_the_views_until_the_forward_fits_and_keeps_the_result(self) -> None:
        image, config = _square_image(20), TTAConfig(shift=8)
        rows: list[int] = []
        actual = _predict_tta(image, config, run=self._oom_run(2, rows))
        assert rows == [8, 4, 2, 2, 2, 2]
        expected = _predict_tta(image, config)
        np.testing.assert_array_equal(actual.xyxy, expected.xyxy)
        np.testing.assert_array_equal(actual.confidence, expected.confidence)

    @pytest.mark.usefixtures("all_views_in_one_forward")
    def test_the_halved_size_holds_for_the_next_group_and_resets_for_the_next_request(self) -> None:
        image, config = _square_image(20), TTAConfig(shift=4, zoom=(1.4, 1.5))
        first: list[int] = []
        _predict_tta(image, config, run=self._oom_run(1, first))
        # Four shifts on one canvas, then two zooms on another that never see a larger chunk than the one that fit.
        assert first == [4, 2, 1, 1, 1, 1, 1, 1]
        second: list[int] = []
        _predict_tta(image, config, run=self._oom_run(1, second))
        assert second == first

    def test_a_single_view_that_runs_out_of_memory_raises(self) -> None:
        rows: list[int] = []
        with pytest.raises(torch.cuda.OutOfMemoryError):
            _predict_tta(_square_image(20), TTAConfig(shift=2), run=self._oom_run(0, rows))
        assert rows == [1]

    @pytest.mark.usefixtures("all_views_in_one_forward")
    @pytest.mark.parametrize("stream", [False, True])
    def test_view_limit_holds_across_batches_and_resets_for_a_new_request(self, stream: bool) -> None:
        """Retry once per request, including when prediction yields results lazily."""
        rows: list[int] = []
        context = partial(_grid_sensitive_context, self._oom_run(2, rows))
        images = [_square_image(20), _square_image(32)]
        tta = TTAConfig(shift=4)

        results = list(core.predict(context, images, batch=1, stream=stream, tta=tta))

        assert len(results) == 2
        assert rows == [4, 2, 2, 2, 2]

        rows.clear()
        list(core.predict(context, images[:1], batch=1, stream=stream, tta=tta))
        assert rows == [4, 2, 2]

    @staticmethod
    def _peak_reserved(predictor: RFDETRInference, images: list[torch.Tensor], recipe: TTAConfig) -> tuple[int, Any]:
        """Predict once from an empty allocator and return the peak reserved CUDA bytes and the result.

        Examples:
            >>> # The call below needs a CUDA device and a model, so the doctest skips it.
            >>> TestChunkedForward._peak_reserved  # doctest: +SKIP
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        (result,) = predictor.predict(images, tta=recipe, threshold=0.0)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_reserved(), result

    @pytest.mark.gpu
    @_requires_cuda
    @pytest.mark.parametrize(
        "model_class", [pytest.param(RFDETRNano, id="detection"), pytest.param(RFDETRSegNano, id="segmentation")]
    )
    def test_real_cuda_out_of_memory_is_retried_with_fewer_views(
        self, model_class: type, mocker: MockerFixture
    ) -> None:
        """Cap the allocator between the peak of one view and the peak of every view, so the grouped forward fails."""
        torch.manual_seed(0)
        model = model_class(pretrain_weights=None, device="cuda", num_queries=4, num_select=4, num_classes=2)
        predictor = RFDETRInference(model)
        images = [_square_image(16)]
        recipe = TTAConfig(shift=7, zoom=(1.5,))
        forward_rows: list[int] = []
        grouped_outputs: list[Any] = []
        run_forward, run_views = model._run_inference_batch, core._run_views

        def record_forward(batch: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            """Run the model and note the rows of every forward that finished."""
            outputs = run_forward(batch, *args, **kwargs)
            forward_rows.append(batch.shape[0])
            return outputs

        def record_views(*args: Any, **kwargs: Any) -> Any:
            """Run the views and keep what they returned, ``None`` for a forward that ran out of memory."""
            outputs = run_views(*args, **kwargs)
            grouped_outputs.append(outputs)
            return outputs

        mocker.patch.object(model, "_run_inference_batch", side_effect=record_forward)
        mocker.patch.object(core, "_run_views", side_effect=record_views)
        # A budget of one pixel gives one view per forward; a huge one gives the CUDA policy of every view at once.
        mocker.patch.object(core, "_MAX_PIXELS_PER_FORWARD", 1)
        self._peak_reserved(predictor, images, recipe)  # warm up kernels and workspaces before measuring
        forward_rows.clear()
        one_view_peak, reference = self._peak_reserved(predictor, images, recipe)
        views = len(forward_rows)
        mocker.patch.object(core, "_MAX_PIXELS_PER_FORWARD", 1 << 40)
        every_view_peak, _ = self._peak_reserved(predictor, images, recipe)
        if every_view_peak < 1.5 * one_view_peak:
            pytest.skip(f"no room between one view ({one_view_peak} B) and every view ({every_view_peak} B) to cap")

        forward_rows.clear()
        grouped_outputs.clear()
        device = torch.cuda.current_device()
        cap = (one_view_peak + every_view_peak) // 2
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(device).total_memory, device)
        try:
            (actual,) = predictor.predict(images, tta=recipe, threshold=0.0)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, device)
            torch.cuda.empty_cache()

        assert any(outputs is None for outputs in grouped_outputs), "the capped allocator never ran out of memory"
        assert max(forward_rows) < views, "a forward of every view finished under the cap"
        assert sum(forward_rows) == views, "a view ran twice or not at all"
        TestGroupedForward._assert_cuda_result(actual, reference, masks=model.model_config.segmentation_head)


class TestMaskSizeReduction:
    """The size every view's mask logits are resampled to, whatever the model's mask downsample ratio."""

    @pytest.mark.parametrize(
        ("ratio", "block", "side", "expected"), [(3, 32, 128, 42), (4, 32, 128, 32), (4, 16, 64, 16), (16, 24, 312, 20)]
    )
    def test_one_canvas_keeps_the_size_the_mask_head_gives(
        self, ratio: int, block: int, side: int, expected: int
    ) -> None:
        canvas = side + block
        masks = torch.zeros(1, 1, canvas // ratio, canvas // ratio)
        assert core._mask_size(masks, (side, side), (canvas, canvas), ratio, groups=1) == (expected, expected)

    @pytest.mark.parametrize(
        ("ratio", "block", "side", "expected"), [(3, 32, 128, 43), (4, 32, 128, 32), (16, 24, 312, 20)]
    )
    def test_several_canvases_share_the_size_of_the_image_itself(
        self, ratio: int, block: int, side: int, expected: int
    ) -> None:
        canvas = side + block
        masks = torch.zeros(1, 1, canvas // ratio, canvas // ratio)
        assert core._mask_size(masks, (side, side), (canvas, canvas), ratio, groups=2) == (expected, expected)


class TestSegmentationViews:
    """Masks of views on different canvases reach the vote at one size."""

    def test_masks_of_two_canvases_stack_at_the_ratio_derived_size(self) -> None:
        side, ratio, block = 312, 16, 24
        size = (side, side)
        mapped_masks = []
        for scale in (1.0, 1.5):
            view = _View(scale=scale)
            canvas = canvas_shape(view, size, block)
            head = torch.randn(1, 3, canvas[0] // ratio, canvas[1] // ratio)
            mask_size = core._mask_size(head, size, canvas, ratio, groups=2)
            outputs = {"pred_boxes": torch.zeros(1, 3, 4), "pred_masks": head}
            mapped_masks.append(map_to_image(outputs, view, size, canvas, mask_size)["pred_masks"])
        assert mapped_masks[0].shape == mapped_masks[1].shape == (1, 3, 20, 20)
        assert torch.stack(mapped_masks).shape == (2, 1, 3, 20, 20)


class TestZoomFusion:
    """No view drops content, so a missed view is a miss of the object and not of its visibility."""

    @pytest.mark.parametrize("scale", [0.75, 1.0, 1.25, 1.5])
    def test_the_placed_batch_holds_every_pixel_of_the_resampled_image(self, scale: float) -> None:
        batch = torch.rand(1, 3, _SIDE, _SIDE) + 1.0
        view = _View(scale=scale, offset=3)
        placed = place_view(batch, view, canvas_shape(view, (_SIDE, _SIDE), _PATCH))
        side = round(scale * _SIDE)
        expected = batch if scale == 1.0 else resize(batch, [side, side], antialias=False)
        torch.testing.assert_close(placed.sum(), expected.sum())

    def test_an_object_seen_by_every_view_keeps_the_mean_and_a_miss_counts_zero(self) -> None:
        seen = _detections([[0, 0, 10, 10]], [0.8], [0])
        missed = sv.Detections.empty()
        fused = fuse_passes([seen, seen, missed, seen])
        np.testing.assert_allclose(fused.confidence, [0.6])


class TestBatchedParity:
    """One forward over stacked views gives the raw outputs of one forward per view, on a real model."""

    @pytest.mark.parametrize("model_class", [RFDETRNano, RFDETRSegNano])
    def test_stacked_views_match_the_per_view_forwards(self, model_class: type) -> None:
        # Five queries keep two-stage top-k ties unlikely: a near-tied proposal can swap under last-bit differences,
        # which is not a batching defect.
        torch.manual_seed(0)
        model = model_class(pretrain_weights=None, num_queries=5, device="cpu")
        network = model.model.model.eval()
        patch_size, resolution = model.model_config.patch_size, model.model.resolution
        block = patch_size * model.model_config.num_windows
        size = (resolution, resolution)
        views = build_views(TTAConfig(shift=3, zoom=(1.25,)), patch_size, size)
        canvas = canvas_shape(views[0], size, block)
        batch = torch.randn(1, 3, *size)
        placed = [place_view(batch, v, canvas) for v in views if canvas_shape(v, size, block) == canvas]
        assert len(placed) > 1
        with torch.no_grad():
            stacked = network(torch.cat(placed))
            looped = [network(view) for view in placed]
        for key in ("pred_logits", "pred_boxes", "pred_masks"):
            if isinstance(stacked.get(key), torch.Tensor):
                torch.testing.assert_close(stacked[key], torch.cat([out[key] for out in looped]), rtol=1e-4, atol=1e-4)


class TestEarlyRefusal:
    """A request with several views is refused before any image is read when the runtime cannot take it."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [("fixed_shape", True), ("batch_size", 1), ("max_batch_size", 4), ("fixed_resolution", _SIDE)],
    )
    def test_stream_is_refused_when_predict_is_called_and_its_source_is_never_opened(
        self, field: str, value: Any, tmp_path: Any
    ) -> None:
        factory = partial(replace, _grid_sensitive_context(), **{field: value})
        with pytest.raises(ValueError, match="cannot run"):
            core.predict(factory, str(tmp_path / "missing.jpg"), stream=True, tta=TTAConfig(shift=2))

    @pytest.mark.parametrize("field", ["fixed_shape", "batch_size", "fixed_resolution"])
    def test_plain_request_is_not_refused(self, field: str) -> None:
        value = {"fixed_shape": True, "batch_size": 1, "fixed_resolution": _SIDE}[field]
        context = replace(_grid_sensitive_context(), **{field: value})
        for tta in (None, TTAConfig(zoom=())):
            _, _, views = core._plan_views(context, tta, None, None)
            assert len(views) == 1

    def test_live_model_context_accepts_several_views(self) -> None:
        _, _, views = core._plan_views(_grid_sensitive_context(), TTAConfig(shift=4), None, None)
        assert len(views) == 4

    def test_keypoint_model_is_refused_for_several_views_and_not_for_the_default(self) -> None:
        context = replace(_grid_sensitive_context(), num_keypoints_per_class=[17])
        with pytest.raises(ValueError, match="tta does not support keypoint models"):
            core._plan_views(context, TTAConfig(shift=2), None, None)
        for tta in (None, TTAConfig(zoom=())):
            assert len(core._plan_views(context, tta, None, None)[2]) == 1
        with pytest.raises(ValueError, match="tta does not support keypoint models"):
            core._plan_views(context, TTAConfig(), None, None)

    def test_several_views_skip_prepare_ahead(self) -> None:
        assert pipelines._may_prepare_ahead(True, "cuda", True, 1)
        assert not pipelines._may_prepare_ahead(True, "cuda", True, 4)
        assert not pipelines._may_prepare_ahead(False, "cuda", True, 1)

    def test_a_stream_runs_the_views_on_a_live_model_context(self) -> None:
        images = [_square_image(20), _square_image(20)]
        results = core.predict(_grid_sensitive_context, images, 0.0, batch=1, tta=TTAConfig(shift=2))
        assert [len(r) for r in results] == [len(_predict_tta(images[0], TTAConfig(shift=2)))] * 2


class TestTTAConfigValidation:
    """``TTAConfig`` rejects a value it cannot run, when it is built."""

    @pytest.mark.parametrize("shift", [0, -1, True, 2.0, "2"])
    def test_shift_must_be_a_positive_int(self, shift: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            TTAConfig(shift=shift)

    @pytest.mark.parametrize("zoom", ["1.25", {1.25}, {"z": 1.25}, 1.25])
    def test_zoom_must_be_a_list_or_tuple(self, zoom: Any) -> None:
        with pytest.raises(TypeError, match="zoom must be a list or tuple"):
            TTAConfig(zoom=zoom)

    @pytest.mark.parametrize(
        "zoom",
        [float("nan"), float("inf"), -float("inf"), 1.0, 0.5, 2.0, -1.0, pytest.param(10**400, id="large-integer")],
    )
    def test_zoom_values_must_be_finite_not_one_and_in_range(self, zoom: float) -> None:
        """Reject invalid values before converting accepted zooms to floats."""
        with pytest.raises(ValueError, match="TTAConfig.zoom values must be finite"):
            TTAConfig(zoom=(zoom,))

    def test_zoom_values_must_not_repeat(self) -> None:
        with pytest.raises(ValueError, match="must not repeat"):
            TTAConfig(zoom=(1.25, 1.25))

    def test_zoom_list_is_stored_as_a_tuple(self) -> None:
        assert TTAConfig(zoom=[1.25, 0.75]).zoom == (1.25, 0.75)

    def test_a_plain_dict_is_rejected_with_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="tta must be a TTAConfig or None"):
            core._predict_batch(
                _grid_sensitive_context, _square_image(16), 0.5, None, None, False, False, tta={"shift": 2}
            )
