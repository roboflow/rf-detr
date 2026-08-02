# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for PostProcess box clamping behaviour."""

from unittest.mock import patch

import pytest
import torch

from rfdetr.models.postprocess import PostProcess


class TestGatherAndScaleBoxes:
    """Tests for :meth:`PostProcess._gather_and_scale_boxes`."""

    def test_clamps_boxes_to_image_bounds(self):
        """Boxes that extrapolate beyond [0, 1] in normalized space are clamped to pixel-space image dimensions after
        scaling."""
        # Three synthetic boxes in cxcywh normalized coords:
        #   [0] cx=0.01, w=0.10 → x1 = (0.01 - 0.05) * 640 = -25.6  ← negative
        #   [1] cx=0.99, w=0.10 → x2 = (0.99 + 0.05) * 640 = 665.6  ← overflow
        #   [2] cx=0.50, w=0.20 → fully in-bounds
        out_bbox = torch.tensor(
            [
                [
                    [0.01, 0.01, 0.10, 0.10],  # negative x1, y1 after scale
                    [0.99, 0.99, 0.10, 0.10],  # x2 > img_w, y2 > img_h after scale
                    [0.50, 0.50, 0.20, 0.20],  # in-bounds control
                ]
            ]
        )  # shape (B=1, Q=3, 4)

        topk_boxes = torch.tensor([[0, 1, 2]])  # select all three
        target_sizes = torch.tensor([[480, 640]])  # (h, w)

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        img_h, img_w = 480, 640

        # All coords must be >= 0
        assert (boxes >= 0).all(), f"Negative coords present: {boxes[boxes < 0]}"
        # x1, x2 must be <= image width
        assert (boxes[..., 0] <= img_w).all()
        assert (boxes[..., 2] <= img_w).all()
        # y1, y2 must be <= image height
        assert (boxes[..., 1] <= img_h).all()
        assert (boxes[..., 3] <= img_h).all()

        # Exact clamped values — bounds-only check cannot catch a clamp returning e.g. 1.0 instead of 0.0
        # box [0]: x1_raw=-25.6, y1_raw=-19.2 → clamped to 0.0
        assert boxes[0, 0, 0].item() == pytest.approx(0.0), "x1 of underflowing box must clamp to 0"
        assert boxes[0, 0, 1].item() == pytest.approx(0.0), "y1 of underflowing box must clamp to 0"
        # box [1]: x2_raw=665.6 → clamped to img_w=640.0; y2_raw=499.2 → clamped to img_h=480.0
        assert boxes[0, 1, 2].item() == pytest.approx(640.0), "x2 of overflowing box must clamp to img_w"
        assert boxes[0, 1, 3].item() == pytest.approx(480.0), "y2 of overflowing box must clamp to img_h"

    def test_in_bounds_boxes_unchanged(self):
        """Boxes already within image bounds are not altered by clamping."""
        out_bbox = torch.tensor(
            [
                [
                    [0.30, 0.30, 0.20, 0.20],
                    [0.70, 0.60, 0.30, 0.40],
                ]
            ]
        )

        topk_boxes = torch.tensor([[0, 1]])
        target_sizes = torch.tensor([[480, 640]])

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        # Manually computed expected values (no clamping needed)
        expected = torch.tensor(
            [
                [
                    [128.0, 96.0, 256.0, 192.0],  # cx=0.30,cy=0.30,w=0.20,h=0.20
                    [352.0, 192.0, 544.0, 384.0],  # cx=0.70,cy=0.60,w=0.30,h=0.40
                ]
            ]
        )

        assert torch.allclose(boxes, expected, atol=1e-4), (
            f"In-bounds boxes were altered.\nExpected:\n{expected}\nGot:\n{boxes}"
        )

    def test_multiple_images_in_batch(self):
        """Clamping works correctly across a batch of mixed image sizes."""
        out_bbox = torch.tensor(
            [
                [
                    [0.01, 0.50, 0.10, 0.20],  # image 0: negative x1
                ],
                [
                    [0.99, 0.50, 0.10, 0.20],  # image 1: x2 overflow
                ],
            ]
        )

        topk_boxes = torch.tensor([[0], [0]])
        target_sizes = torch.tensor(
            [
                [300, 400],  # image 0: 400×300
                [600, 800],  # image 1: 800×600
            ]
        )

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        # Image 0: all coords must be in [0, 400]×[0, 300]
        assert (boxes[0, :, 0] >= 0).all(), "img0 x1: expected >= 0"
        assert (boxes[0, :, 0] <= 400).all(), "img0 x1: expected <= img_w (400)"
        assert (boxes[0, :, 1] >= 0).all(), "img0 y1: expected >= 0"
        assert (boxes[0, :, 1] <= 300).all(), "img0 y1: expected <= img_h (300)"
        assert (boxes[0, :, 2] >= 0).all(), "img0 x2: expected >= 0"
        assert (boxes[0, :, 2] <= 400).all(), "img0 x2: expected <= img_w (400)"
        assert (boxes[0, :, 3] >= 0).all(), "img0 y2: expected >= 0"
        assert (boxes[0, :, 3] <= 300).all(), "img0 y2: expected <= img_h (300)"

        # Image 1: all coords must be in [0, 800]×[0, 600]
        assert (boxes[1, :, 0] >= 0).all(), "img1 x1: expected >= 0"
        assert (boxes[1, :, 0] <= 800).all(), "img1 x1: expected <= img_w (800)"
        assert (boxes[1, :, 1] >= 0).all(), "img1 y1: expected >= 0"
        assert (boxes[1, :, 1] <= 600).all(), "img1 y1: expected <= img_h (600)"
        assert (boxes[1, :, 2] >= 0).all(), "img1 x2: expected >= 0"
        assert (boxes[1, :, 2] <= 800).all(), "img1 x2: expected <= img_w (800)"
        assert (boxes[1, :, 3] >= 0).all(), "img1 y2: expected >= 0"
        assert (boxes[1, :, 3] <= 600).all(), "img1 y2: expected <= img_h (600)"


class TestPostProcessForward:
    """Integration tests for :meth:`PostProcess.forward`."""

    def test_forward_clamps_edge_boxes_to_bounds(self):
        """PostProcess.forward returns non-negative in-bounds boxes for edge-hugging predictions."""
        postprocess = PostProcess(num_select=2)
        outputs = {
            "pred_logits": torch.tensor([[[10.0, -10.0], [9.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.01, 0.01, 0.10, 0.10], [0.99, 0.99, 0.10, 0.10]]]),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = postprocess(outputs, target_sizes)
        boxes = results[0]["boxes"]
        assert (boxes >= 0).all(), f"Negative coords present: {boxes[boxes < 0]}"
        assert (boxes[..., 0] <= 640).all(), "x1 exceeds img_w (640)"
        assert (boxes[..., 2] <= 640).all(), "x2 exceeds img_w (640)"
        assert (boxes[..., 1] <= 480).all(), "y1 exceeds img_h (480)"
        assert (boxes[..., 3] <= 480).all(), "y2 exceeds img_h (480)"


class TestPostProcessMasks:
    """Tests for :meth:`PostProcess._postprocess_masks` mask resizing."""

    def test_chunked_upsample_preserves_shape_for_large_k(self):
        """Chunked upsampling of K=64 masks returns full-resolution boolean masks of shape [K, 1, H, W]."""
        batch, num_queries, mask_h, mask_w = 1, 64, 16, 16
        num_select, img_h, img_w = 64, 512, 512
        out_masks = torch.randn(batch, num_queries, mask_h, mask_w)
        scores = torch.rand(batch, num_select)
        labels = torch.zeros(batch, num_select, dtype=torch.long)
        boxes = torch.zeros(batch, num_select, 4)
        topk_boxes = torch.arange(num_select).unsqueeze(0)
        target_sizes = torch.tensor([[img_h, img_w]])

        results = PostProcess._postprocess_masks(out_masks, scores, labels, boxes, topk_boxes, target_sizes)

        masks = results[0]["masks"]
        assert masks.shape == (num_select, 1, img_h, img_w)
        assert masks.dtype == torch.bool

    def test_native_resolution_skips_upsample_when_flag_false(self):
        """upsample_masks_to_image_size=False returns masks at native mask-head resolution instead of target_sizes (opt-
        in validation-cost reduction, see TrainConfig.eval_masks_head_resolution)."""
        batch, num_queries, mask_h, mask_w = 1, 8, 16, 16
        num_select, img_h, img_w = 8, 512, 512
        out_masks = torch.randn(batch, num_queries, mask_h, mask_w)
        scores = torch.rand(batch, num_select)
        labels = torch.zeros(batch, num_select, dtype=torch.long)
        boxes = torch.zeros(batch, num_select, 4)
        topk_boxes = torch.arange(num_select).unsqueeze(0)
        target_sizes = torch.tensor([[img_h, img_w]])

        results = PostProcess._postprocess_masks(
            out_masks, scores, labels, boxes, topk_boxes, target_sizes, upsample_masks_to_image_size=False
        )

        masks = results[0]["masks"]
        assert masks.shape == (num_select, 1, mask_h, mask_w)
        assert masks.dtype == torch.bool

    def test_native_resolution_thresholds_at_zero_same_as_upsampled_path(self):
        """Native-resolution masks apply the same logit > 0.0 threshold as the upsampled path."""
        out_masks = torch.tensor([[[[5.0, -5.0], [-1.0, 1.0]]]])  # [B=1, Q=1, Hm=2, Wm=2]
        scores = torch.tensor([[0.9]])
        labels = torch.zeros(1, 1, dtype=torch.long)
        boxes = torch.zeros(1, 1, 4)
        topk_boxes = torch.tensor([[0]])
        target_sizes = torch.tensor([[100, 100]])

        results = PostProcess._postprocess_masks(
            out_masks, scores, labels, boxes, topk_boxes, target_sizes, upsample_masks_to_image_size=False
        )

        expected = torch.tensor([[True, False], [False, True]])
        assert torch.equal(results[0]["masks"].squeeze(1)[0], expected)

    def test_duplicate_query_selection_repeats_the_same_mask_rows(self):
        """Top-k can pick the same query under two classes; each pick must yield that query's exact mask.

        The mask gather selects whole planes by query index, so duplicated and out-of-order indices must reproduce the
        source plane verbatim for every occurrence.
        """
        out_masks = torch.randn(1, 4, 8, 8)
        scores = torch.rand(1, 3)
        labels = torch.tensor([[0, 1, 0]])
        boxes = torch.zeros(1, 3, 4)
        topk_boxes = torch.tensor([[2, 2, 1]])  # query 2 selected twice (two classes), then query 1
        target_sizes = torch.tensor([[256, 256]])

        results = PostProcess._postprocess_masks(
            out_masks, scores, labels, boxes, topk_boxes, target_sizes, upsample_masks_to_image_size=False
        )

        masks = results[0]["masks"].squeeze(1)
        expected = out_masks[0] > 0.0
        assert torch.equal(masks[0], expected[2])
        assert torch.equal(masks[1], expected[2])
        assert torch.equal(masks[2], expected[1])

    def test_forward_threads_upsample_flag_from_constructor(self):
        """PostProcess.forward() must respect the constructor's upsample_masks_to_image_size setting."""
        batch, num_queries, mask_h, mask_w = 1, 4, 8, 8
        num_classes = 2
        pp = PostProcess(num_select=4, upsample_masks_to_image_size=False)
        outputs = {
            "pred_logits": torch.randn(batch, num_queries, num_classes),
            "pred_boxes": torch.rand(batch, num_queries, 4),
            "pred_masks": torch.randn(batch, num_queries, mask_h, mask_w),
        }
        target_sizes = torch.tensor([[256, 256]])

        results = pp(outputs, target_sizes)

        assert results[0]["masks"].shape[-2:] == (mask_h, mask_w)

    @staticmethod
    def _mask_case(num_select: int = 16, num_queries: int = 16, mask_hw: int = 8, batch: int = 1):
        """Return (out_masks, scores, labels, boxes, topk_boxes, target_sizes) with known scores.

        Scores descend within each image so a threshold of 0.5 keeps a known prefix, and each image starts lower than
        the previous one so a batch keeps a different number of rows per image.
        """
        out_masks = torch.randn(batch, num_queries, mask_hw, mask_hw)
        scores = torch.stack([torch.linspace(0.95 - 0.3 * i, 0.05, num_select) for i in range(batch)])
        labels = torch.zeros(batch, num_select, dtype=torch.long)
        boxes = torch.zeros(batch, num_select, 4)
        topk_boxes = torch.arange(num_select).unsqueeze(0).repeat(batch, 1)
        target_sizes = torch.tensor([[64, 64]]).repeat(batch, 1)
        return out_masks, scores, labels, boxes, topk_boxes, target_sizes

    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    def test_score_threshold_upsamples_only_the_kept_masks(self, threshold):
        """Masks that the caller's threshold discards must never reach the interpolation.

        The upsample runs at target-image resolution while the kept fraction is small (at ``num_select=100`` a COCO-
        style image keeps a handful), so resizing the discarded rows is work whose result is dropped a few lines later.
        Counting the interpolated rows rather than timing them keeps the test deterministic. A threshold of 1.0 keeps
        nothing and must skip the interpolation altogether rather than resize an empty tensor.

        ``num_select=100`` (> ``_MASK_CHUNK`` = 32) keeps enough rows above 0.5 to span several interpolation chunks,
        so the count also proves the chunked upsample loop stays correct under early filtering.
        """
        args = self._mask_case(num_select=100, num_queries=100)
        scores = args[1]
        expected_kept = int((scores[0] > threshold).sum())

        rows = []
        orig = torch.nn.functional.interpolate

        def counting_interpolate(tensor, *a, **kw):
            rows.append(tensor.shape[0])
            return orig(tensor, *a, **kw)

        with patch("rfdetr.models.postprocess.F.interpolate", counting_interpolate):
            results = PostProcess._postprocess_masks(*args, score_threshold=threshold)

        assert sum(rows) == expected_kept, (
            f"interpolated {sum(rows)} mask rows but only {expected_kept} survive the "
            "caller's threshold; masks below it must be dropped before the resize"
        )
        assert results[0]["masks"].shape[0] == expected_kept

    @pytest.mark.parametrize("upsample", [True, False])
    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    def test_score_threshold_matches_filtering_after_upsampling(self, threshold, upsample):
        """Filtering before the resize must return exactly what filtering after it returns.

        The filter sits above the ``upsample_masks_to_image_size`` branch, so the native-resolution path has to drop the
        same rows as the resized one. A threshold of 1.0 keeps nothing and pins the shape and dtype of the empty result
        on both branches.
        """
        args = self._mask_case()

        filtered_early = PostProcess._postprocess_masks(*args, upsample, score_threshold=threshold)[0]
        full = PostProcess._postprocess_masks(*args, upsample)[0]
        keep = full["scores"] > threshold

        for key in ("scores", "labels", "boxes"):
            assert torch.equal(filtered_early[key], full[key][keep])
        assert torch.equal(filtered_early["masks"], full["masks"][keep])

    def test_score_threshold_filters_each_image_independently(self):
        """A batch keeps a different number of rows per image, not one count applied to all of them."""
        args = self._mask_case(batch=2)
        scores = args[1]
        threshold = 0.5

        results = PostProcess._postprocess_masks(*args, score_threshold=threshold)

        for i, result in enumerate(results):
            keep = scores[i] > threshold
            assert torch.equal(result["scores"], scores[i][keep])
            assert result["masks"].shape[0] == int(keep.sum())
        assert results[0]["masks"].shape[0] > results[1]["masks"].shape[0], (
            "the second image scores lower and must keep fewer masks; equal counts mean the "
            "threshold was applied batch-wide instead of per image"
        )

    @pytest.mark.parametrize("head", ["boxes", "keypoints"])
    def test_score_threshold_is_ignored_outside_the_mask_path(self, head):
        """The box-only and keypoint heads must return the same thing with and without the argument.

        ``predict()`` passes the threshold for every model, so the two heads that cannot use it have to ignore it rather
        than fail. The keypoint path rewrites scores after selection (uncertainty fusion), so filtering on the pre-
        fusion scores would not match the caller's own filter, and the box-only path has no per-detection resize to
        skip.
        """
        batch, num_queries, num_classes = 1, 4, 2
        outputs = {
            "pred_logits": torch.randn(batch, num_queries, num_classes),
            "pred_boxes": torch.rand(batch, num_queries, 4),
        }
        kwargs = {}
        if head == "keypoints":
            # (B, Q, num_keypoint_classes * max_num_keypoints, D) with D >= 7 so precision is emitted.
            outputs["pred_keypoints"] = torch.randn(batch, num_queries, 6, 7)
            kwargs["num_keypoints_per_class"] = [3, 3]
        postprocess = PostProcess(num_select=num_queries, **kwargs)
        target_sizes = torch.tensor([[128, 128]])

        baseline = postprocess(outputs, target_sizes)[0]
        with_threshold = postprocess(outputs, target_sizes, score_threshold=0.9)[0]

        assert baseline.keys() == with_threshold.keys()
        for key in baseline:
            torch.testing.assert_close(baseline[key], with_threshold[key], rtol=0.0, atol=0.0, equal_nan=True)
