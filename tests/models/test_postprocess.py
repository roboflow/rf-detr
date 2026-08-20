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
from rfdetr.utilities import box_ops


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

    def test_gathers_duplicate_and_out_of_order_indices(self):
        """Top-k can pick the same query under two classes; each pick must reproduce that query's exact scaled box.

        The box gather selects whole rows by query index, so duplicated and out-of-order indices must copy the source
        box verbatim for every occurrence. Mirrors the TRT-benchmark twin on the shipped PostProcess path.
        """
        out_bbox = torch.tensor(
            [
                [
                    [0.20, 0.30, 0.10, 0.10],
                    [0.50, 0.50, 0.20, 0.20],
                    [0.70, 0.40, 0.15, 0.25],
                ]
            ]
        )  # (B=1, Q=3, 4); all in-bounds so clamping is a no-op and the copy is exact
        topk_boxes = torch.tensor([[2, 2, 1]])  # query 2 selected twice, then query 1 (out of order)
        target_sizes = torch.tensor([[480, 640]])  # (h, w)

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        # Expected computed independently from raw input, not by re-invoking the function under test.
        scale = torch.tensor([640.0, 480.0, 640.0, 480.0])
        expected = box_ops.box_cxcywh_to_xyxy(out_bbox[0]) * scale
        assert torch.equal(boxes[0, 0], expected[2])
        assert torch.equal(boxes[0, 1], expected[2])
        assert torch.equal(boxes[0, 2], expected[1])

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gather_and_scale_boxes_cuda_matches_cpu_for_duplicated_indices(self):
        """Box selection is an arithmetic-free row copy, so CUDA must reproduce the CPU result bit-for-bit even with
        duplicated and out-of-order indices."""
        out_bbox = torch.tensor(
            [
                [
                    [0.20, 0.30, 0.10, 0.10],
                    [0.50, 0.50, 0.20, 0.20],
                    [0.70, 0.40, 0.15, 0.25],
                ]
            ]
        )
        topk_boxes = torch.tensor([[2, 2, 1]])
        target_sizes = torch.tensor([[480, 640]])

        cpu_boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)
        cuda_boxes = PostProcess._gather_and_scale_boxes(out_bbox.cuda(), topk_boxes.cuda(), target_sizes.cuda())

        assert torch.equal(cpu_boxes, cuda_boxes.cpu())


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

    @pytest.mark.parametrize(
        ("batch", "upsample", "expected_calls"),
        [
            pytest.param(4, True, [(4, 2)], id="upsampled-batch"),
            pytest.param(4, False, [], id="native-resolution"),
            pytest.param(0, True, [], id="empty-upsampled-batch"),
        ],
    )
    def test_target_sizes_are_read_once_per_upsampled_batch(
        self, batch: int, upsample: bool, expected_calls: list[tuple[int, ...]]
    ) -> None:
        """A non-empty upsampled batch reads all target sizes together; other paths do not read them."""
        out_masks = torch.randn(batch, 1, 2, 2)
        scores = torch.ones(batch, 1)
        labels = torch.zeros(batch, 1, dtype=torch.long)
        boxes = torch.zeros(batch, 1, 4)
        topk_boxes = torch.zeros(batch, 1, dtype=torch.long)
        target_sizes = torch.full((batch, 2), 8, dtype=torch.long)
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        with patch.object(torch.Tensor, "tolist", tracked_tolist):
            PostProcess._postprocess_masks(
                out_masks,
                scores,
                labels,
                boxes,
                topk_boxes,
                target_sizes,
                upsample_masks_to_image_size=upsample,
            )

        assert calls == expected_calls

    def test_upsampled_batch_pairs_non_square_target_sizes_with_their_mask_rows(self) -> None:
        """Each image keeps its own non-square target size and mask geometry after batched size conversion.

        This prevents a regression where ``target_sizes.tolist()`` is batched but the resulting rows are swapped or
        height/width are transposed before per-image interpolation.
        """
        out_masks = torch.tensor(
            [
                [[[5.0, -5.0, -5.0], [5.0, -5.0, -5.0]]],
                [[[-5.0, -5.0, 5.0], [-5.0, -5.0, 5.0]]],
            ]
        )
        scores = torch.tensor([[0.9], [0.8]])
        labels = torch.tensor([[1], [2]])
        boxes = torch.zeros(2, 1, 4)
        topk_boxes = torch.zeros(2, 1, dtype=torch.long)
        target_sizes = torch.tensor([[4, 9], [9, 4]])  # (H, W): landscape, then portrait

        results = PostProcess._postprocess_masks(out_masks, scores, labels, boxes, topk_boxes, target_sizes)

        expected_masks = [
            torch.nn.functional.interpolate(
                out_masks[0, 0][None, None], size=(4, 9), mode="bilinear", align_corners=False
            )
            > 0.0,
            torch.nn.functional.interpolate(
                out_masks[1, 0][None, None], size=(9, 4), mode="bilinear", align_corners=False
            )
            > 0.0,
        ]
        assert results[0]["masks"].shape == (1, 1, 4, 9)
        assert results[1]["masks"].shape == (1, 1, 9, 4)
        assert torch.equal(results[0]["masks"], expected_masks[0])
        assert torch.equal(results[1]["masks"], expected_masks[1])

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_target_sizes_are_read_once_per_upsampled_batch_cuda(self) -> None:
        """Same call-count guarantee as ``test_target_sizes_are_read_once_per_upsampled_batch`` above, with every input
        tensor on CUDA — the actual site of the device-to-host synchronization this PR collapses from one per image to
        one per batch."""
        out_masks = torch.randn(4, 1, 2, 2, device="cuda")
        scores = torch.ones(4, 1, device="cuda")
        labels = torch.zeros(4, 1, dtype=torch.long, device="cuda")
        boxes = torch.zeros(4, 1, 4, device="cuda")
        topk_boxes = torch.zeros(4, 1, dtype=torch.long, device="cuda")
        target_sizes = torch.full((4, 2), 8, dtype=torch.long, device="cuda")
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        with patch.object(torch.Tensor, "tolist", tracked_tolist):
            PostProcess._postprocess_masks(
                out_masks,
                scores,
                labels,
                boxes,
                topk_boxes,
                target_sizes,
                upsample_masks_to_image_size=True,
            )

        assert calls == [(4, 2)]

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

    def test_duplicate_query_selection_repeats_upsampled_masks_across_chunk_boundary(self):
        """With upsampling on (production default), duplicated query indices must yield identical resized masks even
        when the two occurrences fall in different _MASK_CHUNK interpolation chunks.

        The gather runs once before the chunked resize, so a duplicate straddling the 32-row chunk boundary exercises
        that the per-chunk interpolation stays row-independent and reproduces the source plane for every occurrence.
        """
        num_queries, num_select = 4, 48  # > _MASK_CHUNK (32) so the resize loop spans two chunks
        out_masks = torch.randn(1, num_queries, 8, 8)
        idx = [0] * num_select
        idx[0] = 2  # query 2 in chunk 0
        idx[40] = 2  # same query in chunk 1 (straddles the 32-row boundary)
        topk_boxes = torch.tensor([idx])
        scores = torch.rand(1, num_select)
        labels = torch.zeros(1, num_select, dtype=torch.long)
        boxes = torch.zeros(1, num_select, 4)
        target_sizes = torch.tensor([[64, 64]])

        results = PostProcess._postprocess_masks(
            out_masks, scores, labels, boxes, topk_boxes, target_sizes, upsample_masks_to_image_size=True
        )

        masks = results[0]["masks"]  # [K, 1, 64, 64] bool
        assert masks.shape == (num_select, 1, 64, 64)
        # Independent reference: resize query 2's source plane the same way the production path does.
        expected = (
            torch.nn.functional.interpolate(
                out_masks[0, 2][None, None], size=(64, 64), mode="bilinear", align_corners=False
            )
            > 0.0
        )[0]
        assert torch.equal(masks[0], masks[40])  # both occurrences identical across the chunk boundary
        assert torch.equal(masks[0], expected)  # and each maps to the correct source query, not query 0

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


class TestAttachEmbeddings:
    """Tests for :meth:`PostProcess._attach_embeddings` and its wiring into ``forward``."""

    def test_forward_omits_embeddings_key_when_absent_from_outputs(self) -> None:
        """No 'embeddings' key is added to results when outputs has no 'embeddings' entry."""
        pp = PostProcess(num_select=2)
        outputs = {
            "pred_logits": torch.tensor([[[10.0, -10.0], [9.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]]),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = pp(outputs, target_sizes)

        assert "embeddings" not in results[0]

    def test_forward_attaches_embeddings_with_expected_shape(self) -> None:
        """Embeddings in outputs is gathered per-selected-query and attached with shape (K, H)."""
        hidden_dim = 4
        pp = PostProcess(num_select=2)
        outputs = {
            "pred_logits": torch.tensor([[[10.0, -10.0], [9.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]]]),
            "embeddings": torch.arange(2 * hidden_dim, dtype=torch.float32).reshape(1, 2, hidden_dim),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = pp(outputs, target_sizes)

        assert "embeddings" in results[0]
        assert results[0]["embeddings"].shape == (2, hidden_dim)

    def test_attach_embeddings_gathers_by_topk_indices(self) -> None:
        """_attach_embeddings must select embeddings for the exact queries in topk_boxes, in that order."""
        hidden_dim = 3
        # 4 queries per image, embeddings identifiable by their first column value == query index.
        out_embeddings = torch.zeros(1, 4, hidden_dim)
        for q in range(4):
            out_embeddings[0, q, 0] = q

        # Select queries [2, 0] (reversed, non-contiguous) as the "top-k" selection.
        topk_boxes = torch.tensor([[2, 0]])
        results: list[dict[str, torch.Tensor]] = [{}]

        PostProcess._attach_embeddings(results, out_embeddings, topk_boxes)

        gathered = results[0]["embeddings"]
        assert gathered.shape == (2, hidden_dim)
        assert gathered[0, 0].item() == 2
        assert gathered[1, 0].item() == 0

    def test_attach_embeddings_batch_of_two_images(self) -> None:
        """Embeddings are gathered independently per image in the batch."""
        out_embeddings = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # image 0
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],  # image 1
            ]
        )
        topk_boxes = torch.tensor([[1, 0], [2, 1]])
        results: list[dict[str, torch.Tensor]] = [{}, {}]

        PostProcess._attach_embeddings(results, out_embeddings, topk_boxes)

        assert torch.equal(results[0]["embeddings"], torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
        assert torch.equal(results[1]["embeddings"], torch.tensor([[12.0, 12.0], [11.0, 11.0]]))

    def test_forward_attaches_embeddings_alongside_masks(self) -> None:
        """Embeddings and masks can coexist in the same forward() call without interfering with each other."""
        batch, num_queries, mask_h, mask_w, hidden_dim, num_classes = 1, 4, 8, 8, 4, 2
        pp = PostProcess(num_select=4)
        outputs = {
            "pred_logits": torch.randn(batch, num_queries, num_classes),
            "pred_boxes": torch.rand(batch, num_queries, 4),
            "pred_masks": torch.randn(batch, num_queries, mask_h, mask_w),
            "embeddings": torch.randn(batch, num_queries, hidden_dim),
        }
        target_sizes = torch.tensor([[256, 256]])

        results = pp(outputs, target_sizes)

        assert "masks" in results[0]
        assert "embeddings" in results[0]
        assert results[0]["embeddings"].shape == (4, hidden_dim)

    def test_forward_filters_embeddings_by_score_threshold_with_masks(self) -> None:
        """When masks are filtered by score_threshold, embeddings must be filtered the same way.

        ``_postprocess_masks`` drops rows scoring at or below ``score_threshold`` before returning results, so
        ``scores``/``labels``/``boxes``/``masks`` all end up with fewer rows than the raw top-k selection. Embeddings
        must be filtered with the same per-image predicate, or the row counts diverge and callers indexing the
        embeddings tensor with a boolean mask sized to the filtered scores hit a shape mismatch.
        """
        hidden_dim = 4
        pp = PostProcess(num_select=3)
        # Deterministic sigmoid scores: query 0 -> ~1.0 (kept), query 1 -> ~0.0 (dropped), query 2 -> ~0.99 (kept).
        logits = torch.tensor([[[10.0], [-10.0], [5.0]]])
        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.rand(1, 3, 4),
            "pred_masks": torch.randn(1, 3, 8, 8),
            "embeddings": torch.arange(3 * hidden_dim, dtype=torch.float32).reshape(1, 3, hidden_dim),
        }
        target_sizes = torch.tensor([[256, 256]])

        results = pp(outputs, target_sizes, score_threshold=0.5)

        assert results[0]["embeddings"].shape[0] == results[0]["scores"].shape[0] == 2
        assert results[0]["masks"].shape[0] == 2
        # Query 1 (logit -10) is below threshold and must be dropped from embeddings too.
        kept_first_values = {row[0].item() for row in results[0]["embeddings"]}
        assert kept_first_values == {0.0, 2 * hidden_dim}
