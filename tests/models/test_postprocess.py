# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for PostProcess box clamping behaviour."""

import weakref

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models import postprocess
from rfdetr.models.postprocess import PostProcess
from rfdetr.utilities import box_ops
from tests._markers import requires_torch_xla

_skip_without_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _cyclic_mask_case(
    num_queries: int,
    num_select: int,
    mask_hw: int = 8,
    size: int = 64,
    device: str | torch.device = "cpu",
    target_device: str | torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return ``_postprocess_masks`` arguments for one image whose top-k cycles through the queries.

    Args:
        num_queries: Number of mask-head queries.
        num_select: Number of selected detections; query ``i % num_queries`` is picked for detection ``i``.
        mask_hw: Side of the square native mask-head resolution.
        size: Side of the square target image size.
        device: Device of every tensor except possibly ``target_sizes``.
        target_device: Device of ``target_sizes``; defaults to ``device``.

    Returns:
        ``(out_masks, scores, labels, boxes, topk_boxes, target_sizes)``.

    Examples:
        >>> out_masks, _, _, _, topk_boxes, target_sizes = _cyclic_mask_case(4, 6)
        >>> out_masks.shape, topk_boxes.tolist(), target_sizes.tolist()
        (torch.Size([1, 4, 8, 8]), [[0, 1, 2, 3, 0, 1]], [[64, 64]])
    """
    target_device = device if target_device is None else target_device
    out_masks = torch.randn(1, num_queries, mask_hw, mask_hw, device=device)
    scores = torch.rand(1, num_select, device=device)
    labels = torch.zeros(1, num_select, dtype=torch.long, device=device)
    boxes = torch.zeros(1, num_select, 4, device=device)
    topk_boxes = (torch.arange(num_select) % num_queries).unsqueeze(0).to(device)
    target_sizes = torch.tensor([[size, size]], device=target_device)
    return out_masks, scores, labels, boxes, topk_boxes, target_sizes


def _chunked_upsampled_masks(out_masks: torch.Tensor, topk_boxes: torch.Tensor, size: int) -> torch.Tensor:
    """Return the selected masks resized chunk by chunk to ``size`` x ``size`` and thresholded at 0 (reference result).

    Examples:
        >>> _chunked_upsampled_masks(torch.ones(1, 2, 4, 4), torch.tensor([[0, 1, 0]]), 8).shape
        torch.Size([3, 1, 8, 8])
    """
    selected = out_masks[0].index_select(0, topk_boxes[0])
    chunk = postprocess._MASK_CHUNK
    return torch.cat(
        [
            F.interpolate(
                selected[start : start + chunk].unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False
            )
            > 0.0
            for start in range(0, selected.shape[0], chunk)
        ],
        dim=0,
    )


class TestGatherAndScaleBoxes:
    """Tests for :meth:`PostProcess._gather_and_scale_boxes`."""

    @pytest.fixture
    def duplicate_query_boxes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return in-bounds ``(out_bbox, topk_boxes, target_sizes)`` selecting query 2 twice, then query 1.

        All boxes are in-bounds, so clamping is a no-op and the gather is an exact row copy.

        Examples:
            Pytest fixtures cannot be called directly outside fixture injection.

            >>> TestGatherAndScaleBoxes.duplicate_query_boxes(None)  # doctest: +SKIP
        """
        out_bbox = torch.tensor(
            [
                [
                    [0.20, 0.30, 0.10, 0.10],
                    [0.50, 0.50, 0.20, 0.20],
                    [0.70, 0.40, 0.15, 0.25],
                ]
            ]
        )  # (B=1, Q=3, 4)
        topk_boxes = torch.tensor([[2, 2, 1]])  # query 2 selected twice, then query 1 (out of order)
        target_sizes = torch.tensor([[480, 640]])  # (h, w)
        return out_bbox, topk_boxes, target_sizes

    def test_clamps_boxes_to_image_bounds(self):
        """Boxes that extrapolate beyond [0, 1] in normalized space are clamped to pixel-space image dimensions after
        scaling."""
        # cxcywh normalized: [0] negative x1/y1 after scale (-25.6, -19.2), [1] x2/y2 overflow (665.6, 499.2),
        # [2] in-bounds control.
        out_bbox = torch.tensor(
            [
                [
                    [0.01, 0.01, 0.10, 0.10],
                    [0.99, 0.99, 0.10, 0.10],
                    [0.50, 0.50, 0.20, 0.20],
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

        # Exact clamped values — a bounds-only check cannot catch a clamp returning e.g. 1.0 instead of 0.0
        assert boxes[0, 0, 0].item() == pytest.approx(0.0), "x1 of underflowing box must clamp to 0"
        assert boxes[0, 0, 1].item() == pytest.approx(0.0), "y1 of underflowing box must clamp to 0"
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

        # Per-image upper bounds for (x1, y1, x2, y2): (img_w, img_h, img_w, img_h)
        upper = torch.tensor([[[400.0, 300.0, 400.0, 300.0]], [[800.0, 600.0, 800.0, 600.0]]])
        assert (boxes >= 0).all(), f"Negative coords present: {boxes[boxes < 0]}"
        assert (boxes <= upper).all(), f"Coords exceed their own image size.\nBounds:\n{upper}\nGot:\n{boxes}"

    def test_gathers_duplicate_and_out_of_order_indices(self, duplicate_query_boxes):
        """Top-k can pick the same query under two classes; each pick must reproduce that query's exact scaled box.

        The box gather selects whole rows by query index, so duplicated and out-of-order indices must copy the source
        box verbatim for every occurrence. Mirrors the TRT-benchmark twin on the shipped PostProcess path.
        """
        out_bbox, topk_boxes, target_sizes = duplicate_query_boxes

        boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)

        # Expected computed independently from raw input, not by re-invoking the function under test.
        scale = torch.tensor([640.0, 480.0, 640.0, 480.0])
        expected = box_ops.box_cxcywh_to_xyxy(out_bbox[0]) * scale
        assert torch.equal(boxes[0, 0], expected[2])
        assert torch.equal(boxes[0, 1], expected[2])
        assert torch.equal(boxes[0, 2], expected[1])

    @pytest.mark.gpu
    @_skip_without_cuda
    def test_cuda_matches_cpu_for_duplicated_indices(self, duplicate_query_boxes):
        """Box selection is an arithmetic-free row copy, so CUDA must reproduce the CPU result bit-for-bit even with
        duplicated and out-of-order indices."""
        out_bbox, topk_boxes, target_sizes = duplicate_query_boxes

        cpu_boxes = PostProcess._gather_and_scale_boxes(out_bbox, topk_boxes, target_sizes)
        cuda_boxes = PostProcess._gather_and_scale_boxes(out_bbox.cuda(), topk_boxes.cuda(), target_sizes.cuda())

        assert torch.equal(cpu_boxes, cuda_boxes.cpu())


class TestPostProcessForward:
    """Integration tests for :meth:`PostProcess.forward`."""

    def test_clamps_edge_boxes_to_bounds(self):
        """PostProcess.forward returns non-negative in-bounds boxes for edge-hugging predictions."""
        post_process = PostProcess(num_select=2)
        outputs = {
            "pred_logits": torch.tensor([[[10.0, -10.0], [9.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.01, 0.01, 0.10, 0.10], [0.99, 0.99, 0.10, 0.10]]]),
        }
        target_sizes = torch.tensor([[480, 640]])
        results = post_process(outputs, target_sizes)
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
        args = _cyclic_mask_case(64, 64, mask_hw=16, size=512)

        results = PostProcess._postprocess_masks(*args)

        masks = results[0]["masks"]
        assert masks.shape == (64, 1, 512, 512)
        assert masks.dtype == torch.bool

    def test_native_resolution_skips_upsample_when_flag_false(self):
        """upsample_masks_to_image_size=False returns masks at native mask-head resolution instead of target_sizes (opt-
        in validation-cost reduction, see TrainConfig.eval_masks_head_resolution)."""
        args = _cyclic_mask_case(8, 8, mask_hw=16, size=512)

        results = PostProcess._postprocess_masks(*args, upsample_masks_to_image_size=False)

        masks = results[0]["masks"]
        assert masks.shape == (8, 1, 16, 16)
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

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, _skip_without_cuda])])
    @pytest.mark.parametrize(
        ("batch", "upsample", "expected_calls"),
        [
            pytest.param(4, True, [(4, 2)], id="upsampled-batch"),
            pytest.param(4, False, [], id="native-resolution"),
            pytest.param(0, True, [], id="empty-upsampled-batch"),
        ],
    )
    def test_target_sizes_are_read_once_per_upsampled_batch(
        self, mocker, device: str, batch: int, upsample: bool, expected_calls: list[tuple[int, ...]]
    ) -> None:
        """A non-empty upsampled batch reads all target sizes together; other paths do not read them.

        On CUDA this is the device-to-host synchronization collapsed from one per image to one per batch.
        """
        out_masks = torch.randn(batch, 1, 2, 2, device=device)
        scores = torch.ones(batch, 1, device=device)
        labels = torch.zeros(batch, 1, dtype=torch.long, device=device)
        boxes = torch.zeros(batch, 1, 4, device=device)
        topk_boxes = torch.zeros(batch, 1, dtype=torch.long, device=device)
        target_sizes = torch.full((batch, 2), 8, dtype=torch.long, device=device)
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        mocker.patch.object(torch.Tensor, "tolist", tracked_tolist)

        PostProcess._postprocess_masks(
            out_masks, scores, labels, boxes, topk_boxes, target_sizes, upsample_masks_to_image_size=upsample
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
            F.interpolate(out_masks[0, 0][None, None], size=(4, 9), mode="bilinear", align_corners=False) > 0.0,
            F.interpolate(out_masks[1, 0][None, None], size=(9, 4), mode="bilinear", align_corners=False) > 0.0,
        ]
        assert results[0]["masks"].shape == (1, 1, 4, 9)
        assert results[1]["masks"].shape == (1, 1, 9, 4)
        assert torch.equal(results[0]["masks"], expected_masks[0])
        assert torch.equal(results[1]["masks"], expected_masks[1])

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
            F.interpolate(out_masks[0, 2][None, None], size=(64, 64), mode="bilinear", align_corners=False) > 0.0
        )[0]
        assert torch.equal(masks[0], masks[40])  # both occurrences identical across the chunk boundary
        assert torch.equal(masks[0], expected)  # and each maps to the correct source query, not query 0

    @pytest.mark.parametrize(
        ("device", "num_select"),
        [
            pytest.param("cpu", 48),
            # Five chunks reach the preallocated CUDA branch; low-K CUDA selections keep the list/concat path.
            pytest.param("cuda", 160, marks=[pytest.mark.gpu, _skip_without_cuda]),
        ],
    )
    def test_upsample_writes_chunks_into_preallocated_result_without_final_cat(
        self, mocker, device: str, num_select: int
    ) -> None:
        """Upsampling must preserve exact chunked interpolation while avoiding a full-size bool concat."""
        args = _cyclic_mask_case(4, num_select, device=device)
        expected = _chunked_upsampled_masks(args[0], args[4], 64)
        mocker.patch.object(
            torch,
            "cat",
            side_effect=AssertionError("mask upsampling must write into the preallocated result without torch.cat"),
        )

        result = PostProcess._postprocess_masks(*args, upsample_masks_to_image_size=True)[0]["masks"]

        assert torch.equal(result, expected)

    def test_upsample_drops_previous_chunk_buffer_before_next_chunk_is_built(self, monkeypatch):
        """The `del interpolated` after each `torch.gt` must free the transient chunk buffer before the next chunk's
        `F.interpolate` call starts, not merely by the time the loop reassigns the variable.

        Without the `del`, the previous chunk stays referenced by the loop variable for the full duration of the next
        `F.interpolate` call, so two chunks are briefly alive at once -- the same double-buffering this whole
        preallocation is meant to avoid, just smaller and confined to one loop iteration.
        """
        args = _cyclic_mask_case(4, 96)  # 3 chunks of 32, so the guard is exercised across two boundaries
        previous_chunk_ref = None
        real_interpolate = F.interpolate

        def _tracking_interpolate(*args: object, **kwargs: object) -> torch.Tensor:
            """Track interpolation output and require the previous chunk to be released before the next call.

            Examples:
                >>> callable(_tracking_interpolate)
                True
            """
            nonlocal previous_chunk_ref
            if previous_chunk_ref is not None:
                assert previous_chunk_ref() is None, (
                    "the previous chunk's buffer is still alive when the next chunk's F.interpolate starts"
                )
            result = real_interpolate(*args, **kwargs)
            previous_chunk_ref = weakref.ref(result)
            return result

        monkeypatch.setattr(postprocess.F, "interpolate", _tracking_interpolate)

        PostProcess._postprocess_masks(*args, upsample_masks_to_image_size=True)

    @pytest.mark.gpu
    @_skip_without_cuda
    def test_upsample_on_cuda_uses_legacy_path_for_low_k_memory_regression(self, mocker):
        """Low-K CUDA selections retain the lower-peak list/concat strategy."""
        args = _cyclic_mask_case(4, 100, device="cuda")  # four chunks cover shipped SegNano/SegSmall defaults
        cat = mocker.patch("torch.cat", wraps=torch.cat)

        result = PostProcess._postprocess_masks(*args, upsample_masks_to_image_size=True)[0]["masks"]

        assert cat.call_count == 1
        assert result.shape == (100, 1, 64, 64)

    @pytest.mark.xla
    @requires_torch_xla
    def test_upsample_on_xla_uses_supported_fallback(self):
        """XLA exercises mask postprocessing without dispatching the unverified ``out=`` overload."""
        import torch_xla

        args = _cyclic_mask_case(4, 48, device=torch_xla.device(), target_device="cpu")
        expected = _chunked_upsampled_masks(args[0], args[4], 64)

        result = PostProcess._postprocess_masks(*args, upsample_masks_to_image_size=True)[0]["masks"]

        assert torch.equal(result, expected)

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

        Examples:
            >>> out_masks, scores, *_ = TestPostProcessMasks._mask_case(num_select=4, batch=2)
            >>> out_masks.shape, scores.shape
            (torch.Size([2, 16, 8, 8]), torch.Size([2, 4]))
        """
        out_masks = torch.randn(batch, num_queries, mask_hw, mask_hw)
        scores = torch.stack([torch.linspace(0.95 - 0.3 * i, 0.05, num_select) for i in range(batch)])
        labels = torch.zeros(batch, num_select, dtype=torch.long)
        boxes = torch.zeros(batch, num_select, 4)
        topk_boxes = torch.arange(num_select).unsqueeze(0).repeat(batch, 1)
        target_sizes = torch.tensor([[64, 64]]).repeat(batch, 1)
        return out_masks, scores, labels, boxes, topk_boxes, target_sizes

    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    def test_score_threshold_upsamples_only_the_kept_masks(self, mocker, threshold):
        """Masks that the caller's threshold discards must never reach the interpolation.

        The upsample runs at target-image resolution while the kept fraction is small (at ``num_select=100`` a COCO-
        style image keeps a handful), so resizing the discarded rows is work whose result is dropped a few lines later.
        Counting the interpolated rows rather than timing them keeps the test deterministic. A threshold of 1.0 keeps
        nothing and must skip the interpolation altogether rather than resize an empty tensor.

        ``num_select=100`` (> ``_MASK_CHUNK`` = 32) keeps enough rows above 0.5 to span several interpolation chunks,
        so the count also proves the chunked upsample loop stays correct under early filtering.
        """
        args = self._mask_case(num_select=100, num_queries=100)
        expected_kept = int((args[1][0] > threshold).sum())
        interpolate = mocker.patch("rfdetr.models.postprocess.F.interpolate", wraps=F.interpolate)

        results = PostProcess._postprocess_masks(*args, score_threshold=threshold)

        interpolated_rows = sum(c.args[0].shape[0] for c in interpolate.call_args_list)
        assert interpolated_rows == expected_kept, (
            f"interpolated {interpolated_rows} mask rows but only {expected_kept} survive the "
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

        keys = ("scores", "labels", "boxes", "masks")
        torch.testing.assert_close(
            {key: filtered_early[key] for key in keys}, {key: full[key][keep] for key in keys}, rtol=0.0, atol=0.0
        )

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

    @staticmethod
    def _box_outputs() -> dict[str, torch.Tensor]:
        """Return the one-image, four-query box outputs the threshold-ignoring tests share.

        Examples:
            >>> sorted(TestPostProcessMasks._box_outputs())
            ['pred_boxes', 'pred_logits']
        """
        return {"pred_logits": torch.randn(1, 4, 2), "pred_boxes": torch.rand(1, 4, 4)}

    def test_score_threshold_is_ignored_by_the_box_head(self):
        """The box-only head returns the same thing with and without the argument.

        ``predict()`` passes the threshold for every model, so the head that cannot use it has to ignore it rather than
        fail; the box-only path has no per-detection resize to skip.
        """
        outputs = self._box_outputs()
        post_process = PostProcess(num_select=4)
        target_sizes = torch.tensor([[128, 128]])

        baseline = post_process(outputs, target_sizes)[0]
        with_threshold = post_process(outputs, target_sizes, score_threshold=0.9)[0]

        torch.testing.assert_close(with_threshold, baseline, rtol=0.0, atol=0.0, equal_nan=True)

    def test_score_threshold_is_ignored_by_the_keypoint_head(self):
        """The keypoint head returns the same thing with and without the argument.

        The keypoint path rewrites scores after selection (uncertainty fusion), so filtering on the pre-fusion scores
        would not match the caller's own filter.
        """
        # pred_keypoints is (B, Q, num_keypoint_classes * max_num_keypoints, D) with D >= 7 so precision is emitted.
        outputs = {**self._box_outputs(), "pred_keypoints": torch.randn(1, 4, 6, 7)}
        post_process = PostProcess(num_select=4, num_keypoints_per_class=[3, 3])
        target_sizes = torch.tensor([[128, 128]])

        baseline = post_process(outputs, target_sizes)[0]
        with_threshold = post_process(outputs, target_sizes, score_threshold=0.9)[0]

        torch.testing.assert_close(with_threshold, baseline, rtol=0.0, atol=0.0, equal_nan=True)
