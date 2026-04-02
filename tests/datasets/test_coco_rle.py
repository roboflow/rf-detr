# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for native RLE annotation support in the COCO dataset pipeline.

Verifies that :func:`convert_coco_poly_to_mask` and :class:`ConvertCoco`
correctly handle compressed RLE, uncompressed RLE, and polygon segmentation
formats — including mixed annotations within the same image.
"""

import numpy as np
import pycocotools.mask as mask_util
import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import ConvertCoco, _is_rle, convert_coco_poly_to_mask

# Shared test dimensions
_H, _W = 100, 100
_IMAGE = Image.new("RGB", (_W, _H))


def _make_reference_mask() -> np.ndarray:
    """Create a deterministic 100x100 binary mask with a rectangular region."""
    mask = np.zeros((_H, _W), dtype=np.uint8)
    mask[20:50, 30:70] = 1
    return mask


def _encode_compressed_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask to compressed RLE with string counts (COCO JSON format)."""
    rle = mask_util.encode(np.asfortranarray(mask))
    # COCO JSON stores counts as a UTF-8 string, not bytes
    rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
    rle["size"] = list(rle["size"])
    return rle


def _encode_uncompressed_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask to uncompressed RLE with integer counts."""
    flat = mask.flatten(order="F")
    counts = []
    current_val = 0
    run_length = 0
    for pixel in flat:
        if pixel == current_val:
            run_length += 1
        else:
            counts.append(run_length)
            current_val = pixel
            run_length = 1
    counts.append(run_length)
    return {"counts": counts, "size": [_H, _W]}


def _make_polygon(mask: np.ndarray) -> list:
    """Create a polygon annotation from a rectangular mask region."""
    # Simple rectangle polygon matching the mask region [20:50, 30:70]
    return [[30, 20, 70, 20, 70, 50, 30, 50]]


class TestIsRle:
    """Tests for the ``_is_rle`` helper."""

    def test_compressed_rle_detected(self) -> None:
        assert _is_rle({"counts": "abc", "size": [100, 100]}) is True

    def test_uncompressed_rle_detected(self) -> None:
        assert _is_rle({"counts": [0, 5, 10], "size": [100, 100]}) is True

    def test_bytes_counts_detected(self) -> None:
        assert _is_rle({"counts": b"abc", "size": [100, 100]}) is True

    def test_polygon_not_detected(self) -> None:
        assert _is_rle([[30, 20, 70, 20, 70, 50, 30, 50]]) is False

    def test_empty_list_not_detected(self) -> None:
        assert _is_rle([]) is False

    def test_none_not_detected(self) -> None:
        assert _is_rle(None) is False


class TestConvertCocoPolyToMaskRle:
    """Tests for RLE support in ``convert_coco_poly_to_mask``."""

    def test_compressed_rle_decodes_correctly(self) -> None:
        """Compressed RLE (string counts) should decode to the expected mask."""
        ref_mask = _make_reference_mask()
        rle = _encode_compressed_rle(ref_mask)

        result = convert_coco_poly_to_mask([rle], _H, _W)

        assert result.shape == (1, _H, _W)
        assert result.dtype == torch.uint8
        assert torch.equal(result[0], torch.as_tensor(ref_mask, dtype=torch.uint8))

    def test_uncompressed_rle_decodes_correctly(self) -> None:
        """Uncompressed RLE (int-list counts) should decode to the expected mask."""
        ref_mask = _make_reference_mask()
        uncompressed = _encode_uncompressed_rle(ref_mask)

        result = convert_coco_poly_to_mask([uncompressed], _H, _W)

        assert result.shape == (1, _H, _W)
        assert result.dtype == torch.uint8
        assert torch.equal(result[0], torch.as_tensor(ref_mask, dtype=torch.uint8))

    def test_polygon_still_works(self) -> None:
        """Polygon annotations should continue to work as before."""
        polygon = _make_polygon(_make_reference_mask())

        result = convert_coco_poly_to_mask([polygon], _H, _W)

        assert result.shape == (1, _H, _W)
        assert result.dtype == torch.uint8
        # The polygon covers the same rectangular region
        assert result[0, 30, 50] == 1  # inside the region
        assert result[0, 0, 0] == 0  # outside

    def test_compressed_rle_matches_polygon(self) -> None:
        """Compressed RLE and polygon for the same region should produce identical masks."""
        polygon = _make_polygon(_make_reference_mask())
        poly_masks = convert_coco_poly_to_mask([polygon], _H, _W)

        # Encode the polygon result as RLE, then decode via our path
        ref_np = poly_masks[0].numpy()
        rle = _encode_compressed_rle(ref_np)
        rle_masks = convert_coco_poly_to_mask([rle], _H, _W)

        assert torch.equal(poly_masks, rle_masks)

    def test_mixed_polygon_and_rle(self) -> None:
        """An image can have both polygon and RLE annotations across instances."""
        ref_mask = _make_reference_mask()
        polygon = _make_polygon(ref_mask)
        rle = _encode_compressed_rle(ref_mask)

        result = convert_coco_poly_to_mask([polygon, rle], _H, _W)

        assert result.shape == (2, _H, _W)
        # Both should produce the same mask
        assert torch.equal(result[0], result[1])

    def test_empty_segmentation_unchanged(self) -> None:
        """Empty segmentation should produce a zero mask."""
        result = convert_coco_poly_to_mask([[]], _H, _W)
        assert result.shape == (1, _H, _W)
        assert result.sum() == 0

    def test_none_segmentation_unchanged(self) -> None:
        """None segmentation should produce a zero mask."""
        result = convert_coco_poly_to_mask([None], _H, _W)
        assert result.shape == (1, _H, _W)
        assert result.sum() == 0

    def test_empty_list_returns_zero_tensor(self) -> None:
        """No segmentations at all should return (0, H, W) tensor."""
        result = convert_coco_poly_to_mask([], _H, _W)
        assert result.shape == (0, _H, _W)

    def test_rle_size_mismatch_behavior(self) -> None:
        """Compressed RLE with mismatched embedded size should raise a decode error."""
        ref_mask = _make_reference_mask()
        rle = _encode_compressed_rle(ref_mask)
        rle["size"] = [50, 50]

        # Observed behavior: pycocotools rejects mismatched RLE metadata during decode.
        with pytest.raises(ValueError, match="Invalid RLE mask representation"):
            convert_coco_poly_to_mask([rle], _H, _W)

    def test_compressed_rle_bytes_counts_decode(self) -> None:
        """Compressed RLE with bytes counts should decode correctly."""
        ref_mask = _make_reference_mask()
        rle = mask_util.encode(np.asfortranarray(ref_mask))
        rle["counts"] = rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"]
        rle["size"] = list(rle["size"])

        result = convert_coco_poly_to_mask([rle], _H, _W)

        assert result.shape == (1, _H, _W)
        assert result[0, 30, 50] == 1
        assert result[0, 0, 0] == 0

    def test_malformed_rle_counts_none_raises_value_error(self) -> None:
        """Malformed RLE with counts=None should raise ValueError."""
        with pytest.raises(ValueError, match="unsupported counts type"):
            convert_coco_poly_to_mask([{"counts": None, "size": [_H, _W]}], _H, _W)


class TestConvertCocoClassWithRle:
    """Tests that ``ConvertCoco`` correctly passes RLE annotations through."""

    def _make_annotation(self, segmentation: object, category_id: int = 0) -> dict:
        return {
            "bbox": [30, 20, 40, 30],
            "category_id": category_id,
            "area": 1200,
            "iscrowd": 0,
            "segmentation": segmentation,
        }

    def _make_target(self, annotations: list) -> dict:
        return {"image_id": 1, "annotations": annotations}

    def test_rle_masks_included_in_target(self) -> None:
        """ConvertCoco with include_masks=True should handle RLE segmentations."""
        ref_mask = _make_reference_mask()
        rle = _encode_compressed_rle(ref_mask)
        anno = self._make_annotation(rle)

        converter = ConvertCoco(include_masks=True)
        _, target = converter(_IMAGE, self._make_target([anno]))

        assert "masks" in target
        assert target["masks"].shape == (1, _H, _W)
        assert target["masks"].dtype == torch.bool
        assert target["masks"][0].any()

    def test_polygon_masks_still_work(self) -> None:
        """ConvertCoco should still handle polygon segmentations."""
        polygon = _make_polygon(_make_reference_mask())
        anno = self._make_annotation(polygon)

        converter = ConvertCoco(include_masks=True)
        _, target = converter(_IMAGE, self._make_target([anno]))

        assert "masks" in target
        assert target["masks"].shape == (1, _H, _W)
        assert target["masks"].dtype == torch.bool

    def test_mixed_rle_and_polygon_in_same_image(self) -> None:
        """An image with both polygon and RLE annotations across instances."""
        ref_mask = _make_reference_mask()
        rle_anno = self._make_annotation(_encode_compressed_rle(ref_mask), category_id=0)
        poly_anno = self._make_annotation(_make_polygon(ref_mask), category_id=1)

        converter = ConvertCoco(include_masks=True)
        _, target = converter(_IMAGE, self._make_target([rle_anno, poly_anno]))

        assert target["masks"].shape == (2, _H, _W)
        assert target["labels"].tolist() == [0, 1]

    def test_no_masks_without_flag(self) -> None:
        """RLE annotations should not produce masks when include_masks=False."""
        rle = _encode_compressed_rle(_make_reference_mask())
        anno = self._make_annotation(rle)

        converter = ConvertCoco(include_masks=False)
        _, target = converter(_IMAGE, self._make_target([anno]))

        assert "masks" not in target


class TestMalformedRle:
    """Documents _is_rle behaviour for structurally malformed inputs.

    Before this PR a bare ``except:`` in the polygon path silently swallowed
    any pycocotools error.  These tests confirm that ``_is_rle`` is a
    *structural* check only (it does not validate values inside the dict) and
    that dicts missing required keys are correctly classified as non-RLE so
    they are routed through the polygon path — where pycocotools will either
    handle them or raise a descriptive error rather than silently falling back.
    """

    def test_missing_size_key_is_not_rle(self) -> None:
        """Dict with 'counts' but no 'size' is not treated as RLE."""
        assert _is_rle({"counts": [1, 2, 3]}) is False

    def test_missing_counts_key_is_not_rle(self) -> None:
        """Dict with 'size' but no 'counts' is not treated as RLE."""
        assert _is_rle({"size": [100, 100]}) is False

    def test_counts_none_is_classified_as_rle(self) -> None:
        """_is_rle is a structural check: presence of both keys suffices regardless of value types."""
        assert _is_rle({"counts": None, "size": [_H, _W]}) is True

    def test_size_mismatch_is_still_classified_as_rle(self) -> None:
        """Dicts with both keys are RLE even when the embedded size mismatches the image dimensions."""
        assert _is_rle({"counts": [1, 2], "size": [50, 50]}) is True
