# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Row selection of PostProcess for a model with both a segmentation and a keypoint head.

Each ``nonzero`` call on a CUDA tensor waits for the device, so the two-head path must pick the rows above
``score_threshold`` once per image and use that selection for the masks and for the keypoints.
"""

import pytest
import torch

from rfdetr.models.postprocess import PostProcess

_TARGET_SIZES = torch.tensor([[4, 4], [4, 4]])


def _count_nonzero_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch ``Tensor.nonzero`` to count its calls and return a one-item list holding the running count.

    Args:
        monkeypatch: Patches ``Tensor.nonzero`` for the duration of the test.

    Examples:
        >>> with pytest.MonkeyPatch.context() as patch:
        ...     calls = _count_nonzero_calls(patch)
        ...     _ = torch.tensor([True, False]).nonzero(as_tuple=True)
        >>> calls
        [1]
    """
    calls = [0]
    original_nonzero = torch.Tensor.nonzero

    def counting_nonzero(tensor: torch.Tensor, *args: object, **kwargs: object) -> object:
        calls[0] += 1
        return original_nonzero(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "nonzero", counting_nonzero)
    return calls


def _two_image_outputs() -> dict[str, torch.Tensor]:
    """Return two-image, three-query outputs of a one-class model with masks and two keypoints per object.

    Query ``q`` has object logit ``[2, -1, 1][q]`` and a mask positive only in column ``q``.

    Examples:
        >>> outputs = _two_image_outputs()
        >>> outputs["pred_masks"].shape, outputs["pred_keypoints"].shape
        (torch.Size([2, 3, 4, 4]), torch.Size([2, 3, 2, 8]))
    """
    masks = torch.full((2, 3, 4, 4), -1.0)
    for query in range(3):
        masks[:, query, :, query] = 1.0
    return {
        "pred_logits": torch.tensor([[[2.0], [-1.0], [1.0]]] * 2),
        "pred_boxes": torch.full((2, 3, 4), 0.5),
        "pred_masks": masks,
        "pred_keypoints": torch.zeros((2, 3, 2, 8)),
    }


class TestMaskAndKeypointSelection:
    """The rows above the threshold are selected once per image and shared by masks and keypoints."""

    def test_threshold_adds_one_selection_per_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A threshold costs one ``nonzero`` per image on top of what keypoint decoding itself needs."""
        postprocess = PostProcess(num_select=3, num_keypoints_per_class=[2])
        outputs = _two_image_outputs()
        calls = _count_nonzero_calls(monkeypatch)
        postprocess(outputs, _TARGET_SIZES)
        unthresholded_calls = calls[0]

        calls[0] = 0
        postprocess(outputs, _TARGET_SIZES, score_threshold=0.3)

        assert calls[0] == unthresholded_calls + len(_TARGET_SIZES)

    def test_mask_selection_is_reported_per_image(self) -> None:
        """``selections`` receives, per image, the indices of the rows scoring above the threshold."""
        scores = torch.tensor([[0.9, 0.2, 0.6], [0.1, 0.8, 0.4]])
        labels = torch.zeros((2, 3), dtype=torch.int64)
        boxes = torch.zeros((2, 3, 4))
        topk_boxes = torch.tensor([[0, 1, 2], [0, 1, 2]])
        selections: list[torch.Tensor] = []

        PostProcess._postprocess_masks(
            torch.zeros((2, 3, 4, 4)),
            scores,
            labels,
            boxes,
            topk_boxes,
            _TARGET_SIZES,
            score_threshold=0.5,
            selections=selections,
        )

        assert [selection.tolist() for selection in selections] == [[0, 2], [1]]

    def test_keypoints_follow_the_mask_rows(self) -> None:
        """Every image keeps masks and keypoints for the same rows, in the same order, as the scores."""
        postprocess = PostProcess(num_select=3, num_keypoints_per_class=[2])
        outputs = _two_image_outputs()
        outputs["pred_keypoints"][..., 0] = torch.tensor([0.1, 0.2, 0.3])[None, :, None]

        results = postprocess(outputs, _TARGET_SIZES, score_threshold=0.3)

        for result in results:
            # Column q of the mask and keypoint x = (q + 1) / 10 * width both identify query q.
            mask_query = result["masks"][:, 0, 0, :].int().argmax(dim=1)
            keypoint_query = (result["keypoints"][:, 0, 0] / 4 * 10).round().long() - 1
            assert mask_query.tolist() == keypoint_query.tolist()
            assert result["keypoints"].shape[0] == result["scores"].shape[0] == result["masks"].shape[0]
