# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""Tensor utilities: NestedTensor, collate_fn, and helpers."""

from typing import Any, List, Optional, Tuple

import torch
import torchvision
from torch import Tensor


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    """Return element-wise maximums of a list of lists.

    Args:
        the_list: List of integer lists, all of the same length.

    Returns:
        List of per-position maximums.
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    """Batch of tensors with variable spatial sizes, padded to a common size.

    Stores both the padded tensor and a boolean mask indicating padding positions.
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device, **kwargs: Any) -> "NestedTensor":
        """Move tensors and mask to *device*.

        Args:
            device: Target device.
            **kwargs: Additional arguments forwarded to ``Tensor.to``.

        Returns:
            New NestedTensor on *device*.
        """
        cast_tensor = self.tensors.to(device, **kwargs)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, **kwargs)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def pin_memory(self) -> "NestedTensor":
        """Pin tensor and mask memory for faster CPU→GPU transfer.

        Returns:
            New NestedTensor with pinned memory.
        """
        return NestedTensor(
            self.tensors.pin_memory(),
            self.mask.pin_memory() if self.mask is not None else None,
        )

    def decompose(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Return ``(tensors, mask)`` tuple.

        Returns:
            Tuple of the padded tensor and the boolean mask.
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """Pad a list of variable-size tensors into a single NestedTensor.

    Args:
        tensor_list: List of 3-D tensors (C, H, W) with possibly different H, W.

    Returns:
        NestedTensor with all images padded to the maximum spatial dimensions.
    """
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """ONNX-tracing-compatible variant of ``nested_tensor_from_tensor_list``.

    Args:
        tensor_list: List of 3-D tensors (C, H, W).

    Returns:
        Padded NestedTensor suitable for ONNX export.
    """
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    """Collate a list of (image, target) pairs into a batched NestedTensor.

    Args:
        batch: List of ``(image, target)`` pairs from a dataset.

    Returns:
        Tuple of ``(NestedTensor_of_images, list_of_targets)``.
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
