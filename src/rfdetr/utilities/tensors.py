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

from typing import Any

import torch
import torchvision
from torch import Tensor


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
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

    def __init__(self, tensors: Tensor, mask: Tensor | None) -> None:
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

    def decompose(self) -> tuple[Tensor, Tensor | None]:
        """Return ``(tensors, mask)`` tuple.

        Returns:
            Tuple of the padded tensor and the boolean mask.
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: list[Tensor]) -> NestedTensor:
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
def _onnx_nested_tensor_from_tensor_list(tensor_list: list[Tensor]) -> NestedTensor:
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


def _bilinear_grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Bilinear grid sampling compatible with all PyTorch backends including MPS.

    Drop-in replacement for ``F.grid_sample(input, grid, mode='bilinear', ...)``.
    On MPS, ``F.grid_sample`` backward (``grid_sampler_2d_backward``) is not yet
    implemented and silently falls back to CPU.  This function uses gather-based
    index arithmetic — natively supported on every backend — for the MPS path,
    while delegating to ``F.grid_sample`` on CUDA/CPU where its fused kernel is
    faster.  The two paths are numerically identical, so model accuracy is
    unaffected.

    Args:
        input: Feature map of shape ``(N, C, H, W)``.
        grid: Sampling grid of shape ``(N, Hg, Wg, 2)`` with values in ``[-1, 1]``.
        padding_mode: ``"zeros"`` returns 0 for out-of-bounds samples;
            ``"border"`` clamps to the nearest border pixel.
        align_corners: If ``True``, grid extremes ``±1`` map to pixel centres at
            positions ``0`` and ``H-1``/``W-1``.

    Returns:
        Sampled tensor of shape ``(N, C, Hg, Wg)``.
    """
    import torch.nn.functional as F

    if input.device.type != "mps":
        return F.grid_sample(input, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)

    if padding_mode not in ("zeros", "border"):
        msg = (
            f"Unsupported padding_mode={padding_mode!r} for manual grid sampling. "
            "Only 'zeros' and 'border' are supported in this path."
        )
        raise ValueError(msg)

    N, C, H, W = input.shape
    Hg, Wg = grid.shape[1], grid.shape[2]

    # Unnormalize [-1, 1] → floating-point pixel coordinates
    if align_corners:
        ix = (grid[..., 0] + 1) * (W - 1) / 2  # [N, Hg, Wg]
        iy = (grid[..., 1] + 1) * (H - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * W / 2 - 0.5
        iy = (grid[..., 1] + 1) * H / 2 - 0.5

    ix0 = ix.floor().long()  # top-left corner
    iy0 = iy.floor().long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Bilinear weights: fractional distance from top-left corner  [N, 1, Hg, Wg]
    # Cast to input.dtype so float16 inputs don't silently upcast to float32.
    wx1 = (ix - ix0.float()).to(input.dtype).unsqueeze(1)
    wy1 = (iy - iy0.float()).to(input.dtype).unsqueeze(1)
    one = wx1.new_tensor(1.0)
    wx0 = one - wx1
    wy0 = one - wy1

    if padding_mode == "border":
        ix0 = ix0.clamp(0, W - 1)
        iy0 = iy0.clamp(0, H - 1)
        ix1 = ix1.clamp(0, W - 1)
        iy1 = iy1.clamp(0, H - 1)
    else:  # zeros: record which corners fall inside the image before clamping
        in_x0 = (ix0 >= 0) & (ix0 < W)  # [N, Hg, Wg]
        in_x1 = (ix1 >= 0) & (ix1 < W)
        in_y0 = (iy0 >= 0) & (iy0 < H)
        in_y1 = (iy1 >= 0) & (iy1 < H)
        ix0 = ix0.clamp(0, W - 1)
        iy0 = iy0.clamp(0, H - 1)
        ix1 = ix1.clamp(0, W - 1)
        iy1 = iy1.clamp(0, H - 1)

    flat = input.flatten(2)  # [N, C, H*W]

    def _gather(iy_: torch.Tensor, ix_: torch.Tensor) -> torch.Tensor:
        idx = (iy_ * W + ix_).flatten(1).unsqueeze(1).expand(N, C, -1)  # [N, C, Hg*Wg]
        return flat.gather(2, idx).view(N, C, Hg, Wg)

    v00 = _gather(iy0, ix0)  # top-left
    v10 = _gather(iy0, ix1)  # top-right
    v01 = _gather(iy1, ix0)  # bottom-left
    v11 = _gather(iy1, ix1)  # bottom-right

    if padding_mode == "zeros":
        v00 = v00 * (in_x0 & in_y0).unsqueeze(1)
        v10 = v10 * (in_x1 & in_y0).unsqueeze(1)
        v01 = v01 * (in_x0 & in_y1).unsqueeze(1)
        v11 = v11 * (in_x1 & in_y1).unsqueeze(1)

    return wx0 * wy0 * v00 + wx1 * wy0 * v10 + wx0 * wy1 * v01 + wx1 * wy1 * v11


def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Collate a list of (image, target) pairs into a batched NestedTensor.

    Args:
        batch: List of ``(image, target)`` pairs from a dataset.

    Returns:
        Tuple of ``(NestedTensor_of_images, list_of_targets)``.
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
