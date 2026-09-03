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

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Any, cast, overload

import torch
import torchvision
from torch import Tensor


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to the next multiple of *multiple*.

    Args:
        value: Non-negative integer to round.
        multiple: Positive integer divisor.

    Returns:
        The smallest integer greater than or equal to *value* that is an exact multiple of *multiple*.

    Raises:
        ValueError: If ``value`` is negative or ``multiple`` is not positive.
    """
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if multiple <= 0:
        raise ValueError(f"multiple must be a positive integer, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
    """Return element-wise maximums of a list of lists.

    Args:
        the_list: List of integer lists, all of the same length.

    Returns:
        List of per-position maximums.
    """
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    """Batch of tensors with variable spatial sizes, padded to a common size.

    Stores both the padded tensor and a boolean mask indicating padding positions.
    """

    def __init__(self, tensors: Tensor, mask: Tensor | None, no_padding: bool = False) -> None:
        self.tensors = tensors
        self.mask = mask
        # Structural claim, not a claim about ``mask``'s contents: True only when the batch was
        # assembled from images that already filled the padded extent, so ``mask`` is all-False.
        # Consumers use it to treat mask-derived values as constants of the batch shape without
        # reading device memory (which would force a device-to-host sync). Defaults to False so
        # every other construction site keeps the conservative behaviour.
        self.no_padding = no_padding

    def to(self, device: torch.device, **kwargs: Any) -> "NestedTensor":
        """Move tensors and mask to *device*.

        Args:
            device: Target device. **kwargs: Additional arguments forwarded to ``Tensor.to``.

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
        return NestedTensor(cast_tensor, cast_mask, self.no_padding)

    def pin_memory(self) -> "NestedTensor":
        """Pin tensor and mask memory for faster CPU→GPU transfer.

        Returns:
            New NestedTensor with pinned memory.
        """
        return NestedTensor(
            self.tensors.pin_memory(),
            self.mask.pin_memory() if self.mask is not None else None,
            self.no_padding,
        )

    def decompose(self) -> tuple[Tensor, Tensor | None]:
        """Return ``(tensors, mask)`` tuple.

        Returns:
            Tuple of the padded tensor and the boolean mask.
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def nested_tensor_from_tensor_list(
    tensor_list: list[Tensor] | Tensor,
    block_size: int | None = None,
) -> NestedTensor:
    """Pack image tensors into a single NestedTensor, padding when needed.

    Args:
        tensor_list: List of 3-D ``(C, H, W)`` tensors with possibly different spatial sizes,
            or one 4-D ``(B, C, H, W)`` batch tensor.
        block_size: When set, round the padded ``H`` and ``W`` up to the next
            multiple of *block_size* before allocating the batch tensor.  Used to satisfy backbone divisibility
            requirements (e.g. windowed-attention backbones require ``H % (patch_size * num_windows) == 0``).  The
            rounded-up strip is explicitly tracked in the ``mask`` as padding.

    Returns:
        NestedTensor with all images padded to the maximum spatial dimensions (rounded up to *block_size* when
        provided). When *tensor_list* is already a contiguous 4-D batch that needs no padding, the returned
        ``.tensors`` is *tensor_list* itself (same storage, not a copy): mutating it after this call also mutates
        the NestedTensor. A non-contiguous 4-D batch instead gets an independent, contiguous copy, like every
        other input shape.
    """
    images = cast(Sequence[Tensor], tensor_list)
    # TODO make this more general
    if images[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(images, block_size=block_size)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in images])
        if block_size is not None:
            max_size[1] = _round_up_to_multiple(max_size[1], block_size)
            max_size[2] = _round_up_to_multiple(max_size[2], block_size)
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(images)] + max_size
        b, c, h, w = batch_shape
        dtype = images[0].dtype
        device = images[0].device
        # Every image already fills the padded extent whenever no image is smaller than the
        # per-axis maximum and block-size rounding added nothing. That is decided here from
        # Python-side shapes alone, so the resulting all-False mask is known without reading it.
        no_padding = all(list(img.shape) == max_size for img in images)
        if no_padding and isinstance(tensor_list, Tensor) and tensor_list.is_contiguous():
            # The batch tensor already *is* the padded batch: torch.zeros(batch_shape) followed by
            # a full-extent copy_ of every image reproduces it element for element. Skip the
            # allocation and the copy. Returned ``.tensors`` is the caller's own tensor (same
            # storage, same strides) rather than an independent copy -- restricted to the
            # already-contiguous case so the result always matches the layout the allocate-and-copy
            # path would have produced. A non-contiguous input falls through to that path instead,
            # which normalizes both layout and ownership like every other branch.
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
            return NestedTensor(tensor_list, mask, True)
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = (
            torch.zeros((b, h, w), dtype=torch.bool, device=device)
            if no_padding
            else torch.ones((b, h, w), dtype=torch.bool, device=device)
        )
        for img, pad_img, m in zip(images, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            if not no_padding:
                m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask, no_padding)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(
    tensor_list: Sequence[Tensor],
    block_size: int | None = None,
) -> NestedTensor:
    """ONNX-tracing-compatible variant of ``nested_tensor_from_tensor_list``.

    Args:
        tensor_list: List of 3-D ``(C, H, W)`` tensors or one 4-D ``(B, C, H, W)`` batch tensor.
        block_size: When set, round ``H`` and ``W`` up to the next multiple of
            this value before padding.  See :func:`nested_tensor_from_tensor_list`.

    Returns:
        Padded NestedTensor suitable for ONNX export.
    """
    max_size: list[Tensor] = []
    for i in range(tensor_list[0].dim()):
        # ``img.shape[i]`` is a symbolic Tensor under ONNX tracing (not the plain
        # int mypy infers for the untraced case); ``as_tensor`` is a no-op on the
        # traced Tensor and correctly wraps the untraced int.
        dim_sizes = torch.stack([torch.as_tensor(img.shape[i]) for img in tensor_list]).to(torch.float32)
        max_size_i = torch.max(dim_sizes).to(torch.int64)
        max_size.append(max_size_i)
    if block_size is not None:
        # Spatial dimensions are indices 1 (H) and 2 (W); index 0 is channels.
        bs = torch.as_tensor(block_size, dtype=torch.int64)
        max_size[1] = ((max_size[1] + bs - 1) // bs) * bs
        max_size[2] = ((max_size[2] + bs - 1) // bs) * bs
    max_size_tuple = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size_tuple, tuple(img.shape))]
        # ``F.pad``'s size argument is typed as Sequence[int], but during ONNX tracing
        # these are symbolic Tensors so the padded size stays dynamic in the exported graph.
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))  # type: ignore[arg-type]
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)  # type: ignore[arg-type]
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def _bilinear_grid_sample(
    input: Tensor,
    grid: Tensor,
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
    """Bilinear grid sampling compatible with all PyTorch backends including MPS and XLA.

    Drop-in replacement for ``F.grid_sample(input, grid, mode='bilinear', ...)``. On MPS, ``F.grid_sample`` backward
    (``grid_sampler_2d_backward``) is not yet implemented and silently falls back to CPU. On XLA, routing through the
    gather path avoids relying on ``F.grid_sample``'s XLA decomposition (host-fallback risk on older torch_xla).  This
    function uses gather-based index arithmetic — natively supported on every backend — for the MPS/XLA path, while
    delegating to ``F.grid_sample`` on CUDA/CPU where its fused kernel is faster.  The two paths are numerically
    identical, so model accuracy is unaffected.

    Args:
        input: Feature map of shape ``(N, C, H, W)``.
        grid: Sampling grid of shape ``(N, Hg, Wg, 2)`` with values in ``[-1, 1]``.
        padding_mode: ``"zeros"`` returns 0 for out-of-bounds samples; ``"border"`` clamps to the nearest border pixel.
        align_corners: If ``True``, grid extremes ``±1`` map to pixel centres at positions ``0`` and ``H-1``/``W-1``.

    Returns:
        Sampled tensor of shape ``(N, C, Hg, Wg)``.
    """
    import torch.nn.functional as F  # noqa: N812

    if input.device.type not in ("mps", "xla"):
        return F.grid_sample(input, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)

    if padding_mode not in ("zeros", "border"):
        msg = (
            f"Unsupported padding_mode={padding_mode!r} for manual grid sampling. "
            "Only 'zeros' and 'border' are supported in this path."
        )
        raise ValueError(msg)

    batch_size, channels, height, width = input.shape
    grid_height, grid_width = grid.shape[1], grid.shape[2]

    # Unnormalize [-1, 1] → floating-point pixel coordinates
    if align_corners:
        ix = (grid[..., 0] + 1) * (width - 1) / 2  # [batch_size, grid_height, grid_width]
        iy = (grid[..., 1] + 1) * (height - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * width / 2 - 0.5
        iy = (grid[..., 1] + 1) * height / 2 - 0.5

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
        ix0 = ix0.clamp(0, width - 1)
        iy0 = iy0.clamp(0, height - 1)
        ix1 = ix1.clamp(0, width - 1)
        iy1 = iy1.clamp(0, height - 1)
    else:  # zeros: record which corners fall inside the image before clamping
        in_x0 = (ix0 >= 0) & (ix0 < width)  # [batch_size, grid_height, grid_width]
        in_x1 = (ix1 >= 0) & (ix1 < width)
        in_y0 = (iy0 >= 0) & (iy0 < height)
        in_y1 = (iy1 >= 0) & (iy1 < height)
        ix0 = ix0.clamp(0, width - 1)
        iy0 = iy0.clamp(0, height - 1)
        ix1 = ix1.clamp(0, width - 1)
        iy1 = iy1.clamp(0, height - 1)

    # MPS path: GatherElements(axis=2) — correct on MPS at runtime.
    flat = input.flatten(2)  # (N, C, H*W)

    def _gather(iy_: Tensor, ix_: Tensor) -> Tensor:
        idx = (iy_ * width + ix_).flatten(1).unsqueeze(1).expand(batch_size, channels, -1)
        return flat.gather(2, idx).view(batch_size, channels, grid_height, grid_width)

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


def _collate_with_block_size(
    batch: list[tuple[Any, ...]],
    block_size: int | None = None,
    pack: bool = False,
) -> tuple[Any, ...]:
    """Module-level collate helper used as the base for :func:`make_collate_fn`.

    Defined at module scope (rather than as a closure inside :func:`make_collate_fn`) so that the resulting
    :class:`functools.partial` is picklable for multi-process DataLoaders and DDP spawn workers.

    Args:
        batch: List of ``(image, target)`` pairs from a dataset.
        block_size: When set, round batch ``H`` and ``W`` up to the next multiple of this value before padding.  See
            :func:`nested_tensor_from_tensor_list`.
        pack: When set, concatenate the target dicts into a :class:`PackedTargets` so that a batch crosses the
            worker-to-main boundary as a handful of tensors instead of one per field per sample.

    Returns:
        Tuple of ``(NestedTensor_of_images, tuple_of_targets)``, the targets packed when *pack* is set.
    """
    columns = list(zip(*batch))
    images = nested_tensor_from_tensor_list(list(columns[0]), block_size=block_size)
    if pack and len(columns) == 2 and all(isinstance(entry, dict) for entry in columns[1]):
        return (images, pack_targets(list(columns[1])))
    return (images, *columns[1:])


def collate_fn(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Collate a list of (image, target) pairs into a batched NestedTensor.

    Uses :func:`nested_tensor_from_tensor_list` with no ``block_size`` rounding. For DataLoaders that need
    backbone-aware rounding (e.g. windowed attention requires divisibility by ``patch_size * num_windows``), use
    :func:`make_collate_fn` instead to obtain a parameterised collate callable.

    Args:
        batch: List of ``(image, target)`` pairs from a dataset.

    Returns:
        Tuple of ``(NestedTensor_of_images, tuple_of_targets)``.
    """
    return _collate_with_block_size(batch, block_size=None)


def make_collate_fn(
    block_size: int | None = None,
    pack: bool = False,
) -> Callable[[list[tuple[Any, ...]]], tuple[Any, ...]]:
    """Build a collate function that rounds batch ``H``/``W`` up to *block_size*.

    Used by the training DataModule to ensure that batched inputs satisfy the backbone's spatial divisibility
    requirement (``patch_size * num_windows``). Passing ``block_size=None`` produces a callable equivalent to
    :func:`collate_fn`.

    The returned callable is a :class:`functools.partial`, not a closure, so it is picklable and safe to use with
    multi-process DataLoaders (``num_workers > 0``) and DDP spawn workers.

    Args:
        block_size: When set, batch ``H`` and ``W`` are rounded up to the next
            multiple of this value before padding.  The rounded-up strip is marked as padding in the NestedTensor mask.
        pack: When set, the collated targets are returned as a :class:`PackedTargets`.

    Returns:
        A collate callable suitable for ``torch.utils.data.DataLoader``.
    """
    return partial(_collate_with_block_size, block_size=block_size, pack=pack)


class PackedTargets:
    """A batch of per-sample target dicts held as one concatenated tensor per field.

    Every tensor a DataLoader worker returns is moved into its own shared-memory segment and handed to the parent
    as a file descriptor. rf-detr returns one dict of small tensors per sample, so a batch of 16 crosses the
    worker-to-main boundary as 114 objects, of which 112 carry a few kilobytes in total yet pay the per-object
    cost. Concatenating each field into a single tensor and carrying the per-sample shapes as plain ints -- which
    pickle by value and never reach shared memory -- takes that to 9 objects without changing a single byte of
    payload.

    Instances behave as a sequence of the original dicts for iteration and indexing, so consumers that merely
    read a batch need no change. The class deliberately does not subclass :class:`collections.abc.Sequence`:
    PyTorch's pin-memory worker dispatches on that ABC *before* it looks for a ``pin_memory`` method, so a
    Sequence subclass would be taken apart and pinned tensor by tensor, rebuilding in the main process exactly
    what the packing avoided. Indexing rebuilds a dict of views: reassigning a key on the returned dict does not
    write back into the packed storage, but an in-place write into one of its tensors does, since the view shares
    the packed field's storage. Call :meth:`as_list` for a same-device materialised copy, or :meth:`to_list` to
    materialise directly on another device; both return independent tensors that are safe to mutate in place, the way
    the unpacked batch already is.

    Args:
        values: One flat tensor per field, holding that field concatenated across the batch.
        shapes: Per-field list of the original per-sample shapes.
        batch: Number of samples packed.
    """

    __slots__ = ("_batch", "_shapes", "_values")

    def __init__(self, values: dict[str, Tensor], shapes: dict[str, list[list[int]]], batch: int) -> None:
        self._values = values
        self._shapes = shapes
        self._batch = batch

    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        return self._batch

    @overload
    def __getitem__(self, index: int) -> dict[str, Tensor]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Tensor]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Rebuild the target dict of one sample, or a list of them for a slice.

        Args:
            index: Sample position, negative or slice as for any sequence.

        Returns:
            The sample's field-to-tensor mapping, each tensor a view into the packed storage.

        Raises:
            IndexError: If *index* is out of range.
        """
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._batch))]
        position = index + self._batch if index < 0 else index
        if not 0 <= position < self._batch:
            raise IndexError(f"index {index} is out of range for a batch of {self._batch}")
        target: dict[str, Tensor] = {}
        for key, flat in self._values.items():
            offset = 0
            for shape in self._shapes[key][:position]:
                offset += _numel(shape)
            shape = self._shapes[key][position]
            target[key] = flat[offset : offset + _numel(shape)].reshape(shape)
        return target

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Iterate the batch one rebuilt target dict at a time."""
        for position in range(self._batch):
            yield self[position]

    def pin_memory(self) -> PackedTargets:
        """Pin every packed field, keeping the batch packed.

        Called by PyTorch's pin-memory worker. Pinning the concatenated fields costs one pinned allocation
        per field rather than one per field per sample, and leaves the batch packed until
        ``transfer_batch_to_device`` materialises its independent destination tensors.

        Returns:
            A packed batch whose fields are in pinned memory.
        """
        return PackedTargets({key: flat.pin_memory() for key, flat in self._values.items()}, self._shapes, self._batch)

    @property
    def fields(self) -> Mapping[str, Tensor]:
        """Return the concatenated tensor of each field, read-only.

        These are the objects that actually cross the process boundary, so this is what to inspect when checking
        that a batch ships as a handful of tensors rather than one per field per sample.

        Returns:
            Mapping of field name to that field concatenated across the batch.
        """
        return MappingProxyType(self._values)

    def as_list(self) -> list[dict[str, Tensor]]:
        """Materialise the batch as the plain list of dicts the unpacked path produces.

        Each tensor is cloned. ``__getitem__`` returns views into the shared per-field storage -- cheap for
        read-only access, but on the unpacked path every sample already owns independent storage, so an
        in-place write on one sample's tensor here must not alias into another sample's view of the same
        field, or into the packed storage itself.

        Returns:
            One dict per sample, ordered as packed, holding independent tensors.
        """
        return [{key: value.clone() for key, value in self[position].items()} for position in range(self._batch)]

    def to_list(self, device: torch.device | str, non_blocking: bool = False) -> list[dict[str, Tensor]]:
        """Materialise independent per-sample tensors directly on *device*.

        Moving the packed object and then calling :meth:`as_list` makes the concatenated destination fields coexist
        with all of their per-sample clones. That transient duplicate is negligible for boxes and labels but can be
        large for segmentation masks. Transferring each packed view directly into its final independently-owned tensor
        avoids the full-field destination copy while preserving the unpacked path's mutation semantics.

        Args:
            device: Destination device.
            non_blocking: Whether to request asynchronous copies.

        Returns:
            One dict per sample, ordered as packed, with independent tensors on *device*.
        """
        materialised: list[dict[str, Tensor]] = [{} for _ in range(self._batch)]
        for key, flat in self._values.items():
            offset = 0
            for target, shape in zip(materialised, self._shapes[key], strict=True):
                elements = _numel(shape)
                value = flat[offset : offset + elements].reshape(shape)
                target[key] = value.to(
                    device=device,
                    non_blocking=non_blocking,
                    copy=True,
                )
                offset += elements
        return materialised

    def to(self, device: torch.device | str, non_blocking: bool = False) -> PackedTargets:
        """Move every packed field to *device* in one transfer per field.

        Args:
            device: Destination device.
            non_blocking: Whether to request an asynchronous copy.

        Returns:
            A packed batch on *device*; ``self`` if it is already there.
        """
        moved = {key: flat.to(device, non_blocking=non_blocking) for key, flat in self._values.items()}
        if all(moved[key] is flat for key, flat in self._values.items()):
            return self
        return PackedTargets(moved, self._shapes, self._batch)


def _numel(shape: list[int]) -> int:
    """Return the number of elements a tensor of *shape* holds.

    Args:
        shape: Tensor shape as a list of dimension sizes.

    Returns:
        The product of the dimensions; 1 for a scalar's empty shape.
    """
    count = 1
    for dim in shape:
        count *= dim
    return count


def pack_targets(targets: Sequence[dict[str, Tensor]]) -> PackedTargets | tuple[dict[str, Tensor], ...]:
    """Concatenate a batch of target dicts into one tensor per field.

    Args:
        targets: One dict per sample, all sharing the same keys.

    Returns:
        The packed batch, or *targets* unchanged as a tuple when it cannot be packed losslessly -- an empty batch,
        samples whose key sets differ, gradient-tracked values, inconsistent field devices, or a value that is not a
        dense (strided), non-quantized tensor. Falling back keeps a custom dataset working at the original throughput
        rather than failing on it.
    """
    if not targets:
        return tuple(targets)
    keys = tuple(sorted(targets[0]))
    values: dict[str, Tensor] = {}
    shapes: dict[str, list[list[int]]] = {}
    for target in targets:
        if tuple(sorted(target)) != keys:
            return tuple(targets)
        for value in target.values():
            # A sparse value has no linear element order for reshape(-1) to preserve, so torch.cat raises
            # RuntimeError instead of falling back. A quantized value can share torch.dtype with another sample
            # while using a different (scale, zero_point), in which case torch.cat silently requantizes it to
            # the first sample's parameters rather than failing -- dtype equality alone does not catch either case.
            if (
                not isinstance(value, Tensor)
                or value.layout != torch.strided
                or value.is_quantized
                or value.requires_grad
            ):
                return tuple(targets)
    for key in keys:
        # torch.cat promotes across mixed dtypes rather than failing, which would silently change a field's
        # dtype and can lose integer precision. A COCO sample with no annotations yields float32 for fields a
        # populated sample yields as int64, so this is reachable on real data, not a theoretical case.
        dtype = targets[0][key].dtype
        device = targets[0][key].device
        if any(target[key].dtype != dtype or target[key].device != device for target in targets):
            return tuple(targets)
    for key in keys:
        column = [target[key] for target in targets]
        values[key] = torch.cat([entry.reshape(-1) for entry in column])
        shapes[key] = [[int(dim) for dim in entry.shape] for entry in column]
    return PackedTargets(values, shapes, len(targets))
