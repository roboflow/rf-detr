# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared image preparation and Supervision result decoding."""

from __future__ import annotations

import io
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image

from rfdetr.models.postprocess import PostProcess
from rfdetr.utilities.class_names import is_coco_pretrained
from rfdetr.utilities.keypoints import _is_bg_first_schema, precision_cholesky_to_pixel_covariance
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from supervision import Detections, KeyPoints

logger = get_logger()


class PredictionConfig(Protocol):
    """Expose the architecture fields needed to validate prediction inputs."""

    @property
    def patch_size(self) -> int:
        """Return the backbone patch size."""
        ...

    @property
    def num_windows(self) -> int:
        """Return the attention window count."""
        ...

    @property
    def num_channels(self) -> int:
        """Return the image channel count."""
        ...


@dataclass(frozen=True)
class PredictionContext:
    """Hold one prediction call's execution and decoding rules."""

    config: PredictionConfig
    device: torch.device
    default_shape: tuple[int, int]
    means: list[float]
    stds: list[float]
    class_names: list[str]
    class_id_to_name: dict[int, str]
    num_classes: int
    num_keypoints_per_class: list[int]
    postprocess: PostProcess
    run: Callable[[torch.Tensor], dict[str, torch.Tensor]]
    runtime_info: dict[str, Any]
    prepare: Callable[[], None] | None = None
    fixed_shape: bool = False
    batch_size: int | None = None
    max_batch_size: int | None = None


def _tensor_to_source_array(image: torch.Tensor) -> np.ndarray[Any, Any]:
    """Convert a normalized CHW tensor into the uint8 HWC source-image representation.

    For a CUDA tensor in ``float16``/``float32``/``float64`` with every dimension greater than one, the
    multiply-then-truncate cast runs on-device before the (now ``uint8``) transfer; every other input,
    including CPU tensors and CUDA tensors of another dtype such as ``bfloat16``, keeps the previous
    NumPy-side conversion. NaN pixels convert successfully on both paths, but only the NumPy-side path
    emits an incidental invalid-cast ``RuntimeWarning`` for them; the on-device path does not.

    Args:
        image: Source tensor in channel-first layout.

    Returns:
        The writable, owning NumPy array stored in prediction metadata.

    Examples:
        >>> source = _tensor_to_source_array(torch.zeros(3, 2, 2))
        >>> source.shape
        (2, 2, 3)
    """
    source_view = image.permute(1, 2, 0)
    if (
        image.device.type == "cuda"
        and image.dtype in (torch.float16, torch.float32, torch.float64)
        and all(size > 1 for size in image.shape)
    ):
        # Preserve the existing multiply-then-truncate result, but transfer one byte per channel instead of a
        # floating-point image before NumPy performs the same conversion on the host.
        # ``copy(order="K")`` retains NumPy ownership and the channel-major strides produced by the existing cast.
        # NumPy types ``ndarray.copy`` as ``Any``, so ``asarray`` restores the element type; it returns the very
        # same object for an ndarray, keeping the copy's ownership, writability, and strides untouched.
        return np.asarray(source_view.mul(255).to(torch.uint8).cpu().numpy().copy(order="K"))
    return (source_view.cpu().numpy() * 255).astype(np.uint8)


def _uint8_image_to_chw_view(image: np.ndarray[Any, Any]) -> torch.Tensor:
    """Return a zero-copy ``uint8`` CHW *view* of a 2-D/3-D HWC image array.

    This is the layout half of ``torchvision.transforms.functional.to_tensor``; the dtype half is
    :func:`_uint8_chw_to_float`. Splitting them lets :meth:`RFDETR.predict` send the
    1-byte-per-channel storage across the host-to-device boundary and widen it on the accelerator,
    instead of widening it 4x on the host and transferring that.

    Args:
        image: A ``(H, W)`` grayscale or ``(H, W, C)`` HWC ``uint8`` array.

    Returns:
        A ``(C, H, W)`` ``uint8`` tensor sharing *image*'s storage.

    Examples:
        >>> arr = np.zeros((2, 2, 3), dtype=np.uint8)
        >>> _uint8_image_to_chw_view(arr).shape
        torch.Size([3, 2, 2])
    """
    if image.ndim == 2:
        image = image[:, :, None]
    return torch.from_numpy(image.transpose((2, 0, 1)))


def _uint8_chw_to_float(chw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Widen a ``uint8`` CHW tensor to the default float dtype and scale it into ``[0, 1]``.

    Dtype and layout are converted together and the division is done in place on that fresh float
    allocation, so the pair costs one allocation rather than ``to_tensor``'s three.

    Args:
        chw: ``(C, H, W)`` ``uint8`` tensor, on any device.
        scale: 0-dim tensor holding ``255``, on ``chw``'s device and in the target dtype. It has to
            be a tensor rather than the Python ``int``: CUDA evaluates ``Tensor.div_(255)`` as a
            multiplication by ``255``'s reciprocal, which rounds differently from the host's
            division for just under half of all byte values (126 of 256, 1 ULP). A tensor divisor
            keeps IEEE-754 correctly rounded division on both, so the result does not depend on
            where it was computed.

    Returns:
        A contiguous ``(C, H, W)`` tensor in the current default floating-point dtype, scaled to
        ``[0, 1]``, byte-for-byte identical to ``to_tensor``'s output. For ``C == 1`` the leading
        dimension's own stride may differ from ``to_tensor``'s, which is immaterial: a size-1
        dimension's stride has no memory-layout effect.

    Examples:
        >>> chw = _uint8_image_to_chw_view(np.zeros((2, 2, 3), dtype=np.uint8))
        >>> _uint8_chw_to_float(chw, torch.tensor(255.0)).dtype
        torch.float32
    """
    widened = chw.to(dtype=torch.get_default_dtype(), memory_format=torch.contiguous_format)
    return widened.div_(scale)


def _validate_shape_dims(
    shape: object,
    block_size: int,
    patch_size: int,
    num_windows: int,
) -> tuple[int, int]:
    """Validate a user-supplied ``(height, width)`` shape tuple and return normalised plain-int dims.

    Args:
        shape: The raw value supplied by the caller (e.g. from ``export(shape=...)`` or
            ``predict(shape=...)``).  Must be a two-element sequence of positive integers (or integer-compatible types
            accepted by :func:`operator.index`).
        block_size: Required divisor for both dimensions.  Equals ``patch_size * num_windows``.
        patch_size: Backbone patch size — used only in error messages.
        num_windows: Number of attention windows — used only in error messages.

    Returns:
        A ``(height, width)`` tuple of plain Python :class:`int` values.

    Raises:
        ValueError: If ``shape`` cannot be unpacked as a two-element sequence, if either
            dimension is a bool, float, or other non-integer type, if either dimension is not positive, or if either
            dimension is not divisible by ``block_size``.
    """
    try:
        raw_height, raw_width = cast("tuple[Any, Any]", shape)
    except (TypeError, ValueError):
        raise ValueError(f"shape must be a sequence of two positive integers (height, width), got {shape!r}.") from None
    for dim_name, dim in (("height", raw_height), ("width", raw_width)):
        if isinstance(dim, bool):
            raise ValueError(f"shape {dim_name} must be an integer, got {type(dim).__name__} (shape={shape!r}).")
        try:
            operator.index(dim)
        except TypeError:
            raise ValueError(
                f"shape {dim_name} must be an integer, got {type(dim).__name__} (shape={shape!r}).",
            ) from None
        if dim <= 0:
            raise ValueError(f"shape must contain positive integers for height and width, got {shape!r}.")
    # Normalise to plain Python ints; also accepts numpy.int64, torch scalars, etc.
    height, width = operator.index(raw_height), operator.index(raw_width)
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"shape must have both dimensions divisible by {block_size} "
            f"(patch_size={patch_size} * num_windows={num_windows}), got {shape!r}.",
        )
    return height, width


def _resolve_patch_size(patch_size: int | None, model_config: object, caller: str) -> int:
    """Resolve and validate the ``patch_size`` argument for :meth:`RFDETR.export` and :meth:`RFDETR.predict`.

    Args:
        patch_size: Value supplied by the caller, or ``None`` to read from ``model_config``.
        model_config: The model's configuration object.  Must expose ``patch_size`` as a
            positive integer attribute when ``patch_size`` is ``None`` or when a mismatch check is needed.
        caller: Name of the calling method (``"export"`` or ``"predict"``) — used in
            error messages to help the caller locate the problem.

    Returns:
        A validated, positive :class:`int` patch size.

    Raises:
        ValueError: If the resolved or provided ``patch_size`` is not a positive integer,
            or if a caller-provided value disagrees with ``model_config.patch_size``.
    """
    if patch_size is None:
        patch_size = getattr(model_config, "patch_size", 14)
    else:
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
        model_patch_size = getattr(model_config, "patch_size", None)
        if model_patch_size is not None and patch_size != model_patch_size:
            raise ValueError(
                f"{caller}(patch_size={patch_size}) does not match the instantiated model's "
                f"patch_size={model_patch_size}. Patch size is an architectural parameter; "
                f"omit patch_size to use the model's configured value.",
            )
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
    return patch_size


@torch.inference_mode()
def predict(
    context: PredictionContext,
    images: str
    | Image.Image
    | np.ndarray[Any, Any]
    | torch.Tensor
    | list[str | Image.Image | np.ndarray[Any, Any] | torch.Tensor],
    threshold: float = 0.5,
    shape: tuple[int, int] | None = None,
    patch_size: int | None = None,
    include_source_image: bool = True,
    **kwargs: Any,
) -> Detections | KeyPoints | list[Detections | KeyPoints]:
    """Prepare images, execute one batch, and decode results using a shared contract.

    Args:
        context: Execution and decoding rules for this call.
        images: An RGB image or a list or tuple of images.
        threshold: Minimum confidence for a result.
        shape: Input height and width.
        patch_size: Backbone patch size for shape validation.
        include_source_image: Store the source image in result metadata.
        **kwargs: Reserved prediction arguments.

    Returns:
        One Supervision result or a list matching the input container.
    """
    from supervision import Detections, KeyPoints

    if context.fixed_shape:
        shape = context.default_shape if shape is None else shape
        batch_size = len(images) if isinstance(images, (list, tuple)) else 1
        if batch_size <= 0 or (context.batch_size is not None and batch_size != context.batch_size):
            raise ValueError(f"Batch size mismatch. Export requires {context.batch_size}, but got {batch_size}.")
        if context.max_batch_size is not None and batch_size > context.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds export maximum {context.max_batch_size}.")

    patch_size = _resolve_patch_size(patch_size, context.config, "predict")
    num_windows = getattr(context.config, "num_windows", 1)
    if isinstance(num_windows, bool) or not isinstance(num_windows, int) or num_windows <= 0:
        raise ValueError(f"model_config.num_windows must be a positive integer, got {num_windows!r}")
    block_size = patch_size * num_windows

    if shape is None:
        default_res = context.default_shape[0]
        if default_res % block_size != 0:
            raise ValueError(
                f"Model's default resolution ({default_res}) is not divisible by "
                f"block_size={block_size} (patch_size={patch_size} * num_windows={num_windows}). "
                f"Provide an explicit shape divisible by {block_size}.",
            )
    else:
        shape = _validate_shape_dims(shape, block_size, patch_size, num_windows)
        if context.fixed_shape and shape != context.default_shape:
            raise ValueError(f"Export requires shape {context.default_shape}, but got {shape}.")

    if context.prepare is not None:
        context.prepare()

    # Determine the return shape from the *input* type, not the runtime batch
    # length: a single image (path / PIL / tensor) yields a bare Detections,
    # while a list/tuple always yields a list — even when it holds one image.
    single_input = not isinstance(images, (list, tuple))
    if not isinstance(images, (list, tuple)):
        images = [images]

    orig_sizes: list[Any] = []
    processed_images: list[Any] = []
    source_images: list[Any] | None = [] if include_source_image else None
    # Tensor range checks stay deferred: `(img > 1).any()` itself is a cheap async kernel launch, but
    # consuming its result in `if ...:` forces Python to call `Tensor.__bool__()`, which blocks
    # the calling thread until the device catches up. For a CUDA tensor passed directly to
    # `predict()` (the documented host-round-trip-free path, see the Note above on pinning), doing
    # that inline inside this loop serializes every image behind its own blocking round-trip,
    # defeating the non-blocking transfers below. Collecting the (still un-synced) result tensors
    # here and only forcing them to Python bools once, after every image has had its conversion,
    # range-check kernels, and transfer all queued, lets the sync for image 1 overlap with the GPU
    # work already queued for images 2..N instead of blocking in front of it. Kept per-image
    # (not `torch.stack`-ed into one combined check) because the images in one `predict()` call
    # are not guaranteed to share a device (e.g. a CPU-tensor image and a CUDA-tensor image mixed
    # in the same list) — stacking would raise instead of validating each on its own device.
    # The shape check below costs no sync (a plain Python int comparison on `.shape[0]`), but its
    # raise is deferred here too, and re-ordered after both range checks in the loop below: the
    # original code checked range before shape for a given image, and raising it eagerly here
    # would flip that precedence for any tensor that is invalid on both axes at once.
    pending_checks: list[tuple[torch.Tensor | bool, torch.Tensor | bool, bool, tuple[int, ...]]] = []
    # Built lazily on the first uint8 image, then shared by the rest of the batch.
    uint8_scale: torch.Tensor | None = None

    for img_input in images:
        img: Any = img_input
        if isinstance(img, str):
            if urlparse(img).scheme in ("http", "https"):
                resp = requests.get(img, timeout=30)
                resp.raise_for_status()
                img = io.BytesIO(resp.content)
            img = Image.open(img)

        range_known_valid = False
        deferred_widen = False
        if not isinstance(img, torch.Tensor):
            # Auto-convert PIL images from any colour mode (L, LA, RGBA, P,
            # etc.) to RGB before converting to tensor.  This matches the
            # standard detector API contract: callers passing a file path or
            # a PIL image should not have to pre-convert; for tensor inputs
            # the channel dimension is the caller's responsibility.
            if isinstance(img, Image.Image) and img.mode != "RGB":
                img = img.convert("RGB")
            pil_image = isinstance(img, Image.Image)
            source_array: np.ndarray[Any, Any] | None = None
            if include_source_image:
                source_array = np.array(img)
                if source_array.dtype != np.uint8:
                    source_array = (source_array * 255).clip(0, 255).astype(np.uint8)
                source_images.append(source_array)  # type: ignore[union-attr]
            uint8_array = isinstance(img, np.ndarray) and img.dtype == np.uint8
            # PIL conversion above guarantees an 8-bit RGB image, and both conversion paths below
            # scale PIL and uint8 NumPy storage into [0, 1]. Their range cannot fail the checks below.
            range_known_valid = pil_image or uint8_array
            if pil_image or (isinstance(img, np.ndarray) and uint8_array and img.ndim in (2, 3)):
                # ``F.to_tensor`` first materializes contiguous CHW uint8 storage, then
                # allocates float storage, then allocates again for division. Convert dtype
                # and layout together and divide that fresh float allocation in place.
                if pil_image:
                    tensor_source = (
                        source_array if source_array is not None else np.array(img, dtype=np.uint8, copy=True)
                    )
                else:
                    tensor_source = cast(np.ndarray[Any, Any], img)
                # Keep the 1-byte-per-channel storage for now: the widening to float is
                # deferred until after the host-to-device transfer below, so only a quarter of
                # the bytes cross the bus and the widen+divide run on the accelerator. The view
                # is already (C, H, W), so every shape check and error message below is
                # unchanged.
                img = _uint8_image_to_chw_view(tensor_source)
                deferred_widen = True
            else:
                img = F.to_tensor(img)
        elif include_source_image and img.dim() == 3:
            # Source extraction requires a (C, H, W) tensor for permute(). Skip malformed ranks so the deferred
            # validation below raises the public shape error instead of an internal RuntimeError.
            source_images.append(_tensor_to_source_array(img))  # type: ignore[union-attr]

        # img.dim() != 3 is checked alongside the channel count (not just deferred as a message
        # detail) because `h, w = img_tensor.shape[1:]` a few lines down unpacks exactly 2 values --
        # deferring only the *raise* while still unconditionally unpacking a non-3D tensor's shape
        # would trade the clear "Invalid tensor image shape" error for a confusing internal
        # `ValueError: not enough values to unpack` (or, for a 0-d/1-d tensor, an IndexError out of
        # `img.shape[0]` itself) the moment a malformed tensor reached this point.
        invalid_shape = img.dim() != 3 or img.shape[0] != context.config.num_channels
        pending_checks.append(
            (
                False if range_known_valid else (img > 1).any(),
                False if range_known_valid else (img < 0).any(),
                invalid_shape,
                tuple(img.shape),
            )
        )
        img_tensor = img

        if invalid_shape:
            # Already known to be un-usable -- record a placeholder so `processed_images`/
            # `orig_sizes` don't silently go missing an entry (kept parallel with `pending_checks`
            # for clarity, even though the loop below is guaranteed to raise on this image's
            # `invalid_shape` before either list is ever read), and skip the size unpacking and
            # transfer that assume a valid (C, H, W) tensor.
            orig_sizes.append(None)
            processed_images.append(None)
            continue

        h, w = img_tensor.shape[1:]
        orig_sizes.append((h, w))

        # A pageable-memory .to(device) copy onto CUDA is slower than a pinned-memory one: the driver has to
        # pin the source buffer itself before it can start the transfer. Pin it explicitly here — but only for a
        # CPU tensor headed to an accelerator; pin_memory() raises on a tensor the caller already placed on the
        # accelerator (a legitimate tensor-input use to skip a host round-trip), and pinning buys nothing when
        # the target device is the CPU itself.
        if img_tensor.device.type == "cpu" and context.device.type == "cuda":
            img_tensor = img_tensor.pin_memory()
        # non_blocking only pays off (and is only safe without an explicit sync) when the destination is CUDA,
        # matching the transfer_batch_to_device() convention in training/module_data.py: a CUDA-tensor-input ->
        # CPU-model transfer with non_blocking=True races the copy — the CPU destination is never pinned, so
        # reads of the tensor's data can observe an in-flight (partially written) copy.
        non_blocking = context.device.type == "cuda"
        img_tensor = img_tensor.to(context.device, non_blocking=non_blocking)
        if deferred_widen:
            if uint8_scale is None:
                uint8_scale = torch.tensor(255, device=img_tensor.device, dtype=torch.get_default_dtype())
            img_tensor = _uint8_chw_to_float(img_tensor, uint8_scale)
        processed_images.append(img_tensor)

    # Force the range-check results to Python bools only now, after every image's conversion,
    # range-check kernels, and transfer have all been queued (see the comment where
    # pending_checks is built). Same nested per-image, per-condition order as the original inline
    # checks (image 0's "above 1", then "below 0", then its shape check; then image 1's, ...), so
    # which of the three messages a given multi-image, multi-violation input raises is unchanged.
    for invalid_high, invalid_low, invalid_shape, img_shape in pending_checks:
        if invalid_high:
            raise ValueError(
                "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1]).",
            )
        if invalid_low:
            raise ValueError(
                "Image has pixel values below 0. Please ensure the image is normalized (scaled to [0, 1]).",
            )
        if invalid_shape:
            raise ValueError(
                "Invalid tensor image shape. Tensor inputs to `predict()` must be in (C, H, W) format "
                f"with C matching the model configuration ({context.config.num_channels} channels). "
                f"Received tensor with shape {img_shape}. "
                "For automatic RGB conversion, pass a PIL Image or a file path instead of a tensor."
            )

    resize_to = list(shape) if shape is not None else list(context.default_shape)
    # antialias=False matches the antialias-free bilinear resize (cv2.INTER_LINEAR)
    # used by Albumentations during training — see issue #1203.
    batch_tensor = torch.stack([F.resize(t, resize_to, antialias=False) for t in processed_images])
    batch_tensor = F.normalize(batch_tensor, context.means, context.stds)

    predictions = context.run(batch_tensor)
    target_sizes = torch.tensor(orig_sizes, device=context.device)
    results = context.postprocess(predictions, target_sizes=target_sizes, score_threshold=threshold)

    _class_id_to_name = context.class_id_to_name
    num_logit_slots = context.num_classes
    _is_coco_pretrained = is_coco_pretrained(context.class_names, num_logit_slots)
    _is_legacy_bgfirst_keypoint = _is_bg_first_schema(context.num_keypoints_per_class)
    predictions_list: list[Detections | KeyPoints] = []
    for i, result in enumerate(results):
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]

        # INVARIANT: this predicate must stay identical (same operator and threshold) to the
        # pre-filter in PostProcess._postprocess_masks (`scores_i > score_threshold`), which is
        # fed `score_threshold=threshold` above. The seg path drops below-threshold masks before
        # upsampling on the strength of that match; diverging here (e.g. `>=`, per-class, top-k)
        # would make it silently drop rows this filter keeps — a behaviour change with no failing test.
        # Materialized as an index vector (not a bool mask) so the bool-mask advanced-indexing
        # sync below runs once per image instead of once per kept tensor: `t[bool_mask]` re-derives
        # its output size via `nonzero(bool_mask)` on every call (a documented CUDA host-device
        # sync), while `t[int64_index]` is index_select-shaped and needs no further sync. Row order
        # is unaffected — `nonzero` returns ascending indices, identical to bool-mask selection order.
        keep_idx = (scores > threshold).nonzero(as_tuple=True)[0]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
        boxes = boxes[keep_idx]
        has_keypoints_result = "keypoints" in result
        has_masks = "masks" in result
        has_kp_precision = "keypoint_precision_cholesky" in result
        # Bound unconditionally (None when absent) rather than left unbound under a guard: these
        # locals live inside `for i, result in enumerate(results)`, so a guard that silently
        # diverges from its transfer/consume counterparts would otherwise carry the *previous*
        # image's tensor forward instead of raising. Rebinding every iteration makes that failure
        # mode a loud AttributeError/TypeError on the stale `None` instead of a silent wrong value.
        keypoints = result["keypoints"][keep_idx] if has_keypoints_result else None
        masks = result["masks"][keep_idx] if has_masks else None
        keypoint_precision = result["keypoint_precision_cholesky"][keep_idx] if has_kp_precision else None

        # PERF: queue every GPU->CPU transfer for this image as non_blocking, then
        # synchronize ONCE instead of once per tensor. Each bare `.cpu()` call below
        # used to impose its own stream synchronization -- up to 4 (5 with keypoints,
        # 6 with keypoint precision) sequential blocking round-trips per image, scaling
        # with the number of kept detections (the mask tensor especially, since it is
        # by far the largest of the group in a crowded/high-detection-count frame).
        # Queuing the copies together lets them share the copy stream and collapses
        # the wait to a single barrier; `.numpy()` is only called after that barrier,
        # so every array below is fully populated exactly as before.
        #
        # Restrict the async path to CUDA and synchronize the tensors' own device's
        # current stream (not the whole device, and not "the current device") -- a bare
        # `torch.cuda.synchronize()` waits on every stream on `torch.cuda.current_device()`,
        # which can differ from `boxes.device` on a multi-GPU setup with a model on a
        # non-default device (e.g. `cuda:1`), and non-CUDA accelerators (e.g. MPS) have
        # no synchronization here at all, so a `non_blocking=True` copy there could be
        # read before it lands. Falling back to the original blocking transfer for
        # non-CUDA devices keeps every other backend exactly as correct as before this
        # change; the coalesced-sync optimization is only claimed for CUDA in the first
        # place.
        # INVARIANT: keypoints/masks/keypoint_precision above all come from this same
        # `result` dict, i.e. one `postprocess()` call on one device — so `is_cuda`,
        # derived from `boxes` alone, applies identically to every field below, and the
        # single stream sync a few lines down (scoped to `boxes.device`) covers all of
        # them. Not enforced here (would be hot-loop validation for a condition that
        # can't happen with today's callers) — holds only as long as `postprocess()`
        # never returns fields split across devices.
        is_cuda = boxes.is_cuda
        boxes_cpu = boxes.float().to("cpu", non_blocking=is_cuda)
        scores_cpu = scores.float().to("cpu", non_blocking=is_cuda)
        labels_cpu = labels.to("cpu", non_blocking=is_cuda)
        keypoints_cpu = keypoints.float().to("cpu", non_blocking=is_cuda) if keypoints is not None else None
        masks_cpu = masks.squeeze(1).to("cpu", non_blocking=is_cuda) if masks is not None else None
        keypoint_precision_cpu = (
            keypoint_precision.float().to("cpu", non_blocking=is_cuda) if keypoint_precision is not None else None
        )
        if is_cuda:
            torch.cuda.current_stream(boxes.device).synchronize()

        keypoints_array = keypoints_cpu.numpy() if keypoints_cpu is not None else None
        has_keypoints = keypoints_array is not None

        if masks_cpu is not None:
            detections = Detections(
                xyxy=boxes_cpu.numpy(),
                confidence=scores_cpu.numpy(),
                class_id=labels_cpu.numpy(),
                mask=masks_cpu.numpy(),
            )
        else:
            detections = Detections(
                xyxy=boxes_cpu.numpy(),
                confidence=scores_cpu.numpy(),
                class_id=labels_cpu.numpy(),
            )
        if keypoint_precision_cpu is not None:
            detections.data["keypoint_precision_cholesky"] = keypoint_precision_cpu.numpy()

        if include_source_image:
            detections.metadata["source_image"] = source_images[i]  # type: ignore[index]
        detections.data["source_shape"] = np.tile(np.array(orig_sizes[i], dtype=np.int64), (len(detections), 1))

        # Attach class names so callers can map class_id → name without a
        # separate lookup. Always set data["class_name"] for a consistent interface.
        #
        # For fine-tuned models, logit index num_logit_slots is the no-object slot —
        # map it to "__background__" without warning. For COCO-pretrained models,
        # background is implicit (filtered by threshold); class ID 90 is "toothbrush".
        # IDs not in _class_id_to_name are genuinely unexpected and produce an empty
        # string with a one-time warning.
        class_ids = detections.class_id if detections.class_id is not None else np.array([], dtype=int)
        # Sentinel for the no-object / background class differs by model type.
        # Legacy background-first keypoint models: slot 0 is background in the keypoint schema.
        # Detection/segmentation models: the no-object slot is at index num_logit_slots.
        _bg_sentinel = 0 if _is_legacy_bgfirst_keypoint else num_logit_slots
        truly_oob = [cid for cid in class_ids if cid not in _class_id_to_name and cid != _bg_sentinel]
        if truly_oob:
            logger.warning_once(
                "predict() encountered unmapped class_id(s): %s — mapping to empty string",
                truly_oob[:5],
            )
        if _is_coco_pretrained:
            class_names = [_class_id_to_name.get(cid, "") for cid in class_ids]
        else:
            class_names = [
                "__background__" if cid == _bg_sentinel else _class_id_to_name.get(cid, "") for cid in class_ids
            ]
        detections.data["class_name"] = np.array(class_names, dtype=object)

        if has_keypoints and keypoints_array is not None:
            keypoint_data = dict(detections.data)
            keypoint_data["xyxy"] = detections.xyxy.astype(np.float32)
            if include_source_image:
                keypoint_data["source_image"] = [
                    source_images[i]  # type: ignore[index]
                    for _ in range(len(detections))
                ]
            raw_precision = keypoint_data.get("keypoint_precision_cholesky")
            raw_source_shape = keypoint_data.get("source_shape")
            if raw_precision is not None and raw_source_shape is not None and len(detections) > 0:
                precision = np.asarray(raw_precision, dtype=np.float32)
                source_shape = np.asarray(raw_source_shape, dtype=np.float32)
                if precision.shape[:2] == keypoints_array.shape[:2] and source_shape.shape == (len(detections), 2):
                    keypoint_data["covariance"] = precision_cholesky_to_pixel_covariance(
                        precision_cholesky=precision, source_shape=source_shape
                    )
            keypoints_array = keypoints_array.astype(np.float32, copy=False)
            keypoint_confidence = keypoints_array[:, :, 2]
            key_points = KeyPoints(
                xy=keypoints_array[:, :, :2],
                keypoint_confidence=keypoint_confidence,
                detection_confidence=detections.confidence.astype(np.float32)
                if detections.confidence is not None
                else None,
                class_id=detections.class_id.astype(int) if detections.class_id is not None else None,
                visible=keypoint_confidence > 0,
                data=keypoint_data,
            )
            predictions_list.append(key_points)
        else:
            predictions_list.append(detections)

    return predictions_list[0] if single_input else predictions_list
