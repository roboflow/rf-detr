# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared image preparation and Supervision result decoding."""

from __future__ import annotations

import functools
import io
import operator
import os
import re
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image, JpegImagePlugin

try:
    # torchvision 0.29 deprecated these codecs for removal in favour of TorchCodec. Once they are gone, local files
    # passed to predict() decode through Pillow instead of ``import rfdetr`` failing.
    from torchvision.io import ImageReadMode, decode_image, decode_jpeg

    _IS_TORCHVISION_IMAGE_DECODE_AVAILABLE = True
except ImportError:
    _IS_TORCHVISION_IMAGE_DECODE_AVAILABLE = False

from rfdetr._prediction.sources import has_only_finite_images, is_expanded_source
from rfdetr._prediction.thresholds import _class_thresholds
from rfdetr._prediction.types import _ImageInput, _PredictionInput, _ThresholdInput
from rfdetr.models.postprocess import PostProcess
from rfdetr.utilities.keypoints import _is_bg_first_schema, precision_cholesky_to_pixel_covariance
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr._prediction.types import _PredictionOutput, _PredictionResult

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
    antialias: bool = False


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
        image: A ``(H, W)`` grayscale or ``(H, W, C)`` HWC ``uint8`` array without negative strides, which
            ``torch.from_numpy`` rejects. :meth:`RFDETR.predict` copies such arrays before calling this.

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


#: Byte offset of the bit depth in a PNG file: the 8-byte signature, then the IHDR chunk's length, type, width and
#: height (4 bytes each).
_PNG_BIT_DEPTH_OFFSET = 24

#: Pillow's chroma subsampling codes for the JPEG files the GPU may decode: 4:4:4, 4:2:2 and 4:2:0.
_GPU_JPEG_SAMPLINGS = frozenset({0, 1, 2})

#: Whether the GPU decode gave the CPU decoder's exact pixels for this process's first JPEG of each (CUDA device index,
#: chroma subsampling). nvjpeg picks a hardware or a CUDA decoder by GPU, so the comparison runs once per process and
#: device, not once per release. ``False`` sends every later JPEG of that kind back to the CPU decoder. Two threads may
#: both run the first comparison; both then record the same result.
_GPU_JPEG_MATCHES: dict[tuple[int | None, int], bool] = {}


@dataclass(frozen=True)
class _EncodedJpeg:
    """A JPEG file read and header-checked ahead of its batch, left for the GPU to decode with the rest of the batch.

    Attributes:
        data: The file's bytes, as the one-dimensional ``uint8`` CPU tensor :func:`~torchvision.io.decode_jpeg` reads.
        sampling: Its chroma subsampling, as Pillow reports it; one of :data:`_GPU_JPEG_SAMPLINGS`.
    """

    data: torch.Tensor
    sampling: int


def _decodes_on_gpu(header: Image.Image, device: torch.device) -> bool:
    """Tell whether a JPEG file's header lets the GPU decode it to the pixels the CPU decoder gives.

    Only baseline colour JPEGs with 4:4:4, 4:2:2 or 4:2:0 chroma qualify. Grayscale, CMYK and RGB-coded Adobe files
    convert colour in steps nvjpeg need not share with libjpeg, and progressive files take nvjpeg's slower
    host-assisted decoder. A kind that :data:`_GPU_JPEG_MATCHES` records as differing on ``device`` stays on the CPU.

    Args:
        header: The file, opened by Pillow; only its header has been read.
        device: The CUDA device that would decode it.

    Returns:
        Whether to leave the file encoded for :func:`_decode_jpegs_on_gpu`.

    Examples:
        >>> buffer = io.BytesIO()
        >>> Image.new("RGB", (16, 16)).save(buffer, "JPEG")
        >>> _decodes_on_gpu(Image.open(buffer), torch.device("cuda", 0))
        True
        >>> Image.new("RGB", (16, 16)).save(buffer := io.BytesIO(), "JPEG", progressive=True)
        >>> _decodes_on_gpu(Image.open(buffer), torch.device("cuda", 0))
        False
    """
    if header.format != "JPEG" or header.mode != "RGB" or header.info.get("progressive"):
        return False
    # An Adobe marker with transform 0 means the components are R, G and B, not Y, Cb and Cr.
    if header.info.get("adobe_transform") == 0:
        return False
    sampling = JpegImagePlugin.get_sampling(header)
    return sampling in _GPU_JPEG_SAMPLINGS and _GPU_JPEG_MATCHES.get((device.index, sampling), True)


def _decode_on_cpu(data: torch.Tensor) -> torch.Tensor | None:
    """Decode a JPEG or PNG file's bytes into RGB ``uint8`` CHW storage on the CPU.

    Args:
        data: The file's bytes, as a one-dimensional ``uint8`` CPU tensor.

    Returns:
        The decoded image, or ``None`` when the caller should open the file with Pillow.

    Examples:
        >>> buffer = io.BytesIO()
        >>> Image.new("RGB", (5, 3), (10, 20, 30)).save(buffer, "PNG")
        >>> _decode_on_cpu(torch.frombuffer(buffer.getbuffer(), dtype=torch.uint8)).shape
        torch.Size([3, 3, 5])
    """
    try:
        decoded = decode_image(data, mode=ImageReadMode.RGB)
    except (RuntimeError, DeprecationWarning, AttributeError):
        # torchvision raises RuntimeError for bytes it cannot decode, and torch raises AttributeError when the image
        # operators were not loaded. torchvision 0.29+ also warns on every call that its codecs are deprecated, which
        # ``-W error`` turns into a raised DeprecationWarning.
        return None
    # Only 8-bit data may reach the uint8 widening in predict(); any other decode stays on the compatibility path.
    return decoded if decoded.dtype == torch.uint8 else None


def _decode_jpegs_on_gpu(encoded: list[_EncodedJpeg], device: torch.device) -> list[torch.Tensor | None]:
    """Decode a batch's encoded JPEG files on the GPU in one nvjpeg call, keeping the CPU decoder's pixels.

    The first file of each chroma subsampling on ``device`` is also decoded on the CPU and compared, and the result is
    recorded in :data:`_GPU_JPEG_MATCHES`. A file whose kind differs, or a batch nvjpeg refuses (a corrupt file the CPU
    decoder tolerates, a torchvision without nvjpeg), decodes on the CPU, and later files of those kinds decode on the
    CPU ahead of their batch, as without this path.

    Args:
        encoded: Files :func:`_decode_local_image` left encoded.
        device: The CUDA device to decode on.

    Returns:
        For each file, in order, its RGB ``uint8`` CHW image on ``device``, or on the CPU when it decoded there, or
        ``None`` when the caller should open it with Pillow.
    """
    try:
        decoded = decode_jpeg([item.data for item in encoded], mode=ImageReadMode.RGB, device=device)
    except (RuntimeError, DeprecationWarning, AttributeError) as error:
        for item in encoded:
            _GPU_JPEG_MATCHES[(device.index, item.sampling)] = False
        logger.info(f"Decoding JPEG files on the CPU: the GPU decoder failed ({error}).")
        return [_decode_on_cpu(item.data) for item in encoded]
    images: list[torch.Tensor | None] = []
    for item, on_gpu in zip(encoded, decoded):
        key = (device.index, item.sampling)
        if key not in _GPU_JPEG_MATCHES:
            on_cpu = _decode_on_cpu(item.data)
            _GPU_JPEG_MATCHES[key] = on_cpu is not None and torch.equal(on_gpu.cpu(), on_cpu)
            if not _GPU_JPEG_MATCHES[key]:
                logger.info(
                    f"Decoding JPEG files with chroma subsampling {item.sampling} on the CPU: the GPU decoder "
                    "gave different pixels on this device."
                )
            images.append(on_gpu if _GPU_JPEG_MATCHES[key] else on_cpu)
        elif _GPU_JPEG_MATCHES[key]:
            images.append(on_gpu)
        else:
            images.append(_decode_on_cpu(item.data))
    return images


def _decode_local_image(path: str, gpu_device: torch.device | None = None) -> torch.Tensor | _EncodedJpeg | None:
    """Decode a local JPEG or PNG directly into RGB ``uint8`` CHW storage.

    Pillow remains the compatibility path for every other format and for files that torchvision
    cannot decode. Restricting the fast path to these two measured formats also avoids changing
    animated-image semantics: torchvision represents animated GIFs as a four-dimensional tensor,
    while Pillow exposes the first frame.

    This is separate from :func:`rfdetr.datasets.io_utils.decode_image`, which the dataset readers use:
    torchvision is a core dependency and yields the CHW tensor ``predict()`` consumes, whereas that
    function's ``simplejpeg`` fast path needs the ``[train]`` extra and returns HWC NumPy arrays.

    The file is read into memory once. Pillow validates the header of that in-memory buffer and
    torchvision decodes the same buffer, so a file changed on disk after the read cannot skip
    Pillow's decompression-bomb limit. The cost is that the whole encoded file is in memory before
    Pillow inspects its header, whereas opening the path with Pillow reads only the header; a file
    that ends up on the Pillow path pays that read before Pillow opens it again.

    Unlike Pillow, torchvision's bundled libjpeg and libpng print their messages about corrupt data
    (``Corrupt JPEG data: ...``) straight to the process's standard error. torchvision offers no hook to
    route them through ``logging``, and redirecting file descriptor 2 would affect the whole process,
    so they are left as is.

    With ``gpu_device``, a JPEG file whose header passes :func:`_decodes_on_gpu` is only read and checked here: the
    batch decodes it on that device with the rest of its JPEG files (:func:`_decode_jpegs_on_gpu`), which takes the
    CPU decode off the worker threads and the copy of the decoded pixels off the bus.

    Args:
        path: Local image path.
        gpu_device: The CUDA device the batch will decode suitable JPEG files on, or ``None`` to decode every file
            here.

    Returns:
        An RGB ``uint8`` CHW tensor, the file's bytes as an :class:`_EncodedJpeg` for the GPU, or ``None`` when the
        caller should use Pillow.

    Raises:
        Image.DecompressionBombError: If Pillow's header check rejects an oversized image.
    """
    if not _IS_TORCHVISION_IMAGE_DECODE_AVAILABLE or Path(path).suffix.lower() not in {".jpeg", ".jpg", ".png"}:
        return None
    try:
        stream = io.BytesIO(Path(path).read_bytes())
        # Opening is lazy: Pillow validates the header (including its decompression-bomb limit),
        # while torchvision still owns the expensive pixel decode below.
        with Image.open(stream) as header:
            if header.format not in {"JPEG", "PNG"} or getattr(header, "n_frames", 1) != 1:
                return None
            # torchvision 0.21+ decodes a 16-bit PNG to uint16 (older releases raise), and either way the result is
            # rejected below, so read the bit depth first rather than pay for a full decode.
            if header.format == "PNG" and stream.getbuffer()[_PNG_BIT_DEPTH_OFFSET] == 16:
                return None
            on_gpu = gpu_device is not None and _decodes_on_gpu(header, gpu_device)
            sampling = JpegImagePlugin.get_sampling(header) if on_gpu else -1
    except OSError:
        # A missing or unreadable path and bytes Pillow cannot identify raise OSError. The caller then opens the path
        # with Pillow, which reports it with its usual error.
        return None
    # ``getbuffer()`` exposes the stream's own storage, so torchvision decodes exactly the bytes Pillow validated.
    # The view is also writable, which ``torch.frombuffer`` expects; it warns on a read-only ``bytes`` object.
    data = torch.frombuffer(stream.getbuffer(), dtype=torch.uint8)
    return _EncodedJpeg(data, sampling) if on_gpu else _decode_on_cpu(data)


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


@functools.lru_cache(maxsize=8)
def _uint8_scale(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the ``255`` divisor :func:`_uint8_chw_to_float` takes, built once per device and dtype.

    Creating it on an accelerator is a blocking host-to-device copy (about 0.1 ms on MPS), paid on every batch before
    this cache. The tensor is only ever read.

    Args:
        device: Device the image tensors are on.
        dtype: Floating-point dtype they are widened to.

    Returns:
        A 0-dim tensor holding ``255``.

    Examples:
        >>> _uint8_scale(torch.device("cpu"), torch.float32)
        tensor(255.)
    """
    return torch.tensor(255, device=device, dtype=dtype)


@functools.lru_cache(maxsize=8)
def _normalization_tensors(
    means: tuple[float, ...], stds: tuple[float, ...], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(C, 1, 1)`` mean and std tensors that normalize a batch, built once per values, dtype and device.

    ``torchvision``'s ``normalize`` builds both on the device for every call and checks the std there, which on a
    synchronous accelerator such as MPS is two blocking copies and a sync per batch. The zero check runs on the host
    instead, after the same conversion to ``dtype``, so it rejects exactly what ``normalize`` rejects.

    Args:
        means: Per-channel means.
        stds: Per-channel standard deviations.
        dtype: Dtype of the batch.
        device: Device of the batch.

    Returns:
        The mean and std tensors, shaped to broadcast over ``(B, C, H, W)``.

    Raises:
        ValueError: If a std is zero after conversion to ``dtype``.

    Examples:
        >>> mean, std = _normalization_tensors((0.5,), (0.25,), torch.float32, torch.device("cpu"))
        >>> mean.shape, std.flatten().tolist()
        (torch.Size([1, 1, 1]), [0.25])
    """
    # Explicitly on the CPU: the process default device could be an accelerator, where the check would sync and a
    # float64 batch would ask MPS for a dtype it lacks.
    std = torch.as_tensor(stds, dtype=dtype, device="cpu")
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    mean = torch.as_tensor(means, dtype=dtype, device="cpu")
    return mean.to(device).view(-1, 1, 1), std.to(device).view(-1, 1, 1)


def _copy_results_to_cpu_once(results: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Copy a batch's boxes, scores and labels to the CPU in a single transfer.

    The per-image filter that follows would otherwise wait for the device twice per image, once to size its ``nonzero``
    index and once to copy the kept rows: MPS copies synchronously, and on CUDA both the ``nonzero`` size and the copy
    to the CPU sync. After this copy it runs on the CPU, where neither waits. The rows are packed in float32, or
    float64 when the scores or boxes are float64, so every value round-trips exactly: float16 and bfloat16 values, and
    class ids below 2**24, fit in float32. Every tensor gets its own dtype back, unlike :func:`_copy_rows_to_cpu_once`,
    which returns float32 boxes and scores for the keypoint and mask paths.

    Args:
        results: Per-image dictionaries holding ``boxes`` ``(N, 4)``, ``scores`` ``(N,)`` and ``labels`` ``(N,)``.

    Returns:
        The same dictionaries with contiguous CPU tensors.

    Examples:
        >>> rows = {"boxes": torch.ones(2, 4), "scores": torch.ones(2), "labels": torch.tensor([3, 90])}
        >>> [result["labels"].tolist() for result in _copy_results_to_cpu_once([rows, rows])]
        [[3, 90], [3, 90]]
    """
    # The threshold filter compares the scores after this copy, so a float32 packing of float64 scores would round a
    # score just above the threshold onto it and drop the row.
    dtype = torch.promote_types(
        torch.promote_types(torch.float32, results[0]["scores"].dtype), results[0]["boxes"].dtype
    )
    packed = torch.cat(
        [
            torch.cat(
                [r["boxes"].to(dtype), r["scores"].to(dtype).unsqueeze(1), r["labels"].to(dtype).unsqueeze(1)], dim=1
            )
            for r in results
        ]
    ).cpu()
    return [
        {
            "boxes": rows[:, :4].to(r["boxes"].dtype).contiguous(),
            "scores": rows[:, 4].to(r["scores"].dtype).contiguous(),
            "labels": rows[:, 5].to(r["labels"].dtype),
        }
        for r, rows in zip(results, torch.split(packed, [len(r["scores"]) for r in results]))
    ]


def _copy_rows_to_cpu_once(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy one image's boxes, scores and labels to the CPU in a single transfer.

    Accelerators other than CUDA (MPS among them) copy to the host synchronously, so each ``.cpu()`` waits on its own;
    packing the three into one float32 tensor pays that wait once. Labels round-trip exactly through float32 because
    class ids stay far below 2**24, the largest integer it represents without rounding. Unlike
    :func:`_copy_results_to_cpu_once`, it runs after the threshold filter and returns float32 boxes and scores, which is
    what the keypoint and mask paths that call it convert to NumPy.

    Args:
        boxes: ``(N, 4)`` boxes.
        scores: ``(N,)`` scores.
        labels: ``(N,)`` integer class ids.

    Returns:
        Contiguous CPU tensors: float32 boxes, float32 scores, and labels in their own dtype.

    Examples:
        >>> boxes, scores, labels = _copy_rows_to_cpu_once(torch.ones(2, 4), torch.ones(2), torch.tensor([3, 90]))
        >>> boxes.shape, scores.dtype, labels.tolist()
        (torch.Size([2, 4]), torch.float32, [3, 90])
    """
    packed = torch.cat([boxes.float(), scores.float().unsqueeze(1), labels.float().unsqueeze(1)], dim=1).cpu()
    return packed[:, :4].contiguous(), packed[:, 4].contiguous(), packed[:, 5].to(labels.dtype)


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


def _mps_lacks_antialiased_resize() -> bool:
    """Report whether the installed torch predates MPS support for antialiased bilinear resize.

    ``aten::_upsample_bilinear2d_aa`` has no MPS kernel before torch 2.7 (pytorch/pytorch#141287 tracks it, and
    #145581 is the PR it credits for 2.7); ``v2.7.0`` is the first tag whose ``native_functions.yaml`` dispatches it to
    MPS. Delete this predicate and the CPU-resize branch in :meth:`RFDETR.predict` once the ``torch`` floor in
    ``pyproject.toml`` reaches 2.7.

    Returns:
        ``True`` when ``torch.__version__`` is older than 2.7. An unparsable version string counts as modern, so
        the MPS path is not detoured through the CPU on an unrecognised build.

    Examples:
        >>> isinstance(_mps_lacks_antialiased_resize(), bool)
        True
    """
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return match is not None and (int(match.group(1)), int(match.group(2))) < (2, 7)


#: Iterable inputs that are one image or source, never a collection of them.
_SCALAR_INPUTS = (str, bytes, os.PathLike, Image.Image, np.ndarray, torch.Tensor)


def _as_collection(images: _PredictionInput) -> _PredictionInput:
    """Read an iterable of inputs into a list so later steps can count and index it.

    Lists, tuples and single inputs pass through unchanged, including strings, arrays and tensors, which are
    iterable but name one image or source.

    Args:
        images: One input or an iterable of inputs.

    Returns:
        The input itself, or a list of the iterable's items.

    Examples:
        >>> _as_collection(iter(["a.jpg", "b.jpg"]))
        ['a.jpg', 'b.jpg']
        >>> _as_collection("a.jpg")
        'a.jpg'
    """
    if isinstance(images, (list, tuple, *_SCALAR_INPUTS)) or not isinstance(images, Iterable):
        return images
    return list(images)


def predict(
    context_factory: Callable[[], PredictionContext],
    images: _PredictionInput,
    threshold: _ThresholdInput = 0.5,
    shape: tuple[int, int] | None = None,
    patch_size: int | None = None,
    include_source_image: bool = True,
    *,
    antialias: bool | None = None,
    stream: bool = False,
    batch: int | None = None,
    vid_stride: int = 1,
    stream_buffer: bool = False,
    prepare_ahead: bool = False,
) -> _PredictionOutput | Generator[_PredictionResult, None, None]:
    """Route both public prediction APIs through the same source and runtime pipeline.

    Empty sources follow one policy (details in :mod:`rfdetr._prediction.sources`). An empty collection holds no
    inputs: ``stream=True`` yields nothing, and an eager call returns ``[]`` for an empty manifest but raises
    ``ValueError`` for an empty list or tuple. A location that must hold media or sources raises in both modes: a
    directory or glob without supported files raises ``FileNotFoundError``, and a ``.streams`` manifest without sources
    raises ``ValueError``.

    Args:
        context_factory: Read current execution rules when an inference batch starts.
        images: Image data, paths, collections, videos, or live sources.
        threshold: The confidence score a detection must exceed to be kept. A single float applies to every class. A
            :class:`~rfdetr.ClassThresholds` applies each class's own threshold to its detections and leaves out every
            detection that belongs to no entry of ``class_names``.
        shape: Input height and width.
        patch_size: Backbone patch size for shape validation.
        include_source_image: Store source pixels in each result.
        antialias: Use antialiasing during image resize. Match the checkpoint training resize.
        stream: Yield results on demand instead of collecting them.
        batch: Maximum batch size. None runs a bare BCHW tensor or an eager image-only list as one batch, an eager
            directory or glob (or a list of them) in batches of eight, or of an export's smaller ``max_batch_size``,
            when the model has no fixed batch size, and other expanded or streamed sources one image or frame at a
            time. An integer bounds every batch.
        vid_stride: Read every Nth video frame.
        stream_buffer: Queue live frames instead of retaining the latest frame.
        prepare_ahead: Let a CUDA context build the next batch's input while the current batch runs. That reads the
            next batch's context before the current batch's results are yielded, so pass it only when nothing changes
            the model between batches, as for a predictor that owns its model or runtime.

    Returns:
        A lazy generator, a scalar image result, or a list of results.
    """
    for name, value in (("batch", 1 if batch is None else batch), ("vid_stride", vid_stride)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(stream_buffer, bool):
        raise ValueError("stream_buffer must be a boolean.")
    images = _as_collection(images)
    # An explicit batch splits an eager image list; an empty list keeps its single-batch error below.
    bounded_image_list = batch is not None and isinstance(images, (list, tuple)) and len(images) > 0
    if stream or bounded_image_list or is_expanded_source(images):
        # The generator below starts only at its first `next()`, after it has opened its source and decoded a batch, so
        # a threshold the model cannot apply is refused here, when `predict` is called. Every batch checks it again
        # against its own context, which a live model may have changed in between.
        context = context_factory()
        _class_thresholds(threshold, context)
        if batch is None:
            batch = _default_batch(images, context, stream=stream)
        # Imported here: the pipelines module builds on this one's batch phases.
        from rfdetr._prediction.pipelines import _predict_stream

        results = _predict_stream(
            context_factory,
            images,
            threshold,
            shape,
            patch_size,
            include_source_image,
            antialias=antialias,
            batch=batch,
            vid_stride=vid_stride,
            stream_buffer=stream_buffer,
            warn_on_live=not stream,
            prepare_ahead=prepare_ahead,
        )
        return results if stream else list(results)
    return _predict_batch(context_factory, images, threshold, shape, patch_size, include_source_image, antialias)


#: Images per batch when ``batch=None`` predicts an eager directory or glob on a model without a fixed batch size.
_DEFAULT_EAGER_BATCH = 8

#: Keys of a detection-only ``PostProcess`` result, the one :func:`_copy_results_to_cpu_once` takes.
_ROW_KEYS = frozenset({"boxes", "scores", "labels"})


def _default_batch(images: _PredictionInput, context: PredictionContext, *, stream: bool) -> int:
    """Choose the batch size for an expanded or streamed source when the caller passes ``batch=None``.

    A BCHW tensor already is one batch. An eager directory or glob runs in batches of :data:`_DEFAULT_EAGER_BATCH` on a
    model or export without a fixed batch size, or of the export's ``max_batch_size`` when that is smaller; its results
    are collected anyway, so batching costs no latency. Streamed sources keep one image per batch, so the first result
    arrives after one image, and a compiled snapshot or fixed-batch export keeps it too.

    Args:
        images: The prediction source.
        context: The model's current prediction rules.
        stream: Whether results are yielded on demand.

    Returns:
        The number of images per batch.

    Examples:
        >>> _default_batch(torch.zeros(3, 3, 8, 8), None, stream=True)
        3
    """
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        # Split a BCHW tensor only when the caller bounds the batch size.
        return max(images.shape[0], 1)
    # A fixed input resolution does not fix the batch axis, so only a fixed batch size keeps one image per batch.
    if not stream and context.batch_size is None and has_only_finite_images(images):
        if context.max_batch_size is None:
            return _DEFAULT_EAGER_BATCH
        return min(_DEFAULT_EAGER_BATCH, context.max_batch_size)
    return 1


@dataclass(frozen=True)
class _PrefetchedImage:
    """A local image path whose RGB decode was submitted to a worker thread ahead of its batch.

    Attributes:
        path: The image path, used as is when the decode returns ``None`` (Pillow fallback) or is not wanted.
        decoded: The pending :func:`_decode_local_image` result.
    """

    path: str
    decoded: Future[torch.Tensor | _EncodedJpeg | None]


def _decode_prefetched_jpegs(images: Sequence[Any], device: torch.device) -> dict[int, torch.Tensor | None]:
    """Decode on ``device`` the batch's JPEG files that the read-ahead left encoded for the GPU.

    On a CUDA device, waits for every prefetched entry of the batch, then decodes the encoded ones in one
    :func:`_decode_jpegs_on_gpu` call. Other devices return at once, so their batch keeps preparing each image as soon
    as its own decode is done.

    Args:
        images: The batch's entries.
        device: The context device; entries are only left encoded when it is a CUDA device.

    Returns:
        The decoded image (``None`` for Pillow) of each entry that was left encoded, keyed by its index in ``images``.

    Examples:
        >>> _decode_prefetched_jpegs(["a.jpg", np.zeros((2, 2, 3), np.uint8)], torch.device("cpu"))
        {}
    """
    if device.type != "cuda":
        return {}
    encoded = {
        index: result
        for index, item in enumerate(images)
        if isinstance(item, _PrefetchedImage) and isinstance(result := item.decoded.result(), _EncodedJpeg)
    }
    if not encoded:
        return {}
    return dict(zip(encoded, _decode_jpegs_on_gpu(list(encoded.values()), device)))


@torch.inference_mode()
def _predict_batch(
    context_factory: Callable[[], PredictionContext],
    images: _PredictionInput | list[_ImageInput | _PrefetchedImage],
    threshold: _ThresholdInput,
    shape: tuple[int, int] | None,
    patch_size: int | None,
    include_source_image: bool,
    antialias: bool | None,
) -> _PredictionOutput:
    """Prepare images, execute one batch, and decode results using a shared contract.

    Args:
        context_factory: Read execution and decoding rules for this batch.
        images: An image, file path, URL string, or list or tuple of those inputs. PIL and file inputs convert to
            grayscale for one-channel models and RGB for three-channel models; arrays and tensors supply their channels.
        threshold: The confidence score a detection must exceed to be kept. A single float applies to every class. A
            :class:`~rfdetr.ClassThresholds` applies each class's own threshold to its detections, so that a class the
            model finds less reliably can be held to a different bar. It leaves out every detection that belongs to no
            entry of ``class_names``: the no-object slot of a fine-tuned model, the background slot of a legacy keypoint
            checkpoint, and the unused category IDs of a pretrained COCO model.
        shape: Input height and width.
        patch_size: Backbone patch size for shape validation.
        include_source_image: Store the source image in result metadata.
        antialias: Use antialiasing during image resize. Match the checkpoint training resize.

    Returns:
        One Supervision result or a list matching the input container.

    Raises:
        TypeError: If ``threshold`` is neither a number nor a ``ClassThresholds``, such as a list or a ``dict``.
        ValueError: If a list or tuple is empty, the batch violates an export's fixed or maximum batch size, the shape
            or patch size is invalid or conflicts with an export's fixed shape, an input fails range or channel
            validation, ``threshold`` is NaN, or a ``ClassThresholds`` does not fit the model: it names a class that
            ``class_names`` does not hold, has a different number of entries than ``class_names``, or is given to a
            model without classes. A ``ClassThresholds`` also raises when the export's ``class_id_to_name`` does not
            match ``class_names`` (more ids than names, names in another order, or an id outside
            ``0..num_classes``).
    """
    prepared = _prepare_batch(context_factory, images, threshold, shape, patch_size, include_source_image, antialias)
    return _decode_batch(prepared, prepared.context.run(prepared.batch_tensor))


@dataclass(frozen=True)
class _PreparedBatch:
    """One batch's model input on the context device, with what decoding its results needs.

    Attributes:
        context: The prediction rules read for this batch.
        batch_tensor: The normalized ``(B, C, H, W)`` model input.
        target_sizes: ``(B, 2)`` source heights and widths, on the context device.
        orig_sizes: The same sizes as Python tuples, for the results' ``source_shape``.
        source_images: Source pixels per image; empty unless ``include_source_image``.
        include_source_image: Store the source image in result metadata.
        single_input: The caller passed one image, so decoding returns one result instead of a list.
        score_floor: The single threshold, or the lowest per-class one.
        threshold_by_class_id: The per-class threshold table, or ``None`` for a single threshold.
    """

    context: PredictionContext
    batch_tensor: torch.Tensor
    target_sizes: torch.Tensor
    orig_sizes: list[Any]
    source_images: list[Any]
    include_source_image: bool
    single_input: bool
    score_floor: float
    threshold_by_class_id: torch.Tensor | None


@torch.inference_mode()
def _prepare_batch(
    context_factory: Callable[[], PredictionContext],
    images: _PredictionInput | list[_ImageInput | _PrefetchedImage],
    threshold: _ThresholdInput,
    shape: tuple[int, int] | None,
    patch_size: int | None,
    include_source_image: bool,
    antialias: bool | None,
) -> _PreparedBatch:
    """Read the batch's context, check its arguments and images, and build the model input on the context device.

    This is the part of :func:`_predict_batch` before the forward pass; its arguments, and what it raises, are the
    same.

    Args:
        context_factory: Read execution and decoding rules for this batch.
        images: An image, file path, URL string, or list or tuple of those inputs.
        threshold: A single confidence threshold, or a ``ClassThresholds``; see :func:`_predict_batch`.
        shape: Input height and width.
        patch_size: Backbone patch size for shape validation.
        include_source_image: Store the source image in result metadata.
        antialias: Use antialiasing during image resize.

    Returns:
        The model input and the data :func:`_decode_batch` needs.
    """
    if not isinstance(images, list):
        # A list, including a batch with decode-ahead entries, already is a collection.
        images = _as_collection(images)
    if isinstance(images, (list, tuple)) and not images:
        raise ValueError("At least one image is required.")

    context = context_factory()
    antialias = context.antialias if antialias is None else antialias
    if context.fixed_shape:
        shape = context.default_shape if shape is None else shape
        batch_size = len(images) if isinstance(images, (list, tuple)) else 1
        if context.batch_size is not None and batch_size != context.batch_size:
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

    # Built before the forward pass, where a class-id map that does not match `class_names` can still be refused.
    score_floor, threshold_by_class_id = _class_thresholds(threshold, context)

    if context.prepare is not None:
        context.prepare()

    # Determine the return shape from the *input* type, not the runtime batch
    # length: a single image (path / PIL / tensor) yields a bare Detections,
    # while a list/tuple always yields a list — even when it holds one image.
    single_input = not isinstance(images, (list, tuple))
    if not isinstance(images, (list, tuple)):
        # ``_as_collection`` already read every other iterable into a list, so this is one image.
        images = [cast("_ImageInput", images)]
    # The batch's JPEG files left encoded for the GPU decode in one nvjpeg call rather than one call per image.
    gpu_decoded = _decode_prefetched_jpegs(images, context.device)

    orig_sizes: list[Any] = []
    processed_images: list[Any] = []
    source_images: list[Any] = []
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

    # torch < 2.7 has no MPS kernel for antialiased bilinear resize (pytorch#141287, #145581). Keep the whole
    # preprocessing path on CPU in that case, so an input is never moved to MPS only to come straight back for
    # resizing. MPS-resident tensor inputs take the same detour (device -> CPU, then the resized batch back to
    # MPS). From torch 2.7 on the MPS kernel exists and this stays False; once the torch floor reaches 2.7,
    # delete this branch together with `_mps_lacks_antialiased_resize`.
    resize_on_cpu = antialias and context.device.type == "mps" and _mps_lacks_antialiased_resize()
    preprocess_device = torch.device("cpu") if resize_on_cpu else context.device

    for index, img_input in enumerate(images):
        img: Any = img_input
        decoded_local_file = False
        prefetched: Future[torch.Tensor | _EncodedJpeg | None] | None = None
        if isinstance(img, _PrefetchedImage):
            img, prefetched = img.path, img.decoded
        if isinstance(img, os.PathLike):
            img = os.fspath(img)
        if isinstance(img, str):
            if urlparse(img).scheme in ("http", "https"):
                resp = requests.get(img, timeout=30)
                resp.raise_for_status()
                img = io.BytesIO(resp.content)
            elif context.config.num_channels == 3:
                # The fast decoder is RGB-only. Keep grayscale on Pillow so files and PIL inputs use the same
                # luminance conversion; a grayscale codec path needs pixel-parity and latency evidence first.
                if index in gpu_decoded:
                    decoded = gpu_decoded[index]
                else:
                    # Without a GPU device the decode never leaves a file encoded.
                    decoded = cast(
                        "torch.Tensor | None",
                        prefetched.result() if prefetched is not None else _decode_local_image(img),
                    )
                if decoded is not None:
                    img = decoded
                    decoded_local_file = True
            if not decoded_local_file:
                img = Image.open(img)

        range_known_valid = decoded_local_file
        deferred_widen = decoded_local_file
        if decoded_local_file:
            if include_source_image:
                source_images.append(img.permute(1, 2, 0).numpy().copy())
        elif not isinstance(img, torch.Tensor):
            # PIL owns the one-/three-channel conversion. Other channel counts require an explicit array or tensor.
            if isinstance(img, Image.Image):
                mode = "L" if context.config.num_channels == 1 else "RGB"
                if img.mode != mode:
                    img = img.convert(mode)
            pil_image = isinstance(img, Image.Image)
            source_array: np.ndarray[Any, Any] | None = None
            if include_source_image:
                source_array = np.array(img)
                if source_array.dtype != np.uint8:
                    source_array = (source_array * 255).clip(0, 255).astype(np.uint8)
                source_images.append(source_array)
            uint8_array = isinstance(img, np.ndarray) and img.dtype == np.uint8
            if isinstance(img, np.ndarray) and any(stride < 0 for stride in img.strides):
                # ``torch.from_numpy``, used by both conversions below, rejects negative strides, e.g. the
                # ``frame[:, :, ::-1]`` BGR-to-RGB view. Copy only these, so contiguous, positive-step and
                # broadcast views stay zero-copy.
                if uint8_array and source_array is not None:
                    # ``np.array(img)`` above already made a positive-stride copy, so reuse it. The model input
                    # then shares storage with ``source_image``, as the PIL path below does.
                    img = source_array
                else:
                    # ``.copy()``, not ``np.ascontiguousarray``: NumPy ignores the stride of a length-1 axis when
                    # it flags an array C-contiguous, so ``ascontiguousarray`` would hand back the same view.
                    img = img.copy()
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
            source_images.append(_tensor_to_source_array(img))

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
        if img_tensor.device.type == "cpu" and preprocess_device.type == "cuda":
            img_tensor = img_tensor.pin_memory()
        # non_blocking only pays off (and is only safe without an explicit sync) when the destination is CUDA,
        # matching the transfer_batch_to_device() convention in training/module_data.py: a CUDA-tensor-input ->
        # CPU-model transfer with non_blocking=True races the copy — the CPU destination is never pinned, so
        # reads of the tensor's data can observe an in-flight (partially written) copy.
        non_blocking = preprocess_device.type == "cuda"
        img_tensor = img_tensor.to(preprocess_device, non_blocking=non_blocking)
        if deferred_widen:
            img_tensor = _uint8_chw_to_float(img_tensor, _uint8_scale(img_tensor.device, torch.get_default_dtype()))
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
    batch_tensor = torch.stack([F.resize(t, resize_to, antialias=antialias) for t in processed_images])
    if resize_on_cpu:
        batch_tensor = batch_tensor.to(context.device)
    # The same subtract-then-divide as torchvision's `normalize`, on a tensor `stack` just allocated, so in place.
    mean, std = _normalization_tensors(
        tuple(context.means), tuple(context.stds), batch_tensor.dtype, batch_tensor.device
    )
    batch_tensor = batch_tensor.sub_(mean).div_(std)

    # Copied to the device before the forward pass is queued: on MPS, building it after the forward measured slower.
    if context.device.type == "cuda":
        # A pageable copy makes the host wait for the transfer; from pinned memory it is queued like the images.
        target_sizes = torch.tensor(orig_sizes, device="cpu").pin_memory().to(context.device, non_blocking=True)
    else:
        target_sizes = torch.tensor(orig_sizes, device=context.device)
    return _PreparedBatch(
        context=context,
        batch_tensor=batch_tensor,
        target_sizes=target_sizes,
        orig_sizes=orig_sizes,
        source_images=source_images,
        include_source_image=include_source_image,
        single_input=single_input,
        score_floor=score_floor,
        threshold_by_class_id=threshold_by_class_id,
    )


@torch.inference_mode()
def _decode_batch(prepared: _PreparedBatch, predictions: dict[str, torch.Tensor]) -> _PredictionOutput:
    """Post-process one batch's model outputs, apply the confidence threshold, and build Supervision results.

    This is the part of :func:`_predict_batch` after the forward pass.

    Args:
        prepared: The batch :func:`_prepare_batch` built.
        predictions: The model's outputs for ``prepared.batch_tensor``.

    Returns:
        One Supervision result or a list matching the input container.
    """
    from supervision import Detections, KeyPoints

    context = prepared.context
    score_floor, threshold_by_class_id = prepared.score_floor, prepared.threshold_by_class_id
    orig_sizes, source_images = prepared.orig_sizes, prepared.source_images
    include_source_image = prepared.include_source_image
    # `score_floor` is the single threshold, or the lowest per-class one, the loosest bound: the pre-filter in
    # ``postprocess`` then never drops a row that the per-class filter below keeps.
    results = context.postprocess(predictions, target_sizes=prepared.target_sizes, score_threshold=score_floor)
    # On an accelerator the filter below would wait for the device twice per image (MPS copies synchronously, and on
    # CUDA `nonzero` and the copy each sync); detection rows are small enough to bring the whole batch to the CPU in
    # one copy first. Masks and keypoints stay on the device until the filter has dropped rows.
    if results[0]["scores"].device.type != "cpu" and all(r.keys() == _ROW_KEYS for r in results):
        results = _copy_results_to_cpu_once(results)

    _class_id_to_name = context.class_id_to_name
    num_logit_slots = context.num_classes
    _is_legacy_bgfirst_keypoint = _is_bg_first_schema(context.num_keypoints_per_class)
    # Rounded to the score dtype on the CPU, once: the float form rounds `t` to the score dtype, so `[t] * n` then
    # compares exactly as `t` does, where a float32 table would compare a float16 score at float32 precision (or a
    # float64 score at float32 precision). On CUDA the copy is non-blocking, so it adds no stream synchronization;
    # other accelerators copy synchronously, as the images do.
    label_thresholds = (
        None
        if threshold_by_class_id is None
        else threshold_by_class_id.to(results[0]["scores"].dtype).to(
            results[0]["scores"].device, non_blocking=results[0]["scores"].device.type == "cuda"
        )
    )
    predictions_list: list[_PredictionResult] = []
    for i, result in enumerate(results):
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]

        # INVARIANT: the pre-filter in PostProcess._postprocess_masks (`scores_i > score_threshold`) must keep
        # every row this predicate may keep, and use the same operator: it is fed `score_threshold=score_floor`
        # above, which is the single threshold, or the lowest per-class threshold. With per-class thresholds this
        # predicate can drop rows that the lowest-threshold pre-filter admits, never the reverse, so the pre-filter may
        # be looser but never stricter. The seg path drops below-threshold masks before upsampling on the strength of
        # that match; a stricter pre-filter, or a different operator here (e.g. `>=`, top-k), would make it
        # silently drop rows this filter keeps — a behaviour change with no failing test.
        # Materialized as an index vector (not a bool mask) so the bool-mask advanced-indexing
        # sync below runs once per image instead of once per kept tensor: `t[bool_mask]` re-derives
        # its output size via `nonzero(bool_mask)` on every call (a documented CUDA host-device
        # sync), while `t[int64_index]` is index_select-shaped and needs no further sync. Row order
        # is unaffected — `nonzero` returns ascending indices, identical to bool-mask selection order.
        if label_thresholds is None:
            keep_idx = (scores > score_floor).nonzero(as_tuple=True)[0]
        else:
            # -1 and every id past the table read the trailing inf sentinel. Clamping only the top would let an
            # id below -1 wrap around in Python indexing and read another class's threshold.
            row_thresholds = label_thresholds[labels.clamp(min=-1, max=label_thresholds.numel() - 1)]
            keep_idx = (scores > row_thresholds).nonzero(as_tuple=True)[0]
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
        if boxes.device.type in ("cuda", "cpu"):
            boxes_cpu = boxes.float().to("cpu", non_blocking=is_cuda)
            scores_cpu = scores.float().to("cpu", non_blocking=is_cuda)
            labels_cpu = labels.to("cpu", non_blocking=is_cuda)
        else:
            # A synchronous accelerator blocks on every copy; pay that wait once for the three row tensors.
            boxes_cpu, scores_cpu, labels_cpu = _copy_rows_to_cpu_once(boxes, scores, labels)
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
            detections.metadata["source_image"] = source_images[i]
        detections.data["source_shape"] = np.tile(np.array(orig_sizes[i], dtype=np.int64), (len(detections), 1))

        # Attach class names so callers can map class_id → name without a
        # separate lookup. Always set data["class_name"] for a consistent interface.
        #
        # Explicit _class_id_to_name entries take precedence, including at the
        # background sentinel. Only an unmapped sentinel maps to "__background__"
        # without warning. Other unmapped IDs produce an empty string with a
        # one-time warning.
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
        class_names = [_class_id_to_name.get(cid, "__background__" if cid == _bg_sentinel else "") for cid in class_ids]
        detections.data["class_name"] = np.array(class_names, dtype=object)

        if has_keypoints and keypoints_array is not None:
            keypoint_data = dict(detections.data)
            keypoint_data["xyxy"] = detections.xyxy.astype(np.float32)
            if include_source_image:
                keypoint_data["source_image"] = [source_images[i] for _ in range(len(detections))]
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

    return predictions_list[0] if prepared.single_input else predictions_list
