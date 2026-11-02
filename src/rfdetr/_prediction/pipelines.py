# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Schedule prediction batches: read and decode finite sources ahead, and overlap CUDA batches.

:func:`~rfdetr._prediction.core.predict` hands every streamed or bounded-batch call to :func:`_predict_stream`, which
runs each batch through the phases in :mod:`rfdetr._prediction.core`.
"""

from __future__ import annotations

import functools
import itertools
import os
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import torch

from rfdetr._prediction.core import (
    PredictionContext,
    _decode_batch,
    _decode_local_image,
    _predict_batch,
    _PrefetchedImage,
    _prepare_batch,
    _PreparedBatch,
)
from rfdetr._prediction.sources import has_only_finite_images, iter_source_batches
from rfdetr._prediction.types import _ImageInput, _PredictionInput, _ThresholdInput
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr._prediction.types import _PredictionResult

logger = get_logger()


def _predict_stream(
    context_factory: Callable[[], PredictionContext],
    images: _PredictionInput,
    threshold: _ThresholdInput,
    shape: tuple[int, int] | None,
    patch_size: int | None,
    include_source_image: bool,
    *,
    antialias: bool | None,
    batch: int,
    vid_stride: int,
    stream_buffer: bool,
    warn_on_live: bool,
    prepare_ahead: bool,
) -> Generator[_PredictionResult, None, None]:
    """Expand sources into bounded batches and predict each lazily, releasing capture on every exit path.

    The model's channel count is read once from the first context so captured frames convert to grayscale for
    one-channel models. Each batch runs the phases of :func:`_predict_batch`, :func:`_prepare_batch` and then
    :func:`_decode_batch`, and re-reads the context when it is prepared. Closing the generator closes the source reader.

    A source with no live input (see :func:`~rfdetr._prediction.sources.has_only_finite_images`) is read ahead for
    three-channel models, except a native model on the CPU: the image files of the batches after the current one
    decode on worker threads while it runs. Those batches hold at least one batch and :data:`_DECODE_WORKERS` images,
    or the rest of the source (see :func:`_decode_ahead`). Live sources, and native models on the CPU, keep reading one
    batch at a time.

    With ``prepare_ahead``, such a source on a CUDA context also builds each batch's model input while the batch before
    it runs (see :func:`_predict_prepared_ahead`). That reads the next batch's context before the current batch's
    results are yielded, so it is set only for a model or runtime that nothing changes between batches; a borrowed
    live model runs each batch through :func:`_predict_batch` once the batch before it is yielded.

    Args:
        context_factory: Read current execution rules when a batch starts.
        images: Image data, paths, collections, videos, or live sources.
        threshold: A single confidence threshold, or a ``ClassThresholds``; see :func:`predict`.
        shape: Input height and width.
        patch_size: Backbone patch size for shape validation.
        include_source_image: Store source pixels in each result.
        antialias: Use antialiasing during image resize.
        batch: Maximum number of images per batch.
        vid_stride: Read every Nth video or live frame.
        stream_buffer: Queue live frames instead of retaining the latest frame.
        warn_on_live: Warn about memory growth from live sources and large finite sources; set for eager callers, whose
            results accumulate.
        prepare_ahead: Let a CUDA context build the next batch's input while the current batch runs; set only when the
            predictor owns its model or runtime.

    Yields:
        One result per input image or frame, in source order.
    """
    context = context_factory()
    num_channels = context.config.num_channels
    source_batches = iter_source_batches(
        images,
        batch=batch,
        vid_stride=vid_stride,
        stream_buffer=stream_buffer,
        warn_on_live=warn_on_live,
        # Frame sources decode RGB; the model's channel count decides whether they convert to grayscale.
        num_channels=num_channels,
        # Eager results each keep a full source image, so large finite sources warn before decoding starts.
        warn_on_retained_frames=warn_on_live and include_source_image,
    )
    # Only the RGB decoder runs ahead; one-channel models open files with Pillow inside `_predict_batch`. A native model
    # on the CPU keeps decoding inline: its forward pass already uses every core, and decode threads competing with it
    # slowed the forward pass by more than the overlap saved. Exported runtimes decode ahead whatever device their
    # context reports (TensorRT reports CUDA, most others the CPU): Core ML gained 18% from it and ONNX Runtime on the
    # CPU lost nothing.
    native_on_cpu = context.device.type == "cpu" and context.runtime_info.get("backend") == "pytorch"
    on_cuda = context.device.type == "cuda"
    may_run_ahead = (prepare_ahead and on_cuda) or (num_channels == 3 and not native_on_cpu)
    # Classified once, and only when a batch could run ahead: it checks every path, directory and glob on disk.
    finite = may_run_ahead and has_only_finite_images(images)
    decode_ahead = num_channels == 3 and not native_on_cpu and finite
    pool = ThreadPoolExecutor(_DECODE_WORKERS, thread_name_prefix="rfdetr-decode") if decode_ahead else None
    # On CUDA, suitable JPEG files decode on the GPU with their batch; a kept source image needs host pixels anyway.
    gpu_device = context.device if on_cuda and not include_source_image else None
    batches = source_batches if pool is None else _decode_ahead(source_batches, pool, gpu_device)
    try:
        if prepare_ahead and on_cuda and finite:
            prepare = functools.partial(
                _prepare_batch,
                context_factory,
                threshold=threshold,
                shape=shape,
                patch_size=patch_size,
                include_source_image=include_source_image,
                antialias=antialias,
            )
            in_worker = context.runtime_info.get("backend") != "pytorch"
            yield from _predict_prepared_ahead(batches, prepare, context.device, in_worker=in_worker)
            return
        for image_batch in batches:
            results = _predict_batch(
                context_factory, image_batch, threshold, shape, patch_size, include_source_image, antialias
            )
            yield from cast("list[_PredictionResult]", results)
    finally:
        if pool is not None:
            # Decodes already queued for a batch nobody will read are dropped; running ones finish first.
            pool.shutdown(wait=True, cancel_futures=True)
        source_batches.close()


#: Worker threads decoding the next batch's image files. File reads and torchvision's decoder release the GIL.
_DECODE_WORKERS = 4


def _decode_ahead(
    source_batches: Iterator[list[_ImageInput]], pool: ThreadPoolExecutor, gpu_device: torch.device | None = None
) -> Generator[list[_ImageInput | _PrefetchedImage], None, None]:
    """Yield each batch after reading ahead from the source and submitting the file decodes of what it read.

    Before a batch is yielded, the batches read after it hold at least one batch and :data:`_DECODE_WORKERS` images,
    or the rest of the source, so small batches keep every decode worker busy: one image at a time left the workers
    idle while the model ran. Local file paths become :class:`_PrefetchedImage` entries; every other item passes
    through unchanged. A source error raised while reading ahead is re-raised after every batch read before it has been
    yielded, where the serial loop raised it.

    Args:
        source_batches: Batches from :func:`~rfdetr._prediction.sources.iter_source_batches`.
        pool: Executor running :func:`_decode_local_image`.
        gpu_device: The CUDA device the batches decode suitable JPEG files on; their workers then only read and
            check those files (see :func:`_decode_local_image`).

    Yields:
        The source's batches, in order, with their local image files decoding in ``pool``.

    Examples:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with ThreadPoolExecutor(1) as pool:
        ...     list(_decode_ahead(iter([["https://example.com/a.jpg"]]), pool))
        [['https://example.com/a.jpg']]
    """

    def submit(image_batch: list[_ImageInput]) -> list[_ImageInput | _PrefetchedImage]:
        scheduled: list[_ImageInput | _PrefetchedImage] = []
        for item in image_batch:
            if isinstance(item, (str, os.PathLike)) and urlparse(os.fspath(item)).scheme not in ("http", "https"):
                path = os.fspath(item)
                scheduled.append(_PrefetchedImage(path, pool.submit(_decode_local_image, path, gpu_device)))
            else:
                scheduled.append(item)
        return scheduled

    scheduled: deque[list[_ImageInput | _PrefetchedImage]] = deque()
    deferred_error: Exception | None = None
    exhausted = False
    while True:
        while not exhausted and deferred_error is None and _needs_read_ahead(scheduled):
            try:
                upcoming = next(source_batches, None)
            except Exception as error:  # re-raised below, after the batches the serial loop would have finished first
                deferred_error = error
                break
            if upcoming is None:
                exhausted = True
            else:
                scheduled.append(submit(upcoming))
        if not scheduled:
            break
        yield scheduled.popleft()
    if deferred_error is not None:
        raise deferred_error


def _needs_read_ahead(scheduled: deque[list[_ImageInput | _PrefetchedImage]]) -> bool:
    """Tell whether the batches read after the next one hold less than a batch or :data:`_DECODE_WORKERS` images.

    Args:
        scheduled: Batches read from the source and not yet yielded, the next one first.

    Returns:
        Whether :func:`_decode_ahead` should read another batch before yielding the next one.

    Examples:
        >>> _needs_read_ahead(deque([["a.jpg"], ["b.jpg"]]))
        True
        >>> _needs_read_ahead(deque([["a.jpg"], ["b.jpg", "c.jpg", "d.jpg", "e.jpg"]]))
        False
    """
    return len(scheduled) < 2 or sum(len(batch) for batch in itertools.islice(scheduled, 1, None)) < _DECODE_WORKERS


def _predict_prepared_ahead(
    batches: Iterator[list[Any]],
    prepare: Callable[[list[Any]], _PreparedBatch],
    device: torch.device,
    *,
    in_worker: bool,
) -> Generator[_PredictionResult, None, None]:
    """Predict CUDA batches, building each batch's model input while the batch before it runs.

    A native model queues its forward pass without waiting for it, so this thread prepares the next batch right after
    queuing the forward pass, before post-processing waits for it. An exported runtime waits for its own execution, so
    a worker thread prepares the next batch on a CUDA stream of its own while it runs; the current stream waits for
    that stream's event before the input is used. A batch holding a CUDA tensor is prepared on this thread and the
    current stream instead: the worker's stream does not wait for the caller's, so it could read the tensor before the
    kernels writing it finish. The overlap helps only inputs decoded on the host, such as image files and arrays: for a
    CUDA tensor, checking its value range and copying it for ``include_source_image`` wait for the batch before it, on
    either path. Each batch runs the same operations as :func:`_predict_batch`, so the results are the same. At most
    one prepared batch waits beside the one running. An error raised while reading or preparing the next batch is
    raised after the current batch's results are yielded, where the serial loop raised it.

    Closing the generator, or an error raised to the caller, waits for a batch the worker is already preparing to
    finish, which can take as long as reading its images (an image URL allows 30 s per read): returning earlier would
    leave a thread queuing work on the side stream after prediction ended. An error raised while preparing a batch
    nobody reads any more is logged at debug level.

    Args:
        batches: The source's batches.
        prepare: :func:`_prepare_batch` with every argument but the images bound.
        device: The CUDA device of the prediction context.
        in_worker: Prepare on a worker thread and its own stream, for runtimes whose execution waits for the device.

    Yields:
        One result per input image, in source order.
    """
    worker = ThreadPoolExecutor(1, thread_name_prefix="rfdetr-prepare") if in_worker else None
    side_stream = torch.cuda.Stream(device) if in_worker else None  # type: ignore[no-untyped-call]

    def prepare_on_side_stream(images: list[Any]) -> tuple[_PreparedBatch, torch.cuda.Event | None]:
        with torch.cuda.stream(side_stream):
            prepared = prepare(images)
        ready = torch.cuda.Event()  # type: ignore[no-untyped-call]
        ready.record(side_stream)
        return prepared, ready

    def start() -> Future[tuple[_PreparedBatch, torch.cuda.Event | None]] | None:
        future: Future[tuple[_PreparedBatch, torch.cuda.Event | None]] = Future()
        try:
            images = next(batches, None)
            if images is None:
                return None
            # Work queued on the caller's stream, such as the kernels producing a CUDA tensor input, is ordered only
            # before work on that same stream.
            if worker is not None and not any(isinstance(image, torch.Tensor) and image.is_cuda for image in images):
                return worker.submit(prepare_on_side_stream, images)
            future.set_result((prepare(images), None))
        except Exception as error:  # raised by `result()` below, after the current batch's results
            future.set_exception(error)
        return future

    pending: Future[tuple[_PreparedBatch, torch.cuda.Event | None]] | None = None
    try:
        pending = start()
        while pending is not None:
            # Cleared first, so `finally` sees only a batch whose result nobody read.
            current, pending = pending, None
            prepared, ready = current.result()
            if ready is not None:
                stream = torch.cuda.current_stream(device)
                stream.wait_event(ready)
                # Allocated on the side stream: without this the allocator could hand their memory back to it while
                # this stream still reads them.
                prepared.batch_tensor.record_stream(stream)
                prepared.target_sizes.record_stream(stream)
                pending = start()
            with torch.inference_mode():
                predictions = prepared.context.run(prepared.batch_tensor)
            if ready is None:
                pending = start()
            yield from cast("list[_PredictionResult]", _decode_batch(prepared, predictions))
    finally:
        if worker is not None:
            # Waits for a prepare already running, so no thread queues side-stream work after prediction ends.
            worker.shutdown(wait=True, cancel_futures=True)
        abandoned = pending.exception() if pending is not None and pending.done() and not pending.cancelled() else None
        if abandoned is not None:
            logger.debug("Dropped the error from preparing a batch that prediction stopped before.", exc_info=abandoned)
