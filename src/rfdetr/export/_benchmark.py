# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Latency and memory benchmarking helpers shared by the per-hardware export cookbooks.

Two timer strategies exist because GPU kernels execute asynchronously: CUDA events measure actual device-side execution,
while ``time.perf_counter`` is correct wall-clock timing for anything that blocks the calling thread — CPU inference,
and every non-CUDA runtime (CoreML, Core AI, ExecuTorch, TensorFlow Lite, LiteRT, OpenVINO) has no CUDA stream to
desynchronize from in the first place. The same split applies to :func:`measure_memory`: device-side weights and buffers
on a CUDA GPU don't show up in host resident memory, so it reads ``torch.cuda.mem_get_info()`` there instead of process
RSS.

Private module: no compatibility guarantee across versions. Formerly duplicated per export format (see the removed
``rfdetr.export._onnx.inference._onnx_runtime``); this is the single home for it.
"""

from __future__ import annotations

import gc
import platform
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import supervision as sv

from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.export._runtime.decode import decode_detections

if TYPE_CHECKING:
    from PIL import Image


class BenchmarkResult(NamedTuple):
    """One latency measurement: mean and standard deviation across timed runs, in milliseconds."""

    label: str
    mean_ms: float
    std_ms: float

    @property
    def fps(self) -> float:
        """Frames per second implied by ``mean_ms``.

        Examples:
            >>> BenchmarkResult("cpu", 10.0, 0.5).fps
            100.0
        """
        return float("inf") if self.mean_ms == 0.0 else 1000.0 / self.mean_ms


def _mean_std(timings: list[float]) -> tuple[float, float]:
    """Mean and population standard deviation of a list of millisecond timings.

    Examples:
        >>> [round(v, 2) for v in _mean_std([1.0, 2.0, 3.0])]
        [2.0, 0.82]
    """
    arr = np.array(timings)
    return float(arr.mean()), float(arr.std())


def _measure_cuda(fn: Callable[[], object], warmup: int, runs: int) -> tuple[float, float]:
    """Time ``fn`` with CUDA events — captures device-side kernel execution, not Python overhead."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    timings: list[float] = []
    for _ in range(runs):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return _mean_std(timings)


def _measure_wall_clock(fn: Callable[[], object], warmup: int, runs: int) -> tuple[float, float]:
    """Time ``fn`` with ``time.perf_counter`` — correct for any call with no CUDA stream to sync."""
    for _ in range(warmup):
        fn()
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return _mean_std(timings)


def measure_latency(
    fn: Callable[[], object],
    *,
    label: str,
    device: str = "cpu",
    warmup: int = 20,
    runs: int = 100,
) -> BenchmarkResult:
    """Measure the latency of a zero-argument callable.

    ``device="cuda"`` times with CUDA events; any other value times with ``time.perf_counter``. Pass
    a thunk wrapping only the runtime's forward call to measure ``forward_ms``, or a thunk wrapping
    preprocess + forward + postprocess to measure ``end2end_ms`` — the caller chooses the scope by
    what it wraps, this function only times whatever it is given.

    Args:
        fn: Zero-argument callable to time.
        label: Name for the resulting :class:`BenchmarkResult` row, e.g. ``"TensorRT forward"``.
        device: ``"cuda"`` selects the CUDA-event timer; any other value uses ``perf_counter``.
        warmup: Untimed warm-up calls before measurement starts, to skip first-call JIT/lazy-init cost.
        runs: Timed calls used to compute the mean and standard deviation.

    Returns:
        A :class:`BenchmarkResult` with ``mean_ms``, ``std_ms``, and the derived ``fps``.

    Examples:
        >>> result = measure_latency(lambda: sum(range(1000)), label="sum", warmup=1, runs=3)
        >>> result.label
        'sum'
        >>> result.mean_ms >= 0.0
        True
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be positive")

    measure = _measure_cuda if device == "cuda" else _measure_wall_clock
    mean_ms, std_ms = measure(fn, warmup, runs)
    return BenchmarkResult(label, mean_ms, std_ms)


#: Seconds between memory samples taken by the background watcher thread.
_SAMPLE_INTERVAL_S = 0.005


@dataclass
class MemoryResult:
    """Mutable holder for the memory measurements taken by :func:`measure_memory`.

    All three fields keep their defaults until the ``with`` block exits. ``delta_mb`` and ``peak_mb``
    answer different questions and routinely differ by a lot: bringing up a CoreML model was measured
    at a 548.7 MB peak but only a 407.2 MB net change, the 141 MB gap being compile scratch space
    released before the block closed.

    Attributes:
        delta_mb: Net change between the start and the end of the block — what the runtime still
            holds once it is up. May be *negative*, which means the block ended with less memory in
            use than it started with, usually because the OS reclaimed an earlier allocation.
        peak_mb: Largest growth above the starting level seen at any sample during the block — what
            it costs to bring the runtime up, including transient scratch space. Never negative.
        samples: Number of samples the watcher thread took. ``0`` means it never got scheduled, so
            ``peak_mb`` is only as good as the two endpoint reads and should not be trusted.
    """

    delta_mb: float = 0.0
    peak_mb: float = 0.0
    samples: int = 0


def _sampled_delta_mb(read_bytes_in_use: Callable[[], int]) -> Iterator[MemoryResult]:
    """Track net and peak growth of ``read_bytes_in_use()`` across a block, sampling in a thread.

    Sampling rather than reading only the endpoints is what makes ``peak_mb`` meaningful: memory
    allocated and released inside the block is invisible to an endpoint-only diff.

    Args:
        read_bytes_in_use: Returns the current bytes-in-use figure for whichever memory is measured.

    Yields:
        The :class:`MemoryResult` filled in when the block exits.
    """
    baseline = read_bytes_in_use()
    peak = baseline
    samples = 0
    stop = threading.Event()

    def watch() -> None:
        nonlocal peak, samples
        while not stop.is_set():
            peak = max(peak, read_bytes_in_use())
            samples += 1
            stop.wait(_SAMPLE_INTERVAL_S)

    watcher = threading.Thread(target=watch, name="measure_memory", daemon=True)
    watcher.start()
    result = MemoryResult()
    try:
        yield result
    finally:
        stop.set()
        watcher.join(timeout=1.0)
        final = read_bytes_in_use()
        result.delta_mb = (final - baseline) / 1e6
        result.peak_mb = (max(peak, final) - baseline) / 1e6
        result.samples = samples


def _rss_delta_mb() -> Iterator[MemoryResult]:
    """Measure host resident memory across a block via ``psutil``."""
    import psutil  # type: ignore[import-untyped]

    gc.collect()
    process = psutil.Process()
    yield from _sampled_delta_mb(lambda: int(process.memory_info().rss))


def _cuda_free_delta_mb() -> Iterator[MemoryResult]:
    """Measure device memory in use across a block via ``torch.cuda.mem_get_info``.

    Device-wide, unlike ``torch.cuda.memory_allocated()`` — it also captures allocations made outside PyTorch's own
    caching allocator, such as ONNX Runtime's CUDA execution provider or a TensorRT engine's own ``cudaMalloc`` calls.
    """
    import torch

    def device_bytes_in_use() -> int:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return int(total - free)

    yield from _sampled_delta_mb(device_bytes_in_use)


@contextmanager
def measure_memory(*, device: str = "cpu") -> Iterator[MemoryResult]:
    """Measure the memory growth caused by the code inside a ``with`` block.

    ``device="cuda"`` reads free-device-memory shrinkage via ``torch.cuda.mem_get_info()``; any
    other value reads host resident-memory growth via ``psutil``. Bracket both the runtime's
    construction *and* its first inference call — several runtimes allocate lazily (an ONNX
    Runtime session grows its arena on first ``run``, an ExecuTorch CoreML program compiles on
    first ``execute``), so closing the block right after construction undercounts the real
    footprint.

    A background thread samples memory every 5 ms for the duration of the block, so
    :attr:`MemoryResult.peak_mb` sees transient scratch space that an endpoint-only reading misses.
    Check :attr:`MemoryResult.samples` before trusting ``peak_mb``: a block that finishes in under a
    few milliseconds, or one that never releases the GIL, can collect no samples at all.

    Args:
        device: ``"cuda"`` selects the device-memory reader; any other value uses host RSS.

    Yields:
        A :class:`MemoryResult` filled in once the block exits.

    Note:
        Both figures are measurements, not guarantees, and neither is a per-runtime sandbox. In one
        shared process the host reader can report ``0.0`` for a large allocation, because RSS counts
        *resident* pages and the allocator may satisfy the request from pages it already holds — no
        sampling rate fixes that, and it cannot be detected from inside this helper. ``delta_mb`` can
        also read negative when the OS reclaims an earlier section's memory during this block. Report
        what comes back; never assert a lower bound on it.

    Note:
        Wrap construction and correctness checks, not the timed loop: the sampler thread adds
        ``psutil`` syscalls that would perturb :func:`measure_latency`'s numbers.

    Examples:
        ``delta_mb`` stays at its ``0.0`` default while the block is open and holds the measurement
        once the block exits, so read it after the ``with``, never inside:

        >>> with measure_memory() as mem:
        ...     reading_inside_the_block = mem.delta_mb
        >>> reading_inside_the_block
        0.0
        >>> isinstance(mem.delta_mb, float)
        True
    """
    reader = _cuda_free_delta_mb if device == "cuda" else _rss_delta_mb
    yield from reader()


def _fmt_ms(result: BenchmarkResult | None) -> str:
    """Format one :class:`BenchmarkResult` as ``"mean ± std"``, or ``"—"`` when the row has no scope for it.

    Examples:
        >>> _fmt_ms(None)
        '—'
        >>> _fmt_ms(BenchmarkResult("cpu", 10.0, 0.5))
        '10.00 ± 0.50'
    """
    if result is None:
        return "—"
    return f"{result.mean_ms:.2f} ± {result.std_ms:.2f}"


def _result_row(
    format_label: str,
    batch: int,
    config: str,
    forward: BenchmarkResult | None,
    end2end: BenchmarkResult | None,
    memory_mb: float | None,
) -> dict[str, str | int | float]:
    """Build one row of a cookbook's results table, keyed the same way across every export cookbook.

    ``forward``/``end2end`` are per-call latency for the whole batch, so FPS is derived as
    ``batch * 1000 / scope.mean_ms`` (images per second), not ``scope.fps`` (calls per second) — a batch-4 call
    that takes 4x as long as batch-1 must still report the same throughput at both rows if the format scales
    perfectly, and only the images/second column makes that comparison read correctly across batch sizes.

    Args:
        format_label: Row label, e.g. ``"ONNX"`` or ``"TensorRT (raw .trt engine)"``.
        batch: Batch size this row was measured at.
        config: Precision/backend/execution-provider string for this row, e.g. ``"CUDA EP"`` or ``"fp16 IR, CPU"``.
        forward: Forward-only timing, or ``None`` if the format has no forward-only scope.
        end2end: End-to-end (preprocess + forward + decode) timing, or ``None`` if the format has no end-to-end
            scope (e.g. a forward-only micro-benchmark).
        memory_mb: Device- or host-memory growth attributed to this row, or ``None`` if this row reuses an
            already-measured number (see each cookbook's own memory-scope note for which rows do).

    Returns:
        A dict with keys ``Format``, ``Batch``, ``Config``, ``forward [ms]``, ``end2end [ms]``,
        ``FPS [img/s] (end2end)``, ``Memory [MB]`` — one row for a :class:`pandas.DataFrame`.
    """
    scope = end2end or forward
    if scope is None:
        raise ValueError(f"_result_row({format_label!r}, batch={batch}) needs at least one of forward/end2end.")
    fps = batch * scope.fps
    return {
        "Format": format_label,
        "Batch": batch,
        "Config": config,
        "forward [ms]": _fmt_ms(forward),
        "end2end [ms]": _fmt_ms(end2end),
        "FPS [img/s] (end2end)": round(fps, 1),
        "Memory [MB]": f"{memory_mb:.1f}" if memory_mb is not None else "—",
    }


def _tile_batch(single_nchw: np.ndarray[Any, Any], batch: int) -> np.ndarray[Any, Any]:
    """Stack *batch* copies of one preprocessed ``(1, C, H, W)`` array into a ``(batch, C, H, W)`` array.

    All *batch* copies are the same image — this measures throughput at a larger batch dimension, not batch diversity.
    """
    return np.concatenate([single_nchw] * batch, axis=0)


def _decode_batch(
    boxes: np.ndarray[Any, Any],
    logits: np.ndarray[Any, Any],
    image_size: tuple[int, int],
    batch: int,
    threshold: float,
) -> None:
    """Run :func:`~rfdetr.export._runtime.decode.decode_detections` over every image in a batched raw-output pair.

    Discards the decoded detections — callers use this inside a :func:`~rfdetr.export._benchmark.measure_latency` thunk,
    where only the wall-clock cost of decoding matters, not the result.
    """
    for i in range(batch):
        decode_detections(boxes[i], logits[i], image_size, threshold=threshold)


def _artifact_size_mb(*paths: Path) -> float:
    """Total on-disk size of *paths* in megabytes, summing a directory's files recursively.

    A single-file export (``.onnx``, ``.trt``, ``.pte``) is one path; a bundle export (OpenVINO's ``.xml`` + ``.bin``,
    CoreML's ``.mlpackage`` directory) is passed as multiple paths or one directory path.
    """
    total_bytes = 0
    for path in paths:
        if path.is_dir():
            total_bytes += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            total_bytes += path.stat().st_size
    return total_bytes / 1e6


#: Reported by :func:`cpu_brand` when the host is not macOS, or when the ``sysctl`` probe cannot run.
_UNKNOWN_CPU_BRAND = "unknown CPU brand"


def cpu_brand() -> str:
    """CPU brand string as reported by macOS ``sysctl``, or a fixed placeholder when it cannot be read.

    The per-hardware cookbooks print this in their Host cell, before any measurement runs, so a raised exception here
    aborts the notebook on the first cell that matters. Two guards are needed rather than one: ``sysctl`` ships only on
    Darwin, and ``subprocess.run(..., check=False)`` suppresses a non-zero *exit status* but not the
    ``FileNotFoundError`` raised when the binary is missing from ``PATH`` entirely — so ``check=False`` alone still
    crashes on the Linux and Windows hosts the CPU and mobile cookbooks claim to support.

    Returns:
        The trimmed brand string on macOS, or :data:`_UNKNOWN_CPU_BRAND` off Darwin, when the probe cannot be spawned,
        or when it reports nothing.

    Examples:
        >>> isinstance(cpu_brand(), str)
        True
    """
    if platform.system() != "Darwin":
        return _UNKNOWN_CPU_BRAND
    try:
        probe = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
        )
    except OSError:
        return _UNKNOWN_CPU_BRAND
    return probe.stdout.strip() or _UNKNOWN_CPU_BRAND


def _enable_notebook_inline_matplotlib() -> None:
    """Enable inline matplotlib figures when running in IPython; a no-op outside a notebook/IPython kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("matplotlib", "inline")
        ipython.run_line_magic("config", "InlineBackend.close_figures = True")


def visualize_detections(detections: sv.Detections, image: Image.Image, save_path: Path | None = None) -> None:
    """Annotate *detections* on *image* and display it inline (and optionally save it) in a notebook.

    Falls back to :data:`~rfdetr.assets.coco_classes.COCO_CLASSES` for label text when *detections* carries no
    ``class_name`` (e.g. a raw decoded output that never went through a class-name-aware decoder).

    Args:
        detections: Detections to draw, already thresholded by the caller.
        image: The source image *detections* was decoded against.
        save_path: When given, also saves the annotated image to this path.
    """
    if detections.class_id is None or detections.confidence is None:
        raise ValueError("visualize_detections requires detections.class_id and detections.confidence to be set.")
    names = detections.data.get("class_name") if detections.data else None
    if names is None:
        names = [COCO_CLASSES.get(int(c), str(c)) for c in detections.class_id]
    labels = [f"{name} {conf:.2f}" for name, conf in zip(names, detections.confidence)]

    annotated = sv.BoxAnnotator(thickness=3).annotate(scene=image.copy(), detections=detections)
    annotated = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=4).annotate(
        scene=annotated, detections=detections, labels=labels
    )
    if save_path is not None:
        annotated.save(save_path)
        print(f"Saved annotated image: {save_path}")
    sv.plot_image(annotated)


#: Annotation archive for COCO 2017; ``instances_val2017.json`` is the only member the accuracy helpers read.
_COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_COCO_VAL_ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
#: Score floor for mAP: low enough to keep the whole precision-recall curve, as ``RFDETR.evaluate`` does.
_COCO_EVAL_THRESHOLD = 0.001


@dataclass(frozen=True)
class CocoValSubset:
    """COCO val2017 images on disk, the annotation file, and the image IDs selected for evaluation.

    Attributes:
        images_dir: Directory holding the ``val2017`` JPEGs.
        annotations_path: Path to ``instances_val2017.json``.
        image_ids: Selected image IDs, in evaluation order.
    """

    images_dir: Path
    annotations_path: Path
    image_ids: tuple[int, ...]


@dataclass(frozen=True)
class CocoMapResult:
    """Box mAP of one runtime on a COCO val2017 subset.

    Attributes:
        map50_95: COCO mAP averaged over IoU 0.50:0.95, in ``[0, 1]``.
        map50: mAP at IoU 0.50, in ``[0, 1]``.
        n_images: Number of images scored.
    """

    map50_95: float
    map50: float
    n_images: int


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, creating parent directories; a separate function so tests can replace it."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def select_coco_val_ids(annotations_path: Path, n_images: int | None = 500, seed: int = 0) -> list[int]:
    """Pick a reproducible subset of COCO val2017 image IDs.

    Args:
        annotations_path: Path to ``instances_val2017.json``.
        n_images: Number of images, or ``None`` for every image in the split.
        seed: Shuffle seed. The same seed selects the same images on every machine.

    Returns:
        Image IDs: sorted when *n_images* is ``None``, otherwise the first *n_images* of a seeded shuffle.

    Raises:
        ValueError: If *n_images* exceeds the number of images in the split.

    Examples:
        >>> select_coco_val_ids.__name__
        'select_coco_val_ids'
    """
    import json

    image_ids = sorted(image["id"] for image in json.loads(Path(annotations_path).read_text())["images"])
    if n_images is None:
        return image_ids
    if n_images > len(image_ids):
        raise ValueError(f"Requested {n_images} images, but the split has only {len(image_ids)}.")
    rng = np.random.default_rng(seed)
    return [int(image_id) for image_id in rng.permutation(image_ids)[:n_images]]


def fetch_coco_val2017(root: Path, n_images: int | None = 500, seed: int = 0) -> CocoValSubset:
    """Download the COCO val2017 annotations and the selected images into *root*, skipping files already present.

    Only the images in the subset are fetched, one by one from their ``coco_url``, so a 500-image subset avoids
    the full 780 MB image archive. The annotation file comes from the 241 MB annotation archive the first time.

    Args:
        root: Directory that receives ``annotations/instances_val2017.json`` and ``val2017/``.
        n_images: Number of images to select, or ``None`` for the full split.
        seed: Selection seed passed to :func:`select_coco_val_ids`.

    Returns:
        The subset, ready for :func:`evaluate_coco_map`.

    Examples:
        >>> fetch_coco_val2017.__name__
        'fetch_coco_val2017'
    """
    import json
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    root = Path(root)
    annotations_path = root / _COCO_VAL_ANNOTATIONS_MEMBER
    if not annotations_path.exists():
        archive = root / "annotations_trainval2017.zip"
        if not archive.exists():
            _download(_COCO_ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extract(_COCO_VAL_ANNOTATIONS_MEMBER, root)
    image_ids = select_coco_val_ids(annotations_path, n_images, seed=seed)
    images_dir = root / "val2017"
    selected = set(image_ids)
    records = [image for image in json.loads(annotations_path.read_text())["images"] if image["id"] in selected]
    missing = [record for record in records if not (images_dir / record["file_name"]).exists()]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda record: _download(record["coco_url"], images_dir / record["file_name"]), missing))
    return CocoValSubset(images_dir=images_dir, annotations_path=annotations_path, image_ids=tuple(image_ids))


def _coco_records(
    output: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | sv.Detections,
    image_id: int,
    image_size: tuple[int, int],
    num_select: int | None,
    background_class_id: int | None,
) -> list[dict[str, Any]]:
    """Turn one image's runtime output into COCO detection records (``bbox`` in pixel ``xywh``)."""
    if isinstance(output, sv.Detections):
        if len(output) == 0:
            return []
        if output.confidence is None or output.class_id is None:
            raise ValueError("Decoded detections need confidence and class_id to be scored.")
        xyxy, scores, class_ids = output.xyxy, output.confidence, output.class_id
    else:
        boxes, logits = (np.asarray(array) for array in output)
        if boxes.ndim == 3:
            boxes, logits = boxes[0], logits[0]
        decoded = decode_detections(
            boxes,
            logits,
            image_size,
            threshold=_COCO_EVAL_THRESHOLD,
            num_select=num_select,
            background_class_id=background_class_id,
        )
        xyxy, scores, class_ids = decoded.xyxy, decoded.confidence, decoded.class_id
    return [
        {
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        }
        for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, class_ids)
    ]


def evaluate_coco_map(
    run: Callable[[Image.Image], tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | sv.Detections],
    subset: CocoValSubset,
    *,
    num_select: int | None = None,
    background_class_id: int | None = None,
    progress: bool = True,
) -> CocoMapResult:
    """Score one runtime's box mAP on a COCO val2017 subset, one image at a time (batch 1).

    *run* receives each image as an RGB PIL image and returns either the raw ``(dets, labels)`` arrays (normalized
    ``cxcywh`` boxes and class logits, with or without a leading batch axis of one) or already-decoded
    :class:`supervision.Detections` in pixel ``xyxy`` (the ``RFDETR.predict()`` path). Raw outputs are decoded with
    :func:`~rfdetr.export._runtime.decode.decode_detections` at a 0.001 score floor.

    Args:
        run: The runtime under test, wrapped to take one PIL image.
        subset: Images and annotations from :func:`fetch_coco_val2017`.
        num_select: Query/class pairs kept per image when decoding raw outputs; ``None`` keeps one per query.
        background_class_id: Class slot dropped when decoding raw outputs. ``None`` (default) suits the official
            COCO checkpoints, whose sparse category IDs use every slot; the decoder's own default ``-1`` would drop
            category 90.
        progress: Whether to show a progress bar.

    Returns:
        mAP@0.50:0.95 and mAP@0.50 over the subset.

    Examples:
        >>> evaluate_coco_map.__name__
        'evaluate_coco_map'
    """
    from faster_coco_eval import COCO, COCOeval_faster
    from PIL import Image as PILImage
    from tqdm.auto import tqdm

    coco_gt = COCO(str(subset.annotations_path))
    records: list[dict[str, Any]] = []
    for image_id in tqdm(subset.image_ids, desc="COCO mAP", disable=not progress):
        file_name = coco_gt.loadImgs([image_id])[0]["file_name"]
        with PILImage.open(subset.images_dir / file_name) as image:
            rgb = image.convert("RGB")
        records += _coco_records(run(rgb), image_id, rgb.size, num_select, background_class_id)
    if not records:
        return CocoMapResult(map50_95=0.0, map50=0.0, n_images=len(subset.image_ids))
    evaluator = COCOeval_faster(coco_gt, coco_gt.loadRes(records), "bbox")
    evaluator.params.imgIds = list(subset.image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return CocoMapResult(
        map50_95=float(evaluator.stats[0]), map50=float(evaluator.stats[1]), n_images=len(subset.image_ids)
    )
