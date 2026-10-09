# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Internal one-pass COCO mean-average-precision adapter.

Purpose:
    Isolate RF-DETR's deliberately narrow dependency on TorchMetrics COCO internals and avoid its per-class evaluator
    reruns. The adapter derives compact per-class AP and AR vectors from the aggregate evaluator arrays produced by one
    global evaluation for each requested IoU type.
Scope:
    Own CPU-backed metric updates, validation of the private TorchMetrics state/backend contract, explicit fixed-order
    distributed state merging, update-state inspection, prediction-score hoisting during COCO-format construction, and
    compact one-pass computation. Lightning lifecycle, EMA voting, logging, checkpoint metrics, F1, keypoint
    evaluation, and terminal rendering remain callback concerns.
Usage:
    Import :class:`OnePassCocoMeanAveragePrecision` only from RF-DETR training code. Construct it with one of the
    backends registered in ``_BACKENDS`` (``hotcoco`` by default, ``faster_coco_eval``, ``ufcoco`` or ``vernier``) and
    ``sync_on_compute=False``; call ``update`` for each batch, explicitly call ``merge_distributed_state`` at
    rank-symmetric callback sites, then call ``compute``.
Outputs:
    Return the same aggregate, per-class, and class-ID tensor keys consumed from TorchMetrics by RF-DETR. Evaluator
    precision, recall, score, and IoU arrays are reduced immediately and are never returned or retained. One
    deliberate divergence: when an IoU type has no images this pass, :meth:`compute` still emits
    ``*_per_class`` sentinel keys (see :meth:`OnePassCocoMeanAveragePrecision._per_class_sentinels`), whereas
    stock TorchMetrics omits them for that branch; this direction is safe for RF-DETR's callback (it always
    expects the per-class keys to exist) but is not covered by parity tests against upstream for that path.
Failure:
    Reject extended summaries, alternative backends, micro averaging, implicit distributed synchronization, and any
    installed TorchMetrics private layout that differs from the verified contract. These failures are intentional and
    actionable; there is no silent slow fallback. The one fallback, ``hotcoco_streaming`` evaluating in one batch
    when streaming cannot apply, is logged and produces the same metrics.
Used by:
    ``rfdetr.training.callbacks.coco_eval.COCOEvalCallback`` for train, validation/test, and EMA COCO accumulators.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib.metadata
import inspect
import io
import operator
import os
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

import numpy as np
import torch
import torchmetrics
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

from rfdetr.config import CocoEvalBackend
from rfdetr.utilities.distributed import all_gather, get_world_size, is_dist_avail_and_initialized
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_METRIC_INPUT_FIELDS = frozenset({"boxes", "scores", "labels", "masks", "iscrowd", "area"})
_MAP_STATE_ATTRS = (
    "detection_box",
    "detection_scores",
    "detection_labels",
    "detection_mask",
    "groundtruth_box",
    "groundtruth_labels",
    "groundtruth_mask",
    "groundtruth_crowds",
    "groundtruth_area",
)
# Parameter names the adapter relies on when calling each backend method. A parameter rename upstream would make
# those calls a raw TypeError rather than an actionable contract failure without this check.
_BACKEND_METHOD_PARAMS: dict[str, tuple[str, ...]] = {
    "_get_coco_datasets": (
        "groundtruth_labels",
        "groundtruth_box",
        "groundtruth_mask",
        "groundtruth_crowds",
        "groundtruth_area",
        "detection_labels",
        "detection_box",
        "detection_mask",
        "detection_scores",
        "iou_type",
        "average",
    ),
    "_coco_stats_to_tensor_dict": ("stats", "prefix", "max_detection_thresholds"),
    "_get_coco_format": ("labels", "all_labels", "boxes", "masks", "scores", "crowds", "area", "iou_type", "average"),
}
# Parameters passed by keyword at this adapter's private-backend call sites. They must not become positional-only in
# a supported TorchMetrics release, because that would otherwise fail as a raw TypeError during metric computation.
_BACKEND_KEYWORD_PARAMS: dict[str, tuple[str, ...]] = {
    "_get_coco_datasets": ("average",),
    "_coco_stats_to_tensor_dict": ("prefix", "max_detection_thresholds"),
    "_get_coco_format": ("labels", "all_labels", "boxes", "masks", "scores", "crowds", "area", "iou_type", "average"),
}
# Evaluator methods compute() calls with no arguments (coco_eval.evaluate() / .accumulate() / .summarize()) — a
# newly-required parameter upstream would make that call fail at compute() time instead of at construction.
_EVALUATOR_ZERO_ARG_METHODS = ("evaluate", "accumulate", "summarize")
_VAR_PARAM_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
# Parity mode every vernier grid runs in; see `_vernier_results` for why it is "corrected" and not "strict".
# Named rather than inlined so a test can drive the same path in both modes, and fail when a vernier release
# starts correcting something that reaches it. Production must stay on "corrected".
_VERNIER_PARITY_MODE: Literal["strict", "corrected"] = "corrected"


def _vernier_thread_budget() -> int:
    """Return the CPU thread budget one DDP-local vernier evaluation should use.

    ``torch.get_num_threads()`` reports the process-wide intra-op thread pool, sized for one process per node.
    Under DDP with multiple ranks sharing a node, handing vernier that same budget on every rank oversubscribes
    the node's CPUs by a factor of ``LOCAL_WORLD_SIZE``; dividing it by the local rank count keeps each rank's
    evaluation within its fair share of the node.

    Returns:
        At least one thread, even when ``LOCAL_WORLD_SIZE`` is unset, zero, or larger than the reported thread
        count.
    """
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    return max(1, torch.get_num_threads() // local_world_size)


def _import_optional_backend(module_name: str, backend_value: str, pip_name: str | None = None) -> Any:
    """Import one of the optional COCO evaluation backend packages.

    ``hotcoco``, ``ultrafast_pycocotools`` and ``vernier`` are required members of the ``train`` extra (see
    pyproject.toml), so a missing import here always means the extra itself was never installed, not that one specific
    backend was left out. Shared by :func:`_hotcoco`, :func:`_ufcoco` and :func:`_vernier` so the backends report a
    missing extra identically.
    Uses the ``__import__`` builtin rather than :func:`importlib.import_module`: the latter bypasses a
    ``patch("builtins.__import__", ...)`` mock, which the missing-dependency regression tests rely on.

    Args:
        module_name: The package's import name (e.g. ``"hotcoco"``).
        backend_value: The ``TrainConfig.eval_backend`` value that selects this package, used in the install hint.
        pip_name: The package's PyPI distribution name, if it differs from ``module_name`` (e.g. a hyphen where the
            import name has an underscore). Defaults to ``module_name``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the optional dependency is not installed.
    """
    try:
        return __import__(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise ImportError(
            f"backend={backend_value!r} requires the {pip_name or module_name} package; "
            "install it with: pip install 'rfdetr[train]'"
        ) from error


def _hotcoco() -> Any:
    """Import the optional ``hotcoco`` backend package.

    Returns:
        The imported ``hotcoco`` module.

    Raises:
        ImportError: If the optional dependency is not installed.
    """
    return _import_optional_backend("hotcoco", "hotcoco")


@contextlib.contextmanager
def _silenced_backend_diagnostics() -> Iterator[None]:
    """Silence hotcoco's intentional ``max_dets`` warning and summary output.

    RF-DETR overrides ``maxDets`` on every evaluation, so hotcoco's warning about that setting describes intentional
    configuration and would otherwise repeat each validation epoch and each IoU type. Since hotcoco 1.2.1, the
    float32 threshold grids no longer produce warnings. The summary table goes through :data:`sys.stdout`, so ordinary
    Python-level redirection catches it.

    Only the known ``max_dets`` :class:`UserWarning` is filtered. Any other warning raised inside the window remains
    visible, and genuine failures still surface as exceptions.

    Yields:
        Nothing; standard output and the warning filter are restored on exit.
    """
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings("ignore", message=r"^hotcoco: max_dets differ from expected", category=UserWarning)
        yield


class _RfdetrCocoBackend(CocoBackend):
    """Typed capability-flag defaults every RF-DETR COCO backend shares.

    :meth:`OnePassCocoMeanAveragePrecision._validate_private_contract` and its constructor read capability flags
    off the active backend -- :attr:`requires_bbox`, :attr:`unused_backend_methods`, :attr:`uses_coco_evaluator`
    and :attr:`streams` (whether ``update()`` matches each batch as it arrives) -- and previously did so through
    ``getattr(backend, name, default)`` at each call site, repeating the same default at every read with no typed
    declaration anywhere a backend could see. Declaring each flag here as a typed class attribute gives every
    backend the same shared default and one place to override it; a new flag belongs here too.
    """

    requires_bbox: bool = False
    unused_backend_methods: tuple[str, ...] = ()
    uses_coco_evaluator: bool = True
    #: Whether :meth:`open_stream` is implemented, so the metric matches each batch in ``update()``.
    streams: bool = False

    def open_stream(
        self,
        categories: list[dict[str, Any]],
        iou_type: str,
        *,
        iou_thresholds: list[float],
        rec_thresholds: list[float],
        max_detection_thresholds: list[int],
    ) -> Any:
        """Return an evaluator that matches each batch as it arrives, for a backend that sets :attr:`streams`.

        Subclass hook. The metric relies on nothing but this duck-typed contract, so a backend that streams through
        another package needs no change to the metric:

        - ``update(images, gt_anns, dt_anns)`` matches one batch: ``images`` and ``gt_anns`` in ``COCO(dict)`` form,
          ``dt_anns`` in a form ``loadRes`` accepts -- the ``(N, 7)`` detection array for ``bbox``, one dict per
          detection carrying only its ``segmentation`` for ``segm``.
        - ``finalize()`` returns a COCO evaluator, already evaluated, that supports ``accumulate()`` and
          ``summarize()`` and then exposes ``stats`` and an ``eval`` dict whose ``precision``/``recall`` arrays index
          their category axis by category id, since ``categories`` holds every id from 0 up.

        Args:
            categories: Every category the stream may see, as COCO category records.
            iou_type: The IoU type the evaluator runs.
            iou_thresholds: IoU thresholds, the grid ``compute()`` evaluates the batch path on.
            rec_thresholds: Recall thresholds of that grid.
            max_detection_thresholds: Maximum-detection thresholds of that grid.

        Returns:
            A fresh streaming evaluator.

        Raises:
            NotImplementedError: If the backend does not stream.
        """
        raise NotImplementedError(f"{type(self).__name__} does not stream; only a backend with streams=True does")


class _PackageCocoBackend(_RfdetrCocoBackend):
    """TorchMetrics COCO backend whose surfaces come from an optional package outside TorchMetrics' backend enum.

    TorchMetrics resolves its COCO, evaluator and mask modules from a closed backend-name enum, so the parent is
    constructed with the supported ``faster_coco_eval`` name and each of the three resolved surfaces is overridden here
    to read from :meth:`_package` instead. Only the surfaces are swapped: every private helper the adapter calls on the
    backend (COCO-format construction, statistics conversion) is TorchMetrics' own and stays shared with the default
    backend. The package is resolved on every access rather than stored on the instance, so the backend pickles with the
    metric under Lightning's DDP spawn and checkpoint plumbing.
    """

    def __init__(self) -> None:
        super().__init__("faster_coco_eval")
        # Import eagerly so a missing optional dependency reports itself. The contract check that runs next
        # resolves `cocoeval` inside an `except ImportError`, which would otherwise swallow the actionable install
        # hint and report a torchmetrics incompatibility instead.
        self._package()

    def _package(self) -> Any:
        """Import and return the package that supplies ``COCO``, ``COCOeval`` and ``mask``.

        Raises:
            ImportError: If the optional dependency is not installed.
        """
        raise NotImplementedError

    @property
    def coco(self) -> object:
        """Return the package's COCO dataset type."""
        return self._package().COCO

    @property
    def cocoeval(self) -> object:
        """Return the package's COCO evaluator type."""
        return self._package().COCOeval

    @property
    def mask_utils(self) -> object:
        """Return the package's RLE mask utilities."""
        return self._package().mask


class _HotCocoBackend(_PackageCocoBackend):
    """TorchMetrics COCO backend that resolves to ``hotcoco`` instead of ``faster-coco-eval``.

    The ``train`` extra's version floor is enforced only when the environment is resolved, so a stale install (an
    older venv, a pinned lockfile elsewhere) would otherwise fail with an ``AttributeError`` deep inside ``compute()``
    after a whole validation epoch. The constructor refuses such an install up front instead. It checks the symbols
    the adapter calls rather than parsing a version string, which also refuses an older release for the single-IoU
    box runs that would not have reached the missing symbol: one actionable error at construction beats a run that
    works or fails depending on the IoU types.
    """

    # hotcoco never reaches `_get_coco_datasets`: it builds its index in the constructor, so this adapter always
    # assembles the COCO-format dictionaries itself on that path. Guarding a call a backend does not make would
    # block its users over an upstream rename that cannot affect them.
    unused_backend_methods = ("_get_coco_datasets",)
    #: hotcoco attributes, dotted from the package, that this backend calls and whose absence means the installed
    #: release predates the ``rfdetr[train]`` floor.
    required_symbols: tuple[str, ...] = ("COCO.update_anns", "COCO.from_arrays")

    def __init__(self) -> None:
        """Import hotcoco and refuse an installed release that lacks a symbol this backend calls.

        Raises:
            ImportError: If hotcoco is not installed, or if the installed release is too old to provide every
                entry of :attr:`required_symbols`.
        """
        super().__init__()
        package = self._package()
        missing = []
        for symbol in self.required_symbols:
            try:
                operator.attrgetter(symbol)(package)
            except AttributeError:
                missing.append(f"hotcoco.{symbol}")
        if missing:
            # Read from the distribution, not the module: hotcoco 1.1 exposes no `__version__`.
            try:
                installed = importlib.metadata.version("hotcoco")
            except importlib.metadata.PackageNotFoundError:
                installed = "(unknown version)"
            raise ImportError(
                f"the installed hotcoco {installed} is too old for RF-DETR's COCO evaluation: it lacks "
                f"{', '.join(missing)}. Upgrade it with: pip install -U 'rfdetr[train]'"
            )

    def _package(self) -> Any:
        """Import and return ``hotcoco``."""
        return _hotcoco()


class _HotCocoStreamingBackend(_HotCocoBackend):
    """Hotcoco backend that matches each batch in ``update()`` through ``hotcoco.StreamingEval``.

    Matching moves from the end of the epoch into the batches: ``compute()`` only accumulates and summarizes, which
    shortens the epoch-end tail, at the cost of more total CPU time inside ``update()``. It does not lower memory:
    the TorchMetrics state the fallback and DDP merge need is kept as well. Streaming is best effort.
    Whenever it cannot apply, :class:`OnePassCocoMeanAveragePrecision` evaluates the stored state in one batch
    exactly as the ``hotcoco`` backend does, so the reported metrics never depend on which path ran.
    """

    streams = True
    required_symbols = (*_HotCocoBackend.required_symbols, "StreamingEval")

    def open_stream(
        self,
        categories: list[dict[str, Any]],
        iou_type: str,
        *,
        iou_thresholds: list[float],
        rec_thresholds: list[float],
        max_detection_thresholds: list[int],
    ) -> Any:
        """Return a ``hotcoco.StreamingEval`` frozen to the grid ``compute()`` sets on the batch path.

        Args:
            categories: Every category the stream may see, as COCO category records.
            iou_type: The IoU type the evaluator runs.
            iou_thresholds: IoU thresholds of the evaluation grid.
            rec_thresholds: Recall thresholds of the evaluation grid.
            max_detection_thresholds: Maximum-detection thresholds of the evaluation grid.

        Returns:
            A fresh ``StreamingEval``; it fills the empty ``cat_ids`` of its ``Params`` from ``categories``.
        """
        package = self._package()
        params = package.Params(iou_type=iou_type)
        params.iou_thrs = np.asarray(iou_thresholds, dtype=np.float64)
        params.rec_thrs = np.asarray(rec_thresholds, dtype=np.float64)
        params.max_dets = max_detection_thresholds
        return package.StreamingEval(categories, iou_type=iou_type, params=params)


#: Key :meth:`OnePassCocoMeanAveragePrecision.__getstate__` sets when it drops an open stream, read back on restore.
_STREAM_DROPPED_ON_COPY = "_stream_dropped_on_copy"


@functools.lru_cache(maxsize=None)
def _warn_streaming_fallback(reason: str) -> None:
    """Log once per process that ``hotcoco_streaming`` evaluates in one batch instead, and why.

    Args:
        reason: Why streaming does not apply; also the deduplication key.
    """
    logger.warning("eval_backend='hotcoco_streaming' evaluates in one batch at epoch end instead: %s.", reason)


def _ufcoco() -> Any:
    """Import the optional ``ultrafast_pycocotools`` backend package.

    Returns:
        The imported ``ultrafast_pycocotools`` module.

    Raises:
        ImportError: If the optional dependency is not installed.
    """
    return _import_optional_backend("ultrafast_pycocotools", "ufcoco", pip_name="ultrafast-pycocotools")


@functools.lru_cache(maxsize=None)
def _ufcoco_evaluator_type() -> type:
    """Return ufcoco's COCO evaluator, with aggregate AP summarized at the largest configured detection limit.

    ufcoco reproduces pycocotools exactly, its summary included: pycocotools reads ``stats[0]`` -- the ``map``
    TorchMetrics reports -- at ``maxDets=100`` whatever the configured thresholds are, and reports ``-1`` when 100 is
    not among them. RF-DETR evaluates at ``eval_max_dets`` (500 by default), and faster-coco-eval and hotcoco both
    read ``stats[0]`` at the largest configured threshold, so this subclass does the same. The other eleven entries
    already use the configured thresholds in pycocotools and are left alone.

    Returns:
        The evaluator class to construct for each IoU type. Built on first use and cached, so the optional import
        stays off the module import path and the adapter sees one class identity.
    """
    evaluator_type = cast(type, _ufcoco().COCOeval)

    # Keep `_collect` inherited: ufcoco's columnar route requires `type(self)._collect is COCOeval._collect`.
    class _UfcocoCocoEval(evaluator_type):  # type: ignore[misc,valid-type]
        def summarize(self) -> None:
            """Summarize results and report aggregate AP at the configured detection limit."""
            super().summarize()
            max_detections = self.params.maxDets[-1]
            if self.params.iouType in ("bbox", "segm") and max_detections != 100:
                self.stats[0] = self._summarize(1, maxDets=max_detections)

    return _UfcocoCocoEval


class _UfcocoMaskTools:
    """Ufcoco's RLE utilities, accepting the boolean masks TorchMetrics hands over.

    TorchMetrics encodes each stored mask as ``np.asfortranarray(mask)`` of the boolean array it keeps, which faster-
    coco-eval and hotcoco accept. ufcoco's ``encode`` holds pycocotools' ``uint8`` contract and rejects a boolean
    array, so it is converted here, one mask at a time; the Fortran-ordered ``uint8`` copy is the array the encoder
    would have been given by pycocotools' own callers. Every other utility -- ``area``, which TorchMetrics calls to size
    annotations -- is the module's own, reached through :meth:`__getattr__`.
    """

    def encode(self, mask: np.ndarray[Any, Any]) -> Any:
        """Encode one binary mask as RLE, converting a boolean array to ``uint8`` first."""
        if mask.dtype == np.bool_:
            mask = np.asfortranarray(mask, dtype=np.uint8)
        return _ufcoco().mask.encode(mask)

    def __getattr__(self, name: str) -> Any:
        """Forward every other RLE utility, such as ``area``, to ultrafast-pycocotools' ``mask`` module."""
        return getattr(_ufcoco().mask, name)


_UFCOCO_MASK_TOOLS = _UfcocoMaskTools()


class _UfcocoBackend(_PackageCocoBackend):
    """TorchMetrics COCO backend that resolves to ``ultrafast-pycocotools`` instead of ``faster-coco-eval``.

    Unlike hotcoco, ufcoco keeps pycocotools' Python-side ``COCO`` object -- ``dataset`` assignment followed by
    ``createIndex()``, annotations read back as the same dictionaries -- so the adapter routes it through the paths it
    takes for faster-coco-eval, with one exception shared with hotcoco: box-only evaluation loads detections through
    ``loadRes`` from one array (see ``OnePassCocoMeanAveragePrecision._loads_detections_from_array``) instead of
    building the prediction dataset. Beyond that, the only adaptations are the two places where ufcoco follows
    pycocotools more literally than the other backends do, :func:`_ufcoco_evaluator_type` and
    :class:`_UfcocoMaskTools`.
    """

    def _package(self) -> Any:
        """Import and return ``ultrafast_pycocotools``."""
        return _ufcoco()

    @property
    def cocoeval(self) -> object:
        """Return ufcoco's COCO evaluator type, summarizing at the configured detection limit."""
        return _ufcoco_evaluator_type()

    @property
    def mask_utils(self) -> object:
        """Return ufcoco's RLE mask utilities, accepting boolean masks."""
        return _UFCOCO_MASK_TOOLS


class _FasterCocoEvalBackend(_RfdetrCocoBackend):
    """TorchMetrics' own ``faster_coco_eval`` backend, the one :class:`MeanAveragePrecision` builds itself.

    Nothing is overridden beyond the shared capability-flag defaults: the class exists so the registry holds one
    backend class per name and the metric builds every backend the same way.
    """

    def __init__(self) -> None:
        super().__init__("faster_coco_eval")


def _vernier() -> Any:
    """Import the optional ``vernier`` backend package.

    Returns:
        The imported ``vernier`` module.

    Raises:
        ImportError: If the optional dependency is not installed.
    """
    return _import_optional_backend("vernier", "vernier")


class _VernierBackend(_RfdetrCocoBackend):
    """TorchMetrics COCO backend built with the ``faster_coco_eval`` name, evaluating on ``vernier``'s native API.

    vernier takes both ground truth and detections as arrays, so
    :meth:`OnePassCocoMeanAveragePrecision._vernier_results` bypasses the COCO dataset and evaluator surfaces
    entirely; only the parent's statistics helper and RLE mask utilities are used. It inherits
    :class:`_RfdetrCocoBackend` directly rather than :class:`_FasterCocoEvalBackend`: the two share only that
    constructor argument, and subclassing the sibling backend it happens to resemble instead of the common base
    both descend from is not a real is-a relationship.
    """

    # vernier evaluates on its own API, reaching neither the COCO dataset and evaluator surfaces nor
    # `_get_coco_format`: one statistics helper is all of TorchMetrics it calls.
    unused_backend_methods = ("_get_coco_datasets", "_get_coco_format")
    uses_coco_evaluator = False
    # vernier needs a box on every annotation, so it cannot evaluate a mask-only run.
    requires_bbox = True

    def __init__(self) -> None:
        # TorchMetrics resolves its COCO modules from a closed backend-name enum with no `vernier` member, the
        # same reason `OnePassCocoMeanAveragePrecision.__init__` builds every backend under the supported
        # `faster_coco_eval` name (see its constructor comment) -- named explicitly here rather than borrowed
        # from `_FasterCocoEvalBackend.__init__` now that this class no longer inherits it.
        super().__init__("faster_coco_eval")
        # Import eagerly for the same reason `_PackageCocoBackend` does: the contract check that runs next would
        # otherwise let a missing package surface only at the first `compute()`.
        _vernier()


#: Registry of every COCO evaluation backend the adapter accepts: `TrainConfig.eval_backend` value -> class of the
#: backend object the metric evaluates with. Adding a backend is one entry here plus its `CocoEvalBackend` member in
#: `rfdetr.config`; the constructor never branches on the name. `pycocotools` is excluded deliberately: it is an order
#: of magnitude slower and RF-DETR never installs it. All of them ship with `rfdetr[train]`.
_BACKENDS: dict[CocoEvalBackend, Callable[[], _RfdetrCocoBackend]] = {
    "faster_coco_eval": _FasterCocoEvalBackend,
    "hotcoco": _HotCocoBackend,
    "hotcoco_streaming": _HotCocoStreamingBackend,
    "ufcoco": _UfcocoBackend,
    "vernier": _VernierBackend,
}


def _rle_dicts(masks: list[Any]) -> list[dict[str, Any]]:
    """Return stored per-image ``(size, counts)`` masks as one flat list of COCO RLE dictionaries, in state order.

    Args:
        masks: Per-image tuples of ``(size, counts)`` pairs, as TorchMetrics stores ``detection_mask`` and
            ``groundtruth_mask``.

    Returns:
        One RLE dictionary for each stored mask.

    Examples:
        >>> _rle_dicts([(((2, 3), b"06"),), ()])
        [{'size': (2, 3), 'counts': b'06'}]
    """
    return [{"size": size, "counts": counts} for image in masks for size, counts in image]


@dataclasses.dataclass(frozen=True)
class _PredictionAreaColumns:
    """Per-IoU-type ``area`` columns of a hotcoco prediction set loaded from the detection array.

    Entry ``i`` of each column belongs to annotation ID ``i + 1``: ``load_res`` numbers array rows from 1 in order.

    Attributes:
        by_iou_type: One float64 area column for each evaluated IoU type, keyed by that type.
    """

    by_iou_type: dict[str, np.ndarray[Any, Any]]


class OnePassCocoMeanAveragePrecision(MeanAveragePrecision):
    """Compute compact COCO AP/AR with CPU state and one global evaluation.

    The subclass is RF-DETR's internal compatibility boundary around private TorchMetrics 1.x COCO state and backend
    helpers. It deliberately supports only the configuration used by the training callback. Per-class AP and AR are
    reduced from the aggregate evaluator's precision and recall arrays, avoiding TorchMetrics' additional evaluator
    construction and one evaluation per observed class.

    Args:
        box_format: Input bounding-box representation.
        iou_type: COCO IoU types to evaluate.
        iou_thresholds: Optional IoU thresholds forwarded to TorchMetrics.
        rec_thresholds: Optional recall thresholds forwarded to TorchMetrics.
        max_detection_thresholds: Three COCO maximum-detection thresholds.
        class_metrics: Whether to return per-class AP and AR.
        extended_summary: Must remain ``False`` so large evaluator arrays do not escape computation.
        average: Must remain ``"macro"`` because RF-DETR logs class-level metrics.
        backend: COCO evaluation backend. ``"vernier"`` is the default; ``"faster_coco_eval"`` selects the previous
            evaluator, ``"ufcoco"`` selects ultrafast-pycocotools and ``"hotcoco"`` selects hotcoco.
            ``"hotcoco_streaming"`` is hotcoco with matching done batch by batch in :meth:`update`. All of them
            ship with ``rfdetr[train]`` and return identical metrics.
        num_classes: The model's class count; predicted and target labels lie in ``[0, num_classes]``. Only
            ``"hotcoco_streaming"`` reads it, because ``StreamingEval`` needs every category before the first
            batch. ``None`` makes that backend evaluate in one batch at epoch end. Its epoch-end accumulate and
            summarize therefore scale with this declared count, not with the classes a run actually observes.
        kwargs: TorchMetrics configuration. ``sync_on_compute`` defaults to and must remain ``False`` because the
            callback invokes :meth:`merge_distributed_state` explicitly at rank-symmetric sites.

    Raises:
        ValueError: If a configuration falls outside the RF-DETR adapter contract.
        RuntimeError: If the installed TorchMetrics private state/backend contract is incompatible.
    """

    def __init__(
        self,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        iou_type: Literal["bbox", "segm"] | tuple[Literal["bbox", "segm"], ...] = "bbox",
        iou_thresholds: list[float] | None = None,
        rec_thresholds: list[float] | None = None,
        max_detection_thresholds: list[int] | None = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        average: Literal["macro", "micro"] = "macro",
        backend: CocoEvalBackend = "hotcoco",
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        if extended_summary:
            raise ValueError("OnePassCocoMeanAveragePrecision does not support extended_summary=True")
        if backend not in _BACKENDS:
            raise ValueError(f"OnePassCocoMeanAveragePrecision requires backend in {tuple(_BACKENDS)}")
        if average != "macro":
            raise ValueError("OnePassCocoMeanAveragePrecision requires average='macro'")
        sync_on_compute = kwargs.pop("sync_on_compute", False)
        if sync_on_compute is not False:
            raise ValueError("OnePassCocoMeanAveragePrecision requires sync_on_compute=False")
        super().__init__(
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            extended_summary=False,
            average=average,
            # TorchMetrics resolves its COCO modules from a closed backend-name enum that has no hotcoco, ufcoco or
            # vernier member, so the supported name is what upstream sees and the backend object is replaced from the
            # registry afterwards.
            backend="faster_coco_eval",
            sync_on_compute=False,
            **kwargs,
        )
        self._coco_backend = _BACKENDS[backend]()
        # Rejected in the constructor rather than at compute(), which would discard a whole validation epoch.
        # Declared on the backend that imposes it, so the registry keeps its promise about the name.
        if self._coco_backend.requires_bbox and "bbox" not in self.iou_type:
            raise ValueError(f"backend={backend!r} requires 'bbox' among the IoU types; it needs a box per annotation")
        self._validate_private_contract()
        # `StreamingEval` takes every category at construction, so the model's whole label range is declared up
        # front: `[0, num_classes]`, background included. Categories nothing ever hits stay at -1 and drop out of
        # every mean, exactly as the batch path's observed-class `catIds` would leave them.
        self._stream_categories: list[dict[str, Any]] | None = None
        if self._coco_backend.streams:
            if num_classes is None:
                _warn_streaming_fallback("the model's num_classes is unknown")
            else:
                self._stream_categories = [{"id": index, "name": str(index)} for index in range(num_classes + 1)]
        self._streams: dict[str, Any] | None = None
        self._streamed_images = 0
        self._stream_stopped = False

    @property
    def has_updates(self) -> bool:
        """Return whether at least one batch has updated this metric."""
        update_count = getattr(self, "_update_count", None)
        if isinstance(update_count, int):
            return update_count > 0
        if torch.is_tensor(update_count):
            return bool(update_count.detach().cpu().item() > 0)
        return True

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Validate inputs and store detached CPU copies of fields consumed by TorchMetrics.

        Args:
            preds: Per-image predictions in TorchMetrics detection format.
            target: Per-image ground-truth annotations in TorchMetrics detection format.
        """
        cpu_preds = [
            {name: value.detach().cpu() for name, value in item.items() if name in _METRIC_INPUT_FIELDS}
            for item in preds
        ]
        cpu_target = [
            {name: value.detach().cpu() for name, value in item.items() if name in _METRIC_INPUT_FIELDS}
            for item in target
        ]
        super().update(cpu_preds, cpu_target)
        if self._stream_categories is not None and not self._stream_stopped and cpu_target:
            self._stream_batch(len(cpu_target))

    def reset(self) -> None:
        """Reset the stored state and start a fresh stream for the next epoch."""
        super().reset()
        self._streams = None
        self._streamed_images = 0
        self._stream_stopped = False

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state, without the open streaming evaluators.

        Returns:
            The metric state. A stream in progress is dropped, and the copy's ``compute()`` then evaluates the stored
            state in one batch; every image streamed so far is still in that state. The state carries a marker so
            :meth:`__setstate__` logs the fallback on the copy, the object that takes it -- this object keeps its
            stream.
        """
        state = super().__getstate__()
        if state.get("_streams") is not None:
            # TODO(hotcoco): keep the stream instead. hotcoco 1.2 pickles `StreamingEval` (proposal 2 landed), and a
            # copy that keeps streaming reports the same metrics; the drop stays only while the documented
            # copy-falls-back contract and its tests are unchanged.
            state["_streams"] = None
            state["_stream_stopped"] = True
            state[_STREAM_DROPPED_ON_COPY] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled or copied metric, logging when its open stream was dropped on the way.

        Args:
            state: The state :meth:`__getstate__` returned.
        """
        stream_dropped = state.pop(_STREAM_DROPPED_ON_COPY, False)
        super().__setstate__(state)
        if stream_dropped:
            _warn_streaming_fallback("the metric was copied or unpickled mid-epoch, which drops its open stream")

    def _stop_streaming(self, reason: str) -> None:
        """Abandon this epoch's stream; ``compute()`` then evaluates the stored state in one batch.

        Args:
            reason: Why streaming does not apply, logged once per process.
        """
        _warn_streaming_fallback(reason)
        self._streams = None
        self._stream_stopped = True

    def _stream_batch(self, num_images: int) -> None:
        """Match the batch just stored by :meth:`update` on each IoU type's ``StreamingEval``.

        Args:
            num_images: How many images the batch added to the end of the stored state.
        """
        if is_dist_avail_and_initialized() and get_world_size() > 1:
            # TODO(hotcoco): stream under DDP once `StreamingEval` can merge state across processes (rfdetr hotcoco
            # proposal 3). Each rank sees only its shard, and the merged state exists only after
            # `merge_distributed_state()` at epoch end.
            self._stop_streaming("StreamingEval cannot merge state across distributed ranks")
            return
        categories = cast(list[dict[str, Any]], self._stream_categories)
        # Unlike detection arrays, target annotations are passed as dictionaries; keep the batch path's explicit
        # validation rather than letting hotcoco report its lower-level conversion error.
        self._validate_detection_labels(self.groundtruth_labels[-num_images:])
        labels = torch.cat([*self.detection_labels[-num_images:], *self.groundtruth_labels[-num_images:]])
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= len(categories)):
            # hotcoco rejects an undeclared category with `KeyError`, but a negative id in the detection array
            # with `ValueError`. One range check ahead of `update()` keeps both ends of the label range on the same
            # fallback without catching a genuine input error. A NaN score is not caught here: `load_res()` raises
            # `ValueError` for it.
            self._stop_streaming(f"a label falls outside the declared categories [0, {len(categories) - 1}]")
            return
        if self._streams is None:
            # TorchMetrics declares the attribute as its own `CocoBackend`; the registry only builds RF-DETR ones.
            backend = cast(_RfdetrCocoBackend, self._coco_backend)
            self._streams = {
                iou_type: backend.open_stream(
                    categories,
                    iou_type,
                    iou_thresholds=self.iou_thresholds,
                    rec_thresholds=self.rec_thresholds,
                    max_detection_thresholds=self.max_detection_thresholds,
                )
                for iou_type in self.iou_type
            }
        images, ground_truth, detections = self._stream_records(num_images)
        # Not wrapped in `_quiet_evaluation()`: `update()` prints and warns nothing, and swapping the process-wide
        # stdout on every validation step would swallow whatever a progress bar wrote meanwhile.
        for iou_type, stream in self._streams.items():
            stream.update(images, ground_truth, detections[iou_type])
        self._streamed_images += num_images

    def _stream_records(self, num_images: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Return the last ``num_images`` stored images as ``StreamingEval.update`` inputs.

        TorchMetrics' ``_get_coco_format`` builds the ground truth, so its ``area`` follows the same rule as the
        batch path. Image ids continue from earlier batches, matching the state-order ids the batch path assigns.

        Args:
            num_images: How many images at the end of the stored state to convert.

        Returns:
            The image records, the ground-truth annotations, and the detections for each IoU type.
        """
        offset = self._streamed_images
        target = self._coco_backend._get_coco_format(
            labels=self.groundtruth_labels[-num_images:],
            boxes=self.groundtruth_box[-num_images:] if self.groundtruth_box else None,
            masks=self.groundtruth_mask[-num_images:] if self.groundtruth_mask else None,
            crowds=self.groundtruth_crowds[-num_images:],
            area=self.groundtruth_area[-num_images:],
            iou_type=self.iou_type,
            all_labels=[],
            average=self.average,
        )
        for image in target["images"]:
            image["id"] += offset
        for annotation in target["annotations"]:
            annotation["image_id"] += offset
        return target["images"], target["annotations"], self._stream_detections(num_images, offset)

    def _stream_detections(self, num_images: int, offset: int) -> dict[str, Any]:
        """Return the last ``num_images`` stored images' detections as raw results, one list per IoU type.

        Built from the stored state directly rather than through ``_get_coco_format``, which converts each
        detection's score separately and dominated streaming's cost. ``StreamingEval`` derives each detection's area
        from the geometry it is given, so each IoU type gets the form that yields the area the batch path switches
        to:

        - ``bbox``: the ``(N, 7)`` array ``[image_id, x, y, width, height, score, category_id]`` that
          ``StreamingEval.update`` takes since hotcoco 1.2, one row per detection, box area, no Python dict per
          detection. Boxes are already COCO ``xywh``, as TorchMetrics stores them.
        - ``segm``: one dict per detection carrying only its ``segmentation``, so its area is the mask's. The array
          with ``segmentation=`` would give every row its box area, moving masks between the area buckets.

        Within an image, detections keep state order, the tie order the batch path ranks equal scores in.

        Args:
            num_images: How many images at the end of the stored state to convert.
            offset: The id of the first of those images.

        Returns:
            Detections in the shape ``load_res`` accepts, keyed by IoU type.

        Raises:
            ValueError: If an image's scores are not a one-dimensional floating-point tensor.
        """
        scores = self.detection_scores[-num_images:]
        self._validate_detection_scores(scores)
        labels = self.detection_labels[-num_images:]
        detections: dict[str, Any] = {}
        if "bbox" in self.iou_type:
            # `_fix_empty_tensors` stores an image without boxes as `(1, 0)`; reshaping restores `(0, 4)`.
            boxes = [image.reshape(-1, 4) for image in self.detection_box[-num_images:]]
            image_ids = torch.repeat_interleave(
                torch.arange(offset, offset + num_images), torch.tensor([len(image) for image in boxes])
            )
            columns = (image_ids, torch.cat(boxes), torch.cat(scores), torch.cat(labels))
            detections["bbox"] = torch.column_stack([column.double() for column in columns]).numpy()
        if "segm" in self.iou_type:
            masks = self.detection_mask[-num_images:]
            detections["segm"] = [
                {
                    "image_id": offset + index,
                    "category_id": label,
                    "segmentation": {"size": size, "counts": counts},
                    "score": score,
                }
                for index, (image_labels, image_scores) in enumerate(zip(labels, scores))
                for label, score, (size, counts) in zip(image_labels.tolist(), image_scores.tolist(), masks[index])
            ]
        return detections

    def merge_distributed_state(self) -> None:
        """Merge all TorchMetrics list states across ranks using a fixed collective order.

        The explicit call site is intentional: every callback rank must enter the same collectives in the same order.
        TorchMetrics' tensor gather can issue a shape-dependent number of collectives for segmentation state, whereas
        RF-DETR's object gather performs exactly one collective for each declared list state.
        """
        if not is_dist_avail_and_initialized() or get_world_size() == 1:
            return
        for attr in _MAP_STATE_ATTRS:
            local = getattr(self, attr)
            local_cpu = [value.detach().cpu() if torch.is_tensor(value) else value for value in local]
            gathered = all_gather(local_cpu)
            setattr(self, attr, [item for rank_items in gathered for item in rank_items])
        self._update_count = max(getattr(self, "_update_count", 0), 1)

    def compute(self) -> dict[str, Tensor]:
        """Return aggregate and compact per-class COCO metrics from one evaluator per IoU type.

        Returns:
            TorchMetrics-compatible aggregate metrics, per-class AP/AR vectors, and observed class IDs.

        Raises:
            RuntimeError: If the installed backend no longer exposes the evaluator arrays required for one-pass
                reduction.
        """
        classes = self._observed_classes()
        logger.debug("Computing one-pass COCO metrics for %d classes and IoU types %s.", len(classes), self.iou_type)
        if isinstance(self._coco_backend, _VernierBackend):
            return {**self._vernier_results(classes), "classes": torch.tensor(classes, dtype=torch.int32)}
        # A stream is used only if it saw every stored image: one opened after a mid-epoch unpickle, or one the
        # state outgrew in `merge_distributed_state()`, falls through to the batch path with the full state. So does
        # a state with no detection or no ground truth at all: the batch path owns the `-1` sentinels for an empty
        # side, and TorchMetrics drops every prediction image of a segm state without masks, which a stream would
        # still finalize against its ground truth. Evaluating an empty side in one batch costs next to nothing.
        if (
            self._streams is not None
            and self._streamed_images == len(self.groundtruth_labels)
            and any(labels.numel() for labels in self.detection_labels)
            and any(labels.numel() for labels in self.groundtruth_labels)
        ):
            return {**self._streaming_results(classes), "classes": torch.tensor(classes, dtype=torch.int32)}
        coco_preds, coco_target, prediction_dataset = self._coco_datasets(classes)

        result: dict[str, Tensor] = {}
        for iou_type in self.iou_type:
            prefix = "" if len(self.iou_type) == 1 else f"{iou_type}_"
            if len(self.iou_type) > 1:
                coco_preds = self._prediction_dataset_for_iou_type(coco_preds, prediction_dataset, iou_type)
            if len(coco_preds.imgs) == 0 or len(coco_target.imgs) == 0:
                result.update(self._empty_iou_type_results(prefix, classes))
                continue

            evaluator_factory = cast(Callable[..., Any], self._coco_backend.cocoeval)
            # By keyword, never positionally: both backends accept `iouType`, but a parameter inserted before it
            # upstream would bind the IoU type to the wrong slot and evaluate a detection run as segmentation,
            # silently. hotcoco spells it `iou_type` natively and accepts `iouType` as a pycocotools alias.
            coco_eval = evaluator_factory(coco_target, coco_preds, iouType=iou_type)
            # Whole-object assignment, not field-by-field mutation: on hotcoco 0.5 the `params` getter returned a
            # copy, so writing a field through it was a silent no-op that left `max_detection_thresholds` at
            # COCO's default 100 with no error. 1.0.0 makes field writes take effect but still documents
            # pull-edit-assign as the supported idiom, and faster-coco-eval returns the live object, where this is
            # equivalent -- so the one form that is correct everywhere is the whole-object assignment.
            params = coco_eval.params
            params.iouThrs = np.asarray(self.iou_thresholds, dtype=np.float64)
            params.recThrs = np.asarray(self.rec_thresholds, dtype=np.float64)
            params.maxDets = self.max_detection_thresholds
            params.catIds = classes
            coco_eval.params = params
            # Only the three evaluator calls are silenced. Widening the window to the rest of this loop would send
            # the adapter's own contract-failure logging to /dev/null along with the backend's chatter.
            with self._quiet_evaluation():
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            result.update(
                self._coco_backend._coco_stats_to_tensor_dict(
                    coco_eval.stats, prefix=prefix, max_detection_thresholds=self.max_detection_thresholds
                )
            )
            result.update(self._reduce_per_class(getattr(coco_eval, "eval", None), prefix, classes))

        result["classes"] = torch.tensor(classes, dtype=torch.int32)
        return result

    def _streaming_results(self, classes: list[int]) -> dict[str, Tensor]:
        """Finalize each IoU type's ``StreamingEval`` and reduce it, keyed for TorchMetrics.

        ``finalize()`` consumes the evaluators, so the stream is closed until :meth:`reset`; a repeated
        ``compute()`` before then takes the batch path over the same stored state.

        Args:
            classes: Sorted class IDs observed in predictions or targets.

        Returns:
            TorchMetrics-compatible aggregate metrics and per-class AP/AR vectors, without ``classes``.
        """
        streams = cast(dict[str, Any], self._streams)
        self._streams = None
        self._stream_stopped = True
        result: dict[str, Tensor] = {}
        for iou_type, stream in streams.items():
            prefix = "" if len(self.iou_type) == 1 else f"{iou_type}_"
            coco_eval = stream.finalize()
            with self._quiet_evaluation():
                coco_eval.accumulate()
                coco_eval.summarize()
            result.update(
                self._coco_backend._coco_stats_to_tensor_dict(
                    coco_eval.stats, prefix=prefix, max_detection_thresholds=self.max_detection_thresholds
                )
            )
            evaluation = getattr(coco_eval, "eval", None)
            if self.class_metrics and isinstance(evaluation, dict) and {"precision", "recall"} <= evaluation.keys():
                # The category axis spans every declared category, whose index is its id; keep the observed ones,
                # the axis the batch path's `catIds` produces.
                evaluation = {
                    "precision": np.asarray(evaluation["precision"])[:, :, classes],
                    "recall": np.asarray(evaluation["recall"])[:, classes],
                }
            result.update(self._reduce_per_class(evaluation, prefix, classes))
        return result

    def _vernier_results(self, classes: list[int]) -> dict[str, Tensor]:
        """Evaluate on vernier's native grid, one per IoU type, keyed for TorchMetrics.

        ``vernier.adapters.coco_inputs_from_columns`` builds the inputs from
        :meth:`_vernier_columns`, once for the whole call, reusing the already concatenated stored state.

        ``iou_thresholds`` / ``rec_thresholds`` are forwarded because TorchMetrics builds them with
        ``torch.linspace`` in ``float32``, unlike vernier.

        Args:
            classes: Sorted class IDs observed in predictions or targets, used as COCO category IDs.

        Returns:
            TorchMetrics-compatible aggregate metrics and per-class AP/AR vectors, without ``classes``.
        """
        vernier = _vernier()
        # Built once, not once per IoU type: the records do not depend on which
        # grid reads them, and at validation scale rebuilding them is the most
        # expensive thing on this path.
        inputs = None
        if self.groundtruth_labels:
            detection_columns, target_columns = self._vernier_columns()
            # Built once for the whole run: the inputs do not name a kernel, so
            # one set serves both passes of a bbox+segm run.
            inputs = vernier.adapters.coco_inputs_from_columns(
                detection_columns,
                target_columns,
                box_format="xywh",
                categories=classes,
                area="auto",
            )
        result: dict[str, Tensor] = {}
        for iou_type in self.iou_type:
            prefix = "" if len(self.iou_type) == 1 else f"{iou_type}_"
            if inputs is None:
                result.update(self._empty_iou_type_results(prefix, classes))
                continue
            ground_truth, detections = inputs
            evaluate_grid = (
                vernier.instance.evaluate_bbox_grid if iou_type == "bbox" else vernier.instance.evaluate_segm_grid
            )
            grid = evaluate_grid(
                ground_truth,
                detections,
                parity_mode=_VERNIER_PARITY_MODE,
                max_dets_per_image=self.max_detection_thresholds[-1],
                use_cats=True,
                iou_thresholds=self.iou_thresholds,
                recall_thresholds=self.rec_thresholds,
                num_threads=_vernier_thread_budget(),
                dt_area="bbox" if iou_type == "bbox" else "mask",
            )
            accumulated = grid.accumulate(self.max_detection_thresholds)
            result.update(
                self._coco_backend._coco_stats_to_tensor_dict(
                    accumulated.summarize().stats, prefix=prefix, max_detection_thresholds=self.max_detection_thresholds
                )
            )
            # Each access materializes a fresh array across the FFI, and `_reduce_per_class` discards both
            # unless `class_metrics` is on, which by default it is not.
            evaluation = None
            if self.class_metrics:
                evaluation = {"precision": accumulated.precision, "recall": accumulated.recall}
            result.update(self._reduce_per_class(evaluation, prefix, classes))
        return result

    def _vernier_columns(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the stored state as the whole columns ``coco_inputs_from_columns`` reads.

        ``rles`` is written only if ``segm`` is in ``self.iou_type``.

        Columns are handed over at their stored dtype; vernier widens them at its own ingest boundary,
        including the ``bfloat16`` an autocast run holds (needs ``vernier>=0.5.3``).

        Returns:
            The detection columns and the target columns.
        """
        detection_columns: dict[str, Any] = {
            # `_fix_empty_tensors` reshapes an empty per-image box tensor to `(1, 0)` rather than `(0, 4)` to
            # avoid a DDP all-reduce hang, which `torch.cat` rejects against a `(N, 4)` tensor from another
            # image and whose `len()` would miscount that image as holding one detection.
            "boxes": torch.cat([image.reshape(-1, 4) for image in self.detection_box]),
            "scores": torch.cat(self.detection_scores),
            "labels": torch.cat(self.detection_labels),
            "counts": torch.tensor([len(image) for image in self.detection_scores]),
        }
        target_columns: dict[str, Any] = {
            "boxes": torch.cat([image.reshape(-1, 4) for image in self.groundtruth_box]),
            "labels": torch.cat(self.groundtruth_labels),
            "iscrowd": torch.cat(self.groundtruth_crowds),
            "area": torch.cat(self.groundtruth_area),
            "counts": torch.tensor([len(image) for image in self.groundtruth_labels]),
        }
        if "segm" in self.iou_type:
            detection_columns["rles"] = [
                {"size": size, "counts": counts} for image in self.detection_mask for size, counts in image
            ]
            target_columns["rles"] = [
                {"size": size, "counts": counts} for image in self.groundtruth_mask for size, counts in image
            ]
        return detection_columns, target_columns

    def _coco_datasets(self, classes: list[int]) -> tuple[Any, Any, dict[str, Any] | _PredictionAreaColumns | None]:
        """Return the COCO prediction and target datasets, hoisting prediction scores out of the annotation loop.

        TorchMetrics' ``_get_coco_format`` hoists boxes and labels to Python lists once per image but reads scores
        one annotation at a time (``scores[image_id][k].cpu().tolist()``, ``helpers.py:563``), which is CPU tensor
        indexing and scalar conversion for every detection because ``update()`` stores detached CPU state. At
        RF-DETR's validation scale that is hundreds of thousands of conversions per ``compute()``. Asking upstream
        for the same annotations with
        ``scores=None`` and assigning per-image score lists afterwards produces byte-identical datasets while
        converting each image's scores once.

        The rewrite needs prediction annotations to appear in state order with no image dropped. Upstream skips an
        image only when it has no masks *and* no boxes (``helpers.py:508-511``), so the hoist is used only when
        boxes are present; a mask-only prediction state falls back to upstream unchanged.

        Args:
            classes: Sorted class IDs observed in predictions or targets, used as COCO category IDs.

        Returns:
            The prediction and target datasets in the order ``_get_coco_datasets`` returns them, followed by the
            source of the per-IoU-type prediction areas a multi-IoU-type evaluation switches between: the
            COCO-format dictionary the prediction dataset was built from (its ``area_bbox``/``area_segm`` values;
            reading them back from hotcoco's ``dataset`` getter would copy the whole prediction set), or
            :class:`_PredictionAreaColumns` when hotcoco loaded a multi-IoU-type set from the detection array.
            ``None`` when the predictions were loaded from the array for a single IoU type, where no area switching
            happens.

        Raises:
            ValueError: If stored detection scores are not one-dimensional floating-point tensors, or, on the
                detection-array path, if a non-empty image's detection labels are not an integer tensor.
            RuntimeError: If upstream stops emitting one annotation for each stored detection score.
        """
        backend = self._coco_backend
        detection_boxes = self.detection_box if len(self.detection_box) > 0 else None
        # hotcoco cannot take the upstream helper's datasets: it builds its index in the constructor, so the
        # `dataset` assignment and `createIndex()` call that helper ends with have nothing to act on. The scores are
        # then handed to `_get_coco_format` the way upstream hands them over, instead of being hoisted.
        if detection_boxes is None and not isinstance(backend, _HotCocoBackend):
            coco_preds, coco_target = cast(
                tuple[Any, Any],
                backend._get_coco_datasets(
                    self.groundtruth_labels,
                    self.groundtruth_box,
                    self.groundtruth_mask,
                    self.groundtruth_crowds,
                    self.groundtruth_area,
                    self.detection_labels,
                    self.detection_box,
                    self.detection_mask,
                    self.detection_scores,
                    self.iou_type,
                    average=self.average,
                ),
            )
            return coco_preds, coco_target, cast(dict[str, Any], coco_preds.dataset)

        # Built before the target: a float detection label is also a category ID there, which hotcoco's `COCO`
        # constructor rejects with a `TypeError` before the array's own label check would have run.
        detections = self._detection_results_array() if self._loads_detections_from_array(detection_boxes) else None
        if detections is not None and isinstance(backend, _HotCocoBackend):
            return self._hotcoco_datasets_from_arrays(detections, classes)
        # `_get_coco_datasets` passes this same list of Python ints (helpers.py:216) even though the parameter is
        # annotated `list[Tensor]`; the values only ever become COCO category IDs.
        all_labels = cast(list[Tensor], classes)
        target_dataset = backend._get_coco_format(
            labels=self.groundtruth_labels,
            boxes=self.groundtruth_box if len(self.groundtruth_box) > 0 else None,
            masks=self.groundtruth_mask if len(self.groundtruth_mask) > 0 else None,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
            iou_type=self.iou_type,
            all_labels=all_labels,
            average=self.average,
        )
        coco_target = self._build_coco(target_dataset)
        if detections is not None:
            # ufcoco's `loadRes` prints pycocotools' progress lines; hotcoco's is silent, so the redirect is harmless
            # there.
            with contextlib.redirect_stdout(io.StringIO()):
                coco_preds = coco_target.loadRes(detections)
            return coco_preds, coco_target, None

        prediction_dataset = backend._get_coco_format(
            labels=self.detection_labels,
            boxes=detection_boxes,
            masks=self.detection_mask if len(self.detection_mask) > 0 else None,
            scores=None if detection_boxes is not None else self.detection_scores,
            iou_type=self.iou_type,
            all_labels=all_labels,
            average=self.average,
        )
        if detection_boxes is not None:
            self._assign_detection_scores(prediction_dataset["annotations"], self.detection_scores)
        return self._build_coco(prediction_dataset), coco_target, prediction_dataset

    def _hotcoco_datasets_from_arrays(
        self, detections: np.ndarray[Any, Any], classes: list[int]
    ) -> tuple[Any, Any, _PredictionAreaColumns | None]:
        """Return hotcoco's prediction and target datasets built from the stored columns, with no dict per annotation.

        The target comes from ``COCO.from_arrays`` (see :meth:`_hotcoco_target_from_arrays`) and the predictions from
        ``load_res`` over the detection array, with the stored masks handed over as ``segmentation=`` when ``segm``
        is evaluated. ``load_res`` keeps each row's box rather than deriving one from its mask and numbers the rows
        ``1..N`` in order, as pycocotools does; it sets every prediction ``area`` to the box area, which a
        multi-IoU-type evaluation then switches per IoU type through :meth:`_prediction_dataset_for_iou_type`.

        Args:
            detections: The detection array from :meth:`_detection_results_array`.
            classes: Sorted class IDs observed in predictions or targets, used as COCO category IDs.

        Returns:
            The prediction dataset, the target dataset, and, for a multi-IoU-type evaluation, the per-IoU-type
            prediction areas; ``None`` for a single IoU type, which keeps the box area ``load_res`` set.

        Raises:
            ValueError: If a non-empty image's target labels are a floating-point or complex tensor.
        """
        coco_target = self._hotcoco_target_from_arrays(classes)
        detection_rles = _rle_dicts(self.detection_mask) if "segm" in self.iou_type else None
        coco_preds = coco_target.load_res(detections, segmentation=detection_rles)
        if len(self.iou_type) == 1:
            return coco_preds, coco_target, None
        # The box area is the product of the array's width and height columns, the same float64 product
        # TorchMetrics' `area_bbox` computes from the converted Python floats.
        area_columns = {"bbox": detections[:, 3] * detections[:, 4], "segm": self._mask_areas(detection_rles or [])}
        return coco_preds, coco_target, _PredictionAreaColumns(area_columns)

    def _hotcoco_target_from_arrays(self, classes: list[int]) -> Any:
        """Return the hotcoco ground-truth dataset built from the stored target columns by ``COCO.from_arrays``.

        Reproduces what TorchMetrics' ``_get_coco_format`` writes for each target annotation
        (``helpers.py:504-571``): annotation IDs ``1..N`` in state order (``from_arrays``' default), the stored
        ``iscrowd``, and the stored ``area`` where it is positive, otherwise the mask area when ``segm`` is evaluated
        and the box area when not. Target areas are never switched per IoU type, so a box evaluation that runs
        beside ``segm`` also reads the mask-area fallback, as upstream's does. Each image takes its height and width
        from its first mask when ``segm`` is evaluated, and every stored image is listed: with boxes present,
        upstream drops none.

        Args:
            classes: Sorted class IDs observed in predictions or targets, used as COCO category IDs.

        Returns:
            The indexed hotcoco target dataset.

        Raises:
            ValueError: If a non-empty image's target labels are a floating-point or complex tensor, which upstream
                rejects per annotation and ``from_arrays`` would reject with a ``TypeError``.
        """
        self._validate_detection_labels(self.groundtruth_labels)
        boxes = torch.cat([image.reshape(-1, 4) for image in self.groundtruth_box]).double().numpy()
        rles = _rle_dicts(self.groundtruth_mask) if "segm" in self.iou_type else None
        fallback_area = self._mask_areas(rles) if rles is not None else boxes[:, 2] * boxes[:, 3]
        stored_area = torch.cat(self.groundtruth_area).double().numpy()
        images: list[dict[str, Any]] = [{"id": image_id} for image_id in range(len(self.groundtruth_labels))]
        if rles is not None:
            for image, masks in zip(images, self.groundtruth_mask):
                if len(masks) > 0:
                    image["height"], image["width"] = masks[0][0]
        annotations_per_image = torch.tensor([len(labels) for labels in self.groundtruth_labels])
        coco_type = cast(Any, self._coco_backend.coco)
        return coco_type.from_arrays(
            images,
            [{"id": class_id, "name": str(class_id)} for class_id in classes],
            torch.repeat_interleave(torch.arange(len(images)), annotations_per_image).numpy(),
            # Validated above, so only an empty floating-point tensor can have promoted the concatenation.
            torch.cat(self.groundtruth_labels).to(torch.int64).numpy(),
            boxes,
            area=np.where(stored_area > 0, stored_area, fallback_area),
            iscrowd=torch.cat(self.groundtruth_crowds).to(torch.int64).numpy(),
            segmentation=rles,
        )

    def _mask_areas(self, rles: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
        """Return the pixel area of each RLE mask as float64, through the backend's own mask utilities.

        Args:
            rles: COCO RLE dictionaries.

        Returns:
            One area for each mask, in order.
        """
        mask_utils = cast(Any, self._coco_backend.mask_utils)
        return np.asarray(mask_utils.area(rles), dtype=np.float64) if rles else np.zeros(0)

    def _loads_detections_from_array(self, detection_boxes: list[Tensor] | None) -> bool:
        """Return whether predictions can be loaded from a detection array instead of built as annotation dicts.

        Building the prediction dataset is the dominant cost of ``compute()`` once the evaluator is fast: at COCO
        validation scale TorchMetrics materializes one Python dict per detection, over a million of them, which
        takes longer than hotcoco or ufcoco needs to evaluate them. ``loadRes`` accepts the same detections as one
        array and parses it in Rust. hotcoco takes every IoU type this way, with masks handed over beside the array
        (``load_res(array, segmentation=...)``); a mask-only state has no boxes to fill the array and keeps the dict
        path. ufcoco keeps the array rows in columns through evaluation even under the adapter's ``COCOeval``
        subclass from ultrafast-pycocotools 0.1.13, the floor ``rfdetr[train]`` pins, but its ``loadRes`` takes no
        masks, so it is box-only. faster-coco-eval is excluded because its own ``loadRes`` is slower than the dict
        path it would replace.

        Args:
            detection_boxes: Stored detection boxes, or ``None`` when the state holds none.

        Returns:
            Whether the detection-array path applies.
        """
        if detection_boxes is None:
            return False
        if isinstance(self._coco_backend, _HotCocoBackend):
            return True
        return isinstance(self._coco_backend, _UfcocoBackend) and tuple(self.iou_type) == ("bbox",)

    def _detection_results_array(self) -> np.ndarray[Any, Any]:
        """Return stored detections as the array COCO's ``loadRes`` accepts, which is also vernier's ``(N, 7)``.

        Columns are ``[image_id, x, y, width, height, score, category_id]``, in stored-state order so that
        equal-scoring detections keep the tie order the annotation-dict path produced. Boxes need no conversion:
        TorchMetrics already converted them to COCO's ``xywh`` when ``update()`` stored them.

        ``torch.cat(..., dim=1)`` writes the seven columns into one C-contiguous float64 buffer, which vernier's
        matrix route requires of the caller rather than copying silently. The ID columns ride as float64; vernier
        checks them back to exact integers, safe from ``int64`` state below 2^53.

        Returns:
            One row for each stored detection.

        Raises:
            ValueError: If stored detection scores are not one-dimensional floating-point tensors, or if a non-empty
                image's detection labels are not an integer tensor.
        """
        self._validate_detection_scores(self.detection_scores)
        self._validate_detection_labels(self.detection_labels)
        # TorchMetrics' `_fix_empty_tensors` reshapes a per-image 1-D empty box tensor to `(1, 0)` rather than
        # `(0, 4)` (avoiding a DDP all-reduce hang), which `torch.cat` rejects against a `(N, 4)` tensor from
        # another image, and whose `len()` would otherwise miscount that image as holding one detection instead
        # of zero. `.reshape(-1, 4)` is a no-op on an already-`(N, 4)` tensor and turns a `(1, 0)` one back into
        # `(0, 4)`, so both the concatenation and the per-image counts below read the same corrected shape.
        reshaped_boxes = [image_boxes.reshape(-1, 4) for image_boxes in self.detection_box]
        boxes = torch.cat(reshaped_boxes).double()
        detections_per_image = torch.tensor([len(image_boxes) for image_boxes in reshaped_boxes])
        image_ids = torch.repeat_interleave(torch.arange(len(detections_per_image)), detections_per_image)
        columns = (
            image_ids.double().unsqueeze(1),
            boxes,
            torch.cat(self.detection_scores).double().unsqueeze(1),
            torch.cat(self.detection_labels).double().unsqueeze(1),
        )
        return torch.cat(columns, dim=1).numpy()

    @staticmethod
    def _validate_detection_scores(scores: list[Tensor]) -> None:
        """Restate the per-annotation score checks TorchMetrics performs during conversion.

        Upstream validates that scores are a tensor but not that each is a one-dimensional floating-point one; that
        check only happens while converting one annotation at a time. RF-DETR's paths convert whole images, a
        streamed batch, or the whole state at once, so the check has to be made here instead.

        Args:
            scores: Per-image detection scores, as stored in ``detection_scores``.

        Raises:
            ValueError: If an image's scores are not a one-dimensional floating-point tensor.
        """
        for image_id, image_scores in enumerate(scores):
            if image_scores.ndim != 1:
                raise ValueError(
                    f"Invalid input score of sample {image_id} "
                    f"(expected one-dimensional tensor, got {image_scores.ndim} dimensions)"
                )
            if not torch.is_floating_point(image_scores):
                raise ValueError(
                    f"Invalid input score of sample {image_id} (expected floating point, got {image_scores.dtype})"
                )

    @staticmethod
    def _validate_detection_labels(labels: list[Tensor]) -> None:
        """Restate the per-annotation label check TorchMetrics performs during conversion, for the array paths.

        Upstream rejects any detection or target label that is not a Python ``int`` once converted, one annotation
        at a time -- a whole-valued ``3.0`` included. The array paths never make that conversion: the detection
        array's category column is float64 whatever the stored dtype was, and hotcoco's ``from_arrays`` rejects a
        float category column with a ``TypeError``, so the check is restated here on the stored tensor's dtype, which
        every element shares. Integer and boolean tensors pass, as their elements do upstream; an empty tensor
        passes whatever its dtype, because upstream has no element to check.

        Args:
            labels: Per-image labels, as stored in ``detection_labels`` or ``groundtruth_labels``.

        Raises:
            ValueError: If a non-empty image's labels are a floating-point or complex tensor.
        """
        for image_id, image_labels in enumerate(labels):
            if image_labels.numel() > 0 and (torch.is_floating_point(image_labels) or torch.is_complex(image_labels)):
                raise ValueError(
                    f"Invalid input class of sample {image_id} (expected integer labels, got {image_labels.dtype})"
                )

    def _build_coco(self, dataset: dict[str, Any]) -> Any:
        """Return an indexed backend COCO dataset for a TorchMetrics COCO-format dictionary.

        Args:
            dataset: A COCO-format dictionary as produced by TorchMetrics' ``_get_coco_format``.

        Returns:
            The backend's COCO dataset object, with its annotation index already built.
        """
        coco_factory = cast(Callable[..., Any], self._coco_backend.coco)
        if not isinstance(self._coco_backend, _HotCocoBackend):
            coco = coco_factory()
            coco.dataset = dataset
            with contextlib.redirect_stdout(io.StringIO()):
                coco.createIndex()
            return coco

        return coco_factory(dataset)

    def _prediction_dataset_for_iou_type(
        self, coco_preds: Any, prediction_dataset: dict[str, Any] | _PredictionAreaColumns | None, iou_type: str
    ) -> Any:
        """Point prediction annotation areas at one IoU type of a multi-type evaluation.

        Args:
            coco_preds: The prediction COCO dataset built by :meth:`_coco_datasets`.
            prediction_dataset: What :meth:`_coco_datasets` returned beside ``coco_preds``: the COCO-format
                dictionary it was built from, or the per-IoU-type area columns of a hotcoco set loaded from the
                detection array. Never ``None`` here: ``None`` comes only from a single-IoU-type array load, which
                never reaches this method.
            iou_type: The IoU type whose per-annotation area should become the active ``area``.

        Returns:
            The prediction dataset to evaluate for this IoU type.

        Raises:
            RuntimeError: If a multi-IoU-type evaluation reached a prediction dataset loaded from the detection array
                with no area columns, which only the box-only ufcoco path builds.
        """
        if isinstance(prediction_dataset, _PredictionAreaColumns):
            areas = prediction_dataset.by_iou_type[iou_type]
            # Its evaluator keeps the annotations it was built with, so this must run before `compute()` constructs
            # the evaluator for this IoU type.
            coco_preds.update_anns(ids=np.arange(1, areas.size + 1), area=areas)
            return coco_preds
        if prediction_dataset is None:
            raise RuntimeError(
                "OnePassCocoMeanAveragePrecision cannot switch annotation areas on an array-built prediction "
                "dataset without area columns; only hotcoco loads multi-IoU-type predictions from an array."
            )
        annotations = prediction_dataset["annotations"]
        if isinstance(self._coco_backend, _HotCocoBackend):
            # hotcoco copies the dictionary into its own index at construction, so writing `area` into it would be
            # invisible there; the column form of `update_anns` edits that index in place, with no dict per
            # annotation. Its evaluator keeps the annotations it was built with, so this must run before `compute()`
            # constructs the evaluator for this IoU type.
            coco_preds.update_anns(
                ids=[annotation["id"] for annotation in annotations],
                area=[annotation[f"area_{iou_type}"] for annotation in annotations],
            )
            return coco_preds
        # faster-coco-eval indexes the assigned dictionary itself, so the areas written here are live there at once.
        for annotation in annotations:
            annotation["area"] = annotation[f"area_{iou_type}"]
        return coco_preds

    def _quiet_evaluation(self) -> contextlib.AbstractContextManager[Any]:
        """Return the standard-output suppression the active backend needs while evaluating.

        Returns:
            A context manager that silences the backend's COCO summary output.
        """
        if isinstance(self._coco_backend, _HotCocoBackend):
            return _silenced_backend_diagnostics()
        return contextlib.redirect_stdout(io.StringIO())

    def _assign_detection_scores(self, annotations: list[dict[str, Any]], scores: list[Tensor]) -> None:
        """Attach stored detection scores to prediction annotations, converting one image of scores at a time.

        Args:
            annotations: Prediction annotations built by TorchMetrics with ``scores=None``, in state order.
            scores: The per-image detection scores those annotations were built from.

        Raises:
            ValueError: If an image's scores are not one-dimensional floating-point tensors, which upstream rejects
                per annotation.
            RuntimeError: If the annotation count no longer matches the stored score count.
        """
        self._validate_detection_scores(scores)
        flat_scores = [score for image_scores in scores for score in image_scores.cpu().tolist()]
        if len(flat_scores) != len(annotations):
            # TorchMetrics validates one score for each prediction box and label at update time
            # (`_input_validator`, helpers.py:95-102), so a mismatch here means its annotation loop changed shape.
            message = (
                f"OnePassCocoMeanAveragePrecision built {len(annotations)} prediction annotations for "
                f"{len(flat_scores)} detection scores with torchmetrics {torchmetrics.__version__}. "
                "Re-verify rfdetr.training.coco_map before upgrade."
            )
            logger.error(message)
            raise RuntimeError(message)
        for annotation, score in zip(annotations, flat_scores):
            annotation["score"] = score

    @staticmethod
    def _mismatched_backend_signatures(backend: Any, present_methods: list[str]) -> list[str]:
        """Return backend methods whose required parameters cannot accept this adapter's calls."""
        mismatched = []
        for name in present_methods:
            parameters = inspect.signature(getattr(backend, name)).parameters
            if not set(_BACKEND_METHOD_PARAMS[name]) <= set(parameters) or any(
                parameters[parameter].kind is inspect.Parameter.POSITIONAL_ONLY
                for parameter in _BACKEND_KEYWORD_PARAMS[name]
            ):
                mismatched.append(name)
        return mismatched

    @staticmethod
    def _evaluator_methods_now_requiring_args(evaluator_type: Any, present_methods: list[str]) -> list[str]:
        """Return evaluator method names that now require an argument this adapter never passes."""
        mismatched = []
        for name in present_methods:
            # `evaluator_type` is the backend's evaluator class, not an instance, so the unbound
            # function's first parameter is `self` — skip it before checking for new required args.
            params = list(inspect.signature(getattr(evaluator_type, name)).parameters.values())[1:]
            if any(p.default is inspect.Parameter.empty and p.kind not in _VAR_PARAM_KINDS for p in params):
                mismatched.append(name)
        return mismatched

    def _validate_private_contract(self) -> None:
        """Fail fast when installed TorchMetrics internals differ from the verified adapter boundary."""
        installed_states = {name for name, default in self._defaults.items() if isinstance(default, list)}
        expected_states = set(_MAP_STATE_ATTRS)
        missing_states = sorted(expected_states - installed_states)
        stale_states = sorted(installed_states - expected_states)
        backend: _RfdetrCocoBackend | None = getattr(self, "_coco_backend", None)
        # Which surfaces to check is the backend's own typed declaration (`_RfdetrCocoBackend`), read against the
        # module constant so a contract change reaches every backend that did not opt out of it. `backend` itself
        # can still be absent -- guarded rather than defaulted through the flags below -- when `_coco_backend`
        # was never assigned.
        unused = backend.unused_backend_methods if backend is not None else ()
        backend_methods = tuple(name for name in _BACKEND_METHOD_PARAMS if name not in unused)
        evaluator_methods = _EVALUATOR_ZERO_ARG_METHODS if backend is None or backend.uses_coco_evaluator else ()
        missing_methods = (
            ["_coco_backend"]
            if backend is None
            else [name for name in backend_methods if not callable(getattr(backend, name, None))]
        )
        present_backend_methods = [name for name in backend_methods if name not in missing_methods]
        mismatched_signatures = (
            self._mismatched_backend_signatures(backend, present_backend_methods) if present_backend_methods else []
        )
        evaluator_type = None
        missing_evaluator_methods: list[str] = []
        mismatched_evaluator_signatures: list[str] = []
        if backend is not None and evaluator_methods:
            try:
                # `_coco_datasets` calls the `coco` dataset factory directly, so a rename upstream must fail here
                # rather than at compute() time. A backend that calls no evaluator method reaches neither.
                evaluator_type, coco_factory = backend.cocoeval, backend.coco
            except (AttributeError, TypeError, ImportError):
                # `CocoBackend.cocoeval` lazily imports the backend package (e.g. `faster_coco_eval`) and
                # raises `ModuleNotFoundError` (an `ImportError`) when it is absent; without catching it
                # here that exception propagates raw instead of the actionable RuntimeError below.
                evaluator_type = coco_factory = None
            if not callable(coco_factory):
                missing_methods = [*missing_methods, "coco"]
            missing_evaluator_methods = (
                ["cocoeval"]
                if evaluator_type is None
                else [name for name in evaluator_methods if not callable(getattr(evaluator_type, name, None))]
            )
            present_evaluator_methods = [name for name in evaluator_methods if name not in missing_evaluator_methods]
            mismatched_evaluator_signatures = (
                self._evaluator_methods_now_requiring_args(evaluator_type, present_evaluator_methods)
                if present_evaluator_methods
                else []
            )
        if not (
            missing_states
            or stale_states
            or missing_methods
            or missing_evaluator_methods
            or mismatched_signatures
            or mismatched_evaluator_signatures
        ):
            return
        message = (
            "OnePassCocoMeanAveragePrecision is incompatible with installed "
            f"torchmetrics {torchmetrics.__version__}. Missing list states: {missing_states}; "
            f"unexpected list states: {stale_states}; missing backend methods: {missing_methods}; "
            f"missing evaluator methods: {missing_evaluator_methods}; "
            f"backend methods with an incompatible signature: {mismatched_signatures}; "
            f"evaluator methods now requiring an argument: {mismatched_evaluator_signatures}. "
            "Re-verify rfdetr.training.coco_map before upgrade."
        )
        logger.error(message)
        raise RuntimeError(message)

    def _observed_classes(self) -> list[int]:
        """Return sorted class IDs observed in predictions or targets."""
        labels = self.detection_labels + self.groundtruth_labels
        if not labels:
            return []
        return cast(list[int], torch.unique(torch.cat(labels)).cpu().tolist())

    def _per_class_sentinels(self, prefix: str, classes: list[int]) -> dict[str, Tensor]:
        """Return compact negative per-class sentinels when COCO evaluation has no images."""
        count = len(classes) if self.class_metrics else 1
        values = torch.full((count,), -1.0, dtype=torch.float32)
        return {
            f"{prefix}map_per_class": values,
            f"{prefix}mar_{self.max_detection_thresholds[-1]}_per_class": values.clone(),
        }

    def _empty_iou_type_results(self, prefix: str, classes: list[int]) -> dict[str, Tensor]:
        """Return every metric key for an IoU type with no images to evaluate, all at COCO's ``-1``.

        Args:
            prefix: Key prefix for this IoU type, empty when only one is evaluated.
            classes: Sorted class IDs observed in predictions or targets.

        Returns:
            The aggregate sentinels upstream emits, plus the per-class ones it omits -- a divergence from stock
            TorchMetrics (module docstring's Outputs section), which the callback relies on.
        """
        return {
            **self._coco_backend._coco_stats_to_tensor_dict(
                12 * [-1.0], prefix=prefix, max_detection_thresholds=self.max_detection_thresholds
            ),
            **self._per_class_sentinels(prefix, classes),
        }

    def _reduce_per_class(self, evaluation: Any, prefix: str, classes: list[int]) -> dict[str, Tensor]:
        """Reduce precision and recall arrays to TorchMetrics-compatible per-class vectors."""
        if not self.class_metrics:
            return self._per_class_sentinels(prefix, classes)
        if not isinstance(evaluation, dict) or "precision" not in evaluation or "recall" not in evaluation:
            message = (
                "OnePassCocoMeanAveragePrecision requires COCO evaluator eval['precision'] and eval['recall'] arrays "
                f"with torchmetrics {torchmetrics.__version__}."
            )
            logger.error(message)
            raise RuntimeError(message)
        precision = np.asarray(evaluation["precision"])
        recall = np.asarray(evaluation["recall"])
        if (
            precision.ndim != 5
            or recall.ndim != 4
            or precision.shape[2] != len(classes)
            or recall.shape[1] != len(classes)
        ):
            message = (
                "OnePassCocoMeanAveragePrecision received incompatible COCO evaluator shapes: "
                f"precision={precision.shape}, recall={recall.shape}, classes={len(classes)}."
            )
            logger.error(message)
            raise RuntimeError(message)

        map_per_class: list[float] = []
        mar_per_class: list[float] = []
        for class_index in range(len(classes)):
            class_precision = precision[:, :, class_index, 0, -1]
            valid_precision = class_precision[class_precision > -1]
            map_per_class.append(float(valid_precision.mean()) if valid_precision.size else -1.0)
            class_recall = recall[:, class_index, 0, -1]
            valid_recall = class_recall[class_recall > -1]
            mar_per_class.append(float(valid_recall.mean()) if valid_recall.size else -1.0)
        return {
            f"{prefix}map_per_class": torch.tensor(map_per_class, dtype=torch.float32),
            f"{prefix}mar_{self.max_detection_thresholds[-1]}_per_class": torch.tensor(
                mar_per_class, dtype=torch.float32
            ),
        }
