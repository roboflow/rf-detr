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
    Import :class:`OnePassCocoMeanAveragePrecision` only from RF-DETR training code. Construct it with the
    ``faster_coco_eval`` backend (or the optional ``hotcoco`` backend) and ``sync_on_compute=False``, call
    ``update`` for each batch, explicitly call ``merge_distributed_state`` at rank-symmetric callback sites, then
    call ``compute``.
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
    actionable; there is no silent slow fallback.
Used by:
    ``rfdetr.training.callbacks.coco_eval.COCOEvalCallback`` for train, validation/test, and EMA COCO accumulators.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import sys
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

import numpy as np
import torch
import torchmetrics
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

from rfdetr.utilities.distributed import all_gather, get_world_size, is_dist_avail_and_initialized
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_METRIC_INPUT_FIELDS = frozenset({"boxes", "scores", "labels", "masks", "iscrowd", "area"})
# COCO evaluation backends this adapter supports. `pycocotools` is excluded deliberately: it is an order of
# magnitude slower and RF-DETR never installs it.
_SUPPORTED_BACKENDS = ("faster_coco_eval", "hotcoco")
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
# hotcoco requires every image entry to declare its size, but TorchMetrics emits bare ``{"id": ...}`` entries.
# Mask sizes are recovered from each image's RLE; box-only evaluation never reads the image size (COCO area ranges
# come from the annotations), so a placeholder stands in when no mask reveals the real one.
_PLACEHOLDER_IMAGE_SIZE = 1


def _hotcoco() -> Any:
    """Import the optional ``hotcoco`` backend package.

    Returns:
        The imported ``hotcoco`` module.

    Raises:
        ImportError: If the optional dependency is not installed.
    """
    try:
        import hotcoco
    except ImportError as error:
        raise ImportError(
            "backend='hotcoco' requires the hotcoco package; install it with: pip install 'rfdetr[train]'"
        ) from error
    return hotcoco


@contextlib.contextmanager
def _silenced_rust_output() -> Iterator[None]:
    """Silence writes to the standard output and error file descriptors, plus Python-level standard output.

    hotcoco prints from Rust, straight to the file descriptors, so :func:`contextlib.redirect_stdout` does not
    reach it: ``summarize()`` puts a twelve-line COCO summary table on descriptor 1 and, on descriptor 2, one
    warning for every evaluator parameter that differs from the COCO defaults. RF-DETR deliberately overrides
    ``maxDets`` and ``iouThrs`` on every evaluation, so those warnings would repeat each validation epoch and each
    IoU type while describing intended configuration. Genuine failures still surface: the bindings raise Python
    exceptions rather than reporting errors on descriptor 2.

    Yields:
        Nothing; both descriptors are restored on exit.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {descriptor: os.dup(descriptor) for descriptor in (1, 2)}
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        for descriptor in saved:
            os.dup2(devnull, descriptor)
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for descriptor, saved_descriptor in saved.items():
            os.dup2(saved_descriptor, descriptor)
            os.close(saved_descriptor)
        os.close(devnull)


class _HotCocoMaskUtils:
    """Hotcoco's RLE mask utilities with the input coercion TorchMetrics' encode call needs.

    TorchMetrics hands ``encode`` the boolean array it gets straight from the mask tensor. faster-coco-eval accepts
    that; hotcoco's Rust binding accepts ``uint8`` only and rejects anything else with a bare
    ``TypeError: 'ndarray' object is not an instance of 'ndarray'``. Every other utility is forwarded untouched.

    Args:
        mask_utils: hotcoco's ``mask`` module.
    """

    def __init__(self, mask_utils: Any) -> None:
        self._mask_utils = mask_utils

    def encode(self, mask: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Return the run-length encoding of one binary mask.

        Args:
            mask: A two-dimensional binary mask.

        Returns:
            The COCO run-length encoding of the mask.
        """
        return cast(dict[str, Any], self._mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))

    def __getattr__(self, name: str) -> Any:
        """Forward every other mask utility to hotcoco unchanged.

        Args:
            name: The attribute to resolve on hotcoco's ``mask`` module.

        Returns:
            The resolved hotcoco attribute.
        """
        return getattr(self._mask_utils, name)


class _HotCocoBackend(CocoBackend):
    """TorchMetrics COCO backend that resolves to ``hotcoco`` instead of ``faster-coco-eval``.

    TorchMetrics resolves its COCO, evaluator and mask modules from a closed backend-name enum, so the parent is
    constructed with the supported ``faster_coco_eval`` name and each of the three resolved surfaces is overridden here.
    Only the surfaces are swapped: every private helper the adapter calls on the backend (COCO-format construction,
    statistics conversion) is TorchMetrics' own and stays shared with the default backend.
    """

    def __init__(self) -> None:
        super().__init__("faster_coco_eval")
        # Import eagerly so a missing optional dependency reports itself. The contract check that runs next
        # resolves `cocoeval` inside an `except ImportError`, which would otherwise swallow the actionable install
        # hint and report a torchmetrics incompatibility instead.
        _hotcoco()

    @property
    def coco(self) -> object:
        """Return hotcoco's COCO dataset type."""
        return _hotcoco().COCO

    @property
    def cocoeval(self) -> object:
        """Return hotcoco's COCO evaluator type."""
        return _hotcoco().COCOeval

    @property
    def mask_utils(self) -> object:
        """Return hotcoco's RLE mask utilities."""
        return _HotCocoMaskUtils(_hotcoco().mask)


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
        backend: COCO evaluation backend. ``"hotcoco"`` is the default; ``"faster_coco_eval"`` selects the previous
            evaluator. Both ship with ``rfdetr[train]`` and return identical metrics.
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
        backend: Literal["faster_coco_eval", "hotcoco"] = "hotcoco",
        **kwargs: Any,
    ) -> None:
        if extended_summary:
            raise ValueError("OnePassCocoMeanAveragePrecision does not support extended_summary=True")
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"OnePassCocoMeanAveragePrecision requires backend in {_SUPPORTED_BACKENDS}")
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
            # TorchMetrics resolves its COCO modules from a closed backend-name enum that has no hotcoco member, so
            # the supported name is what upstream sees and the resolved surfaces are replaced afterwards.
            backend="faster_coco_eval",
            sync_on_compute=False,
            **kwargs,
        )
        if backend == "hotcoco":
            self._coco_backend = _HotCocoBackend()
        self._validate_private_contract()

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
        coco_preds, coco_target, prediction_dataset = self._coco_datasets(classes)

        result: dict[str, Tensor] = {}
        for iou_type in self.iou_type:
            prefix = "" if len(self.iou_type) == 1 else f"{iou_type}_"
            if len(self.iou_type) > 1:
                coco_preds = self._prediction_dataset_for_iou_type(coco_preds, prediction_dataset, iou_type)
            if len(coco_preds.imgs) == 0 or len(coco_target.imgs) == 0:
                result.update(
                    self._coco_backend._coco_stats_to_tensor_dict(
                        12 * [-1.0], prefix=prefix, max_detection_thresholds=self.max_detection_thresholds
                    )
                )
                # Divergence from stock TorchMetrics (module docstring's Outputs section): stock omits
                # *_per_class keys on this empty-images branch, but the callback always expects them.
                result.update(self._per_class_sentinels(prefix, classes))
                continue

            evaluator_factory = cast(Callable[..., Any], self._coco_backend.cocoeval)
            # The two backends spell the parameter differently and neither accepts the other's spelling. Passing it
            # positionally would work on both, but then a parameter inserted before it upstream would bind the IoU
            # type to the wrong slot and evaluate a detection run as segmentation, silently.
            iou_type_keyword = "iou_type" if isinstance(self._coco_backend, _HotCocoBackend) else "iouType"
            coco_eval = evaluator_factory(coco_target, coco_preds, **{iou_type_keyword: iou_type})
            # Whole-object assignment, not field-by-field mutation: hotcoco's `params` getter returns a copy, so
            # writing a field through it is a silent no-op that would leave `max_detection_thresholds` at COCO's
            # default 100 with no error. faster-coco-eval returns the live object, where this is equivalent.
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
            result.update(self._reduce_per_class(coco_eval, prefix, classes))

        result["classes"] = torch.tensor(classes, dtype=torch.int32)
        return result

    def _coco_datasets(self, classes: list[int]) -> tuple[Any, Any, dict[str, Any] | None]:
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
            COCO-format dictionary the prediction dataset was built from, or ``None`` when it was loaded from a
            detection array instead. That dictionary is returned rather than read back from the dataset because
            hotcoco keeps only the COCO fields it knows, dropping the ``area_bbox``/``area_segm`` values a
            multi-IoU-type evaluation switches between; ``None`` is safe because the array path is taken only for
            single-IoU-type evaluation, where no area switching happens.

        Raises:
            ValueError: If stored detection scores are not one-dimensional floating-point tensors.
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
        if self._loads_detections_from_array(detection_boxes):
            return coco_target.loadRes(self._detection_results_array()), coco_target, None

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
            self._assign_detection_scores(prediction_dataset["annotations"])
        # A multi-IoU-type evaluation on hotcoco rebuilds the prediction dataset for its first IoU type before
        # reading it, so building one here too would index the whole prediction set an extra time per epoch and
        # throw it away. `compute()` only touches the returned object after that rebuild.
        rebuilt_per_iou_type = len(self.iou_type) > 1 and isinstance(backend, _HotCocoBackend)
        coco_preds = None if rebuilt_per_iou_type else self._build_coco(prediction_dataset)
        return coco_preds, coco_target, prediction_dataset

    def _loads_detections_from_array(self, detection_boxes: list[Tensor] | None) -> bool:
        """Return whether predictions can be loaded from a detection array instead of built as annotation dicts.

        Building the prediction dataset is the dominant cost of ``compute()`` once the evaluator is fast: at COCO
        validation scale TorchMetrics materializes one Python dict per detection, over a million of them, which
        takes longer than hotcoco needs to evaluate them. ``loadRes`` accepts the same detections as one array and
        parses it in Rust. faster-coco-eval is excluded because its own ``loadRes`` is slower than the dict path it
        would replace, and mask evaluation is excluded because an array carries no segmentation.

        Args:
            detection_boxes: Stored detection boxes, or ``None`` when the state holds none.

        Returns:
            Whether the detection-array path applies.
        """
        return (
            isinstance(self._coco_backend, _HotCocoBackend)
            and detection_boxes is not None
            and tuple(self.iou_type) == ("bbox",)
        )

    def _detection_results_array(self) -> np.ndarray[Any, Any]:
        """Return stored detections as the array COCO's ``loadRes`` accepts.

        Columns are ``[image_id, x, y, width, height, score, category_id]``, in stored-state order so that
        equal-scoring detections keep the tie order the annotation-dict path produced. Boxes need no conversion:
        TorchMetrics already converted them to COCO's ``xywh`` when ``update()`` stored them.

        Returns:
            One row for each stored detection.

        Raises:
            ValueError: If stored detection scores are not one-dimensional floating-point tensors.
        """
        self._validate_detection_scores()
        boxes = torch.cat(self.detection_box).double()
        detections_per_image = torch.tensor([len(image_boxes) for image_boxes in self.detection_box])
        image_ids = torch.repeat_interleave(torch.arange(len(detections_per_image)), detections_per_image)
        columns = (
            image_ids.double().unsqueeze(1),
            boxes,
            torch.cat(self.detection_scores).double().unsqueeze(1),
            torch.cat(self.detection_labels).double().unsqueeze(1),
        )
        return torch.cat(columns, dim=1).numpy()

    def _validate_detection_scores(self) -> None:
        """Restate the per-annotation score checks TorchMetrics performs during conversion.

        Upstream validates that scores are a tensor but not that each is a one-dimensional floating-point one; that
        check only happens while converting one annotation at a time. Both of RF-DETR's paths convert whole images
        or the whole state at once, so the check has to be made here instead.

        Raises:
            ValueError: If an image's scores are not a one-dimensional floating-point tensor.
        """
        for image_id, image_scores in enumerate(self.detection_scores):
            if image_scores.ndim != 1:
                raise ValueError(
                    f"Invalid input score of sample {image_id} "
                    f"(expected one-dimensional tensor, got {image_scores.ndim} dimensions)"
                )
            if not torch.is_floating_point(image_scores):
                raise ValueError(
                    f"Invalid input score of sample {image_id} (expected floating point, got {image_scores.dtype})"
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

        image_sizes: dict[int, list[int]] = {}
        for annotation in dataset["annotations"]:
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, dict):
                continue
            # hotcoco's own `mask.encode` returns RLE counts as bytes, but its COCO constructor decodes only the
            # string form and silently treats a bytes payload as an empty mask -- mask AP collapses to 0.0 with no
            # error raised anywhere.
            if isinstance(segmentation["counts"], bytes):
                segmentation["counts"] = segmentation["counts"].decode("utf-8")
            image_sizes[annotation["image_id"]] = segmentation["size"]
        for image in dataset["images"]:
            height, width = image_sizes.get(image["id"], [_PLACEHOLDER_IMAGE_SIZE, _PLACEHOLDER_IMAGE_SIZE])
            image["height"], image["width"] = height, width
        return coco_factory(dataset)

    def _prediction_dataset_for_iou_type(
        self, coco_preds: Any, prediction_dataset: dict[str, Any] | None, iou_type: str
    ) -> Any:
        """Point prediction annotation areas at one IoU type of a multi-type evaluation.

        Args:
            coco_preds: The prediction COCO dataset built by :meth:`_coco_datasets`.
            prediction_dataset: The COCO-format dictionary ``coco_preds`` was built from. Never ``None`` here: the
                array path that returns ``None`` is restricted to single-IoU-type evaluation, which never reaches
                this method.
            iou_type: The IoU type whose per-annotation area should become the active ``area``.

        Returns:
            The prediction dataset to evaluate for this IoU type.

        Raises:
            RuntimeError: If a multi-IoU-type evaluation reached the array-built prediction dataset.
        """
        if prediction_dataset is None:
            raise RuntimeError(
                "OnePassCocoMeanAveragePrecision cannot switch annotation areas on an array-built prediction "
                "dataset; the detection-array path must stay restricted to single-IoU-type evaluation."
            )
        for annotation in prediction_dataset["annotations"]:
            annotation["area"] = annotation[f"area_{iou_type}"]
        if not isinstance(self._coco_backend, _HotCocoBackend):
            # faster-coco-eval indexes the assigned dictionary itself, so the areas just written are already live.
            return coco_preds
        # hotcoco copies the dictionary into its own index at construction, so the areas just written are invisible
        # to the existing dataset and the evaluator has to be handed a rebuilt one.
        return self._build_coco(prediction_dataset)

    def _quiet_evaluation(self) -> contextlib.AbstractContextManager[Any]:
        """Return the standard-output suppression the active backend needs while evaluating.

        Returns:
            A context manager that silences the backend's COCO summary output.
        """
        if isinstance(self._coco_backend, _HotCocoBackend):
            return _silenced_rust_output()
        return contextlib.redirect_stdout(io.StringIO())

    def _assign_detection_scores(self, annotations: list[dict[str, Any]]) -> None:
        """Attach stored detection scores to prediction annotations, converting one image of scores at a time.

        Args:
            annotations: Prediction annotations built by TorchMetrics with ``scores=None``, in state order.

        Raises:
            ValueError: If an image's scores are not one-dimensional floating-point tensors, which upstream rejects
                per annotation.
            RuntimeError: If the annotation count no longer matches the stored score count.
        """
        self._validate_detection_scores()
        flat_scores = [score for image_scores in self.detection_scores for score in image_scores.cpu().tolist()]
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
        backend = getattr(self, "_coco_backend", None)
        backend_methods = tuple(_BACKEND_METHOD_PARAMS)
        if isinstance(backend, _HotCocoBackend):
            # hotcoco never reaches `_get_coco_datasets`: it builds its index in the constructor, so this adapter
            # always assembles the COCO-format dictionaries itself on that path. Guarding a call this backend does
            # not make would block hotcoco users over an upstream rename that cannot affect them.
            backend_methods = tuple(name for name in backend_methods if name != "_get_coco_datasets")
        missing_methods = (
            ["_coco_backend"]
            if backend is None
            else [name for name in backend_methods if not callable(getattr(backend, name, None))]
        )
        present_backend_methods = [name for name in backend_methods if name not in missing_methods]
        mismatched_signatures = (
            self._mismatched_backend_signatures(backend, present_backend_methods) if present_backend_methods else []
        )
        try:
            # `_coco_datasets` calls the `coco` dataset factory directly, so a rename upstream must fail here
            # rather than at compute() time.
            evaluator_type = backend.cocoeval if backend is not None else None
            coco_factory = backend.coco if backend is not None else None
        except (AttributeError, TypeError, ImportError):
            # `CocoBackend.cocoeval` lazily imports the backend package (e.g. `faster_coco_eval`) and
            # raises `ModuleNotFoundError` (an `ImportError`) when it is absent; without catching it
            # here that exception propagates raw instead of the actionable RuntimeError below.
            evaluator_type = coco_factory = None
        if backend is not None and not callable(coco_factory):
            missing_methods = [*missing_methods, "coco"]
        missing_evaluator_methods = (
            ["cocoeval"]
            if evaluator_type is None
            else [name for name in _EVALUATOR_ZERO_ARG_METHODS if not callable(getattr(evaluator_type, name, None))]
        )
        present_evaluator_methods = [
            name for name in _EVALUATOR_ZERO_ARG_METHODS if name not in missing_evaluator_methods
        ]
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

    def _reduce_per_class(self, coco_eval: Any, prefix: str, classes: list[int]) -> dict[str, Tensor]:
        """Reduce COCO evaluator precision and recall arrays to TorchMetrics-compatible class vectors."""
        if not self.class_metrics:
            return self._per_class_sentinels(prefix, classes)
        evaluation = getattr(coco_eval, "eval", None)
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
