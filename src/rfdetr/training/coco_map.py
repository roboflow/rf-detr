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
    ``faster_coco_eval`` backend and ``sync_on_compute=False``, call ``update`` for each batch, explicitly call
    ``merge_distributed_state`` at rank-symmetric callback sites, then call ``compute``.
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
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import torch
import torchmetrics
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

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
# Parameter names `compute()` (coco_map.py) relies on when calling each backend method — by keyword for
# `average`/`prefix`/`max_detection_thresholds`, by position for the rest. A parameter rename upstream would
# make those calls a silent TypeError rather than an actionable contract failure without this check.
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
# Evaluator methods compute() calls with no arguments (coco_eval.evaluate() / .accumulate() / .summarize()) — a
# newly-required parameter upstream would make that call fail at compute() time instead of at construction.
_EVALUATOR_ZERO_ARG_METHODS = ("evaluate", "accumulate", "summarize")
_VAR_PARAM_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


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
        backend: Must remain ``"faster_coco_eval"``; this is the callback's supported backend.
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
        backend: Literal["pycocotools", "faster_coco_eval"] = "faster_coco_eval",
        **kwargs: Any,
    ) -> None:
        if extended_summary:
            raise ValueError("OnePassCocoMeanAveragePrecision does not support extended_summary=True")
        if backend != "faster_coco_eval":
            raise ValueError("OnePassCocoMeanAveragePrecision requires backend='faster_coco_eval'")
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
            backend=backend,
            sync_on_compute=False,
            **kwargs,
        )
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
        coco_preds, coco_target = self._coco_datasets(classes)

        result: dict[str, Tensor] = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for iou_type in self.iou_type:
                prefix = "" if len(self.iou_type) == 1 else f"{iou_type}_"
                if len(self.iou_type) > 1:
                    for annotation in coco_preds.dataset["annotations"]:
                        annotation["area"] = annotation[f"area_{iou_type}"]
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
                coco_eval = evaluator_factory(coco_target, coco_preds, iouType=iou_type)
                coco_eval.params.iouThrs = np.asarray(self.iou_thresholds, dtype=np.float64)
                coco_eval.params.recThrs = np.asarray(self.rec_thresholds, dtype=np.float64)
                coco_eval.params.maxDets = self.max_detection_thresholds
                coco_eval.params.catIds = classes
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

    def _coco_datasets(self, classes: list[int]) -> tuple[Any, Any]:
        """Return the COCO prediction and target datasets, hoisting prediction scores out of the annotation loop.

        TorchMetrics' ``_get_coco_format`` hoists boxes and labels to Python lists once per image but reads scores
        one annotation at a time (``scores[image_id][k].cpu().tolist()``, ``helpers.py:563``), which is a tensor
        index, a device copy and a scalar conversion for every detection. At RF-DETR's validation scale that is
        hundreds of thousands of conversions per ``compute()``. Asking upstream for the same annotations with
        ``scores=None`` and assigning per-image score lists afterwards produces byte-identical datasets while
        converting each image's scores once.

        The rewrite needs prediction annotations to appear in state order with no image dropped. Upstream skips an
        image only when it has no masks *and* no boxes (``helpers.py:508-511``), so the hoist is used only when
        boxes are present; a mask-only prediction state falls back to upstream unchanged.

        Args:
            classes: Sorted class IDs observed in predictions or targets, used as COCO category IDs.

        Returns:
            The prediction and target datasets, in the order ``_get_coco_datasets`` returns them.

        Raises:
            RuntimeError: If upstream stops emitting one annotation for each stored detection score.
        """
        backend = self._coco_backend
        detection_boxes = self.detection_box if len(self.detection_box) > 0 else None
        if detection_boxes is None:
            return cast(
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

        coco_factory = cast(Callable[[], Any], backend.coco)
        coco_target, coco_preds = coco_factory(), coco_factory()
        # `_get_coco_datasets` passes this same list of Python ints (helpers.py:216) even though the parameter is
        # annotated `list[Tensor]`; the values only ever become COCO category IDs.
        all_labels = cast(list[Tensor], classes)
        coco_target.dataset = backend._get_coco_format(
            labels=self.groundtruth_labels,
            boxes=self.groundtruth_box if len(self.groundtruth_box) > 0 else None,
            masks=self.groundtruth_mask if len(self.groundtruth_mask) > 0 else None,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
            iou_type=self.iou_type,
            all_labels=all_labels,
            average=self.average,
        )
        coco_preds.dataset = backend._get_coco_format(
            labels=self.detection_labels,
            boxes=detection_boxes,
            masks=self.detection_mask if len(self.detection_mask) > 0 else None,
            scores=None,
            iou_type=self.iou_type,
            all_labels=all_labels,
            average=self.average,
        )
        self._assign_detection_scores(coco_preds.dataset["annotations"])

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()
        return coco_preds, coco_target

    def _assign_detection_scores(self, annotations: list[dict[str, Any]]) -> None:
        """Attach stored detection scores to prediction annotations, converting one image of scores at a time.

        Args:
            annotations: Prediction annotations built by TorchMetrics with ``scores=None``, in state order.

        Raises:
            ValueError: If an image's scores are not floating point, which upstream rejects per annotation.
            RuntimeError: If the annotation count no longer matches the stored score count.
        """
        for image_id, image_scores in enumerate(self.detection_scores):
            if not torch.is_floating_point(image_scores):
                raise ValueError(
                    f"Invalid input score of sample {image_id} (expected floating point, got {image_scores.dtype})"
                )
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
        """Return backend method names missing a parameter this adapter's call sites rely on."""
        return [
            name
            for name in present_methods
            if not set(_BACKEND_METHOD_PARAMS[name]) <= set(inspect.signature(getattr(backend, name)).parameters)
        ]

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
        backend_methods = tuple(_BACKEND_METHOD_PARAMS)
        backend = getattr(self, "_coco_backend", None)
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
