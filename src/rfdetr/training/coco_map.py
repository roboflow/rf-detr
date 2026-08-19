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
    distributed state merging, update-state inspection, and compact one-pass computation. Lightning lifecycle, EMA
    voting, logging, checkpoint metrics, F1, keypoint evaluation, and terminal rendering remain callback concerns.
Usage:
    Import :class:`OnePassCocoMeanAveragePrecision` only from RF-DETR training code. Construct it with the
    ``faster_coco_eval`` backend and ``sync_on_compute=False``, call ``update`` for each batch, explicitly call
    ``merge_distributed_state`` at rank-symmetric callback sites, then call ``compute``.
Outputs:
    Return the same aggregate, per-class, and class-ID tensor keys consumed from TorchMetrics by RF-DETR. Evaluator
    precision, recall, score, and IoU arrays are reduced immediately and are never returned or retained.
Failure:
    Reject extended summaries, alternative backends, micro averaging, implicit distributed synchronization, and any
    installed TorchMetrics private layout that differs from the verified contract. These failures are intentional and
    actionable; there is no silent slow fallback.
Used by:
    ``rfdetr.training.callbacks.coco_eval.COCOEvalCallback`` for train, validation/test, and EMA COCO accumulators.
"""

from __future__ import annotations

import contextlib
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
        coco_preds, coco_target = cast(
            tuple[Any, Any],
            self._coco_backend._get_coco_datasets(
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

    def _validate_private_contract(self) -> None:
        """Fail fast when installed TorchMetrics internals differ from the verified adapter boundary."""
        installed_states = {name for name, default in self._defaults.items() if isinstance(default, list)}
        expected_states = set(_MAP_STATE_ATTRS)
        missing_states = sorted(expected_states - installed_states)
        stale_states = sorted(installed_states - expected_states)
        backend_methods = ("_get_coco_datasets", "_coco_stats_to_tensor_dict")
        backend = getattr(self, "_coco_backend", None)
        missing_methods = (
            ["_coco_backend"]
            if backend is None
            else [name for name in backend_methods if not callable(getattr(backend, name, None))]
        )
        try:
            evaluator_type = backend.cocoeval if backend is not None else None
        except (AttributeError, TypeError):
            evaluator_type = None
        missing_evaluator_methods = (
            ["cocoeval"]
            if evaluator_type is None
            else [
                name
                for name in ("evaluate", "accumulate", "summarize")
                if not callable(getattr(evaluator_type, name, None))
            ]
        )
        if not (missing_states or stale_states or missing_methods or missing_evaluator_methods):
            return
        message = (
            "OnePassCocoMeanAveragePrecision is incompatible with installed "
            f"torchmetrics {torchmetrics.__version__}. Missing list states: {missing_states}; "
            f"unexpected list states: {stale_states}; missing backend methods: {missing_methods}; "
            f"missing evaluator methods: {missing_evaluator_methods}. "
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
