# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""OKS keypoint mAP metric backed by :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`."""

from typing import Any

from rfdetr.evaluation.coco_eval import CocoEvaluator


class MetricKeypointOKS:
    """OKS keypoint mAP metric backed by CocoEvaluator.

    Plain Python facade over :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`
    with a :meth:`reset` / :meth:`update` / :meth:`compute` interface that mirrors
    the torchmetrics API shape without subclassing it.

    DDP synchronisation is handled inside :meth:`compute` via
    :meth:`~rfdetr.evaluation.coco_eval.CocoEvaluator.synchronize_between_processes`,
    which uses the repo's pickle-based ``all_gather`` — avoiding the torchmetrics
    deadlock bugs #931 / #449 that affect variable-shape state tensors.

    Supports arbitrary keypoint counts and per-category OKS sigmas through the
    underlying :class:`~rfdetr.evaluation.coco_eval._GroupedKeypointCOCOeval`.

    When TorchMetrics ships production-quality arbitrary-keypoint support (tracked
    in upstream PR #3348), the internals of :meth:`compute` can delegate to
    ``MeanAveragePrecision(iou_type="keypoints", keypoint_format="xyv")`` without
    any change to callers.

    Args:
        coco_gt: COCO ground-truth object (any type accepted by
            :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`).
        keypoint_oks_sigmas: Per-keypoint OKS sigmas. When ``None``, falls back to
            COCO person sigmas for 17-keypoint datasets or a uniform 0.05 sigma for
            other counts.
        max_dets: Maximum detections per image. Defaults to 500.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> metric = MetricKeypointOKS(MagicMock(), max_dets=100)
        >>> metric.has_updates
        False
        >>> metric.reset()  # idempotent on empty state
    """

    def __init__(
        self,
        coco_gt: Any,
        keypoint_oks_sigmas: list[float] | None = None,
        max_dets: int = 500,
    ) -> None:
        self._coco_gt = coco_gt
        self._keypoint_oks_sigmas = keypoint_oks_sigmas
        self._max_dets = max_dets
        # List of per-batch prediction dicts — NOT merged into a single dict.
        # Using a list preserves all predictions when the same image_id appears in
        # multiple batches (e.g. DDP DistributedSampler padding), matching the
        # original CocoEvaluator.update()-per-batch append semantics.
        self._batches: list[dict[int, dict[str, Any]]] = []

    @property
    def has_updates(self) -> bool:
        """Return whether any predictions have been accumulated since last reset.

        Returns:
            ``True`` if :meth:`update` has been called at least once since the
            last :meth:`reset`.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> metric = MetricKeypointOKS(MagicMock())
            >>> metric.has_updates
            False
            >>> metric.update({1: {}})
            >>> metric.has_updates
            True
        """
        return bool(self._batches)

    def reset(self) -> None:
        """Clear accumulated predictions.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> metric = MetricKeypointOKS(MagicMock())
            >>> metric.update({1: {}})
            >>> metric.reset()
            >>> metric.has_updates
            False
        """
        self._batches.clear()

    def update(self, predictions: dict[int, dict[str, Any]]) -> None:
        """Accumulate per-batch predictions.

        Each call appends one batch; predictions are replayed in order inside
        :meth:`compute`.  Predictions for the same ``image_id`` across different
        calls are preserved as separate entries — no overwrite.

        Args:
            predictions: Mapping from ``image_id`` to a prediction dict with keys
                ``boxes`` (``[N, 4]`` xyxy pixel coords), ``scores`` (``[N]``),
                ``labels`` (``[N]`` int), and ``keypoints`` (``[N, K, 3]``
                x/y/confidence in pixel coords). Pass an empty dict for images
                with no predictions.

        Examples:
            >>> from unittest.mock import MagicMock
            >>> metric = MetricKeypointOKS(MagicMock())
            >>> metric.update({1: {}, 2: {}})
            >>> metric.has_updates
            True
        """
        self._batches.append(predictions)

    def compute(self) -> dict[str, float]:
        """Run OKS keypoint evaluation and return metric dict.

        Constructs a fresh :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`,
        replays all accumulated per-batch predictions in order (matching the
        original per-batch ``CocoEvaluator.update()`` call pattern), synchronises
        across DDP ranks via
        :meth:`~rfdetr.evaluation.coco_eval.CocoEvaluator.synchronize_between_processes`,
        and accumulates COCO keypoint statistics.

        Returns:
            Dict with float values for keys ``"map"`` (mAP@50:95), ``"map_50"``
            (AP@50), ``"map_75"`` (AP@75), and ``"mar"`` (AR@50:95).  Any
            unavailable statistic is reported as ``-1.0``.

        Examples:
            >>> from unittest.mock import MagicMock, patch
            >>> import numpy as np
            >>> metric = MetricKeypointOKS(MagicMock(), max_dets=500)
            >>> fake_eval = MagicMock()
            >>> fake_eval.coco_eval = {"keypoints": MagicMock(stats=np.array([0.5, 0.7, 0.4, -1, -1, 0.6]))}
            >>> with patch("rfdetr.evaluation.keypoint_oks.CocoEvaluator", return_value=fake_eval):
            ...     result = metric.compute()
            >>> result["map"]
            0.5
            >>> result["map_50"]
            0.7
        """
        evaluator = CocoEvaluator(
            self._coco_gt,
            ["keypoints"],
            max_dets=self._max_dets,
            keypoint_oks_sigmas=self._keypoint_oks_sigmas,
            log_summary=False,
        )
        for batch in self._batches:
            evaluator.update(batch)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        stats = evaluator.coco_eval["keypoints"].stats
        return {
            "map": float(stats[0]) if len(stats) > 0 else -1.0,
            "map_50": float(stats[1]) if len(stats) > 1 else -1.0,
            "map_75": float(stats[2]) if len(stats) > 2 else -1.0,
            "mar": float(stats[5]) if len(stats) > 5 else -1.0,
        }
