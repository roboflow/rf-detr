# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCOEvalCallback — torchmetrics-based mAP and F1 evaluation."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import Callback
from torch import Tensor

from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.keypoint_oks import (
    DEFAULT_KEYPOINT_MAX_DETS,
    MetricKeypointOKS,
    OKSKey,
)
from rfdetr.evaluation.matching import (
    build_matching_data,
    distributed_merge_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy
from rfdetr.utilities.console import (
    _IS_RICH_AVAILABLE,
    _get_rich_console,
    _has_progress_bar,
    _render_overall_merged,
    _render_summary_tables,
)
from rfdetr.utilities.distributed import is_dist_avail_and_initialized
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _warn_missing_rich_once(warning_emitted: bool) -> bool:
    """Warn once when metric table rendering is skipped because Rich is unavailable.

    Args:
        warning_emitted: Whether this warning has already been emitted.

    Returns:
        Always ``True``; caller assigns back to suppress future warnings.
    """
    if warning_emitted:
        return True
    logger.warning("Rich is not installed; skipping metric table rendering. Install `rich` to enable tables.")
    return True


def _get_ema_inner_module(ema_cb: Any) -> Any:
    """Return the inner ``nn.Module`` wrapped by an EMA callback.

    ``RFDETREMACallback._average_model`` is a private attribute holding a ``torch.optim.swa_utils.AveragedModel``
    (which exposes the actual module on ``.module``).  This helper centralises the access so that consumers degrade
    gracefully when the EMA model has not yet been initialised — preferable to reaching through two layers of
    private attributes at every call site.

    Args:
        ema_cb: EMA callback instance (or ``None``).

    Returns:
        The inner module wrapped by ``AveragedModel``, or ``None`` when no EMA model is available.
    """
    if ema_cb is None:
        return None
    averaged = getattr(ema_cb, "_average_model", None)
    if averaged is None:
        return None
    return getattr(averaged, "module", averaged)


def _is_running_in_notebook() -> bool:
    """Return whether an active IPython shell is available."""
    with contextlib.suppress(ImportError):
        ipython = importlib.import_module("IPython")
        get_ipython = cast(Callable[[], Any], getattr(ipython, "get_ipython"))
        return get_ipython() is not None
    return False


class COCOEvalCallback(Callback):
    """Validation callback that computes mAP (via torchmetrics) and macro-F1.

    Accumulates predictions and targets across validation batches, then at epoch end computes:

    - ``val/mAP_50_95``, ``val/mAP_50``, ``val/mAP_75``, ``val/mAR`` using
      ``torchmetrics.detection.MeanAveragePrecision``.
    - Per-class ``val/AP/<name>`` when class names are available.
    - ``val/F1``, ``val/precision``, ``val/recall`` from a confidence-threshold
      sweep over compact per-class matching data (DDP-safe).

    For segmentation models (``segmentation=True``) additional metrics ``val/segm_mAP_50_95`` and ``val/segm_mAP_50``
    are logged.

    Args:
        max_dets: Maximum detections per image passed to
            ``MeanAveragePrecision``. Defaults to :data:`~rfdetr.evaluation.keypoint_oks.DEFAULT_KEYPOINT_MAX_DETS`.
        segmentation: When ``True``, evaluate both bbox and segm IoU using
            ``backend="faster_coco_eval"``. Defaults to ``False``.
        eval_interval: Run validation metrics every N epochs. Test metrics are
            always computed when ``trainer.test()`` is called.
        log_per_class_metrics: When ``False``, skip per-class AP computation
            (``MeanAveragePrecision(class_metrics=False)``) as well as the per-class logging/table.
        eval_base_model: When ``False`` (default), ``validation_step`` already forwarded through
            the EMA model directly (see ``TrainConfig.eval_base_model``), so the independent
            duplicate EMA forward pass this callback would otherwise run every validation batch
            is skipped and its predictions are routed to the EMA track. When ``True``,
            ``validation_step`` forwards the base model and this callback runs the second, EMA
            forward pass, so both models are evaluated from independent predictions.
    """

    def __init__(
        self,
        max_dets: int = DEFAULT_KEYPOINT_MAX_DETS,
        segmentation: bool = False,
        eval_interval: int = 1,
        log_per_class_metrics: bool = True,
        keypoint_oks_sigmas: list[float] | None = None,
        in_notebook: bool | None = None,
        eval_base_model: bool = False,
    ) -> None:
        super().__init__()
        self._max_dets = max_dets
        self._segmentation = segmentation
        self._eval_interval = max(1, int(eval_interval))
        self._log_per_class_metrics = bool(log_per_class_metrics)
        self._eval_base_model = bool(eval_base_model)
        self._class_names: list[str] = []
        self._cat_id_to_name: dict[int, str] = {}
        self._f1_local: dict[int, dict[str, Any]] = init_matching_accumulator()
        self._f1_train_local: dict[int, dict[str, Any]] = init_matching_accumulator()
        # Whether the EMA metric received ≥1 update this epoch.  Gates the EMA cross-rank
        # sync so it is issued symmetrically on all DDP ranks (see _should_compute_ema).
        self._ema_has_updates: bool = False
        self._missing_rich_warning_emitted: bool = False
        self._output_widget: Any = None  # ipywidgets.Output, created lazily
        self._keypoint_mode: bool = False
        self._use_segm_metrics: bool = segmentation
        self._train_segm_skip_warned: bool = False
        self._keypoint_oks_metrics: dict[str, MetricKeypointOKS] = {}
        self._keypoint_oks_sigmas = keypoint_oks_sigmas
        self._in_notebook: bool
        if in_notebook is None:
            self._in_notebook = _is_running_in_notebook()
        else:
            self._in_notebook = in_notebook

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Instantiate ``MeanAveragePrecision`` after DDP device placement.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``.
        """
        model_config = getattr(pl_module, "model_config", None)
        # Some callback unit shims omit model_config; missing keypoint flag means bbox/segm evaluation.
        use_grouppose_keypoints = (
            getattr(model_config, "use_grouppose_keypoints", False) if model_config is not None else False
        )
        self._keypoint_mode = use_grouppose_keypoints is True
        self._use_segm_metrics = self._segmentation and not self._keypoint_mode
        iou_type: Any = ["bbox", "segm"] if self._use_segm_metrics else "bbox"
        kwargs: dict[str, Any] = dict(
            # Per-class AP is genuinely skipped (compute + state memory) when per-class logging is
            # off — with class_metrics=True the metric would still pay the per-class cost and the
            # flag would only gate result consumption (#416).
            class_metrics=self._log_per_class_metrics,
            max_detection_thresholds=[1, 10, self._max_dets],
            # Disable torchmetrics' built-in cross-rank sync: its `gather_all_tensors` requires every
            # state tensor to have the same ndim on all ranks, but DDP seg validation produces
            # per-rank states that are scalar on some ranks and vectors on others, so the internal
            # sync issues a different number of collectives per rank and deadlocks (known torchmetrics
            # bug, #931/#449). The adapter merges state with the repo's fixed-order
            # `all_gather`, then compute() runs locally on the full set.
            sync_on_compute=False,
        )
        kwargs["backend"] = "faster_coco_eval"
        self.map_metric = OnePassCocoMeanAveragePrecision(iou_type=iou_type, **kwargs)
        self.map_metric_train = OnePassCocoMeanAveragePrecision(iou_type=iou_type, **kwargs)
        # Separate metric for the EMA model.  Created deterministically on EVERY rank in
        # on_validation_epoch_start / on_test_epoch_start (see _prepare_ema_metric) so its
        # cross-rank compute() sync is issued symmetrically and cannot deadlock DDP val.
        self.map_metric_ema: Any = None

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Release the notebook output widget when the trainer exits.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``.
        """
        self._output_widget = None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Resolve per-class names from the DataModule at the start of training.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self._resolve_class_names(trainer)

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        """Resolve per-class names for a standalone ``trainer.validate()`` run.

        ``on_fit_start`` does not fire on validate-only runs, so per-class AP would otherwise be labelled by numeric id.
        Skipped when names are already resolved (e.g. validation inside ``fit``).

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        if not self._cat_id_to_name:
            self._resolve_class_names(trainer)

    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        """Resolve per-class names for a standalone ``trainer.test()`` run.

        ``on_fit_start`` does not fire on test-only runs (e.g. :meth:`rfdetr.detr.RFDETR.evaluate`), so per-class AP
        would otherwise be labelled by numeric id. Skipped when names are already resolved.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        if not self._cat_id_to_name:
            self._resolve_class_names(trainer)

    def _resolve_class_names(self, trainer: Any) -> None:
        """Build the ``category_id → name`` mapping from the DataModule's COCO metadata.

        Resolves names from the first available dataset split (train, val, or test) so per-class AP is logged under the
        class name regardless of whether the dataset uses sequential or non-sequential category IDs, and regardless of
        which loop (fit / validate / test) is running.

        Args:
            trainer: The PTL Trainer.
        """
        dm = trainer.datamodule
        if dm is None:
            return
        if hasattr(dm, "class_names"):
            self._class_names = dm.class_names or []
        # Build cat_id → name from the COCO annotation object when available.
        for attr in ("_dataset_train", "_dataset_val", "_dataset_test"):
            dataset = getattr(dm, attr, None)
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                if hasattr(coco, "label2cat"):
                    # remap_category_ids=True: dataset labels are 0-based contiguous
                    # indices.  label2cat maps remapped_label → original_cat_id;
                    # use it to build label → name so class IDs match predictions.
                    self._cat_id_to_name = {
                        label: coco.cats[cat_id]["name"] for label, cat_id in coco.label2cat.items()
                    }
                else:
                    # Raw COCO category IDs used as labels (standard COCO dataset).
                    self._cat_id_to_name = {k: v["name"] for k, v in coco.cats.items()}
                return
        # Fallback: treat class_names as 0-based sequential labels.
        self._cat_id_to_name = {i: name for i, name in enumerate(self._class_names)}

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Prepare the EMA metric on every rank before validation (keeps DDP collectives symmetric).

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()
        self._reset_keypoint_split("val")
        self._reset_keypoint_split("val_ema")
        self._prepare_ema_metric(trainer)

    def on_test_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset ``_ema_has_updates`` before test to prevent stale validation state from triggering EMA compute.

        ``on_test_batch_end`` never sets ``_ema_has_updates = True``, so EMA compute is always skipped during
        test (test metrics already reflect the EMA model via checkpoint loading in
        :class:`~rfdetr.training.callbacks.best_model.BestModelCallback`).  Without this hook a stale ``True`` value
        left by a preceding validation epoch would make ``_should_compute_ema`` return ``True``, causing an
        empty-state EMA compute pass that logs sentinel ``-1`` values.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()
        self._reset_keypoint_split("test")
        self._prepare_ema_metric(trainer)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate train predictions for optional train-split mAP logging.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``training_step``.
            batch: The device-transferred batch (unused here).
            batch_idx: Batch index within the training epoch.
        """
        if getattr(getattr(pl_module, "train_config", None), "compute_train_metrics", False) is not True:
            return
        if self._eval_interval > 1 and not self._is_metric_epoch(trainer):
            return
        if not isinstance(outputs, dict) or "results" not in outputs or "targets" not in outputs:
            return

        preds: list[dict[str, Tensor]] = self._convert_preds(outputs["results"])
        # preds omitted: training pred_masks is a sparse dict lacking "masks", so passing it here is inert.
        targets = self._convert_targets(outputs["targets"])
        # In training mode pred_masks is a sparse dict, excluded from postprocess inputs, so
        # preds have no masks key.  torchmetrics requires it when iou_type includes "segm" → skip.
        if self._use_segm_metrics and preds and "masks" not in preds[0]:
            if not self._train_segm_skip_warned:
                logger.info(
                    "Train-split segmentation mAP skipped: pred_masks is a sparse dict during training "
                    "(sparse_forward).  Only val/test segm mAP is available."
                )
                self._train_segm_skip_warned = True
            return
        self.map_metric_train.update(preds, targets)

        iou_type = "segm" if self._use_segm_metrics else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_train_local, batch_matching)
        self._update_keypoint_oks_metric(trainer, outputs, split="train")

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute optional train-split mAP at the end of the training epoch.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        if getattr(getattr(pl_module, "train_config", None), "compute_train_metrics", False) is not True:
            self.map_metric_train.reset()
            self._f1_train_local = init_matching_accumulator()
            self._reset_keypoint_split("train")
            return
        if self._eval_interval > 1 and not self._is_metric_epoch(trainer):
            self.map_metric_train.reset()
            self._f1_train_local = init_matching_accumulator()
            self._reset_keypoint_split("train")
            return
        self._compute_and_log(trainer, pl_module, "train", metric=self.map_metric_train)

    def _is_metric_epoch(self, trainer: Any) -> bool:
        """Decide whether the current epoch falls on an ``_eval_interval`` boundary (or is the final epoch).

        Shared by :meth:`on_train_batch_end` (skip accumulation on non-eval epochs) and
        :meth:`on_train_epoch_end` (skip compute/log and reset accumulators instead).

        Args:
            trainer: The PTL Trainer.

        Returns:
            ``True`` when this epoch should accumulate and log train metrics.
        """
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        is_last_epoch = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
        return current_epoch % self._eval_interval == 0 or is_last_epoch

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions and matching data for one validation batch.

        Expects ``outputs`` to be the dict returned by ``RFDETRModelModule.validation_step``: ``{"results": list[dict],
        "targets": list[dict]}``.

        When an EMA callback is present the EMA model is run on the same batch in a separate ``torch.no_grad()`` forward
        pass so that base and EMA metrics are computed from independent predictions.

        Unless ``eval_base_model`` is set, and once the EMA model has warmed up, ``validation_step`` already
        forwarded through the EMA-averaged weights (see ``RFDETRModelModule._resolve_eval_model``) — these
        predictions are routed to the EMA mAP/checkpoint track (``map_metric_ema`` / ``val/ema_*``) instead of
        the regular one, which never ran a base-model forward pass this batch. Without this routing, the regular
        ``val/mAP_50_95`` key would silently reflect EMA quality while ``BestModelCallback`` checkpoints the
        (unevaluated) base weights under that key — a metric/weights mismatch. ``_compute_and_log`` then mirrors
        the EMA score onto the primary key so monitors keep receiving a real number, and ``BestModelCallback``
        suppresses its base-weights track for the same reason (see its ``evaluates_base_model`` argument).

        The macro-F1 sweep (``val/F1``) has no parallel EMA-tracked accumulator and always follows
        ``validation_step``'s own forward — which under the default is the same EMA model the mirrored
        ``val/mAP_50_95`` reports, so the two agree. Under ``eval_base_model=True`` both follow the base model
        instead, and the EMA track has no F1 counterpart.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``validation_step``.
            batch: The device-transferred batch ``(samples, targets)``.
            batch_idx: Batch index within the validation epoch.
            dataloader_idx: Index of the validation dataloader (unused here).
        """
        if not isinstance(outputs, Mapping):
            return
        preds: list[dict[str, Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"], preds if self._use_segm_metrics else None)
        # ema_cb._average_model availability is rank-invariant (EMA updates fire on the same
        # global step on every rank), so per-rank EMA-forward decisions stay consistent.
        ema_cb = self._get_ema_callback(trainer)
        ema_inner = _get_ema_inner_module(ema_cb)
        used_ema_forward = not self._eval_base_model and ema_inner is not None
        if used_ema_forward:
            if self.map_metric_ema is not None:
                self.map_metric_ema.update(preds, targets)
                self._ema_has_updates = True
        else:
            self.map_metric.update(preds, targets)

        iou_type = "segm" if self._use_segm_metrics else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_local, batch_matching)
        self._update_keypoint_oks_metric(trainer, outputs, split="val_ema" if used_ema_forward else "val")

        # Run EMA model separately on the same batch so that base and EMA metrics
        # are computed from independent forward passes rather than being aliases.
        # The EMA metric object itself is created on every rank in
        # on_validation_epoch_start (_prepare_ema_metric); here we only run the EMA
        # forward pass + update when the averaged model is available.
        # Skipped entirely unless eval_base_model=True: validation_step already forwarded through
        # the EMA model directly (RFDETRModelModule._resolve_eval_model) and the primary preds
        # above are already routed to the EMA track, so this second, independent EMA forward
        # pass would be pure duplicate compute (#416) — the ~3-3.5%-of-epoch saving PR12 claims.
        if self._eval_base_model and ema_cb is not None and ema_inner is not None and self.map_metric_ema is not None:
            samples, _ = batch
            orig_sizes = torch.stack([t["orig_size"] for t in outputs["targets"]]).to(pl_module.device)
            ema_underlying = ema_inner.model
            with torch.no_grad():
                ema_underlying.eval()  # AveragedModel deepcopy is not managed by PTL
                ema_outputs = ema_underlying(samples)
                ema_results = pl_module.postprocess(ema_outputs, orig_sizes)
            ema_preds = self._convert_preds(ema_results)
            ema_targets = self._convert_targets(outputs["targets"], ema_preds if self._use_segm_metrics else None)
            self.map_metric_ema.update(ema_preds, ema_targets)
            self._update_keypoint_oks_metric(
                trainer,
                {"results": ema_results, "targets": outputs["targets"]},
                split="val_ema",
            )
            self._ema_has_updates = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log mAP and F1 metrics at the end of the validation epoch.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        if self._eval_interval > 1:
            current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            max_epochs = getattr(trainer, "max_epochs", None)
            is_last_epoch = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
            if current_epoch % self._eval_interval != 0 and not is_last_epoch:
                self.map_metric.reset()
                if self.map_metric_ema is not None:
                    self.map_metric_ema.reset()
                self._f1_local = init_matching_accumulator()
                self._reset_keypoint_split("val")
                self._reset_keypoint_split("val_ema")
                return
        self._compute_and_log(trainer, pl_module, "val")

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions and matching data for one test batch.

        Mirrors :meth:`on_validation_batch_end` for the test evaluation loop triggered by ``trainer.test()`` at the end
        of training.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``test_step``.
            batch: Raw batch (unused here).
            batch_idx: Batch index within the test epoch.
            dataloader_idx: Index of the test dataloader (unused here).
        """
        if not isinstance(outputs, Mapping):
            return
        preds: list[dict[str, Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"], preds if self._use_segm_metrics else None)

        self.map_metric.update(preds, targets)

        iou_type = "segm" if self._use_segm_metrics else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_local, batch_matching)
        self._update_keypoint_oks_metric(trainer, outputs, split="test")

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log mAP and F1 under ``test/`` prefix at end of test epoch.

        Mirrors :meth:`on_validation_epoch_end` for the test evaluation loop.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self._compute_and_log(trainer, pl_module, "test")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_and_log_ema_metrics(
        self, trainer: Any, pl_module: Any, split: str, pfx: str, mar_key: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """Compute, log, and reset ``map_metric_ema`` if every rank agrees it has data this epoch.

        Extracted out of :meth:`_compute_and_log` so its early-return branch (base ``metric`` empty)
        can call it too — when the base model is not evaluated it never accumulates updates (see
        ``on_validation_batch_end``), so gating EMA logging on the base metric's own guard silently
        drops the EMA metrics as well, leaving the epoch with no validation output at all (#1285).

        The EMA ``compute()`` triggers a cross-rank metric sync, so it must be issued by EVERY rank
        or none: a rank whose EMA metric is empty/absent would otherwise skip this collective and
        desync the DDP collective sequence, deadlocking validation (#931 / #449).
        ``_should_compute_ema`` makes the decision unanimous across ranks.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            split: Metric namespace — ``"val"`` or ``"test"``.
            pfx: torchmetrics key prefix (``"bbox_"`` when ``iou_type`` is a list, else ``""``).
            mar_key: Prefixed AR metric key for ``self._max_dets``.

        Returns:
            ``(should_compute_ema, ema_metrics)`` — whether the EMA metrics were actually computed
            and logged this call (callers use this to decide whether the parallel EMA keypoint split
            should also be logged or reset), and the raw ``compute()`` output when they were, else
            ``None``. Callers that also need per-class EMA data (see ``_print_ema_only_summary``)
            reuse this instead of calling ``compute()`` a second time.
        """
        should_compute_ema = self._should_compute_ema(pl_module)
        ema_metrics: dict[str, Any] | None = None
        if should_compute_ema:
            self.map_metric_ema.merge_distributed_state()
            ema_metrics = self._compute_map_metric(trainer, self.map_metric_ema)
            pl_module.log(
                f"{split}/ema_mAP_50_95",
                ema_metrics[f"{pfx}map"],
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(f"{split}/ema_mAP_50", ema_metrics[f"{pfx}map_50"], logger=True, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/ema_mAR", ema_metrics[mar_key], logger=True, on_step=False, on_epoch=True)
            trainer.callback_metrics[f"{split}/ema_mAP_50_95"] = ema_metrics[f"{pfx}map"].detach().cpu()
            trainer.callback_metrics[f"{split}/ema_mAP_50"] = ema_metrics[f"{pfx}map_50"].detach().cpu()
            trainer.callback_metrics[f"{split}/ema_mAR"] = ema_metrics[mar_key].detach().cpu()
            if self._use_segm_metrics:
                pl_module.log(
                    f"{split}/ema_segm_mAP_50_95", ema_metrics["segm_map"], logger=True, on_step=False, on_epoch=True
                )
                pl_module.log(
                    f"{split}/ema_segm_mAP_50", ema_metrics["segm_map_50"], logger=True, on_step=False, on_epoch=True
                )
                trainer.callback_metrics[f"{split}/ema_segm_mAP_50_95"] = ema_metrics["segm_map"].detach().cpu()
                trainer.callback_metrics[f"{split}/ema_segm_mAP_50"] = ema_metrics["segm_map_50"].detach().cpu()
            self.map_metric_ema.reset()
            self._ema_has_updates = False
        elif self.map_metric_ema is not None:
            # Not all ranks have EMA data this epoch (e.g. EMA not yet warmed up) → skip the
            # sync uniformly on every rank, but clear local state so the next epoch is clean.
            self.map_metric_ema.reset()
            self._ema_has_updates = False
        return should_compute_ema, ema_metrics

    def _mirror_ema_metrics_to_primary_keys(self, trainer: Any, pl_module: Any, split: str) -> None:
        """Copy every ``{split}/ema_<name>`` scalar onto ``{split}/<name>`` when the EMA track is the only one.

        Called only from the branch where the base ``metric`` accumulated nothing, so the primary keys are
        genuinely unwritten this epoch and nothing can be overwritten. Without the mirror, the epoch's real
        score reaches ``val/ema_mAP_50_95`` alone and every scheduler, early-stopping hook, checkpoint monitor
        and dashboard watching ``val/mAP_50_95`` silently sees nothing — tolerable for the opt-in
        ``eval_ema_only`` flag this replaces, not for a default (``TrainConfig.eval_base_model``).

        Mirroring the whole ``ema_`` prefix rather than an enumerated key list keeps the primary namespace in
        step with whatever the EMA track logged for the task at hand — box, segmentation and keypoint keys
        alike, including the task-specific key ``BestModelCallback``/``RFDETREarlyStopping`` monitor. Only
        ``trainer.callback_metrics`` entries are mirrored, which is exactly the headline scalar set: per-class
        AP goes through ``pl_module.log`` alone and deliberately stays under ``{split}/ema_AP/<class>``.

        The mirrored score comes from the EMA weights, so ``BestModelCallback`` must not run its base-weights
        track against it — see its ``evaluates_base_model`` argument, which trainer.py wires from the same
        config field.

        Args:
            trainer: The PTL Trainer, whose ``callback_metrics`` supplies and receives the mirrored values.
            pl_module: The LightningModule used to log the mirrored scalars to external loggers.
            split: Metric namespace — ``"val"`` (the only split that reaches this method).
        """
        ema_prefix = f"{split}/ema_"
        for key in [k for k in trainer.callback_metrics if k.startswith(ema_prefix)]:
            metric_name = key[len(ema_prefix) :]
            primary_key = f"{split}/{metric_name}"
            value = trainer.callback_metrics[key]
            trainer.callback_metrics[primary_key] = value
            # Headline AP keeps its progress-bar slot: `_compute_and_log` shows `{split}/mAP_50_95` there on a
            # base-model epoch, so a default (EMA-only) epoch would otherwise silently lose it from the bar.
            prog_bar = metric_name.lower().endswith("map_50_95")
            pl_module.log(primary_key, value, prog_bar=prog_bar, logger=True, on_step=False, on_epoch=True)

    def _compute_and_log_f1_metrics(
        self, trainer: Any, pl_module: Any, split: str, f1_local: dict[int, dict[str, Any]]
    ) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
        """Sweep confidence thresholds over ``f1_local``, log ``{split}/F1`` (+precision/recall), return per-class F1.

        Independent of ``self.map_metric``/``self.map_metric_ema``: ``f1_local`` accumulates every batch's matching
        data via ``merge_matching_data`` in ``on_validation_batch_end`` unconditionally, regardless of which mAP
        track (base vs EMA) that batch's predictions were routed to. Extracted so the empty-``metric`` early-return
        branch of :meth:`_compute_and_log` can also call it — when the base model is not evaluated ``f1_local`` is the
        only accumulator with real data this epoch, so discarding it via ``_reset_f1_local`` without computing would
        silently drop ``val/F1`` too, even though real matching data was collected (#1285).

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            split: Metric namespace — ``"val"``, ``"test"``, or ``"train"``.
            f1_local: Per-category matching accumulator for this split.

        Returns:
            ``(overall, f1_by_cid)`` — ``overall`` has keys ``"F1"``, ``"Precision"``, ``"Recall"``; ``f1_by_cid``
            maps category_id to its per-class ``f1``/``precision``/``recall`` at the best macro-F1 threshold
            (empty when no matching data was accumulated).
        """
        merged = distributed_merge_matching_data(f1_local)
        f1_by_cid: dict[int, dict[str, float]] = {}
        if merged:
            sorted_ids = sorted(merged.keys())
            per_class_list = [merged[cid] for cid in sorted_ids]
            classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged[cid]["total_gt"] > 0]
            f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
            best = max(f1_results, key=lambda x: x["macro_f1"])
            overall = {
                "F1": float(best["macro_f1"]),
                "Precision": float(best["macro_precision"]),
                "Recall": float(best["macro_recall"]),
            }
            for k, cid in enumerate(sorted_ids):
                f1_by_cid[cid] = {
                    "f1": float(best["per_class_f1"][k]),
                    "precision": float(best["per_class_prec"][k]),
                    "recall": float(best["per_class_rec"][k]),
                }
        else:
            overall = {"F1": 0.0, "Precision": 0.0, "Recall": 0.0}
        pl_module.log(f"{split}/F1", overall["F1"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        pl_module.log(f"{split}/precision", overall["Precision"], logger=True, on_step=False, on_epoch=True)
        pl_module.log(f"{split}/recall", overall["Recall"], logger=True, on_step=False, on_epoch=True)
        trainer.callback_metrics[f"{split}/F1"] = torch.tensor(overall["F1"])
        trainer.callback_metrics[f"{split}/precision"] = torch.tensor(overall["Precision"])
        trainer.callback_metrics[f"{split}/recall"] = torch.tensor(overall["Recall"])
        return overall, f1_by_cid

    def _any_rank_has_updates(self, metric: Any, pl_module: Any) -> bool:
        """Vote — identically on every rank — whether *any* rank has updates for *metric*.

        ``metric.has_updates`` only reflects local per-rank state; branching on it directly lets ranks
        diverge on whether they enter ``merge_distributed_state()``'s ``all_gather`` collectives next,
        desynchronising the DDP collective sequence and deadlocking validation. This makes the decision
        collectively instead. Unlike ``_should_compute_ema``'s ``all_reduce(MIN)`` (unanimous — skip
        whenever any rank is empty, intentionally conservative to avoid discarding EMA state on an
        uneven epoch), this uses ``all_reduce(MAX)``: the base/train mAP path must not silently drop a
        populated rank's data just because a sibling rank's shard was empty, so any single rank voting 1
        makes every rank enter the merge (``merge_distributed_state()`` treats an empty local shard as a
        no-op contribution to the gather).

        Args:
            metric: The mAP accumulator to check (``self.map_metric`` or a split-specific variant).
            pl_module: The LightningModule (provides the device for the reduction).

        Returns:
            ``True`` iff at least one rank accumulated updates this epoch, making it safe — and
            necessary — for every rank to enter ``merge_distributed_state()`` identically.
        """
        vote = 1 if metric.has_updates else 0
        if is_dist_avail_and_initialized():
            flag = torch.tensor([vote], device=getattr(pl_module, "device", "cpu"))
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            vote = int(flag.item())
        return bool(vote)

    def _compute_and_log(self, trainer: Any, pl_module: Any, split: str, *, metric: Any | None = None) -> None:
        """Shared epoch-end logic for validation and test evaluation loops.

        Computes mAP (via ``self.map_metric``), runs the F1 confidence-threshold sweep, logs all scalar metrics via
        ``pl_module.log``, prints two summary tables to the terminal, and resets internal accumulators.  When
        ``self.map_metric_ema`` is set, EMA variants of all metrics (including ``ema_segm_mAP_50_95`` and
        ``ema_segm_mAP_50`` for segmentation models) are logged under the same ``split/`` namespace.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            split: Metric namespace — ``"val"`` or ``"test"``.
            metric: Optional split-specific mAP accumulator. Defaults to the validation/test accumulator.
        """
        metric = self.map_metric if metric is None else metric
        f1_local = self._f1_train_local if split == "train" else self._f1_local
        # torchmetrics prefixes all keys when iou_type is a list (e.g. "bbox_map"). Computed
        # up front (pure, independent of `metric`) so the early-return branch below can also
        # use it to log the EMA-only track when the base model is not evaluated (#1285).
        pfx = "bbox_" if self._use_segm_metrics else ""
        mar_key = f"{pfx}mar_{self._max_dets}"
        if not self._any_rank_has_updates(metric, pl_module):
            metric.reset()
            self._reset_keypoint_split(split)
            # Unless `eval_base_model` is set, on_validation_batch_end routes every prediction to
            # map_metric_ema instead of `metric` (see its docstring), so `metric` never
            # accumulates any update this epoch — that must not also suppress the EMA metrics,
            # or such a run logs no validation output at all (#1285). train/test never
            # populate map_metric_ema this way (on_test_epoch_start resets _ema_has_updates for
            # test, and on_train_epoch_end never reaches this branch with stale EMA state from a
            # prior validation epoch under normal use), so this is scoped to "val" only.
            if split == "val":
                should_compute_ema, ema_metrics = self._compute_and_log_ema_metrics(
                    trainer, pl_module, split, pfx, mar_key
                )
                if should_compute_ema:
                    self._compute_and_log_keypoint_map(
                        "val_ema", pl_module, trainer, log_split="val", metric_prefix="ema_"
                    )
                    self._mirror_ema_metrics_to_primary_keys(trainer, pl_module, split)
                else:
                    self._reset_keypoint_split("val_ema")
                # f1_local accumulates every batch's matching data unconditionally in
                # on_validation_batch_end (merge_matching_data runs outside the used_ema_forward
                # branch), independent of which mAP track a batch's predictions were routed to.
                # When the base model is not evaluated `metric` never updates but f1_local does — computing it here
                # instead of via the unconditional _reset_f1_local below prevents val/F1 from
                # silently going unlogged even though real matching data was collected (#1285).
                f1_overall, f1_by_cid = self._compute_and_log_f1_metrics(trainer, pl_module, split, f1_local)
                # The normal per-class/table path below only ever reads from the base `metric`,
                # which never has data here — without this, an EMA-only epoch prints no console table
                # for the whole run even though ema_metrics has real per-class data (#1285).
                if should_compute_ema and ema_metrics is not None:
                    self._print_ema_only_summary(
                        trainer, pl_module, split, pfx, mar_key, ema_metrics, f1_overall, f1_by_cid
                    )
            self._reset_f1_local(split)
            logger.debug("Skipping %s COCO metric compute because no predictions were accumulated.", split)
            return

        # Merge per-rank state across ranks ourselves (DDP-safe, fixed-shape gather) before the
        # metric computes locally — replaces torchmetrics' deadlock-prone internal sync. No-op when
        # not distributed. Every rank reaches this call once the vote above agrees at least one rank
        # has updates, so the collectives stay symmetric even on a rank whose own shard was empty.
        metric.merge_distributed_state()
        metrics = self._compute_map_metric(trainer, metric)

        overall: dict[str, float] = {
            "mAP 50:95": float(metrics[f"{pfx}map"]),
            "mAP 50": float(metrics[f"{pfx}map_50"]),
            "mAP 75": float(metrics[f"{pfx}map_75"]),
            f"mAR @{self._max_dets}": float(metrics[mar_key]),
        }

        pl_module.log(
            f"{split}/mAP_50_95", metrics[f"{pfx}map"], prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        pl_module.log(
            f"{split}/mAP_50", metrics[f"{pfx}map_50"], prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        pl_module.log(f"{split}/mAP_75", metrics[f"{pfx}map_75"], logger=True, on_step=False, on_epoch=True)
        pl_module.log(f"{split}/mAR", metrics[mar_key], logger=True, on_step=False, on_epoch=True)

        # Write directly into callback_metrics so ModelCheckpoint / EarlyStopping
        # read fresh values each epoch.  pl_module.log() from a callback's
        # on_*_epoch_end goes only to logged_metrics (external loggers), not to
        # callback_metrics, so checkpointing would see stale values otherwise.
        trainer.callback_metrics[f"{split}/mAP_50_95"] = metrics[f"{pfx}map"].detach().cpu()
        trainer.callback_metrics[f"{split}/mAP_50"] = metrics[f"{pfx}map_50"].detach().cpu()
        trainer.callback_metrics[f"{split}/mAP_75"] = metrics[f"{pfx}map_75"].detach().cpu()
        trainer.callback_metrics[f"{split}/mAR"] = metrics[mar_key].detach().cpu()

        should_compute_ema, _ema_metrics = self._compute_and_log_ema_metrics(trainer, pl_module, split, pfx, mar_key)

        if self._use_segm_metrics:
            overall["segm mAP 50:95"] = float(metrics["segm_map"])
            overall["segm mAP 50"] = float(metrics["segm_map_50"])
            pl_module.log(f"{split}/segm_mAP_50_95", metrics["segm_map"], logger=True, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/segm_mAP_50", metrics["segm_map_50"], logger=True, on_step=False, on_epoch=True)
            trainer.callback_metrics[f"{split}/segm_mAP_50_95"] = metrics["segm_map"].detach().cpu()
            trainer.callback_metrics[f"{split}/segm_mAP_50"] = metrics["segm_map_50"].detach().cpu()

        # F1 sweep — run first so per-class F1/prec/rec are available when
        # building the unified per-class table rows below.
        f1_overall, f1_by_cid = self._compute_and_log_f1_metrics(trainer, pl_module, split, f1_local)
        overall.update(f1_overall)

        # Defensive normalization, not currently triggered: OnePassCocoMeanAveragePrecision.compute()
        # (coco_map.py) always returns `classes` and `*_per_class` as 1-d tensors, even for a single
        # class (`torch.tensor([id])` / `torch.full((1,), ...)`), so this branch is dead against the
        # installed torchmetrics 1.8.2 adapter today. Kept as a guard in case that invariant ever
        # changes; ensure it is always 1-d before iterating.
        if "classes" in metrics and metrics["classes"].ndim == 0:
            metrics = dict(metrics)
            metrics["classes"] = metrics["classes"].unsqueeze(0)
            for metric_key in list(metrics):
                value = metrics[metric_key]
                if isinstance(value, Tensor) and value.ndim == 0 and "per_class" in metric_key:
                    metrics[metric_key] = value.unsqueeze(0)

        # Per-class AR from torchmetrics (keyed by category_id).  Gated on
        # self._log_per_class_metrics like the AP path (_build_per_class_rows) —
        # with the flag off, torchmetrics still emits a 0-d mar_*_per_class
        # (class_metrics=False collapses per-class state), which the ndim==0
        # normalizer above only unsqueezes for a genuinely single-class batch,
        # so zip() against a 1-d `classes` would raise TypeError: iteration
        # over a 0-d tensor.  Skipping also avoids computing ar_by_cid when
        # _build_per_class_rows would discard it anyway.
        ar_pc_key = f"{pfx}mar_{self._max_dets}_per_class"
        ar_by_cid: dict[int, float] = {}
        if self._log_per_class_metrics and ar_pc_key in metrics and "classes" in metrics:
            for class_id, ar in zip(metrics["classes"], metrics[ar_pc_key]):
                ar_by_cid[int(class_id)] = float(ar)

        # Unified per-class rows: AP 50:95 | AR | F1 | Precision | Recall
        # Classes with no ground-truth annotations are skipped (pycocotools
        # returns -1 for AP and torchmetrics returns NaN for AR on such classes,
        # so they would show as all dashes in the table).
        per_class = self._build_per_class_rows(
            metrics=metrics, pfx=pfx, split=split, pl_module=pl_module, ar_by_cid=ar_by_cid, f1_by_cid=f1_by_cid
        )

        self._print_metrics_tables(trainer, split, overall, per_class)
        self._compute_and_log_keypoint_map(split, pl_module, trainer)
        if split == "val" and should_compute_ema:
            self._compute_and_log_keypoint_map("val_ema", pl_module, trainer, log_split="val", metric_prefix="ema_")
        elif split == "val":
            self._reset_keypoint_split("val_ema")
        metric.reset()
        self._reset_f1_local(split)

    def _reset_f1_local(self, split: str) -> None:
        """Reset the F1 accumulator for a metric split."""
        if split == "train":
            self._f1_train_local = init_matching_accumulator()
        else:
            self._f1_local = init_matching_accumulator()

    def _get_ema_callback(self, trainer: Any) -> Any:
        """Return the EMA callback instance, or ``None`` if not present."""
        for callback in getattr(trainer, "callbacks", []):
            if callable(getattr(callback, "get_ema_model_state_dict", None)):
                return callback
        return None

    def _compute_map_metric(self, trainer: Any, metric: Any) -> dict[str, Any]:
        """Compute a torchmetrics mAP metric while suppressing duplicate terminal summaries under progress bars."""
        if not _has_progress_bar(trainer):
            result: dict[str, Any] = metric.compute()
            return result

        metric_loggers = (logger, logging.getLogger("faster_coco_eval"), logging.getLogger("faster_coco_eval.core"))
        previous_levels = [(metric_logger, metric_logger.level) for metric_logger in metric_loggers]
        try:
            for metric_logger in metric_loggers:
                if metric_logger.getEffectiveLevel() < logging.WARNING:
                    metric_logger.setLevel(logging.WARNING)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = metric.compute()
                return result
        finally:
            for metric_logger, previous_level in previous_levels:
                metric_logger.setLevel(previous_level)

    def _prepare_ema_metric(self, trainer: Any) -> None:
        """Ensure ``map_metric_ema`` exists (and is reset) on EVERY rank when EMA is active.

        Driven by the rank-invariant presence of the EMA callback rather than by per-batch state, so any cross-rank
        state merge (via the metric adapter) is issued symmetrically across DDP ranks. Previously
        the metric was created lazily in :meth:`on_validation_batch_end`, so a rank with an empty/uneven shard could
        finish without it, skip the merge/compute path, and deadlock validation (#931 / #449).

        Args:
            trainer: The PTL Trainer.
        """
        self._ema_has_updates = False
        if self._get_ema_callback(trainer) is None:
            self.map_metric_ema = None
            return
        if self.map_metric_ema is None:
            ema_iou_type: Any = ["bbox", "segm"] if self._use_segm_metrics else "bbox"
            self.map_metric_ema = OnePassCocoMeanAveragePrecision(
                iou_type=ema_iou_type,
                class_metrics=self._log_per_class_metrics,
                max_detection_thresholds=[1, 10, self._max_dets],
                backend="faster_coco_eval",
                sync_on_compute=False,  # we merge state across ranks ourselves (see map_metric in setup)
            )
        else:
            self.map_metric_ema.reset()

    def _should_compute_ema(self, pl_module: Any) -> bool:
        """Decide — identically on every rank — whether to run the EMA metric ``compute()``.

        Under DDP, ``map_metric_ema.merge_distributed_state()`` issues cross-rank collectives that every rank must
        participate in, or none may — a rank that skips desynchronises the NCCL collective sequence and deadlocks
        validation (#931 / #449).  Each rank votes ``1`` only when its EMA metric exists and received at least
        one batch update this epoch; a cross-rank ``all_reduce(MIN)`` makes the decision unanimous — a single
        rank voting 0 suppresses EMA compute on all ranks.

        Args:
            pl_module: The LightningModule (provides the device for the reduction).

        Returns:
            ``True`` iff every rank both holds an EMA metric object and received at least one batch update this
            epoch, making ``compute()`` safe to run identically on all ranks; ``False`` otherwise (EMA compute
            skipped uniformly on all ranks).
        """
        has_ema = self.map_metric_ema is not None and self._ema_has_updates
        vote = 1 if has_ema else 0
        if is_dist_avail_and_initialized():
            flag = torch.tensor([vote], device=getattr(pl_module, "device", "cpu"))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            vote = int(flag.item())
        return bool(vote)

    def _get_or_create_keypoint_oks_metric(self, trainer: Any, split: str) -> MetricKeypointOKS | None:
        """Return the :class:`~rfdetr.evaluation.keypoint_oks.MetricKeypointOKS` for *split*, creating it if needed.

        The metric is created lazily on first access per split and reused across epochs (state is reset
        at epoch boundaries via :meth:`_reset_keypoint_split`).

        Args:
            trainer: The PTL Trainer (provides access to the datamodule).
            split: One of ``"train"``, ``"val"``, ``"val_ema"``, or ``"test"``.

        Returns:
            A :class:`~rfdetr.evaluation.keypoint_oks.MetricKeypointOKS` bound to the split's COCO
            ground-truth, or ``None`` when no dataset is available.
        """
        if split in self._keypoint_oks_metrics:
            return self._keypoint_oks_metrics[split]

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None

        source_split = split.removesuffix("_ema")
        split_attrs = {
            "train": ("_dataset_train",),
            "val": ("_dataset_val",),
            "test": ("_dataset_test",),
        }.get(source_split, ("_dataset_val", "_dataset_test", "_dataset_train"))
        for attr in split_attrs:
            dataset = getattr(datamodule, attr, None)
            if dataset is None:
                continue
            coco_api = get_coco_api_from_dataset(dataset)
            if coco_api is None:
                continue
            metric = MetricKeypointOKS(
                coco_api,
                keypoint_oks_sigmas=self._keypoint_oks_sigmas,
                max_dets=self._max_dets,
            )
            self._keypoint_oks_metrics[split] = metric
            return metric
        return None

    def _reset_keypoint_split(self, split: str) -> None:
        """Reset accumulated keypoint predictions for *split*.

        Args:
            split: One of ``"train"``, ``"val"``, ``"val_ema"``, or ``"test"``.
        """
        metric = self._keypoint_oks_metrics.get(split)
        if metric is not None:
            metric.reset()

    def _update_keypoint_oks_metric(self, trainer: Any, outputs: Mapping[str, Any], split: str) -> None:
        """Accumulate batch predictions into the keypoint OKS metric.

        Args:
            trainer: The PTL Trainer.
            outputs: Batch output dict with ``"results"`` and ``"targets"`` keys.
            split: Metric split (``"train"``, ``"val"``, ``"val_ema"``, or ``"test"``).
        """
        if not self._keypoint_mode:
            return

        metric = self._get_or_create_keypoint_oks_metric(trainer, split)
        if metric is None:
            return

        predictions: dict[int, dict[str, Tensor]] = {}
        results = outputs["results"]
        targets = outputs["targets"]
        for result, target in zip(results, targets):
            image_id_tensor = target.get("image_id")
            if image_id_tensor is None:
                continue
            image_id = int(image_id_tensor.item()) if torch.is_tensor(image_id_tensor) else int(image_id_tensor)
            if "keypoints" not in result:
                predictions[image_id] = {}
                continue
            predictions[image_id] = {
                "boxes": result["boxes"].detach().cpu(),
                "scores": result["scores"].detach().cpu(),
                "labels": result["labels"].detach().cpu(),
                "keypoints": result["keypoints"].detach().cpu(),
            }

        if not predictions:
            return
        metric.update(predictions)

    def _compute_and_log_keypoint_map(
        self,
        split: str,
        pl_module: Any,
        trainer: Any,
        *,
        log_split: str | None = None,
        metric_prefix: str = "",
    ) -> None:
        """Compute and log OKS keypoint AP/AR metrics when keypoint mode is active.

        Args:
            split: Internal metric split (``"val"``, ``"val_ema"``, ``"train"``, ``"test"``).
            pl_module: The LightningModule used to log scalar metrics.
            trainer: The PTL Trainer (provides ``callback_metrics``).
            log_split: Namespace prefix for logged keys. Defaults to *split*.
            metric_prefix: Optional string prepended to each metric name (e.g. ``"ema_"``).
        """
        metric = self._keypoint_oks_metrics.get(split)
        if not self._keypoint_mode or metric is None:
            return
        # Cross-rank vote before entering compute(): metric.compute() calls
        # synchronize_between_processes() which issues an all_gather collective. If any
        # rank short-circuits here without joining that collective the process group
        # deadlocks. Use the same all_reduce(MIN) pattern as _should_compute_ema.
        has_updates_vote = 1 if metric.has_updates else 0
        if is_dist_avail_and_initialized():
            flag = torch.tensor([has_updates_vote], device=getattr(pl_module, "device", "cpu"))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            has_updates_vote = int(flag.item())
        if not has_updates_vote:
            return

        log_split = split if log_split is None else log_split
        try:
            stats = metric.compute()
            keypoint_metrics = {
                "keypoint_map_50_95": (OKSKey.MAP, True),
                "keypoint_map_50": (OKSKey.MAP_50, True),
                "keypoint_map_75": (OKSKey.MAP_75, False),
                "keypoint_mAR": (OKSKey.MAR, False),
            }
            for metric_name, (stat_key, prog_bar) in keypoint_metrics.items():
                value = stats.get(stat_key, -1.0)
                if value < 0:
                    continue
                log_key = f"{log_split}/{metric_prefix}{metric_name}"
                pl_module.log(log_key, value, prog_bar=prog_bar, logger=True, on_step=False, on_epoch=True)
                trainer.callback_metrics[log_key] = torch.tensor(value)
        finally:
            metric.reset()

    def _print_ema_only_summary(
        self,
        trainer: Any,
        pl_module: Any,
        split: str,
        pfx: str,
        mar_key: str,
        ema_metrics: dict[str, Any],
        f1_overall: dict[str, float],
        f1_by_cid: dict[int, dict[str, float]],
    ) -> None:
        """Print the Rich summary table from ``ema_metrics`` when it is the epoch's only track with data.

        The normal table/per-class path in :meth:`_compute_and_log` only ever reads from the base
        ``metric`` — when the base model is not evaluated that metric never accumulates a single update (see the
        early-return branch), so that path never runs and no table is ever printed for the entire run,
        even though ``map_metric_ema`` has real per-class data (built with the same ``class_metrics``
        setting as the base metric). Without this, fixing the logged scalars alone leaves the console
        table half of the original "no validation output at all" complaint (#1285) unfixed. Per-class
        AP is logged under ``ema_`` keys (via ``_build_per_class_rows``'s ``metric_prefix``) so it never
        collides with a base-track ``{split}/AP/{name}`` key — consistent with ``val/ema_mAP_50_95``
        staying separate from ``val/mAP_50_95`` elsewhere in this callback.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            split: Metric namespace — ``"val"`` (the only split that reaches this method).
            pfx: torchmetrics key prefix (``"bbox_"`` when ``iou_type`` is a list, else ``""``).
            mar_key: Prefixed AR metric key for ``self._max_dets``.
            ema_metrics: Raw ``map_metric_ema.compute()`` output, from ``_compute_and_log_ema_metrics``.
            f1_overall: ``{"F1", "Precision", "Recall"}`` from ``_compute_and_log_f1_metrics``; not
                EMA-prefixed by existing design (see ``TrainConfig.eval_base_model`` docstring) since
                ``f1_local`` has no parallel EMA-tracked accumulator.
            f1_by_cid: Per-class F1/precision/recall keyed by ``category_id``, from the same call.
        """
        if "classes" in ema_metrics and ema_metrics["classes"].ndim == 0:
            ema_metrics = dict(ema_metrics)
            ema_metrics["classes"] = ema_metrics["classes"].unsqueeze(0)
            for metric_key in list(ema_metrics):
                value = ema_metrics[metric_key]
                if isinstance(value, Tensor) and value.ndim == 0 and "per_class" in metric_key:
                    ema_metrics[metric_key] = value.unsqueeze(0)

        overall_ema: dict[str, float] = {
            "mAP 50:95": float(ema_metrics[f"{pfx}map"]),
            "mAP 50": float(ema_metrics[f"{pfx}map_50"]),
            "mAP 75": float(ema_metrics[f"{pfx}map_75"]),
            f"mAR @{self._max_dets}": float(ema_metrics[mar_key]),
        }
        if self._use_segm_metrics and "segm_map" in ema_metrics:
            overall_ema["segm mAP 50:95"] = float(ema_metrics["segm_map"])
            overall_ema["segm mAP 50"] = float(ema_metrics["segm_map_50"])
        overall_ema.update(f1_overall)

        ar_pc_key = f"{pfx}mar_{self._max_dets}_per_class"
        ar_by_cid: dict[int, float] = {}
        if self._log_per_class_metrics and ar_pc_key in ema_metrics and "classes" in ema_metrics:
            for class_id, ar in zip(ema_metrics["classes"], ema_metrics[ar_pc_key]):
                ar_by_cid[int(class_id)] = float(ar)

        per_class_ema = self._build_per_class_rows(
            metrics=ema_metrics,
            pfx=pfx,
            split=split,
            pl_module=pl_module,
            ar_by_cid=ar_by_cid,
            f1_by_cid=f1_by_cid,
            metric_prefix="ema_",
        )
        self._print_metrics_tables(trainer, "val (ema)", overall_ema, per_class_ema)

    def _build_per_class_rows(
        self,
        metrics: dict[str, Any],
        pfx: str,
        split: str,
        pl_module: Any,
        ar_by_cid: dict[int, float],
        f1_by_cid: dict[int, dict[str, float]],
        metric_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Build per-class rows and emit per-class AP metrics.

        Args:
            metrics: Output of ``MeanAveragePrecision.compute()``.
            pfx: Key prefix for bbox metrics when segmentation mode is enabled.
            split: Metric namespace (``"val"`` or ``"test"``).
            pl_module: LightningModule used for metric logging.
            ar_by_cid: Per-class AR keyed by ``category_id``.
            f1_by_cid: Per-class F1/precision/recall keyed by ``category_id``.
            metric_prefix: Prepended to the logged key (``f"{split}/{metric_prefix}AP/{name}"``), e.g.
                ``"ema_"`` so per-class EMA AP (see ``_print_ema_only_summary``) never collides with
                the regular track's ``{split}/AP/{name}`` keys.

        Returns:
            Per-class rows for table rendering.
        """
        per_class: list[dict[str, Any]] = []
        if not self._log_per_class_metrics:
            return per_class

        pc_key = f"{pfx}map_per_class"
        if pc_key not in metrics or "classes" not in metrics:
            return per_class

        for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
            ap_f = float(ap)
            ar_f = ar_by_cid.get(int(class_id), float("nan"))
            if ap_f < 0 and (ar_f != ar_f or ar_f < 0):  # no ground-truth: skip ghost class
                continue
            idx = int(class_id)
            name = self._cat_id_to_name.get(idx, str(idx))
            pl_module.log(f"{split}/{metric_prefix}AP/{name}", ap)
            row: dict[str, Any] = {"name": name, "ap": ap_f, "ar": ar_f}
            row.update(f1_by_cid.get(idx, {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}))
            per_class.append(row)
        return per_class

    def _print_metrics_tables(
        self,
        trainer: Any,
        split: str,
        overall: dict[str, float],
        per_class: list[dict[str, Any]],
    ) -> None:
        """Print two tables to the terminal: overall metrics and per-class metrics.

        The overall table is transposed (metrics as columns, one value row) with true merged group-header cells rendered
        via box-drawing characters: ``mAP`` spans sub-columns 50:95 / 50 / 75, ``mAR`` spans ``@N``, and ``F1 sweep``
        spans F1 / Prec / Recall.  The per-class table uses a standard Rich ``Table`` with columns for AP 50:95, AR, F1,
        Prec, Recall.

        Only runs on the global-zero rank to avoid duplicate output in DDP.

        Args:
            trainer: The PTL Trainer (used to check ``is_global_zero``).
            split: ``"val"`` or ``"test"``.
            overall: Ordered mapping of metric label → scalar value.
            per_class: Per-class dicts with keys ``name``, ``ap``, ``ar``,
                ``f1``, ``precision``, ``recall``; skipped when empty.
        """
        if not getattr(trainer, "is_global_zero", True):
            return
        if not _IS_RICH_AVAILABLE:
            self._missing_rich_warning_emitted = _warn_missing_rich_once(self._missing_rich_warning_emitted)
            return

        console = _get_rich_console(trainer)
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        epoch_sfx = (
            f" (Epoch {current_epoch}/{max_epochs})"
            if isinstance(max_epochs, int) and max_epochs > 0
            else f" (Epoch {current_epoch})"
        )
        title_pfx = split.capitalize() + epoch_sfx
        overall_rendered = _render_overall_merged(title_pfx, overall, self._max_dets)

        if self._in_notebook:
            # Lazily create an ipywidgets.Output on the first table print so it
            # anchors below the progress bar that is already visible.  Subsequent
            # epochs clear only the widget's isolated slot — the main cell output
            # (and PTL's progress bar) is never touched, so there is no flicker.
            if self._output_widget is None:
                with contextlib.suppress(ImportError):
                    widgets = importlib.import_module("ipywidgets")

                    ipython_display = importlib.import_module("IPython.display")
                    display = cast(Callable[..., Any], getattr(ipython_display, "display"))

                    self._output_widget = widgets.Output()
                    display(self._output_widget)

            if self._output_widget is not None:
                self._output_widget.clear_output(wait=True)
                with self._output_widget:
                    _render_summary_tables(console, title_pfx, overall_rendered, per_class)
                return

            # ipywidgets not installed — fall back to IPython cell-level clear so
            # tables replace each other instead of accumulating across epochs.
            with contextlib.suppress(ImportError):
                ipython_display = importlib.import_module("IPython.display")
                clear_output = cast(Callable[..., Any], getattr(ipython_display, "clear_output"))
                clear_output(wait=True)
            _render_summary_tables(console, title_pfx, overall_rendered, per_class)
            return

        # Print directly through the console.  A second rich.live.Live on the same
        # console as RichProgressBar would silently nest (Live._nested=True) and
        # delegate all refresh() calls to the progress-bar renderable, so metric
        # tables would never appear.  console.print() avoids that nesting issue.
        _render_summary_tables(console, title_pfx, overall_rendered, per_class)

    def _convert_preds(self, preds: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        """Normalise prediction dicts from ``PostProcess`` for torchmetrics.

        ``PostProcess.forward`` returns masks with shape ``[K, 1, H, W]`` (the extra channel is introduced by
        ``F.interpolate`` which requires 4-D input).  Both ``torchmetrics.MeanAveragePrecision`` and
        ``engine.build_matching_data`` expect ``[K, H, W]``, so squeeze the channel dim when present.

        ``PostProcess.forward`` currently returns ``[K, 1, H, W]`` masks. Keep this callback-local squeeze for metric
        code paths because ``RFDETR.predict`` and other inference-facing callers still consume the 4-D representation
        and apply ``.squeeze(1)`` at their boundary.

        Args:
            preds: Raw per-image prediction dicts from ``PostProcess``.

        Returns:
            Per-image dicts with ``masks`` squeezed to ``[K, H, W]`` when applicable; all other keys are passed through
            unchanged.
        """
        out = []
        for p in preds:
            entry = dict(p)
            if "masks" in entry and entry["masks"].ndim == 4 and entry["masks"].shape[1] == 1:
                entry["masks"] = entry["masks"].squeeze(1)
            out.append(entry)
        return out

    def _convert_targets(
        self, targets: list[dict[str, Tensor]], preds: list[dict[str, Tensor]] | None = None
    ) -> list[dict[str, Tensor]]:
        """Convert targets from normalised CxCyWH to absolute xyxy boxes.

        Masks use each prediction's pixel grid when available, avoiding a lossy
        model-resolution -> original-resolution -> mask-head-resolution round trip.

        Args:
            targets: Per-image target dicts with ``boxes`` in normalised
                CxCyWH format and ``orig_size`` as ``[H, W]``.
            preds: Converted per-image predictions. Their mask shapes select the
                target mask grid during segmentation evaluation. When provided,
                ``preds`` must have the same length and order as ``targets``:
                the two are paired positionally 1:1 (``preds[i]`` describes the
                same image as ``targets[i]``).

        Returns:
            Per-image dicts with ``boxes`` in absolute xyxy, ``labels``, and optionally ``masks`` and ``iscrowd``.
        """
        if preds is not None:
            assert len(preds) == len(targets), (
                f"preds and targets must be positionally paired 1:1; got {len(preds)} preds vs {len(targets)} targets"
            )
        out = []
        # Stack every target's orig_size into one device-to-host synchronization instead of
        # one per target inside the loop (same fix as PostProcess._postprocess_masks).
        orig_sizes_list = torch.stack([t["orig_size"] for t in targets]).tolist() if targets else []
        for index, t in enumerate(targets):
            h, w = orig_sizes_list[index]
            scale = t["boxes"].new_tensor([w, h, w, h])
            boxes = box_cxcywh_to_xyxy(t["boxes"]) * scale
            entry: dict[str, Tensor] = {"boxes": boxes, "labels": t["labels"]}
            if "masks" in t:
                masks = t["masks"].bool()
                mask_size = (int(h), int(w))
                # Native-grid path assumes a uniform (square-resized) batch: every image shares one
                # grid, so reusing the prediction's mask resolution is safe. Under non-square
                # mixed-size padded batches, mask_size would be the batch-wide padded grid while
                # masks is the unpadded GT, and resizing to it would stretch content — that
                # configuration is unsupported (WAD, see issue #481).
                if preds is not None and "masks" in preds[index]:
                    pred_mask_shape = preds[index]["masks"].shape
                    mask_size = (int(pred_mask_shape[-2]), int(pred_mask_shape[-1]))
                if masks.shape[-2:] != mask_size:
                    masks = (
                        F.interpolate(
                            masks.float().unsqueeze(1),
                            size=mask_size,
                            mode="nearest",
                        )
                        .squeeze(1)
                        .bool()
                    )
                entry["masks"] = masks
            if "iscrowd" in t:
                entry["iscrowd"] = t["iscrowd"]
            out.append(entry)
        return out
